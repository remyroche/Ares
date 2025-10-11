"""
Directional Specialist Model for Analyst Ensemble

This module provides a LightGBM-based directional specialist model that serves
as the 4th base model in the Analyst ensemble. It specializes in directional
prediction with asymmetric objectives optimized for long vs short scenarios.

Key Features:
- LightGBM with asymmetric objective function
- Directional feature engineering and weighting
- Enhanced sample weighting for directional clarity
- Integration with existing ensemble architecture
- Optimized for directional prediction accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from collections.abc import Sequence
import logging
from dataclasses import dataclass
from enum import Enum

# Core ML imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Import utilities
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.utils.math_validation import (
    safe_divide, validate_finite, validate_positive, safe_mean, safe_std
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
)

# Import base ensemble
from .regime_ensembles.base_ensemble import BaseEnsemble

logger = get_logger('DirectionalSpecialistModel')


class DirectionType(Enum):
    """Direction types for specialist optimization."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class StreamingQuantileTransformer:
    """Streaming quantile transformer storing state for incremental updates."""

    def __init__(self):
        self._state: Dict[str, np.ndarray] = {}

    def _update_state(self, feature: str, values: np.ndarray) -> None:
        if values.size == 0:
            return

        existing = self._state.get(feature)
        if existing is not None and existing.size:
            combined = np.concatenate([existing, values])
            combined.sort()
            self._state[feature] = combined
        else:
            self._state[feature] = np.sort(values)

    def _transform_values(self, feature: str, values: np.ndarray) -> np.ndarray:
        state = self._state.get(feature)
        if state is None or state.size == 0:
            return np.zeros_like(values, dtype=float)

        ranks = np.searchsorted(state, values, side='right').astype(float)
        transformed = ranks / float(len(state))
        return np.clip(transformed, 0.0, 1.0)

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        transformed_df = df.copy()
        for col in columns:
            series = transformed_df[col]
            mask = series.notna()
            values = series[mask].astype(float).values
            self._update_state(col, values)
            transformed = self._transform_values(col, values)
            series.loc[mask] = transformed
            transformed_df[col] = series
        return transformed_df

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        transformed_df = df.copy()
        for col in columns:
            series = transformed_df[col]
            mask = series.notna()
            values = series[mask].astype(float).values
            transformed = self._transform_values(col, values)
            series.loc[mask] = transformed
            transformed_df[col] = series
        return transformed_df


@dataclass
class DirectionalConfig:
    """Configuration for directional specialist model."""

    # LightGBM parameters optimized for directional prediction
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42

    # Cyclic noise injection parameters
    enable_cyclic_noise: bool = True
    cyclic_noise_scale: float = 1e-3
    cyclic_noise_period: int = 512
    
    # Directional optimization parameters
    directional_weight_boost: float = 1.5  # Boost weight for strong directional moves
    asymmetric_loss_alpha: float = 0.7     # Asymmetric loss parameter
    min_directional_threshold: float = 0.1  # Minimum threshold for directional moves
    temporal_decay_tau: float = 7 * 24 * 3600  # Temporal decay constant (seconds)
    
    tau: float = 7.0  # Exponential decay time constant in days for recency weighting

    # Feature engineering parameters
    enable_directional_features: bool = True
    directional_lookback_periods: List[int] = None
    momentum_periods: List[int] = None

    # Feature normalization parameters
    enable_directional_quantiles: bool = False
    
    def __post_init__(self):
        if self.directional_lookback_periods is None:
            self.directional_lookback_periods = [5, 10, 15, 20, 30]
        if self.momentum_periods is None:
            self.momentum_periods = [3, 5, 8, 13, 21]
        validate_positive(self.tau, "tau")


class DirectionalFeatureEngineer:
    """Feature engineering specialized for directional prediction."""
    
    def __init__(self, config: DirectionalConfig):
        self.config = config
        self.logger = logger.getChild('DirectionalFeatureEngineer')
        self.quantile_transformer: Optional[StreamingQuantileTransformer] = None
        if self.config.enable_directional_quantiles:
            self.quantile_transformer = StreamingQuantileTransformer()

    def create_directional_features(
        self,
        df: pd.DataFrame,
        target: np.ndarray,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Create features optimized for directional prediction.

        Args:
            df: Input dataframe with OHLCV data
            target: Target values for directional optimization
            fit: Whether to update quantile transformer state

        Returns:
            DataFrame with enhanced directional features
        """
        if not self.config.enable_directional_features:
            return df.copy()
        
        self.logger.debug("Creating directional features...")
        
        df_enhanced = df.copy()
        
        # Add directional momentum features
        df_enhanced = self._add_directional_momentum_features(df_enhanced)
        
        # Add asymmetric volatility features
        df_enhanced = self._add_asymmetric_volatility_features(df_enhanced)
        
        # Add regime directional bias features
        df_enhanced = self._add_regime_directional_features(df_enhanced, target)
        
        # Add volume directional features
        df_enhanced = self._add_volume_directional_features(df_enhanced)
        
        self.logger.debug(f"Created {df_enhanced.shape[1] - df.shape[1]} directional features")
        
        if self.quantile_transformer is not None:
            df_enhanced = self._apply_quantile_transform(df_enhanced, fit=fit)

        return df_enhanced

    def _apply_quantile_transform(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            return df

        if fit:
            return self.quantile_transformer.fit_transform(df, numeric_columns)

        return self.quantile_transformer.transform(df, numeric_columns)
    
    def _add_directional_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features optimized for directional prediction."""
        
        for period in self.config.momentum_periods:
            # Long-biased momentum (emphasizes sustained moves)
            df[f'momentum_long_{period}'] = (
                df['close'].pct_change(period).rolling(period//2).mean()
            )
            
            # Short-biased momentum (emphasizes rapid moves)
            df[f'momentum_short_{period}'] = (
                df['close'].pct_change(period//2).rolling(period//4).mean()
            )
            
            # Directional strength
            price_change = df['close'].pct_change(period)
            volatility = df['close'].pct_change().rolling(period).std()
            df[f'directional_strength_{period}'] = safe_divide(
                np.abs(price_change), volatility, default=0.0
            )
        
        return df
    
    def _add_asymmetric_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features that capture directional asymmetries."""
        
        returns = df['close'].pct_change()
        
        for period in self.config.directional_lookback_periods:
            # Upside volatility (for long predictions)
            upside_returns = returns.where(returns > 0, 0)
            df[f'upside_volatility_{period}'] = upside_returns.rolling(period).std()
            
            # Downside volatility (for short predictions) 
            downside_returns = returns.where(returns < 0, 0)
            df[f'downside_volatility_{period}'] = downside_returns.rolling(period).std()
            
            # Volatility ratio (asymmetry measure)
            df[f'volatility_asymmetry_{period}'] = safe_divide(
                df[f'upside_volatility_{period}'],
                df[f'downside_volatility_{period}'],
                default=1.0
            )
        
        return df
    
    def _add_regime_directional_features(self, df: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
        """Add features that capture regime-specific directional biases."""
        
        # Calculate rolling directional bias
        target_series = pd.Series(target, index=df.index[:len(target)])
        
        for period in self.config.directional_lookback_periods:
            # Long bias strength
            long_signals = (target_series > self.config.min_directional_threshold).astype(int)
            df[f'long_bias_strength_{period}'] = long_signals.rolling(period).mean()
            
            # Short bias strength
            short_signals = (target_series < -self.config.min_directional_threshold).astype(int)
            df[f'short_bias_strength_{period}'] = short_signals.rolling(period).mean()
            
            # Directional consistency
            directional_changes = np.abs(target_series.diff())
            df[f'directional_consistency_{period}'] = (
                1.0 / (1.0 + directional_changes.rolling(period).std())
            )
        
        return df
    
    def _add_volume_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features optimized for directional prediction."""
        
        if 'volume' not in df.columns:
            return df
        
        price_change = df['close'].pct_change()
        
        for period in self.config.directional_lookback_periods:
            # Volume on up moves (accumulation)
            up_moves = price_change > 0
            df[f'accumulation_volume_{period}'] = (
                (df['volume'] * up_moves).rolling(period).sum() /
                df['volume'].rolling(period).sum()
            )
            
            # Volume on down moves (distribution)
            down_moves = price_change < 0
            df[f'distribution_volume_{period}'] = (
                (df['volume'] * down_moves).rolling(period).sum() /
                df['volume'].rolling(period).sum()
            )
            
            # Volume directional bias
            df[f'volume_directional_bias_{period}'] = (
                df[f'accumulation_volume_{period}'] - df[f'distribution_volume_{period}']
            )
        
        return df


class DirectionalSpecialistModel:
    """
    LightGBM-based directional specialist model for Analyst ensemble.
    
    This model serves as the 4th base model in the ensemble and specializes
    in directional prediction with asymmetric objectives.
    """
    
    def __init__(self, config: Optional[DirectionalConfig] = None):
        """Initialize the directional specialist model."""
        self.config = config or DirectionalConfig()
        self.logger = logger.getChild('DirectionalSpecialistModel')
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required but not available. Please install lightgbm.")
        
        # Initialize feature engineer
        self.feature_engineer = DirectionalFeatureEngineer(self.config)
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Model components
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
        
        # Directional statistics
        self.directional_stats = {}
        
        self.logger.info("🎯 Directional Specialist Model initialized")
        self.logger.info(f"   LightGBM parameters: n_estimators={self.config.n_estimators}")
        self.logger.info(f"   Directional features: {self.config.enable_directional_features}")
        self.logger.info(f"   Asymmetric loss alpha: {self.config.asymmetric_loss_alpha}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        timestamps: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]] = None
    ) -> 'DirectionalSpecialistModel':
        """
        Fit the directional specialist model.
        
        Args:
            X: Input features
            y: Target values
            sample_weight: Optional sample weights
            timestamps: Optional timestamps aligned with samples for temporal decay
            
        Returns:
            Self for method chaining
        """
        self.logger.info("🚀 Training directional specialist model...")
        
        # Create directional features
        X_enhanced = self.feature_engineer.create_directional_features(X, y)

        if self.config.enable_cyclic_noise:
            from src.training.steps.model_training.utils.noise_injection import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
                CyclicNoiseConfig,
                add_cyclic_noise,
            )

            noise_config = CyclicNoiseConfig(
                noise_scale=self.config.cyclic_noise_scale,
                cycle_length=self.config.cyclic_noise_period,
                random_state=self.config.random_state,
            )
            X_prepared = add_cyclic_noise(X_enhanced, noise_config)
        else:
            X_prepared = X_enhanced

        # Create directional sample weights with recency decay
        timestamps = X.index if isinstance(X, pd.DataFrame) else None
        directional_weights = self._create_directional_sample_weights(y, timestamps=timestamps)

        if sample_weight is None:
            sample_weight = directional_weights
        else:
            base_weights = np.asarray(sample_weight, dtype=float)
            if base_weights.shape[0] != directional_weights.shape[0]:
                raise ValueError(
                    "Provided sample_weight must have the same length as y for directional weighting"
                )
            sample_weight = base_weights * directional_weights

        half_life_days = self.config.tau * np.log(2)
        half_life_td = pd.to_timedelta(half_life_days, unit='D')
        self.logger.info(
            "   Recency half-life: %s (tau=%.2f days)",
            half_life_td,
            self.config.tau,
        )
        
        # Initialize LightGBM model
        self.model = lgb.LGBMRegressor(
            objective='regression',
            metric='l1',
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            feature_fraction=self.config.feature_fraction,
            bagging_fraction=self.config.bagging_fraction,
            bagging_freq=self.config.bagging_freq,
            min_child_samples=self.config.min_child_samples,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        # Fit model with directional optimization
        self.model.fit(
            X_prepared,
            y,
            sample_weight=sample_weight,
            eval_set=[(X_prepared, y)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Store feature columns and statistics
        self.feature_columns = X_prepared.columns.tolist()
        self.directional_stats = self._calculate_directional_statistics(y, sample_weight)
        self.is_fitted = True
        
        self.logger.info("✅ Directional specialist model training completed")
        self.logger.info(f"   Features used: {len(self.feature_columns)}")
        self.logger.info(f"   Long samples: {self.directional_stats['long_samples']}")
        self.logger.info(f"   Short samples: {self.directional_stats['short_samples']}")
        self.logger.info(f"   Directional clarity: {self.directional_stats['directional_clarity']:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the directional specialist model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create directional features (using dummy target for feature engineering)
        dummy_target = np.zeros(len(X))
        X_enhanced = self.feature_engineer.create_directional_features(X, dummy_target, fit=False)
        
        # Ensure feature alignment
        X_aligned = X_enhanced.reindex(columns=self.feature_columns, fill_value=0)
        
        # Make predictions
        predictions = self.model.predict(X_aligned)
        
        return predictions
    
    def predict_with_direction_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Make predictions with directional confidence assessment.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_scores, dominant_direction)
        """
        predictions = self.predict(X)
        
        # Calculate directional confidence
        confidence_scores = self._calculate_prediction_confidence(X, predictions)
        
        # Determine dominant direction
        dominant_direction = 'long' if np.mean(predictions) > 0 else 'short'
        
        return predictions, confidence_scores, dominant_direction
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'weight')
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_values = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_columns, importance_values))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_directional_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance categorized by directional type."""
        all_importance = self.get_feature_importance()
        
        categorized_importance = {
            'long_features': {},
            'short_features': {},
            'general_features': {}
        }
        
        for feature, importance in all_importance.items():
            if 'long' in feature or 'accumulation' in feature or 'upside' in feature:
                categorized_importance['long_features'][feature] = importance
            elif 'short' in feature or 'distribution' in feature or 'downside' in feature:
                categorized_importance['short_features'][feature] = importance
            else:
                categorized_importance['general_features'][feature] = importance
        
        return categorized_importance
    
    def _create_directional_sample_weights(
        self,
        y: np.ndarray,
        timestamps: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]] = None
    ) -> np.ndarray:
        """Create sample weights that emphasize directional clarity with temporal decay."""

        weights = np.ones(len(y))

        if len(y) == 0:
            return weights

        # Higher weight for strong directional moves
        strong_threshold = np.percentile(np.abs(y), 75)
        strong_moves = np.abs(y) > strong_threshold
        weights[strong_moves] *= self.config.directional_weight_boost

        # Slightly higher weight for clear directional signals
        clear_long = y > self.config.min_directional_threshold
        clear_short = y < -self.config.min_directional_threshold
        weights[clear_long | clear_short] *= 1.2


        if timestamps is not None:
            timestamp_index = pd.Index(timestamps)

            if len(timestamp_index) != len(y):
                self.logger.warning(
                    "Timestamp index length %d does not match target length %d; skipping recency decay.",
                    len(timestamp_index),
                    len(y),
                )
            else:
                if not pd.api.types.is_datetime64_any_dtype(timestamp_index):
                    timestamp_index = pd.to_datetime(timestamp_index, errors='coerce')
                else:
                    timestamp_index = pd.DatetimeIndex(timestamp_index)

                valid_mask = ~timestamp_index.isna()
                if valid_mask.any():
                    latest_time = timestamp_index[valid_mask].max()
                    age = latest_time - timestamp_index[valid_mask]
                    age_in_days = age / np.timedelta64(1, 'D')
                    tau_days = max(self.config.tau, np.finfo(float).eps)
                    recency_decay = np.exp(-age_in_days / tau_days)

                    recency_factor = np.ones(len(y), dtype=float)
                    recency_factor[valid_mask] = recency_decay
                    if (~valid_mask).any():
                        self.logger.debug(
                            "Encountered %d NaT timestamps; assigning neutral recency weight of 1.0.",
                            np.count_nonzero(~valid_mask),
                        )

                    weights *= recency_factor
                else:
                    self.logger.warning(
                        "All provided timestamps are NaT after conversion; skipping recency decay."
                    )

        return weights

    @staticmethod
    def _extract_timestamps(X: pd.DataFrame) -> Optional[pd.Series]:
        """Extract timestamp information from feature dataframe if available."""

        if isinstance(X.index, pd.DatetimeIndex):
            return pd.Series(X.index, index=X.index)

        for candidate in ['timestamp', 'time', 'datetime']:
            if candidate in X.columns:
                try:
                    return pd.to_datetime(X[candidate], errors='coerce')
                except Exception:
                    continue

        return None

    def _compute_temporal_decay(
        self,
        timestamps: Optional[Union[pd.Series, np.ndarray, Sequence[Any]]],
        n_samples: int
    ) -> Optional[np.ndarray]:
        """Compute exponential decay factors based on sample timestamps."""

        if timestamps is None:
            return None

        try:
            if isinstance(timestamps, pd.Series):
                ts_series = timestamps.reset_index(drop=True)
            elif isinstance(timestamps, (pd.Index, pd.DatetimeIndex)):
                ts_series = pd.Series(timestamps)
            else:
                if isinstance(timestamps, (str, bytes)):
                    raise TypeError("Timestamps must be an iterable of values, not a single string/bytes object.")
                ts_series = pd.Series(list(timestamps))
        except Exception as exc:
            self.logger.warning(f"⚠️ Failed to interpret timestamps for temporal decay: {exc}")
            return None

        if len(ts_series) != n_samples:
            self.logger.warning(
                "⚠️ Timestamp count (%s) does not match sample count (%s); skipping temporal decay.",
                len(ts_series),
                n_samples
            )
            return None

        # Attempt to coerce timestamps into datetime values with heuristics for numeric units
        ts_datetime = self._coerce_to_datetime(ts_series)
        if ts_datetime is None or ts_datetime.isna().all():
            self.logger.warning("⚠️ Unable to parse timestamps for temporal decay; skipping decay application.")
            return None

        valid_mask = ~ts_datetime.isna()
        if not valid_mask.any():
            return None

        valid_timestamps = ts_datetime[valid_mask]
        reference_time = valid_timestamps.max()
        tau = max(float(self.config.temporal_decay_tau), 1.0)

        delta_seconds = (reference_time - valid_timestamps).dt.total_seconds().clip(lower=0.0)
        decay_values = np.exp(-delta_seconds / tau)

        decay = np.ones(n_samples)
        decay[valid_mask.to_numpy()] = decay_values.to_numpy()

        if not valid_mask.all():
            self.logger.warning(
                "⚠️ %d timestamps were invalid for temporal decay and received neutral weight.",
                (~valid_mask).sum()
            )

        return decay

    @staticmethod
    def _coerce_to_datetime(series: pd.Series) -> Optional[pd.Series]:
        """Convert a series of timestamps to datetime, handling common numeric units."""

        if series.empty:
            return None

        if np.issubdtype(series.dtype, np.datetime64):
            return pd.to_datetime(series, utc=True, errors='coerce').tz_convert(None)

        # Handle pandas Period
        if isinstance(series.dtype, pd.PeriodDtype):
            period_converted = series.dt.to_timestamp().astype('datetime64[ns]')
            return pd.to_datetime(period_converted, errors='coerce')

        # Try direct conversion first
        converted = pd.to_datetime(series, errors='coerce', utc=True)
        if not converted.isna().all():
            return converted.tz_convert(None)

        # Numeric fallback with unit detection
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
        except Exception:
            return None

        if numeric_series.isna().all():
            return None

        max_abs = np.nanmax(np.abs(numeric_series.to_numpy()))
        if max_abs > 1e18:
            unit = 'ns'
        elif max_abs > 1e15:
            unit = 'us'
        elif max_abs > 1e12:
            unit = 'ms'
        else:
            unit = 's'

        converted_numeric = pd.to_datetime(numeric_series, unit=unit, errors='coerce', utc=True)
        return converted_numeric.tz_convert(None)
    
    def _calculate_directional_statistics(self, y: np.ndarray, sample_weight: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics about directional distribution."""

        long_mask = y > self.config.min_directional_threshold
        short_mask = y < -self.config.min_directional_threshold

        weight_sum = float(np.sum(sample_weight))
        weight_mean = float(weight_sum / len(y)) if len(y) > 0 else 0.0
        weight_sq_sum = float(np.sum(np.square(sample_weight)))
        effective_sample_count = (
            (weight_sum ** 2) / weight_sq_sum if weight_sq_sum > 0 else float(len(y))
        )

        stats = {
            'total_samples': len(y),
            'long_samples': np.sum(long_mask),
            'short_samples': np.sum(short_mask),
            'neutral_samples': len(y) - np.sum(long_mask) - np.sum(short_mask),
            'long_ratio': np.mean(long_mask),
            'short_ratio': np.mean(short_mask),
            'directional_clarity': (np.sum(long_mask) + np.sum(short_mask)) / len(y),
            'weighted_long_ratio': np.sum(sample_weight[long_mask]) / np.sum(sample_weight),
            'weighted_short_ratio': np.sum(sample_weight[short_mask]) / np.sum(sample_weight),
            'mean_target': np.mean(y),
            'std_target': np.std(y),
            'directional_bias': 'long' if np.mean(y) > 0 else 'short',
            'weight_sum': weight_sum,
            'weight_mean': weight_mean,
            'effective_sample_count': effective_sample_count
        }

        return stats

    def _validate_quantile_feature_range(self, df: pd.DataFrame) -> None:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return

        min_val = numeric_df.min().min()
        max_val = numeric_df.max().max()

        if min_val < -1e-6 or max_val > 1.0 + 1e-6:
            raise ValueError(
                "Quantile-transformed features must be within [0, 1]. "
                f"Observed range: [{min_val}, {max_val}]"
            )

    def _calculate_prediction_confidence(self, X: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        
        # Simple confidence based on prediction magnitude and consistency
        pred_magnitude = np.abs(predictions)
        pred_consistency = 1.0 / (1.0 + np.std(predictions))
        
        # Normalize to 0-1 range
        magnitude_norm = pred_magnitude / (np.max(pred_magnitude) + 1e-8)
        confidence_scores = (magnitude_norm + pred_consistency) / 2
        
        return np.clip(confidence_scores, 0.0, 1.0)


# Integration with existing ensemble architecture
class DirectionalSpecialistEnsemble(BaseEnsemble):
    """
    Ensemble wrapper for directional specialist model.
    
    This allows the directional specialist to integrate seamlessly
    with the existing ensemble architecture.
    """
    
    def __init__(self, config: Optional[DirectionalConfig] = None):
        super().__init__()
        self.directional_model = DirectionalSpecialistModel(config)
        self.logger = logger.getChild('DirectionalSpecialistEnsemble')
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> 'DirectionalSpecialistEnsemble':
        """Fit the directional specialist ensemble."""
        timestamps = kwargs.get('timestamps')
        sample_weight = kwargs.get('sample_weight')
        self.directional_model.fit(X, y, sample_weight=sample_weight, timestamps=timestamps)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the directional specialist."""
        return self.directional_model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the directional specialist model."""
        return {
            'model_type': 'DirectionalSpecialist',
            'base_algorithm': 'LightGBM',
            'specialization': 'Directional Prediction',
            'feature_count': len(self.directional_model.feature_columns) if self.directional_model.is_fitted else 0,
            'directional_stats': self.directional_model.directional_stats if self.directional_model.is_fitted else {}
        }


# Convenience functions for easy integration
def create_directional_specialist_model(config: Optional[DirectionalConfig] = None) -> DirectionalSpecialistModel:
    """Create a directional specialist model with optional configuration."""
    return DirectionalSpecialistModel(config)

def create_directional_specialist_ensemble(config: Optional[DirectionalConfig] = None) -> DirectionalSpecialistEnsemble:
    """Create a directional specialist ensemble with optional configuration."""
    return DirectionalSpecialistEnsemble(config)


# Example usage and testing
if __name__ == '__main__':
    tprint("🧪 Testing Directional Specialist Model")
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic OHLCV data
    test_data = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Add additional features
    for i in range(n_features - 5):
        test_data[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Create directional targets
    returns = test_data['close'].pct_change()
    target = returns.rolling(5).mean().shift(-5).fillna(0)  # Future directional movement
    
    # Test directional specialist
    config = DirectionalConfig(n_estimators=100)  # Reduced for testing
    model = DirectionalSpecialistModel(config)
    
    # Fit model
    model.fit(test_data, target.values)
    
    # Make predictions
    predictions = model.predict(test_data)
    
    # Get directional predictions with confidence
    pred, conf, direction = model.predict_with_direction_confidence(test_data)
    
    # Get feature importance
    importance = model.get_feature_importance()
    directional_importance = model.get_directional_feature_importance()
    
    tprint("✅ Directional Specialist Model test completed!")
    tprint(f"   Predictions shape: {predictions.shape}")
    tprint(f"   Dominant direction: {direction}")
    tprint(f"   Mean confidence: {np.mean(conf):.3f}")
    tprint(f"   Top features: {list(importance.keys())[:5]}")
    tprint(f"   Long features: {len(directional_importance['long_features'])}")
    tprint(f"   Short features: {len(directional_importance['short_features'])}")

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
