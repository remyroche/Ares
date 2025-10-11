"""
Normalization & Stationarity Feature Generator

This module provides comprehensive normalization and stationarity features
for making market data more learnable and interpretable.

Features implemented:
- Rolling z-score normalization
- Volatility scaling
- Regime normalization
- Cross-sectional normalization
- Stationarity transformations

Note: Individual normalization methods now inherit from BaseScaler (features_common)
for consistency with feature_engineering_roadmap.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig
from src.features_common.transforms.base_scaler import BaseScaler

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

from ..base_calculations import (

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
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

logger = logging.getLogger(__name__)


class NormalizationFeatureGenerator(FeatureGenerator):
    """Feature generator for normalization and stationarity features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="normalization_features",
            category=FeatureCategory.NORMALIZATION,
            description="Comprehensive normalization and stationarity features",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters={
                "rolling_windows": [20, 50, 100],
                "volatility_windows": [10, 20, 50],
                "regime_windows": [30, 60, 120],
                "cross_sectional_groups": ["price", "volume", "momentum"],
                "normalization_methods": ["zscore", "robust", "minmax", "quantile"],
                "regime_detection_methods": ["volatility", "momentum", "volume", "hybrid"],
                "stationarity_tests": True,
                "adaptive_normalization": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate normalization features."""
        try:
            # Generate all normalization features
            features_dict = self.generate_normalization_features(data, **kwargs)

            # Return first feature as representative for base class
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.error(f"Error generating normalization features: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_normalization_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive normalization features."""
        features = {}

        try:
            # Rolling z-score normalization
            features.update(self._generate_rolling_zscore_features(data))

            # Volatility scaling features
            features.update(self._generate_volatility_scaling_features(data))

            # Regime normalization features
            features.update(self._generate_regime_normalization_features(data))

            # Cross-sectional normalization features
            features.update(self._generate_cross_sectional_features(data))

            # Stationarity transformation features
            features.update(self._generate_stationarity_features(data))

            logger.info(f"Generated {len(features)} normalization features")
            return features

        except Exception as e:
            logger.error(f"Error in generate_normalization_features: {e}")
            return {}

    def _generate_rolling_zscore_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate enhanced rolling z-score normalization features."""
        features = {}
        rolling_windows = self.config.parameters.get("rolling_windows", [20, 50, 100])
        normalization_methods = self.config.parameters.get("normalization_methods", ["zscore", "robust", "minmax", "quantile"])

        for window in rolling_windows:
            for column in ["close", "volume", "high", "low", "open"]:
                if column in data.columns:
                    values = data[column]
                    
                    for method in normalization_methods:
                        if method == "zscore":
                            # Standard z-score
                            rolling_mean = self._vectorbt_rolling_operation(values, "mean", window)
                            rolling_std = self._vectorbt_rolling_operation(values, "std", window)
                            zscore = (values - rolling_mean) / rolling_std
                            features[f"zscore_{column}_{window}"] = zscore.fillna(0).values

                        elif method == "robust":
                            # Robust z-score using median and MAD
                            rolling_median = values.rolling(window=window).median()
                            rolling_mad = (values - rolling_median).abs().rolling(window=window).median()
                            robust_zscore = (values - rolling_median) / (1.4826 * rolling_mad)  # 1.4826 for consistency with std
                            features[f"robust_zscore_{column}_{window}"] = robust_zscore.fillna(0).values

                        elif method == "minmax":
                            # Min-max normalization
                            rolling_min = self._vectorbt_rolling_operation(values, "min", window)
                            rolling_max = self._vectorbt_rolling_operation(values, "max", window)
                            minmax_norm = (values - rolling_min) / (rolling_max - rolling_min + 1e-8)
                            features[f"minmax_{column}_{window}"] = minmax_norm.fillna(0).values

                        elif method == "quantile":
                            # Quantile normalization
                            rolling_q25 = self._calculate_rolling_quantile_vectorized(values, window, 0.25)
                            rolling_q75 = self._calculate_rolling_quantile_vectorized(values, window, 0.75)
                            quantile_norm = (values - rolling_q25) / (rolling_q75 - rolling_q25 + 1e-8)
                            features[f"quantile_{column}_{window}"] = quantile_norm.fillna(0).values

                    # Adaptive z-score with regime awareness
                    features.update(self._generate_adaptive_zscore_features(values, column, window))

        return features

    def _generate_adaptive_zscore_features(self, values: pd.Series, column: str, window: int) -> Dict[str, np.ndarray]:
        """Generate adaptive z-score features that adjust to market regimes."""
        features = {}

        # Calculate volatility regime
        returns = values.pct_change()
        vol_regime = returns.rolling(window=window//2).std()
        
        # Define regime thresholds
        low_vol_threshold = vol_regime.quantile(0.33)
        high_vol_threshold = vol_regime.quantile(0.67)

        # Low volatility regime normalization
        low_vol_mask = vol_regime <= low_vol_threshold
        if low_vol_mask.sum() > 0:
            low_vol_values = values[low_vol_mask]
            if len(low_vol_values) > window:
                low_vol_mean = self._vectorbt_rolling_operation(low_vol_values, "mean", window)
                low_vol_std = self._vectorbt_rolling_operation(low_vol_values, "std", window)
                adaptive_zscore = np.zeros(len(values))
                adaptive_zscore[low_vol_mask] = (low_vol_values - low_vol_mean) / low_vol_std
                features[f"adaptive_zscore_{column}_{window}_low_vol"] = adaptive_zscore

        # High volatility regime normalization
        high_vol_mask = vol_regime >= high_vol_threshold
        if high_vol_mask.sum() > 0:
            high_vol_values = values[high_vol_mask]
            if len(high_vol_values) > window:
                high_vol_mean = self._vectorbt_rolling_operation(high_vol_values, "mean", window)
                high_vol_std = self._vectorbt_rolling_operation(high_vol_values, "std", window)
                adaptive_zscore = np.zeros(len(values))
                adaptive_zscore[high_vol_mask] = (high_vol_values - high_vol_mean) / high_vol_std
                features[f"adaptive_zscore_{column}_{window}_high_vol"] = adaptive_zscore

        return features

    def _generate_volatility_scaling_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate enhanced volatility scaling features."""
        features = {}
        volatility_windows = self.config.parameters.get("volatility_windows", [10, 20, 50])

        for window in volatility_windows:
            # Calculate returns and volatility
            returns = data["close"].pct_change()
            rolling_vol = self._vectorbt_rolling_operation(returns, "std", window)
            
            # GARCH-like volatility estimation
            garch_vol = self._estimate_garch_volatility(returns, window)
            
            for column in ["close", "volume", "high", "low", "open"]:
                if column in data.columns:
                    if column == "close":
                        # Volatility-scaled returns
                        vol_scaled_returns = returns / rolling_vol
                        features[f"vol_scaled_returns_{window}"] = vol_scaled_returns.fillna(0).values
                        
                        # GARCH-scaled returns
                        garch_scaled_returns = returns / garch_vol
                        features[f"garch_scaled_returns_{window}"] = garch_scaled_returns.fillna(0).values
                        
                    else:
                        # Volatility-scaled price changes
                        price_changes = data[column].pct_change()
                        vol_scaled_changes = price_changes / rolling_vol
                        features[f"vol_scaled_{column}_{window}"] = vol_scaled_changes.fillna(0).values
                        
                        # GARCH-scaled changes
                        garch_scaled_changes = price_changes / garch_vol
                        features[f"garch_scaled_{column}_{window}"] = garch_scaled_changes.fillna(0).values

                    # Volatility regime scaling
                    features.update(self._generate_volatility_regime_scaling(data[column], column, window, rolling_vol))

        return features

    def _estimate_garch_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Estimate GARCH-like volatility using exponential weighting."""
        # Simple GARCH(1,1) approximation
        alpha = 0.1  # Weight for recent returns
        beta = 0.85  # Weight for previous volatility
        omega = 0.05  # Long-term variance
        
        garch_vol = pd.Series(index=returns.index, dtype=float)
        garch_vol.iloc[0] = self._vectorbt_rolling_operation(returns, "std", window).iloc[0] ** 2
        
        for i in range(1, len(returns)):
            if not pd.isna(returns.iloc[i-1]):
                garch_vol.iloc[i] = omega + alpha * (returns.iloc[i-1] ** 2) + beta * garch_vol.iloc[i-1]
            else:
                garch_vol.iloc[i] = garch_vol.iloc[i-1]
        
        return np.sqrt(garch_vol)

    def _generate_volatility_regime_scaling(self, values: pd.Series, column: str, window: int, rolling_vol: pd.Series) -> Dict[str, np.ndarray]:
        """Generate volatility regime-aware scaling features."""
        features = {}

        # Define volatility regimes
        vol_percentiles = rolling_vol.quantile([0.25, 0.75])
        low_vol_threshold = vol_percentiles.iloc[0]
        high_vol_threshold = vol_percentiles.iloc[1]

        # Low volatility regime scaling
        low_vol_mask = rolling_vol <= low_vol_threshold
        if low_vol_mask.sum() > 0:
            low_vol_values = values[low_vol_mask]
            low_vol_vol = rolling_vol[low_vol_mask]
            if len(low_vol_values) > 0:
                low_vol_scaled = low_vol_values.pct_change() / low_vol_vol
                features[f"low_vol_scaled_{column}_{window}"] = low_vol_scaled.fillna(0).values

        # High volatility regime scaling
        high_vol_mask = rolling_vol >= high_vol_threshold
        if high_vol_mask.sum() > 0:
            high_vol_values = values[high_vol_mask]
            high_vol_vol = rolling_vol[high_vol_mask]
            if len(high_vol_values) > 0:
                high_vol_scaled = high_vol_values.pct_change() / high_vol_vol
                features[f"high_vol_scaled_{column}_{window}"] = high_vol_scaled.fillna(0).values

        return features

    def _generate_regime_normalization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime-based normalization features."""
        features = {}
        regime_windows = self.config.parameters.get("regime_windows", [30, 60, 120])

        for window in regime_windows:
            # Detect regime using volatility regime detection
            returns = data["close"].pct_change()
            vol_regime = self._vectorbt_rolling_operation(returns, "std", window)

            # Define regimes based on volatility percentiles
            low_vol_threshold = vol_regime.quantile(0.3)
            high_vol_threshold = vol_regime.quantile(0.7)

            # Create regime indicators
            low_vol_regime = (vol_regime <= low_vol_threshold).astype(int)
            high_vol_regime = (vol_regime >= high_vol_threshold).astype(int)
            normal_regime = ((vol_regime > low_vol_threshold) & (vol_regime < high_vol_threshold)).astype(int)

            features[f"low_vol_regime_{window}"] = low_vol_regime.values
            features[f"high_vol_regime_{window}"] = high_vol_regime.values
            features[f"normal_regime_{window}"] = normal_regime.values

            # Regime-normalized features
            for column in ["close", "volume"]:
                if column in data.columns:
                    # Normalize within each regime
                    regime_normalized = np.zeros(len(data))
                    for regime_val in [0, 1]:  # normal and high vol regimes
                        regime_mask = (high_vol_regime == regime_val) if regime_val == 1 else (normal_regime == regime_val)
                        if regime_mask.sum() > 0:
                            regime_data = data[column][regime_mask]
                            regime_mean = regime_data.mean()
                            regime_std = regime_data.std()
                            if regime_std > 0:
                                regime_normalized[regime_mask] = (data[column][regime_mask] - regime_mean) / regime_std

                    features[f"regime_norm_{column}_{window}"] = regime_normalized

        return features

    def _generate_cross_sectional_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-sectional normalization features."""
        features = {}
        groups = self.config.parameters.get("cross_sectional_groups", ["price", "volume", "momentum"])

        # Note: This would typically require multiple assets data
        # For now, we'll create proxy cross-sectional features based on different price levels

        for group in groups:
            if group == "price":
                # Cross-sectional rank of price levels
                price_cols = ["open", "high", "low", "close"]
                available_cols = [col for col in price_cols if col in data.columns]

                if len(available_cols) > 1:
                    # Create a combined price metric
                    price_combined = data[available_cols].mean(axis=1)
                    features[f"price_cross_rank"] = price_combined.rank().values

            elif group == "volume" and "volume" in data.columns:
                # Volume cross-sectional rank (proxy using rolling quantiles)
                volume_rank = data["volume"].rank(pct=True)
                features[f"volume_cross_rank"] = volume_rank.values

            elif group == "momentum":
                # Momentum cross-sectional rank
                momentum_cols = []
                for window in [5, 10, 20]:
                    momentum = data["close"].pct_change(window)
                    momentum_cols.append(momentum)

                if momentum_cols:
                    momentum_combined = pd.concat(momentum_cols, axis=1).mean(axis=1)
                    features[f"momentum_cross_rank"] = momentum_combined.rank().values

        return features

    def _generate_stationarity_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate stationarity transformation features."""
        features = {}

        # Fractional differencing approximation using rolling windows
        for column in ["close", "volume"]:
            if column in data.columns:
                # First difference
                diff1 = data[column].diff()
                features[f"diff1_{column}"] = diff1.fillna(0).values

                # Second difference (acceleration)
                diff2 = diff1.diff()
                features[f"diff2_{column}"] = diff2.fillna(0).values

                # Rolling detrending
                for window in [20, 50]:
                    rolling_mean = data[column].rolling(window=window).mean()
                    detrended = data[column] - rolling_mean
                    features[f"detrended_{column}_{window}"] = detrended.fillna(0).values

        return featuresclass RollingZScoreGenerator(FeatureGenerator):
    """Generator for rolling z-score normalization features."""

    def __init__(self, window: int = 50, column: str = "close"):
        config = FeatureConfig(
            name=f"rolling_zscore_{column}_{window}",
            category=FeatureCategory.NORMALIZATION,
            description=f"Rolling z-score normalization of {column} over {window} periods",
            required_columns=[column],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "column": column}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.column = column

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate rolling z-score feature."""
        if self.column not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        values = data[self.column]
        rolling_mean = values.rolling(window=self.window).mean()
        rolling_std = values.rolling(window=self.window).std()

        zscore = (values - rolling_mean) / rolling_std
        return zscore.fillna(0)class VolatilityScalingGenerator(FeatureGenerator):
    """Generator for volatility scaling features."""

    def __init__(self, window: int = 20, column: str = "close"):
        config = FeatureConfig(
            name=f"volatility_scaling_{column}_{window}",
            category=FeatureCategory.NORMALIZATION,
            description=f"Volatility scaling of {column} using {window}-period rolling volatility",
            required_columns=["close", column],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "column": column}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.column = column

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility scaling feature."""
        if self.column not in data.columns or "close" not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        returns = data["close"].pct_change()
        rolling_vol = returns.rolling(window=self.window).std()

        if self.column == "close":
            scaled = returns / rolling_vol
        else:
            price_changes = data[self.column].pct_change()
            scaled = price_changes / rolling_vol

        return scaled.fillna(0)class CrossSectionalNormalizer(FeatureGenerator):
    """Generator for cross-sectional normalization features."""

    def __init__(self, group_by: str = "price", method: str = "rank"):
        config = FeatureConfig(
            name=f"cross_sectional_{group_by}_{method}",
            category=FeatureCategory.NORMALIZATION,
            description=f"Cross-sectional {method} normalization for {group_by} features",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=30,
            min_lookback=10,
            max_lookback=100,
            parameters={"group_by": group_by, "method": method}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.group_by = group_by
        self.method = method

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-sectional normalization feature."""
        # This is a simplified version - in practice would need multiple assets
        if self.group_by == "price":
            price_cols = ["open", "high", "low", "close"]
            available_cols = [col for col in price_cols if col in data.columns]

            if len(available_cols) > 1:
                price_combined = data[available_cols].mean(axis=1)
                if self.method == "rank":
                    return price_combined.rank(pct=True)
                elif self.method == "zscore":
                    return (price_combined - price_combined.mean()) / price_combined.std()

        return pd.Series(np.zeros(len(data)), index=data.index)

# ============================================================================
# BaseScaler-based Normalizers
# These classes provide consistent interface with feature_engineering_roadmap
# ============================================================================

class ZScoreNormalizer(BaseScaler):
    """
    Z-score normalizer that inherits from BaseScaler.
    
    Provides consistent interface with feature_engineering_roadmap transforms.
    Uses tprint for better UX and math_validation for robustness.
    """
    
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit mean/std and transform data with enhanced logging and validation."""
        self._log_info(f"🔧 [ZScoreNormalizer] Fitting on {len(data)} samples")
        
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            self._log_warning("⚠️  No valid data to fit, using defaults")
            self.mean = 0.0
            self.std = 1.0
        else:
            self.mean = float(clean_data.mean())
            self.std = float(clean_data.std())
            
            if self.std == 0 or np.isnan(self.std):
                self._log_warning("⚠️  Zero std detected, using 1.0")
                self.std = 1.0
        
        self.fitted = True
        self._log_success(f"✅ [ZScoreNormalizer] Fitted: mean={self.mean:.4f}, std={self.std:.4f}")
        
        transformed = self.transform(data)
        
        # Validate output
        self._check_output_validity(transformed, "transformed data")
        
        return transformed
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform data using fitted mean/std with safe division."""
        self._validate_fitted()
        
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer state is invalid")
        
        # Use safe division to prevent inf/nan
        return self._safe_divide(data - self.mean, self.std, default=0.0)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        return {
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        self.mean = state.get('mean')
        self.std = state.get('std')
        self.fitted = state.get('fitted', False)


class RobustScaler(BaseScaler):
    """
    Robust scaler using median and MAD.
    
    More robust to outliers than standard z-score normalization.
    Uses tprint for better UX and math_validation for robustness.
    """
    
    def __init__(self):
        super().__init__()
        self.median = None
        self.mad = None
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit median/MAD and transform data with enhanced logging and validation."""
        self._log_info(f"🔧 [RobustScaler] Fitting on {len(data)} samples")
        
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            self._log_warning("⚠️  No valid data to fit, using defaults")
            self.median = 0.0
            self.mad = 1.0
        else:
            self.median = float(clean_data.median())
            deviations = np.abs(clean_data - self.median)
            self.mad = float(deviations.median())
            
            if self.mad == 0 or np.isnan(self.mad):
                self._log_warning("⚠️  Zero MAD detected, using 1.0")
                self.mad = 1.0
        
        self.fitted = True
        self._log_success(f"✅ [RobustScaler] Fitted: median={self.median:.4f}, mad={self.mad:.4f}")
        
        transformed = self.transform(data)
        
        # Validate output
        self._check_output_validity(transformed, "transformed data")
        
        return transformed
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform data using fitted median/MAD with safe division."""
        self._validate_fitted()
        
        if self.median is None or self.mad is None:
            raise ValueError("Scaler state is invalid")
        
        # 1.4826 factor for consistency with standard deviation
        # Use safe division to prevent inf/nan
        return self._safe_divide(data - self.median, 1.4826 * self.mad, default=0.0)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        return {
            'median': self.median,
            'mad': self.mad,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        self.median = state.get('median')
        self.mad = state.get('mad')
        self.fitted = state.get('fitted', False)


class MinMaxScaler(BaseScaler):
    """
    Min-max scaler that scales data to [0, 1] range.
    
    Uses tprint for better UX and math_validation for robustness.
    """
    
    def __init__(self):
        super().__init__()
        self.min_val = None
        self.max_val = None
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit min/max and transform data with enhanced logging and validation."""
        self._log_info(f"🔧 [MinMaxScaler] Fitting on {len(data)} samples")
        
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            self._log_warning("⚠️  No valid data to fit, using defaults")
            self.min_val = 0.0
            self.max_val = 1.0
        else:
            self.min_val = float(clean_data.min())
            self.max_val = float(clean_data.max())
            
            if self.min_val == self.max_val:
                self._log_warning("⚠️  Min equals max, adjusting range")
                self.max_val = self.min_val + 1.0
        
        self.fitted = True
        self._log_success(f"✅ [MinMaxScaler] Fitted: min={self.min_val:.4f}, max={self.max_val:.4f}")
        
        transformed = self.transform(data)
        
        # Validate output
        self._check_output_validity(transformed, "transformed data")
        
        return transformed
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform data using fitted min/max with safe division."""
        self._validate_fitted()
        
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler state is invalid")
        
        # Use safe division to prevent inf/nan
        return self._safe_divide(data - self.min_val, self.max_val - self.min_val, default=0.0)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for persistence."""
        return {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        self.min_val = state.get('min_val')
        self.max_val = state.get('max_val')
        self.fitted = state.get('fitted', False)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
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
            return self._calculate_sma_vectorized(data, window)
        elif operation == 'std':
            return self._calculate_rolling_std_vectorized(data, window)
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return self._calculate_rolling_min_vectorized(data, window)
        elif operation == 'max':
            return self._calculate_rolling_max_vectorized(data, window)
        elif operation == 'sum':
            return self._calculate_rolling_sum_vectorized(data, window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
