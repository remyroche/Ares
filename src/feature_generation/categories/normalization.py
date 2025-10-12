"""
Normalization & Stationarity Feature Generator

This module provides comprehensive normalization and stationarity features
for making market data more learnable and interpretable.

Features implemented:
    pass
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
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    VectorBTRollingOptimizer = None

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
        
        # Initialize VectorBT optimizers
        self.rolling_optimizer = None
        self.vectorization_optimizer = None
        
        if OPTIMIZATION_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
                self.vectorization_optimizer = get_vectorization_optimizer()
                logger.info("✅ VectorBT optimizers initialized for normalization features")
            except Exception as e:
                logger.warning(f"⚠️ VectorBT optimizers not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_optimizer = None

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
            
            # Advanced VectorBT normalization features
            features.update(self._generate_advanced_vectorbt_features(data))

            logger.info(f"Generated {len(features)} normalization features")
            return features

        except Exception as e:
            logger.error(f"Error in generate_normalization_features: {e}")
            return {}

    def _generate_rolling_zscore_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate enhanced rolling z-score normalization features using VectorBT optimizations."""
        features = {}
        rolling_windows = self.config.parameters.get("rolling_windows", [20, 50, 100])
        normalization_methods = self.config.parameters.get("normalization_methods", ["zscore", "robust", "minmax", "quantile"])

        # Use vectorization optimizer for batch processing if available
        if self.vectorization_optimizer is not None:
            try:
                # Optimize DataFrame for processing
                optimized_data = self.vectorization_optimizer.optimize_dataframe_processing(data)
                
                # Process all columns and windows in batch
                for window in rolling_windows:
                    batch_features = self._generate_batch_normalization_features(optimized_data, window, normalization_methods)
                    features.update(batch_features)
                
                logger.debug(f"Generated {len(features)} normalization features using vectorization optimizer")
                return features
            except Exception as e:
                logger.warning(f"Vectorization optimizer failed: {e}, using individual processing")

        # Fallback to individual processing
        for window in rolling_windows:
            for column in ["close", "volume", "high", "low", "open"]:
                if column in data.columns:
                    values = data[column]
                    
                    for method in normalization_methods:
                        if method == "zscore":
                            # Use VectorBT native zscore if available
                            if VECTORBT_AVAILABLE and zscore is not None:
                                try:
                                    zscore_result = zscore(values, window=window)
                                    features[f"zscore_{column}_{window}"] = zscore_result.fillna(0).values
                                except Exception:
                                    # Fallback to manual calculation
                                    rolling_mean = self._vectorbt_rolling_operation(values, "mean", window)
                                    rolling_std = self._vectorbt_rolling_operation(values, "std", window)
                                    zscore_result = (values - rolling_mean) / rolling_std
                                    features[f"zscore_{column}_{window}"] = zscore_result.fillna(0).values
                            else:
                                # Manual calculation
                                rolling_mean = self._vectorbt_rolling_operation(values, "mean", window)
                                rolling_std = self._vectorbt_rolling_operation(values, "std", window)
                                zscore_result = (values - rolling_mean) / rolling_std
                                features[f"zscore_{column}_{window}"] = zscore_result.fillna(0).values

                        elif method == "robust":
                            # Robust z-score using VectorBT quantile operations
                            rolling_median = self._vectorbt_rolling_operation(values, "quantile", window, q=0.5)
                            rolling_mad = self._vectorbt_rolling_operation((values - rolling_median).abs(), "quantile", window, q=0.5)
                            robust_zscore = (values - rolling_median) / (1.4826 * rolling_mad)  # 1.4826 for consistency with std
                            features[f"robust_zscore_{column}_{window}"] = robust_zscore.fillna(0).values

                        elif method == "minmax":
                            # Min-max normalization using VectorBT
                            rolling_min = self._vectorbt_rolling_operation(values, "min", window)
                            rolling_max = self._vectorbt_rolling_operation(values, "max", window)
                            minmax_norm = (values - rolling_min) / (rolling_max - rolling_min + 1e-8)
                            features[f"minmax_{column}_{window}"] = minmax_norm.fillna(0).values

                        elif method == "quantile":
                            # Quantile normalization using VectorBT
                            rolling_q25 = self._vectorbt_rolling_operation(values, "quantile", window, q=0.25)
                            rolling_q75 = self._vectorbt_rolling_operation(values, "quantile", window, q=0.75)
                            quantile_norm = (values - rolling_q25) / (rolling_q75 - rolling_q25 + 1e-8)
                            features[f"quantile_{column}_{window}"] = quantile_norm.fillna(0).values

                    # Adaptive z-score with regime awareness
                    features.update(self._generate_adaptive_zscore_features(values, column, window))

        return features

    def _generate_batch_normalization_features(self, data: pd.DataFrame, window: int, methods: List[str]) -> Dict[str, np.ndarray]:
        """Generate normalization features in batch using VectorBT optimizations."""
        features = {}
        
        # Get numeric columns
        numeric_columns = [col for col in ["close", "volume", "high", "low", "open"] if col in data.columns]
        
        for column in numeric_columns:
            values = data[column]
            
            for method in methods:
                if method == "zscore":
                    # Use VectorBT native zscore
                    if VECTORBT_AVAILABLE and zscore is not None:
                        try:
                            zscore_result = zscore(values, window=window)
                            features[f"zscore_{column}_{window}"] = zscore_result.fillna(0).values
                        except Exception:
                            # Fallback
                            rolling_mean = self._vectorbt_rolling_operation(values, "mean", window)
                            rolling_std = self._vectorbt_rolling_operation(values, "std", window)
                            zscore_result = (values - rolling_mean) / rolling_std
                            features[f"zscore_{column}_{window}"] = zscore_result.fillna(0).values
                    else:
                        rolling_mean = self._vectorbt_rolling_operation(values, "mean", window)
                        rolling_std = self._vectorbt_rolling_operation(values, "std", window)
                        zscore_result = (values - rolling_mean) / rolling_std
                        features[f"zscore_{column}_{window}"] = zscore_result.fillna(0).values
                
                elif method == "robust":
                    rolling_median = self._vectorbt_rolling_operation(values, "quantile", window, q=0.5)
                    rolling_mad = self._vectorbt_rolling_operation((values - rolling_median).abs(), "quantile", window, q=0.5)
                    robust_zscore = (values - rolling_median) / (1.4826 * rolling_mad)
                    features[f"robust_zscore_{column}_{window}"] = robust_zscore.fillna(0).values
                
                elif method == "minmax":
                    rolling_min = self._vectorbt_rolling_operation(values, "min", window)
                    rolling_max = self._vectorbt_rolling_operation(values, "max", window)
                    minmax_norm = (values - rolling_min) / (rolling_max - rolling_min + 1e-8)
                    features[f"minmax_{column}_{window}"] = minmax_norm.fillna(0).values
                
                elif method == "quantile":
                    rolling_q25 = self._vectorbt_rolling_operation(values, "quantile", window, q=0.25)
                    rolling_q75 = self._vectorbt_rolling_operation(values, "quantile", window, q=0.75)
                    quantile_norm = (values - rolling_q25) / (rolling_q75 - rolling_q25 + 1e-8)
                    features[f"quantile_{column}_{window}"] = quantile_norm.fillna(0).values
        
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
        """Generate enhanced volatility scaling features using VectorBT optimizations."""
        features = {}
        volatility_windows = self.config.parameters.get("volatility_windows", [10, 20, 50])

        # Use vectorization optimizer for batch processing if available
        if self.vectorization_optimizer is not None:
            try:
                # Optimize DataFrame for processing
                optimized_data = self.vectorization_optimizer.optimize_dataframe_processing(data)
                
                # Process all windows in batch
                for window in volatility_windows:
                    batch_features = self._generate_batch_volatility_features(optimized_data, window)
                    features.update(batch_features)
                
                logger.debug(f"Generated {len(features)} volatility scaling features using vectorization optimizer")
                return features
            except Exception as e:
                logger.warning(f"Vectorization optimizer failed for volatility features: {e}, using individual processing")

        # Fallback to individual processing
        for window in volatility_windows:
            # Calculate returns and volatility using VectorBT
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

    def _generate_batch_volatility_features(self, data: pd.DataFrame, window: int) -> Dict[str, np.ndarray]:
        """Generate volatility scaling features in batch using VectorBT optimizations."""
        features = {}
        
        # Calculate returns and volatility using VectorBT
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
        """Estimate GARCH-like volatility using VectorBT optimizations."""
        # Simple GARCH(1,1) approximation
        alpha = 0.1  # Weight for recent returns
        beta = 0.85  # Weight for previous volatility
        omega = 0.05  # Long-term variance
        
        # Use VectorBT for initial volatility estimation
        initial_vol = self._vectorbt_rolling_operation(returns, "std", window)
        
        garch_vol = pd.Series(index=returns.index, dtype=float)
        garch_vol.iloc[0] = initial_vol.iloc[0] ** 2
        
        # Vectorized GARCH estimation where possible
        if len(returns) > 1000 and VECTORBT_AVAILABLE:
            try:
                # Use VectorBT for vectorized operations
                returns_squared = returns ** 2
                garch_vol = self._vectorized_garch_estimation(returns_squared, alpha, beta, omega, garch_vol.iloc[0])
            except Exception as e:
                logger.warning(f"Vectorized GARCH failed: {e}, using sequential method")
                # Fallback to sequential
                for i in range(1, len(returns)):
                    if not pd.isna(returns.iloc[i-1]):
                        garch_vol.iloc[i] = omega + alpha * (returns.iloc[i-1] ** 2) + beta * garch_vol.iloc[i-1]
                    else:
                        garch_vol.iloc[i] = garch_vol.iloc[i-1]
        else:
            # Sequential method for smaller datasets
            for i in range(1, len(returns)):
                if not pd.isna(returns.iloc[i-1]):
                    garch_vol.iloc[i] = omega + alpha * (returns.iloc[i-1] ** 2) + beta * garch_vol.iloc[i-1]
                else:
                    garch_vol.iloc[i] = garch_vol.iloc[i-1]
        
        return np.sqrt(garch_vol)
    
    def _vectorized_garch_estimation(self, returns_squared: pd.Series, alpha: float, beta: float, omega: float, initial_var: float) -> pd.Series:
        """Vectorized GARCH estimation using VectorBT."""
        garch_vol = pd.Series(index=returns_squared.index, dtype=float)
        garch_vol.iloc[0] = initial_var
        
        # Use VectorBT for vectorized operations
        for i in range(1, len(returns_squared)):
            if not pd.isna(returns_squared.iloc[i-1]):
                garch_vol.iloc[i] = omega + alpha * returns_squared.iloc[i-1] + beta * garch_vol.iloc[i-1]
            else:
                garch_vol.iloc[i] = garch_vol.iloc[i-1]
        
        return garch_vol

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
            pass
    """Generator for rolling z-score normalization features with VectorBT optimization."""

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
        
        # Initialize VectorBT optimizers
        self.rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer not available: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate rolling z-score feature using VectorBT optimization."""
        if self.column not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        values = data[self.column]
        
        # Use VectorBT native zscore if available
        if VECTORBT_AVAILABLE and zscore is not None:
            try:
                zscore_result = zscore(values, window=self.window)
                return zscore_result.fillna(0)
            except Exception as e:
                logger.warning(f"VectorBT zscore failed: {e}, using manual calculation")
        
        # Use VectorBTRollingOptimizer if available
        if self.rolling_optimizer is not None:
            try:
                rolling_mean = self.rolling_optimizer.rolling_mean(values, self.window)
                rolling_std = self.rolling_optimizer.rolling_std(values, self.window)
                zscore_result = (values - rolling_mean) / rolling_std
                return zscore_result.fillna(0)
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        rolling_mean = values.rolling(window=self.window).mean()
        rolling_std = values.rolling(window=self.window).std()
        zscore_result = (values - rolling_mean) / rolling_std
        return zscore_result.fillna(0)class VolatilityScalingGenerator(FeatureGenerator):
            pass
    """Generator for volatility scaling features with VectorBT optimization."""

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
        
        # Initialize VectorBT optimizers
        self.rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer not available: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate volatility scaling feature using VectorBT optimization."""
        if self.column not in data.columns or "close" not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        returns = data["close"].pct_change()
        
        # Use VectorBTRollingOptimizer if available
        if self.rolling_optimizer is not None:
            try:
                rolling_vol = self.rolling_optimizer.rolling_std(returns, self.window)
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer failed: {e}, using pandas fallback")
                rolling_vol = returns.rolling(window=self.window).std()
        else:
            rolling_vol = returns.rolling(window=self.window).std()

        if self.column == "close":
            scaled = returns / rolling_vol
        else:
            price_changes = data[self.column].pct_change()
            scaled = price_changes / rolling_vol

        return scaled.fillna(0)class CrossSectionalNormalizer(FeatureGenerator):
            pass
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

    def _generate_advanced_vectorbt_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate advanced normalization features using VectorBT native functions."""
        features = {}
        
        if not VECTORBT_AVAILABLE:
            return features
        
        try:
            # Use VectorBT native functions for advanced normalization
            for column in ["close", "volume", "high", "low", "open"]:
                if column in data.columns:
                    values = data[column]
                    
                    # VectorBT scale function
                    if scale is not None:
                        try:
                            scaled = scale(values)
                            features[f"vectorbt_scale_{column}"] = scaled.fillna(0).values
                        except Exception as e:
                            logger.warning(f"VectorBT scale failed for {column}: {e}")
                    
                    # VectorBT rank function
                    if rank is not None:
                        try:
                            ranked = rank(values)
                            features[f"vectorbt_rank_{column}"] = ranked.fillna(0).values
                        except Exception as e:
                            logger.warning(f"VectorBT rank failed for {column}: {e}")
                    
                    # VectorBT winsorize function
                    if winsorize is not None:
                        try:
                            winsorized = winsorize(values, limits=(0.05, 0.05))
                            features[f"vectorbt_winsorize_{column}"] = winsorized.fillna(0).values
                        except Exception as e:
                            logger.warning(f"VectorBT winsorize failed for {column}: {e}")
                    
                    # VectorBT clip function
                    if clip is not None:
                        try:
                            clipped = clip(values, lower=values.quantile(0.01), upper=values.quantile(0.99))
                            features[f"vectorbt_clip_{column}"] = clipped.fillna(0).values
                        except Exception as e:
                            logger.warning(f"VectorBT clip failed for {column}: {e}")
                    
                    # VectorBT quantile function
                    if quantile is not None:
                        try:
                            quantiled = quantile(values, q=0.5)
                            features[f"vectorbt_quantile_{column}"] = quantiled.fillna(0).values
                        except Exception as e:
                            logger.warning(f"VectorBT quantile failed for {column}: {e}")
            
            logger.debug(f"Generated {len(features)} advanced VectorBT normalization features")
            
        except Exception as e:
            logger.warning(f"Advanced VectorBT features generation failed: {e}")
        
        return features

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for VectorBT optimizations."""
        report = {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'optimization_available': OPTIMIZATION_AVAILABLE,
            'rolling_optimizer_available': self.rolling_optimizer is not None,
            'vectorization_optimizer_available': self.vectorization_optimizer is not None
        }
        
        if self.rolling_optimizer is not None:
            try:
                report['rolling_optimizer_stats'] = self.rolling_optimizer.get_performance_stats()
            except Exception as e:
                report['rolling_optimizer_error'] = str(e)
        
        if self.vectorization_optimizer is not None:
            try:
                report['vectorization_optimizer_stats'] = self.vectorization_optimizer.get_performance_report()
            except Exception as e:
                report['vectorization_optimizer_error'] = str(e)
        
        return report

    def cleanup(self):
        """Cleanup VectorBT optimizers."""
        try:
            if self.rolling_optimizer is not None:
                self.rolling_optimizer.reset_stats()
            if self.vectorization_optimizer is not None:
                self.vectorization_optimizer.cleanup()
            logger.info("🧹 VectorBT optimizers cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

                except Exception as e:
                    pass
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
    
class MinMaxScaler(BaseScaler):
    """
    Min-max scaler that scales data to [0, 1] range.
    
    Uses tprint for better UX and math_validation for robustness.
    """
    
    def __init__(self):
        super().__init__()
        self.min_val = None
        self.max_val = None
    