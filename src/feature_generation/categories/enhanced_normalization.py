"""
Enhanced Normalization & Stationarity Feature Generator

This module provides comprehensive normalization and stationarity features
for making market data more learnable and interpretable, with advanced
techniques for regime-aware normalization and cross-sectional analysis.

Features implemented:
- Advanced rolling z-score normalization with multiple windows
- Volatility scaling with regime-aware adjustments
- Regime normalization with structural bias removal
- Cross-sectional normalization across assets/timeframes
- Stationarity transformations with fractional differencing
- Adaptive normalization based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureConfig,
    FeatureCategory,
    VectorizedFeatureGenerator
)

logger = logging.getLogger(__name__)


class EnhancedNormalizationFeatureGenerator(VectorizedFeatureGenerator):
    """Enhanced feature generator for normalization and stationarity features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="enhanced_normalization_features",
            category=FeatureCategory.NORMALIZATION,
            description="Enhanced normalization and stationarity features with regime awareness",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=100,
            min_lookback=50,
            max_lookback=500,
            parameters={
                "rolling_windows": [10, 20, 50, 100, 200],
                "volatility_windows": [5, 10, 20, 50],
                "regime_windows": [30, 60, 120, 240],
                "cross_sectional_groups": ["price", "volume", "momentum", "volatility"],
                "normalization_methods": ["zscore", "robust", "minmax", "quantile"],
                "regime_detection_methods": ["volatility", "momentum", "volume", "hybrid"],
                "stationarity_tests": True,
                "adaptive_normalization": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced normalization features."""
        try:
            # Generate all enhanced normalization features
            features_dict = self.generate_enhanced_normalization_features(data, **kwargs)

            # Return first feature as representative for base class
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.error(f"Error generating enhanced normalization features: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_enhanced_normalization_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive enhanced normalization features."""
        features = {}

        try:
            # Advanced rolling z-score normalization
            features.update(self._generate_advanced_rolling_zscore_features(data))

            # Enhanced volatility scaling features
            features.update(self._generate_enhanced_volatility_scaling_features(data))

            # Regime-aware normalization features
            features.update(self._generate_regime_aware_normalization_features(data))

            # Cross-sectional normalization features
            features.update(self._generate_enhanced_cross_sectional_features(data))

            # Stationarity transformation features
            features.update(self._generate_enhanced_stationarity_features(data))

            # Adaptive normalization features
            features.update(self._generate_adaptive_normalization_features(data))

            # Multi-scale normalization features
            features.update(self._generate_multi_scale_normalization_features(data))

            logger.info(f"Generated {len(features)} enhanced normalization features")
            return features

        except Exception as e:
            logger.error(f"Error in generate_enhanced_normalization_features: {e}")
            return {}

    def _generate_advanced_rolling_zscore_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate advanced rolling z-score normalization features."""
        features = {}
        rolling_windows = self.config.parameters.get("rolling_windows", [10, 20, 50, 100, 200])
        normalization_methods = self.config.parameters.get("normalization_methods", ["zscore", "robust", "minmax"])

        for window in rolling_windows:
            for column in ["close", "volume", "high", "low", "open"]:
                if column in data.columns:
                    values = data[column]
                    
                    for method in normalization_methods:
                        if method == "zscore":
                            # Standard z-score
                            rolling_mean = values.rolling(window=window).mean()
                            rolling_std = values.rolling(window=window).std()
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
                            rolling_min = values.rolling(window=window).min()
                            rolling_max = values.rolling(window=window).max()
                            minmax_norm = (values - rolling_min) / (rolling_max - rolling_min + 1e-8)
                            features[f"minmax_{column}_{window}"] = minmax_norm.fillna(0).values

                        elif method == "quantile":
                            # Quantile normalization
                            rolling_q25 = values.rolling(window=window).quantile(0.25)
                            rolling_q75 = values.rolling(window=window).quantile(0.75)
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
                low_vol_mean = low_vol_values.rolling(window=window).mean()
                low_vol_std = low_vol_values.rolling(window=window).std()
                adaptive_zscore = np.zeros(len(values))
                adaptive_zscore[low_vol_mask] = (low_vol_values - low_vol_mean) / low_vol_std
                features[f"adaptive_zscore_{column}_{window}_low_vol"] = adaptive_zscore

        # High volatility regime normalization
        high_vol_mask = vol_regime >= high_vol_threshold
        if high_vol_mask.sum() > 0:
            high_vol_values = values[high_vol_mask]
            if len(high_vol_values) > window:
                high_vol_mean = high_vol_values.rolling(window=window).mean()
                high_vol_std = high_vol_values.rolling(window=window).std()
                adaptive_zscore = np.zeros(len(values))
                adaptive_zscore[high_vol_mask] = (high_vol_values - high_vol_mean) / high_vol_std
                features[f"adaptive_zscore_{column}_{window}_high_vol"] = adaptive_zscore

        return features

    def _generate_enhanced_volatility_scaling_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate enhanced volatility scaling features."""
        features = {}
        volatility_windows = self.config.parameters.get("volatility_windows", [5, 10, 20, 50])

        for window in volatility_windows:
            # Calculate returns and volatility
            returns = data["close"].pct_change()
            rolling_vol = returns.rolling(window=window).std()
            
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
        garch_vol.iloc[0] = returns.rolling(window=window).std().iloc[0] ** 2
        
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

    def _generate_regime_aware_normalization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime-aware normalization features."""
        features = {}
        regime_windows = self.config.parameters.get("regime_windows", [30, 60, 120, 240])
        regime_methods = self.config.parameters.get("regime_detection_methods", ["volatility", "momentum", "volume", "hybrid"])

        for window in regime_windows:
            for method in regime_methods:
                regime_indicators = self._detect_regimes(data, window, method)
                
                for column in ["close", "volume"]:
                    if column in data.columns:
                        # Regime-normalized features
                        regime_normalized = self._normalize_within_regimes(data[column], regime_indicators, window)
                        features[f"regime_norm_{column}_{window}_{method}"] = regime_normalized

                        # Regime-relative features
                        regime_relative = self._calculate_regime_relative_features(data[column], regime_indicators, window)
                        features[f"regime_relative_{column}_{window}_{method}"] = regime_relative

        return features

    def _detect_regimes(self, data: pd.DataFrame, window: int, method: str) -> Dict[str, pd.Series]:
        """Detect market regimes using various methods."""
        regimes = {}

        if method == "volatility":
            returns = data["close"].pct_change()
            vol = returns.rolling(window=window).std()
            regimes["low_vol"] = (vol <= vol.quantile(0.33)).astype(int)
            regimes["high_vol"] = (vol >= vol.quantile(0.67)).astype(int)
            regimes["normal_vol"] = ((vol > vol.quantile(0.33)) & (vol < vol.quantile(0.67))).astype(int)

        elif method == "momentum":
            momentum = data["close"].pct_change(window)
            regimes["uptrend"] = (momentum > momentum.quantile(0.67)).astype(int)
            regimes["downtrend"] = (momentum < momentum.quantile(0.33)).astype(int)
            regimes["sideways"] = ((momentum >= momentum.quantile(0.33)) & (momentum <= momentum.quantile(0.67))).astype(int)

        elif method == "volume":
            if "volume" in data.columns:
                vol_ratio = data["volume"] / data["volume"].rolling(window=window).mean()
                regimes["high_volume"] = (vol_ratio >= vol_ratio.quantile(0.67)).astype(int)
                regimes["low_volume"] = (vol_ratio <= vol_ratio.quantile(0.33)).astype(int)
                regimes["normal_volume"] = ((vol_ratio > vol_ratio.quantile(0.33)) & (vol_ratio < vol_ratio.quantile(0.67))).astype(int)

        elif method == "hybrid":
            # Combine volatility and momentum
            returns = data["close"].pct_change()
            vol = returns.rolling(window=window).std()
            momentum = data["close"].pct_change(window)
            
            # High volatility + high momentum = trending
            regimes["trending"] = ((vol >= vol.quantile(0.5)) & (momentum.abs() >= momentum.abs().quantile(0.5))).astype(int)
            # Low volatility + low momentum = ranging
            regimes["ranging"] = ((vol <= vol.quantile(0.5)) & (momentum.abs() <= momentum.abs().quantile(0.5))).astype(int)
            # Other combinations
            regimes["mixed"] = (1 - regimes["trending"] - regimes["ranging"]).astype(int)

        return regimes

    def _normalize_within_regimes(self, values: pd.Series, regime_indicators: Dict[str, pd.Series], window: int) -> np.ndarray:
        """Normalize values within each regime."""
        normalized = np.zeros(len(values))
        
        for regime_name, regime_mask in regime_indicators.items():
            if regime_mask.sum() > 0:
                regime_values = values[regime_mask]
                if len(regime_values) > window:
                    regime_mean = regime_values.rolling(window=window).mean()
                    regime_std = regime_values.rolling(window=window).std()
                    regime_normalized = (regime_values - regime_mean) / regime_std
                    normalized[regime_mask] = regime_normalized.fillna(0).values

        return normalized

    def _calculate_regime_relative_features(self, values: pd.Series, regime_indicators: Dict[str, pd.Series], window: int) -> np.ndarray:
        """Calculate regime-relative features."""
        regime_relative = np.zeros(len(values))
        
        for regime_name, regime_mask in regime_indicators.items():
            if regime_mask.sum() > 0:
                regime_values = values[regime_mask]
                if len(regime_values) > 0:
                    # Calculate relative position within regime
                    regime_percentile = regime_values.rank(pct=True)
                    regime_relative[regime_mask] = regime_percentile.values

        return regime_relative

    def _generate_enhanced_cross_sectional_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate enhanced cross-sectional normalization features."""
        features = {}
        groups = self.config.parameters.get("cross_sectional_groups", ["price", "volume", "momentum", "volatility"])

        for group in groups:
            if group == "price":
                # Cross-sectional rank of price levels
                price_cols = ["open", "high", "low", "close"]
                available_cols = [col for col in price_cols if col in data.columns]

                if len(available_cols) > 1:
                    # Create combined price metrics
                    price_combined = data[available_cols].mean(axis=1)
                    features[f"price_cross_rank"] = price_combined.rank(pct=True).values
                    features[f"price_cross_zscore"] = (price_combined - price_combined.mean()) / price_combined.std()

            elif group == "volume" and "volume" in data.columns:
                # Volume cross-sectional features
                volume_rank = data["volume"].rank(pct=True)
                features[f"volume_cross_rank"] = volume_rank.values
                
                # Volume momentum cross-sectional
                volume_momentum = data["volume"].pct_change(5)
                features[f"volume_momentum_cross_rank"] = volume_momentum.rank(pct=True).values

            elif group == "momentum":
                # Momentum cross-sectional features
                momentum_features = []
                for window in [5, 10, 20]:
                    momentum = data["close"].pct_change(window)
                    momentum_features.append(momentum)

                if momentum_features:
                    momentum_combined = pd.concat(momentum_features, axis=1).mean(axis=1)
                    features[f"momentum_cross_rank"] = momentum_combined.rank(pct=True).values
                    features[f"momentum_cross_zscore"] = (momentum_combined - momentum_combined.mean()) / momentum_combined.std()

            elif group == "volatility":
                # Volatility cross-sectional features
                returns = data["close"].pct_change()
                vol_features = []
                for window in [5, 10, 20]:
                    volatility = returns.rolling(window=window).std()
                    vol_features.append(volatility)

                if vol_features:
                    vol_combined = pd.concat(vol_features, axis=1).mean(axis=1)
                    features[f"volatility_cross_rank"] = vol_combined.rank(pct=True).values
                    features[f"volatility_cross_zscore"] = (vol_combined - vol_combined.mean()) / vol_combined.std()

        return features

    def _generate_enhanced_stationarity_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate enhanced stationarity transformation features."""
        features = {}

        for column in ["close", "volume"]:
            if column in data.columns:
                values = data[column]
                
                # First difference
                diff1 = values.diff()
                features[f"diff1_{column}"] = diff1.fillna(0).values

                # Second difference (acceleration)
                diff2 = diff1.diff()
                features[f"diff2_{column}"] = diff2.fillna(0).values

                # Fractional differencing approximation
                features.update(self._generate_fractional_differencing_features(values, column))

                # Rolling detrending
                for window in [20, 50, 100]:
                    rolling_mean = values.rolling(window=window).mean()
                    detrended = values - rolling_mean
                    features[f"detrended_{column}_{window}"] = detrended.fillna(0).values

                # Log transformation
                if (values > 0).all():
                    log_values = np.log(values)
                    features[f"log_{column}"] = log_values.values

                # Box-Cox transformation
                features.update(self._generate_box_cox_features(values, column))

        return features

    def _generate_fractional_differencing_features(self, values: pd.Series, column: str) -> Dict[str, np.ndarray]:
        """Generate fractional differencing features."""
        features = {}

        # Fractional differencing approximation using rolling windows
        for d in [0.1, 0.3, 0.5, 0.7]:
            # Simple fractional differencing approximation
            frac_diff = values.copy()
            for i in range(1, len(values)):
                frac_diff.iloc[i] = values.iloc[i] - d * values.iloc[i-1]
            
            features[f"frac_diff_{column}_{d}"] = frac_diff.fillna(0).values

        return features

    def _generate_box_cox_features(self, values: pd.Series, column: str) -> Dict[str, np.ndarray]:
        """Generate Box-Cox transformation features."""
        features = {}

        try:
            # Box-Cox transformation
            if (values > 0).all():
                box_cox_values, lambda_param = stats.boxcox(values)
                features[f"box_cox_{column}"] = box_cox_values
                features[f"box_cox_lambda_{column}"] = np.full(len(values), lambda_param)

        except Exception as e:
            logger.warning(f"Box-Cox transformation failed for {column}: {e}")

        return features

    def _generate_adaptive_normalization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate adaptive normalization features that adjust to market conditions."""
        features = {}

        # Market stress indicator
        returns = data["close"].pct_change()
        stress_indicator = self._calculate_market_stress(returns)

        for column in ["close", "volume"]:
            if column in data.columns:
                values = data[column]
                
                # Adaptive window based on market stress
                adaptive_window = self._calculate_adaptive_window(stress_indicator)
                
                # Adaptive normalization
                adaptive_mean = values.rolling(window=adaptive_window).mean()
                adaptive_std = values.rolling(window=adaptive_window).std()
                adaptive_norm = (values - adaptive_mean) / adaptive_std
                features[f"adaptive_norm_{column}"] = adaptive_norm.fillna(0).values

                # Stress-adjusted normalization
                stress_adjusted = adaptive_norm * (1 + stress_indicator)
                features[f"stress_adjusted_{column}"] = stress_adjusted.fillna(0).values

        return features

    def _calculate_market_stress(self, returns: pd.Series) -> pd.Series:
        """Calculate market stress indicator."""
        # VIX-like indicator based on rolling volatility
        vol = returns.rolling(window=20).std()
        vol_percentile = vol.rolling(window=100).rank(pct=True)
        
        # Stress indicator: 0 (low stress) to 1 (high stress)
        stress = vol_percentile.fillna(0.5)
        
        return stress

    def _calculate_adaptive_window(self, stress_indicator: pd.Series) -> pd.Series:
        """Calculate adaptive window size based on market stress."""
        # Low stress: longer window (more stable)
        # High stress: shorter window (more responsive)
        min_window = 10
        max_window = 100
        
        adaptive_window = min_window + (1 - stress_indicator) * (max_window - min_window)
        return adaptive_window.astype(int)

    def _generate_multi_scale_normalization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate multi-scale normalization features."""
        features = {}

        for column in ["close", "volume"]:
            if column in data.columns:
                values = data[column]
                
                # Multi-scale z-scores
                scales = [5, 10, 20, 50, 100]
                for scale in scales:
                    if len(values) >= scale:
                        rolling_mean = values.rolling(window=scale).mean()
                        rolling_std = values.rolling(window=scale).std()
                        zscore = (values - rolling_mean) / rolling_std
                        features[f"multiscale_zscore_{column}_{scale}"] = zscore.fillna(0).values

                # Scale-relative features
                if len(scales) >= 2:
                    short_scale = scales[0]
                    long_scale = scales[-1]
                    
                    short_mean = values.rolling(window=short_scale).mean()
                    long_mean = values.rolling(window=long_scale).mean()
                    
                    scale_ratio = short_mean / (long_mean + 1e-8)
                    features[f"scale_ratio_{column}_{short_scale}_{long_scale}"] = scale_ratio.fillna(1).values

        return features


# Individual enhanced normalization generators

class AdvancedRollingZScoreGenerator(FeatureGenerator):
    """Generator for advanced rolling z-score normalization features."""

    def __init__(self, window: int = 50, column: str = "close", method: str = "zscore"):
        config = FeatureConfig(
            name=f"advanced_rolling_zscore_{column}_{window}_{method}",
            category=FeatureCategory.NORMALIZATION,
            description=f"Advanced rolling {method} normalization of {column} over {window} periods",
            required_columns=[column],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "column": column, "method": method}
        )
        super().__init__(config)
        self.window = window
        self.column = column
        self.method = method

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate advanced rolling z-score feature."""
        if self.column not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        values = data[self.column]
        
        if self.method == "zscore":
            rolling_mean = values.rolling(window=self.window).mean()
            rolling_std = values.rolling(window=self.window).std()
            result = (values - rolling_mean) / rolling_std
        elif self.method == "robust":
            rolling_median = values.rolling(window=self.window).median()
            rolling_mad = (values - rolling_median).abs().rolling(window=self.window).median()
            result = (values - rolling_median) / (1.4826 * rolling_mad)
        elif self.method == "minmax":
            rolling_min = values.rolling(window=self.window).min()
            rolling_max = values.rolling(window=self.window).max()
            result = (values - rolling_min) / (rolling_max - rolling_min + 1e-8)
        else:
            result = pd.Series(np.zeros(len(data)), index=data.index)

        return result.fillna(0)


class RegimeAwareNormalizer(FeatureGenerator):
    """Generator for regime-aware normalization features."""

    def __init__(self, window: int = 60, column: str = "close", regime_method: str = "volatility"):
        config = FeatureConfig(
            name=f"regime_aware_norm_{column}_{window}_{regime_method}",
            category=FeatureCategory.NORMALIZATION,
            description=f"Regime-aware normalization of {column} over {window} periods using {regime_method}",
            required_columns=[column],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "column": column, "regime_method": regime_method}
        )
        super().__init__(config)
        self.window = window
        self.column = column
        self.regime_method = regime_method

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime-aware normalization feature."""
        if self.column not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        values = data[self.column]
        
        # Detect regimes
        if self.regime_method == "volatility":
            returns = data["close"].pct_change()
            vol = returns.rolling(window=self.window//2).std()
            regime_mask = vol >= vol.quantile(0.5)
        elif self.regime_method == "momentum":
            momentum = values.pct_change(self.window//2)
            regime_mask = momentum >= momentum.quantile(0.5)
        else:
            regime_mask = pd.Series([False] * len(values), index=values.index)

        # Normalize within regimes
        result = np.zeros(len(values))
        
        # High regime
        high_mask = regime_mask
        if high_mask.sum() > 0:
            high_values = values[high_mask]
            if len(high_values) > self.window//2:
                high_mean = high_values.rolling(window=self.window//2).mean()
                high_std = high_values.rolling(window=self.window//2).std()
                result[high_mask] = ((high_values - high_mean) / high_std).fillna(0).values

        # Low regime
        low_mask = ~regime_mask
        if low_mask.sum() > 0:
            low_values = values[low_mask]
            if len(low_values) > self.window//2:
                low_mean = low_values.rolling(window=self.window//2).mean()
                low_std = low_values.rolling(window=self.window//2).std()
                result[low_mask] = ((low_values - low_mean) / low_std).fillna(0).values

        return pd.Series(result, index=data.index)


def create_enhanced_normalization_generators() -> List[FeatureGenerator]:
    """Create all enhanced normalization feature generators."""
    generators = []

    # Main enhanced normalization generator
    generators.append(EnhancedNormalizationFeatureGenerator())

    # Individual generators
    for window in [20, 50, 100]:
        for column in ["close", "volume"]:
            for method in ["zscore", "robust", "minmax"]:
                generators.append(AdvancedRollingZScoreGenerator(window=window, column=column, method=method))

    # Regime-aware generators
    for window in [30, 60, 120]:
        for column in ["close", "volume"]:
            for regime_method in ["volatility", "momentum"]:
                generators.append(RegimeAwareNormalizer(window=window, column=column, regime_method=regime_method))

    return generators


def create_default_enhanced_normalization_generators() -> List[FeatureGenerator]:
    """Create default set of enhanced normalization generators."""
    return create_enhanced_normalization_generators()


# Export all generators
__all__ = [
    'EnhancedNormalizationFeatureGenerator',
    'AdvancedRollingZScoreGenerator',
    'RegimeAwareNormalizer',
    'create_enhanced_normalization_generators',
    'create_default_enhanced_normalization_generators'
]