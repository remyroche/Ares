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
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureConfig,
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

logger = logging.getLogger(__name__)


class NormalizationFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for normalization and stationarity features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)

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
                "cross_sectional_groups": ["price", "volume", "momentum"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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
        """Generate rolling z-score normalization features."""
        features = {}
        rolling_windows = self.config.parameters.get("rolling_windows", [20, 50, 100])

        for window in rolling_windows:
            for column in ["close", "volume", "high", "low"]:
                if column in data.columns:
                    # Rolling mean and std
                    rolling_mean = data[column].rolling(window=window).mean()
                    rolling_std = data[column].rolling(window=window).std()

                    # Z-score normalization
                    zscore = (data[column] - rolling_mean) / rolling_std
                    features[f"zscore_{column}_{window}"] = zscore.fillna(0).values

                    # Robust z-score (using median and MAD)
                    rolling_median = data[column].rolling(window=window).median()
                    rolling_mad = (data[column] - rolling_median).abs().rolling(window=window).median()
                    robust_zscore = (data[column] - rolling_median) / rolling_mad
                    features[f"robust_zscore_{column}_{window}"] = robust_zscore.fillna(0).values

        return features

    def _generate_volatility_scaling_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility scaling features."""
        features = {}
        volatility_windows = self.config.parameters.get("volatility_windows", [10, 20, 50])

        for window in volatility_windows:
            # Calculate rolling volatility
            returns = data["close"].pct_change()
            rolling_vol = returns.rolling(window=window).std()

            for column in ["close", "volume", "high", "low"]:
                if column in data.columns:
                    # Volatility-scaled returns
                    if column == "close":
                        scaled_returns = returns / rolling_vol
                        features[f"vol_scaled_returns_{window}"] = scaled_returns.fillna(0).values
                    else:
                        # Volatility-scaled price changes
                        price_changes = data[column].pct_change()
                        scaled_changes = price_changes / rolling_vol
                        features[f"vol_scaled_{column}_{window}"] = scaled_changes.fillna(0).values

        return features

    def _generate_regime_normalization_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime-based normalization features."""
        features = {}
        regime_windows = self.config.parameters.get("regime_windows", [30, 60, 120])

        for window in regime_windows:
            # Detect regime using volatility regime detection
            returns = data["close"].pct_change()
            vol_regime = returns.rolling(window=window).std()

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

        return features


class RollingZScoreGenerator(FeatureGenerator):
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
        super().__init__(config)
        self.window = window
        self.column = column

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate rolling z-score feature."""
        if self.column not in data.columns:
            return pd.Series(np.zeros(len(data)), index=data.index)

        values = data[self.column]
        rolling_mean = values.rolling(window=self.window).mean()
        rolling_std = values.rolling(window=self.window).std()

        zscore = (values - rolling_mean) / rolling_std
        return zscore.fillna(0)


class VolatilityScalingGenerator(FeatureGenerator):
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
        super().__init__(config)
        self.window = window
        self.column = column

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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

        return scaled.fillna(0)


class CrossSectionalNormalizer(FeatureGenerator):
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
        super().__init__(config)
        self.group_by = group_by
        self.method = method

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
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