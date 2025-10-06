"""
Enhanced Interaction & Composite Feature Generators

This module provides advanced interaction and composite features that capture
complex relationships between different market indicators and create
regime-dependent features for better model performance.

Features implemented:
- Pairwise interactions between strong features
- Regime-dependent features (momentum strength only in trending regimes)
- Structural ratios that encode market context
- Cointegration residuals for pairs trading
- Polynomial and non-linear interactions
- Cross-asset interaction features
- Microstructure interaction features
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureConfig,
    FeatureCategory,
    VectorizedFeatureGenerator
)

logger = logging.getLogger(__name__)


class EnhancedInteractionFeatureGenerator(VectorizedFeatureGenerator):
    """Enhanced feature generator for interaction and composite features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="enhanced_interaction_features",
            category=FeatureCategory.INTERACTION,
            description="Enhanced interaction and composite features with regime awareness",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters={
                "interaction_types": ["pairwise", "regime_dependent", "structural_ratios", "cointegration"],
                "regime_detection_methods": ["volatility", "momentum", "volume", "hybrid"],
                "polynomial_degrees": [2, 3],
                "cointegration_pairs": ["BTCUSDT_ETHUSDT", "SPY_QQQ"],
                "microstructure_features": True,
                "cross_asset_features": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced interaction features."""
        try:
            # Generate all enhanced interaction features
            features_dict = self.generate_enhanced_interaction_features(data, **kwargs)

            # Return first feature as representative for base class
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.error(f"Error generating enhanced interaction features: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_enhanced_interaction_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive enhanced interaction features."""
        features = {}

        try:
            # Pairwise interaction features
            features.update(self._generate_pairwise_interaction_features(data))

            # Regime-dependent features
            features.update(self._generate_regime_dependent_features(data))

            # Structural ratio features
            features.update(self._generate_structural_ratio_features(data))

            # Cointegration residual features
            features.update(self._generate_cointegration_features(data))

            # Polynomial interaction features
            features.update(self._generate_polynomial_interaction_features(data))

            # Cross-asset interaction features
            features.update(self._generate_cross_asset_interaction_features(data))

            # Microstructure interaction features
            features.update(self._generate_microstructure_interaction_features(data))

            # Non-linear transformation features
            features.update(self._generate_nonlinear_transformation_features(data))

            logger.info(f"Generated {len(features)} enhanced interaction features")
            return features

        except Exception as e:
            logger.error(f"Error in generate_enhanced_interaction_features: {e}")
            return {}

    def _generate_pairwise_interaction_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate pairwise interaction features between different indicators."""
        features = {}

        # Price-volume interactions
        if "volume" in data.columns:
            # Price-volume momentum interaction
            price_momentum = data["close"].pct_change(5)
            volume_momentum = data["volume"].pct_change(5)
            features["price_volume_momentum_interaction"] = (price_momentum * volume_momentum).fillna(0).values

            # Price-volume divergence
            price_volume_divergence = price_momentum - volume_momentum
            features["price_volume_divergence"] = price_volume_divergence.fillna(0).values

            # Volume-weighted price momentum
            volume_weighted_momentum = price_momentum * (data["volume"] / data["volume"].rolling(20).mean())
            features["volume_weighted_momentum"] = volume_weighted_momentum.fillna(0).values

        # Volatility-momentum interactions
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=20).std()
        momentum = data["close"].pct_change(10)

        # Volatility-momentum interaction
        vol_momentum_interaction = momentum * volatility
        features["volatility_momentum_interaction"] = vol_momentum_interaction.fillna(0).values

        # Volatility-scaled momentum
        vol_scaled_momentum = momentum / (volatility + 1e-8)
        features["volatility_scaled_momentum"] = vol_scaled_momentum.fillna(0).values

        # High-low range interactions
        if "high" in data.columns and "low" in data.columns:
            hl_range = data["high"] - data["low"]
            hl_range_pct = hl_range / data["close"]

            # Range-momentum interaction
            range_momentum_interaction = momentum * hl_range_pct
            features["range_momentum_interaction"] = range_momentum_interaction.fillna(0).values

            # Range-volatility interaction
            range_vol_interaction = volatility * hl_range_pct
            features["range_volatility_interaction"] = range_vol_interaction.fillna(0).values

        # Trend-momentum interactions
        trend_strength = self._calculate_trend_strength(data["close"], 20)
        trend_momentum_interaction = momentum * trend_strength
        features["trend_momentum_interaction"] = trend_momentum_interaction.fillna(0).values

        return features

    def _calculate_trend_strength(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def calc_slope(x):
            if len(x) < 2:
                return 0.0
            try:
                return np.polyfit(range(len(x)), x, 1)[0]
            except:
                return 0.0

        return series.rolling(window=window).apply(calc_slope, raw=False)

    def _generate_regime_dependent_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime-dependent features that adapt to market conditions."""
        features = {}

        # Detect market regimes
        regimes = self._detect_market_regimes(data)

        # Regime-dependent momentum features
        momentum = data["close"].pct_change(10)
        
        for regime_name, regime_mask in regimes.items():
            if regime_mask.sum() > 0:
                # Momentum strength only in trending regimes
                if "trending" in regime_name or "uptrend" in regime_name:
                    regime_momentum = np.zeros(len(data))
                    regime_momentum[regime_mask] = momentum[regime_mask]
                    features[f"regime_dependent_momentum_{regime_name}"] = regime_momentum

                # Volatility features only in high volatility regimes
                if "high_vol" in regime_name or "volatile" in regime_name:
                    returns = data["close"].pct_change()
                    volatility = returns.rolling(window=20).std()
                    regime_vol = np.zeros(len(data))
                    regime_vol[regime_mask] = volatility[regime_mask]
                    features[f"regime_dependent_volatility_{regime_name}"] = regime_vol

                # Volume features only in high volume regimes
                if "high_volume" in regime_name and "volume" in data.columns:
                    volume_ratio = data["volume"] / data["volume"].rolling(20).mean()
                    regime_volume = np.zeros(len(data))
                    regime_volume[regime_mask] = volume_ratio[regime_mask]
                    features[f"regime_dependent_volume_{regime_name}"] = regime_volume

        # Regime transition features
        features.update(self._generate_regime_transition_features(regimes))

        return features

    def _detect_market_regimes(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect market regimes using multiple methods."""
        regimes = {}

        # Volatility regime
        returns = data["close"].pct_change()
        vol = returns.rolling(window=20).std()
        vol_percentiles = vol.quantile([0.33, 0.67])
        
        regimes["low_vol"] = (vol <= vol_percentiles.iloc[0]).astype(int)
        regimes["high_vol"] = (vol >= vol_percentiles.iloc[1]).astype(int)
        regimes["normal_vol"] = ((vol > vol_percentiles.iloc[0]) & (vol < vol_percentiles.iloc[1])).astype(int)

        # Momentum regime
        momentum = data["close"].pct_change(20)
        mom_percentiles = momentum.quantile([0.33, 0.67])
        
        regimes["uptrend"] = (momentum >= mom_percentiles.iloc[1]).astype(int)
        regimes["downtrend"] = (momentum <= mom_percentiles.iloc[0]).astype(int)
        regimes["sideways"] = ((momentum > mom_percentiles.iloc[0]) & (momentum < mom_percentiles.iloc[1])).astype(int)

        # Volume regime
        if "volume" in data.columns:
            volume_ratio = data["volume"] / data["volume"].rolling(20).mean()
            vol_ratio_percentiles = volume_ratio.quantile([0.33, 0.67])
            
            regimes["high_volume"] = (volume_ratio >= vol_ratio_percentiles.iloc[1]).astype(int)
            regimes["low_volume"] = (volume_ratio <= vol_ratio_percentiles.iloc[0]).astype(int)
            regimes["normal_volume"] = ((volume_ratio > vol_ratio_percentiles.iloc[0]) & (volume_ratio < vol_ratio_percentiles.iloc[1])).astype(int)

        # Combined regimes
        regimes["trending"] = (regimes["uptrend"] | regimes["downtrend"]).astype(int)
        regimes["ranging"] = (regimes["sideways"] & regimes["low_vol"]).astype(int)

        return regimes

    def _generate_regime_transition_features(self, regimes: Dict[str, pd.Series]) -> Dict[str, np.ndarray]:
        """Generate regime transition features."""
        features = {}

        for regime_name, regime_mask in regimes.items():
            # Regime persistence
            regime_persistence = regime_mask.rolling(window=10).sum()
            features[f"regime_persistence_{regime_name}"] = regime_persistence.values

            # Regime transitions
            regime_transitions = regime_mask.diff().abs()
            features[f"regime_transitions_{regime_name}"] = regime_transitions.fillna(0).values

        return features

    def _generate_structural_ratio_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate structural ratio features that encode market context."""
        features = {}

        # Bid-ask imbalance proxy (using high-low range)
        if "high" in data.columns and "low" in data.columns:
            hl_range = data["high"] - data["low"]
            close_position = (data["close"] - data["low"]) / (hl_range + 1e-8)
            features["close_position_in_range"] = close_position.fillna(0.5).values

            # Range efficiency
            true_range = np.maximum(
                data["high"] - data["low"],
                np.maximum(
                    (data["high"] - data["close"].shift(1)).abs(),
                    (data["low"] - data["close"].shift(1)).abs()
                )
            )
            range_efficiency = hl_range / (true_range + 1e-8)
            features["range_efficiency"] = range_efficiency.fillna(0).values

        # Price-volume efficiency
        if "volume" in data.columns:
            price_change = data["close"].pct_change().abs()
            volume_change = data["volume"].pct_change().abs()
            pv_efficiency = price_change / (volume_change + 1e-8)
            features["price_volume_efficiency"] = pv_efficiency.fillna(0).values

        # Momentum efficiency
        returns = data["close"].pct_change()
        momentum = data["close"].pct_change(10)
        momentum_efficiency = momentum / (returns.rolling(window=10).std() + 1e-8)
        features["momentum_efficiency"] = momentum_efficiency.fillna(0).values

        # Volatility clustering ratio
        vol_short = returns.rolling(window=5).std()
        vol_long = returns.rolling(window=20).std()
        vol_clustering_ratio = vol_short / (vol_long + 1e-8)
        features["volatility_clustering_ratio"] = vol_clustering_ratio.fillna(1).values

        return features

    def _generate_cointegration_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cointegration residual features for pairs trading."""
        features = {}

        # For single asset, create proxy cointegration features
        # In practice, this would use multiple assets
        
        # Price-mean reversion residual
        price_ma = data["close"].rolling(window=20).mean()
        price_residual = data["close"] - price_ma
        features["price_mean_reversion_residual"] = price_residual.fillna(0).values

        # Volume-price cointegration residual
        if "volume" in data.columns:
            volume_ma = data["volume"].rolling(window=20).mean()
            volume_residual = data["volume"] - volume_ma
            
            # Simple cointegration test
            cointegration_residual = price_residual - 0.1 * volume_residual
            features["volume_price_cointegration_residual"] = cointegration_residual.fillna(0).values

        # Volatility cointegration residual
        returns = data["close"].pct_change()
        vol_short = returns.rolling(window=5).std()
        vol_long = returns.rolling(window=20).std()
        vol_cointegration_residual = vol_short - vol_long
        features["volatility_cointegration_residual"] = vol_cointegration_residual.fillna(0).values

        return features

    def _generate_polynomial_interaction_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate polynomial interaction features."""
        features = {}

        # Create base features
        returns = data["close"].pct_change().fillna(0)
        momentum = data["close"].pct_change(10).fillna(0)
        volatility = returns.rolling(window=20).std().fillna(0)

        # Polynomial features
        polynomial_degrees = self.config.parameters.get("polynomial_degrees", [2, 3])

        for degree in polynomial_degrees:
            # Returns polynomial
            features[f"returns_poly_{degree}"] = np.power(returns, degree).values

            # Momentum polynomial
            features[f"momentum_poly_{degree}"] = np.power(momentum, degree).values

            # Volatility polynomial
            features[f"volatility_poly_{degree}"] = np.power(volatility, degree).values

            # Cross-polynomial features
            features[f"returns_momentum_poly_{degree}"] = (returns * momentum ** (degree-1)).values
            features[f"returns_volatility_poly_{degree}"] = (returns * volatility ** (degree-1)).values
            features[f"momentum_volatility_poly_{degree}"] = (momentum * volatility ** (degree-1)).values

        return features

    def _generate_cross_asset_interaction_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-asset interaction features."""
        features = {}

        # For single asset, create proxy cross-asset features using different timeframes
        # In practice, this would use multiple assets

        # Cross-timeframe momentum interaction
        momentum_5m = data["close"].pct_change(5)
        momentum_20m = data["close"].pct_change(20)
        cross_tf_momentum_interaction = momentum_5m * momentum_20m
        features["cross_tf_momentum_interaction"] = cross_tf_momentum_interaction.fillna(0).values

        # Cross-timeframe volatility interaction
        vol_5m = data["close"].pct_change().rolling(window=5).std()
        vol_20m = data["close"].pct_change().rolling(window=20).std()
        cross_tf_vol_interaction = vol_5m * vol_20m
        features["cross_tf_volatility_interaction"] = cross_tf_vol_interaction.fillna(0).values

        # Cross-timeframe correlation
        cross_tf_correlation = momentum_5m.rolling(window=20).corr(momentum_20m)
        features["cross_tf_correlation"] = cross_tf_correlation.fillna(0).values

        return features

    def _generate_microstructure_interaction_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate microstructure interaction features."""
        features = {}

        if "high" in data.columns and "low" in data.columns and "open" in data.columns:
            # Intraday range interaction
            intraday_range = data["high"] - data["low"]
            overnight_gap = data["open"] - data["close"].shift(1)
            
            # Range-gap interaction
            range_gap_interaction = intraday_range * overnight_gap.abs()
            features["range_gap_interaction"] = range_gap_interaction.fillna(0).values

            # Gap efficiency
            gap_efficiency = overnight_gap.abs() / (intraday_range + 1e-8)
            features["gap_efficiency"] = gap_efficiency.fillna(0).values

            # Price position efficiency
            close_position = (data["close"] - data["low"]) / (data["high"] - data["low"] + 1e-8)
            open_position = (data["open"] - data["low"]) / (data["high"] - data["low"] + 1e-8)
            position_efficiency = close_position - open_position
            features["position_efficiency"] = position_efficiency.fillna(0).values

        # Volume-price microstructure interaction
        if "volume" in data.columns:
            returns = data["close"].pct_change()
            volume_ratio = data["volume"] / data["volume"].rolling(20).mean()
            
            # Volume-weighted returns
            vw_returns = returns * volume_ratio
            features["volume_weighted_returns"] = vw_returns.fillna(0).values

            # Volume momentum interaction
            volume_momentum = data["volume"].pct_change(5)
            price_momentum = data["close"].pct_change(5)
            vw_momentum_interaction = price_momentum * volume_momentum
            features["volume_momentum_interaction"] = vw_momentum_interaction.fillna(0).values

        return features

    def _generate_nonlinear_transformation_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate non-linear transformation features."""
        features = {}

        returns = data["close"].pct_change().fillna(0)
        momentum = data["close"].pct_change(10).fillna(0)
        volatility = returns.rolling(window=20).std().fillna(0)

        # Logarithmic transformations
        if (data["close"] > 0).all():
            log_price = np.log(data["close"])
            log_returns = log_price.diff()
            features["log_returns"] = log_returns.fillna(0).values

        # Exponential transformations
        features["exp_momentum"] = np.exp(momentum).values
        features["exp_volatility"] = np.exp(volatility).values

        # Trigonometric transformations
        features["sin_momentum"] = np.sin(momentum * np.pi).values
        features["cos_momentum"] = np.cos(momentum * np.pi).values

        # Hyperbolic transformations
        features["tanh_momentum"] = np.tanh(momentum).values
        features["tanh_volatility"] = np.tanh(volatility).values

        # Power transformations
        features["sqrt_volatility"] = np.sqrt(volatility).values
        features["cbrt_momentum"] = np.cbrt(momentum).values

        return features


# Individual enhanced interaction generators

class PairwiseInteractionGenerator(FeatureGenerator):
    """Generator for pairwise interaction features."""

    def __init__(self, feature1: str = "rsi", feature2: str = "volume", interaction_type: str = "product"):
        config = FeatureConfig(
            name=f"pairwise_interaction_{feature1}_{feature2}_{interaction_type}",
            category=FeatureCategory.INTERACTION,
            description=f"Pairwise interaction between {feature1} and {feature2} using {interaction_type}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=50,
            parameters={"feature1": feature1, "feature2": feature2, "interaction_type": interaction_type}
        )
        super().__init__(config)
        self.feature1 = feature1
        self.feature2 = feature2
        self.interaction_type = interaction_type

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate pairwise interaction feature."""
        # Calculate features
        feature1_values = self._calculate_feature(data, self.feature1)
        feature2_values = self._calculate_feature(data, self.feature2)

        if feature1_values is None or feature2_values is None:
            return pd.Series(np.zeros(len(data)), index=data.index)

        # Apply interaction
        if self.interaction_type == "product":
            interaction = feature1_values * feature2_values
        elif self.interaction_type == "ratio":
            interaction = feature1_values / (feature2_values + 1e-8)
        elif self.interaction_type == "difference":
            interaction = feature1_values - feature2_values
        elif self.interaction_type == "sum":
            interaction = feature1_values + feature2_values
        else:
            interaction = feature1_values * feature2_values

        return interaction.fillna(0)

    def _calculate_feature(self, data: pd.DataFrame, feature_name: str) -> Optional[pd.Series]:
        """Calculate feature based on name."""
        if feature_name == "rsi":
            return self._calculate_rsi(data["close"])
        elif feature_name == "volume":
            return data["volume"] if "volume" in data.columns else None
        elif feature_name == "momentum":
            return data["close"].pct_change(10)
        elif feature_name == "volatility":
            returns = data["close"].pct_change()
            return returns.rolling(window=20).std()
        else:
            return None

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi


class RegimeDependentFeatureGenerator(FeatureGenerator):
    """Generator for regime-dependent features."""

    def __init__(self, regime_detector: str = "volatility", feature_type: str = "momentum"):
        config = FeatureConfig(
            name=f"regime_dependent_{feature_type}_{regime_detector}",
            category=FeatureCategory.INTERACTION,
            description=f"Regime-dependent {feature_type} feature using {regime_detector} regime detection",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=30,
            min_lookback=20,
            max_lookback=100,
            parameters={"regime_detector": regime_detector, "feature_type": feature_type}
        )
        super().__init__(config)
        self.regime_detector = regime_detector
        self.feature_type = feature_type

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime-dependent feature."""
        # Detect regimes
        regimes = self._detect_regimes(data, self.regime_detector)
        
        # Calculate base feature
        base_feature = self._calculate_base_feature(data, self.feature_type)
        
        if base_feature is None:
            return pd.Series(np.zeros(len(data)), index=data.index)

        # Apply regime-dependent logic
        regime_feature = np.zeros(len(data))
        
        for regime_name, regime_mask in regimes.items():
            if regime_mask.sum() > 0:
                # Only apply feature in appropriate regimes
                if self._should_apply_in_regime(regime_name, self.feature_type):
                    regime_feature[regime_mask] = base_feature[regime_mask]

        return pd.Series(regime_feature, index=data.index)

    def _detect_regimes(self, data: pd.DataFrame, detector: str) -> Dict[str, pd.Series]:
        """Detect regimes using specified detector."""
        regimes = {}

        if detector == "volatility":
            returns = data["close"].pct_change()
            vol = returns.rolling(window=20).std()
            vol_percentiles = vol.quantile([0.33, 0.67])
            
            regimes["low_vol"] = (vol <= vol_percentiles.iloc[0]).astype(int)
            regimes["high_vol"] = (vol >= vol_percentiles.iloc[1]).astype(int)
            regimes["normal_vol"] = ((vol > vol_percentiles.iloc[0]) & (vol < vol_percentiles.iloc[1])).astype(int)

        elif detector == "momentum":
            momentum = data["close"].pct_change(20)
            mom_percentiles = momentum.quantile([0.33, 0.67])
            
            regimes["uptrend"] = (momentum >= mom_percentiles.iloc[1]).astype(int)
            regimes["downtrend"] = (momentum <= mom_percentiles.iloc[0]).astype(int)
            regimes["sideways"] = ((momentum > mom_percentiles.iloc[0]) & (momentum < mom_percentiles.iloc[1])).astype(int)

        return regimes

    def _calculate_base_feature(self, data: pd.DataFrame, feature_type: str) -> Optional[pd.Series]:
        """Calculate base feature."""
        if feature_type == "momentum":
            return data["close"].pct_change(10)
        elif feature_type == "volatility":
            returns = data["close"].pct_change()
            return returns.rolling(window=20).std()
        elif feature_type == "volume" and "volume" in data.columns:
            return data["volume"] / data["volume"].rolling(20).mean()
        else:
            return None

    def _should_apply_in_regime(self, regime_name: str, feature_type: str) -> bool:
        """Determine if feature should be applied in given regime."""
        if feature_type == "momentum":
            return "trend" in regime_name or "uptrend" in regime_name
        elif feature_type == "volatility":
            return "vol" in regime_name or "high" in regime_name
        elif feature_type == "volume":
            return "volume" in regime_name or "high" in regime_name
        else:
            return True


class StructuralRatioGenerator(FeatureGenerator):
    """Generator for structural ratio features."""

    def __init__(self, ratio_type: str = "bid_ask_imbalance", window: int = 20):
        config = FeatureConfig(
            name=f"structural_ratio_{ratio_type}_{window}",
            category=FeatureCategory.INTERACTION,
            description=f"Structural {ratio_type} ratio with {window} period window",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"ratio_type": ratio_type, "window": window}
        )
        super().__init__(config)
        self.ratio_type = ratio_type
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate structural ratio feature."""
        if self.ratio_type == "bid_ask_imbalance":
            return self._calculate_bid_ask_imbalance(data)
        elif self.ratio_type == "range_efficiency":
            return self._calculate_range_efficiency(data)
        elif self.ratio_type == "volume_efficiency":
            return self._calculate_volume_efficiency(data)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _calculate_bid_ask_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """Calculate bid-ask imbalance proxy using high-low range."""
        if "high" in data.columns and "low" in data.columns:
            hl_range = data["high"] - data["low"]
            close_position = (data["close"] - data["low"]) / (hl_range + 1e-8)
            return close_position.fillna(0.5)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _calculate_range_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """Calculate range efficiency ratio."""
        if "high" in data.columns and "low" in data.columns:
            hl_range = data["high"] - data["low"]
            true_range = np.maximum(
                data["high"] - data["low"],
                np.maximum(
                    (data["high"] - data["close"].shift(1)).abs(),
                    (data["low"] - data["close"].shift(1)).abs()
                )
            )
            efficiency = hl_range / (true_range + 1e-8)
            return efficiency.fillna(0)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _calculate_volume_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume efficiency ratio."""
        if "volume" in data.columns:
            price_change = data["close"].pct_change().abs()
            volume_change = data["volume"].pct_change().abs()
            efficiency = price_change / (volume_change + 1e-8)
            return efficiency.fillna(0)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)


def create_enhanced_interaction_generators() -> List[FeatureGenerator]:
    """Create all enhanced interaction feature generators."""
    generators = []

    # Main enhanced interaction generator
    generators.append(EnhancedInteractionFeatureGenerator())

    # Individual generators
    feature_pairs = [
        ("rsi", "volume", "product"),
        ("momentum", "volatility", "ratio"),
        ("price", "volume", "product"),
        ("volatility", "trend", "product")
    ]

    for feature1, feature2, interaction_type in feature_pairs:
        generators.append(PairwiseInteractionGenerator(feature1=feature1, feature2=feature2, interaction_type=interaction_type))

    # Regime-dependent generators
    for regime_detector in ["volatility", "momentum"]:
        for feature_type in ["momentum", "volatility", "volume"]:
            generators.append(RegimeDependentFeatureGenerator(regime_detector=regime_detector, feature_type=feature_type))

    # Structural ratio generators
    for ratio_type in ["bid_ask_imbalance", "range_efficiency", "volume_efficiency"]:
        for window in [10, 20, 50]:
            generators.append(StructuralRatioGenerator(ratio_type=ratio_type, window=window))

    return generators


def create_default_enhanced_interaction_generators() -> List[FeatureGenerator]:
    """Create default set of enhanced interaction generators."""
    return create_enhanced_interaction_generators()


# Export all generators
__all__ = [
    'EnhancedInteractionFeatureGenerator',
    'PairwiseInteractionGenerator',
    'RegimeDependentFeatureGenerator',
    'StructuralRatioGenerator',
    'create_enhanced_interaction_generators',
    'create_default_enhanced_interaction_generators'
]