"""
Consolidated Microstructure Features Module

This module provides comprehensive microstructure and order flow feature generators
for high-frequency trading analysis, combining all microstructure-related features
with full VectorBT optimization.

Key Features:
- Bid-ask spread analysis and proxies
- Order flow imbalance and aggression metrics
- Market depth and liquidity analysis
- Trade size and intensity analysis
- Price impact and volume-weighted features
- VectorBT-optimized order flow features
- High-frequency microstructure indicators
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

from ..base_calculations import BaseCalculationType, create_base_calculator
from src.feature_generation.utils.math_validation import safe_divide, validate_finite, safe_percentage_change

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
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
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

# Import scipy for advanced statistical functions
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint, tprint_warning
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)

logger = logging.getLogger(__name__)

# ============================================================================
# CORE MICROSTRUCTURE FEATURES
# ============================================================================

class MicrostructureFeatureGenerator(VectorizedFeatureGenerator):
    """
    Comprehensive microstructure feature generator for high-frequency trading analysis.

    This generator creates a wide range of microstructure features that capture
    market dynamics, order flow patterns, and liquidity conditions. It's designed
    for high-frequency trading analysis and market microstructure research.

    Key Feature Categories:
    - Bid-ask spread analysis and proxies
    - Order flow imbalance and aggression metrics
    - Market depth and liquidity analysis
    - Trade size and intensity analysis
    - Price impact and volume-weighted features
    - Market microstructure indicators

    Parameters:
    - config: FeatureConfig object with generator parameters

    Returns:
    - Dict[str, np.ndarray]: Dictionary of microstructure features

    Example:
        >>> generator = MicrostructureFeatureGenerator()
        >>> features = generator.generate_features(data)
        >>> print(f"Generated {len(features)} microstructure features")
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self) -> FeatureConfig:
        """Create default configuration for microstructure features."""
        return FeatureConfig(
            name="microstructure_features",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Comprehensive microstructure features for high-frequency analysis",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open", "bid", "ask", "bid_size", "ask_size"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate microstructure features."""
        features = self.generate_features(data, **kwargs)

        # Return the first feature as representative
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index[:len(features[first_feature_name])])
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate all microstructure features."""
        features = {}

        try:
            # Bid-ask spread features
            if 'bid' in data.columns and 'ask' in data.columns:
                features.update(self._calculate_bid_ask_features(data))
            else:
                missing = [col for col in ['bid', 'ask'] if col not in data.columns]
                if missing:
                    tprint_warning(
                        f"⚠️ Skipping bid/ask microstructure features due to missing columns: {missing}"
                    )

            # Order flow features
            features.update(self._calculate_order_flow_features(data))

            # Volume-weighted features
            features.update(self._calculate_volume_weighted_features(data))

            # Trade intensity features
            features.update(self._calculate_trade_intensity_features(data))

            # Liquidity proxy features
            features.update(self._calculate_liquidity_proxy_features(data))

            # Market depth features
            if 'bid_size' in data.columns and 'ask_size' in data.columns:
                features.update(self._calculate_market_depth_features(data))

        except Exception as e:
            tprint(f"⚠️ Microstructure features generation failed: {e}")

        return features

    def _calculate_bid_ask_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate bid-ask spread related features."""
        features = {}

        try:
            bid = data['bid']
            ask = data['ask']

            # Basic spread
            spread = ask - bid
            features['bid_ask_spread'] = spread.values

            # Relative spread
            mid_price = (bid + ask) / 2
            relative_spread = spread / mid_price
            features['relative_spread'] = relative_spread.values

            # Spread volatility
            if VECTORBT_AVAILABLE:
                spread_vol = rolling_std(spread, window=20)
                features['spread_volatility'] = spread_vol.values
            else:
                features['spread_volatility'] = spread.rolling(window=20).std().values

        except Exception as e:
            tprint(f"⚠️ Bid-ask features calculation failed: {e}")

        return features

    def _calculate_order_flow_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate order flow related features."""
        features = {}

        try:
            close = data['close']
            volume = data['volume']

            # Price-volume relationship
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Order flow imbalance (simplified)
            if VECTORBT_AVAILABLE:
                price_volume_corr = rolling_corr(price_change, volume_change, window=20)
                features['order_flow_imbalance'] = price_volume_corr.values
            else:
                features['order_flow_imbalance'] = price_change.rolling(window=20).corr(volume_change).values

            # Volume-weighted price change
            vw_price_change = (price_change * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
            features['vw_price_change'] = vw_price_change.values

        except Exception as e:
            tprint(f"⚠️ Order flow features calculation failed: {e}")

        return features

    def _calculate_volume_weighted_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volume-weighted features."""
        features = {}

        try:
            close = data['close']
            volume = data['volume']

            # VWAP
            if VECTORBT_AVAILABLE:
                vwap = rolling_sum(close * volume, window=20) / rolling_sum(volume, window=20)
                features['vwap'] = vwap.values
                features['vwap_ratio'] = (close / vwap).values
            else:
                vwap = (close * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
                features['vwap'] = vwap.values
                features['vwap_ratio'] = (close / vwap).values

            # Volume-weighted volatility
            returns = close.pct_change()
            if VECTORBT_AVAILABLE:
                vw_volatility = rolling_std(returns, window=20) * np.sqrt(volume.rolling(window=20).mean())
                features['vw_volatility'] = vw_volatility.values
            else:
                vw_volatility = returns.rolling(window=20).std() * np.sqrt(volume.rolling(window=20).mean())
                features['vw_volatility'] = vw_volatility.values

        except Exception as e:
            tprint(f"⚠️ Volume-weighted features calculation failed: {e}")

        return features

    def _calculate_trade_intensity_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate trade intensity features."""
        features = {}

        try:
            volume = data['volume']

            # Trade intensity (volume per period)
            if VECTORBT_AVAILABLE:
                intensity = rolling_mean(volume, window=20)
                intensity_volatility = rolling_std(volume, window=20)
                features['trade_intensity'] = intensity.values
                features['intensity_volatility'] = intensity_volatility.values
            else:
                intensity = volume.rolling(window=20).mean()
                intensity_volatility = volume.rolling(window=20).std()
                features['trade_intensity'] = intensity.values
                features['intensity_volatility'] = intensity_volatility.values

            # Volume acceleration
            volume_change = volume.pct_change()
            if VECTORBT_AVAILABLE:
                volume_acceleration = rolling_mean(volume_change, window=5)
                features['volume_acceleration'] = volume_acceleration.values
            else:
                volume_acceleration = volume_change.rolling(window=5).mean()
                features['volume_acceleration'] = volume_acceleration.values

        except Exception as e:
            tprint(f"⚠️ Trade intensity features calculation failed: {e}")

        return features

    def _calculate_liquidity_proxy_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate liquidity proxy features."""
        features = {}

        try:
            close = data['close']
            volume = data['volume']

            # Amihud illiquidity measure
            returns = close.pct_change()
            amihud_illiquidity = np.abs(returns) / volume
            if VECTORBT_AVAILABLE:
                features['amihud_illiquidity'] = rolling_mean(amihud_illiquidity, window=20).values
            else:
                features['amihud_illiquidity'] = amihud_illiquidity.rolling(window=20).mean().values

            # Roll's lambda (simplified)
            from ...utils.error_handling import safe_diff
            price_changes = safe_diff(close)
            if VECTORBT_AVAILABLE:
                roll_lambda = rolling_std(price_changes, window=20)
                features['roll_lambda'] = roll_lambda.values
            else:
                roll_lambda = price_changes.rolling(window=20).std()
                features['roll_lambda'] = roll_lambda.values

        except Exception as e:
            tprint(f"⚠️ Liquidity proxy features calculation failed: {e}")

        return features

    def _calculate_market_depth_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate market depth features."""
        features = {}

        try:
            bid_size = data['bid_size']
            ask_size = data['ask_size']

            # Market depth imbalance
            depth_imbalance = (bid_size - ask_size) / (bid_size + ask_size)
            features['depth_imbalance'] = depth_imbalance.values

            # Total market depth
            total_depth = bid_size + ask_size
            features['total_depth'] = total_depth.values

            # Depth volatility
            if VECTORBT_AVAILABLE:
                depth_volatility = rolling_std(total_depth, window=20)
                features['depth_volatility'] = depth_volatility.values
            else:
                depth_volatility = total_depth.rolling(window=20).std()
                features['depth_volatility'] = depth_volatility.values

        except Exception as e:
            tprint(f"⚠️ Market depth features calculation failed: {e}")

        return features

# DISABLED: Requires bid/ask columns which are not available
# class BidAskSpreadGenerator(VectorizedFeatureGenerator):
#     """Generator for bid-ask spread features."""
#
#     def __init__(self, window: int = 20):
#         config = FeatureConfig(
#             name="bid_ask_spread",
#             category=FeatureCategory.MICROSTRUCTURE,
#             description="Bid-ask spread analysis for market microstructure",
#             required_columns=[],
#             optional_columns=["bid", "ask"],
#             default_lookback=window,
#             min_lookback=5,
#             max_lookback=100,
#             parameters={"window": window}
#         )
#         super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
#         self.window = window
#
#     def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
#         """Generate bid-ask spread feature."""
#         try:
#             # Check if bid and ask columns exist
#             if 'bid' not in data.columns or 'ask' not in data.columns:
#                 missing = [col for col in ['bid', 'ask'] if col not in data.columns]
#                 tprint_warning(f"⚠️ Skipping bid-ask spread feature due to missing columns: {missing}")
#                 return pd.Series(np.zeros(len(data)), index=data.index)
#
#             bid = data['bid']
#             ask = data['ask']
#
#             # Calculate spread
#             spread = ask - bid
#
#             # Calculate relative spread
#             mid_price = (bid + ask) / 2
#             relative_spread = spread / mid_price
#
#             # Calculate spread volatility
#             if VECTORBT_AVAILABLE and rolling_std is not None:
#                 spread_vol = rolling_std(relative_spread, window=self.window)
#             else:
#                 spread_vol = relative_spread.rolling(window=self.window).std()
#
#             return spread_vol
#
#         except Exception as e:
#             tprint_warning(f"⚠️ Bid-ask spread calculation failed: {e}")
#             return pd.Series(np.zeros(len(data)), index=data.index)

class OrderFlowImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for order flow imbalance features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="order_flow_imbalance",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Order flow imbalance analysis",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow imbalance feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price and volume changes
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Calculate correlation as proxy for order flow imbalance
            if VECTORBT_AVAILABLE:
                imbalance = rolling_corr(price_change, volume_change, window=self.window)
            else:
                imbalance = price_change.rolling(window=self.window).corr(volume_change)

            return imbalance.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow imbalance calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class TradeSizeImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for trade size imbalance features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="trade_size_imbalance",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Trade size imbalance analysis",
            required_columns=["volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trade size imbalance feature."""
        try:
            volume = data['volume']

            # Calculate volume percentiles
            if VECTORBT_AVAILABLE:
                volume_median = rolling_quantile(volume, window=self.window, q=0.5)
                large_trades = (volume > volume_median).astype(int)
                imbalance = rolling_mean(large_trades, window=self.window) - 0.5
            else:
                volume_median = volume.rolling(window=self.window).quantile(0.5)
                large_trades = (volume > volume_median).astype(int)
                imbalance = large_trades.rolling(window=self.window).mean() - 0.5

            return imbalance.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Trade size imbalance calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class PriceImpactGenerator(VectorizedFeatureGenerator):
    """Generator for price impact features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="price_impact",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Price impact analysis for market microstructure",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price impact feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate returns
            returns = close.pct_change()

            # Calculate volume-weighted price impact
            if VECTORBT_AVAILABLE:
                price_impact = rolling_mean(np.abs(returns) / volume, window=self.window)
            else:
                price_impact = (np.abs(returns) / volume).rolling(window=self.window).mean()

            return price_impact.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Price impact calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VolumeWeightedPriceGenerator(VectorizedFeatureGenerator):
    """Generator for volume-weighted price features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="volume_weighted_price",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Volume-weighted average price analysis",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume-weighted price feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate VWAP
            if VECTORBT_AVAILABLE:
                vwap = rolling_sum(close * volume, window=self.window) / rolling_sum(volume, window=self.window)
                vwap_ratio = close / vwap
            else:
                vwap = (close * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                vwap_ratio = close / vwap

            return vwap_ratio.fillna(1)

        except Exception as e:
            tprint(f"⚠️ Volume-weighted price calculation failed: {e}")
            return pd.Series(np.ones(len(data)), index=data.index)

class TradeIntensityGenerator(VectorizedFeatureGenerator):
    """Generator for trade intensity features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="trade_intensity",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Trade intensity analysis",
            required_columns=["volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trade intensity feature."""
        try:
            volume = data['volume']

            # Calculate trade intensity
            if VECTORBT_AVAILABLE:
                intensity = rolling_mean(volume, window=self.window)
                intensity_volatility = rolling_std(volume, window=self.window)
                normalized_intensity = intensity / intensity_volatility
            else:
                intensity = volume.rolling(window=self.window).mean()
                intensity_volatility = volume.rolling(window=self.window).std()
                normalized_intensity = intensity / intensity_volatility

            return normalized_intensity.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Trade intensity calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class LiquidityProxyGenerator(VectorizedFeatureGenerator):
    """Generator for liquidity proxy features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="liquidity_proxy",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Liquidity proxy analysis",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate liquidity proxy feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate Amihud illiquidity measure
            returns = close.pct_change()
            amihud_illiquidity = np.abs(returns) / volume

            if VECTORBT_AVAILABLE:
                liquidity_proxy = rolling_mean(amihud_illiquidity, window=self.window)
            else:
                liquidity_proxy = amihud_illiquidity.rolling(window=self.window).mean()

            # Invert to get liquidity (higher values = more liquid)
            liquidity = 1 / (1 + liquidity_proxy)

            return liquidity.fillna(0.5)

        except Exception as e:
            tprint(f"⚠️ Liquidity proxy calculation failed: {e}")
            return pd.Series(np.ones(len(data)) * 0.5, index=data.index)

class MarketDepthGenerator(VectorizedFeatureGenerator):
    """Generator for market depth features."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="market_depth",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Market depth analysis",
            required_columns=[],  # Changed to optional to prevent fast-fail
            optional_columns=["bid_size", "ask_size"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market depth feature."""
        try:
            # Check if bid_size and ask_size columns exist
            if 'bid_size' not in data.columns or 'ask_size' not in data.columns:
                missing = [col for col in ['bid_size', 'ask_size'] if col not in data.columns]
                tprint(f"⚠️ Skipping market depth feature due to missing columns: {missing}", "warning")
                return pd.Series(np.zeros(len(data)), index=data.index)
            
            bid_size = data['bid_size']
            ask_size = data['ask_size']

            # Calculate depth imbalance
            total_depth = bid_size + ask_size
            depth_imbalance = (bid_size - ask_size) / total_depth

            if VECTORBT_AVAILABLE:
                depth_volatility = rolling_std(depth_imbalance, window=self.window)
            else:
                depth_volatility = depth_imbalance.rolling(window=self.window).std()

            return depth_volatility.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Market depth calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

# ============================================================================
# VECTORBT-OPTIMIZED ORDER FLOW FEATURES
# ============================================================================

class VectorBTTakerBuyRatioGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized taker buy ratio generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_taker_buy_ratio",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized taker buy ratio",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate taker buy ratio feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Taker buy ratio (simplified: positive price change with volume)
            taker_buy_ratio = np.where(price_change > 0, volume, 0)

            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                # Use VectorBT rolling operations
                taker_buy_sum = rolling_sum(taker_buy_ratio, window=self.window)
                total_volume = rolling_sum(volume, window=self.window)
                ratio = safe_divide(taker_buy_sum, total_volume)
            else:
                # Fallback to pandas
                taker_buy_sum = pd.Series(taker_buy_ratio).rolling(window=self.window).sum()
                total_volume = volume.rolling(window=self.window).sum()
                ratio = safe_divide(taker_buy_sum, total_volume)

            return ratio.fillna(0.5)

        except Exception as e:
            tprint(f"⚠️ Taker buy ratio calculation failed: {e}")
            return pd.Series(np.ones(len(data)) * 0.5, index=data.index)

class VectorBTTakerSellRatioGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized taker sell ratio generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_taker_sell_ratio",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized taker sell ratio",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate taker sell ratio feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Taker sell ratio (simplified: negative price change with volume)
            taker_sell_ratio = np.where(price_change < 0, volume, 0)

            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                # Use VectorBT rolling operations
                taker_sell_sum = rolling_sum(taker_sell_ratio, window=self.window)
                total_volume = rolling_sum(volume, window=self.window)
                ratio = safe_divide(taker_sell_sum, total_volume)
            else:
                # Fallback to pandas
                taker_sell_sum = pd.Series(taker_sell_ratio).rolling(window=self.window).sum()
                total_volume = volume.rolling(window=self.window).sum()
                ratio = safe_divide(taker_sell_sum, total_volume)

            return ratio.fillna(0.5)

        except Exception as e:
            tprint(f"⚠️ Taker sell ratio calculation failed: {e}")
            return pd.Series(np.ones(len(data)) * 0.5, index=data.index)

class VectorBTMarketAggressionIndexGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized market aggression index generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_market_aggression_index",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized market aggression index",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market aggression index feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change and volume change
            price_change = close.pct_change()
            volume_change = volume.pct_change()

            # Market aggression index (correlation between price and volume changes)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                aggression_index = rolling_corr(price_change, volume_change, window=self.window)
            else:
                aggression_index = price_change.rolling(window=self.window).corr(volume_change)

            return aggression_index.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Market aggression index calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowImbalanceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow imbalance generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_imbalance",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow imbalance",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow imbalance feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow imbalance (volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()

            return vw_price_change.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow imbalance calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

# DISABLED: Requires bid/ask columns which are not available
# class VectorBTBidAskImbalanceGenerator(VectorBTFeatureGenerator):
#     """VectorBT-optimized bid-ask imbalance generator."""
#
#     def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
#         if config is None:
#             config = self._create_default_config(window)
#         super().__init__(config)
#         self.window = window
#
#         # Initialize VectorBT optimizer
#         self.vectorbt_rolling_optimizer = None
#         if OPTIMIZATION_AVAILABLE:
#             try:
#                 self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
#             except Exception as e:
#                 tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
#
#     def _create_default_config(self, window: int) -> FeatureConfig:
#         """Create default configuration."""
#         return FeatureConfig(
#             name="vectorbt_bid_ask_imbalance",
#             category=FeatureCategory.MICROSTRUCTURE,
#             description="VectorBT-optimized bid-ask imbalance",
#             required_columns=["bid", "ask"],
#             default_lookback=window,
#             min_lookback=5,
#             max_lookback=100,
#             parameters={"window": window}
#         )
#
#     def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
#         """Generate bid-ask imbalance feature."""
#         try:
#             # Check if bid and ask columns exist
#             if 'bid' not in data.columns or 'ask' not in data.columns:
#                 missing = [col for col in ['bid', 'ask'] if col not in data.columns]
#                 tprint_warning(f"⚠️ Skipping bid-ask imbalance feature due to missing columns: {missing}")
#                 return pd.Series(np.zeros(len(data)), index=data.index)
#
#             bid = data['bid']
#             ask = data['ask']
#
#             # Calculate spread and mid price
#             spread = ask - bid
#             mid_price = (bid + ask) / 2
#
#             # Relative spread as imbalance measure
#             relative_spread = spread / mid_price
#
#             if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer and rolling_mean is not None:
#                 imbalance = rolling_mean(relative_spread, window=self.window)
#             else:
#                 imbalance = relative_spread.rolling(window=self.window).mean()
#
#             return imbalance.fillna(0)
#
#         except Exception as e:
#             tprint_warning(f"⚠️ Bid-ask imbalance calculation failed: {e}")
#             return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTMarketOrderFlowGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized market order flow generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_market_order_flow",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized market order flow",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market order flow feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Market order flow (volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                order_flow = rolling_sum(price_change * volume, window=self.window)
            else:
                order_flow = (price_change * volume).rolling(window=self.window).sum()

            return order_flow.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Market order flow calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTVolumeWeightedOrderFlowGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume-weighted order flow generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_volume_weighted_order_flow",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized volume-weighted order flow",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume-weighted order flow feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Volume-weighted order flow
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_order_flow = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
            else:
                vw_order_flow = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()

            return vw_order_flow.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Volume-weighted order flow calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow momentum generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_momentum",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow momentum",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow momentum feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow momentum (rate of change of volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                # Ensure we have a Series for diff operation using safe_diff
                from ...utils.error_handling import safe_diff
                if isinstance(vw_price_change, pd.Series):
                    momentum = safe_diff(vw_price_change)
                else:
                    momentum = safe_diff(pd.Series(vw_price_change))
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                from ...utils.error_handling import safe_diff
                momentum = safe_diff(vw_price_change)

            return momentum.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow momentum calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow volatility generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_volatility",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow volatility",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow volatility feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow volatility (volatility of volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                volatility = rolling_std(vw_price_change, window=self.window)
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                volatility = vw_price_change.rolling(window=self.window).std()

            return volatility.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow volatility calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow trend strength generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_trend_strength",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow trend strength",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow trend strength feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow trend strength (autocorrelation of volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                trend_strength = rolling_corr(vw_price_change, vw_price_change.shift(1), window=self.window)
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                trend_strength = vw_price_change.rolling(window=self.window).corr(vw_price_change.shift(1))

            return trend_strength.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow trend strength calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow consistency generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_consistency",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow consistency",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow consistency feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow consistency (inverse of volatility of volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                volatility = rolling_std(vw_price_change, window=self.window)
                consistency = 1 / (1 + volatility)
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                volatility = vw_price_change.rolling(window=self.window).std()
                consistency = 1 / (1 + volatility)

            return consistency.fillna(0.5)

        except Exception as e:
            tprint(f"⚠️ Order flow consistency calculation failed: {e}")
            return pd.Series(np.ones(len(data)) * 0.5, index=data.index)

class VectorBTOrderFlowAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow acceleration generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_acceleration",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow acceleration",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow acceleration feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow acceleration (second derivative of volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                # Ensure we have Series for diff operations using safe_diff
                from ...utils.error_handling import safe_diff
                if isinstance(vw_price_change, pd.Series):
                    velocity = safe_diff(vw_price_change)
                    acceleration = safe_diff(velocity)
                else:
                    vw_series = pd.Series(vw_price_change)
                    velocity = safe_diff(vw_series)
                    acceleration = safe_diff(velocity)
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                from ...utils.error_handling import safe_diff
                velocity = safe_diff(vw_price_change)
                acceleration = safe_diff(velocity)

            return acceleration.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow acceleration calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowJerkGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow jerk generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_jerk",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow jerk",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow jerk feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow jerk (third derivative of volume-weighted price change)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                # Ensure we have Series for diff operations using safe_diff
                from ...utils.error_handling import safe_diff
                if isinstance(vw_price_change, pd.Series):
                    velocity = safe_diff(vw_price_change)
                    acceleration = safe_diff(velocity)
                    jerk = safe_diff(acceleration)
                else:
                    vw_series = pd.Series(vw_price_change)
                    velocity = safe_diff(vw_series)
                    acceleration = safe_diff(velocity)
                    jerk = safe_diff(acceleration)
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                from ...utils.error_handling import safe_diff
                velocity = safe_diff(vw_price_change)
                acceleration = safe_diff(velocity)
                jerk = safe_diff(acceleration)

            return jerk.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow jerk calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class VectorBTOrderFlowRegimeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow regime generator."""

    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window

        # Initialize VectorBT optimizer
        self.vectorbt_rolling_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")

    def _create_default_config(self, window: int) -> FeatureConfig:
        """Create default configuration."""
        return FeatureConfig(
            name="vectorbt_order_flow_regime",
            category=FeatureCategory.MICROSTRUCTURE,
            description="VectorBT-optimized order flow regime detection",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow regime feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate price change
            price_change = close.pct_change()

            # Order flow regime (based on volume-weighted price change and volatility)
            if VECTORBT_AVAILABLE and self.vectorbt_rolling_optimizer:
                vw_price_change = rolling_sum(price_change * volume, window=self.window) / rolling_sum(volume, window=self.window)
                volatility = rolling_std(vw_price_change, window=self.window)
                regime = np.where(vw_price_change > volatility, 1, np.where(vw_price_change < -volatility, -1, 0))
            else:
                vw_price_change = (price_change * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
                volatility = vw_price_change.rolling(window=self.window).std()
                regime = np.where(vw_price_change > volatility, 1, np.where(vw_price_change < -volatility, -1, 0))

            return pd.Series(regime, index=data.index).fillna(0)

        except Exception as e:
            tprint(f"⚠️ Order flow regime calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

# ============================================================================
# ANALYST FEATURES
# ============================================================================

# DISABLED: Requires bid/ask columns which are not available
# class AnalystSpreadNormalizedGenerator(VectorizedFeatureGenerator):
#     """Generator for analyst spread normalized feature."""
#
#     def __init__(self, window: int = 20):
#         config = FeatureConfig(
#             name="analyst_spread_normalized",
#             category=FeatureCategory.MICROSTRUCTURE,
#             description="Analyst spread normalized for market microstructure analysis",
#             required_columns=["bid", "ask"],
#             default_lookback=window,
#             min_lookback=5,
#             max_lookback=100,
#             parameters={"window": window}
#         )
#         super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
#         self.window = window
#
#     def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
#         """Generate analyst spread normalized feature."""
#         try:
#             # Check if bid and ask columns exist
#             if 'bid' not in data.columns or 'ask' not in data.columns:
#                 missing = [col for col in ['bid', 'ask'] if col not in data.columns]
#                 tprint_warning(f"⚠️ Skipping analyst spread normalized feature due to missing columns: {missing}")
#                 return pd.Series(np.ones(len(data)), index=data.index)
#
#             bid = data['bid']
#             ask = data['ask']
#
#             # Calculate spread and normalize
#             spread = ask - bid
#             mid_price = (bid + ask) / 2
#             normalized_spread = spread / mid_price
#
#             # Calculate rolling mean for normalization
#             if VECTORBT_AVAILABLE and rolling_mean is not None:
#                 normalized_spread_mean = rolling_mean(normalized_spread, window=self.window)
#                 spread_normalized = normalized_spread / normalized_spread_mean
#             else:
#                 normalized_spread_mean = normalized_spread.rolling(window=self.window).mean()
#                 spread_normalized = normalized_spread / normalized_spread_mean
#
#             return spread_normalized.fillna(1)
#
#         except Exception as e:
#             tprint_warning(f"⚠️ Analyst spread normalized calculation failed: {e}")
#             return pd.Series(np.ones(len(data)), index=data.index)

class AnalystTickImbalanceGenerator(VectorizedFeatureGenerator):
    """Generator for analyst tick imbalance feature."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="analyst_tick_imbalance",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Analyst tick imbalance for market microstructure analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate analyst tick imbalance feature."""
        try:
            close = data['close']

            # Calculate price changes
            from ...utils.error_handling import safe_diff
            price_change = safe_diff(close)

            # Tick imbalance (simplified: positive vs negative changes)
            tick_imbalance = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))

            # Calculate rolling sum
            if VECTORBT_AVAILABLE:
                tick_imbalance_sum = rolling_sum(tick_imbalance, window=self.window)
            else:
                tick_imbalance_sum = pd.Series(tick_imbalance).rolling(window=self.window).sum()

            return tick_imbalance_sum.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Analyst tick imbalance calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class CorwinSchultzSpreadMomentumGenerator(VectorizedFeatureGenerator):
    """Generator for Corwin-Schultz spread momentum feature."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="corwin_schultz_spread_momentum",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Corwin-Schultz spread momentum for market microstructure analysis",
            required_columns=["high", "low"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Corwin-Schultz spread momentum feature."""
        try:
            high = data['high']
            low = data['low']

            # Calculate high-low range
            hl_range = high - low

            # Calculate Corwin-Schultz spread (simplified)
            if VECTORBT_AVAILABLE:
                spread = rolling_mean(hl_range, window=self.window)
                # Ensure we have Series for diff operation using safe_diff
                from ...utils.error_handling import safe_diff
                if isinstance(spread, pd.Series):
                    spread_momentum = safe_diff(spread)
                else:
                    spread_momentum = safe_diff(pd.Series(spread))
            else:
                spread = hl_range.rolling(window=self.window).mean()
                from ...utils.error_handling import safe_diff
                spread_momentum = safe_diff(spread)

            return spread_momentum.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Corwin-Schultz spread momentum calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class AmihudIlliquidityVWAPDistanceGenerator(VectorizedFeatureGenerator):
    """Generator for Amihud illiquidity VWAP distance feature."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="amihud_illiquidity_vwap_distance",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Amihud illiquidity VWAP distance for market microstructure analysis",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Amihud illiquidity VWAP distance feature."""
        try:
            close = data['close']
            volume = data['volume']

            # Calculate VWAP
            if VECTORBT_AVAILABLE:
                vwap = rolling_sum(close * volume, window=self.window) / rolling_sum(volume, window=self.window)
            else:
                vwap = (close * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()

            # Calculate distance from VWAP
            vwap_distance = np.abs(close - vwap) / vwap

            # Calculate Amihud illiquidity
            returns = close.pct_change()
            amihud_illiquidity = np.abs(returns) / volume

            # Combine features
            combined_feature = vwap_distance * amihud_illiquidity

            if VECTORBT_AVAILABLE:
                feature = rolling_mean(combined_feature, window=self.window)
            else:
                feature = combined_feature.rolling(window=self.window).mean()

            return feature.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Amihud illiquidity VWAP distance calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class RollLambdaRVShortGenerator(VectorizedFeatureGenerator):
    """Generator for Roll lambda RV short feature."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name="roll_lambda_rv_short",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Roll lambda RV short for market microstructure analysis",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Roll lambda RV short feature."""
        try:
            close = data['close']

            # Calculate returns
            returns = close.pct_change()

            # Calculate realized variance
            if VECTORBT_AVAILABLE:
                rv = rolling_sum(returns ** 2, window=self.window)
            else:
                rv = (returns ** 2).rolling(window=self.window).sum()

            # Calculate Roll's lambda (simplified)
            roll_lambda = np.sqrt(rv)

            return roll_lambda.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Roll lambda RV short calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

class RangeVolumeShockOpen30Generator(VectorizedFeatureGenerator):
    """Generator for range volume shock open 30 feature."""

    def __init__(self, window: int = 30):
        config = FeatureConfig(
            name="range_volume_shock_open_30",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Range volume shock open 30 for market microstructure analysis",
            required_columns=["high", "low", "volume"],
            default_lookback=window,
            min_lookback=5,
            max_lookback=100,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate range volume shock open 30 feature."""
        try:
            high = data['high']
            low = data['low']
            volume = data['volume']

            # Calculate range
            price_range = high - low

            # Calculate volume shock
            if VECTORBT_AVAILABLE:
                volume_mean = rolling_mean(volume, window=self.window)
                volume_shock = volume / volume_mean
            else:
                volume_mean = volume.rolling(window=self.window).mean()
                volume_shock = volume / volume_mean

            # Combine range and volume shock
            combined_feature = price_range * volume_shock

            return combined_feature.fillna(0)

        except Exception as e:
            tprint(f"⚠️ Range volume shock open 30 calculation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_microstructure_feature_generators() -> List[FeatureGenerator]:
    """Create all microstructure feature generators (numpy/numba optimized, no orderbook dependency)."""
    generators = []

    # Core microstructure generators (numpy-optimized, work without orderbook data)
    generators.append(MicrostructureFeatureGenerator())
    # generators.append(BidAskSpreadGenerator())  # DISABLED: Requires bid/ask columns
    generators.append(OrderFlowImbalanceGenerator())  # Works with OHLCV
    generators.append(TradeSizeImbalanceGenerator())
    generators.append(PriceImpactGenerator())  # Works with OHLCV
    generators.append(VolumeWeightedPriceGenerator())  # Works with OHLCV
    generators.append(TradeIntensityGenerator())  # Works with OHLCV
    generators.append(LiquidityProxyGenerator())  # Works with OHLCV
    generators.append(MarketDepthGenerator())  # Optional: uses bid_size/ask_size if available

    # VectorBT-optimized generators (optional, only if VectorBT is available)
    if VECTORBT_AVAILABLE:
        try:
            for window in [10, 20, 30]:
                generators.append(VectorBTTakerBuyRatioGenerator(window))
                generators.append(VectorBTTakerSellRatioGenerator(window))
                generators.append(VectorBTMarketAggressionIndexGenerator(window))
                generators.append(VectorBTOrderFlowImbalanceGenerator(window))
                # generators.append(VectorBTBidAskImbalanceGenerator(window))  # DISABLED: Requires bid/ask columns
                generators.append(VectorBTMarketOrderFlowGenerator(window))
                generators.append(VectorBTVolumeWeightedOrderFlowGenerator(window))
                generators.append(VectorBTOrderFlowMomentumGenerator(window))
                generators.append(VectorBTOrderFlowVolatilityGenerator(window))
                generators.append(VectorBTOrderFlowTrendStrengthGenerator(window))
                generators.append(VectorBTOrderFlowConsistencyGenerator(window))
                generators.append(VectorBTOrderFlowAccelerationGenerator(window))
                generators.append(VectorBTOrderFlowJerkGenerator(window))
                generators.append(VectorBTOrderFlowRegimeGenerator(window))
        except Exception as e:
            tprint(f"⚠️ VectorBT microstructure generators failed to initialize: {e}")
    else:
        tprint("ℹ️ VectorBT not available, using core microstructure generators only (numpy-optimized)")

    # Analyst generators
    # generators.append(AnalystSpreadNormalizedGenerator())  # DISABLED: Requires bid/ask columns
    generators.append(AnalystTickImbalanceGenerator())
    generators.append(CorwinSchultzSpreadMomentumGenerator())
    generators.append(AmihudIlliquidityVWAPDistanceGenerator())
    generators.append(RollLambdaRVShortGenerator())
    generators.append(RangeVolumeShockOpen30Generator())

    return generators

def create_core_microstructure_generators() -> List[FeatureGenerator]:
    """Create core microstructure feature generators."""
    generators = []

    generators.append(MicrostructureFeatureGenerator())
    # generators.append(BidAskSpreadGenerator())  # DISABLED: Requires bid/ask columns
    generators.append(OrderFlowImbalanceGenerator())
    generators.append(TradeSizeImbalanceGenerator())
    generators.append(PriceImpactGenerator())
    generators.append(VolumeWeightedPriceGenerator())
    generators.append(TradeIntensityGenerator())
    generators.append(LiquidityProxyGenerator())
    generators.append(MarketDepthGenerator())

    return generators

def create_vectorbt_microstructure_generators() -> List[FeatureGenerator]:
    """Create VectorBT-optimized microstructure feature generators."""
    generators = []

    for window in [10, 20, 30]:
        generators.append(VectorBTTakerBuyRatioGenerator(window))
        generators.append(VectorBTTakerSellRatioGenerator(window))
        generators.append(VectorBTMarketAggressionIndexGenerator(window))
        generators.append(VectorBTOrderFlowImbalanceGenerator(window))
        # generators.append(VectorBTBidAskImbalanceGenerator(window))  # DISABLED: Requires bid/ask columns
        generators.append(VectorBTMarketOrderFlowGenerator(window))
        generators.append(VectorBTVolumeWeightedOrderFlowGenerator(window))
        generators.append(VectorBTOrderFlowMomentumGenerator(window))
        generators.append(VectorBTOrderFlowVolatilityGenerator(window))
        generators.append(VectorBTOrderFlowTrendStrengthGenerator(window))
        generators.append(VectorBTOrderFlowConsistencyGenerator(window))
        generators.append(VectorBTOrderFlowAccelerationGenerator(window))
        generators.append(VectorBTOrderFlowJerkGenerator(window))
        generators.append(VectorBTOrderFlowRegimeGenerator(window))

    return generators

def create_analyst_microstructure_generators() -> List[FeatureGenerator]:
    """Create analyst microstructure feature generators."""
    generators = []

    # generators.append(AnalystSpreadNormalizedGenerator())  # DISABLED: Requires bid/ask columns
    generators.append(AnalystTickImbalanceGenerator())
    generators.append(CorwinSchultzSpreadMomentumGenerator())
    generators.append(AmihudIlliquidityVWAPDistanceGenerator())
    generators.append(RollLambdaRVShortGenerator())
    generators.append(RangeVolumeShockOpen30Generator())

    return generators

def create_default_microstructure_generators() -> List[FeatureGenerator]:
    """Create default microstructure feature generators."""
    return create_microstructure_feature_generators()

def process_microstructure_features_batch(data: pd.DataFrame,
                                        generators: Optional[List[FeatureGenerator]] = None,
                                        use_vectorbt: bool = True,
                                        **kwargs) -> pd.DataFrame:
    """
    Process microstructure features in batch using VectorBT optimizations.

    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters

    Returns:
        DataFrame with generated microstructure features
    """
    if generators is None:
        generators = create_microstructure_feature_generators()

    if use_vectorbt and OPTIMIZATION_AVAILABLE:
        try:
            # Use unified optimization system for batch processing
            from src.feature_generation.utils.unified_optimization_system import get_unified_optimization_system
            unified_optimizer = get_unified_optimization_system()

            # Process features in batch
            result = unified_optimizer.process_features_batch(data, generators, **kwargs)
            return result

        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_microstructure_features_sequential(data, generators, **kwargs)
    else:
        return _process_microstructure_features_sequential(data, generators, **kwargs)

def _process_microstructure_features_sequential(data: pd.DataFrame,
                                              generators: List[FeatureGenerator],
                                              **kwargs) -> pd.DataFrame:
    """Process microstructure features sequentially (fallback)."""
    results = []

    for generator in generators:
        try:
            feature_result = generator._generate_feature(data, **kwargs)
            if not feature_result.empty:
                results.append(feature_result)
        except Exception as e:
            warnings.warn(f"Generator {generator.__class__.__name__} failed: {e}")
            continue

    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame(index=data.index)

__all__ = [
    # Core Microstructure Features
    'MicrostructureFeatureGenerator',
    'BidAskSpreadGenerator',
    'OrderFlowImbalanceGenerator',
    'TradeSizeImbalanceGenerator',
    'PriceImpactGenerator',
    'VolumeWeightedPriceGenerator',
    'TradeIntensityGenerator',
    'LiquidityProxyGenerator',
    'MarketDepthGenerator',

    # VectorBT-Optimized Features
    'VectorBTTakerBuyRatioGenerator',
    'VectorBTTakerSellRatioGenerator',
    'VectorBTMarketAggressionIndexGenerator',
    'VectorBTOrderFlowImbalanceGenerator',
    'VectorBTBidAskImbalanceGenerator',
    'VectorBTMarketOrderFlowGenerator',
    'VectorBTVolumeWeightedOrderFlowGenerator',
    'VectorBTOrderFlowMomentumGenerator',
    'VectorBTOrderFlowVolatilityGenerator',
    'VectorBTOrderFlowTrendStrengthGenerator',
    'VectorBTOrderFlowConsistencyGenerator',
    'VectorBTOrderFlowAccelerationGenerator',
    'VectorBTOrderFlowJerkGenerator',
    'VectorBTOrderFlowRegimeGenerator',

    # Analyst Features
    # 'AnalystSpreadNormalizedGenerator',  # DISABLED: Requires bid/ask columns
    'AnalystTickImbalanceGenerator',
    'CorwinSchultzSpreadMomentumGenerator',
    'AmihudIlliquidityVWAPDistanceGenerator',
    'RollLambdaRVShortGenerator',
    'RangeVolumeShockOpen30Generator',

    # Factory Functions
    'create_microstructure_feature_generators',
    'create_core_microstructure_generators',
    'create_vectorbt_microstructure_generators',
    'create_analyst_microstructure_generators',
    'create_default_microstructure_generators',
    'process_microstructure_features_batch'
]
