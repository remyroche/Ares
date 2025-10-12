"""
VectorBT-Optimized Order Flow Feature Generators

This module provides high-performance order flow feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Taker buy/sell ratios
- Market aggression index
- Order flow imbalance
- Bid-ask spread analysis
- Market order flow analysis
- Volume-weighted order flow
- Order flow momentum
- Order flow volatility
- Order flow trend strength
- Order flow consistency
- Order flow acceleration
- Order flow jerk
- Order flow regime detection
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union
from scipy import stats

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

logger = logging.getLogger(__name__)

class VectorBTTakerBuyRatioGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized taker buy ratio generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
        
        # Initialize VectorBT rolling optimizer for enhanced performance
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_taker_buy_ratio_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized taker buy ratio over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate taker buy ratio using VectorBT operations."""
        tprint(f"Generating VectorBT taker buy ratio feature with window {self.window}")
        
        if data.empty:
            tprint("Warning: Empty data provided for taker buy ratio calculation")
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_taker_buy_ratio_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change using VectorBT
        price_change = close.pct_change()
        
        # Simulate taker buy ratio based on price movement and volume
        buy_pressure = (price_change > 0).astype(int) * volume
        
        # Use VectorBT rolling optimizer if available, otherwise fallback to base class method
        if self.rolling_optimizer:
            try:
                total_volume = self.rolling_optimizer.rolling_sum(volume, window=self.window)
                buy_volume = self.rolling_optimizer.rolling_sum(buy_pressure, window=self.window)
            except Exception as e:
                logger.warning(f"VectorBT rolling optimizer failed: {e}, using base class method")
                total_volume = self._vectorbt_rolling_operation(volume, 'sum', window=self.window)
                buy_volume = self._vectorbt_rolling_operation(buy_pressure, 'sum', window=self.window)
        else:
            total_volume = self._vectorbt_rolling_operation(volume, 'sum', window=self.window)
            buy_volume = self._vectorbt_rolling_operation(buy_pressure, 'sum', window=self.window)
        
        # Calculate ratio with safe division
        ratio = safe_divide(buy_volume, total_volume)
        
        return ratio.rename(f'vectorbt_taker_buy_ratio_{self.window}')


class VectorBTTakerSellRatioGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized taker sell ratio generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_taker_sell_ratio_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized taker sell ratio over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate taker sell ratio using VectorBT operations."""
        tprint(f"Generating VectorBT taker sell ratio feature with window {self.window}")
        
        if data.empty:
            tprint("Warning: Empty data provided for taker sell ratio calculation")
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_taker_sell_ratio_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change using VectorBT
        price_change = close.pct_change()
        
        # Simulate taker sell ratio based on price movement and volume
        sell_pressure = (price_change < 0).astype(int) * volume
        
        # Use VectorBT rolling operations
        total_volume = self._vectorbt_rolling_operation(volume, 'sum', window=self.window)
        sell_volume = self._vectorbt_rolling_operation(sell_pressure, 'sum', window=self.window)
        
        # Calculate ratio with safe division
        ratio = safe_divide(sell_volume, total_volume)
        
        return ratio.rename(f'vectorbt_taker_sell_ratio_{self.window}')


class VectorBTMarketAggressionIndexGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized market aggression index generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_market_aggression_index_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized market aggression index over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market aggression index using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_market_aggression_index_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price velocity using VectorBT
        price_velocity = close.pct_change().abs()
        
        # Calculate aggression (price velocity * volume)
        aggression = price_velocity * volume
        
        # Use VectorBT rolling mean
        aggression_index = self._vectorbt_rolling_operation(aggression, 'mean', window=self.window)
        
        return aggression_index.rename(f'vectorbt_market_aggression_index_{self.window}')


class VectorBTOrderFlowImbalanceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow imbalance generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_imbalance_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow imbalance over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow imbalance using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_imbalance_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change using VectorBT
        price_change = close.pct_change()
        
        # Calculate buy and sell volumes
        buy_volume = (price_change > 0).astype(int) * volume
        sell_volume = (price_change < 0).astype(int) * volume
        
        # Use VectorBT rolling operations
        buy_sum = self._vectorbt_rolling_operation(buy_volume, 'sum', window=self.window)
        sell_sum = self._vectorbt_rolling_operation(sell_volume, 'sum', window=self.window)
        
        # Calculate imbalance with safe division
        imbalance = safe_divide(buy_sum - sell_sum, buy_sum + sell_sum)
        
        return imbalance.rename(f'vectorbt_order_flow_imbalance_{self.window}')


class VectorBTBidAskImbalanceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized bid-ask imbalance generator."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="vectorbt_bid_ask_imbalance",
            category=FeatureCategory.ORDER_FLOW,
            description="VectorBT-optimized bid-ask imbalance ((bid_size - ask_size) / (bid_size + ask_size))",
            required_columns=["bid", "ask"],
            optional_columns=["close", "volume"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=100,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate bid-ask imbalance using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='vectorbt_bid_ask_imbalance')
        
        if 'bid' in data.columns and 'ask' in data.columns:
            bid_size = data['bid']
            ask_size = data['ask']
            
            # Calculate imbalance with safe division
            imbalance = safe_divide(bid_size - ask_size, bid_size + ask_size)
            
            return imbalance.rename('vectorbt_bid_ask_imbalance')
        else:
            # Return neutral value if bid/ask data not available
            return pd.Series([0.0] * len(data), index=data.index, name='vectorbt_bid_ask_imbalance')


class VectorBTMarketOrderFlowGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized market order flow generator."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="vectorbt_market_order_flow",
            category=FeatureCategory.ORDER_FLOW,
            description="VectorBT-optimized market order flow (market_buys - market_sells)",
            required_columns=["market_buys", "market_sells"],
            optional_columns=["close", "volume"],
            default_lookback=20,
            min_lookback=10,
            max_lookback=100,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market order flow using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='vectorbt_market_order_flow')
        
        if 'market_buys' in data.columns and 'market_sells' in data.columns:
            market_buys = data['market_buys']
            market_sells = data['market_sells']
            
            # Calculate market order flow
            order_flow = market_buys - market_sells
            
            return order_flow.rename('vectorbt_market_order_flow')
        else:
            # Return neutral value if market order data not available
            return pd.Series([0.0] * len(data), index=data.index, name='vectorbt_market_order_flow')


class VectorBTVolumeWeightedOrderFlowGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume-weighted order flow generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volume_weighted_order_flow_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized volume-weighted order flow over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume-weighted order flow using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volume_weighted_order_flow_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate volume-weighted order flow
        vw_order_flow = price_change * volume
        
        # Use VectorBT rolling sum
        vw_order_flow_sum = self._vectorbt_rolling_operation(vw_order_flow, 'sum', window=self.window)
        
        return vw_order_flow_sum.rename(f'vectorbt_volume_weighted_order_flow_{self.window}')


class VectorBTOrderFlowMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow momentum generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_momentum_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow momentum over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow momentum using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_momentum_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow momentum
        order_flow = price_change * volume
        
        # Use VectorBT rolling mean for momentum
        momentum = self._vectorbt_rolling_operation(order_flow, 'mean', window=self.window)
        
        return momentum.rename(f'vectorbt_order_flow_momentum_{self.window}')


class VectorBTOrderFlowVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow volatility generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_volatility_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow volatility over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow volatility using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_volatility_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow
        order_flow = price_change * volume
        
        # Use VectorBT rolling std for volatility
        volatility = self._vectorbt_rolling_operation(order_flow, 'std', window=self.window)
        
        return volatility.rename(f'vectorbt_order_flow_volatility_{self.window}')


class VectorBTOrderFlowTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow trend strength generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_trend_strength_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow trend strength over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow trend strength using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_trend_strength_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow
        order_flow = price_change * volume
        
        # Calculate trend strength using rolling correlation with time
        time_index = pd.Series(range(len(order_flow)), index=order_flow.index)
        trend_strength = self._vectorbt_rolling_operation(
            order_flow, 'corr', window=self.window, other=time_index
        )
        
        return trend_strength.rename(f'vectorbt_order_flow_trend_strength_{self.window}')


class VectorBTOrderFlowConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow consistency generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_consistency_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow consistency over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow consistency using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_consistency_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow
        order_flow = price_change * volume
        
        # Calculate consistency as inverse of volatility
        volatility = self._vectorbt_rolling_operation(order_flow, 'std', window=self.window)
        consistency = 1.0 / (volatility + 1e-8)  # Add small epsilon to avoid division by zero
        
        return consistency.rename(f'vectorbt_order_flow_consistency_{self.window}')


class VectorBTOrderFlowAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow acceleration generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_acceleration_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow acceleration over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow acceleration using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_acceleration_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow
        order_flow = price_change * volume
        
        # Calculate acceleration (second derivative)
        order_flow_diff = order_flow.diff()
        acceleration = order_flow_diff.diff()
        
        # Use VectorBT rolling mean for smoothing
        acceleration_smooth = self._vectorbt_rolling_operation(acceleration, 'mean', window=self.window)
        
        return acceleration_smooth.rename(f'vectorbt_order_flow_acceleration_{self.window}')


class VectorBTOrderFlowJerkGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow jerk generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_jerk_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow jerk over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow jerk using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_jerk_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow
        order_flow = price_change * volume
        
        # Calculate jerk (third derivative)
        order_flow_diff = order_flow.diff()
        order_flow_diff2 = order_flow_diff.diff()
        jerk = order_flow_diff2.diff()
        
        # Use VectorBT rolling mean for smoothing
        jerk_smooth = self._vectorbt_rolling_operation(jerk, 'mean', window=self.window)
        
        return jerk_smooth.rename(f'vectorbt_order_flow_jerk_{self.window}')


class VectorBTOrderFlowRegimeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized order flow regime detection generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_order_flow_regime_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"VectorBT-optimized order flow regime detection over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow regime using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_order_flow_regime_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate price change
        price_change = close.pct_change()
        
        # Calculate order flow
        order_flow = price_change * volume
        
        # Calculate regime based on order flow momentum
        momentum = self._vectorbt_rolling_operation(order_flow, 'mean', window=self.window)
        volatility = self._vectorbt_rolling_operation(order_flow, 'std', window=self.window)
        
        # Regime classification: 1 for bullish, -1 for bearish, 0 for neutral
        regime = np.where(momentum > volatility, 1, np.where(momentum < -volatility, -1, 0))
        
        return pd.Series(regime, index=data.index, name=f'vectorbt_order_flow_regime_{self.window}')


def create_vectorbt_order_flow_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized order flow feature generators."""
    generators = []
    
    # Basic order flow features
    for window in [5, 10, 20]:
        generators.extend([
            VectorBTTakerBuyRatioGenerator(window),
            VectorBTTakerSellRatioGenerator(window),
            VectorBTMarketAggressionIndexGenerator(window),
            VectorBTOrderFlowImbalanceGenerator(window),
            VectorBTVolumeWeightedOrderFlowGenerator(window),
            VectorBTOrderFlowMomentumGenerator(window),
            VectorBTOrderFlowVolatilityGenerator(window),
            VectorBTOrderFlowTrendStrengthGenerator(window),
            VectorBTOrderFlowConsistencyGenerator(window),
            VectorBTOrderFlowAccelerationGenerator(window),
            VectorBTOrderFlowJerkGenerator(window),
            VectorBTOrderFlowRegimeGenerator(window),
        ])
    
    # Specialized features
    generators.extend([
        VectorBTBidAskImbalanceGenerator(),
        VectorBTMarketOrderFlowGenerator(),
    ])
    
    return generators


def create_default_vectorbt_order_flow_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized order flow feature generators."""
    return create_vectorbt_order_flow_generators()


class VectorBTOrderFlowBatchProcessor:
    """Batch processor for VectorBT order flow features with unified vectorization management."""
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        """Initialize batch processor with VectorBT rolling optimizer."""
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=enable_gpu, 
                enable_parallel=enable_parallel
            )
        else:
            self.rolling_optimizer = None
            logger.warning("VectorBT rolling optimizer not available, using pandas fallback")
        
        # Performance tracking
        self.batch_stats = {
            'total_batches': 0,
            'total_features': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'processing_time': 0.0
        }
    
    def process_batch_rolling_operations(self, data: pd.DataFrame, 
                                       operations_config: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process multiple rolling operations in batch for efficiency.
        
        Args:
            data: Input OHLCV data
            operations_config: List of operation configurations
            
        Returns:
            DataFrame with all computed features
        """
        import time
        start_time = time.time()
        
        results = {}
        
        if self.rolling_optimizer:
            try:
                # Use VectorBT rolling optimizer for batch processing
                for config in operations_config:
                    feature_name = config['name']
                    operation = config['operation']
                    window = config['window']
                    column = config.get('column', 'close')
                    
                    if column not in data.columns:
                        logger.warning(f"Column {column} not found for feature {feature_name}")
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                        continue
                    
                    try:
                        if operation == 'mean':
                            result = self.rolling_optimizer.rolling_mean(data[column], window=window)
                        elif operation == 'std':
                            result = self.rolling_optimizer.rolling_std(data[column], window=window)
                        elif operation == 'sum':
                            result = self.rolling_optimizer.rolling_sum(data[column], window=window)
                        elif operation == 'min':
                            result = self.rolling_optimizer.rolling_min(data[column], window=window)
                        elif operation == 'max':
                            result = self.rolling_optimizer.rolling_max(data[column], window=window)
                        else:
                            logger.warning(f"Unknown operation: {operation}")
                            result = pd.Series(np.nan, index=data.index)
                        
                        results[feature_name] = result
                        self.batch_stats['vectorbt_operations'] += 1
                        
                    except Exception as e:
                        logger.warning(f"VectorBT operation {operation} failed: {e}, using pandas fallback")
                        # Fallback to pandas
                        result = self._pandas_rolling_operation(data[column], operation, window)
                        results[feature_name] = result
                        self.batch_stats['pandas_fallbacks'] += 1
                
            except Exception as e:
                logger.error(f"VectorBT batch processing failed: {e}, using pandas fallback")
                # Fallback to pandas for all operations
                for config in operations_config:
                    feature_name = config['name']
                    operation = config['operation']
                    window = config['window']
                    column = config.get('column', 'close')
                    
                    if column in data.columns:
                        result = self._pandas_rolling_operation(data[column], operation, window)
                        results[feature_name] = result
                        self.batch_stats['pandas_fallbacks'] += 1
                    else:
                        results[feature_name] = pd.Series(np.nan, index=data.index)
        else:
            # Use pandas fallback
            for config in operations_config:
                feature_name = config['name']
                operation = config['operation']
                window = config['window']
                column = config.get('column', 'close')
                
                if column in data.columns:
                    result = self._pandas_rolling_operation(data[column], operation, window)
                    results[feature_name] = result
                    self.batch_stats['pandas_fallbacks'] += 1
                else:
                    results[feature_name] = pd.Series(np.nan, index=data.index)
        
        # Update statistics
        self.batch_stats['total_batches'] += 1
        self.batch_stats['total_features'] += len(operations_config)
        self.batch_stats['processing_time'] += time.time() - start_time
        
        return pd.DataFrame(results, index=data.index)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, window: int) -> pd.Series:
        """Fallback pandas rolling operation."""
        rolling_obj = data.rolling(window=window)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        else:
            return pd.Series(np.nan, index=data.index)
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        stats = self.batch_stats.copy()
        if stats['total_batches'] > 0:
            stats['avg_time_per_batch'] = stats['processing_time'] / stats['total_batches']
            stats['avg_features_per_batch'] = stats['total_features'] / stats['total_batches']
        return stats
    
    def reset_stats(self):
        """Reset batch processing statistics."""
        self.batch_stats = {
            'total_batches': 0,
            'total_features': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'processing_time': 0.0
        }


def create_unified_vectorization_manager(enable_gpu: bool = False, enable_parallel: bool = True) -> VectorBTOrderFlowBatchProcessor:
    """Create a unified vectorization manager for order flow features."""
    return VectorBTOrderFlowBatchProcessor(enable_gpu=enable_gpu, enable_parallel=enable_parallel)


# Export all generators
__all__ = [
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
    'VectorBTOrderFlowBatchProcessor',
    'create_vectorbt_order_flow_generators',
    'create_default_vectorbt_order_flow_generators',
    'create_unified_vectorization_manager'
]