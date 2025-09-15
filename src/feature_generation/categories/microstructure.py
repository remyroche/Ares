"""
Microstructure Feature Generator

This module provides feature generators for microstructure-based indicators,
including bid-ask spread, order flow, and other high-frequency features.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

class MicrostructureFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for microstructure-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="microstructure_features",
            category=FeatureCategory.MICROSTRUCTURE,
            description="Comprehensive microstructure features including bid-ask spread, order flow, and trade intensity",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume", "bid", "ask"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "spread_windows": [5, 10, 20],
                "order_flow_windows": [5, 10, 20],
                "trade_intensity_windows": [5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'MicrostructureFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        ms = np.zeros_like(close_prices)
        return pd.Series(ms, index=data.index, name='ms_placeholder')

# Bid-Ask Spread Generator
class BidAskSpreadGenerator(FeatureGenerator):
    """Generator for bid-ask spread features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'bid' not in required_columns:
            required_columns.append('bid')
        if 'ask' not in required_columns:
            required_columns.append('ask')
        
        config = FeatureConfig(
            name=f"bid_ask_spread_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Bid-ask spread over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate bid-ask spread."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            bid = data['bid']
            ask = data['ask']
            spread = ask - bid
        else:
            base_values = self.base_calculator.calculate(data)
            spread = base_values.rolling(window=self.window).std()
        return spread

# Order Flow Imbalance Generator
class OrderFlowImbalanceGenerator(FeatureGenerator):
    """Generator for order flow imbalance features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"order_flow_imbalance_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Order flow imbalance over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow imbalance."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        order_flow_imbalance = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return order_flow_imbalance

# Trade Size Imbalance Generator
class TradeSizeImbalanceGenerator(FeatureGenerator):
    """Generator for trade size imbalance features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"trade_size_imbalance_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Trade size imbalance over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trade size imbalance."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        trade_size_imbalance = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return trade_size_imbalance

# Price Impact Generator
class PriceImpactGenerator(FeatureGenerator):
    """Generator for price impact features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"price_impact_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Price impact over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price impact."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        price_impact = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return price_impact

# Volume Weighted Price Generator
class VolumeWeightedPriceGenerator(FeatureGenerator):
    """Generator for volume weighted price features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"volume_weighted_price_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Volume weighted price over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume weighted price."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        volume_weighted_price = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return volume_weighted_price

# Trade Intensity Generator
class TradeIntensityGenerator(FeatureGenerator):
    """Generator for trade intensity features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"trade_intensity_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Trade intensity over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trade intensity."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        trade_intensity = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return trade_intensity

# Liquidity Proxy Generator
class LiquidityProxyGenerator(FeatureGenerator):
    """Generator for liquidity proxy features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"liquidity_proxy_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Liquidity proxy over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate liquidity proxy."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        liquidity_proxy = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return liquidity_proxy

# Market Depth Generator
class MarketDepthGenerator(FeatureGenerator):
    """Generator for market depth features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"market_depth_{window}_{base_calculation.value}",
            category=FeatureCategory.MICROSTRUCTURE,
            description=f"Market depth over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate market depth."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        market_depth = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return market_depth

def create_default_microstructure_generators() -> List[FeatureGenerator]:
    """Create default microstructure feature generators."""
    windows = [5, 10, 20]
    
    generators = []
    
    # Create generators for each window
    for window in windows:
        generators.extend([
            BidAskSpreadGenerator(window),
            OrderFlowImbalanceGenerator(window),
            TradeSizeImbalanceGenerator(window),
            PriceImpactGenerator(window),
            VolumeWeightedPriceGenerator(window),
            TradeIntensityGenerator(window),
            LiquidityProxyGenerator(window),
            MarketDepthGenerator(window),
        ])
    
    return generators