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
    
    @staticmethod
    def _normalize_series(series: pd.Series, window: int) -> pd.Series:
        rolling_std = series.rolling(window=max(window, 5), min_periods=1).std().replace(0.0, np.nan)
        normalized = series / rolling_std
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return np.tanh(normalized.clip(-6.0, 6.0))

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='microstructure_signal')

        close = data['close'].astype(float)
        high = data['high'].astype(float) if 'high' in data.columns else close
        low = data['low'].astype(float) if 'low' in data.columns else close
        volume = data['volume'].astype(float) if 'volume' in data.columns else None
        bid = data['bid'].astype(float) if 'bid' in data.columns else None
        ask = data['ask'].astype(float) if 'ask' in data.columns else None

        params = self.config.parameters or {}
        spread_windows = [window for window in params.get('spread_windows', [self.config.default_lookback]) if window and window > 1]
        order_flow_windows = [window for window in params.get('order_flow_windows', [self.config.default_lookback]) if window and window > 1]
        intensity_windows = [window for window in params.get('trade_intensity_windows', [self.config.default_lookback]) if window and window > 1]

        aggregated = pd.Series(0.0, index=data.index, dtype=float)
        contributions = 0

        for window in spread_windows:
            if bid is not None and ask is not None:
                raw_spread = (ask - bid).rolling(window=window, min_periods=window).mean()
            else:
                raw_spread = (high - low).rolling(window=window, min_periods=window).mean()
            spread_signal = self._normalize_series(raw_spread.fillna(0.0), window)
            aggregated = aggregated.add(spread_signal, fill_value=0.0)
            contributions += 1

        base_volume = volume if volume is not None else pd.Series(1.0, index=data.index)
        signed_volume = base_volume * np.sign(close.diff().fillna(0.0))
        for window in order_flow_windows:
            flow_signal = signed_volume.rolling(window=window, min_periods=window).sum()
            normalized_flow = self._normalize_series(flow_signal.fillna(0.0), window)
            aggregated = aggregated.add(normalized_flow, fill_value=0.0)
            contributions += 1

        for window in intensity_windows:
            if volume is not None:
                intensity = volume.rolling(window=window, min_periods=window).mean()
            else:
                intensity = close.diff().abs().rolling(window=window, min_periods=window).mean()
            normalized_intensity = self._normalize_series(intensity.fillna(0.0), window)
            aggregated = aggregated.add(normalized_intensity, fill_value=0.0)
            contributions += 1

        if not contributions:
            return pd.Series(0.0, index=data.index, name='microstructure_signal')

        signal = aggregated / float(contributions)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.clip(-1.0, 1.0).rename('microstructure_signal')

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