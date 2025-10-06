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
from ...utils.math_validation import validate_finite

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
class BidAskSpreadGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate bid-ask spread or fallback to price volatility when bid/ask not available."""
        # Check if bid and ask columns are available
        if 'bid' in data.columns and 'ask' in data.columns:
            if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
                bid = data['bid']
                ask = data['ask']
                spread = ask - bid
            else:
                base_values = self.base_calculator.calculate(data)
                spread = base_values.rolling(window=self.window).std()
        else:
            # Fallback: use high-low spread as proxy for bid-ask spread
            self.logger.warning(f"⚠️ Bid/ask columns not available, using high-low spread as proxy")
            if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
                spread = data['high'] - data['low']
            else:
                base_values = self.base_calculator.calculate(data)
                spread = base_values.rolling(window=self.window).std()

        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(spread.values, f"BidAskSpread_{self.window}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(spread.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)

                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"

                # Only log once per feature globally to reduce verbosity
                feature_key = f"BidAskSpread_{self.window}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(BidAskSpreadGenerator, '_logged_warnings'):
                    BidAskSpreadGenerator._logged_warnings = set()
                if feature_key not in BidAskSpreadGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    BidAskSpreadGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")

        return spread

# Order Flow Imbalance Generator
class OrderFlowImbalanceGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate order flow imbalance."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        order_flow_imbalance = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return order_flow_imbalance

# Trade Size Imbalance Generator
class TradeSizeImbalanceGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trade size imbalance."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        trade_size_imbalance = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return trade_size_imbalance

# Price Impact Generator
class PriceImpactGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate price impact."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        price_impact = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return price_impact

# Volume Weighted Price Generator
class VolumeWeightedPriceGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume weighted price."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        volume_weighted_price = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return volume_weighted_price

# Trade Intensity Generator
class TradeIntensityGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trade intensity."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        trade_intensity = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return trade_intensity

# Liquidity Proxy Generator
class LiquidityProxyGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate liquidity proxy."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        liquidity_proxy = (base_values * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        return liquidity_proxy

# Market Depth Generator
class MarketDepthGenerator(VectorizedFeatureGenerator):
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
        super().__init__(config, enable_matrix_ops=True)
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

    # Analyst Features - Microstructure generators
    class AnalystSpreadNormalizedGenerator(VectorizedFeatureGenerator):
        """Generator for normalized spread feature."""

        def __init__(self):
            config = FeatureConfig(
                name="analyst_spread_normalized",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Analyst normalized bid-ask spread using ATR",
                required_columns=["high", "low", "close"],
                default_lookback=20,
                min_lookback=10,
                max_lookback=100,
                parameters={}
            )
            super().__init__(config, enable_matrix_ops=True)

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            """Generate normalized spread feature."""
            # Spread calculation (using high-low as proxy)
            spread = (data['high'] - data['low']) / data['close']

            # ATR for normalization (using simplified ATR calculation)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()

            spread_normalized = spread / atr.replace(0, 1)
            return spread_normalized

    class AnalystTickImbalanceGenerator(VectorizedFeatureGenerator):
        """Generator for tick imbalance feature."""

        def __init__(self, lookback: int = 100):
            config = FeatureConfig(
                name="analyst_tick_imbalance",
                category=FeatureCategory.MICROSTRUCTURE,
                description="Analyst tick imbalance ((upticks - downticks) / total_ticks)",
                required_columns=["close"],
                default_lookback=lookback,
                min_lookback=50,
                max_lookback=200,
                parameters={"lookback": lookback}
            )
            super().__init__(config, enable_matrix_ops=True)
            self.lookback = lookback

        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            """Generate tick imbalance feature."""
            price_changes = data['close'].diff()

            # Count upticks vs downticks in rolling window
            upticks = (price_changes > 0).rolling(self.lookback).sum()
            downticks = (price_changes < 0).rolling(self.lookback).sum()
            total_ticks = upticks + downticks

            tick_imbalance = (upticks - downticks) / total_ticks.replace(0, 1)
            return tick_imbalance

    generators.append(AnalystSpreadNormalizedGenerator())
    generators.append(AnalystTickImbalanceGenerator())

    return generators