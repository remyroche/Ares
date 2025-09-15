"""Order Flow features"""
import pandas as pd
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator

class TakerBuyRatioGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"taker_buy_ratio_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Taker buy ratio over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # Simulate taker buy ratio based on price movement and volume
        price_change = close.pct_change()
        buy_pressure = (price_change > 0).astype(int) * volume
        total_volume = volume.rolling(window=self.window).sum()
        buy_volume = buy_pressure.rolling(window=self.window).sum()
        
        return buy_volume / total_volume.replace(0, 1)

class TakerSellRatioGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"taker_sell_ratio_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Taker sell ratio over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # Simulate taker sell ratio based on price movement and volume
        price_change = close.pct_change()
        sell_pressure = (price_change < 0).astype(int) * volume
        total_volume = volume.rolling(window=self.window).sum()
        sell_volume = sell_pressure.rolling(window=self.window).sum()
        
        return sell_volume / total_volume.replace(0, 1)

class MarketAggressionIndexGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"market_aggression_index_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Market aggression index over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # Calculate market aggression based on price velocity and volume
        price_velocity = close.pct_change().abs()
        aggression = price_velocity * volume
        
        return aggression.rolling(window=self.window).mean()

class OrderFlowImbalanceGenerator(FeatureGenerator):
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"order_flow_imbalance_{window}",
            category=FeatureCategory.ORDER_FLOW,
            description=f"Order flow imbalance over {window} periods",
            required_columns=["close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window
        )
        super().__init__(config)
        self.window = window
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # Calculate order flow imbalance
        price_change = close.pct_change()
        buy_volume = (price_change > 0).astype(int) * volume
        sell_volume = (price_change < 0).astype(int) * volume
        
        buy_sum = buy_volume.rolling(window=self.window).sum()
        sell_sum = sell_volume.rolling(window=self.window).sum()
        
        return (buy_sum - sell_sum) / (buy_sum + sell_sum).replace(0, 1)

def create_default_order_flow_generators() -> List[FeatureGenerator]:
    generators = []
    windows = [5, 10, 20]
    
    for window in windows:
        generators.extend([
            TakerBuyRatioGenerator(window),
            TakerSellRatioGenerator(window),
            MarketAggressionIndexGenerator(window),
            OrderFlowImbalanceGenerator(window),
        ])
    
    return generators