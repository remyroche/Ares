"""Legacy features

Legacy features are traditional technical indicators that have been used in 
financial analysis for decades. These include classic indicators like:
- Traditional RSI implementations
- Classic MACD calculations
- Original Bollinger Bands formulations
- Standard moving averages
- Conventional oscillators

These features maintain backward compatibility with existing trading systems
and provide a baseline for comparison with newer, enhanced indicators.
"""
import pandas as pd
import numpy as np
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory

class LegacyRSIGenerator(FeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_rsi_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy RSI {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Traditional RSI calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class LegacyMACDGenerator(FeatureGenerator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        config = FeatureConfig(
            name=f"legacy_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy MACD {fast}/{slow}/{signal} - traditional implementation",
            required_columns=["close"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Traditional MACD calculation
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        
        return macd

class LegacyBollingerBandsGenerator(FeatureGenerator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        config = FeatureConfig(
            name=f"legacy_bollinger_{period}_{std_dev}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Bollinger Bands {period}/{std_dev} - traditional implementation",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Traditional Bollinger Bands calculation
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        
        return upper_band

class LegacySMAGenerator(FeatureGenerator):
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"legacy_sma_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy SMA {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        return close.rolling(window=self.period).mean()

class LegacyEMAGenerator(FeatureGenerator):
    def __init__(self, period: int = 21):
        config = FeatureConfig(
            name=f"legacy_ema_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy EMA {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        return close.ewm(span=self.period).mean()

class LegacyATRGenerator(FeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_atr_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy ATR {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Traditional ATR calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=self.period).mean()
        return atr

class LegacyStochasticGenerator(FeatureGenerator):
    def __init__(self, k_period: int = 14, d_period: int = 3):
        config = FeatureConfig(
            name=f"legacy_stochastic_{k_period}_{d_period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Stochastic {k_period}/{d_period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period
        )
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Traditional Stochastic calculation
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        return k_percent

class LegacyOBVGenerator(FeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="legacy_obv",
            category=FeatureCategory.LEGACY,
            description="Legacy OBV - traditional implementation",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # Traditional OBV calculation
        price_change = close.diff()
        obv = volume.copy()
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        
        return obv.cumsum()

def create_default_legacy_generators() -> List[FeatureGenerator]:
    """
    Create default legacy feature generators.
    
    Legacy features include traditional implementations of classic indicators
    that have been used in technical analysis for decades. These provide
    backward compatibility and serve as benchmarks for enhanced versions.
    """
    generators = []
    
    # Classic indicators with standard parameters
    generators.extend([
        LegacyRSIGenerator(14),
        LegacyMACDGenerator(12, 26, 9),
        LegacyBollingerBandsGenerator(20, 2.0),
        LegacySMAGenerator(20),
        LegacyEMAGenerator(21),
        LegacyATRGenerator(14),
        LegacyStochasticGenerator(14, 3),
        LegacyOBVGenerator(),
    ])
    
    # Additional legacy moving averages
    sma_periods = [5, 10, 50, 100, 200]
    for period in sma_periods:
        generators.append(LegacySMAGenerator(period))
    
    # Additional legacy EMAs
    ema_periods = [8, 12, 26, 50, 100]
    for period in ema_periods:
        generators.append(LegacyEMAGenerator(period))
    
    # Additional legacy RSI periods
    rsi_periods = [9, 21, 25]
    for period in rsi_periods:
        generators.append(LegacyRSIGenerator(period))
    
    return generators