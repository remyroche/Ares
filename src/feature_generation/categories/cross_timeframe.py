"""Cross-timeframe features"""
import pandas as pd
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator

class CrossTimeframeRSIGenerator(FeatureGenerator):
    def __init__(self, period: int = 14, timeframe: str = "1m"):
        config = FeatureConfig(
            name=f"rsi_{period}_{timeframe}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"RSI {period} on {timeframe} timeframe",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config)
        self.period = period
        self.timeframe = timeframe
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class CrossTimeframeMACDGenerator(FeatureGenerator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, timeframe: str = "1m"):
        config = FeatureConfig(
            name=f"macd_{fast}_{slow}_{signal}_{timeframe}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"MACD {fast}/{slow}/{signal} on {timeframe} timeframe",
            required_columns=["close"],
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.timeframe = timeframe
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate MACD
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        
        return macd

class CrossTimeframeBollingerBandsGenerator(FeatureGenerator):
    def __init__(self, period: int = 20, std_dev: float = 2.0, timeframe: str = "1m"):
        config = FeatureConfig(
            name=f"bb_upper_{period}_{timeframe}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Bollinger Bands upper {period} on {timeframe} timeframe",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev
        self.timeframe = timeframe
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        
        # Calculate Bollinger Bands
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        
        return upper_band

class CrossTimeframeSMAGenerator(FeatureGenerator):
    def __init__(self, period: int = 20, timeframe: str = "1m"):
        config = FeatureConfig(
            name=f"sma_{period}_{timeframe}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"SMA {period} on {timeframe} timeframe",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
        self.timeframe = timeframe
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        return close.rolling(window=self.period).mean()

class CrossTimeframeEMAGenerator(FeatureGenerator):
    def __init__(self, period: int = 20, timeframe: str = "1m"):
        config = FeatureConfig(
            name=f"ema_{period}_{timeframe}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"EMA {period} on {timeframe} timeframe",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config)
        self.period = period
        self.timeframe = timeframe
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        return close.ewm(span=self.period).mean()

class CrossTimeframeATRGenerator(FeatureGenerator):
    def __init__(self, period: int = 14, timeframe: str = "1m"):
        config = FeatureConfig(
            name=f"atr_{period}_{timeframe}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"ATR {period} on {timeframe} timeframe",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
        self.timeframe = timeframe
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=self.period).mean()
        return atr

def create_default_cross_timeframe_generators() -> List[FeatureGenerator]:
    generators = []
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    # RSI across timeframes
    for timeframe in timeframes:
        generators.append(CrossTimeframeRSIGenerator(14, timeframe))
    
    # MACD across timeframes
    for timeframe in timeframes:
        generators.append(CrossTimeframeMACDGenerator(12, 26, 9, timeframe))
    
    # Bollinger Bands across timeframes
    for timeframe in timeframes:
        generators.append(CrossTimeframeBollingerBandsGenerator(20, 2.0, timeframe))
    
    # SMA across timeframes
    periods = [5, 10, 20, 50, 100, 200]
    for timeframe in timeframes:
        for period in periods:
            generators.append(CrossTimeframeSMAGenerator(period, timeframe))
    
    # EMA across timeframes
    ema_periods = [8, 12, 21, 26, 50, 100]
    for timeframe in timeframes:
        for period in ema_periods:
            generators.append(CrossTimeframeEMAGenerator(period, timeframe))
    
    # ATR across timeframes
    atr_periods = [7, 14, 21, 30]
    for timeframe in timeframes:
        for period in atr_periods:
            generators.append(CrossTimeframeATRGenerator(period, timeframe))
    
    return generators