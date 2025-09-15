"""
Volatility Feature Generator

This module provides feature generators for volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

class VolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Comprehensive volatility-based features including Bollinger Bands, ATR, and volatility measures",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "bb_periods": [20],
                "bb_std": [2.0],
                "atr_periods": [14],
                "volatility_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'VolatilityFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        volatility = self._calculate_volatility(close_prices, period=20)
        return pd.Series(volatility, index=data.index, name='volatility_20')
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        returns = np.diff(np.log(prices))
        volatility = pd.Series(returns).rolling(window=period-1).std().values
        return np.concatenate([[np.nan], volatility])

class BollingerBandsGenerator(FeatureGenerator):
    """Generator for Bollinger Bands."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        config = FeatureConfig(
            name=f"bb_upper_{period}_{std_dev}",
            category=FeatureCategory.VOLATILITY,
            description=f"Bollinger Bands Upper with period={period}, std={std_dev}",
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
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper_band = sma + (std * self.std_dev)
        return upper_band

class ATRGenerator(FeatureGenerator):
    """Generator for Average True Range."""
    
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"atr_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Average True Range over {period} periods",
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
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=self.period).mean()
        
        return atr

def create_volatility_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of volatility feature generators."""
    if periods is None:
        periods = {
            'bb': [20],
            'atr': [14],
            'volatility': [10, 20]
        }
    
    generators = []
    
    # Bollinger Bands generators
    for period in periods.get('bb', [20]):
        generators.append(BollingerBandsGenerator(period))
    
    # ATR generators
    for period in periods.get('atr', [14]):
        generators.append(ATRGenerator(period))
    
    return generators

def create_default_volatility_generators() -> List[FeatureGenerator]:
    return create_volatility_generators()