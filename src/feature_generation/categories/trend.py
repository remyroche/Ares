"""
Trend Feature Generator

This module provides feature generators for trend-based indicators,
including moving averages, trend lines, and trend strength measures.
Supports different base calculations: price returns, returns-based VWAP, etc.
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
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class TrendFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for trend-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="trend_features",
            category=FeatureCategory.TREND,
            description="Comprehensive trend-based features including moving averages and trend indicators",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "sma_periods": [5, 10, 20, 50],
                "ema_periods": [12, 26],
                "trend_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'TrendFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close_prices = data['close'].values
        sma = self._calculate_sma(close_prices, period=20)
        return pd.Series(sma, index=data.index, name='sma_20')
    
    def _calculate_sma(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        sma = pd.Series(prices).rolling(window=period).mean().values
        return sma

class SMAGenerator(FeatureGenerator):
    """Generator for Simple Moving Average with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 **base_kwargs):
        """
        Initialize SMA generator.
        
        Args:
            period: SMA period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"sma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Simple Moving Average over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate SMA on base values
        sma = base_values.rolling(window=self.period).mean()
        
        return sma

class EMAGenerator(FeatureGenerator):
    """Generator for Exponential Moving Average."""
    
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"ema_{period}",
            category=FeatureCategory.TREND,
            description=f"Exponential Moving Average over {period} periods",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        ema = close.ewm(span=self.period).mean()
        return ema

def create_trend_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of trend feature generators."""
    if periods is None:
        periods = {
            'sma': [5, 10, 20, 50],
            'ema': [12, 26]
        }
    
    generators = []
    
    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))
    
    # EMA generators
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))
    
    return generators

def create_default_trend_generators() -> List[FeatureGenerator]:
    return create_trend_generators()