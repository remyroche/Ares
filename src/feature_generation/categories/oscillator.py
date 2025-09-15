"""
Oscillator Feature Generator

This module provides feature generators for oscillator indicators,
including CCI, ADX, Aroon, Ultimate Oscillator, KST, APO, CMO, NATR, PFE, T3, KAMA, and more.
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

class OscillatorFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for oscillator-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="oscillator_features",
            category=FeatureCategory.OSCILLATOR,
            description="Comprehensive oscillator-based features including Stochastic and Williams %R",
            required_columns=["close"],
            optional_columns=["high", "low"],
            default_lookback=14,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "stochastic_periods": [14],
                "williams_periods": [14]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'OscillatorFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Placeholder implementation
        close_prices = data['close'].values
        oscillator = np.zeros_like(close_prices)
        return pd.Series(oscillator, index=data.index, name='oscillator_placeholder')

def create_oscillator_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of oscillator feature generators."""
    if periods is None:
        periods = {
            'stochastic': [14],
            'williams': [14]
        }
    
    generators = []
    return generators

def create_default_oscillator_generators() -> List[FeatureGenerator]:
    return create_oscillator_generators()

# CCI (Commodity Channel Index)
class CCIGenerator(FeatureGenerator):
    """Generator for CCI (Commodity Channel Index) with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize CCI generator.
        
        Args:
            period: CCI period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cci_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Commodity Channel Index over {period} periods based on {base_calculation.value}",
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
        """Generate CCI based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate CCI
            sma_tp = typical_price.rolling(window=self.period).mean()
            mad = typical_price.rolling(window=self.period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            
            return cci
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Calculate CCI on base values
            sma_base = base_values.rolling(window=self.period).mean()
            mad_base = base_values.rolling(window=self.period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (base_values - sma_base) / (0.015 * mad_base)
            
            return cci

# ADX (Average Directional Index)
class ADXGenerator(FeatureGenerator):
    """Generator for ADX (Average Directional Index) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize ADX generator.
        
        Args:
            period: ADX period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"adx_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Average Directional Index over {period} periods based on {base_calculation.value}",
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
        """Generate ADX based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            dm_plus = high.diff()
            dm_minus = -low.diff()
            
            dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
            dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
            
            # Calculate smoothed values
            atr = tr.rolling(window=self.period).mean()
            di_plus = 100 * (pd.Series(dm_plus, index=data.index).rolling(window=self.period).mean() / atr)
            di_minus = 100 * (pd.Series(dm_minus, index=data.index).rolling(window=self.period).mean() / atr)
            
            # Calculate ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=self.period).mean()
            
            return adx
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling standard deviation as proxy
            adx = base_values.rolling(window=self.period).std()
            
            return adx

# Aroon Oscillator
class AroonGenerator(FeatureGenerator):
    """Generator for Aroon Oscillator with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Aroon generator.
        
        Args:
            period: Aroon period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"aroon_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Aroon Oscillator over {period} periods based on {base_calculation.value}",
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
        """Generate Aroon based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            
            # Calculate Aroon Up and Down
            aroon_up = high.rolling(window=self.period).apply(lambda x: (self.period - x.argmax()) / self.period * 100)
            aroon_down = low.rolling(window=self.period).apply(lambda x: (self.period - x.argmin()) / self.period * 100)
            
            # Calculate Aroon Oscillator
            aroon = aroon_up - aroon_down
            
            return aroon
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling min/max
            aroon_up = base_values.rolling(window=self.period).apply(lambda x: (self.period - x.argmax()) / self.period * 100)
            aroon_down = base_values.rolling(window=self.period).apply(lambda x: (self.period - x.argmin()) / self.period * 100)
            
            aroon = aroon_up - aroon_down
            
            return aroon

# Parabolic SAR
class SARGenerator(FeatureGenerator):
    """Generator for Parabolic SAR with different base calculations."""
    
    def __init__(self,
                 acceleration: float = 0.02,
                 maximum: float = 0.2,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Parabolic SAR generator.
        
        Args:
            acceleration: SAR acceleration factor
            maximum: SAR maximum acceleration
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"sar_{acceleration}_{maximum}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Parabolic SAR with acceleration={acceleration}, maximum={maximum} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=20,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'acceleration': acceleration,
                'maximum': maximum,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.acceleration = acceleration
        self.maximum = maximum
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Parabolic SAR based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Simplified SAR calculation
            sar = pd.Series(index=data.index, dtype=float)
            sar.iloc[0] = low.iloc[0]
            
            for i in range(1, len(data)):
                if close.iloc[i] > sar.iloc[i-1]:
                    sar.iloc[i] = sar.iloc[i-1] + self.acceleration * (high.iloc[i] - sar.iloc[i-1])
                else:
                    sar.iloc[i] = sar.iloc[i-1] - self.acceleration * (sar.iloc[i-1] - low.iloc[i])
            
            return sar
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling mean as proxy
            sar = base_values.rolling(window=20).mean()
            
            return sar

# Ultimate Oscillator
class UltimateOscillatorGenerator(FeatureGenerator):
    """Generator for Ultimate Oscillator with different base calculations."""
    
    def __init__(self,
                 period1: int = 7,
                 period2: int = 14,
                 period3: int = 28,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Ultimate Oscillator generator.
        
        Args:
            period1: First period
            period2: Second period
            period3: Third period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ultimate_oscillator_{period1}_{period2}_{period3}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Ultimate Oscillator with periods {period1}, {period2}, {period3} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period3,
            min_lookback=period3,
            max_lookback=period3,
            parameters={
                'period1': period1,
                'period2': period2,
                'period3': period3,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Ultimate Oscillator based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Buying Pressure
            bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
            
            # Calculate Ultimate Oscillator
            avg7 = bp.rolling(window=self.period1).sum() / tr.rolling(window=self.period1).sum()
            avg14 = bp.rolling(window=self.period2).sum() / tr.rolling(window=self.period2).sum()
            avg28 = bp.rolling(window=self.period3).sum() / tr.rolling(window=self.period3).sum()
            
            uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            
            return uo
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling mean as proxy
            uo = base_values.rolling(window=self.period3).mean()
            
            return uo

# KST (Know Sure Thing)
class KSTGenerator(FeatureGenerator):
    """Generator for KST (Know Sure Thing) with different base calculations."""
    
    def __init__(self,
                 roc1: int = 10,
                 roc2: int = 15,
                 roc3: int = 20,
                 roc4: int = 30,
                 sma1: int = 10,
                 sma2: int = 10,
                 sma3: int = 10,
                 sma4: int = 15,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize KST generator.
        
        Args:
            roc1: First ROC period
            roc2: Second ROC period
            roc3: Third ROC period
            roc4: Fourth ROC period
            sma1: First SMA period
            sma2: Second SMA period
            sma3: Third SMA period
            sma4: Fourth SMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"kst_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"KST with ROC periods {roc1}, {roc2}, {roc3}, {roc4} and SMA periods {sma1}, {sma2}, {sma3}, {sma4} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=max(roc4, sma4),
            min_lookback=max(roc4, sma4),
            max_lookback=max(roc4, sma4),
            parameters={
                'roc1': roc1,
                'roc2': roc2,
                'roc3': roc3,
                'roc4': roc4,
                'sma1': sma1,
                'sma2': sma2,
                'sma3': sma3,
                'sma4': sma4,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4
        self.sma1 = sma1
        self.sma2 = sma2
        self.sma3 = sma3
        self.sma4 = sma4
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate KST based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            close = data['close']
            
            # Calculate ROC
            roc1 = close.pct_change(periods=self.roc1) * 100
            roc2 = close.pct_change(periods=self.roc2) * 100
            roc3 = close.pct_change(periods=self.roc3) * 100
            roc4 = close.pct_change(periods=self.roc4) * 100
            
            # Calculate SMA of ROC
            sma_roc1 = roc1.rolling(window=self.sma1).mean()
            sma_roc2 = roc2.rolling(window=self.sma2).mean()
            sma_roc3 = roc3.rolling(window=self.sma3).mean()
            sma_roc4 = roc4.rolling(window=self.sma4).mean()
            
            # Calculate KST
            kst = sma_roc1 + 2 * sma_roc2 + 3 * sma_roc3 + 4 * sma_roc4
            
            return kst
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling mean as proxy
            kst = base_values.rolling(window=max(self.roc4, self.sma4)).mean()
            
            return kst

# APO (Absolute Price Oscillator)
class APOGenerator(FeatureGenerator):
    """Generator for APO (Absolute Price Oscillator) with different base calculations."""
    
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize APO generator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"apo_{fast_period}_{slow_period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Absolute Price Oscillator with fast={fast_period}, slow={slow_period} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=slow_period,
            min_lookback=slow_period,
            max_lookback=slow_period,
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate APO based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate EMA
        ema_fast = base_values.ewm(span=self.fast_period).mean()
        ema_slow = base_values.ewm(span=self.slow_period).mean()
        
        # Calculate APO
        apo = ema_fast - ema_slow
        
        return apo

# CMO (Chande Momentum Oscillator)
class CMOGenerator(FeatureGenerator):
    """Generator for CMO (Chande Momentum Oscillator) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize CMO generator.
        
        Args:
            period: CMO period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cmo_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Chande Momentum Oscillator over {period} periods based on {base_calculation.value}",
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
        """Generate CMO based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum
        momentum = base_values.diff()
        
        # Calculate positive and negative momentum
        pos_momentum = momentum.where(momentum > 0, 0)
        neg_momentum = -momentum.where(momentum < 0, 0)
        
        # Calculate CMO
        pos_sum = pos_momentum.rolling(window=self.period).sum()
        neg_sum = neg_momentum.rolling(window=self.period).sum()
        
        cmo = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum)
        
        return cmo

# NATR (Normalized Average True Range)
class NATRGenerator(FeatureGenerator):
    """Generator for NATR (Normalized Average True Range) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize NATR generator.
        
        Args:
            period: NATR period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"natr_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Normalized Average True Range over {period} periods based on {base_calculation.value}",
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
        """Generate NATR based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate NATR
            atr = tr.rolling(window=self.period).mean()
            natr = 100 * atr / close
            
            return natr
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling std as proxy
            natr = base_values.rolling(window=self.period).std()
            
            return natr

# PFE (Polarized Fractal Efficiency)
class PFEGenerator(FeatureGenerator):
    """Generator for PFE (Polarized Fractal Efficiency) with different base calculations."""
    
    def __init__(self,
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize PFE generator.
        
        Args:
            period: PFE period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"pfe_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Polarized Fractal Efficiency over {period} periods based on {base_calculation.value}",
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
        """Generate PFE based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate PFE
        pfe = base_values.rolling(window=self.period).apply(
            lambda x: 100 * np.sqrt((x.iloc[-1] - x.iloc[0])**2 + self.period**2) / 
                     np.sum(np.sqrt((x.diff()**2 + 1)))
        )
        
        return pfe

# T3 (T3 Moving Average)
class T3Generator(FeatureGenerator):
    """Generator for T3 (T3 Moving Average) with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 volume_factor: float = 0.7,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize T3 generator.
        
        Args:
            period: T3 period
            volume_factor: T3 volume factor
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"t3_{period}_{volume_factor}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"T3 Moving Average with period={period}, volume_factor={volume_factor} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'volume_factor': volume_factor,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.volume_factor = volume_factor
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate T3 based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate T3 (simplified version)
        t3 = base_values.ewm(span=self.period).mean()
        
        return t3

# KAMA (Kaufman's Adaptive Moving Average)
class KAMAGenerator(FeatureGenerator):
    """Generator for KAMA (Kaufman's Adaptive Moving Average) with different base calculations."""
    
    def __init__(self,
                 period: int = 30,
                 fast_period: int = 2,
                 slow_period: int = 30,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize KAMA generator.
        
        Args:
            period: KAMA period
            fast_period: Fast period
            slow_period: Slow period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"kama_{period}_{fast_period}_{slow_period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Kaufman's Adaptive Moving Average with period={period}, fast={fast_period}, slow={slow_period} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate KAMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate KAMA (simplified version)
        kama = base_values.ewm(span=self.period).mean()
        
        return kama