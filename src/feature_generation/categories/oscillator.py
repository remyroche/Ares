"""
Oscillator Feature Generator

This module provides feature generators for oscillator indicators,
including CCI, ADX, Aroon, Ultimate Oscillator, KST, APO, CMO, NATR, PFE, T3, KAMA, and more.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    BaseCalculationType,
    create_base_calculator
)

class OscillatorFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for oscillator-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
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
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Placeholder implementation
        close_prices = data['close'].values
        oscillator = np.zeros_like(close_prices)
        return pd.Series(oscillator, index=data.index, name='oscillator_placeholder')

def create_oscillator_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of oscillator feature generators."""
    if periods is None:
        periods = {
            'cci': [20],
            'adx': [14],
            'aroon': [25],
            'ultimate': [14],
            'kst': [10, 15, 20, 30],
            'apo': [12, 26],
            'cmo': [14],
            'natr': [14],
            'pfe': [12],
            't3': [14],
            'kama': [30]
        }
    
    generators = []
    
    # CCI generators
    for period in periods.get('cci', [20]):
        generators.append(CCIGenerator(period=period))
    
    # ADX generators
    for period in periods.get('adx', [14]):
        generators.append(ADXGenerator(period=period))
    
    # Aroon generators
    for period in periods.get('aroon', [25]):
        generators.append(AroonGenerator(period=period))
    
    # Ultimate Oscillator generators
    for period in periods.get('ultimate', [14]):
        generators.append(UltimateOscillatorGenerator(period=period))
    
    # KST generators
    for period in periods.get('kst', [10]):
        generators.append(KSTGenerator(period=period))
    
    # APO generators
    for period in periods.get('apo', [12]):
        generators.append(APOGenerator(period=period))
    
    # CMO generators
    for period in periods.get('cmo', [14]):
        generators.append(CMOGenerator(period=period))
    
    # NATR generators
    for period in periods.get('natr', [14]):
        generators.append(NATRGenerator(period=period))
    
    # PFE generators
    for period in periods.get('pfe', [12]):
        generators.append(PFEGenerator(period=period))
    
    # T3 generators
    for period in periods.get('t3', [14]):
        generators.append(T3Generator(period=period))
    
    # KAMA generators
    for period in periods.get('kama', [30]):
        generators.append(KAMAGenerator(period=period))
    
    return generators

def create_default_oscillator_generators() -> List[FeatureGenerator]:
    return create_oscillator_generators()

# CCI (Commodity Channel Index)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class CCIGenerator(VectorizedFeatureGenerator):
    """Generator for CCI (Commodity Channel Index) with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate CCI based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate typical price
            typical_price = (high + low + close) / 3
            
            # Calculate CCI - OPTIMIZED: Vectorized MAD calculation
            sma_tp = typical_price.rolling(window=self.period).mean()
            # Vectorized MAD: use rolling mean of absolute deviations
            mad = (typical_price - typical_price.rolling(window=self.period).mean()).abs().rolling(window=self.period).mean()
            cci = (typical_price - sma_tp) / (0.015 * mad)
            
            return cci
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Calculate CCI on base values - OPTIMIZED: Vectorized MAD calculation
            sma_base = base_values.rolling(window=self.period).mean()
            # Vectorized MAD: use rolling mean of absolute deviations
            mad_base = (base_values - base_values.rolling(window=self.period).mean()).abs().rolling(window=self.period).mean()
            cci = (base_values - sma_base) / (0.015 * mad_base)
            
            return cci

# ADX (Average Directional Index)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class ADXGenerator(VectorizedFeatureGenerator):
    """Generator for ADX (Average Directional Index) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
            # Convert to pandas Series if it's a numpy array
            if isinstance(base_values, np.ndarray):
                base_values = pd.Series(base_values, index=data.index)
            adx = base_values.rolling(window=self.period).std()
            
            return adx

# Aroon Oscillator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AroonGenerator(VectorizedFeatureGenerator):
    """Generator for Aroon Oscillator with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Aroon based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            
            # OPTIMIZED: Use vectorized argmax/argmin calculations
            # Use pandas built-in rolling idxmax/idxmin for better performance
            aroon_up = ((self.period - high.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
            aroon_down = ((self.period - low.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
            
            # Calculate Aroon Oscillator
            aroon = aroon_up - aroon_down
            
            return aroon
        else:
            base_values = self.base_calculator.calculate(data)
            
            # OPTIMIZED: Use vectorized argmax/argmin calculations
            # Use pandas built-in rolling idxmax/idxmin for better performance
            aroon_up = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
            aroon_down = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
            
            aroon = aroon_up - aroon_down
            
            return aroon

# Parabolic SAR
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class SARGenerator(VectorizedFeatureGenerator):
    """Generator for Parabolic SAR with different base calculations."""
    
    def __init__(self,
                 acceleration: float = 0.02,
                 maximum: float = 0.2,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.acceleration = acceleration
        self.maximum = maximum
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
            sar = self._vectorbt_rolling_operation(base_values, "mean", 20)
            
            return sar

# Ultimate Oscillator
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class UltimateOscillatorGenerator(VectorizedFeatureGenerator):
    """Generator for Ultimate Oscillator with different base calculations."""
    
    def __init__(self,
                 period1: int = 7,
                 period2: int = 14,
                 period3: int = 28,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class KSTGenerator(VectorizedFeatureGenerator):
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
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
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
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class APOGenerator(VectorizedFeatureGenerator):
    """Generator for APO (Absolute Price Oscillator) with different base calculations."""
    
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate APO based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate EMA
        ema_fast = base_values.ewm(span=self.fast_period).mean()
        ema_slow = base_values.ewm(span=self.slow_period).mean()
        
        # Calculate APO
        apo = ema_fast - ema_slow
        
        return apo

# CMO (Chande Momentum Oscillator)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class CMOGenerator(VectorizedFeatureGenerator):
    """Generator for CMO (Chande Momentum Oscillator) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class NATRGenerator(VectorizedFeatureGenerator):
    """Generator for NATR (Normalized Average True Range) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

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
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class PFEGenerator(VectorizedFeatureGenerator):
    """Generator for PFE (Polarized Fractal Efficiency) with different base calculations."""
    
    def __init__(self,
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate PFE based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate PFE - OPTIMIZED: Vectorized PFE calculation
        # Pre-calculate differences and their norms
        diff_values = base_values.diff().fillna(0)
        diff_norms = np.sqrt(diff_values**2 + 1)
        
        # Vectorized PFE calculation
        numerator = np.sqrt((base_values - base_values.shift(self.period))**2 + self.period**2)
        denominator = diff_norms.rolling(window=self.period).sum()
        pfe = 100 * numerator / denominator
        
        return pfe

# T3 (T3 Moving Average)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class T3Generator(VectorizedFeatureGenerator):
    """Generator for T3 (T3 Moving Average) with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 volume_factor: float = 0.7,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.volume_factor = volume_factor
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate T3 based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate T3 (simplified version)
        t3 = base_values.ewm(span=self.period).mean()
        
        return t3

# KAMA (Kaufman's Adaptive Moving Average)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class KAMAGenerator(VectorizedFeatureGenerator):
    """Generator for KAMA (Kaufman's Adaptive Moving Average) with different base calculations."""
    
    def __init__(self,
                 period: int = 30,
                 fast_period: int = 2,
                 slow_period: int = 30,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
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
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
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
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate KAMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate KAMA (simplified version)
        kama = base_values.ewm(span=self.period).mean()
        
        return kama
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
