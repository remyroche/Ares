"""
VectorBT-Optimized Support/Resistance Feature Generators

This module provides high-performance support/resistance feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Support level detection
- Resistance level detection
- Pivot point calculations
- Fibonacci level analysis
- Volume profile analysis
- Dynamic support/resistance levels
- Multi-timeframe support/resistance
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ..base_calculations import BaseCalculationType, create_base_calculator
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

logger = logging.getLogger(__name__)


class VectorBTSupportLevelGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized support level generator."""
    
    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'low' not in required_columns:
            required_columns.append('low')
        
        config = FeatureConfig(
            name=f"vectorbt_support_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized support level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "open", "close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate support level using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_support_level_{self.level}_{self.window}_{self.base_calculation.value}')
        
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            low = data['low']
            # Use VectorBT for optimized rolling operations
            support_level = self._vectorbt_rolling_operation(low, 'min', window=self.window)
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations
            support_level = self._vectorbt_rolling_operation(base_values, 'min', window=self.window)
        
        return support_level.rename(f'vectorbt_support_level_{self.level}_{self.window}_{self.base_calculation.value}')


class VectorBTResistanceLevelGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized resistance level generator."""
    
    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        
        config = FeatureConfig(
            name=f"vectorbt_resistance_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized resistance level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["low", "open", "close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate resistance level using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_resistance_level_{self.level}_{self.window}_{self.base_calculation.value}')
        
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            # Use VectorBT for optimized rolling operations
            resistance_level = self._vectorbt_rolling_operation(high, 'max', window=self.window)
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations
            resistance_level = self._vectorbt_rolling_operation(base_values, 'max', window=self.window)
        
        return resistance_level.rename(f'vectorbt_resistance_level_{self.level}_{self.window}_{self.base_calculation.value}')


class VectorBTPivotPointGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized pivot point generator."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')
        if 'close' not in required_columns:
            required_columns.append('close')
        
        config = FeatureConfig(
            name=f"vectorbt_pivot_point_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized pivot point over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate pivot point using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_pivot_point_{self.window}_{self.base_calculation.value}')
        
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            pivot_point = (high + low + close) / 3
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations
            pivot_point = self._vectorbt_rolling_operation(base_values, 'mean', window=self.window)
        
        return pivot_point.rename(f'vectorbt_pivot_point_{self.window}_{self.base_calculation.value}')


class VectorBTFibonacciLevelGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Fibonacci level generator."""
    
    def __init__(self, level: float = 0.618, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')
        
        config = FeatureConfig(
            name=f"vectorbt_fibonacci_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized Fibonacci level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["open", "close", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Fibonacci level using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_fibonacci_{self.level}_{self.window}_{self.base_calculation.value}')
        
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            # Use VectorBT for optimized rolling operations
            high_max = self._vectorbt_rolling_operation(high, 'max', window=self.window)
            low_min = self._vectorbt_rolling_operation(low, 'min', window=self.window)
            range_size = high_max - low_min
            fibonacci_level = low_min + (range_size * self.level)
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations
            fibonacci_level = self._vectorbt_rolling_operation(base_values, 'quantile', window=self.window, q=self.level)
        
        return fibonacci_level.rename(f'vectorbt_fibonacci_{self.level}_{self.window}_{self.base_calculation.value}')


class VectorBTVolumeProfileGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume profile generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volume_profile_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized volume profile over {window} periods",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume profile using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volume_profile_{self.window}')
        
        close = data['close']
        volume = data['volume']
        
        # Calculate volume-weighted average price
        vwap = (close * volume).rolling(window=self.window).sum() / volume.rolling(window=self.window).sum()
        
        # Use VectorBT for optimized rolling operations
        volume_profile = self._vectorbt_rolling_operation(vwap, 'mean', window=self.window)
        
        return volume_profile.rename(f'vectorbt_volume_profile_{self.window}')


class VectorBTDynamicSupportResistanceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized dynamic support/resistance generator."""
    
    def __init__(self, window: int = 20, sensitivity: float = 0.1, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window, sensitivity)
        super().__init__(config)
        self.window = window
        self.sensitivity = sensitivity
    
    @classmethod
    def _create_default_config(cls, window: int = 20, sensitivity: float = 0.1) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_dynamic_sr_{window}_{sensitivity}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized dynamic support/resistance over {window} periods with sensitivity {sensitivity}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window, "sensitivity": sensitivity},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate dynamic support/resistance using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_dynamic_sr_{self.window}_{self.sensitivity}')
        
        close = data['close']
        
        # Calculate price range
        price_range = close.rolling(window=self.window).max() - close.rolling(window=self.window).min()
        
        # Calculate dynamic levels based on price volatility
        volatility = self._vectorbt_rolling_operation(close, 'std', window=self.window)
        dynamic_level = close.rolling(window=self.window).mean() + (volatility * self.sensitivity)
        
        return dynamic_level.rename(f'vectorbt_dynamic_sr_{self.window}_{self.sensitivity}')


class VectorBTMultiTimeframeSupportResistanceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized multi-timeframe support/resistance generator."""
    
    def __init__(self, short_window: int = 10, long_window: int = 50, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(short_window, long_window)
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window
    
    @classmethod
    def _create_default_config(cls, short_window: int = 10, long_window: int = 50) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_multi_tf_sr_{short_window}_{long_window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized multi-timeframe support/resistance with {short_window} and {long_window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=long_window,
            min_lookback=long_window,
            max_lookback=long_window,
            parameters={"short_window": short_window, "long_window": long_window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate multi-timeframe support/resistance using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_multi_tf_sr_{self.short_window}_{self.long_window}')
        
        close = data['close']
        
        # Calculate short-term and long-term levels
        short_term_level = self._vectorbt_rolling_operation(close, 'mean', window=self.short_window)
        long_term_level = self._vectorbt_rolling_operation(close, 'mean', window=self.long_window)
        
        # Combine levels with weights
        combined_level = (short_term_level * 0.3 + long_term_level * 0.7)
        
        return combined_level.rename(f'vectorbt_multi_tf_sr_{self.short_window}_{self.long_window}')


def create_vectorbt_support_resistance_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized support/resistance feature generators."""
    generators = []
    
    # Support level generators
    for level in [1, 2, 3, 4, 5]:
        for window in [5, 10, 20]:
            generators.append(VectorBTSupportLevelGenerator(level, window))
    
    # Resistance level generators
    for level in [1, 2, 3, 4, 5]:
        for window in [5, 10, 20]:
            generators.append(VectorBTResistanceLevelGenerator(level, window))
    
    # Pivot point generators
    for window in [5, 10, 20]:
        generators.append(VectorBTPivotPointGenerator(window))
    
    # Fibonacci level generators
    fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    for level in fibonacci_levels:
        for window in [5, 10, 20]:
            generators.append(VectorBTFibonacciLevelGenerator(level, window))
    
    # Volume profile generators
    for window in [5, 10, 20]:
        generators.append(VectorBTVolumeProfileGenerator(window))
    
    # Dynamic support/resistance generators
    for window in [10, 20]:
        for sensitivity in [0.1, 0.2]:
            generators.append(VectorBTDynamicSupportResistanceGenerator(window, sensitivity))
    
    # Multi-timeframe generators
    generators.extend([
        VectorBTMultiTimeframeSupportResistanceGenerator(short_window=10, long_window=50),
        VectorBTMultiTimeframeSupportResistanceGenerator(short_window=5, long_window=20),
    ])
    
    return generators


def create_default_vectorbt_support_resistance_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized support/resistance feature generators."""
    return create_vectorbt_support_resistance_generators()


# Export all generators
__all__ = [
    'VectorBTSupportLevelGenerator',
    'VectorBTResistanceLevelGenerator',
    'VectorBTPivotPointGenerator',
    'VectorBTFibonacciLevelGenerator',
    'VectorBTVolumeProfileGenerator',
    'VectorBTDynamicSupportResistanceGenerator',
    'VectorBTMultiTimeframeSupportResistanceGenerator',
    'create_vectorbt_support_resistance_generators',
    'create_default_vectorbt_support_resistance_generators'
]