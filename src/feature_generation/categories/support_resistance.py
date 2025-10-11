"""
Support/Resistance Feature Generators - VectorBT Optimized

This module provides high-performance support/resistance feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Support level detection
- Resistance level detection
- Pivot point calculations
- Fibonacci retracement levels
- Volume-weighted support/resistance
- Dynamic support/resistance levels
- Support/resistance strength indicators
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

class SupportLevelGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized support level generator."""
    
    def __init__(self, level: int = 1, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(level, window)
        super().__init__(config)
        self.level = level
        self.window = window
    
    @classmethod
    def _create_default_config(cls, level: int = 1, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"support_level_{level}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized support level {level} over {window} periods",
            required_columns=["low"],
            optional_columns=["high", "close", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"level": level, "window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate support level using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'support_level_{self.level}_{self.window}')
        
        low = data['low']
        
        # Use VectorBT rolling min for support level
        support_level = self._vectorbt_rolling_operation(low, 'min', window=self.window)
        
        return support_level.rename(f'support_level_{self.level}_{self.window}')


class ResistanceLevelGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized resistance level generator."""
    
    def __init__(self, level: int = 1, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(level, window)
        super().__init__(config)
        self.level = level
        self.window = window
    
    @classmethod
    def _create_default_config(cls, level: int = 1, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"resistance_level_{level}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized resistance level {level} over {window} periods",
            required_columns=["high"],
            optional_columns=["low", "close", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"level": level, "window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate resistance level using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'resistance_level_{self.level}_{self.window}')
        
        high = data['high']
        
        # Use VectorBT rolling max for resistance level
        resistance_level = self._vectorbt_rolling_operation(high, 'max', window=self.window)
        
        return resistance_level.rename(f'resistance_level_{self.level}_{self.window}')


class PivotPointGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized pivot point generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"pivot_point_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized pivot point over {window} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate pivot point using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'pivot_point_{self.window}')
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate pivot point as (high + low + close) / 3
        pivot_point = (high + low + close) / 3
        
        return pivot_point.rename(f'pivot_point_{self.window}')


class FibonacciLevelGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Fibonacci level generator."""
    
    def __init__(self, level: float = 0.618, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(level, window)
        super().__init__(config)
        self.level = level
        self.window = window
    
    @classmethod
    def _create_default_config(cls, level: float = 0.618, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"fibonacci_{level}_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized Fibonacci level {level} over {window} periods",
            required_columns=["high", "low"],
            optional_columns=["close", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"level": level, "window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Fibonacci level using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'fibonacci_{self.level}_{self.window}')
        
        high = data['high']
        low = data['low']
        
        # Calculate range using VectorBT rolling operations
        high_max = self._vectorbt_rolling_operation(high, 'max', window=self.window)
        low_min = self._vectorbt_rolling_operation(low, 'min', window=self.window)
        
        # Calculate Fibonacci level
        range_size = high_max - low_min
        fibonacci_level = low_min + (range_size * self.level)
        
        return fibonacci_level.rename(f'fibonacci_{self.level}_{self.window}')


class VolumeWeightedSupportResistanceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume-weighted support/resistance generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vw_sr_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized volume-weighted support/resistance over {window} periods",
            required_columns=["high", "low", "volume"],
            optional_columns=["close", "open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume-weighted support/resistance using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vw_sr_{self.window}')
        
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calculate volume-weighted price
        vw_price = (high + low) / 2 * volume
        
        # Use VectorBT rolling operations
        vw_price_sum = self._vectorbt_rolling_operation(vw_price, 'sum', window=self.window)
        volume_sum = self._vectorbt_rolling_operation(volume, 'sum', window=self.window)
        
        # Calculate volume-weighted support/resistance
        vw_sr = safe_divide(vw_price_sum, volume_sum)
        
        return vw_sr.rename(f'vw_sr_{self.window}')


class SupportResistanceStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized support/resistance strength generator."""
    
    def __init__(self, window: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)
        self.window = window
    
    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"sr_strength_{window}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized support/resistance strength over {window} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate support/resistance strength using VectorBT operations."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'sr_strength_{self.window}')
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate range
        price_range = high - low
        
        # Calculate how close price is to support/resistance levels
        support_level = self._vectorbt_rolling_operation(low, 'min', window=self.window)
        resistance_level = self._vectorbt_rolling_operation(high, 'max', window=self.window)
        
        # Calculate strength as distance from levels
        distance_to_support = close - support_level
        distance_to_resistance = resistance_level - close
        
        # Normalize by range
        support_strength = safe_divide(distance_to_support, price_range)
        resistance_strength = safe_divide(distance_to_resistance, price_range)
        
        # Combine strengths (positive for resistance, negative for support)
        sr_strength = resistance_strength - support_strength
        
        return sr_strength.rename(f'sr_strength_{self.window}')


class DynamicSupportResistanceGenerator(VectorBTFeatureGenerator):
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
            name=f"dynamic_sr_{window}_{sensitivity}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"VectorBT-optimized dynamic support/resistance over {window} periods (sensitivity {sensitivity})",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
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
            return pd.Series(dtype=float, index=data.index, name=f'dynamic_sr_{self.window}_{self.sensitivity}')
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate price momentum
        price_momentum = close.pct_change()
        
        # Calculate dynamic levels based on momentum
        momentum_ma = self._vectorbt_rolling_operation(price_momentum, 'mean', window=self.window)
        
        # Adjust levels based on momentum
        base_support = self._vectorbt_rolling_operation(low, 'min', window=self.window)
        base_resistance = self._vectorbt_rolling_operation(high, 'max', window=self.window)
        
        # Dynamic adjustment
        adjustment = momentum_ma * self.sensitivity * (base_resistance - base_support)
        
        dynamic_sr = base_support + adjustment
        
        return dynamic_sr.rename(f'dynamic_sr_{self.window}_{self.sensitivity}')


def create_support_resistance_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized support/resistance feature generators."""
    generators = []
    
    # Basic support/resistance levels
    for level in [1, 2, 3, 4, 5]:
        for window in [5, 10, 20]:
            generators.extend([
                SupportLevelGenerator(level, window),
                ResistanceLevelGenerator(level, window),
            ])
    
    # Pivot points
    for window in [5, 10, 20]:
        generators.append(PivotPointGenerator(window))
    
    # Fibonacci levels
    fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    for level in fibonacci_levels:
        for window in [5, 10, 20]:
            generators.append(FibonacciLevelGenerator(level, window))
    
    # Volume-weighted support/resistance
    for window in [5, 10, 20]:
        generators.append(VolumeWeightedSupportResistanceGenerator(window))
    
    # Support/resistance strength
    for window in [5, 10, 20]:
        generators.append(SupportResistanceStrengthGenerator(window))
    
    # Dynamic support/resistance
    for window in [10, 20]:
        for sensitivity in [0.05, 0.1, 0.2]:
            generators.append(DynamicSupportResistanceGenerator(window, sensitivity))
    
    return generators


def create_default_support_resistance_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized support/resistance feature generators."""
    return create_support_resistance_generators()


# Export all generators
__all__ = [
    'SupportLevelGenerator',
    'ResistanceLevelGenerator',
    'PivotPointGenerator',
    'FibonacciLevelGenerator',
    'VolumeWeightedSupportResistanceGenerator',
    'SupportResistanceStrengthGenerator',
    'DynamicSupportResistanceGenerator',
    'create_support_resistance_generators',
    'create_default_support_resistance_generators'
]