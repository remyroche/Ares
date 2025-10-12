"""
Support/Resistance Feature Generator

This module provides feature generators for support/resistance-based indicators.
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
    BaseCalculationType,
    create_base_calculator
)

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
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# VectorBT-optimized support/resistance generators will be defined inline below
VECTORBT_SUPPORT_RESISTANCE_AVAILABLE = VECTORBT_AVAILABLE

class SupportResistanceFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for support/resistance-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="support_resistance_features",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description="Comprehensive support/resistance features including pivot points, Fibonacci, and volume profile",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=1,
            max_lookback=100,
            parameters={
                "pivot_windows": [5, 10, 20],
                "fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
                "volume_profile_windows": [5, 10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'SupportResistanceFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close_prices = data['close'].values
        sr = np.zeros_like(close_prices)
        return pd.Series(sr, index=data.index, name='sr_placeholder')

# Support Level Generator
    
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

class SupportLevelGenerator(VectorizedFeatureGenerator):
    """Generator for support level features."""
    
    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'low' not in required_columns:
            required_columns.append('low')
        
        config = FeatureConfig(
            name=f"support_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Support level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate support level."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            low = data['low']
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(low) >= 1000:
                support_level = rolling_min(low, window=self.window)
            else:
                support_level = low.rolling(window=self.window).min()
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(base_values) >= 1000:
                support_level = rolling_min(base_values, window=self.window)
            else:
                support_level = base_values.rolling(window=self.window).min()
        return support_level
    
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

# Resistance Level Generator
    
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

class ResistanceLevelGenerator(VectorizedFeatureGenerator):
    """Generator for resistance level features."""
    
    def __init__(self, level: int = 1, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        
        config = FeatureConfig(
            name=f"resistance_level_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Resistance level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate resistance level."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(high) >= 1000:
                resistance_level = rolling_max(high, window=self.window)
            else:
                resistance_level = high.rolling(window=self.window).max()
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(base_values) >= 1000:
                resistance_level = rolling_max(base_values, window=self.window)
            else:
                resistance_level = base_values.rolling(window=self.window).max()
        return resistance_level

# Pivot Point Generator
    
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

class PivotPointGenerator(VectorizedFeatureGenerator):
    """Generator for pivot point features."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
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
            name=f"pivot_point_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Pivot point over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate pivot point."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            pivot_point = (high + low + close) / 3
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(base_values) >= 1000:
                pivot_point = rolling_mean(base_values, window=self.window)
            else:
                pivot_point = base_values.rolling(window=self.window).mean()
        return pivot_point

# Fibonacci Level Generator
    
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

class FibonacciLevelGenerator(VectorizedFeatureGenerator):
    """Generator for Fibonacci level features."""
    
    def __init__(self, level: float = 0.618, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        if 'high' not in required_columns:
            required_columns.append('high')
        if 'low' not in required_columns:
            required_columns.append('low')
        
        config = FeatureConfig(
            name=f"fibonacci_{level}_{window}_{base_calculation.value}",
            category=FeatureCategory.SUPPORT_RESISTANCE,
            description=f"Fibonacci level {level} over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'level': level, 'window': window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.level = level
        self.window = window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Fibonacci level."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(high) >= 1000:
                high_max = rolling_max(high, window=self.window)
                low_min = rolling_min(low, window=self.window)
                range_size = high_max - low_min
                fibonacci_level = low_min + (range_size * self.level)
            else:
                range_size = high.rolling(window=self.window).max() - low.rolling(window=self.window).min()
                fibonacci_level = low.rolling(window=self.window).min() + (range_size * self.level)
        else:
            base_values = self.base_calculator.calculate(data)
            # Use VectorBT for optimized rolling operations if available
            if VECTORBT_AVAILABLE and len(base_values) >= 1000:
                fibonacci_level = quantile(base_values, q=self.level, window=self.window)
            else:
                fibonacci_level = base_values.rolling(window=self.window).quantile(self.level)
        return fibonacci_level

def create_default_support_resistance_generators() -> List[FeatureGenerator]:
    """Create default support/resistance feature generators."""
    generators = []
    
    # Use VectorBT generators if available, otherwise fall back to legacy generators
    if VECTORBT_SUPPORT_RESISTANCE_AVAILABLE and VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        generators.extend(create_default_vectorbt_support_resistance_generators())
    else:
        # Fall back to legacy generators
        windows = [5, 10, 20]
        fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        # Create generators for each window
        for window in windows:
            generators.extend([
                SupportLevelGenerator(1, window),
                SupportLevelGenerator(2, window),
                SupportLevelGenerator(3, window),
                SupportLevelGenerator(4, window),
                SupportLevelGenerator(5, window),
                ResistanceLevelGenerator(1, window),
                ResistanceLevelGenerator(2, window),
                ResistanceLevelGenerator(3, window),
                ResistanceLevelGenerator(4, window),
                ResistanceLevelGenerator(5, window),
                PivotPointGenerator(window),
            ])
        
        # Create Fibonacci level generators
        for level in fibonacci_levels:
            for window in windows:
                generators.append(FibonacciLevelGenerator(level, window))
    
    return generators
