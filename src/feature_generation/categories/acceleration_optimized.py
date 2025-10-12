"""
Optimized Acceleration Feature Generators with Full VectorBT Integration

This module provides fully optimized acceleration feature generators using
VectorBTRollingOptimizer and UnifiedVectorizationManager for maximum performance.

Key Features:
- Unified Vectorization Manager integration
- VectorBTRollingOptimizer for all rolling operations
- Batch processing optimization
- Memory management
- Performance monitoring
- Hardware acceleration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Unified Vectorization Manager
try:
    from ..utils.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

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

# Import VectorBT-optimized acceleration generators
try:
    from .vectorbt_acceleration import (
        create_vectorbt_acceleration_generators,
        create_default_vectorbt_acceleration_generators,
        VectorBTMomentumGenerator,
        VectorBTPriceAccelerationGenerator,
        VectorBTPriceJerkGenerator,
        VectorBTTrendStrengthGenerator,
        VectorBTTrendConsistencyGenerator,
        VectorBTVolumeAccelerationGenerator,
        VectorBTVolatilityAccelerationGenerator,
        VectorBTMomentumAccelerationGenerator,
        VectorBTAccelerationMomentumGenerator,
        VectorBTAccelerationVolatilityGenerator,
        VectorBTAccelerationTrendStrengthGenerator,
        VectorBTAccelerationConsistencyGenerator,
        VectorBTAccelerationRegimeGenerator,
        VectorBTMultiTimeframeAccelerationGenerator,
        VectorBTAccelerationCorrelationGenerator,
        VectorBTAccelerationDivergenceGenerator
    )
    VECTORBT_ACCELERATION_AVAILABLE = True
except ImportError:
    VECTORBT_ACCELERATION_AVAILABLE = False

class OptimizedAccelerationFeatureGenerator(VectorizedFeatureGenerator):
    """Fully optimized acceleration feature generator with UnifiedVectorizationManager."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
        
        # Initialize VectorBT Rolling Optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="optimized_acceleration_features",
            category=FeatureCategory.ACCELERATION,
            description="Fully optimized acceleration features with VectorBT integration",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "acceleration_windows": [5, 10, 20],
                "momentum_windows": [5, 10, 20, 50]
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing using UnifiedVectorizationManager."""
        if self.vectorization_manager:
            return self.vectorization_manager.optimize_dataframe(data)
        elif hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with UnifiedVectorizationManager."""
        if self.vectorization_manager:
            # Convert to batch operations format
            batch_operations = []
            for window in windows:
                for operation in operations:
                    for column in (columns or data.select_dtypes(include=[np.number]).columns):
                        batch_operations.append({
                            'name': f'{column}_{operation}_{window}',
                            'column': column,
                            'operation': operation,
                            'window': window
                        })
            return self.vectorization_manager.batch_rolling_operations(data, batch_operations)
        elif hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, 
                                   window: int, **kwargs) -> pd.Series:
        """Perform rolling operation using UnifiedVectorizationManager."""
        if self.vectorization_manager:
            return self.vectorization_manager.rolling_operation(data, operation, window, **kwargs)
        elif self.rolling_optimizer:
            return self.rolling_optimizer._rolling_operation(data, operation, window, **kwargs)
        else:
            # Fallback to pandas
            rolling_obj = data.rolling(window=window, **kwargs)
            if operation == 'mean':
                return rolling_obj.mean()
            elif operation == 'std':
                return rolling_obj.std()
            elif operation == 'var':
                return rolling_obj.var()
            elif operation == 'min':
                return rolling_obj.min()
            elif operation == 'max':
                return rolling_obj.max()
            elif operation == 'sum':
                return rolling_obj.sum()
            else:
                raise ValueError(f"Unsupported operation: {operation}")

class OptimizedMomentumGenerator(FeatureGenerator):
    """Fully optimized momentum generator with VectorBT integration."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"optimized_momentum_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Fully optimized momentum over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        
        # Initialize optimizers
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
        
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        else:
            self.rolling_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum with full VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'optimized_momentum_{self.period}_{self.base_calculation.value}')
        
        # Optimize DataFrame for processing
        if self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe(data)
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum using optimized rolling operations
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        
        # Validate finite values
        try:
            validate_finite(momentum.values, f"Optimized_Momentum_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            self.logger.warning(f"⚠️ {e}")
        
        return momentum.rename(f'optimized_momentum_{self.period}_{self.base_calculation.value}')

class OptimizedPriceAccelerationGenerator(FeatureGenerator):
    """Fully optimized price acceleration generator with VectorBT integration."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"optimized_acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Fully optimized acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        
        # Initialize optimizers
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
        
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        else:
            self.rolling_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration with full VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'optimized_acceleration_{self.period}_{self.base_calculation.value}')
        
        # Optimize DataFrame for processing
        if self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe(data)
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum first
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        
        # Calculate acceleration (second derivative) using optimized operations
        if self.vectorization_manager:
            acceleration = self.vectorization_manager.rolling_operation(momentum, 'diff', self.period)
        elif self.rolling_optimizer:
            acceleration = self.rolling_optimizer._rolling_operation(momentum, 'diff', self.period)
        else:
            acceleration = momentum.diff(self.period)
        
        # Validate finite values
        try:
            validate_finite(acceleration.values, f"Optimized_Acceleration_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            self.logger.warning(f"⚠️ {e}")
        
        return acceleration.rename(f'optimized_acceleration_{self.period}_{self.base_calculation.value}')

class OptimizedTrendStrengthGenerator(FeatureGenerator):
    """Fully optimized trend strength generator with VectorBT integration."""
    
    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"optimized_trend_strength_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Fully optimized trend strength over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation
        
        # Initialize optimizers
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.vectorization_manager = get_unified_vectorization_manager()
        else:
            self.vectorization_manager = None
        
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        else:
            self.rolling_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend strength with full VectorBT optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'optimized_trend_strength_{self.window}_{self.base_calculation.value}')
        
        # Optimize DataFrame for processing
        if self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe(data)
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate trend strength using optimized rolling correlation
        time_index = pd.Series(range(len(base_values)), index=base_values.index)
        
        if self.vectorization_manager:
            trend_strength = self.vectorization_manager.rolling_operation(
                base_values, 'corr', window=self.window, other=time_index
            )
        elif self.rolling_optimizer:
            trend_strength = self.rolling_optimizer.rolling_corr(
                base_values, time_index, window=self.window
            )
        else:
            # Fallback to pandas
            trend_strength = base_values.rolling(window=self.window).corr(time_index)
        
        return trend_strength.rename(f'optimized_trend_strength_{self.window}_{self.base_calculation.value}')

def create_optimized_acceleration_generators() -> List[FeatureGenerator]:
    """Create all optimized acceleration-based feature generators."""
    generators = []
    
    # Use VectorBT generators if available, otherwise use optimized fallback generators
    if VECTORBT_ACCELERATION_AVAILABLE and VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        generators.extend(create_vectorbt_acceleration_generators())
    else:
        # Use optimized fallback generators
        # Momentum generators for different periods
        for period in [5, 10, 20, 50]:
            generators.append(OptimizedMomentumGenerator(period=period))
        
        # Acceleration generators
        for period in [5, 10]:
            generators.append(OptimizedPriceAccelerationGenerator(period=period))
        
        # Trend strength generators
        for window in [5, 10, 20, 50]:
            generators.append(OptimizedTrendStrengthGenerator(window=window))
    
    return generators

def create_default_optimized_acceleration_generators() -> List[FeatureGenerator]:
    """Create default optimized acceleration-based feature generators."""
    return create_optimized_acceleration_generators()

# Export all generators
__all__ = [
    'OptimizedAccelerationFeatureGenerator',
    'OptimizedMomentumGenerator',
    'OptimizedPriceAccelerationGenerator',
    'OptimizedTrendStrengthGenerator',
    'create_optimized_acceleration_generators',
    'create_default_optimized_acceleration_generators'
]