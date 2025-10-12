"""
Backwards Compatible Acceleration Feature Generators

This module maintains full backwards compatibility while adding optional VectorBT optimizations.
All existing APIs and functionality remain unchanged.

Key Features:
- Full backwards compatibility
- Optional VectorBT optimizations
- Graceful fallbacks
- No breaking changes
- Enhanced performance when optimizations are available
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities (existing)
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Optional VectorBT optimizations (new)
try:
    from ..utils.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None

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

# Import VectorBT-optimized acceleration generators (existing)
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

class AccelerationFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for acceleration-based features with optional VectorBT optimizations.
    
    This class maintains full backwards compatibility while adding optional performance enhancements.
    All existing APIs and functionality remain unchanged.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, enable_optimizations: bool = True):
        """
        Initialize acceleration feature generator.
        
        Args:
            config: Feature configuration (optional)
            enable_optimizations: Whether to enable optional VectorBT optimizations (default: True)
        """
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize optional optimizations (backwards compatible)
        self.enable_optimizations = enable_optimizations
        self.vectorization_manager = None
        self.rolling_optimizer = None
        
        if enable_optimizations:
            # Initialize Unified Vectorization Manager (optional)
            if UNIFIED_VECTORIZATION_AVAILABLE:
                try:
                    self.vectorization_manager = get_unified_vectorization_manager()
                except Exception as e:
                    self.logger.warning(f"UnifiedVectorizationManager not available: {e}")
            
            # Initialize VectorBT Rolling Optimizer (optional)
            if ROLLING_OPTIMIZER_AVAILABLE:
                try:
                    self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                except Exception as e:
                    self.logger.warning(f"VectorBTRollingOptimizer not available: {e}")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="acceleration_features",
            category=FeatureCategory.ACCELERATION,
            description="Comprehensive acceleration features including momentum, acceleration, and jerk",
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
            gpu_accelerated=False
        )
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing with optional enhancements.
        
        This method maintains backwards compatibility while adding optional optimizations.
        """
        # Try new optimizations first (if available)
        if self.enable_optimizations and self.vectorization_manager:
            try:
                return self.vectorization_manager.optimize_dataframe(data)
            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager optimization failed: {e}")
        
        # Fallback to existing optimization (backwards compatible)
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        
        # No optimization available
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with optional enhancements.
        
        This method maintains backwards compatibility while adding optional optimizations.
        """
        # Try new optimizations first (if available)
        if self.enable_optimizations and self.vectorization_manager:
            try:
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
            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager batch operations failed: {e}")
        
        # Fallback to existing optimization (backwards compatible)
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        
        # No optimization available - return original data
        return data
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, 
                                   window: int, **kwargs) -> pd.Series:
        """Perform rolling operation with optional optimizations.
        
        This method maintains backwards compatibility while adding optional optimizations.
        """
        # Try new optimizations first (if available)
        if self.enable_optimizations and self.vectorization_manager:
            try:
                return self.vectorization_manager.rolling_operation(data, operation, window, **kwargs)
            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager rolling operation failed: {e}")
        
        # Try VectorBT Rolling Optimizer (if available)
        if self.enable_optimizations and self.rolling_optimizer:
            try:
                return self.rolling_optimizer._rolling_operation(data, operation, window, **kwargs)
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer rolling operation failed: {e}")
        
        # Fallback to existing implementation (backwards compatible)
        if hasattr(self, '_vectorbt_rolling_operation'):
            return self._vectorbt_rolling_operation(data, operation, window, **kwargs)
        
        # Final fallback to pandas
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

class MomentumGenerator(FeatureGenerator):
    """Generator for price momentum features with optional optimizations."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, 
                 enable_optimizations: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"momentum_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Price momentum over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        self.enable_optimizations = enable_optimizations
        
        # Initialize optional optimizations
        self.vectorization_manager = None
        self.rolling_optimizer = None
        
        if enable_optimizations:
            if UNIFIED_VECTORIZATION_AVAILABLE:
                try:
                    self.vectorization_manager = get_unified_vectorization_manager()
                except Exception as e:
                    self.logger.warning(f"UnifiedVectorizationManager not available: {e}")
            
            if ROLLING_OPTIMIZER_AVAILABLE:
                try:
                    self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                except Exception as e:
                    self.logger.warning(f"VectorBTRollingOptimizer not available: {e}")
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum with optional optimizations."""
        # Optimize DataFrame for processing (if optimizations enabled)
        if self.enable_optimizations and self.vectorization_manager:
            try:
                data = self.vectorization_manager.optimize_dataframe(data)
            except Exception as e:
                self.logger.warning(f"DataFrame optimization failed: {e}")
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum with proper handling using safe math utilities
        shifted_values = base_values.shift(self.period)
        
        # Use safe percentage change calculation
        momentum_values = []
        for i in range(len(base_values)):
            current_val = base_values.iloc[i]
            shifted_val = shifted_values.iloc[i]
            
            # Use safe percentage change function
            momentum_val = safe_percentage_change(shifted_val, current_val)
            momentum_values.append(momentum_val)
        
        momentum_series = pd.Series(momentum_values, index=data.index, name=f'momentum_{self.period}_{self.base_calculation.value}')
        
        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(momentum_series.values, f"Momentum_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(momentum_series.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)
                
                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"
                
                # Only log once per feature globally to reduce verbosity
                feature_key = f"Momentum_{self.period}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(MomentumGenerator, '_logged_warnings'):
                    MomentumGenerator._logged_warnings = set()
                if feature_key not in MomentumGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    MomentumGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")
        
        return momentum_series

class PriceAccelerationGenerator(FeatureGenerator):
    """Generator for price acceleration features with optional optimizations."""
    
    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, 
                 enable_optimizations: bool = True, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"Price acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        self.enable_optimizations = enable_optimizations
        
        # Initialize optional optimizations
        self.vectorization_manager = None
        self.rolling_optimizer = None
        
        if enable_optimizations:
            if UNIFIED_VECTORIZATION_AVAILABLE:
                try:
                    self.vectorization_manager = get_unified_vectorization_manager()
                except Exception as e:
                    self.logger.warning(f"UnifiedVectorizationManager not available: {e}")
            
            if ROLLING_OPTIMIZER_AVAILABLE:
                try:
                    self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                except Exception as e:
                    self.logger.warning(f"VectorBTRollingOptimizer not available: {e}")
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration with optional optimizations."""
        # Optimize DataFrame for processing (if optimizations enabled)
        if self.enable_optimizations and self.vectorization_manager:
            try:
                data = self.vectorization_manager.optimize_dataframe(data)
            except Exception as e:
                self.logger.warning(f"DataFrame optimization failed: {e}")
        
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum with proper handling using safe math utilities
        shifted_values = base_values.shift(self.period)
        
        # Use safe percentage change calculation for momentum
        momentum_values = []
        for i in range(len(base_values)):
            current_val = base_values.iloc[i]
            shifted_val = shifted_values.iloc[i]
            
            # Use safe percentage change function
            momentum_val = safe_percentage_change(shifted_val, current_val)
            momentum_values.append(momentum_val)
        
        momentum_series = pd.Series(momentum_values, index=data.index)
        
        # Calculate acceleration (second derivative) using optimized operations if available
        if self.enable_optimizations and self.vectorization_manager:
            try:
                acceleration = self.vectorization_manager.rolling_operation(momentum_series, 'diff', self.period)
            except Exception as e:
                self.logger.warning(f"Optimized acceleration calculation failed: {e}")
                acceleration = momentum_series.diff(self.period)
        else:
            acceleration = momentum_series.diff(self.period)
        
        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(acceleration.values, f"Acceleration_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(acceleration.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)
                
                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"
                
                # Only log once per feature globally to reduce verbosity
                feature_key = f"Acceleration_{self.period}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(PriceAccelerationGenerator, '_logged_warnings'):
                    PriceAccelerationGenerator._logged_warnings = set()
                if feature_key not in PriceAccelerationGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    PriceAccelerationGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")
        
        return acceleration

def create_acceleration_generators(enable_optimizations: bool = True) -> List[FeatureGenerator]:
    """Create all acceleration-based feature generators with optional optimizations.
    
    Args:
        enable_optimizations: Whether to enable optional VectorBT optimizations (default: True)
    
    Returns:
        List of acceleration feature generators
    """
    generators = []
    
    # Use VectorBT generators if available, otherwise fall back to legacy generators
    if VECTORBT_ACCELERATION_AVAILABLE and VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        generators.extend(create_vectorbt_acceleration_generators())
    else:
        # Fall back to legacy generators with optional optimizations
        # Momentum generators for different periods
        for period in [5, 10, 20, 50]:
            generators.append(MomentumGenerator(period=period, enable_optimizations=enable_optimizations))
        
        # Acceleration generators
        for period in [5, 10]:
            generators.append(PriceAccelerationGenerator(period=period, enable_optimizations=enable_optimizations))
    
    return generators

def create_default_acceleration_generators(enable_optimizations: bool = True) -> List[FeatureGenerator]:
    """Create default acceleration-based feature generators with optional optimizations.
    
    Args:
        enable_optimizations: Whether to enable optional VectorBT optimizations (default: True)
    
    Returns:
        List of acceleration feature generators
    """
    return create_acceleration_generators(enable_optimizations)

# Export all generators (maintaining backwards compatibility)
__all__ = [
    'AccelerationFeatureGenerator',
    'MomentumGenerator',
    'PriceAccelerationGenerator',
    'create_acceleration_generators',
    'create_default_acceleration_generators'
]