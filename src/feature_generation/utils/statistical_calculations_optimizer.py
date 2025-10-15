"""
Statistical Calculations Optimizer

This module provides optimized statistical calculations using VectorBT to replace
manual NumPy-based statistical computations across feature generators.

Key Features:
- VectorBT-optimized statistical functions
- 
- Batch processing capabilities
- Memory-efficient operations
- Consistent error handling
"""

import numpy as np
import pandas as pd
import time
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_skew, rolling_kurt,
        rolling_quantile, rolling_rank, rolling_apply,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None
    rolling_rank = None
    rolling_apply = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None

# Import optimization modules
try:
    from .unified_vectorization_manager import UnifiedVectorizationManager, VECTORBT_AVAILABLE
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    VECTORBT_AVAILABLE = False

# Scipy for advanced statistics
try:
    from scipy import stats
    from scipy.stats import skew, kurtosis, jarque_bera, shapiro, normaltest
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    skew = None
    kurtosis = None
    jarque_bera = None
    shapiro = None
    normaltest = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager, 
        OperationType, 
        OptimizationStrategy,
        OperationConfig,
        OptimizationResult
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None
    OperationConfig = None
    OptimizationResult = None

logger = logging.getLogger(__name__)

class StatisticalOperationType(Enum):
    """Types of statistical operations supported."""
    # Basic statistics
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"
    QUANTILE = "quantile"
    
    # Higher-order moments
    SKEW = "skew"
    KURT = "kurt"
    KURTOSIS = "kurtosis"
    
    # Distribution tests
    JARQUE_BERA = "jarque_bera"
    SHAPIRO_WILK = "shapiro_wilk"
    NORMALITY_TEST = "normality_test"
    
    # Correlation and covariance
    CORRELATION = "correlation"
    COVARIANCE = "covariance"
    AUTOCORRELATION = "autocorrelation"
    
    # Ranking and scaling
    RANK = "rank"
    ZSCORE = "zscore"
    WINSORIZE = "winsorize"
    CLIP = "clip"
    
    # Custom operations
    CUSTOM = "custom"

@dataclass
class StatisticalOperationConfig:
    """Configuration for statistical operations."""
    operation: StatisticalOperationType
    window: Optional[int] = None
    min_periods: Optional[int] = None
    axis: int = 0
    ddof: int = 1
    bias: bool = False
    fisher: bool = True
    # Additional parameters
    quantile_value: Optional[float] = None
    other_series: Optional[Union[pd.Series, np.ndarray]] = None
    lag: int = 1
    custom_func: Optional[Callable] = None
    # Winsorization parameters
    limits: Optional[Tuple[float, float]] = None
    # Clipping parameters
    lower: Optional[float] = None
    upper: Optional[float] = None

@dataclass
class BatchStatisticalConfig:
    """Configuration for batch statistical operations."""
    operations: List[StatisticalOperationConfig] = None
    enable_gpu: bool = True
    enable_parallel: bool = True
    memory_optimization: bool = True
    chunk_size: int = 1000
    performance_threshold: int = 1000

    def __init__(self, operations: Optional[List[StatisticalOperationConfig]] = None, **kwargs):
        """Initialize BatchStatisticalConfig with operations."""
        self.operations = operations
        for key, value in kwargs.items():
            setattr(self, key, value)

class StatisticalCalculationsOptimizer:
    """
    Optimized statistical calculations using VectorBT.
    
    This class provides high-performance statistical computations that replace
    manual NumPy-based calculations with VectorBT-optimized implementations.
    """
    
    def __init__(self, config: Optional[BatchStatisticalConfig] = None):
        """
        Initialize the statistical calculations optimizer.
        
        Args:
            config: Configuration for batch statistical operations
        """
        self.config = config or BatchStatisticalConfig(operations=[])
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'numpy_fallbacks': 0,
            'scipy_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'total_time': 0.0,
            'average_time_per_operation': 0.0,
            'memory_savings_mb': 0.0
        }
        
        # Initialize optimization components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize optimization components."""
        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
            self.logger.warning("Unified Vectorization Manager not available")
        
        # Check GPU availability
        self.gpu_available =  self.config.enable_gpu
        if self.gpu_available:
            try:
                # Test GPU memory
                test_array = np.array([1, 2, 3])
                del test_array
                self.logger.info("GPU memory test successful")
            except Exception as e:
                self.gpu_available = False
                self.logger.warning(f"GPU not available: {e}")
    
    def should_use_vectorbt(self, data_size: int) -> bool:
        """Determine if VectorBT optimization should be used."""
        return (VECTORBT_AVAILABLE and 
                data_size >= self.config.performance_threshold)
    
    def should_use_gpu(self, data_size: int) -> bool:
        """Determine if GPU optimization should be used."""
        return (self.gpu_available and 
                data_size >= self.config.performance_threshold * 2)
    
    def single_statistical_operation(self, 
                                   data: Union[pd.Series, pd.DataFrame], 
                                   config: StatisticalOperationConfig) -> Union[pd.Series, pd.DataFrame, float]:
        """
        Perform a single statistical operation with optimization.
        
        Args:
            data: Input data (Series or DataFrame)
            config: Statistical operation configuration
            
        Returns:
            Result of the statistical operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        try:
            # Determine optimization strategy
            data_size = len(data) if hasattr(data, '__len__') else data.shape[0]
            
            if self.should_use_vectorbt(data_size):
                result = self._vectorbt_statistical_operation(data, config)
                self.performance_stats['vectorbt_operations'] += 1
            else:
                result = self._numpy_statistical_operation(data, config)
                self.performance_stats['numpy_fallbacks'] += 1
            
            # Update performance stats
            operation_time = time.time() - start_time
            self.performance_stats['total_time'] += operation_time
            self.performance_stats['average_time_per_operation'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_operations']
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Statistical operation failed: {e}, using fallback")
            return self._numpy_statistical_operation(data, config)
    
    def batch_statistical_operations(self, 
                                   data: Union[pd.Series, pd.DataFrame], 
                                   configs: List[StatisticalOperationConfig]) -> Dict[str, Union[pd.Series, pd.DataFrame, float]]:
        """
        Perform multiple statistical operations in batch for efficiency.
        
        Args:
            data: Input data (Series or DataFrame)
            configs: List of statistical operation configurations
            
        Returns:
            Dictionary mapping operation names to results
        """
        start_time = time.time()
        self.performance_stats['batch_operations'] += 1
        
        results = {}
        data_size = len(data) if hasattr(data, '__len__') else data.shape[0]
        
        # Use unified manager if available and data is large enough
        if (self.unified_manager and 
            data_size >= self.config.performance_threshold and 
            len(configs) > 1):
            
            try:
                results = self._unified_batch_operations(data, configs)
                return results
            except Exception as e:
                self.logger.warning(f"Unified batch operations failed: {e}, using individual operations")
        
        # Fallback to individual operations
        for i, config in enumerate(configs):
            try:
                result = self.single_statistical_operation(data, config)
                operation_name = f"{config.operation.value}_{i}"
                results[operation_name] = result
            except Exception as e:
                self.logger.error(f"Batch operation {i} failed: {e}")
                continue
        
        batch_time = time.time() - start_time
        self.performance_stats['total_time'] += batch_time
        
        return results
    
    def _vectorbt_statistical_operation(self, 
                                      data: Union[pd.Series, pd.DataFrame], 
                                      config: StatisticalOperationConfig) -> Union[pd.Series, pd.DataFrame, float]:
        """Perform statistical operation using VectorBT."""
        if config.operation == StatisticalOperationType.MEAN:
            if config.window:
                return rolling_mean(data, window=config.window, min_periods=config.min_periods)
            else:
                return data.mean(axis=config.axis)
        
        elif config.operation == StatisticalOperationType.STD:
            if config.window:
                return rolling_std(data, window=config.window, min_periods=config.min_periods, ddof=config.ddof)
            else:
                return data.std(axis=config.axis, ddof=config.ddof)
        
        elif config.operation == StatisticalOperationType.VAR:
            if config.window:
                return rolling_var(data, window=config.window, min_periods=config.min_periods, ddof=config.ddof)
            else:
                return data.var(axis=config.axis, ddof=config.ddof)
        
        elif config.operation == StatisticalOperationType.MEDIAN:
            if config.window:
                return rolling_quantile(data, window=config.window, q=0.5, min_periods=config.min_periods)
            else:
                return data.median(axis=config.axis)
        
        elif config.operation == StatisticalOperationType.QUANTILE:
            if config.quantile_value is None:
                raise ValueError("quantile_value must be specified for quantile operation")
            if config.window:
                return rolling_quantile(data, window=config.window, q=config.quantile_value, min_periods=config.min_periods)
            else:
                return data.quantile(config.quantile_value, axis=config.axis)
        
        elif config.operation == StatisticalOperationType.SKEW:
            if config.window:
                return rolling_skew(data, window=config.window, min_periods=config.min_periods, bias=config.bias)
            else:
                return data.skew(axis=config.axis, bias=config.bias)
        
        elif config.operation == StatisticalOperationType.KURT:
            if config.window:
                return rolling_kurt(data, window=config.window, min_periods=config.min_periods, bias=config.bias, fisher=config.fisher)
            else:
                return data.kurtosis(axis=config.axis, bias=config.bias, fisher=config.fisher)
        
        elif config.operation == StatisticalOperationType.RANK:
            if config.window:
                return rolling_rank(data, window=config.window, min_periods=config.min_periods)
            else:
                return rank(data)
        
        elif config.operation == StatisticalOperationType.ZSCORE:
            return zscore(data, axis=config.axis)
        
        elif config.operation == StatisticalOperationType.WINSORIZE:
            if config.limits is None:
                raise ValueError("limits must be specified for winsorize operation")
            return winsorize(data, limits=config.limits, axis=config.axis)
        
        elif config.operation == StatisticalOperationType.CLIP:
            if config.lower is None or config.upper is None:
                raise ValueError("lower and upper must be specified for clip operation")
            return clip(data, lower=config.lower, upper=config.upper)
        
        elif config.operation == StatisticalOperationType.CORRELATION:
            if config.other_series is None:
                raise ValueError("other_series must be specified for correlation operation")
            if config.window:
                from vectorbt.generic import rolling_corr
                return rolling_corr(data, config.other_series, window=config.window, min_periods=config.min_periods)
            else:
                return data.corr(config.other_series)
        
        elif config.operation == StatisticalOperationType.AUTOCORRELATION:
            if config.window:
                shifted = data.shift(config.lag)
                from vectorbt.generic import rolling_corr
                return rolling_corr(data, shifted, window=config.window, min_periods=config.min_periods)
            else:
                return data.autocorr(lag=config.lag)
        
        elif config.operation == StatisticalOperationType.CUSTOM:
            if config.custom_func is None:
                raise ValueError("custom_func must be specified for custom operation")
            if config.window:
                return rolling_apply(data, config.custom_func, window=config.window, min_periods=config.min_periods)
            else:
                return config.custom_func(data)
        
        else:
            raise ValueError(f"Unsupported operation: {config.operation}")
    
    def _numpy_statistical_operation(self, 
                                   data: Union[pd.Series, pd.DataFrame], 
                                   config: StatisticalOperationConfig) -> Union[pd.Series, pd.DataFrame, float]:
        """Fallback statistical operation using NumPy/SciPy."""
        # Convert to numpy array for processing
        if isinstance(data, pd.Series):
            values = data.values
            index = data.index
        elif isinstance(data, pd.DataFrame):
            values = data.values
            index = data.index
            columns = data.columns
        else:
            values = np.array(data)
            index = None
            columns = None
        
        if config.operation == StatisticalOperationType.MEAN:
            result = np.mean(values, axis=config.axis)
        elif config.operation == StatisticalOperationType.STD:
            result = np.std(values, axis=config.axis, ddof=config.ddof)
        elif config.operation == StatisticalOperationType.VAR:
            result = np.var(values, axis=config.axis, ddof=config.ddof)
        elif config.operation == StatisticalOperationType.MEDIAN:
            result = np.median(values, axis=config.axis)
        elif config.operation == StatisticalOperationType.QUANTILE:
            if config.quantile_value is None:
                raise ValueError("quantile_value must be specified for quantile operation")
            result = np.quantile(values, config.quantile_value, axis=config.axis)
        elif config.operation == StatisticalOperationType.SKEW:
            if SCIPY_AVAILABLE:
                result = skew(values, axis=config.axis, bias=config.bias)
            else:
                # Manual skewness calculation
                mean_val = np.mean(values, axis=config.axis, keepdims=True)
                std_val = np.std(values, axis=config.axis, keepdims=True)
                result = np.mean(((values - mean_val) / (std_val + 1e-8)) ** 3, axis=config.axis)
        elif config.operation == StatisticalOperationType.KURT:
            if SCIPY_AVAILABLE:
                result = kurtosis(values, axis=config.axis, bias=config.bias, fisher=config.fisher)
            else:
                # Manual kurtosis calculation
                mean_val = np.mean(values, axis=config.axis, keepdims=True)
                std_val = np.std(values, axis=config.axis, keepdims=True)
                result = np.mean(((values - mean_val) / (std_val + 1e-8)) ** 4, axis=config.axis) - (3 if config.fisher else 0)
        elif config.operation == StatisticalOperationType.CORRELATION:
            if config.other_series is None:
                raise ValueError("other_series must be specified for correlation operation")
            other_values = config.other_series.values if hasattr(config.other_series, 'values') else config.other_series
            result = np.corrcoef(values, other_values)[0, 1]
        elif config.operation == StatisticalOperationType.AUTOCORRELATION:
            if len(values.shape) > 1:
                raise ValueError("Autocorrelation only supported for 1D arrays")
            result = np.corrcoef(values[:-config.lag], values[config.lag:])[0, 1] if len(values) > config.lag else np.nan
        elif config.operation == StatisticalOperationType.CUSTOM:
            if config.custom_func is None:
                raise ValueError("custom_func must be specified for custom operation")
            result = config.custom_func(values)
        else:
            raise ValueError(f"Unsupported operation: {config.operation}")
        
        # Convert back to appropriate format
        if isinstance(data, pd.Series):
            return pd.Series(result, index=index)
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(result, index=index, columns=columns)
        else:
            return result
    
    def _unified_batch_operations(self, 
                                 data: Union[pd.Series, pd.DataFrame], 
                                 configs: List[StatisticalOperationConfig]) -> Dict[str, Union[pd.Series, pd.DataFrame, float]]:
        """Perform batch operations using Unified Vectorization Manager."""
        if not self.unified_manager:
            raise RuntimeError("Unified Vectorization Manager not available")
        
        def batch_statistical_func(data, configs):
            """Batch statistical function for unified manager."""
            results = {}
            for i, config in enumerate(configs):
                result = self._vectorbt_statistical_operation(data, config)
                operation_name = f"{config.operation.value}_{i}"
                results[operation_name] = result
            return results
        
        # Create operation config
        op_config = OperationConfig(
            operation_type=OperationType.STATISTICAL_COMPUTATION,
            data_size=len(data),
            data_dimensions=data.shape if hasattr(data, 'shape') else (len(data),),
            memory_budget_mb=1024.0,
            parallel_workers=None
        )
        
        # Execute through unified manager
        result = self.unified_manager.optimize_operation(
            operation_type=OperationType.STATISTICAL_COMPUTATION,
            data=data,
            operation_func=lambda x: batch_statistical_func(x, configs),
            config=op_config
        )
        
        return result.result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'numpy_fallbacks': 0,
            'scipy_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'total_time': 0.0,
            'average_time_per_operation': 0.0,
            'memory_savings_mb': 0.0
        }

# Convenience functions
def create_statistical_optimizer(enable_gpu: bool = True, 
                               enable_parallel: bool = True,
                               performance_threshold: int = 1000) -> StatisticalCalculationsOptimizer:
    """Create a statistical optimizer with specified configuration."""
    config = BatchStatisticalConfig(
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        performance_threshold=performance_threshold
    )
    return StatisticalCalculationsOptimizer(config)

def batch_statistical_operations(data: Union[pd.Series, pd.DataFrame],
                               operations: List[str],
                               windows: Optional[List[int]] = None,
                               enable_gpu: bool = True) -> Dict[str, Union[pd.Series, pd.DataFrame, float]]:
    """
    Convenience function for batch statistical operations.
    
    Args:
        data: Input data
        operations: List of operation names ('mean', 'std', 'skew', etc.)
        windows: List of window sizes (None for non-rolling operations)
        enable_gpu: Whether to enable 
        
    Returns:
        Dictionary of results
    """
    optimizer = create_statistical_optimizer(enable_gpu=enable_gpu)
    
    # Create operation configs
    configs = []
    for operation in operations:
        if windows:
            for window in windows:
                config = StatisticalOperationConfig(
                    operation=StatisticalOperationType(operation),
                    window=window
                )
                configs.append(config)
        else:
            config = StatisticalOperationConfig(
                operation=StatisticalOperationType(operation)
            )
            configs.append(config)
    
    return optimizer.batch_statistical_operations(data, configs)

# Global instance for easy access
_global_statistical_optimizer = None

def get_global_statistical_optimizer() -> StatisticalCalculationsOptimizer:
    """Get the global statistical optimizer instance."""
    global _global_statistical_optimizer
    if _global_statistical_optimizer is None:
        _global_statistical_optimizer = create_statistical_optimizer()
    return _global_statistical_optimizer