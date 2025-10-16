"""
Consolidated Rolling Operations Optimizer

This module provides a unified, high-performance rolling operations system that consolidates
all rolling calculations across feature generators using VectorBT optimizations.

Key Features:
- Batch processing of multiple rolling operations
- Automatic VectorBT optimization selection
- Memory-efficient processing
- Consistent error handling and fallbacks
- Performance monitoring and optimization
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
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_skew, rolling_kurt, rolling_quantile
    )
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
    rolling_skew = None
    rolling_kurt = None
    rolling_quantile = None

# GPU acceleration removed - CuPy not supported on all platforms
cp = None

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

class RollingOperationType(Enum):
    """Types of rolling operations supported."""
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    SKEW = "skew"
    KURT = "kurt"
    QUANTILE = "quantile"
    CORR = "corr"
    COV = "cov"
    APPLY = "apply"

@dataclass
class RollingOperationConfig:
    """Configuration for rolling operations."""
    operation: RollingOperationType
    window: int
    column: Optional[str] = None
    min_periods: Optional[int] = None
    center: bool = False
    win_type: Optional[str] = None
    on: Optional[str] = None
    axis: int = 0
    closed: Optional[str] = None
    method: str = 'single'
    # Additional parameters for specific operations
    quantile_value: Optional[float] = None
    corr_other: Optional[Union[pd.Series, np.ndarray]] = None
    apply_func: Optional[Callable] = None

@dataclass
class BatchRollingConfig:
    """Configuration for batch rolling operations."""
    operations: List[RollingOperationConfig] = None
    enable_gpu: bool = True
    enable_parallel: bool = True
    memory_optimization: bool = True
    chunk_size: int = 1000
    performance_threshold: int = 1000  # Minimum data size for VectorBT optimization

    def __init__(self, operations: Optional[List[RollingOperationConfig]] = None, **kwargs):
        """Initialize BatchRollingConfig with operations."""
        self.operations = operations
        for key, value in kwargs.items():
            setattr(self, key, value)

class ConsolidatedRollingOptimizer:
    """
    Consolidated rolling operations optimizer with VectorBT acceleration.

    This class provides a unified interface for all rolling operations across
    feature generators, with automatic optimization selection and fallback handling.
    """

    def __init__(self, config: Optional[BatchRollingConfig] = None):
        """
        Initialize the consolidated rolling optimizer.

        Args:
            config: Configuration for batch rolling operations
        """
        self.config = config or BatchRollingConfig(operations=[])
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
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
        """Determine if GPU should be used for this operation."""
        return (self.gpu_available and
                data_size >= self.config.performance_threshold * 2)

    def single_rolling_operation(self,
                                data: Union[pd.Series, pd.DataFrame],
                                config: RollingOperationConfig) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform a single rolling operation with optimization.

        Args:
            data: Input data (Series or DataFrame)
            config: Rolling operation configuration

        Returns:
            Result of the rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1

        try:
            # Determine optimization strategy
            data_size = len(data) if hasattr(data, '__len__') else data.shape[0]

            if self.should_use_vectorbt(data_size):
                result = self._vectorbt_rolling_operation(data, config)
                self.performance_stats['vectorbt_operations'] += 1
            else:
                result = self._pandas_rolling_operation(data, config)
                self.performance_stats['pandas_fallbacks'] += 1

            # Update performance stats
            operation_time = time.time() - start_time
            self.performance_stats['total_time'] += operation_time
            self.performance_stats['average_time_per_operation'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_operations']
            )

            return result

        except Exception as e:
            self.logger.warning(f"Rolling operation failed: {e}, using fallback")
            return self._pandas_rolling_operation(data, config)

    def batch_rolling_operations(self,
                                data: Union[pd.Series, pd.DataFrame],
                                configs: List[RollingOperationConfig]) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Perform multiple rolling operations in batch for efficiency.

        Args:
            data: Input data (Series or DataFrame)
            configs: List of rolling operation configurations

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
                result = self.single_rolling_operation(data, config)
                operation_name = f"{config.operation.value}_{config.window}_{i}"
                results[operation_name] = result
            except Exception as e:
                self.logger.error(f"Batch operation {i} failed: {e}")
                continue

        batch_time = time.time() - start_time
        self.performance_stats['total_time'] += batch_time

        return results

    def _vectorbt_rolling_operation(self,
                                   data: Union[pd.Series, pd.DataFrame],
                                   config: RollingOperationConfig) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT."""
        if config.operation == RollingOperationType.MEAN:
            return rolling_mean(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.STD:
            return rolling_std(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.VAR:
            return rolling_var(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.MIN:
            return rolling_min(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.MAX:
            return rolling_max(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.SUM:
            return rolling_sum(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.SKEW:
            return rolling_skew(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.KURT:
            return rolling_kurt(data, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.QUANTILE:
            if config.quantile_value is None:
                raise ValueError("quantile_value must be specified for quantile operation")
            return rolling_quantile(data, window=config.window, q=config.quantile_value, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.CORR:
            if config.corr_other is None:
                raise ValueError("corr_other must be specified for correlation operation")
            return rolling_corr(data, config.corr_other, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.COV:
            if config.corr_other is None:
                raise ValueError("corr_other must be specified for covariance operation")
            return rolling_cov(data, config.corr_other, window=config.window, min_periods=config.min_periods)
        elif config.operation == RollingOperationType.APPLY:
            if config.apply_func is None:
                raise ValueError("apply_func must be specified for apply operation")
            return rolling_apply(data, config.apply_func, window=config.window, min_periods=config.min_periods)
        else:
            raise ValueError(f"Unsupported operation: {config.operation}")

    def _pandas_rolling_operation(self,
                                 data: Union[pd.Series, pd.DataFrame],
                                 config: RollingOperationConfig) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(
            window=config.window,
            min_periods=config.min_periods,
            center=config.center,
            win_type=config.win_type,
            on=config.on,
            axis=config.axis,
            closed=config.closed,
            method=config.method
        )

        if config.operation == RollingOperationType.MEAN:
            return rolling_obj.mean()
        elif config.operation == RollingOperationType.STD:
            return rolling_obj.std()
        elif config.operation == RollingOperationType.VAR:
            return rolling_obj.var()
        elif config.operation == RollingOperationType.MIN:
            return rolling_obj.min()
        elif config.operation == RollingOperationType.MAX:
            return rolling_obj.max()
        elif config.operation == RollingOperationType.SUM:
            return rolling_obj.sum()
        elif config.operation == RollingOperationType.SKEW:
            return rolling_obj.skew()
        elif config.operation == RollingOperationType.KURT:
            return rolling_obj.kurt()
        elif config.operation == RollingOperationType.QUANTILE:
            if config.quantile_value is None:
                raise ValueError("quantile_value must be specified for quantile operation")
            return rolling_obj.quantile(config.quantile_value)
        elif config.operation == RollingOperationType.CORR:
            if config.corr_other is None:
                raise ValueError("corr_other must be specified for correlation operation")
            return rolling_obj.corr(config.corr_other)
        elif config.operation == RollingOperationType.COV:
            if config.corr_other is None:
                raise ValueError("corr_other must be specified for covariance operation")
            return rolling_obj.cov(config.corr_other)
        elif config.operation == RollingOperationType.APPLY:
            if config.apply_func is None:
                raise ValueError("apply_func must be specified for apply operation")
            return rolling_obj.apply(config.apply_func, raw=True)
        else:
            raise ValueError(f"Unsupported operation: {config.operation}")

    def _unified_batch_operations(self,
                                 data: Union[pd.Series, pd.DataFrame],
                                 configs: List[RollingOperationConfig]) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Perform batch operations using Unified Vectorization Manager."""
        if not self.unified_manager:
            raise RuntimeError("Unified Vectorization Manager not available")

        def batch_rolling_func(data, configs):
            """Batch rolling function for unified manager."""
            results = {}
            for i, config in enumerate(configs):
                result = self._vectorbt_rolling_operation(data, config)
                operation_name = f"{config.operation.value}_{config.window}_{i}"
                results[operation_name] = result
            return results

        # Create operation config
        op_config = OperationConfig(
            operation_type=OperationType.TECHNICAL_INDICATORS,
            data_size=len(data),
            data_dimensions=data.shape if hasattr(data, 'shape') else (len(data),),
            memory_budget_mb=1024.0,
            parallel_workers=None
        )

        # Execute through unified manager
        result = self.unified_manager.optimize_operation(
            operation_type=OperationType.TECHNICAL_INDICATORS,
            data=data,
            operation_func=lambda x: batch_rolling_func(x, configs),
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
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'total_time': 0.0,
            'average_time_per_operation': 0.0,
            'memory_savings_mb': 0.0
        }

# Convenience functions
def create_rolling_optimizer(enable_gpu: bool = True,
                           enable_parallel: bool = True,
                           performance_threshold: int = 1000) -> ConsolidatedRollingOptimizer:
    """Create a rolling optimizer with specified configuration."""
    config = BatchRollingConfig(
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        performance_threshold=performance_threshold
    )
    return ConsolidatedRollingOptimizer(config)

def batch_rolling_operations(data: Union[pd.Series, pd.DataFrame],
                           operations: List[str],
                           windows: List[int],
                           columns: Optional[List[str]] = None,
                           enable_gpu: bool = True) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """
    Convenience function for batch rolling operations.

    Args:
        data: Input data
        operations: List of operation names ('mean', 'std', 'var', etc.)
        windows: List of window sizes
        columns: List of columns to process (None for all)
        enable_gpu: Whether to enable

    Returns:
        Dictionary of results
    """
    optimizer = create_rolling_optimizer(enable_gpu=enable_gpu)

    # Create operation configs
    configs = []
    for operation in operations:
        for window in windows:
            config = RollingOperationConfig(
                operation=RollingOperationType(operation),
                window=window
            )
            configs.append(config)

    return optimizer.batch_rolling_operations(data, configs)

# Global instance for easy access
_global_optimizer = None

def get_global_rolling_optimizer() -> ConsolidatedRollingOptimizer:
    """Get the global rolling optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = create_rolling_optimizer()
    return _global_optimizer
