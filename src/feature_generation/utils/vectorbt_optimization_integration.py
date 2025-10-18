"""
VectorBT Optimization Integration Module

This module provides a unified interface for all VectorBT optimizations,
integrating enhanced scaling, rolling operations, and memory management.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
from contextlib import contextmanager

# Import our optimization modules
from .vectorbt_memory_optimizer import (
    get_memory_optimizer, get_performance_profiler,
    optimize_dataframe_memory, process_with_memory_management
)
from .vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
# Lazy import to avoid circular dependencies
def _get_scaler_imports():
    """Direct scaling functions to avoid circular dependencies."""
    # Define available scaling methods directly
    def get_available_scaling_methods() -> List[str]:
        return ['zscore', 'minmax', 'robust']

    # Simple scaler class replacement
    class SimpleScaler:
        def __init__(self, method: str = 'zscore'):
            self.method = method

        def fit_transform(self, data: pd.Series) -> pd.Series:
            if self.method == 'zscore':
                return (data - data.mean()) / data.std()
            elif self.method == 'minmax':
                return (data - data.min()) / (data.max() - data.min())
            elif self.method == 'robust':
                median = data.median()
                mad = (data - median).abs().median()
                return (data - median) / mad
            else:
                return (data - data.mean()) / data.std()

    class SimpleBatchScaler:
        def __init__(self, method: str = 'zscore'):
            self.method = method
            self.scalers = {}

        def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            for col in data.columns:
                scaler = SimpleScaler(self.method)
                result[col] = scaler.fit_transform(data[col])
            return result

    return SimpleScaler, SimpleBatchScaler, get_available_scaling_methods

logger = logging.getLogger(__name__)

class VectorBTOptimizationManager:
    """
    Unified manager for all VectorBT optimizations.

    This class provides a single interface for:
    - Enhanced scaling and transforms
    - Optimized rolling operations
    - Memory and performance optimization
    - Batch processing
    - Performance monitoring
    """

    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True,
                 memory_efficient: bool = True, max_memory_gb: float = 8.0,
                 chunk_size: int = 1000, enable_monitoring: bool = True):
        """
        Initialize VectorBT optimization manager.

        Args:
            enable_gpu: Enable GPU acceleration
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            max_memory_gb: Maximum memory usage in GB
            chunk_size: Default chunk size for processing
            enable_monitoring: Enable performance monitoring
        """
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.max_memory_gb = max_memory_gb
        self.chunk_size = chunk_size
        self.enable_monitoring = enable_monitoring

        # Initialize components
        self.memory_optimizer = get_memory_optimizer(
            max_memory_gb=max_memory_gb,
            enable_gpu=enable_gpu,
            chunk_size=chunk_size,
            enable_monitoring=enable_monitoring
        )

        self.rolling_optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=enable_gpu,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient,
            chunk_size=chunk_size
        )

        self.performance_profiler = get_performance_profiler(
            enable_detailed_profiling=enable_monitoring
        )

        # Performance tracking
        self.optimization_stats = {
            'total_operations': 0,
            'scaling_operations': 0,
            'rolling_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0
        }

        # Reduced verbosity - only log once per session
        if not hasattr(VectorBTOptimizationManager, '_logged_initialization'):
            logger.info(f"VectorBTOptimizationManager initialized: "
                       f"GPU={enable_gpu}, Memory={memory_efficient}, "
                       f"Parallel={enable_parallel}")
            VectorBTOptimizationManager._logged_initialization = True

    def scale_data(self, data: Union[pd.Series, pd.DataFrame],
                   method: str = 'zscore', **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Scale data using enhanced VectorBT scaler.

        Args:
            data: Input data
            method: Scaling method
            **kwargs: Additional parameters

        Returns:
            Scaled data
        """
        start_time = time.time()

        # Optimize data for processing
        if self.memory_efficient:
            data = self.memory_optimizer.optimize_dataframe(data)

        # Create scaler
        VectorBTScaler, VectorBTBatchScaler, get_available_scaling_methods = _get_scaler_imports()
        if VectorBTBatchScaler is None or VectorBTScaler is None:
            raise ImportError("VectorBT scaler modules not available")

        if isinstance(data, pd.DataFrame):
            scaler = VectorBTBatchScaler(
                method=method,
                enable_gpu=self.enable_gpu,
                memory_efficient=self.memory_efficient,
                **kwargs
            )
        else:
            scaler = VectorBTScaler(
                method=method,
                enable_gpu=self.enable_gpu,
                memory_efficient=self.memory_efficient,
                **kwargs
            )

        # Scale data
        result = scaler.fit_transform(data)

        # Update stats
        self.optimization_stats['scaling_operations'] += 1
        self.optimization_stats['total_operations'] += 1
        self.optimization_stats['total_time'] += time.time() - start_time

        if hasattr(scaler, 'performance_stats'):
            if scaler.performance_stats.get('gpu_operations', 0) > 0:
                self.optimization_stats['gpu_operations'] += 1
            if scaler.performance_stats.get('memory_optimizations', 0) > 0:
                self.optimization_stats['memory_optimizations'] += 1

        return result

    def rolling_operation(self, data: Union[pd.Series, pd.DataFrame],
                         operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform rolling operation with optimization.

        Args:
            data: Input data
            operation: Operation type
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of rolling operation
        """
        start_time = time.time()

        # Optimize data for processing
        if self.memory_efficient:
            data = self.memory_optimizer.optimize_dataframe(data)

        # Perform rolling operation
        if operation == 'mean':
            result = self.rolling_optimizer.rolling_mean(data, window, **kwargs)
        elif operation == 'std':
            result = self.rolling_optimizer.rolling_std(data, window, **kwargs)
        elif operation == 'var':
            result = self.rolling_optimizer.rolling_var(data, window, **kwargs)
        elif operation == 'min':
            result = self.rolling_optimizer.rolling_min(data, window, **kwargs)
        elif operation == 'max':
            result = self.rolling_optimizer.rolling_max(data, window, **kwargs)
        elif operation == 'sum':
            result = self.rolling_optimizer.rolling_sum(data, window, **kwargs)
        elif operation == 'quantile':
            q = kwargs.pop('q', 0.5)
            result = self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
        elif operation == 'skew':
            result = self.rolling_optimizer.rolling_skew(data, window, **kwargs)
        elif operation == 'kurt':
            result = self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
        elif operation == 'corr':
            other = kwargs.pop('other', None)
            result = self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
        elif operation == 'cov':
            other = kwargs.pop('other', None)
            result = self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
        elif operation == 'apply':
            func = kwargs.pop('func', None)
            result = self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}")

        # Update stats
        self.optimization_stats['rolling_operations'] += 1
        self.optimization_stats['total_operations'] += 1
        self.optimization_stats['total_time'] += time.time() - start_time

        return result

    def batch_process_features(self, data: pd.DataFrame,
                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process multiple features in batch with optimization.

        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated features
        """
        start_time = time.time()

        # Optimize data for processing
        if self.memory_efficient:
            data = self.memory_optimizer.optimize_dataframe(data)

        # Process features in chunks if data is large
        if len(data) > self.chunk_size and self.memory_efficient:
            result = self.memory_optimizer.process_in_chunks(
                data, self._process_features_chunk, feature_configs
            )
        else:
            result = self._process_features_chunk(data, feature_configs)

        # Update stats
        self.optimization_stats['batch_operations'] += 1
        self.optimization_stats['total_operations'] += 1
        self.optimization_stats['total_time'] += time.time() - start_time

        return result

    def _process_features_chunk(self, data: pd.DataFrame,
                              feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a chunk of features."""
        results = {}

        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'rolling')
            params = config.get('params', {})

            try:
                if feature_type == 'rolling':
                    operation = params.get('operation', 'mean')
                    window = params.get('window', 20)
                    column = params.get('column', 'close')

                    if column in data.columns:
                        results[feature_name] = self.rolling_operation(
                            data[column], operation, window, **params
                        )

                elif feature_type == 'scaling':
                    method = params.get('method', 'zscore')
                    column = params.get('column', 'close')

                    if column in data.columns:
                        results[feature_name] = self.scale_data(
                            data[column], method, **params
                        )

                elif feature_type == 'custom':
                    # Custom feature processing
                    func = params.get('function')
                    if callable(func):
                        results[feature_name] = func(data, **params)

            except Exception as e:
                logger.warning(f"Feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

    def optimize_dataframe(self, data: pd.DataFrame,
                          target_dtype: str = 'auto') -> pd.DataFrame:
        """
        Optimize DataFrame for memory efficiency.

        Args:
            data: Input DataFrame
            target_dtype: Target data type

        Returns:
            Optimized DataFrame
        """
        return self.memory_optimizer.optimize_dataframe(data, target_dtype)

    def get_available_scaling_methods(self) -> List[str]:
        """Get list of available scaling methods."""
        VectorBTScaler, VectorBTBatchScaler, get_available_scaling_methods = _get_scaler_imports()
        if get_available_scaling_methods is None:
            return []
        return get_available_scaling_methods()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = self.optimization_stats.copy()

        # Add memory stats
        memory_stats = self.memory_optimizer.get_memory_usage()
        stats.update(memory_stats)

        # Add rolling optimizer stats
        rolling_stats = self.rolling_optimizer.get_performance_stats()
        stats.update(rolling_stats)

        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['scaling_percentage'] = stats['scaling_operations'] / stats['total_operations'] * 100
            stats['rolling_percentage'] = stats['rolling_operations'] / stats['total_operations'] * 100
            stats['batch_percentage'] = stats['batch_operations'] / stats['total_operations'] * 100
            stats['gpu_percentage'] = stats['gpu_operations'] / stats['total_operations'] * 100
        else:
            stats['average_operation_time'] = 0
            stats['scaling_percentage'] = 0
            stats['rolling_percentage'] = 0
            stats['batch_percentage'] = 0
            stats['gpu_percentage'] = 0

        return stats

    def reset_stats(self):
        """Reset all statistics."""
        self.optimization_stats = {
            'total_operations': 0,
            'scaling_operations': 0,
            'rolling_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0
        }

        self.memory_optimizer.reset_stats()
        self.rolling_optimizer.reset_performance_stats()
        self.performance_profiler.reset_profile_data()

    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        if not self.enable_monitoring:
            yield
            return

        start_time = time.time()
        start_memory = self.memory_optimizer.get_memory_usage()['current_memory_gb']

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.memory_optimizer.get_memory_usage()['current_memory_gb']

            execution_time = end_time - start_time
            memory_used = end_memory - start_memory

            logger.info(f"Operation {operation_name}: {execution_time:.3f}s, "
                       f"Memory: {memory_used:.3f}GB")

# Global instance
_optimization_manager = None

def get_optimization_manager(**kwargs) -> VectorBTOptimizationManager:
    """Get global optimization manager instance."""
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = VectorBTOptimizationManager(**kwargs)
    return _optimization_manager

def optimize_vectorbt_operation(operation: Callable, data: Union[pd.Series, pd.DataFrame],
                              **kwargs) -> Any:
    """
    Optimize a VectorBT operation with full optimization pipeline.

    Args:
        operation: Operation function
        data: Input data
        **kwargs: Additional parameters

    Returns:
        Result of the operation
    """
    manager = get_optimization_manager()

    with manager.performance_monitoring(operation.__name__):
        # Optimize data
        if isinstance(data, pd.DataFrame):
            data = manager.optimize_dataframe(data)

        # Perform operation
        return operation(data, **kwargs)

def create_optimized_feature_pipeline(enable_gpu: bool = False,
                                    memory_efficient: bool = True) -> VectorBTOptimizationManager:
    """
    Create an optimized feature pipeline.

    Args:
        enable_gpu: Enable GPU acceleration
        memory_efficient: Enable memory optimization

    Returns:
        Optimization manager
    """
    return VectorBTOptimizationManager(
        enable_gpu=enable_gpu,
        enable_parallel=True,
        memory_efficient=memory_efficient,
        max_memory_gb=8.0,
        chunk_size=1000,
        enable_monitoring=True
    )

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': np.random.randn(10000).astype(np.float64),
        'volume': np.random.randint(1000, 10000, 10000).astype(np.int64),
        'high': np.random.randn(10000).astype(np.float64),
        'low': np.random.randn(10000).astype(np.float64)
    })

    print("Original data shape:", data.shape)
    print("Original memory usage:", data.memory_usage(deep=True).sum() / (1024**3), "GB")

    # Create optimization manager
    manager = get_optimization_manager(
        enable_gpu=False,
        memory_efficient=True,
        enable_monitoring=True
    )

    # Test scaling
    print("\nTesting scaling...")
    scaled_close = manager.scale_data(data['close'], method='adaptive')
    print(f"Scaled close shape: {scaled_close.shape}")

    # Test rolling operations
    print("\nTesting rolling operations...")
    rolling_mean = manager.rolling_operation(data['close'], 'mean', window=20)
    rolling_std = manager.rolling_operation(data['close'], 'std', window=20)
    print(f"Rolling mean shape: {rolling_mean.shape}")
    print(f"Rolling std shape: {rolling_std.shape}")

    # Test batch processing
    print("\nTesting batch processing...")
    feature_configs = [
        {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
        {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
        {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
        {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}},
        {'name': 'volume_scaled', 'type': 'scaling', 'params': {'method': 'minmax', 'column': 'volume'}}
    ]

    features = manager.batch_process_features(data, feature_configs)
    print(f"Generated features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")

    # Get optimization stats
    stats = manager.get_optimization_stats()
    print(f"\nOptimization stats: {stats}")

    # Test memory optimization
    print("\nTesting memory optimization...")
    optimized_data = manager.optimize_dataframe(data)
    print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum() / (1024**3):.3f}GB")

    print("\nOptimization pipeline test completed successfully!")
