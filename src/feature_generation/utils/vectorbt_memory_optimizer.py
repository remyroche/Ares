"""
VectorBT Memory and Performance Optimizer

This module provides comprehensive memory and performance optimization utilities
for VectorBT operations, including memory management, GPU utilization, and
performance monitoring.
"""

import numpy as np
import pandas as pd
import logging
import time
import gc
import psutil
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from contextlib import contextmanager
import warnings
from functools import wraps

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# GPU acceleration removed - CuPy not supported on all platforms
cp = None
CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorBTMemoryOptimizer:
    """
    Comprehensive memory and performance optimizer for VectorBT operations.

    This class provides:
    - Memory usage monitoring and optimization
    - GPU memory management
    - Data type optimization
    - Chunked processing for large datasets
    - Performance monitoring and profiling
    """

    def __init__(self, max_memory_gb: float = 8.0, enable_gpu: bool = False,
                 chunk_size: int = 1000, enable_monitoring: bool = True):
        """
        Initialize VectorBT memory optimizer.

        Args:
            max_memory_gb: Maximum memory usage in GB
            enable_gpu: Enable
            chunk_size: Default chunk size for processing
            enable_monitoring: Enable performance monitoring
        """
        self.max_memory_gb = max_memory_gb
        self.enable_gpu = False  # GPU support removed
        self.chunk_size = chunk_size
        self.enable_monitoring = enable_monitoring

        # Memory tracking
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'optimizations_applied': 0,
            'chunks_processed': 0,
            'memory_savings': 0.0
        }

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'chunked_operations': 0,
            'total_time': 0.0,
            'average_operation_time': 0.0,
            'memory_efficiency': 0.0
        }

        # Initialize memory monitoring
        if self.enable_monitoring:
            self._start_memory_monitoring()

    def _start_memory_monitoring(self):
        """Start memory monitoring background process."""
        self._initial_memory = psutil.Process().memory_info().rss / (1024**3)
        self.memory_stats['current_memory_usage'] = self._initial_memory

    def optimize_dataframe(self, data: pd.DataFrame,
                          target_dtype: str = 'auto') -> pd.DataFrame:
        """
        Optimize DataFrame for memory efficiency.

        Args:
            data: Input DataFrame
            target_dtype: Target data type ('auto', 'float32', 'float64', 'int32', 'int64')

        Returns:
            Optimized DataFrame
        """
        if data.empty:
            return data

        optimized_data = data.copy()
        original_memory = data.memory_usage(deep=True).sum() / (1024**3)

        for column in optimized_data.columns:
            if optimized_data[column].dtype == 'float64':
                if target_dtype == 'auto':
                    # Auto-optimize based on data range
                    if self._can_convert_to_float32(optimized_data[column]):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.memory_stats['optimizations_applied'] += 1
                elif target_dtype == 'float32':
                    if self._can_convert_to_float32(optimized_data[column]):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.memory_stats['optimizations_applied'] += 1

            elif optimized_data[column].dtype == 'int64':
                if target_dtype == 'auto':
                    if self._can_convert_to_int32(optimized_data[column]):
                        optimized_data[column] = optimized_data[column].astype(np.int32)
                        self.memory_stats['optimizations_applied'] += 1
                elif target_dtype == 'int32':
                    if self._can_convert_to_int32(optimized_data[column]):
                        optimized_data[column] = optimized_data[column].astype(np.int32)
                        self.memory_stats['optimizations_applied'] += 1

        # Calculate memory savings
        optimized_memory = optimized_data.memory_usage(deep=True).sum() / (1024**3)
        memory_savings = original_memory - optimized_memory
        self.memory_stats['memory_savings'] += memory_savings

        logger.info(f"DataFrame optimized: {original_memory:.3f}GB -> {optimized_memory:.3f}GB "
                   f"(saved {memory_savings:.3f}GB)")

        return optimized_data

    def _can_convert_to_float32(self, series: pd.Series) -> bool:
        """Check if series can be safely converted to float32."""
        if series.isna().all():
            return True

        min_val = series.min()
        max_val = series.max()

        return (min_val >= np.finfo(np.float32).min and
                max_val <= np.finfo(np.float32).max)

    def _can_convert_to_int32(self, series: pd.Series) -> bool:
        """Check if series can be safely converted to int32."""
        if series.isna().all():
            return True

        min_val = series.min()
        max_val = series.max()

        return (min_val >= np.iinfo(np.int32).min and
                max_val <= np.iinfo(np.int32).max)

    def enable_gpu_processing(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Enable GPU processing for data.

        Args:
            data: Input data

        Returns:
            GPU-accelerated data
        """
        if not self.enable_gpu or True:
            return data

        try:
            if isinstance(data, pd.Series):
                gpu_data = np.asarray(data.values)
                return pd.Series(gpu_data, index=data.index)
            elif isinstance(data, pd.DataFrame):
                gpu_data = {}
                for column in data.columns:
                    gpu_data[column] = np.asarray(data[column].values)
                return pd.DataFrame(gpu_data, index=data.index)
        except Exception as e:
            logger.warning(f"GPU processing failed: {e}")
            return data

    def process_in_chunks(self, data: Union[pd.Series, pd.DataFrame],
                         operation: Callable, chunk_size: Optional[int] = None,
                         **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Process data in chunks for memory efficiency.

        Args:
            data: Input data
            operation: Operation function to apply
            chunk_size: Size of chunks (uses default if None)
            **kwargs: Additional arguments for operation

        Returns:
            Processed data
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        if len(data) <= chunk_size:
            return operation(data, **kwargs)

        results = []
        total_chunks = (len(data) + chunk_size - 1) // chunk_size

        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]

            # Process chunk
            chunk_result = operation(chunk, **kwargs)
            results.append(chunk_result)

            # Update stats
            self.memory_stats['chunks_processed'] += 1
            self.performance_stats['chunked_operations'] += 1

            # Force garbage collection every few chunks
            if i % (chunk_size * 5) == 0:
                gc.collect()

        # Combine results
        if isinstance(data, pd.Series):
            return pd.concat(results, ignore_index=False)
        else:
            return pd.concat(results, ignore_index=False)

    @contextmanager
    def memory_managed_operation(self, estimated_memory_gb: float,
                               operation_name: str = "operation"):
        """
        Context manager for memory-managed operations.

        Args:
            estimated_memory_gb: Estimated memory usage in GB
            operation_name: Name of the operation for logging
        """
        start_memory = psutil.Process().memory_info().rss / (1024**3)
        start_time = time.time()

        try:
            # Check if we have enough memory
            if estimated_memory_gb > self.max_memory_gb:
                logger.warning(f"Operation {operation_name} may exceed memory limit: "
                             f"{estimated_memory_gb:.2f}GB > {self.max_memory_gb}GB")

            yield

        finally:
            # Clean up and monitor
            gc.collect()

            end_memory = psutil.Process().memory_info().rss / (1024**3)
            end_time = time.time()

            memory_used = end_memory - start_memory
            operation_time = end_time - start_time

            # Update stats
            self.memory_stats['current_memory_usage'] = end_memory
            if end_memory > self.memory_stats['peak_memory_usage']:
                self.memory_stats['peak_memory_usage'] = end_memory

            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_time'] += operation_time
            self.performance_stats['average_operation_time'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_operations']
            )

            logger.debug(f"Operation {operation_name}: {operation_time:.3f}s, "
                        f"Memory: {memory_used:.3f}GB")

    def monitor_performance(self, operation: Callable) -> Callable:
        """
        Decorator to monitor performance of operations.

        Args:
            operation: Operation function to monitor

        Returns:
            Wrapped function with performance monitoring
        """
        @wraps(operation)
        def wrapper(*args, **kwargs):
            if not self.enable_monitoring:
                return operation(*args, **kwargs)

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**3)

            try:
                result = operation(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**3)

                # Update performance stats
                operation_time = end_time - start_time
                memory_used = end_memory - start_memory

                self.performance_stats['total_operations'] += 1
                self.performance_stats['total_time'] += operation_time
                self.performance_stats['average_operation_time'] = (
                    self.performance_stats['total_time'] / self.performance_stats['total_operations']
                )

                # Update memory stats
                if end_memory > self.memory_stats['peak_memory_usage']:
                    self.memory_stats['peak_memory_usage'] = end_memory

                self.memory_stats['current_memory_usage'] = end_memory

                return result

            except Exception as e:
                logger.error(f"Operation failed: {e}")
                raise

        return wrapper

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        current_memory = psutil.Process().memory_info().rss / (1024**3)

        return {
            'current_memory_gb': current_memory,
            'peak_memory_gb': self.memory_stats['peak_memory_usage'],
            'memory_savings_gb': self.memory_stats['memory_savings'],
            'optimizations_applied': self.memory_stats['optimizations_applied'],
            'chunks_processed': self.memory_stats['chunks_processed']
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['total_operations'] * 100
            )
            stats['gpu_usage_percentage'] = (
                stats['gpu_operations'] / stats['total_operations'] * 100
            )
            stats['chunked_usage_percentage'] = (
                stats['chunked_operations'] / stats['total_operations'] * 100
            )

        return stats

    def reset_stats(self):
        """Reset all statistics."""
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'optimizations_applied': 0,
            'chunks_processed': 0,
            'memory_savings': 0.0
        }

        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'chunked_operations': 0,
            'total_time': 0.0,
            'average_operation_time': 0.0,
            'memory_efficiency': 0.0
        }

    def optimize_vectorbt_settings(self):
        """Optimize VectorBT global settings for performance."""
        if not VECTORBT_AVAILABLE:
            return

        # Configure VectorBT for optimal performance using newer API
        # Check if array_wrapper structure exists and set wrapper if available
        if hasattr(vbt.settings, 'array_wrapper') and 'wrapper' in vbt.settings['array_wrapper']:
            vbt.settings['array_wrapper']['wrapper'] = 'pandas'
        vbt.settings['caching']['enabled'] = True

        if self.enable_gpu:
            try:
                # Check if GPU settings are available in this VectorBT version
                if hasattr(vbt.settings, 'gpu') and 'enabled' in vbt.settings['gpu']:
                    vbt.settings['gpu']['enabled'] = True
                    logger.info("VectorBT GPU acceleration enabled")
                else:
                    logger.warning("VectorBT GPU acceleration not available in this version")
            except Exception as e:
                logger.warning(f"VectorBT GPU acceleration not available: {e}")

        # Configure parallel processing
        try:
            # Check if parallel settings are available in this VectorBT version
            if hasattr(vbt.settings, 'parallel') and 'enabled' in vbt.settings['parallel']:
                vbt.settings['parallel']['enabled'] = True
                logger.info("VectorBT parallel processing enabled")
        except Exception as e:
            logger.warning(f"VectorBT parallel processing not available: {e}")

class VectorBTPerformanceProfiler:
    """
    Performance profiler for VectorBT operations.

    This class provides detailed performance profiling and analysis
    for VectorBT operations.
    """

    def __init__(self, enable_detailed_profiling: bool = False):
        """
        Initialize performance profiler.

        Args:
            enable_detailed_profiling: Enable detailed profiling
        """
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profile_data = []

    def profile_operation(self, operation_name: str, operation: Callable,
                         *args, **kwargs) -> Any:
        """
        Profile a single operation.

        Args:
            operation_name: Name of the operation
            operation: Operation function
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of the operation
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**3)

        try:
            result = operation(*args, **kwargs)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**3)

            # Record profile data
            profile_entry = {
                'operation_name': operation_name,
                'execution_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'success': True,
                'timestamp': time.time()
            }

            self.profile_data.append(profile_entry)

            return result

        except Exception as e:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**3)

            # Record failed operation
            profile_entry = {
                'operation_name': operation_name,
                'execution_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

            self.profile_data.append(profile_entry)
            raise

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of profiling data."""
        if not self.profile_data:
            return {'total_operations': 0}

        successful_ops = [op for op in self.profile_data if op['success']]
        failed_ops = [op for op in self.profile_data if not op['success']]

        if successful_ops:
            avg_time = np.mean([op['execution_time'] for op in successful_ops])
            avg_memory = np.mean([op['memory_used'] for op in successful_ops])
            total_time = sum([op['execution_time'] for op in successful_ops])
            total_memory = sum([op['memory_used'] for op in successful_ops])
        else:
            avg_time = 0
            avg_memory = 0
            total_time = 0
            total_memory = 0

        return {
            'total_operations': len(self.profile_data),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'average_execution_time': avg_time,
            'average_memory_usage': avg_memory,
            'total_execution_time': total_time,
            'total_memory_usage': total_memory,
            'success_rate': len(successful_ops) / len(self.profile_data) * 100
        }

    def get_operation_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get breakdown by operation type."""
        if not self.profile_data:
            return {}

        operation_stats = {}

        for op in self.profile_data:
            name = op['operation_name']
            if name not in operation_stats:
                operation_stats[name] = {
                    'count': 0,
                    'total_time': 0,
                    'total_memory': 0,
                    'successes': 0,
                    'failures': 0
                }

            operation_stats[name]['count'] += 1
            operation_stats[name]['total_time'] += op['execution_time']
            operation_stats[name]['total_memory'] += op['memory_used']

            if op['success']:
                operation_stats[name]['successes'] += 1
            else:
                operation_stats[name]['failures'] += 1

        # Calculate averages
        for name, stats in operation_stats.items():
            stats['average_time'] = stats['total_time'] / stats['count']
            stats['average_memory'] = stats['total_memory'] / stats['count']
            stats['success_rate'] = stats['successes'] / stats['count'] * 100

        return operation_stats

    def reset_profile_data(self):
        """Reset profiling data."""
        self.profile_data = []

# Global instances
_memory_optimizer = None
_performance_profiler = None

def get_memory_optimizer(**kwargs) -> VectorBTMemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = VectorBTMemoryOptimizer(**kwargs)
    return _memory_optimizer

def get_performance_profiler(**kwargs) -> VectorBTPerformanceProfiler:
    """Get global performance profiler instance."""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = VectorBTPerformanceProfiler(**kwargs)
    return _performance_profiler

def optimize_dataframe_memory(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    optimizer = get_memory_optimizer()
    return optimizer.optimize_dataframe(data, **kwargs)

def process_with_memory_management(operation: Callable, data: Union[pd.Series, pd.DataFrame],
                                 **kwargs) -> Any:
    """Process data with memory management."""
    optimizer = get_memory_optimizer()

    with optimizer.memory_managed_operation(
        estimated_memory_gb=data.memory_usage(deep=True).sum() / (1024**3),
        operation_name=operation.__name__
    ):
        return operation(data, **kwargs)

def profile_operation(operation_name: str, operation: Callable, *args, **kwargs) -> Any:
    """Profile an operation."""
    profiler = get_performance_profiler()
    return profiler.profile_operation(operation_name, operation, *args, **kwargs)

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

    print("Original data memory usage:")
    print(f"  Memory: {data.memory_usage(deep=True).sum() / (1024**3):.3f}GB")
    print(f"  Dtypes: {data.dtypes.to_dict()}")

    # Optimize memory
    optimizer = get_memory_optimizer(max_memory_gb=4.0, enable_monitoring=True)
    optimized_data = optimizer.optimize_dataframe(data)

    print("\nOptimized data memory usage:")
    print(f"  Memory: {optimized_data.memory_usage(deep=True).sum() / (1024**3):.3f}GB")
    print(f"  Dtypes: {optimized_data.dtypes.to_dict()}")

    # Get memory stats
    memory_stats = optimizer.get_memory_usage()
    print(f"\nMemory stats: {memory_stats}")

    # Test chunked processing
    def test_operation(df):
        return df.rolling(window=20).mean()

    result = optimizer.process_in_chunks(optimized_data, test_operation, chunk_size=1000)
    print(f"\nChunked processing result shape: {result.shape}")

    # Get performance stats
    perf_stats = optimizer.get_performance_stats()
    print(f"\nPerformance stats: {perf_stats}")
