"""
Unified Vectorization Manager

This module provides a centralized manager for VectorBT optimization across all feature generators.
It coordinates VectorBTRollingOptimizer usage and provides unified optimization strategies.

Key Features:
- Centralized VectorBT optimization management
- Intelligent resource allocation
- Performance monitoring and optimization
- Batch processing coordination
- Memory management
- GPU acceleration coordination
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
from functools import wraps
import threading
from contextlib import contextmanager

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
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
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class UnifiedVectorizationManager:
    """
    Centralized manager for VectorBT optimization across all feature generators.
    
    This manager coordinates VectorBT usage, optimizes resource allocation,
    and provides unified optimization strategies for maximum performance.
    """
    
    def __init__(self, 
                 enable_gpu: bool = False, 
                 enable_parallel: bool = True,
                 memory_efficient: bool = True,
                 max_workers: int = 4,
                 cache_size_mb: int = 1000):
        """
        Initialize Unified Vectorization Manager.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            max_workers: Maximum number of worker threads
            cache_size_mb: Cache size in MB
        """
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.memory_efficient = memory_efficient
        self.max_workers = max_workers
        self.cache_size_mb = cache_size_mb
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'memory_optimizations': 0,
            'parallel_operations': 0
        }
        
        # Resource management
        self.active_operations = 0
        self.memory_usage_mb = 0
        self.operation_lock = threading.Lock()
        
        # Cache management
        self.cache = {}
        self.cache_access_times = {}
        self.cache_size_bytes = 0
        
        # Configure VectorBT settings
        if VECTORBT_AVAILABLE:
            vbt.settings.parallel['enabled'] = self.enable_parallel
            vbt.settings.parallel['num_threads'] = self.max_workers
            if self.enable_gpu:
                vbt.settings.array_wrapper['freq'] = '1min'
        
        logger.info(f"UnifiedVectorizationManager initialized: VectorBT={VECTORBT_AVAILABLE}, GPU={self.enable_gpu}, Parallel={self.enable_parallel}")
    
    def should_use_vectorbt(self, data_size: int, operation_complexity: str = 'medium') -> bool:
        """
        Determine if VectorBT should be used based on data size and operation complexity.
        
        Args:
            data_size: Size of the data
            operation_complexity: Complexity level ('low', 'medium', 'high')
            
        Returns:
            True if VectorBT should be used
        """
        if not VECTORBT_AVAILABLE:
            return False
        
        # Thresholds based on operation complexity
        thresholds = {
            'low': 100,
            'medium': 500,
            'high': 1000
        }
        
        threshold = thresholds.get(operation_complexity, 500)
        return data_size >= threshold
    
    def should_use_gpu(self, data_size: int, operation_complexity: str = 'medium') -> bool:
        """
        Determine if GPU acceleration should be used.
        
        Args:
            data_size: Size of the data
            operation_complexity: Complexity level ('low', 'medium', 'high')
            
        Returns:
            True if GPU should be used
        """
        if not self.enable_gpu or not CUPY_AVAILABLE:
            return False
        
        # GPU thresholds are higher due to memory transfer overhead
        thresholds = {
            'low': 5000,
            'medium': 10000,
            'high': 20000
        }
        
        threshold = thresholds.get(operation_complexity, 10000)
        return data_size >= threshold
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for VectorBT processing.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        if not self.memory_efficient:
            return data
        
        optimized_data = data.copy()
        
        # Optimize data types
        for column in optimized_data.columns:
            if optimized_data[column].dtype == 'float64':
                if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                    optimized_data[column].max() <= np.finfo(np.float32).max):
                    optimized_data[column] = optimized_data[column].astype(np.float32)
                    self.performance_stats['memory_optimizations'] += 1
        
        # Ensure index is datetime for time series operations
        if not isinstance(optimized_data.index, pd.DatetimeIndex):
            try:
                optimized_data.index = pd.to_datetime(optimized_data.index)
            except (ValueError, TypeError):
                pass
        
        return optimized_data
    
    def rolling_operation(self, 
                         data: Union[pd.Series, pd.DataFrame], 
                         operation: str, 
                         window: int, 
                         **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation using VectorBT.
        
        Args:
            data: Input data
            operation: Operation to perform
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of the rolling operation
        """
        start_time = time.time()
        
        with self.operation_lock:
            self.active_operations += 1
            self.performance_stats['total_operations'] += 1
        
        try:
            # Optimize data
            if isinstance(data, pd.DataFrame):
                data = self.optimize_dataframe(data)
            
            # Determine optimal processing method
            data_size = len(data)
            operation_complexity = kwargs.get('complexity', 'medium')
            
            if self.should_use_gpu(data_size, operation_complexity):
                result = self._gpu_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            elif self.should_use_vectorbt(data_size, operation_complexity):
                result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            else:
                result = self._pandas_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
            
            # Update timing
            self.performance_stats['total_time'] += time.time() - start_time
            
            return result
            
        finally:
            with self.operation_lock:
                self.active_operations -= 1
    
    def batch_operations(self, 
                        data: pd.DataFrame, 
                        operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform multiple operations in batch for efficiency.
        
        Args:
            data: Input DataFrame
            operations: List of operation specifications
            
        Returns:
            DataFrame with results of all operations
        """
        start_time = time.time()
        
        with self.operation_lock:
            self.active_operations += 1
            self.performance_stats['batch_operations'] += 1
        
        try:
            # Optimize data
            data = self.optimize_dataframe(data)
            
            results = {}
            
            # Group operations by type for batch processing
            rolling_ops = [op for op in operations if op.get('type') == 'rolling']
            indicator_ops = [op for op in operations if op.get('type') == 'indicator']
            
            # Process rolling operations in batch
            if rolling_ops:
                rolling_results = self._batch_rolling_operations(data, rolling_ops)
                results.update(rolling_results)
            
            # Process indicator operations in batch
            if indicator_ops:
                indicator_results = self._batch_indicator_operations(data, indicator_ops)
                results.update(indicator_results)
            
            # Update timing
            self.performance_stats['total_time'] += time.time() - start_time
            
            return pd.DataFrame(results, index=data.index)
            
        finally:
            with self.operation_lock:
                self.active_operations -= 1
    
    def _vectorbt_rolling_operation(self, 
                                   data: Union[pd.Series, pd.DataFrame], 
                                   operation: str, 
                                   window: int, 
                                   **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT."""
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
            elif operation == 'quantile':
                q = kwargs.get('q', 0.5)
                return rolling_quantile(data, window=window, q=q, **kwargs)
            elif operation == 'skew':
                return rolling_skew(data, window=window, **kwargs)
            elif operation == 'kurt':
                return rolling_kurt(data, window=window, **kwargs)
            elif operation == 'apply':
                func = kwargs.get('func')
                return rolling_apply(data, window=window, func=func, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                return rolling_corr(data, other, window=window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                return rolling_cov(data, other, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported VectorBT operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation {operation} failed: {e}")
            raise
    
    def _gpu_rolling_operation(self, 
                              data: Union[pd.Series, pd.DataFrame], 
                              operation: str, 
                              window: int, 
                              **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using GPU acceleration."""
        try:
            # Convert to CuPy arrays
            if isinstance(data, pd.Series):
                gpu_data = cp.asarray(data.values)
                result = self._gpu_rolling_series(gpu_data, operation, window, **kwargs)
                return pd.Series(result, index=data.index, name=data.name)
            else:
                gpu_data = cp.asarray(data.values)
                result = self._gpu_rolling_dataframe(gpu_data, operation, window, **kwargs)
                return pd.DataFrame(result, index=data.index, columns=data.columns)
        except Exception as e:
            logger.warning(f"GPU operation {operation} failed: {e}")
            raise
    
    def _gpu_rolling_series(self, data: cp.ndarray, operation: str, window: int, **kwargs) -> cp.ndarray:
        """GPU rolling operation for Series."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones(window) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones(window), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._pandas_rolling_operation(pd.Series(data.get()), operation, window, **kwargs).values
    
    def _gpu_rolling_dataframe(self, data: cp.ndarray, operation: str, window: int, **kwargs) -> cp.ndarray:
        """GPU rolling operation for DataFrame."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones((window, 1)), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._pandas_rolling_operation(pd.DataFrame(data.get()), operation, window, **kwargs).values
    
    def _pandas_rolling_operation(self, 
                                 data: Union[pd.Series, pd.DataFrame], 
                                 operation: str, 
                                 window: int, 
                                 **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using pandas."""
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
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _batch_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process multiple rolling operations in batch."""
        results = {}
        
        for op in operations:
            op_name = op.get('name')
            op_params = op.get('params', {})
            column = op_params.get('column', 'close')
            operation = op_params.get('operation')
            window = op_params.get('window')
            
            if column in data.columns:
                results[op_name] = self.rolling_operation(
                    data[column], operation, window, **op_params
                )
        
        return results
    
    def _batch_indicator_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process multiple indicator operations in batch."""
        results = {}
        
        for op in operations:
            op_name = op.get('name')
            op_params = op.get('params', {})
            indicator = op_params.get('indicator')
            
            # Implement indicator calculations here
            # This would be expanded based on specific indicator requirements
            pass
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['avg_time_per_operation'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['gpu_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'memory_optimizations': 0,
            'parallel_operations': 0
        }
    
    @contextmanager
    def batch_context(self):
        """Context manager for batch operations."""
        try:
            yield self
        finally:
            pass


# Global manager instance
_global_manager = None

def get_unified_vectorization_manager(enable_gpu: bool = False, 
                                    enable_parallel: bool = True,
                                    memory_efficient: bool = True) -> UnifiedVectorizationManager:
    """Get global Unified Vectorization Manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedVectorizationManager(
            enable_gpu=enable_gpu,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient
        )
    return _global_manager


def optimized_rolling_operation(data: Union[pd.Series, pd.DataFrame], 
                              operation: str, 
                              window: int, 
                              **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Perform optimized rolling operation using Unified Vectorization Manager."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_operation(data, operation, window, **kwargs)


def optimized_batch_operations(data: pd.DataFrame, 
                              operations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Perform optimized batch operations using Unified Vectorization Manager."""
    manager = get_unified_vectorization_manager()
    return manager.batch_operations(data, operations)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=10000, freq='1min')
    np.random.seed(42)
    
    # Generate sample data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(10000) * 0.01),
        'volume': np.random.lognormal(10, 1, 10000)
    }, index=dates)
    
    # Test manager
    manager = get_unified_vectorization_manager(enable_gpu=False, enable_parallel=True)
    
    # Test various operations
    print("Testing Unified Vectorization Manager...")
    
    # Rolling mean
    mean_result = manager.rolling_operation(data['close'], 'mean', window=20)
    print(f"Rolling mean shape: {mean_result.shape}")
    
    # Rolling std
    std_result = manager.rolling_operation(data['close'], 'std', window=20)
    print(f"Rolling std shape: {std_result.shape}")
    
    # Batch operations
    operations = [
        {'type': 'rolling', 'name': 'sma_20', 'params': {'column': 'close', 'operation': 'mean', 'window': 20}},
        {'type': 'rolling', 'name': 'std_20', 'params': {'column': 'close', 'operation': 'std', 'window': 20}},
        {'type': 'rolling', 'name': 'volume_sma_10', 'params': {'column': 'volume', 'operation': 'mean', 'window': 10}}
    ]
    
    batch_result = manager.batch_operations(data, operations)
    print(f"Batch operations result shape: {batch_result.shape}")
    
    # Performance stats
    stats = manager.get_performance_stats()
    print(f"Performance stats: {stats}")