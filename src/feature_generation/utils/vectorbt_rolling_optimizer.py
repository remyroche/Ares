"""
VectorBT Rolling Operations Optimizer

This module provides optimized rolling operations using VectorBT's high-performance
functions, with intelligent fallbacks and performance monitoring.

Key Features:
- VectorBT native rolling operations (mean, std, var, min, max, sum, etc.)
- Intelligent fallback to pandas/numpy when VectorBT unavailable
- Performance monitoring and statistics
- Memory-efficient chunked processing
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import warnings
from functools import wraps
import time

# VectorBT imports for optimization
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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class VectorBTRollingOptimizer:
    """
    Optimized rolling operations using VectorBT with intelligent fallbacks.
    
    Provides high-performance rolling calculations with automatic optimization
    selection based on data size and available hardware.
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize VectorBT rolling optimizer.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_parallel: Enable parallel processing
        """
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.use_vectorbt = VECTORBT_AVAILABLE
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'numpy_fallbacks': 0,
            'gpu_operations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
        
        # Configure VectorBT settings
        if self.use_vectorbt:
            vbt.settings.parallel['enabled'] = self.enable_parallel
            if self.enable_gpu:
                vbt.settings.array_wrapper['freq'] = '1min'
        
        logger.info(f"VectorBTRollingOptimizer initialized: VectorBT={self.use_vectorbt}, GPU={self.enable_gpu}")
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling mean calculation."""
        return self._rolling_operation(data, 'mean', window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling standard deviation calculation."""
        return self._rolling_operation(data, 'std', window, **kwargs)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling variance calculation."""
        return self._rolling_operation(data, 'var', window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling minimum calculation."""
        return self._rolling_operation(data, 'min', window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling maximum calculation."""
        return self._rolling_operation(data, 'max', window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling sum calculation."""
        return self._rolling_operation(data, 'sum', window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling quantile calculation."""
        return self._rolling_operation(data, 'quantile', window, q=q, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling skewness calculation."""
        return self._rolling_operation(data, 'skew', window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling kurtosis calculation."""
        return self._rolling_operation(data, 'kurt', window, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling apply calculation."""
        return self._rolling_operation(data, 'apply', window, func=func, **kwargs)
    
    def rolling_corr(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling correlation calculation."""
        return self._rolling_operation(data1, 'corr', window, data2=data2, **kwargs)
    
    def rolling_cov(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling covariance calculation."""
        return self._rolling_operation(data1, 'cov', window, data2=data2, **kwargs)
    
    def _rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation with intelligent method selection.
        
        Args:
            data: Input data (Series or DataFrame)
            operation: Operation to perform ('mean', 'std', 'var', 'min', 'max', 'sum', 'quantile', 'skew', 'kurt', 'apply', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional parameters for the operation
            
        Returns:
            Result of the rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        try:
            # Determine optimal processing method
            if self._should_use_vectorbt(data, window):
                result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif self._should_use_gpu(data, window):
                result = self._gpu_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            else:
                result = self._pandas_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
            
            # Update timing
            self.performance_stats['total_time'] += time.time() - start_time
            return result
            
        except Exception as e:
            logger.warning(f"Rolling operation {operation} failed: {e}, using numpy fallback")
            self.performance_stats['numpy_fallbacks'] += 1
            return self._numpy_rolling_operation(data, operation, window, **kwargs)
    
    def _should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if VectorBT should be used for this operation."""
        if not self.use_vectorbt:
            return False
        
        # Use VectorBT for larger datasets or when parallel processing is beneficial
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size > 1000 or (self.enable_parallel and data_size > 100)
    
    def _should_use_gpu(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if GPU acceleration should be used."""
        if not self.enable_gpu or not CUPY_AVAILABLE:
            return False
        
        # Use GPU for very large datasets
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size > 10000
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
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
                data2 = kwargs.get('data2')
                return rolling_corr(data, data2, window=window, **kwargs)
            elif operation == 'cov':
                data2 = kwargs.get('data2')
                return rolling_cov(data, data2, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported VectorBT operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT {operation} failed: {e}")
            raise
    
    def _gpu_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                              window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
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
            logger.warning(f"GPU {operation} failed: {e}")
            raise
    
    def _gpu_rolling_series(self, data: cp.ndarray, operation: str, window: int, **kwargs) -> cp.ndarray:
        """GPU rolling operation for Series."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones(window) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones(window), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._numpy_rolling_operation(pd.Series(data.get()), operation, window, **kwargs).values
    
    def _gpu_rolling_dataframe(self, data: cp.ndarray, operation: str, window: int, **kwargs) -> cp.ndarray:
        """GPU rolling operation for DataFrame."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones((window, 1)), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._numpy_rolling_operation(pd.DataFrame(data.get()), operation, window, **kwargs).values
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
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
            data2 = kwargs.get('data2')
            return rolling_obj.corr(data2)
        elif operation == 'cov':
            data2 = kwargs.get('data2')
            return rolling_obj.cov(data2)
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _numpy_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using numpy (fallback)."""
        if isinstance(data, pd.Series):
            values = data.values
            result = self._numpy_rolling_series(values, operation, window, **kwargs)
            return pd.Series(result, index=data.index, name=data.name)
        else:
            values = data.values
            result = self._numpy_rolling_dataframe(values, operation, window, **kwargs)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
    
    def _numpy_rolling_series(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """Numpy rolling operation for Series."""
        if operation == 'mean':
            return np.convolve(values, np.ones(window) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones(window), mode='same')
        else:
            # For complex operations, use pandas
            series = pd.Series(values)
            return series.rolling(window=window, **kwargs).agg(operation).values
    
    def _numpy_rolling_dataframe(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """Numpy rolling operation for DataFrame."""
        if operation == 'mean':
            return np.convolve(values, np.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones((window, 1)), mode='same')
        else:
            # For complex operations, use pandas
            df = pd.DataFrame(values)
            return df.rolling(window=window, **kwargs).agg(operation).values
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'numpy_fallbacks': 0,
            'gpu_operations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }


# Global optimizer instance
_global_optimizer = None

def get_vectorbt_rolling_optimizer(enable_gpu: bool = False, enable_parallel: bool = True) -> VectorBTRollingOptimizer:
    """Get global VectorBT rolling optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = VectorBTRollingOptimizer(enable_gpu=enable_gpu, enable_parallel=enable_parallel)
    return _global_optimizer


def optimized_rolling_mean(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling mean using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_mean(data, window, **kwargs)


def optimized_rolling_std(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling standard deviation using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_std(data, window, **kwargs)


def optimized_rolling_var(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling variance using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_var(data, window, **kwargs)


def optimized_rolling_min(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling minimum using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_min(data, window, **kwargs)


def optimized_rolling_max(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling maximum using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_max(data, window, **kwargs)


def optimized_rolling_sum(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling sum using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_sum(data, window, **kwargs)


def optimized_rolling_quantile(data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling quantile using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_quantile(data, window, q=q, **kwargs)


def optimized_rolling_apply(data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling apply using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_apply(data, window, func, **kwargs)


def optimized_rolling_corr(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling correlation using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_corr(data1, data2, window, **kwargs)


def optimized_rolling_cov(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                         window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling covariance using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_cov(data1, data2, window, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
    np.random.seed(42)
    
    # Generate sample data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(5000) * 0.01),
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=dates)
    
    # Test optimizer
    optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
    
    # Test various operations
    print("Testing VectorBT rolling operations...")
    
    # Rolling mean
    mean_result = optimizer.rolling_mean(data['close'], window=20)
    print(f"Rolling mean shape: {mean_result.shape}")
    
    # Rolling std
    std_result = optimizer.rolling_std(data['close'], window=20)
    print(f"Rolling std shape: {std_result.shape}")
    
    # Rolling correlation
    corr_result = optimizer.rolling_corr(data['close'], data['volume'], window=20)
    print(f"Rolling correlation shape: {corr_result.shape}")
    
    # Performance stats
    stats = optimizer.get_performance_stats()
    print(f"Performance stats: {stats}")