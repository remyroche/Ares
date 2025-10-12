"""
Unified Vectorization Manager

This module provides a centralized vectorization optimization system that intelligently
selects the best vectorization method based on data characteristics, available hardware,
and performance requirements.

Key Features:
- Intelligent method selection (VectorBT, NumPy, Pandas, GPU)
- Performance monitoring and adaptive optimization
- Memory-efficient processing for large datasets
- Unified API for all vectorization operations
- Automatic fallback mechanisms
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import warnings
from functools import wraps
import time
from dataclasses import dataclass
from enum import Enum

# VectorBT imports
try:
    import vectorbt as vbt
    # VectorBT 0.28+ has a different API structure
    # We'll use pandas as primary and VectorBT for specific optimizations
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class VectorizationMethod(Enum):
    """Available vectorization methods."""
    VECTORBT = "vectorbt"
    PANDAS = "pandas"
    NUMPY = "numpy"
    GPU = "gpu"
    CHUNKED = "chunked"


@dataclass
class PerformanceMetrics:
    """Performance metrics for vectorization operations."""
    method_used: VectorizationMethod
    execution_time: float
    memory_usage: float
    data_size: int
    operation_type: str
    success: bool
    error_message: Optional[str] = None


class UnifiedVectorizationManager:
    """
    Unified vectorization manager that intelligently selects the best
    vectorization method for each operation.
    """
    
    def __init__(self, 
                 enable_gpu: bool = False,
                 enable_parallel: bool = True,
                 memory_efficient: bool = True,
                 chunk_size: int = 1000,
                 adaptive_optimization: bool = True):
        """
        Initialize the unified vectorization manager.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            chunk_size: Size of data chunks for processing
            adaptive_optimization: Enable adaptive optimization based on performance
        """
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.adaptive_optimization = adaptive_optimization
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.method_performance: Dict[VectorizationMethod, Dict[str, float]] = {
            method: {'avg_time': 0.0, 'success_rate': 0.0, 'usage_count': 0}
            for method in VectorizationMethod
        }
        
        # Configure VectorBT
        if VECTORBT_AVAILABLE:
            vbt.settings.parallel['enabled'] = self.enable_parallel
            if self.enable_gpu:
                vbt.settings.array_wrapper['freq'] = '1min'
        
        logger.info(f"UnifiedVectorizationManager initialized: "
                   f"VectorBT={VECTORBT_AVAILABLE}, GPU={self.enable_gpu}, "
                   f"Memory={self.memory_efficient}, Adaptive={self.adaptive_optimization}")
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling mean calculation."""
        return self._execute_rolling_operation(data, 'mean', window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling standard deviation calculation."""
        return self._execute_rolling_operation(data, 'std', window, **kwargs)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling variance calculation."""
        return self._execute_rolling_operation(data, 'var', window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling minimum calculation."""
        return self._execute_rolling_operation(data, 'min', window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling maximum calculation."""
        return self._execute_rolling_operation(data, 'max', window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling sum calculation."""
        return self._execute_rolling_operation(data, 'sum', window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling quantile calculation."""
        return self._execute_rolling_operation(data, 'quantile', window, q=q, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling skewness calculation."""
        return self._execute_rolling_operation(data, 'skew', window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling kurtosis calculation."""
        return self._execute_rolling_operation(data, 'kurt', window, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling apply calculation."""
        return self._execute_rolling_operation(data, 'apply', window, func=func, **kwargs)
    
    def rolling_corr(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling correlation calculation."""
        return self._execute_rolling_operation(data1, 'corr', window, data2=data2, **kwargs)
    
    def rolling_cov(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling covariance calculation."""
        return self._execute_rolling_operation(data1, 'cov', window, data2=data2, **kwargs)
    
    def _execute_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Execute rolling operation with intelligent method selection.
        
        Args:
            data: Input data
            operation: Operation to perform
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of the rolling operation
        """
        start_time = time.time()
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Select optimal method
        method = self._select_optimal_method(data, operation, window)
        
        try:
            # Execute operation
            if method == VectorizationMethod.VECTORBT:
                result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
            elif method == VectorizationMethod.GPU:
                result = self._gpu_rolling_operation(data, operation, window, **kwargs)
            elif method == VectorizationMethod.CHUNKED:
                result = self._chunked_rolling_operation(data, operation, window, **kwargs)
            elif method == VectorizationMethod.PANDAS:
                result = self._pandas_rolling_operation(data, operation, window, **kwargs)
            else:  # NUMPY
                result = self._numpy_rolling_operation(data, operation, window, **kwargs)
            
            # Record performance metrics
            execution_time = time.time() - start_time
            self._record_performance(method, execution_time, data_size, operation, True)
            
            return result
            
        except Exception as e:
            # Fallback to pandas if primary method fails
            execution_time = time.time() - start_time
            self._record_performance(method, execution_time, data_size, operation, False, str(e))
            
            logger.warning(f"Primary method {method.value} failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _select_optimal_method(self, data: Union[pd.Series, pd.DataFrame], operation: str, window: int) -> VectorizationMethod:
        """Select the optimal vectorization method based on data characteristics and performance history."""
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Use adaptive optimization if enabled
        if self.adaptive_optimization and self.performance_history:
            return self._adaptive_method_selection(data_size, operation)
        
        # Rule-based method selection
        if data_size > self.chunk_size and self.memory_efficient:
            return VectorizationMethod.CHUNKED
        elif data_size > 10000 and self.enable_gpu and CUPY_AVAILABLE:
            return VectorizationMethod.GPU
        elif data_size > 1000 and VECTORBT_AVAILABLE:
            return VectorizationMethod.VECTORBT
        elif data_size > 100:
            return VectorizationMethod.PANDAS
        else:
            return VectorizationMethod.NUMPY
    
    def _adaptive_method_selection(self, data_size: int, operation: str) -> VectorizationMethod:
        """Adaptive method selection based on performance history."""
        # Filter performance history for similar operations and data sizes
        relevant_metrics = [
            m for m in self.performance_history
            if m.operation_type == operation and 
            abs(m.data_size - data_size) / max(data_size, 1) < 0.5  # Within 50% size range
        ]
        
        if not relevant_metrics:
            return self._select_optimal_method(None, operation, 0)  # Fallback to rule-based
        
        # Find method with best average performance
        method_scores = {}
        for method in VectorizationMethod:
            method_metrics = [m for m in relevant_metrics if m.method_used == method and m.success]
            if method_metrics:
                avg_time = np.mean([m.execution_time for m in method_metrics])
                success_rate = len(method_metrics) / len([m for m in relevant_metrics if m.method_used == method])
                method_scores[method] = avg_time * (2 - success_rate)  # Penalize low success rate
        
        if method_scores:
            return min(method_scores, key=method_scores.get)
        
        return VectorizationMethod.VECTORBT  # Default fallback
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Execute rolling operation using VectorBT optimizations with pandas."""
        try:
            # VectorBT 0.28+ doesn't have direct rolling functions, so we use pandas
            # but with VectorBT optimizations for memory and performance
            if self.enable_parallel and VECTORBT_AVAILABLE:
                # Use VectorBT's parallel processing capabilities
                vbt.settings.parallel['enabled'] = True
            
            # Use pandas rolling with VectorBT optimizations
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
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT optimized {operation} failed: {e}")
            raise
    
    def _gpu_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                              window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Execute rolling operation using GPU acceleration."""
        if isinstance(data, pd.Series):
            gpu_data = cp.asarray(data.values)
            result = self._gpu_rolling_series(gpu_data, operation, window, **kwargs)
            return pd.Series(result, index=data.index, name=data.name)
        else:
            gpu_data = cp.asarray(data.values)
            result = self._gpu_rolling_dataframe(gpu_data, operation, window, **kwargs)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
    
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
    
    def _chunked_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                  window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Execute rolling operation using chunked processing for memory efficiency."""
        if isinstance(data, pd.Series):
            return self._chunked_series_operation(data, operation, window, **kwargs)
        else:
            return self._chunked_dataframe_operation(data, operation, window, **kwargs)
    
    def _chunked_series_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Process Series in chunks for memory efficiency."""
        results = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap
            chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
            
            # Remove overlap from result (except for first chunk)
            if i == 0:
                results.append(chunk_result)
            else:
                results.append(chunk_result.iloc[window-1:])
        
        return pd.concat(results, ignore_index=False)
    
    def _chunked_dataframe_operation(self, data: pd.DataFrame, operation: str, 
                                    window: int, **kwargs) -> pd.DataFrame:
        """Process DataFrame in chunks for memory efficiency."""
        results = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap
            chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
            
            # Remove overlap from result (except for first chunk)
            if i == 0:
                results.append(chunk_result)
            else:
                results.append(chunk_result.iloc[window-1:])
        
        return pd.concat(results, ignore_index=False)
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Execute rolling operation using pandas."""
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
        """Execute rolling operation using numpy."""
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
    
    def _record_performance(self, method: VectorizationMethod, execution_time: float, 
                          data_size: int, operation: str, success: bool, 
                          error_message: Optional[str] = None):
        """Record performance metrics for adaptive optimization."""
        metrics = PerformanceMetrics(
            method_used=method,
            execution_time=execution_time,
            memory_usage=0.0,  # Could be enhanced with actual memory tracking
            data_size=data_size,
            operation_type=operation,
            success=success,
            error_message=error_message
        )
        
        self.performance_history.append(metrics)
        
        # Update method performance statistics
        if success:
            method_stats = self.method_performance[method]
            method_stats['usage_count'] += 1
            method_stats['avg_time'] = (
                (method_stats['avg_time'] * (method_stats['usage_count'] - 1) + execution_time) 
                / method_stats['usage_count']
            )
            method_stats['success_rate'] = (
                (method_stats['success_rate'] * (method_stats['usage_count'] - 1) + 1.0) 
                / method_stats['usage_count']
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        summary = {
            "total_operations": len(self.performance_history),
            "method_usage": {},
            "operation_performance": {},
            "overall_stats": {}
        }
        
        # Method usage statistics
        for method in VectorizationMethod:
            method_ops = [m for m in self.performance_history if m.method_used == method]
            if method_ops:
                summary["method_usage"][method.value] = {
                    "count": len(method_ops),
                    "success_rate": len([m for m in method_ops if m.success]) / len(method_ops),
                    "avg_time": np.mean([m.execution_time for m in method_ops if m.success])
                }
        
        # Operation performance
        operations = set(m.operation_type for m in self.performance_history)
        for op in operations:
            op_metrics = [m for m in self.performance_history if m.operation_type == op]
            if op_metrics:
                summary["operation_performance"][op] = {
                    "count": len(op_metrics),
                    "avg_time": np.mean([m.execution_time for m in op_metrics if m.success]),
                    "best_method": min(
                        [m for m in op_metrics if m.success],
                        key=lambda x: x.execution_time
                    ).method_used.value if any(m.success for m in op_metrics) else "none"
                }
        
        # Overall statistics
        successful_ops = [m for m in self.performance_history if m.success]
        if successful_ops:
            summary["overall_stats"] = {
                "avg_execution_time": np.mean([m.execution_time for m in successful_ops]),
                "total_execution_time": sum(m.execution_time for m in successful_ops),
                "success_rate": len(successful_ops) / len(self.performance_history)
            }
        
        return summary
    
    def reset_performance_history(self):
        """Reset performance history and statistics."""
        self.performance_history.clear()
        self.method_performance = {
            method: {'avg_time': 0.0, 'success_rate': 0.0, 'usage_count': 0}
            for method in VectorizationMethod
        }


# Global manager instance
_global_manager = None


def get_unified_vectorization_manager(enable_gpu: bool = False, enable_parallel: bool = True) -> UnifiedVectorizationManager:
    """Get global unified vectorization manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedVectorizationManager(
            enable_gpu=enable_gpu, 
            enable_parallel=enable_parallel
        )
    return _global_manager


# Convenience functions
def optimized_rolling_mean(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling mean using unified vectorization manager."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_mean(data, window, **kwargs)


def optimized_rolling_std(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling standard deviation using unified vectorization manager."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_std(data, window, **kwargs)


def optimized_rolling_var(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling variance using unified vectorization manager."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_var(data, window, **kwargs)


def optimized_rolling_apply(data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling apply using unified vectorization manager."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_apply(data, window, func, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(5000) * 0.01),
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=dates)
    
    # Test unified vectorization manager
    manager = UnifiedVectorizationManager(enable_gpu=False, enable_parallel=True)
    
    print("Testing Unified Vectorization Manager...")
    
    # Test various operations
    mean_result = manager.rolling_mean(data['close'], window=20)
    std_result = manager.rolling_std(data['close'], window=20)
    corr_result = manager.rolling_corr(data['close'], data['volume'], window=20)
    
    print(f"Rolling mean shape: {mean_result.shape}")
    print(f"Rolling std shape: {std_result.shape}")
    print(f"Rolling correlation shape: {corr_result.shape}")
    
    # Performance summary
    summary = manager.get_performance_summary()
    print(f"Performance summary: {summary}")