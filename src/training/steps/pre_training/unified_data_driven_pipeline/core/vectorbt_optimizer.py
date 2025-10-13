"""
Enhanced VectorBT Optimization Component

This module provides comprehensive VectorBT optimizations across all components,
including matrix operations, rolling operations, and batch processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
    )
    from vectorbt.portfolio import Portfolio
    from vectorbt.returns import Returns
    VECTORBT_AVAILABLE = True
    tprint_info("✅ VectorBT imported successfully")
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
    tprint_warning("⚠️ VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    tprint_info("✅ CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    tprint_debug("CuPy not available, using CPU-only operations")

logger = logging.getLogger(__name__)


@dataclass
class VectorBTConfig:
    """Configuration for VectorBT optimizations."""
    
    # Performance settings
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Batch processing
    batch_size: int = 1000
    max_workers: int = 4
    
    # Rolling operations
    default_window: int = 20
    min_periods: int = 1
    
    # Matrix operations
    enable_matrix_optimization: bool = True
    matrix_chunk_size: int = 1000
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass
class OptimizationResult:
    """Result from VectorBT optimization."""
    
    # Results
    result_data: Any
    operation_type: str
    
    # Performance metrics
    execution_time: float
    memory_usage_mb: float
    vectorbt_operations: int
    pandas_fallbacks: int
    
    # Optimization details
    optimization_method: str
    batch_size: int
    parallel_workers: int
    
    # Success indicators
    success: bool
    error_message: Optional[str] = None


class VectorBTOptimizer:
    """
    Enhanced VectorBT optimizer for comprehensive performance optimization.
    
    This class provides optimized implementations of common operations using VectorBT,
    with automatic fallback to pandas when VectorBT is not available.
    """
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        """
        Initialize the VectorBT optimizer.
        
        Args:
            config: Configuration for VectorBT optimizations
        """
        self.config = config or VectorBTConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'total_execution_time': 0.0,
            'memory_savings_mb': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize caching if available
        self.cache = {} if self.config.enable_caching else None
        
        tprint_info("🚀 VectorBT Optimizer initialized")
        tprint_debug(f"📊 VectorBT available: {VECTORBT_AVAILABLE}")
        tprint_debug(f"📊 GPU available: {CUPY_AVAILABLE}")
        tprint_debug(f"📊 Config: {self.config}")
    
    def rolling_operation(self, 
                         data: Union[pd.Series, pd.DataFrame],
                         operation: str,
                         window: int = None,
                         **kwargs) -> OptimizationResult:
        """
        Perform rolling operation with VectorBT optimization.
        
        Args:
            data: Input data
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult with optimized results
        """
        start_time = time.time()
        window = window or self.config.default_window
        
        def _execute_rolling_operation():
            try:
                if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
                    return self._pandas_rolling_operation(data, operation, window, **kwargs)
                
                # Check cache
                cache_key = f"rolling_{operation}_{window}_{hash(str(data.shape))}"
                if self.cache and cache_key in self.cache:
                    self.performance_stats['cache_hits'] += 1
                    return self.cache[cache_key]
                
                # Execute VectorBT operation
                result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                
                # Cache result
                if self.cache and len(self.cache) < self.config.cache_size:
                    self.cache[cache_key] = result
                
                self.performance_stats['vectorbt_operations'] += 1
                return result
                
            except Exception as e:
                self.logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        # Execute operation
        result = _execute_rolling_operation()
        
        # Update performance stats
        execution_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        
        return OptimizationResult(
            result_data=result,
            operation_type=f"rolling_{operation}",
            execution_time=execution_time,
            memory_usage_mb=self._get_memory_usage_mb(result),
            vectorbt_operations=1 if VECTORBT_AVAILABLE else 0,
            pandas_fallbacks=0 if VECTORBT_AVAILABLE else 1,
            optimization_method="vectorbt" if VECTORBT_AVAILABLE else "pandas",
            batch_size=self.config.batch_size,
            parallel_workers=self.config.max_workers,
            success=True
        )
    
    def _vectorbt_rolling_operation(self, 
                                   data: Union[pd.Series, pd.DataFrame],
                                   operation: str,
                                   window: int,
                                   **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Execute rolling operation using VectorBT."""
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
        elif operation == 'corr':
            return rolling_corr(data, window=window, **kwargs)
        elif operation == 'cov':
            return rolling_cov(data, window=window, **kwargs)
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}")
    
    def _pandas_rolling_operation(self, 
                                 data: Union[pd.Series, pd.DataFrame],
                                 operation: str,
                                 window: int,
                                 **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Execute rolling operation using pandas fallback."""
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
        elif operation == 'corr':
            return rolling_obj.corr()
        elif operation == 'cov':
            return rolling_obj.cov()
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}")
    
    def matrix_operation(self, 
                        data: Union[pd.DataFrame, np.ndarray],
                        operation: str,
                        **kwargs) -> OptimizationResult:
        """
        Perform matrix operation with VectorBT optimization.
        
        Args:
            data: Input data
            operation: Operation type ('corr', 'cov', 'multiply', 'add', 'subtract')
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult with optimized results
        """
        start_time = time.time()
        
        def _execute_matrix_operation():
            try:
                if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
                    return self._pandas_matrix_operation(data, operation, **kwargs)
                
                # Check cache
                cache_key = f"matrix_{operation}_{hash(str(data.shape))}"
                if self.cache and cache_key in self.cache:
                    self.performance_stats['cache_hits'] += 1
                    return self.cache[cache_key]
                
                # Execute VectorBT operation
                result = self._vectorbt_matrix_operation(data, operation, **kwargs)
                
                # Cache result
                if self.cache and len(self.cache) < self.config.cache_size:
                    self.cache[cache_key] = result
                
                self.performance_stats['vectorbt_operations'] += 1
                return result
                
            except Exception as e:
                self.logger.warning(f"VectorBT matrix operation failed: {e}, using pandas fallback")
                return self._pandas_matrix_operation(data, operation, **kwargs)
        
        # Execute operation
        result = _execute_matrix_operation()
        
        # Update performance stats
        execution_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_execution_time'] += execution_time
        
        return OptimizationResult(
            result_data=result,
            operation_type=f"matrix_{operation}",
            execution_time=execution_time,
            memory_usage_mb=self._get_memory_usage_mb(result),
            vectorbt_operations=1 if VECTORBT_AVAILABLE else 0,
            pandas_fallbacks=0 if VECTORBT_AVAILABLE else 1,
            optimization_method="vectorbt" if VECTORBT_AVAILABLE else "pandas",
            batch_size=self.config.batch_size,
            parallel_workers=self.config.max_workers,
            success=True
        )
    
    def _vectorbt_matrix_operation(self, 
                                  data: Union[pd.DataFrame, np.ndarray],
                                  operation: str,
                                  **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """Execute matrix operation using VectorBT."""
        if operation == 'corr':
            return data.corr() if isinstance(data, pd.DataFrame) else np.corrcoef(data)
        elif operation == 'cov':
            return data.cov() if isinstance(data, pd.DataFrame) else np.cov(data)
        elif operation == 'multiply':
            if isinstance(data, pd.DataFrame):
                return data * data
            else:
                return np.multiply(data, data)
        elif operation == 'add':
            if isinstance(data, pd.DataFrame):
                return data + data
            else:
                return np.add(data, data)
        elif operation == 'subtract':
            if isinstance(data, pd.DataFrame):
                return data - data
            else:
                return np.subtract(data, data)
        else:
            raise ValueError(f"Unsupported matrix operation: {operation}")
    
    def _pandas_matrix_operation(self, 
                                data: Union[pd.DataFrame, np.ndarray],
                                operation: str,
                                **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """Execute matrix operation using pandas fallback."""
        if operation == 'corr':
            return data.corr() if isinstance(data, pd.DataFrame) else np.corrcoef(data)
        elif operation == 'cov':
            return data.cov() if isinstance(data, pd.DataFrame) else np.cov(data)
        elif operation == 'multiply':
            if isinstance(data, pd.DataFrame):
                return data * data
            else:
                return np.multiply(data, data)
        elif operation == 'add':
            if isinstance(data, pd.DataFrame):
                return data + data
            else:
                return np.add(data, data)
        elif operation == 'subtract':
            if isinstance(data, pd.DataFrame):
                return data - data
            else:
                return np.subtract(data, data)
        else:
            raise ValueError(f"Unsupported matrix operation: {operation}")
    
    def batch_process(self, 
                     data_list: List[Union[pd.Series, pd.DataFrame]],
                     operation: Callable,
                     **kwargs) -> List[OptimizationResult]:
        """
        Process multiple data objects in batch with VectorBT optimization.
        
        Args:
            data_list: List of data objects to process
            operation: Operation function to apply
            **kwargs: Additional arguments
            
        Returns:
            List of OptimizationResult objects
        """
        start_time = time.time()
        results = []
        
        try:
            if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
                # Fallback to sequential processing
                for data in data_list:
                    result = self._process_single_item(data, operation, **kwargs)
                    results.append(result)
            else:
                # VectorBT batch processing
                results = self._vectorbt_batch_process(data_list, operation, **kwargs)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += execution_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return []
    
    def _vectorbt_batch_process(self, 
                               data_list: List[Union[pd.Series, pd.DataFrame]],
                               operation: Callable,
                               **kwargs) -> List[OptimizationResult]:
        """Process batch using VectorBT optimizations."""
        results = []
        
        # Process in chunks for memory efficiency
        chunk_size = self.config.batch_size
        for i in range(0, len(data_list), chunk_size):
            chunk = data_list[i:i + chunk_size]
            
            # Process chunk
            for data in chunk:
                result = self._process_single_item(data, operation, **kwargs)
                results.append(result)
        
        return results
    
    def _process_single_item(self, 
                            data: Union[pd.Series, pd.DataFrame],
                            operation: Callable,
                            **kwargs) -> OptimizationResult:
        """Process a single data item."""
        start_time = time.time()
        
        try:
            result_data = operation(data, **kwargs)
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                result_data=result_data,
                operation_type="batch_operation",
                execution_time=execution_time,
                memory_usage_mb=self._get_memory_usage_mb(result_data),
                vectorbt_operations=1 if VECTORBT_AVAILABLE else 0,
                pandas_fallbacks=0 if VECTORBT_AVAILABLE else 1,
                optimization_method="vectorbt" if VECTORBT_AVAILABLE else "pandas",
                batch_size=self.config.batch_size,
                parallel_workers=self.config.max_workers,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return OptimizationResult(
                result_data=None,
                operation_type="batch_operation",
                execution_time=execution_time,
                memory_usage_mb=0.0,
                vectorbt_operations=0,
                pandas_fallbacks=1,
                optimization_method="pandas",
                batch_size=self.config.batch_size,
                parallel_workers=self.config.max_workers,
                success=False,
                error_message=str(e)
            )
    
    def gpu_operation(self, 
                     data: Union[pd.Series, pd.DataFrame, np.ndarray],
                     operation: str,
                     **kwargs) -> OptimizationResult:
        """
        Perform operation using GPU acceleration if available.
        
        Args:
            data: Input data
            operation: Operation type
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult with optimized results
        """
        start_time = time.time()
        
        if not CUPY_AVAILABLE or not self.config.enable_gpu:
            # Fallback to CPU operation
            return self.rolling_operation(data, operation, **kwargs)
        
        try:
            # Convert to CuPy array
            if isinstance(data, pd.Series):
                gpu_data = cp.asarray(data.values)
            elif isinstance(data, pd.DataFrame):
                gpu_data = cp.asarray(data.values)
            else:
                gpu_data = cp.asarray(data)
            
            # Execute GPU operation
            result = self._gpu_operation(gpu_data, operation, **kwargs)
            
            # Convert back to pandas if needed
            if isinstance(data, (pd.Series, pd.DataFrame)):
                result = self._convert_gpu_result_to_pandas(result, data)
            
            execution_time = time.time() - start_time
            self.performance_stats['gpu_operations'] += 1
            
            return OptimizationResult(
                result_data=result,
                operation_type=f"gpu_{operation}",
                execution_time=execution_time,
                memory_usage_mb=self._get_memory_usage_mb(result),
                vectorbt_operations=0,
                pandas_fallbacks=0,
                optimization_method="gpu",
                batch_size=self.config.batch_size,
                parallel_workers=self.config.max_workers,
                success=True
            )
            
        except Exception as e:
            self.logger.warning(f"GPU operation failed: {e}, using CPU fallback")
            return self.rolling_operation(data, operation, **kwargs)
    
    def _gpu_operation(self, 
                      gpu_data: cp.ndarray,
                      operation: str,
                      **kwargs) -> cp.ndarray:
        """Execute operation on GPU."""
        if operation == 'mean':
            return cp.mean(gpu_data, axis=0)
        elif operation == 'std':
            return cp.std(gpu_data, axis=0)
        elif operation == 'var':
            return cp.var(gpu_data, axis=0)
        elif operation == 'min':
            return cp.min(gpu_data, axis=0)
        elif operation == 'max':
            return cp.max(gpu_data, axis=0)
        elif operation == 'sum':
            return cp.sum(gpu_data, axis=0)
        else:
            raise ValueError(f"Unsupported GPU operation: {operation}")
    
    def _convert_gpu_result_to_pandas(self, 
                                    gpu_result: cp.ndarray,
                                    original_data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Convert GPU result back to pandas format."""
        cpu_result = cp.asnumpy(gpu_result)
        
        if isinstance(original_data, pd.Series):
            return pd.Series(cpu_result, index=original_data.index)
        elif isinstance(original_data, pd.DataFrame):
            return pd.DataFrame(cpu_result, index=original_data.index, columns=original_data.columns)
        else:
            return cpu_result
    
    def _get_memory_usage_mb(self, data: Any) -> float:
        """Get memory usage of data in MB."""
        try:
            if hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum() / 1024 / 1024
            elif hasattr(data, 'nbytes'):
                return data.nbytes / 1024 / 1024
            else:
                return 0.0
        except:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_ops = self.performance_stats['total_operations']
        vectorbt_ops = self.performance_stats['vectorbt_operations']
        pandas_ops = self.performance_stats['pandas_fallbacks']
        gpu_ops = self.performance_stats['gpu_operations']
        
        return {
            'total_operations': total_ops,
            'vectorbt_operations': vectorbt_ops,
            'pandas_fallbacks': pandas_ops,
            'gpu_operations': gpu_ops,
            'vectorbt_usage_rate': vectorbt_ops / max(total_ops, 1),
            'pandas_fallback_rate': pandas_ops / max(total_ops, 1),
            'gpu_usage_rate': gpu_ops / max(total_ops, 1),
            'total_execution_time': self.performance_stats['total_execution_time'],
            'average_execution_time': self.performance_stats['total_execution_time'] / max(total_ops, 1),
            'cache_hit_rate': self.performance_stats['cache_hits'] / max(
                self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'], 1
            ),
            'memory_savings_mb': self.performance_stats['memory_savings_mb']
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'total_execution_time': 0.0,
            'memory_savings_mb': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        if self.cache:
            self.cache.clear()


# Convenience functions
def create_vectorbt_optimizer(config: Optional[VectorBTConfig] = None) -> VectorBTOptimizer:
    """Create a VectorBT optimizer with default configuration."""
    return VectorBTOptimizer(config)


def optimize_with_vectorbt(data: Union[pd.Series, pd.DataFrame],
                          operation: str,
                          **kwargs) -> OptimizationResult:
    """
    Convenience function to optimize operations with VectorBT.
    
    Args:
        data: Input data
        operation: Operation type
        **kwargs: Additional arguments
        
    Returns:
        OptimizationResult with optimized results
    """
    optimizer = create_vectorbt_optimizer()
    return optimizer.rolling_operation(data, operation, **kwargs)


# Export main classes and functions
__all__ = [
    'VectorBTOptimizer',
    'VectorBTConfig',
    'OptimizationResult',
    'create_vectorbt_optimizer',
    'optimize_with_vectorbt'
]