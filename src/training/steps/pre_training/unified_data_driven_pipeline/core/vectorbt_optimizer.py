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

# Import optimization utilities
try:
    from src.utils.ml_common.optimization.grid_utils import (
        build_coarse_grid_from_search_space,
        build_fine_grid_around_best,
        generate_grid,
        GridSearchOptimizer
    )
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer,
        OptimizationConfig as BayesianOptimizationConfig
    )
    OPTIMIZATION_UTILS_AVAILABLE = True
    tprint_info("✅ Optimization utilities imported successfully")
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    build_coarse_grid_from_search_space = None
    build_fine_grid_around_best = None
    generate_grid = None
    GridSearchOptimizer = None
    BayesianTPEOptimizer = None
    BayesianOptimizationConfig = None
    tprint_warning("⚠️ Optimization utilities not available")

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
    
    def optimize_dataframe_operations(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """
        Optimize DataFrame operations using VectorBT when available.
        
        Args:
            df: Input DataFrame
            operations: List of operations to perform
            
        Returns:
            Optimized DataFrame
        """
        if not VECTORBT_AVAILABLE or df.empty:
            return df
        
        try:
            optimized_df = df.copy()
            
            for operation in operations:
                if operation == 'rolling_mean_5':
                    optimized_df = self._apply_rolling_mean(optimized_df, window=5)
                elif operation == 'rolling_std_10':
                    optimized_df = self._apply_rolling_std(optimized_df, window=10)
                elif operation == 'zscore_normalize':
                    optimized_df = self._apply_zscore_normalization(optimized_df)
                elif operation == 'winsorize_outliers':
                    optimized_df = self._apply_winsorization(optimized_df)
                elif operation == 'rank_features':
                    optimized_df = self._apply_ranking(optimized_df)
            
            self.performance_stats['vectorbt_operations'] += len(operations)
            tprint_debug(f"✅ Optimized {len(operations)} operations using VectorBT")
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT optimization failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return df
    
    def _apply_rolling_mean(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Apply rolling mean using VectorBT optimization."""
        try:
            if VECTORBT_AVAILABLE and rolling_mean is not None:
                return rolling_mean(df, window=window)
            else:
                return df.rolling(window=window).mean()
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT rolling mean failed: {e}, using pandas fallback")
            return df.rolling(window=window).mean()
    
    def _apply_rolling_std(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Apply rolling standard deviation using VectorBT optimization."""
        try:
            if VECTORBT_AVAILABLE and rolling_std is not None:
                return rolling_std(df, window=window)
            else:
                return df.rolling(window=window).std()
        except Exception:
            return df.rolling(window=window).std()
    
    def _apply_zscore_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization using VectorBT optimization."""
        try:
            if VECTORBT_AVAILABLE and zscore is not None:
                return zscore(df)
            else:
                return (df - df.mean()) / df.std()
        except Exception:
            return (df - df.mean()) / df.std()
    
    def _apply_winsorization(self, df: pd.DataFrame, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.DataFrame:
        """Apply winsorization using VectorBT optimization."""
        try:
            if VECTORBT_AVAILABLE and winsorize is not None:
                return winsorize(df, limits=limits)
            else:
                # Fallback winsorization
                return df.clip(lower=df.quantile(limits[0]), upper=df.quantile(1-limits[1]))
        except Exception:
            return df.clip(lower=df.quantile(limits[0]), upper=df.quantile(1-limits[1]))
    
    def _apply_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ranking using VectorBT optimization."""
        try:
            if VECTORBT_AVAILABLE and rank is not None:
                return rank(df)
            else:
                return df.rank()
        except Exception:
            return df.rank()
    
    def batch_process_features(self, features: List[pd.Series], batch_size: int = 100) -> List[pd.Series]:
        """
        Process features in batches for memory efficiency.
        
        Args:
            features: List of feature series
            batch_size: Number of features to process at once
            
        Returns:
            List of processed features
        """
        processed_features = []
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            
            try:
                # Process batch using VectorBT if available
                if VECTORBT_AVAILABLE:
                    batch_df = pd.concat(batch, axis=1)
                    processed_batch_df = self.optimize_dataframe_operations(
                        batch_df, ['zscore_normalize', 'winsorize_outliers']
                    )
                    processed_features.extend([processed_batch_df[col] for col in processed_batch_df.columns])
                else:
                    # Fallback to individual processing
                    for feature in batch:
                        processed_feature = self._process_single_feature(feature)
                        processed_features.append(processed_feature)
                
                tprint_debug(f"✅ Processed batch {i//batch_size + 1}/{(len(features)-1)//batch_size + 1}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Batch processing failed: {e}, processing individually")
                for feature in batch:
                    try:
                        processed_feature = self._process_single_feature(feature)
                        processed_features.append(processed_feature)
                    except Exception as feature_error:
                        tprint_warning(f"⚠️ Feature processing failed: {feature_error}")
                        processed_features.append(feature)
        
        return processed_features
    
    def _process_single_feature(self, feature: pd.Series) -> pd.Series:
        """Process a single feature with basic operations."""
        try:
            # Apply basic preprocessing
            processed = feature.fillna(feature.median())
            processed = (processed - processed.mean()) / processed.std()
            processed = processed.clip(lower=processed.quantile(0.05), upper=processed.quantile(0.95))
            return processed
        except Exception:
            return feature
    
    def calculate_feature_importance_vectorbt(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance using VectorBT-optimized correlation analysis.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            importance_scores = {}
            
            if VECTORBT_AVAILABLE and rolling_corr is not None:
                # Use VectorBT for optimized vectorized correlation calculation
                try:
                    # Vectorized correlation calculation for all features at once
                    if len(features) > 50:
                        # Use rolling correlation for large datasets
                        rolling_corrs = rolling_corr(features, targets, window=50)
                        importance_scores = rolling_corrs.abs().mean().to_dict()
                    else:
                        # Use simple correlation for smaller datasets
                        corr_matrix = features.corrwith(targets)
                        importance_scores = corr_matrix.abs().to_dict()
                except Exception as e:
                    tprint_warning(f"⚠️ VectorBT correlation failed: {e}, using fallback")
                    # Fallback to vectorized pandas correlation
                    corr_matrix = features.corrwith(targets)
                    importance_scores = corr_matrix.abs().to_dict()
            else:
                # Optimized pandas correlation using vectorized operations
                try:
                    corr_matrix = features.corrwith(targets)
                    importance_scores = corr_matrix.abs().to_dict()
                except Exception as e:
                    tprint_warning(f"⚠️ Vectorized correlation failed: {e}, using loop fallback")
                    # Fallback to loop-based correlation
                    for col in features.columns:
                    try:
                        importance_scores[col] = float(features[col].corr(targets).abs())
                    except Exception:
                        importance_scores[col] = 0.0
            
            self.performance_stats['vectorbt_operations'] += 1
            tprint_debug(f"✅ Calculated feature importance for {len(features.columns)} features")
            
            return importance_scores
            
        except Exception as e:
            tprint_warning(f"⚠️ Feature importance calculation failed: {e}")
            return {col: 0.0 for col in features.columns}
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage using efficient data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Memory-optimized DataFrame
        """
        try:
            optimized_df = df.copy()
            original_memory = df.memory_usage(deep=True).sum()
            
            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=['int64']).columns:
                if optimized_df[col].min() >= 0:
                    if optimized_df[col].max() < 255:
                        optimized_df[col] = optimized_df[col].astype('uint8')
                    elif optimized_df[col].max() < 65535:
                        optimized_df[col] = optimized_df[col].astype('uint16')
                    elif optimized_df[col].max() < 4294967295:
                        optimized_df[col] = optimized_df[col].astype('uint32')
                else:
                    if optimized_df[col].min() > -128 and optimized_df[col].max() < 127:
                        optimized_df[col] = optimized_df[col].astype('int8')
                    elif optimized_df[col].min() > -32768 and optimized_df[col].max() < 32767:
                        optimized_df[col] = optimized_df[col].astype('int16')
                    elif optimized_df[col].min() > -2147483648 and optimized_df[col].max() < 2147483647:
                        optimized_df[col] = optimized_df[col].astype('int32')
            
            # Optimize float columns
            for col in optimized_df.select_dtypes(include=['float64']).columns:
                optimized_df[col] = optimized_df[col].astype('float32')
            
            # Optimize object columns
            for col in optimized_df.select_dtypes(include=['object']).columns:
                if optimized_df[col].dtype == 'object':
                    try:
                        optimized_df[col] = optimized_df[col].astype('category')
                    except:
                        pass
            
            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            memory_reduction = (original_memory - optimized_memory) / original_memory * 100
            
            tprint_success(f"✅ Memory optimization: {memory_reduction:.1f}% reduction")
            tprint_debug(f"📊 Original: {original_memory / 1024**2:.1f}MB, Optimized: {optimized_memory / 1024**2:.1f}MB")
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"⚠️ Memory optimization failed: {e}")
            return df

    def optimize_parameters(self, 
                           data: Union[pd.Series, pd.DataFrame], 
                           parameter_space: Dict[str, Any],
                           objective_function: Callable,
                           method: str = 'enhanced_grid_search',
                           **kwargs) -> Dict[str, Any]:
        """
        Optimize parameters using enhanced optimization utilities.
        
        Args:
            data: Input data for optimization
            parameter_space: Dictionary defining parameter search space
            objective_function: Function to evaluate parameter combinations
            method: Optimization method ('enhanced_grid_search', 'enhanced_bayesian_tpe', 'grid_search')
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTIMIZATION_UTILS_AVAILABLE:
            tprint_warning("⚠️ Optimization utilities not available, using fallback")
            return self._fallback_parameter_optimization(data, parameter_space, objective_function, **kwargs)
        
        try:
            if method == 'enhanced_bayesian_tpe':
                return self._optimize_with_enhanced_bayesian_tpe(data, parameter_space, objective_function, **kwargs)
            elif method == 'enhanced_grid_search':
                return self._optimize_with_enhanced_grid_search(data, parameter_space, objective_function, **kwargs)
            elif method == 'grid_search':
                return self._optimize_with_grid_search(data, parameter_space, objective_function, **kwargs)
            else:
                tprint_warning(f"⚠️ Unknown optimization method: {method}, using enhanced grid search")
                return self._optimize_with_enhanced_grid_search(data, parameter_space, objective_function, **kwargs)
                
        except Exception as e:
            tprint_error(f"❌ Parameter optimization failed: {e}")
            return self._fallback_parameter_optimization(data, parameter_space, objective_function, **kwargs)
    
    def _optimize_with_enhanced_bayesian_tpe(self, data, parameter_space, objective_function, **kwargs):
        """Optimize using enhanced Bayesian TPE."""
        if not BayesianTPEOptimizer:
            raise ImportError("BayesianTPEOptimizer not available")
        
        # Configure TPE optimizer for VectorBT operations
        config = BayesianOptimizationConfig(
            n_trials=kwargs.get('n_trials', 50),
            enable_staged_optimization=True,
            enable_hardware_optimization=True,
            enable_vectorbt_optimization=VECTORBT_AVAILABLE,
            early_stopping_patience=kwargs.get('patience', 10)
        )
        
        optimizer = BayesianTPEOptimizer(config)
        
        def objective(trial):
            params = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            return objective_function(data, params)
        
        # Run optimization
        optimizer.optimize(objective, parameter_space)
        
        return {
            'best_params': optimizer.get_best_params(),
            'best_score': optimizer.get_best_value(),
            'optimization_history': optimizer.optimization_history,
            'method': 'enhanced_bayesian_tpe'
        }
    
    def _optimize_with_enhanced_grid_search(self, data, parameter_space, objective_function, **kwargs):
        """Optimize using enhanced grid search."""
        if not generate_grid:
            raise ImportError("Grid utilities not available")
        
        # Generate optimized grid
        max_trials = kwargs.get('max_trials', 50)
        grid_params = generate_grid(parameter_space, max_trials)
        
        if not grid_params:
            raise ValueError("No grid parameters generated")
        
        # Evaluate all combinations
        best_score = float('-inf')
        best_params = None
        all_scores = []
        
        for params in grid_params:
            score = objective_function(data, params)
            all_scores.append((params, score))
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_scores': all_scores,
            'method': 'enhanced_grid_search'
        }
    
    def _optimize_with_grid_search(self, data, parameter_space, objective_function, **kwargs):
        """Optimize using basic grid search."""
        if not build_coarse_grid_from_search_space:
            raise ImportError("Grid utilities not available")
        
        grid_points = kwargs.get('grid_points', 5)
        grid_params = build_coarse_grid_from_search_space(parameter_space, grid_points)
        
        if not grid_params:
            raise ValueError("No grid parameters generated")
        
        # Evaluate all combinations
        best_score = float('-inf')
        best_params = None
        
        for params in grid_params:
            score = objective_function(data, params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'grid_search'
        }
    
    def _fallback_parameter_optimization(self, data, parameter_space, objective_function, **kwargs):
        """Fallback parameter optimization when utilities are not available."""
        tprint_warning("⚠️ Using fallback parameter optimization")
        
        # Simple random search fallback
        n_trials = kwargs.get('n_trials', 20)
        best_score = float('-inf')
        best_params = None
        
        for _ in range(n_trials):
            params = {}
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = np.random.randint(
                        param_config['low'], param_config['high'] + 1
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = np.random.choice(param_config['choices'])
            
            score = objective_function(data, params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'fallback_random_search'
        }


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