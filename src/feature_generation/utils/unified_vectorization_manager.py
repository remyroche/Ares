"""
Unified Vectorization Manager

This module provides a comprehensive vectorization management system that combines
VectorBT rolling optimizer, vectorization optimizer, and hardware acceleration
for maximum performance in feature generation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import time
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov, rolling_quantile,
        scale, rank, zscore, winsorize, clip, quantile
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None

# Import optimization components
try:
    from .vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from .vectorization_optimizer import get_vectorization_optimizer, VectorizationOptimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    get_vectorization_optimizer = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

@dataclass
class UnifiedVectorizationConfig:
    """Configuration for unified vectorization management."""
    # VectorBT Configuration
    enable_vectorbt: bool = True
    vectorbt_threshold: int = 1000
    enable_vectorbt_parallel: bool = True
    enable_vectorbt_gpu: bool = False
    
    # Memory Management
    memory_limit_gb: float = 8.0
    enable_memory_optimization: bool = True
    chunk_size: int = 10000
    
    # Performance Optimization
    enable_batch_processing: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Hardware Acceleration
    enable_gpu_acceleration: bool = False
    enable_simd_optimization: bool = True
    
    # Monitoring
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = True

class UnifiedVectorizationManager:
    """
    Unified vectorization manager that combines all optimization strategies
    for maximum performance in feature generation.
    """
    
    def __init__(self, config: Optional[UnifiedVectorizationConfig] = None):
        """Initialize the unified vectorization manager."""
        self.config = config or UnifiedVectorizationConfig()
        self.logger = logger.getChild('UnifiedVectorizationManager')
        
        # Initialize components
        self.vectorbt_optimizer = None
        self.vectorization_optimizer = None
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimizations': 0,
            'total_execution_time': 0.0,
            'memory_usage': 0.0
        }
        
        # Cache for computed results
        self._cache = {}
        self._cache_enabled = self.config.enable_caching
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ Unified Vectorization Manager initialized")
    
    def _initialize_components(self):
        """Initialize optimization components."""
        try:
            # Initialize VectorBT rolling optimizer
            if self.config.enable_vectorbt and OPTIMIZATION_AVAILABLE and get_vectorbt_rolling_optimizer:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_vectorbt_gpu,
                    enable_parallel=self.config.enable_vectorbt_parallel
                )
                self.logger.info("✅ VectorBT rolling optimizer initialized")
            
            # Initialize vectorization optimizer
            if OPTIMIZATION_AVAILABLE and get_vectorization_optimizer:
                self.vectorization_optimizer = get_vectorization_optimizer()
                self.logger.info("✅ Vectorization optimizer initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some components not available: {e}")
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        start_time = time.time()
        
        try:
            # Use vectorization optimizer if available
            if self.vectorization_optimizer:
                optimized_data = self.vectorization_optimizer.optimize_dataframe_processing(data)
            else:
                optimized_data = self._basic_dataframe_optimization(data)
            
            # Track performance
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += time.time() - start_time
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return data
    
    def _basic_dataframe_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame optimization without vectorization optimizer."""
        optimized_df = data.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            if optimized_df[col].dtype == np.float64:
                if (optimized_df[col].max() < np.finfo(np.float32).max and
                    optimized_df[col].min() > np.finfo(np.float32).min):
                    optimized_df[col] = optimized_df[col].astype(np.float32)
                    self.performance_stats['memory_optimizations'] += 1
            elif optimized_df[col].dtype == np.int64:
                if (optimized_df[col].max() < np.iinfo(np.int32).max and
                    optimized_df[col].min() > np.iinfo(np.int32).min):
                    optimized_df[col] = optimized_df[col].astype(np.int32)
                    self.performance_stats['memory_optimizations'] += 1
        
        return optimized_df
    
    def vectorized_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                   operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform vectorized rolling operation with optimal method selection."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"rolling_{operation}_{window}_{hash(str(data.index))}"
        if self._cache_enabled and cache_key in self._cache:
            self.performance_stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        try:
            # Use VectorBT optimizer if available and data is large enough
            if (self.vectorbt_optimizer and 
                len(data) >= self.config.vectorbt_threshold and 
                VECTORBT_AVAILABLE):
                
                result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
                
            else:
                # Use pandas fallback
                result = self._pandas_rolling_operation(data, operation, window, **kwargs)
            
            # Cache result
            if self._cache_enabled and len(self._cache) < self.config.cache_size:
                self._cache[cache_key] = result
            elif self._cache_enabled:
                self.performance_stats['cache_misses'] += 1
            
            # Track performance
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Rolling operation failed: {e}")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                  operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")
        
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
                raise ValueError(f"Unsupported operation: {operation}")
        
        except Exception as e:
            self.logger.warning(f"VectorBT rolling operation failed: {e}")
            raise
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using pandas."""
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
            raise ValueError(f"Unsupported operation: {operation}")
    
    def vectorized_scale(self, data: pd.Series, method: str = 'zscore', **kwargs) -> pd.Series:
        """Scale data using VectorBT scaling functions."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"scale_{method}_{hash(str(data.index))}"
        if self._cache_enabled and cache_key in self._cache:
            self.performance_stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= self.config.vectorbt_threshold:
                result = self._vectorbt_scale(data, method, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            else:
                result = self._pandas_scale(data, method, **kwargs)
            
            # Cache result
            if self._cache_enabled and len(self._cache) < self.config.cache_size:
                self._cache[cache_key] = result
            elif self._cache_enabled:
                self.performance_stats['cache_misses'] += 1
            
            # Track performance
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Scaling operation failed: {e}")
            return self._pandas_scale(data, method, **kwargs)
    
    def _vectorbt_scale(self, data: pd.Series, method: str, **kwargs) -> pd.Series:
        """Scale data using VectorBT."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")
        
        try:
            if method == 'zscore':
                return zscore(data, **kwargs)
            elif method == 'minmax':
                return scale(data, method='minmax', **kwargs)
            elif method == 'robust':
                return scale(data, method='robust', **kwargs)
            elif method == 'quantile':
                return quantile(data, **kwargs)
            elif method == 'winsorize':
                return winsorize(data, **kwargs)
            elif method == 'rank':
                return rank(data, **kwargs)
            elif method == 'clip':
                return clip(data, **kwargs)
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
        
        except Exception as e:
            self.logger.warning(f"VectorBT scaling failed: {e}")
            raise
    
    def _pandas_scale(self, data: pd.Series, method: str, **kwargs) -> pd.Series:
        """Fallback scaling using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    def batch_vectorized_operations(self, data: pd.DataFrame, 
                                  operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Perform multiple vectorized operations in batch for efficiency."""
        start_time = time.time()
        
        try:
            results = {}
            
            for op in operations:
                op_type = op.get('type')
                op_params = op.get('params', {})
                op_name = op.get('name', f"{op_type}_{len(results)}")
                
                if op_type == 'rolling':
                    operation = op_params.get('operation')
                    window = op_params.get('window')
                    column = op_params.get('column', 'close')
                    results[op_name] = self.vectorized_rolling_operation(
                        data[column], operation, window, **op_params
                    )
                elif op_type == 'scale':
                    method = op_params.get('method', 'zscore')
                    column = op_params.get('column', 'close')
                    results[op_name] = self.vectorized_scale(
                        data[column], method, **op_params
                    )
            
            # Track performance
            self.performance_stats['batch_operations'] += 1
            self.performance_stats['total_operations'] += len(operations)
            self.performance_stats['total_execution_time'] += time.time() - start_time
            
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Batch operations failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_execution_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimizations': 0,
            'total_execution_time': 0.0,
            'memory_usage': 0.0
        }
    
    def cleanup(self):
        """Cleanup resources and clear cache."""
        self._cache.clear()
        self.logger.info("🧹 Unified Vectorization Manager cleanup completed")

# Global instance
_unified_vectorization_manager: Optional[UnifiedVectorizationManager] = None

def get_unified_vectorization_manager(config: Optional[UnifiedVectorizationConfig] = None) -> UnifiedVectorizationManager:
    """Get or create the global unified vectorization manager instance."""
    global _unified_vectorization_manager
    
    if _unified_vectorization_manager is None:
        _unified_vectorization_manager = UnifiedVectorizationManager(config)
    
    return _unified_vectorization_manager

def optimize_dataframe_processing(df: pd.DataFrame, 
                                config: Optional[UnifiedVectorizationConfig] = None) -> pd.DataFrame:
    """Convenience function to optimize DataFrame processing."""
    manager = get_unified_vectorization_manager(config)
    return manager.optimize_dataframe(df)

def vectorized_rolling_operation(data: Union[pd.Series, pd.DataFrame],
                               operation: str,
                               window: int,
                               config: Optional[UnifiedVectorizationConfig] = None,
                               **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Convenience function for vectorized rolling operations."""
    manager = get_unified_vectorization_manager(config)
    return manager.vectorized_rolling_operation(data, operation, window, **kwargs)

def vectorized_scale(data: pd.Series,
                    method: str = 'zscore',
                    config: Optional[UnifiedVectorizationConfig] = None,
                    **kwargs) -> pd.Series:
    """Convenience function for vectorized scaling."""
    manager = get_unified_vectorization_manager(config)
    return manager.vectorized_scale(data, method, **kwargs)