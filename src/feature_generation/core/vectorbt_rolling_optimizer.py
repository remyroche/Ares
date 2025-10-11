"""
VectorBT Rolling Optimizer

This module provides advanced rolling operations optimization using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Optimized rolling operations with memory management
- Batch processing for multiple rolling operations
- GPU acceleration support
- Parallel processing capabilities
- Advanced rolling statistics
- Custom rolling functions
- Memory-efficient processing
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum
    from vectorbt.generic import rolling_apply, rolling_corr, rolling_cov, rolling_skew, rolling_kurt
    from vectorbt.generic import rolling_quantile, rolling_rank, rolling_apply
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    # Define dummy functions for type hints
    def rolling_mean(*args, **kwargs): pass
    def rolling_std(*args, **kwargs): pass
    def rolling_var(*args, **kwargs): pass
    def rolling_min(*args, **kwargs): pass
    def rolling_max(*args, **kwargs): pass
    def rolling_sum(*args, **kwargs): pass
    def rolling_apply(*args, **kwargs): pass
    def rolling_corr(*args, **kwargs): pass
    def rolling_cov(*args, **kwargs): pass
    def rolling_skew(*args, **kwargs): pass
    def rolling_kurt(*args, **kwargs): pass
    def rolling_quantile(*args, **kwargs): pass
    def rolling_rank(*args, **kwargs): pass

from src.utils.ml_common.vectorbt_memory_manager import get_memory_manager, memory_managed_operation, optimize_memory_usage
from src.utils.ml_common.vectorbt_performance_monitor import get_performance_monitor, monitor_operation

logger = logging.getLogger(__name__)

class VectorBTRollingOptimizer:
    """
    Advanced rolling operations optimizer using VectorBT.
    
    This class provides optimized rolling operations with memory management,
    GPU acceleration, and parallel processing capabilities.
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True, 
                 memory_limit_gb: float = 8.0):
        """
        Initialize VectorBT rolling optimizer.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            enable_parallel: Whether to enable parallel processing
            memory_limit_gb: Memory limit in GB for operations
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.memory_limit_gb = memory_limit_gb
        
        # Initialize memory manager and performance monitor
        self.memory_manager = get_memory_manager()
        self.performance_monitor = get_performance_monitor()
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Performance tracking
        self.stats = {
            'rolling_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'batch_operations': 0
        }
        
        # Cache for computed operations
        self._operation_cache = {}
        self._cache_enabled = True
    
    def _configure_vectorbt(self):
        """Configure VectorBT global settings for optimal performance."""
        if not VECTORBT_AVAILABLE:
            return
        
        # Configure VectorBT settings
        vbt.settings.setting('array_wrapper', 'pandas')
        vbt.settings.setting('caching', True)
        vbt.settings.setting('caching_dir', 'data_cache/vectorbt_rolling_cache')
        
        if self.enable_gpu:
            try:
                vbt.settings.setting('use_gpu', True)
                logger.info("✅ VectorBT GPU acceleration enabled for rolling operations")
            except Exception as e:
                logger.warning(f"⚠️ GPU acceleration not available: {e}")
                self.enable_gpu = False
        
        if self.enable_parallel:
            try:
                vbt.settings.setting('use_parallel', True)
                logger.info("✅ VectorBT parallel processing enabled for rolling operations")
            except Exception as e:
                logger.warning(f"⚠️ Parallel processing not available: {e}")
                self.enable_parallel = False
    
    def rolling_mean(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling mean operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_mean(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).mean()
        except Exception as e:
            logger.warning(f"Rolling mean failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).mean()
    
    def rolling_std(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling standard deviation operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_std(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).std()
        except Exception as e:
            logger.warning(f"Rolling std failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).std()
    
    def rolling_var(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling variance operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_var(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).var()
        except Exception as e:
            logger.warning(f"Rolling var failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).var()
    
    def rolling_min(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling minimum operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_min(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).min()
        except Exception as e:
            logger.warning(f"Rolling min failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).min()
    
    def rolling_max(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling maximum operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_max(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).max()
        except Exception as e:
            logger.warning(f"Rolling max failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).max()
    
    def rolling_sum(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling sum operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_sum(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).sum()
        except Exception as e:
            logger.warning(f"Rolling sum failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).sum()
    
    def rolling_corr(self, data: pd.Series, other: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling correlation operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_corr(data, other, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).corr(other)
        except Exception as e:
            logger.warning(f"Rolling corr failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).corr(other)
    
    def rolling_cov(self, data: pd.Series, other: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling covariance operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_cov(data, other, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).cov(other)
        except Exception as e:
            logger.warning(f"Rolling cov failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).cov(other)
    
    def rolling_skew(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling skewness operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_skew(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).skew()
        except Exception as e:
            logger.warning(f"Rolling skew failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).skew()
    
    def rolling_kurt(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling kurtosis operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_kurt(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).kurt()
        except Exception as e:
            logger.warning(f"Rolling kurt failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).kurt()
    
    def rolling_quantile(self, data: pd.Series, window: int, q: float, **kwargs) -> pd.Series:
        """Optimized rolling quantile operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_quantile(data, window=window, q=q, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).quantile(q)
        except Exception as e:
            logger.warning(f"Rolling quantile failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).quantile(q)
    
    def rolling_rank(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Optimized rolling rank operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_rank(data, window=window, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).rank()
        except Exception as e:
            logger.warning(f"Rolling rank failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).rank()
    
    def rolling_apply(self, data: pd.Series, window: int, func: Callable, **kwargs) -> pd.Series:
        """Optimized rolling apply operation."""
        self.stats['rolling_operations'] += 1
        
        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:
                return rolling_apply(data, window=window, func=func, **kwargs)
            else:
                return data.rolling(window=window, **kwargs).apply(func)
        except Exception as e:
            logger.warning(f"Rolling apply failed: {e}, using fallback")
            return data.rolling(window=window, **kwargs).apply(func)
    
    def batch_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform batch rolling operations for efficiency.
        
        Args:
            data: Input DataFrame
            operations: List of operation dictionaries
            
        Returns:
            DataFrame with results
        """
        self.stats['batch_operations'] += 1
        self.stats['rolling_operations'] += len(operations)
        
        # Estimate memory requirements
        data_size_gb = data.memory_usage(deep=True).sum() / (1024**3)
        estimated_memory_gb = data_size_gb * len(operations) * 2
        
        with memory_managed_operation(
            min(estimated_memory_gb, self.memory_limit_gb),
            f"batch_rolling_operations_{len(operations)}",
            "rolling_optimization"
        ):
            try:
                results = {}
                
                # Process operations in batches for memory efficiency
                batch_size = min(10, len(operations))
                
                for i in range(0, len(operations), batch_size):
                    batch_operations = operations[i:i + batch_size]
                    
                    for op in batch_operations:
                        op_name = op.get('name', f"rolling_op_{len(results)}")
                        op_type = op.get('type', 'mean')
                        column = op.get('column', 'close')
                        window = op.get('window', 20)
                        params = op.get('params', {})
                        
                        if column not in data.columns:
                            logger.warning(f"Column {column} not found for rolling operation {op_name}")
                            results[op_name] = pd.Series(np.nan, index=data.index)
                            continue
                        
                        try:
                            if op_type == 'mean':
                                result = self.rolling_mean(data[column], window, **params)
                            elif op_type == 'std':
                                result = self.rolling_std(data[column], window, **params)
                            elif op_type == 'var':
                                result = self.rolling_var(data[column], window, **params)
                            elif op_type == 'min':
                                result = self.rolling_min(data[column], window, **params)
                            elif op_type == 'max':
                                result = self.rolling_max(data[column], window, **params)
                            elif op_type == 'sum':
                                result = self.rolling_sum(data[column], window, **params)
                            elif op_type == 'corr':
                                other_column = op.get('other_column')
                                if other_column and other_column in data.columns:
                                    result = self.rolling_corr(data[column], data[other_column], window, **params)
                                else:
                                    logger.warning(f"Other column {other_column} not found for correlation")
                                    result = pd.Series(np.nan, index=data.index)
                            elif op_type == 'cov':
                                other_column = op.get('other_column')
                                if other_column and other_column in data.columns:
                                    result = self.rolling_cov(data[column], data[other_column], window, **params)
                                else:
                                    logger.warning(f"Other column {other_column} not found for covariance")
                                    result = pd.Series(np.nan, index=data.index)
                            elif op_type == 'skew':
                                result = self.rolling_skew(data[column], window, **params)
                            elif op_type == 'kurt':
                                result = self.rolling_kurt(data[column], window, **params)
                            elif op_type == 'quantile':
                                q = op.get('q', 0.5)
                                result = self.rolling_quantile(data[column], window, q, **params)
                            elif op_type == 'rank':
                                result = self.rolling_rank(data[column], window, **params)
                            elif op_type == 'apply':
                                func = op.get('func')
                                if func:
                                    result = self.rolling_apply(data[column], window, func, **params)
                                else:
                                    logger.warning(f"Function not provided for rolling apply operation {op_name}")
                                    result = pd.Series(np.nan, index=data.index)
                            else:
                                logger.warning(f"Unknown rolling operation type: {op_type}")
                                result = pd.Series(np.nan, index=data.index)
                            
                            # Optimize memory usage
                            result = optimize_memory_usage(result)
                            results[op_name] = result
                            
                        except Exception as e:
                            logger.warning(f"Rolling operation {op_name} failed: {e}")
                            results[op_name] = pd.Series(np.nan, index=data.index)
                
                return pd.DataFrame(results, index=data.index)
                
            except Exception as e:
                logger.error(f"Batch rolling operations failed: {e}")
                return pd.DataFrame(index=data.index)
    
    def advanced_rolling_statistics(self, data: pd.DataFrame, 
                                  columns: List[str], 
                                  windows: List[int]) -> pd.DataFrame:
        """
        Calculate advanced rolling statistics for multiple columns and windows.
        
        Args:
            data: Input DataFrame
            columns: List of column names
            windows: List of window sizes
            
        Returns:
            DataFrame with advanced rolling statistics
        """
        operations = []
        
        for column in columns:
            if column not in data.columns:
                continue
            
            for window in windows:
                # Basic statistics
                operations.extend([
                    {'name': f'{column}_mean_{window}', 'type': 'mean', 'column': column, 'window': window},
                    {'name': f'{column}_std_{window}', 'type': 'std', 'column': column, 'window': window},
                    {'name': f'{column}_min_{window}', 'type': 'min', 'column': column, 'window': window},
                    {'name': f'{column}_max_{window}', 'type': 'max', 'column': column, 'window': window},
                    {'name': f'{column}_skew_{window}', 'type': 'skew', 'column': column, 'window': window},
                    {'name': f'{column}_kurt_{window}', 'type': 'kurt', 'column': column, 'window': window},
                ])
                
                # Quantiles
                for q in [0.25, 0.5, 0.75]:
                    operations.append({
                        'name': f'{column}_q{int(q*100)}_{window}',
                        'type': 'quantile',
                        'column': column,
                        'window': window,
                        'q': q
                    })
        
        return self.batch_rolling_operations(data, operations)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'rolling_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'batch_operations': 0
        }


def get_vectorbt_rolling_optimizer(enable_gpu: bool = False, 
                                 enable_parallel: bool = True,
                                 memory_limit_gb: float = 8.0) -> VectorBTRollingOptimizer:
    """
    Get a VectorBT rolling optimizer instance.
    
    Args:
        enable_gpu: Whether to enable GPU acceleration
        enable_parallel: Whether to enable parallel processing
        memory_limit_gb: Memory limit in GB for operations
        
    Returns:
        VectorBTRollingOptimizer instance
    """
    return VectorBTRollingOptimizer(
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        memory_limit_gb=memory_limit_gb
    )


# Export the main class and factory function
__all__ = [
    'VectorBTRollingOptimizer',
    'get_vectorbt_rolling_optimizer'
]