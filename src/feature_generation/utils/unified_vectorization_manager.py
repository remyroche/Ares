"""
Unified Vectorization Manager

This module provides a unified interface for VectorBT optimizations across all
feature generation categories, consolidating VectorBTRollingOptimizer and other
VectorBT utilities into a single, efficient manager.

Key Features:
- Centralized VectorBT configuration
- Unified rolling operations interface
- Batch processing optimization
- Memory management
- Performance monitoring
- Hardware acceleration
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import time
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import existing optimizers
try:
    from .vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

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
    vectorbt_threshold: int = 1000  # Minimum data size for VectorBT
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_limit_gb: float = 8.0
    
    # Rolling Operations
    enable_rolling_optimizer: bool = True
    rolling_chunk_size: int = 10000
    enable_rolling_caching: bool = True
    
    # Batch Processing
    enable_batch_processing: bool = True
    batch_size: int = 1000
    max_batch_size: int = 10000
    
    # Memory Management
    enable_memory_optimization: bool = True
    memory_efficiency_threshold: float = 0.8
    enable_memory_pooling: bool = True
    
    # Performance Monitoring
    enable_performance_monitoring: bool = True
    enable_detailed_logging: bool = False

class UnifiedVectorizationManager:
    """
    Unified manager for all VectorBT optimizations in feature generation.
    
    This class provides a single interface for:
    - VectorBTRollingOptimizer operations
    - Batch processing
    - Memory management
    - Performance monitoring
    - Hardware acceleration
    """
    
    def __init__(self, config: Optional[UnifiedVectorizationConfig] = None):
        """Initialize the unified vectorization manager."""
        self.config = config or UnifiedVectorizationConfig()
        self.logger = logger.getChild('UnifiedVectorizationManager')
        
        # Initialize components
        self.rolling_optimizer = None
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'rolling_operations': 0,
            'batch_operations': 0,
            'gpu_operations': 0,
            'memory_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0,
            'memory_savings': 0.0
        }
        
        # Initialize VectorBT settings
        self._configure_vectorbt()
        
        # Initialize rolling optimizer
        self._initialize_rolling_optimizer()
        
        # Initialize caching
        self._cache = {} if self.config.enable_rolling_caching else None
        
        self.logger.info("✅ Unified Vectorization Manager initialized")
    
    def _configure_vectorbt(self):
        """Configure VectorBT global settings."""
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
            return
        
        try:
            # Configure VectorBT settings
            vbt.settings.setting('array_wrapper', 'pandas')
            vbt.settings.setting('caching', True)
            vbt.settings.setting('caching_dir', 'data_cache/vectorbt_unified_cache')
            vbt.settings.setting('cache_size', 1000)  # 1GB cache
            vbt.settings.setting('cache_ttl', 3600)  # 1 hour TTL
            vbt.settings.setting('cache_compression', True)
            vbt.settings.setting('memory_limit', self.config.memory_limit_gb * 1024**3)
            vbt.settings.setting('chunk_size', self.config.rolling_chunk_size)
            
            if self.config.enable_gpu and CUPY_AVAILABLE:
                vbt.settings.setting('use_gpu', True)
                self.logger.info("✅ VectorBT GPU acceleration enabled")
            
            if self.config.enable_parallel:
                vbt.settings.setting('use_parallel', True)
                vbt.settings.setting('n_threads', min(8, os.cpu_count()))
                self.logger.info("✅ VectorBT parallel processing enabled")
                
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT configuration failed: {e}")
    
    def _initialize_rolling_optimizer(self):
        """Initialize the VectorBT rolling optimizer."""
        if not ROLLING_OPTIMIZER_AVAILABLE or not self.config.enable_rolling_optimizer:
            self.logger.warning("VectorBT Rolling Optimizer not available")
            return
        
        try:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel
            )
            self.logger.info("✅ VectorBT Rolling Optimizer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Rolling optimizer initialization failed: {e}")
            self.rolling_optimizer = None
    
    def should_use_vectorbt(self, data: Union[pd.DataFrame, pd.Series]) -> bool:
        """Determine if VectorBT should be used for this data."""
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
            return False
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Check data size threshold
        if data_size < self.config.vectorbt_threshold:
            return False
        
        # Check memory limit
        if hasattr(data, 'memory_usage'):
            memory_usage_gb = data.memory_usage(deep=True).sum() / (1024**3)
            if memory_usage_gb > self.config.memory_limit_gb:
                self.logger.warning(f"Data size ({memory_usage_gb:.2f}GB) exceeds memory limit ({self.config.memory_limit_gb}GB)")
                return False
        
        return True
    
    def rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """
        Perform rolling operation using the best available method.
        
        Args:
            data: Input data series
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', 'corr', 'cov', 'quantile', 'skew', 'kurt')
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['rolling_operations'] += 1
        
        # Check cache first
        if self._cache is not None:
            cache_key = self._generate_cache_key(data, operation, window, **kwargs)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        try:
            # Use rolling optimizer if available
            if self.rolling_optimizer and self.should_use_vectorbt(data):
                result = self._execute_rolling_optimizer_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            else:
                # Fallback to direct VectorBT or pandas
                result = self._execute_direct_operation(data, operation, window, **kwargs)
            
            # Cache result
            if self._cache is not None:
                self._cache[cache_key] = result
                # Limit cache size
                if len(self._cache) > 1000:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
            
            # Update timing
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Rolling operation {operation} failed: {e}, using pandas fallback")
            return self._pandas_fallback_operation(data, operation, window, **kwargs)
    
    def _execute_rolling_optimizer_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Execute operation using VectorBT rolling optimizer."""
        if operation == 'mean':
            return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
        elif operation == 'std':
            return self.rolling_optimizer.rolling_std(data, window, **kwargs)
        elif operation == 'var':
            return self.rolling_optimizer.rolling_var(data, window, **kwargs)
        elif operation == 'min':
            return self.rolling_optimizer.rolling_min(data, window, **kwargs)
        elif operation == 'max':
            return self.rolling_optimizer.rolling_max(data, window, **kwargs)
        elif operation == 'sum':
            return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
        elif operation == 'skew':
            return self.rolling_optimizer.rolling_skew(data, window, **kwargs)
        elif operation == 'kurt':
            return self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
        elif operation == 'corr':
            other = kwargs.get('other')
            return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
        elif operation == 'cov':
            other = kwargs.get('other')
            return self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _execute_direct_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Execute operation using direct VectorBT calls."""
        if not VECTORBT_AVAILABLE:
            return self._pandas_fallback_operation(data, operation, window, **kwargs)
        
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
            elif operation == 'corr':
                other = kwargs.get('other')
                return rolling_corr(data, other, window=window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                return rolling_cov(data, other, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"Direct VectorBT operation failed: {e}")
            return self._pandas_fallback_operation(data, operation, window, **kwargs)
    
    def _pandas_fallback_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Fallback to pandas operations."""
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
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def batch_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform multiple rolling operations in batch for efficiency.
        
        Args:
            data: Input DataFrame
            operations: List of operation dictionaries with keys:
                - name: Feature name
                - column: Column to process
                - operation: Operation type
                - window: Window size
                - **kwargs: Additional parameters
                
        Returns:
            DataFrame with rolling features
        """
        if not self.config.enable_batch_processing:
            return self._sequential_rolling_operations(data, operations)
        
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['batch_operations'] += 1
        
        try:
            results = {}
            
            # Group operations by window for efficiency
            operations_by_window = {}
            for op in operations:
                window = op.get('window', 20)
                if window not in operations_by_window:
                    operations_by_window[window] = []
                operations_by_window[window].append(op)
            
            # Process each window group
            for window, window_ops in operations_by_window.items():
                for op in window_ops:
                    name = op['name']
                    column = op.get('column', 'close')
                    operation = op['operation']
                    op_kwargs = {k: v for k, v in op.items() if k not in ['name', 'column', 'operation', 'window']}
                    
                    if column in data.columns:
                        results[name] = self.rolling_operation(
                            data[column], operation, window, **op_kwargs
                        )
                    else:
                        self.logger.warning(f"Column {column} not found for operation {name}")
                        results[name] = pd.Series(np.nan, index=data.index)
            
            execution_time = time.time() - start_time
            self.performance_stats['total_execution_time'] += execution_time
            
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Batch rolling operations failed: {e}")
            return self._sequential_rolling_operations(data, operations)
    
    def _sequential_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process rolling operations sequentially."""
        results = {}
        
        for op in operations:
            name = op['name']
            column = op.get('column', 'close')
            operation = op['operation']
            window = op.get('window', 20)
            op_kwargs = {k: v for k, v in op.items() if k not in ['name', 'column', 'operation', 'window']}
            
            if column in data.columns:
                results[name] = self.rolling_operation(
                    data[column], operation, window, **op_kwargs
                )
            else:
                results[name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for VectorBT processing."""
        if not self.config.enable_memory_optimization:
            return data
        
        try:
            optimized_data = data.copy()
            
            # Optimize data types
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1
                elif optimized_data[column].dtype == 'int64':
                    if (optimized_data[column].min() >= np.iinfo(np.int32).min and
                        optimized_data[column].max() <= np.iinfo(np.int32).max):
                        optimized_data[column] = optimized_data[column].astype(np.int32)
                        self.performance_stats['memory_optimizations'] += 1
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return data
    
    def _generate_cache_key(self, data: pd.Series, operation: str, window: int, **kwargs) -> str:
        """Generate cache key for operation."""
        import hashlib
        
        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        
        return f"{operation}_{window}_{data_hash}_{params_hash}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['total_operations'] * 100
            )
            stats['rolling_operations_percentage'] = (
                stats['rolling_operations'] / stats['total_operations'] * 100
            )
            stats['batch_operations_percentage'] = (
                stats['batch_operations'] / stats['total_operations'] * 100
            )
            stats['average_execution_time'] = (
                stats['total_execution_time'] / stats['total_operations']
            )
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['rolling_operations_percentage'] = 0
            stats['batch_operations_percentage'] = 0
            stats['average_execution_time'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'rolling_operations': 0,
            'batch_operations': 0,
            'gpu_operations': 0,
            'memory_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0,
            'memory_savings': 0.0
        }
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.rolling_optimizer:
                self.rolling_optimizer.reset_stats()
            if self._cache:
                self._cache.clear()
            self.logger.info("🧹 Unified Vectorization Manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Global instance
_unified_vectorization_manager: Optional[UnifiedVectorizationManager] = None

def get_unified_vectorization_manager(config: Optional[UnifiedVectorizationConfig] = None) -> UnifiedVectorizationManager:
    """Get or create the global unified vectorization manager."""
    global _unified_vectorization_manager
    
    if _unified_vectorization_manager is None:
        _unified_vectorization_manager = UnifiedVectorizationManager(config)
    
    return _unified_vectorization_manager

def unified_rolling_operation(data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
    """Convenience function for unified rolling operations."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_operation(data, operation, window, **kwargs)

def unified_batch_rolling_operations(data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convenience function for unified batch rolling operations."""
    manager = get_unified_vectorization_manager()
    return manager.batch_rolling_operations(data, operations)

def unified_optimize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for unified DataFrame optimization."""
    manager = get_unified_vectorization_manager()
    return manager.optimize_dataframe(data)