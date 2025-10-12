"""
Optimized Feature Generator Base Class

This module provides an optimized base class for feature generators that leverages
VectorBTRollingOptimizer and UnifiedVectorizationManager for maximum performance.

Key Features:
- Batch rolling operations using VectorBTRollingOptimizer
- UnifiedVectorizationManager integration for cross-category features
- Memory optimization with data type optimization
- Smart caching for frequently computed operations
- Performance monitoring and statistics
- Cross-category feature generation optimization
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
import hashlib

from .feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from .vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, UnifiedVectorizationManager, 
        OperationType, OptimizationStrategy, OperationConfig
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None
    OperationConfig = None

logger = logging.getLogger(__name__)


class OptimizedFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """
    Optimized feature generator base class with comprehensive VectorBT optimization.
    
    This class provides:
    - Batch rolling operations using VectorBTRollingOptimizer
    - UnifiedVectorizationManager integration
    - Memory optimization
    - Smart caching
    - Performance monitoring
    - Cross-category feature generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimized feature generator."""
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT components
        self.rolling_optimizer = None
        self.unified_manager = None
        
        # Initialize VectorBT Rolling Optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=False, 
                    enable_parallel=True,
                    memory_efficient=True,
                    chunk_size=1000
                )
                self.logger.info("✅ VectorBTRollingOptimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBTRollingOptimizer initialization failed: {e}")
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            try:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ UnifiedVectorizationManager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager initialization failed: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'memory_savings': 0.0
        }
        
        # Cache for computed results
        self._result_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 1000
        
        self.logger.info("✅ OptimizedFeatureGenerator initialized with comprehensive VectorBT optimization")
    
    def batch_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform multiple rolling operations in batch for better performance.
        
        Args:
            data: Input DataFrame
            operations: List of operation configurations
            
        Returns:
            DataFrame with results of all operations
        """
        if not self.rolling_optimizer:
            return self._fallback_batch_rolling_operations(data, operations)
        
        start_time = time.time()
        self.performance_stats['batch_operations'] += 1
        self.performance_stats['total_operations'] += 1
        
        try:
            results = {}
            
            for op_config in operations:
                name = op_config['name']
                operation = op_config['operation']
                window = op_config['window']
                column = op_config.get('column', 'close')
                
                if column not in data.columns:
                    self.logger.warning(f"Column {column} not found in data, skipping {name}")
                    continue
                
                # Check cache first
                cache_key = self._generate_cache_key(data[column], operation, window, op_config.get('kwargs', {}))
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result is not None:
                    results[name] = cached_result
                    self.performance_stats['cache_hits'] += 1
                else:
                    # Perform rolling operation
                    result = self._perform_rolling_operation(
                        data[column], operation, window, op_config.get('kwargs', {})
                    )
                    results[name] = result
                    self.performance_stats['cache_misses'] += 1
                    
                    # Cache result
                    self._put_in_cache(cache_key, result)
            
            self.performance_stats['total_time'] += time.time() - start_time
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            self.logger.warning(f"Batch rolling operations failed: {e}, using fallback")
            return self._fallback_batch_rolling_operations(data, operations)
    
    def _perform_rolling_operation(self, data: pd.Series, operation: str, window: int, kwargs: Dict[str, Any]) -> pd.Series:
        """Perform a single rolling operation using VectorBTRollingOptimizer."""
        try:
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
            elif operation == 'apply':
                func = kwargs.get('func')
                return self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
            elif operation == 'ewm':
                alpha = kwargs.get('alpha')
                span = kwargs.get('span')
                return self.rolling_optimizer.rolling_ewm(data, window, alpha=alpha, span=span, **kwargs)
            else:
                raise ValueError(f"Unsupported rolling operation: {operation}")
                
        except Exception as e:
            self.logger.warning(f"VectorBT rolling operation {operation} failed: {e}, using pandas fallback")
            return self._pandas_fallback_rolling(data, operation, window, kwargs)
    
    def generate_cross_category_features(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate features across multiple categories using UnifiedVectorizationManager.
        
        Args:
            data: Input DataFrame
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated features
        """
        if not self.unified_manager:
            return self._fallback_cross_category_features(data, feature_configs)
        
        start_time = time.time()
        self.performance_stats['batch_operations'] += 1
        self.performance_stats['total_operations'] += 1
        
        try:
            # Optimize DataFrame for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Use UnifiedVectorizationManager for batch processing
            features = self.unified_manager.batch_process_features(optimized_data, feature_configs)
            
            self.performance_stats['total_time'] += time.time() - start_time
            return features
            
        except Exception as e:
            self.logger.warning(f"Cross-category feature generation failed: {e}, using fallback")
            return self._fallback_cross_category_features(data, feature_configs)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for VectorBT processing with memory efficiency.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        if not self.unified_manager:
            return self._optimize_data_types_manual(data)
        
        try:
            # Use UnifiedVectorizationManager's optimization
            optimized_data = self.unified_manager.optimize_dataframe(data)
            self.performance_stats['memory_optimizations'] += 1
            return optimized_data
        except Exception as e:
            self.logger.warning(f"UnifiedVectorizationManager optimization failed: {e}, using manual optimization")
            return self._optimize_data_types_manual(data)
    
    def _optimize_data_types_manual(self, data: pd.DataFrame) -> pd.DataFrame:
        """Manual data type optimization as fallback."""
        optimized_data = data.copy()
        
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
    
    def get_cached_rolling_result(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """
        Get cached rolling result or compute and cache it.
        
        Args:
            data: Input data
            operation: Rolling operation
            window: Window size
            **kwargs: Additional parameters
            
        Returns:
            Rolling operation result
        """
        cache_key = self._generate_cache_key(data, operation, window, kwargs)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            self.performance_stats['cache_hits'] += 1
            return cached_result
        
        # Compute result
        result = self._perform_rolling_operation(data, operation, window, kwargs)
        
        # Cache result
        self._put_in_cache(cache_key, result)
        self.performance_stats['cache_misses'] += 1
        
        return result
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        initial_stats = self.performance_stats.copy()
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate performance metrics
            final_stats = self.performance_stats
            operations_delta = final_stats['total_operations'] - initial_stats['total_operations']
            vectorbt_delta = final_stats['vectorbt_operations'] - initial_stats['vectorbt_operations']
            cache_hits_delta = final_stats['cache_hits'] - initial_stats['cache_hits']
            cache_misses_delta = final_stats['cache_misses'] - initial_stats['cache_misses']
            
            self.logger.info(f"Performance monitoring - {operation_name}:")
            self.logger.info(f"  - Execution time: {execution_time:.3f}s")
            self.logger.info(f"  - Operations: {operations_delta}")
            self.logger.info(f"  - VectorBT operations: {vectorbt_delta}")
            self.logger.info(f"  - Cache hits: {cache_hits_delta}")
            self.logger.info(f"  - Cache misses: {cache_misses_delta}")
            
            if cache_hits_delta + cache_misses_delta > 0:
                cache_hit_rate = (cache_hits_delta / (cache_hits_delta + cache_misses_delta)) * 100
                self.logger.info(f"  - Cache hit rate: {cache_hit_rate:.2f}%")
    
    def generate_features_with_monitoring(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate features with comprehensive performance monitoring."""
        with self.performance_monitoring("feature_generation"):
            return self.generate_cross_category_features(data, feature_configs)
    
    def _generate_cache_key(self, data: pd.Series, operation: str, window: int, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for operation."""
        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        
        return f"{operation}_{window}_{data_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.Series]:
        """Get result from cache."""
        if not self._cache_enabled:
            return None
        
        try:
            if cache_key in self._result_cache:
                return self._result_cache[cache_key]
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: pd.Series):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        try:
            # Limit cache size
            if len(self._result_cache) >= self._max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[cache_key] = result
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def _fallback_batch_rolling_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback batch rolling operations using pandas."""
        results = {}
        
        for op_config in operations:
            name = op_config['name']
            operation = op_config['operation']
            window = op_config['window']
            column = op_config.get('column', 'close')
            
            if column not in data.columns:
                continue
            
            result = self._pandas_fallback_rolling(data[column], operation, window, op_config.get('kwargs', {}))
            results[name] = result
        
        return pd.DataFrame(results, index=data.index)
    
    def _pandas_fallback_rolling(self, data: pd.Series, operation: str, window: int, kwargs: Dict[str, Any]) -> pd.Series:
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
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        elif operation == 'ewm':
            alpha = kwargs.get('alpha')
            span = kwargs.get('span')
            if alpha is not None:
                return data.ewm(alpha=alpha, **kwargs).mean()
            elif span is not None:
                return data.ewm(span=span, **kwargs).mean()
            else:
                return data.ewm(span=window, **kwargs).mean()
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _fallback_cross_category_features(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback cross-category feature generation."""
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
                        results[feature_name] = self._pandas_fallback_rolling(
                            data[column], operation, window, params
                        )
                
                elif feature_type == 'scaling':
                    method = params.get('method', 'zscore')
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        if method == 'zscore':
                            results[feature_name] = (data[column] - data[column].mean()) / data[column].std()
                        elif method == 'minmax':
                            results[feature_name] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
                
                elif feature_type == 'custom':
                    func = params.get('function')
                    if callable(func):
                        results[feature_name] = func(data, **params)
                
            except Exception as e:
                self.logger.warning(f"Feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add rolling optimizer stats if available
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats.update(rolling_stats)
        
        # Add unified manager stats if available
        if self.unified_manager:
            unified_stats = self.unified_manager.get_performance_stats()
            stats.update(unified_stats)
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            stats['rolling_usage_rate'] = stats['rolling_operations'] / stats['total_operations']
            stats['scaling_usage_rate'] = stats['scaling_operations'] / stats['total_operations']
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['average_operation_time'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['gpu_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
            stats['rolling_usage_rate'] = 0
            stats['scaling_usage_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'memory_savings': 0.0
        }
        
        if self.rolling_optimizer:
            self.rolling_optimizer.reset_stats()
        
        if self.unified_manager:
            self.unified_manager.reset_stats()
        
        self._result_cache.clear()