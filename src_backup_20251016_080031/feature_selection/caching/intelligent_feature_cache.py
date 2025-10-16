"""
Intelligent Feature Selection Cache

This module provides intelligent caching for feature selection operations using
hardware optimization tools and the unified cache system.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Import hardware optimization tools
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
from src.utils.unified_cache import UnifiedCache, cached
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error, tprint_performance

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for feature selection caching."""
    # Cache settings
    enable_caching: bool = True
    cache_dir: str = "data_cache/feature_selection"
    max_memory_mb: int = 1024
    default_ttl_seconds: int = 3600  # 1 hour
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_limit_gb: float = 8.0
    enable_compression: bool = True
    
    # Cache invalidation
    enable_smart_invalidation: bool = True
    invalidation_threshold: float = 0.8  # Invalidate when 80% full
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_cache_stats: bool = True

class IntelligentFeatureCache:
    """Intelligent cache for feature selection operations with hardware optimization."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the intelligent feature cache."""
        self.config = config or CacheConfig()
        self.logger = logger.getChild('IntelligentFeatureCache')
        
        # Initialize hardware manager
        if self.config.enable_hardware_optimization:
            hw_config = HardwareConfig(
                memory_limit_gb=self.config.memory_limit_gb,
                enable_compression=self.config.enable_compression,
                memory_optimization_level='balanced'
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
            self.memory_optimizer = M1MemoryOptimizer(self.config.memory_limit_gb)
        else:
            self.hardware_manager = None
            self.memory_optimizer = None
        
        # Initialize unified cache
        self.cache = UnifiedCache(
            cache_dir=self.config.cache_dir,
            max_memory_mb=self.config.max_memory_mb,
            enable_disk=True,
            enable_compression=self.config.enable_compression,
            default_ttl_seconds=self.config.default_ttl_seconds,
            namespace="feature_selection"
        )
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'cache_size_mb': 0
        }
        
        tprint_success("🚀 IntelligentFeatureCache initialized with hardware optimization")
    
    def _generate_cache_key(self, 
                           X_hash: str, 
                           y_hash: str, 
                           method: str, 
                           params: Dict[str, Any]) -> str:
        """Generate a unique cache key for feature selection operation."""
        try:
            # Create deterministic key from parameters
            key_data = {
                'X_hash': X_hash,
                'y_hash': y_hash,
                'method': method,
                'params': sorted(params.items()) if params else {}
            }
            
            key_string = str(key_data)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Cache key generation failed: {e}")
            # Fallback to simple hash
            return hashlib.md5(f"{X_hash}_{y_hash}_{method}".encode()).hexdigest()
    
    def _compute_data_hash(self, data: Union[np.ndarray, pd.DataFrame]) -> str:
        """Compute hash for data to detect changes."""
        try:
            if isinstance(data, pd.DataFrame):
                # Hash shape, column names, and sample of data
                sample_data = data.head(100).values
                hash_data = f"{data.shape}_{list(data.columns)}_{sample_data.tobytes()}"
            else:
                # Hash shape and sample of data
                sample_data = data[:100] if len(data) > 100 else data
                hash_data = f"{data.shape}_{sample_data.tobytes()}"
            
            return hashlib.md5(hash_data.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Data hash computation failed: {e}")
            return hashlib.md5(str(data.shape).encode()).hexdigest()
    
    def get_cached_result(self, 
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         method: str,
                         params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached feature selection result."""
        if not self.config.enable_caching:
            return None
        
        try:
            # Generate cache key
            X_hash = self._compute_data_hash(X)
            y_hash = self._compute_data_hash(y)
            cache_key = self._generate_cache_key(X_hash, y_hash, method, params)
            
            # Check cache
            result = self.cache.get(cache_key)
            
            if result is not None:
                self.stats['hits'] += 1
                tprint_performance(f"💾 Cache HIT for {method} selection")
                return result
            else:
                self.stats['misses'] += 1
                tprint_performance(f"💾 Cache MISS for {method} selection")
                return None
                
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
            self.stats['misses'] += 1
            return None
    
    def cache_result(self, 
                    X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series],
                    method: str,
                    params: Dict[str, Any],
                    result: Dict[str, Any]) -> None:
        """Cache feature selection result."""
        if not self.config.enable_caching:
            return
        
        try:
            # Generate cache key
            X_hash = self._compute_data_hash(X)
            y_hash = self._compute_data_hash(y)
            cache_key = self._generate_cache_key(X_hash, y_hash, method, params)
            
            # Add metadata
            result_with_metadata = {
                **result,
                'cached_at': time.time(),
                'method': method,
                'params': params,
                'data_shape': X.shape if hasattr(X, 'shape') else len(X)
            }
            
            # Cache with TTL
            ttl = self.config.default_ttl_seconds
            self.cache.set(cache_key, result_with_metadata, ttl=ttl)
            
            tprint_success(f"💾 Cached {method} selection result")
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            if pattern:
                # Invalidate specific pattern
                cleared = self.cache.clear_namespace()
            else:
                # Clear all cache
                cleared = self.cache.clear_namespace()
            
            tprint_success(f"🗑️ Invalidated {cleared} cache entries")
            return cleared
            
        except Exception as e:
            self.logger.error(f"Cache invalidation failed: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_stats = self.cache.get_stats()
            
            stats = {
                **self.stats,
                'cache_hits': cache_stats.get('hits', 0),
                'cache_misses': cache_stats.get('misses', 0),
                'cache_evictions': cache_stats.get('evictions', 0),
                'hit_rate': self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses']),
                'memory_usage_mb': cache_stats.get('memory_usage_mb', 0)
            }
            
            if self.config.log_cache_stats:
                tprint_performance(f"📊 Cache Stats: {stats['hit_rate']:.2%} hit rate, "
                                 f"{stats['memory_usage_mb']:.1f}MB used")
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Cache stats retrieval failed: {e}")
            return self.stats
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage using hardware tools."""
        if not self.memory_optimizer:
            return {'optimized': False, 'reason': 'Memory optimizer not available'}
        
        try:
            # Get memory pressure
            memory_pressure = self.memory_optimizer.get_memory_pressure()
            
            if memory_pressure > self.config.invalidation_threshold:
                # Clear old cache entries
                cleared = self.invalidate_cache()
                
                # Apply memory optimizations
                optimization_result = self.memory_optimizer.optimize_memory()
                
                tprint_success(f"🧠 Memory optimized: {cleared} entries cleared, "
                             f"pressure: {memory_pressure:.2f}")
                
                return {
                    'optimized': True,
                    'cleared_entries': cleared,
                    'memory_pressure_before': memory_pressure,
                    'optimization_result': optimization_result
                }
            else:
                return {
                    'optimized': False,
                    'reason': f'Memory pressure ({memory_pressure:.2f}) below threshold'
                }
                
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return {'optimized': False, 'reason': str(e)}

class FeatureSelectionCacheManager:
    """Manager for feature selection caching with hardware optimization."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the cache manager."""
        self.config = config or CacheConfig()
        self.cache = IntelligentFeatureCache(self.config)
        
        # Start hardware monitoring if enabled
        if self.config.enable_hardware_optimization and self.cache.memory_optimizer:
            self.cache.memory_optimizer.start_monitoring()
            tprint_success("🔧 Hardware monitoring started")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.cache.memory_optimizer:
            self.cache.memory_optimizer.stop_monitoring()
        tprint_success("🔧 Hardware monitoring stopped")
    
    def get_cached_selection(self, 
                           X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           method: str,
                           params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached feature selection result."""
        return self.cache.get_cached_result(X, y, method, params)
    
    def cache_selection_result(self, 
                             X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             method: str,
                             params: Dict[str, Any],
                             result: Dict[str, Any]) -> None:
        """Cache feature selection result."""
        self.cache.cache_result(X, y, method, params, result)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.cache.get_cache_stats()

def cached_feature_selection(cache_manager: FeatureSelectionCacheManager):
    """Decorator for caching feature selection operations."""
    def decorator(func: Callable) -> Callable:
        def wrapper(X, y, method='comprehensive', **kwargs):
            # Check cache first
            cached_result = cache_manager.get_cached_selection(X, y, method, kwargs)
            if cached_result:
                return cached_result
            
            # Execute function
            result = func(X, y, method=method, **kwargs)
            
            # Cache result
            cache_manager.cache_selection_result(X, y, method, kwargs, result)
            
            return result
        
        return wrapper
    return decorator

def create_feature_cache(config: Optional[CacheConfig] = None) -> FeatureSelectionCacheManager:
    """Create a feature selection cache manager."""
    return FeatureSelectionCacheManager(config)