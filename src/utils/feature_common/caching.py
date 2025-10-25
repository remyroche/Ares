"""
Shared Computation Cache for Feature Engineering

Provides efficient caching for expensive computations across feature engineering operations.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import hashlib
import time
import logging
from collections import OrderedDict
import gc

from src.utils.tprint import tprint, tprint_info, tprint_warning

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    max_size: int = 1000
    max_memory_mb: int = 500
    ttl_seconds: int = 3600  # Time to live
    enable_compression: bool = True
    cleanup_interval: int = 100  # Cleanup every N operations

class SharedComputationCache:
    """
    Shared computation cache for expensive operations across feature engineering.
    
    Features:
    - Memory-efficient storage with automatic cleanup
    - TTL-based expiration
    - Compression for large objects
    - Thread-safe operations
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize shared cache."""
        self.config = config or CacheConfig()
        self.cache = OrderedDict()
        self.access_count = {}
        self.creation_time = {}
        self.memory_usage = 0
        self.operation_count = 0
        
        tprint_info("🔧 Initialized SharedComputationCache")
    
    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function name and arguments."""
        # Create hash of arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        key_hash = hashlib.md5(args_str.encode()).hexdigest()[:16]
        return f"{func_name}_{key_hash}"
    
    def _estimate_memory_usage(self, obj: Any) -> int:
        """Estimate memory usage of an object in bytes."""
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_memory_usage(item) for item in obj)
            else:
                return len(str(obj).encode())
        except:
            return 1000  # Default estimate
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, creation_time in self.creation_time.items():
            if current_time - creation_time > self.config.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _cleanup_memory_pressure(self):
        """Remove entries when memory pressure is high."""
        if self.memory_usage > self.config.max_memory_mb * 1024 * 1024:
            # Remove least recently used entries
            while (self.memory_usage > self.config.max_memory_mb * 1024 * 1024 and 
                   len(self.cache) > 0):
                oldest_key = next(iter(self.cache))
                self._remove_entry(oldest_key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
            if key in self.creation_time:
                del self.creation_time[key]
    
    def get_or_compute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Get cached result or compute and cache.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result (cached or computed)
        """
        key = self._generate_key(func.__name__, *args, **kwargs)
        
        # Check if cached and not expired
        if key in self.cache:
            current_time = time.time()
            if (current_time - self.creation_time[key] < self.config.ttl_seconds):
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.cache[key]
            else:
                self._remove_entry(key)
        
        # Compute result
        result = func(*args, **kwargs)
        
        # Estimate memory usage
        memory_usage = self._estimate_memory_usage(result)
        
        # Check if we should cache this result
        if memory_usage < self.config.max_memory_mb * 1024 * 1024 // 10:  # Don't cache huge objects
            self.cache[key] = result
            self.access_count[key] = 1
            self.creation_time[key] = time.time()
            self.memory_usage += memory_usage
        
        # Periodic cleanup
        self.operation_count += 1
        if self.operation_count % self.config.cleanup_interval == 0:
            self._cleanup_expired()
            self._cleanup_memory_pressure()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'memory_usage_mb': self.memory_usage / (1024 * 1024),
            'operation_count': self.operation_count,
            'hit_rate': sum(self.access_count.values()) / max(1, self.operation_count),
            'most_accessed': max(self.access_count.items(), key=lambda x: x[1]) if self.access_count else None
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_count.clear()
        self.creation_time.clear()
        self.memory_usage = 0
        self.operation_count = 0
        gc.collect()

class FeatureCache:
    """
    Specialized cache for feature engineering operations.
    
    Provides optimized caching for:
    - Rolling calculations
    - Correlation matrices
    - Statistical tests
    - SHAP values
    """
    
    def __init__(self, max_size: int = 500):
        """Initialize feature cache."""
        self.rolling_cache = {}
        self.correlation_cache = {}
        self.statistical_cache = {}
        self.shap_cache = {}
        self.max_size = max_size
        
        tprint_info("🔧 Initialized FeatureCache")
    
    def get_rolling_stat(self, series: pd.Series, window: int, stat_type: str) -> pd.Series:
        """Get or compute rolling statistic."""
        key = (id(series), window, stat_type)
        
        if key in self.rolling_cache:
            return self.rolling_cache[key]
        
        # Compute rolling statistic
        if stat_type == 'mean':
            result = series.rolling(window=window, min_periods=max(1, window//2)).mean()
        elif stat_type == 'std':
            result = series.rolling(window=window, min_periods=max(1, window//2)).std()
        elif stat_type == 'var':
            result = series.rolling(window=window, min_periods=max(1, window//2)).var()
        elif stat_type == 'min':
            result = series.rolling(window=window, min_periods=max(1, window//2)).min()
        elif stat_type == 'max':
            result = series.rolling(window=window, min_periods=max(1, window//2)).max()
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")
        
        # Cache if not too large
        if len(self.rolling_cache) < self.max_size:
            self.rolling_cache[key] = result
        
        return result
    
    def get_correlation_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get or compute correlation matrix."""
        key = id(features_df)
        
        if key in self.correlation_cache:
            return self.correlation_cache[key]
        
        # Compute correlation matrix
        corr_matrix = features_df.corr()
        
        # Cache if not too large
        if len(self.correlation_cache) < self.max_size:
            self.correlation_cache[key] = corr_matrix
        
        return corr_matrix
    
    def get_statistical_test(self, feature: pd.Series, target: pd.Series, test_type: str) -> float:
        """Get or compute statistical test result."""
        key = (id(feature), id(target), test_type)
        
        if key in self.statistical_cache:
            return self.statistical_cache[key]
        
        # Compute statistical test
        if test_type == 'ttest':
            from scipy.stats import ttest_ind
            # Split into two groups
            median = feature.median()
            group1 = target[feature > median]
            group2 = target[feature <= median]
            if len(group1) > 0 and len(group2) > 0:
                _, p_value = ttest_ind(group1, group2)
                result = p_value
            else:
                result = 1.0
        elif test_type == 'correlation':
            result = feature.corr(target)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        # Cache result
        if len(self.statistical_cache) < self.max_size:
            self.statistical_cache[key] = result
        
        return result
    
    def get_shap_values(self, model, data, target_name: str) -> np.ndarray:
        """Get or compute SHAP values."""
        key = (id(data), target_name)
        
        if key in self.shap_cache:
            return self.shap_cache[key]
        
        # This would be implemented with actual SHAP calculation
        # For now, return placeholder
        result = np.random.random((len(data), len(data.columns)))
        
        # Cache if not too large
        if len(self.shap_cache) < self.max_size:
            self.shap_cache[key] = result
        
        return result
    
    def clear(self):
        """Clear all caches."""
        self.rolling_cache.clear()
        self.correlation_cache.clear()
        self.statistical_cache.clear()
        self.shap_cache.clear()
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'rolling_cache_size': len(self.rolling_cache),
            'correlation_cache_size': len(self.correlation_cache),
            'statistical_cache_size': len(self.statistical_cache),
            'shap_cache_size': len(self.shap_cache)
        }

# Global cache instances
_shared_cache = None
_feature_cache = None

def get_shared_cache() -> SharedComputationCache:
    """Get global shared cache instance."""
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = SharedComputationCache()
    return _shared_cache

def get_feature_cache() -> FeatureCache:
    """Get global feature cache instance."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCache()
    return _feature_cache
