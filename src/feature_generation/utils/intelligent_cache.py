"""
Intelligent Result Caching with TTL

This module provides intelligent caching with time-to-live (TTL)
and automatic cleanup for feature generation results.
"""

import logging
import time
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import gc

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    def access(self):
        """Record access to the cache entry."""
        self.access_count += 1
        self.last_access = time.time()

class IntelligentCache:
    """
    Intelligent cache with TTL, LRU eviction, and automatic cleanup.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 500.0,
                 default_ttl_seconds: Optional[int] = 3600,
                 cleanup_interval: int = 300):  # 5 minutes
        """Initialize the intelligent cache."""
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self.logger = logger.getChild('IntelligentCache')
        
        # Thread-safe cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_entries': 0,
            'total_size_bytes': 0,
            'last_cleanup': time.time()
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        self.logger.info(f"🚀 IntelligentCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._cache[key]
                    self.stats['expired_entries'] += 1
                    self.stats['misses'] += 1
                    return None
                
                # Record access and move to end (LRU)
                entry.access()
                self._cache.move_to_end(key)
                self.stats['hits'] += 1
                
                return entry.data
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = self._estimate_size(value)
            except:
                size_bytes = 1024  # Default estimate
            
            # Check if we need to evict
            self._check_and_evict(size_bytes)
            
            # Create cache entry
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            entry = CacheEntry(
                data=value,
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )
            
            # Store entry
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self.stats['total_size_bytes'] += size_bytes
            
            return True
    
    def _check_and_evict(self, new_size_bytes: int):
        """Check if eviction is needed and evict if necessary."""
        # Check size limit
        while (len(self._cache) >= self.max_size or 
               self.stats['total_size_bytes'] + new_size_bytes > self.max_memory_bytes):
            
            if not self._cache:
                break
            
            # Remove least recently used
            key, entry = self._cache.popitem(last=False)
            self.stats['total_size_bytes'] -= entry.size_bytes
            self.stats['evictions'] += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            if hasattr(value, 'memory_usage'):
                # Pandas DataFrame/Series
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in value.items())
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self.stats['total_size_bytes'] -= entry.size_bytes
                self.stats['expired_entries'] += 1
            
            self.stats['last_cleanup'] = time.time()
            
            if expired_keys:
                self.logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired entries")
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a deterministic key from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats['total_size_bytes'] = 0
            self.logger.info("🧹 Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self.stats['hits'] / 
                       max(1, self.stats['hits'] + self.stats['misses']))
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'cache_size': len(self._cache),
                'memory_usage_mb': self.stats['total_size_bytes'] / 1024 / 1024,
                'memory_usage_percent': (self.stats['total_size_bytes'] / 
                                       max(1, self.max_memory_bytes)) * 100
            }

class FeatureResultCache:
    """
    Specialized cache for feature generation results.
    """
    
    def __init__(self, 
                 max_size: int = 500,
                 max_memory_mb: float = 1000.0,
                 default_ttl_seconds: int = 1800):  # 30 minutes
        """Initialize feature result cache."""
        self.cache = IntelligentCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            default_ttl_seconds=default_ttl_seconds
        )
        
        self.logger = logger.getChild('FeatureResultCache')
        
        # Feature-specific TTL settings
        self.feature_ttls = {
            'rolling_features': 3600,  # 1 hour
            'technical_indicators': 1800,  # 30 minutes
            'statistical_features': 7200,  # 2 hours
            'volume_features': 1800,  # 30 minutes
            'volatility_features': 3600  # 1 hour
        }
        
        self.logger.info("🚀 FeatureResultCache initialized")
    
    def get_feature_result(self, 
                          data_hash: str, 
                          feature_specs: Dict[str, Any],
                          **kwargs) -> Optional[Any]:
        """Get cached feature result."""
        key = self._generate_feature_key(data_hash, feature_specs, **kwargs)
        return self.cache.get(key)
    
    def put_feature_result(self, 
                          data_hash: str, 
                          feature_specs: Dict[str, Any],
                          result: Any,
                          feature_type: str = None,
                          **kwargs) -> bool:
        """Put feature result in cache."""
        key = self._generate_feature_key(data_hash, feature_specs, **kwargs)
        
        # Determine TTL based on feature type
        ttl = None
        if feature_type in self.feature_ttls:
            ttl = self.feature_ttls[feature_type]
        
        return self.cache.put(key, result, ttl)
    
    def _generate_feature_key(self, 
                             data_hash: str, 
                             feature_specs: Dict[str, Any],
                             **kwargs) -> str:
        """Generate cache key for feature result."""
        return self.cache.generate_key(
            data_hash,
            feature_specs,
            **kwargs
        )
    
    def calculate_data_hash(self, data: Any) -> str:
        """Calculate hash of input data for caching."""
        try:
            if hasattr(data, 'shape'):
                # For DataFrames/arrays, use shape and sample of data
                shape_info = str(data.shape)
                sample_info = str(data.head(5).values.tobytes()) if hasattr(data, 'head') else str(data.flatten()[:100])
                return hashlib.md5((shape_info + sample_info).encode()).hexdigest()
            else:
                return hashlib.md5(str(data).encode()).hexdigest()
        except:
            return hashlib.md5(str(id(data)).encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.cache.clear()
        self.logger.info("🧹 Feature result cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

# Global instances
_feature_cache: Optional[FeatureResultCache] = None
_general_cache: Optional[IntelligentCache] = None

def get_feature_cache() -> FeatureResultCache:
    """Get the global feature result cache."""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureResultCache()
    return _feature_cache

def get_general_cache() -> IntelligentCache:
    """Get the global general cache."""
    global _general_cache
    if _general_cache is None:
        _general_cache = IntelligentCache()
    return _general_cache
