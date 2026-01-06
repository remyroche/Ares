"""
Memory-Efficient Caching System

Implements smart caching with LRU eviction, size limits,
and memory monitoring for enhanced causal framework.
"""

import os
import sys
import time
import pickle
import hashlib
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import lru_cache
import threading
import weakref

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class MemoryEfficientCache:
    """
    Memory-efficient caching system with LRU eviction and size limits.
    
    Provides intelligent caching for enhanced causal framework components
    with automatic memory management and cleanup.
    """
    
    def __init__(self, max_cache_size_mb: int = 500, max_items: int = 1000, verbose: bool = True):
        """
        Initialize Memory-Efficient Cache.
        
        Args:
            max_cache_size_mb: Maximum cache size in MB
            max_items: Maximum number of cached items
            verbose: Whether to print progress information
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.max_items = max_items
        self.verbose = verbose
        
        # Cache storage
        self.cache_ = {}
        self.access_order_ = []
        self.item_sizes_ = {}
        
        # Memory tracking
        self.current_size_mb = 0.0
        self.memory_monitor_enabled = True
        
        # Statistics
        self.cache_hits_ = 0
        self.cache_misses_ = 0
        self.evictions_ = 0
        self.total_requests_ = 0
        
        # Thread safety
        self.lock_ = threading.RLock()
        
        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        
        if self.verbose:
            tprint_info(f"🧠 Memory Cache: Initialized (max_size: {max_cache_size_mb}MB, max_items: {max_items})")
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            # Use pickle for size estimation
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback to sys.getsizeof (less accurate)
            return sys.getsizeof(obj)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _should_evict_for_memory(self) -> bool:
        """Check if eviction is needed for memory constraints."""
        if not self.memory_monitor_enabled:
            return False
        
        current_memory = self._get_memory_usage()
        
        # Evict if we're using more than 80% of available memory or cache limit
        system_memory = psutil.virtual_memory()
        memory_pressure = current_memory / (system_memory.total / 1024 / 1024)
        
        return (self.current_size_mb > self.max_cache_size_mb or 
                len(self.cache_) > self.max_items or
                memory_pressure > 0.8)
    
    def _evict_lru_items(self, count: int = 1) -> int:
        """Evict LRU items from cache."""
        evicted_count = 0
        
        with self.lock_:
            for _ in range(count):
                if not self.access_order_:
                    break
                
                # Remove oldest item
                oldest_key = self.access_order_.pop(0)
                
                if oldest_key in self.cache_:
                    # Update size tracking
                    item_size = self.item_sizes_.get(oldest_key, 0)
                    self.current_size_mb -= item_size / 1024 / 1024
                    
                    # Remove from cache
                    del self.cache_[oldest_key]
                    del self.item_sizes_[oldest_key]
                    
                    evicted_count += 1
                    self.evictions_ += 1
        
        return evicted_count
    
    def _generate_cache_key(self, key: Any) -> str:
        """Generate a cache key from any hashable object."""
        if isinstance(key, str):
            return key
        else:
            # Create hash for complex objects
            key_str = str(key) if hasattr(key, '__str__') else repr(key)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        cache_key = self._generate_cache_key(key)
        
        with self.lock_:
            self.total_requests_ += 1
            
            if cache_key in self.cache_:
                # Cache hit
                self.cache_hits_ += 1
                
                # Update access order
                self.access_order_.remove(cache_key)
                self.access_order_.append(cache_key)
                
                if self.verbose and self.cache_hits_ % 100 == 0:
                    hit_rate = self.cache_hits_ / self.total_requests_
                    tprint_info(f"🎯 Cache Stats: {hit_rate:.2%} hit rate, {len(self.cache_)} items, {self.current_size_mb:.1f}MB")
                
                return self.cache_[cache_key]
            else:
                # Cache miss
                self.cache_misses_ += 1
                return None
    
    def put(self, key: Any, value: Any) -> bool:
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if item was cached, False if evicted
        """
        cache_key = self._generate_cache_key(key)
        
        # Estimate item size
        item_size = self._get_object_size(value)
        item_size_mb = item_size / 1024 / 1024
        
        # Check if item is too large
        if item_size_mb > self.max_cache_size_mb * 0.5:  # Don't cache items > 50% of cache size
            if self.verbose:
                tprint_warning(f"⚠️ Item too large for cache: {item_size_mb:.1f}MB")
            return False
        
        with self.lock_:
            # Evict items if necessary
            while self._should_evict_for_memory():
                evicted = self._evict_lru_items(1)
                if evicted == 0:
                    break
            
            # Remove existing item if present
            if cache_key in self.cache_:
                old_size = self.item_sizes_.get(cache_key, 0)
                self.current_size_mb -= old_size / 1024 / 1024
                self.access_order_.remove(cache_key)
            
            # Add new item
            self.cache_[cache_key] = value
            self.access_order_.append(cache_key)
            self.item_sizes_[cache_key] = item_size
            self.current_size_mb += item_size_mb
            
            return True
    
    def remove(self, key: Any) -> bool:
        """
        Remove item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed, False if not found
        """
        cache_key = self._generate_cache_key(key)
        
        with self.lock_:
            if cache_key in self.cache_:
                # Update size tracking
                item_size = self.item_sizes_.get(cache_key, 0)
                self.current_size_mb -= item_size / 1024 / 1024
                
                # Remove from cache
                del self.cache_[cache_key]
                del self.item_sizes_[cache_key]
                self.access_order_.remove(cache_key)
                
                return True
            
            return False
    
    def clear(self):
        """Clear all cached items."""
        with self.lock_:
            self.cache_.clear()
            self.access_order_.clear()
            self.item_sizes_.clear()
            self.current_size_mb = 0.0
        
        if self.verbose:
            tprint_info("🗑️ Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock_:
            total_requests = self.cache_hits_ + self.cache_misses_
            hit_rate = self.cache_hits_ / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_hits': self.cache_hits_,
                'cache_misses': self.cache_misses_,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'current_items': len(self.cache_),
                'max_items': self.max_items,
                'current_size_mb': self.current_size_mb,
                'max_size_mb': self.max_cache_size_mb,
                'evictions': self.evictions_,
                'memory_usage_mb': self._get_memory_usage(),
                'memory_pressure': self.current_size_mb / self.max_cache_size_mb
            }
    
    def optimize_cache(self):
        """Optimize cache by evicting least useful items."""
        with self.lock_:
            original_size = len(self.cache_)
            
            # Evict items until we're under limits
            while self._should_evict_for_memory():
                evicted = self._evict_lru_items(5)  # Evict in batches
                if evicted == 0:
                    break
            
            evicted_count = original_size - len(self.cache_)
            
            if self.verbose and evicted_count > 0:
                tprint_info(f"🧹 Cache optimization: evicted {evicted_count} items")
    
    def set_memory_limit(self, max_size_mb: int):
        """Update memory limit and optimize if necessary."""
        self.max_cache_size_mb = max_size_mb
        
        if self.verbose:
            tprint_info(f"📊 Memory limit updated: {max_size_mb}MB")
        
        # Optimize cache if over new limit
        if self.current_size_mb > max_size_mb:
            self.optimize_cache()


class CacheManager:
    """
    Global cache manager for enhanced causal framework.
    
    Manages multiple specialized caches with coordinated memory management.
    """
    
    def __init__(self, max_total_memory_mb: int = 1000, verbose: bool = True):
        """
        Initialize Cache Manager.
        
        Args:
            max_total_memory_mb: Maximum total memory for all caches
            verbose: Whether to print progress information
        """
        self.max_total_memory_mb = max_total_memory_mb
        self.verbose = verbose
        
        # Specialized caches
        self.caches_ = {}
        
        # Create default caches
        self.create_cache('mdi_features', max_size_mb=200, max_items=500)
        self.create_cache('causal_graphs', max_size_mb=300, max_items=100)
        self.create_cache('sem_models', max_size_mb=200, max_items=50)
        self.create_cache('quality_metrics', max_size_mb=100, max_items=200)
        
        if self.verbose:
            tprint_success(f"🧠 Cache Manager: Initialized with {len(self.caches_)} specialized caches")
    
    def create_cache(self, name: str, max_size_mb: int = 100, max_items: int = 100) -> MemoryEfficientCache:
        """
        Create a specialized cache.
        
        Args:
            name: Cache name
            max_size_mb: Maximum size in MB
            max_items: Maximum number of items
            
        Returns:
            Created cache instance
        """
        cache = MemoryEfficientCache(
            max_cache_size_mb=max_size_mb,
            max_items=max_items,
            verbose=self.verbose
        )
        
        self.caches_[name] = cache
        
        if self.verbose:
            tprint_info(f"   📦 Created cache '{name}': {max_size_mb}MB, {max_items} items")
        
        return cache
    
    def get_cache(self, name: str) -> Optional[MemoryEfficientCache]:
        """Get a specialized cache by name."""
        return self.caches_.get(name)
    
    def optimize_all_caches(self):
        """Optimize all caches."""
        if self.verbose:
            tprint_info("🧹 Optimizing all caches...")
        
        for name, cache in self.caches_.items():
            cache.optimize_cache()
        
        if self.verbose:
            tprint_success("✅ All caches optimized")
    
    def get_total_statistics(self) -> Dict[str, Any]:
        """Get combined statistics for all caches."""
        total_stats = {
            'total_hits': 0,
            'total_misses': 0,
            'total_requests': 0,
            'total_items': 0,
            'total_size_mb': 0.0,
            'total_evictions': 0,
            'caches': {}
        }
        
        for name, cache in self.caches_.items():
            stats = cache.get_statistics()
            
            # Add to totals
            total_stats['total_hits'] += stats['cache_hits']
            total_stats['total_misses'] += stats['cache_misses']
            total_stats['total_requests'] += stats['total_requests']
            total_stats['total_items'] += stats['current_items']
            total_stats['total_size_mb'] += stats['current_size_mb']
            total_stats['total_evictions'] += stats['evictions']
            
            # Add per-cache stats
            total_stats['caches'][name] = stats
        
        # Calculate overall hit rate
        if total_stats['total_requests'] > 0:
            total_stats['overall_hit_rate'] = total_stats['total_hits'] / total_stats['total_requests']
        else:
            total_stats['overall_hit_rate'] = 0.0
        
        return total_stats
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache in self.caches_.values():
            cache.clear()
        
        if self.verbose:
            tprint_info("🗑️ All caches cleared")


# Global cache manager instance
_global_cache_manager = None

def get_cache_manager(max_total_memory_mb: int = 1000, verbose: bool = True) -> CacheManager:
    """Get or create the global cache manager."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(max_total_memory_mb, verbose)
    
    return _global_cache_manager


# Convenience decorators
def cached_result(cache_name: str, ttl_seconds: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cache_name)
            
            if cache is None:
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = (func.__name__, args, frozenset(kwargs.items()))
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator
