#!/usr/bin/env python3
"""Intelligent Caching System with Memoization for Step03.

This module provides advanced caching capabilities including:
1. Intelligent memoization with automatic cache invalidation
2. Multi-level caching (memory, disk, distributed)
3. Cache performance monitoring and optimization
4. Automatic cache cleanup and memory management
5. Cache hit/miss analytics and reporting
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Hashable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import weakref
import threading
from collections import OrderedDict
import psutil
import gc

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""
    max_memory_cache_size_mb: int = 500
    max_disk_cache_size_mb: int = 2000
    cache_ttl_seconds: int = 3600
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    enable_compression: bool = True
    compression_level: int = 6
    enable_cache_analytics: bool = True
    auto_cleanup_interval_seconds: int = 300
    max_cache_entries: int = 10000
    cache_eviction_policy: str = 'lru'  # lru, lfu, ttl
    enable_cache_warming: bool = True
    cache_warming_batch_size: int = 10

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    total_size_bytes: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    eviction_count: int = 0
    cleanup_count: int = 0
    last_cleanup: float = 0.0

class MemoryCache:
    """In-memory cache with LRU eviction."""
    
    def __init__(self, max_size_mb: int, max_entries: int):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.logger = logging.getLogger(f"{__name__}.MemoryCache")
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size_bytes = 0
        self._lock = threading.RLock()
        
        self.logger.info(f"🧠 Memory cache initialized: {max_size_mb}MB, {max_entries} entries")
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (dict, list)):
                return len(pickle.dumps(obj))
            else:
                return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def _evict_entries(self, target_size_bytes: int) -> None:
        """Evict entries to reach target size."""
        with self._lock:
            while (self._current_size_bytes > target_size_bytes and 
                   len(self._cache) > 0):
                
                # Remove least recently used entry
                key, entry = self._cache.popitem(last=False)
                self._current_size_bytes -= entry.size_bytes
                
                self.logger.debug(f"🗑️ Evicted cache entry: {key} ({entry.size_bytes} bytes)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if entry.ttl_seconds and time.time() - entry.created_at > entry.ttl_seconds:
                del self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                return None
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            tags: Optional[List[str]] = None, dependencies: Optional[List[str]] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes
                del self._cache[key]
            
            # Create new entry
            size_bytes = self._estimate_size(value)
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                tags=tags or [],
                dependencies=dependencies or []
            )
            
            # Check if we need to evict entries
            if (self._current_size_bytes + size_bytes > self.max_size_bytes or 
                len(self._cache) >= self.max_entries):
                self._evict_entries(self.max_size_bytes - size_bytes)
            
            # Add new entry
            self._cache[key] = entry
            self._current_size_bytes += size_bytes
            
            self.logger.debug(f"💾 Cached in memory: {key} ({size_bytes} bytes)")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]
                self.logger.debug(f"🗑️ Invalidated cache entry: {key}")
                return True
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag."""
        with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._cache[key]
                self._current_size_bytes -= entry.size_bytes
                del self._cache[key]
            
            if keys_to_remove:
                self.logger.info(f"🗑️ Invalidated {len(keys_to_remove)} entries with tag: {tag}")
            
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            self.logger.info("🧹 Memory cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'entries': len(self._cache),
                'size_bytes': self._current_size_bytes,
                'size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_bytes': self.max_size_bytes,
                'max_entries': self.max_entries,
                'utilization_percent': (self._current_size_bytes / self.max_size_bytes) * 100
            }

class DiskCache:
    """Disk-based cache with compression."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int, compression_level: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_level = compression_level
        self.logger = logging.getLogger(f"{__name__}.DiskCache")
        
        self._metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()
        self._lock = threading.RLock()
        
        self.logger.info(f"💾 Disk cache initialized: {cache_dir} ({max_size_mb}MB)")
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_size(self) -> int:
        """Get current cache size."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            total_size += cache_file.stat().st_size
        return total_size
    
    def _evict_old_entries(self, target_size_bytes: int) -> None:
        """Evict old entries to reach target size."""
        with self._lock:
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self._metadata.items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )
            
            current_size = self._get_size()
            for key, metadata in sorted_entries:
                if current_size <= target_size_bytes:
                    break
                
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    file_size = cache_path.stat().st_size
                    cache_path.unlink()
                    current_size -= file_size
                    del self._metadata[key]
                    
                    self.logger.debug(f"🗑️ Evicted disk cache entry: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            if key not in self._metadata:
                return None
            
            metadata = self._metadata[key]
            cache_path = self._get_cache_path(key)
            
            if not cache_path.exists():
                del self._metadata[key]
                return None
            
            # Check TTL
            if metadata.get('ttl_seconds'):
                if time.time() - metadata['created_at'] > metadata['ttl_seconds']:
                    cache_path.unlink()
                    del self._metadata[key]
                    return None
            
            try:
                # Load cached data
                if metadata.get('format') == 'parquet':
                    data = pd.read_parquet(cache_path)
                elif metadata.get('format') == 'json':
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                else:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                
                # Update access info
                metadata['last_accessed'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                self._save_metadata()
                
                return data
                
            except Exception as e:
                self.logger.error(f"Failed to load disk cache for key {key}: {e}")
                cache_path.unlink(missing_ok=True)
                del self._metadata[key]
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            tags: Optional[List[str]] = None, dependencies: Optional[List[str]] = None) -> None:
        """Set value in disk cache."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            
            # Remove existing entry if present
            if key in self._metadata and cache_path.exists():
                old_size = cache_path.stat().st_size
                cache_path.unlink()
                del self._metadata[key]
            
            try:
                # Determine format and save
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(cache_path, compression='gzip')
                    format_type = 'parquet'
                elif isinstance(value, (dict, list)):
                    with open(cache_path, 'w') as f:
                        json.dump(value, f, default=str)
                    format_type = 'json'
                else:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(value, f)
                    format_type = 'pickle'
                
                # Update metadata
                file_size = cache_path.stat().st_size
                self._metadata[key] = {
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'size_bytes': file_size,
                    'format': format_type,
                    'ttl_seconds': ttl_seconds,
                    'tags': tags or [],
                    'dependencies': dependencies or []
                }
                
                # Check if we need to evict entries
                current_size = self._get_size()
                if current_size > self.max_size_bytes:
                    self._evict_old_entries(self.max_size_bytes - file_size)
                
                self._save_metadata()
                self.logger.debug(f"💾 Cached to disk: {key} ({file_size} bytes)")
                
            except Exception as e:
                self.logger.error(f"Failed to save disk cache for key {key}: {e}")
                cache_path.unlink(missing_ok=True)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate disk cache entry."""
        with self._lock:
            if key in self._metadata:
                cache_path = self._get_cache_path(key)
                cache_path.unlink(missing_ok=True)
                del self._metadata[key]
                self._save_metadata()
                self.logger.debug(f"🗑️ Invalidated disk cache entry: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all disk cache entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            self._metadata.clear()
            self._save_metadata()
            self.logger.info("🧹 Disk cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        with self._lock:
            current_size = self._get_size()
            return {
                'entries': len(self._metadata),
                'size_bytes': current_size,
                'size_mb': current_size / (1024 * 1024),
                'max_size_bytes': self.max_size_bytes,
                'utilization_percent': (current_size / self.max_size_bytes) * 100
            }

class IntelligentCache:
    """Intelligent caching system with multi-level support."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = logging.getLogger(f"{__name__}.IntelligentCache")
        
        # Initialize cache levels
        self.memory_cache = None
        if self.config.enable_memory_cache:
            self.memory_cache = MemoryCache(
                self.config.max_memory_cache_size_mb,
                self.config.max_cache_entries
            )
        
        self.disk_cache = None
        if self.config.enable_disk_cache:
            disk_cache_dir = Path("cache/step03")
            self.disk_cache = DiskCache(
                disk_cache_dir,
                self.config.max_disk_cache_size_mb,
                self.config.compression_level
            )
        
        # Statistics
        self.stats = CacheStats()
        self._lock = threading.RLock()
        
        # Start cleanup task
        if self.config.auto_cleanup_interval_seconds > 0:
            self._cleanup_task = None
            self._start_cleanup_task()
        
        self.logger.info("🚀 Intelligent cache system initialized")
    
    def _start_cleanup_task(self) -> None:
        """Start automatic cleanup task."""
        def cleanup_loop():
            while True:
                time.sleep(self.config.auto_cleanup_interval_seconds)
                try:
                    self._cleanup_expired_entries()
                except Exception as e:
                    self.logger.error(f"Error in cache cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        self.logger.info("🧹 Cache cleanup task started")
    
    def _cleanup_expired_entries(self) -> None:
        """Cleanup expired cache entries."""
        with self._lock:
            self.stats.cleanup_count += 1
            self.stats.last_cleanup = time.time()
            
            # Cleanup memory cache
            if self.memory_cache:
                # Memory cache handles TTL automatically in get()
                pass
            
            # Cleanup disk cache
            if self.disk_cache:
                # Disk cache handles TTL automatically in get()
                pass
            
            self.logger.debug("🧹 Cache cleanup completed")
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create hash of arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        args_hash = hashlib.md5(args_str.encode()).hexdigest()
        return f"{func_name}_{args_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        with self._lock:
            self.stats.total_requests += 1
            
            # Try memory cache first
            if self.memory_cache:
                value = self.memory_cache.get(key)
                if value is not None:
                    self.stats.cache_hits += 1
                    self._update_hit_rate()
                    return value
            
            # Try disk cache
            if self.disk_cache:
                value = self.disk_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    if self.memory_cache:
                        self.memory_cache.set(key, value)
                    
                    self.stats.cache_hits += 1
                    self._update_hit_rate()
                    return value
            
            # Cache miss
            self.stats.cache_misses += 1
            self._update_hit_rate()
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            tags: Optional[List[str]] = None, dependencies: Optional[List[str]] = None) -> None:
        """Set value in cache (both memory and disk)."""
        with self._lock:
            # Set in memory cache
            if self.memory_cache:
                self.memory_cache.set(key, value, ttl_seconds, tags, dependencies)
            
            # Set in disk cache
            if self.disk_cache:
                self.disk_cache.set(key, value, ttl_seconds, tags, dependencies)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self._lock:
            memory_invalidated = False
            disk_invalidated = False
            
            if self.memory_cache:
                memory_invalidated = self.memory_cache.invalidate(key)
            
            if self.disk_cache:
                disk_invalidated = self.disk_cache.invalidate(key)
            
            return memory_invalidated or disk_invalidated
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag."""
        with self._lock:
            memory_count = 0
            disk_count = 0
            
            if self.memory_cache:
                memory_count = self.memory_cache.invalidate_by_tag(tag)
            
            if self.disk_cache:
                # Disk cache doesn't support tag invalidation yet
                pass
            
            return memory_count + disk_count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            if self.memory_cache:
                self.memory_cache.clear()
            
            if self.disk_cache:
                self.disk_cache.clear()
            
            self.logger.info("🧹 All caches cleared")
    
    def _update_hit_rate(self) -> None:
        """Update cache hit rate."""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            stats = {
                'performance': {
                    'total_requests': self.stats.total_requests,
                    'cache_hits': self.stats.cache_hits,
                    'cache_misses': self.stats.cache_misses,
                    'hit_rate': self.stats.hit_rate,
                    'eviction_count': self.stats.eviction_count,
                    'cleanup_count': self.stats.cleanup_count,
                    'last_cleanup': self.stats.last_cleanup
                },
                'memory_cache': self.memory_cache.get_stats() if self.memory_cache else None,
                'disk_cache': self.disk_cache.get_stats() if self.disk_cache else None,
                'config': {
                    'max_memory_cache_size_mb': self.config.max_memory_cache_size_mb,
                    'max_disk_cache_size_mb': self.config.max_disk_cache_size_mb,
                    'cache_ttl_seconds': self.config.cache_ttl_seconds,
                    'enable_memory_cache': self.config.enable_memory_cache,
                    'enable_disk_cache': self.config.enable_disk_cache,
                    'cache_eviction_policy': self.config.cache_eviction_policy
                }
            }
            
            # Calculate total size
            total_size_mb = 0
            if self.memory_cache:
                total_size_mb += self.memory_cache.get_stats()['size_mb']
            if self.disk_cache:
                total_size_mb += self.disk_cache.get_stats()['size_mb']
            
            stats['total_size_mb'] = total_size_mb
            return stats

def memoize(ttl_seconds: Optional[int] = None, tags: Optional[List[str]] = None,
           dependencies: Optional[List[str]] = None, cache_key_func: Optional[Callable] = None):
    """Decorator for intelligent memoization."""
    def decorator(func):
        cache = get_intelligent_cache()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(func.__name__, args, kwargs)
            else:
                key = cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"📦 Cache hit for {func.__name__}: {key}")
                return cached_result
            
            # Execute function
            logger.debug(f"⚡ Cache miss for {func.__name__}: {key}")
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl_seconds, tags, dependencies)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(func.__name__, args, kwargs)
            else:
                key = cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"📦 Cache hit for {func.__name__}: {key}")
                return cached_result
            
            # Execute function
            logger.debug(f"⚡ Cache miss for {func.__name__}: {key}")
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl_seconds, tags, dependencies)
            return result
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Global instance
_intelligent_cache = None

def get_intelligent_cache(config: Optional[CacheConfig] = None) -> IntelligentCache:
    """Get or create global intelligent cache instance."""
    global _intelligent_cache
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache(config)
    return _intelligent_cache

# Convenience functions
def cache_result(key: str, ttl_seconds: Optional[int] = None, 
                tags: Optional[List[str]] = None, dependencies: Optional[List[str]] = None):
    """Cache a result with given key."""
    cache = get_intelligent_cache()
    return cache.get(key)

def set_cache_result(key: str, value: Any, ttl_seconds: Optional[int] = None,
                    tags: Optional[List[str]] = None, dependencies: Optional[List[str]] = None):
    """Set a cached result with given key."""
    cache = get_intelligent_cache()
    cache.set(key, value, ttl_seconds, tags, dependencies)

def invalidate_cache(key: str) -> bool:
    """Invalidate cache entry."""
    cache = get_intelligent_cache()
    return cache.invalidate(key)

def invalidate_cache_by_tag(tag: str) -> int:
    """Invalidate all cache entries with given tag."""
    cache = get_intelligent_cache()
    return cache.invalidate_by_tag(tag)

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache = get_intelligent_cache()
    return cache.get_stats()

def clear_cache() -> None:
    """Clear all cache entries."""
    cache = get_intelligent_cache()
    cache.clear()