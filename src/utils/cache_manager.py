"""Cache Manager Module.

Handles caching strategies for artifacts with LRU eviction and memory management.
"""

import time
import threading
from collections import OrderedDict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

from .logger import system_logger


@dataclass
class CacheConfig:
    """Configuration for cache management."""
    max_size_mb: float = 512.0
    enable_lru_eviction: bool = True
    enable_ttl: bool = True
    ttl_hours: float = 24.0
    cleanup_interval_seconds: float = 300.0
    enable_thread_safety: bool = True


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0


class CacheManager:
    """Handles caching strategies for artifacts."""
    
    def __init__(self, config: CacheConfig):
        """Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.logger = system_logger.getChild("CacheManager")
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_size_bytes = 0
        self._max_cache_size_bytes = int(config.max_size_mb * 1024 * 1024)
        
        # Thread safety
        if config.enable_thread_safety:
            self._lock = threading.RLock()
        else:
            self._lock = None
        
        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Background cleanup
        self._last_cleanup = time.time()
    
    def _lock_context(self):
        """Get lock context manager."""
        if self._lock is not None:
            return self._lock
        return nullcontext()
    
    def put(self, key: str, data: Any, size_bytes: Optional[int] = None) -> bool:
        """Put data into cache.
        
        Args:
            key: Cache key
            data: Data to cache
            size_bytes: Size of data in bytes (estimated if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock_context():
            try:
                # Estimate size if not provided
                if size_bytes is None:
                    size_bytes = self._estimate_size(data)
                
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                
                # Evict entries if cache is full
                while (self._cache_size_bytes + size_bytes) > self._max_cache_size_bytes and self._cache:
                    self._evict_lru()
                
                # Add new entry
                entry = CacheEntry(
                    key=key,
                    data=data,
                    size_bytes=size_bytes,
                    created_at=datetime.now(),
                    last_accessed=datetime.now()
                )
                
                self._cache[key] = entry
                self._cache_size_bytes += size_bytes
                
                self.logger.debug(f"Cached {key} ({size_bytes} bytes)")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache {key}: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        with self._lock_context():
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if self.config.enable_ttl:
                    if datetime.now() - entry.created_at > timedelta(hours=self.config.ttl_hours):
                        self._remove_entry(key)
                        self._misses += 1
                        return None
                
                # Update access info
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                self._hits += 1
                self.logger.debug(f"Cache hit for {key}")
                return entry.data
            else:
                self._misses += 1
                self.logger.debug(f"Cache miss for {key}")
                return None
    
    def remove(self, key: str) -> bool:
        """Remove data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if removed, False if not found
        """
        with self._lock_context():
            if key in self._cache:
                self._remove_entry(key)
                self.logger.debug(f"Removed {key} from cache")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock_context():
            self._cache.clear()
            self._cache_size_bytes = 0
            self.logger.debug("Cache cleared")
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._cache_size_bytes -= entry.size_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Remove oldest entry
        oldest_key, oldest_entry = self._cache.popitem(last=False)
        self._cache_size_bytes -= oldest_entry.size_bytes
        self._evictions += 1
        
        self.logger.debug(f"Evicted {oldest_key} from cache")
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        try:
            import sys
            return sys.getsizeof(data)
        except:
            return 1024  # Default estimate
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries.
        
        Returns:
            Number of entries cleaned up
        """
        if not self.config.enable_ttl:
            return 0
        
        with self._lock_context():
            current_time = datetime.now()
            ttl_delta = timedelta(hours=self.config.ttl_hours)
            expired_keys = []
            
            for key, entry in self._cache.items():
                if current_time - entry.created_at > ttl_delta:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cleanup if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.config.cleanup_interval_seconds:
            self.cleanup_expired()
            self._last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock_context():
            total_requests = self._hits + self._misses
            hit_ratio = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
                "size_bytes": self._cache_size_bytes,
                "max_size_bytes": self._max_cache_size_bytes,
                "size_mb": self._cache_size_bytes / (1024 * 1024),
                "max_size_mb": self._max_cache_size_bytes / (1024 * 1024),
                "utilization": (self._cache_size_bytes / self._max_cache_size_bytes) * 100,
                "entries": len(self._cache),
                "evictions": self._evictions,
                "config": {
                    "max_size_mb": self.config.max_size_mb,
                    "enable_lru_eviction": self.config.enable_lru_eviction,
                    "enable_ttl": self.config.enable_ttl,
                    "ttl_hours": self.config.ttl_hours,
                    "enable_thread_safety": self.config.enable_thread_safety
                }
            }