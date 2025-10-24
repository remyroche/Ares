"""
Optimization caching system.

This module provides intelligent caching for optimization results and intermediate
computations to improve performance and avoid redundant work.
"""

from typing import Any, Dict, Optional, Hashable, Union
import time
import hashlib
import pickle
import logging
from dataclasses import dataclass
from collections import OrderedDict

from ..exceptions import CacheError


@dataclass
class CacheEntry:
    """Single cache entry."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


class OptimizationCache:
    """
    Intelligent caching system for optimization results.
    
    Features:
    - TTL-based expiration
    - LRU eviction policy
    - Size-based limits
    - Hash-based keys for complex objects
    - Memory usage tracking
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, 
                 max_memory_mb: int = 100):
        """
        Initialize optimization cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self.cache: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: Hashable) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._evict(key)
                self.misses += 1
                return None
            
            # Update access count and move to end (LRU)
            entry.access_count += 1
            self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
            
        except Exception as e:
            raise CacheError(f"Failed to get from cache: {e}") from e
    
    def put(self, key: Hashable, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        try:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except (pickle.PickleError, TypeError):
                # Fallback for non-picklable objects
                size_bytes = 1000  # Estimate
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.current_memory_bytes + size_bytes > self.max_memory_bytes):
                if not self.cache:
                    break  # Can't evict anything
                self._evict_oldest()
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes
            
        except Exception as e:
            raise CacheError(f"Failed to put in cache: {e}") from e
    
    def _evict(self, key: Hashable) -> None:
        """Evict specific key from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache[key]
            self.evictions += 1
    
    def _evict_oldest(self) -> None:
        """Evict oldest (least recently used) entry."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.current_memory_bytes = 0
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_bytes": self.current_memory_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_usage_percent": (self.current_memory_bytes / self.max_memory_bytes) * 100,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._evict(key)
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")
        
        return len(expired_keys)


def create_cache_key(*args, **kwargs) -> str:
    """
    Create a hash-based cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Hash string suitable as cache key
    """
    try:
        # Create a deterministic string representation
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        
        # Convert to string and hash
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
        
    except Exception:
        # Fallback to simple string representation
        return str(hash(str(args) + str(sorted(kwargs.items()))))


class ModelEvaluationCache:
    """Specialized cache for model evaluation results."""
    
    def __init__(self, cache: Optional[OptimizationCache] = None):
        """Initialize with optional underlying cache."""
        self.cache = cache or OptimizationCache(max_size=500, ttl_seconds=1800)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_model_score(self, model_params: Dict[str, Any], 
                       data_hash: str) -> Optional[float]:
        """Get cached model score."""
        key = create_cache_key(model_params, data_hash)
        return self.cache.get(key)
    
    def put_model_score(self, model_params: Dict[str, Any], 
                       data_hash: str, score: float) -> None:
        """Cache model score."""
        key = create_cache_key(model_params, data_hash)
        self.cache.put(key, score)
    
    def create_data_hash(self, X, y) -> str:
        """Create hash for data to detect changes."""
        try:
            import numpy as np
            # Use shape and a sample of data for hashing
            data_info = {
                'X_shape': X.shape if hasattr(X, 'shape') else len(X),
                'y_shape': y.shape if hasattr(y, 'shape') else len(y),
                'X_sample': X.flat[:100].tolist() if hasattr(X, 'flat') else X[:100],
                'y_sample': y.flat[:100].tolist() if hasattr(y, 'flat') else y[:100]
            }
            return create_cache_key(data_info)
        except Exception:
            # Fallback to simple hash
            return str(hash(str(X) + str(y)))