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
from collections import OrderedDict, deque

from ..exceptions import CacheError

# Import tprint functions
try:
    from ...tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer, tprint_data_preview,
        tprint_data_format, LogLevel, TPrintConfig
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def tprint_success(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_performance(*args, **kwargs): pass
    def tprint_data_preview(*args, **kwargs): pass
    def tprint_data_format(*args, **kwargs): pass


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
        self.access_order: deque = deque()  # Track access order for O(1) LRU
        self.current_memory_bytes = 0
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        if TPRINT_AVAILABLE:
            tprint_success(f"💾 OptimizationCache initialized: max_size={max_size}, ttl={ttl_seconds}s, memory_limit={max_memory_mb}MB")
    
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
                if TPRINT_AVAILABLE:
                    tprint_debug(f"🔍 Cache miss for key: {str(key)[:50]}...")
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._evict(key)
                self.misses += 1
                if TPRINT_AVAILABLE:
                    tprint_debug(f"⏰ Cache entry expired for key: {str(key)[:50]}...")
                return None
            
            # Update access count and move to end (LRU)
            entry.access_count += 1
            self.cache.move_to_end(key)
            
            # Update access order for O(1) LRU
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hits += 1
            if TPRINT_AVAILABLE:
                tprint_debug(f"✅ Cache hit for key: {str(key)[:50]}... (access count: {entry.access_count})")
            return entry.value
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Cache get failed: {e}")
            raise CacheError(f"Failed to get from cache: {e}") from e
    
    def put(self, key: Hashable, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        try:
            if TPRINT_AVAILABLE:
                tprint_debug(f"💾 Caching value for key: {str(key)[:50]}...")
                tprint_data_preview(value, f"cache_value_{str(key)[:20]}")
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except (pickle.PickleError, TypeError):
                # Better fallback for non-picklable objects
                try:
                    import sys
                    size_bytes = sys.getsizeof(value)
                    # Add overhead for complex objects
                    if hasattr(value, '__dict__'):
                        size_bytes += sum(sys.getsizeof(v) for v in value.__dict__.values())
                except:
                    # Final fallback with better estimation
                    if isinstance(value, (list, tuple)):
                        size_bytes = len(value) * 100  # Estimate 100 bytes per item
                    elif isinstance(value, dict):
                        size_bytes = len(value) * 200  # Estimate 200 bytes per key-value pair
                    else:
                        size_bytes = 1000  # Default estimate
            
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
                if TPRINT_AVAILABLE:
                    tprint_debug(f"🔄 Replacing existing cache entry for key: {str(key)[:50]}...")
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Cached value ({size_bytes} bytes) for key: {str(key)[:50]}...")
                tprint_info(f"📊 Cache stats: {len(self.cache)}/{self.max_size} entries, {self.current_memory_bytes/1024/1024:.1f}MB used")
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Cache put failed: {e}")
            raise CacheError(f"Failed to put in cache: {e}") from e
    
    def _evict(self, key: Hashable) -> None:
        """Evict specific key from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache[key]
            # Remove from access order if present
            if key in self.access_order:
                self.access_order.remove(key)
            self.evictions += 1
    
    def _evict_oldest(self) -> None:
        """Evict oldest (least recently used) entry using O(1) deque operations."""
        if self.cache and self.access_order:
            # Get the least recently used key from the front of the deque
            oldest_key = self.access_order.popleft()
            # Remove from cache if it still exists
            if oldest_key in self.cache:
                self._evict(oldest_key)
        elif self.cache:
            # Fallback to OrderedDict method if deque is empty
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        self.current_memory_bytes = 0
        if TPRINT_AVAILABLE:
            tprint_success("🧹 Cache cleared")
        else:
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
            if TPRINT_AVAILABLE:
                tprint_success(f"🧹 Cleaned up {len(expired_keys)} expired entries")
            else:
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
        # Use more efficient key generation
        key_parts = []
        
        # Handle positional arguments
        for arg in args:
            if hasattr(arg, '__hash__') and not isinstance(arg, (list, dict, set)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                # Preserve order for lists and tuples - use tuple() to maintain order
                key_parts.append(str(tuple(arg)))
            else:
                # Use a more robust hash for complex objects
                try:
                    key_parts.append(str(hash(arg)))
                except TypeError:
                    # For unhashable objects, use string representation
                    key_parts.append(str(arg))
        
        # Handle keyword arguments
        for key, value in sorted(kwargs.items()):
            if hasattr(value, '__hash__') and not isinstance(value, (list, dict, set)):
                key_parts.append(f"{key}={value}")
            elif isinstance(value, (list, tuple)):
                # Preserve order for lists and tuples
                key_parts.append(f"{key}={tuple(value)}")
            else:
                try:
                    key_parts.append(f"{key}={hash(value)}")
                except TypeError:
                    key_parts.append(f"{key}={str(value)}")
        
        # Create hash from combined parts using SHA-256 for better collision resistance
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()
        
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
        if TPRINT_AVAILABLE:
            tprint_debug(f"🔍 Looking up model score for data hash: {data_hash[:20]}...")
        score = self.cache.get(key)
        if score is not None and TPRINT_AVAILABLE:
            tprint_success(f"✅ Found cached model score: {score:.4f}")
        return score
    
    def put_model_score(self, model_params: Dict[str, Any], 
                       data_hash: str, score: float) -> None:
        """Cache model score."""
        key = create_cache_key(model_params, data_hash)
        if TPRINT_AVAILABLE:
            tprint_info(f"💾 Caching model score {score:.4f} for data hash: {data_hash[:20]}...")
        self.cache.put(key, score)
    
    def create_data_hash(self, X, y) -> str:
        """Create hash for data to detect changes."""
        try:
            import numpy as np
            # Use shape and a sample of data for hashing
            data_info = {
                'X_shape': X.shape if hasattr(X, 'shape') else len(X),
                'y_shape': y.shape if hasattr(y, 'shape') else len(y),
                'X_dtype': str(X.dtype) if hasattr(X, 'dtype') else str(type(X)),
                'y_dtype': str(y.dtype) if hasattr(y, 'dtype') else str(type(y)),
                'X_sample': X.flat[:100].tolist() if hasattr(X, 'flat') else X[:100],
                'y_sample': y.flat[:100].tolist() if hasattr(y, 'flat') else y[:100]
            }
            hash_key = create_cache_key(data_info)
            if TPRINT_AVAILABLE:
                tprint_debug(f"🔑 Created data hash: {hash_key[:20]}...")
                tprint_data_format(data_info, "data_hash_info")
            return hash_key
        except Exception:
            # Fallback to simple hash
            hash_key = str(hash(str(X) + str(y)))
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Using fallback hash: {hash_key[:20]}...")
            return hash_key