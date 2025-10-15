"""
Advanced Caching Strategies for ML Common Operations

This module provides intelligent caching strategies for expensive ML operations,
integrating with VectorBTRollingOptimizer and M1 hardware optimizations.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Import M1 optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategy types."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    HYBRID = "hybrid"
    VECTORBT = "vectorbt"
    M1_OPTIMIZED = "m1_optimized"

class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

@dataclass
class CacheConfig:
    """Configuration for caching strategies."""
    
    # Basic settings
    strategy: CacheStrategy = CacheStrategy.MEMORY
    max_size: int = 1000
    max_memory_mb: float = 500.0
    ttl_seconds: int = 3600  # 1 hour
    enable_compression: bool = True
    enable_serialization: bool = True
    
    # Eviction settings
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    eviction_threshold: float = 0.8  # Evict when 80% full
    
    # Performance settings
    enable_async: bool = True
    enable_parallel_loading: bool = True
    max_workers: int = 4
    
    # M1 optimizations
    enable_m1_optimizations: bool = True
    use_m1_memory_optimizer: bool = True
    use_m1_cpu_optimizer: bool = True
    
    # VectorBT optimizations
    enable_vectorbt_optimizations: bool = True
    use_vectorbt_rolling: bool = True
    
    # Advanced settings
    enable_weak_references: bool = True
    enable_garbage_collection: bool = True
    gc_threshold: int = 100  # Trigger GC every N operations
    
    # Redis settings (if using Redis)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Disk cache settings
    cache_dir: str = "data_cache/ml_cache"
    enable_disk_compression: bool = True

@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[int] = None
    strategy: CacheStrategy = CacheStrategy.MEMORY
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheStrategyBase(ABC):
    """Base class for cache strategies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0,
            'total_operations': 0
        }
        self._lock = threading.RLock()
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get current cache size."""
        pass
    
    def _update_stats(self, hit: bool = None, eviction: bool = False, error: bool = False):
        """Update cache statistics."""
        with self._lock:
            self._stats['total_operations'] += 1
            if hit is not None:
                if hit:
                    self._stats['hits'] += 1
                else:
                    self._stats['misses'] += 1
            if eviction:
                self._stats['evictions'] += 1
            if error:
                self._stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a deterministic key
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        # Serialize to JSON for hashing
        try:
            key_str = json.dumps(key_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to string representation
            key_str = str(key_data)
        
        # Generate hash
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if not self.config.enable_serialization:
            return value
        
        try:
            if self.config.enable_compression:
                import gzip
                serialized = pickle.dumps(value)
                return gzip.compress(serialized)
            else:
                return pickle.dumps(value)
        except Exception as e:
            self.logger.warning(f"Serialization failed: {e}")
            return value
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if not self.config.enable_serialization:
            return data
        
        try:
            if self.config.enable_compression:
                import gzip
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
        except Exception as e:
            self.logger.warning(f"Deserialization failed: {e}")
            return data

class MemoryCacheStrategy(CacheStrategyBase):
    """In-memory cache strategy with M1 optimizations."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._memory_optimizer = None
        self._cpu_optimizer = None
        
        if M1_OPTIMIZATIONS_AVAILABLE and config.enable_m1_optimizations:
            if config.use_m1_memory_optimizer:
                self._memory_optimizer = get_m1_memory_optimizer()
            if config.use_m1_cpu_optimizer:
                self._cpu_optimizer = get_m1_cpu_optimizer()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        try:
            with self._lock:
                if key not in self._cache:
                    self._update_stats(hit=False)
                    return None
                
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._update_stats(hit=False, eviction=True)
                    return None
                
                # Update access statistics
                entry.update_access()
                
                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                
                self._update_stats(hit=True)
                return entry.value
                
        except Exception as e:
            self.logger.error(f"Error getting from memory cache: {e}")
            self._update_stats(error=True)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            with self._lock:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check memory limits
                if self._should_evict(size_bytes):
                    await self._evict_entries()
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    size_bytes=size_bytes,
                    ttl=ttl or self.config.ttl_seconds,
                    strategy=CacheStrategy.MEMORY
                )
                
                # Store entry
                self._cache[key] = entry
                self._access_order.append(key)
                
                # Apply M1 memory optimization if available
                if self._memory_optimizer and M1_OPTIMIZATIONS_AVAILABLE:
                    self._memory_optimizer.optimize_dataframe_memory(value)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting memory cache: {e}")
            self._update_stats(error=True)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error deleting from memory cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all memory cache entries."""
        try:
            with self._lock:
                self._cache.clear()
                self._access_order.clear()
                
                # Force garbage collection if enabled
                if self.config.enable_garbage_collection:
                    gc.collect()
                
                return True
        except Exception as e:
            self.logger.error(f"Error clearing memory cache: {e}")
            return False
    
    async def size(self) -> int:
        """Get current memory cache size."""
        with self._lock:
            return len(self._cache)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if hasattr(value, 'memory_usage'):
                # Pandas DataFrame/Series
                return value.memory_usage(deep=True).sum()
            elif hasattr(value, 'nbytes'):
                # NumPy array
                return value.nbytes
            elif isinstance(value, (list, tuple, dict)):
                # Estimate for Python objects
                return len(str(value).encode('utf-8'))
            else:
                # Fallback estimation
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def _should_evict(self, new_size: int) -> bool:
        """Check if we should evict entries."""
        current_size = sum(entry.size_bytes for entry in self._cache.values())
        total_size = current_size + new_size
        
        # Check memory limit
        if total_size > self.config.max_memory_mb * 1024 * 1024:
            return True
        
        # Check count limit
        if len(self._cache) >= self.config.max_size:
            return True
        
        return False
    
    async def _evict_entries(self):
        """Evict entries based on policy."""
        if self.config.eviction_policy == CacheEvictionPolicy.LRU:
            await self._evict_lru()
        elif self.config.eviction_policy == CacheEvictionPolicy.LFU:
            await self._evict_lfu()
        elif self.config.eviction_policy == CacheEvictionPolicy.TTL:
            await self._evict_expired()
        elif self.config.eviction_policy == CacheEvictionPolicy.SIZE:
            await self._evict_largest()
        else:  # ADAPTIVE
            await self._evict_adaptive()
    
    async def _evict_lru(self):
        """Evict least recently used entries."""
        with self._lock:
            evict_count = max(1, int(len(self._cache) * self.config.eviction_threshold))
            
            for _ in range(evict_count):
                if self._access_order:
                    key_to_evict = self._access_order.pop(0)
                    if key_to_evict in self._cache:
                        del self._cache[key_to_evict]
                        self._update_stats(eviction=True)
    
    async def _evict_lfu(self):
        """Evict least frequently used entries."""
        with self._lock:
            evict_count = max(1, int(len(self._cache) * self.config.eviction_threshold))
            
            # Sort by access count
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
            
            for i in range(min(evict_count, len(sorted_entries))):
                key_to_evict = sorted_entries[i][0]
                del self._cache[key_to_evict]
                if key_to_evict in self._access_order:
                    self._access_order.remove(key_to_evict)
                self._update_stats(eviction=True)
    
    async def _evict_expired(self):
        """Evict expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._update_stats(eviction=True)
    
    async def _evict_largest(self):
        """Evict largest entries."""
        with self._lock:
            evict_count = max(1, int(len(self._cache) * self.config.eviction_threshold))
            
            # Sort by size
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )
            
            for i in range(min(evict_count, len(sorted_entries))):
                key_to_evict = sorted_entries[i][0]
                del self._cache[key_to_evict]
                if key_to_evict in self._access_order:
                    self._access_order.remove(key_to_evict)
                self._update_stats(eviction=True)
    
    async def _evict_adaptive(self):
        """Adaptive eviction based on usage patterns."""
        with self._lock:
            # Combine LRU and LFU with weights
            current_time = time.time()
            
            def score_entry(entry: CacheEntry) -> float:
                # Time-based score (older is worse)
                age_score = (current_time - entry.last_accessed) / 3600  # Hours
                
                # Frequency-based score (less frequent is worse)
                freq_score = 1.0 / max(1, entry.access_count)
                
                # Size-based score (larger is worse)
                size_score = entry.size_bytes / (1024 * 1024)  # MB
                
                # Combined score (higher is worse)
                return age_score * 0.4 + freq_score * 0.4 + size_score * 0.2
            
            evict_count = max(1, int(len(self._cache) * self.config.eviction_threshold))
            
            # Sort by combined score
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: score_entry(x[1]),
                reverse=True
            )
            
            for i in range(min(evict_count, len(sorted_entries))):
                key_to_evict = sorted_entries[i][0]
                del self._cache[key_to_evict]
                if key_to_evict in self._access_order:
                    self._access_order.remove(key_to_evict)
                self._update_stats(eviction=True)

class VectorBTCacheStrategy(CacheStrategyBase):
    """VectorBT-optimized cache strategy for financial operations."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._vectorbt_optimizer = None
        self._cache: Dict[str, CacheEntry] = {}
        
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and config.enable_vectorbt_optimizations:
            self._vectorbt_optimizer = get_vectorbt_rolling_optimizer()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from VectorBT cache."""
        try:
            with self._lock:
                if key not in self._cache:
                    self._update_stats(hit=False)
                    return None
                
                entry = self._cache[key]
                
                if entry.is_expired():
                    del self._cache[key]
                    self._update_stats(hit=False, eviction=True)
                    return None
                
                entry.update_access()
                self._update_stats(hit=True)
                return entry.value
                
        except Exception as e:
            self.logger.error(f"Error getting from VectorBT cache: {e}")
            self._update_stats(error=True)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in VectorBT cache with optimizations."""
        try:
            with self._lock:
                # Apply VectorBT optimizations if available
                if self._vectorbt_optimizer and hasattr(value, 'values'):
                    # Optimize DataFrame/Series for VectorBT operations
                    optimized_value = self._vectorbt_optimizer.optimize_dataframe(value)
                else:
                    optimized_value = value
                
                # Calculate size
                size_bytes = self._calculate_size(optimized_value)
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=optimized_value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    size_bytes=size_bytes,
                    ttl=ttl or self.config.ttl_seconds,
                    strategy=CacheStrategy.VECTORBT
                )
                
                self._cache[key] = entry
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting VectorBT cache: {e}")
            self._update_stats(error=True)
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from VectorBT cache."""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error deleting from VectorBT cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all VectorBT cache entries."""
        try:
            with self._lock:
                self._cache.clear()
                return True
        except Exception as e:
            self.logger.error(f"Error clearing VectorBT cache: {e}")
            return False
    
    async def size(self) -> int:
        """Get current VectorBT cache size."""
        with self._lock:
            return len(self._cache)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if hasattr(value, 'memory_usage'):
                return value.memory_usage(deep=True).sum()
            elif hasattr(value, 'nbytes'):
                return value.nbytes
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024

class HybridCacheStrategy(CacheStrategyBase):
    """Hybrid cache strategy combining multiple strategies."""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._strategies: Dict[CacheStrategy, CacheStrategyBase] = {}
        self._strategy_priority = [
            CacheStrategy.MEMORY,
            CacheStrategy.VECTORBT,
            CacheStrategy.DISK
        ]
        
        # Initialize strategies
        self._strategies[CacheStrategy.MEMORY] = MemoryCacheStrategy(config)
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            self._strategies[CacheStrategy.VECTORBT] = VectorBTCacheStrategy(config)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache."""
        for strategy_type in self._strategy_priority:
            if strategy_type in self._strategies:
                value = await self._strategies[strategy_type].get(key)
                if value is not None:
                    # Promote to higher priority cache
                    await self._promote_to_higher_cache(key, value)
                    return value
        
        self._update_stats(hit=False)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in hybrid cache."""
        success = False
        
        # Set in all available strategies
        for strategy_type in self._strategy_priority:
            if strategy_type in self._strategies:
                try:
                    await self._strategies[strategy_type].set(key, value, ttl)
                    success = True
                except Exception as e:
                    self.logger.warning(f"Failed to set in {strategy_type.value} cache: {e}")
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from hybrid cache."""
        success = True
        
        for strategy in self._strategies.values():
            try:
                await strategy.delete(key)
            except Exception as e:
                self.logger.warning(f"Failed to delete from cache: {e}")
                success = False
        
        return success
    
    async def clear(self) -> bool:
        """Clear all hybrid cache entries."""
        success = True
        
        for strategy in self._strategies.values():
            try:
                await strategy.clear()
            except Exception as e:
                self.logger.warning(f"Failed to clear cache: {e}")
                success = False
        
        return success
    
    async def size(self) -> int:
        """Get total hybrid cache size."""
        total_size = 0
        
        for strategy in self._strategies.values():
            try:
                total_size += await strategy.size()
            except Exception as e:
                self.logger.warning(f"Failed to get cache size: {e}")
        
        return total_size
    
    async def _promote_to_higher_cache(self, key: str, value: Any):
        """Promote value to higher priority cache."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated promotion logic
        pass

class MLCommonCache:
    """Main cache manager for ML Common operations."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = logger.getChild('MLCommonCache')
        self._strategy: Optional[CacheStrategyBase] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the cache strategy."""
        if self._initialized:
            return
        
        try:
            if self.config.strategy == CacheStrategy.MEMORY:
                self._strategy = MemoryCacheStrategy(self.config)
            elif self.config.strategy == CacheStrategy.VECTORBT:
                self._strategy = VectorBTCacheStrategy(self.config)
            elif self.config.strategy == CacheStrategy.HYBRID:
                self._strategy = HybridCacheStrategy(self.config)
            else:
                # Default to memory cache
                self._strategy = MemoryCacheStrategy(self.config)
            
            self._initialized = True
            self.logger.info(f"Cache initialized with {self.config.strategy.value} strategy")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._initialized:
            await self.initialize()
        
        return await self._strategy.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._initialized:
            await self.initialize()
        
        return await self._strategy.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._initialized:
            await self.initialize()
        
        return await self._strategy.delete(key)
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        if not self._initialized:
            await self.initialize()
        
        return await self._strategy.clear()
    
    async def size(self) -> int:
        """Get current cache size."""
        if not self._initialized:
            await self.initialize()
        
        return await self._strategy.size()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized or not self._strategy:
            return {}
        
        return self._strategy.get_stats()

# Global cache instance
_global_cache: Optional[MLCommonCache] = None

def get_ml_common_cache(config: Optional[CacheConfig] = None) -> MLCommonCache:
    """Get the global ML Common cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = MLCommonCache(config)
    
    return _global_cache

def cached(
    strategy: CacheStrategy = CacheStrategy.MEMORY,
    ttl: Optional[int] = None,
    key_prefix: str = "",
    enable_async: bool = True
):
    """
    Decorator for caching function results.
    
    Args:
        strategy: Cache strategy to use
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        enable_async: Whether to use async caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache instance
            cache = get_ml_common_cache()
            await cache.initialize()
            
            # Generate cache key
            key = f"{key_prefix}{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Store in cache
            await cache.set(key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run in an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, create a new one
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, async_wrapper(*args, **kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(async_wrapper(*args, **kwargs))
        
        if enable_async and asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Convenience functions
async def cache_get(key: str) -> Optional[Any]:
    """Get value from global cache."""
    cache = get_ml_common_cache()
    await cache.initialize()
    return await cache.get(key)

async def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in global cache."""
    cache = get_ml_common_cache()
    await cache.initialize()
    return await cache.set(key, value, ttl)

async def cache_delete(key: str) -> bool:
    """Delete value from global cache."""
    cache = get_ml_common_cache()
    await cache.initialize()
    return await cache.delete(key)

async def cache_clear() -> bool:
    """Clear global cache."""
    cache = get_ml_common_cache()
    await cache.initialize()
    return await cache.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    cache = get_ml_common_cache()
    return cache.get_stats()