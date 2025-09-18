from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import functools
import hashlib
import pickle
import time
from pathlib import Path

"""Caching utilities for the Ares project."""

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple cache manager for the Ares project."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_cache/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        # Check memory cache first
        if key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                    self.memory_cache[key] = value
                    self.cache_stats["hits"] += 1
                    return value
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set value in cache."""
        self.memory_cache[key] = value
        
        if persist:
            try:
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.cache_stats.copy()


# Global cache manager instance
_cache_manager = None


def get_cache_manager(cache_dir: Optional[str] = None) -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir)
    return _cache_manager


def intelligent_caching(*args, **kwargs) -> None:
    """Intelligent caching decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(ttl: int = 3600, use_disk: bool = True):
    """
    Cache function results with optional TTL and disk persistence.
    
    Args:
        ttl: Time to live in seconds
        use_disk: Whether to persist to disk
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            cache_manager = get_cache_manager()
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                cached_value, timestamp = cached_result
                if time.time() - timestamp < ttl:
                    return cached_value
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, (result, time.time()), persist=use_disk)
            
            return result
        return wrapper
    return decorator