from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import functools
import hashlib
import pickle
import time
from pathlib import Path
import asyncio

from src.utils.unified_cache import UnifiedCache, get_unified_cache

"""Caching utilities for the Ares project."""

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple cache manager for the Ares project."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data_cache/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_stats = {"hits": 0, "misses": 0}
        # Backed by unified cache (namespace-scoped)
        self._ucache = UnifiedCache(
            cache_dir=str(self.cache_dir), namespace="cache_manager", enable_disk=True, enable_compression=True
        )
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        value = self._ucache.get(key)
        if value is not None:
            self.cache_stats["hits"] += 1
            return value
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set value in cache."""
        self._ucache.set(key, value, persist=persist)
    
    def clear(self) -> None:
        """Clear all caches."""
        self._ucache.clear_namespace()
    
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


def intelligent_caching(
    ttl: Optional[int] = None,
    cache_key: Optional[str] = None,
    key_func: Optional[Callable[..., str]] = None,
    namespace: str = "decorators",
    use_disk: bool = True,
):
    """Intelligent caching decorator using UnifiedCache.

    Supports both sync and async functions. Key selection priority:
    cache_key > key_func(args, kwargs) > function+args hash.
    """

    def _compute_key(func: Callable, f_args: Tuple[Any, ...], f_kwargs: Dict[str, Any]) -> str:
        if cache_key:
            return str(cache_key)
        if key_func is not None:
            try:
                return str(key_func(*f_args, **f_kwargs))
            except Exception:
                pass
        # Fallback: stable key from function name and args
        uc = get_unified_cache(namespace=namespace)
        return uc.build_cache_key(func.__name__, f_args, f_kwargs)

    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        cache = get_unified_cache(namespace=namespace)

        if is_async:
            async def async_wrapper(*f_args: Any, **f_kwargs: Any):
                key = _compute_key(func, f_args, f_kwargs)
                cached = await cache.aget(key)
                if cached is not None:
                    return cached
                result = await func(*f_args, **f_kwargs)
                await cache.aset(key, result, ttl_seconds=ttl, persist=use_disk)
                return result
            return functools.wraps(func)(async_wrapper)

        def sync_wrapper(*f_args: Any, **f_kwargs: Any):
            key = _compute_key(func, f_args, f_kwargs)
            cached = cache.get(key)
            if cached is not None:
                return cached
            result = func(*f_args, **f_kwargs)
            cache.set(key, result, ttl_seconds=ttl, persist=use_disk)
            return result
        return functools.wraps(func)(sync_wrapper)

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


class IntelligentCache:
    """Async-friendly cache wrapper built on UnifiedCache.

    Used by feature generation and other async code paths that call get/set directly.
    """

    def __init__(
        self,
        *,
        ttl_seconds: Optional[int] = None,
        cache_dir: str = "data_cache/feature_cache",
        namespace: str = "intelligent",
        max_memory_mb: int = 2048,
        enable_compression: bool = True,
        enable_disk: bool = True,
    ) -> None:
        self._cache = UnifiedCache(
            cache_dir=cache_dir,
            max_memory_mb=max_memory_mb,
            enable_disk=enable_disk,
            enable_compression=enable_compression,
            default_ttl_seconds=ttl_seconds,
            namespace=namespace,
        )

    async def get(self, key: str) -> Any | None:
        return await self._cache.aget(key)

    async def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._cache.aset(key, value, metadata)

    def clear(self) -> None:
        self._cache.clear_namespace()

    def get_stats(self) -> Dict[str, Any]:
        return self._cache.get_stats()