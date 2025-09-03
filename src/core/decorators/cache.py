"""
Caching decorators with flexible policies.

Provides decorators for caching function results with support for
per-request caching, cross-request caching, TTL, and invalidation.
"""

import functools
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .compose import P, R, uniform_wrapper
from .logging import get_correlation_id

# Context variable for request-scoped cache
request_cache_var: ContextVar[dict[str, Any] | None] = ContextVar("request_cache", default=None)


class CachePolicy(Enum):
    """Cache policy types."""
    PER_REQUEST = "per_request"  # Cache only within a single request
    CROSS_REQUEST = "cross_request"  # Cache across requests
    DISTRIBUTED = "distributed"  # Use distributed cache (Redis, etc.)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    created_at: float
    expires_at: float | None
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def increment_hits(self) -> None:
        """Increment hit counter."""
        self.hit_count += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value from cache."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache with optional TTL."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.access_order: list[str] = []

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        entry = self.cache.get(key)
        if entry is None:
            return None

        if entry.is_expired:
            self.delete(key)
            return None

        # Update access order for LRU
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        entry.increment_hits()
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache with optional TTL."""
        # Enforce max size with LRU eviction
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                lru_key = self.access_order[0]
                self.delete(lru_key)

        expires_at = time.time() + ttl if ttl else None
        self.cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
        )

        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self.cache:
            return False

        entry = self.cache[key]
        if entry.is_expired:
            self.delete(key)
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(entry.hit_count for entry in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "keys": list(self.cache.keys()),
        }


# Global cache backends
_memory_cache = MemoryCacheBackend()
_cache_backends: dict[CachePolicy, CacheBackend] = {
    CachePolicy.CROSS_REQUEST: _memory_cache,
}


def get_request_cache() -> dict[str, Any]:
    """Get or create request-scoped cache."""
    cache = request_cache_var.get()
    if cache is None:
        cache = {}
        request_cache_var.set(cache)
    return cache


def clear_request_cache() -> None:
    """Clear request-scoped cache."""
    request_cache_var.set(None)


def make_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    include_correlation_id: bool = False,
) -> str:
    """
    Create a cache key from function and arguments.

    Args:
        func: Function being cached
        args: Function arguments
        kwargs: Function keyword arguments
        include_correlation_id: Whether to include correlation ID in key

    Returns:
        Cache key string
    """
    # Create key components
    key_parts = [
        func.__module__,
        func.__name__,
    ]

    if include_correlation_id:
        key_parts.append(get_correlation_id())

    # Serialize arguments
    try:
        # Try JSON first (faster)
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        key_parts.extend([args_str, kwargs_str])
        key = ":".join(key_parts)
    except (TypeError, ValueError):
        # Fall back to pickle for complex objects
        key_data = pickle.dumps((key_parts, args, kwargs))
        key = hashlib.sha256(key_data).hexdigest()

    return key


def cached(
    *,
    policy: CachePolicy = CachePolicy.PER_REQUEST,
    ttl: float | None = None,
    key_func: Callable | None = None,
    condition: Callable[[Any], bool] | None = None,
    cache_none: bool = False,
    cache_exceptions: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Cache function results based on policy.

    Args:
        policy: Caching policy (per-request, cross-request, distributed)
        ttl: Time-to-live in seconds
        key_func: Custom function to generate cache keys
        condition: Function to determine if result should be cached
        cache_none: Whether to cache None results
        cache_exceptions: Whether to cache exceptions

    Example:
        @cached(policy=CachePolicy.CROSS_REQUEST, ttl=300)
        def get_user(user_id: str) -> dict:
            return database.fetch_user(user_id)

        @cached(
            policy=CachePolicy.PER_REQUEST,
            condition=lambda result: result["status"] == "success"
        )
        def api_call(endpoint: str) -> dict:
            return requests.get(endpoint).json()
    """
    def get_cache_backend() -> dict[str, Any] | CacheBackend:
        """Get appropriate cache backend based on policy."""
        if policy == CachePolicy.PER_REQUEST:
            return get_request_cache()
        if policy in _cache_backends:
            return _cache_backends[policy]
        msg = f"No backend configured for policy {policy}"
        raise ValueError(msg)

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Generate cache key
        if key_func:
            cache_key = key_func(func, args, kwargs)
        else:
            include_correlation = policy == CachePolicy.PER_REQUEST
            cache_key = make_cache_key(func, args, kwargs, include_correlation)

        # Get cache backend
        cache = get_cache_backend()

        # Check cache
        if policy == CachePolicy.PER_REQUEST:
            # Simple dict interface for request cache
            if cache_key in cache:
                return cache[cache_key]
        else:
            # CacheBackend interface
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

        # Execute function
        try:
            result = func(*args, **kwargs)

            # Check if we should cache the result
            should_cache = True
            if result is None and not cache_none:
                should_cache = False
            if condition and not condition(result):
                should_cache = False

            # Cache the result
            if should_cache:
                if policy == CachePolicy.PER_REQUEST:
                    cache[cache_key] = result
                else:
                    cache.set(cache_key, result, ttl)

            return result

        except Exception as e:
            # Cache exception if requested
            if cache_exceptions:
                if policy == CachePolicy.PER_REQUEST:
                    cache[cache_key] = e
                else:
                    cache.set(cache_key, e, ttl)
            raise

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Generate cache key
        if key_func:
            cache_key = key_func(func, args, kwargs)
        else:
            include_correlation = policy == CachePolicy.PER_REQUEST
            cache_key = make_cache_key(func, args, kwargs, include_correlation)

        # Get cache backend
        cache = get_cache_backend()

        # Check cache
        if policy == CachePolicy.PER_REQUEST:
            if cache_key in cache:
                return cache[cache_key]
        else:
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Check if we should cache the result
            should_cache = True
            if result is None and not cache_none:
                should_cache = False
            if condition and not condition(result):
                should_cache = False

            # Cache the result
            if should_cache:
                if policy == CachePolicy.PER_REQUEST:
                    cache[cache_key] = result
                else:
                    cache.set(cache_key, result, ttl)

            return result

        except Exception as e:
            # Cache exception if requested
            if cache_exceptions:
                if policy == CachePolicy.PER_REQUEST:
                    cache[cache_key] = e
                else:
                    cache.set(cache_key, e, ttl)
            raise

    return uniform_wrapper(f"cached({policy.value})", sync_handler, async_handler)


def cache_invalidate(
    func: Callable | None = None,
    *,
    pattern: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """
    Invalidate cache entries.

    Args:
        func: Function whose cache to invalidate
        pattern: Pattern to match cache keys
        tags: Tags to match (if supported by backend)

    Example:
        # Invalidate specific function's cache
        cache_invalidate(get_user)

        # Invalidate by pattern
        cache_invalidate(pattern="user:*")
    """
    # For now, just clear the entire cache
    # This could be enhanced with pattern matching
    for backend in _cache_backends.values():
        backend.clear()

    # Clear request cache
    clear_request_cache()


def memoize(
    maxsize: int = 128,
    typed: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Simple memoization decorator using functools.lru_cache.

    This is a convenience wrapper for simple in-memory caching
    without the full caching policy system.

    Args:
        maxsize: Maximum cache size
        typed: Whether to cache separately based on argument types

    Example:
        @memoize(maxsize=100)
        def fibonacci(n: int) -> int:
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Use functools.lru_cache for the implementation
        cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        # Preserve the original function for introspection
        cached_func.__wrapped__ = func

        return cached_func

    return decorator


def cache_stats() -> dict[str, Any]:
    """
    Get cache statistics for all backends.

    Returns:
        Dictionary with stats for each cache backend
    """
    stats = {}

    # Request cache stats
    request_cache = request_cache_var.get()
    if request_cache:
        stats["per_request"] = {
            "size": len(request_cache),
            "keys": list(request_cache.keys()),
        }

    # Backend stats
    for policy, backend in _cache_backends.items():
        if hasattr(backend, "get_stats"):
            stats[policy.value] = backend.get_stats()
        else:
            stats[policy.value] = {"type": type(backend).__name__}

    return stats
