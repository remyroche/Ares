from typing import Any, Dict, Optional
import logging

from src.utils.unified_cache import UnifiedCache

"""Caching utilities for the Ares project."""

logger = logging.getLogger(__name__)


"""
NOTE: This module now only provides IntelligentCache. Use src.utils.unified_cache
for the core UnifiedCache and decorators (cached).
"""


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

    # Sync API
    def get(self, key: str) -> Any | None:
        """Retrieve a cached value synchronously."""
        return self._cache.get(key)

    def set(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a value synchronously."""
        self._cache.set(key, value, metadata)

    # Async API
    async def async_get(self, key: str) -> Any | None:
        """Retrieve a cached value asynchronously."""
        return await self._cache.aget(key)

    async def async_set(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a value asynchronously."""
        await self._cache.aset(key, value, metadata)

    def clear(self) -> None:
        self._cache.clear_namespace()

    def get_stats(self) -> Dict[str, Any]:
        return self._cache.get_stats()