from __future__ import annotations

from typing import Dict, Optional, Any
import asyncio
import logging

from src.utils.unified_cache import UnifiedCache, cached

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
        return self._cache.get(key)

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._cache.set(key, value, metadata)

    async def get(self, key: str) -> Any | None:
        return await self._cache.aget(key)

    async def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._cache.aset(key, value, metadata)

    def clear(self) -> None:
        self._cache.clear_namespace()

    def get_stats(self) -> Dict[str, Any]:
        return self._cache.get_stats()

# Export the cached decorator as intelligent_caching for backwards compatibility
intelligent_caching = cached
