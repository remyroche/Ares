from __future__ import annotations

"""
Unified cache module for the Ares project.

This cache consolidates behaviors from:
- SharedMLCache (process-local memory with LRU cleanup)
- IntelligentFeatureCache (memory + on-disk compressed persistence, size-aware)
- CacheManager/Decorators (simple disk persistence and TTL via decorators)

Key features:
- In-memory cache with LRU metadata (last_access, access_count)
- Optional disk persistence with gzip compression
- Optional per-key TTL and default TTL
- Namespaces for logical separation within a single cache directory
- Size estimation for numpy arrays and pandas DataFrames
- Simple async helpers for compatibility with async call sites
"""

import gzip
import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Optional heavy deps; handle gracefully if missing
try:  # numpy
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional
    np = None  # type: ignore

try:  # pandas
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore


class UnifiedCache:
    """Unified cache with in-memory LRU and optional disk persistence."""

    def __init__(
        self,
        cache_dir: str = "data_cache/unified_cache",
        max_memory_mb: int = 2048,
        enable_disk: bool = True,
        enable_compression: bool = True,
        default_ttl_seconds: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.base_dir = Path(cache_dir)
        self.enable_disk = enable_disk
        self.enable_compression = enable_compression
        self.default_ttl_seconds = default_ttl_seconds
        self.max_memory_mb = float(max_memory_mb)
        self.namespace = namespace or "default"

        # Memory structures
        self._memory_cache: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Stats
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

        # Create namespace directory
        self._ns_dir = self.base_dir / self.namespace
        self._ns_dir.mkdir(parents=True, exist_ok=True)

    # --------------- Public API ---------------
    def build_cache_key(self, function_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        """Create a stable hash key from function name and arguments."""
        try:
            key_data = {"function": function_name, "args": self._make_pickle_safe(args), "kwargs": self._make_pickle_safe(kwargs)}
            key_bytes = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            key_bytes = f"{function_name}:{repr(args)}:{repr(sorted(kwargs.items()))}".encode()
        return hashlib.md5(key_bytes).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get value from memory or disk. Respects TTL if present."""
        with self._lock:
            value = self._memory_cache.get(key)
            if value is not None:
                if self._is_expired(key):
                    self._delete_in_memory(key)
                else:
                    self._touch(key)
                    self._hits += 1
                    return value

        # Disk fallback (outside lock for I/O but protect critical sections)
        if self.enable_disk:
            loaded = self._load_from_disk(key)
            if loaded is not None:
                value, metadata = loaded
                with self._lock:
                    if not self._is_expired_metadata(metadata):
                        self._place_in_memory(key, value, metadata)
                        self._hits += 1
                        return value
        with self._lock:
            self._misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        ttl_seconds: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        """Store value in memory and optionally on disk."""
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

        meta = metadata.copy() if metadata else {}
        now = time.time()
        meta.update(
            {
                "created": meta.get("created", now),
                "last_access": now,
                "access_count": meta.get("access_count", 0) + 1,
                "size_mb": self._estimate_size_mb(value),
            }
        )
        if effective_ttl is not None:
            meta["expires_at"] = now + float(effective_ttl)

        with self._lock:
            self._place_in_memory(key, value, meta)

        if persist and self.enable_disk:
            self._save_to_disk(key, value, meta)

    async def aget(self, key: str) -> Any | None:  # Async helper
        return self.get(key)

    async def aset(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        ttl_seconds: Optional[int] = None,
        persist: bool = True,
    ) -> None:  # Async helper
        self.set(key, value, metadata, ttl_seconds=ttl_seconds, persist=persist)

    def clear_namespace(self) -> None:
        """Clear memory and on-disk entries for this namespace."""
        with self._lock:
            self._memory_cache.clear()
            self._metadata.clear()
        if self.enable_disk and self._ns_dir.exists():
            for f in self._ns_dir.glob("*.pkl*"):
                try:
                    f.unlink()
                except Exception as e:  # pragma: no cover - best effort
                    logger.warning(f"Failed to delete cache file {f}: {e}")

    def clear_all(self) -> None:
        """Alias to clear_namespace for the unified cache instance."""
        self.clear_namespace()

    def get_stats(self) -> Dict[str, Any]:
        """Return basic cache statistics for this namespace."""
        disk_usage_mb = 0.0
        if self.enable_disk and self._ns_dir.exists():
            try:
                disk_usage_mb = sum(f.stat().st_size for f in self._ns_dir.glob("*.pkl*")) / (1024 * 1024)
            except Exception:  # pragma: no cover
                disk_usage_mb = 0.0
        with self._lock:
            return {
                "namespace": self.namespace,
                "memory_cache_size": len(self._memory_cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "disk_usage_mb": disk_usage_mb,
                "max_memory_mb": self.max_memory_mb,
                "enable_disk": self.enable_disk,
                "enable_compression": self.enable_compression,
                "default_ttl_seconds": self.default_ttl_seconds,
            }

    # --------------- Internal helpers ---------------
    def _ns_file(self, key: str) -> Path:
        suffix = ".pkl.gz" if self.enable_compression else ".pkl"
        return self._ns_dir / f"{key}{suffix}"

    def _is_expired(self, key: str) -> bool:
        metadata = self._metadata.get(key)
        return self._is_expired_metadata(metadata)

    @staticmethod
    def _is_expired_metadata(metadata: Optional[Dict[str, Any]]) -> bool:
        if not metadata:
            return False
        expires_at = metadata.get("expires_at")
        return bool(expires_at and time.time() > float(expires_at))

    def _touch(self, key: str) -> None:
        md = self._metadata.get(key)
        if md is None:
            md = {}
            self._metadata[key] = md
        md["last_access"] = time.time()
        md["access_count"] = int(md.get("access_count", 0)) + 1

    def _place_in_memory(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        # Evict if necessary
        self._maybe_evict_for(value)
        self._memory_cache[key] = value
        self._metadata[key] = metadata

    def _maybe_evict_for(self, incoming_value: Any) -> None:
        current_mb = self._memory_usage_mb()
        incoming_mb = self._estimate_size_mb(incoming_value)
        if current_mb + incoming_mb <= self.max_memory_mb:
            return
        # Evict up to 20% by LRU
        target_mb = max(0.0, self.max_memory_mb * 0.8 - incoming_mb)
        self._evict_lru_until(target_mb)

    def _evict_lru_until(self, target_free_mb: float) -> None:
        if not self._metadata:
            return
        # Sort keys by last_access ascending (oldest first)
        items = sorted(self._metadata.items(), key=lambda kv: kv[1].get("last_access", 0.0))
        while items and self._memory_usage_mb() > target_free_mb:
            key, _ = items.pop(0)
            self._delete_in_memory(key)
            self._evictions += 1

    def _delete_in_memory(self, key: str) -> None:
        self._memory_cache.pop(key, None)
        self._metadata.pop(key, None)

    def _memory_usage_mb(self) -> float:
        total_bytes = 0
        for v in self._memory_cache.values():
            total_bytes += self._estimate_size_bytes(v)
        return total_bytes / (1024 * 1024)

    @staticmethod
    def _estimate_size_bytes(value: Any) -> int:
        try:
            if pd is not None and isinstance(value, pd.DataFrame):
                return int(value.memory_usage(deep=True).sum())
            if np is not None and isinstance(value, np.ndarray):
                return int(value.nbytes)
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            try:
                return len(pickle.dumps(repr(value)))
            except Exception:
                return 0

    def _estimate_size_mb(self, value: Any) -> float:
        return float(self._estimate_size_bytes(value)) / (1024 * 1024)

    def _save_to_disk(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        file_path = self._ns_file(key)
        payload = {"data": value, "metadata": metadata, "timestamp": time.time()}
        try:
            if self.enable_compression:
                with gzip.open(file_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:  # pragma: no cover - best effort
            logger.warning(f"Failed to save cache file {file_path}: {e}")

    def _load_from_disk(self, key: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        file_path = self._ns_file(key)
        if not file_path.exists():
            return None
        try:
            if self.enable_compression:
                with gzip.open(file_path, "rb") as f:
                    payload = pickle.load(f)
            else:
                with open(file_path, "rb") as f:
                    payload = pickle.load(f)
            return payload.get("data"), payload.get("metadata", {})
        except Exception as e:  # pragma: no cover - best effort
            logger.warning(f"Failed to load cache file {file_path}: {e}")
            return None

    # --------------- Utilities ---------------
    @staticmethod
    def _make_pickle_safe(obj: Any) -> Any:
        try:
            import asyncio
        except Exception:
            asyncio = None  # type: ignore

        if isinstance(obj, dict):
            return {k: UnifiedCache._make_pickle_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            seq = [UnifiedCache._make_pickle_safe(v) for v in obj]
            return type(obj)(seq)
        if asyncio is not None and (hasattr(obj, "__await__") or (callable(obj) and getattr(obj, "__name__", "").startswith("<async"))):
            return f"<async_like_{type(obj).__name__}>"
        return obj


# Convenience factory for callers who only need a simple, namespaced cache
def get_unified_cache(
    *,
    namespace: str,
    cache_dir: str = "data_cache/unified_cache",
    max_memory_mb: int = 2048,
    enable_disk: bool = True,
    enable_compression: bool = True,
    default_ttl_seconds: Optional[int] = None,
) -> UnifiedCache:
    return UnifiedCache(
        cache_dir=cache_dir,
        max_memory_mb=max_memory_mb,
        enable_disk=enable_disk,
        enable_compression=enable_compression,
        default_ttl_seconds=default_ttl_seconds,
        namespace=namespace,
    )


def cached(
    ttl: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
    namespace: str = "decorators",
    use_disk: bool = True,
):
    """Decorator for caching function results via UnifiedCache.

    Supports both sync and async functions. Key selection priority:
    key_func(args, kwargs) > function+args hash.
    """

    def _compute_key(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        if key_func is not None:
            try:
                return str(key_func(*args, **kwargs))
            except Exception as e:
                raise ValueError(f"Custom cache key function failed: {e}")
        uc = get_unified_cache(namespace=namespace)
        return uc.build_cache_key(func.__name__, args, kwargs)

    def decorator(func: Callable) -> Callable:
        import asyncio as _asyncio  # local import to avoid global requirement
        is_async = _asyncio.iscoroutinefunction(func)
        cache = get_unified_cache(namespace=namespace)

        if is_async:
            async def async_wrapper(*f_args: Any, **f_kwargs: Any):
                key = _compute_key(func, f_args, f_kwargs)
                cached_value = await cache.aget(key)
                if cached_value is not None:
                    return cached_value
                result = await func(*f_args, **f_kwargs)
                await cache.aset(key, result, ttl_seconds=ttl, persist=use_disk)
                return result
            return async_wrapper  # type: ignore[return-value]

        def sync_wrapper(*f_args: Any, **f_kwargs: Any):
            key = _compute_key(func, f_args, f_kwargs)
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
            result = func(*f_args, **f_kwargs)
            cache.set(key, result, ttl_seconds=ttl, persist=use_disk)
            return result
        return sync_wrapper  # type: ignore[return-value]

    return decorator

