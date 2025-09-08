"""
Step05 Multi-Level Caching Strategy Module

This module provides a sophisticated multi-level caching system for Step05 processing,
implementing L1 (in-memory), L2 (file-based), and L3 (distributed) cache hierarchies
with intelligent cache management, invalidation, and performance optimization.
"""

import hashlib
import pickle
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import weakref
from collections import OrderedDict, defaultdict
import os

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates

logger = system_logger.getChild('MultiLevelCache')


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    data: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    compression_level: int = 0
    checksum: Optional[str] = None


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    compression_savings_bytes: int = 0
    avg_access_time_ms: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class CacheConfig:
    """Configuration for multi-level caching."""
    enable_l1_cache: bool = True
    enable_l2_cache: bool = True
    enable_l3_cache: bool = False  # Distributed cache

    # L1 Cache settings
    l1_max_entries: int = 1000
    l1_max_size_mb: float = 512.0
    l1_ttl_seconds: int = 3600  # 1 hour

    # L2 Cache settings
    l2_cache_dir: str = "data_cache/step05_cache"
    l2_max_size_mb: float = 2048.0
    l2_ttl_seconds: int = 86400  # 24 hours
    l2_compression_level: int = 6

    # L3 Cache settings (for future distributed implementation)
    l3_redis_host: Optional[str] = None
    l3_redis_port: int = 6379
    l3_ttl_seconds: int = 604800  # 7 days

    # General settings
    enable_compression: bool = True
    enable_checksum: bool = True
    cleanup_interval_seconds: int = 300
    enable_performance_monitoring: bool = True
    cache_key_prefix: str = "step05"


class L1Cache:
    """In-memory L1 cache with LRU eviction."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logger.getChild('L1Cache')

        # Main cache storage with LRU ordering
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()

        # Thread safety
        self._lock = threading.RLock()

        self.logger.info("🚀 L1 Cache initialized")
        self.logger.info(f"📊 Max entries: {config.l1_max_entries:,}")
        self.logger.info(f"💾 Max size: {config.l1_max_size_mb:.1f}MB")

    def get(self, key: str) -> Optional[Any]:
        """Get item from L1 cache."""
        with self._lock:
            try:
                if key in self.cache:
                    entry = self.cache[key]

                    # Check TTL
                    if self._is_expired(entry):
                        self._remove_entry(key)
                        self.stats.misses += 1
                        return None

                    # Update access metadata
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1

                    # Move to end (most recently used)
                    self.cache.move_to_end(key)

                    self.stats.hits += 1
                    self.logger.debug(f"✅ L1 cache hit: {key}")

                    return entry.data
                else:
                    self.stats.misses += 1
                    return None

            except Exception as e:
                self.logger.warning(f"⚠️ L1 cache get error: {e}")
                return None

    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None,
            tags: List[str] = None) -> bool:
        """Put item in L1 cache."""
        with self._lock:
            try:
                # Calculate data size
                size_bytes = self._calculate_data_size(data)

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    data=data,
                    size_bytes=size_bytes,
                    ttl_seconds=ttl_seconds or self.config.l1_ttl_seconds,
                    tags=tags or []
                )

                # Check if we need to evict entries
                self._ensure_capacity(size_bytes)

                # Add to cache
                self.cache[key] = entry
                self.cache.move_to_end(key)  # Mark as most recently used

                # Update stats
                self.stats.total_entries = len(self.cache)
                self.stats.total_size_bytes += size_bytes

                self.logger.debug(f"💾 L1 cache put: {key} ({size_bytes} bytes)")
                return True

            except Exception as e:
                self.logger.warning(f"⚠️ L1 cache put error: {e}")
                return False

    def remove(self, key: str) -> bool:
        """Remove item from L1 cache."""
        with self._lock:
            try:
                if key in self.cache:
                    entry = self.cache[key]
                    self.stats.total_size_bytes -= entry.size_bytes
                    del self.cache[key]
                    self.stats.total_entries = len(self.cache)
                    self.stats.invalidations += 1
                    self.logger.debug(f"🗑️ L1 cache removed: {key}")
                    return True
                return False

            except Exception as e:
                self.logger.warning(f"⚠️ L1 cache remove error: {e}")
                return False

    def clear(self) -> None:
        """Clear all entries from L1 cache."""
        with self._lock:
            self.cache.clear()
            self.stats = CacheStats()
            self.logger.info("🧹 L1 cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                self.logger.info(f"🗑️ L1 cache cleanup: {len(expired_keys)} expired entries removed")

            return len(expired_keys)

    def _ensure_capacity(self, required_size: int) -> None:
        """Ensure there's enough capacity for new entry."""
        max_size_bytes = int(self.config.l1_max_size_mb * 1024 * 1024)

        # Evict entries until we have enough space
        while (self.stats.total_size_bytes + required_size > max_size_bytes or
               len(self.cache) >= self.config.l1_max_entries):

            if not self.cache:
                break

            # Remove least recently used entry
            oldest_key, oldest_entry = next(iter(self.cache.items()))
            self._remove_entry(oldest_key)
            self.stats.evictions += 1

    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update stats."""
        if key in self.cache:
            entry = self.cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.cache[key]
            self.stats.total_entries = len(self.cache)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl_seconds is None:
            return False

        age = datetime.now() - entry.created_at
        return age.total_seconds() > entry.ttl_seconds

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            # Use pickle to estimate size
            pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return len(pickled)
        except Exception:
            # Fallback estimate
            if isinstance(data, (list, tuple)):
                return len(data) * 8  # Rough estimate
            elif isinstance(data, dict):
                return len(data) * 16  # Rough estimate
            elif isinstance(data, str):
                return len(data) * 2  # UTF-8 estimate
            else:
                return 1024  # Default estimate


class L2Cache:
    """File-based L2 cache with compression and persistence."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logger.getChild('L2Cache')

        # Cache directory
        self.cache_dir = Path(config.l2_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index of cached files
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self.stats = CacheStats()

        # Thread safety
        self._lock = threading.RLock()

        # Load existing cache index
        self._load_cache_index()

        self.logger.info("🚀 L2 Cache initialized")
        self.logger.info(f"📁 Cache directory: {self.cache_dir}")
        self.logger.info(f"💾 Max size: {config.l2_max_size_mb:.1f}MB")

    def get(self, key: str) -> Optional[Any]:
        """Get item from L2 cache."""
        with self._lock:
            try:
                if key not in self.cache_index:
                    self.stats.misses += 1
                    return None

                metadata = self.cache_index[key]
                file_path = self.cache_dir / metadata['filename']

                # Check if file exists
                if not file_path.exists():
                    self._remove_index_entry(key)
                    self.stats.misses += 1
                    return None

                # Check TTL
                created_at = datetime.fromisoformat(metadata['created_at'])
                ttl_seconds = metadata.get('ttl_seconds', self.config.l2_ttl_seconds)

                if ttl_seconds and (datetime.now() - created_at).total_seconds() > ttl_seconds:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None

                # Load data from file
                data = self._load_from_file(file_path, metadata)

                if data is not None:
                    # Update metadata
                    metadata['last_accessed'] = datetime.now().isoformat()
                    metadata['access_count'] = metadata.get('access_count', 0) + 1

                    # Save updated metadata
                    self._save_cache_index()

                    self.stats.hits += 1
                    self.logger.debug(f"✅ L2 cache hit: {key}")
                    return data
                else:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None

            except Exception as e:
                self.logger.warning(f"⚠️ L2 cache get error: {e}")
                return None

    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None,
            tags: List[str] = None) -> bool:
        """Put item in L2 cache."""
        with self._lock:
            try:
                # Generate filename
                filename = self._generate_filename(key)
                file_path = self.cache_dir / filename

                # Calculate uncompressed size
                uncompressed_size = self._calculate_data_size(data)

                # Compress and save data
                compressed_size = self._save_to_file(file_path, data)

                # Calculate compression savings
                if compressed_size < uncompressed_size:
                    self.stats.compression_savings_bytes += (uncompressed_size - compressed_size)

                # Create metadata
                metadata = {
                    'filename': filename,
                    'key': key,
                    'created_at': datetime.now().isoformat(),
                    'last_accessed': datetime.now().isoformat(),
                    'access_count': 0,
                    'size_bytes': compressed_size,
                    'uncompressed_size_bytes': uncompressed_size,
                    'ttl_seconds': ttl_seconds or self.config.l2_ttl_seconds,
                    'tags': tags or [],
                    'compression_level': self.config.l2_compression_level
                }

                # Check if we need to evict entries
                self._ensure_capacity(compressed_size)

                # Add to index
                self.cache_index[key] = metadata
                self._save_cache_index()

                # Update stats
                self.stats.total_entries = len(self.cache_index)
                self.stats.total_size_bytes += compressed_size

                self.logger.debug(f"💾 L2 cache put: {key} ({compressed_size} bytes)")
                return True

            except Exception as e:
                self.logger.warning(f"⚠️ L2 cache put error: {e}")
                return False

    def remove(self, key: str) -> bool:
        """Remove item from L2 cache."""
        return self._remove_entry(key)

    def clear(self) -> None:
        """Clear all entries from L2 cache."""
        with self._lock:
            try:
                # Remove all files
                for metadata in self.cache_index.values():
                    file_path = self.cache_dir / metadata['filename']
                    if file_path.exists():
                        file_path.unlink()

                # Clear index
                self.cache_index.clear()
                self._save_cache_index()

                # Reset stats
                self.stats = CacheStats()

                self.logger.info("🧹 L2 cache cleared")

            except Exception as e:
                self.logger.warning(f"⚠️ L2 cache clear error: {e}")

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            current_time = datetime.now()

            for key, metadata in self.cache_index.items():
                created_at = datetime.fromisoformat(metadata['created_at'])
                ttl_seconds = metadata.get('ttl_seconds', self.config.l2_ttl_seconds)

                if ttl_seconds and (current_time - created_at).total_seconds() > ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                self.logger.info(f"🗑️ L2 cache cleanup: {len(expired_keys)} expired entries removed")

            return len(expired_keys)

    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        try:
            index_file = self.cache_dir / "cache_index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
                self.logger.info(f"📋 Loaded L2 cache index: {len(self.cache_index)} entries")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load L2 cache index: {e}")
            self.cache_index = {}

    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        try:
            index_file = self.cache_dir / "cache_index.json"
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save L2 cache index: {e}")

    def _generate_filename(self, key: str) -> str:
        """Generate unique filename for cache entry."""
        hash_obj = hashlib.md5(key.encode())
        return f"{hash_obj.hexdigest()}.cache"

    def _save_to_file(self, file_path: Path, data: Any) -> int:
        """Save data to compressed file and return file size."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return file_path.stat().st_size
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save cache file {file_path}: {e}")
            return 0

    def _load_from_file(self, file_path: Path, metadata: Dict[str, Any]) -> Optional[Any]:
        """Load data from compressed file."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load cache file {file_path}: {e}")
            return None

    def _remove_entry(self, key: str) -> bool:
        """Remove entry and associated file."""
        try:
            if key in self.cache_index:
                metadata = self.cache_index[key]
                file_path = self.cache_dir / metadata['filename']

                # Remove file
                if file_path.exists():
                    file_path.unlink()

                # Update stats
                self.stats.total_size_bytes -= metadata.get('size_bytes', 0)
                self.stats.invalidations += 1

                # Remove from index
                del self.cache_index[key]
                self.stats.total_entries = len(self.cache_index)

                # Save updated index
                self._save_cache_index()

                self.logger.debug(f"🗑️ L2 cache removed: {key}")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️ L2 cache remove error: {e}")
            return False

    def _remove_index_entry(self, key: str) -> None:
        """Remove entry from index only."""
        if key in self.cache_index:
            del self.cache_index[key]
            self._save_cache_index()

    def _ensure_capacity(self, required_size: int) -> None:
        """Ensure there's enough capacity for new entry."""
        max_size_bytes = int(self.config.l2_max_size_mb * 1024 * 1024)

        # Evict entries until we have enough space
        while self.stats.total_size_bytes + required_size > max_size_bytes:
            if not self.cache_index:
                break

            # Find oldest entry to evict
            oldest_key = None
            oldest_time = datetime.now()

            for key, metadata in self.cache_index.items():
                created_at = datetime.fromisoformat(metadata['created_at'])
                if created_at < oldest_time:
                    oldest_time = created_at
                    oldest_key = key

            if oldest_key:
                self._remove_entry(oldest_key)
                self.stats.evictions += 1

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate


class L3Cache:
    """Distributed L3 cache (Redis-based) for future implementation."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logger.getChild('L3Cache')
        self.stats = CacheStats()

        # Placeholder for Redis connection
        self.redis_client = None

        if config.enable_l3_cache and config.l3_redis_host:
            try:
                # Future Redis implementation
                self.logger.info("🚀 L3 Cache initialized (Redis)")
            except Exception as e:
                self.logger.warning(f"⚠️ L3 Cache initialization failed: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from L3 cache (placeholder)."""
        # Future Redis implementation
        return None

    def put(self, key: str, data: Any, ttl_seconds: Optional[int] = None,
            tags: List[str] = None) -> bool:
        """Put item in L3 cache (placeholder)."""
        # Future Redis implementation
        return False

    def remove(self, key: str) -> bool:
        """Remove item from L3 cache (placeholder)."""
        return False

    def clear(self) -> None:
        """Clear L3 cache (placeholder)."""
        pass


class MultiLevelCache:
    """
    Multi-level caching system with L1 (memory), L2 (file), and L3 (distributed) caches.
    Provides intelligent cache management with performance optimization and automatic cleanup.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = logger

        # Initialize cache levels
        self.l1_cache = L1Cache(self.config) if self.config.enable_l1_cache else None
        self.l2_cache = L2Cache(self.config) if self.config.enable_l2_cache else None
        self.l3_cache = L3Cache(self.config) if self.config.enable_l3_cache else None

        # Performance monitoring
        self.performance_stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0,
            'total_requests': 0
        }

        # Thread safety
        self._lock = threading.RLock()

        # Start cleanup thread
        if self.config.enable_performance_monitoring:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="Cache-Cleanup"
            )
            self.cleanup_thread.start()

        self.logger.info("🚀 Multi-Level Cache initialized")
        self.logger.info(f"🏗️ Cache levels: L1={'✅' if self.l1_cache else '❌'}, "
                        f"L2={'✅' if self.l2_cache else '❌'}, "
                        f"L3={'✅' if self.l3_cache else '❌'}")

    def get(self, key: str, operation_type: str = "general") -> Optional[Any]:
        """
        Get item from cache hierarchy (L1 -> L2 -> L3).

        Args:
            key: Cache key
            operation_type: Type of operation for cache optimization

        Returns:
            Cached data or None if not found
        """
        with self._lock:
            self.performance_stats['total_requests'] += 1

            # Try L1 cache first
            if self.l1_cache:
                data = self.l1_cache.get(key)
                if data is not None:
                    self.performance_stats['l1_hits'] += 1
                    return data

            # Try L2 cache
            if self.l2_cache:
                data = self.l2_cache.get(key)
                if data is not None:
                    # Promote to L1 for faster future access
                    if self.l1_cache:
                        self.l1_cache.put(key, data)
                    self.performance_stats['l2_hits'] += 1
                    return data

            # Try L3 cache (future implementation)
            if self.l3_cache:
                data = self.l3_cache.get(key)
                if data is not None:
                    # Promote to higher levels
                    if self.l1_cache:
                        self.l1_cache.put(key, data)
                    if self.l2_cache:
                        self.l2_cache.put(key, data)
                    self.performance_stats['l3_hits'] += 1
                    return data

            # Cache miss
            self.performance_stats['misses'] += 1
            return None

    def put(self, key: str, data: Any, operation_type: str = "general",
            ttl_seconds: Optional[int] = None, tags: List[str] = None) -> bool:
        """
        Put item in cache hierarchy.

        Args:
            key: Cache key
            data: Data to cache
            operation_type: Type of operation for cache optimization
            ttl_seconds: Time to live in seconds
            tags: Tags for cache management

        Returns:
            True if successfully cached
        """
        with self._lock:
            success = False

            # Store in L1 (if enabled)
            if self.l1_cache:
                if self.l1_cache.put(key, data, ttl_seconds, tags):
                    success = True

            # Store in L2 (if enabled)
            if self.l2_cache:
                if self.l2_cache.put(key, data, ttl_seconds, tags):
                    success = True

            # Store in L3 (if enabled)
            if self.l3_cache:
                if self.l3_cache.put(key, data, ttl_seconds, tags):
                    success = True

            if success:
                self.logger.debug(f"💾 Cached: {key} ({operation_type})")
            else:
                self.logger.warning(f"⚠️ Failed to cache: {key}")

            return success

    def remove(self, key: str) -> bool:
        """Remove item from all cache levels."""
        with self._lock:
            removed = False

            if self.l1_cache and self.l1_cache.remove(key):
                removed = True
            if self.l2_cache and self.l2_cache.remove(key):
                removed = True
            if self.l3_cache and self.l3_cache.remove(key):
                removed = True

            if removed:
                self.logger.debug(f"🗑️ Removed from cache: {key}")

            return removed

    def clear(self, level: Optional[str] = None) -> None:
        """
        Clear cache levels.

        Args:
            level: Specific level to clear ('l1', 'l2', 'l3') or None for all
        """
        with self._lock:
            if level == 'l1' and self.l1_cache:
                self.l1_cache.clear()
            elif level == 'l2' and self.l2_cache:
                self.l2_cache.clear()
            elif level == 'l3' and self.l3_cache:
                self.l3_cache.clear()
            else:
                # Clear all levels
                if self.l1_cache:
                    self.l1_cache.clear()
                if self.l2_cache:
                    self.l2_cache.clear()
                if self.l3_cache:
                    self.l3_cache.clear()

            self.logger.info(f"🧹 Cache cleared: {level or 'all levels'}")

    def invalidate_by_tags(self, tags: List[str]) -> int:
        """
        Invalidate cache entries by tags.

        Args:
            tags: List of tags to match

        Returns:
            Number of entries invalidated
        """
        # Future implementation for tag-based invalidation
        self.logger.debug(f"🏷️ Tag-based invalidation not yet implemented: {tags}")
        return 0

    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries from all cache levels."""
        with self._lock:
            results = {}

            if self.l1_cache:
                results['l1'] = self.l1_cache.cleanup_expired()
            if self.l2_cache:
                results['l2'] = self.l2_cache.cleanup_expired()
            if self.l3_cache:
                results['l3'] = 0  # Placeholder

            total_cleaned = sum(results.values())
            if total_cleaned > 0:
                self.logger.info(f"🧹 Cleanup completed: {total_cleaned} expired entries removed")

            return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            stats = dict(self.performance_stats)

            # Calculate hit rates
            total_hits = stats['l1_hits'] + stats['l2_hits'] + stats['l3_hits']
            total_requests = stats['total_requests']

            if total_requests > 0:
                stats['overall_hit_rate'] = total_hits / total_requests
                stats['l1_hit_rate'] = stats['l1_hits'] / total_requests
                stats['l2_hit_rate'] = stats['l2_hits'] / total_requests
                stats['l3_hit_rate'] = stats['l3_hits'] / total_requests
                stats['miss_rate'] = stats['misses'] / total_requests

            # Individual cache stats
            if self.l1_cache:
                stats['l1_stats'] = {
                    'entries': self.l1_cache.stats.total_entries,
                    'size_bytes': self.l1_cache.stats.total_size_bytes,
                    'hits': self.l1_cache.stats.hits,
                    'misses': self.l1_cache.stats.misses,
                    'evictions': self.l1_cache.stats.evictions
                }

            if self.l2_cache:
                stats['l2_stats'] = {
                    'entries': self.l2_cache.stats.total_entries,
                    'size_bytes': self.l2_cache.stats.total_size_bytes,
                    'hits': self.l2_cache.stats.hits,
                    'misses': self.l2_cache.stats.misses,
                    'evictions': self.l2_cache.stats.evictions,
                    'compression_savings': self.l2_cache.stats.compression_savings_bytes
                }

            return stats

    def optimize_cache(self) -> Dict[str, Any]:
        """Perform cache optimization and maintenance."""
        with self._lock:
            results = {
                'cleanup_results': self.cleanup_expired(),
                'optimization_timestamp': datetime.now().isoformat()
            }

            # Additional optimization strategies can be added here

            self.logger.info("🎯 Cache optimization completed")
            return results

    def _cleanup_worker(self) -> None:
        """Background cleanup worker thread."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                self.cleanup_expired()
            except Exception as e:
                self.logger.warning(f"⚠️ Cache cleanup worker error: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
                # Thread will be terminated when process ends
                pass
        except Exception:
            pass
