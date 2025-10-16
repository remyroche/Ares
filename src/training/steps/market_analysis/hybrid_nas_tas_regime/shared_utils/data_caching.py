"""
Data Caching Utilities for Hybrid NAS-TAS Regime Detection.

Provides comprehensive data caching and persistence utilities
using existing serialization utils for efficient data management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
from enum import Enum
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing utilities
try:
    from src.utils.serialization_utils import (
        save_dataframe, load_dataframe, save_object, load_object,
        get_serialization_manager
    )
    SERIALIZATION_UTILS_AVAILABLE = True
except ImportError:
    SERIALIZATION_UTILS_AVAILABLE = False

try:
    from src.utils.common_operations import (
        get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategies available."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based
    MANUAL = "manual"  # Manual control

@dataclass
class CacheConfig:
    """Configuration for data caching."""
    cache_dir: str = "cache"
    max_cache_size_mb: float = 1000.0
    max_items: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    strategy: CacheStrategy = CacheStrategy.LRU
    compression: bool = True
    encryption: bool = False
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    auto_cleanup: bool = True
    cleanup_interval_seconds: int = 300  # 5 minutes

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_expires: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CacheResult:
    """Result from cache operations."""
    success: bool
    data: Optional[Any] = None
    cache_hit: bool = False
    operation_time: float = 0.0
    cache_size_mb: float = 0.0
    cache_items: int = 0
    error_message: Optional[str] = None

class DataCache:
    """Advanced data cache with hardware acceleration and multiple strategies."""

    def __init__(self, config: CacheConfig):
        """Initialize the data cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize cache directory
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache storage
        self.cache_storage: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.access_counts: Dict[str, int] = {}  # For LFU

        # Initialize hardware acceleration if available
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for data caching")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for data caching")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        # Initialize serialization manager if available
        self.serialization_manager = None
        if SERIALIZATION_UTILS_AVAILABLE:
            try:
                self.serialization_manager = get_serialization_manager()
                self.logger.info("✅ Serialization manager initialized for data caching")
            except Exception as e:
                self.logger.warning(f"⚠️ Serialization manager not available: {e}")

        # Start auto cleanup if enabled
        if config.auto_cleanup:
            self._start_auto_cleanup()

        self.logger.info("✅ Data Cache initialized")
        self.logger.info(f"   Cache directory: {self.cache_dir}")
        self.logger.info(f"   Max size: {config.max_cache_size_mb}MB")
        self.logger.info(f"   Strategy: {config.strategy.value}")

    def get(self, key: str) -> CacheResult:
        """Get data from cache.

        Args:
            key: Cache key

        Returns:
            CacheResult with data if found
        """
        start_time = time.time()

        try:
            # Check if key exists in memory cache
            if key in self.cache_storage:
                entry = self.cache_storage[key]

                # Check TTL if applicable
                if entry.ttl_expires and datetime.now() > entry.ttl_expires:
                    self.logger.info(f"🗑️ Cache entry expired: {key}")
                    self._remove_entry(key)
                    return CacheResult(
                        success=False,
                        cache_hit=False,
                        operation_time=time.time() - start_time,
                        cache_size_mb=self._get_cache_size_mb(),
                        cache_items=len(self.cache_storage)
                    )

                # Update access information
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.access_counts[key] = self.access_counts.get(key, 0) + 1

                # Update access order for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                operation_time = time.time() - start_time

                self.logger.info(f"✅ Cache hit: {key}")

                return CacheResult(
                    success=True,
                    data=entry.data,
                    cache_hit=True,
                    operation_time=operation_time,
                    cache_size_mb=self._get_cache_size_mb(),
                    cache_items=len(self.cache_storage)
                )

            # Try to load from disk cache
            disk_result = self._load_from_disk(key)
            if disk_result.success:
                # Store in memory cache
                self._store_in_memory(key, disk_result.data)
                return disk_result

            # Cache miss
            operation_time = time.time() - start_time

            self.logger.info(f"❌ Cache miss: {key}")

            return CacheResult(
                success=False,
                cache_hit=False,
                operation_time=operation_time,
                cache_size_mb=self._get_cache_size_mb(),
                cache_items=len(self.cache_storage)
            )

        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error(f"❌ Cache get failed for {key}: {e}")

            return CacheResult(
                success=False,
                operation_time=operation_time,
                error_message=str(e)
            )

    def put(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> CacheResult:
        """Store data in cache.

        Args:
            key: Cache key
            data: Data to store
            metadata: Optional metadata

        Returns:
            CacheResult indicating success
        """
        start_time = time.time()

        try:
            # Calculate data size
            size_bytes = self._calculate_data_size(data)

            # Check if we need to make space
            if self._should_evict(size_bytes):
                self._evict_entries(size_bytes)

            # Create cache entry
            now = datetime.now()
            ttl_expires = None
            if self.config.strategy == CacheStrategy.TTL:
                ttl_expires = now + timedelta(seconds=self.config.ttl_seconds)

            entry = CacheEntry(
                key=key,
                data=data,
                created_at=now,
                last_accessed=now,
                access_count=1,
                size_bytes=size_bytes,
                ttl_expires=ttl_expires,
                metadata=metadata
            )

            # Store in memory
            self._store_in_memory(key, data, entry)

            # Store on disk if serialization is available
            if self.serialization_manager:
                self._store_on_disk(key, data, metadata)

            operation_time = time.time() - start_time

            self.logger.info(f"✅ Data cached: {key} ({size_bytes} bytes)")

            return CacheResult(
                success=True,
                operation_time=operation_time,
                cache_size_mb=self._get_cache_size_mb(),
                cache_items=len(self.cache_storage)
            )

        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error(f"❌ Cache put failed for {key}: {e}")

            return CacheResult(
                success=False,
                operation_time=operation_time,
                error_message=str(e)
            )

    def remove(self, key: str) -> CacheResult:
        """Remove data from cache.

        Args:
            key: Cache key to remove

        Returns:
            CacheResult indicating success
        """
        start_time = time.time()

        try:
            removed = self._remove_entry(key)

            # Remove from disk if exists
            disk_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            if disk_file.exists():
                disk_file.unlink()

            operation_time = time.time() - start_time

            if removed:
                self.logger.info(f"✅ Cache entry removed: {key}")
                return CacheResult(
                    success=True,
                    operation_time=operation_time,
                    cache_size_mb=self._get_cache_size_mb(),
                    cache_items=len(self.cache_storage)
                )
            else:
                self.logger.info(f"⚠️ Cache entry not found: {key}")
                return CacheResult(
                    success=False,
                    operation_time=operation_time,
                    cache_size_mb=self._get_cache_size_mb(),
                    cache_items=len(self.cache_storage)
                )

        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error(f"❌ Cache remove failed for {key}: {e}")

            return CacheResult(
                success=False,
                operation_time=operation_time,
                error_message=str(e)
            )

    def clear(self) -> CacheResult:
        """Clear all cache data.

        Returns:
            CacheResult indicating success
        """
        start_time = time.time()

        try:
            # Clear memory cache
            self.cache_storage.clear()
            self.access_order.clear()
            self.access_counts.clear()

            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()

            operation_time = time.time() - start_time

            self.logger.info("✅ Cache cleared")

            return CacheResult(
                success=True,
                operation_time=operation_time,
                cache_size_mb=0.0,
                cache_items=0
            )

        except Exception as e:
            operation_time = time.time() - start_time
            self.logger.error(f"❌ Cache clear failed: {e}")

            return CacheResult(
                success=False,
                operation_time=operation_time,
                error_message=str(e)
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        try:
            total_size_bytes = sum(entry.size_bytes for entry in self.cache_storage.values())
            total_accesses = sum(entry.access_count for entry in self.cache_storage.values())

            stats = {
                'total_items': len(self.cache_storage),
                'total_size_mb': total_size_bytes / (1024 * 1024),
                'max_size_mb': self.config.max_cache_size_mb,
                'max_items': self.config.max_items,
                'total_accesses': total_accesses,
                'average_accesses_per_item': total_accesses / len(self.cache_storage) if self.cache_storage else 0,
                'strategy': self.config.strategy.value,
                'compression_enabled': self.config.compression,
                'encryption_enabled': self.config.encryption,
                'auto_cleanup_enabled': self.config.auto_cleanup
            }

            # Add strategy-specific stats
            if self.config.strategy == CacheStrategy.LRU:
                stats['lru_order'] = self.access_order[-10:]  # Last 10 accessed
            elif self.config.strategy == CacheStrategy.LFU:
                stats['lfu_counts'] = dict(sorted(self.access_counts.items(), key=lambda x: x[1], reverse=True)[:10])

            return stats

        except Exception as e:
            self.logger.warning(f"⚠️ Statistics calculation failed: {e}")
            return {'error': str(e)}

    def _store_in_memory(self, key: str, data: Any, entry: Optional[CacheEntry] = None):
        """Store data in memory cache."""
        try:
            if entry is None:
                now = datetime.now()
                entry = CacheEntry(
                    key=key,
                    data=data,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    size_bytes=self._calculate_data_size(data)
                )

            self.cache_storage[key] = entry
            self.access_counts[key] = 1

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

        except Exception as e:
            self.logger.warning(f"⚠️ Memory storage failed for {key}: {e}")

    def _store_on_disk(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Store data on disk cache."""
        try:
            if not self.serialization_manager:
                return

            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"

            # Create cache metadata
            cache_metadata = {
                'key': key,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }

            # Save data and metadata
            if isinstance(data, pd.DataFrame):
                save_dataframe(data, str(cache_file))
            else:
                save_object(data, str(cache_file))

            # Save metadata
            metadata_file = self.cache_dir / f"{self._hash_key(key)}.meta"
            with open(metadata_file, 'w') as f:
                json.dump(cache_metadata, f, indent=2)

        except Exception as e:
            self.logger.warning(f"⚠️ Disk storage failed for {key}: {e}")

    def _load_from_disk(self, key: str) -> CacheResult:
        """Load data from disk cache."""
        try:
            if not self.serialization_manager:
                return CacheResult(success=False)

            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            metadata_file = self.cache_dir / f"{self._hash_key(key)}.meta"

            if not cache_file.exists():
                return CacheResult(success=False)

            # Load metadata
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

            # Load data
            try:
                data = load_dataframe(str(cache_file))
            except:
                data = load_object(str(cache_file))

            return CacheResult(
                success=True,
                data=data,
                cache_hit=True
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Disk load failed for {key}: {e}")
            return CacheResult(success=False, error_message=str(e))

    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        try:
            if key in self.cache_storage:
                del self.cache_storage[key]

            if key in self.access_order:
                self.access_order.remove(key)

            if key in self.access_counts:
                del self.access_counts[key]

            return True

        except Exception as e:
            self.logger.warning(f"⚠️ Entry removal failed for {key}: {e}")
            return False

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate data size in bytes."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (list, dict)):
                return len(str(data).encode('utf-8'))
            else:
                return len(str(data).encode('utf-8'))

        except Exception:
            return 0

    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        try:
            total_bytes = sum(entry.size_bytes for entry in self.cache_storage.values())
            return total_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def _should_evict(self, new_size_bytes: int) -> bool:
        """Check if we need to evict entries."""
        try:
            current_size_mb = self._get_cache_size_mb()
            new_size_mb = new_size_bytes / (1024 * 1024)

            # Check size limit
            if current_size_mb + new_size_mb > self.config.max_cache_size_mb:
                return True

            # Check item count limit
            if len(self.cache_storage) >= self.config.max_items:
                return True

            return False

        except Exception:
            return True

    def _evict_entries(self, required_size_bytes: int):
        """Evict entries based on strategy."""
        try:
            required_size_mb = required_size_bytes / (1024 * 1024)
            current_size_mb = self._get_cache_size_mb()

            # Calculate how much to free
            target_size_mb = max(0, current_size_mb + required_size_mb - self.config.max_cache_size_mb * 0.8)

            if self.config.strategy == CacheStrategy.LRU:
                self._evict_lru(target_size_mb)
            elif self.config.strategy == CacheStrategy.LFU:
                self._evict_lfu(target_size_mb)
            elif self.config.strategy == CacheStrategy.SIZE:
                self._evict_largest(target_size_mb)
            else:
                self._evict_oldest(target_size_mb)

        except Exception as e:
            self.logger.warning(f"⚠️ Eviction failed: {e}")

    def _evict_lru(self, target_size_mb: float):
        """Evict least recently used entries."""
        try:
            freed_size_mb = 0.0

            while self.access_order and freed_size_mb < target_size_mb:
                key = self.access_order[0]
                if key in self.cache_storage:
                    entry = self.cache_storage[key]
                    freed_size_mb += entry.size_bytes / (1024 * 1024)
                    self._remove_entry(key)
                else:
                    self.access_order.pop(0)

        except Exception as e:
            self.logger.warning(f"⚠️ LRU eviction failed: {e}")

    def _evict_lfu(self, target_size_mb: float):
        """Evict least frequently used entries."""
        try:
            freed_size_mb = 0.0

            # Sort by access count
            sorted_items = sorted(self.access_counts.items(), key=lambda x: x[1])

            for key, _ in sorted_items:
                if freed_size_mb >= target_size_mb:
                    break

                if key in self.cache_storage:
                    entry = self.cache_storage[key]
                    freed_size_mb += entry.size_bytes / (1024 * 1024)
                    self._remove_entry(key)

        except Exception as e:
            self.logger.warning(f"⚠️ LFU eviction failed: {e}")

    def _evict_largest(self, target_size_mb: float):
        """Evict largest entries."""
        try:
            freed_size_mb = 0.0

            # Sort by size
            sorted_entries = sorted(
                self.cache_storage.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )

            for key, _ in sorted_entries:
                if freed_size_mb >= target_size_mb:
                    break

                entry = self.cache_storage[key]
                freed_size_mb += entry.size_bytes / (1024 * 1024)
                self._remove_entry(key)

        except Exception as e:
            self.logger.warning(f"⚠️ Largest eviction failed: {e}")

    def _evict_oldest(self, target_size_mb: float):
        """Evict oldest entries."""
        try:
            freed_size_mb = 0.0

            # Sort by creation time
            sorted_entries = sorted(
                self.cache_storage.items(),
                key=lambda x: x[1].created_at
            )

            for key, _ in sorted_entries:
                if freed_size_mb >= target_size_mb:
                    break

                entry = self.cache_storage[key]
                freed_size_mb += entry.size_bytes / (1024 * 1024)
                self._remove_entry(key)

        except Exception as e:
            self.logger.warning(f"⚠️ Oldest eviction failed: {e}")

    def _hash_key(self, key: str) -> str:
        """Create hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _start_auto_cleanup(self):
        """Start automatic cleanup process."""
        try:
            import threading

            def cleanup_worker():
                while True:
                    time.sleep(self.config.cleanup_interval_seconds)
                    self._cleanup_expired()

            cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            cleanup_thread.start()

        except Exception as e:
            self.logger.warning(f"⚠️ Auto cleanup start failed: {e}")

    def _cleanup_expired(self):
        """Clean up expired entries."""
        try:
            now = datetime.now()
            expired_keys = []

            for key, entry in self.cache_storage.items():
                if entry.ttl_expires and now > entry.ttl_expires:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                self.logger.info(f"🧹 Cleaned up {len(expired_keys)} expired entries")

        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")

def create_data_cache(config: Optional[CacheConfig] = None) -> DataCache:
    """Create a data cache instance.

    Args:
        config: Optional cache configuration

    Returns:
        DataCache instance
    """
    if config is None:
        config = CacheConfig()
    return DataCache(config)

def quick_cache(data: Any, key: str,
                cache_dir: str = "cache",
                max_size_mb: float = 100.0) -> CacheResult:
    """Quick data caching with default settings.

    Args:
        data: Data to cache
        key: Cache key
        cache_dir: Cache directory
        max_size_mb: Maximum cache size in MB

    Returns:
        CacheResult
    """
    config = CacheConfig(
        cache_dir=cache_dir,
        max_cache_size_mb=max_size_mb
    )
    cache = DataCache(config)
    return cache.put(key, data)
