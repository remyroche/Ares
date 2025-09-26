"""CLVSA Element Caching System for Efficient Reuse Across ML Models.

This module provides a comprehensive caching system for CLVSA (Cross-View Learning with Self-Attention)
elements, allowing computed features, attention weights, and predictions to be shared across
different machine learning models to improve efficiency and reduce computational overhead.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class CLVSACacheEntry:
    """Cache entry for CLVSA elements."""
    key: str
    features: Dict[str, torch.Tensor]
    predictions: torch.Tensor
    attention_weights: Dict[str, np.ndarray]
    market_data_hash: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    memory_size: int = 0


@dataclass
class CLVSACacheConfig:
    """Configuration for CLVSA caching system."""
    max_cache_size: int = 100  # Maximum number of entries
    max_memory_mb: float = 500.0  # Maximum memory usage in MB
    ttl_seconds: int = 3600  # Time to live for cache entries (1 hour)
    enable_persistence: bool = True
    cache_dir: Optional[str] = None
    cleanup_interval: int = 300  # Cleanup every 5 minutes
    enable_compression: bool = True
    compression_level: int = 6


class CLVSACacheManager:
    """Manager for CLVSA element caching with memory and disk optimization."""

    def __init__(self, config: CLVSACacheConfig):
        self.config = config
        self.cache: Dict[str, CLVSACacheEntry] = {}
        self.lock = threading.RLock()

        # Memory tracking
        self.current_memory_usage = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

        # GPU memory tracking
        self.gpu_memory_usage = 0
        self.gpu_cache: Dict[str, Any] = {}  # Store GPU tensors when available

        # Persistence
        if config.enable_persistence:
            self.cache_dir = Path(config.cache_dir or "./clvsa_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Memory optimization
        self.memory_pool: Dict[str, torch.Tensor] = {}
        self.tensor_references: Dict[str, weakref.WeakSet] = {}

        # Background cleanup
        self.cleanup_thread: Optional[threading.Thread] = None
        self.stop_cleanup = threading.Event()

        # Initialize GPU if available
        self._init_gpu_support()

        self._start_cleanup_thread()

        logger.info(f"🔄 CLVSA Cache Manager initialized (max_size: {config.max_cache_size}, max_memory: {config.max_memory_mb}MB)")

    def _init_gpu_support(self):
        """Initialize GPU support for memory optimization."""
        try:
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_device = torch.device('cuda')
                logger.info(f"✅ GPU support enabled for CLVSA cache (device: {torch.cuda.get_device_name()})")
            else:
                self.gpu_available = False
                self.gpu_device = None
                logger.info("⚠️ GPU not available, using CPU-only caching")
        except Exception as e:
            self.gpu_available = False
            self.gpu_device = None
            logger.warning(f"GPU initialization failed: {e}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self.config.cleanup_interval > 0:
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            logger.info("✅ CLVSA cache cleanup thread started")

    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while not self.stop_cleanup.wait(self.config.cleanup_interval):
            try:
                self._cleanup_expired_entries()
                self._cleanup_memory_pressure()
                self._cleanup_memory_pool()
                self._cleanup_gpu_memory()
            except Exception as e:
                logger.warning(f"Cache cleanup failed: {e}")

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory if usage is high."""
        if not self.gpu_available:
            return

        try:
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated(self.gpu_device)
                gpu_memory_reserved = torch.cuda.memory_reserved(self.gpu_device)
                total_gpu_memory = torch.cuda.get_device_properties(self.gpu_device).total_memory

                memory_usage_ratio = (gpu_memory_allocated + gpu_memory_reserved) / total_gpu_memory

                if memory_usage_ratio > 0.8:  # High GPU memory usage
                    # Move some GPU entries back to CPU
                    gpu_keys = list(self.gpu_cache.keys())
                    moved_count = 0

                    for key in gpu_keys[:len(gpu_keys)//2]:  # Move half of GPU entries to CPU
                        if key in self.cache:
                            entry = self.cache[key]
                            # Move tensors back to CPU
                            cpu_features = {}
                            for modality, tensor in entry.features.items():
                                if tensor.device.type == 'cuda':
                                    cpu_features[modality] = tensor.cpu()
                                else:
                                    cpu_features[modality] = tensor

                            entry.features = cpu_features
                            entry.predictions = entry.predictions.cpu() if entry.predictions.device.type == 'cuda' else entry.predictions
                            moved_count += 1

                    if moved_count > 0:
                        logger.info(f"🔄 Moved {moved_count} entries from GPU to CPU to free memory")

        except Exception as e:
            logger.debug(f"GPU memory cleanup failed: {e}")

    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []

        with self.lock:
            for key, entry in self.cache.items():
                if current_time - entry.created_at > self.config.ttl_seconds:
                    expired_keys.append(key)
                    self.current_memory_usage -= entry.memory_size

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

    def _cleanup_memory_pressure(self):
        """Remove entries when memory pressure is high."""
        if self.current_memory_usage > self.config.max_memory_mb * 1024 * 1024:
            # Sort by access time (oldest first)
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )

            # Remove oldest entries until memory is under limit
            removed_count = 0
            for key, entry in sorted_entries:
                if self.current_memory_usage <= self.config.max_memory_mb * 1024 * 1024 * 0.8:
                    break

                del self.cache[key]
                self.current_memory_usage -= entry.memory_size
                removed_count += 1

            if removed_count > 0:
                self.eviction_count += removed_count
                logger.info(f"🗑️ Evicted {removed_count} entries due to memory pressure")

    def _generate_cache_key(self, market_data: pd.DataFrame, feature_config: Dict[str, Any]) -> str:
        """Generate cache key from market data and configuration."""
        # Create hash of market data
        market_data_str = market_data.to_string().encode('utf-8')
        config_str = str(sorted(feature_config.items())).encode('utf-8')

        combined = market_data_str + config_str
        return hashlib.md5(combined).hexdigest()

    def _calculate_memory_size(self, entry: CLVSACacheEntry) -> int:
        """Calculate memory size of cache entry."""
        size = 0

        # Features size
        for modality, tensor in entry.features.items():
            size += tensor.numel() * tensor.element_size()

        # Predictions size
        size += entry.predictions.numel() * entry.predictions.element_size()

        # Attention weights size
        for name, weights in entry.attention_weights.items():
            size += weights.nbytes

        return size

    def _optimize_tensor_memory(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Optimize tensor memory usage using memory pool."""
        # Check if we have a suitable tensor in the pool
        tensor_shape = tuple(tensor.shape)
        tensor_dtype = tensor.dtype

        pool_key = f"{tensor_shape}_{tensor_dtype}"

        if pool_key in self.memory_pool:
            # Reuse tensor from pool
            pooled_tensor = self.memory_pool[pool_key]
            if pooled_tensor.is_shared():
                # Copy data to pooled tensor
                pooled_tensor.copy_(tensor)
                return pooled_tensor
            else:
                # Remove from pool if not shared
                del self.memory_pool[pool_key]

        # Move to GPU if available and beneficial
        if self.gpu_available and tensor.numel() > 10000:  # Only for large tensors
            try:
                gpu_tensor = tensor.to(self.gpu_device, non_blocking=True)
                return gpu_tensor
            except Exception as e:
                logger.debug(f"Failed to move tensor to GPU: {e}")

        # Store in memory pool for future reuse
        if tensor.is_contiguous():
            self.memory_pool[pool_key] = tensor.detach()

        return tensor

    def _move_to_gpu_if_beneficial(self, entry: CLVSACacheEntry) -> CLVSACacheEntry:
        """Move tensors to GPU if beneficial for performance."""
        if not self.gpu_available:
            return entry

        # Check if GPU memory is available
        try:
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated(self.gpu_device)
                gpu_memory_reserved = torch.cuda.memory_reserved(self.gpu_device)
                total_gpu_memory = torch.cuda.get_device_properties(self.gpu_device).total_memory

                # Only use GPU if we have enough memory
                memory_usage_ratio = (gpu_memory_allocated + gpu_memory_reserved) / total_gpu_memory
                if memory_usage_ratio < 0.7:  # Keep 30% free
                    # Move large tensors to GPU
                    optimized_features = {}
                    for modality, tensor in entry.features.items():
                        if tensor.numel() > 5000:  # Only large tensors
                            optimized_features[modality] = tensor.to(self.gpu_device, non_blocking=True)
                        else:
                            optimized_features[modality] = tensor

                    # Update entry
                    entry.features = optimized_features
                    entry.predictions = entry.predictions.to(self.gpu_device, non_blocking=True)

                    # Update memory tracking
                    entry.memory_size = self._calculate_memory_size(entry)
                    self.gpu_memory_usage += entry.memory_size

                    return entry
        except Exception as e:
            logger.debug(f"GPU memory optimization failed: {e}")

        return entry

    def _cleanup_memory_pool(self):
        """Clean up memory pool to free unused tensors."""
        current_time = time.time()

        # Remove tensors that haven't been accessed recently
        keys_to_remove = []
        for pool_key, tensor in self.memory_pool.items():
            # Remove if tensor is not shared or hasn't been accessed
            if not tensor.is_shared():
                keys_to_remove.append(pool_key)

        for key in keys_to_remove:
            del self.memory_pool[key]

        # Clean up weak references
        for ref_key in list(self.tensor_references.keys()):
            weak_set = self.tensor_references[ref_key]
            if len(weak_set) == 0:
                del self.tensor_references[ref_key]

    def _save_to_disk(self, key: str, entry: CLVSACacheEntry):
        """Save cache entry to disk."""
        if not self.cache_dir:
            return

        try:
            cache_file = self.cache_dir / f"{key}.pkl"

            # Convert tensors to numpy for serialization
            serializable_entry = CLVSACacheEntry(
                key=entry.key,
                features={k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                         for k, v in entry.features.items()},
                predictions=entry.predictions.cpu().numpy() if isinstance(entry.predictions, torch.Tensor) else entry.predictions,
                attention_weights={k: v for k, v in entry.attention_weights.items()},
                market_data_hash=entry.market_data_hash,
                created_at=entry.created_at,
                last_accessed=entry.last_accessed,
                access_count=entry.access_count,
                memory_size=entry.memory_size
            )

            with open(cache_file, 'wb') as f:
                if self.config.enable_compression:
                    import gzip
                    with gzip.GzipFile(fileobj=f, compresslevel=self.config.compression_level) as gf:
                        pickle.dump(serializable_entry, gf)
                else:
                    pickle.dump(serializable_entry, f)

        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")

    def _load_from_disk(self, key: str) -> Optional[CLVSACacheEntry]:
        """Load cache entry from disk."""
        if not self.cache_dir:
            return None

        try:
            cache_file = self.cache_dir / f"{key}.pkl"

            if not cache_file.exists():
                return None

            with open(cache_file, 'rb') as f:
                if self.config.enable_compression:
                    import gzip
                    with gzip.GzipFile(fileobj=f, mode='rb') as gf:
                        entry = pickle.load(gf)
                else:
                    entry = pickle.load(f)

            # Convert numpy arrays back to tensors
            entry.features = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                             for k, v in entry.features.items()}
            if isinstance(entry.predictions, np.ndarray):
                entry.predictions = torch.from_numpy(entry.predictions)

            return entry

        except Exception as e:
            logger.warning(f"Failed to load cache entry from disk: {e}")
            return None

    def get_cache_key(self, market_data: pd.DataFrame, feature_config: Dict[str, Any]) -> str:
        """Get cache key for given data and configuration."""
        return self._generate_cache_key(market_data, feature_config)

    def store(self, market_data: pd.DataFrame, feature_config: Dict[str, Any],
              features: Dict[str, torch.Tensor], predictions: torch.Tensor,
              attention_weights: Dict[str, np.ndarray]) -> str:
        """Store CLVSA elements in cache with memory optimization."""
        key = self._generate_cache_key(market_data, feature_config)

        with self.lock:
            # Check if already exists
            if key in self.cache:
                return key

            # Optimize tensor memory usage
            optimized_features = {}
            for modality, tensor in features.items():
                optimized_features[modality] = self._optimize_tensor_memory(tensor, f"{key}_{modality}")

            optimized_predictions = self._optimize_tensor_memory(predictions, f"{key}_predictions")

            # Create entry
            entry = CLVSACacheEntry(
                key=key,
                features=optimized_features,
                predictions=optimized_predictions,
                attention_weights=attention_weights,
                market_data_hash=hashlib.md5(market_data.to_string().encode()).hexdigest(),
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0
            )

            entry.memory_size = self._calculate_memory_size(entry)

            # Move to GPU if beneficial
            entry = self._move_to_gpu_if_beneficial(entry)

            # Check size limits
            if len(self.cache) >= self.config.max_cache_size:
                self._cleanup_expired_entries()
                self._cleanup_memory_pool()
                if len(self.cache) >= self.config.max_cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
                    oldest_entry = self.cache[oldest_key]
                    self.current_memory_usage -= oldest_entry.memory_size
                    if oldest_entry in self.gpu_cache:
                        del self.gpu_cache[oldest_key]
                    del self.cache[oldest_key]
                    self.eviction_count += 1

            # Check memory limits
            if self.current_memory_usage + entry.memory_size > self.config.max_memory_mb * 1024 * 1024:
                self._cleanup_memory_pressure()
                self._cleanup_memory_pool()

            # Add to cache
            self.cache[key] = entry
            self.current_memory_usage += entry.memory_size

            # Store in GPU cache if on GPU
            if self.gpu_available and any(tensor.device.type == 'cuda' for tensor in entry.features.values()):
                self.gpu_cache[key] = entry

            # Save to disk if persistence is enabled
            if self.config.enable_persistence:
                self._save_to_disk(key, entry)

            logger.debug(f"💾 Stored CLVSA elements in cache (key: {key[:8]}..., size: {entry.memory_size / (1024*1024)".2f"}MB)")
            return key

    def retrieve(self, market_data: pd.DataFrame, feature_config: Dict[str, Any]) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, np.ndarray]]]:
        """Retrieve CLVSA elements from cache."""
        key = self._generate_cache_key(market_data, feature_config)

        with self.lock:
            # Try memory cache first
            if key in self.cache:
                entry = self.cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.hit_count += 1

                logger.debug(f"🎯 Cache hit for key: {key[:8]}...")
                return entry.features, entry.predictions, entry.attention_weights

            # Try disk cache
            entry = self._load_from_disk(key)
            if entry:
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.hit_count += 1

                # Add to memory cache
                entry.memory_size = self._calculate_memory_size(entry)
                self.cache[key] = entry
                self.current_memory_usage += entry.memory_size

                logger.debug(f"💿 Cache hit from disk for key: {key[:8]}...")
                return entry.features, entry.predictions, entry.attention_weights

            self.miss_count += 1
            logger.debug(f"❌ Cache miss for key: {key[:8]}...")
            return None

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_memory_usage = 0
            self.hit_count = 0
            self.miss_count = 0
            self.eviction_count = 0

            # Clear disk cache
            if self.cache_dir:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info("🧹 Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            cache_size = len(self.cache)
            hit_rate = self.hit_count / max(self.hit_count + self.miss_count, 1)
            memory_mb = self.current_memory_usage / (1024 * 1024)

            return {
                'cache_size': cache_size,
                'max_cache_size': self.config.max_cache_size,
                'memory_usage_mb': memory_mb,
                'max_memory_mb': self.config.max_memory_mb,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'eviction_count': self.eviction_count,
                'hit_rate': hit_rate,
                'total_requests': self.hit_count + self.miss_count
            }

    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        logger.info("🛑 Shutting down CLVSA cache manager...")

        if self.cleanup_thread:
            self.stop_cleanup.set()
            self.cleanup_thread.join(timeout=5)

        # Persist all cache entries to disk
        if self.config.enable_persistence:
            with self.lock:
                for key, entry in self.cache.items():
                    self._save_to_disk(key, entry)

        logger.info("✅ CLVSA cache manager shutdown complete")


# Global cache manager instance
_global_cache_manager: Optional[CLVSACacheManager] = None
_cache_lock = threading.Lock()


def get_global_clvsa_cache(config: Optional[CLVSACacheConfig] = None) -> CLVSACacheManager:
    """Get global CLVSA cache manager instance."""
    global _global_cache_manager

    if _global_cache_manager is None:
        with _cache_lock:
            if _global_cache_manager is None:
                cache_config = config or CLVSACacheConfig()
                _global_cache_manager = CLVSACacheManager(cache_config)

    return _global_cache_manager


@contextmanager
def clvsa_cache_context(config: Optional[CLVSACacheConfig] = None):
    """Context manager for CLVSA caching."""
    cache_manager = get_global_clvsa_cache(config)
    try:
        yield cache_manager
    finally:
        pass


def create_cache_config(**overrides) -> CLVSACacheConfig:
    """Create cache configuration with overrides."""
    return CLVSACacheConfig(**overrides)


__all__ = [
    "CLVSACacheManager",
    "CLVSACacheConfig",
    "CLVSACacheEntry",
    "get_global_clvsa_cache",
    "clvsa_cache_context",
    "create_cache_config"
]