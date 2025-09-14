from __future__ import annotations

from src.utils.tprint import tprint

"""
Shared ML Cache Utilities

Provides a lightweight, process-local cache for reusing expensive ML artifacts
across modules: CV splits, MI scores, correlation matrices, and RF importances.

Optional integration with joblib.Memory when available.
Enhanced with aggressive memory management and comprehensive error handling.
"""

import gc
import os
import psutil
import threading
import time
import weakref
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from contextlib import contextmanager

import numpy as np

# Enhanced dependency management with fast fail
try:
    from joblib import Memory
    from joblib.hashing import hash as joblib_hash
    JOBLIB_AVAILABLE = True
    tprint("✅ joblib available for persistent caching")
except ImportError as e:
    JOBLIB_AVAILABLE = False
    Memory = None  # type: ignore
    joblib_hash = None  # type: ignore
    tprint(f"⚠️ joblib not available: {e}. Using in-memory cache only.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
    tprint("✅ psutil available for memory monitoring")
except ImportError as e:
    PSUTIL_AVAILABLE = False
    tprint(f"⚠️ psutil not available: {e}. Memory monitoring disabled.")


def _default_cache_dir() -> str:
    base = os.environ.get("ARES_ML_CACHE_DIR", "/tmp/ares_ml_cache")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return base


def _safe_hash(obj: Any) -> str:
    try:
        if JOBLIB_AVAILABLE:
            return str(joblib_hash(obj))
    except Exception:
        pass
    try:
        if isinstance(obj, np.ndarray):
            return str(hash((obj.shape, obj.dtype.str, obj.tobytes())))
        return str(hash(repr(obj)))
    except Exception:
        return str(id(obj))


class SharedMLCache:
    """Thread-safe in-memory cache with optional joblib persistence and aggressive memory management."""

    _instance: Optional["SharedMLCache"] = None
    _lock = threading.Lock()

    def __init__(self, cache_dir: Optional[str] = None, max_memory_mb: int = 1024):
        tprint("🚀 Initializing SharedMLCache...")
        start_time = time.time()
        
        self._cache_dir = cache_dir or _default_cache_dir()
        self._memory = Memory(self._cache_dir, verbose=0) if JOBLIB_AVAILABLE else None
        self._max_memory_mb = max_memory_mb
        self._memory_threshold = 0.8  # Cleanup when 80% of max memory used
        
        tprint(f"📊 Cache directory: {self._cache_dir}")
        tprint(f"📊 Max memory: {self._max_memory_mb}MB")
        tprint(f"📊 Memory threshold: {self._memory_threshold*100:.1f}%")
        tprint(f"📊 Joblib available: {JOBLIB_AVAILABLE}")
        
        # Process-local dictionaries with weak references for automatic cleanup
        self.cv_splits: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.mi_scores: Dict[str, Dict[str, float]] = {}
        self.corr_matrices: Dict[str, np.ndarray] = {}
        self.rf_importances: Dict[str, Dict[str, float]] = {}
        tprint("✅ Cache dictionaries initialized")
        
        # Memory tracking
        self._cache_sizes: Dict[str, int] = {}
        self._access_counts: Dict[str, int] = {}
        self._last_access: Dict[str, float] = {}
        tprint("✅ Memory tracking initialized")
        
        # Memory monitoring
        self._initial_memory = self._get_memory_usage()
        init_time = time.time() - start_time
        tprint(f"✅ SharedMLCache initialized in {init_time:.3f}s with {self._max_memory_mb}MB limit")

    @classmethod
    def get(cls) -> "SharedMLCache":
        with cls._lock:
            if cls._instance is None:
                cls._instance = SharedMLCache()
            return cls._instance

    # ---- Hash helpers ----
    @staticmethod
    def hash_array(X: np.ndarray) -> str:
        return _safe_hash((X.shape, X.dtype.str, X.tobytes()))

    @staticmethod
    def hash_two_arrays(X: np.ndarray, y: np.ndarray) -> str:
        return _safe_hash((
            (X.shape, X.dtype.str, X.tobytes()),
            (y.shape, y.dtype.str, y.tobytes())
        ))

    # ---- CV splits ----
    def get_cv_splits(self, key: str) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        result = self.cv_splits.get(key)
        if result is not None:
            self._update_access_stats(key)
        return result

    def set_cv_splits(self, key: str, splits: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        self.cv_splits[key] = splits
        self._update_cache_stats(key, splits)
        self._check_memory_usage()

    # ---- MI scores ----
    def get_mi_scores(self, key: str) -> Optional[Dict[str, float]]:
        result = self.mi_scores.get(key)
        if result is not None:
            self._update_access_stats(key)
        return result

    def set_mi_scores(self, key: str, scores: Dict[str, float]) -> None:
        self.mi_scores[key] = scores
        self._update_cache_stats(key, scores)
        self._check_memory_usage()

    # ---- Correlation matrix ----
    def get_corr_matrix(self, key: str) -> Optional[np.ndarray]:
        result = self.corr_matrices.get(key)
        if result is not None:
            self._update_access_stats(key)
        return result

    def set_corr_matrix(self, key: str, corr: np.ndarray) -> None:
        self.corr_matrices[key] = corr
        self._update_cache_stats(key, corr)
        self._check_memory_usage()

    # ---- RF importances ----
    def get_rf_importances(self, key: str) -> Optional[Dict[str, float]]:
        result = self.rf_importances.get(key)
        if result is not None:
            self._update_access_stats(key)
        return result

    def set_rf_importances(self, key: str, imp: Dict[str, float]) -> None:
        self.rf_importances[key] = imp
        self._update_cache_stats(key, imp)
        self._check_memory_usage()

    def set_cached_value(self, key: str, data: Any) -> None:
        """Generic cache setter for arbitrary data."""
        try:
            # Use a general cache dictionary for arbitrary data
            if not hasattr(self, '_generic_cache'):
                self._generic_cache: Dict[str, Any] = {}

            self._generic_cache[key] = data
            self._update_cache_stats(key, data)
            self._check_memory_usage()
        except Exception as e:
            tprint(f"❌ Failed to cache data: {e}")

    def get_cached_value(self, key: str) -> Any:
        """Generic cache getter for arbitrary data."""
        try:
            if hasattr(self, '_generic_cache') and key in self._generic_cache:
                # Update access stats for LRU
                self._update_access_stats(key)
                return self._generic_cache[key]
            return None
        except Exception as e:
            tprint(f"❌ Failed to get cached data: {e}")
            return None

    # ---- Memory Management Methods ----
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
            else:
                # Fallback: estimate based on cache sizes
                total_size = sum(self._cache_sizes.values())
                return total_size / 1024 / 1024
        except Exception as e:
            tprint(f"⚠️ Memory monitoring failed: {e}")
            return 0.0

    def _update_access_stats(self, key: str) -> None:
        """Update access statistics for LRU eviction."""
        import time
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
        self._last_access[key] = time.time()

    def _update_cache_stats(self, key: str, data: Any) -> None:
        """Update cache size statistics."""
        try:
            if isinstance(data, np.ndarray):
                size = data.nbytes
            elif isinstance(data, (list, tuple)):
                size = sum(item.nbytes if isinstance(item, np.ndarray) else 0 for item in data)
            elif isinstance(data, dict):
                size = sum(len(str(k)) + len(str(v)) for k, v in data.items())
            else:
                size = len(str(data))
            
            self._cache_sizes[key] = size
            self._update_access_stats(key)
        except Exception as e:
            tprint(f"⚠️ Cache stats update failed for key {key}: {e}")

    def _check_memory_usage(self) -> None:
        """Check memory usage and trigger cleanup if needed."""
        current_memory = self._get_memory_usage()
        memory_limit = self._max_memory_mb * self._memory_threshold
        
        if current_memory > memory_limit:
            tprint(f"🧹 Memory usage {current_memory:.1f}MB exceeds threshold {memory_limit:.1f}MB, triggering cleanup")
            self._aggressive_cleanup()

    def _aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup using LRU eviction."""
        try:
            # Sort by last access time (oldest first)
            sorted_keys = sorted(
                self._last_access.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest 50% of entries
            remove_count = max(1, len(sorted_keys) // 2)
            removed_keys = []
            
            for key, _ in sorted_keys[:remove_count]:
                removed_keys.append(key)
                self._remove_key(key)
            
            # Force garbage collection
            gc.collect()
            
            current_memory = self._get_memory_usage()
            tprint(f"🧹 Cleanup completed: removed {len(removed_keys)} entries, memory now {current_memory:.1f}MB")
            
        except Exception as e:
            tprint(f"❌ Aggressive cleanup failed: {e}")

    def _remove_key(self, key: str) -> None:
        """Remove a key from all caches."""
        try:
            # Remove from all cache dictionaries
            self.cv_splits.pop(key, None)
            self.mi_scores.pop(key, None)
            self.corr_matrices.pop(key, None)
            self.rf_importances.pop(key, None)

            # Remove from generic cache if it exists
            if hasattr(self, '_generic_cache'):
                self._generic_cache.pop(key, None)

            # Remove from tracking dictionaries
            self._cache_sizes.pop(key, None)
            self._access_counts.pop(key, None)
            self._last_access.pop(key, None)

        except Exception as e:
            tprint(f"⚠️ Failed to remove key {key}: {e}")

    def clear_all(self) -> None:
        """Clear all caches and force garbage collection."""
        try:
            tprint("🧹 Clearing all caches...")
            
            # Clear all cache dictionaries
            self.cv_splits.clear()
            self.mi_scores.clear()
            self.corr_matrices.clear()
            self.rf_importances.clear()

            # Clear generic cache if it exists
            if hasattr(self, '_generic_cache'):
                self._generic_cache.clear()
            
            # Clear tracking dictionaries
            self._cache_sizes.clear()
            self._access_counts.clear()
            self._last_access.clear()
            
            # Force garbage collection
            gc.collect()
            
            current_memory = self._get_memory_usage()
            tprint(f"✅ All caches cleared, memory now {current_memory:.1f}MB")
            
        except Exception as e:
            tprint(f"❌ Cache clearing failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            current_memory = self._get_memory_usage()
            total_entries = sum(len(cache) for cache in [
                self.cv_splits, self.mi_scores,
                self.corr_matrices, self.rf_importances
            ])

            # Include generic cache if it exists
            if hasattr(self, '_generic_cache'):
                total_entries += len(self._generic_cache)

            return {
                'memory_usage_mb': current_memory,
                'max_memory_mb': self._max_memory_mb,
                'memory_threshold': self._memory_threshold,
                'total_entries': total_entries,
                'cv_splits_count': len(self.cv_splits),
                'mi_scores_count': len(self.mi_scores),
                'corr_matrices_count': len(self.corr_matrices),
                'rf_importances_count': len(self.rf_importances),
                'generic_cache_count': len(self._generic_cache) if hasattr(self, '_generic_cache') else 0,
                'cache_enabled': True
            }

        except Exception as e:
            return {
                'error': str(e),
                'cache_enabled': False
            }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            current_memory = self._get_memory_usage()
            total_entries = sum(len(cache) for cache in [
                self.cv_splits, self.mi_scores,
                self.corr_matrices, self.rf_importances
            ])

            return {
                'current_memory_mb': current_memory,
                'max_memory_mb': self._max_memory_mb,
                'memory_usage_percent': (current_memory / self._max_memory_mb) * 100,
                'total_cache_entries': total_entries,
                'cv_splits_count': len(self.cv_splits),
                'mi_scores_count': len(self.mi_scores),
                'corr_matrices_count': len(self.corr_matrices),
                'rf_importances_count': len(self.rf_importances),
                'cache_sizes_total_mb': sum(self._cache_sizes.values()) / 1024 / 1024
            }
        except Exception as e:
            tprint(f"❌ Memory stats failed: {e}")
            return {'error': str(e)}

    @contextmanager
    def memory_context(self, operation_name: str) -> Any:
        """Context manager for memory-intensive operations."""
        initial_memory = self._get_memory_usage()
        tprint(f"🔄 Starting {operation_name}, initial memory: {initial_memory:.1f}MB")
        
        try:
            yield self
        finally:
            final_memory = self._get_memory_usage()
            memory_delta = final_memory - initial_memory
            tprint(f"✅ Completed {operation_name}, memory delta: {memory_delta:+.1f}MB")
            
            # Trigger cleanup if memory usage increased significantly
            if memory_delta > 100:  # More than 100MB increase
                self._check_memory_usage()


# Convenience module-level singleton with enhanced error handling
try:
    shared_cache = SharedMLCache.get()
    tprint("✅ SharedMLCache singleton initialized successfully")
except Exception as e:
    tprint(f"❌ Failed to initialize SharedMLCache: {e}")
    # Create a minimal fallback
    shared_cache = SharedMLCache()


