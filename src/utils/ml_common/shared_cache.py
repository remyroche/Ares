"""
Shared ML Cache Utilities

Provides a lightweight, process-local cache for reusing expensive ML artifacts
across modules: CV splits, MI scores, correlation matrices, and RF importances.

Optional integration with joblib.Memory when available.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from joblib import Memory
    from joblib.hashing import hash as joblib_hash
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False
    Memory = None  # type: ignore
    joblib_hash = None  # type: ignore


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
    """Thread-safe in-memory cache with optional joblib persistence."""

    _instance: Optional["SharedMLCache"] = None
    _lock = threading.Lock()

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or _default_cache_dir()
        self._memory = Memory(self._cache_dir, verbose=0) if JOBLIB_AVAILABLE else None

        # Process-local dictionaries
        self.cv_splits: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.mi_scores: Dict[str, Dict[str, float]] = {}
        self.corr_matrices: Dict[str, np.ndarray] = {}
        self.rf_importances: Dict[str, Dict[str, float]] = {}

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
        return self.cv_splits.get(key)

    def set_cv_splits(self, key: str, splits: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        self.cv_splits[key] = splits

    # ---- MI scores ----
    def get_mi_scores(self, key: str) -> Optional[Dict[str, float]]:
        return self.mi_scores.get(key)

    def set_mi_scores(self, key: str, scores: Dict[str, float]) -> None:
        self.mi_scores[key] = scores

    # ---- Correlation matrix ----
    def get_corr_matrix(self, key: str) -> Optional[np.ndarray]:
        return self.corr_matrices.get(key)

    def set_corr_matrix(self, key: str, corr: np.ndarray) -> None:
        self.corr_matrices[key] = corr

    # ---- RF importances ----
    def get_rf_importances(self, key: str) -> Optional[Dict[str, float]]:
        return self.rf_importances.get(key)

    def set_rf_importances(self, key: str, imp: Dict[str, float]) -> None:
        self.rf_importances[key] = imp


# Convenience module-level singleton
shared_cache = SharedMLCache.get()

