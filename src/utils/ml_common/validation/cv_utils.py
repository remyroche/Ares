"""
Compatibility utilities for cross-validation (legacy API layer).

This module provides light shims that map older ml_common validation names to
the unified implementations now used across the codebase.

Exposed symbols:
- TemporalCrossValidator: Thin wrapper delegating to the unified temporal CV
- PurgedKFold: Alias to the time-aware purged/embargoed splitter
- CrossValidationUtilities: Minimal utilities with walk_forward_validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    SkTimeSeriesSplit = None  # type: ignore

# Purged K-Fold time-aware splitter (existing implementation)
try:
    from src.utils.purged_kfold import PurgedKFoldTime as PurgedKFold  # type: ignore
except Exception:  # pragma: no cover - fallback if unavailable
    PurgedKFold = None  # type: ignore


class TemporalCrossValidator:
    """Backwards-compatible temporal cross-validator wrapper.

    Delegates to sklearn's TimeSeriesSplit if available, otherwise provides
    a simple sequential splitter. This class is intended to satisfy legacy
    imports while the canonical API lives in validation.unified_cv and
    validation.universal_temporal_validation.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0, test_size: Optional[int] = None) -> None:
        self.n_splits = max(2, int(n_splits))
        self.gap = max(0, int(gap))
        self.test_size = test_size

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if SkTimeSeriesSplit is not None:
            try:
                import inspect
                if self.test_size is not None and 'test_size' in inspect.signature(SkTimeSeriesSplit).parameters:
                    cv = SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap, test_size=self.test_size)
                else:
                    cv = SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)  # type: ignore[arg-type]
                for tr, te in cv.split(X, y):
                    yield tr, te
                return
            except Exception:
                pass

        # Fallback: naive sequential splits
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for fs in fold_sizes:
            stop = start + fs
            test_idx = np.arange(start, stop)
            train_end = max(0, start - self.gap)
            train_idx = np.arange(0, train_end)
            yield train_idx, test_idx
            start = stop


@dataclass
class _WalkForwardConfig:
    initial_train_size: float = 0.6
    step_size: float = 0.1
    min_test_size: float = 0.1


class CrossValidationUtilities:
    """Minimal CV utilities used by memory integration shims.

    The canonical, richer API remains in validation.unified_cv and
    validation.universal_temporal_validation. This class exists to
    preserve backwards compatibility for integrations that monkey-patch
    walk_forward_validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.config = _WalkForwardConfig(
            initial_train_size=float(cfg.get('initial_train_size', 0.6)),
            step_size=float(cfg.get('step_size', 0.1)),
            min_test_size=float(cfg.get('min_test_size', 0.1)),
        )

    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n == 0:
            return []
        initial = max(1, int(n * self.config.initial_train_size))
        step = max(1, int(n * self.config.step_size))
        min_test = max(1, int(n * self.config.min_test_size))
        indices: List[Tuple[np.ndarray, np.ndarray]] = []
        train_end = initial
        while train_end < n - min_test:
            test_end = min(n, train_end + min_test)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            indices.append((train_idx, test_idx))
            train_end = min(n, train_end + step)
        return indices


__all__ = [
    'TemporalCrossValidator',
    'PurgedKFold',
    'CrossValidationUtilities',
]

