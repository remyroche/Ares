"""
Cross-validation helpers with purged/embargoed time series splits and integrity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..purged_kfold import PurgedKFoldTime
except Exception:
    PurgedKFoldTime = None  # fallback handled below

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.CV")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.CV")


@dataclass
class PurgedSplitConfig:
    n_splits: int = 5
    purge_minutes: int = 30
    embargo_minutes: int = 15


def purged_time_series_splits(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    config: PurgedSplitConfig = PurgedSplitConfig(),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) using purged/embargoed logic.

    Falls back to naive sequential splits if dedicated splitter unavailable.
    """
    if PurgedKFoldTime is None:
        _LOGGER.warning("PurgedKFoldTime not available, using naive sequential splits")
        n = len(X)
        fold_sizes = np.full(config.n_splits, n // config.n_splits, dtype=int)
        fold_sizes[: n % config.n_splits] += 1
        start = 0
        for fs in fold_sizes:
            stop = start + fs
            val_idx = np.arange(start, stop)
            train_idx = np.concatenate([np.arange(0, start), np.arange(stop, n)])
            yield train_idx, val_idx
            start = stop
        return

    splitter = PurgedKFoldTime(
        n_splits=config.n_splits,
        purge=pd.Timedelta(minutes=int(config.purge_minutes)),
        embargo=pd.Timedelta(minutes=int(config.embargo_minutes)),
    )
    for tr, va in splitter.split(X, y):
        yield tr, va


def analyze_splits(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Return per-fold diagnostics: sizes, class counts, temporal ordering checks."""
    results: Dict[str, Any] = {"folds": []}
    is_time = isinstance(X.index, pd.DatetimeIndex)

    for i, (tr, va) in enumerate(splits):
        fold: Dict[str, Any] = {"fold": i}
        fold["train_samples"] = int(len(tr))
        fold["val_samples"] = int(len(va))
        try:
            u, c = np.unique(y.iloc[tr], return_counts=True)
            fold["train_class_counts"] = {int(k): int(v) for k, v in zip(u, c)}
        except Exception:
            pass
        try:
            u, c = np.unique(y.iloc[va], return_counts=True)
            fold["val_class_counts"] = {int(k): int(v) for k, v in zip(u, c)}
        except Exception:
            pass
        if is_time and len(tr) > 0 and len(va) > 0:
            fold["temporal_ok"] = bool(X.index[tr][-1] < X.index[va][0])
        results["folds"].append(fold)

    # Aggregate
    results["n_folds"] = len(splits)
    results["min_train"] = int(min(f["train_samples"] for f in results["folds"])) if splits else 0
    results["min_val"] = int(min(f["val_samples"] for f in results["folds"])) if splits else 0
    return results


def validate_cv_integrity(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    min_train: int = 100,
    min_val: int = 50,
    require_two_classes: bool = True,
) -> Dict[str, Any]:
    """Validate CV splits for minimum sizes, class diversity, temporal order."""
    issues: List[str] = []
    fold_ok: List[bool] = []
    is_time = isinstance(X.index, pd.DatetimeIndex)

    for i, (tr, va) in enumerate(splits):
        ok = True
        if len(tr) < min_train:
            ok = False
            issues.append(f"fold_{i}: train too small {len(tr)} < {min_train}")
        if len(va) < min_val:
            ok = False
            issues.append(f"fold_{i}: val too small {len(va)} < {min_val}")
        if require_two_classes:
            try:
                if len(np.unique(y.iloc[tr])) < 2:
                    ok = False
                    issues.append(f"fold_{i}: single-class train")
                if len(np.unique(y.iloc[va])) < 2:
                    ok = False
                    issues.append(f"fold_{i}: single-class val")
            except Exception:
                pass
        if is_time and len(tr) > 0 and len(va) > 0:
            if not (X.index[tr][-1] < X.index[va][0]):
                ok = False
                issues.append(f"fold_{i}: temporal ordering violated")
        fold_ok.append(ok)

    return {
        'is_valid': all(fold_ok) if fold_ok else False,
        'fold_validity': fold_ok,
        'issues': issues,
        'n_folds': len(splits),
    }


__all__ = [
    'PurgedSplitConfig',
    'purged_time_series_splits',
    'analyze_splits',
    'validate_cv_integrity',
]


