from __future__ import annotations

from src.utils.tprint import tprint

"""
Cross-validation helpers with purged/embargoed time series splits and integrity checks.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .consolidated_cv import ConsolidatedCrossValidator as PurgedKFoldTime
except Exception:
    PurgedKFoldTime = None  # fallback handled below

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.CV")
    tprint("✅ Custom logger available for MLCommon.CV")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    import logging
    _LOGGER = logging.getLogger("MLCommon.CV")
    _LOGGER.setLevel(logging.INFO)

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
    _LOGGER.info(f"🔄 Starting purged time series splits...")
    _LOGGER.info(f"📊 Parameters - Splits: {config.n_splits}, Purge: {config.purge_minutes}min, Embargo: {config.embargo_minutes}min")
    _LOGGER.info(f"📊 Data shape: {X.shape}")

    if PurgedKFoldTime is None:
        _LOGGER.warning("⚠️ PurgedKFoldTime not available, using naive sequential splits")
        n = len(X)
        fold_sizes = np.full(config.n_splits, n // config.n_splits, dtype=int)
        fold_sizes[: n % config.n_splits] += 1
        start = 0
        for i, fs in enumerate(fold_sizes):
            stop = start + fs
            val_idx = np.arange(start, stop)
            train_idx = np.concatenate([np.arange(0, start), np.arange(stop, n)])
            _LOGGER.debug(f"📊 Fold {i+1}/{config.n_splits} - Train: {len(train_idx)}, Val: {len(val_idx)}")
            yield train_idx, val_idx
            start = stop
        _LOGGER.info(f"✅ Naive sequential splits completed - {config.n_splits} folds")
        return

    _LOGGER.info("🔧 Using dedicated PurgedKFoldTime splitter")
    splitter = PurgedKFoldTime(
        n_splits=config.n_splits,
        purge=pd.Timedelta(minutes=int(config.purge_minutes)),
        embargo=pd.Timedelta(minutes=int(config.embargo_minutes)),
    )
    for i, (tr, va) in enumerate(splitter.split(X, y)):
        _LOGGER.debug(f"📊 Fold {i+1}/{config.n_splits} - Train: {len(tr)}, Val: {len(va)}")
        yield tr, va
    _LOGGER.info(f"✅ Purged time series splits completed - {config.n_splits} folds")

def analyze_splits(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Return per-fold diagnostics: sizes, class counts, temporal ordering checks."""
    _LOGGER.info(f"🔍 Starting split analysis for {len(splits)} folds...")
    _LOGGER.info(f"📊 Data shape: {X.shape}, Target shape: {y.shape}")

    results: Dict[str, Any] = {"folds": []}
    is_time = isinstance(X.index, pd.DatetimeIndex)
    _LOGGER.debug(f"📊 Time series data: {is_time}")

    for i, (tr, va) in enumerate(splits):
        _LOGGER.debug(f"📊 Analyzing fold {i+1}/{len(splits)} - Train: {len(tr)}, Val: {len(va)}")

        fold: Dict[str, Any] = {"fold": i}
        fold["train_samples"] = int(len(tr))
        fold["val_samples"] = int(len(va))

        try:
            u, c = np.unique(y.iloc[tr], return_counts=True)
            fold["train_class_counts"] = {int(k): int(v) for k, v in zip(u, c)}
            _LOGGER.debug(f"📊 Train classes: {fold['train_class_counts']}")
        except Exception as e:
            _LOGGER.warning(f"Failed to analyze train class counts for fold {i}: {e}")
            fold["train_class_counts"] = {}  # Continue without class counts

        try:
            u, c = np.unique(y.iloc[va], return_counts=True)
            fold["val_class_counts"] = {int(k): int(v) for k, v in zip(u, c)}
            _LOGGER.debug(f"📊 Val classes: {fold['val_class_counts']}")
        except Exception as e:
            _LOGGER.warning(f"Failed to analyze val class counts for fold {i}: {e}")
            fold["val_class_counts"] = {}  # Continue without class counts

        if is_time and len(tr) > 0 and len(va) > 0:
            temporal_ok = bool(X.index[tr][-1] < X.index[va][0])
            fold["temporal_ok"] = temporal_ok
            if not temporal_ok:
                _LOGGER.warning(f"⚠️ Temporal ordering violation in fold {i}")
            else:
                _LOGGER.debug(f"✅ Temporal ordering OK for fold {i}")

        results["folds"].append(fold)

    # Aggregate
    results["n_folds"] = len(splits)
    results["min_train"] = int(min(f["train_samples"] for f in results["folds"])) if splits else 0
    results["min_val"] = int(min(f["val_samples"] for f in results["folds"])) if splits else 0

    _LOGGER.info(f"✅ Split analysis completed - {len(splits)} folds analyzed")
    _LOGGER.info(f"📊 Summary - Min train: {results['min_train']}, Min val: {results['min_val']}")

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
    _LOGGER.info(f"🔍 Starting CV integrity validation...")
    _LOGGER.info(f"📊 Parameters - Min train: {min_train}, Min val: {min_val}, Require two classes: {require_two_classes}")
    _LOGGER.info(f"📊 Validating {len(splits)} folds")

    issues: List[str] = []
    fold_ok: List[bool] = []
    is_time = isinstance(X.index, pd.DatetimeIndex)
    _LOGGER.debug(f"📊 Time series data: {is_time}")

    for i, (tr, va) in enumerate(splits):
        _LOGGER.debug(f"📊 Validating fold {i+1}/{len(splits)} - Train: {len(tr)}, Val: {len(va)}")

        ok = True
        if len(tr) < min_train:
            ok = False
            issue_msg = f"fold_{i}: train too small {len(tr)} < {min_train}"
            issues.append(issue_msg)
            _LOGGER.warning(f"⚠️ {issue_msg}")
        if len(va) < min_val:
            ok = False
            issue_msg = f"fold_{i}: val too small {len(va)} < {min_val}"
            issues.append(issue_msg)
            _LOGGER.warning(f"⚠️ {issue_msg}")
        if require_two_classes:
            try:
                if len(np.unique(y.iloc[tr])) < 2:
                    ok = False
                    issue_msg = f"fold_{i}: single-class train"
                    issues.append(issue_msg)
                    _LOGGER.warning(f"⚠️ {issue_msg}")
                if len(np.unique(y.iloc[va])) < 2:
                    ok = False
                    issues.append(f"fold_{i}: single-class val")
            except Exception as e:
                _LOGGER.warning(f"Failed to check class counts for fold {i}: {e}")
                ok = False  # Mark as invalid but continue
        if is_time and len(tr) > 0 and len(va) > 0:
            if not (X.index[tr][-1] < X.index[va][0]):
                ok = False
                issue_msg = f"fold_{i}: temporal ordering violated"
                issues.append(issue_msg)
                _LOGGER.warning(f"⚠️ {issue_msg}")
        fold_ok.append(ok)

        if ok:
            _LOGGER.debug(f"✅ Fold {i+1} validation passed")
        else:
            _LOGGER.warning(f"❌ Fold {i+1} validation failed")

    is_valid = all(fold_ok) if fold_ok else False
    _LOGGER.info(f"✅ CV integrity validation completed")
    _LOGGER.info(f"📊 Results - Valid: {is_valid}, Issues: {len(issues)}, Valid folds: {sum(fold_ok)}/{len(splits)}")

    if issues:
        _LOGGER.warning(f"⚠️ Found {len(issues)} validation issues")
    else:
        _LOGGER.info("✅ No validation issues found")

    return {
        'is_valid': is_valid,
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
