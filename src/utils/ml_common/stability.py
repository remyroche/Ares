"""
Stability utilities: selection stability across folds/time and aggregation helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Stability")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Stability")


def feature_selection_stability(
    fold_selections: List[List[str]],
    all_features: List[str],
) -> Dict[str, Any]:
    """Compute selection frequency and stability score for each feature."""
    counts: Dict[str, int] = {f: 0 for f in all_features}
    for sel in fold_selections:
        for f in sel:
            if f in counts:
                counts[f] += 1
    n_folds = max(1, len(fold_selections))
    stability = {f: counts[f] / n_folds for f in all_features}
    return {
        'selection_counts': counts,
        'stability_scores': stability,
        'n_folds': n_folds,
    }


def aggregate_time_blocks(
    block_metrics: List[Dict[str, float]],
    keys: List[str],
) -> Dict[str, Any]:
    """Aggregate metrics across time blocks and compute variability."""
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [b.get(k) for b in block_metrics if k in b]
        if vals:
            arr = np.array(vals, dtype=float)
            agg[k] = {
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'cv': float(np.nanstd(arr) / (np.nanmean(arr) if np.nanmean(arr) != 0 else 1.0)),
                'min': float(np.nanmin(arr)),
                'max': float(np.nanmax(arr)),
            }
    return agg


__all__ = [
    'feature_selection_stability',
    'aggregate_time_blocks',
]

