"""
Ensembling utilities: blending, stacking, and dynamic regime ensembles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Ensembling")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Ensembling")


def simple_blend(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    normalize_weights: bool = True,
) -> np.ndarray:
    """Weighted mean of predictions. Supports class probabilities or regression outputs."""
    if not predictions:
        return np.array([])
    P = np.stack(predictions, axis=0)
    if weights is None:
        weights = [1.0 / P.shape[0]] * P.shape[0]
    w = np.array(weights, dtype=float)
    if normalize_weights:
        s = np.sum(w)
        w = w / (s if s > 0 else 1.0)
    # broadcast weights across samples/classes
    while w.ndim < P.ndim:
        w = w[:, None]
    return np.sum(P * w, axis=0)


def learn_blend_weights(
    val_predictions: List[np.ndarray],
    y_val: np.ndarray,
    metric: str = 'balanced_accuracy',
) -> List[float]:
    """Grid-search small simplex to pick blend weights maximizing a metric."""
    if not val_predictions:
        return []
    K = len(val_predictions)
    grid = _simplex_grid(K, step=0.1)
    best_w = [1.0 / K] * K
    best_s = -np.inf
    for w in grid:
        blended = simple_blend(val_predictions, w)
        s = _eval_metric(y_val, blended, metric)
        if s > best_s:
            best_s = s
            best_w = w
    return best_w


def dynamic_regime_ensemble(
    regime_ids: np.ndarray,
    regime_to_model_preds: Dict[int, np.ndarray],
    default_pred: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select predictions based on regime id per sample."""
    n = len(regime_ids)
    if not regime_to_model_preds:
        return default_pred if default_pred is not None else np.zeros(n)
    # Assume each array in dict has shape (n, ...) aligned with samples
    out = None
    for i in range(n):
        rid = int(regime_ids[i])
        pred_mat = regime_to_model_preds.get(rid)
        if pred_mat is None:
            sel = default_pred[i] if default_pred is not None else 0.0
        else:
            sel = pred_mat[i]
        if out is None:
            out = np.zeros_like(pred_mat[0] if pred_mat is not None else np.array(sel))
            out = np.tile(out, (n, *([1] * (np.ndim(out)))))
        out[i] = sel
    return out


def _simplex_grid(k: int, step: float = 0.1) -> List[List[float]]:
    """Generate coarse weight combinations that sum to 1.0."""
    if k == 1:
        return [[1.0]]
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    combos: List[List[float]] = []
    def rec(prefix: List[float], depth: int):
        if depth == k - 1:
            rem = 1.0 - sum(prefix)
            if rem >= -1e-9:
                combos.append(prefix + [max(0.0, rem)])
            return
        for v in vals:
            if sum(prefix) + v <= 1.0 + 1e-9:
                rec(prefix + [float(v)], depth + 1)
    rec([], 0)
    return combos


def _eval_metric(y_true: np.ndarray, pred: np.ndarray, metric: str) -> float:
    try:
        if pred.ndim == 1 or (pred.ndim == 2 and pred.shape[1] == 1):
            # regression or binary scores -> threshold at 0.5
            y_pred = (pred.ravel() >= 0.5).astype(int)
        elif pred.ndim == 2:
            y_pred = np.argmax(pred, axis=1)
        else:
            y_pred = pred
        if metric == 'accuracy':
            return float(np.mean(y_true == y_pred))
        if metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric == 'f1_macro':
            from sklearn.metrics import f1_score
            return float(f1_score(y_true, y_pred, average='macro'))
        return float(np.mean(y_true == y_pred))
    except Exception:
        return 0.0


__all__ = [
    'simple_blend',
    'learn_blend_weights',
    'dynamic_regime_ensemble',
]

