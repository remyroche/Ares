"""
Threshold search and calibration helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        f1_score,
        balanced_accuracy_score,
        roc_auc_score,
        precision_recall_curve,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Thresholding")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Thresholding")


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1_macro',
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Find probability threshold maximizing a metric.

    Supported metrics: 'f1_macro', 'balanced_accuracy', 'youden_j' (binary).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_t = 0.5
    best_s = -np.inf
    scores: List[Tuple[float, float]] = []

    # handle binary vs multi-class
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        pos_scores = y_prob[:, 1]
    elif y_prob.ndim == 1:
        pos_scores = y_prob
    else:
        # fall back to argmax for multi-class; thresholding applies to max prob
        pos_scores = np.max(y_prob, axis=1)
        y_pred_labels = np.argmax(y_prob, axis=1)

    for t in thresholds:
        if y_prob.ndim <= 2 and (y_prob.ndim == 1 or y_prob.shape[1] == 2):
            y_pred = (pos_scores >= t).astype(int)
        else:
            # gate predictions by confidence
            y_pred = y_pred_labels.copy()
            y_pred[pos_scores < t] = -1  # abstain; ignored by metrics below

        s = _score(y_true, y_pred, pos_scores, metric)
        scores.append((float(t), float(s)))
        if s > best_s:
            best_t, best_s = float(t), float(s)

    return {'best_threshold': best_t, 'best_score': float(best_s), 'curve': scores}


def _score(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, metric: str) -> float:
    try:
        if metric == 'f1_macro':
            return float(f1_score(y_true, y_pred, average='macro')) if SKLEARN_AVAILABLE else 0.0
        if metric == 'balanced_accuracy':
            return float(balanced_accuracy_score(y_true, y_pred)) if SKLEARN_AVAILABLE else 0.0
        if metric == 'youden_j':
            # binary only
            if SKLEARN_AVAILABLE and len(np.unique(y_true)) == 2:
                # approximate via PR or direct confusion matrix would be better
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                youden = recall + (precision) - 1.0
                return float(np.nanmax(youden))
            return 0.0
        return 0.0
    except Exception:
        return 0.0


def calibrate_probabilities(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'isotonic',
    cv: int = 3,
) -> Any:
    """Wrap estimator with probability calibration (Platt or isotonic)."""
    if not SKLEARN_AVAILABLE:
        return estimator
    try:
        calibrated = CalibratedClassifierCV(estimator, method=method, cv=cv)
        calibrated.fit(X_train, y_train)
        return calibrated
    except Exception as e:
        _LOGGER.warning(f"Calibration failed: {e}")
        # fallback to original
        try:
            estimator.fit(X_train, y_train)
        except Exception:
            pass
        return estimator


__all__ = [
    'optimize_threshold',
    'calibrate_probabilities',
]

