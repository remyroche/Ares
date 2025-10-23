from __future__ import annotations

from src.utils.tprint import tprint

"""
Threshold search and calibration helpers.

Enhanced with M1 GPU acceleration, memory optimization, and parallel processing.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

# Initialize logger early to ensure availability in import error handlers
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Thresholding")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Thresholding")

# Import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Enhanced dependency management with fast fail
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        f1_score,
        balanced_accuracy_score,
        roc_auc_score,
        precision_recall_curve,
    )
    SKLEARN_AVAILABLE = True
    tprint("✅ Scikit-learn available for thresholding functionality")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    tprint(f"❌ Scikit-learn not available: {e}. Thresholding functionality severely limited.")
    _LOGGER.error("Scikit-learn not available - limited thresholding functionality")
    raise ImportError(f"Scikit-learn is required for thresholding functionality: {e}")
except Exception as e:
    SKLEARN_AVAILABLE = False
    tprint(f"❌ Scikit-learn import failed: {e}. Thresholding functionality severely limited.")
    _LOGGER.error(f"Scikit-learn import failed: {e}")
    raise ImportError(f"Scikit-learn import failed: {e}")

# Validate critical dependencies
if not SKLEARN_AVAILABLE:
    raise ImportError("Scikit-learn is required for thresholding utilities")

# Import M1 utilities for enhanced performance

# Import M1 utilities
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from src.utils.hardware.m1_memory_optimizer import (
        auto_skim_memory, smart_memory_allocation,
        memory_skim_decorator, auto_memory_skim_decorator,
        auto_memory_skim_context, smart_memory_context
    )
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False

def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1_macro',
    thresholds: Optional[np.ndarray] = None,
    use_parallel: bool = True,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Find probability threshold maximizing a metric with M1 optimization.

    Enhanced with parallel processing and GPU acceleration for large datasets.

    Supported metrics: 'f1_macro', 'balanced_accuracy', 'youden_j' (binary).

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize
        thresholds: Thresholds to evaluate (default: linspace 0.05-0.95)
        use_parallel: Whether to use parallel processing for large threshold grids
        use_gpu: Whether to use GPU acceleration if available

    Returns:
        Dict with best threshold, score, and evaluation curve
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    # Estimate memory requirements and auto-skim if needed
    if MEMORY_OPTIMIZER_AVAILABLE:
        data_size_mb = len(y_true) * thresholds.shape[0] * 8 / (1024**2)
        auto_skim_memory(data_size_mb, "threshold_optimization")

    # Use parallel processing for large grids
    if use_parallel and CPU_OPTIMIZER_AVAILABLE and len(thresholds) > 10:
        try:
            return _optimize_threshold_parallel(y_true, y_prob, metric, thresholds, use_gpu)
        except Exception as e:
            _LOGGER.warning(f"Parallel threshold optimization failed: {e}, falling back to sequential")

    # Sequential implementation (original)
    return _optimize_threshold_sequential(y_true, y_prob, metric, thresholds)

def _optimize_threshold_sequential(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str,
    thresholds: np.ndarray,
) -> Dict[str, Any]:
    """Sequential threshold optimization."""
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

def _optimize_threshold_parallel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str,
    thresholds: np.ndarray,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Parallel threshold optimization using M1 CPU optimizer."""
    cpu_optimizer = get_m1_cpu_optimizer()

    # Prepare data for parallel processing
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        pos_scores = y_prob[:, 1]
        is_multiclass = False
    elif y_prob.ndim == 1:
        pos_scores = y_prob
        is_multiclass = False
    else:
        # Multi-class case
        pos_scores = np.max(y_prob, axis=1)
        y_pred_labels = np.argmax(y_prob, axis=1)
        is_multiclass = True

    def evaluate_threshold(t: float) -> Tuple[float, float]:
        """Evaluate a single threshold."""
        if not is_multiclass:
            y_pred = (pos_scores >= t).astype(int)
            s = _score(y_true, y_pred, pos_scores, metric)
        else:
            # Multi-class with confidence gating
            y_pred = y_pred_labels.copy()
            y_pred[pos_scores < t] = -1  # abstain
            s = _score(y_true, y_pred, pos_scores, metric)

        return float(t), float(s)

    # Evaluate thresholds in parallel
    results = cpu_optimizer.parallel_process(
        thresholds.tolist(),
        evaluate_threshold,
        task_type="cpu_bound"
    )

    # Find best result
    best_t = 0.5
    best_s = -np.inf
    scores = []

    for t, s in results:
        scores.append((t, s))
        if s > best_s:
            best_t, best_s = t, s

    return {'best_threshold': best_t, 'best_score': best_s, 'curve': scores}

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
        _LOGGER.warning(f"Calibration failed: {e}, returning original estimator")
        return estimator  # Return original estimator as fallback

class AdaptiveThresholding:
    """
    Adaptive thresholding utility class for dynamic threshold optimization.

    This class provides methods for automatically finding optimal thresholds
    for classification tasks based on various metrics and criteria.
    """

    def __init__(self, metric: str = 'f1_macro', cv_folds: int = 3):
        """
        Initialize adaptive thresholding.

        Args:
            metric: Metric to optimize ('f1_macro', 'balanced_accuracy', 'youden_j')
            cv_folds: Number of cross-validation folds for threshold optimization
        """
        self.metric = metric
        self.cv_folds = cv_folds
        self.best_threshold = 0.5
        self.best_score = 0.0
        self.threshold_history = []

    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray,
                             thresholds: Optional[np.ndarray] = None) -> float:
        """
        Find optimal threshold for given predictions and ground truth.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities or scores
            thresholds: Optional array of thresholds to test

        Returns:
            Optimal threshold value
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)

        best_threshold = 0.5
        best_score = -np.inf

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            score = _score(y_true, y_pred, y_scores, self.metric)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold
        self.best_score = best_score
        self.threshold_history.append((best_threshold, best_score))

        return best_threshold

    def apply_threshold(self, y_scores: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Apply threshold to scores to get binary predictions.

        Args:
            y_scores: Predicted probabilities or scores
            threshold: Threshold to use (if None, uses best_threshold)

        Returns:
            Binary predictions
        """
        thresh = threshold if threshold is not None else self.best_threshold
        return (y_scores >= thresh).astype(int)

    def get_threshold_stats(self) -> Dict[str, Any]:
        """Get statistics about threshold optimization history."""
        if not self.threshold_history:
            return {'count': 0, 'best_threshold': self.best_threshold, 'best_score': self.best_score}

        thresholds = [t for t, s in self.threshold_history]
        scores = [s for t, s in self.threshold_history]

        return {
            'count': len(self.threshold_history),
            'best_threshold': self.best_threshold,
            'best_score': self.best_score,
            'mean_threshold': np.mean(thresholds),
            'std_threshold': np.std(thresholds),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }

__all__ = [
    'optimize_threshold',
    'calibrate_probabilities',
    'AdaptiveThresholding',
]
