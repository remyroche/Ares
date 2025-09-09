"""
Ensembling utilities: blending, stacking, and dynamic regime ensembles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Ensembling")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Ensembling")

# Import memory skimming utilities
try:
    from ..m1_memory_optimizer import (
        auto_skim_memory, smart_memory_allocation,
        memory_skim_decorator, auto_memory_skim_decorator,
        auto_memory_skim_context, smart_memory_context
    )
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

# Import GPU utilities
try:
    from ..m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Import CPU optimizer
try:
    from ..m1_cpu_optimizer import get_m1_cpu_optimizer
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False


def simple_blend(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    normalize_weights: bool = True,
    use_gpu: bool = True,
) -> np.ndarray:
    """Weighted mean of predictions with optional GPU acceleration.

    Supports class probabilities or regression outputs.

    Args:
        predictions: List of prediction arrays to blend
        weights: Optional weights for each prediction array
        normalize_weights: Whether to normalize weights to sum to 1
        use_gpu: Whether to use GPU acceleration if available and beneficial

    Returns:
        Blended predictions array
    """
    if not predictions:
        return np.array([])

    # Check if GPU acceleration is beneficial
    if use_gpu and GPU_AVAILABLE and len(predictions) > 5:
        try:
            return _simple_blend_gpu(predictions, weights, normalize_weights)
        except Exception as e:
            _LOGGER.warning(f"GPU blending failed: {e}, falling back to CPU")

    # CPU implementation
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


def _simple_blend_gpu(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    normalize_weights: bool = True,
) -> np.ndarray:
    """GPU-accelerated simple blending."""
    gpu_manager = M1GPUManager()

    with gpu_manager.gpu_context("ensemble_blending"):
        # Stack predictions on GPU
        P_cpu = np.stack(predictions, axis=0)
        P_gpu = gpu_manager.to_device(P_cpu, "matrix_mult")

        # Handle weights
        if weights is None:
            weights = [1.0 / P_gpu.shape[0]] * P_gpu.shape[0]

        w_cpu = np.array(weights, dtype=float)
        if normalize_weights:
            s = np.sum(w_cpu)
            w_cpu = w_cpu / (s if s > 0 else 1.0)

        w_gpu = gpu_manager.to_device(w_cpu, "general")

        # Broadcast weights across samples/classes
        while w_gpu.ndim < P_gpu.ndim:
            w_gpu = w_gpu.unsqueeze(-1)

        # Compute weighted sum
        result_gpu = torch.sum(P_gpu * w_gpu, dim=0)

        # Convert back to CPU
        return result_gpu.cpu().numpy()


def learn_blend_weights(
    val_predictions: List[np.ndarray],
    y_val: np.ndarray,
    metric: str = 'balanced_accuracy',
    use_parallel: bool = True,
) -> List[float]:
    """Grid-search small simplex to pick blend weights maximizing a metric.

    Enhanced with parallel processing for large grids.

    Args:
        val_predictions: List of validation predictions
        y_val: Validation targets
        metric: Metric to optimize
        use_parallel: Whether to use parallel processing if beneficial

    Returns:
        Optimal blend weights
    """
    if not val_predictions:
        return []

    K = len(val_predictions)
    grid = _simplex_grid(K, step=0.1)

    # Use parallel processing for large grids if beneficial
    if use_parallel and CPU_OPTIMIZER_AVAILABLE and len(grid) > 50:
        try:
            return _learn_blend_weights_parallel(val_predictions, y_val, grid, metric)
        except Exception as e:
            _LOGGER.warning(f"Parallel weight learning failed: {e}, falling back to sequential")

    # Sequential implementation
    best_w = [1.0 / K] * K
    best_s = -np.inf
    for w in grid:
        blended = simple_blend(val_predictions, w, use_gpu=False)  # Avoid nested GPU calls
        s = _eval_metric(y_val, blended, metric)
        if s > best_s:
            best_s = s
            best_w = w
    return best_w


def _learn_blend_weights_parallel(
    val_predictions: List[np.ndarray],
    y_val: np.ndarray,
    grid: List[List[float]],
    metric: str,
) -> List[float]:
    """Parallel implementation of blend weight learning."""
    cpu_optimizer = get_m1_cpu_optimizer()

    def evaluate_weight_combination(weights):
        """Evaluate a single weight combination."""
        blended = simple_blend(val_predictions, weights, use_gpu=False)
        return _eval_metric(y_val, blended, metric), weights

    # Evaluate weight combinations in parallel
    results = cpu_optimizer.parallel_process(
        grid,
        evaluate_weight_combination,
        task_type="cpu_bound"
    )

    # Find best result
    best_score = -np.inf
    best_weights = None

    for score, weights in results:
        if score > best_score:
            best_score = score
            best_weights = weights

    return best_weights if best_weights else [1.0 / len(val_predictions)] * len(val_predictions)


def dynamic_regime_ensemble(
    regime_ids: np.ndarray,
    regime_to_model_preds: Dict[int, np.ndarray],
    default_pred: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select predictions based on regime id per sample with automatic memory skimming."""
    n = len(regime_ids)
    
    # Estimate memory requirements and auto-skim if needed
    if MEMORY_OPTIMIZER_AVAILABLE:
        estimated_memory_mb = n * len(regime_to_model_preds) * 8 / (1024**2)  # Rough estimate
        auto_skim_memory(estimated_memory_mb, "model_inference")
    
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
    """Evaluate metric with comprehensive error handling and sklearn dependency management."""
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
            try:
                from sklearn.metrics import balanced_accuracy_score
                return float(balanced_accuracy_score(y_true, y_pred))
            except ImportError as e:
                _LOGGER.warning(f"⚠️ sklearn not available for balanced_accuracy: {e}")
                return float(np.mean(y_true == y_pred))
            except Exception as e:
                _LOGGER.error(f"❌ balanced_accuracy calculation failed: {e}")
                return float(np.mean(y_true == y_pred))
                
        if metric == 'f1_macro':
            try:
                from sklearn.metrics import f1_score
                return float(f1_score(y_true, y_pred, average='macro'))
            except ImportError as e:
                _LOGGER.warning(f"⚠️ sklearn not available for f1_macro: {e}")
                return float(np.mean(y_true == y_pred))
            except Exception as e:
                _LOGGER.error(f"❌ f1_macro calculation failed: {e}")
                return float(np.mean(y_true == y_pred))
                
        return float(np.mean(y_true == y_pred))
        
    except Exception as e:
        _LOGGER.error(f"❌ Metric evaluation failed: {e}")
        return 0.0


__all__ = [
    'simple_blend',
    'learn_blend_weights',
    'dynamic_regime_ensemble',
]


