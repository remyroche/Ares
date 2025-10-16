"""
Ensembling utilities: blending, stacking, and dynamic regime ensembles.
"""

from __future__ import annotations

import time
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
    from src.utils.hardware.m1_memory_optimizer import (  # type: ignore
        auto_skim_memory, smart_memory_allocation,
        memory_skim_decorator, auto_memory_skim_decorator,
        auto_memory_skim_context, smart_memory_context
    )
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

# Import GPU utilities
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Import CPU optimizer
try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer  # type: ignore
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
    _LOGGER.info(f"🔄 Starting simple blend with {len(predictions)} predictions")
    start_time = time.time()

    if not predictions:
        _LOGGER.warning("⚠️ No predictions provided for blending")
        return np.array([])

    _LOGGER.debug(f"📊 Prediction shapes: {[p.shape for p in predictions]}")
    _LOGGER.debug(f"📊 Weights provided: {weights is not None}")
    _LOGGER.debug(f"📊 Normalize weights: {normalize_weights}")
    _LOGGER.debug(f"📊 Use GPU: {use_gpu}")

    # Check if GPU acceleration is beneficial
    if use_gpu and GPU_AVAILABLE and len(predictions) > 5:
        _LOGGER.debug("🚀 Attempting GPU-accelerated blending...")
        try:
            result = _simple_blend_gpu(predictions, weights, normalize_weights)
            blend_time = time.time() - start_time
            _LOGGER.info(f"✅ GPU blending completed in {blend_time:.3f}s")
            return result
        except Exception as e:
            _LOGGER.warning(f"⚠️ GPU blending failed: {e}, falling back to CPU")

    # CPU implementation
    _LOGGER.debug("🔄 Using CPU implementation for blending...")
    P = np.stack(predictions, axis=0)
    _LOGGER.debug(f"📊 Stacked predictions shape: {P.shape}")

    if weights is None:
        weights = [1.0 / P.shape[0]] * P.shape[0]
        _LOGGER.debug(f"📊 Generated uniform weights: {weights}")

    w = np.array(weights, dtype=float)
    _LOGGER.debug(f"📊 Weight array shape: {w.shape}")

    if normalize_weights:
        s = np.sum(w)
        w = w / (s if s > 0 else 1.0)
        _LOGGER.debug(f"📊 Normalized weights sum: {np.sum(w):.6f}")

    # broadcast weights across samples/classes
    while w.ndim < P.ndim:
        w = w[:, None]
        _LOGGER.debug(f"📊 Broadcasting weights, new shape: {w.shape}")

    result = np.sum(P * w, axis=0)
    blend_time = time.time() - start_time

    _LOGGER.info(f"✅ CPU blending completed in {blend_time:.3f}s")
    _LOGGER.info(f"📊 Result shape: {result.shape}")
    _LOGGER.info(f"📊 Result range: {result.min():.4f} - {result.max():.4f}")

    return result

def _simple_blend_gpu(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    normalize_weights: bool = True,
) -> np.ndarray:
    """GPU-accelerated simple blending."""
    _LOGGER.debug("🚀 Starting GPU-accelerated blending...")
    gpu_start_time = time.time()

    gpu_manager = M1GPUManager()
    _LOGGER.debug("✅ GPU manager initialized")

    with gpu_manager.gpu_context("ensemble_blending"):
        _LOGGER.debug("🔄 GPU context established")

        # Stack predictions on GPU
        _LOGGER.debug("🔄 Stacking predictions on CPU...")
        P_cpu = np.stack(predictions, axis=0)
        _LOGGER.debug(f"📊 CPU predictions shape: {P_cpu.shape}")

        _LOGGER.debug("🔄 Transferring predictions to GPU...")
        P_gpu = gpu_manager.to_device(P_cpu, "matrix_mult")
        _LOGGER.debug(f"📊 GPU predictions shape: {P_gpu.shape}")

        # Handle weights
        if weights is None:
            weights = [1.0 / P_gpu.shape[0]] * P_gpu.shape[0]
            _LOGGER.debug(f"📊 Generated uniform weights: {weights}")

        w_cpu = np.array(weights, dtype=float)
        _LOGGER.debug(f"📊 CPU weights shape: {w_cpu.shape}")

        if normalize_weights:
            s = np.sum(w_cpu)
            w_cpu = w_cpu / (s if s > 0 else 1.0)
            _LOGGER.debug(f"📊 Normalized weights sum: {np.sum(w_cpu):.6f}")

        _LOGGER.debug("🔄 Transferring weights to GPU...")
        w_gpu = gpu_manager.to_device(w_cpu, "general")
        _LOGGER.debug(f"📊 GPU weights shape: {w_gpu.shape}")

        # Broadcast weights across samples/classes
        while w_gpu.ndim < P_gpu.ndim:
            w_gpu = w_gpu.unsqueeze(-1)
            _LOGGER.debug(f"📊 Broadcasting weights, new shape: {w_gpu.shape}")

        # Compute weighted sum
        _LOGGER.debug("🔄 Computing weighted sum on GPU...")
        result_gpu = torch.sum(P_gpu * w_gpu, dim=0)
        _LOGGER.debug(f"📊 GPU result shape: {result_gpu.shape}")

        # Convert back to CPU
        _LOGGER.debug("🔄 Converting result back to CPU...")
        result_cpu = result_gpu.cpu().numpy()

        gpu_time = time.time() - gpu_start_time
        _LOGGER.info(f"✅ GPU blending completed in {gpu_time:.3f}s")
        _LOGGER.info(f"📊 Result shape: {result_cpu.shape}")
        _LOGGER.info(f"📊 Result range: {result_cpu.min():.4f} - {result_cpu.max():.4f}")

        return result_cpu

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
    _LOGGER.info(f"🔄 Starting blend weight learning with {len(val_predictions)} predictions")
    start_time = time.time()

    if not val_predictions:
        _LOGGER.warning("⚠️ No validation predictions provided")
        return []

    K = len(val_predictions)
    _LOGGER.debug(f"📊 Number of models: {K}")
    _LOGGER.debug(f"📊 Validation targets shape: {y_val.shape}")
    _LOGGER.debug(f"📊 Optimization metric: {metric}")
    _LOGGER.debug(f"📊 Use parallel processing: {use_parallel}")

    _LOGGER.debug("🔄 Generating simplex grid...")
    grid = _simplex_grid(K, step=0.1)
    _LOGGER.info(f"📊 Generated {len(grid)} weight combinations to evaluate")

    # Use parallel processing for large grids if beneficial
    if use_parallel and CPU_OPTIMIZER_AVAILABLE and len(grid) > 50:
        _LOGGER.debug("🚀 Attempting parallel weight learning...")
        try:
            result = _learn_blend_weights_parallel(val_predictions, y_val, grid, metric)
            learn_time = time.time() - start_time
            _LOGGER.info(f"✅ Parallel weight learning completed in {learn_time:.3f}s")
            return result
        except Exception as e:
            _LOGGER.warning(f"⚠️ Parallel weight learning failed: {e}, falling back to sequential")

    # Sequential implementation
    _LOGGER.debug("🔄 Using sequential weight learning...")
    best_w = [1.0 / K] * K
    best_s = -np.inf

    _LOGGER.debug(f"🔄 Evaluating {len(grid)} weight combinations...")
    for i, w in enumerate(grid):
        if i % 100 == 0:  # Log progress every 100 iterations
            progress = (i / len(grid)) * 100
            _LOGGER.debug(f"📊 Progress: {progress:.1f}% ({i}/{len(grid)})")

        blended = simple_blend(val_predictions, w, use_gpu=False)  # Avoid nested GPU calls
        s = _eval_metric(y_val, blended, metric)

        if s > best_s:
            best_s = s
            best_w = w
            _LOGGER.debug(f"📊 New best score: {best_s:.4f} with weights: {[f'{w:.3f}' for w in best_w]}")

    learn_time = time.time() - start_time
    _LOGGER.info(f"✅ Sequential weight learning completed in {learn_time:.3f}s")
    _LOGGER.info(f"📊 Best score: {best_s:.4f}")
    _LOGGER.info(f"📊 Best weights: {[f'{w:.3f}' for w in best_w]}")
    _LOGGER.info(f"📊 Weight sum: {sum(best_w):.6f}")

    return best_w

def _learn_blend_weights_parallel(
    val_predictions: List[np.ndarray],
    y_val: np.ndarray,
    grid: List[List[float]],
    metric: str,
) -> List[float]:
    """Parallel implementation of blend weight learning."""
    _LOGGER.debug("🚀 Starting parallel blend weight learning...")
    parallel_start_time = time.time()

    cpu_optimizer = get_m1_cpu_optimizer()
    _LOGGER.debug("✅ CPU optimizer initialized")

    def evaluate_weight_combination(weights):
        """Evaluate a single weight combination."""
        blended = simple_blend(val_predictions, weights, use_gpu=False)
        return _eval_metric(y_val, blended, metric), weights

    _LOGGER.debug(f"🔄 Processing {len(grid)} weight combinations in parallel...")

    # Evaluate weight combinations in parallel
    results = cpu_optimizer.parallel_process(
        grid,
        evaluate_weight_combination,
        task_type="cpu_bound"
    )

    _LOGGER.debug(f"✅ Parallel processing completed, got {len(results)} results")

    # Find best result
    _LOGGER.debug("🔍 Finding best weight combination...")
    best_score = -np.inf
    best_weights = None

    for score, weights in results:
        if score > best_score:
            best_score = score
            best_weights = weights
            _LOGGER.debug(f"📊 New best score: {best_score:.4f}")

    if best_weights is None:
        _LOGGER.warning("⚠️ No valid weights found, using uniform weights")
        best_weights = [1.0 / len(val_predictions)] * len(val_predictions)

    parallel_time = time.time() - parallel_start_time
    _LOGGER.info(f"✅ Parallel weight learning completed in {parallel_time:.3f}s")
    _LOGGER.info(f"📊 Best score: {best_score:.4f}")
    _LOGGER.info(f"📊 Best weights: {[f'{w:.3f}' for w in best_weights]}")

    return best_weights

def dynamic_regime_ensemble(
    regime_ids: np.ndarray,
    regime_to_model_preds: Dict[int, np.ndarray],
    default_pred: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select predictions based on regime id per sample with automatic memory skimming."""
    _LOGGER.info(f"🔄 Starting dynamic regime ensemble with {len(regime_ids)} samples")
    start_time = time.time()

    n = len(regime_ids)
    _LOGGER.debug(f"📊 Number of samples: {n}")
    _LOGGER.debug(f"📊 Number of regime models: {len(regime_to_model_preds)}")
    _LOGGER.debug(f"📊 Default prediction provided: {default_pred is not None}")

    # Estimate memory requirements and auto-skim if needed
    if MEMORY_OPTIMIZER_AVAILABLE:
        estimated_memory_mb = n * len(regime_to_model_preds) * 8 / (1024**2)  # Rough estimate
        _LOGGER.debug(f"📊 Estimated memory requirement: {estimated_memory_mb:.1f} MB")
        auto_skim_memory(estimated_memory_mb, "model_inference")
        _LOGGER.debug("✅ Memory optimization applied")

    if not regime_to_model_preds:
        _LOGGER.warning("⚠️ No regime model predictions provided")
        result = default_pred if default_pred is not None else np.zeros(n)
        _LOGGER.info(f"✅ Returning default predictions: shape {result.shape}")
        return result

    # Assume each array in dict has shape (n, ...) aligned with samples
    _LOGGER.debug("🔄 Processing regime-based predictions...")
    out = None

    # Log regime distribution
    unique_regimes, regime_counts = np.unique(regime_ids, return_counts=True)
    _LOGGER.info(f"📊 Regime distribution: {dict(zip(unique_regimes, regime_counts))}")

    for i in range(n):
        if i % 1000 == 0:  # Log progress every 1000 samples
            progress = (i / n) * 100
            _LOGGER.debug(f"📊 Progress: {progress:.1f}% ({i}/{n})")

        rid = int(regime_ids[i])
        pred_mat = regime_to_model_preds.get(rid)

        if pred_mat is None:
            sel = default_pred[i] if default_pred is not None else 0.0
            _LOGGER.debug(f"📊 Sample {i}: regime {rid} not found, using default")
        else:
            sel = pred_mat[i]
            _LOGGER.debug(f"📊 Sample {i}: regime {rid} prediction selected")

        if out is None:
            out = np.zeros_like(pred_mat[0] if pred_mat is not None else np.array(sel))
            out = np.tile(out, (n, *([1] * (np.ndim(out)))))
            _LOGGER.debug(f"📊 Initialized output array with shape: {out.shape}")

        out[i] = sel

    ensemble_time = time.time() - start_time
    _LOGGER.info(f"✅ Dynamic regime ensemble completed in {ensemble_time:.3f}s")
    _LOGGER.info(f"📊 Output shape: {out.shape}")
    _LOGGER.info(f"📊 Output range: {out.min():.4f} - {out.max():.4f}")

    return out

def _simplex_grid(k: int, step: float = 0.1) -> List[List[float]]:
    """Generate coarse weight combinations that sum to 1.0."""
    _LOGGER.debug(f"🔄 Generating simplex grid for k={k}, step={step}")
    grid_start_time = time.time()

    if k == 1:
        _LOGGER.debug("📊 Single dimension, returning [[1.0]]")
        return [[1.0]]

    vals = np.arange(0.0, 1.0 + 1e-9, step)
    _LOGGER.debug(f"📊 Value range: {len(vals)} values from {vals[0]:.3f} to {vals[-1]:.3f}")

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

    _LOGGER.debug("🔄 Generating combinations recursively...")
    rec([], 0)

    grid_time = time.time() - grid_start_time
    _LOGGER.info(f"✅ Simplex grid generated in {grid_time:.3f}s: {len(combos)} combinations")
    _LOGGER.debug(f"📊 Grid size: {len(combos)} combinations")

    return combos

def _eval_metric(y_true: np.ndarray, pred: np.ndarray, metric: str) -> float:
    """Evaluate metric with comprehensive error handling and sklearn dependency management."""
    _LOGGER.debug(f"🔍 Evaluating metric '{metric}' with shapes: y_true={y_true.shape}, pred={pred.shape}")

    try:
        # Convert predictions to appropriate format
        if pred.ndim == 1 or (pred.ndim == 2 and pred.shape[1] == 1):
            # regression or binary scores -> threshold at 0.5
            y_pred = (pred.ravel() >= 0.5).astype(int)
            _LOGGER.debug("📊 Using threshold-based prediction conversion")
        elif pred.ndim == 2:
            y_pred = np.argmax(pred, axis=1)
            _LOGGER.debug("📊 Using argmax prediction conversion")
        else:
            y_pred = pred
            _LOGGER.debug("📊 Using direct prediction")

        _LOGGER.debug(f"📊 Converted predictions shape: {y_pred.shape}")
        _LOGGER.debug(f"📊 Prediction range: {y_pred.min()} - {y_pred.max()}")

        if metric == 'accuracy':
            score = float(np.mean(y_true == y_pred))
            _LOGGER.debug(f"📊 Accuracy score: {score:.4f}")
            return score

        if metric == 'balanced_accuracy':
            try:
                from sklearn.metrics import balanced_accuracy_score
                score = float(balanced_accuracy_score(y_true, y_pred))
                _LOGGER.debug(f"📊 Balanced accuracy score: {score:.4f}")
                return score
            except ImportError as e:
                _LOGGER.warning(f"⚠️ sklearn not available for balanced_accuracy: {e}")
                fallback_score = float(np.mean(y_true == y_pred))
                _LOGGER.debug(f"📊 Using fallback accuracy: {fallback_score:.4f}")
                return fallback_score
            except Exception as e:
                _LOGGER.error(f"❌ balanced_accuracy calculation failed: {e}")
                fallback_score = float(np.mean(y_true == y_pred))
                _LOGGER.debug(f"📊 Using fallback accuracy: {fallback_score:.4f}")
                return fallback_score

        if metric == 'f1_macro':
            try:
                from sklearn.metrics import f1_score
                score = float(f1_score(y_true, y_pred, average='macro'))
                _LOGGER.debug(f"📊 F1 macro score: {score:.4f}")
                return score
            except ImportError as e:
                _LOGGER.warning(f"⚠️ sklearn not available for f1_macro: {e}")
                fallback_score = float(np.mean(y_true == y_pred))
                _LOGGER.debug(f"📊 Using fallback accuracy: {fallback_score:.4f}")
                return fallback_score
            except Exception as e:
                _LOGGER.error(f"❌ f1_macro calculation failed: {e}")
                fallback_score = float(np.mean(y_true == y_pred))
                _LOGGER.debug(f"📊 Using fallback accuracy: {fallback_score:.4f}")
                return fallback_score

        # Default to accuracy
        score = float(np.mean(y_true == y_pred))
        _LOGGER.debug(f"📊 Default accuracy score: {score:.4f}")
        return score

    except Exception as e:
        _LOGGER.error(f"❌ Metric evaluation failed: {e}")
        _LOGGER.error(f"📋 y_true shape: {y_true.shape}, pred shape: {pred.shape}, metric: {metric}")
        return 0.0

__all__ = [
    'simple_blend',
    'learn_blend_weights',
    'dynamic_regime_ensemble',
]
