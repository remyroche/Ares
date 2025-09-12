"""
Stability utilities: selection stability across folds/time and aggregation helpers.

Enhanced with M1 GPU acceleration, memory optimization, and parallel processing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# Import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Stability")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Stability")

# Import M1 utilities
try:
    from ..hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from ..hardware.m1_memory_optimizer import (
        auto_skim_memory, smart_memory_allocation,
        memory_skim_decorator, auto_memory_skim_decorator,
        auto_memory_skim_context, smart_memory_context
    )
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

try:
    from ..hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False


def feature_selection_stability(
    fold_selections: List[List[str]],
    all_features: List[str],
    use_parallel: bool = True,
) -> Dict[str, Any]:
    """Compute selection frequency and stability score for each feature.

    Enhanced with parallel processing for large datasets.

    Args:
        fold_selections: List of selected features for each fold
        all_features: List of all possible features
        use_parallel: Whether to use parallel processing for large datasets

    Returns:
        Dict with selection counts, stability scores, and fold count
    """
    _LOGGER.info(f"🔄 Computing feature selection stability for {len(all_features)} features across {len(fold_selections)} folds")
    start_time = time.time()
    
    n_folds = max(1, len(fold_selections))
    _LOGGER.debug(f"📊 Number of folds: {n_folds}")
    _LOGGER.debug(f"📊 Total features: {len(all_features)}")
    _LOGGER.debug(f"📊 Use parallel processing: {use_parallel}")

    # Use parallel processing for large feature sets
    if use_parallel and CPU_OPTIMIZER_AVAILABLE and len(all_features) > 100:
        _LOGGER.debug("🚀 Attempting parallel stability calculation...")
        try:
            result = _feature_selection_stability_parallel(fold_selections, all_features, n_folds)
            stability_time = time.time() - start_time
            _LOGGER.info(f"✅ Parallel stability calculation completed in {stability_time:.3f}s")
            return result
        except Exception as e:
            _LOGGER.warning(f"⚠️ Parallel stability calculation failed: {e}, falling back to sequential")

    # Sequential implementation
    _LOGGER.debug("🔄 Using sequential stability calculation...")
    counts: Dict[str, int] = {f: 0 for f in all_features}
    
    for i, sel in enumerate(fold_selections):
        if i % 10 == 0:  # Log progress every 10 folds
            progress = (i / len(fold_selections)) * 100
            _LOGGER.debug(f"📊 Processing fold {i+1}/{len(fold_selections)} ({progress:.1f}%)")
        
        for f in sel:
            if f in counts:
                counts[f] += 1

    stability = {f: counts[f] / n_folds for f in all_features}
    
    stability_time = time.time() - start_time
    _LOGGER.info(f"✅ Sequential stability calculation completed in {stability_time:.3f}s")
    _LOGGER.info(f"📊 Average stability score: {np.mean(list(stability.values())):.4f}")
    _LOGGER.info(f"📊 Max stability score: {max(stability.values()):.4f}")
    _LOGGER.info(f"📊 Min stability score: {min(stability.values()):.4f}")
    
    return {
        'selection_counts': counts,
        'stability_scores': stability,
        'n_folds': n_folds,
    }


def _feature_selection_stability_parallel(
    fold_selections: List[List[str]],
    all_features: List[str],
    n_folds: int,
) -> Dict[str, Any]:
    """Parallel implementation of feature selection stability calculation."""
    cpu_optimizer = get_m1_cpu_optimizer()

    # Initialize counts dictionary
    counts: Dict[str, int] = {f: 0 for f in all_features}

    def count_feature_selections(feature: str) -> Tuple[str, int]:
        """Count how many folds selected this feature."""
        count = sum(1 for sel in fold_selections if feature in sel)
        return feature, count

    # Count selections in parallel
    results = cpu_optimizer.parallel_process(
        all_features,
        count_feature_selections,
        task_type="cpu_bound"
    )

    # Update counts
    for feature, count in results:
        counts[feature] = count

    # Calculate stability scores
    stability = {f: counts[f] / n_folds for f in all_features}

    return {
        'selection_counts': counts,
        'stability_scores': stability,
        'n_folds': n_folds,
    }


def aggregate_time_blocks(
    block_metrics: List[Dict[str, float]],
    keys: List[str],
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Aggregate metrics across time blocks and compute variability.

    Enhanced with GPU acceleration for large datasets.

    Args:
        block_metrics: List of metric dictionaries for each time block
        keys: List of metric keys to aggregate
        use_gpu: Whether to use GPU acceleration for large datasets

    Returns:
        Dict with aggregated statistics for each metric
    """
    # Use GPU acceleration for large datasets
    if use_gpu and GPU_AVAILABLE and len(block_metrics) > 100:
        try:
            return _aggregate_time_blocks_gpu(block_metrics, keys)
        except Exception as e:
            _LOGGER.warning(f"GPU aggregation failed: {e}, falling back to CPU")

    # CPU implementation
    return _aggregate_time_blocks_cpu(block_metrics, keys)


def _aggregate_time_blocks_cpu(
    block_metrics: List[Dict[str, float]],
    keys: List[str],
) -> Dict[str, Any]:
    """CPU implementation of time block aggregation."""
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


def _aggregate_time_blocks_gpu(
    block_metrics: List[Dict[str, float]],
    keys: List[str],
) -> Dict[str, Any]:
    """GPU-accelerated time block aggregation."""
    gpu_manager = M1GPUManager()

    with gpu_manager.gpu_context("time_block_aggregation"):
        agg: Dict[str, Any] = {}

        for k in keys:
            vals = [b.get(k) for b in block_metrics if k in b]
            if vals:
                # Convert to tensor and move to GPU
                arr_cpu = np.array(vals, dtype=float)
                arr_gpu = gpu_manager.to_device(arr_cpu, "general")

                # Compute statistics on GPU
                mean_gpu = torch.mean(arr_gpu)
                std_gpu = torch.std(arr_gpu)
                min_gpu = torch.min(arr_gpu)
                max_gpu = torch.max(arr_gpu)

                # Convert back to CPU for coefficient of variation
                mean_val = float(mean_gpu.cpu().numpy())
                std_val = float(std_gpu.cpu().numpy())
                min_val = float(min_gpu.cpu().numpy())
                max_val = float(max_gpu.cpu().numpy())

                # Calculate coefficient of variation
                cv_val = std_val / (mean_val if mean_val != 0 else 1.0)

                agg[k] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv_val,
                    'min': min_val,
                    'max': max_val,
                }

    return agg


__all__ = [
    'feature_selection_stability',
    'aggregate_time_blocks',
]


