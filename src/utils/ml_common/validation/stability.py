from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)

"""
Stability utilities: selection stability across folds/time and aggregation helpers.

Enhanced with M1 GPU acceleration, memory optimization, and parallel processing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
import time

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
        GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
            auto_skim_memory, smart_memory_allocation,
        memory_skim_decorator, auto_memory_skim_decorator,
        auto_memory_skim_context, smart_memory_context
    )
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

try:
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
    cpu_optimizer = get_comprehensive_optimizer()

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
    gpu_manager = get_integrated_hardware_manager().gpu_manager()

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

class StabilityAnalyzer:
    """Comprehensive stability analysis for feature selection.

    Enhanced with M1 GPU acceleration, memory optimization, and parallel processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stability analyzer."""
        self.config = config or {}
        self.logger = _LOGGER.getChild('StabilityAnalyzer')

        # Bootstrap parameters
        self.n_bootstraps = self.config.get('n_bootstraps', 100)
        self.bootstrap_fraction = self.config.get('bootstrap_fraction', 0.8)
        # Threshold for considering a feature stable. Honour the new
        # ``stable_feature_threshold`` configuration option when provided while
        # still supporting the legacy ``stability_threshold`` key.
        self.stability_threshold = self.config.get(
            'stable_feature_threshold',
            self.config.get('stability_threshold', 0.7)
        )

        # Temporal analysis parameters
        self.temporal_windows = self.config.get('temporal_windows', [0.5, 0.7, 0.9])
        self.min_window_size = self.config.get('min_window_size', 100)

        # Cross-dataset parameters
        self.min_dataset_overlap = self.config.get('min_dataset_overlap', 0.3)

        _LOGGER.info("📈 StabilityAnalyzer initialized")
        _LOGGER.info(f"⚙️ Bootstrap samples: {self.n_bootstraps}")
        _LOGGER.info(f"⚙️ Bootstrap fraction: {self.bootstrap_fraction}")
        _LOGGER.info(f"⚙️ Stability threshold: {self.stability_threshold}")

    def analyze_bootstrap_stability(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   selection_method: callable,
                                   method_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feature selection stability using bootstrap sampling."""
        start_time = time.time()
        _LOGGER.info(f"📈 Starting bootstrap stability analysis...")
        _LOGGER.info(f"📊 Parameters - Bootstrap samples: {self.n_bootstraps}, Data shape: {X.shape}")

        try:
            n_samples, n_features = X.shape
            bootstrap_size = int(n_samples * self.bootstrap_fraction)

            # Track feature selection across bootstraps
            feature_selection_counts = defaultdict(int)
            feature_scores = defaultdict(list)
            bootstrap_results = []

            # Perform bootstrap sampling
            np.random.seed(42)  # For reproducibility

            for bootstrap_idx in range(self.n_bootstraps):
                _LOGGER.debug(f"🔄 Bootstrap {bootstrap_idx + 1}/{self.n_bootstraps}")

                # Sample bootstrap data
                bootstrap_indices = np.random.choice(n_samples, size=bootstrap_size, replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                try:
                    # Apply selection method
                    result = selection_method(X_bootstrap, y_bootstrap, feature_names, **method_params)

                    if result.get('success', False):
                        selected_features = result.get('selected_features', [])
                        scores = result.get('scores', {})

                        # Count feature selections
                        for feature in selected_features:
                            feature_selection_counts[feature] += 1

                        # Collect scores
                        for feature, score in scores.items():
                            feature_scores[feature].append(score)

                        bootstrap_results.append({
                            'bootstrap_idx': bootstrap_idx,
                            'selected_features': selected_features,
                            'scores': scores,
                            'success': True
                        })
                    else:
                        _LOGGER.warning(f"⚠️ Bootstrap {bootstrap_idx + 1} failed: {result.get('error', 'Unknown error')}")
                        bootstrap_results.append({
                            'bootstrap_idx': bootstrap_idx,
                            'success': False,
                            'error': result.get('error', 'Unknown error')
                        })

                except Exception as e:
                    _LOGGER.warning(f"⚠️ Bootstrap {bootstrap_idx + 1} failed: {e}")
                    bootstrap_results.append({
                        'bootstrap_idx': bootstrap_idx,
                        'success': False,
                        'error': str(e)
                    })

            # Calculate stability scores
            stability_scores = {}
            for feature in feature_names:
                selection_count = feature_selection_counts[feature]
                stability_score = selection_count / self.n_bootstraps
                stability_scores[feature] = stability_score

            # Select stable features and surface unstable ones for reporting
            stable_features = [
                feature for feature, score in stability_scores.items()
                if score >= self.stability_threshold
            ]
            unstable_features = {
                feature: score for feature, score in stability_scores.items()
                if score < self.stability_threshold
            }

            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(bootstrap_results, stability_scores)

            execution_time = time.time() - start_time

            result = {
                'stable_features': stable_features,
                'unstable_features': unstable_features,
                'stability_scores': stability_scores,
                'bootstrap_results': bootstrap_results,
                'stability_metrics': stability_metrics,
                'method': 'bootstrap_stability',
                'parameters': {
                    'n_bootstraps': self.n_bootstraps,
                    'bootstrap_fraction': self.bootstrap_fraction,
                    'stability_threshold': self.stability_threshold
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Bootstrap stability analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Found {len(stable_features)} stable features out of {len(feature_names)}")
            if unstable_features:
                _LOGGER.info(
                    "🚩 Features below stability threshold %.2f: %s",
                    self.stability_threshold,
                    sorted(unstable_features.keys())
                )

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Bootstrap stability analysis failed: {e}")
            return {
                'stable_features': [],
                'stability_scores': {},
                'bootstrap_results': [],
                'stability_metrics': {},
                'method': 'bootstrap_stability',
                'error': str(e),
                'success': False
            }

    def _calculate_stability_metrics(self, bootstrap_results: List[Dict[str, Any]],
                                   stability_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comprehensive stability metrics."""
        try:
            successful_bootstraps = [r for r in bootstrap_results if r.get('success', False)]

            if not successful_bootstraps:
                return {'error': 'No successful bootstrap results'}

            # Calculate feature overlap metrics
            all_selected_features = [r['selected_features'] for r in successful_bootstraps]

            # Jaccard similarity between bootstrap results
            jaccard_similarities = []
            for i in range(len(all_selected_features)):
                for j in range(i + 1, len(all_selected_features)):
                    set1 = set(all_selected_features[i])
                    set2 = set(all_selected_features[j])
                    if set1 or set2:  # Avoid division by zero
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        jaccard_similarities.append(jaccard)

            # Stability distribution
            stability_values = list(stability_scores.values())

            metrics = {
                'n_successful_bootstraps': len(successful_bootstraps),
                'n_total_bootstraps': len(bootstrap_results),
                'success_rate': len(successful_bootstraps) / len(bootstrap_results),
                'mean_jaccard_similarity': np.mean(jaccard_similarities) if jaccard_similarities else 0,
                'std_jaccard_similarity': np.std(jaccard_similarities) if jaccard_similarities else 0,
                'mean_stability_score': np.mean(stability_values),
                'std_stability_score': np.std(stability_values),
                'min_stability_score': np.min(stability_values),
                'max_stability_score': np.max(stability_values),
                'features_above_threshold': sum(1 for score in stability_values if score >= self.stability_threshold),
                'features_below_threshold': sum(1 for score in stability_values if score < self.stability_threshold),
                'stability_threshold': self.stability_threshold
            }

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to calculate stability metrics: {e}")
            return {'error': str(e)}

__all__ = [
    'feature_selection_stability',
    'aggregate_time_blocks',
    'StabilityAnalyzer',
]
