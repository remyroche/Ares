from src.utils.tprint import tprint

"""
Stability Analysis Component

This module provides comprehensive stability analysis for feature selection,
including bootstrap validation, temporal stability, and cross-dataset validation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
from collections import defaultdict
import warnings

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.StabilityAnalysis")
    tprint("✅ Custom logger available for FeatureSelection.StabilityAnalysis")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.StabilityAnalysis")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

class StabilityAnalyzer:
    """Comprehensive stability analysis for feature selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stability analyzer."""
        self.config = config or {}
        self.logger = logger.getChild('StabilityAnalyzer')

        # Bootstrap parameters
        self.n_bootstraps = self._get_bootstrap_count()
        self.bootstrap_fraction = self.config.get('bootstrap_fraction', 0.8)
        # Threshold for considering a feature stable. Prefer the new
        # ``stable_feature_threshold`` configuration key but fall back to the
        # legacy ``stability_threshold`` option to maintain backwards
        # compatibility with existing pipelines.
        self.stability_threshold = self.config.get(
            'stable_feature_threshold',
            self.config.get('stability_threshold', 0.8)  # Increased from 0.7 to 0.8 for more stringent selection
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

    def _get_bootstrap_count(self) -> int:
        """Get bootstrap count based on execution mode."""
        # Get mode from config, default to 'blank' for backward compatibility
        mode = self.config.get('mode', 'blank').lower()

        # Define bootstrap counts per mode
        bootstrap_counts = {
            'full': 100,   # FULL mode: 100 bootstrap samples
            'blank': 30,   # BLANK mode: 30 bootstrap samples (increased for better stability)
            'light': 2     # LIGHT mode: 2 bootstrap samples
        }

        bootstrap_count = bootstrap_counts.get(mode, 5)  # Default to 5 if unknown mode

        _LOGGER.info(f"📊 Bootstrap count for mode '{mode}': {bootstrap_count}")
        return bootstrap_count

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

            # Select stable features and capture unstable ones for reporting
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

    def analyze_temporal_stability(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str],
                                  selection_method: callable,
                                  method_params: Dict[str, Any],
                                  temporal_indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze temporal stability of feature selection."""
        start_time = time.time()
        _LOGGER.info(f"📈 Starting temporal stability analysis...")
        _LOGGER.info(f"📊 Parameters - Temporal windows: {self.temporal_windows}, Data shape: {X.shape}")

        try:
            n_samples, n_features = X.shape

            # Use provided temporal indices or create default
            if temporal_indices is None:
                temporal_indices = np.arange(n_samples)

            # Analyze stability across temporal windows
            window_results = []
            feature_temporal_scores = defaultdict(list)

            for window_fraction in self.temporal_windows:
                window_size = int(n_samples * window_fraction)

                if window_size < self.min_window_size:
                    _LOGGER.warning(f"⚠️ Window size {window_size} too small, skipping")
                    continue

                _LOGGER.debug(f"🔄 Analyzing temporal window: {window_fraction} ({window_size} samples)")

                # Use the first window_size samples
                X_window = X[:window_size]
                y_window = y[:window_size]

                try:
                    # Apply selection method
                    result = selection_method(X_window, y_window, feature_names, **method_params)

                    if result.get('success', False):
                        selected_features = result.get('selected_features', [])
                        scores = result.get('scores', {})

                        # Record temporal scores
                        for feature, score in scores.items():
                            feature_temporal_scores[feature].append(score)

                        window_results.append({
                            'window_fraction': window_fraction,
                            'window_size': window_size,
                            'selected_features': selected_features,
                            'scores': scores,
                            'success': True
                        })
                    else:
                        _LOGGER.warning(f"⚠️ Temporal window {window_fraction} failed: {result.get('error', 'Unknown error')}")
                        window_results.append({
                            'window_fraction': window_fraction,
                            'window_size': window_size,
                            'success': False,
                            'error': result.get('error', 'Unknown error')
                        })

                except Exception as e:
                    _LOGGER.warning(f"⚠️ Temporal window {window_fraction} failed: {e}")
                    window_results.append({
                        'window_fraction': window_fraction,
                        'window_size': window_size,
                        'success': False,
                        'error': str(e)
                    })

            # Calculate temporal stability metrics
            temporal_stability_metrics = self._calculate_temporal_stability_metrics(
                window_results, feature_temporal_scores
            )

            execution_time = time.time() - start_time

            result = {
                'window_results': window_results,
                'feature_temporal_scores': dict(feature_temporal_scores),
                'temporal_stability_metrics': temporal_stability_metrics,
                'method': 'temporal_stability',
                'parameters': {
                    'temporal_windows': self.temporal_windows,
                    'min_window_size': self.min_window_size
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Temporal stability analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Analyzed {len(window_results)} temporal windows")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Temporal stability analysis failed: {e}")
            return {
                'window_results': [],
                'feature_temporal_scores': {},
                'temporal_stability_metrics': {},
                'method': 'temporal_stability',
                'error': str(e),
                'success': False
            }

    def analyze_cross_dataset_stability(self, datasets: List[Dict[str, Any]],
                                       selection_method: callable,
                                       method_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stability across multiple datasets."""
        start_time = time.time()
        _LOGGER.info(f"📈 Starting cross-dataset stability analysis...")
        _LOGGER.info(f"📊 Parameters - Datasets: {len(datasets)}")

        try:
            dataset_results = []
            feature_dataset_scores = defaultdict(list)
            all_feature_names = set()

            # Analyze each dataset
            for dataset_idx, dataset in enumerate(datasets):
                _LOGGER.debug(f"🔄 Analyzing dataset {dataset_idx + 1}/{len(datasets)}")

                X = dataset.get('X')
                y = dataset.get('y')
                feature_names = dataset.get('feature_names', [])
                dataset_name = dataset.get('name', f'dataset_{dataset_idx}')

                if X is None or y is None:
                    _LOGGER.warning(f"⚠️ Dataset {dataset_name} missing X or y")
                    continue

                all_feature_names.update(feature_names)

                try:
                    # Apply selection method
                    result = selection_method(X, y, feature_names, **method_params)

                    if result.get('success', False):
                        selected_features = result.get('selected_features', [])
                        scores = result.get('scores', {})

                        # Record dataset scores
                        for feature, score in scores.items():
                            feature_dataset_scores[feature].append(score)

                        dataset_results.append({
                            'dataset_name': dataset_name,
                            'dataset_idx': dataset_idx,
                            'selected_features': selected_features,
                            'scores': scores,
                            'success': True
                        })
                    else:
                        _LOGGER.warning(f"⚠️ Dataset {dataset_name} failed: {result.get('error', 'Unknown error')}")
                        dataset_results.append({
                            'dataset_name': dataset_name,
                            'dataset_idx': dataset_idx,
                            'success': False,
                            'error': result.get('error', 'Unknown error')
                        })

                except Exception as e:
                    _LOGGER.warning(f"⚠️ Dataset {dataset_name} failed: {e}")
                    dataset_results.append({
                        'dataset_name': dataset_name,
                        'dataset_idx': dataset_idx,
                        'success': False,
                        'error': str(e)
                    })

            # Calculate cross-dataset stability metrics
            cross_dataset_metrics = self._calculate_cross_dataset_stability_metrics(
                dataset_results, feature_dataset_scores, list(all_feature_names)
            )

            execution_time = time.time() - start_time

            result = {
                'dataset_results': dataset_results,
                'feature_dataset_scores': dict(feature_dataset_scores),
                'cross_dataset_metrics': cross_dataset_metrics,
                'method': 'cross_dataset_stability',
                'parameters': {
                    'n_datasets': len(datasets),
                    'min_dataset_overlap': self.min_dataset_overlap
                },
                'execution_time': execution_time,
                'success': True
            }

            _LOGGER.info(f"✅ Cross-dataset stability analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Analyzed {len(dataset_results)} datasets")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Cross-dataset stability analysis failed: {e}")
            return {
                'dataset_results': [],
                'feature_dataset_scores': {},
                'cross_dataset_metrics': {},
                'method': 'cross_dataset_stability',
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

    def _calculate_temporal_stability_metrics(self, window_results: List[Dict[str, Any]],
                                            feature_temporal_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate temporal stability metrics."""
        try:
            successful_windows = [r for r in window_results if r.get('success', False)]

            if not successful_windows:
                return {'error': 'No successful temporal windows'}

            # Calculate temporal consistency
            temporal_consistency = {}
            for feature, scores in feature_temporal_scores.items():
                if len(scores) > 1:
                    # Calculate coefficient of variation
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    cv = std_score / mean_score if mean_score > 0 else 0
                    temporal_consistency[feature] = 1 - cv  # Higher is more consistent
                else:
                    temporal_consistency[feature] = 1.0

            # Calculate feature selection consistency across windows
            all_selected_features = [r['selected_features'] for r in successful_windows]

            # Jaccard similarity between temporal windows
            jaccard_similarities = []
            for i in range(len(all_selected_features)):
                for j in range(i + 1, len(all_selected_features)):
                    set1 = set(all_selected_features[i])
                    set2 = set(all_selected_features[j])
                    if set1 or set2:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        jaccard_similarities.append(jaccard)

            metrics = {
                'n_successful_windows': len(successful_windows),
                'n_total_windows': len(window_results),
                'success_rate': len(successful_windows) / len(window_results),
                'mean_temporal_consistency': np.mean(list(temporal_consistency.values())),
                'std_temporal_consistency': np.std(list(temporal_consistency.values())),
                'mean_jaccard_similarity': np.mean(jaccard_similarities) if jaccard_similarities else 0,
                'std_jaccard_similarity': np.std(jaccard_similarities) if jaccard_similarities else 0,
                'temporal_consistency_by_feature': temporal_consistency
            }

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to calculate temporal stability metrics: {e}")
            return {'error': str(e)}

    def _calculate_cross_dataset_stability_metrics(self, dataset_results: List[Dict[str, Any]],
                                                 feature_dataset_scores: Dict[str, List[float]],
                                                 all_feature_names: List[str]) -> Dict[str, Any]:
        """Calculate cross-dataset stability metrics."""
        try:
            successful_datasets = [r for r in dataset_results if r.get('success', False)]

            if not successful_datasets:
                return {'error': 'No successful datasets'}

            # Calculate cross-dataset consistency
            cross_dataset_consistency = {}
            for feature, scores in feature_dataset_scores.items():
                if len(scores) > 1:
                    # Calculate coefficient of variation
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    cv = std_score / mean_score if mean_score > 0 else 0
                    cross_dataset_consistency[feature] = 1 - cv  # Higher is more consistent
                else:
                    cross_dataset_consistency[feature] = 1.0

            # Calculate feature selection consistency across datasets
            all_selected_features = [r['selected_features'] for r in successful_datasets]

            # Jaccard similarity between datasets
            jaccard_similarities = []
            for i in range(len(all_selected_features)):
                for j in range(i + 1, len(all_selected_features)):
                    set1 = set(all_selected_features[i])
                    set2 = set(all_selected_features[j])
                    if set1 or set2:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        jaccard_similarities.append(jaccard)

            # Calculate feature overlap
            feature_overlap = {}
            for feature in all_feature_names:
                overlap_count = sum(1 for features in all_selected_features if feature in features)
                feature_overlap[feature] = overlap_count / len(successful_datasets)

            metrics = {
                'n_successful_datasets': len(successful_datasets),
                'n_total_datasets': len(dataset_results),
                'success_rate': len(successful_datasets) / len(dataset_results),
                'mean_cross_dataset_consistency': np.mean(list(cross_dataset_consistency.values())),
                'std_cross_dataset_consistency': np.std(list(cross_dataset_consistency.values())),
                'mean_jaccard_similarity': np.mean(jaccard_similarities) if jaccard_similarities else 0,
                'std_jaccard_similarity': np.std(jaccard_similarities) if jaccard_similarities else 0,
                'mean_feature_overlap': np.mean(list(feature_overlap.values())),
                'std_feature_overlap': np.std(list(feature_overlap.values())),
                'cross_dataset_consistency_by_feature': cross_dataset_consistency,
                'feature_overlap_by_feature': feature_overlap
            }

            return metrics

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to calculate cross-dataset stability metrics: {e}")
            return {'error': str(e)}

    def get_stability_summary(self, stability_results: Dict[str, Any]) -> str:
        """Generate stability analysis summary."""
        try:
            method = stability_results.get('method', 'unknown')

            if method == 'bootstrap_stability':
                metrics = stability_results.get('stability_metrics', {})
                stable_features = stability_results.get('stable_features', [])

                summary = f"""
=== Bootstrap Stability Analysis Summary ===
Method: {method}
Execution Time: {stability_results.get('execution_time', 0):.3f}s

=== Results ===
Stable Features: {len(stable_features)}
Total Features: {len(stability_results.get('stability_scores', {}))}
Success Rate: {metrics.get('success_rate', 0):.2%}

=== Stability Metrics ===
Mean Stability Score: {metrics.get('mean_stability_score', 0):.3f}
Std Stability Score: {metrics.get('std_stability_score', 0):.3f}
Mean Jaccard Similarity: {metrics.get('mean_jaccard_similarity', 0):.3f}
Features Above Threshold: {metrics.get('features_above_threshold', 0)}
"""

            elif method == 'temporal_stability':
                metrics = stability_results.get('temporal_stability_metrics', {})

                summary = f"""
=== Temporal Stability Analysis Summary ===
Method: {method}
Execution Time: {stability_results.get('execution_time', 0):.3f}s

=== Results ===
Successful Windows: {metrics.get('n_successful_windows', 0)}
Total Windows: {metrics.get('n_total_windows', 0)}
Success Rate: {metrics.get('success_rate', 0):.2%}

=== Temporal Metrics ===
Mean Temporal Consistency: {metrics.get('mean_temporal_consistency', 0):.3f}
Mean Jaccard Similarity: {metrics.get('mean_jaccard_similarity', 0):.3f}
"""

            elif method == 'cross_dataset_stability':
                metrics = stability_results.get('cross_dataset_metrics', {})

                summary = f"""
=== Cross-Dataset Stability Analysis Summary ===
Method: {method}
Execution Time: {stability_results.get('execution_time', 0):.3f}s

=== Results ===
Successful Datasets: {metrics.get('n_successful_datasets', 0)}
Total Datasets: {metrics.get('n_total_datasets', 0)}
Success Rate: {metrics.get('success_rate', 0):.2%}

=== Cross-Dataset Metrics ===
Mean Cross-Dataset Consistency: {metrics.get('mean_cross_dataset_consistency', 0):.3f}
Mean Jaccard Similarity: {metrics.get('mean_jaccard_similarity', 0):.3f}
Mean Feature Overlap: {metrics.get('mean_feature_overlap', 0):.3f}
"""

            else:
                summary = f"Unknown stability analysis method: {method}"

            return summary

        except Exception as e:
            _LOGGER.error(f"❌ Failed to generate stability summary: {e}")
            return f"Error generating stability summary: {e}"
