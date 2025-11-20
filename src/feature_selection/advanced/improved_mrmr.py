"""
Improved mRMR with 70% MI + 30% Spearman (Rank-based)

This module implements the improved mRMR approach with:
- 70% Mutual Information + 30% Spearman correlation
- Rank-based scoring with z-score normalization
- Quantile binning for MI calculation
- Greedy selection until 50% of original features
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug, tprint_info
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

# Import MI proxy for optimized mutual information computation
from .mi_proxy import MIProxy, create_mi_proxy

logger = logging.getLogger(__name__)

class ImprovedMRMR:
    """Improved mRMR with rank-based scoring and quantile binning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize improved mRMR with MI proxy optimization."""
        default_config = {
            'mi_weight': 0.7,
            'spearman_weight': 0.3,
            'target_ratio': 0.5,  # Select top 50% of features
            'quantile_bins': 10,  # Number of quantile bins for MI
            'epsilon': 1e-8,  # Small constant for division
            'use_rank_based': True,
            'enable_cv_relevance': True,
            'cv_folds': 5,
            'enable_hardware_optimization': True,
            'use_mi_proxy': True,  # Enable MI proxy optimization
            'n_jobs': -1,
            'random_state': 42,
            'verbose': True,
            'export_per_feature_csv': False  # Export per-feature scores to CSV
        }

        if config is not None:
            merged_config: Dict[str, Any] = default_config.copy()
            merged_config.update(config)
            self.config = merged_config
        else:
            self.config = default_config

        self.logger = logger.getChild('ImprovedMRMR')

        # Initialize hardware optimization
        if self.config.get('enable_hardware_optimization', True):
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='balanced',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None

        # Initialize MI proxy for optimized MI computation
        if self.config.get('use_mi_proxy', True):
            mi_proxy_config = {
                'n_bins': self.config.get('quantile_bins', 10),
                'cv_folds': self.config.get('cv_folds', 5),
                'use_numba': True,
                'quantization_strategy': 'quantile'
            }
            self.mi_proxy = create_mi_proxy(mi_proxy_config)
        else:
            self.mi_proxy = None

        # Performance tracking
        self.performance_stats = {
            'total_selections': 0,
            'avg_selection_time': 0.0,
            'features_removed': 0,
            'mi_calculations': 0,
            'spearman_calculations': 0,
            'quantile_binnings': 0,
            'mi_proxy_stats': {}
        }

        # Store per-feature results for CSV export
        self.per_feature_results = {}

        tprint_success("🔧 ImprovedMRMR initialized with MI proxy optimization")

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       feature_names: Optional[List[str]] = None,
                       target_ratio: Optional[float] = None) -> Dict[str, Any]:
        """Select features using improved mRMR approach."""
        tprint_info(f"🔧 Improved mRMR selection: {X.shape}")

        start_time = time.time()

        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Use provided target ratio or default
            target_ratio = target_ratio or self.config['target_ratio']
            n_target = max(1, int(X.shape[1] * target_ratio))

            tprint_debug(f"🔧 Target features: {n_target} (ratio: {target_ratio})")

            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            # Step 1: Compute relevance scores
            relevance_scores = self._compute_relevance_scores(X, y, feature_names, is_classification)

            # Step 2: Greedy selection with redundancy calculation
            selected_features = self._greedy_selection(
                X, y, feature_names, relevance_scores, n_target, is_classification
            )

            # Create feature mask
            feature_mask = np.zeros(X.shape[1], dtype=bool)
            feature_mask[selected_features] = True

            # Get selected features data
            X_selected = X[:, selected_features]
            selected_feature_names = [feature_names[i] for i in selected_features]

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_selections'] += 1
            self.performance_stats['features_removed'] += X.shape[1] - len(selected_features)
            self.performance_stats['avg_selection_time'] = (
                (self.performance_stats['avg_selection_time'] * (self.performance_stats['total_selections'] - 1) +
                 execution_time) / self.performance_stats['total_selections']
            )

            # Store per-feature results
            self.per_feature_results = {
                'feature_names': feature_names,
                'relevance_scores': relevance_scores,
                'selected_indices': selected_features,
                'selected_features': selected_feature_names,
                'n_original': X.shape[1],
                'n_selected': len(selected_features)
            }

            # Export per-feature CSV if requested
            if self.config.get('export_per_feature_csv', False):
                csv_path = self._export_per_feature_csv(feature_names, relevance_scores, selected_features)
                tprint_info(f"📊 Per-feature scores exported to: {csv_path}")
            else:
                csv_path = None

            result = {
                'success': True,
                'selected_features': selected_feature_names,
                'selected_indices': selected_features,
                'feature_mask': feature_mask,
                'X_selected': X_selected,
                'n_original': X.shape[1],
                'n_selected': len(selected_features),
                'selection_ratio': len(selected_features) / X.shape[1],
                'relevance_scores': relevance_scores,
                'execution_time': execution_time,
                'method': 'improved_mrmr',
                'mi_proxy_stats': self.performance_stats.get('mi_proxy_stats', {}),
                'per_feature_csv_path': csv_path
            }

            tprint_success(f"✅ Improved mRMR completed: {X.shape[1]} -> {len(selected_features)} features in {execution_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Improved mRMR selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _compute_relevance_scores(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str], is_classification: bool) -> Dict[str, float]:
        """Compute relevance scores using MI and Spearman correlation."""
        tprint_debug("🔧 Computing relevance scores")

        try:
            n_features = X.shape[1]

            # Compute MI relevance
            mi_scores = self._compute_mi_relevance(X, y, is_classification)
            self.performance_stats['mi_calculations'] += 1

            # Compute Spearman relevance
            spearman_scores = self._compute_spearman_relevance(X, y)
            self.performance_stats['spearman_calculations'] += 1

            # Combine scores with rank-based approach
            if self.config['use_rank_based']:
                relevance_scores = self._combine_rank_based_scores(
                    mi_scores, spearman_scores, feature_names
                )
            else:
                relevance_scores = self._combine_direct_scores(
                    mi_scores, spearman_scores, feature_names
                )

            return relevance_scores

        except Exception as e:
            self.logger.error(f"Relevance score computation failed: {e}")
            # Fallback to simple MI scores
            return self._compute_mi_relevance(X, y, is_classification)

    def _compute_mi_relevance(self, X: np.ndarray, y: np.ndarray,
                            is_classification: bool) -> Dict[str, float]:
        """Compute mutual information relevance scores using MI proxy when available."""
        try:
            # Use MI proxy if available and enabled
            if self.mi_proxy is not None:
                tprint_debug("🚀 Using MI proxy for optimized mutual information computation")
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                mi_dict = self.mi_proxy.compute_mi_target(X, y, feature_names)

                # Store MI proxy stats
                self.performance_stats['mi_proxy_stats'] = self.mi_proxy.get_performance_stats()
                return mi_dict

            # Fallback to sklearn's MI functions
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=self.config['random_state'])
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.config['random_state'])

            # Convert to dictionary
            mi_dict = {}
            for i, score in enumerate(mi_scores):
                mi_dict[f"feature_{i}"] = float(score)

            return mi_dict

        except Exception as e:
            self.logger.warning(f"MI calculation failed: {e}")
            # Fallback to simple correlation
            correlations = np.abs(np.corrcoef(X.T, y)[-1, :-1])
            return {f"feature_{i}": float(correlations[i]) for i in range(X.shape[1])}

    def _compute_spearman_relevance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute Spearman correlation relevance scores."""
        try:
            spearman_scores = {}

            for i in range(X.shape[1]):
                correlation, _ = spearmanr(X[:, i], y)
                spearman_scores[f"feature_{i}"] = float(abs(correlation))

            return spearman_scores

        except Exception as e:
            self.logger.warning(f"Spearman calculation failed: {e}")
            # Fallback to simple correlation
            correlations = np.abs(np.corrcoef(X.T, y)[-1, :-1])
            return {f"feature_{i}": float(correlations[i]) for i in range(X.shape[1])}

    def _combine_rank_based_scores(self, mi_scores: Dict[str, float],
                                 spearman_scores: Dict[str, float],
                                 feature_names: List[str]) -> Dict[str, float]:
        """Combine scores using rank-based approach with z-score normalization."""
        tprint_debug("🔧 Combining scores with rank-based approach")

        try:
            # Get feature names in order
            feature_keys = [f"feature_{i}" for i in range(len(feature_names))]

            # Extract scores in order
            mi_values = np.array([mi_scores.get(key, 0.0) for key in feature_keys])
            spearman_values = np.array([spearman_scores.get(key, 0.0) for key in feature_keys])

            # Rank scores (descending order)
            mi_ranks = rankdata(-mi_values, method='dense')  # Negative for descending
            spearman_ranks = rankdata(-spearman_values, method='dense')

            # Z-score normalize ranks
            mi_ranks_z = (mi_ranks - np.mean(mi_ranks)) / (np.std(mi_ranks) + 1e-8)
            spearman_ranks_z = (spearman_ranks - np.mean(spearman_ranks)) / (np.std(spearman_ranks) + 1e-8)

            # Combine with weights
            mi_weight = self.config['mi_weight']
            spearman_weight = self.config['spearman_weight']

            combined_scores = mi_weight * mi_ranks_z + spearman_weight * spearman_ranks_z

            # Convert back to dictionary
            relevance_scores = {}
            for i, feature_name in enumerate(feature_names):
                relevance_scores[feature_name] = float(combined_scores[i])

            return relevance_scores

        except Exception as e:
            self.logger.warning(f"Rank-based combination failed: {e}")
            return self._combine_direct_scores(mi_scores, spearman_scores, feature_names)

    def _combine_direct_scores(self, mi_scores: Dict[str, float],
                             spearman_scores: Dict[str, float],
                             feature_names: List[str]) -> Dict[str, float]:
        """Combine scores using direct weighted average."""
        try:
            relevance_scores = {}
            mi_weight = self.config['mi_weight']
            spearman_weight = self.config['spearman_weight']

            for i, feature_name in enumerate(feature_names):
                key = f"feature_{i}"
                mi_score = mi_scores.get(key, 0.0)
                spearman_score = spearman_scores.get(key, 0.0)

                combined_score = mi_weight * mi_score + spearman_weight * spearman_score
                relevance_scores[feature_name] = float(combined_score)

            return relevance_scores

        except Exception as e:
            self.logger.warning(f"Direct score combination failed: {e}")
            return mi_scores  # Fallback to MI scores only

    def _greedy_selection(self, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str], relevance_scores: Dict[str, float],
                        n_target: int, is_classification: bool) -> List[int]:
        """Greedy selection with redundancy calculation."""
        tprint_debug(f"🔧 Greedy selection: {n_target} features")

        try:
            # Create a mapping from feature names to indices
            feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

            n_features = X.shape[1]
            selected_features = []
            remaining_features = list(range(n_features))

            # Start with the feature with highest relevance
            best_feature_name = max(relevance_scores.items(), key=lambda x: x[1])[0]
            best_idx = feature_name_to_idx.get(best_feature_name, 0)
            selected_features.append(best_idx)
            remaining_features.remove(best_idx)

            # Greedy selection loop
            while len(selected_features) < n_target and remaining_features:
                best_score = -np.inf
                best_candidate = None

                for candidate_idx in remaining_features:
                    # Calculate score for this candidate
                    score = self._calculate_candidate_score(
                        X, y, selected_features, candidate_idx,
                        relevance_scores, is_classification
                    )

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate_idx

                if best_candidate is not None:
                    selected_features.append(best_candidate)
                    remaining_features.remove(best_candidate)
                else:
                    # No improvement found, break
                    break

            return selected_features

        except Exception as e:
            self.logger.error(f"Greedy selection failed: {e}")
            # Fallback to simple relevance-based selection
            # Create a mapping from feature names to indices for fallback
            feature_name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            sorted_features = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
            return [feature_name_to_idx.get(f[0], idx) for idx, (f, score) in enumerate(sorted_features[:n_target]) if f[0] in feature_name_to_idx]

    def _calculate_candidate_score(self, X: np.ndarray, y: np.ndarray,
                                 selected_features: List[int], candidate_idx: int,
                                 relevance_scores: Dict[str, float],
                                 is_classification: bool) -> float:
        """Calculate score for a candidate feature."""
        try:
            # Get relevance score - use the candidate_idx directly as key
            # since relevance_scores uses feature names, not "feature_N" format
            feature_keys = list(relevance_scores.keys())
            relevance_key = feature_keys[candidate_idx] if candidate_idx < len(feature_keys) else f"feature_{candidate_idx}"
            relevance = relevance_scores.get(relevance_key, 0.0)

            # Calculate redundancy with selected features
            redundancy = self._calculate_redundancy(
                X, selected_features, candidate_idx, is_classification
            )

            # Use mRMR criterion: relevance - redundancy
            # Or relevance / (redundancy + epsilon) for highly collinear features
            epsilon = self.config['epsilon']

            if redundancy > 0.5:  # High collinearity threshold
                score = relevance / (redundancy + epsilon)
            else:
                score = relevance - redundancy

            return float(score)

        except Exception as e:
            self.logger.warning(f"Candidate score calculation failed: {e}")
            # Fallback to relevance only
            feature_keys = list(relevance_scores.keys())
            relevance_key = feature_keys[candidate_idx] if candidate_idx < len(feature_keys) else f"feature_{candidate_idx}"
            return relevance_scores.get(relevance_key, 0.0)

    def _calculate_redundancy(self, X: np.ndarray, selected_features: List[int],
                            candidate_idx: int, is_classification: bool) -> float:
        """Calculate redundancy between candidate and selected features."""
        try:
            if not selected_features:
                return 0.0

            # Calculate MI and Spearman redundancy
            mi_redundancy = self._calculate_mi_redundancy(
                X, selected_features, candidate_idx, is_classification
            )
            spearman_redundancy = self._calculate_spearman_redundancy(
                X, selected_features, candidate_idx
            )

            # Combine with weights (same as relevance)
            mi_weight = self.config['mi_weight']
            spearman_weight = self.config['spearman_weight']

            redundancy = mi_weight * mi_redundancy + spearman_weight * spearman_redundancy

            return float(redundancy)

        except Exception as e:
            self.logger.warning(f"Redundancy calculation failed: {e}")
            return 0.0

    def _calculate_mi_redundancy(self, X: np.ndarray, selected_features: List[int],
                               candidate_idx: int, is_classification: bool) -> float:
        """Calculate MI redundancy between candidate and selected features."""
        try:
            mi_scores = []

            for selected_idx in selected_features:
                # Use quantile binning for MI calculation
                X_candidate = self._quantile_bin(X[:, candidate_idx])
                X_selected = self._quantile_bin(X[:, selected_idx])

                # Calculate MI between features
                if is_classification:
                    mi_score = mutual_info_classif(
                        X_candidate.reshape(-1, 1), X_selected,
                        random_state=self.config['random_state']
                    )[0]
                else:
                    mi_score = mutual_info_regression(
                        X_candidate.reshape(-1, 1), X_selected,
                        random_state=self.config['random_state']
                    )[0]

                mi_scores.append(mi_score)

            return float(np.mean(mi_scores))

        except Exception as e:
            self.logger.warning(f"MI redundancy calculation failed: {e}")
            return 0.0

    def _calculate_spearman_redundancy(self, X: np.ndarray, selected_features: List[int],
                                     candidate_idx: int) -> float:
        """Calculate Spearman redundancy between candidate and selected features."""
        try:
            spearman_scores = []

            for selected_idx in selected_features:
                correlation, _ = spearmanr(X[:, candidate_idx], X[:, selected_idx])
                spearman_scores.append(abs(correlation))

            return float(np.mean(spearman_scores))

        except Exception as e:
            self.logger.warning(f"Spearman redundancy calculation failed: {e}")
            return 0.0

    def _quantile_bin(self, data: np.ndarray) -> np.ndarray:
        """Apply quantile binning to data."""
        try:
            n_bins = self.config.get('quantile_bins', 10)

            # Handle edge case
            if len(np.unique(data)) <= n_bins:
                return data

            # Create quantile bins
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = np.quantile(data, quantiles)

            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)

            # Digitize data
            binned_data = np.digitize(data, bin_edges[1:-1])

            self.performance_stats['quantile_binnings'] += 1
            return binned_data

        except Exception as e:
            self.logger.warning(f"Quantile binning failed: {e}")
            return data  # Return original data if binning fails

    def _export_per_feature_csv(self, feature_names: List[str],
                              relevance_scores: Dict[str, float],
                              selected_indices: List[int],
                              output_dir: Optional[str] = None) -> str:
        """
        Export per-feature scores and metadata to CSV.

        Args:
            feature_names: List of feature names
            relevance_scores: Dictionary of feature relevance scores
            selected_indices: Indices of selected features
            output_dir: Output directory (uses current working directory if None)

        Returns:
            Path to the saved CSV file
        """
        try:
            import os
            from pathlib import Path

            # Create output directory if not specified
            if output_dir is None:
                output_dir = Path.cwd() / "mrmr_results"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data for CSV
            selected_set = set(selected_indices)
            data = []

            for i, feature_name in enumerate(feature_names):
                relevance_key = f"feature_{i}"
                relevance = relevance_scores.get(relevance_key, 0.0)
                is_selected = i in selected_set

                data.append({
                    'feature_index': i,
                    'feature_name': feature_name,
                    'relevance_score': float(relevance),
                    'is_selected': is_selected,
                    'selection_rank': list(selected_indices).index(i) + 1 if i in selected_set else -1
                })

            # Create DataFrame and sort by relevance score
            df = pd.DataFrame(data)
            df = df.sort_values('relevance_score', ascending=False).reset_index(drop=True)

            # Save to CSV
            csv_file = output_dir / f"per_feature_mrmr_scores.csv"
            df.to_csv(csv_file, index=False)

            tprint_debug(f"✅ Per-feature scores exported to: {csv_file}")

            return str(csv_file)

        except Exception as e:
            self.logger.error(f"Failed to export per-feature CSV: {e}")
            return ""

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_selections'] > 0:
            stats['avg_features_removed'] = stats['features_removed'] / stats['total_selections']
            stats['mi_usage_ratio'] = stats['mi_calculations'] / stats['total_selections']
            stats['spearman_usage_ratio'] = stats['spearman_calculations'] / stats['total_selections']
            stats['quantile_binning_ratio'] = stats['quantile_binnings'] / stats['total_selections']
        else:
            stats['avg_features_removed'] = 0.0
            stats['mi_usage_ratio'] = 0.0
            stats['spearman_usage_ratio'] = 0.0
            stats['quantile_binning_ratio'] = 0.0

        return stats

    def get_selection_insights(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights about selection results."""
        if not result.get('success', False):
            return {'error': 'Selection failed'}

        insights = {
            'n_original': result['n_original'],
            'n_selected': result['n_selected'],
            'selection_ratio': result['selection_ratio'],
            'execution_time': result['execution_time'],
            'relevance_distribution': {},
            'selected_feature_names': result['selected_features'][:10]  # Top 10
        }

        # Analyze relevance distribution
        if 'relevance_scores' in result:
            scores = list(result['relevance_scores'].values())
            if scores:
                insights['relevance_distribution'] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores))
                }

        return insights

def create_improved_mrmr(config: Optional[Dict[str, Any]] = None) -> ImprovedMRMR:
    """Create an improved mRMR selector."""
    return ImprovedMRMR(config)
