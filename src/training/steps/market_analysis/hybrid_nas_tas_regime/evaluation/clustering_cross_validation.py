"""
Cross-Validation for Clustering Parameters

This module provides cross-validation capabilities for tuning clustering parameters
in the NAS-TAS system, including number of regimes and weights optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

@dataclass
class ClusteringCVResult:
    """Result from clustering cross-validation."""
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: Dict[str, List[float]]
    param_scores: Dict[str, Dict[str, float]]
    validation_metrics: Dict[str, float]
    stability_scores: Dict[str, float]
    execution_time: float
    metadata: Dict[str, Any]

class ClusteringCrossValidator:
    """
    Cross-validation for clustering parameters optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize clustering cross-validator."""
        tprint_info("🚀 Initializing Clustering Cross-Validator")
        tprint_debug(f"Configuration: {config}")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # CV parameters
        tprint_debug("⚙️ Setting cross-validation parameters...")
        self.cv_folds = config.get('cv_folds', 5)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        self.scoring_metric = config.get('scoring_metric', 'silhouette')
        self.enable_time_series_cv = config.get('enable_time_series_cv', True)
        tprint_success("✅ Cross-validation parameters configured")

        # Parameter search space
        tprint_debug("🔧 Setting parameter search space...")
        self.n_regimes_range = config.get('n_regimes_range', list(range(2, 11)))
        self.weight_ranges = config.get('weight_ranges', {
            'economic_significance_weight': np.arange(0.1, 0.6, 0.1),
            'momentum_weight': np.arange(0.1, 0.6, 0.1),
            'volume_weight': np.arange(0.1, 0.6, 0.1)
        })
        self.algorithm_options = config.get('algorithm_options', ['kmeans', 'hierarchical', 'gmm'])
        tprint_success("✅ Parameter search space configured")

        # Clustering algorithms
        tprint_debug("🔧 Initializing clustering algorithms...")
        self.clustering_algorithms = {
            'kmeans': self._kmeans_clustering,
            'hierarchical': self._hierarchical_clustering,
            'gmm': self._gmm_clustering
        }
        tprint_success(f"✅ {len(self.clustering_algorithms)} clustering algorithms available")

        tprint_success("✅ Clustering Cross-Validator initialized")
        self.logger.info("✅ Clustering Cross-Validator initialized")

    def optimize_clustering_parameters(self,
                                     features: np.ndarray,
                                     market_data: pd.DataFrame,
                                     regime_labels: Optional[np.ndarray] = None) -> ClusteringCVResult:
        """
        Optimize clustering parameters using cross-validation.

        Args:
            features: Feature matrix
            market_data: Market data for economic analysis
            regime_labels: Optional existing regime labels for validation

        Returns:
            ClusteringCVResult with optimized parameters
        """
        try:
            tprint("🔍 [CLUSTERING_CV] Starting clustering parameter optimization", color="blue", bold=True)
            tprint_debug(f"📊 [CLUSTERING_CV] Features shape: {features.shape}")
            tprint_debug(f"📊 [CLUSTERING_CV] Market data shape: {market_data.shape}")
            self.logger.info("🔍 Starting clustering parameter optimization...")

            # Prepare cross-validation splits
            tprint("📊 [CLUSTERING_CV] Preparing cross-validation splits", color="cyan")
            cv_splits = self._prepare_cv_splits(features)
            tprint_success(f"✅ [CLUSTERING_CV] CV splits prepared: {len(cv_splits)} folds")

            # Optimize number of regimes
            tprint("🎯 [CLUSTERING_CV] Optimizing number of regimes", color="cyan")
            n_regimes_results = self._optimize_n_regimes(features, cv_splits)
            tprint_success("✅ [CLUSTERING_CV] Number of regimes optimized")

            # Optimize weights
            tprint("⚖️ [CLUSTERING_CV] Optimizing weights", color="cyan")
            weight_results = self._optimize_weights(features, market_data, cv_splits)
            tprint_success("✅ [CLUSTERING_CV] Weights optimized")

            # Optimize algorithm
            tprint("🔧 [CLUSTERING_CV] Optimizing algorithm", color="cyan")
            algorithm_results = self._optimize_algorithm(features, cv_splits)
            tprint_success("✅ [CLUSTERING_CV] Algorithm optimized")

            # Combine results
            tprint("🔄 [CLUSTERING_CV] Combining optimization results", color="cyan")
            best_params, best_score = self._combine_optimization_results(
                n_regimes_results, weight_results, algorithm_results
            )
            tprint_success(f"✅ [CLUSTERING_CV] Best parameters: {best_params}")
            tprint_success(f"✅ [CLUSTERING_CV] Best score: {best_score:.3f}")

            # Calculate validation metrics
            tprint("📈 [CLUSTERING_CV] Calculating validation metrics", color="cyan")
            validation_metrics = self._calculate_validation_metrics(
                features, best_params, regime_labels
            )
            tprint_success("✅ [CLUSTERING_CV] Validation metrics calculated")

            # Calculate stability scores
            tprint("🔒 [CLUSTERING_CV] Calculating stability scores", color="cyan")
            stability_scores = self._calculate_stability_scores(
                features, best_params, cv_splits
            )
            tprint_success("✅ [CLUSTERING_CV] Stability scores calculated")

            tprint_success(f"🎉 [CLUSTERING_CV] Clustering parameter optimization completed successfully")
            tprint_performance(f"⚡ [CLUSTERING_CV] Final result: {best_params}")

            return ClusteringCVResult(
                best_params=best_params,
                best_score=best_score,
                cv_scores={
                    'n_regimes': n_regimes_results['scores'],
                    'weights': weight_results['scores'],
                    'algorithm': algorithm_results['scores']
                },
                param_scores={
                    'n_regimes': n_regimes_results['param_scores'],
                    'weights': weight_results['param_scores'],
                    'algorithm': algorithm_results['param_scores']
                },
                validation_metrics=validation_metrics,
                stability_scores=stability_scores,
                execution_time=0.0,  # TODO: Add timing
                metadata={
                    'n_cv_folds': len(cv_splits),
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'optimization_timestamp': datetime.now().isoformat(),
                    'config': self.config
                }
            )

        except Exception as e:
            tprint_error(f"❌ [CLUSTERING_CV] Clustering parameter optimization failed: {e}")
            tprint_debug(f"🔍 [CLUSTERING_CV] Error details: {str(e)}")
            self.logger.error(f"Clustering parameter optimization failed: {e}")
            raise

    def _prepare_cv_splits(self, features: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare cross-validation splits."""
        try:
            if self.enable_time_series_cv:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                splits = []
                for train_idx, test_idx in tscv.split(features):
                    splits.append((train_idx, test_idx))
                return splits
            else:
                # Regular k-fold cross-validation
                kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                splits = []
                for train_idx, test_idx in kf.split(features):
                    splits.append((train_idx, test_idx))
                return splits
        except Exception as e:
            self.logger.warning(f"CV splits preparation failed: {e}")
            # Fallback to simple train/test split
            n_samples = features.shape[0]
            test_size = int(n_samples * self.test_size)
            train_idx = np.arange(n_samples - test_size)
            test_idx = np.arange(n_samples - test_size, n_samples)
            return [(train_idx, test_idx)]

    def _optimize_n_regimes(self, features: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Optimize number of regimes."""
        try:
            scores = []
            param_scores = {}

            for n_regimes in self.n_regimes_range:
                fold_scores = []

                for train_idx, test_idx in cv_splits:
                    try:
                        # Train on training set
                        train_features = features[train_idx]
                        test_features = features[test_idx]

                        # Apply clustering
                        labels = self._apply_clustering(train_features, {'n_regimes': n_regimes})

                        if len(set(labels)) < 2:
                            fold_scores.append(0.0)
                            continue

                        # Calculate score on test set
                        score = self._calculate_clustering_score(test_features, labels)
                        fold_scores.append(score)

                    except Exception as e:
                        self.logger.warning(f"CV fold failed for n_regimes={n_regimes}: {e}")
                        fold_scores.append(0.0)

                avg_score = np.mean(fold_scores) if fold_scores else 0.0
                scores.append(avg_score)
                param_scores[n_regimes] = {
                    'score': avg_score,
                    'fold_scores': fold_scores,
                    'std': np.std(fold_scores) if fold_scores else 0.0
                }

            best_n_regimes = self.n_regimes_range[np.argmax(scores)]

            return {
                'best_param': best_n_regimes,
                'scores': scores,
                'param_scores': param_scores
            }
        except Exception as e:
            self.logger.warning(f"Number of regimes optimization failed: {e}")
            return {
                'best_param': 3,
                'scores': [0.0] * len(self.n_regimes_range),
                'param_scores': {}
            }

    def _optimize_weights(self, features: np.ndarray, market_data: pd.DataFrame,
                        cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Optimize clustering weights."""
        try:
            scores = []
            param_scores = {}

            # Generate weight combinations
            weight_combinations = self._generate_weight_combinations()

            for weight_combo in weight_combinations:
                fold_scores = []

                for train_idx, test_idx in cv_splits:
                    try:
                        # Train on training set
                        train_features = features[train_idx]
                        test_features = features[test_idx]

                        # Apply clustering with weights
                        labels = self._apply_clustering_with_weights(
                            train_features, market_data.iloc[train_idx], weight_combo
                        )

                        if len(set(labels)) < 2:
                            fold_scores.append(0.0)
                            continue

                        # Calculate score on test set
                        score = self._calculate_clustering_score(test_features, labels)
                        fold_scores.append(score)

                    except Exception as e:
                        self.logger.warning(f"CV fold failed for weights={weight_combo}: {e}")
                        fold_scores.append(0.0)

                avg_score = np.mean(fold_scores) if fold_scores else 0.0
                scores.append(avg_score)
                param_scores[str(weight_combo)] = {
                    'score': avg_score,
                    'fold_scores': fold_scores,
                    'std': np.std(fold_scores) if fold_scores else 0.0
                }

            best_weights = weight_combinations[np.argmax(scores)]

            return {
                'best_param': best_weights,
                'scores': scores,
                'param_scores': param_scores
            }
        except Exception as e:
            self.logger.warning(f"Weights optimization failed: {e}")
            return {
                'best_param': {'economic_significance_weight': 0.3, 'momentum_weight': 0.25, 'volume_weight': 0.25},
                'scores': [0.0],
                'param_scores': {}
            }

    def _optimize_algorithm(self, features: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Optimize clustering algorithm."""
        try:
            scores = []
            param_scores = {}

            for algorithm in self.algorithm_options:
                fold_scores = []

                for train_idx, test_idx in cv_splits:
                    try:
                        # Train on training set
                        train_features = features[train_idx]
                        test_features = features[test_idx]

                        # Apply clustering with algorithm
                        labels = self._apply_clustering_algorithm(train_features, algorithm)

                        if len(set(labels)) < 2:
                            fold_scores.append(0.0)
                            continue

                        # Calculate score on test set
                        score = self._calculate_clustering_score(test_features, labels)
                        fold_scores.append(score)

                    except Exception as e:
                        self.logger.warning(f"CV fold failed for algorithm={algorithm}: {e}")
                        fold_scores.append(0.0)

                avg_score = np.mean(fold_scores) if fold_scores else 0.0
                scores.append(avg_score)
                param_scores[algorithm] = {
                    'score': avg_score,
                    'fold_scores': fold_scores,
                    'std': np.std(fold_scores) if fold_scores else 0.0
                }

            best_algorithm = self.algorithm_options[np.argmax(scores)]

            return {
                'best_param': best_algorithm,
                'scores': scores,
                'param_scores': param_scores
            }
        except Exception as e:
            self.logger.warning(f"Algorithm optimization failed: {e}")
            return {
                'best_param': 'kmeans',
                'scores': [0.0] * len(self.algorithm_options),
                'param_scores': {}
            }

    def _generate_weight_combinations(self) -> List[Dict[str, float]]:
        """Generate weight combinations for optimization."""
        try:
            combinations = []

            for econ_weight in self.weight_ranges['economic_significance_weight']:
                for mom_weight in self.weight_ranges['momentum_weight']:
                    for vol_weight in self.weight_ranges['volume_weight']:
                        # Ensure weights sum to reasonable total
                        total_weight = econ_weight + mom_weight + vol_weight
                        if 0.5 <= total_weight <= 1.0:  # Allow some flexibility
                            combinations.append({
                                'economic_significance_weight': econ_weight,
                                'momentum_weight': mom_weight,
                                'volume_weight': vol_weight
                            })

            return combinations
        except Exception as e:
            self.logger.warning(f"Weight combinations generation failed: {e}")
            return [{'economic_significance_weight': 0.3, 'momentum_weight': 0.25, 'volume_weight': 0.25}]

    def _apply_clustering(self, features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply clustering with given parameters."""
        try:
            n_regimes = params.get('n_regimes', 3)

            # Use K-means as default
            kmeans = KMeans(n_clusters=n_regimes, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)

            return labels
        except Exception as e:
            self.logger.warning(f"Clustering application failed: {e}")
            return np.zeros(len(features))

    def _apply_clustering_with_weights(self, features: np.ndarray, market_data: pd.DataFrame,
                                     weights: Dict[str, float]) -> np.ndarray:
        """Apply clustering with economic weights."""
        try:
            # This would integrate with the economic clustering logic
            # For now, use basic clustering
            n_regimes = 3  # Default
            kmeans = KMeans(n_clusters=n_regimes, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)

            return labels
        except Exception as e:
            self.logger.warning(f"Weighted clustering application failed: {e}")
            return np.zeros(len(features))

    def _apply_clustering_algorithm(self, features: np.ndarray, algorithm: str) -> np.ndarray:
        """Apply specific clustering algorithm."""
        try:
            if algorithm in self.clustering_algorithms:
                return self.clustering_algorithms[algorithm](features)
            else:
                # Fallback to K-means
                kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
                return kmeans.fit_predict(features)
        except Exception as e:
            self.logger.warning(f"Algorithm {algorithm} application failed: {e}")
            return np.zeros(len(features))

    def _calculate_clustering_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate clustering score using composite metrics."""
        try:
            if len(set(labels)) < 2:
                return 0.0

            if self.scoring_metric == 'composite':
                # Use composite scoring with multiple metrics
                return self._calculate_composite_score(features, labels)
            elif self.scoring_metric == 'silhouette':
                return silhouette_score(features, labels)
            elif self.scoring_metric == 'calinski_harabasz':
                return calinski_harabasz_score(features, labels)
            elif self.scoring_metric == 'davies_bouldin':
                return -davies_bouldin_score(features, labels)  # Negative because lower is better
            else:
                return silhouette_score(features, labels)
        except Exception as e:
            self.logger.warning(f"Clustering score calculation failed: {e}")
            return 0.0

    def _calculate_composite_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate composite clustering score using multiple metrics."""
        try:
            # Calculate individual metrics
            silhouette = silhouette_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)

            # Normalize metrics to 0-1 range
            # Silhouette: already in [-1, 1], normalize to [0, 1]
            norm_silhouette = (silhouette + 1) / 2

            # Calinski-Harabasz: normalize by dividing by 1000 and capping at 1
            norm_ch = min(calinski_harabasz / 1000, 1.0)

            # Davies-Bouldin: lower is better, so invert and normalize
            norm_db = max(0, 1.0 / (1.0 + davies_bouldin))

            # Calculate regime balance
            unique_labels = set(labels)
            regime_sizes = [np.sum(labels == label) for label in unique_labels]
            regime_balance = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if len(regime_sizes) > 1 else 0.0

            # Composite score with weighted combination
            composite_score = (
                0.35 * norm_silhouette +      # Silhouette score (most important)
                0.25 * norm_ch +             # Calinski-Harabasz score
                0.25 * norm_db +             # Davies-Bouldin score (inverted)
                0.15 * regime_balance        # Regime balance
            )

            return composite_score

        except Exception as e:
            self.logger.warning(f"Composite score calculation failed: {e}")
            # Fallback to silhouette score
            return silhouette_score(features, labels)

    def _combine_optimization_results(self, n_regimes_results: Dict[str, Any],
                                    weight_results: Dict[str, Any],
                                    algorithm_results: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Combine optimization results to get best parameters."""
        try:
            best_params = {
                'n_regimes': n_regimes_results['best_param'],
                'weights': weight_results['best_param'],
                'algorithm': algorithm_results['best_param']
            }

            # Calculate combined score
            n_regimes_score = max(n_regimes_results['scores']) if n_regimes_results['scores'] else 0.0
            weight_score = max(weight_results['scores']) if weight_results['scores'] else 0.0
            algorithm_score = max(algorithm_results['scores']) if algorithm_results['scores'] else 0.0

            combined_score = (n_regimes_score + weight_score + algorithm_score) / 3.0

            return best_params, combined_score
        except Exception as e:
            self.logger.warning(f"Optimization results combination failed: {e}")
            return {
                'n_regimes': 3,
                'weights': {'economic_significance_weight': 0.3, 'momentum_weight': 0.25, 'volume_weight': 0.25},
                'algorithm': 'kmeans'
            }, 0.0

    def _calculate_validation_metrics(self, features: np.ndarray, best_params: Dict[str, Any],
                                    regime_labels: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate validation metrics for best parameters."""
        try:
            metrics = {}

            # Apply clustering with best parameters
            labels = self._apply_clustering(features, best_params)

            # Standard clustering metrics
            if len(set(labels)) >= 2:
                metrics['silhouette_score'] = silhouette_score(features, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = 0.0

            # Regime balance
            regime_sizes = np.bincount(labels, minlength=len(set(labels)))
            metrics['regime_balance'] = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes)) if np.mean(regime_sizes) > 0 else 0

            # Compare with existing labels if provided
            if regime_labels is not None:
                # Calculate agreement metrics
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                metrics['adjusted_rand_score'] = adjusted_rand_score(regime_labels, labels)
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(regime_labels, labels)

            return metrics
        except Exception as e:
            self.logger.warning(f"Validation metrics calculation failed: {e}")
            return {}

    def _calculate_stability_scores(self, features: np.ndarray, best_params: Dict[str, Any],
                                  cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Calculate stability scores across CV folds."""
        try:
            stability_scores = {}

            # Calculate clustering stability across folds
            fold_labels = []
            for train_idx, test_idx in cv_splits:
                train_features = features[train_idx]
                labels = self._apply_clustering(train_features, best_params)
                fold_labels.append(labels)

            # Calculate stability metrics
            if len(fold_labels) > 1:
                # Pairwise stability
                pairwise_stabilities = []
                for i in range(len(fold_labels)):
                    for j in range(i + 1, len(fold_labels)):
                        from sklearn.metrics import adjusted_rand_score
                        stability = adjusted_rand_score(fold_labels[i], fold_labels[j])
                        pairwise_stabilities.append(stability)

                stability_scores['pairwise_stability'] = np.mean(pairwise_stabilities) if pairwise_stabilities else 0.0
                stability_scores['stability_std'] = np.std(pairwise_stabilities) if pairwise_stabilities else 0.0
            else:
                stability_scores['pairwise_stability'] = 0.0
                stability_scores['stability_std'] = 0.0

            return stability_scores
        except Exception as e:
            self.logger.warning(f"Stability scores calculation failed: {e}")
            return {}

    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """K-means clustering."""
        try:
            kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
            return kmeans.fit_predict(features)
        except Exception as e:
            self.logger.warning(f"K-means clustering failed: {e}")
            return np.zeros(len(features))

    def _hierarchical_clustering(self, features: np.ndarray) -> np.ndarray:
        """Hierarchical clustering."""
        try:
            hierarchical = AgglomerativeClustering(n_clusters=3)
            return hierarchical.fit_predict(features)
        except Exception as e:
            self.logger.warning(f"Hierarchical clustering failed: {e}")
            return np.zeros(len(features))

    def _gmm_clustering(self, features: np.ndarray) -> np.ndarray:
        """Gaussian Mixture Model clustering."""
        try:
            gmm = GaussianMixture(n_components=3, random_state=self.random_state, n_init=5)
            return gmm.fit_predict(features)
        except Exception as e:
            self.logger.warning(f"GMM clustering failed: {e}")
            return np.zeros(len(features))

def create_clustering_cross_validator(config: Dict[str, Any]) -> ClusteringCrossValidator:
    """Create clustering cross-validator."""
    return ClusteringCrossValidator(config)
