"""
Advanced Clustering for Hybrid NAS-TAS Regime System

This module provides advanced clustering algorithms inspired by hmm_clustering.py
including multiple clustering algorithms, ensemble methods, and regime-specific clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

@dataclass
class AdvancedClusteringResult:
    """Result from advanced clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    probabilities: np.ndarray
    quality_metrics: Dict[str, float]
    frontier_metrics: Dict[str, Any]
    regime_transfers: List[Dict[str, Any]]
    optimization_iterations: int
    algorithm_used: str
    execution_time: float
    metadata: Dict[str, Any]

class FrontierType(Enum):
    """Types of frontiers between clusters."""
    VOLUME_VOLATILITY = "volume_volatility"
    MOMENTUM_TREND = "momentum_trend"
    VOLUME_MOMENTUM = "volume_momentum"
    VOLATILITY_TREND = "volatility_trend"
    CROSS_DIMENSIONAL = "cross_dimensional"

@dataclass
class FrontierBoundary:
    """Boundary information for 4D frontiers."""
    cluster_a: int
    cluster_b: int
    frontier_type: FrontierType
    boundary_points: np.ndarray
    similarity_score: float
    cv_ratio: float
    size_ratio: float

class AdvancedHybridClusterer:
    """
    Advanced clustering with multiple algorithms and ensemble methods.
    Inspired by hmm_clustering.py matrix optimization and frontier analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize advanced clusterer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Available clustering algorithms
        self.clustering_algorithms = {
            'kmeans': KMeans(n_clusters=config.get('n_regimes', 8), random_state=42),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gmm': GaussianMixture(n_components=config.get('n_regimes', 8), random_state=42),
            'agglomerative': AgglomerativeClustering(n_clusters=config.get('n_regimes', 8)),
            'hierarchical': AgglomerativeClustering(n_clusters=config.get('n_regimes', 8), linkage='ward')
        }

        # Performance tracking
        self.performance_history = []

        self.logger.info("✅ Advanced Hybrid Clusterer initialized")

    def cluster_features(self,
                        features: np.ndarray,
                        economic_weights: Optional[np.ndarray] = None,
                        financial_weights: Optional[np.ndarray] = None) -> AdvancedClusteringResult:
        """
        Perform advanced clustering with multiple algorithms and ensemble methods.

        Args:
            features: Feature matrix
            economic_weights: Economic significance weights for each feature
            financial_weights: Financial relevance weights for each feature

        Returns:
            AdvancedClusteringResult with comprehensive clustering results
        """
        try:
            self.logger.info("🔍 Starting advanced clustering...")

            # Apply weights if provided
            weighted_features = self._apply_weights(features, economic_weights, financial_weights)

            # Choose clustering strategy
            primary_algorithm = self.config.get('primary_algorithm', 'adaptive')

            if primary_algorithm == 'adaptive':
                result = self._adaptive_clustering(weighted_features)
            elif primary_algorithm == 'ensemble':
                result = self._ensemble_clustering(weighted_features)
            else:
                result = self._single_algorithm_clustering(weighted_features, primary_algorithm)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(weighted_features, result.labels)

            # Perform frontier analysis
            frontier_metrics = self._frontier_analysis(weighted_features, result.labels)

            # Optimize regime transfers
            regime_transfers = self._regime_transfer_optimization(
                weighted_features, result.labels, frontier_metrics
            )

            return AdvancedClusteringResult(
                labels=result.labels,
                cluster_centers=result.cluster_centers,
                probabilities=result.probabilities,
                quality_metrics=quality_metrics,
                frontier_metrics=frontier_metrics,
                regime_transfers=regime_transfers,
                optimization_iterations=self.config.get('optimization_iterations', 5),
                algorithm_used=primary_algorithm,
                execution_time=result.execution_time,
                metadata={
                    'n_features': features.shape[1],
                    'n_samples': features.shape[0],
                    'n_regimes': len(set(result.labels)),
                    'timestamp': datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Advanced clustering failed: {e}")
            raise

    def _apply_weights(self,
                      features: np.ndarray,
                      economic_weights: Optional[np.ndarray] = None,
                      financial_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply economic and financial weights to features."""
        try:
            weighted_features = features.copy()

            # Apply economic weights
            if economic_weights is not None and len(economic_weights) == features.shape[1]:
                weighted_features *= economic_weights.reshape(1, -1)

            # Apply financial weights
            if financial_weights is not None and len(financial_weights) == features.shape[1]:
                weighted_features *= financial_weights.reshape(1, -1)

            return weighted_features

        except Exception as e:
            self.logger.warning(f"Weight application failed: {e}, using unweighted features")
            return features

    def _adaptive_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Adaptive clustering that selects best algorithm."""
        try:
            self.logger.info("🔍 Performing adaptive clustering...")

            best_score = -1
            best_result = None
            best_algorithm = None

            # Try different algorithms
            algorithms_to_try = ['kmeans', 'gmm', 'agglomerative', 'hierarchical']

            for algorithm in algorithms_to_try:
                try:
                    result = self._single_algorithm_clustering(features, algorithm)

                    # Calculate quality score
                    score = self._calculate_algorithm_score(features, result)

                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_algorithm = algorithm

                except Exception as e:
                    self.logger.warning(f"Algorithm {algorithm} failed: {e}")
                    continue

            if best_result is None:
                raise ValueError("All clustering algorithms failed")

            self.logger.info(f"   Selected algorithm: {best_algorithm} (score: {best_score:.3f})")
            return best_result

        except Exception as e:
            self.logger.error(f"Adaptive clustering failed: {e}")
            # Fallback to kmeans
            return self._single_algorithm_clustering(features, 'kmeans')

    def _ensemble_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Ensemble clustering combining multiple algorithms."""
        try:
            self.logger.info("🔍 Performing ensemble clustering...")

            ensemble_method = self.config.get('ensemble_method', 'voting')

            if ensemble_method == 'voting':
                return self._voting_ensemble_clustering(features)
            elif ensemble_method == 'stacking':
                return self._stacking_ensemble_clustering(features)
            elif ensemble_method == 'bagging':
                return self._bagging_ensemble_clustering(features)
            else:
                return self._voting_ensemble_clustering(features)  # Default

        except Exception as e:
            self.logger.error(f"Ensemble clustering failed: {e}")
            return self._single_algorithm_clustering(features, 'kmeans')

    def _voting_ensemble_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Voting-based ensemble clustering."""
        try:
            # Get predictions from multiple algorithms
            predictions = []

            for algorithm in ['kmeans', 'gmm', 'agglomerative']:
                try:
                    result = self._single_algorithm_clustering(features, algorithm)
                    predictions.append(result.labels)
                except:
                    continue

            if not predictions:
                raise ValueError("No algorithms succeeded")

            # Combine predictions using voting
            n_samples = len(features)
            n_regimes = self.config.get('n_regimes', 8)
            votes = np.zeros((n_samples, n_regimes))

            for pred in predictions:
                for i, label in enumerate(pred):
                    if 0 <= label < n_regimes:
                        votes[i, label] += 1

            # Final labels based on majority vote
            final_labels = np.argmax(votes, axis=1)

            # Use KMeans on final predictions for refinement
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            refined_labels = kmeans.fit_predict(votes)

            return {
                'labels': refined_labels,
                'cluster_centers': kmeans.cluster_centers_,
                'probabilities': votes / len(predictions),
                'execution_time': 0.0
            }

        except Exception as e:
            self.logger.error(f"Voting ensemble failed: {e}")
            raise

    def _single_algorithm_clustering(self, features: np.ndarray, algorithm: str) -> Dict[str, Any]:
        """Single algorithm clustering."""
        try:
            if algorithm not in self.clustering_algorithms:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            clusterer = self.clustering_algorithms[algorithm]

            # Fit and predict
            if algorithm == 'gmm':
                labels = clusterer.fit_predict(features)
                probabilities = clusterer.predict_proba(features)
                centers = clusterer.means_
            else:
                labels = clusterer.fit_predict(features)
                probabilities = np.zeros((len(features), len(set(labels))))
                centers = clusterer.cluster_centers_ if hasattr(clusterer, 'cluster_centers_') else np.array([])

            return {
                'labels': labels,
                'cluster_centers': centers,
                'probabilities': probabilities,
                'execution_time': 0.0
            }

        except Exception as e:
            self.logger.error(f"Single algorithm clustering failed for {algorithm}: {e}")
            raise

    def _calculate_algorithm_score(self, features: np.ndarray, result: Dict[str, Any]) -> float:
        """Calculate quality score for algorithm selection."""
        try:
            labels = result['labels']

            if len(set(labels)) < 2:
                return 0.0

            # Silhouette score (higher is better)
            try:
                silhouette = silhouette_score(features, labels)
            except:
                silhouette = 0.0

            # Calinski-Harabasz score (higher is better)
            try:
                ch_score = calinski_harabasz_score(features, labels)
            except:
                ch_score = 0.0

            # Combine scores (normalize to 0-1 range)
            max_silhouette = 1.0
            max_ch = 1000.0  # Reasonable upper bound

            normalized_silhouette = min(silhouette / max_silhouette, 1.0)
            normalized_ch = min(ch_score / max_ch, 1.0)

            score = 0.6 * normalized_silhouette + 0.4 * normalized_ch
            return score

        except Exception as e:
            self.logger.warning(f"Algorithm score calculation failed: {e}")
            return 0.0

    def _calculate_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        try:
            metrics = {}

            unique_labels = set(labels)
            n_clusters = len(unique_labels)

            if n_clusters < 2:
                return {'error': 'Insufficient clusters'}

            # Standard clustering metrics
            try:
                metrics['silhouette_score'] = silhouette_score(features, labels)
            except:
                metrics['silhouette_score'] = 0.0

            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)
            except:
                metrics['calinski_harabasz_score'] = 0.0

            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
            except:
                metrics['davies_bouldin_score'] = 0.0

            # Regime-specific metrics
            regime_sizes = np.bincount(labels, minlength=n_clusters)
            metrics['regime_balance'] = 1.0 - (np.std(regime_sizes) / np.mean(regime_sizes))
            metrics['min_regime_size'] = np.min(regime_sizes)
            metrics['max_regime_size'] = np.max(regime_sizes)

            return metrics

        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {'error': str(e)}

    def _frontier_analysis(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Perform frontier analysis between clusters."""
        try:
            self.logger.info("🔍 Performing frontier analysis...")

            frontiers = {}
            unique_labels = sorted(set(labels))

            # Calculate frontiers between cluster pairs
            for i, label_a in enumerate(unique_labels):
                for label_b in unique_labels[i+1:]:
                    frontier = self._calculate_frontier(features, labels, label_a, label_b)
                    if frontier:
                        frontier_key = f"{label_a}_{label_b}"
                        frontiers[frontier_key] = frontier

            # Calculate overall frontier metrics
            frontier_metrics = {
                'n_frontiers': len(frontiers),
                'avg_similarity': np.mean([f.similarity_score for f in frontiers.values()]),
                'avg_cv_ratio': np.mean([f.cv_ratio for f in frontiers.values()]),
                'frontier_boundaries': frontiers
            }

            self.logger.info(f"   Found {len(frontiers)} frontiers")
            return frontier_metrics

        except Exception as e:
            self.logger.warning(f"Frontier analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_frontier(self, features: np.ndarray, labels: np.ndarray, label_a: int, label_b: int) -> Optional[FrontierBoundary]:
        """Calculate frontier between two clusters."""
        try:
            # Get points for each cluster
            points_a = features[labels == label_a]
            points_b = features[labels == label_b]

            if len(points_a) == 0 or len(points_b) == 0:
                return None

            # Calculate centroids
            centroid_a = np.mean(points_a, axis=0)
            centroid_b = np.mean(points_b, axis=0)

            # Find boundary points (points closest to the other cluster)
            distances_to_b = np.min(np.linalg.norm(points_a[:, np.newaxis] - points_b, axis=2), axis=1)
            distances_to_a = np.min(np.linalg.norm(points_b[:, np.newaxis] - points_a, axis=2), axis=1)

            # Boundary points are those with minimum distance to the other cluster
            boundary_threshold = np.percentile(distances_to_b, 10)  # Top 10% closest
            boundary_a = points_a[distances_to_b <= boundary_threshold]

            # Calculate similarity and ratios
            similarity_score = 1.0 / (1.0 + np.linalg.norm(centroid_a - centroid_b))
            cv_ratio = np.std(points_a) / np.std(points_b) if np.std(points_b) > 0 else 1.0
            size_ratio = len(points_a) / len(points_b)

            return FrontierBoundary(
                cluster_a=label_a,
                cluster_b=label_b,
                frontier_type=FrontierType.CROSS_DIMENSIONAL,
                boundary_points=boundary_a,
                similarity_score=similarity_score,
                cv_ratio=cv_ratio,
                size_ratio=size_ratio
            )

        except Exception as e:
            self.logger.warning(f"Frontier calculation failed for {label_a}-{label_b}: {e}")
            return None

    def _regime_transfer_optimization(self,
                                    features: np.ndarray,
                                    labels: np.ndarray,
                                    frontier_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize regime transfers between clusters."""
        try:
            self.logger.info("🔍 Performing regime transfer optimization...")

            transfers = []
            unique_labels = sorted(set(labels))

            # Identify potential transfers
            for i, label_a in enumerate(unique_labels):
                for label_b in unique_labels[i+1:]:
                    transfer = self._evaluate_regime_transfer(
                        features, labels, label_a, label_b, frontier_metrics
                    )
                    if transfer:
                        transfers.append(transfer)

            # Sort by benefit
            transfers.sort(key=lambda x: x.get('benefit', 0), reverse=True)

            self.logger.info(f"   Identified {len(transfers)} potential transfers")
            return transfers

        except Exception as e:
            self.logger.warning(f"Regime transfer optimization failed: {e}")
            return []

    def _evaluate_regime_transfer(self,
                                features: np.ndarray,
                                labels: np.ndarray,
                                label_a: int,
                                label_b: int,
                                frontier_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate potential transfer between regimes."""
        try:
            points_a = features[labels == label_a]
            points_b = features[labels == label_b]

            if len(points_a) == 0 or len(points_b) == 0:
                return None

            # Calculate transfer metrics
            centroid_a = np.mean(points_a, axis=0)
            centroid_b = np.mean(points_b, axis=0)

            distance = np.linalg.norm(centroid_a - centroid_b)
            similarity = 1.0 / (1.0 + distance)

            # Calculate potential benefit
            benefit = similarity * min(len(points_a), len(points_b)) / max(len(points_a), len(points_b))

            return {
                'regime_a': label_a,
                'regime_b': label_b,
                'distance': distance,
                'similarity': similarity,
                'benefit': benefit,
                'size_a': len(points_a),
                'size_b': len(points_b)
            }

        except Exception as e:
            self.logger.warning(f"Transfer evaluation failed for {label_a}-{label_b}: {e}")
            return None

def create_advanced_clusterer(config: Dict[str, Any]) -> AdvancedHybridClusterer:
    """Create advanced hybrid clusterer."""
    return AdvancedHybridClusterer(config)
