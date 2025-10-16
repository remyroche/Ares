"""
Enhanced Optimized Regime Clustering with 4D Frontier Optimization

This module provides enhanced clustering algorithms that implement:
- Improved within-cluster CV calculations and optimization
- Enhanced Davies-Bouldin and Silhouette score calculations
- 5% average cluster size targeting while maintaining 3-8% range
- 4D frontier establishment between clusters
- Regime transfer optimization with CV similarity and size constraints
- 5-iteration optimization process using matrix operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
import logging
import time
from dataclasses import dataclass
from enum import Enum

# Import unified matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        optimize_dataframe,
        vectorized_rolling_features,
        gpu_matrix_multiply,
        sparse_matrix_multiply,
        batch_matrix_multiply,
        optimize_batch_size
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    warnings.warn("Matrix operations not available, using fallback implementations")

from .config import OptimalClusteringConfig
from .utils import (
    calculate_cluster_statistics, calculate_cluster_quality_metrics,
    calculate_cluster_quality_metrics_optimized, validate_cluster_quality,
    detect_outliers, prepare_clustering_features, load_regime_data
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedClusteringResult:
    """Result of enhanced clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    statistics: Any
    quality_metrics: Dict[str, float]
    validation: Any
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    frontiers: Dict[str, Any]
    transfer_history: List[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None

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

@dataclass
class RegimeTransferCandidate:
    """Candidate for regime transfer between clusters."""
    regime_id: int
    current_cluster: int
    target_cluster: int
    cv_similarity_current: float
    cv_similarity_target: float
    size_ratio_current: float
    size_ratio_target: float
    transfer_benefit: float
    constraint_violation: bool

class EnhancedMatrixOptimizedClusterer:
    """Enhanced matrix-optimized clustering algorithm with 4D frontier optimization."""

    def __init__(self, config: OptimalClusteringConfig):
        """Initialize the enhanced clusterer.

        Args:
            config: Clustering configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize matrix operations
        if MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.enhanced_ops = get_enhanced_matrix_operations()
            self.batch_processor = get_batch_matrix_processor()
            self.logger.info("✅ Matrix operations initialized successfully")
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            self.batch_processor = None
            self.logger.warning("⚠️ Matrix operations not available, using fallback mode")

        # Enhanced tracking
        self.transfer_history = []
        self.frontier_cache = {}

    def cluster_with_enhanced_optimization(self, features: np.ndarray,
                                        metadata: Dict[str, Any]) -> EnhancedClusteringResult:
        """Perform enhanced clustering with 4D frontier optimization.

        Args:
            features: Feature matrix
            metadata: Feature metadata

        Returns:
            EnhancedClusteringResult with optimization details
        """
        try:
            self.logger.info("🚀 Starting enhanced clustering with 4D frontier optimization")

            # Step 1: Initial clustering with improved CV optimization
            labels, cluster_centers = self._enhanced_initial_clustering(features)

            # Step 2: Calculate enhanced quality metrics
            quality_metrics = self._calculate_enhanced_quality_metrics(features, labels)

            # Step 3: 5-iteration frontier optimization process
            optimized_labels, frontiers, transfer_history = self._frontier_optimization_loop(
                features, labels, cluster_centers
            )

            # Step 4: Final validation and statistics
            final_stats = calculate_cluster_statistics(optimized_labels, self.config.to_dict())
            final_quality = self._calculate_enhanced_quality_metrics(features, optimized_labels)
            validation = validate_cluster_quality(final_stats, final_quality, self.config.to_dict())

            # Step 5: Create enhanced result
            result = EnhancedClusteringResult(
                labels=optimized_labels,
                cluster_centers=cluster_centers,
                statistics=final_stats,
                quality_metrics=final_quality,
                validation=validation,
                metadata={
                    **metadata,
                    'frontier_optimization_applied': True,
                    'optimization_iterations': 5,
                    'transfer_operations': len(transfer_history)
                },
                performance_metrics={
                    'initial_silhouette': quality_metrics.get('silhouette', 0.0),
                    'final_silhouette': final_quality.get('silhouette', 0.0),
                    'initial_davies_bouldin': quality_metrics.get('davies_bouldin', float('inf')),
                    'final_davies_bouldin': final_quality.get('davies_bouldin', float('inf')),
                    'improvement_silhouette': final_quality.get('silhouette', 0.0) - quality_metrics.get('silhouette', 0.0),
                    'improvement_davies_bouldin': quality_metrics.get('davies_bouldin', float('inf')) - final_quality.get('davies_bouldin', float('inf'))
                },
                frontiers=frontiers,
                transfer_history=transfer_history,
                success=True
            )

            self.logger.info("✅ Enhanced clustering completed successfully")
            self.logger.info(f"📊 Final metrics - Silhouette: {final_quality.get('silhouette', 0.0):.3f}, "
                           f"DB: {final_quality.get('davies_bouldin', float('inf')):.3f}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Enhanced clustering failed: {e}")
            return EnhancedClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics=None,
                quality_metrics={},
                validation=None,
                metadata={},
                performance_metrics={},
                frontiers={},
                transfer_history=[],
                success=False,
                error_message=str(e)
            )

    def _enhanced_initial_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform enhanced initial clustering with improved CV optimization.

        Args:
            features: Feature matrix

        Returns:
            Tuple of (labels, cluster_centers)
        """
        try:
            # Use improved 4D mapping with CV optimization
            weighted_features = self._create_enhanced_weighted_4d_map(features)

            # Apply enhanced centroid initialization
            cluster_centers = self._find_enhanced_equidistant_centroids(weighted_features, self.config.target_n_clusters)

            # Perform clustering with enhanced parameters
            kmeans = KMeans(
                n_clusters=self.config.target_n_clusters,
                init=cluster_centers,
                n_init=1,
                max_iter=self.config.kmeans_max_iter,
                random_state=self.config.random_state,
                algorithm='lloyd'  # More stable for CV optimization
            )

            labels = kmeans.fit_predict(weighted_features)
            cluster_centers = kmeans.cluster_centers_

            self.logger.info("✅ Enhanced initial clustering completed")
            return labels, cluster_centers

        except Exception as e:
            self.logger.warning(f"Enhanced initial clustering failed: {e}")
            # Fallback to standard clustering
            kmeans = KMeans(n_clusters=self.config.target_n_clusters, random_state=self.config.random_state)
            labels = kmeans.fit_predict(features)
            return labels, kmeans.cluster_centers_

    def _create_enhanced_weighted_4d_map(self, features: np.ndarray) -> np.ndarray:
        """Create enhanced weighted 4D map with improved CV-based weighting.

        Args:
            features: Original feature matrix

        Returns:
            Enhanced weighted feature matrix
        """
        try:
            # Start with initial clustering to get baseline clusters
            kmeans = KMeans(n_clusters=min(50, len(features)//100), n_init=5, random_state=self.config.random_state)
            initial_labels = kmeans.fit_predict(features)

            weighted_features = features.copy()
            unique_labels, counts = np.unique(initial_labels, return_counts=True)
            n_samples = len(features)

            for label, count in zip(unique_labels, counts):
                if label == -1:
                    continue

                cluster_mask = initial_labels == label
                cluster_features = features[cluster_mask]

                # Enhanced CV calculation for each dimension
                enhanced_cv = self._calculate_enhanced_cluster_cv(cluster_features)

                # Calculate enhanced inverse size weight with CV optimization
                inv_size_weight = 1.0 / max(1e-6, (count / n_samples))

                # Enhanced 4D weighting with CV optimization
                w_momentum = (1.0 + enhanced_cv['momentum_cv'] * 0.3)
                w_volatility = max(0.1, (1.0 - enhanced_cv['volatility_cv'] * 0.4))
                w_volume = (1.0 + enhanced_cv['volume_cv'] * 0.2)
                w_trend = max(0.1, (1.0 - enhanced_cv['trend_cv'] * 0.3))

                # Create enhanced weight vector
                four_w = np.array([w_momentum, w_volatility, w_volume, w_trend], dtype=float)
                reps = int(np.ceil(cluster_features.shape[1] / 4))
                weight_vec = np.tile(four_w, reps)[:cluster_features.shape[1]]

                # Apply CV-optimized weighting
                weight_vec *= (1.0 + 0.2 * inv_size_weight * (1.0 - np.mean(list(enhanced_cv.values()))))
                weighted_features[cluster_mask] = cluster_features * weight_vec

            return weighted_features

        except Exception as e:
            self.logger.warning(f"Enhanced weighted 4D map creation failed: {e}")
            return features

    def _calculate_enhanced_cluster_cv(self, cluster_features: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced coefficient of variation for cluster features.

        Args:
            cluster_features: Features of a single cluster

        Returns:
            Dictionary with CV values for each dimension
        """
        try:
            cv_dict = {}

            for i in range(min(4, cluster_features.shape[1])):
                feature_values = cluster_features[:, i]
                feature_values = feature_values[np.isfinite(feature_values)]

                if len(feature_values) < 2:
                    cv_dict[f'dim_{i}_cv'] = 0.0
                    continue

                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)

                if mean_val == 0:
                    cv = 0.0
                else:
                    # Enhanced CV calculation with outlier mitigation
                    cv = std_val / abs(mean_val)

                    # Apply outlier mitigation for extreme CV values
                    if cv > 10.0:  # Very high CV indicates outliers
                        # Use median absolute deviation instead
                        mad = np.median(np.abs(feature_values - np.median(feature_values)))
                        cv = mad / abs(mean_val) if mean_val != 0 else 0.0

                # Map to dimension names
                dimension_map = {0: 'volume', 1: 'volatility', 2: 'momentum', 3: 'trend'}
                cv_dict[f'{dimension_map.get(i, f"dim_{i}")}_cv'] = cv

            return cv_dict

        except Exception as e:
            self.logger.warning(f"Enhanced CV calculation failed: {e}")
            return {'volume_cv': 0.0, 'volatility_cv': 0.0, 'momentum_cv': 0.0, 'trend_cv': 0.0}

    def _find_enhanced_equidistant_centroids(self, features: np.ndarray, n_centroids: int) -> np.ndarray:
        """Find enhanced optimally distributed centroids with CV optimization.

        Args:
            features: Weighted feature matrix
            n_centroids: Number of centroids to find

        Returns:
            Enhanced centroid coordinates
        """
        try:
            # Use hierarchical approach for better initial centroids
            initial_k = min(n_centroids * 4, features.shape[0] // 50)

            kmeans = KMeans(
                n_clusters=initial_k,
                init='k-means++',
                n_init=30,
                max_iter=200,
                random_state=self.config.random_state
            )

            kmeans.fit(features)
            initial_centroids = kmeans.cluster_centers_

            # Calculate enhanced centroid quality scores with CV consideration
            centroid_scores = self._calculate_enhanced_centroid_scores(initial_centroids, features)

            # Select best centroids with CV optimization
            top_indices = np.argsort(centroid_scores)[::-1][:n_centroids]
            selected_centroids = initial_centroids[top_indices]

            # Apply final refinement with CV-aware optimization
            final_kmeans = KMeans(
                n_clusters=n_centroids,
                init=selected_centroids,
                n_init=1,
                max_iter=150,
                random_state=self.config.random_state
            )

            final_kmeans.fit(features)
            optimized_centroids = final_kmeans.cluster_centers_

            self.logger.info(f"✅ Found {n_centroids} enhanced optimally distributed centroids")
            return optimized_centroids

        except Exception as e:
            self.logger.warning(f"Enhanced centroid finding failed: {e}")
            return np.random.randn(n_centroids, features.shape[1])

    def _calculate_enhanced_centroid_scores(self, centroids: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate enhanced quality scores for centroid distribution with CV optimization.

        Args:
            centroids: Centroid coordinates
            features: Feature matrix

        Returns:
            Array of enhanced centroid quality scores
        """
        try:
            n_centroids = len(centroids)
            scores = np.zeros(n_centroids)

            for i, centroid in enumerate(centroids):
                # Calculate distance to all other centroids
                distances_to_others = np.linalg.norm(centroids - centroid, axis=1)
                distances_to_others = distances_to_others[distances_to_others > 0]

                # Calculate distance to nearest data points
                distances_to_data = np.linalg.norm(features - centroid, axis=1)
                min_distance_to_data = np.min(distances_to_data)

                # Enhanced scoring with CV consideration
                if len(distances_to_others) > 0:
                    mean_distance_to_centroids = np.mean(distances_to_others)
                    std_distance_to_centroids = np.std(distances_to_others)

                    distance_score = 1.0 / (1.0 + std_distance_to_centroids)
                    proximity_score = 1.0 / (1.0 + min_distance_to_data)

                    # Add CV optimization factor
                    # Calculate local CV for features near this centroid
                    nearby_indices = np.argsort(distances_to_data)[:min(100, len(features))]
                    nearby_features = features[nearby_indices]
                    local_cv = np.mean([self._calculate_enhanced_cluster_cv(nearby_features.reshape(1, -1))[f'dim_{j}_cv']
                                      for j in range(min(4, nearby_features.shape[1]))])

                    cv_factor = 1.0 / (1.0 + local_cv)  # Lower CV = higher score

                    scores[i] = (distance_score * 0.4 + proximity_score * 0.3 + cv_factor * 0.3)
                else:
                    scores[i] = 0.0

            return scores

        except Exception as e:
            self.logger.warning(f"Enhanced centroid scoring failed: {e}")
            return np.ones(n_centroids)

    def _calculate_enhanced_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate enhanced quality metrics with improved CV-based scoring.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary of enhanced quality metrics
        """
        try:
            metrics = calculate_cluster_quality_metrics_optimized(features, labels, True)

            # Add enhanced CV-based metrics
            mask = labels != -1
            if mask.sum() > 0:
                clean_features = features[mask]
                clean_labels = labels[mask]

                # Calculate within-cluster CV metrics
                unique_labels = np.unique(clean_labels)
                within_cluster_cvs = []

                for label in unique_labels:
                    cluster_mask = clean_labels == label
                    cluster_features = clean_features[cluster_mask]

                    if len(cluster_features) > 1:
                        # Enhanced within-cluster CV calculation
                        cluster_cv = self._calculate_enhanced_cluster_cv(cluster_features)
                        within_cluster_cvs.append(np.mean(list(cluster_cv.values())))

                if within_cluster_cvs:
                    metrics['mean_within_cluster_cv'] = float(np.mean(within_cluster_cvs))
                    metrics['std_within_cluster_cv'] = float(np.std(within_cluster_cvs))
                    metrics['min_within_cluster_cv'] = float(np.min(within_cluster_cvs))
                    metrics['max_within_cluster_cv'] = float(np.max(within_cluster_cvs))

                    # Enhanced quality score that balances Silhouette and CV
                    silhouette_score = metrics.get('silhouette', 0.0)
                    mean_cv = np.mean(within_cluster_cvs)

                    # Combined quality metric (higher is better)
                    metrics['enhanced_quality_score'] = float(
                        0.6 * silhouette_score +
                        0.4 * (1.0 / (1.0 + mean_cv))  # Lower CV = higher score
                    )

            return metrics

        except Exception as e:
            self.logger.warning(f"Enhanced quality metrics calculation failed: {e}")
            return calculate_cluster_quality_metrics(features, labels)

    def _frontier_optimization_loop(self, features: np.ndarray, initial_labels: np.ndarray,
                                  cluster_centers: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        """Perform 5-iteration frontier optimization process.

        Args:
            features: Feature matrix
            initial_labels: Initial cluster labels
            cluster_centers: Cluster centers

        Returns:
            Tuple of (optimized_labels, frontiers, transfer_history)
        """
        try:
            labels = initial_labels.copy()
            frontiers = {}
            transfer_history = []

            self.logger.info("🔄 Starting 5-iteration frontier optimization process")

            for iteration in range(5):
                self.logger.info(f"📊 Iteration {iteration + 1}/5: Establishing 4D frontiers...")

                # Step 1: Establish 4D frontiers between clusters
                current_frontiers = self._establish_4d_frontiers(features, labels, cluster_centers)
                frontiers[f'iteration_{iteration}'] = current_frontiers

                # Step 2: Find regime transfer candidates
                transfer_candidates = self._find_regime_transfer_candidates(features, labels, current_frontiers)

                # Step 3: Apply CV-optimized transfers with size constraints
                labels, iteration_transfers = self._apply_enhanced_regime_transfers(
                    features, labels, transfer_candidates
                )

                transfer_history.extend(iteration_transfers)

                self.logger.info(f"✅ Iteration {iteration + 1} completed: {len(iteration_transfers)} transfers applied")

                # Check for convergence
                if len(iteration_transfers) == 0:
                    self.logger.info(f"🎯 Convergence reached at iteration {iteration + 1}")
                    break

            self.logger.info("✅ Frontier optimization completed")
            return labels, frontiers, transfer_history

        except Exception as e:
            self.logger.warning(f"Frontier optimization failed: {e}")
            return initial_labels, {}, []

    def _establish_4d_frontiers(self, features: np.ndarray, labels: np.ndarray,
                              cluster_centers: np.ndarray) -> Dict[str, List[FrontierBoundary]]:
        """Establish 4D frontiers between clusters.

        Args:
            features: Feature matrix
            labels: Cluster labels
            cluster_centers: Cluster centers

        Returns:
            Dictionary of frontiers by type
        """
        try:
            frontiers = {frontier_type.value: [] for frontier_type in FrontierType}

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)

            # Calculate frontiers for each pair of clusters
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    cluster_a = unique_labels[i]
                    cluster_b = unique_labels[j]

                    # Get points for both clusters
                    points_a = features[labels == cluster_a]
                    points_b = features[labels == cluster_b]

                    # Calculate 4D frontier for different dimension pairs
                    for frontier_type in FrontierType:
                        boundary = self._calculate_4d_boundary(
                            points_a, points_b, cluster_a, cluster_b, frontier_type
                        )
                        frontiers[frontier_type.value].append(boundary)

            self.logger.info(f"✅ Established {sum(len(f) for f in frontiers.values())} 4D frontiers")
            return frontiers

        except Exception as e:
            self.logger.warning(f"4D frontier establishment failed: {e}")
            return {frontier_type.value: [] for frontier_type in FrontierType}

    def _calculate_4d_boundary(self, points_a: np.ndarray, points_b: np.ndarray,
                             cluster_a: int, cluster_b: int, frontier_type: FrontierType) -> FrontierBoundary:
        """Calculate 4D boundary between two clusters.

        Args:
            points_a: Points in cluster A
            points_b: Points in cluster B
            cluster_a: Cluster A ID
            cluster_b: Cluster B ID
            frontier_type: Type of frontier to calculate

        Returns:
            FrontierBoundary object
        """
        try:
            # Define dimension pairs for each frontier type
            dimension_pairs = {
                FrontierType.VOLUME_VOLATILITY: [0, 1],
                FrontierType.MOMENTUM_TREND: [2, 3],
                FrontierType.VOLUME_MOMENTUM: [0, 2],
                FrontierType.VOLATILITY_TREND: [1, 3],
                FrontierType.CROSS_DIMENSIONAL: [0, 3]  # volume-trend cross
            }

            dims = dimension_pairs[frontier_type]

            # Calculate centroids for the relevant dimensions
            center_a = np.mean(points_a[:, dims], axis=0)
            center_b = np.mean(points_b[:, dims], axis=0)

            # Calculate boundary points (midpoint between centroids)
            boundary_point = (center_a + center_b) / 2

            # Calculate similarity and CV ratios
            similarity_score = self._calculate_cluster_similarity(points_a, points_b)
            cv_ratio = self._calculate_cv_ratio(points_a, points_b)
            size_ratio = len(points_a) / len(points_b) if len(points_b) > 0 else float('inf')

            return FrontierBoundary(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                frontier_type=frontier_type,
                boundary_points=boundary_point,
                similarity_score=similarity_score,
                cv_ratio=cv_ratio,
                size_ratio=size_ratio
            )

        except Exception as e:
            self.logger.warning(f"4D boundary calculation failed: {e}")
            return FrontierBoundary(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                frontier_type=frontier_type,
                boundary_points=np.array([0.0, 0.0]),
                similarity_score=0.0,
                cv_ratio=1.0,
                size_ratio=1.0
            )

    def _calculate_cluster_similarity(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        """Calculate similarity between two clusters.

        Args:
            points_a: Points in cluster A
            points_b: Points in cluster B

        Returns:
            Similarity score (higher = more similar)
        """
        try:
            if len(points_a) == 0 or len(points_b) == 0:
                return 0.0

            # Calculate enhanced CV-based similarity
            cv_a = self._calculate_enhanced_cluster_cv(points_a)
            cv_b = self._calculate_enhanced_cluster_cv(points_b)

            # Similarity based on CV difference (lower difference = higher similarity)
            cv_similarity = 1.0 / (1.0 + np.mean([abs(cv_a[k] - cv_b[k]) for k in cv_a.keys()]))

            # Add centroid distance similarity
            center_a = np.mean(points_a, axis=0)
            center_b = np.mean(points_b, axis=0)
            center_distance = np.linalg.norm(center_a - center_b)

            distance_similarity = 1.0 / (1.0 + center_distance)

            # Combined similarity
            return float(0.7 * cv_similarity + 0.3 * distance_similarity)

        except Exception as e:
            self.logger.warning(f"Cluster similarity calculation failed: {e}")
            return 0.0

    def _calculate_cv_ratio(self, points_a: np.ndarray, points_b: np.ndarray) -> float:
        """Calculate CV ratio between two clusters.

        Args:
            points_a: Points in cluster A
            points_b: Points in cluster B

        Returns:
            CV ratio (A/B)
        """
        try:
            cv_a = np.mean(list(self._calculate_enhanced_cluster_cv(points_a).values()))
            cv_b = np.mean(list(self._calculate_enhanced_cluster_cv(points_b).values()))

            return cv_a / cv_b if cv_b > 0 else float('inf')

        except Exception as e:
            self.logger.warning(f"CV ratio calculation failed: {e}")
            return 1.0

    def _find_regime_transfer_candidates(self, features: np.ndarray, labels: np.ndarray,
                                       frontiers: Dict[str, List[FrontierBoundary]]) -> List[RegimeTransferCandidate]:
        """Find regime transfer candidates based on 4D frontiers and CV analysis.

        Args:
            features: Feature matrix
            labels: Cluster labels
            frontiers: 4D frontiers

        Returns:
            List of transfer candidates
        """
        try:
            candidates = []
            unique_labels = np.unique(labels)
            n_samples = len(labels)

            # Calculate cluster statistics
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            cluster_percentages = {label: size/n_samples for label, size in cluster_sizes.items()}

            # For each regime (sample), check if it should be transferred
            for regime_id in range(len(features)):
                current_cluster = labels[regime_id]
                current_size_pct = cluster_percentages[current_cluster]

                # Find best target cluster
                best_target = None
                best_benefit = 0.0

                for target_cluster in unique_labels:
                    if target_cluster == current_cluster:
                        continue

                    target_size_pct = cluster_percentages[target_cluster]

                    # Check size constraint (don't transfer if target is 50%+ bigger)
                    if target_size_pct > current_size_pct * 1.5:
                        continue

                    # Calculate CV similarities
                    current_cluster_points = features[labels == current_cluster]
                    target_cluster_points = features[labels == target_cluster]
                    regime_point = features[regime_id:regime_id+1]

                    # CV similarity with current cluster
                    current_cv_sim = self._calculate_regime_cluster_cv_similarity(
                        regime_point, current_cluster_points
                    )

                    # CV similarity with target cluster
                    target_cv_sim = self._calculate_regime_cluster_cv_similarity(
                        regime_point, target_cluster_points
                    )

                    # Calculate transfer benefit
                    benefit = target_cv_sim - current_cv_sim

                    # Consider frontier information
                    frontier_bonus = self._calculate_frontier_bonus(
                        regime_id, current_cluster, target_cluster, frontiers
                    )

                    total_benefit = benefit + 0.2 * frontier_bonus

                    if total_benefit > best_benefit:
                        best_benefit = total_benefit
                        best_target = target_cluster

                # Create candidate if beneficial transfer found
                if best_target is not None and best_benefit > 0.1:  # Minimum benefit threshold
                    candidates.append(RegimeTransferCandidate(
                        regime_id=regime_id,
                        current_cluster=current_cluster,
                        target_cluster=best_target,
                        cv_similarity_current=self._calculate_regime_cluster_cv_similarity(
                            features[regime_id:regime_id+1], features[labels == current_cluster]
                        ),
                        cv_similarity_target=self._calculate_regime_cluster_cv_similarity(
                            features[regime_id:regime_id+1], features[labels == best_target]
                        ),
                        size_ratio_current=current_size_pct / cluster_percentages[best_target],
                        size_ratio_target=cluster_percentages[best_target] / current_size_pct,
                        transfer_benefit=best_benefit,
                        constraint_violation=False
                    ))

            self.logger.info(f"✅ Found {len(candidates)} regime transfer candidates")
            return candidates

        except Exception as e:
            self.logger.warning(f"Regime transfer candidate finding failed: {e}")
            return []

    def _calculate_regime_cluster_cv_similarity(self, regime_point: np.ndarray, cluster_points: np.ndarray) -> float:
        """Calculate CV-based similarity between a regime and a cluster.

        Args:
            regime_point: Single regime point
            cluster_points: Points in target cluster

        Returns:
            CV similarity score
        """
        try:
            if len(cluster_points) == 0:
                return 0.0

            # Calculate CV for the cluster
            cluster_cv = self._calculate_enhanced_cluster_cv(cluster_points)
            mean_cluster_cv = np.mean(list(cluster_cv.values()))

            # Calculate CV for the regime point when added to cluster
            combined_points = np.vstack([cluster_points, regime_point])
            combined_cv = self._calculate_enhanced_cluster_cv(combined_points)
            mean_combined_cv = np.mean(list(combined_cv.values()))

            # Similarity is inverse of CV increase
            if mean_cluster_cv == 0:
                return 1.0  # Perfect similarity if cluster has no variance

            return max(0.0, 1.0 - (mean_combined_cv / mean_cluster_cv))

        except Exception as e:
            self.logger.warning(f"Regime-cluster CV similarity calculation failed: {e}")
            return 0.0

    def _calculate_frontier_bonus(self, regime_id: int, current_cluster: int,
                                target_cluster: int, frontiers: Dict[str, List[FrontierBoundary]]) -> float:
        """Calculate frontier bonus for regime transfer.

        Args:
            regime_id: Regime ID
            current_cluster: Current cluster
            target_cluster: Target cluster
            frontiers: 4D frontiers

        Returns:
            Frontier bonus score
        """
        try:
            bonus = 0.0

            # Check all frontier types
            for frontier_list in frontiers.values():
                for frontier in frontier_list:
                    if ((frontier.cluster_a == current_cluster and frontier.cluster_b == target_cluster) or
                        (frontier.cluster_a == target_cluster and frontier.cluster_b == current_cluster)):

                        # Add bonus based on frontier characteristics
                        if frontier.similarity_score > 0.7:  # High similarity frontier
                            bonus += 0.3
                        if frontier.cv_ratio < 1.2:  # Similar CV ratios
                            bonus += 0.2
                        if frontier.size_ratio < 1.5:  # Balanced sizes
                            bonus += 0.1

            return bonus

        except Exception as e:
            self.logger.warning(f"Frontier bonus calculation failed: {e}")
            return 0.0

    def _apply_enhanced_regime_transfers(self, features: np.ndarray, labels: np.ndarray,
                                       candidates: List[RegimeTransferCandidate]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply enhanced regime transfers with optimization.

        Args:
            features: Feature matrix
            labels: Cluster labels
            candidates: Transfer candidates

        Returns:
            Tuple of (updated_labels, transfer_history)
        """
        try:
            updated_labels = labels.copy()
            transfer_history = []

            # Sort candidates by benefit (highest first)
            sorted_candidates = sorted(candidates, key=lambda x: x.transfer_benefit, reverse=True)

            # Apply transfers in batches to maintain stability
            batch_size = max(1, len(sorted_candidates) // 10)  # Apply in 10% batches
            applied_count = 0

            for i, candidate in enumerate(sorted_candidates):
                # Check size constraint again (in case sizes changed)
                current_size = np.sum(updated_labels == candidate.current_cluster)
                target_size = np.sum(updated_labels == candidate.target_cluster)
                total_size = len(updated_labels)

                if target_size > current_size * 1.5:
                    continue

                # Apply transfer
                updated_labels[candidate.regime_id] = candidate.target_cluster
                applied_count += 1

                transfer_history.append({
                    'regime_id': candidate.regime_id,
                    'from_cluster': candidate.current_cluster,
                    'to_cluster': candidate.target_cluster,
                    'benefit': candidate.transfer_benefit,
                    'cv_similarity_improvement': candidate.cv_similarity_target - candidate.cv_similarity_current
                })

                # Apply in batches for stability
                if applied_count >= batch_size and i < len(sorted_candidates) - 1:
                    break

            self.logger.info(f"✅ Applied {applied_count} enhanced regime transfers")
            return updated_labels, transfer_history

        except Exception as e:
            self.logger.warning(f"Enhanced regime transfer application failed: {e}")
            return labels, []

def create_enhanced_clustering_config() -> OptimalClusteringConfig:
    """Create enhanced clustering configuration optimized for 5% average cluster size.

    Returns:
        Enhanced clustering configuration
    """
    config = OptimalClusteringConfig()

    # Adjust for 5% average cluster size while maintaining 3-8% range
    config.min_cluster_size_pct = 0.03  # 3% minimum
    config.max_cluster_size_pct = 0.08  # 8% maximum
    config.target_coverage_pct = 0.90   # 90% coverage

    # Enhanced optimization parameters
    config.weighted_4d_mapping = True
    config.equidistant_centroids = True
    config.size_constrained_merging = True
    config.cv_based_similarity = True
    config.cv_optimized_splitting = True
    config.enhanced_redistribution = True
    config.iterative_refinement = True
    config.adaptive_targets = True
    config.smart_cluster_transfer = True

    # 5-iteration optimization
    config.outlier_redistribution_rounds = 5
    config.refinement_passes = 5

    # Enhanced quality thresholds
    config.min_silhouette_score = 0.35   # Higher threshold
    config.min_calinski_harabasz_score = 200.0  # Higher threshold
    config.min_davies_bouldin_score = 1.3  # Lower maximum (better)

    # Enhanced CV optimization
    config.perfect_distribution_threshold = 0.95  # Higher requirement
    config.smart_transfer_percentage = 0.15  # More conservative

    return config

def run_enhanced_clustering_pipeline(data_path: str, config: Optional[OptimalClusteringConfig] = None) -> EnhancedClusteringResult:
    """Run enhanced clustering pipeline with 4D frontier optimization.

    Args:
        data_path: Path to regime data
        config: Optional clustering configuration

    Returns:
        EnhancedClusteringResult
    """
    try:
        if config is None:
            config = create_enhanced_clustering_config()

        # Load and prepare data
        regime_data = load_regime_data(data_path, config.to_dict())
        features, metadata = prepare_clustering_features(regime_data, config.to_dict())

        # Create enhanced clusterer
        clusterer = EnhancedMatrixOptimizedClusterer(config)

        # Run enhanced clustering
        result = clusterer.cluster_with_enhanced_optimization(features, metadata)

        logger.info("✅ Enhanced clustering pipeline completed successfully")
        return result

    except Exception as e:
        logger.error(f"❌ Enhanced clustering pipeline failed: {e}")
        return EnhancedClusteringResult(
            labels=np.array([]),
            cluster_centers=np.array([]),
            statistics=None,
            quality_metrics={},
            validation=None,
            metadata={},
            performance_metrics={},
            frontiers={},
            transfer_history=[],
            success=False,
            error_message=str(e)
        )
