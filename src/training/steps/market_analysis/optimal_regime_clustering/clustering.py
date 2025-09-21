"""
Optimal Regime Clustering Algorithm

This module implements the hybrid clustering algorithm for creating 20 optimal clusters
from HMM regime discovery output with noise reduction and quality validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
import logging
from dataclasses import dataclass
from .config import OptimalClusteringConfig
from .utils import (
    calculate_cluster_statistics, calculate_cluster_quality_metrics,
    validate_cluster_quality, bootstrap_cluster_stability, detect_outliers,
    prepare_clustering_features, load_hmm_regime_data
)
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor,
        gpu_matrix_multiply,
        batch_matrix_multiply,
    )
    MATRIX_OPS = True
except Exception:
    MATRIX_OPS = False

logger = logging.getLogger(__name__)

@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    statistics: Any
    quality_metrics: Dict[str, float]
    validation: Any
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class OptimalRegimeClusterer:
    """Optimal clustering algorithm for HMM regime data."""

    def __init__(self, config: OptimalClusteringConfig):
        """Initialize the clusterer.

        Args:
            config: Clustering configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        # optional matrix ops for performance
        if MATRIX_OPS:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
            except Exception:
                self.matrix_ops = None
                self.vectorized_core = None
                self.enhanced_ops = None
                self.batch_processor = None
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            self.batch_processor = None

    def cluster(self, data: Union[str, pd.DataFrame], **kwargs) -> ClusteringResult:
        """Perform optimal clustering on HMM regime data.

        Args:
            data: Path to data file or DataFrame containing regime data
            **kwargs: Additional parameters

        Returns:
            ClusteringResult object
        """
        try:
            self.logger.info("Starting optimal regime clustering...")

            # Load and prepare data
            if isinstance(data, str):
                regime_data = load_hmm_regime_data(data, self.config.to_dict())
            else:
                regime_data = data

            features, feature_metadata = prepare_clustering_features(regime_data, self.config.to_dict())

            # Detect and remove outliers
            outlier_mask = detect_outliers(
                features,
                method=self.config.outlier_detection_method,
                contamination=0.05
            )

            if outlier_mask.sum() > 0:
                self.logger.info(f"Removing {outlier_mask.sum()} outliers")
                features = features[~outlier_mask]

            # Multi-stage clustering approach
            if self.config.multi_stage_clustering:
                result = self._multi_stage_clustering(features, feature_metadata)
            else:
                result = self._single_stage_clustering(features, feature_metadata)

            self.logger.info("Optimal regime clustering completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in optimal regime clustering: {e}")
            return ClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics=None,
                quality_metrics={},
                validation=None,
                metadata={},
                success=False,
                error_message=str(e)
            )

    def _multi_stage_clustering(self, features: np.ndarray, feature_metadata: Dict[str, Any]) -> ClusteringResult:
        """Perform multi-stage clustering for optimal results.

        Args:
            features: Feature matrix
            feature_metadata: Feature metadata

        Returns:
            ClusteringResult
        """
        try:
            self.logger.info("Starting multi-stage clustering...")

            # Stage 1: Noise reduction using HDBSCAN/DBSCAN
            noise_labels = self._noise_reduction_clustering(features)

            # Stage 2: Main clustering using K-means
            main_labels = self._main_clustering(features)

            # Stage 3: Combine results and optimize
            final_labels = self._combine_and_optimize_clusters(
                features, noise_labels, main_labels, feature_metadata
            )

            # Calculate statistics and metrics
            statistics = calculate_cluster_statistics(final_labels, self.config.to_dict())
            quality_metrics = calculate_cluster_quality_metrics(features, final_labels)

            # Validate results
            validation = validate_cluster_quality(statistics, quality_metrics, self.config.to_dict())

            # Create cluster centers
            cluster_centers = self._calculate_cluster_centers(features, final_labels)

            # Create metadata
            metadata = {
                'feature_metadata': feature_metadata,
                'n_iterations': getattr(self, '_iteration_count', 1),
                'clustering_method': 'multi_stage',
                'noise_reduction_applied': True
            }

            result = ClusteringResult(
                labels=final_labels,
                cluster_centers=cluster_centers,
                statistics=statistics,
                quality_metrics=quality_metrics,
                validation=validation,
                metadata=metadata,
                success=True
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in multi-stage clustering: {e}")
            raise

    def _single_stage_clustering(self, features: np.ndarray, feature_metadata: Dict[str, Any]) -> ClusteringResult:
        """Perform single-stage clustering.

        Args:
            features: Feature matrix
            feature_metadata: Feature metadata

        Returns:
            ClusteringResult
        """
        try:
            self.logger.info("Starting single-stage clustering...")

            # Choose clustering method based on configuration
            if self.config.clustering_method == "hdbscan":
                labels = self._hdbscan_clustering(features)
            elif self.config.clustering_method == "dbscan":
                labels = self._dbscan_clustering(features)
            elif self.config.clustering_method == "kmeans":
                labels = self._kmeans_clustering(features)
            else:  # hybrid
                labels = self._hybrid_clustering(features)

            # Calculate statistics and metrics
            statistics = calculate_cluster_statistics(labels, self.config.to_dict())
            quality_metrics = calculate_cluster_quality_metrics(features, labels)

            # Validate results
            validation = validate_cluster_quality(statistics, quality_metrics, self.config.to_dict())

            # Create cluster centers
            cluster_centers = self._calculate_cluster_centers(features, labels)

            # Create metadata
            metadata = {
                'feature_metadata': feature_metadata,
                'n_iterations': getattr(self, '_iteration_count', 1),
                'clustering_method': self.config.clustering_method,
                'noise_reduction_applied': False
            }

            result = ClusteringResult(
                labels=labels,
                cluster_centers=cluster_centers,
                statistics=statistics,
                quality_metrics=quality_metrics,
                validation=validation,
                metadata=metadata,
                success=True
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in single-stage clustering: {e}")
            raise

    def _noise_reduction_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform noise reduction clustering using HDBSCAN.

        Args:
            features: Feature matrix

        Returns:
            Noise-reduced labels
        """
        try:
            self.logger.info("Performing noise reduction clustering...")

            # Use HDBSCAN for noise reduction
            try:
                from hdbscan import HDBSCAN
                clusterer = HDBSCAN(
                    min_cluster_size=self.config.min_cluster_size,
                    min_samples=self.config.min_samples,
                    cluster_selection_epsilon=self.config.cluster_selection_epsilon
                )
                labels = clusterer.fit_predict(features)
                self.logger.info(f"HDBSCAN found {len(np.unique(labels[labels != -1]))} clusters")
                return labels
            except ImportError:
                self.logger.warning("HDBSCAN not available, using DBSCAN for noise reduction")
                return self._dbscan_clustering(features)

        except Exception as e:
            self.logger.warning(f"Error in noise reduction clustering: {e}")
            return np.full(len(features), -1)  # All noise

    def _main_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform main clustering using K-means.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            self.logger.info("Performing main clustering...")

            # Use K-means with target number of clusters
            kmeans = KMeans(
                n_clusters=self.config.target_n_clusters,
                n_init=10,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                init='k-means++'
            )

            labels = kmeans.fit_predict(features)
            self.logger.info(f"K-means created {len(np.unique(labels))} clusters")
            return labels

        except Exception as e:
            self.logger.error(f"Error in main clustering: {e}")
            raise

    def _hdbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN clustering.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            from hdbscan import HDBSCAN

            clusterer = HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_epsilon=self.config.cluster_selection_epsilon
            )

            labels = clusterer.fit_predict(features)
            self.logger.info(f"HDBSCAN found {len(np.unique(labels[labels != -1]))} clusters")
            return labels

        except ImportError:
            self.logger.error("HDBSCAN not available")
            raise

    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            clusterer = DBSCAN(
                eps=self.config.cluster_selection_epsilon,
                min_samples=self.config.min_samples
            )

            labels = clusterer.fit_predict(features)
            self.logger.info(f"DBSCAN found {len(np.unique(labels[labels != -1]))} clusters")
            return labels

        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            raise

    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform K-means clustering.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            kmeans = KMeans(
                n_clusters=self.config.target_n_clusters,
                n_init=10,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                init='k-means++'
            )

            labels = kmeans.fit_predict(features)
            self.logger.info(f"K-means created {len(np.unique(labels))} clusters")
            return labels

        except Exception as e:
            self.logger.error(f"Error in K-means clustering: {e}")
            raise

    def _hybrid_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform hybrid clustering (DBSCAN + K-means).

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            # First use DBSCAN for noise reduction
            dbscan_labels = self._dbscan_clustering(features)

            # Identify core points (non-noise)
            core_mask = dbscan_labels != -1
            noise_mask = ~core_mask

            if core_mask.sum() == 0:
                self.logger.warning("No core points found, using K-means on all data")
                return self._kmeans_clustering(features)

            # Use K-means on core points
            core_features = features[core_mask]
            core_labels = self._kmeans_clustering(core_features)

            # Combine results
            final_labels = np.full(len(features), -1)
            final_labels[core_mask] = core_labels

            self.logger.info(f"Hybrid clustering: {len(np.unique(core_labels))} clusters on {core_mask.sum()} core points")
            return final_labels

        except Exception as e:
            self.logger.error(f"Error in hybrid clustering: {e}")
            raise

    def _combine_and_optimize_clusters(self, features: np.ndarray, noise_labels: np.ndarray,
                                     main_labels: np.ndarray, feature_metadata: Dict[str, Any]) -> np.ndarray:
        """Combine and optimize clustering results.

        Args:
            features: Feature matrix
            noise_labels: Labels from noise reduction
            main_labels: Labels from main clustering
            feature_metadata: Feature metadata

        Returns:
            Optimized cluster labels
        """
        try:
            self.logger.info("Combining and optimizing cluster results...")

            # Start with main clustering results
            final_labels = main_labels.copy()

            # Remove noise points identified by noise reduction
            if len(np.unique(noise_labels)) > 1:
                noise_mask = noise_labels == -1
                if noise_mask.any():
                    final_labels[noise_mask] = -1

            # Optimize cluster sizes if needed
            if self.config.adaptive_clustering:
                final_labels = self._optimize_cluster_sizes(features, final_labels)

            self.logger.info(f"Final clustering: {len(np.unique(final_labels[final_labels != -1]))} clusters")
            return final_labels

        except Exception as e:
            self.logger.error(f"Error combining and optimizing clusters: {e}")
            return main_labels

    def _optimize_cluster_sizes(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Optimize cluster sizes to meet target distribution.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Optimized cluster labels
        """
        try:
            self.logger.info("Optimizing cluster sizes...")

            # Calculate current cluster statistics
            stats = calculate_cluster_statistics(labels, self.config.to_dict())

            target_clusters = self.config.target_n_clusters
            min_pct = float(getattr(self.config, 'min_cluster_size_pct', 0.03))
            max_pct = float(getattr(self.config, 'max_cluster_size_pct', 0.08))

            # Adjust k if needed
            if len(np.unique(labels)) != target_clusters:
                # quick GMM to reach k, then bounded assignment
                gmm = GaussianMixture(
                    n_components=target_clusters,
                    random_state=self.config.random_state,
                    max_iter=self.config.max_iter
                )
                gmm_labels = gmm.fit_predict(features)
                return self._capacity_constrained_assignment(features, gmm_labels, min_pct, max_pct)

            # Already at k; balance sizes via assignment
            return self._capacity_constrained_assignment(features, labels, min_pct, max_pct)

        except Exception as e:
            self.logger.warning(f"Error optimizing cluster sizes: {e}")
            return labels

    def _capacity_constrained_assignment(self, features: np.ndarray, labels: np.ndarray,
                                         min_pct: float, max_pct: float,
                                         *, distance_metric: str = "euclidean",
                                         whiten: bool = False) -> np.ndarray:
        """Greedy bounded assignment to enforce cluster size bounds and full coverage."""
        n = labels.shape[0]
        lower = max(1, int(np.ceil(min_pct * n)))
        upper = max(1, int(np.floor(max_pct * n)))
        current = labels.copy()

        # optional whitening
        X = features
        if whiten:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-12
            X = (X - mean) / std

        def compute_centroids(lbls: np.ndarray) -> np.ndarray:
            uniq = np.unique(lbls)
            k = len(uniq)
            idx_map = {lab: i for i, lab in enumerate(uniq)}
            lid = np.vectorize(idx_map.get)(lbls)
            onehot = np.zeros((n, k), dtype=np.float64)
            onehot[np.arange(n), lid] = 1.0
            try:
                if MATRIX_OPS and gpu_matrix_multiply is not None:
                    sums = gpu_matrix_multiply(onehot.T, X)
                elif MATRIX_OPS and batch_matrix_multiply is not None:
                    sums = batch_matrix_multiply(onehot.T, X)
                else:
                    sums = onehot.T @ X
            except Exception:
                sums = onehot.T @ X
            cnts_local = onehot.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                centers = sums / np.maximum(cnts_local[:, None], 1.0)
            return centers

        def compute_dists(lbls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            ctrs = compute_centroids(lbls)
            ctrs_w = ctrs
            if whiten:
                ctrs_w = (ctrs - mean) / std
            if distance_metric == "cosine":
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                Cn = ctrs_w / (np.linalg.norm(ctrs_w, axis=1, keepdims=True) + 1e-12)
                try:
                    if MATRIX_OPS and gpu_matrix_multiply is not None:
                        sims = gpu_matrix_multiply(Xn, Cn.T)
                    elif MATRIX_OPS and batch_matrix_multiply is not None:
                        sims = batch_matrix_multiply(Xn, Cn.T)
                    else:
                        sims = Xn @ Cn.T
                except Exception:
                    sims = Xn @ Cn.T
                d = 1.0 - sims
                return d, ctrs
            else:
                try:
                    if MATRIX_OPS and gpu_matrix_multiply is not None:
                        dots = gpu_matrix_multiply(X, ctrs_w.T)
                    elif MATRIX_OPS and batch_matrix_multiply is not None:
                        dots = batch_matrix_multiply(X, ctrs_w.T)
                    else:
                        dots = X @ ctrs_w.T
                except Exception:
                    dots = X @ ctrs_w.T
                x2 = np.sum(X * X, axis=1, keepdims=True)
                c2 = np.sum(ctrs_w * ctrs_w, axis=1, keepdims=True).T
                d2 = x2 + c2 - 2.0 * dots
                np.maximum(d2, 0.0, out=d2)
                d = np.sqrt(d2, where=(d2>=0))
                return d, ctrs

        dists, centroids = compute_dists(current)
        uniq = np.unique(current)
        k = len(uniq)
        topk = np.argsort(dists, axis=1)[:, :min(3, k)]
        cnts = np.bincount(current, minlength=k)

        # Phase A: raise to lower bound
        for _ in range(5):
            deficits = [(c, lower - cnts[c]) for c in range(k) if cnts[c] < lower]
            if not deficits:
                break
            deficits.sort(key=lambda x: x[1], reverse=True)
            moved_any = False
            for c, need in deficits:
                candidates = np.where((current != c) & ((topk[:, 0] == c) | (topk[:, 1] == c) | (topk[:, 2] == c) if topk.shape[1] >= 3 else (topk[:, 0] == c)))[0]
                donor_ok = candidates[cnts[current[candidates]] > lower]
                if donor_ok.size == 0:
                    donor_ok = candidates
                if donor_ok.size == 0:
                    continue
                deltas = dists[donor_ok, c] - dists[donor_ok, current[donor_ok]]
                order = np.argsort(deltas)
                to_move = donor_ok[order][:need]
                for idx in to_move:
                    old = current[idx]
                    if cnts[old] <= lower:
                        continue
                    current[idx] = c
                    cnts[old] -= 1
                    cnts[c] += 1
                    moved_any = True
            if not moved_any:
                break
            dists, centroids = compute_dists(current)
            topk = np.argsort(dists, axis=1)[:, :min(3, k)]

        # Phase B: reduce above upper
        for _ in range(5):
            overs = [(c, cnts[c] - upper) for c in range(k) if cnts[c] > upper]
            if not overs:
                break
            moved_any = False
            overs.sort(key=lambda x: x[1], reverse=True)
            for c, excess in overs:
                indices = np.where(current == c)[0]
                if indices.size == 0:
                    continue
                best_alt = np.full(indices.shape[0], -1, dtype=int)
                alt_cost = np.full(indices.shape[0], np.inf, dtype=float)
                for idx_i, i in enumerate(indices):
                    for alt in topk[i]:
                        if alt == c or cnts[alt] >= upper:
                            continue
                        cost = dists[i, alt]
                        if cost < alt_cost[idx_i]:
                            alt_cost[idx_i] = cost
                            best_alt[idx_i] = alt
                stay_cost = dists[indices, c]
                regret = alt_cost - stay_cost
                order = np.argsort(regret)
                moved = 0
                for idx in order:
                    if moved >= excess:
                        break
                    i = indices[idx]
                    alt = best_alt[idx]
                    if alt == -1 or cnts[alt] >= upper or cnts[c] - 1 < lower:
                        continue
                    current[i] = alt
                    cnts[c] -= 1
                    cnts[alt] += 1
                    moved += 1
                    moved_any = True
            if not moved_any:
                break
            dists, centroids = compute_dists(current)
            topk = np.argsort(dists, axis=1)[:, :min(3, k)]

        return current

    def _split_large_clusters(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Split large clusters to achieve better size distribution.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Labels with split clusters
        """
        try:
            self.logger.info("Splitting large clusters...")

            # Find large clusters
            unique_labels, counts = np.unique(labels, return_counts=True)
            large_clusters = unique_labels[counts > self.config.max_cluster_size_pct * len(labels)]

            if len(large_clusters) == 0:
                return labels

            # For each large cluster, split it
            final_labels = labels.copy()

            for cluster_id in large_clusters:
                mask = labels == cluster_id
                if mask.sum() < 2 * self.config.min_cluster_size:
                    continue

                cluster_features = features[mask]

                # Split into 2 sub-clusters
                kmeans = KMeans(n_clusters=2, random_state=self.config.random_state)
                sub_labels = kmeans.fit_predict(cluster_features)

                # Assign new cluster IDs
                new_cluster_ids = np.max(final_labels) + np.arange(1, 3)
                final_labels[mask] = np.where(sub_labels == 0, cluster_id, new_cluster_ids[0])

            self.logger.info(f"Split {len(large_clusters)} large clusters")
            return final_labels

        except Exception as e:
            self.logger.warning(f"Error splitting large clusters: {e}")
            return labels

    def _calculate_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers efficiently (batched; uses matrix ops if available).

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Cluster centers array
        """
        try:
            unique_labels = np.unique(labels)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]
            if len(unique_labels) == 0:
                return np.array([])
            # Map labels to 0..k-1
            n = features.shape[0]
            k = len(unique_labels)
            idx_map = {lab: i for i, lab in enumerate(unique_labels)}
            lid = np.vectorize(idx_map.get)(labels)
            onehot = np.zeros((n, k), dtype=np.float64)
            onehot[np.arange(n), lid] = 1.0
            # Sums via matrix ops when available
            try:
                if MATRIX_OPS and gpu_matrix_multiply is not None:
                    sums = gpu_matrix_multiply(onehot.T, features)
                elif MATRIX_OPS and batch_matrix_multiply is not None:
                    sums = batch_matrix_multiply(onehot.T, features)
                else:
                    sums = onehot.T @ features
            except Exception:
                sums = onehot.T @ features
            cnts = onehot.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ctrs = sums / np.maximum(cnts[:, None], 1.0)
            return ctrs

        except Exception as e:
            self.logger.warning(f"Error calculating cluster centers: {e}")
            return np.array([])

def create_optimal_clusterer(config: Optional[OptimalClusteringConfig] = None) -> OptimalRegimeClusterer:
    """Create optimal regime clusterer.

    Args:
        config: Clustering configuration (default: None)

    Returns:
        OptimalRegimeClusterer instance
    """
    if config is None:
        config = OptimalClusteringConfig()

    return OptimalRegimeClusterer(config)

def cluster_hmm_regimes(data_path: str, config: Optional[OptimalClusteringConfig] = None,
                       **kwargs) -> ClusteringResult:
    """Convenience function to cluster HMM regimes.

    Args:
        data_path: Path to HMM regime data
        config: Clustering configuration
        **kwargs: Additional parameters

    Returns:
        ClusteringResult
    """
    clusterer = create_optimal_clusterer(config)
    return clusterer.cluster(data_path, **kwargs)