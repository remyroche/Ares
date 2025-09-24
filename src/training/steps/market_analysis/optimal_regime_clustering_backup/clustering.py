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
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import logging
from dataclasses import dataclass
from .config import OptimalClusteringConfig
from .utils import (
    calculate_cluster_statistics, calculate_cluster_quality_metrics,
    validate_cluster_quality, detect_outliers,
    prepare_clustering_features, load_regime_data
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

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unique_labels = np.array([l for l in np.unique(labels) if l >= 0], dtype=int)
        if unique_labels.size == 0:
            return np.array([]), unique_labels
        centroids = []
        for lab in unique_labels:
            idx = labels == lab
            if not np.any(idx):
                centroids.append(np.zeros((X.shape[1],), dtype=float))
            else:
                centroids.append(np.mean(X[idx], axis=0))
        return np.vstack(centroids), unique_labels

    def _assign_noise_to_nearest(self, X: np.ndarray, labels: np.ndarray, centers: Optional[np.ndarray] = None,
                                 center_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Reassign any negative labels to nearest existing clusters to ensure full coverage."""
        if labels is None or labels.size == 0:
            return labels
        new_labels = labels.copy()
        noise_idx = np.where(new_labels < 0)[0]
        if noise_idx.size == 0:
            return new_labels
        if centers is None or center_labels is None or centers.size == 0 or center_labels.size == 0:
            centers, center_labels = self._compute_centroids(X, new_labels)
        if centers is None or centers.size == 0 or center_labels is None or center_labels.size == 0:
            # Fallback: assign all to 0 if we cannot compute centers
            new_labels[noise_idx] = 0
            return new_labels
        # Compute distances from noise points to centers
        pts = X[noise_idx]
        # Euclidean distances
        x2 = np.sum(pts * pts, axis=1, keepdims=True)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T
        dots = pts @ centers.T
        d2 = x2 + c2 - 2.0 * dots
        np.maximum(d2, 0.0, out=d2)
        nn = np.argmin(d2, axis=1)
        new_labels[noise_idx] = center_labels[nn]
        return new_labels

    def _size_cv(self, labels: np.ndarray) -> float:
        counts = [np.sum(labels == l) for l in np.unique(labels) if l >= 0]
        if len(counts) == 0:
            return 0.0
        mean = float(np.mean(counts))
        std = float(np.std(counts))
        return float(std / (mean + 1e-12))

    def _compute_transition_matrix(self, labels: np.ndarray, uniq_labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[int, int]]:
        """Compute transition count matrix across consecutive labels (non-negative only)."""
        if labels is None or labels.size < 2:
            return np.zeros((0, 0), dtype=np.int64), {}
        if uniq_labels is None:
            uniq_labels = np.array([l for l in np.unique(labels) if l >= 0], dtype=int)
        label_to_idx = {int(l): i for i, l in enumerate(uniq_labels)}
        k = len(uniq_labels)
        T = np.zeros((k, k), dtype=np.int64)
        prev = labels[:-1]
        next_ = labels[1:]
        for a, b in zip(prev, next_):
            if a < 0 or b < 0:
                continue
            ia = label_to_idx.get(int(a))
            ib = label_to_idx.get(int(b))
            if ia is None or ib is None:
                continue
            T[ia, ib] += 1
        return T, label_to_idx

    def _mutual_transition_strength(self, T: np.ndarray, i: int, j: int) -> float:
        """Return symmetric flow strength normalized by external flow: s / (s + ext)."""
        if T.size == 0:
            return 0.0
        s = float(T[i, j] + T[j, i])
        out_i = float(T[i, :].sum())
        in_i = float(T[:, i].sum())
        out_j = float(T[j, :].sum())
        in_j = float(T[:, j].sum())
        ext = max(0.0, (out_i + in_i + out_j + in_j) - 2.0 * s)
        denom = s + ext
        return float(s / denom) if denom > 0 else 0.0

    def _temporal_coherence_gain(self, T: np.ndarray, i: int, j: int) -> float:
        """Gain in self-coherence if i and j are merged: (i→j + j→i) / (out_i + out_j)."""
        if T.size == 0:
            return 0.0
        s = float(T[i, j] + T[j, i])
        out_i = float(T[i, :].sum())
        out_j = float(T[j, :].sum())
        denom = out_i + out_j
        return float(s / denom) if denom > 0 else 0.0

    def _stable_kmeans_selection(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, KMeans]:
        """Run multiple seeds and pick the best KMeans by size-penalized objective."""
        best_obj = np.inf
        best_labels: Optional[np.ndarray] = None
        best_model: Optional[KMeans] = None
        base_seed = int(self.config.random_state)
        seeds = [base_seed + i for i in range(int(self.config.kmeans_num_seeds))]
        for seed in seeds:
            model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=int(self.config.kmeans_n_init),
                max_iter=int(self.config.kmeans_max_iter),
                random_state=seed,
            )
            labels = model.fit_predict(X)
            inertia = float(getattr(model, 'inertia_', 0.0))
            size_pen = self._size_cv(labels)
            obj = inertia * 1.0 + float(self.config.size_penalty_weight) * size_pen * X.shape[0]
            if obj < best_obj:
                best_obj = obj
                best_labels = labels
                best_model = model
        # Fallback single run if something went wrong
        if best_labels is None or best_model is None:
            best_model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=int(max(10, self.config.kmeans_n_init)),
                max_iter=int(self.config.kmeans_max_iter),
                random_state=base_seed,
            )
            best_labels = best_model.fit_predict(X)
        return best_labels, best_model

    def _constrained_kmeans(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Constrained K-Means using k-means-constrained if available, else fallback to assignment."""
        n = X.shape[0]
        size_min = max(1, int(np.floor(self.config.min_cluster_size_pct * n)))
        size_max = max(1, int(np.ceil(self.config.max_cluster_size_pct * n)))
        labels: Optional[np.ndarray] = None
        if bool(self.config.constrained_kmeans_enabled):
            try:
                from k_means_constrained import KMeansConstrained  # type: ignore
                model = KMeansConstrained(
                    n_clusters=n_clusters,
                    size_min=size_min,
                    size_max=size_max,
                    init='k-means++',
                    n_init=int(self.config.kmeans_n_init),
                    max_iter=int(self.config.kmeans_max_iter),
                    random_state=int(self.config.random_state),
                )
                labels = model.fit_predict(X)
            except Exception:
                labels = None
        if labels is None:
            # Fallback: stable KMeans then bounded reassignment
            labels, _ = self._stable_kmeans_selection(X, n_clusters)
            labels = self._capacity_constrained_assignment(
                X, labels, float(self.config.min_cluster_size_pct), float(self.config.max_cluster_size_pct)
            )
        return labels

    def _merge_two_clusters(self, labels: np.ndarray, a: int, b: int) -> np.ndarray:
        new_labels = labels.copy()
        new_labels[labels == b] = a
        # Reindex to compact [0..k-1]
        uniq = [l for l in np.unique(new_labels) if l >= 0]
        remap = {lab: i for i, lab in enumerate(sorted(uniq))}
        mapped = np.array([remap[l] if l in remap else l for l in new_labels], dtype=int)
        return mapped

    def _pairwise_centroid_similarities(self, C: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        if C.size == 0 or C.shape[0] < 2:
            return np.array([])
        if metric == 'cosine':
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            sims = Cn @ Cn.T
            mask = ~np.eye(C.shape[0], dtype=bool)
            return sims[mask]
        # euclidean -> convert to negative distance similarity surrogate
        dists = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
        mask = ~np.eye(C.shape[0], dtype=bool)
        return -dists[mask]

    def _overcluster_then_merge(self, X: np.ndarray, target_k: int) -> np.ndarray:
        k_min = int(self.config.overcluster_k_min)
        k_max = int(self.config.overcluster_k_max)
        best_obj = np.inf
        best_labels: Optional[np.ndarray] = None
        best_centers: Optional[np.ndarray] = None
        # Try several k values and pick best by size-penalized inertia
        for k in range(k_min, k_max + 1):
            labels_k, model_k = self._stable_kmeans_selection(X, k)
            inertia = float(getattr(model_k, 'inertia_', 0.0))
            size_pen = self._size_cv(labels_k)
            obj = inertia * 1.0 + float(self.config.size_penalty_weight) * size_pen * X.shape[0]
            if obj < best_obj:
                best_obj = obj
                best_labels = labels_k
                best_centers = getattr(model_k, 'cluster_centers_', None)
        if best_labels is None:
            # Fallback: single run at k_max
            best_labels, model = self._stable_kmeans_selection(X, k_max)
            best_centers = getattr(model, 'cluster_centers_', None)

        labels = best_labels
        if best_centers is None:
            C, labs = self._compute_centroids(X, labels)
        else:
            C = best_centers
            labs = np.arange(C.shape[0])

        # Pre-compute similarity distribution for gating
        sims = self._pairwise_centroid_similarities(C, metric=getattr(self.config, 'merge_similarity_metric', 'cosine'))
        easy_thr = np.quantile(sims, float(self.config.easy_merge_top_percentile)) if sims.size > 0 else 0.0

        n = X.shape[0]
        lower = float(self.config.min_cluster_size_pct) * n
        upper = float(self.config.max_cluster_size_pct) * n

        while len(np.unique(labels)) > target_k:
            C, uniq = self._compute_centroids(X, labels)
            if C.size == 0:
                break
            metric = getattr(self.config, 'merge_similarity_metric', 'cosine')
            if metric == 'cosine':
                Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                sims_mat = Cn @ Cn.T
                np.fill_diagonal(sims_mat, -np.inf)
            else:
                dists = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
                np.fill_diagonal(dists, np.inf)
            # Transition-aware scoring
            T, label_to_idx = self._compute_transition_matrix(labels, uniq)
            counts = {int(l): int(np.sum(labels == l)) for l in uniq}
            # Build candidate list with weighted objective
            candidates = []
            for i in range(C.shape[0]):
                for j in range(i + 1, C.shape[0]):
                    a, b = int(uniq[i]), int(uniq[j])
                    merged_size = counts[a] + counts[b]
                    size_ok = merged_size <= max(upper, 1.0)
                    if metric == 'cosine':
                        pair_sim = float(sims_mat[i, j])
                        feature_score = (pair_sim + 1.0) * 0.5
                    else:
                        dist = float(dists[i, j])
                        feature_score = 1.0 / (1.0 + max(dist, 0.0))
                    # Temporal components
                    mi = self._mutual_transition_strength(T, i, j)
                    tg = self._temporal_coherence_gain(T, i, j)
                    # Weights: 80% feature similarity, 10% mutual transition, 10% temporal coherence gain
                    merge_score = 0.80 * feature_score + 0.10 * mi + 0.10 * tg
                    candidates.append((merge_score, i, j, size_ok))
            if not candidates:
                break
            # Prefer size_ok candidates; then by highest score
            candidates.sort(key=lambda x: (not x[3], -x[0]))
            _, i, j, _ = candidates[0]
            a, b = int(uniq[i]), int(uniq[j])

            new_labels = self._merge_two_clusters(labels, a, b)
            try:
                sil_before = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1.0
                sil_after = silhouette_score(X, new_labels) if len(np.unique(new_labels)) > 1 else -1.0
                degrade = sil_after - sil_before
                if degrade >= -0.01:
                    labels = new_labels
                else:
                    # Remove this pair and try next best
                    # Mark this pair as unusable
                    for idx, cand in enumerate(candidates):
                        if cand[1] == i and cand[2] == j:
                            candidates.pop(idx)
                            break
                    if not candidates:
                        break
                    _, i2, j2, _ = candidates[0]
                    a2, b2 = int(uniq[i2]), int(uniq[j2])
                    labels = self._merge_two_clusters(labels, a2, b2)
            except Exception:
                labels = new_labels

        # Final size enforcement
        labels = self._capacity_constrained_assignment(
            X, labels, float(self.config.min_cluster_size_pct), float(self.config.max_cluster_size_pct)
        )
        return labels

    def _postprocess_split_merge(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Split oversized, merge undersized with percentile-gated criteria and silhouette/DB checks."""
        n = X.shape[0]
        min_pct = float(self.config.min_cluster_size_pct)
        max_pct = float(self.config.max_cluster_size_pct)
        lower = min_pct * n
        upper = max_pct * n
        metric = getattr(self.config, 'merge_similarity_metric', 'cosine')
        try:
            sil_global_before = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1.0
        except Exception:
            sil_global_before = -1.0

        for _ in range(int(self.config.split_merge_max_iters)):
            changed = False
            # Recompute
            uniq = [l for l in np.unique(labels) if l >= 0]
            if not uniq:
                break
            C, uniq_arr = self._compute_centroids(X, labels)
            sims = self._pairwise_centroid_similarities(C, metric=metric)
            easy_thr = np.quantile(sims, float(self.config.easy_merge_top_percentile)) if sims.size > 0 else 0.0
            strict_thr = np.quantile(sims, float(self.config.strict_split_bottom_percentile)) if sims.size > 0 else 0.0

            counts = {int(c): int(np.sum(labels == c)) for c in uniq}

            # Oversized -> try split
            for c in list(uniq):
                if counts[c] > upper and counts[c] >= 4:
                    idx = np.where(labels == c)[0]
                    try:
                        sub_labels, _ = self._stable_kmeans_selection(X[idx], 2)
                    except Exception:
                        km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=self.config.random_state)
                        sub_labels = km.fit_predict(X[idx])
                    # Compute sub-centroid similarity
                    subC0 = np.mean(X[idx][sub_labels == 0], axis=0)
                    subC1 = np.mean(X[idx][sub_labels == 1], axis=0)
                    if metric == 'cosine':
                        s = float(np.dot(subC0, subC1) / ((np.linalg.norm(subC0) + 1e-12) * (np.linalg.norm(subC1) + 1e-12)))
                    else:
                        s = float(-np.linalg.norm(subC0 - subC1))
                    # Build candidate labels
                    new_labels = labels.copy()
                    new_id = max(uniq) + 1
                    new_labels[idx[sub_labels == 1]] = new_id
                    try:
                        sil_after = silhouette_score(X, new_labels) if len(np.unique(new_labels)) > 1 else sil_global_before
                    except Exception:
                        sil_after = sil_global_before
                    # Accept if subclusters are sufficiently distinct (s <= strict_thr for euclidean surrogate negative) or if silhouette doesn't degrade
                    accept = (metric == 'cosine' and s <= strict_thr) or (metric != 'cosine' and s >= strict_thr) or (sil_after >= sil_global_before - 0.005)
                    if accept:
                        labels = new_labels
                        sil_global_before = sil_after
                        changed = True
            if changed:
                continue

            # Undersized -> try merge to nearest neighbor
            C, uniq_arr = self._compute_centroids(X, labels)
            counts = {int(c): int(np.sum(labels == c)) for c in [int(u) for u in uniq_arr]}
            for i, c in enumerate(uniq_arr):
                if counts[int(c)] < lower:
                    # choose most similar neighbor
                    if metric == 'cosine':
                        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                        sims_mat = Cn @ Cn.T
                        sims_mat[i, i] = -np.inf
                        j = int(np.argmax(sims_mat[i]))
                        pair_sim = float(sims_mat[i, j])
                    else:
                        d = np.linalg.norm(C - C[i], axis=1)
                        d[i] = np.inf
                        j = int(np.argmin(d))
                        pair_sim = float(-d[j])
                    a = int(uniq_arr[i])
                    b = int(uniq_arr[j])
                    new_labels = self._merge_two_clusters(labels, a, b)
                    try:
                        sil_after = silhouette_score(X, new_labels) if len(np.unique(new_labels)) > 1 else sil_global_before
                    except Exception:
                        sil_after = sil_global_before
                    easy_pair = pair_sim >= easy_thr
                    # Accept if easy pair or global silhouette doesn't degrade materially
                    if easy_pair or (sil_after >= sil_global_before - 0.01):
                        labels = new_labels
                        sil_global_before = sil_after
                        changed = True
            if not changed:
                break

        # Final size enforcement pass
        labels = self._capacity_constrained_assignment(X, labels, min_pct, max_pct)
        return labels

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
            quality_metrics = calculate_cluster_quality_metrics(features, final_labels, feature_metadata)

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
            quality_metrics = calculate_cluster_quality_metrics(features, labels, feature_metadata)

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
        """Perform initial clustering pass; all points remain covered by reassignment.

        Args:
            features: Feature matrix

        Returns:
            Noise-reduced labels
        """
        try:
            self.logger.info("Performing initial clustering pass...")

            # Use HDBSCAN for noise reduction
            try:
                from hdbscan import HDBSCAN
                clusterer = HDBSCAN(
                    min_cluster_size=self.config.min_cluster_size,
                    min_samples=self.config.min_samples,
                    cluster_selection_epsilon=self.config.cluster_selection_epsilon
                )
                labels = clusterer.fit_predict(features)
                labels = self._assign_noise_to_nearest(features, labels)
                self.logger.info(f"Initial pass produced {len(np.unique(labels))} clusters")
                return labels
            except ImportError:
                self.logger.warning("HDBSCAN not available, using DBSCAN for initial pass")
                return self._dbscan_clustering(features)

        except Exception as e:
            self.logger.warning(f"Error in noise reduction clustering: {e}")
            return self._assign_noise_to_nearest(features, np.zeros(len(features), dtype=int))

    def _main_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform main clustering using K-means.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            self.logger.info("Performing main clustering...")

            # Constrained path preferred if enabled
            if bool(self.config.constrained_kmeans_enabled):
                labels = self._constrained_kmeans(features, int(self.config.target_n_clusters))
            # Else overcluster and merge if enabled
            elif bool(self.config.overcluster_enabled):
                labels = self._overcluster_then_merge(features, int(self.config.target_n_clusters))
            else:
                # Stable KMeans selection
                labels, _ = self._stable_kmeans_selection(features, int(self.config.target_n_clusters))
            self.logger.info(f"Main clustering produced {len(np.unique(labels))} clusters")
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
            labels = self._assign_noise_to_nearest(features, labels)
            self.logger.info(f"DBSCAN produced {len(np.unique(labels))} clusters")
            return labels

        except Exception as e:
            self.logger.error(f"Error in DBSCAN clustering: {e}")
            raise

    def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform K-means clustering using the stable selection/constraints pipeline."""
        try:
            return self._main_clustering(features)
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
            # First clustering pass
            dbscan_labels = self._dbscan_clustering(features)

            # Identify core points (treat all as covered)
            core_mask = np.ones(len(features), dtype=bool)
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
            final_labels = self._assign_noise_to_nearest(features, final_labels)
            self.logger.info(f"Hybrid clustering: {len(np.unique(final_labels))} clusters")
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

            # Ensure full coverage by reassigning any negative labels
            if noise_labels is not None and noise_labels.size == len(final_labels):
                if np.any(noise_labels < 0):
                    C, uniq = self._compute_centroids(features, final_labels)
                    final_labels = self._assign_noise_to_nearest(features, final_labels, centers=C, center_labels=uniq)

            # Optimize cluster sizes if needed
            if self.config.adaptive_clustering:
                final_labels = self._optimize_cluster_sizes(features, final_labels)
            # Postprocess split-merge with percentile gating
            if bool(self.config.split_merge_enabled):
                final_labels = self._postprocess_split_merge(features, final_labels)

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
            elif distance_metric == "mahalanobis":
                # Regularized covariance inverse
                try:
                    cov = np.cov(X.T)
                    eps = 1e-6
                    cov.flat[:: cov.shape[0] + 1] += eps
                    inv = np.linalg.inv(cov)
                except Exception:
                    inv = np.eye(X.shape[1], dtype=np.float64)
                try:
                    if MATRIX_OPS and gpu_matrix_multiply is not None:
                        X_inv = gpu_matrix_multiply(X, inv)
                        C_inv = gpu_matrix_multiply(ctrs_w, inv)
                    elif MATRIX_OPS and batch_matrix_multiply is not None:
                        X_inv = batch_matrix_multiply(X, inv)
                        C_inv = batch_matrix_multiply(ctrs_w, inv)
                    else:
                        X_inv = X @ inv
                        C_inv = ctrs_w @ inv
                except Exception:
                    X_inv = X @ inv
                    C_inv = ctrs_w @ inv
                x_term = np.sum(X * X_inv, axis=1, keepdims=True)
                c_term = np.sum(ctrs_w * C_inv, axis=1, keepdims=True).T
                cross = X @ C_inv.T
                d2 = x_term + c_term - 2.0 * cross
                np.maximum(d2, 0.0, out=d2)
                d = np.sqrt(d2, where=(d2>=0))
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

        # ---- Temporal coherence helpers (time-aware costs) ----
        def temporal_penalty_for_assignment(i: int, target: int) -> float:
            # Penalty for increasing boundaries around position i when assigning to target
            left = current[i - 1] if i > 0 else target
            right = current[i + 1] if i < n - 1 else target
            before = int(current[i] != left) + int(current[i] != right)
            after = int(target != left) + int(target != right)
            delta = after - before
            return float(max(0, delta) / 2.0)  # in [0,1]

        def dwell_alignment_penalty(i: int, target: int) -> float:
            # Lower penalty if neighbors already belong to target cluster
            same = 0
            if i > 0 and current[i - 1] == target:
                same += 1
            if i < n - 1 and current[i + 1] == target:
                same += 1
            return float(1.0 - same / 2.0)  # 0 if both neighbors same as target, 0.5 if one, 1.0 if none

        def composite_cost(i: int, target: int) -> float:
            base = float(dists[i, target])
            denom = float(dists[i, topk[i][0]]) if topk.shape[1] >= 1 else (float(np.max(dists[i])) + 1e-12)
            base_norm = base / (denom + 1e-12)
            tpen = temporal_penalty_for_assignment(i, target)
            dpen = dwell_alignment_penalty(i, target)
            return 0.80 * base_norm + 0.15 * tpen + 0.05 * dpen

        # Phase A: raise to lower bound (time-aware cost)
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
                # Select by composite time-aware cost
                comp = np.array([composite_cost(int(idx), int(c)) for idx in donor_ok])
                order = np.argsort(comp)
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

        # Phase B: reduce above upper (time-aware cost)
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
                    # evaluate a few nearest alternatives
                    for alt in topk[i]:
                        if alt == c or cnts[alt] >= upper:
                            continue
                        cost = composite_cost(int(i), int(alt))
                        if cost < alt_cost[idx_i]:
                            alt_cost[idx_i] = cost
                            best_alt[idx_i] = alt
                # prefer lowest composite cost moves
                order = np.argsort(alt_cost)
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
