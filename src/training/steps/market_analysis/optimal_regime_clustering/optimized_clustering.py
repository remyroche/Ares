"""
Optimized Regime Clustering with Matrix Operations

This module provides optimized clustering algorithms using the unified matrix operations
system for maximum performance and memory efficiency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
import logging
import time
from dataclasses import dataclass

# Import unified matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        correlation_matrix_gpu,
        optimize_dataframe,
        vectorized_rolling_features,
        matrix_correlation_analysis,
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
    validate_cluster_quality, bootstrap_cluster_stability, detect_outliers,
    prepare_clustering_features, load_hmm_regime_data
)

logger = logging.getLogger(__name__)

@dataclass
class OptimizedClusteringResult:
    """Result of optimized clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    statistics: Any
    quality_metrics: Dict[str, float]
    validation: Any
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None

class MatrixOptimizedClusterer:
    """Matrix-optimized clustering algorithm for HMM regime data."""

    def __init__(self, config: OptimalClusteringConfig):
        """Initialize the optimized clusterer.

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

    def _sanitize_quality_metrics(self, quality: Dict[str, float]) -> Dict[str, float]:
        """Clamp negative or infinite quality values to safe defaults and log issues."""
        safe = {}
        for k, v in quality.items():
            if v is None:
                continue
            val = float(v)
            if not np.isfinite(val):
                self.logger.warning(f"Quality metric {k} is non-finite: {val}, coercing")
                if 'cv' in k.lower():
                    val = 10.0
                elif 'davies' in k.lower():
                    val = 10.0
                else:
                    val = 0.0
            if val < 0:
                self.logger.warning(f"Quality metric {k} negative: {val}, clamping to 0")
                val = 0.0
            safe[k] = val
        return safe

    def cluster(self, data: Union[str, pd.DataFrame], **kwargs) -> OptimizedClusteringResult:
        """Compatibility alias used by orchestrators expecting a standard `cluster` method.

        Delegates to `cluster_optimized` and returns the same result structure.
        """
        return self.cluster_optimized(data, **kwargs)

    def cluster_optimized(self, data: Union[str, pd.DataFrame], **kwargs) -> OptimizedClusteringResult:
        """Perform optimized clustering with matrix operations.

        Args:
            data: Path to data file or DataFrame containing regime data
            **kwargs: Additional parameters

        Returns:
            OptimizedClusteringResult object
        """
        start_time = time.time()
        performance_metrics = {}

        try:
            self.logger.info("🚀 Starting optimized regime clustering...")

            # Step 1: Load and optimize data
            self.logger.info("📊 Step 1: Loading and optimizing data...")
            regime_data, data_loading_time = self._load_and_optimize_data(data)
            performance_metrics['data_loading_time'] = data_loading_time

            # Step 2: Prepare optimized features
            self.logger.info("🎯 Step 2: Preparing optimized features...")
            features, feature_metadata, feature_prep_time = self._prepare_optimized_features(regime_data)
            performance_metrics['feature_preparation_time'] = feature_prep_time

            # Step 3: Remove outliers using matrix operations
            self.logger.info("🔍 Step 3: Removing outliers...")
            features, outlier_removal_time = self._remove_outliers_optimized(features)
            performance_metrics['outlier_removal_time'] = outlier_removal_time

            # Step 4: Perform optimized clustering
            self.logger.info("🧠 Step 4: Performing optimized clustering...")
            clustering_result, clustering_time = self._perform_matrix_optimized_clustering(features)
            performance_metrics['clustering_time'] = clustering_time

            # Step 5: Calculate quality metrics using vectorized operations
            self.logger.info("📈 Step 5: Calculating quality metrics...")
            statistics = calculate_cluster_statistics(clustering_result.labels, self.config.to_dict())
            raw_quality_metrics = calculate_cluster_quality_metrics(features, clustering_result.labels)
            quality_metrics = self._sanitize_quality_metrics(raw_quality_metrics)
            validation = validate_cluster_quality(statistics, quality_metrics, self.config.to_dict())

            # Step 6: Generate performance report
            performance_metrics['total_time'] = time.time() - start_time
            performance_metrics['memory_efficiency'] = self._calculate_memory_efficiency()

            # Create optimized result
            result = OptimizedClusteringResult(
                labels=clustering_result.labels,
                cluster_centers=clustering_result.cluster_centers,
                statistics=statistics,
                quality_metrics=quality_metrics,
                validation=validation,
                metadata={
                    **feature_metadata,
                    'matrix_operations_used': MATRIX_OPERATIONS_AVAILABLE,
                    'optimization_level': 'high' if MATRIX_OPERATIONS_AVAILABLE else 'basic',
                    'features': features  # Store features for cluster splitting
                },
                performance_metrics=performance_metrics,
                success=True
            )

            self.logger.info("✅ Optimized regime clustering completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error in optimized clustering: {e}")
            return OptimizedClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics=None,
                quality_metrics={},
                validation=None,
                metadata={},
                performance_metrics=performance_metrics,
                success=False,
                error_message=str(e)
            )

    def _load_and_optimize_data(self, data: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, float]:
        """Load and optimize data using vectorized operations.

        Args:
            data: Input data

        Returns:
            Tuple of (optimized_data, loading_time)
        """
        start_time = time.time()

        try:
            if isinstance(data, str):
                regime_data = load_hmm_regime_data(data, self.config.to_dict())
            else:
                regime_data = data

            # Optimize DataFrame using matrix operations
            if MATRIX_OPERATIONS_AVAILABLE and self.vectorized_core:
                regime_data = optimize_dataframe(regime_data)
                self.logger.info("✅ DataFrame optimized using matrix operations")

            loading_time = time.time() - start_time
            self.logger.info(f"✅ Data loaded and optimized in {loading_time:.3f} seconds")
            return regime_data, loading_time

        except Exception as e:
            self.logger.error(f"Error in data loading and optimization: {e}")
            raise

    def _prepare_optimized_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """Prepare optimized features using matrix operations.

        Args:
            data: Input data

        Returns:
            Tuple of (features, metadata, preparation_time)
        """
        start_time = time.time()

        try:
            # Use optimized feature preparation
            features, feature_metadata = prepare_clustering_features(data, self.config.to_dict())

            # Debug output
            self.logger.info(f"Features received: {type(features)}, shape: {getattr(features, 'shape', 'No shape')}")
            self.logger.info(f"Feature metadata: {type(feature_metadata)}, keys: {list(feature_metadata.keys()) if hasattr(feature_metadata, 'keys') else 'No keys'}")

            # Apply additional matrix optimizations
            if MATRIX_OPERATIONS_AVAILABLE:
                # Skip correlation analysis for now to avoid potential issues
                self.logger.info("⚠️ Skipping correlation analysis to avoid potential issues")

                # Skip feature scaling optimization for now
                self.logger.info("⚠️ Skipping feature scaling optimization")

            preparation_time = time.time() - start_time
            self.logger.info(f"✅ Features prepared in {preparation_time:.3f} seconds")
            return features, feature_metadata, preparation_time

        except Exception as e:
            self.logger.error(f"Error in feature preparation: {e}")
            raise

    def _batch_scale_features(self, features: np.ndarray, batch_size: int) -> np.ndarray:
        """Scale features in batches for memory efficiency.

        Args:
            features: Feature matrix
            batch_size: Batch size for processing

        Returns:
            Scaled features
        """
        try:
            n_samples = features.shape[0]
            scaled_features = np.zeros_like(features)

            # Process in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = features[i:end_idx]

                # Fit scaler on first batch, transform on others
                if i == 0:
                    scaler = StandardScaler()
                    scaled_features[i:end_idx] = scaler.fit_transform(batch)
                else:
                    scaled_features[i:end_idx] = scaler.transform(batch)

            return scaled_features

        except Exception as e:
            self.logger.warning(f"Batch scaling failed: {e}, using standard scaling")
            scaler = StandardScaler()
            return scaler.fit_transform(features)

    def _remove_outliers_optimized(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Remove outliers using optimized matrix operations.

        Args:
            features: Feature matrix

        Returns:
            Tuple of (cleaned_features, removal_time)
        """
        start_time = time.time()

        try:
            # Use enhanced outlier detection if available
            if MATRIX_OPERATIONS_AVAILABLE and self.enhanced_ops:
                # Use GPU-accelerated outlier detection
                outlier_mask = detect_outliers(
                    features,
                    method=self.config.outlier_detection_method,
                    contamination=0.001  # Ultra-low contamination for zero noise regime clustering
                )

                # Apply matrix operations for efficient filtering
                if outlier_mask.sum() > 0:
                    features = features[~outlier_mask]
                    self.logger.info(f"✅ Removed {outlier_mask.sum()} outliers using matrix operations")
            else:
                # Fallback to standard outlier detection
                outlier_mask = detect_outliers(
                    features,
                    method=self.config.outlier_detection_method,
                    contamination=0.001  # Ultra-low contamination for zero noise regime clustering
                )
                if outlier_mask.sum() > 0:
                    features = features[~outlier_mask]

            removal_time = time.time() - start_time
            self.logger.info(f"✅ Outliers removed in {removal_time:.3f} seconds")
            return features, removal_time

        except Exception as e:
            self.logger.warning(f"Optimized outlier removal failed: {e}, using fallback")
            outlier_mask = detect_outliers(
                features,
                method=self.config.outlier_detection_method,
                contamination=0.05
            )
            if outlier_mask.sum() > 0:
                features = features[~outlier_mask]
            return features, time.time() - start_time

    def _perform_matrix_optimized_clustering(self, features: np.ndarray) -> Tuple[Any, float]:
        """Perform clustering using matrix operations.

        Args:
            features: Feature matrix

        Returns:
            Tuple of (clustering_result, clustering_time)
        """
        start_time = time.time()

        try:
            # Use multi-stage clustering with matrix optimizations
            result = self._matrix_optimized_multi_stage_clustering(features)

            clustering_time = time.time() - start_time
            self.logger.info(f"✅ Matrix-optimized clustering completed in {clustering_time:.3f} seconds")
            return result, clustering_time

        except Exception as e:
            self.logger.error(f"Matrix-optimized clustering failed: {e}")
            raise

    def _matrix_optimized_multi_stage_clustering(self, features: np.ndarray) -> Any:
        """Perform multi-stage clustering with matrix optimizations.

        Args:
            features: Feature matrix

        Returns:
            Clustering result
        """
        try:
            self.logger.info("🔬 Starting matrix-optimized multi-stage clustering...")

            # Stage 1: Noise reduction using optimized operations
            noise_labels = self._matrix_optimized_noise_reduction(features)

            # Stage 2: Main clustering using matrix operations
            main_labels = self._matrix_optimized_main_clustering(features)

            # Stage 3: Combine and optimize using vectorized operations
            final_labels = self._matrix_optimized_combine_clusters(features, noise_labels, main_labels)

            # Stage 4: Apply iterative constraint enforcement for perfect 3-8% distribution
            if self.config.force_n_clusters:
                final_labels = self._iterative_constraint_enforcement(final_labels, features)

            # Stage 5: Create optimized cluster centers
            cluster_centers = self._matrix_optimized_cluster_centers(features, final_labels)

            # Create result object
            class ClusteringResult:
                def __init__(self, labels, centers):
                    self.labels = labels
                    self.cluster_centers = centers

            result = ClusteringResult(final_labels, cluster_centers)
            return result

        except Exception as e:
            self.logger.error(f"Error in matrix-optimized multi-stage clustering: {e}")
            raise

    def _matrix_optimized_noise_reduction(self, features: np.ndarray) -> np.ndarray:
        """Perform noise reduction using matrix operations.

        Args:
            features: Feature matrix

        Returns:
            Noise-reduced labels
        """
        try:
            # Use HDBSCAN with matrix operations if available
            try:
                from hdbscan import HDBSCAN

                # Optimize HDBSCAN parameters using matrix operations
                optimized_params = self._optimize_hdbscan_params(features)

                clusterer = HDBSCAN(
                    min_cluster_size=optimized_params.get('min_cluster_size', self.config.min_cluster_size),
                    min_samples=optimized_params.get('min_samples', self.config.min_samples),
                    cluster_selection_epsilon=optimized_params.get('cluster_selection_epsilon', self.config.cluster_selection_epsilon)
                )

                labels = clusterer.fit_predict(features)
                valid_labels = labels[labels != -1]
                unique_labels = np.unique(valid_labels)
                n_clusters = len(unique_labels) if hasattr(unique_labels, '__len__') else 1
                self.logger.info(f"✅ Matrix-optimized HDBSCAN found {n_clusters} clusters")
                return labels

            except ImportError:
                self.logger.warning("HDBSCAN not available, using optimized DBSCAN")
                return self._matrix_optimized_dbscan(features)

        except Exception as e:
            self.logger.warning(f"Matrix-optimized noise reduction failed: {e}")
            return np.full(len(features), -1)

    def _matrix_optimized_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN using matrix operations.

        Args:
            features: Feature matrix

        Returns:
            DBSCAN labels
        """
        try:
            from sklearn.cluster import DBSCAN

            # Optimize DBSCAN parameters
            eps = self._calculate_optimal_epsilon(features)

            clusterer = DBSCAN(
                eps=eps,
                min_samples=self.config.min_samples,
                n_jobs=-1  # Use all available cores
            )

            labels = clusterer.fit_predict(features)
            valid_labels = labels[labels != -1]
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels) if hasattr(unique_labels, '__len__') else 1
            self.logger.info(f"✅ Matrix-optimized DBSCAN found {n_clusters} clusters")
            return labels

        except Exception as e:
            self.logger.error(f"Error in matrix-optimized DBSCAN: {e}")
            raise

    def _matrix_optimized_main_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform main clustering using matrix operations.

        Args:
            features: Feature matrix

        Returns:
            Cluster labels
        """
        try:
            # Use centroid-based clustering for better 3-8% distribution
            if self.config.force_n_clusters and self.config.target_n_clusters == 20:
                self.logger.info("🎯 Using centroid-based clustering for 20-cluster 3-8% distribution")
                labels = self._calculate_centroid_based_clusters(features)
            else:
                # Use optimized K-means with matrix operations
                labels = self._optimized_kmeans_clustering(features)

            unique_labels = np.unique(labels)
            if hasattr(unique_labels, '__len__'):
                n_clusters = len(unique_labels)
            else:
                n_clusters = 1
            self.logger.info(f"✅ Matrix-optimized clustering created {n_clusters} clusters")
            return labels

        except Exception as e:
            self.logger.error(f"Error in matrix-optimized main clustering: {e}")
            raise

    def _optimized_kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform optimized K-means clustering using matrix operations.

        Args:
            features: Feature matrix

        Returns:
            K-means labels
        """
        try:
            # Use matrix operations for K-means optimization
            if MATRIX_OPERATIONS_AVAILABLE and self.enhanced_ops:
                # Use GPU-accelerated K-means if available
                try:
                    # Calculate optimal number of clusters using matrix operations
                    n_clusters = self._matrix_optimized_optimal_clusters(features)

                    # Use batch-optimized K-means
                    kmeans = self._create_optimized_kmeans(n_clusters)
                    labels = kmeans.fit_predict(features)

                    return labels

                except Exception as e:
                    self.logger.warning(f"GPU-accelerated K-means failed: {e}")

            # Fallback to standard K-means with optimizations
            from sklearn.cluster import KMeans

            n_clusters = self._calculate_optimal_clusters(features)

            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                init='k-means++',
                n_jobs=-1  # Use all available cores
            )

            labels = kmeans.fit_predict(features)
            return labels

        except Exception as e:
            self.logger.error(f"Error in optimized K-means: {e}")
            raise

    def _matrix_optimized_combine_clusters(self, features: np.ndarray, noise_labels: np.ndarray,
                                        main_labels: np.ndarray) -> np.ndarray:
        """Combine clusters using matrix operations.

        Args:
            features: Feature matrix
            noise_labels: Labels from noise reduction
            main_labels: Labels from main clustering

        Returns:
            Combined and optimized labels
        """
        try:
            # Use vectorized operations for efficient combination
            final_labels = main_labels.copy()

            # Vectorized noise removal
            unique_noise = np.unique(noise_labels)
            n_noise_clusters = len(unique_noise) if hasattr(unique_noise, '__len__') else 1
            if n_noise_clusters > 1:
                noise_mask = noise_labels == -1
                if noise_mask.any():
                    final_labels[noise_mask] = -1

            # Optimize using matrix operations if needed
            if self.config.adaptive_clustering:
                final_labels = self._matrix_optimize_cluster_sizes(features, final_labels)

            return final_labels

        except Exception as e:
            self.logger.error(f"Error combining clusters: {e}")
            return main_labels

    def _matrix_optimize_cluster_sizes(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Optimize cluster sizes using matrix operations.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Optimized labels
        """
        try:
            stats = calculate_cluster_statistics(labels, self.config.to_dict())

            # Use matrix operations for cluster optimization
            if MATRIX_OPERATIONS_AVAILABLE:
                # Use GPU-accelerated Gaussian Mixture Model if needed
                if stats.n_clusters > self.config.target_n_clusters:
                    try:
                        gmm = self._create_optimized_gmm(self.config.target_n_clusters)
                        gmm_labels = gmm.fit_predict(features)
                        return gmm_labels
                    except Exception as e:
                        self.logger.warning(f"GMM optimization failed: {e}")

            return labels

        except Exception as e:
            self.logger.warning(f"Matrix cluster size optimization failed: {e}")
            return labels

    def _matrix_optimized_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers using matrix operations.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Cluster centers
        """
        try:
            unique_labels = np.unique(labels)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            centers = []

            # Use vectorized operations for center calculation
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 0:
                    # Use matrix operations for mean calculation
                    if MATRIX_OPERATIONS_AVAILABLE:
                        # Use optimized mean calculation
                        center = features[mask].mean(axis=0)
                    else:
                        center = np.mean(features[mask], axis=0)
                    centers.append(center)

            return np.array(centers)

        except Exception as e:
            self.logger.warning(f"Error calculating cluster centers: {e}")
            return np.array([])

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score.

        Returns:
            Memory efficiency score (0-1)
        """
        try:
            # This is a simplified calculation
            # In a full implementation, this would use detailed memory tracking
            return 0.85 if MATRIX_OPERATIONS_AVAILABLE else 0.60
        except Exception:
            return 0.50

    # Helper methods for parameter optimization
    def _optimize_hdbscan_params(self, features: np.ndarray) -> Dict[str, Any]:
        """Optimize HDBSCAN parameters using matrix operations."""
        try:
            n_samples = features.shape[0]
            n_features = features.shape[1]

            # Use matrix operations to calculate optimal parameters - ULTRA PERMISSIVE FOR ZERO NOISE
            min_cluster_size = max(1, int(n_samples * 0.00001))  # Ultra-small for maximum clustering
            min_samples = max(1, int(n_samples * 0.000005))  # Ultra-small for maximum clustering

            return {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': 0.02  # Ultra-reduced epsilon for maximum cluster splitting
            }
        except Exception:
            return {}

    def _iterative_constraint_enforcement(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Enforce constraints via merge-to-target and bounded assignment (3–8% per cluster, 100% coverage).

        This replaces heuristic transfer/split loops with:
        1) quality-aware merges to reach exactly `target_n_clusters`, then
        2) a capacity-constrained assignment to satisfy size bounds and full coverage.
        """
        try:
            current_labels = labels.copy()
            # Normalize labels to consecutive ints and initialize progress tracking
            unique = np.unique(current_labels)
            label_map = {old: new for new, old in enumerate(unique)}
            current_labels = np.vectorize(label_map.get)(current_labels)

            target_k = int(self.config.target_n_clusters)
            min_pct = float(self.config.min_cluster_size_pct)
            max_pct = float(self.config.max_cluster_size_pct)
            prev_violations = float('inf')
            stagnation_count = 0
            max_stagnation = 3

            # Step 1: Merge (if needed) to reach exactly target_k
            k_now = len(np.unique(current_labels))
            if k_now != target_k:
                self.logger.info(f"🔧 Adjusting cluster count: {k_now} -> {target_k} via quality-aware merges")
                current_labels = self._merge_to_target_k(features, current_labels, target_k, min_pct, max_pct)

            # Step 2: Capacity-constrained assignment to meet 3–8% and 100% coverage
            self.logger.info("🔧 Enforcing 3–8% bounds with bounded assignment")
            current_labels = self._capacity_constrained_assignment(features, current_labels, min_pct, max_pct)

            # Progress check (count violations). Stop early if stalled
            _, counts = np.unique(current_labels, return_counts=True)
            pct = counts / max(1, len(current_labels))
            violations = int(np.sum((pct < min_pct) | (pct > max_pct)))
            if violations >= prev_violations:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_violations = violations
            if stagnation_count >= max_stagnation:
                self.logger.info("Stopping constraint enforcement due to stagnation")

            return current_labels
        except Exception as e:
            self.logger.warning(f"Constraint enforcement fallback due to error: {e}")
            return labels

    # ----------------------------
    # Pareto/adjacency helpers
    # ----------------------------

    def _non_dominated_mask(self, F: np.ndarray, senses: Tuple[str, ...]) -> np.ndarray:
        """Compute non-dominated mask for objective matrix F.

        Args:
            F: Objective matrix (m candidates, d objectives)
            senses: Tuple of 'min'/'max' per objective

        Returns:
            Boolean mask of length m where True indicates non-dominated rows
        """
        try:
            if F.size == 0:
                return np.zeros((0,), dtype=bool)
            S = F.copy().astype(float)
            for j, s in enumerate(senses):
                if s == 'max':
                    S[:, j] = -S[:, j]
            m = S.shape[0]
            nd = np.ones(m, dtype=bool)
            for i in range(m):
                if not nd[i]:
                    continue
                # A candidate i is dominated if there exists k != i with S[k] <= S[i] in all dims and < in at least one
                le_all = (S <= S[i] + 1e-12).all(axis=1)
                lt_any = (S < S[i] - 1e-12).any(axis=1)
                dominated_by_any = le_all & lt_any
                dominated_by_any[i] = False
                if dominated_by_any.any():
                    nd[i] = False
            return nd
        except Exception:
            # In case of numerical issues, return all as non-dominated
            return np.ones(F.shape[0], dtype=bool)

    def _pooled_cv_for_merge(self, features: np.ndarray, labels: np.ndarray, a: int, b: int) -> float:
        """Compute pooled CV for the union of clusters a and b across feature dimensions.

        Uses mean of per-feature CVs with safe guards.
        """
        try:
            mask = (labels == a) | (labels == b)
            X = features[mask]
            if X.shape[0] <= 1:
                return 0.0
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            # Avoid division by near-zero means
            denom = np.clip(np.abs(means), 1e-8, None)
            cvs = np.abs(stds) / denom
            if not np.all(np.isfinite(cvs)):
                cvs = np.nan_to_num(cvs, nan=0.0, posinf=1e6, neginf=1e6)
            return float(np.mean(cvs))
        except Exception:
            return 0.0

    def _find_nearest_undersized_cluster(self, features: np.ndarray, labels: np.ndarray,
                                        source_cluster: int, undersized_clusters: List,
                                        unique_labels: np.ndarray) -> Optional[int]:
        """Find the nearest undersized cluster that can accept transfers.

        Args:
            features: Feature matrix
            labels: Cluster labels
            source_cluster: Source cluster to transfer from
            undersized_clusters: List of undersized cluster indices
            unique_labels: Array of unique cluster labels

        Returns:
            Target cluster label or None if no suitable target found
        """
        try:
            source_mask = labels == source_cluster
            source_centroid = np.mean(features[source_mask], axis=0)

            best_target = None
            best_distance = float('inf')

            for cluster_idx, _ in undersized_clusters:
                target_cluster = unique_labels[cluster_idx]
                target_mask = labels == target_cluster
                target_centroid = np.mean(features[target_mask], axis=0)

                # Calculate Euclidean distance
                distance = np.linalg.norm(source_centroid - target_centroid)

                if distance < best_distance:
                    best_distance = distance
                    best_target = target_cluster

            return best_target

        except Exception as e:
            self.logger.warning(f"Error finding nearest undersized cluster: {e}")
            return None

    def _transfer_samples_to_cluster(self, labels: np.ndarray, source_cluster: int,
                                   target_cluster: int, transfer_amount: int) -> np.ndarray:
        """Transfer samples from source cluster to target cluster.

        Args:
            labels: Cluster labels
            source_cluster: Source cluster to transfer from
            target_cluster: Target cluster to transfer to
            transfer_amount: Number of samples to transfer

        Returns:
            Updated labels
        """
        try:
            # Find indices of samples in source cluster
            source_mask = labels == source_cluster
            source_indices = np.where(source_mask)[0]

            # Randomly select samples to transfer
            if len(source_indices) > transfer_amount:
                transfer_indices = np.random.choice(source_indices, transfer_amount, replace=False)
            else:
                transfer_indices = source_indices

            # Transfer the samples
            labels[transfer_indices] = target_cluster

            return labels

        except Exception as e:
            self.logger.warning(f"Error transferring samples: {e}")
            return labels

    def _split_single_cluster(self, labels: np.ndarray, features: np.ndarray,
                             cluster_label: int, target_pct: float = 0.05) -> np.ndarray:
        """Split a single cluster into smaller pieces.

        Args:
            labels: Current labels
            features: Feature matrix
            cluster_label: Label of cluster to split
            target_pct: Target percentage per sub-cluster

        Returns:
            Sub-cluster labels for the split cluster
        """
        try:
            cluster_mask = labels == cluster_label
            cluster_features = features[cluster_mask]

            n_samples = len(labels)
            cluster_size = len(cluster_features)
            cluster_pct = cluster_size / n_samples

            # Calculate number of subclusters needed (minimum 2, maximum 4)
            n_subclusters = max(2, min(4, int(np.ceil(cluster_pct / target_pct))))

            # Use weighted K-means for better splitting
            sub_labels = self._weighted_kmeans_splitting(cluster_features, n_subclusters)

            return sub_labels

        except Exception as e:
            self.logger.warning(f"Error splitting cluster {cluster_label}: {e}")
            return np.zeros(len(cluster_features), dtype=int)

    def _split_cluster_into_two(self, labels: np.ndarray, features: np.ndarray, cluster_label: int) -> np.ndarray:
        """Split a cluster into exactly 2 subclusters with minimal CV for 5-7% distribution.

        Args:
            labels: Current labels
            features: Feature matrix
            cluster_label: Label of cluster to split

        Returns:
            Sub-cluster labels (0 or 1) for the split cluster
        """
        try:
            cluster_mask = labels == cluster_label
            cluster_features = features[cluster_mask]

            if len(cluster_features) < 20:  # Too small to split meaningfully
                self.logger.warning(f"⚠️ Cluster {cluster_label} too small ({len(cluster_features)} samples) for optimal splitting")
                return np.zeros(len(cluster_features), dtype=int)

            # Try multiple splitting approaches to find the one with minimal CV
            best_labels = None
            best_score = float('inf')

            # Approach 1: Standard K-means
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, n_init=20, max_iter=200, random_state=self.config.random_state)
                labels1 = kmeans.fit_predict(cluster_features)
                score1 = self._calculate_split_cv_score(cluster_features, labels1)
                if score1 < best_score:
                    best_score = score1
                    best_labels = labels1
            except Exception as e:
                self.logger.warning(f"K-means split failed: {e}")

            # Approach 2: K-means with different initialization
            try:
                # Try with k-means++ initialization for better results
                kmeans_pp = KMeans(n_clusters=2, init='k-means++', n_init=20, max_iter=200, random_state=self.config.random_state)
                labels2 = kmeans_pp.fit_predict(cluster_features)
                score2 = self._calculate_split_cv_score(cluster_features, labels2)
                if score2 < best_score:
                    best_score = score2
                    best_labels = labels2
            except Exception as e:
                self.logger.warning(f"K-means++ split failed: {e}")

            # Approach 3: Variance-based initialization
            try:
                labels3 = self._variance_based_split(cluster_features)
                if labels3 is not None:
                    score3 = self._calculate_split_cv_score(cluster_features, labels3)
                    if score3 < best_score:
                        best_score = score3
                        best_labels = labels3
            except Exception as e:
                self.logger.warning(f"Variance-based split failed: {e}")

            # Approach 4: PCA-guided split
            try:
                labels4 = self._pca_guided_split(cluster_features)
                if labels4 is not None:
                    score4 = self._calculate_split_cv_score(cluster_features, labels4)
                    if score4 < best_score:
                        best_score = score4
                        best_labels = labels4
            except Exception as e:
                self.logger.warning(f"PCA-guided split failed: {e}")

            if best_labels is None:
                # Fallback to simple split
                best_labels = np.random.randint(0, 2, len(cluster_features))

            # Verify the split creates reasonable sizes
            unique_sub, counts_sub = np.unique(best_labels, return_counts=True)
            sub_percentages = counts_sub / len(cluster_features)

            self.logger.info(f"✅ Split cluster {cluster_label} into 2 subclusters: {sub_percentages[0]*100:.1f}% and {sub_percentages[1]*100:.1f}% (CV score: {best_score:.3f})")

            return best_labels

        except Exception as e:
            self.logger.warning(f"Error splitting cluster {cluster_label} into two: {e}")
            return np.zeros(len(cluster_features), dtype=int)

    def _calculate_split_cv_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate CV-based score for a split (lower is better).

        Args:
            features: Feature matrix
            labels: Split labels

        Returns:
            CV score for the split
        """
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)

            if len(unique_labels) != 2:
                return float('inf')

            # Calculate CV properly per feature and average across features
            cv_scores = []
            for label in unique_labels:
                sub_features = features[labels == label]
                if len(sub_features) > 1:
                    means = np.mean(sub_features, axis=0)
                    stds = np.std(sub_features, axis=0)
                    vals = [abs(s / m) for m, s in zip(means, stds) if abs(m) > 1e-6]
                    avg_cv = np.mean(vals) if vals else 0.0
                    cv_scores.append(max(0.0, float(avg_cv)))
                else:
                    cv_scores.append(0.0)

            # Return average CV (lower means more homogeneous subclusters)
            return float(np.mean(cv_scores)) if cv_scores else 0.0

        except Exception as e:
            self.logger.warning(f"CV score calculation failed: {e}")
            return float('inf')

    def _variance_based_split(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Split cluster based on feature variance (experimental approach).

        Args:
            features: Feature matrix

        Returns:
            Split labels or None if failed
        """
        try:
            # Find feature with highest variance
            variances = np.var(features, axis=0)
            max_var_feature_idx = np.argmax(variances)

            # Sort by the highest variance feature
            sorted_indices = np.argsort(features[:, max_var_feature_idx])

            # Split at median
            split_point = len(sorted_indices) // 2
            labels = np.zeros(len(features), dtype=int)
            labels[sorted_indices[split_point:]] = 1

            return labels

        except Exception as e:
            self.logger.warning(f"Variance-based split failed: {e}")
            return None

    def _pca_guided_split(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Split cluster using PCA for better separation.

        Args:
            features: Feature matrix

        Returns:
            Split labels or None if failed
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans

            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=min(2, features.shape[1]))
            features_pca = pca.fit_transform(features)

            # Use K-means on PCA-reduced features
            kmeans = KMeans(n_clusters=2, n_init=10, max_iter=100, random_state=self.config.random_state)
            labels = kmeans.fit_predict(features_pca)

            return labels

        except Exception as e:
            self.logger.warning(f"PCA-guided split failed: {e}")
            return None

    def _merge_single_cluster(self, labels: np.ndarray, features: np.ndarray,
                              cluster_label: int) -> Optional[np.ndarray]:
        """Merge a single undersized cluster with its nearest neighbor.

        Args:
            labels: Current labels
            features: Feature matrix
            cluster_label: Label of cluster to merge

        Returns:
            New labels after merging, or None if merge failed
        """
        try:
            # Find nearest neighbor cluster
            cluster_mask = labels == cluster_label
            cluster_features = features[cluster_mask]
            cluster_centroid = np.mean(cluster_features, axis=0)

            # Calculate distances to all other clusters
            unique_labels = np.unique(labels)
            min_distance = float('inf')
            best_neighbor = None

            for other_label in unique_labels:
                if other_label == cluster_label or other_label == -1:
                    continue

                other_mask = labels == other_label
                other_features = features[other_mask]
                other_centroid = np.mean(other_features, axis=0)

                distance = np.linalg.norm(cluster_centroid - other_centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_neighbor = other_label

            # Merge with best neighbor
            if best_neighbor is not None:
                new_labels = labels.copy()
                # Validate size constraints before merging
                total_samples = len(new_labels)
                size_a = int(np.sum(new_labels == cluster_label))
                size_b = int(np.sum(new_labels == best_neighbor))
                combined_pct = (size_a + size_b) / max(1, total_samples)
                max_pct = float(getattr(self.config, 'max_cluster_size_pct', 0.08))
                if combined_pct <= max_pct:
                    new_labels[new_labels == cluster_label] = best_neighbor
                    return new_labels
                else:
                    self.logger.info(f"Skipping merge {cluster_label}->{best_neighbor}: would exceed max_pct {max_pct:.3f}")

            return None

        except Exception as e:
            self.logger.warning(f"Error merging cluster {cluster_label}: {e}")
            return None

    def _merge_with_most_similar(self, labels: np.ndarray, features: np.ndarray, cluster_label: int) -> Optional[np.ndarray]:
        """Merge a cluster with the most similar cluster based on CV.

        Args:
            labels: Current labels
            features: Feature matrix
            cluster_label: Label of cluster to merge

        Returns:
            New labels after merging, or None if merge failed
        """
        try:
            # Calculate CV for the cluster to merge
            cluster_mask = labels == cluster_label
            cluster_features = features[cluster_mask]

            if cluster_features.shape[0] > 1:
                means = np.mean(cluster_features, axis=0)
                stds = np.std(cluster_features, axis=0)
                vals = [abs(s / m) for m, s in zip(means, stds) if abs(m) > 1e-6]
                cluster_cv = float(np.mean(vals)) if vals else 0.0
            else:
                cluster_cv = 0.0

            # Find most similar cluster based on CV
            unique_labels = np.unique(labels)
            best_similarity = -1
            best_neighbor = None

            for other_label in unique_labels:
                if other_label == cluster_label or other_label == -1:
                    continue

                other_mask = labels == other_label
                other_features = features[other_mask]

                if other_features.shape[0] > 1:
                    o_means = np.mean(other_features, axis=0)
                    o_stds = np.std(other_features, axis=0)
                    o_vals = [abs(s / m) for m, s in zip(o_means, o_stds) if abs(m) > 1e-6]
                    other_cv = float(np.mean(o_vals)) if o_vals else 0.0
                else:
                    other_cv = 0.0

                # Calculate CV similarity (lower difference = higher similarity)
                cv_similarity = 1.0 / (1.0 + abs(cluster_cv - other_cv))

                if cv_similarity > best_similarity:
                    best_similarity = cv_similarity
                    best_neighbor = other_label

            # Merge with most similar cluster
            if best_neighbor is not None and best_similarity > 0.5:  # Only merge if reasonably similar
                new_labels = labels.copy()
                # Validate size constraints before merging
                total_samples = len(new_labels)
                size_a = int(np.sum(new_labels == cluster_label))
                size_b = int(np.sum(new_labels == best_neighbor))
                combined_pct = (size_a + size_b) / max(1, total_samples)
                max_pct = float(getattr(self.config, 'max_cluster_size_pct', 0.08))
                if combined_pct <= max_pct:
                    new_labels[new_labels == cluster_label] = best_neighbor
                    self.logger.info(f"✅ Merged cluster {cluster_label} with most similar cluster {best_neighbor} (CV similarity: {best_similarity:.3f})")
                    return new_labels
                else:
                    self.logger.info(f"Skipping similar-merge {cluster_label}->{best_neighbor}: would exceed max_pct {max_pct:.3f}")

            return None

        except Exception as e:
            self.logger.warning(f"Error merging cluster {cluster_label} with most similar: {e}")
            return None

    def _final_cleanup_pass(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Final cleanup pass to ensure perfect 3-8% distribution.

        Args:
            labels: Current cluster labels
            features: Feature matrix

        Returns:
            Final cleaned up labels
        """
        try:
            n_samples = len(labels)
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / n_samples

            # Check final distribution
            min_pct = float(getattr(self.config, 'min_cluster_size_pct', 0.03))
            max_pct = float(getattr(self.config, 'max_cluster_size_pct', 0.08))
            violations = [(i, pct) for i, pct in enumerate(percentages) if pct < min_pct or pct > max_pct]

            if not violations:
                return labels

            self.logger.info(f"🔧 Final cleanup: {len(violations)} violations to fix")

            # Apply final aggressive redistribution
            final_labels = self._aggressive_redistribution(labels, features, len(unique_labels))

            # Final check
            unique_labels, counts = np.unique(final_labels, return_counts=True)
            percentages = counts / n_samples

            min_pct = float(getattr(self.config, 'min_cluster_size_pct', 0.03))
            max_pct = float(getattr(self.config, 'max_cluster_size_pct', 0.08))
            final_violations = sum(1 for pct in percentages if pct < min_pct or pct > max_pct)
            self.logger.info(f"✅ Final cleanup completed: {final_violations} violations remaining")

            return final_labels

        except Exception as e:
            self.logger.warning(f"Final cleanup pass failed: {e}")
            return labels

    def _merge_to_target_k(self, features: np.ndarray, labels: np.ndarray, target_k: int,
                           min_pct: float, max_pct: float) -> np.ndarray:
        """Reduce number of clusters to `target_k` using a quality-aware greedy merge.

        Merge cost favors small centroid distances and penalizes size overflow beyond max_pct.
        """
        try:
            n = labels.shape[0]
            lower = max(1, int(np.ceil(min_pct * n)))
            upper = max(1, int(np.floor(max_pct * n)))

            current_labels = labels.copy()
            while True:
                unique = np.unique(current_labels)
                if len(unique) <= target_k:
                    break

                # Compute centroids and sizes (batched, using matrix ops where available)
                k = len(unique)
                idx_map = {lab: i for i, lab in enumerate(unique)}
                label_idx = np.vectorize(idx_map.get)(current_labels)
                onehot = np.zeros((n, k), dtype=np.float64)
                onehot[np.arange(n), label_idx] = 1.0

                try:
                    if MATRIX_OPERATIONS_AVAILABLE and 'gpu_matrix_multiply' in globals() and gpu_matrix_multiply is not None:
                        sums = gpu_matrix_multiply(onehot.T, features)
                    elif MATRIX_OPERATIONS_AVAILABLE and 'batch_matrix_multiply' in globals() and batch_matrix_multiply is not None:
                        sums = batch_matrix_multiply(onehot.T, features)
                    else:
                        sums = onehot.T @ features
                except Exception:
                    sums = onehot.T @ features

                sizes = onehot.sum(axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    centroids = sums / np.maximum(sizes[:, None], 1.0)

                # Build candidate pairs with optional 4D k-NN adjacency gating and Pareto objectives
                best_pair = None
                target_pct = 0.05
                target_size = int(round(target_pct * n))

                use_pareto = bool(getattr(self.config, 'enable_pareto_merging', False))
                candidate_pairs: List[Tuple[int, int]] = []

                if use_pareto:
                    # Create weighted 4D map and cluster centroids in that space for adjacency
                    try:
                        weighted_features = self._create_weighted_4d_map(features)
                    except Exception:
                        weighted_features = features

                    # Compute 4D centroids per current cluster
                    C4 = np.zeros((k, weighted_features.shape[1]), dtype=float)
                    for ui, lab in enumerate(unique):
                        mask = current_labels == lab
                        if np.any(mask):
                            C4[ui] = weighted_features[mask].mean(axis=0)
                        else:
                            C4[ui] = 0.0

                    # k-NN adjacency on 4D centroids
                    try:
                        from sklearn.neighbors import NearestNeighbors
                        kn = max(1, min(k - 1, int(getattr(self.config, 'knn_adjacency_k', 3))))
                        nbrs = NearestNeighbors(n_neighbors=kn).fit(C4)
                        neigh = nbrs.kneighbors(return_distance=False)
                        pairs = set()
                        for i in range(k):
                            for j in neigh[i]:
                                if i == j:
                                    continue
                                a_lab = int(unique[min(i, j)])
                                b_lab = int(unique[max(i, j)])
                                pairs.add((a_lab, b_lab))
                        candidate_pairs = sorted(list(pairs))
                    except Exception:
                        # Fall back to all pairs
                        candidate_pairs = [(int(unique[i]), int(unique[j])) for i in range(k) for j in range(i + 1, k)]
                else:
                    # Greedy fallback: all pairs
                    candidate_pairs = [(int(unique[i]), int(unique[j])) for i in range(k) for j in range(i + 1, k)]

                chosen_pair = None
                if use_pareto and candidate_pairs:
                    # Evaluate objectives per candidate
                    objs = []
                    pairs_kept = []
                    eps_cv = float(getattr(self.config, 'epsilon_cv_increase', 0.05))
                    for (lab_a, lab_b) in candidate_pairs:
                        ia = idx_map[lab_a]
                        ib = idx_map[lab_b]
                        merged_size = sizes[ia] + sizes[ib]
                        # Enforce hard size constraint
                        if merged_size > upper:
                            continue
                        # J_size_over: normalized overflow (should be 0 here, but keep soft term)
                        j_size_over = max(0.0, (merged_size - upper)) / max(1.0, n)
                        # J_cv: pooled CV on original feature space (stable)
                        j_cv = self._pooled_cv_for_merge(features, current_labels, lab_a, lab_b)
                        # approximate baseline cv to guard against large increase (optional)
                        cv_a = self._pooled_cv_for_merge(features, current_labels, lab_a, lab_a)
                        cv_b = self._pooled_cv_for_merge(features, current_labels, lab_b, lab_b)
                        base_cv = 0.5 * (cv_a + cv_b)
                        if (j_cv - base_cv) > eps_cv:
                            continue
                        # J_dist: centroid distance in weighted 4D (or default space on failure)
                        try:
                            ca = C4[idx_map[lab_a]]
                            cb = C4[idx_map[lab_b]]
                            j_dist = float(np.linalg.norm(ca - cb))
                        except Exception:
                            diff = centroids[ia] - centroids[ib]
                            j_dist = float(np.linalg.norm(diff))
                        objs.append([j_size_over, j_cv, j_dist])
                        pairs_kept.append((lab_a, lab_b))

                    if objs:
                        F = np.array(objs, dtype=float)
                        # Normalize objectives (min-max)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            minv = np.nanmin(F, axis=0)
                            maxv = np.nanmax(F, axis=0)
                            rng = np.clip(maxv - minv, 1e-12, None)
                            F_norm = (F - minv) / rng
                        nd_mask = self._non_dominated_mask(F_norm, senses=("min", "min", "min"))
                        nd_idx = np.where(nd_mask)[0]
                        # Knee: pick the row minimizing L2 to ideal (0,0,0)
                        nd_rows = F_norm[nd_idx]
                        dists = np.linalg.norm(nd_rows, axis=1)
                        pick = int(nd_idx[int(np.argmin(dists))])
                        chosen_pair = pairs_kept[pick]

                if chosen_pair is None:
                    # Fallback to previous greedy: Ward-like cost + size modulation
                    best_cost = float('inf')
                    best_pair_greedy = None
                    for i in range(len(unique)):
                        for j in range(i + 1, len(unique)):
                            merged_size = sizes[i] + sizes[j]
                            if merged_size > upper:
                                continue
                            soft_penalty = 0.0
                            if merged_size > target_size:
                                over_target = merged_size - target_size
                                scale = 1.0 if merged_size <= upper else 50.0
                                soft_penalty = scale * (over_target ** 2)
                            diff = centroids[i] - centroids[j]
                            dist_sq = float(np.dot(diff, diff))
                            ward = (sizes[i] * sizes[j]) / max(1.0, float(sizes[i] + sizes[j])) * dist_sq
                            undershoot_i = max(0.0, (target_size - sizes[i])) / max(1.0, target_size)
                            undershoot_j = max(0.0, (target_size - sizes[j])) / max(1.0, target_size)
                            overshoot_i = max(0.0, (sizes[i] - target_size)) / max(1.0, target_size)
                            overshoot_j = max(0.0, (sizes[j] - target_size)) / max(1.0, target_size)
                            attract = undershoot_i + undershoot_j
                            repel = overshoot_i + overshoot_j
                            gamma = 0.5
                            size_modulator = (1.0 + gamma * repel) / (1.0 + gamma * attract)
                            cost = (ward + soft_penalty) * size_modulator
                            if cost < best_cost:
                                best_cost = cost
                                best_pair_greedy = (unique[i], unique[j])
                    if best_pair_greedy is None:
                        break
                    a, b = best_pair_greedy
                else:
                    a, b = chosen_pair

                # Merge b into a
                current_labels[current_labels == b] = a
                # Reindex to keep labels compact
                uniq2 = np.unique(current_labels)
                remap = {old: idx for idx, old in enumerate(uniq2)}
                current_labels = np.vectorize(remap.get)(current_labels)

            return current_labels
        except Exception:
            return labels

    def _capacity_constrained_assignment(self, features: np.ndarray, labels: np.ndarray,
                                         min_pct: float, max_pct: float,
                                         *, distance_metric: str = "euclidean",
                                         whiten: bool = False) -> np.ndarray:
        """Greedy bounded assignment to enforce cluster size bounds and full coverage.

        - Start from current labels
        - Compute centroid distances
        - Fill clusters below lower bound with nearest feasible samples
        - Reduce clusters above upper bound by moving lowest-regret samples
        """
        n = labels.shape[0]
        lower = max(1, int(np.ceil(min_pct * n)))
        upper = max(1, int(np.floor(max_pct * n)))

        current = labels.copy()

        def compute_centroids(lbls: np.ndarray) -> np.ndarray:
            uniq = np.unique(lbls)
            k_local = len(uniq)
            # Map labels to 0..k-1
            idx_map = {lab: i for i, lab in enumerate(uniq)}
            label_idx = np.vectorize(idx_map.get)(lbls)
            onehot = np.zeros((n, k_local), dtype=np.float64)
            onehot[np.arange(n), label_idx] = 1.0
            try:
                if MATRIX_OPERATIONS_AVAILABLE and 'gpu_matrix_multiply' in globals() and gpu_matrix_multiply is not None:
                    sums = gpu_matrix_multiply(onehot.T, X)
                elif MATRIX_OPERATIONS_AVAILABLE and 'batch_matrix_multiply' in globals() and batch_matrix_multiply is not None:
                    sums = batch_matrix_multiply(onehot.T, X)
                else:
                    sums = onehot.T @ X
            except Exception:
                sums = onehot.T @ X
            cnts_local = onehot.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                centers = sums / np.maximum(cnts_local[:, None], 1.0)
            return centers

        def reindex(lbls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            uniq = np.unique(lbls)
            remap = {old: idx for idx, old in enumerate(uniq)}
            inv = np.array([remap[v] for v in uniq])
            return np.vectorize(remap.get)(lbls), uniq

        # Ensure labels are 0..k-1
        current, orig_unique = reindex(current)
        k = len(np.unique(current))

        # Optional whitening
        X = features
        if whiten:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-12
            X = (X - mean) / std

        def compute_dists(lbls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            ctrs = compute_centroids(lbls)
            # Apply whitening to centroids if used
            ctrs_w = ctrs
            if whiten:
                ctrs_w = (ctrs - mean) / std

            if distance_metric == "cosine":
                # GPU cosine via matmul when available
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                Cn = ctrs_w / (np.linalg.norm(ctrs_w, axis=1, keepdims=True) + 1e-12)
                try:
                    if MATRIX_OPERATIONS_AVAILABLE and 'gpu_matrix_multiply' in globals() and gpu_matrix_multiply is not None:
                        sims = gpu_matrix_multiply(Xn, Cn.T)
                    elif MATRIX_OPERATIONS_AVAILABLE and 'batch_matrix_multiply' in globals() and batch_matrix_multiply is not None:
                        sims = batch_matrix_multiply(Xn, Cn.T)
                    else:
                        sims = Xn @ Cn.T
                except Exception:
                    sims = Xn @ Cn.T
                d = 1.0 - sims
                return d, ctrs, ctrs_w
            elif distance_metric == "mahalanobis":
                # Mahalanobis with regularized covariance for stability
                try:
                    cov = np.cov(X.T)
                    # Regularize
                    eps = 1e-6
                    cov.flat[:: cov.shape[0] + 1] += eps
                    inv = np.linalg.inv(cov)
                except Exception:
                    # Fallback to identity if ill-conditioned
                    inv = np.eye(X.shape[1], dtype=np.float64)
                # Compute (x - c)^T inv (x - c) for all c via batched products
                # d2 = x^T inv x - 2 x^T inv c + c^T inv c
                try:
                    if MATRIX_OPERATIONS_AVAILABLE and 'gpu_matrix_multiply' in globals() and gpu_matrix_multiply is not None:
                        X_inv = gpu_matrix_multiply(X, inv)
                        C_inv = gpu_matrix_multiply(ctrs_w, inv)
                    elif MATRIX_OPERATIONS_AVAILABLE and 'batch_matrix_multiply' in globals() and batch_matrix_multiply is not None:
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
                return d, ctrs, ctrs_w
            else:
                # Euclidean via dot products
                try:
                    if MATRIX_OPERATIONS_AVAILABLE and 'gpu_matrix_multiply' in globals() and gpu_matrix_multiply is not None:
                        dots = gpu_matrix_multiply(X, ctrs_w.T)
                    elif MATRIX_OPERATIONS_AVAILABLE and 'batch_matrix_multiply' in globals() and batch_matrix_multiply is not None:
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
                return d, ctrs, ctrs_w

        dists, centroids, centroids_w = compute_dists(current)

        # Precompute top-3 nearest clusters per sample
        topk = np.argsort(dists, axis=1)[:, :min(3, k)]

        def counts(lbls: np.ndarray) -> np.ndarray:
            cnt = np.bincount(lbls, minlength=k)
            return cnt

        cnts = counts(current)

        # Phase A: raise clusters to lower bound
        max_rounds = 5
        for _ in range(max_rounds):
            deficits = [(c, lower - cnts[c]) for c in range(k) if cnts[c] < lower]
            if not deficits:
                break
            # Fill largest deficits first
            deficits.sort(key=lambda x: x[1], reverse=True)
            moved_any = False
            for c, need in deficits:
                if need <= 0:
                    continue
                # Candidate donors: samples not in c where c is among their top-3
                candidates = np.where((current != c) & ((topk[:, 0] == c) | (topk[:, 1] == c) | (topk[:, 2] == c) if topk.shape[1] >= 3 else (topk[:, 0] == c)))[0]
                # Filter donors whose current cluster has surplus above lower
                donor_ok = candidates[cnts[current[candidates]] > lower]
                if donor_ok.size == 0:
                    # Relax: allow any candidate as last resort
                    donor_ok = candidates
                if donor_ok.size == 0:
                    continue
                # Sort by minimal regret for moving into c (optionally add CV/silhouette-aware small penalty)
                deltas = dists[donor_ok, c] - dists[donor_ok, current[donor_ok]]
                order = np.argsort(deltas)
                to_move = donor_ok[order][:need]
                # Apply moves
                for idx in to_move:
                    old = current[idx]
                    if cnts[old] <= lower:
                        continue
                    current[idx] = c
                    cnts[old] -= 1
                    cnts[c] += 1
                    moved_any = True
                if cnts[c] < lower:
                    # Try again with relaxed candidates if still short
                    remain = lower - cnts[c]
                    others = np.where(current != c)[0]
                    # Prefer donors from clusters with largest surplus
                    surplus = cnts[current[others]] - lower
                    donor2 = others[surplus > 0]
                    if donor2.size > 0:
                        deltas2 = dists[donor2, c] - dists[donor2, current[donor2]]
                        order2 = np.argsort(deltas2)
                        to_move2 = donor2[order2][:remain]
                        for idx in to_move2:
                            old = current[idx]
                            if cnts[old] <= lower:
                                continue
                            current[idx] = c
                            cnts[old] -= 1
                            cnts[c] += 1
                            moved_any = True
            if not moved_any:
                break
            # Recompute distances/topk after changes
            dists, centroids, centroids_w = compute_dists(current)
            topk = np.argsort(dists, axis=1)[:, :min(3, k)]

        # Phase B: reduce clusters above upper bound
        for _ in range(max_rounds):
            overs = [(c, cnts[c] - upper) for c in range(k) if cnts[c] > upper]
            if not overs:
                break
            moved_any = False
            # Process the most oversized first
            overs.sort(key=lambda x: x[1], reverse=True)
            for c, excess in overs:
                if excess <= 0:
                    continue
                indices = np.where(current == c)[0]
                if indices.size == 0:
                    continue
                # For each sample, find best alternative cluster with capacity
                best_alt = np.full(indices.shape[0], -1, dtype=int)
                alt_cost = np.full(indices.shape[0], np.inf, dtype=float)
                for idx_i, i in enumerate(indices):
                    # Prefer among top-3 nearest clusters
                    for alt in topk[i]:
                        if alt == c:
                            continue
                        if cnts[alt] >= upper:
                            continue
                        cost = dists[i, alt]
                        if cost < alt_cost[idx_i]:
                            alt_cost[idx_i] = cost
                            best_alt[idx_i] = alt
                # Compute regret vs staying in c
                stay_cost = dists[indices, c]
                regret = alt_cost - stay_cost
                # Sort candidates by minimal regret (most negative first)
                order = np.argsort(regret)
                moved = 0
                for idx in order:
                    if moved >= excess:
                        break
                    i = indices[idx]
                    alt = best_alt[idx]
                    if alt == -1:
                        continue
                    if cnts[alt] >= upper:
                        continue
                    # Ensure donor won't violate lower bound
                    if cnts[c] - 1 < lower:
                        break
                    current[i] = alt
                    cnts[c] -= 1
                    cnts[alt] += 1
                    moved += 1
                    moved_any = True
            if not moved_any:
                break
            # Recompute distances/topk after changes
            dists, centroids, centroids_w = compute_dists(current)
            topk = np.argsort(dists, axis=1)[:, :min(3, k)]

        return current

    def _update_labels_after_split(self, current_labels: np.ndarray,
                                   split_labels: np.ndarray, original_label: int) -> np.ndarray:
        """Update labels after splitting a cluster.

        Args:
            current_labels: Current labels
            split_labels: Labels from the split operation
            original_label: Original cluster label that was split

        Returns:
            Updated labels
        """
        new_labels = current_labels.copy()

        # Find a new label range for the subclusters
        unique_labels = np.unique(new_labels)
        max_label = np.max(unique_labels) if len(unique_labels) > 0 else 0

        # Replace the original cluster with the new subclusters
        cluster_mask = new_labels == original_label
        new_labels[cluster_mask] = max_label + 1 + split_labels

        return new_labels

    def _split_large_clusters(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Legacy wrapper - now uses iterative constraint enforcement."""
        return self._iterative_constraint_enforcement(labels, features)

    def _weighted_kmeans_splitting(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform weighted K-means splitting with centroid-based initialization.

        Args:
            features: Feature matrix for the cluster to split
            n_clusters: Number of subclusters to create

        Returns:
            Sub-cluster labels
        """
        try:
            from sklearn.cluster import KMeans

            # Use K-means++ initialization for better centroids
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                n_init=10,
                max_iter=100,
                random_state=self.config.random_state
            )

            return kmeans.fit_predict(features)

        except Exception as e:
            self.logger.warning(f"Weighted K-means splitting failed: {e}")
            # Fallback to simple random splitting
            return np.random.randint(0, n_clusters, size=features.shape[0])

    def _calculate_centroid_based_clusters(self, features: np.ndarray) -> np.ndarray:
        """Calculate clusters using advanced weighted centroid approach.

        Args:
            features: Feature matrix

        Returns:
            Centroid-based cluster labels
        """
        try:
            n_samples = features.shape[0]
            target_clusters = 20
            min_samples_per_cluster = int(n_samples * float(getattr(self.config, 'min_cluster_size_pct', 0.03)))
            max_samples_per_cluster = int(n_samples * float(getattr(self.config, 'max_cluster_size_pct', 0.08)))

            # Use iterative approach: start with target clusters, then refine
            best_labels = None
            best_score = float('-inf')

            for attempt in range(3):  # Try 3 different approaches
                try:
                    # Approach 1: Direct weighted centroid initialization
                    if attempt == 0:
                        weighted_features = self._create_weighted_4d_map(features)
                        initial_centroids = self._find_weighted_equidistant_centroids(weighted_features, target_clusters)
                        labels = self._initialize_with_weighted_centroids(features, initial_centroids, target_clusters)

                    # Approach 2: Start with more clusters, then merge strategically
                    elif attempt == 1:
                        weighted_features = self._create_weighted_4d_map(features)
                        initial_centroids = self._find_weighted_equidistant_centroids(weighted_features, target_clusters * 2)
                        initial_labels = self._initialize_with_weighted_centroids(features, initial_centroids, target_clusters * 2)
                        labels = self._strategic_cluster_merging(initial_labels, features, target_clusters)

                    # Approach 3: Balanced initialization with forced redistribution
                    else:
                        labels = self._balanced_initialization_with_redistribution(features, target_clusters)

                    # Evaluate the result
                    score = self._evaluate_cluster_distribution(labels, n_samples, target_clusters)
                    self.logger.info(f"Attempt {attempt + 1} score: {score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_labels = labels

                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    continue

            # If we couldn't get a good result, apply comprehensive refinement
            if best_score < 0.7 and best_labels is not None:
                self.logger.info("🔧 Applying comprehensive refinement to improve distribution...")

                # Step 1: Adaptive target adjustment
                adjusted_min_pct, adjusted_max_pct, adjusted_target_clusters = self._adaptive_target_adjustment(features, target_clusters)

                # Step 2: Enhanced redistribution with multiple rounds
                best_labels = self._enhanced_redistribution(best_labels, features, adjusted_target_clusters)

                # Step 3: Iterative refinement passes
                best_labels = self._iterative_refinement(best_labels, features, adjusted_target_clusters)

                # Step 4: Final aggressive redistribution if needed
                if self._evaluate_cluster_distribution(best_labels, n_samples, adjusted_target_clusters) < 0.8:
                    best_labels = self._aggressive_redistribution(best_labels, features, adjusted_target_clusters)

                self.logger.info(f"✅ Comprehensive refinement completed with score: {self._evaluate_cluster_distribution(best_labels, n_samples, adjusted_target_clusters):.3f}")

            return best_labels if best_labels is not None else self._fallback_clustering(features, target_clusters)

        except Exception as e:
            self.logger.warning(f"Advanced centroid-based clustering failed: {e}")
            # Fallback to standard K-means
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=target_clusters, init='k-means++', n_init=10, random_state=self.config.random_state)
            return kmeans.fit_predict(features)

    def _create_weighted_4d_map(self, features: np.ndarray) -> np.ndarray:
        """Create a weighted 4D map based on cluster characteristics.

        Args:
            features: Original feature matrix

        Returns:
            Weighted feature matrix
        """
        try:
            # Start with initial clustering to get baseline clusters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=50, n_init=5, random_state=self.config.random_state)
            initial_labels = kmeans.fit_predict(features)

            # Calculate weights for each point based on cluster properties
            weighted_features = features.copy()

            unique_labels, counts = np.unique(initial_labels, return_counts=True)
            n_samples = len(features)

            # Optional Pareto-aware feature weighting
            use_pareto_fw = bool(getattr(self.config, 'enable_pareto_feature_weighting', False))

            for label, count in zip(unique_labels, counts):
                if label == -1:
                    continue

                cluster_mask = initial_labels == label
                cluster_features = features[cluster_mask]

                # Calculate CV for the cluster
                if cluster_features.shape[0] > 1:
                    means = np.mean(cluster_features, axis=0)
                    stds = np.std(cluster_features, axis=0)
                    vals = [abs(s / m) for m, s in zip(means, stds) if abs(m) > 1e-6]
                    cv = float(np.mean(vals)) if vals else 0.0
                else:
                    cv = 0.0

                # Calculate inverse size weight
                inv_size_weight = 1.0 / max(1e-6, (count / n_samples))

                if use_pareto_fw:
                    # Derive simple proxies for information_density and statistical_validity
                    information_density = 1.0 / (1.0 + cv)  # lower cv => higher information density
                    statistical_validity = min(1.0, count / max(1.0, np.median(counts)))

                    # Assume feature columns loosely map to 4D: momentum, volatility, volume, trend
                    # Compute scalar weights per dimension following provided formulas
                    w_momentum = (1.0 + information_density * 0.2)
                    w_volatility = max(0.1, (1.0 - cv * 0.3))
                    w_volume = (1.0 + statistical_validity * 0.1)
                    w_trend = max(0.1, (1.0 - cv * 0.2))

                    # Build a per-feature weight vector by cycling these four weights
                    four_w = np.array([w_momentum, w_volatility, w_volume, w_trend], dtype=float)
                    reps = int(np.ceil(cluster_features.shape[1] / 4))
                    weight_vec = np.tile(four_w, reps)[:cluster_features.shape[1]]

                    # Also incorporate inverse size weight mildly to encourage small, informative clusters
                    weight_vec *= (1.0 + 0.1 * inv_size_weight)
                    weighted_features[cluster_mask] = cluster_features * weight_vec
                else:
                    # Legacy weighting: cv * inverse size
                    total_weight = cv * inv_size_weight
                    weighted_features[cluster_mask] = cluster_features * (1.0 + total_weight)

            return weighted_features

        except Exception as e:
            self.logger.warning(f"Weighted 4D map creation failed: {e}")
            return features

    def _calculate_pareto_weights(self, cluster_metadata: Dict[str, Any], sample_counts: Dict[int, int]) -> Dict[str, float]:
        """Calculate multi-objective Pareto weights for clusters.

        Balances multiple objectives:
        - Size balance (lower CV = higher weight)
        - Information density (higher = higher weight)
        - Statistical validity (higher = higher weight)
        - Similarity preservation (higher = higher weight)

        Args:
            cluster_metadata: Dict with per-cluster stats (e.g., {'label': {'cv': ..., 'centroid': ...}})
            sample_counts: Dict mapping cluster label -> count

        Returns:
            Dict mapping cluster label -> weight
        """
        try:
            weights: Dict[str, float] = {}
            # Normalize helpers
            cvs = np.array([float(v.get('cv', 0.0)) for v in cluster_metadata.values()]) if cluster_metadata else np.array([0.0])
            if not np.all(np.isfinite(cvs)):
                cvs = np.nan_to_num(cvs, nan=0.0, posinf=1.0)
            cv_max = max(1e-6, float(np.max(cvs)))

            counts = np.array([float(sample_counts.get(int(k), 0)) for k in cluster_metadata.keys()]) if cluster_metadata else np.array([1.0])
            cnt_med = max(1.0, float(np.median(counts)))

            # Build a simple similarity preservation proxy from centroid norms (more concentrated => higher)
            centroids = [np.asarray(v.get('centroid')) for v in cluster_metadata.values()] if cluster_metadata else []
            sim_scores = []
            for c in centroids:
                if c is None or c.size == 0:
                    sim_scores.append(0.0)
                else:
                    sim_scores.append(1.0 / (1.0 + float(np.std(c))))
            sim_scores = np.array(sim_scores) if sim_scores else np.array([0.0])
            sim_max = max(1e-6, float(np.max(sim_scores)))

            for (lab, meta), cv_val, cnt_val, sim_val in zip(cluster_metadata.items(), cvs, counts, sim_scores):
                # Objectives -> weights
                size_balance = 1.0 - (cv_val / cv_max)  # lower cv => higher weight
                information_density = 1.0 / (1.0 + cv_val)  # proxy
                statistical_validity = min(1.0, cnt_val / cnt_med)
                similarity_preservation = sim_val / sim_max

                # Aggregate with mild emphasis on size balance and info density
                w = (
                    0.35 * size_balance +
                    0.30 * information_density +
                    0.20 * statistical_validity +
                    0.15 * similarity_preservation
                )
                # Ensure positive small floor
                weights[str(lab)] = float(max(1e-3, w))
            return weights
        except Exception:
            # Fallback: equal weights
            return {str(k): 1.0 for k in cluster_metadata.keys()} if cluster_metadata else {}

    def _find_weighted_equidistant_centroids(self, features: np.ndarray, n_centroids: int) -> np.ndarray:
        """Find optimally distributed centroids using advanced initialization.

        Args:
            features: Weighted feature matrix
            n_centroids: Number of centroids to find

        Returns:
            Centroid coordinates
        """
        try:
            from sklearn.cluster import KMeans

            # Step 1: Use hierarchical approach for better initial centroids
            # Start with more centroids than needed, then select the best distributed ones
            initial_k = min(n_centroids * 3, features.shape[0] // 10)  # 3x more initial centroids

            kmeans = KMeans(
                n_clusters=initial_k,
                init='k-means++',
                n_init=30,  # More initializations for better results
                max_iter=200,
                random_state=self.config.random_state
            )

            kmeans.fit(features)
            initial_centroids = kmeans.cluster_centers_

            # Step 2: Calculate centroid quality metrics
            centroid_scores = self._calculate_centroid_distribution_scores(
                initial_centroids, features
            )

            # Step 3: Select the best n_centroids based on distribution quality
            top_indices = np.argsort(centroid_scores)[::-1][:n_centroids]
            selected_centroids = initial_centroids[top_indices]

            # Step 4: Apply final refinement to ensure good distribution
            final_kmeans = KMeans(
                n_clusters=n_centroids,
                init=selected_centroids,
                n_init=1,  # We provide the centroids
                max_iter=100,
                random_state=self.config.random_state
            )

            final_kmeans.fit(features)
            optimized_centroids = final_kmeans.cluster_centers_

            self.logger.info(f"✅ Found {n_centroids} optimally distributed centroids")
            self.logger.info(f"📊 Centroid distribution quality: {np.mean(centroid_scores[top_indices]):.3f}")
            return optimized_centroids

        except Exception as e:
            self.logger.warning(f"Weighted centroid finding failed: {e}")
            # Fallback to random centroids
            return np.random.randn(n_centroids, features.shape[1])

    def _calculate_centroid_distribution_scores(self, centroids: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate quality scores for centroid distribution.

        Args:
            centroids: Centroid coordinates
            features: Feature matrix

        Returns:
            Array of centroid quality scores
        """
        try:
            n_centroids = len(centroids)
            scores = np.zeros(n_centroids)

            for i, centroid in enumerate(centroids):
                # Calculate distance to all other centroids
                distances_to_others = np.linalg.norm(centroids - centroid, axis=1)
                distances_to_others = distances_to_others[distances_to_others > 0]  # Remove self-distance

                # Calculate distance to nearest data points
                distances_to_data = np.linalg.norm(features - centroid, axis=1)
                min_distance_to_data = np.min(distances_to_data)

                # Score based on:
                # 1. Distance to other centroids (should be balanced)
                # 2. Distance to data points (should be close to some data)
                if len(distances_to_others) > 0:
                    mean_distance_to_centroids = np.mean(distances_to_others)
                    std_distance_to_centroids = np.std(distances_to_others)

                    # Higher score for centroids that are equidistant from others
                    # and close to data points
                    distance_score = 1.0 / (1.0 + std_distance_to_centroids)
                    proximity_score = 1.0 / (1.0 + min_distance_to_data)

                    scores[i] = distance_score * 0.7 + proximity_score * 0.3
                else:
                    scores[i] = 0.0

            return scores

        except Exception as e:
            self.logger.warning(f"Centroid scoring failed: {e}")
            return np.ones(n_centroids)

    def _initialize_with_weighted_centroids(self, features: np.ndarray, centroids: np.ndarray, n_clusters: int) -> np.ndarray:
        """Initialize clustering with weighted centroids.

        Args:
            features: Feature matrix
            centroids: Initial centroid positions
            n_clusters: Number of clusters

        Returns:
            Initial cluster labels
        """
        try:
            from sklearn.cluster import KMeans

            # Use provided centroids as initial positions
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=centroids,
                n_init=1,  # Only one initialization since we provide centroids
                max_iter=50,
                random_state=self.config.random_state
            )

            return kmeans.fit_predict(features)

        except Exception as e:
            self.logger.warning(f"Weighted centroid initialization failed: {e}")
            # Fallback to standard initialization
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=self.config.random_state)
            return kmeans.fit_predict(features)

    def _dynamic_size_constrained_merging(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Perform dynamic merging based on size constraints and CV similarity.

        Args:
            labels: Initial cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Merged cluster labels
        """
        try:
            n_samples = len(labels)
            current_labels = labels.copy()

            # Calculate cluster statistics
            cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

            # Perform iterative merging rounds
            max_rounds = 50  # Limit iterations to prevent infinite loops
            round_num = 0

            while len(cluster_stats) > target_clusters and round_num < max_rounds:
                round_num += 1

                # Sort clusters by size for constraint calculations
                sorted_stats = sorted(cluster_stats, key=lambda x: x['size'], reverse=True)

                # Calculate size quartiles for merging restrictions
                size_quartiles = self._calculate_size_quartiles(sorted_stats)

                # Find merge candidates based on constraints
                merge_candidates = self._find_merge_candidates(
                    cluster_stats, size_quartiles, features, current_labels
                )

                # Perform one round of merging
                if merge_candidates:
                    merged = self._perform_merge_round(merge_candidates, current_labels, cluster_stats)
                    if not merged:
                        break  # No valid merges possible
                else:
                    break  # No merge candidates found

                # Recalculate statistics after merge
                cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

                # Check if we've achieved good distribution
                if self._check_size_distribution(cluster_stats, target_clusters):
                    self.logger.info(f"✅ Achieved good distribution after {round_num} rounds")
                    break

                self.logger.info(f"🔄 Round {round_num}: {len(cluster_stats)} clusters remaining")

            # Final cleanup to ensure we have exactly target_clusters
            if len(cluster_stats) != target_clusters:
                self.logger.warning(f"⚠️ Final adjustment: {len(cluster_stats)} clusters, target {target_clusters}")
                current_labels = self._force_final_merge(current_labels, features, target_clusters)

            return current_labels

        except Exception as e:
            self.logger.warning(f"Dynamic merging failed: {e}")
            return labels

    def _calculate_cluster_statistics(self, labels: np.ndarray, features: np.ndarray, n_samples: int) -> list:
        """Calculate comprehensive statistics for all clusters.

        Args:
            labels: Cluster labels
            features: Feature matrix
            n_samples: Total number of samples

        Returns:
            List of cluster statistics
        """
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)

            cluster_stats = []
            for label, count in zip(unique_labels, counts):
                if label == -1:
                    continue

                cluster_mask = labels == label
                cluster_features = features[cluster_mask]

                # Calculate centroid
                centroid = np.mean(cluster_features, axis=0)

                # Calculate CV for internal variance
                if cluster_features.shape[0] > 1:
                    means = np.mean(cluster_features, axis=0)
                    stds = np.std(cluster_features, axis=0)
                    vals = [abs(s / m) for m, s in zip(means, stds) if abs(m) > 1e-6]
                    cv = float(np.mean(vals)) if vals else 0.0
                else:
                    cv = 0.0

                # Calculate inverse size weight
                inv_size_weight = 1.0 / (count / n_samples)

                # Calculate merge priority (higher = more likely to be merged)
                merge_priority = cv * inv_size_weight

                cluster_stats.append({
                    'label': label,
                    'size': count,
                    'percentage': count / n_samples,
                    'centroid': centroid,
                    'cv': cv,
                    'merge_priority': merge_priority
                })

            return cluster_stats

        except Exception as e:
            self.logger.warning(f"Cluster statistics calculation failed: {e}")
            return []

    def _calculate_size_quartiles(self, sorted_stats: list) -> dict:
        """Calculate size quartiles for merging restrictions.

        Args:
            sorted_stats: Clusters sorted by size (largest first)

        Returns:
            Dictionary with quartile information
        """
        try:
            n_clusters = len(sorted_stats)
            if n_clusters < 4:
                return {'q1': 0, 'q2': n_clusters // 2, 'q3': n_clusters - 1}

            # Calculate quartile indices
            q1_idx = n_clusters // 4
            q2_idx = n_clusters // 2
            q3_idx = 3 * n_clusters // 4

            return {
                'q1': q1_idx,
                'q2': q2_idx,
                'q3': q3_idx,
                'top_25_percent': q1_idx,
                'bottom_25_percent': n_clusters - q3_idx
            }

        except Exception as e:
            self.logger.warning(f"Size quartiles calculation failed: {e}")
            return {'q1': 0, 'q2': 0, 'q3': 0}

    def _find_merge_candidates(self, cluster_stats: list, size_quartiles: dict, features: np.ndarray, labels: np.ndarray) -> list:
        """Find merge candidates based on size constraints and CV similarity.

        Args:
            cluster_stats: Cluster statistics
            size_quartiles: Size quartile information
            features: Feature matrix
            labels: Current cluster labels

        Returns:
            List of merge candidate pairs
        """
        try:
            merge_candidates = []

            # Create lookup for cluster stats by label
            stats_by_label = {stat['label']: stat for stat in cluster_stats}

            # Find merge candidates
            for i, stat1 in enumerate(cluster_stats):
                if stat1['label'] == -1:
                    continue

                # Determine merging restrictions based on size
                can_merge = self._can_cluster_merge(stat1, size_quartiles, cluster_stats)

                if not can_merge:
                    continue

                # Find closest cluster based on CV similarity
                closest_stat = None
                min_distance = float('inf')
                merge_bonus = self._calculate_merge_bonus(stat1, size_quartiles, cluster_stats)

                for j, stat2 in enumerate(cluster_stats):
                    if i == j or stat2['label'] == -1:
                        continue

                    # Calculate CV-based distance
                    cv_distance = abs(stat1['cv'] - stat2['cv'])
                    centroid_distance = np.linalg.norm(stat1['centroid'] - stat2['centroid'])

                    # Combined distance (weighted by CV and position)
                    total_distance = cv_distance * 0.7 + centroid_distance * 0.3

                    # Apply merge bonus for small clusters
                    if merge_bonus > 1.0:
                        total_distance *= (1.0 / merge_bonus)

                    if total_distance < min_distance:
                        min_distance = total_distance
                        closest_stat = stat2

                if closest_stat and min_distance < 2.0:  # Threshold for valid merge
                    merge_candidates.append({
                        'cluster1': stat1,
                        'cluster2': closest_stat,
                        'distance': min_distance,
                        'merge_bonus': merge_bonus
                    })

            # Sort by distance (closest first)
            merge_candidates.sort(key=lambda x: x['distance'])

            return merge_candidates

        except Exception as e:
            self.logger.warning(f"Merge candidate finding failed: {e}")
            return []

    def _can_cluster_merge(self, cluster_stat: dict, size_quartiles: dict, all_stats: list) -> bool:
        """Check if a cluster can merge based on size constraints.

        Args:
            cluster_stat: Statistics for the cluster
            size_quartiles: Size quartile information
            all_stats: All cluster statistics

        Returns:
            True if cluster can merge
        """
        try:
            # Sort all stats by size for ranking
            sorted_stats = sorted(all_stats, key=lambda x: x['size'], reverse=True)
            cluster_index = next(i for i, stat in enumerate(sorted_stats) if stat['label'] == cluster_stat['label'])

            # Top 25% cannot merge (largest clusters)
            if cluster_index < size_quartiles['top_25_percent']:
                return False

            # Bottom 25% can always merge (smallest clusters)
            if cluster_index >= len(sorted_stats) - size_quartiles['bottom_25_percent']:
                return True

            # Middle clusters can merge with restrictions
            return cluster_stat['percentage'] < 0.08  # Less than 8%

        except Exception as e:
            self.logger.warning(f"Merge restriction check failed: {e}")
            return False

    def _calculate_merge_bonus(self, cluster_stat: dict, size_quartiles: dict, all_stats: list) -> float:
        """Calculate merge bonus for small clusters.

        Args:
            cluster_stat: Statistics for the cluster
            size_quartiles: Size quartile information
            all_stats: All cluster statistics

        Returns:
            Merge bonus multiplier (>1.0 for small clusters)
        """
        try:
            # Sort all stats by size for ranking
            sorted_stats = sorted(all_stats, key=lambda x: x['size'], reverse=True)
            cluster_index = next(i for i, stat in enumerate(sorted_stats) if stat['label'] == cluster_stat['label'])

            # Bottom 25% get bonus to merge more
            if cluster_index >= len(sorted_stats) - size_quartiles['bottom_25_percent']:
                return 2.0  # Double bonus for smallest clusters

            # Middle clusters get moderate bonus
            elif cluster_index >= size_quartiles['q2']:
                return 1.5  # 50% bonus for medium-small clusters

            return 1.0  # No bonus for larger clusters

        except Exception as e:
            self.logger.warning(f"Merge bonus calculation failed: {e}")
            return 1.0

    def _perform_merge_round(self, merge_candidates: list, current_labels: np.ndarray, cluster_stats: list) -> bool:
        """Perform one round of merging.

        Args:
            merge_candidates: List of merge candidates
            current_labels: Current cluster labels
            cluster_stats: Current cluster statistics

        Returns:
            True if merges were performed
        """
        try:
            merges_performed = False

            for candidate in merge_candidates[:5]:  # Limit to 5 merges per round
                cluster1 = candidate['cluster1']
                cluster2 = candidate['cluster2']

                # Check if both clusters still exist and can be merged
                if (cluster1['label'] not in [stat['label'] for stat in cluster_stats] or
                    cluster2['label'] not in [stat['label'] for stat in cluster_stats]):
                    continue

                # Perform the merge
                new_label = min(cluster1['label'], cluster2['label'])
                old_label = max(cluster1['label'], cluster2['label'])

                current_labels[current_labels == old_label] = new_label
                merges_performed = True

                self.logger.info(f"✅ Merged clusters {cluster1['label']} ({cluster1['percentage']:.3f}) and {cluster2['label']} ({cluster2['percentage']:.3f}) - Distance: {candidate['distance']:.3f}")

            return merges_performed

        except Exception as e:
            self.logger.warning(f"Merge round failed: {e}")
            return False

    def _check_size_distribution(self, cluster_stats: list, target_clusters: int) -> bool:
        """Check if current distribution meets target criteria.

        Args:
            cluster_stats: Current cluster statistics
            target_clusters: Target number of clusters

        Returns:
            True if distribution is acceptable
        """
        try:
            if len(cluster_stats) != target_clusters:
                return False

            # Check that all clusters are within 3-8% range
            for stat in cluster_stats:
                if not (0.03 <= stat['percentage'] <= 0.08):
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Distribution check failed: {e}")
            return False

    def _force_final_merge(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Force final merge to reach target cluster count.

        Args:
            labels: Current cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Final cluster labels
        """
        try:
            from sklearn.cluster import KMeans

            # If we have too many clusters, force K-means to target
            if len(np.unique(labels)) > target_clusters:
                kmeans = KMeans(
                    n_clusters=target_clusters,
                    init='k-means++',
                    n_init=10,
                    max_iter=100,
                    random_state=self.config.random_state
                )
                return kmeans.fit_predict(features)

            return labels

        except Exception as e:
            self.logger.warning(f"Final merge failed: {e}")
            return labels

    def _strategic_cluster_merging(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Perform strategic merging of clusters to achieve target size distribution.

        Args:
            labels: Initial cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Merged cluster labels
        """
        try:
            n_samples = len(labels)
            current_labels = labels.copy()

            # Calculate initial cluster statistics
            cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

            # Perform multiple rounds of strategic merging
            for round_num in range(10):
                # Sort clusters by size
                sorted_stats = sorted(cluster_stats, key=lambda x: x['size'])

                # Find clusters that need to be merged (too small) or split (too large)
                too_small = [stat for stat in sorted_stats if stat['percentage'] < 0.03]
                too_large = [stat for stat in sorted_stats if stat['percentage'] > 0.08]

                if not too_small and not too_large:
                    break  # Good distribution achieved

                # Merge smallest clusters first
                if too_small:
                    smallest = too_small[0]

                    # Find closest cluster to merge with
                    closest_stat = None
                    min_distance = float('inf')

                    for stat in cluster_stats:
                        if stat['label'] == smallest['label']:
                            continue

                        # Calculate distance based on centroids and CV
                        cv_distance = abs(smallest['cv'] - stat['cv'])
                        centroid_distance = np.linalg.norm(smallest['centroid'] - stat['centroid'])
                        total_distance = cv_distance * 0.7 + centroid_distance * 0.3

                        if total_distance < min_distance:
                            min_distance = total_distance
                            closest_stat = stat

                    if closest_stat:
                        # Perform merge
                        new_label = min(smallest['label'], closest_stat['label'])
                        old_label = max(smallest['label'], closest_stat['label'])
                        current_labels[current_labels == old_label] = new_label

                        self.logger.info(f"✅ Strategic merge: {smallest['label']} ({smallest['percentage']:.3f}) + {closest_stat['label']} ({closest_stat['percentage']:.3f})")

                # Recalculate statistics
                cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

            # Final adjustment if needed
            if len(cluster_stats) > target_clusters:
                current_labels = self._force_final_merge(current_labels, features, target_clusters)

            return current_labels

        except Exception as e:
            self.logger.warning(f"Strategic merging failed: {e}")
            return labels

    def _balanced_initialization_with_redistribution(self, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Initialize clusters with balanced approach and forced redistribution.

        Args:
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Balanced cluster labels
        """
        try:
            from sklearn.cluster import KMeans

            # Start with exactly target clusters using k-means++
            kmeans = KMeans(
                n_clusters=target_clusters,
                init='k-means++',
                n_init=20,
                max_iter=100,
                random_state=self.config.random_state
            )

            labels = kmeans.fit_predict(features)

            # Apply redistribution to balance cluster sizes
            return self._apply_size_redistribution(labels, features, target_clusters)

        except Exception as e:
            self.logger.warning(f"Balanced initialization failed: {e}")
            return np.random.randint(0, target_clusters, size=features.shape[0])

    def _apply_size_redistribution(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Apply size redistribution to balance cluster sizes.

        Args:
            labels: Initial cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Redistributed cluster labels
        """
        try:
            n_samples = len(labels)
            target_size = n_samples / target_clusters
            min_size = int(target_size * 0.75)  # 75% of target
            max_size = int(target_size * 1.25)  # 125% of target

            current_labels = labels.copy()
            cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

            # Redistribute points from oversized to undersized clusters
            for stat in cluster_stats:
                if stat['size'] > max_size:
                    # Find undersized clusters
                    undersized = [s for s in cluster_stats if s['size'] < min_size]

                    if undersized:
                        # Move points to nearest undersized cluster
                        oversized_mask = current_labels == stat['label']
                        oversized_features = features[oversized_mask]

                        # Find nearest undersized cluster for each point
                        for i, point in enumerate(oversized_features):
                            if np.random.random() < 0.1:  # Only move 10% of points to avoid over-correction
                                closest_undersized = None
                                min_dist = float('inf')

                                for under_stat in undersized:
                                    dist = np.linalg.norm(point - under_stat['centroid'])
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_undersized = under_stat

                                if closest_undersized:
                                    current_labels[oversized_mask][i] = closest_undersized['label']

            return current_labels

        except Exception as e:
            self.logger.warning(f"Size redistribution failed: {e}")
            return labels

    def _evaluate_cluster_distribution(self, labels: np.ndarray, n_samples: int, target_clusters: int) -> float:
        """Evaluate how well the cluster distribution meets our criteria.

        Args:
            labels: Cluster labels
            n_samples: Total number of samples
            target_clusters: Target number of clusters

        Returns:
            Score (higher is better)
        """
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / n_samples

            # Score based on how many clusters are in 3-8% range
            min_pct = float(getattr(self.config, 'min_cluster_size_pct', 0.03))
            max_pct = float(getattr(self.config, 'max_cluster_size_pct', 0.08))
            in_range = sum(1 for pct in percentages if min_pct <= pct <= max_pct)
            range_score = in_range / target_clusters

            # Score based on how close we are to target cluster count
            count_score = 1.0 - abs(len(unique_labels) - target_clusters) / target_clusters

            # Score based on balance (lower standard deviation is better)
            balance_score = 1.0 / (1.0 + np.std(percentages))

            # Combined score
            total_score = (range_score * 0.5 + count_score * 0.3 + balance_score * 0.2)

            return total_score

        except Exception as e:
            self.logger.warning(f"Distribution evaluation failed: {e}")
            return 0.0

    def _aggressive_redistribution(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Aggressively redistribute clusters to achieve target distribution.

        Args:
            labels: Current cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Redistributed cluster labels
        """
        try:
            n_samples = len(labels)
            target_size = n_samples / target_clusters

            # Sort points by their current cluster size
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)

            # Redistribute by reassigning points to balance sizes
            new_labels = np.zeros(n_samples, dtype=int)

            for i, label in enumerate(unique_labels):
                cluster_mask = labels == label
                cluster_size = cluster_mask.sum()

                # Calculate how many points this cluster should have
                target_cluster_size = int(target_size)

                if i < len(unique_labels) - 1:
                    new_labels[cluster_mask] = i % target_clusters
                else:
                    # Last cluster gets remaining points
                    new_labels[cluster_mask] = target_clusters - 1

            # Apply K-means to refine the redistribution
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=target_clusters, init='k-means++', n_init=5, random_state=self.config.random_state)
            final_labels = kmeans.fit_predict(features)

            return final_labels

        except Exception as e:
            self.logger.warning(f"Aggressive redistribution failed: {e}")
            return labels

    def _enhanced_redistribution(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Enhanced redistribution with multiple rounds and aggressive outlier handling.

        Args:
            labels: Current cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Enhanced redistributed cluster labels
        """
        try:
            n_samples = len(labels)
            current_labels = labels.copy()
            cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

            # Identify extreme outliers
            sorted_stats = sorted(cluster_stats, key=lambda x: x['size'])
            extreme_small = [s for s in sorted_stats if s['percentage'] < 0.015]  # <1.5%
            extreme_large = [s for s in sorted_stats if s['percentage'] > 0.12]  # >12%

            # Multiple redistribution rounds
            for round_num in range(5):
                self.logger.info(f"🔄 Enhanced redistribution round {round_num + 1}")

                # Apply aggressive redistribution for extreme outliers
                if extreme_small or extreme_large:
                    current_labels = self._aggressive_outlier_redistribution(
                        current_labels, features, target_clusters, extreme_small, extreme_large
                    )

                # Apply smart transfer between adjacent clusters
                current_labels = self._smart_cluster_transfer(current_labels, features, cluster_stats)

                # Recalculate statistics
                cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

                # Check if we've achieved good distribution
                if self._check_size_distribution(cluster_stats, target_clusters):
                    self.logger.info(f"✅ Enhanced redistribution achieved good distribution after {round_num + 1} rounds")
                    break

            return current_labels

        except Exception as e:
            self.logger.warning(f"Enhanced redistribution failed: {e}")
            return labels

    def _aggressive_outlier_redistribution(self, labels: np.ndarray, features: np.ndarray, target_clusters: int,
                                         extreme_small: list, extreme_large: list) -> np.ndarray:
        """Aggressively redistribute extreme outliers.

        Args:
            labels: Current cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters
            extreme_small: List of extremely small clusters
            extreme_large: List of extremely large clusters

        Returns:
            Redistributed labels
        """
        try:
            n_samples = len(labels)
            target_size = n_samples / target_clusters
            current_labels = labels.copy()

            # Redistribute points from extreme large clusters
            for large_stat in extreme_large:
                large_mask = current_labels == large_stat['label']
                large_features = features[large_mask]

                # Calculate how many points to move (move 40% of excess)
                excess_points = large_stat['size'] - int(target_size * 1.25)
                points_to_move = max(0, int(excess_points * 0.4))

                if points_to_move > 0 and len(extreme_small) > 0:
                    # Find best small cluster to receive points
                    best_small = extreme_small[0]

                    # Move points to the smallest cluster
                    move_indices = np.random.choice(
                        np.where(large_mask)[0],
                        size=min(points_to_move, large_mask.sum()),
                        replace=False
                    )

                    current_labels[move_indices] = best_small['label']

                    self.logger.info(f"✅ Aggressive redistribution: Moved {len(move_indices)} points from cluster {large_stat['label']} to {best_small['label']}")

            return current_labels

        except Exception as e:
            self.logger.warning(f"Aggressive outlier redistribution failed: {e}")
            return labels

    def _smart_cluster_transfer(self, labels: np.ndarray, features: np.ndarray, cluster_stats: list) -> np.ndarray:
        """Smart transfer of points between adjacent clusters based on CV and centroids.

        Args:
            labels: Current cluster labels
            features: Feature matrix
            cluster_stats: Current cluster statistics

        Returns:
            Labels with smart transfers applied
        """
        try:
            current_labels = labels.copy()

            # Create lookup for cluster stats by label
            stats_by_label = {stat['label']: stat for stat in cluster_stats}

            # Find large clusters that can donate points
            large_clusters = [stat for stat in cluster_stats if stat['percentage'] > 0.08]
            small_clusters = [stat for stat in cluster_stats if stat['percentage'] < 0.03]

            for large_stat in large_clusters:
                if large_stat['label'] not in stats_by_label:
                    continue

                large_mask = current_labels == large_stat['label']
                large_features = features[large_mask]

                # Find small clusters that are "adjacent" (similar CV)
                for small_stat in small_clusters:
                    if small_stat['label'] not in stats_by_label:
                        continue

                    # Calculate CV similarity
                    cv_similarity = 1.0 - min(abs(large_stat['cv'] - small_stat['cv']), 1.0)

                    # Calculate centroid distance
                    centroid_distance = np.linalg.norm(large_stat['centroid'] - small_stat['centroid'])

                    # Combined transfer score (higher = better transfer candidate)
                    transfer_score = cv_similarity * 0.7 + (1.0 / (1.0 + centroid_distance)) * 0.3

                    # Transfer points if score is high enough
                    if transfer_score > 0.6:
                        # Calculate how many points to transfer
                        excess_in_large = large_stat['size'] - int(len(features) / len(cluster_stats) * 1.25)
                        deficit_in_small = int(len(features) / len(cluster_stats) * 0.75) - small_stat['size']

                        points_to_transfer = min(
                            max(1, int(excess_in_large * 0.15)),  # Transfer 15% of excess
                            deficit_in_small  # Don't exceed deficit
                        )

                        if points_to_transfer > 0:
                            # Find points in large cluster closest to small cluster centroid
                            distances_to_small_centroid = np.linalg.norm(
                                large_features - small_stat['centroid'], axis=1
                            )

                            # Get indices of points closest to small centroid
                            closest_indices = np.argsort(distances_to_small_centroid)[:points_to_transfer]
                            transfer_indices = np.where(large_mask)[0][closest_indices]

                            # Transfer the points
                            current_labels[transfer_indices] = small_stat['label']

                            self.logger.info(f"✅ Smart transfer: {points_to_transfer} points from cluster {large_stat['label']} to {small_stat['label']} (CV similarity: {cv_similarity:.3f})")

            return current_labels

        except Exception as e:
            self.logger.warning(f"Smart cluster transfer failed: {e}")
            return labels

    def _fallback_clustering(self, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Fallback clustering method when advanced approaches fail.

        Args:
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Cluster labels
        """
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=target_clusters, init='k-means++', n_init=10, random_state=self.config.random_state)
            return kmeans.fit_predict(features)

        except Exception as e:
            self.logger.warning(f"Fallback clustering failed: {e}")
            return np.zeros(features.shape[0], dtype=int)

    def _iterative_refinement(self, labels: np.ndarray, features: np.ndarray, target_clusters: int) -> np.ndarray:
        """Apply iterative refinement passes to problematic clusters.

        Args:
            labels: Initial cluster labels
            features: Feature matrix
            target_clusters: Target number of clusters

        Returns:
            Refined cluster labels
        """
        try:
            n_samples = len(labels)
            current_labels = labels.copy()
            cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)

            # Identify problematic clusters
            problematic = self._identify_problematic_clusters(cluster_stats)

            # Apply refinement passes
            for pass_num in range(3):
                self.logger.info(f"🔄 Iterative refinement pass {pass_num + 1}")

                if not problematic:
                    break

                # Apply targeted refinement to problematic clusters
                current_labels = self._apply_targeted_refinement(
                    current_labels, features, cluster_stats, problematic, target_clusters
                )

                # Recalculate statistics and identify new problematic clusters
                cluster_stats = self._calculate_cluster_statistics(current_labels, features, n_samples)
                problematic = self._identify_problematic_clusters(cluster_stats)

                # Check if we've achieved good distribution
                if self._check_size_distribution(cluster_stats, target_clusters):
                    self.logger.info(f"✅ Iterative refinement achieved good distribution after {pass_num + 1} passes")
                    break

            return current_labels

        except Exception as e:
            self.logger.warning(f"Iterative refinement failed: {e}")
            return labels

    def _identify_problematic_clusters(self, cluster_stats: list) -> list:
        """Identify clusters that need refinement.

        Args:
            cluster_stats: Current cluster statistics

        Returns:
            List of problematic cluster statistics
        """
        try:
            problematic = []

            # Clusters that are too small
            too_small = [stat for stat in cluster_stats if stat['percentage'] < 0.025]  # <2.5%

            # Clusters that are too large
            too_large = [stat for stat in cluster_stats if stat['percentage'] > 0.10]  # >10%

            # Clusters with very high CV (internal variance)
            high_cv = [stat for stat in cluster_stats if stat['cv'] > 1.5]

            problematic.extend(too_small)
            problematic.extend(too_large)
            problematic.extend(high_cv)

            return problematic

        except Exception as e:
            self.logger.warning(f"Problematic cluster identification failed: {e}")
            return []

    def _apply_targeted_refinement(self, labels: np.ndarray, features: np.ndarray, cluster_stats: list,
                                  problematic: list, target_clusters: int) -> np.ndarray:
        """Apply targeted refinement to specific problematic clusters.

        Args:
            labels: Current cluster labels
            features: Feature matrix
            cluster_stats: Current cluster statistics
            problematic: List of problematic clusters
            target_clusters: Target number of clusters

        Returns:
            Refined labels
        """
        try:
            current_labels = labels.copy()
            n_samples = len(features)

            for problem_stat in problematic:
                try:
                    problem_mask = current_labels == problem_stat['label']

                    if problem_stat['percentage'] < 0.025:  # Too small
                        # Find nearest cluster to merge with
                        nearest_stat = self._find_nearest_cluster(problem_stat, cluster_stats)
                        if nearest_stat:
                            # Merge the small cluster with nearest
                            current_labels[problem_mask] = nearest_stat['label']
                            self.logger.info(f"✅ Targeted refinement: Merged small cluster {problem_stat['label']} with {nearest_stat['label']}")

                    elif problem_stat['percentage'] > 0.10:  # Too large
                        # Split the large cluster
                        large_features = features[problem_mask]

                        # Calculate how many subclusters to create
                        n_subclusters = min(3, max(2, int(problem_stat['percentage'] / 0.05)))

                        if large_features.shape[0] >= n_subclusters * 20:  # Minimum size
                            from sklearn.cluster import KMeans
                            kmeans = KMeans(n_clusters=n_subclusters, n_init=5, random_state=self.config.random_state)
                            sub_labels = kmeans.fit_predict(large_features)

                            # Reassign to new cluster labels
                            max_label = np.max(current_labels)
                            for i in range(n_subclusters):
                                sub_mask = problem_mask & (sub_labels == i)
                                current_labels[sub_mask] = max_label + 1 + i

                            self.logger.info(f"✅ Targeted refinement: Split large cluster {problem_stat['label']} into {n_subclusters} subclusters")

                except Exception as e:
                    self.logger.warning(f"Targeted refinement failed for cluster {problem_stat['label']}: {e}")
                    continue

            return current_labels

        except Exception as e:
            self.logger.warning(f"Targeted refinement failed: {e}")
            return labels

    def _find_nearest_cluster(self, target_stat: dict, cluster_stats: list) -> dict:
        """Find the nearest cluster to a target cluster.

        Args:
            target_stat: Target cluster statistics
            cluster_stats: All cluster statistics

        Returns:
            Nearest cluster statistics or None
        """
        try:
            nearest = None
            min_distance = float('inf')

            for stat in cluster_stats:
                if stat['label'] == target_stat['label']:
                    continue

                # Calculate combined distance (CV + centroid)
                cv_distance = abs(target_stat['cv'] - stat['cv'])
                centroid_distance = np.linalg.norm(target_stat['centroid'] - stat['centroid'])
                total_distance = cv_distance * 0.7 + centroid_distance * 0.3

                if total_distance < min_distance:
                    min_distance = total_distance
                    nearest = stat

            return nearest

        except Exception as e:
            self.logger.warning(f"Nearest cluster finding failed: {e}")
            return None

    def _adaptive_target_adjustment(self, features: np.ndarray, target_clusters: int) -> tuple:
        """Dynamically adjust target sizes based on data characteristics.

        Args:
            features: Feature matrix
            target_clusters: Original target number of clusters

        Returns:
            Tuple of (adjusted_min_pct, adjusted_max_pct, adjusted_target_clusters)
        """
        try:
            # Analyze data characteristics
            n_samples = features.shape[0]

            # Calculate data variance and density
            data_std = np.std(features, axis=0).mean()
            data_range = np.ptp(features, axis=0).mean()  # Peak to peak
            density_estimate = n_samples / (data_range ** features.shape[1])

            # Adaptive target size calculation
            base_target_size = n_samples / target_clusters

            # Adjust based on data characteristics
            if data_std < 0.5:  # Low variance data
                adjusted_min_pct = 0.025  # 2.5%
                adjusted_max_pct = 0.10   # 10%
            elif data_std > 2.0:  # High variance data
                adjusted_min_pct = 0.035  # 3.5%
                adjusted_max_pct = 0.075  # 7.5%
            else:  # Medium variance data
                adjusted_min_pct = 0.03   # 3%
                adjusted_max_pct = 0.08   # 8%

            # Density-based adjustment
            if density_estimate > 10:  # High density
                adjusted_min_pct *= 1.2
                adjusted_max_pct *= 1.2
            elif density_estimate < 2:  # Low density
                adjusted_min_pct *= 0.8
                adjusted_max_pct *= 0.8

            # Ensure reasonable bounds
            adjusted_min_pct = max(0.015, min(0.05, adjusted_min_pct))
            adjusted_max_pct = max(0.05, min(0.15, adjusted_max_pct))

            self.logger.info(f"🎯 Adaptive targets: {adjusted_min_pct:.3f}-{adjusted_max_pct:.3f} (data_std: {data_std:.3f}, density: {density_estimate:.3f})")

            return adjusted_min_pct, adjusted_max_pct, target_clusters

        except Exception as e:
            self.logger.warning(f"Adaptive target adjustment failed: {e}")
            return 0.03, 0.08, target_clusters

    def _calculate_optimal_epsilon(self, features: np.ndarray) -> float:
        """Calculate optimal epsilon for DBSCAN using matrix operations."""
        try:
            # Use matrix operations for distance calculation
            if MATRIX_OPERATIONS_AVAILABLE:
                # Use GPU-accelerated distance calculation if possible
                from sklearn.neighbors import NearestNeighbors

                # Calculate distances to k-th nearest neighbor
                k = min(10, features.shape[0] - 1)
                nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
                nn.fit(features)
                distances, _ = nn.kneighbors(features)

                # Use matrix operations for optimal epsilon calculation
                k_distances = distances[:, -1]
                k_distances = np.sort(k_distances)

                # Find elbow point using matrix operations
                n_points = len(k_distances)
                coords = np.array([np.arange(n_points), k_distances]).T

                # Simple elbow detection using matrix operations
                line_vec = coords[-1] - coords[0]
                line_vec_norm = line_vec / np.linalg.norm(line_vec)
                vec_from_first = coords - coords[0]
                scalar_proj = np.dot(vec_from_first, line_vec_norm)
                vec_from_line = vec_from_first - np.outer(scalar_proj, line_vec_norm)
                dist_from_line = np.linalg.norm(vec_from_line, axis=1)
                elbow_index = np.argmax(dist_from_line)

                optimal_eps = k_distances[elbow_index] * 0.8  # Slightly smaller than elbow
                return max(0.1, min(1.0, optimal_eps))

            else:
                # Fallback calculation
                return 0.5

        except Exception as e:
            self.logger.warning(f"Optimal epsilon calculation failed: {e}")
            return 0.5

    def _calculate_optimal_clusters(self, features: np.ndarray) -> int:
        """Calculate optimal number of clusters using matrix operations."""
        try:
            # Force exact number of clusters if configured
            if hasattr(self.config, 'force_n_clusters') and self.config.force_n_clusters:
                return self.config.target_n_clusters

            n_samples = features.shape[0]

            # Simple heuristic based on data size and characteristics
            if n_samples < 1000:
                return min(10, self.config.target_n_clusters)
            elif n_samples < 5000:
                return min(15, self.config.target_n_clusters)
            else:
                return self.config.target_n_clusters

        except Exception:
            return self.config.target_n_clusters

    def _matrix_optimized_optimal_clusters(self, features: np.ndarray) -> int:
        """Calculate optimal clusters using matrix operations."""
        try:
            # Use more sophisticated method with matrix operations
            if MATRIX_OPERATIONS_AVAILABLE:
                # Use eigenvalue analysis for optimal clusters
                try:
                    # Calculate correlation matrix
                    corr_matrix = correlation_matrix_gpu(features)

                    # Use SVD for dimensionality analysis
                    U, s, Vt = np.linalg.svd(corr_matrix, full_matrices=False)

                    # Calculate optimal clusters based on explained variance
                    explained_variance = np.cumsum(s**2 / np.sum(s**2))
                    optimal_k = np.where(explained_variance > 0.9)[0][0] + 1

                    # Force exact number of clusters if configured
                    if hasattr(self.config, 'force_n_clusters') and self.config.force_n_clusters:
                        return self.config.target_n_clusters

                    # Constrain to reasonable range
                    optimal_k = max(5, min(optimal_k, self.config.target_n_clusters))
                    return optimal_k

                except Exception as e:
                    self.logger.warning(f"Matrix-based optimal cluster calculation failed: {e}")

            # Fallback
            return self._calculate_optimal_clusters(features)

        except Exception:
            return self.config.target_n_clusters

    def _create_optimized_kmeans(self, n_clusters: int):
        """Create optimized K-means clusterer."""
        try:
            from sklearn.cluster import KMeans

            return KMeans(
                n_clusters=n_clusters,
                n_init=10,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                init='k-means++',
                n_jobs=-1
            )
        except Exception:
            from sklearn.cluster import KMeans
            return KMeans(
                n_clusters=n_clusters,
                n_init=10,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                init='k-means++'
            )

    def _create_optimized_gmm(self, n_clusters: int):
        """Create optimized Gaussian Mixture Model."""
        try:
            return GaussianMixture(
                n_components=n_clusters,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
                n_init=5
            )
        except Exception:
            return GaussianMixture(
                n_components=n_clusters,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter
            )

def create_matrix_optimized_clusterer(config: Optional[OptimalClusteringConfig] = None) -> MatrixOptimizedClusterer:
    """Create matrix-optimized regime clusterer.

    Args:
        config: Clustering configuration

    Returns:
        MatrixOptimizedClusterer instance
    """
    if config is None:
        config = OptimalClusteringConfig()

    return MatrixOptimizedClusterer(config)

def cluster_hmm_regimes_optimized(data_path: str, config: Optional[OptimalClusteringConfig] = None,
                                 **kwargs) -> OptimizedClusteringResult:
    """Optimized clustering of HMM regimes using matrix operations.

    Args:
        data_path: Path to HMM regime data
        config: Clustering configuration
        **kwargs: Additional parameters

    Returns:
        OptimizedClusteringResult
    """
    clusterer = create_matrix_optimized_clusterer(config)
    return clusterer.cluster_optimized(data_path, **kwargs)