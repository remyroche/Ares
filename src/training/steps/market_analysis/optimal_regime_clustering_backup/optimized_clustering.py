"""
Optimized HMM Clustering with Vectorized Operations and Memory Efficiency

This module provides optimized clustering algorithms using:
- Vectorized operations from matrix_operations/
- Memory-efficient data structures from hardware/
- Caching for frequently computed metrics
- GPU acceleration when available
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
import gc
from functools import lru_cache
from dataclasses import dataclass
from contextlib import contextmanager

# Import matrix operations and hardware optimizations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor,
        gpu_matrix_multiply,
        batch_matrix_multiply,
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_advanced_memory_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_cpu_optimizer,
        get_unified_hardware_manager,
        optimize_dataframe_advanced,
        AdvancedM1MemoryOptimizer,
        EnhancedM1GPUManager,
        AdvancedM1CPUOptimizer,
        ADVANCED_MEMORY_AVAILABLE,
        ENHANCED_GPU_AVAILABLE,
        ADVANCED_CPU_AVAILABLE
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    ADVANCED_MEMORY_AVAILABLE = False
    ENHANCED_GPU_AVAILABLE = False
    ADVANCED_CPU_AVAILABLE = False

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from .config import OptimalClusteringConfig
from .utils import (
    calculate_cluster_statistics, calculate_cluster_quality_metrics,
    validate_cluster_quality, detect_outliers,
    prepare_clustering_features, load_regime_data
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
    success: bool
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None

class OptimizedRegimeClusterer:
    """
    Optimized clustering algorithm for HMM regime data with:
    - Vectorized operations
    - Memory-efficient data structures
    - Caching for frequently computed metrics
    - GPU acceleration when available
    """

    def __init__(self, config: OptimalClusteringConfig):
        """Initialize the optimized clusterer."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize hardware optimizations
        self._init_hardware_optimizations()

        # Initialize matrix operations
        self._init_matrix_operations()

        # Initialize caching
        self._init_caching()

        # Performance tracking
        self.performance_metrics = {}
        self._operation_times = {}

    def _init_hardware_optimizations(self):
        """Initialize hardware optimization components."""
        try:
            if HARDWARE_AVAILABLE:
                self.memory_optimizer = get_advanced_memory_optimizer()
                self.gpu_manager = get_enhanced_gpu_manager()
                self.cpu_optimizer = get_advanced_cpu_optimizer()
                self.hardware_manager = get_unified_hardware_manager()
                self.logger.info("✅ Hardware optimizations initialized")
            else:
                self.memory_optimizer = None
                self.gpu_manager = None
                self.cpu_optimizer = None
                self.hardware_manager = None
                self.logger.warning("⚠️ Hardware optimizations not available")
        except Exception as e:
            self.logger.warning(f"Hardware initialization failed: {e}")
            self.memory_optimizer = None
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.hardware_manager = None

    def _init_matrix_operations(self):
        """Initialize matrix operations components."""
        try:
            if MATRIX_OPS_AVAILABLE:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized")
            else:
                self.matrix_ops = None
                self.vectorized_core = None
                self.enhanced_ops = None
                self.batch_processor = None
                self.logger.warning("⚠️ Matrix operations not available")
        except Exception as e:
            self.logger.warning(f"Matrix operations initialization failed: {e}")
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            self.batch_processor = None

    def _init_caching(self):
        """Initialize caching for frequently computed metrics."""
        self._distance_cache = {}
        self._centroid_cache = {}
        self._similarity_cache = {}
        self._quality_cache = {}

        # Cache configuration
        self.cache_size = 1000
        self.cache_ttl = 300  # 5 minutes

    @contextmanager
    def _performance_timer(self, operation_name: str):
        """Context manager for performance timing."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._operation_times[operation_name] = elapsed
            self.logger.debug(f"⏱️ {operation_name}: {elapsed:.4f}s")

    def _get_cached_result(self, cache_key: str, compute_func: Callable, *args, **kwargs):
        """Get cached result or compute and cache."""
        if cache_key in self._quality_cache:
            cached_result, timestamp = self._quality_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result

        # Compute result
        result = compute_func(*args, **kwargs)

        # Cache result
        if len(self._quality_cache) < self.cache_size:
            self._quality_cache[cache_key] = (result, time.time())

        return result

    def _vectorized_distance_calculation(self, X: np.ndarray, centers: np.ndarray,
                                        metric: str = "euclidean") -> np.ndarray:
        """
        Vectorized distance calculation using optimized matrix operations.

        Args:
            X: Data points (n_samples, n_features)
            centers: Cluster centers (n_clusters, n_features)
            metric: Distance metric ('euclidean', 'cosine', 'mahalanobis')

        Returns:
            Distance matrix (n_samples, n_clusters)
        """
        cache_key = f"dist_{hash(X.tobytes())}_{hash(centers.tobytes())}_{metric}"

        def compute_distances():
            if metric == "euclidean":
                return self._vectorized_euclidean_distance(X, centers)
            elif metric == "cosine":
                return self._vectorized_cosine_distance(X, centers)
            elif metric == "mahalanobis":
                return self._vectorized_mahalanobis_distance(X, centers)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return self._get_cached_result(cache_key, compute_distances)

    def _vectorized_euclidean_distance(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Vectorized Euclidean distance calculation."""
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_euclidean_distance'):
                return self.matrix_ops.vectorized_euclidean_distance(X, centers)

            # Fallback to optimized numpy implementation
            # Use broadcasting for efficient computation
            X_expanded = X[:, np.newaxis, :]  # (n_samples, 1, n_features)
            centers_expanded = centers[np.newaxis, :, :]  # (1, n_clusters, n_features)

            # Compute squared differences
            diff = X_expanded - centers_expanded
            squared_diff = np.sum(diff ** 2, axis=2)

            # Return distances
            return np.sqrt(squared_diff)

        except Exception as e:
            self.logger.warning(f"Vectorized Euclidean distance failed: {e}")
            # Fallback to sklearn implementation
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(X, centers)

    def _vectorized_cosine_distance(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Vectorized cosine distance calculation."""
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_cosine_distance'):
                return self.matrix_ops.vectorized_cosine_distance(X, centers)

            # Normalize vectors
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)

            # Compute cosine similarities
            similarities = np.dot(X_norm, centers_norm.T)

            # Convert to distances
            return 1 - similarities

        except Exception as e:
            self.logger.warning(f"Vectorized cosine distance failed: {e}")
            from sklearn.metrics.pairwise import cosine_distances
            return cosine_distances(X, centers)

    def _vectorized_mahalanobis_distance(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Vectorized Mahalanobis distance calculation."""
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_mahalanobis_distance'):
                return self.matrix_ops.vectorized_mahalanobis_distance(X, centers)

            # Compute covariance matrix
            cov = np.cov(X.T)

            # Add regularization for numerical stability
            reg_param = 1e-6
            cov_reg = cov + reg_param * np.eye(cov.shape[0])

            # Compute inverse covariance
            try:
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                inv_cov = np.linalg.pinv(cov_reg)

            # Compute Mahalanobis distances
            diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=2))

            return distances

        except Exception as e:
            self.logger.warning(f"Vectorized Mahalanobis distance failed: {e}")
            from sklearn.metrics.pairwise import pairwise_distances
            return pairwise_distances(X, centers, metric='mahalanobis')

    def _memory_efficient_clustering(self, features: np.ndarray,
                                   chunk_size: int = 10000) -> np.ndarray:
        """
        Memory-efficient clustering for large datasets using chunking.

        Args:
            features: Feature matrix
            chunk_size: Size of chunks for processing

        Returns:
            Cluster labels
        """
        n_samples = features.shape[0]

        if n_samples <= chunk_size:
            # Small dataset - process normally
            return self._standard_clustering(features)

        self.logger.info(f"🔄 Processing large dataset ({n_samples} samples) in chunks of {chunk_size}")

        # Initialize memory optimizer if available
        if self.memory_optimizer:
            self.memory_optimizer.optimize_for_clustering(n_samples, chunk_size)

        # Process in chunks
        all_labels = np.zeros(n_samples, dtype=int)

        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = features[i:end_idx]

            # Process chunk
            chunk_labels = self._standard_clustering(chunk)
            all_labels[i:end_idx] = chunk_labels

            # Memory cleanup
            del chunk, chunk_labels
            if self.memory_optimizer:
                self.memory_optimizer.cleanup_memory()

        return all_labels

    def _standard_clustering(self, features: np.ndarray) -> np.ndarray:
        """Standard clustering implementation."""
        if self.config.clustering_method == "kmeans":
            return self._optimized_kmeans_clustering(features)
        elif self.config.clustering_method == "dbscan":
            return self._optimized_dbscan_clustering(features)
        else:
            return self._optimized_hybrid_clustering(features)

    def _optimized_kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """Optimized K-means clustering with vectorized operations."""
        with self._performance_timer("kmeans_clustering"):
            n_clusters = self.config.target_n_clusters

            # Use multiple seeds for stability
            best_labels = None
            best_inertia = float('inf')

            for seed in range(self.config.kmeans_num_seeds):
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    init='k-means++',
                    n_init=1,  # Single run per seed
                    max_iter=self.config.kmeans_max_iter,
                    random_state=seed,
                    algorithm='lloyd'  # Use Lloyd's algorithm for efficiency
                )

                labels = kmeans.fit_predict(features)
                inertia = kmeans.inertia_

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels

            return best_labels

    def _optimized_dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Optimized DBSCAN clustering with vectorized operations."""
        with self._performance_timer("dbscan_clustering"):
            dbscan = DBSCAN(
                eps=self.config.cluster_selection_epsilon,
                min_samples=self.config.min_samples,
                algorithm='ball_tree',  # More efficient for high dimensions
                metric='euclidean'
            )

            labels = dbscan.fit_predict(features)

            # Handle noise points
            if np.any(labels == -1):
                labels = self._assign_noise_to_nearest(features, labels)

            return labels

    def _optimized_hybrid_clustering(self, features: np.ndarray) -> np.ndarray:
        """Optimized hybrid clustering combining multiple algorithms."""
        with self._performance_timer("hybrid_clustering"):
            # First pass: DBSCAN for noise reduction
            dbscan_labels = self._optimized_dbscan_clustering(features)

            # Second pass: K-means on core points
            core_mask = dbscan_labels != -1
            if np.sum(core_mask) > 0:
                core_features = features[core_mask]
                kmeans_labels = self._optimized_kmeans_clustering(core_features)

                # Combine results
                final_labels = np.full(len(features), -1)
                final_labels[core_mask] = kmeans_labels
                final_labels = self._assign_noise_to_nearest(features, final_labels)
            else:
                final_labels = dbscan_labels

            return final_labels

    def _assign_noise_to_nearest(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Assign noise points to nearest clusters using vectorized operations."""
        new_labels = labels.copy()
        noise_idx = np.where(new_labels == -1)[0]

        if len(noise_idx) == 0:
            return new_labels

        # Get cluster centers
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) == 0:
            new_labels[noise_idx] = 0
            return new_labels

        # Compute centers
        centers = np.array([np.mean(features[labels == label], axis=0)
                           for label in unique_labels])

        # Compute distances from noise points to centers
        noise_features = features[noise_idx]
        distances = self._vectorized_distance_calculation(noise_features, centers)

        # Assign to nearest cluster
        nearest_clusters = unique_labels[np.argmin(distances, axis=1)]
        new_labels[noise_idx] = nearest_clusters

        return new_labels

    def _cached_quality_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics with caching."""
        cache_key = f"quality_{hash(features.tobytes())}_{hash(labels.tobytes())}"

        def compute_metrics():
            return calculate_cluster_quality_metrics(features, labels)

        return self._get_cached_result(cache_key, compute_metrics)

    def _cached_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score with caching."""
        cache_key = f"silhouette_{hash(features.tobytes())}_{hash(labels.tobytes())}"

        def compute_silhouette():
            if len(np.unique(labels)) > 1:
                return silhouette_score(features, labels)
            return 0.0

        return self._get_cached_result(cache_key, compute_silhouette)

    def cluster(self, data: Union[str, pd.DataFrame], **kwargs) -> OptimizedClusteringResult:
        """
        Perform optimized clustering on HMM regime data.

        Args:
            data: Path to data file or DataFrame containing regime data
            **kwargs: Additional parameters

        Returns:
            OptimizedClusteringResult object
        """
        try:
            self.logger.info("🚀 Starting optimized regime clustering...")

            # Load and prepare data
            if isinstance(data, str):
                regime_data = load_regime_data(data, self.config.to_dict())
            else:
                regime_data = data

            # Optimize data for memory efficiency
            if self.memory_optimizer:
                regime_data = self.memory_optimizer.optimize_dataframe(regime_data)

            features, feature_metadata = prepare_clustering_features(regime_data, self.config.to_dict())

            # Detect and remove outliers
            outlier_mask = detect_outliers(
                features,
                method=self.config.outlier_detection_method,
                contamination=0.05
            )

            if outlier_mask.sum() > 0:
                self.logger.info(f"🗑️ Removing {outlier_mask.sum()} outliers")
                features = features[~outlier_mask]

            # Perform clustering
            if self.config.multi_stage_clustering:
                labels = self._multi_stage_optimized_clustering(features)
            else:
                labels = self._single_stage_optimized_clustering(features)

            # Calculate statistics and metrics
            statistics = calculate_cluster_statistics(labels, self.config.to_dict())
            quality_metrics = self._cached_quality_metrics(features, labels)

            # Validate results
            validation = validate_cluster_quality(statistics, quality_metrics, self.config.to_dict())

            # Create cluster centers
            cluster_centers = self._calculate_optimized_cluster_centers(features, labels)

            # Create metadata
            metadata = {
                'feature_metadata': feature_metadata,
                'clustering_method': self.config.clustering_method,
                'performance_metrics': self._operation_times,
                'hardware_optimizations': {
                    'memory_optimizer_available': self.memory_optimizer is not None,
                    'gpu_manager_available': self.gpu_manager is not None,
                    'matrix_ops_available': self.matrix_ops is not None
                }
            }

            result = OptimizedClusteringResult(
                labels=labels,
                cluster_centers=cluster_centers,
                statistics=statistics,
                quality_metrics=quality_metrics,
                validation=validation,
                metadata=metadata,
                success=True,
                performance_metrics=self._operation_times
            )

            self.logger.info("✅ Optimized regime clustering completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error in optimized regime clustering: {e}")
            return OptimizedClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics=None,
                quality_metrics={},
                validation=None,
                metadata={},
                success=False,
                performance_metrics=self._operation_times,
                error_message=str(e)
            )

    def _multi_stage_optimized_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform multi-stage optimized clustering."""
        with self._performance_timer("multi_stage_clustering"):
            # Stage 1: Noise reduction
            noise_labels = self._optimized_dbscan_clustering(features)

            # Stage 2: Main clustering
            main_labels = self._optimized_kmeans_clustering(features)

            # Stage 3: Combine and optimize
            final_labels = self._combine_and_optimize_clusters(features, noise_labels, main_labels)

            return final_labels

    def _single_stage_optimized_clustering(self, features: np.ndarray) -> np.ndarray:
        """Perform single-stage optimized clustering."""
        with self._performance_timer("single_stage_clustering"):
            return self._standard_clustering(features)

    def _combine_and_optimize_clusters(self, features: np.ndarray, noise_labels: np.ndarray,
                                     main_labels: np.ndarray) -> np.ndarray:
        """Combine and optimize clustering results."""
        with self._performance_timer("combine_optimize"):
            # Start with main clustering results
            final_labels = main_labels.copy()

            # Ensure full coverage
            if np.any(noise_labels == -1):
                final_labels = self._assign_noise_to_nearest(features, final_labels)

            return final_labels

    def _calculate_optimized_cluster_centers(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers using optimized operations."""
        with self._performance_timer("calculate_centers"):
            unique_labels = np.unique(labels)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            if len(unique_labels) == 0:
                return np.array([])

            # Use vectorized operations for center calculation
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_centers'):
                return self.matrix_ops.vectorized_centers(features, labels, unique_labels)

            # Fallback to standard implementation
            centers = []
            for label in unique_labels:
                cluster_points = features[labels == label]
                centers.append(np.mean(cluster_points, axis=0))

            return np.array(centers)

    def cleanup(self):
        """Cleanup resources and clear caches."""
        self._distance_cache.clear()
        self._centroid_cache.clear()
        self._similarity_cache.clear()
        self._quality_cache.clear()

        if self.memory_optimizer:
            self.memory_optimizer.cleanup_memory()

        gc.collect()

def create_optimized_clusterer(config: Optional[OptimalClusteringConfig] = None) -> OptimizedRegimeClusterer:
    """Create optimized regime clusterer."""
    if config is None:
        config = OptimalClusteringConfig()

    return OptimizedRegimeClusterer(config)

def cluster_hmm_regimes_optimized(data_path: str, config: Optional[OptimalClusteringConfig] = None,
                                 **kwargs) -> OptimizedClusteringResult:
    """Convenience function to cluster HMM regimes with optimizations."""
    clusterer = create_optimized_clusterer(config)
    result = clusterer.cluster(data_path, **kwargs)
    clusterer.cleanup()
    return result
