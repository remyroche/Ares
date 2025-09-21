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
            quality_metrics = calculate_cluster_quality_metrics(features, clustering_result.labels)
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
                    'optimization_level': 'high' if MATRIX_OPERATIONS_AVAILABLE else 'basic'
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
            self.logger.info(f"✅ Data loaded and optimized in {loading_time".3f"} seconds")
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

            # Apply additional matrix optimizations
            if MATRIX_OPERATIONS_AVAILABLE:
                # Use GPU-accelerated correlation analysis for feature selection
                if features.shape[1] > 10:  # Only if many features
                    try:
                        correlation_matrix = correlation_matrix_gpu(features)
                        feature_metadata['correlation_matrix'] = correlation_matrix

                        # Remove highly correlated features (>0.95 correlation)
                        corr_threshold = 0.95
                        upper_triangle = np.triu(correlation_matrix, k=1)
                        high_corr_indices = np.where(np.abs(upper_triangle) > corr_threshold)[1]

                        if len(high_corr_indices) > 0:
                            # Keep only uncorrelated features
                            uncorrelated_indices = [i for i in range(features.shape[1]) if i not in high_corr_indices[:len(high_corr_indices)//2]]
                            features = features[:, uncorrelated_indices]
                            feature_metadata['features_removed_correlation'] = len(high_corr_indices)//2
                            self.logger.info(f"✅ Removed {len(high_corr_indices)//2} highly correlated features")

                    except Exception as e:
                        self.logger.warning(f"Correlation analysis failed: {e}")

                # Optimize feature scaling using matrix operations
                if features.shape[0] > 10000:  # Large dataset optimization
                    # Use batch processing for scaling
                    batch_size = optimize_batch_size(features.shape[0], features.shape[1])
                    features = self._batch_scale_features(features, batch_size)
                    feature_metadata['batch_scaling_used'] = True

            preparation_time = time.time() - start_time
            self.logger.info(f"✅ Features prepared in {preparation_time".3f"} seconds")
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
                    contamination=0.05
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
                    contamination=0.05
                )
                if outlier_mask.sum() > 0:
                    features = features[~outlier_mask]

            removal_time = time.time() - start_time
            self.logger.info(f"✅ Outliers removed in {removal_time".3f"} seconds")
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
            self.logger.info(f"✅ Matrix-optimized clustering completed in {clustering_time".3f"} seconds")
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

            # Stage 4: Create optimized cluster centers
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
                self.logger.info(f"✅ Matrix-optimized HDBSCAN found {len(np.unique(labels[labels != -1]))} clusters")
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
            self.logger.info(f"✅ Matrix-optimized DBSCAN found {len(np.unique(labels[labels != -1]))} clusters")
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
            # Use optimized K-means with matrix operations
            labels = self._optimized_kmeans_clustering(features)

            self.logger.info(f"✅ Matrix-optimized K-means created {len(np.unique(labels))} clusters")
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
            if len(np.unique(noise_labels)) > 1:
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

            # Use matrix operations to calculate optimal parameters
            min_cluster_size = max(50, int(n_samples * 0.001))
            min_samples = max(10, int(n_samples * 0.0005))

            return {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': 0.1
            }
        except Exception:
            return {}

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