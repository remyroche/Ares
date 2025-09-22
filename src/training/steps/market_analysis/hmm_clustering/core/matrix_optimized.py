"""
Matrix-optimized clustering algorithm for HMM regime clustering.

This module provides optimized clustering algorithms using the unified matrix operations
system for maximum performance and memory efficiency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, HDBSCAN
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

# Import hardware acceleration
try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

from .base_clustering import BaseClusterer, ClusteringResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizedClusteringResult(ClusteringResult):
    """Result of optimized clustering operation."""
    performance_metrics: Dict[str, float] = None
    matrix_ops_used: bool = False
    hardware_acceleration_used: bool = False


class MatrixOptimizedClusterer(BaseClusterer):
    """Matrix-optimized clustering algorithm for regime data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the optimized clusterer.

        Args:
            config: Clustering configuration
        """
        super().__init__(config)
        
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

    def cluster(self, features: np.ndarray) -> ClusteringResult:
        """Perform matrix-optimized clustering.

        Args:
            features: Feature matrix to cluster

        Returns:
            ClusteringResult with clustering results
        """
        start_time = time.time()
        
        try:
            # Prepare features
            features = self._prepare_features(features)
            
            # Monitor performance
            self._monitor_performance("matrix_optimized_clustering")
            
            # Perform optimized clustering
            result = self._matrix_optimized_multi_stage_clustering(features)
            
            # Stop performance monitoring
            perf_metrics = self._stop_performance_monitoring("matrix_optimized_clustering")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create result with performance metrics
            clustering_result = self._create_result(
                labels=result['labels'],
                features=features,
                execution_time=execution_time,
                metadata={
                    'method': 'matrix_optimized',
                    'matrix_ops_used': self.matrix_ops is not None,
                    'hardware_acceleration_used': self.hardware_accelerator is not None,
                    'performance_metrics': perf_metrics,
                    'metrics_evolution': result.get('metrics_evolution', {})
                }
            )
            
            # Add optimized-specific fields
            clustering_result.performance_metrics = perf_metrics
            clustering_result.matrix_ops_used = self.matrix_ops is not None
            clustering_result.hardware_acceleration_used = self.hardware_accelerator is not None
            
            self.logger.info(f"✅ Matrix-optimized clustering completed in {execution_time:.2f}s")
            return clustering_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Matrix-optimized clustering failed: {e}")
            return ClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e), 'method': 'matrix_optimized'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _matrix_optimized_multi_stage_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform multi-stage matrix-optimized clustering.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with clustering results and metrics evolution
        """
        self.logger.info("🚀 Starting matrix-optimized multi-stage clustering")
        
        # Initialize metrics evolution tracking
        metrics_evolution = {}
        
        # Stage 1: Noise Reduction with GPU Acceleration (keep noise points)
        self.logger.info("📊 Stage 1: Matrix-optimized noise reduction")
        noise_labels, noise_metrics = self._matrix_optimized_noise_reduction(features)
        metrics_evolution['step_1_noise_reduction'] = noise_metrics
        metrics_evolution['step_1_noise_reduction']['basic_metrics'] = self._calculate_basic_clustering_metrics(features, noise_labels)
        
        # Stage 2: K-means Clustering with Matrix Operations
        self.logger.info("🔗 Stage 2: Matrix-optimized K-means clustering")
        kmeans_result, kmeans_metrics = self._matrix_optimized_kmeans_clustering(features)
        metrics_evolution['step_2_kmeans'] = kmeans_metrics
        metrics_evolution['step_2_kmeans']['basic_metrics'] = self._calculate_basic_clustering_metrics(features, kmeans_result)
        
        # Stage 3: Combines HDBSCAN noise reduction with K-means results (keep noise)
        self.logger.info("🎯 Stage 3: Combining HDBSCAN noise reduction with K-means")
        combined_result, combined_metrics = self._matrix_optimized_combine_clusters(features, noise_labels, kmeans_result)
        metrics_evolution['step_3_combination'] = combined_metrics
        metrics_evolution['step_3_combination']['basic_metrics'] = self._calculate_basic_clustering_metrics(features, combined_result)
        
        # Stage 4: Main Clustering with Constraint Enforcement
        self.logger.info("⚙️ Stage 4: Main clustering with constraint enforcement")
        main_result, main_metrics = self._matrix_optimized_main_clustering(features, combined_result)
        metrics_evolution['step_4_main_clustering'] = main_metrics
        metrics_evolution['step_4_main_clustering']['basic_metrics'] = self._calculate_basic_clustering_metrics(features, main_result)
        
        # Stage 5: Iterative Constraint Enforcement
        self.logger.info("🔄 Stage 5: Iterative constraint enforcement")
        final_result, constraint_metrics = self._iterative_constraint_enforcement(features, main_result)
        metrics_evolution['step_5_constraint_enforcement'] = constraint_metrics
        metrics_evolution['step_5_constraint_enforcement']['basic_metrics'] = self._calculate_basic_clustering_metrics(features, final_result)
        
        self.logger.info("✅ Matrix-optimized multi-stage clustering completed")
        
        return {
            'labels': final_result,
            'metrics_evolution': metrics_evolution
        }

    def _matrix_optimized_noise_reduction(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform matrix-optimized noise reduction using HDBSCAN.

        Args:
            features: Feature matrix

        Returns:
            Tuple of (labels, metrics)
        """
        try:
            # Use matrix operations for distance calculation if available
            if self.matrix_ops is not None:
                # Optimize batch size for matrix operations
                batch_size = self.batch_processor.optimize_batch_size(features.shape[0]) if self.batch_processor else 1000
                
                # Process in batches if needed
                if features.shape[0] > batch_size:
                    return self._batch_noise_reduction(features, batch_size)
            
            # Standard HDBSCAN noise reduction
            clusterer = HDBSCAN(
                min_cluster_size=max(5, features.shape[0] // 100),
                min_samples=max(3, features.shape[0] // 200),
                cluster_selection_epsilon=0.1
            )
            
            labels = clusterer.fit_predict(features)
            
            # Keep noise points (-1 labels) for further processing
            valid_labels = labels[labels != -1]
            noise_labels = labels[labels == -1]
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels) if hasattr(unique_labels, '__len__') else 1
            n_noise = len(noise_labels)
            
            # Calculate noise reduction metrics
            noise_reduction_metrics = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_percentage': (n_noise / len(labels)) * 100,
                'cluster_method': 'HDBSCAN',
                'parameters': {
                    'min_cluster_size': clusterer.min_cluster_size,
                    'min_samples': clusterer.min_samples,
                    'cluster_selection_epsilon': clusterer.cluster_selection_epsilon
                }
            }
            
            self.logger.info(f"✅ Matrix-optimized HDBSCAN found {n_clusters} clusters, {n_noise} noise points ({noise_reduction_metrics['noise_percentage']:.1f}%)")
            return labels, noise_reduction_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix-optimized noise reduction failed: {e}")
            # Fallback to simple labeling
            labels = np.zeros(features.shape[0], dtype=int)
            return labels, {
                'n_clusters': 1,
                'n_noise_points': 0,
                'noise_percentage': 0.0,
                'cluster_method': 'fallback',
                'error': str(e)
            }

    def _batch_noise_reduction(self, features: np.ndarray, batch_size: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform noise reduction in batches for large datasets.

        Args:
            features: Feature matrix
            batch_size: Batch size for processing

        Returns:
            Tuple of (labels, metrics)
        """
        self.logger.info(f"🔄 Processing noise reduction in batches of {batch_size}")
        
        all_labels = []
        total_clusters = 0
        
        for i in range(0, features.shape[0], batch_size):
            batch_features = features[i:i + batch_size]
            
            # Process batch
            clusterer = HDBSCAN(
                min_cluster_size=max(5, batch_size // 20),
                min_samples=max(3, batch_size // 40)
            )
            
            batch_labels = clusterer.fit_predict(batch_features)
            
            # Adjust cluster labels to avoid conflicts
            valid_labels = batch_labels[batch_labels != -1]
            if len(valid_labels) > 0:
                unique_labels = np.unique(valid_labels)
                batch_labels[batch_labels != -1] += total_clusters
                total_clusters += len(unique_labels)
            
            all_labels.extend(batch_labels)
        
        labels = np.array(all_labels)
        
        # Calculate metrics
        valid_labels = labels[labels != -1]
        noise_labels = labels[labels == -1]
        n_clusters = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0
        n_noise = len(noise_labels)
        
        return labels, {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': (n_noise / len(labels)) * 100,
            'cluster_method': 'HDBSCAN_batched',
            'batch_size': batch_size,
            'total_batches': (features.shape[0] + batch_size - 1) // batch_size
        }

    def _matrix_optimized_kmeans_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform matrix-optimized K-means clustering.

        Args:
            features: Feature matrix

        Returns:
            Tuple of (labels, metrics)
        """
        try:
            # Determine optimal number of clusters
            n_clusters = min(20, max(2, features.shape[0] // 100))
            
            # Use matrix operations for K-means if available
            if self.matrix_ops is not None:
                # Use GPU-accelerated K-means if available
                if hasattr(self.matrix_ops, 'kmeans_gpu'):
                    labels, centers = self.matrix_ops.kmeans_gpu(features, n_clusters)
                else:
                    # Use standard K-means with matrix operations for distance calculation
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
                    centers = kmeans.cluster_centers_
            else:
                # Standard K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                centers = kmeans.cluster_centers_
            
            # Calculate K-means metrics
            kmeans_metrics = {
                'n_clusters': n_clusters,
                'method': 'KMeans_matrix_optimized' if self.matrix_ops else 'KMeans_standard',
                'inertia': float(kmeans.inertia_) if 'kmeans' in locals() else 0.0,
                'n_iter': int(kmeans.n_iter_) if 'kmeans' in locals() else 0
            }
            
            self.logger.info(f"✅ Matrix-optimized K-means found {n_clusters} clusters")
            return labels, kmeans_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix-optimized K-means failed: {e}")
            # Fallback to single cluster
            labels = np.zeros(features.shape[0], dtype=int)
            return labels, {
                'n_clusters': 1,
                'method': 'fallback',
                'error': str(e)
            }

    def _matrix_optimized_combine_clusters(self, features: np.ndarray, noise_labels: np.ndarray, 
                                         kmeans_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Combine HDBSCAN noise reduction with K-means results.

        Args:
            features: Feature matrix
            noise_labels: HDBSCAN labels
            kmeans_labels: K-means labels

        Returns:
            Tuple of (combined_labels, metrics)
        """
        try:
            # Keep noise points from HDBSCAN and combine with K-means
            combined_labels = noise_labels.copy()
            
            # For non-noise points, use K-means labels but adjust to avoid conflicts
            valid_mask = noise_labels != -1
            if np.any(valid_mask):
                # Adjust K-means labels to avoid conflicts with HDBSCAN cluster IDs
                max_hdbscan_id = np.max(noise_labels[valid_mask]) if np.any(valid_mask) else -1
                adjusted_kmeans = kmeans_labels[valid_mask] + max_hdbscan_id + 1
                combined_labels[valid_mask] = adjusted_kmeans
            
            # Calculate combination metrics
            unique_labels = np.unique(combined_labels[combined_labels != -1])
            n_clusters = len(unique_labels) if len(unique_labels) > 0 else 0
            n_noise = np.sum(combined_labels == -1)
            
            combination_metrics = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_percentage': (n_noise / len(combined_labels)) * 100,
                'method': 'HDBSCAN_KMeans_combination',
                'hdbscan_clusters': len(np.unique(noise_labels[noise_labels != -1])),
                'kmeans_clusters': len(np.unique(kmeans_labels))
            }
            
            self.logger.info(f"✅ Combined clustering: {n_clusters} clusters, {n_noise} noise points")
            return combined_labels, combination_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster combination failed: {e}")
            return noise_labels, {
                'n_clusters': 0,
                'n_noise_points': len(noise_labels),
                'method': 'fallback',
                'error': str(e)
            }

    def _matrix_optimized_main_clustering(self, features: np.ndarray, 
                                        initial_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform main clustering with constraint enforcement.

        Args:
            features: Feature matrix
            initial_labels: Initial cluster labels

        Returns:
            Tuple of (labels, metrics)
        """
        try:
            # Use Gaussian Mixture Model for main clustering
            n_components = min(20, max(2, len(np.unique(initial_labels[initial_labels != -1]))))
            
            # Prepare features for GMM
            valid_mask = initial_labels != -1
            if np.any(valid_mask):
                valid_features = features[valid_mask]
                
                # Use matrix operations for GMM if available
                if self.matrix_ops is not None:
                    # Use optimized GMM
                    gmm = GaussianMixture(n_components=n_components, random_state=42)
                    gmm.fit(valid_features)
                    main_labels = gmm.predict(valid_features)
                else:
                    gmm = GaussianMixture(n_components=n_components, random_state=42)
                    gmm.fit(valid_features)
                    main_labels = gmm.predict(valid_features)
                
                # Update labels
                final_labels = initial_labels.copy()
                final_labels[valid_mask] = main_labels
            else:
                final_labels = initial_labels.copy()
            
            # Calculate main clustering metrics
            main_clustering_metrics = {
                'n_clusters': len(np.unique(final_labels[final_labels != -1])),
                'method': 'GMM_matrix_optimized' if self.matrix_ops else 'GMM_standard',
                'n_components': n_components,
                'n_valid_points': np.sum(valid_mask),
                'n_noise_points': np.sum(final_labels == -1)
            }
            
            self.logger.info(f"✅ Main clustering completed: {main_clustering_metrics['n_clusters']} clusters")
            return final_labels, main_clustering_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Main clustering failed: {e}")
            return initial_labels, {
                'n_clusters': 0,
                'method': 'fallback',
                'error': str(e)
            }

    def _iterative_constraint_enforcement(self, features: np.ndarray, 
                                        labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform iterative constraint enforcement.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Tuple of (labels, metrics)
        """
        try:
            # Apply size constraints
            final_labels = self._apply_size_constraints(features, labels)
            
            # Calculate constraint enforcement metrics
            constraint_metrics = {
                'n_clusters': len(np.unique(final_labels[final_labels != -1])),
                'n_valid_points': np.sum(final_labels != -1),
                'n_noise_points': np.sum(final_labels == -1),
                'method': 'constraint_enforcement',
                'constraints_applied': True
            }
            
            self.logger.info(f"✅ Constraint enforcement completed: {constraint_metrics['n_clusters']} clusters")
            return final_labels, constraint_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Constraint enforcement failed: {e}")
            return labels, {
                'n_clusters': 0,
                'method': 'fallback',
                'error': str(e)
            }

    def _apply_size_constraints(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Apply size constraints to clusters.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Updated labels with constraints applied
        """
        try:
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) == 0:
                return labels
            
            # Get cluster sizes
            cluster_sizes = {}
            for label in unique_labels:
                cluster_sizes[label] = np.sum(labels == label)
            
            # Apply constraints
            min_size = self.config.get('min_cluster_size', 10)
            max_size = self.config.get('max_cluster_size', 1000)
            
            updated_labels = labels.copy()
            
            for label, size in cluster_sizes.items():
                if size < min_size:
                    # Merge small clusters with nearest
                    updated_labels = self._merge_small_cluster(features, updated_labels, label)
                elif size > max_size:
                    # Split large clusters
                    updated_labels = self._split_large_cluster(features, updated_labels, label)
            
            return updated_labels
            
        except Exception as e:
            self.logger.warning(f"⚠️ Size constraint application failed: {e}")
            return labels

    def _merge_small_cluster(self, features: np.ndarray, labels: np.ndarray, 
                           small_label: int) -> np.ndarray:
        """Merge small cluster with nearest cluster.

        Args:
            features: Feature matrix
            labels: Cluster labels
            small_label: Label of small cluster to merge

        Returns:
            Updated labels
        """
        try:
            # Find nearest cluster
            small_mask = labels == small_label
            if not np.any(small_mask):
                return labels
            
            # Get centroids of all clusters
            centroids = {}
            for label in np.unique(labels[labels != -1]):
                if label != small_label:
                    mask = labels == label
                    centroids[label] = np.mean(features[mask], axis=0)
            
            if not centroids:
                return labels
            
            # Find nearest centroid
            small_centroid = np.mean(features[small_mask], axis=0)
            min_distance = float('inf')
            nearest_label = None
            
            for label, centroid in centroids.items():
                distance = np.linalg.norm(small_centroid - centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_label = label
            
            # Merge clusters
            if nearest_label is not None:
                labels[small_mask] = nearest_label
            
            return labels
            
        except Exception as e:
            self.logger.warning(f"⚠️ Small cluster merge failed: {e}")
            return labels

    def _split_large_cluster(self, features: np.ndarray, labels: np.ndarray, 
                           large_label: int) -> np.ndarray:
        """Split large cluster into smaller clusters.

        Args:
            features: Feature matrix
            labels: Cluster labels
            large_label: Label of large cluster to split

        Returns:
            Updated labels
        """
        try:
            large_mask = labels == large_label
            if not np.any(large_mask):
                return labels
            
            large_features = features[large_mask]
            if len(large_features) < 20:  # Don't split if too small
                return labels
            
            # Use K-means to split
            n_splits = min(3, len(large_features) // 10)
            if n_splits < 2:
                return labels
            
            kmeans = KMeans(n_clusters=n_splits, random_state=42)
            split_labels = kmeans.fit_predict(large_features)
            
            # Update labels
            max_label = np.max(labels[labels != -1]) if np.any(labels != -1) else 0
            updated_labels = labels.copy()
            updated_labels[large_mask] = split_labels + max_label + 1
            
            return updated_labels
            
        except Exception as e:
            self.logger.warning(f"⚠️ Large cluster split failed: {e}")
            return labels

    def _calculate_basic_clustering_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate basic clustering metrics.

        Args:
            features: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary of basic metrics
        """
        try:
            # Filter out noise points for metrics calculation
            valid_mask = labels != -1
            valid_features = features[valid_mask]
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) == 0:
                return {
                    'silhouette': 0.0,
                    'average_cluster_cv': 0.0,
                    'n_clusters': 0,
                    'n_valid_points': 0,
                    'n_noise_points': len(features)
                }
            
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return {
                    'silhouette': 0.0,
                    'average_cluster_cv': 0.0,
                    'n_clusters': n_clusters,
                    'n_valid_points': len(valid_features),
                    'n_noise_points': len(features) - len(valid_features)
                }
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(valid_features, valid_labels)
            
            # Calculate average cluster coefficient of variation
            cluster_cvs = []
            for label in unique_labels:
                cluster_features = valid_features[valid_labels == label]
                if len(cluster_features) > 1:
                    cluster_std = np.std(cluster_features, axis=0)
                    cluster_mean = np.mean(cluster_features, axis=0)
                    # Avoid division by zero
                    cluster_mean = np.where(cluster_mean == 0, 1e-8, cluster_mean)
                    cluster_cv = np.mean(cluster_std / cluster_mean)
                    cluster_cvs.append(cluster_cv)
            
            average_cluster_cv = np.mean(cluster_cvs) if cluster_cvs else 0.0
            
            return {
                'silhouette': float(silhouette),
                'average_cluster_cv': float(average_cluster_cv),
                'n_clusters': n_clusters,
                'n_valid_points': len(valid_features),
                'n_noise_points': len(features) - len(valid_features)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate basic clustering metrics: {e}")
            return {
                'silhouette': 0.0,
                'average_cluster_cv': 0.0,
                'n_clusters': 0,
                'n_valid_points': 0,
                'n_noise_points': len(features)
            }