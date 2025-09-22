"""
Base clustering interfaces and common functionality for HMM regime clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from datetime import datetime

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

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_OPS = True
except Exception:
    HARDWARE_OPS = False

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
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None


class BaseClusterer(ABC):
    """Base class for all clustering algorithms."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the clusterer.

        Args:
            config: Clustering configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_OPS:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPS:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

    @abstractmethod
    def cluster(self, features: np.ndarray) -> ClusteringResult:
        """Perform clustering on the given features.

        Args:
            features: Feature matrix to cluster

        Returns:
            ClusteringResult with clustering results
        """
        pass

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for clustering.

        Args:
            features: Raw feature matrix

        Returns:
            Prepared feature matrix
        """
        if features is None or features.size == 0:
            raise ValueError("Features cannot be empty")
        
        # Ensure features are numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        # Handle NaN values
        if np.any(np.isnan(features)):
            self.logger.warning("⚠️ Features contain NaN values, replacing with 0")
            features = np.nan_to_num(features, nan=0.0)
        
        # Handle infinite values
        if np.any(np.isinf(features)):
            self.logger.warning("⚠️ Features contain infinite values, replacing with 0")
            features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
        
        return features

    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cluster centroids.

        Args:
            X: Feature matrix
            labels: Cluster labels

        Returns:
            Tuple of (centroids, unique_labels)
        """
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

    def _assign_noise_to_nearest(self, X: np.ndarray, labels: np.ndarray, 
                                centers: Optional[np.ndarray] = None,
                                center_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Reassign noise points to nearest clusters.

        Args:
            X: Feature matrix
            labels: Cluster labels (may contain -1 for noise)
            centers: Cluster centers
            center_labels: Labels for cluster centers

        Returns:
            Updated labels with noise points reassigned
        """
        if labels is None or labels.size == 0:
            return labels
        
        # Find noise points
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        # Get valid clusters
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return labels
        
        # Compute centroids if not provided
        if centers is None:
            centers, center_labels = self._compute_centroids(X[~noise_mask], valid_labels)
        
        if centers is None or len(centers) == 0:
            return labels
        
        # Use matrix operations for distance calculation if available
        if self.matrix_ops is not None:
            try:
                noise_features = X[noise_mask]
                distances = self.matrix_ops.euclidean_distances(noise_features, centers)
                nearest_clusters = center_labels[np.argmin(distances, axis=1)]
                
                # Update labels
                updated_labels = labels.copy()
                updated_labels[noise_mask] = nearest_clusters
                return updated_labels
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations failed, falling back to CPU: {e}")
        
        # Fallback to CPU implementation
        noise_features = X[noise_mask]
        distances = np.sqrt(((noise_features[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
        nearest_clusters = center_labels[np.argmin(distances, axis=1)]
        
        # Update labels
        updated_labels = labels.copy()
        updated_labels[noise_mask] = nearest_clusters
        return updated_labels

    def _calculate_basic_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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
                    'n_clusters': 0,
                    'n_valid_points': 0,
                    'n_noise_points': len(features)
                }
            
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return {
                    'silhouette': 0.0,
                    'n_clusters': n_clusters,
                    'n_valid_points': len(valid_features),
                    'n_noise_points': len(features) - len(valid_features)
                }
            
            # Calculate silhouette score
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(valid_features, valid_labels)
            
            return {
                'silhouette': float(silhouette),
                'n_clusters': n_clusters,
                'n_valid_points': len(valid_features),
                'n_noise_points': len(features) - len(valid_features)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate basic metrics: {e}")
            return {
                'silhouette': 0.0,
                'n_clusters': 0,
                'n_valid_points': 0,
                'n_noise_points': len(features)
            }

    def _monitor_performance(self, operation_name: str) -> None:
        """Monitor performance of clustering operations.

        Args:
            operation_name: Name of the operation being monitored
        """
        if self.performance_monitor is not None:
            try:
                self.performance_monitor.start_monitoring(operation_name)
            except Exception as e:
                self.logger.warning(f"⚠️ Performance monitoring failed: {e}")

    def _stop_performance_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """Stop performance monitoring and get results.

        Args:
            operation_name: Name of the operation being monitored

        Returns:
            Performance monitoring results
        """
        if self.performance_monitor is not None:
            try:
                return self.performance_monitor.stop_monitoring(operation_name)
            except Exception as e:
                self.logger.warning(f"⚠️ Performance monitoring stop failed: {e}")
        return {}

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage.

        Returns:
            Memory usage information
        """
        if self.memory_manager is not None:
            try:
                return self.memory_manager.get_memory_usage()
            except Exception as e:
                self.logger.warning(f"⚠️ Memory check failed: {e}")
        return {}

    def _optimize_memory(self) -> None:
        """Optimize memory usage."""
        if self.memory_manager is not None:
            try:
                self.memory_manager.optimize_memory()
            except Exception as e:
                self.logger.warning(f"⚠️ Memory optimization failed: {e}")

    def _create_result(self, labels: np.ndarray, features: np.ndarray, 
                      execution_time: float, metadata: Dict[str, Any] = None) -> ClusteringResult:
        """Create a clustering result object.

        Args:
            labels: Cluster labels
            features: Feature matrix
            execution_time: Execution time in seconds
            metadata: Additional metadata

        Returns:
            ClusteringResult object
        """
        try:
            # Compute centroids
            centroids, unique_labels = self._compute_centroids(features, labels)
            
            # Calculate basic metrics
            basic_metrics = self._calculate_basic_metrics(features, labels)
            
            # Create statistics
            statistics = {
                'n_clusters': len(unique_labels),
                'n_points': len(features),
                'n_valid_points': basic_metrics['n_valid_points'],
                'n_noise_points': basic_metrics['n_noise_points'],
                'unique_labels': unique_labels.tolist()
            }
            
            # Create quality metrics
            quality_metrics = basic_metrics.copy()
            
            # Create validation
            validation = {
                'valid': basic_metrics['n_clusters'] > 0,
                'silhouette_threshold_met': basic_metrics['silhouette'] >= 0.2,
                'cluster_count_valid': 2 <= basic_metrics['n_clusters'] <= 25
            }
            
            # Create metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'hardware_acceleration': self.hardware_accelerator is not None,
                'matrix_operations': self.matrix_ops is not None
            })
            
            return ClusteringResult(
                labels=labels,
                cluster_centers=centroids,
                statistics=statistics,
                quality_metrics=quality_metrics,
                validation=validation,
                metadata=metadata,
                success=True,
                execution_time=execution_time,
                timestamp=metadata['timestamp']
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create clustering result: {e}")
            return ClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )