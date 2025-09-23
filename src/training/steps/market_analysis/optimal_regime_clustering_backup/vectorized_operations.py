"""
Vectorized Operations for HMM Clustering

This module provides optimized vectorized operations specifically designed
for clustering algorithms with GPU acceleration and memory efficiency.
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging
from contextlib import contextmanager

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        gpu_matrix_multiply,
        batch_matrix_multiply,
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorizedClusteringOperations:
    """
    Vectorized operations optimized for clustering algorithms.
    
    Provides:
    - Vectorized distance calculations
    - Optimized centroid computations
    - Batch processing for large datasets
    - GPU acceleration when available
    """

    def __init__(self, enable_gpu: bool = True, chunk_size: int = 10000):
        """Initialize vectorized operations."""
        self.enable_gpu = enable_gpu
        self.chunk_size = chunk_size
        self.matrix_ops = None
        self.vectorized_core = None
        
        # Initialize matrix operations
        self._init_matrix_operations()

    def _init_matrix_operations(self):
        """Initialize matrix operations components."""
        try:
            if MATRIX_OPS_AVAILABLE:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                logger.info("✅ Vectorized operations initialized")
            else:
                logger.warning("⚠️ Matrix operations not available")
        except Exception as e:
            logger.warning(f"Matrix operations initialization failed: {e}")

    def vectorized_euclidean_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized Euclidean distance calculation.
        
        Args:
            X: First set of points (n_samples_x, n_features)
            Y: Second set of points (n_samples_y, n_features)
            
        Returns:
            Distance matrix (n_samples_x, n_samples_y)
        """
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_euclidean_distance'):
                return self.matrix_ops.vectorized_euclidean_distance(X, Y)
            
            # Optimized numpy implementation using broadcasting
            # Compute ||X - Y||^2 = ||X||^2 + ||Y||^2 - 2*X*Y^T
            X_norm_squared = np.sum(X**2, axis=1, keepdims=True)
            Y_norm_squared = np.sum(Y**2, axis=1, keepdims=True)
            dot_product = np.dot(X, Y.T)
            
            # Broadcasting for efficient computation
            distances_squared = X_norm_squared + Y_norm_squared.T - 2 * dot_product
            
            # Ensure non-negative (numerical stability)
            distances_squared = np.maximum(distances_squared, 0)
            
            return np.sqrt(distances_squared)
            
        except Exception as e:
            logger.warning(f"Vectorized Euclidean distance failed: {e}")
            # Fallback to sklearn
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(X, Y)

    def vectorized_cosine_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized cosine distance calculation.
        
        Args:
            X: First set of points (n_samples_x, n_features)
            Y: Second set of points (n_samples_y, n_features)
            
        Returns:
            Distance matrix (n_samples_x, n_samples_y)
        """
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_cosine_distance'):
                return self.matrix_ops.vectorized_cosine_distance(X, Y)
            
            # Normalize vectors
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
            
            # Compute cosine similarities
            similarities = np.dot(X_norm, Y_norm.T)
            
            # Convert to distances
            return 1 - similarities
            
        except Exception as e:
            logger.warning(f"Vectorized cosine distance failed: {e}")
            from sklearn.metrics.pairwise import cosine_distances
            return cosine_distances(X, Y)

    def vectorized_mahalanobis_distance(self, X: np.ndarray, Y: np.ndarray, 
                                       cov: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Vectorized Mahalanobis distance calculation.
        
        Args:
            X: First set of points (n_samples_x, n_features)
            Y: Second set of points (n_samples_y, n_features)
            cov: Covariance matrix (if None, computed from X)
            
        Returns:
            Distance matrix (n_samples_x, n_samples_y)
        """
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_mahalanobis_distance'):
                return self.matrix_ops.vectorized_mahalanobis_distance(X, Y, cov)
            
            # Compute covariance matrix if not provided
            if cov is None:
                cov = np.cov(X.T)
            
            # Add regularization for numerical stability
            reg_param = 1e-6
            cov_reg = cov + reg_param * np.eye(cov.shape[0])
            
            # Compute inverse covariance
            try:
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_reg)
            
            # Compute Mahalanobis distances
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            distances_squared = np.sum(diff @ inv_cov * diff, axis=2)
            
            return np.sqrt(np.maximum(distances_squared, 0))
            
        except Exception as e:
            logger.warning(f"Vectorized Mahalanobis distance failed: {e}")
            from sklearn.metrics.pairwise import pairwise_distances
            return pairwise_distances(X, Y, metric='mahalanobis')

    def vectorized_centers(self, features: np.ndarray, labels: np.ndarray, 
                          unique_labels: np.ndarray) -> np.ndarray:
        """
        Vectorized cluster center calculation.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster labels (n_samples,)
            unique_labels: Unique cluster labels
            
        Returns:
            Cluster centers (n_clusters, n_features)
        """
        try:
            if self.matrix_ops and hasattr(self.matrix_ops, 'vectorized_centers'):
                return self.matrix_ops.vectorized_centers(features, labels, unique_labels)
            
            # Use matrix operations for efficient center calculation
            n_samples, n_features = features.shape
            n_clusters = len(unique_labels)
            
            # Create one-hot encoding for labels
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            label_indices = np.array([label_to_idx[label] for label in labels])
            
            # Create one-hot matrix
            onehot = np.zeros((n_samples, n_clusters), dtype=np.float64)
            onehot[np.arange(n_samples), label_indices] = 1.0
            
            # Compute cluster sizes
            cluster_sizes = np.sum(onehot, axis=0)
            
            # Compute weighted sums
            if self.matrix_ops and hasattr(self.matrix_ops, 'gpu_matrix_multiply'):
                weighted_sums = self.matrix_ops.gpu_matrix_multiply(onehot.T, features)
            else:
                weighted_sums = onehot.T @ features
            
            # Compute centers
            with np.errstate(divide='ignore', invalid='ignore'):
                centers = weighted_sums / np.maximum(cluster_sizes[:, np.newaxis], 1.0)
            
            return centers
            
        except Exception as e:
            logger.warning(f"Vectorized centers calculation failed: {e}")
            # Fallback to standard implementation
            centers = []
            for label in unique_labels:
                cluster_points = features[labels == label]
                centers.append(np.mean(cluster_points, axis=0))
            return np.array(centers)

    def batch_distance_calculation(self, X: np.ndarray, Y: np.ndarray, 
                                  metric: str = "euclidean", 
                                  batch_size: int = None) -> np.ndarray:
        """
        Batch distance calculation for large datasets.
        
        Args:
            X: First set of points
            Y: Second set of points
            metric: Distance metric
            batch_size: Batch size for processing
            
        Returns:
            Distance matrix
        """
        if batch_size is None:
            batch_size = self.chunk_size
        
        n_samples_x = X.shape[0]
        n_samples_y = Y.shape[0]
        
        # If dataset is small enough, compute directly
        if n_samples_x * n_samples_y < batch_size * batch_size:
            if metric == "euclidean":
                return self.vectorized_euclidean_distance(X, Y)
            elif metric == "cosine":
                return self.vectorized_cosine_distance(X, Y)
            elif metric == "mahalanobis":
                return self.vectorized_mahalanobis_distance(X, Y)
        
        # Process in batches
        distances = np.zeros((n_samples_x, n_samples_y))
        
        for i in range(0, n_samples_x, batch_size):
            end_i = min(i + batch_size, n_samples_x)
            X_batch = X[i:end_i]
            
            for j in range(0, n_samples_y, batch_size):
                end_j = min(j + batch_size, n_samples_y)
                Y_batch = Y[j:end_j]
                
                # Compute distances for this batch
                if metric == "euclidean":
                    batch_distances = self.vectorized_euclidean_distance(X_batch, Y_batch)
                elif metric == "cosine":
                    batch_distances = self.vectorized_cosine_distance(X_batch, Y_batch)
                elif metric == "mahalanobis":
                    batch_distances = self.vectorized_mahalanobis_distance(X_batch, Y_batch)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                distances[i:end_i, j:end_j] = batch_distances
        
        return distances

    def optimized_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Optimized silhouette score calculation using vectorized operations.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            
        Returns:
            Silhouette score
        """
        try:
            from sklearn.metrics import silhouette_score
            
            # For large datasets, use sampling
            n_samples = len(features)
            if n_samples > 10000:
                # Sample for efficiency
                sample_size = min(5000, n_samples)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                sample_features = features[indices]
                sample_labels = labels[indices]
                return silhouette_score(sample_features, sample_labels)
            else:
                return silhouette_score(features, labels)
                
        except Exception as e:
            logger.warning(f"Optimized silhouette score failed: {e}")
            return 0.0

    def cleanup(self):
        """Cleanup resources."""
        if self.matrix_ops and hasattr(self.matrix_ops, 'cleanup'):
            self.matrix_ops.cleanup()

# Factory function
def create_vectorized_operations(enable_gpu: bool = True, chunk_size: int = 10000) -> VectorizedClusteringOperations:
    """Create vectorized operations instance."""
    return VectorizedClusteringOperations(enable_gpu=enable_gpu, chunk_size=chunk_size)