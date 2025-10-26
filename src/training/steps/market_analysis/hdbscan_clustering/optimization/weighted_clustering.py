"""
Weighted Clustering Module

This module implements weighted clustering techniques to balance regime sizes
and improve cluster quality by giving more importance to certain samples or features.

Key Features:
- Sample weighting based on regime importance
- Feature weighting for better discrimination
- Balanced clustering with size constraints
- Integration with HDBSCAN and other clustering algorithms
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class WeightedClusteringConfig:
    """Configuration for weighted clustering."""
    target_cluster_count: int = 6
    min_cluster_size: float = 0.03  # 3% of data
    max_cluster_size: float = 0.15  # 15% of data
    balance_weight: float = 0.5  # Weight for cluster balance
    quality_weight: float = 0.3  # Weight for clustering quality
    temporal_weight: float = 0.2  # Weight for temporal consistency
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    random_state: int = 42

class WeightedClustering:
    """
    Weighted clustering system for balanced regime detection.
    
    This class implements various weighting strategies to achieve balanced
    cluster sizes while maintaining clustering quality.
    """
    
    def __init__(self, config: Optional[WeightedClusteringConfig] = None):
        """Initialize the weighted clustering system."""
        self.config = config or WeightedClusteringConfig()
        self.sample_weights = None
        self.feature_weights = None
        self.cluster_centers = None
        self.cluster_labels = None
        
    def fit_predict(self, X: pd.DataFrame, initial_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit weighted clustering and return cluster labels.
        
        Args:
            X: Feature matrix
            initial_labels: Initial cluster labels (optional)
            
        Returns:
            Cluster labels
        """
        try:
            logger.info("Starting weighted clustering...")
            
            # Calculate sample weights
            self.sample_weights = self._calculate_sample_weights(X, initial_labels)
            
            # Calculate feature weights
            self.feature_weights = self._calculate_feature_weights(X, initial_labels)
            
            # Apply weighted clustering
            self.cluster_labels = self._apply_weighted_clustering(X)
            
            # Post-process for balance
            self.cluster_labels = self._balance_clusters(X, self.cluster_labels)
            
            logger.info(f"Weighted clustering completed: {len(np.unique(self.cluster_labels))} clusters")
            return self.cluster_labels
            
        except Exception as e:
            logger.error(f"Weighted clustering failed: {e}")
            # Fallback to simple clustering
            return self._fallback_clustering(X)
    
    def _calculate_sample_weights(self, X: pd.DataFrame, initial_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate sample weights based on regime importance and balance."""
        try:
            n_samples = len(X)
            base_weights = np.ones(n_samples)
            
            if initial_labels is not None:
                # Weight samples based on current cluster sizes
                unique_labels, counts = np.unique(initial_labels, return_counts=True)
                non_noise_mask = unique_labels != -1
                cluster_labels = unique_labels[non_noise_mask]
                cluster_sizes = counts[non_noise_mask]
                
                if len(cluster_labels) > 0:
                    # Calculate target cluster size
                    target_size = n_samples / self.config.target_cluster_count
                    
                    # Weight samples inversely to their cluster size
                    for i, label in enumerate(cluster_labels):
                        cluster_mask = initial_labels == label
                        cluster_size = np.sum(cluster_mask)
                        
                        if cluster_size > 0:
                            # Smaller clusters get higher weights
                            weight_factor = target_size / cluster_size
                            base_weights[cluster_mask] *= weight_factor
            
            # Normalize weights
            base_weights = base_weights / np.mean(base_weights)
            
            # Apply temporal weighting if data has temporal structure
            if hasattr(X, 'index') and len(X.index) > 1:
                temporal_weights = self._calculate_temporal_weights(X)
                base_weights *= temporal_weights
            
            # Clip extreme weights
            base_weights = np.clip(base_weights, 0.1, 10.0)
            
            return base_weights
            
        except Exception as e:
            logger.warning(f"Error calculating sample weights: {e}")
            return np.ones(len(X))
    
    def _calculate_feature_weights(self, X: pd.DataFrame, initial_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate feature weights based on discriminative power."""
        try:
            n_features = X.shape[1]
            base_weights = np.ones(n_features)
            
            if initial_labels is not None and len(np.unique(initial_labels)) > 1:
                # Calculate feature importance based on cluster separation
                for i, feature in enumerate(X.columns):
                    feature_values = X[feature].values
                    
                    # Calculate between-cluster variance
                    unique_labels = np.unique(initial_labels)
                    non_noise_labels = unique_labels[unique_labels != -1]
                    
                    if len(non_noise_labels) > 1:
                        cluster_means = []
                        for label in non_noise_labels:
                            cluster_mask = initial_labels == label
                            if np.sum(cluster_mask) > 0:
                                cluster_means.append(np.mean(feature_values[cluster_mask]))
                        
                        if len(cluster_means) > 1:
                            between_var = np.var(cluster_means)
                            within_var = np.var(feature_values)
                            
                            if within_var > 0:
                                # F-ratio: higher is better
                                f_ratio = between_var / within_var
                                base_weights[i] = min(10.0, max(0.1, f_ratio))
            
            # Normalize feature weights
            base_weights = base_weights / np.mean(base_weights)
            
            return base_weights
            
        except Exception as e:
            logger.warning(f"Error calculating feature weights: {e}")
            return np.ones(X.shape[1])
    
    def _calculate_temporal_weights(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate temporal weights for time series data."""
        try:
            n_samples = len(X)
            temporal_weights = np.ones(n_samples)
            
            # Weight recent samples more heavily
            if hasattr(X.index, 'to_pydatetime'):
                # If index is datetime, weight by recency
                max_time = X.index.max()
                time_diffs = (max_time - X.index).total_seconds()
                max_diff = time_diffs.max()
                
                if max_diff > 0:
                    # Recent samples get higher weights
                    temporal_weights = 1.0 + (max_diff - time_diffs) / max_diff
            else:
                # If no datetime index, weight by position (recent = higher index)
                temporal_weights = 1.0 + np.arange(n_samples) / n_samples
            
            return temporal_weights
            
        except Exception as e:
            logger.warning(f"Error calculating temporal weights: {e}")
            return np.ones(len(X))
    
    def _apply_weighted_clustering(self, X: pd.DataFrame) -> np.ndarray:
        """Apply weighted clustering algorithm."""
        try:
            # Convert to numpy array
            X_array = X.values
            
            # Apply feature weights
            if self.feature_weights is not None:
                X_weighted = X_array * self.feature_weights
            else:
                X_weighted = X_array
            
            # Use weighted K-means as base clustering
            kmeans = KMeans(
                n_clusters=self.config.target_cluster_count,
                random_state=self.config.random_state,
                n_init=10,
                max_iter=self.config.max_iterations
            )
            
            # Apply sample weights by duplicating samples
            if self.sample_weights is not None:
                X_weighted, sample_indices = self._apply_sample_weights(X_weighted, self.sample_weights)
                cluster_labels = kmeans.fit_predict(X_weighted)
                
                # Map back to original samples
                original_labels = np.zeros(len(X))
                for i, idx in enumerate(sample_indices):
                    original_labels[idx] = cluster_labels[i]
                
                return original_labels
            else:
                return kmeans.fit_predict(X_weighted)
                
        except Exception as e:
            logger.error(f"Error in weighted clustering: {e}")
            return self._fallback_clustering(X)
    
    def _apply_sample_weights(self, X: np.ndarray, sample_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sample weights by duplicating samples."""
        try:
            # Round weights to integers for duplication
            int_weights = np.round(sample_weights * 10).astype(int)
            int_weights = np.clip(int_weights, 1, 50)  # Cap at 50x duplication
            
            # Create weighted dataset
            weighted_samples = []
            sample_indices = []
            
            for i, weight in enumerate(int_weights):
                for _ in range(weight):
                    weighted_samples.append(X[i])
                    sample_indices.append(i)
            
            return np.array(weighted_samples), np.array(sample_indices)
            
        except Exception as e:
            logger.warning(f"Error applying sample weights: {e}")
            return X, np.arange(len(X))
    
    def _balance_clusters(self, X: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        """Balance cluster sizes using iterative refinement."""
        try:
            n_samples = len(X)
            target_size = n_samples / self.config.target_cluster_count
            min_size = int(n_samples * self.config.min_cluster_size)
            max_size = int(n_samples * self.config.max_cluster_size)
            
            balanced_labels = labels.copy()
            max_iterations = 10
            
            for iteration in range(max_iterations):
                unique_labels, counts = np.unique(balanced_labels, return_counts=True)
                non_noise_labels = unique_labels[unique_labels != -1]
                cluster_sizes = counts[unique_labels != -1]
                
                if len(non_noise_labels) == 0:
                    break
                
                # Check if clusters are balanced
                balanced = True
                for size in cluster_sizes:
                    if size < min_size or size > max_size:
                        balanced = False
                        break
                
                if balanced:
                    break
                
                # Balance clusters
                balanced_labels = self._iterative_balance(
                    X, balanced_labels, non_noise_labels, cluster_sizes, 
                    target_size, min_size, max_size
                )
            
            return balanced_labels
            
        except Exception as e:
            logger.warning(f"Error balancing clusters: {e}")
            return labels
    
    def _iterative_balance(self, X: pd.DataFrame, labels: np.ndarray, 
                          cluster_labels: np.ndarray, cluster_sizes: np.ndarray,
                          target_size: float, min_size: int, max_size: int) -> np.ndarray:
        """Iteratively balance cluster sizes."""
        try:
            new_labels = labels.copy()
            
            # Handle clusters that are too small
            small_clusters = cluster_labels[cluster_sizes < min_size]
            for small_cluster in small_clusters:
                # Find the most similar larger cluster
                best_target = self._find_merge_target(small_cluster, cluster_labels, cluster_sizes, X, labels)
                if best_target is not None:
                    small_mask = new_labels == small_cluster
                    new_labels[small_mask] = best_target
            
            # Handle clusters that are too large
            large_clusters = cluster_labels[cluster_sizes > max_size]
            for large_cluster in large_clusters:
                # Split the large cluster
                split_labels = self._split_cluster(large_cluster, X, new_labels, target_size)
                if len(split_labels) > 1:
                    # Update labels with split results
                    large_mask = new_labels == large_cluster
                    for i, split_label in enumerate(split_labels):
                        if i == 0:
                            # Keep original cluster ID for first split
                            continue
                        else:
                            # Assign new cluster ID
                            new_cluster_id = np.max(cluster_labels) + i
                            split_mask = split_label
                            new_labels[large_mask & split_mask] = new_cluster_id
            
            return new_labels
            
        except Exception as e:
            logger.warning(f"Error in iterative balance: {e}")
            return labels
    
    def _find_merge_target(self, small_cluster: int, cluster_labels: np.ndarray, 
                          cluster_sizes: np.ndarray, X: pd.DataFrame, 
                          labels: np.ndarray) -> Optional[int]:
        """Find the best cluster to merge a small cluster with."""
        try:
            # Get clusters that are large enough to accept the small cluster
            suitable_clusters = cluster_labels[cluster_sizes >= 20]
            
            if len(suitable_clusters) == 0:
                return cluster_labels[np.argmax(cluster_sizes)]
            
            # Find the most similar cluster
            small_mask = labels == small_cluster
            small_center = X[small_mask].mean()
            
            best_target = None
            best_similarity = -1
            
            for target_cluster in suitable_clusters:
                target_mask = labels == target_cluster
                target_center = X[target_mask].mean()
                
                # Calculate similarity (cosine similarity)
                similarity = np.dot(small_center, target_center) / (
                    np.linalg.norm(small_center) * np.linalg.norm(target_center) + 1e-8
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target = target_cluster
            
            return best_target
            
        except Exception as e:
            logger.warning(f"Error finding merge target: {e}")
            return cluster_labels[np.argmax(cluster_sizes)]
    
    def _split_cluster(self, large_cluster: int, X: pd.DataFrame, 
                      labels: np.ndarray, target_size: float) -> List[np.ndarray]:
        """Split a large cluster into smaller ones."""
        try:
            large_mask = labels == large_cluster
            large_indices = np.where(large_mask)[0]
            
            if len(large_indices) <= target_size:
                return [large_mask]
            
            # Use K-means to split the cluster
            cluster_data = X[large_mask]
            n_splits = max(2, int(len(large_indices) / target_size))
            
            kmeans = KMeans(n_clusters=n_splits, random_state=self.config.random_state)
            split_labels = kmeans.fit_predict(cluster_data)
            
            # Convert to boolean masks
            split_masks = []
            for i in range(n_splits):
                split_mask = np.zeros(len(labels), dtype=bool)
                split_indices = large_indices[split_labels == i]
                split_mask[split_indices] = True
                split_masks.append(split_mask)
            
            return split_masks
            
        except Exception as e:
            logger.warning(f"Error splitting cluster: {e}")
            return [labels == large_cluster]
    
    def _fallback_clustering(self, X: pd.DataFrame) -> np.ndarray:
        """Fallback to simple K-means clustering."""
        try:
            kmeans = KMeans(
                n_clusters=self.config.target_cluster_count,
                random_state=self.config.random_state
            )
            return kmeans.fit_predict(X.values)
        except Exception as e:
            logger.error(f"Fallback clustering failed: {e}")
            return np.zeros(len(X), dtype=int)
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available."""
        return self.cluster_centers
    
    def get_sample_weights(self) -> Optional[np.ndarray]:
        """Get sample weights used in clustering."""
        return self.sample_weights
    
    def get_feature_weights(self) -> Optional[np.ndarray]:
        """Get feature weights used in clustering."""
        return self.feature_weights


def create_weighted_clustering(config: Optional[WeightedClusteringConfig] = None) -> WeightedClustering:
    """Factory function to create a WeightedClustering instance."""
    return WeightedClustering(config)
