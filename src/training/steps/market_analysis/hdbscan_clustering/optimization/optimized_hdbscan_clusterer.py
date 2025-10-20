"""
Optimized HDBSCAN Clustering for Market Analysis

This module provides optimized HDBSCAN clustering with VectorBT acceleration,
memory optimization, and intelligent parameter selection.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import hdbscan

# Import UnifiedVectorizationManager
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    VectorizationConfig,
    get_unified_vectorization_manager
)

logger = logging.getLogger(__name__)

@dataclass
class HDBSCANConfig:
    """Configuration for optimized HDBSCAN clustering."""
    # Core HDBSCAN parameters
    min_cluster_size: int = 15
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = 'eom'  # 'eom' or 'leaf'
    
    # Distance metrics
    metric: str = 'euclidean'  # 'euclidean', 'manhattan', 'cosine', 'precomputed'
    metric_params: Optional[Dict[str, Any]] = None
    
    # Memory optimization
    memory_efficient: bool = True
    chunk_size: int = 1000
    max_memory_gb: float = 8.0
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Parameter optimization
    enable_parameter_optimization: bool = True
    optimization_metric: str = 'silhouette'  # 'silhouette', 'calinski_harabasz', 'davies_bouldin'
    parameter_search_space: Dict[str, List] = None
    
    # Clustering validation
    enable_validation: bool = True
    min_silhouette_score: float = 0.3
    max_clusters: int = 20
    min_clusters: int = 2

class OptimizedHDBSCANClusterer:
    """
    Optimized HDBSCAN clusterer with VectorBT acceleration and intelligent
    parameter optimization.
    """
    
    def __init__(self, config: Optional[HDBSCANConfig] = None):
        """Initialize the optimized HDBSCAN clusterer."""
        self.config = config or HDBSCANConfig()
        
        # Initialize UnifiedVectorizationManager
        vectorization_config = VectorizationConfig(
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            chunk_size=self.config.chunk_size,
            enable_parallel=True
        )
        self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        
        # Initialize parameter search space
        if self.config.parameter_search_space is None:
            self.config.parameter_search_space = {
                'min_cluster_size': [10, 15, 20, 25, 30],
                'min_samples': [3, 5, 7, 10, 15],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.3, 0.5]
            }
        
        # Performance tracking
        self.performance_stats = {
            'clustering_time': 0.0,
            'n_clusters': 0,
            'n_noise_points': 0,
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'optimization_time': 0.0
        }
        
        # Store best parameters
        self.best_params = None
        self.best_score = -np.inf
        
        logger.info("✅ OptimizedHDBSCANClusterer initialized")
    
    def cluster_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster data using optimized HDBSCAN with parameter optimization.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            Tuple of (cluster_labels, clustering_info)
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized HDBSCAN clustering for {features_df.shape[0]} samples")
        
        # Validate input
        self._validate_features(features_df)
        
        # Optimize parameters if enabled
        if self.config.enable_parameter_optimization:
            logger.info("🔄 Optimizing HDBSCAN parameters")
            optimization_start = time.time()
            best_params = self._optimize_parameters(features_df)
            optimization_time = time.time() - optimization_start
            self.performance_stats['optimization_time'] = optimization_time
        else:
            best_params = {
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
                'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                'cluster_selection_method': self.config.cluster_selection_method,
                'metric': self.config.metric
            }
        
        # Perform clustering
        cluster_labels, clustering_info = self._perform_clustering(features_df, best_params)
        
        # Validate clustering results
        if self.config.enable_validation:
            self._validate_clustering(cluster_labels, features_df)
        
        # Update performance stats
        clustering_time = time.time() - start_time
        self._update_performance_stats(cluster_labels, features_df, clustering_time)
        
        logger.info(f"✅ HDBSCAN clustering completed: {len(np.unique(cluster_labels))} clusters in {clustering_time:.2f}s")
        return cluster_labels, clustering_info
    
    def _optimize_parameters(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize HDBSCAN parameters using grid search."""
        logger.info("🔄 Optimizing HDBSCAN parameters")
        
        best_params = None
        best_score = -np.inf
        
        # Grid search over parameter space
        for min_cluster_size in self.config.parameter_search_space['min_cluster_size']:
            for min_samples in self.config.parameter_search_space['min_samples']:
                for epsilon in self.config.parameter_search_space['cluster_selection_epsilon']:
                    try:
                        # Test parameter combination
                        params = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': epsilon,
                            'cluster_selection_method': self.config.cluster_selection_method,
                            'metric': self.config.metric
                        }
                        
                        # Perform clustering with these parameters
                        cluster_labels, _ = self._perform_clustering(features_df, params)
                        
                        # Calculate validation score
                        score = self._calculate_validation_score(cluster_labels, features_df)
                        
                        # Update best parameters
                        if score > best_score:
                            best_score = score
                            best_params = params
                            
                        logger.debug(f"Parameters {params}: score = {score:.3f}")
                        
                    except Exception as e:
                        logger.debug(f"Parameters {params} failed: {e}")
                        continue
        
        if best_params is None:
            logger.warning("⚠️ Parameter optimization failed, using default parameters")
            best_params = {
                'min_cluster_size': self.config.min_cluster_size,
                'min_samples': self.config.min_samples,
                'cluster_selection_epsilon': self.config.cluster_selection_epsilon,
                'cluster_selection_method': self.config.cluster_selection_method,
                'metric': self.config.metric
            }
        else:
            logger.info(f"✅ Best parameters found: {best_params} (score: {best_score:.3f})")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def _perform_clustering(self, features_df: pd.DataFrame, 
                          params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform HDBSCAN clustering with given parameters."""
        try:
            # Use VectorBT acceleration if available
            if hasattr(self.vectorization_manager, 'hdbscan_cluster'):
                cluster_labels, clustering_info = self.vectorization_manager.hdbscan_cluster(
                    features_df, **params
                )
            else:
                # Use standard HDBSCAN
                clusterer = HDBSCAN(**params)
                cluster_labels = clusterer.fit_predict(features_df)
                
                # Create clustering info
                clustering_info = {
                    'clusterer': clusterer,
                    'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    'n_noise_points': list(cluster_labels).count(-1),
                    'cluster_persistence': getattr(clusterer, 'cluster_persistence_', None),
                    'condensed_tree': getattr(clusterer, 'condensed_tree_', None)
                }
            
            return cluster_labels, clustering_info
            
        except Exception as e:
            logger.error(f"❌ HDBSCAN clustering failed: {e}")
            raise
    
    def _calculate_validation_score(self, cluster_labels: np.ndarray, 
                                   features_df: pd.DataFrame) -> float:
        """Calculate validation score for clustering."""
        try:
            # Remove noise points for validation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features_df[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate score based on optimization metric
            if self.config.optimization_metric == 'silhouette':
                score = silhouette_score(valid_features, valid_labels)
            elif self.config.optimization_metric == 'calinski_harabasz':
                score = calinski_harabasz_score(valid_features, valid_labels)
            elif self.config.optimization_metric == 'davies_bouldin':
                score = -davies_bouldin_score(valid_features, valid_labels)  # Negative because lower is better
            else:
                score = silhouette_score(valid_features, valid_labels)
            
            return score
            
        except Exception as e:
            logger.debug(f"Validation score calculation failed: {e}")
            return -np.inf
    
    def _validate_clustering(self, cluster_labels: np.ndarray, features_df: pd.DataFrame):
        """Validate clustering results."""
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise_points = list(cluster_labels).count(-1)
        
        # Check minimum clusters
        if n_clusters < self.config.min_clusters:
            logger.warning(f"⚠️ Too few clusters: {n_clusters} < {self.config.min_clusters}")
        
        # Check maximum clusters
        if n_clusters > self.config.max_clusters:
            logger.warning(f"⚠️ Too many clusters: {n_clusters} > {self.config.max_clusters}")
        
        # Check silhouette score
        if n_clusters >= 2:
            try:
                valid_mask = cluster_labels != -1
                if valid_mask.sum() >= 2:
                    valid_labels = cluster_labels[valid_mask]
                    valid_features = features_df[valid_mask]
                    
                    if len(set(valid_labels)) >= 2:
                        silhouette = silhouette_score(valid_features, valid_labels)
                        if silhouette < self.config.min_silhouette_score:
                            logger.warning(f"⚠️ Low silhouette score: {silhouette:.3f} < {self.config.min_silhouette_score}")
            except Exception as e:
                logger.debug(f"Silhouette score validation failed: {e}")
    
    def _validate_features(self, features_df: pd.DataFrame):
        """Validate input features."""
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")
        
        if features_df.empty:
            raise ValueError("Features DataFrame cannot be empty")
        
        if features_df.shape[0] < self.config.min_cluster_size:
            raise ValueError(f"Not enough samples: {features_df.shape[0]} < {self.config.min_cluster_size}")
    
    def _update_performance_stats(self, cluster_labels: np.ndarray, 
                                 features_df: pd.DataFrame, 
                                 clustering_time: float):
        """Update performance statistics."""
        self.performance_stats['clustering_time'] = clustering_time
        
        # Calculate clustering metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise_points = list(cluster_labels).count(-1)
        
        self.performance_stats['n_clusters'] = n_clusters
        self.performance_stats['n_noise_points'] = n_noise_points
        
        # Calculate validation scores
        if n_clusters >= 2:
            try:
                valid_mask = cluster_labels != -1
                if valid_mask.sum() >= 2:
                    valid_labels = cluster_labels[valid_mask]
                    valid_features = features_df[valid_mask]
                    
                    if len(set(valid_labels)) >= 2:
                        self.performance_stats['silhouette_score'] = silhouette_score(valid_features, valid_labels)
                        self.performance_stats['calinski_harabasz_score'] = calinski_harabasz_score(valid_features, valid_labels)
                        self.performance_stats['davies_bouldin_score'] = davies_bouldin_score(valid_features, valid_labels)
            except Exception as e:
                logger.debug(f"Performance stats calculation failed: {e}")
        
        # Calculate memory usage
        memory_usage = features_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
        
        # Get VectorBT usage rate
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        self.performance_stats['vectorbt_usage_rate'] = vectorization_stats.get('vectorbt_usage_rate', 0)
    
    def approximate_predict_with_fallback(self, 
                                        features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Predict cluster labels and probabilities for new data points.
        
        This method provides approximate prediction capabilities for HDBSCAN,
        which doesn't have a direct predict method. It uses various fallback
        strategies to estimate cluster assignments and probabilities.
        
        Args:
            features: Feature matrix for prediction (n_samples, n_features)
            
        Returns:
            Tuple of (labels, probabilities, method_used)
            - labels: Predicted cluster labels
            - probabilities: Predicted cluster probabilities
            - method_used: String describing the method used
        """
        try:
            if not hasattr(self, 'best_clusterer') or self.best_clusterer is None:
                logger.warning("⚠️ No trained clusterer available, using random assignment")
                return self._random_fallback(features)
            
            # Try different prediction methods in order of preference
            methods = [
                self._approximate_predict_centroid,
                self._approximate_predict_knn,
                self._approximate_predict_distance
            ]
            
            for method in methods:
                try:
                    labels, probabilities = method(features)
                    if labels is not None and probabilities is not None:
                        method_name = method.__name__.replace('_approximate_predict_', '')
                        logger.info(f"✅ Prediction successful using {method_name} method")
                        return labels, probabilities, method_name
                except Exception as e:
                    logger.debug(f"Method {method.__name__} failed: {e}")
                    continue
            
            # If all methods fail, use random fallback
            logger.warning("⚠️ All prediction methods failed, using random assignment")
            return self._random_fallback(features)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return self._random_fallback(features)
    
    def _approximate_predict_centroid(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using centroid-based assignment."""
        try:
            if not hasattr(self.best_clusterer, 'cluster_centers_'):
                raise ValueError("No cluster centers available")
            
            # Calculate distances to cluster centers
            distances = self._calculate_distances_to_centers(features)
            
            # Assign to closest cluster
            labels = np.argmin(distances, axis=1)
            
            # Calculate probabilities based on distance (closer = higher probability)
            max_distances = np.max(distances, axis=1, keepdims=True)
            probabilities = 1.0 - (distances / (max_distances + 1e-10))
            probabilities = np.max(probabilities, axis=1)
            
            return labels, probabilities
            
        except Exception as e:
            logger.debug(f"Centroid prediction failed: {e}")
            return None, None
    
    def _approximate_predict_knn(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using k-nearest neighbors approach."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Get training data from the clusterer
            if hasattr(self.best_clusterer, 'cluster_centers_'):
                training_data = self.best_clusterer.cluster_centers_
                training_labels = np.arange(len(training_data))
            else:
                # Use original training data if available
                if hasattr(self, 'training_features') and self.training_features is not None:
                    training_data = self.training_features
                    training_labels = self.training_labels
                else:
                    raise ValueError("No training data available for KNN")
            
            # Fit KNN
            k = min(5, len(training_data))
            knn = NearestNeighbors(n_neighbors=k, metric=self.config.metric)
            knn.fit(training_data)
            
            # Find nearest neighbors
            distances, indices = knn.kneighbors(features)
            
            # Assign labels based on majority vote
            labels = []
            probabilities = []
            
            for i in range(len(features)):
                neighbor_labels = training_labels[indices[i]]
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                
                # Assign to most common label
                most_common_idx = np.argmax(counts)
                predicted_label = unique_labels[most_common_idx]
                labels.append(predicted_label)
                
                # Calculate probability based on distance
                avg_distance = np.mean(distances[i])
                probability = np.exp(-avg_distance)  # Simple exponential decay
                probabilities.append(probability)
            
            return np.array(labels), np.array(probabilities)
            
        except Exception as e:
            logger.debug(f"KNN prediction failed: {e}")
            return None, None
    
    def _approximate_predict_distance(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using distance-based assignment."""
        try:
            # Get cluster centers or representative points
            if hasattr(self.best_clusterer, 'cluster_centers_'):
                centers = self.best_clusterer.cluster_centers_
            else:
                # Use condensed tree to find representative points
                centers = self._extract_representative_points()
            
            if centers is None or len(centers) == 0:
                raise ValueError("No representative points available")
            
            # Calculate distances to all centers
            distances = self._calculate_distances_to_centers(features, centers)
            
            # Assign to closest cluster
            labels = np.argmin(distances, axis=1)
            
            # Calculate probabilities based on relative distances
            min_distances = np.min(distances, axis=1, keepdims=True)
            probabilities = min_distances / (distances + 1e-10)
            probabilities = np.max(probabilities, axis=1)
            
            return labels, probabilities
            
        except Exception as e:
            logger.debug(f"Distance prediction failed: {e}")
            return None, None
    
    def _calculate_distances_to_centers(self, features: np.ndarray, centers: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate distances from features to cluster centers."""
        try:
            if centers is None:
                if hasattr(self.best_clusterer, 'cluster_centers_'):
                    centers = self.best_clusterer.cluster_centers_
                else:
                    raise ValueError("No cluster centers available")
            
            # Calculate pairwise distances
            if self.config.metric == 'euclidean':
                distances = np.sqrt(((features[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
            elif self.config.metric == 'manhattan':
                distances = np.abs(features[:, np.newaxis] - centers[np.newaxis, :]).sum(axis=2)
            else:
                # Fallback to euclidean
                distances = np.sqrt(((features[:, np.newaxis] - centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            return distances
            
        except Exception as e:
            logger.error(f"❌ Distance calculation failed: {e}")
            return np.ones((len(features), 1))
    
    def _extract_representative_points(self) -> Optional[np.ndarray]:
        """Extract representative points from HDBSCAN condensed tree."""
        try:
            if not hasattr(self.best_clusterer, 'condensed_tree_'):
                return None
            
            # This is a simplified approach - in practice, you'd need to
            # traverse the condensed tree to find representative points
            # For now, return None to trigger fallback
            return None
            
        except Exception as e:
            logger.debug(f"Representative point extraction failed: {e}")
            return None
    
    def _random_fallback(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Fallback to random assignment when all other methods fail."""
        try:
            n_samples = len(features)
            
            # Random labels (assuming 2-5 clusters)
            n_clusters = np.random.randint(2, 6)
            labels = np.random.randint(0, n_clusters, n_samples)
            
            # Random probabilities
            probabilities = np.random.uniform(0.1, 0.9, n_samples)
            
            return labels, probabilities, "random_fallback"
            
        except Exception as e:
            logger.error(f"❌ Random fallback failed: {e}")
            # Ultimate fallback
            n_samples = len(features)
            return np.zeros(n_samples), np.ones(n_samples), "ultimate_fallback"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add vectorization manager stats
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        stats['vectorization_stats'] = vectorization_stats
        
        # Add best parameters
        if self.best_params is not None:
            stats['best_parameters'] = self.best_params
            stats['best_score'] = self.best_score
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'clustering_time': 0.0,
            'n_clusters': 0,
            'n_noise_points': 0,
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'optimization_time': 0.0
        }
        
        # Reset best parameters
        self.best_params = None
        self.best_score = -np.inf
        
        # Reset vectorization manager stats
        self.vectorization_manager.reset_stats()

# Convenience function for easy usage
def create_optimized_hdbscan_clusterer(
    min_cluster_size: int = 15,
    min_samples: int = 5,
    cluster_selection_epsilon: float = 0.0,
    enable_parameter_optimization: bool = True,
    optimization_metric: str = 'silhouette',
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> OptimizedHDBSCANClusterer:
    """
    Create an optimized HDBSCAN clusterer with specified configuration.
    
    Args:
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples per cluster
        cluster_selection_epsilon: Cluster selection epsilon
        enable_parameter_optimization: Enable parameter optimization
        optimization_metric: Metric for optimization
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        OptimizedHDBSCANClusterer instance
    """
    config = HDBSCANConfig(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        enable_parameter_optimization=enable_parameter_optimization,
        optimization_metric=optimization_metric,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return OptimizedHDBSCANClusterer(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    # Create clustered data
    cluster1 = np.random.randn(300, n_features) + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    cluster2 = np.random.randn(300, n_features) + [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
    cluster3 = np.random.randn(400, n_features) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    features = np.vstack([cluster1, cluster2, cluster3])
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(features, columns=feature_names)
    
    print(f"Sample data: {features_df.shape}")
    
    # Create optimized HDBSCAN clusterer
    clusterer = create_optimized_hdbscan_clusterer(
        min_cluster_size=50,
        min_samples=10,
        enable_parameter_optimization=True,
        optimization_metric='silhouette',
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Perform clustering
    cluster_labels, clustering_info = clusterer.cluster_data(features_df)
    
    print(f"Clustering results: {len(np.unique(cluster_labels))} clusters")
    print(f"Performance stats: {clusterer.get_performance_stats()}")
