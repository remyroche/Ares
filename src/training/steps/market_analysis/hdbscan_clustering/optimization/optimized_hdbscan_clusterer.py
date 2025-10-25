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
    get_unified_vectorization_manager
)

# Import VectorizationConfig from the correct module
from src.feature_generation.utils.unified_vectorization_manager import VectorizationConfig

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
    metric: str = 'euclidean'  # 'euclidean', 'manhattan', 'precomputed'
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
                # Ensure only numeric data is passed to HDBSCAN
                numeric_features_df = features_df.select_dtypes(include=[np.number])
                if len(numeric_features_df.columns) < len(features_df.columns):
                    logger.warning(f"⚠️ Filtered out {len(features_df.columns) - len(numeric_features_df.columns)} non-numeric columns for HDBSCAN")
                
                # Convert to float64 and handle any remaining data type issues
                numeric_features_df = numeric_features_df.astype(np.float64)
                
                # Remove any infinite or NaN values
                numeric_features_df = numeric_features_df.replace([np.inf, -np.inf], np.nan)
                numeric_features_df = numeric_features_df.fillna(0)
                
                # Ensure all values are finite
                if not np.all(np.isfinite(numeric_features_df.values)):
                    logger.error("❌ Non-finite values found in features after cleaning")
                    raise ValueError("Non-finite values found in features")
                
                cluster_labels = clusterer.fit_predict(numeric_features_df)
                
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
