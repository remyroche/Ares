"""
HDBSCAN Clustering Integration

This module provides integration between HDBSCAN clustering features and the clustering task.
It ensures 50-100 features are properly selected and optimized for density-based clustering.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Import feature categorization
from .regime_feature_categorization import FeatureUseCase, get_hdbscan_features
from .clustering_features import ClusteringIntegration, ClusteringFeatureConfig
from .feature_task_integration import FeatureTaskIntegrator, MLTask

# Import HDBSCAN if available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available. Install with: pip install hdbscan")


class HDBSCANClusteringIntegration:
    """
    HDBSCAN Clustering Integration.
    
    Provides 50-100 features optimized for density-based clustering using HDBSCAN.
    """
    
    def __init__(self, 
                 min_features: int = 50,
                 max_features: int = 100,
                 clustering_config: Optional[ClusteringFeatureConfig] = None):
        self.min_features = min_features
        self.max_features = max_features
        self.clustering_config = clustering_config or ClusteringFeatureConfig()
        
        # Initialize feature integrator
        self.feature_integrator = FeatureTaskIntegrator()
        
        # Initialize clustering feature generator
        self.clustering_generator = ClusteringIntegration(self.clustering_config)
    
    def get_clustering_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get features optimized for HDBSCAN clustering.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing features and metadata
        """
        # Get features from the task integrator
        feature_result = self.feature_integrator.get_features_for_task(
            MLTask.HDBSCAN_CLUSTERING, data
        )
        
        # Generate actual clustering features
        clustering_features = self.clustering_generator.generate_features(data)
        
        # Ensure we have the right number of features
        feature_names = list(clustering_features.keys())
        if len(feature_names) > self.max_features:
            # Select top features by variance (most informative)
            feature_variances = {}
            for name, values in clustering_features.items():
                feature_variances[name] = np.var(values)
            
            # Sort by variance and select top features
            sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [name for name, _ in sorted_features[:self.max_features]]
            
            # Filter clustering features
            filtered_features = {name: clustering_features[name] for name in selected_features}
            clustering_features = filtered_features
            feature_names = selected_features
        
        # Ensure minimum features
        if len(feature_names) < self.min_features:
            warnings.warn(f"Only {len(feature_names)} features available, minimum is {self.min_features}")
        
        return {
            'features': clustering_features,
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'target_range': (self.min_features, self.max_features),
            'clustering_optimized': True,
            'description': 'Features optimized for HDBSCAN density-based clustering'
        }
    
    def prepare_data_for_clustering(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for HDBSCAN clustering.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Get clustering features
        feature_result = self.get_clustering_features(data)
        features = feature_result['features']
        feature_names = feature_result['feature_names']
        
        # Convert to numpy array
        feature_matrix = np.column_stack([features[name] for name in feature_names])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_matrix, feature_names
    
    def cluster_with_hdbscan(self, data: pd.DataFrame, 
                           min_cluster_size: int = 5,
                           min_samples: int = 3,
                           cluster_selection_epsilon: float = 0.0) -> Dict[str, Any]:
        """
        Perform HDBSCAN clustering on the data.
        
        Args:
            data: Market data DataFrame
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            cluster_selection_epsilon: Cluster selection epsilon for HDBSCAN
            
        Returns:
            Dictionary containing clustering results
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        # Prepare data
        feature_matrix, feature_names = self.prepare_data_for_clustering(data)
        
        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        
        cluster_labels = clusterer.fit_predict(feature_matrix)
        
        # Calculate clustering metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        return {
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'feature_names': feature_names,
            'feature_matrix': feature_matrix,
            'clusterer': clusterer,
            'clustering_parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': cluster_selection_epsilon
            }
        }
    
    def analyze_clustering_quality(self, clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of HDBSCAN clustering.
        
        Args:
            clustering_result: Result from cluster_with_hdbscan
            
        Returns:
            Dictionary containing quality metrics
        """
        cluster_labels = clustering_result['cluster_labels']
        feature_matrix = clustering_result['feature_matrix']
        
        # Basic statistics
        n_clusters = clustering_result['n_clusters']
        n_noise = clustering_result['n_noise']
        total_samples = len(cluster_labels)
        
        # Cluster size distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Noise ratio
        noise_ratio = n_noise / total_samples if total_samples > 0 else 0
        
        # Average cluster size (excluding noise)
        non_noise_clusters = [size for label, size in cluster_sizes.items() if label != -1]
        avg_cluster_size = np.mean(non_noise_clusters) if non_noise_clusters else 0
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'total_samples': total_samples,
            'noise_ratio': noise_ratio,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': avg_cluster_size,
            'clustering_quality': 'good' if noise_ratio < 0.3 and n_clusters > 1 else 'poor'
        }
    
    def get_feature_importance_for_clustering(self, data: pd.DataFrame, 
                                            clustering_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Get feature importance for clustering.
        
        Args:
            data: Market data DataFrame
            clustering_result: Result from cluster_with_hdbscan
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_names = clustering_result['feature_names']
        feature_matrix = clustering_result['feature_matrix']
        cluster_labels = clustering_result['cluster_labels']
        
        # Calculate feature importance based on variance within clusters
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = feature_matrix[:, i]
            
            # Calculate variance within each cluster
            cluster_variances = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_values = feature_values[cluster_mask]
                
                if len(cluster_values) > 1:
                    cluster_var = np.var(cluster_values)
                    cluster_variances.append(cluster_var)
            
            # Feature importance is inverse of average cluster variance
            if cluster_variances:
                avg_cluster_variance = np.mean(cluster_variances)
                importance_scores[feature_name] = 1 / (avg_cluster_variance + 1e-8)
            else:
                importance_scores[feature_name] = 0.0
        
        return importance_scores


# Convenience functions
def get_hdbscan_clustering_features(data: pd.DataFrame) -> Dict[str, Any]:
    """Get features for HDBSCAN clustering."""
    integrator = HDBSCANClusteringIntegration()
    return integrator.get_clustering_features(data)


def perform_hdbscan_clustering(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Perform HDBSCAN clustering on the data."""
    integrator = HDBSCANClusteringIntegration()
    return integrator.cluster_with_hdbscan(data, **kwargs)


__all__ = [
    'HDBSCANClusteringIntegration',
    'get_hdbscan_clustering_features',
    'perform_hdbscan_clustering'
]