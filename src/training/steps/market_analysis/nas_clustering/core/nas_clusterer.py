"""
NAS Clusterer

Implementation for Neural Architecture Search clustering.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class ClusteringAlgorithm(Enum):
    """Clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"


@dataclass
class ClusteringConfig:
    """Configuration for NAS clustering."""
    algorithm: ClusteringAlgorithm
    n_clusters: Optional[int] = None
    eps: float = 0.5
    min_samples: int = 5
    linkage: str = "ward"
    random_state: int = 42


class NASClusterer:
    """Neural Architecture Search Clusterer."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize NAS clusterer.
        
        Args:
            config: Clustering configuration
        """
        self.config = config
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.clustering_metrics = {}
        
    def fit(self, architectures: List[Dict], features: Optional[np.ndarray] = None) -> Dict:
        """Fit clustering model to architectures.
        
        Args:
            architectures: List of architecture specifications
            features: Optional pre-computed features
            
        Returns:
            Dictionary containing clustering results
        """
        # Extract features if not provided
        if features is None:
            features = self._extract_features(architectures)
        
        # Initialize clusterer
        self._initialize_clusterer()
        
        # Fit clustering model
        self.cluster_labels = self.clusterer.fit_predict(features)
        
        # Calculate cluster centers
        self.cluster_centers = self._calculate_cluster_centers(features)
        
        # Calculate clustering metrics
        self.clustering_metrics = self._calculate_metrics(features)
        
        return {
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'metrics': self.clustering_metrics,
            'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        }
    
    def _extract_features(self, architectures: List[Dict]) -> np.ndarray:
        """Extract features from architectures."""
        features = []
        
        for architecture in architectures:
            feature_vector = []
            
            # Extract layer features
            layers = architecture.get('layers', [])
            
            # Number of layers
            feature_vector.append(len(layers))
            
            # Total parameters
            total_params = sum(layer.get('width', 64) for layer in layers)
            feature_vector.append(total_params)
            
            # Average layer width
            avg_width = total_params / len(layers) if layers else 0
            feature_vector.append(avg_width)
            
            # Layer width variance
            widths = [layer.get('width', 64) for layer in layers]
            width_variance = np.var(widths) if len(widths) > 1 else 0
            feature_vector.append(width_variance)
            
            # Activation diversity
            activations = [layer.get('activation', 'relu') for layer in layers]
            unique_activations = len(set(activations))
            feature_vector.append(unique_activations)
            
            # Dropout rate
            dropout_rates = [layer.get('dropout', 0) for layer in layers]
            avg_dropout = np.mean(dropout_rates) if dropout_rates else 0
            feature_vector.append(avg_dropout)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _initialize_clusterer(self):
        """Initialize clustering algorithm."""
        if self.config.algorithm == ClusteringAlgorithm.KMEANS:
            self.clusterer = KMeans(
                n_clusters=self.config.n_clusters or 3,
                random_state=self.config.random_state
            )
        elif self.config.algorithm == ClusteringAlgorithm.DBSCAN:
            self.clusterer = DBSCAN(
                eps=self.config.eps,
                min_samples=self.config.min_samples
            )
        elif self.config.algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.config.n_clusters or 3,
                linkage=self.config.linkage
            )
        else:
            self.clusterer = KMeans(n_clusters=3, random_state=self.config.random_state)
    
    def _calculate_cluster_centers(self, features: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        if self.config.algorithm == ClusteringAlgorithm.DBSCAN:
            # For DBSCAN, calculate centers of non-noise clusters
            unique_labels = set(self.cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            centers = []
            for label in unique_labels:
                cluster_points = features[self.cluster_labels == label]
                if len(cluster_points) > 0:
                    centers.append(np.mean(cluster_points, axis=0))
            
            return np.array(centers) if centers else np.array([])
        else:
            # For other algorithms, use cluster centers
            if hasattr(self.clusterer, 'cluster_centers_'):
                return self.clusterer.cluster_centers_
            else:
                # Calculate centers manually
                unique_labels = set(self.cluster_labels)
                centers = []
                for label in unique_labels:
                    cluster_points = features[self.cluster_labels == label]
                    if len(cluster_points) > 0:
                        centers.append(np.mean(cluster_points, axis=0))
                return np.array(centers)
    
    def _calculate_metrics(self, features: np.ndarray) -> Dict:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        # Remove noise points for metric calculation
        valid_labels = self.cluster_labels[self.cluster_labels != -1]
        valid_features = features[self.cluster_labels != -1]
        
        if len(valid_labels) > 1 and len(set(valid_labels)) > 1:
            # Silhouette score
            try:
                metrics['silhouette_score'] = silhouette_score(valid_features, valid_labels)
            except:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_features, valid_labels)
            except:
                metrics['calinski_harabasz_score'] = 0.0
        else:
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz_score'] = 0.0
        
        # Number of clusters
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        metrics['n_clusters'] = n_clusters
        
        # Number of noise points
        n_noise = np.sum(self.cluster_labels == -1)
        metrics['n_noise'] = n_noise
        
        return metrics
    
    def predict(self, architectures: List[Dict], features: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict cluster labels for new architectures.
        
        Args:
            architectures: List of architecture specifications
            features: Optional pre-computed features
            
        Returns:
            Array of cluster labels
        """
        if features is None:
            features = self._extract_features(architectures)
        
        if self.config.algorithm == ClusteringAlgorithm.DBSCAN:
            # DBSCAN doesn't have predict method, need to use fit_predict
            return self.clusterer.fit_predict(features)
        else:
            return self.clusterer.predict(features)
    
    def get_cluster_summary(self, architectures: List[Dict]) -> Dict:
        """Get summary of clusters.
        
        Args:
            architectures: List of architecture specifications
            
        Returns:
            Dictionary containing cluster summary
        """
        if self.cluster_labels is None:
            return {}
        
        summary = {}
        unique_labels = set(self.cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise cluster
                continue
            
            cluster_architectures = [architectures[i] for i in range(len(architectures)) 
                                   if self.cluster_labels[i] == label]
            
            # Calculate cluster statistics
            cluster_stats = {
                'size': len(cluster_architectures),
                'architectures': cluster_architectures,
                'avg_layers': np.mean([len(arch.get('layers', [])) for arch in cluster_architectures]),
                'avg_params': np.mean([sum(layer.get('width', 64) for layer in arch.get('layers', [])) 
                                     for arch in cluster_architectures])
            }
            
            summary[f'cluster_{label}'] = cluster_stats
        
        return summary
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        return self.cluster_centers
    
    def get_cluster_labels(self) -> np.ndarray:
        """Get cluster labels."""
        return self.cluster_labels
    
    def get_metrics(self) -> Dict:
        """Get clustering metrics."""
        return self.clustering_metrics
