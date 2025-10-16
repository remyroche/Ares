"""
NAS Clusterer

Implementation for Neural Architecture Search clustering.
"""

print("🔍 [NAS_CLUSTERER] Loading NAS Clusterer module")
print("🔍 [NAS_CLUSTERER] Module path: /workspace/src/training/steps/market_analysis/nas_clustering/core/nas_clusterer.py")
print("🔍 [NAS_CLUSTERER] Purpose: Implementation for Neural Architecture Search clustering")
print("🔍 [NAS_CLUSTERER] Status: Starting module import")

import numpy as np
print("🔍 [NAS_CLUSTERER] ✓ NumPy imported successfully")

from typing import Dict, List, Any, Optional, Tuple
print("🔍 [NAS_CLUSTERER] ✓ Typing imports completed")

from dataclasses import dataclass
print("🔍 [NAS_CLUSTERER] ✓ Dataclasses imported successfully")

from enum import Enum
print("🔍 [NAS_CLUSTERER] ✓ Enum imported successfully")

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
print("🔍 [NAS_CLUSTERER] ✓ Scikit-learn clustering algorithms imported")

from sklearn.metrics import silhouette_score, calinski_harabasz_score
print("🔍 [NAS_CLUSTERER] ✓ Clustering metrics imported")

print("🔍 [NAS_CLUSTERER] All imports completed successfully")


class ClusteringAlgorithm(Enum):
    """Clustering algorithms."""
    print("🔍 [CLUSTERING_ALGORITHM] Defining ClusteringAlgorithm enum")
    KMEANS = "kmeans"
    print("🔍 [CLUSTERING_ALGORITHM] ✓ KMEANS defined")
    DBSCAN = "dbscan"
    print("🔍 [CLUSTERING_ALGORITHM] ✓ DBSCAN defined")
    AGGLOMERATIVE = "agglomerative"
    print("🔍 [CLUSTERING_ALGORITHM] ✓ AGGLOMERATIVE defined")
    print("🔍 [CLUSTERING_ALGORITHM] All clustering algorithms defined successfully")


@dataclass
class ClusteringConfig:
    """Configuration for NAS clustering."""
    print("🔍 [CLUSTERING_CONFIG] Defining ClusteringConfig dataclass")
    algorithm: ClusteringAlgorithm
    print("🔍 [CLUSTERING_CONFIG] ✓ algorithm field defined")
    n_clusters: Optional[int] = None
    print("🔍 [CLUSTERING_CONFIG] ✓ n_clusters field defined (default: None)")
    eps: float = 0.5
    print("🔍 [CLUSTERING_CONFIG] ✓ eps field defined (default: 0.5)")
    min_samples: int = 5
    print("🔍 [CLUSTERING_CONFIG] ✓ min_samples field defined (default: 5)")
    linkage: str = "ward"
    print("🔍 [CLUSTERING_CONFIG] ✓ linkage field defined (default: 'ward')")
    random_state: int = 42
    print("🔍 [CLUSTERING_CONFIG] ✓ random_state field defined (default: 42)")
    print("🔍 [CLUSTERING_CONFIG] All configuration fields defined successfully")


class NASClusterer:
    """Neural Architecture Search Clusterer."""
    
    def __init__(self, config: ClusteringConfig):
        """Initialize NAS clusterer.
        
        Args:
            config: Clustering configuration
        """
        print("🔍 [NAS_CLUSTERER_INIT] Initializing NASClusterer")
        print(f"🔍 [NAS_CLUSTERER_INIT] Config received: {config}")
        print(f"🔍 [NAS_CLUSTERER_INIT] Config type: {type(config)}")
        print(f"🔍 [NAS_CLUSTERER_INIT] Algorithm: {config.algorithm}")
        print(f"🔍 [NAS_CLUSTERER_INIT] N clusters: {config.n_clusters}")
        print(f"🔍 [NAS_CLUSTERER_INIT] Eps: {config.eps}")
        print(f"🔍 [NAS_CLUSTERER_INIT] Min samples: {config.min_samples}")
        print(f"🔍 [NAS_CLUSTERER_INIT] Linkage: {config.linkage}")
        print(f"🔍 [NAS_CLUSTERER_INIT] Random state: {config.random_state}")
        
        self.config = config
        print("🔍 [NAS_CLUSTERER_INIT] ✓ Config assigned to self.config")
        
        self.clusterer = None
        print("🔍 [NAS_CLUSTERER_INIT] ✓ clusterer initialized as None")
        
        self.cluster_labels = None
        print("🔍 [NAS_CLUSTERER_INIT] ✓ cluster_labels initialized as None")
        
        self.cluster_centers = None
        print("🔍 [NAS_CLUSTERER_INIT] ✓ cluster_centers initialized as None")
        
        self.clustering_metrics = {}
        print("🔍 [NAS_CLUSTERER_INIT] ✓ clustering_metrics initialized as empty dict")
        
        print("🔍 [NAS_CLUSTERER_INIT] Initialization complete!")
        
    def fit(self, architectures: List[Dict], features: Optional[np.ndarray] = None) -> Dict:
        """Fit clustering model to architectures.
        
        Args:
            architectures: List of architecture specifications
            features: Optional pre-computed features
            
        Returns:
            Dictionary containing clustering results
        """
        print("🔍 [NAS_CLUSTERER_FIT] Starting clustering fit")
        print(f"🔍 [NAS_CLUSTERER_FIT] Number of architectures: {len(architectures)}")
        print(f"🔍 [NAS_CLUSTERER_FIT] Features provided: {features is not None}")
        
        if features is not None:
            print(f"🔍 [NAS_CLUSTERER_FIT] Features shape: {features.shape}")
            print(f"🔍 [NAS_CLUSTERER_FIT] Features type: {type(features)}")
            print(f"🔍 [NAS_CLUSTERER_FIT] Features dtype: {features.dtype}")
        else:
            print("🔍 [NAS_CLUSTERER_FIT] No features provided - will extract from architectures")
        
        # Extract features if not provided
        if features is None:
            print("🔍 [NAS_CLUSTERER_FIT] Extracting features from architectures...")
            features = self._extract_features(architectures)
            print(f"🔍 [NAS_CLUSTERER_FIT] ✓ Features extracted - shape: {features.shape}")
        else:
            print("🔍 [NAS_CLUSTERER_FIT] Using provided features")
        
        # Initialize clusterer
        print("🔍 [NAS_CLUSTERER_FIT] Initializing clusterer...")
        self._initialize_clusterer()
        print(f"🔍 [NAS_CLUSTERER_FIT] ✓ Clusterer initialized: {type(self.clusterer)}")
        
        # Fit clustering model
        print("🔍 [NAS_CLUSTERER_FIT] Fitting clustering model...")
        self.cluster_labels = self.clusterer.fit_predict(features)
        print(f"🔍 [NAS_CLUSTERER_FIT] ✓ Clustering completed - labels shape: {self.cluster_labels.shape}")
        print(f"🔍 [NAS_CLUSTERER_FIT] Unique labels: {np.unique(self.cluster_labels)}")
        print(f"🔍 [NAS_CLUSTERER_FIT] Number of clusters: {len(np.unique(self.cluster_labels))}")
        
        # Calculate cluster centers
        print("🔍 [NAS_CLUSTERER_FIT] Calculating cluster centers...")
        self.cluster_centers = self._calculate_cluster_centers(features)
        print(f"🔍 [NAS_CLUSTERER_FIT] ✓ Cluster centers calculated - shape: {self.cluster_centers.shape}")
        
        # Calculate clustering metrics
        print("🔍 [NAS_CLUSTERER_FIT] Calculating clustering metrics...")
        self.clustering_metrics = self._calculate_metrics(features)
        print(f"🔍 [NAS_CLUSTERER_FIT] ✓ Metrics calculated: {self.clustering_metrics}")
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        print(f"🔍 [NAS_CLUSTERER_FIT] Final number of clusters: {n_clusters}")
        
        result = {
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'metrics': self.clustering_metrics,
            'n_clusters': n_clusters
        }
        print(f"🔍 [NAS_CLUSTERER_FIT] ✓ Fit completed successfully")
        print(f"🔍 [NAS_CLUSTERER_FIT] Result: {result}")
        return result
    
    def _extract_features(self, architectures: List[Dict]) -> np.ndarray:
        """Extract features from architectures."""
        print("🔍 [NAS_CLUSTERER_EXTRACT] Starting feature extraction")
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] Number of architectures: {len(architectures)}")
        
        features = []
        
        for i, architecture in enumerate(architectures):
            if i % 10 == 0:  # Print progress every 10 architectures
                print(f"🔍 [NAS_CLUSTERER_EXTRACT] Processing architecture {i+1}/{len(architectures)}")
            
            feature_vector = []
            
            # Extract layer features
            layers = architecture.get('layers', [])
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: {len(layers)} layers")
            
            # Number of layers
            n_layers = len(layers)
            feature_vector.append(n_layers)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Number of layers = {n_layers}")
            
            # Total parameters
            total_params = sum(layer.get('width', 64) for layer in layers)
            feature_vector.append(total_params)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Total parameters = {total_params}")
            
            # Average layer width
            avg_width = total_params / len(layers) if layers else 0
            feature_vector.append(avg_width)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Average width = {avg_width:.2f}")
            
            # Layer width variance
            widths = [layer.get('width', 64) for layer in layers]
            width_variance = np.var(widths) if len(widths) > 1 else 0
            feature_vector.append(width_variance)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Width variance = {width_variance:.2f}")
            
            # Activation diversity
            activations = [layer.get('activation', 'relu') for layer in layers]
            unique_activations = len(set(activations))
            feature_vector.append(unique_activations)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Unique activations = {unique_activations}")
            
            # Dropout rate
            dropout_rates = [layer.get('dropout', 0) for layer in layers]
            avg_dropout = np.mean(dropout_rates) if dropout_rates else 0
            feature_vector.append(avg_dropout)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Average dropout = {avg_dropout:.3f}")
            
            features.append(feature_vector)
            print(f"🔍 [NAS_CLUSTERER_EXTRACT] Architecture {i}: Feature vector = {feature_vector}")
        
        result = np.array(features)
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] ✓ Feature extraction completed")
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] Features shape: {result.shape}")
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] Features dtype: {result.dtype}")
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] Features min: {np.min(result):.6f}")
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] Features max: {np.max(result):.6f}")
        print(f"🔍 [NAS_CLUSTERER_EXTRACT] Features mean: {np.mean(result):.6f}")
        return result
    
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
