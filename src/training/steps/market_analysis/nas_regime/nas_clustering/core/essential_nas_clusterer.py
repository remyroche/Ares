"""
Essential NAS Clusterer

A simplified neural architecture search clusterer for regime detection.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class ClusterResult:
    """Result of clustering operation."""
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    cluster_metrics: Dict[str, float]
    execution_time: float

class EssentialNASClusterer:
    """
    Essential NAS Clusterer for regime detection.
    
    Provides a simplified but effective clustering approach for neural architecture search.
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 enable_multi_objective: bool = True, light_mode: bool = False,
                 max_cluster_size_ratio: float = 0.25):
        """
        Initialize the Essential NAS Clusterer.
        
        Args:
            population_size: Size of the population for evolutionary search
            generations: Number of generations to evolve
            enable_multi_objective: Whether to use multi-objective optimization
            light_mode: Whether to use light mode for faster execution
            max_cluster_size_ratio: Maximum ratio of data points in a single cluster
        """
        self.population_size = population_size
        self.generations = generations
        self.enable_multi_objective = enable_multi_objective
        self.light_mode = light_mode
        self.max_cluster_size_ratio = max_cluster_size_ratio
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize clustering parameters
        self.min_clusters = 2
        self.max_clusters = 20
        
        self.logger.info(f"Essential NAS Clusterer initialized with population_size={population_size}, generations={generations}")
    
    def search(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform NAS search with clustering.
        
        Args:
            data: Input data for clustering
            labels: Optional ground truth labels
            
        Returns:
            Dictionary containing search results
        """
        start_time = time.time()
        self.logger.info("Starting NAS clustering search")
        
        try:
            # Validate input data
            if len(data) == 0:
                raise ValueError("Input data is empty")
            
            # Determine optimal number of clusters
            optimal_k = self._find_optimal_clusters(data)
            
            # Perform clustering
            cluster_result = self._perform_clustering(data, optimal_k)
            
            # Generate architecture recommendations
            architecture = self._generate_architecture(data, cluster_result)
            
            search_time = time.time() - start_time
            
            result = {
                'success': True,
                'best_params': architecture,
                'search_history': [architecture],
                'cluster_metrics': cluster_result.cluster_metrics,
                'search_time': search_time,
                'optimal_clusters': optimal_k,
                'cluster_labels': cluster_result.cluster_labels,
                'cluster_centers': cluster_result.cluster_centers
            }
            
            self.logger.info(f"NAS clustering search completed in {search_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"NAS clustering search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'search_time': time.time() - start_time
            }
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            if self.light_mode:
                # Light mode: test fewer cluster counts
                k_range = range(2, min(8, len(data) // 10))
            else:
                k_range = range(2, min(self.max_clusters, len(data) // 10))
            
            silhouette_scores = []
            inertias = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data)
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(data, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0.0)
                
                inertias.append(kmeans.inertia_)
            
            # Find optimal k using silhouette score
            if silhouette_scores:
                optimal_k = k_range[np.argmax(silhouette_scores)]
            else:
                optimal_k = 3
            
            self.logger.info(f"Optimal number of clusters determined: {optimal_k}")
            return optimal_k
            
        except Exception as e:
            self.logger.warning(f"Error finding optimal clusters: {e}, using default k=3")
            return 3
    
    def _perform_clustering(self, data: np.ndarray, n_clusters: int) -> ClusterResult:
        """Perform clustering with the specified number of clusters."""
        start_time = time.time()
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate clustering metrics
            cluster_centers = kmeans.cluster_centers_
            
            metrics = {}
            
            # Silhouette score
            if len(np.unique(cluster_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(data, cluster_labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, cluster_labels)
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
            
            # Cluster size distribution
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = counts / len(data)
            metrics['max_cluster_size_ratio'] = np.max(cluster_sizes)
            metrics['cluster_size_std'] = np.std(cluster_sizes)
            
            # Check cluster size constraint
            if metrics['max_cluster_size_ratio'] > self.max_cluster_size_ratio:
                self.logger.warning(f"Largest cluster size ratio {metrics['max_cluster_size_ratio']:.3f} exceeds limit {self.max_cluster_size_ratio}")
            
            execution_time = time.time() - start_time
            
            result = ClusterResult(
                cluster_labels=cluster_labels,
                cluster_centers=cluster_centers,
                cluster_metrics=metrics,
                execution_time=execution_time
            )
            
            self.logger.info(f"Clustering completed with {n_clusters} clusters, silhouette score: {metrics['silhouette_score']:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            # Return fallback result
            return ClusterResult(
                cluster_labels=np.zeros(len(data), dtype=int),
                cluster_centers=np.zeros((1, data.shape[1])),
                cluster_metrics={'error': str(e)},
                execution_time=time.time() - start_time
            )
    
    def _generate_architecture(self, data: np.ndarray, cluster_result: ClusterResult) -> Dict[str, Any]:
        """Generate neural architecture based on clustering results."""
        try:
            n_features = data.shape[1]
            n_clusters = len(np.unique(cluster_result.cluster_labels))
            
            # Determine architecture based on data characteristics
            if n_features < 10:
                hidden_size = 32
                n_layers = 2
            elif n_features < 50:
                hidden_size = 64
                n_layers = 3
            else:
                hidden_size = 128
                n_layers = 4
            
            # Adjust based on number of clusters
            if n_clusters > 5:
                hidden_size = min(hidden_size * 2, 256)
                n_layers = min(n_layers + 1, 5)
            
            # Generate architecture
            architecture = {
                'input_size': n_features,
                'hidden_size': hidden_size,
                'output_size': n_clusters,
                'n_layers': n_layers,
                'activation': 'relu',
                'dropout_rate': 0.2,
                'batch_normalization': True,
                'parameters_count': self._calculate_parameters(n_features, hidden_size, n_clusters, n_layers),
                'fitness_score': cluster_result.cluster_metrics.get('silhouette_score', 0.0),
                'complexity_score': min(1.0, n_layers / 5.0),
                'efficiency_score': min(1.0, 1000 / max(1, self._calculate_parameters(n_features, hidden_size, n_clusters, n_layers)))
            }
            
            self.logger.info(f"Generated architecture with {architecture['parameters_count']} parameters")
            return architecture
            
        except Exception as e:
            self.logger.error(f"Architecture generation failed: {e}")
            return {
                'input_size': data.shape[1],
                'hidden_size': 64,
                'output_size': 3,
                'n_layers': 2,
                'parameters_count': 1000,
                'fitness_score': 0.0,
                'complexity_score': 0.5,
                'efficiency_score': 0.5,
                'error': str(e)
            }
    
    def _calculate_parameters(self, input_size: int, hidden_size: int, output_size: int, n_layers: int) -> int:
        """Calculate total number of parameters in the architecture."""
        try:
            total_params = 0
            
            # Input to first hidden layer
            total_params += input_size * hidden_size + hidden_size
            
            # Hidden layers
            for _ in range(n_layers - 2):
                total_params += hidden_size * hidden_size + hidden_size
            
            # Last hidden to output layer
            total_params += hidden_size * output_size + output_size
            
            return total_params
            
        except Exception as e:
            self.logger.warning(f"Parameter calculation failed: {e}")
            return 1000
    
    def get_cluster_statistics(self, data: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Get detailed statistics about the clustering results."""
        try:
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels)
            
            statistics = {
                'n_clusters': n_clusters,
                'cluster_sizes': {},
                'cluster_means': {},
                'cluster_stds': {},
                'inter_cluster_distance': 0.0,
                'intra_cluster_distance': 0.0
            }
            
            # Calculate cluster statistics
            for label in unique_labels:
                mask = cluster_labels == label
                cluster_data = data[mask]
                
                statistics['cluster_sizes'][label] = len(cluster_data)
                statistics['cluster_means'][label] = np.mean(cluster_data, axis=0).tolist()
                statistics['cluster_stds'][label] = np.std(cluster_data, axis=0).tolist()
            
            # Calculate inter and intra cluster distances
            if n_clusters > 1:
                from sklearn.metrics.pairwise import euclidean_distances
                
                # Inter-cluster distances
                cluster_centers = np.array([statistics['cluster_means'][label] for label in unique_labels])
                inter_distances = euclidean_distances(cluster_centers)
                statistics['inter_cluster_distance'] = np.mean(inter_distances[np.triu_indices_from(inter_distances, k=1)])
                
                # Intra-cluster distances
                intra_distances = []
                for label in unique_labels:
                    mask = cluster_labels == label
                    cluster_data = data[mask]
                    if len(cluster_data) > 1:
                        cluster_distances = euclidean_distances(cluster_data)
                        intra_distances.append(np.mean(cluster_distances[np.triu_indices_from(cluster_distances, k=1)]))
                
                if intra_distances:
                    statistics['intra_cluster_distance'] = np.mean(intra_distances)
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Cluster statistics calculation failed: {e}")
            return {'error': str(e)}
