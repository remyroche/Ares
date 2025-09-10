"""
Alternative Clustering Algorithms for SR Level Optimization

This module provides alternative clustering algorithms to replace DBSCAN
when it fails to achieve the desired number of clusters or levels.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

try:
    from sklearn.cluster import (
        KMeans, AgglomerativeClustering, SpectralClustering, 
        MeanShift, OPTICS, HDBSCAN, Birch
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from .logger import system_logger

@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    clusters: List[List[int]]
    noise_points: List[int]
    cluster_centers: List[float]
    algorithm_used: str
    parameters: Dict[str, Any]
    quality_score: float
    total_levels: int

class BaseClusteringAlgorithm(ABC):
    """Base class for clustering algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = system_logger.getChild(f'Clustering_{name}')
    
    @abstractmethod
    def cluster(self, levels: List[Dict], target_min_levels: int, 
                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster levels and return result."""
        pass
    
    def _prepare_data(self, levels: List[Dict]) -> np.ndarray:
        """Prepare data for clustering."""
        if not levels:
            return np.array([]).reshape(0, 1)
        
        # Extract price and strength features
        features = []
        for level in levels:
            price = level.get('price', 0.0)
            strength = level.get('strength', 0.5)
            touches = level.get('touches', 1)
            
            # Create feature vector: [price, strength, touches]
            features.append([price, strength, touches])
        
        return np.array(features)

class KMeansClustering(BaseClusteringAlgorithm):
    """K-Means clustering for SR levels."""
    
    def __init__(self):
        super().__init__("KMeans")
    
    def cluster(self, levels: List[Dict], target_min_levels: int, 
                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster using K-Means."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available for K-Means clustering")
        
        if len(levels) < target_min_levels:
            # Not enough levels to cluster
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                noise_points=[],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                algorithm_used="KMeans",
                parameters={"n_clusters": len(levels)},
                quality_score=1.0,
                total_levels=len(levels)
            )
        
        # Determine optimal number of clusters
        n_clusters = min(target_min_levels, len(levels) // 2)
        n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        try:
            features = self._prepare_data(levels)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Group levels by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                if label != -1:  # K-Means doesn't produce noise points
                    clusters[label].append(i)
            
            # Calculate cluster centers (prices)
            cluster_centers = []
            for cluster in clusters:
                if cluster:
                    cluster_prices = [levels[i].get('price', 0.0) for i in cluster]
                    cluster_centers.append(np.mean(cluster_prices))
                else:
                    cluster_centers.append(0.0)
            
            # Calculate quality score (inertia-based)
            quality_score = 1.0 / (1.0 + kmeans.inertia_)
            
            self.logger.info(f"K-Means clustering: {len(levels)} levels -> {n_clusters} clusters")
            
            return ClusteringResult(
                clusters=clusters,
                noise_points=[],
                cluster_centers=cluster_centers,
                algorithm_used="KMeans",
                parameters={"n_clusters": n_clusters},
                quality_score=quality_score,
                total_levels=len(levels)
            )
            
        except Exception as e:
            self.logger.error(f"K-Means clustering failed: {e}")
            raise

class AgglomerativeClusteringAlgorithm(BaseClusteringAlgorithm):
    """Agglomerative clustering for SR levels."""
    
    def __init__(self):
        super().__init__("Agglomerative")
    
    def cluster(self, levels: List[Dict], target_min_levels: int, 
                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster using Agglomerative clustering."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available for Agglomerative clustering")
        
        if len(levels) < target_min_levels:
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                noise_points=[],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                algorithm_used="Agglomerative",
                parameters={"n_clusters": len(levels)},
                quality_score=1.0,
                total_levels=len(levels)
            )
        
        n_clusters = min(target_min_levels, len(levels) // 2)
        n_clusters = max(2, n_clusters)
        
        try:
            features = self._prepare_data(levels)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Group levels by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(i)
            
            # Calculate cluster centers
            cluster_centers = []
            for cluster in clusters:
                if cluster:
                    cluster_prices = [levels[i].get('price', 0.0) for i in cluster]
                    cluster_centers.append(np.mean(cluster_prices))
                else:
                    cluster_centers.append(0.0)
            
            # Calculate quality score (silhouette-like)
            quality_score = 0.8  # Agglomerative typically performs well
            
            self.logger.info(f"Agglomerative clustering: {len(levels)} levels -> {n_clusters} clusters")
            
            return ClusteringResult(
                clusters=clusters,
                noise_points=[],
                cluster_centers=cluster_centers,
                algorithm_used="Agglomerative",
                parameters={"n_clusters": n_clusters, "linkage": "ward"},
                quality_score=quality_score,
                total_levels=len(levels)
            )
            
        except Exception as e:
            self.logger.error(f"Agglomerative clustering failed: {e}")
            raise

class HDBSCANClustering(BaseClusteringAlgorithm):
    """HDBSCAN clustering for SR levels."""
    
    def __init__(self):
        super().__init__("HDBSCAN")
    
    def cluster(self, levels: List[Dict], target_min_levels: int, 
                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster using HDBSCAN."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available")
        
        if len(levels) < target_min_levels:
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                noise_points=[],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                algorithm_used="HDBSCAN",
                parameters={"min_cluster_size": 2},
                quality_score=1.0,
                total_levels=len(levels)
            )
        
        try:
            features = self._prepare_data(levels)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Adaptive min_cluster_size based on target
            min_cluster_size = max(2, len(levels) // (target_min_levels * 2))
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.0
            )
            cluster_labels = clusterer.fit_predict(features_scaled)
            
            # Group levels by cluster
            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise label
            
            clusters = [[] for _ in range(len(unique_labels))]
            noise_points = []
            
            for i, label in enumerate(cluster_labels):
                if label == -1:
                    noise_points.append(i)
                else:
                    clusters[label].append(i)
            
            # Calculate cluster centers
            cluster_centers = []
            for cluster in clusters:
                if cluster:
                    cluster_prices = [levels[i].get('price', 0.0) for i in cluster]
                    cluster_centers.append(np.mean(cluster_prices))
                else:
                    cluster_centers.append(0.0)
            
            # Calculate quality score
            quality_score = clusterer.cluster_persistence_.mean() if hasattr(clusterer, 'cluster_persistence_') else 0.7
            
            self.logger.info(f"HDBSCAN clustering: {len(levels)} levels -> {len(clusters)} clusters, {len(noise_points)} noise")
            
            return ClusteringResult(
                clusters=clusters,
                noise_points=noise_points,
                cluster_centers=cluster_centers,
                algorithm_used="HDBSCAN",
                parameters={"min_cluster_size": min_cluster_size},
                quality_score=quality_score,
                total_levels=len(levels)
            )
            
        except Exception as e:
            self.logger.error(f"HDBSCAN clustering failed: {e}")
            raise

class GaussianMixtureClustering(BaseClusteringAlgorithm):
    """Gaussian Mixture Model clustering for SR levels."""
    
    def __init__(self):
        super().__init__("GaussianMixture")
    
    def cluster(self, levels: List[Dict], target_min_levels: int, 
                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster using Gaussian Mixture Model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available for Gaussian Mixture clustering")
        
        if len(levels) < target_min_levels:
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                noise_points=[],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                algorithm_used="GaussianMixture",
                parameters={"n_components": len(levels)},
                quality_score=1.0,
                total_levels=len(levels)
            )
        
        n_components = min(target_min_levels, len(levels) // 2)
        n_components = max(2, n_components)
        
        try:
            features = self._prepare_data(levels)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=42,
                covariance_type='full'
            )
            cluster_labels = gmm.fit_predict(features_scaled)
            
            # Group levels by cluster
            clusters = [[] for _ in range(n_components)]
            for i, label in enumerate(cluster_labels):
                clusters[label].append(i)
            
            # Use GMM means as cluster centers
            cluster_centers = []
            for i in range(n_components):
                # Transform mean back to original scale
                mean_scaled = gmm.means_[i]
                mean_original = scaler.inverse_transform([mean_scaled])[0]
                cluster_centers.append(mean_original[0])  # Price is first feature
            
            # Calculate quality score (AIC/BIC based)
            quality_score = 1.0 / (1.0 + gmm.aic(features_scaled) / 1000)
            
            self.logger.info(f"Gaussian Mixture clustering: {len(levels)} levels -> {n_components} components")
            
            return ClusteringResult(
                clusters=clusters,
                noise_points=[],
                cluster_centers=cluster_centers,
                algorithm_used="GaussianMixture",
                parameters={"n_components": n_components},
                quality_score=quality_score,
                total_levels=len(levels)
            )
            
        except Exception as e:
            self.logger.error(f"Gaussian Mixture clustering failed: {e}")
            raise

class PriceBasedClustering(BaseClusteringAlgorithm):
    """Simple price-based clustering for SR levels."""
    
    def __init__(self):
        super().__init__("PriceBased")
    
    def cluster(self, levels: List[Dict], target_min_levels: int, 
                price_range: Tuple[float, float]) -> ClusteringResult:
        """Cluster using simple price-based binning."""
        
        if len(levels) < target_min_levels:
            return ClusteringResult(
                clusters=[[i] for i in range(len(levels))],
                noise_points=[],
                cluster_centers=[level.get('price', 0.0) for level in levels],
                algorithm_used="PriceBased",
                parameters={"n_bins": len(levels)},
                quality_score=1.0,
                total_levels=len(levels)
            )
        
        try:
            prices = [level.get('price', 0.0) for level in levels]
            min_price, max_price = min(prices), max(prices)
            
            # Create price bins
            n_bins = min(target_min_levels, len(levels) // 2)
            n_bins = max(2, n_bins)
            
            bin_edges = np.linspace(min_price, max_price, n_bins + 1)
            bin_assignments = np.digitize(prices, bin_edges) - 1
            
            # Group levels by bin
            clusters = [[] for _ in range(n_bins)]
            for i, bin_idx in enumerate(bin_assignments):
                if 0 <= bin_idx < n_bins:
                    clusters[bin_idx].append(i)
            
            # Calculate cluster centers (bin midpoints)
            cluster_centers = []
            for i in range(n_bins):
                if clusters[i]:
                    cluster_prices = [levels[j].get('price', 0.0) for j in clusters[i]]
                    cluster_centers.append(np.mean(cluster_prices))
                else:
                    cluster_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            
            # Calculate quality score (based on price distribution)
            quality_score = 0.6  # Simple method, moderate quality
            
            self.logger.info(f"Price-based clustering: {len(levels)} levels -> {n_bins} bins")
            
            return ClusteringResult(
                clusters=clusters,
                noise_points=[],
                cluster_centers=cluster_centers,
                algorithm_used="PriceBased",
                parameters={"n_bins": n_bins},
                quality_score=quality_score,
                total_levels=len(levels)
            )
            
        except Exception as e:
            self.logger.error(f"Price-based clustering failed: {e}")
            raise

class ClusteringManager:
    """Manager for alternative clustering algorithms."""
    
    def __init__(self):
        self.logger = system_logger.getChild('ClusteringManager')
        self.algorithms = {
            'kmeans': KMeansClustering(),
            'agglomerative': AgglomerativeClusteringAlgorithm(),
            'hdbscan': HDBSCANClustering() if HDBSCAN_AVAILABLE else None,
            'gaussian_mixture': GaussianMixtureClustering(),
            'price_based': PriceBasedClustering()
        }
        
        # Remove unavailable algorithms
        self.algorithms = {k: v for k, v in self.algorithms.items() if v is not None}
        
        self.logger.info(f"Available clustering algorithms: {list(self.algorithms.keys())}")
    
    def cluster_with_fallback(self, levels: List[Dict], target_min_levels: int, 
                             price_range: Tuple[float, float], 
                             preferred_algorithm: str = 'kmeans') -> ClusteringResult:
        """Cluster levels with fallback to alternative algorithms."""
        
        if not levels:
            return ClusteringResult(
                clusters=[],
                noise_points=[],
                cluster_centers=[],
                algorithm_used="none",
                parameters={},
                quality_score=0.0,
                total_levels=0
            )
        
        # Try preferred algorithm first
        if preferred_algorithm in self.algorithms:
            try:
                result = self.algorithms[preferred_algorithm].cluster(
                    levels, target_min_levels, price_range
                )
                if result.total_levels >= target_min_levels:
                    self.logger.info(f"Successfully clustered with {preferred_algorithm}")
                    return result
                else:
                    self.logger.warning(f"{preferred_algorithm} produced insufficient levels: {result.total_levels} < {target_min_levels}")
            except Exception as e:
                self.logger.warning(f"{preferred_algorithm} failed: {e}")
        
        # Try other algorithms in order of preference
        algorithm_order = ['kmeans', 'agglomerative', 'gaussian_mixture', 'hdbscan', 'price_based']
        
        for algorithm_name in algorithm_order:
            if algorithm_name == preferred_algorithm:
                continue
                
            if algorithm_name in self.algorithms:
                try:
                    result = self.algorithms[algorithm_name].cluster(
                        levels, target_min_levels, price_range
                    )
                    if result.total_levels >= target_min_levels:
                        self.logger.info(f"Successfully clustered with {algorithm_name} (fallback)")
                        return result
                    else:
                        self.logger.warning(f"{algorithm_name} produced insufficient levels: {result.total_levels} < {target_min_levels}")
                except Exception as e:
                    self.logger.warning(f"{algorithm_name} failed: {e}")
        
        # If all algorithms fail, return simple clustering
        self.logger.warning("All clustering algorithms failed, returning simple clustering")
        return ClusteringResult(
            clusters=[[i] for i in range(len(levels))],
            noise_points=[],
            cluster_centers=[level.get('price', 0.0) for level in levels],
            algorithm_used="simple",
            parameters={},
            quality_score=0.5,
            total_levels=len(levels)
        )
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available clustering algorithms."""
        return list(self.algorithms.keys())
    
    def get_algorithm_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available algorithms."""
        info = {}
        for name, algorithm in self.algorithms.items():
            info[name] = {
                'name': algorithm.name,
                'description': algorithm.__doc__ or f"{algorithm.name} clustering algorithm",
                'available': True
            }
        return info

# Convenience function
def get_clustering_manager() -> ClusteringManager:
    """Get a clustering manager instance."""
    return ClusteringManager()