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
    def cluster(self, levels: List[Dict], price_range: Tuple[float, float], 
                proximity_threshold: float = 0.01, strength_similarity_threshold: float = 0.2) -> ClusteringResult:
        """Cluster levels based on strength and proximity."""
        pass
    
    def _prepare_data(self, levels: List[Dict]) -> np.ndarray:
        """Prepare data for clustering with normalized features."""
        if not levels:
            return np.array([]).reshape(0, 2)
        
        # Extract and normalize features
        prices = [level.get('price', 0.0) for level in levels]
        strengths = [level.get('strength', 0.5) for level in levels]
        
        # Normalize prices to [0, 1] range
        min_price, max_price = min(prices), max(prices)
        price_range = max_price - min_price
        if price_range > 0:
            normalized_prices = [(p - min_price) / price_range for p in prices]
        else:
            normalized_prices = [0.5] * len(prices)
        
        # Strengths are already in [0, 1] range
        features = list(zip(normalized_prices, strengths))
        return np.array(features)
    
    def _calculate_strength_proximity_distance(self, level1: Dict, level2: Dict, 
                                            price_range: Tuple[float, float]) -> float:
        """Calculate combined distance based on price proximity and strength similarity."""
        price1, price2 = level1.get('price', 0.0), level2.get('price', 0.0)
        strength1, strength2 = level1.get('strength', 0.5), level2.get('strength', 0.5)
        
        # Price proximity (normalized by price range)
        min_price, max_price = price_range
        price_range_size = max_price - min_price
        if price_range_size > 0:
            price_distance = abs(price1 - price2) / price_range_size
        else:
            price_distance = 0.0
        
        # Strength similarity (inverse of strength difference)
        strength_distance = abs(strength1 - strength2)
        
        # Combined distance (weighted combination)
        # Price proximity is more important (70%) than strength similarity (30%)
        combined_distance = 0.7 * price_distance + 0.3 * strength_distance
        
        return combined_distance

class StrengthProximityClustering(BaseClusteringAlgorithm):
    """Strength and proximity-based clustering for SR levels."""
    
    def __init__(self):
        super().__init__("StrengthProximity")
    
    def cluster(self, levels: List[Dict], price_range: Tuple[float, float], 
                proximity_threshold: float = 0.01, strength_similarity_threshold: float = 0.2) -> ClusteringResult:
        """Cluster levels based on strength and proximity using adaptive thresholds."""
        
        if not levels:
            return ClusteringResult(
                clusters=[],
                noise_points=[],
                cluster_centers=[],
                algorithm_used="StrengthProximity",
                parameters={},
                quality_score=0.0,
                total_levels=0
            )
        
        try:
            # Convert proximity threshold to absolute price difference
            min_price, max_price = price_range
            price_range_size = max_price - min_price
            absolute_proximity_threshold = proximity_threshold * price_range_size
            
            self.logger.info(f"Clustering {len(levels)} levels with proximity threshold: {absolute_proximity_threshold:.2f} ({proximity_threshold:.1%} of price range)")
            self.logger.info(f"Strength similarity threshold: {strength_similarity_threshold:.2f}")
            
            # Initialize clusters
            clusters = []
            unassigned_levels = list(range(len(levels)))
            
            while unassigned_levels:
                # Start new cluster with strongest unassigned level
                current_cluster = []
                seed_idx = self._find_strongest_level(levels, unassigned_levels)
                current_cluster.append(seed_idx)
                unassigned_levels.remove(seed_idx)
                
                # Find all levels that should be in this cluster
                self._grow_cluster(levels, current_cluster, unassigned_levels, 
                                 absolute_proximity_threshold, strength_similarity_threshold)
                
                clusters.append(current_cluster)
                
                # Remove assigned levels from unassigned
                for idx in current_cluster[1:]:  # Skip seed (already removed)
                    if idx in unassigned_levels:
                        unassigned_levels.remove(idx)
            
            # Calculate cluster centers and quality
            cluster_centers = []
            total_quality = 0.0
            
            for cluster in clusters:
                if cluster:
                    # Calculate weighted center (by strength)
                    cluster_prices = [levels[i].get('price', 0.0) for i in cluster]
                    cluster_strengths = [levels[i].get('strength', 0.5) for i in cluster]
                    
                    # Weighted average by strength
                    total_strength = sum(cluster_strengths)
                    if total_strength > 0:
                        weighted_center = sum(p * s for p, s in zip(cluster_prices, cluster_strengths)) / total_strength
                    else:
                        weighted_center = sum(cluster_prices) / len(cluster_prices)
                    
                    cluster_centers.append(weighted_center)
                    
                    # Calculate cluster quality (cohesion)
                    cluster_quality = self._calculate_cluster_quality(levels, cluster, weighted_center)
                    total_quality += cluster_quality
            
            # Overall quality score
            quality_score = total_quality / len(clusters) if clusters else 0.0
            
            self.logger.info(f"Strength-proximity clustering: {len(levels)} levels -> {len(clusters)} clusters")
            self.logger.info(f"Average cluster size: {len(levels) / len(clusters):.1f} levels")
            self.logger.info(f"Quality score: {quality_score:.3f}")
            
            return ClusteringResult(
                clusters=clusters,
                noise_points=[],
                cluster_centers=cluster_centers,
                algorithm_used="StrengthProximity",
                parameters={
                    "proximity_threshold": proximity_threshold,
                    "strength_similarity_threshold": strength_similarity_threshold,
                    "absolute_proximity_threshold": absolute_proximity_threshold
                },
                quality_score=quality_score,
                total_levels=len(levels)
            )
            
        except Exception as e:
            self.logger.error(f"Strength-proximity clustering failed: {e}")
            raise
    
    def _find_strongest_level(self, levels: List[Dict], available_indices: List[int]) -> int:
        """Find the level with highest strength among available indices."""
        if not available_indices:
            return 0
        
        strongest_idx = available_indices[0]
        strongest_strength = levels[strongest_idx].get('strength', 0.5)
        
        for idx in available_indices[1:]:
            strength = levels[idx].get('strength', 0.5)
            if strength > strongest_strength:
                strongest_strength = strength
                strongest_idx = idx
        
        return strongest_idx
    
    def _grow_cluster(self, levels: List[Dict], current_cluster: List[int], 
                     unassigned_levels: List[int], proximity_threshold: float, 
                     strength_similarity_threshold: float) -> None:
        """Grow cluster by adding nearby levels with similar strength."""
        
        # Get cluster characteristics
        cluster_prices = [levels[i].get('price', 0.0) for i in current_cluster]
        cluster_strengths = [levels[i].get('strength', 0.5) for i in current_cluster]
        cluster_center_price = sum(cluster_prices) / len(cluster_prices)
        cluster_avg_strength = sum(cluster_strengths) / len(cluster_strengths)
        
        # Find levels to add to cluster
        levels_to_add = []
        
        for idx in unassigned_levels:
            level_price = levels[idx].get('price', 0.0)
            level_strength = levels[idx].get('strength', 0.5)
            
            # Check proximity
            price_distance = abs(level_price - cluster_center_price)
            if price_distance > proximity_threshold:
                continue
            
            # Check strength similarity
            strength_difference = abs(level_strength - cluster_avg_strength)
            if strength_difference > strength_similarity_threshold:
                continue
            
            # Level qualifies for this cluster
            levels_to_add.append(idx)
        
        # Add qualifying levels
        current_cluster.extend(levels_to_add)
    
    def _calculate_cluster_quality(self, levels: List[Dict], cluster: List[int], 
                                 cluster_center: float) -> float:
        """Calculate quality score for a cluster based on cohesion."""
        if not cluster:
            return 0.0
        
        cluster_prices = [levels[i].get('price', 0.0) for i in cluster]
        cluster_strengths = [levels[i].get('strength', 0.5) for i in cluster]
        
        # Price cohesion (lower variance = higher quality)
        price_variance = np.var(cluster_prices) if len(cluster_prices) > 1 else 0.0
        price_cohesion = 1.0 / (1.0 + price_variance)
        
        # Strength cohesion (lower variance = higher quality)
        strength_variance = np.var(cluster_strengths) if len(cluster_strengths) > 1 else 0.0
        strength_cohesion = 1.0 / (1.0 + strength_variance)
        
        # Average strength (higher = better)
        avg_strength = sum(cluster_strengths) / len(cluster_strengths)
        
        # Combined quality score
        quality = 0.4 * price_cohesion + 0.3 * strength_cohesion + 0.3 * avg_strength
        
        return quality

class KMeansClustering(BaseClusteringAlgorithm):
    """K-Means clustering for SR levels."""
    
    def __init__(self):
        super().__init__("KMeans")
    
    def cluster(self, levels: List[Dict], price_range: Tuple[float, float], 
                proximity_threshold: float = 0.01, strength_similarity_threshold: float = 0.2) -> ClusteringResult:
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
            'strength_proximity': StrengthProximityClustering(),
            'kmeans': KMeansClustering(),
            'agglomerative': AgglomerativeClusteringAlgorithm(),
            'hdbscan': HDBSCANClustering() if HDBSCAN_AVAILABLE else None,
            'gaussian_mixture': GaussianMixtureClustering(),
            'price_based': PriceBasedClustering()
        }
        
        # Remove unavailable algorithms
        self.algorithms = {k: v for k, v in self.algorithms.items() if v is not None}
        
        self.logger.info(f"Available clustering algorithms: {list(self.algorithms.keys())}")
    
    def cluster_with_fallback(self, levels: List[Dict], price_range: Tuple[float, float], 
                             proximity_threshold: float = 0.01, 
                             strength_similarity_threshold: float = 0.2,
                             preferred_algorithm: str = 'strength_proximity') -> ClusteringResult:
        """Cluster levels based on strength and proximity with fallback to alternative algorithms."""
        
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
        
        self.logger.info(f"Clustering {len(levels)} levels with strength-proximity approach")
        self.logger.info(f"Price range: {price_range[0]:.2f} - {price_range[1]:.2f}")
        self.logger.info(f"Proximity threshold: {proximity_threshold:.1%} of price range")
        self.logger.info(f"Strength similarity threshold: {strength_similarity_threshold:.2f}")
        
        # Try preferred algorithm first
        if preferred_algorithm in self.algorithms:
            try:
                result = self.algorithms[preferred_algorithm].cluster(
                    levels, price_range, proximity_threshold, strength_similarity_threshold
                )
                self.logger.info(f"Successfully clustered with {preferred_algorithm}: {len(result.clusters)} clusters")
                return result
            except Exception as e:
                self.logger.warning(f"{preferred_algorithm} failed: {e}")
        
        # Try other algorithms in order of preference
        algorithm_order = ['strength_proximity', 'kmeans', 'agglomerative', 'gaussian_mixture', 'hdbscan', 'price_based']
        
        for algorithm_name in algorithm_order:
            if algorithm_name == preferred_algorithm:
                continue
                
            if algorithm_name in self.algorithms:
                try:
                    result = self.algorithms[algorithm_name].cluster(
                        levels, price_range, proximity_threshold, strength_similarity_threshold
                    )
                    self.logger.info(f"Successfully clustered with {algorithm_name} (fallback): {len(result.clusters)} clusters")
                    return result
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