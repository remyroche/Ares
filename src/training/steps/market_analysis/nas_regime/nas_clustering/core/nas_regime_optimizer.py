"""
NAS Regime Optimizer

Optimizes regime detection parameters for neural architecture search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of regime optimization."""
    optimal_n_regimes: int
    optimization_scores: Dict[str, float]
    regime_quality_metrics: Dict[str, float]
    execution_time: float

class NASRegimeOptimizer:
    """
    NAS Regime Optimizer for optimizing regime detection parameters.
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 enable_hardware_optimization: bool = True, enable_matrix_optimization: bool = True):
        """
        Initialize the NAS Regime Optimizer.
        
        Args:
            population_size: Size of the population for optimization
            generations: Number of generations to optimize
            enable_hardware_optimization: Whether to enable hardware optimization
            enable_matrix_optimization: Whether to enable matrix optimization
        """
        self.population_size = population_size
        self.generations = generations
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_matrix_optimization = enable_matrix_optimization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(f"NAS Regime Optimizer initialized with population_size={population_size}")
    
    def optimize_regime_count(self, data: np.ndarray, max_regimes: int = 20) -> OptimizationResult:
        """
        Optimize the number of regimes for the given data.
        
        Args:
            data: Input data for regime optimization
            max_regimes: Maximum number of regimes to test
            
        Returns:
            OptimizationResult with optimal regime count and metrics
        """
        start_time = time.time()
        self.logger.info(f"Starting regime count optimization with max_regimes={max_regimes}")
        
        try:
            # Determine range of regime counts to test
            min_regimes = 2
            max_regimes = min(max_regimes, len(data) // 10)
            regime_counts = range(min_regimes, max_regimes + 1)
            
            optimization_scores = {}
            regime_quality_metrics = {}
            
            # Test different regime counts
            for n_regimes in regime_counts:
                try:
                    scores = self._evaluate_regime_count(data, n_regimes)
                    optimization_scores[n_regimes] = scores
                    
                    # Calculate quality metrics
                    quality = self._calculate_regime_quality(data, n_regimes)
                    regime_quality_metrics[n_regimes] = quality
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate regime count {n_regimes}: {e}")
                    optimization_scores[n_regimes] = {'silhouette': 0.0, 'calinski_harabasz': 0.0}
                    regime_quality_metrics[n_regimes] = {'stability': 0.0, 'separation': 0.0}
            
            # Find optimal regime count
            optimal_n_regimes = self._find_optimal_regime_count(optimization_scores, regime_quality_metrics)
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                optimal_n_regimes=optimal_n_regimes,
                optimization_scores=optimization_scores.get(optimal_n_regimes, {}),
                regime_quality_metrics=regime_quality_metrics.get(optimal_n_regimes, {}),
                execution_time=execution_time
            )
            
            self.logger.info(f"Regime optimization completed. Optimal regimes: {optimal_n_regimes}")
            return result
            
        except Exception as e:
            self.logger.error(f"Regime optimization failed: {e}")
            execution_time = time.time() - start_time
            return OptimizationResult(
                optimal_n_regimes=3,
                optimization_scores={},
                regime_quality_metrics={},
                execution_time=execution_time
            )
    
    def _evaluate_regime_count(self, data: np.ndarray, n_regimes: int) -> Dict[str, float]:
        """Evaluate clustering quality for a specific number of regimes."""
        try:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate clustering metrics
            scores = {}
            
            # Silhouette score
            if len(np.unique(cluster_labels)) > 1:
                scores['silhouette'] = silhouette_score(data, cluster_labels)
                scores['calinski_harabasz'] = calinski_harabasz_score(data, cluster_labels)
            else:
                scores['silhouette'] = 0.0
                scores['calinski_harabasz'] = 0.0
            
            # Inertia (within-cluster sum of squares)
            scores['inertia'] = kmeans.inertia_
            
            # Cluster size balance
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            cluster_sizes = counts / len(data)
            scores['cluster_balance'] = 1.0 - np.std(cluster_sizes)
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Error evaluating regime count {n_regimes}: {e}")
            return {'silhouette': 0.0, 'calinski_harabasz': 0.0, 'inertia': float('inf'), 'cluster_balance': 0.0}
    
    def _calculate_regime_quality(self, data: np.ndarray, n_regimes: int) -> Dict[str, float]:
        """Calculate regime quality metrics."""
        try:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            quality_metrics = {}
            
            # Regime stability (how consistent the clustering is)
            stability_scores = []
            for _ in range(5):  # Test stability with multiple runs
                kmeans_temp = KMeans(n_clusters=n_regimes, random_state=None, n_init=1)
                temp_labels = kmeans_temp.fit_predict(data)
                # Calculate similarity with original clustering
                similarity = self._calculate_clustering_similarity(cluster_labels, temp_labels)
                stability_scores.append(similarity)
            
            quality_metrics['stability'] = np.mean(stability_scores)
            
            # Regime separation (how well separated the regimes are)
            cluster_centers = kmeans.cluster_centers_
            if len(cluster_centers) > 1:
                from sklearn.metrics.pairwise import euclidean_distances
                center_distances = euclidean_distances(cluster_centers)
                # Average distance between cluster centers
                quality_metrics['separation'] = np.mean(center_distances[np.triu_indices_from(center_distances, k=1)])
            else:
                quality_metrics['separation'] = 0.0
            
            # Regime compactness (how tight the clusters are)
            intra_cluster_distances = []
            for i in range(n_regimes):
                mask = cluster_labels == i
                cluster_data = data[mask]
                if len(cluster_data) > 1:
                    center = cluster_centers[i]
                    distances = np.linalg.norm(cluster_data - center, axis=1)
                    intra_cluster_distances.append(np.mean(distances))
            
            if intra_cluster_distances:
                quality_metrics['compactness'] = 1.0 / (1.0 + np.mean(intra_cluster_distances))
            else:
                quality_metrics['compactness'] = 0.0
            
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime quality for {n_regimes}: {e}")
            return {'stability': 0.0, 'separation': 0.0, 'compactness': 0.0}
    
    def _calculate_clustering_similarity(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate similarity between two clusterings."""
        try:
            from sklearn.metrics import adjusted_rand_score
            return adjusted_rand_score(labels1, labels2)
        except Exception:
            # Fallback similarity calculation
            if len(labels1) != len(labels2):
                return 0.0
            
            # Simple accuracy-based similarity
            matches = np.sum(labels1 == labels2)
            return matches / len(labels1)
    
    def _find_optimal_regime_count(self, optimization_scores: Dict[int, Dict[str, float]], 
                                 regime_quality_metrics: Dict[int, Dict[str, float]]) -> int:
        """Find the optimal number of regimes based on scores and quality metrics."""
        try:
            best_score = -float('inf')
            best_n_regimes = 3
            
            for n_regimes, scores in optimization_scores.items():
                quality = regime_quality_metrics.get(n_regimes, {})
                
                # Combined score: silhouette + stability + compactness - cluster imbalance
                combined_score = (
                    scores.get('silhouette', 0.0) * 0.4 +
                    quality.get('stability', 0.0) * 0.3 +
                    quality.get('compactness', 0.0) * 0.2 +
                    scores.get('cluster_balance', 0.0) * 0.1
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_n_regimes = n_regimes
            
            self.logger.info(f"Optimal regime count: {best_n_regimes} (score: {best_score:.3f})")
            return best_n_regimes
            
        except Exception as e:
            self.logger.warning(f"Error finding optimal regime count: {e}")
            return 3
