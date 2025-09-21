#!/usr/bin/env python3
"""
Cluster Validator for Regime Clustering Quality Assessment.

This module provides comprehensive validation of regime clustering results,
including internal coherence, validity, and distinction metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import Counter
import logging

from src.utils.logger import system_logger
from src.utils.tprint import tprint


class ClusterValidator:
    """
    Validates the quality of regime clustering results.
    
    Provides metrics for:
    - Internal coherence (within-cluster similarity)
    - Validity (cluster separation and compactness)
    - Distinction (between-cluster differences)
    - Size distribution compliance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the cluster validator."""
        self.config = config
        self.logger = system_logger.getChild('ClusterValidator')
        
        # Validation thresholds
        self.min_silhouette_score = config.get('min_silhouette_score', 0.3)
        self.max_size_variance = config.get('max_size_variance', 0.01)
        self.min_constraint_satisfaction = config.get('min_constraint_satisfaction', 0.8)
        
        tprint("🔧 ClusterValidator initialized")
    
    def validate_clustering_results(self, 
                                  cluster_labels: np.ndarray,
                                  regime_coordinates: np.ndarray,
                                  cluster_stats: Dict[str, Any],
                                  regime_assignments: List[int]) -> Dict[str, Any]:
        """
        Perform comprehensive validation of clustering results.
        
        Args:
            cluster_labels: Cluster assignment for each regime
            regime_coordinates: 3D coordinates for each regime
            cluster_stats: Statistics for each cluster
            regime_assignments: Sample assignments to regimes
            
        Returns:
            Dictionary with validation results
        """
        tprint("🔍 Starting comprehensive cluster validation")
        
        validation_results = {}
        
        # 1. Internal Coherence Validation
        validation_results['internal_coherence'] = self._validate_internal_coherence(
            cluster_labels, regime_coordinates, cluster_stats
        )
        
        # 2. Validity Validation
        validation_results['validity'] = self._validate_cluster_validity(
            cluster_labels, regime_coordinates
        )
        
        # 3. Distinction Validation
        validation_results['distinction'] = self._validate_cluster_distinction(
            cluster_labels, regime_coordinates, cluster_stats
        )
        
        # 4. Size Distribution Validation
        validation_results['size_distribution'] = self._validate_size_distribution(
            cluster_stats, regime_assignments
        )
        
        # 5. Overall Quality Score
        validation_results['overall_quality'] = self._calculate_overall_quality(
            validation_results
        )
        
        tprint("✅ Cluster validation completed")
        return validation_results
    
    def _validate_internal_coherence(self, 
                                   cluster_labels: np.ndarray,
                                   regime_coordinates: np.ndarray,
                                   cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate internal coherence of clusters (within-cluster similarity).
        
        Returns metrics for how similar regimes are within each cluster.
        """
        tprint("🔍 Validating internal coherence")
        
        coherence_metrics = {}
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            # Get regimes in this cluster
            regime_mask = cluster_labels == cluster_id
            cluster_coords = regime_coordinates[regime_mask]
            
            if len(cluster_coords) < 2:
                coherence_metrics[cluster_id] = {
                    'intra_cluster_distance': 0.0,
                    'coherence_score': 1.0,
                    'regime_count': len(cluster_coords)
                }
                continue
            
            # Calculate intra-cluster distances
            distances = []
            for i in range(len(cluster_coords)):
                for j in range(i + 1, len(cluster_coords)):
                    dist = np.linalg.norm(cluster_coords[i] - cluster_coords[j])
                    distances.append(dist)
            
            mean_intra_distance = np.mean(distances) if distances else 0.0
            coherence_score = 1.0 / (1.0 + mean_intra_distance)  # Higher is better
            
            coherence_metrics[cluster_id] = {
                'intra_cluster_distance': mean_intra_distance,
                'coherence_score': coherence_score,
                'regime_count': len(cluster_coords),
                'distance_std': np.std(distances) if distances else 0.0
            }
        
        # Overall coherence metrics
        all_coherence_scores = [metrics['coherence_score'] for metrics in coherence_metrics.values()]
        overall_coherence = np.mean(all_coherence_scores)
        
        return {
            'cluster_metrics': coherence_metrics,
            'overall_coherence_score': overall_coherence,
            'coherence_variance': np.var(all_coherence_scores),
            'min_coherence': min(all_coherence_scores),
            'max_coherence': max(all_coherence_scores)
        }
    
    def _validate_cluster_validity(self, 
                                 cluster_labels: np.ndarray,
                                 regime_coordinates: np.ndarray) -> Dict[str, Any]:
        """
        Validate cluster validity using standard clustering metrics.
        
        Returns silhouette score, Calinski-Harabasz index, and Davies-Bouldin index.
        """
        tprint("🔍 Validating cluster validity")
        
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        # Standardize coordinates
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(regime_coordinates)
        
        # Calculate metrics
        silhouette = silhouette_score(scaled_coords, cluster_labels)
        calinski = calinski_harabasz_score(scaled_coords, cluster_labels)
        davies_bouldin = davies_bouldin_score(scaled_coords, cluster_labels)
        
        # Determine if metrics meet quality thresholds
        quality_flags = {
            'silhouette_good': silhouette >= self.min_silhouette_score,
            'calinski_good': calinski > 0,  # Always positive for valid clustering
            'davies_bouldin_good': davies_bouldin < 2.0  # Lower is better
        }
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'quality_flags': quality_flags,
            'overall_validity_score': self._combine_validity_scores(silhouette, calinski, davies_bouldin)
        }
    
    def _validate_cluster_distinction(self, 
                                    cluster_labels: np.ndarray,
                                    regime_coordinates: np.ndarray,
                                    cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate distinction between clusters (between-cluster differences).
        
        Returns metrics for how different clusters are from each other.
        """
        tprint("🔍 Validating cluster distinction")
        
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        if n_clusters < 2:
            return {
                'inter_cluster_distances': {},
                'distinction_score': 1.0,
                'cluster_separation': 0.0
            }
        
        # Calculate inter-cluster distances
        inter_distances = {}
        cluster_centroids = {}
        
        # Calculate centroids
        for cluster_id in unique_clusters:
            regime_mask = cluster_labels == cluster_id
            cluster_coords = regime_coordinates[regime_mask]
            centroid = np.mean(cluster_coords, axis=0)
            cluster_centroids[cluster_id] = centroid
        
        # Calculate pairwise distances between centroids
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i < j:  # Avoid duplicates
                    dist = np.linalg.norm(cluster_centroids[cluster1] - cluster_centroids[cluster2])
                    inter_distances[f"{cluster1}_{cluster2}"] = dist
        
        # Calculate distinction metrics
        all_inter_distances = list(inter_distances.values())
        mean_inter_distance = np.mean(all_inter_distances)
        min_inter_distance = min(all_inter_distances)
        
        # Calculate intra-cluster distances for comparison
        intra_distances = []
        for cluster_id in unique_clusters:
            regime_mask = cluster_labels == cluster_id
            cluster_coords = regime_coordinates[regime_mask]
            if len(cluster_coords) > 1:
                centroid = cluster_centroids[cluster_id]
                intra_dist = np.mean([np.linalg.norm(coord - centroid) for coord in cluster_coords])
                intra_distances.append(intra_dist)
        
        mean_intra_distance = np.mean(intra_distances) if intra_distances else 0.0
        
        # Separation ratio (higher is better)
        separation_ratio = mean_inter_distance / (mean_intra_distance + 1e-8)
        
        return {
            'inter_cluster_distances': inter_distances,
            'mean_inter_cluster_distance': mean_inter_distance,
            'min_inter_cluster_distance': min_inter_distance,
            'mean_intra_cluster_distance': mean_intra_distance,
            'separation_ratio': separation_ratio,
            'distinction_score': min(1.0, separation_ratio / 2.0)  # Normalize to 0-1
        }
    
    def _validate_size_distribution(self, 
                                  cluster_stats: Dict[str, Any],
                                  regime_assignments: List[int]) -> Dict[str, Any]:
        """
        Validate that cluster sizes meet the specified constraints.
        
        Returns metrics for size distribution compliance.
        """
        tprint("🔍 Validating size distribution")
        
        total_samples = len(regime_assignments)
        cluster_percentages = [stats['percentage'] for stats in cluster_stats.values()]
        
        # Check constraint satisfaction
        min_size_pct = self.config.get('min_cluster_size_pct', 0.03) * 100
        max_size_pct = self.config.get('max_cluster_size_pct', 0.08) * 100
        max_noise_pct = self.config.get('max_noise_pct', 0.05) * 100
        
        clusters_in_range = sum(1 for pct in cluster_percentages 
                              if min_size_pct <= pct <= max_size_pct)
        total_clusters = len(cluster_percentages)
        
        # Identify problematic clusters
        too_small = [i for i, pct in enumerate(cluster_percentages) if pct < min_size_pct]
        too_large = [i for i, pct in enumerate(cluster_percentages) if pct > max_size_pct]
        
        # Check for noise cluster
        noise_clusters = [i for i, pct in enumerate(cluster_percentages) 
                         if pct < min_size_pct * 0.5]  # Very small clusters
        
        # Size distribution metrics
        size_variance = np.var(cluster_percentages)
        size_range = max(cluster_percentages) - min(cluster_percentages)
        
        # Compliance scores
        size_compliance = clusters_in_range / total_clusters if total_clusters > 0 else 0
        noise_compliance = len(noise_clusters) / total_clusters if total_clusters > 0 else 0
        
        return {
            'total_clusters': total_clusters,
            'clusters_in_range': clusters_in_range,
            'size_compliance': size_compliance,
            'cluster_percentages': cluster_percentages,
            'size_variance': size_variance,
            'size_range': size_range,
            'too_small_clusters': too_small,
            'too_large_clusters': too_large,
            'noise_clusters': noise_clusters,
            'noise_compliance': noise_compliance,
            'constraint_satisfaction': size_compliance
        }
    
    def _calculate_overall_quality(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall quality score from all validation metrics.
        
        Returns weighted combination of all quality measures.
        """
        tprint("🔍 Calculating overall quality score")
        
        # Extract individual scores
        coherence_score = validation_results['internal_coherence']['overall_coherence_score']
        validity_score = validation_results['validity']['overall_validity_score']
        distinction_score = validation_results['distinction']['distinction_score']
        size_compliance = validation_results['size_distribution']['constraint_satisfaction']
        
        # Weighted combination (adjust weights based on importance)
        weights = {
            'coherence': 0.25,
            'validity': 0.30,
            'distinction': 0.25,
            'size_compliance': 0.20
        }
        
        overall_score = (
            weights['coherence'] * coherence_score +
            weights['validity'] * validity_score +
            weights['distinction'] * distinction_score +
            weights['size_compliance'] * size_compliance
        )
        
        # Quality assessment
        if overall_score >= 0.8:
            quality_level = "Excellent"
        elif overall_score >= 0.6:
            quality_level = "Good"
        elif overall_score >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        return {
            'overall_score': overall_score,
            'quality_level': quality_level,
            'component_scores': {
                'coherence': coherence_score,
                'validity': validity_score,
                'distinction': distinction_score,
                'size_compliance': size_compliance
            },
            'weights': weights,
            'recommendations': self._generate_recommendations(validation_results, overall_score)
        }
    
    def _combine_validity_scores(self, silhouette: float, calinski: float, davies_bouldin: float) -> float:
        """
        Combine validity scores into a single metric.
        
        Args:
            silhouette: Silhouette score (-1 to 1, higher is better)
            calinski: Calinski-Harabasz score (0 to inf, higher is better)
            davies_bouldin: Davies-Bouldin score (0 to inf, lower is better)
            
        Returns:
            Combined validity score (0 to 1, higher is better)
        """
        # Normalize silhouette to 0-1
        silhouette_norm = (silhouette + 1) / 2
        
        # Normalize Calinski (use log to compress range)
        calinski_norm = min(1.0, np.log(calinski + 1) / 10)
        
        # Normalize Davies-Bouldin (invert and normalize)
        davies_bouldin_norm = max(0.0, 1.0 - min(1.0, davies_bouldin / 5))
        
        # Weighted average
        combined_score = (0.4 * silhouette_norm + 0.3 * calinski_norm + 0.3 * davies_bouldin_norm)
        
        return combined_score
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], overall_score: float) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Returns list of actionable recommendations.
        """
        recommendations = []
        
        # Check individual components
        coherence_score = validation_results['internal_coherence']['overall_coherence_score']
        if coherence_score < 0.5:
            recommendations.append("Consider increasing cluster compactness - some clusters may be too spread out")
        
        validity_score = validation_results['validity']['overall_validity_score']
        if validity_score < 0.5:
            recommendations.append("Cluster separation could be improved - consider adjusting clustering parameters")
        
        distinction_score = validation_results['distinction']['distinction_score']
        if distinction_score < 0.5:
            recommendations.append("Clusters may be too similar - consider increasing the number of clusters")
        
        size_compliance = validation_results['size_distribution']['constraint_satisfaction']
        if size_compliance < 0.8:
            recommendations.append("Cluster sizes don't meet constraints - consider merging small clusters or splitting large ones")
        
        # Overall recommendations
        if overall_score < 0.4:
            recommendations.append("Overall clustering quality is poor - consider re-running with different parameters")
        elif overall_score < 0.6:
            recommendations.append("Clustering quality is fair - minor adjustments may improve results")
        else:
            recommendations.append("Clustering quality is good - results are suitable for ML training")
        
        return recommendations