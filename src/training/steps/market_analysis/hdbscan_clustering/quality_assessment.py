"""
Quality Assessment Module for HDBSCAN Clustering

This module provides quality assessment utilities for HDBSCAN clustering results.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

@dataclass
class QualityMetrics:
    """Quality metrics for clustering results."""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    noise_ratio: float
    cluster_count: int
    stability_score: float
    separation_score: float
    compactness_score: float

class QualityAssessor:
    """Quality assessment for HDBSCAN clustering results."""
    
    def __init__(self):
        """Initialize the quality assessor."""
        pass
    
    def assess_quality(self, 
                      data: np.ndarray, 
                      labels: np.ndarray, 
                      cluster_centers: Optional[np.ndarray] = None) -> QualityMetrics:
        """
        Assess the quality of clustering results.
        
        Args:
            data: Input data used for clustering
            labels: Cluster labels (-1 for noise)
            cluster_centers: Optional cluster centers
            
        Returns:
            QualityMetrics object with quality scores
        """
        # Calculate basic metrics
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0.0
        
        # Calculate silhouette score
        try:
            from sklearn.metrics import silhouette_score
            if n_clusters > 1 and len(unique_labels) > 1:
                # Only calculate for non-noise points
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(data[non_noise_mask], labels[non_noise_mask])
                else:
                    silhouette = -1.0
            else:
                silhouette = -1.0
        except ImportError:
            silhouette = -1.0
        
        # Calculate Calinski-Harabasz score
        try:
            from sklearn.metrics import calinski_harabasz_score
            if n_clusters > 1 and len(unique_labels) > 1:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    ch_score = calinski_harabasz_score(data[non_noise_mask], labels[non_noise_mask])
                else:
                    ch_score = 0.0
            else:
                ch_score = 0.0
        except ImportError:
            ch_score = 0.0
        
        # Calculate Davies-Bouldin score
        try:
            from sklearn.metrics import davies_bouldin_score
            if n_clusters > 1 and len(unique_labels) > 1:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    db_score = davies_bouldin_score(data[non_noise_mask], labels[non_noise_mask])
                else:
                    db_score = float('inf')
            else:
                db_score = float('inf')
        except ImportError:
            db_score = float('inf')
        
        # Calculate stability score (based on cluster size distribution)
        stability_score = self._calculate_stability_score(labels)
        
        # Calculate separation score
        separation_score = self._calculate_separation_score(data, labels)
        
        # Calculate compactness score
        compactness_score = self._calculate_compactness_score(data, labels)
        
        return QualityMetrics(
            silhouette_score=silhouette,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            noise_ratio=noise_ratio,
            cluster_count=n_clusters,
            stability_score=stability_score,
            separation_score=separation_score,
            compactness_score=compactness_score
        )
    
    def _calculate_stability_score(self, labels: np.ndarray) -> float:
        """Calculate stability score based on cluster size distribution."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Remove noise label
        if -1 in unique_labels:
            noise_idx = np.where(unique_labels == -1)[0][0]
            unique_labels = np.delete(unique_labels, noise_idx)
            counts = np.delete(counts, noise_idx)
        
        if len(counts) <= 1:
            return 0.0
        
        # Calculate coefficient of variation (lower is more stable)
        mean_size = np.mean(counts)
        std_size = np.std(counts)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        # Convert to stability score (0-1, higher is better)
        stability_score = max(0.0, 1.0 - cv)
        
        return stability_score
    
    def _calculate_separation_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate separation score between clusters."""
        unique_labels = np.unique(labels)
        
        # Remove noise label
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) <= 1:
            return 0.0
        
        # Calculate minimum distance between cluster centers
        cluster_centers = []
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                center = np.mean(data[mask], axis=0)
                cluster_centers.append(center)
        
        if len(cluster_centers) <= 1:
            return 0.0
        
        cluster_centers = np.array(cluster_centers)
        
        # Calculate minimum distance between any two cluster centers
        min_distance = float('inf')
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                min_distance = min(min_distance, distance)
        
        # Normalize by data scale
        data_std = np.std(data)
        if data_std > 0:
            separation_score = min_distance / data_std
        else:
            separation_score = 0.0
        
        return separation_score
    
    def _calculate_compactness_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate compactness score within clusters."""
        unique_labels = np.unique(labels)
        
        # Remove noise label
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) == 0:
            return 0.0
        
        compactness_scores = []
        
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                cluster_data = data[mask]
                center = np.mean(cluster_data, axis=0)
                
                # Calculate average distance from center
                distances = np.linalg.norm(cluster_data - center, axis=1)
                avg_distance = np.mean(distances)
                
                # Normalize by data scale
                data_std = np.std(data)
                if data_std > 0:
                    compactness = 1.0 / (1.0 + avg_distance / data_std)
                else:
                    compactness = 1.0
                
                compactness_scores.append(compactness)
        
        if len(compactness_scores) == 0:
            return 0.0
        
        return np.mean(compactness_scores)

def assess_clustering_quality(data: np.ndarray, 
                            labels: np.ndarray, 
                            cluster_centers: Optional[np.ndarray] = None) -> QualityMetrics:
    """
    Convenience function to assess clustering quality.
    
    Args:
        data: Input data used for clustering
        labels: Cluster labels (-1 for noise)
        cluster_centers: Optional cluster centers
        
    Returns:
        QualityMetrics object with quality scores
    """
    assessor = QualityAssessor()
    return assessor.assess_quality(data, labels, cluster_centers)
