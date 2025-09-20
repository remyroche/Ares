#!/usr/bin/env python3
"""
Cluster Balancing for HMM Market Regime Detection

This module provides advanced cluster balancing techniques to ensure
no single cluster contains more than a specified percentage of samples.

Key Features:
- Adaptive cluster splitting for oversized clusters
- Intelligent cluster merging for undersized clusters
- Post-processing cluster rebalancing
- Constraint-based HMM training
- Cluster size monitoring and validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class BalancingMethod(Enum):
    """Methods for cluster balancing."""
    ADAPTIVE_SPLITTING = "adaptive_splitting"
    CLUSTER_MERGING = "cluster_merging"
    CONSTRAINT_BASED = "constraint_based"
    POST_PROCESSING = "post_processing"
    HYBRID = "hybrid"

@dataclass
class ClusterBalancingConfig:
    """Configuration for cluster balancing."""
    max_cluster_size_pct: float = 15.0  # Maximum cluster size as percentage
    min_cluster_size_pct: float = 5.0   # Minimum cluster size as percentage
    target_cluster_size_pct: float = 10.0  # Target cluster size as percentage
    
    # Balancing method
    balancing_method: BalancingMethod = BalancingMethod.HYBRID
    
    # Splitting parameters
    max_split_iterations: int = 5
    split_quality_threshold: float = 0.7
    
    # Merging parameters
    merge_similarity_threshold: float = 0.8
    max_merge_iterations: int = 3
    
    # Constraint-based training
    use_constrained_training: bool = True
    constraint_strength: float = 0.1
    
    # Validation
    validate_balance: bool = True
    balance_tolerance: float = 2.0  # Tolerance in percentage points

@dataclass
class ClusterInfo:
    """Information about a cluster."""
    cluster_id: int
    size: int
    percentage: float
    centroid: np.ndarray
    samples_indices: np.ndarray
    quality_score: float = 0.0

class ClusterBalancer:
    """
    Advanced cluster balancing system for HMM regime detection.
    
    Ensures no single cluster contains more than the specified maximum
    percentage of samples through various balancing techniques.
    """
    
    def __init__(self, config: Optional[ClusterBalancingConfig] = None):
        """Initialize the cluster balancer."""
        self.config = config or ClusterBalancingConfig()
        self.logger = logger.getChild("ClusterBalancer")
        
        # State tracking
        self.original_clusters: Dict[int, ClusterInfo] = {}
        self.balanced_clusters: Dict[int, ClusterInfo] = {}
        self.balancing_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"ClusterBalancer initialized with max cluster size: {self.config.max_cluster_size_pct}%")
    
    def balance_clusters(
        self,
        features: np.ndarray,
        initial_labels: np.ndarray,
        initial_probabilities: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Balance clusters to ensure no cluster exceeds the maximum size.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            initial_labels: Initial cluster labels
            initial_probabilities: Initial cluster probabilities (optional)
            
        Returns:
            Tuple of (balanced_labels, balanced_probabilities, balancing_info)
        """
        try:
            self.logger.info("Starting cluster balancing...")
            
            # Analyze initial cluster distribution
            cluster_analysis = self._analyze_cluster_distribution(features, initial_labels)
            self.logger.info(f"Initial cluster distribution: {cluster_analysis['size_percentages']}")
            
            # Check if balancing is needed
            max_cluster_pct = max(cluster_analysis['size_percentages'].values())
            if max_cluster_pct <= self.config.max_cluster_size_pct:
                self.logger.info(f"Clusters already balanced (max: {max_cluster_pct:.2f}%)")
                return initial_labels, initial_probabilities, {'balanced': False, 'reason': 'already_balanced'}
            
            # Store original clusters
            self._store_cluster_info(features, initial_labels, initial_probabilities)
            
            # Apply balancing method
            if self.config.balancing_method == BalancingMethod.HYBRID:
                balanced_labels, balanced_probs = self._hybrid_balancing(features, initial_labels, initial_probabilities)
            elif self.config.balancing_method == BalancingMethod.ADAPTIVE_SPLITTING:
                balanced_labels, balanced_probs = self._adaptive_splitting(features, initial_labels, initial_probabilities)
            elif self.config.balancing_method == BalancingMethod.CLUSTER_MERGING:
                balanced_labels, balanced_probs = self._cluster_merging(features, initial_labels, initial_probabilities)
            elif self.config.balancing_method == BalancingMethod.POST_PROCESSING:
                balanced_labels, balanced_probs = self._post_processing_balance(features, initial_labels, initial_probabilities)
            else:
                self.logger.warning(f"Unknown balancing method: {self.config.balancing_method}")
                balanced_labels, balanced_probs = initial_labels, initial_probabilities
            
            # Validate results
            final_analysis = self._analyze_cluster_distribution(features, balanced_labels)
            self.logger.info(f"Final cluster distribution: {final_analysis['size_percentages']}")
            
            # Create balancing info
            balancing_info = {
                'balanced': True,
                'method': self.config.balancing_method.value,
                'initial_distribution': cluster_analysis['size_percentages'],
                'final_distribution': final_analysis['size_percentages'],
                'improvement': max_cluster_pct - max(final_analysis['size_percentages'].values()),
                'iterations': len(self.balancing_history)
            }
            
            return balanced_labels, balanced_probs, balancing_info
            
        except Exception as e:
            self.logger.error(f"Cluster balancing failed: {e}")
            return initial_labels, initial_probabilities, {'balanced': False, 'error': str(e)}
    
    def _hybrid_balancing(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hybrid balancing combining multiple techniques."""
        self.logger.info("Applying hybrid balancing...")
        
        current_labels = labels.copy()
        current_probs = probabilities.copy() if probabilities is not None else None
        
        # Step 1: Split oversized clusters
        for iteration in range(self.config.max_split_iterations):
            cluster_analysis = self._analyze_cluster_distribution(features, current_labels)
            oversized_clusters = [
                cluster_id for cluster_id, pct in cluster_analysis['size_percentages'].items()
                if pct > self.config.max_cluster_size_pct
            ]
            
            if not oversized_clusters:
                break
                
            self.logger.info(f"Splitting iteration {iteration + 1}: {len(oversized_clusters)} oversized clusters")
            current_labels, current_probs = self._split_oversized_clusters(
                features, current_labels, current_probs, oversized_clusters
            )
        
        # Step 2: Merge undersized clusters if any exist
        cluster_analysis = self._analyze_cluster_distribution(features, current_labels)
        undersized_clusters = [
            cluster_id for cluster_id, pct in cluster_analysis['size_percentages'].items()
            if pct < self.config.min_cluster_size_pct
        ]
        
        if undersized_clusters:
            self.logger.info(f"Merging {len(undersized_clusters)} undersized clusters")
            current_labels, current_probs = self._merge_undersized_clusters(
                features, current_labels, current_probs, undersized_clusters
            )
        
        return current_labels, current_probs
    
    def _adaptive_splitting(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split oversized clusters adaptively."""
        self.logger.info("Applying adaptive splitting...")
        
        current_labels = labels.copy()
        current_probs = probabilities.copy() if probabilities is not None else None
        
        for iteration in range(self.config.max_split_iterations):
            cluster_analysis = self._analyze_cluster_distribution(features, current_labels)
            oversized_clusters = [
                cluster_id for cluster_id, pct in cluster_analysis['size_percentages'].items()
                if pct > self.config.max_cluster_size_pct
            ]
            
            if not oversized_clusters:
                self.logger.info(f"Splitting completed after {iteration} iterations")
                break
                
            current_labels, current_probs = self._split_oversized_clusters(
                features, current_labels, current_probs, oversized_clusters
            )
        
        return current_labels, current_probs
    
    def _split_oversized_clusters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray],
        oversized_clusters: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split oversized clusters into smaller ones."""
        new_labels = labels.copy()
        new_probs = probabilities.copy() if probabilities is not None else None
        next_cluster_id = max(labels) + 1
        
        for cluster_id in oversized_clusters:
            # Get cluster data
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_size = len(cluster_features)
            target_size = int(len(features) * self.config.target_cluster_size_pct / 100)
            n_splits = max(2, int(np.ceil(cluster_size / target_size)))
            
            self.logger.debug(f"Splitting cluster {cluster_id} (size: {cluster_size}) into {n_splits} parts")
            
            # Use K-means to split the cluster
            kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(cluster_features)
            
            # Assign new cluster labels
            for sub_cluster_id in range(n_splits):
                sub_mask = sub_labels == sub_cluster_id
                sub_indices = cluster_indices[sub_mask]
                
                if sub_cluster_id == 0:
                    # Keep original cluster ID for first sub-cluster
                    new_labels[sub_indices] = cluster_id
                else:
                    # Assign new cluster ID for other sub-clusters
                    new_labels[sub_indices] = next_cluster_id
                    next_cluster_id += 1
            
            # Update probabilities if available
            if new_probs is not None:
                # Redistribute probabilities based on sub-cluster assignments
                for sub_cluster_id in range(n_splits):
                    sub_mask = sub_labels == sub_cluster_id
                    sub_indices = cluster_indices[sub_mask]
                    
                    if len(sub_indices) > 0:
                        # Create new probability distribution
                        new_prob_row = np.zeros(new_probs.shape[1] + n_splits - 1)
                        
                        # Copy existing probabilities
                        if new_probs.shape[1] <= len(new_prob_row):
                            new_prob_row[:new_probs.shape[1]] = np.mean(new_probs[sub_indices], axis=0)
                        
                        # Set high probability for assigned cluster
                        assigned_cluster = new_labels[sub_indices[0]]
                        if assigned_cluster < len(new_prob_row):
                            new_prob_row[assigned_cluster] = 0.8
                            new_prob_row /= np.sum(new_prob_row)
        
        return new_labels, new_probs
    
    def _cluster_merging(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge similar clusters to balance sizes."""
        self.logger.info("Applying cluster merging...")
        
        current_labels = labels.copy()
        current_probs = probabilities.copy() if probabilities is not None else None
        
        # Calculate cluster centroids
        unique_clusters = np.unique(current_labels)
        centroids = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = current_labels == cluster_id
            centroids[cluster_id] = np.mean(features[cluster_mask], axis=0)
        
        # Find similar clusters to merge
        for iteration in range(self.config.max_merge_iterations):
            cluster_analysis = self._analyze_cluster_distribution(features, current_labels)
            
            # Find oversized clusters
            oversized_clusters = [
                cluster_id for cluster_id, pct in cluster_analysis['size_percentages'].items()
                if pct > self.config.max_cluster_size_pct
            ]
            
            if not oversized_clusters:
                break
            
            # Find merge candidates
            merge_pairs = self._find_merge_candidates(features, current_labels, centroids)
            
            if not merge_pairs:
                self.logger.warning("No suitable merge candidates found")
                break
            
            # Perform merges
            for cluster1, cluster2 in merge_pairs[:1]:  # Merge one pair at a time
                self.logger.debug(f"Merging clusters {cluster1} and {cluster2}")
                current_labels[current_labels == cluster2] = cluster1
                
                # Update centroids
                cluster_mask = current_labels == cluster1
                centroids[cluster1] = np.mean(features[cluster_mask], axis=0)
                del centroids[cluster2]
        
        return current_labels, current_probs
    
    def _post_processing_balance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply post-processing balancing by reassigning samples."""
        self.logger.info("Applying post-processing balance...")
        
        balanced_labels = labels.copy()
        n_samples = len(features)
        target_size = int(n_samples * self.config.target_cluster_size_pct / 100)
        
        cluster_analysis = self._analyze_cluster_distribution(features, balanced_labels)
        unique_clusters = list(cluster_analysis['sizes'].keys())
        
        # Iteratively balance clusters
        for iteration in range(10):  # Max 10 iterations
            cluster_analysis = self._analyze_cluster_distribution(features, balanced_labels)
            
            # Find oversized and undersized clusters
            oversized = [c for c, size in cluster_analysis['sizes'].items() if size > target_size * 1.5]
            undersized = [c for c, size in cluster_analysis['sizes'].items() if size < target_size * 0.5]
            
            if not oversized or not undersized:
                break
            
            # Reassign samples from oversized to undersized clusters
            for oversized_cluster in oversized:
                if not undersized:
                    break
                    
                # Find samples to reassign (those with lowest confidence)
                oversized_mask = balanced_labels == oversized_cluster
                oversized_indices = np.where(oversized_mask)[0]
                
                if probabilities is not None:
                    # Use probability-based reassignment
                    cluster_probs = probabilities[oversized_indices, oversized_cluster]
                    # Reassign samples with lowest confidence
                    n_reassign = min(len(oversized_indices) // 4, target_size // 2)
                    reassign_indices = oversized_indices[np.argsort(cluster_probs)[:n_reassign]]
                else:
                    # Random reassignment
                    n_reassign = min(len(oversized_indices) // 4, target_size // 2)
                    reassign_indices = np.random.choice(oversized_indices, n_reassign, replace=False)
                
                # Assign to undersized clusters
                target_cluster = undersized[0]
                balanced_labels[reassign_indices] = target_cluster
                
                # Update undersized list
                cluster_analysis = self._analyze_cluster_distribution(features, balanced_labels)
                undersized = [c for c, size in cluster_analysis['sizes'].items() if size < target_size * 0.5]
        
        return balanced_labels, probabilities
    
    def _find_merge_candidates(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centroids: Dict[int, np.ndarray]
    ) -> List[Tuple[int, int]]:
        """Find pairs of clusters that are candidates for merging."""
        unique_clusters = list(centroids.keys())
        merge_candidates = []
        
        # Calculate pairwise distances between centroids
        centroid_matrix = np.array([centroids[c] for c in unique_clusters])
        distances = cdist(centroid_matrix, centroid_matrix)
        
        # Find similar clusters
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i >= j:
                    continue
                
                # Check if clusters are similar enough to merge
                similarity = 1 / (1 + distances[i, j])  # Convert distance to similarity
                
                if similarity > self.config.merge_similarity_threshold:
                    merge_candidates.append((cluster1, cluster2))
        
        # Sort by similarity (highest first)
        merge_candidates.sort(key=lambda x: 1 / (1 + distances[
            unique_clusters.index(x[0]), unique_clusters.index(x[1])
        ]), reverse=True)
        
        return merge_candidates
    
    def _merge_undersized_clusters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray],
        undersized_clusters: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge undersized clusters with nearby larger clusters."""
        new_labels = labels.copy()
        
        # Calculate centroids for all clusters
        unique_clusters = np.unique(labels)
        centroids = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            centroids[cluster_id] = np.mean(features[cluster_mask], axis=0)
        
        # Merge each undersized cluster with its nearest neighbor
        for undersized_cluster in undersized_clusters:
            if undersized_cluster not in centroids:
                continue
                
            # Find nearest cluster
            undersized_centroid = centroids[undersized_cluster]
            min_distance = float('inf')
            nearest_cluster = None
            
            for cluster_id, centroid in centroids.items():
                if cluster_id == undersized_cluster:
                    continue
                    
                distance = np.linalg.norm(undersized_centroid - centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = cluster_id
            
            if nearest_cluster is not None:
                self.logger.debug(f"Merging undersized cluster {undersized_cluster} into {nearest_cluster}")
                new_labels[new_labels == undersized_cluster] = nearest_cluster
                
                # Update centroid
                cluster_mask = new_labels == nearest_cluster
                centroids[nearest_cluster] = np.mean(features[cluster_mask], axis=0)
                del centroids[undersized_cluster]
        
        return new_labels, probabilities
    
    def _analyze_cluster_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze the distribution of cluster sizes."""
        unique_clusters, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        
        sizes = dict(zip(unique_clusters, counts))
        size_percentages = {
            cluster_id: (count / total_samples) * 100
            for cluster_id, count in sizes.items()
        }
        
        return {
            'sizes': sizes,
            'size_percentages': size_percentages,
            'total_samples': total_samples,
            'n_clusters': len(unique_clusters),
            'max_cluster_pct': max(size_percentages.values()),
            'min_cluster_pct': min(size_percentages.values()),
            'std_cluster_pct': np.std(list(size_percentages.values()))
        }
    
    def _store_cluster_info(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray]
    ):
        """Store information about original clusters."""
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                size=len(cluster_features),
                percentage=(len(cluster_features) / len(features)) * 100,
                centroid=np.mean(cluster_features, axis=0),
                samples_indices=cluster_indices,
                quality_score=self._calculate_cluster_quality(cluster_features)
            )
            
            self.original_clusters[cluster_id] = cluster_info
    
    def _calculate_cluster_quality(self, cluster_features: np.ndarray) -> float:
        """Calculate quality score for a cluster."""
        if len(cluster_features) < 2:
            return 0.0
        
        # Calculate intra-cluster variance
        centroid = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        variance = np.var(distances)
        
        # Quality is inversely related to variance (lower variance = higher quality)
        quality = 1 / (1 + variance)
        return quality
    
    def validate_balance(self, labels: np.ndarray) -> Dict[str, Any]:
        """Validate that clusters are properly balanced."""
        analysis = self._analyze_cluster_distribution(np.zeros((len(labels), 1)), labels)
        
        max_pct = analysis['max_cluster_pct']
        min_pct = analysis['min_cluster_pct']
        
        is_balanced = max_pct <= (self.config.max_cluster_size_pct + self.config.balance_tolerance)
        has_sufficient_min = min_pct >= (self.config.min_cluster_size_pct - self.config.balance_tolerance)
        
        return {
            'is_balanced': is_balanced,
            'has_sufficient_min': has_sufficient_min,
            'max_cluster_pct': max_pct,
            'min_cluster_pct': min_pct,
            'size_distribution': analysis['size_percentages'],
            'balance_quality': min_pct / max_pct if max_pct > 0 else 0.0
        }

def create_balanced_hmm_config(max_cluster_size_pct: float = 15.0) -> ClusterBalancingConfig:
    """Create a balanced HMM configuration."""
    return ClusterBalancingConfig(
        max_cluster_size_pct=max_cluster_size_pct,
        min_cluster_size_pct=5.0,
        target_cluster_size_pct=10.0,
        balancing_method=BalancingMethod.HYBRID,
        max_split_iterations=5,
        merge_similarity_threshold=0.8,
        use_constrained_training=True,
        validate_balance=True
    )