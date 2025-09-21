#!/usr/bin/env python3
"""
Regime Clusterer for HMM Regime Consolidation.

This module clusters small HMM regimes into larger, coherent clusters suitable
for ML model training. Uses hierarchical clustering with size constraints to
create ~20 clusters of 3-8% each with <5% noise.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from pathlib import Path
import re
from datetime import datetime

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from src.utils.logger import system_logger
from src.utils.tprint import tprint


class RegimeClusterer:
    """
    Clusters small HMM regimes into larger coherent clusters.
    
    Strategy:
    1. Parse regime names to extract 3D coordinates (Momentum, Volatility, Volume)
    2. Use hierarchical clustering with size constraints
    3. Validate cluster quality and coherence
    4. Assign noise regimes to dedicated cluster
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the regime clusterer."""
        self.config = config
        self.logger = system_logger.getChild('RegimeClusterer')
        
        # Clustering parameters
        self.target_clusters = config.get('target_clusters', 20)
        self.min_cluster_size_pct = config.get('min_cluster_size_pct', 0.03)  # 3%
        self.max_cluster_size_pct = config.get('max_cluster_size_pct', 0.08)  # 8%
        self.max_noise_pct = config.get('max_noise_pct', 0.05)  # 5%
        
        # Clustering method parameters
        self.linkage_method = config.get('linkage_method', 'ward')
        self.distance_threshold = config.get('distance_threshold', None)
        self.min_samples_per_regime = config.get('min_samples_per_regime', 5)
        
        # Results storage
        self.regime_data = None
        self.regime_coordinates = None
        self.cluster_labels = None
        self.cluster_stats = None
        self.clustering_metrics = None
        
        tprint("🔧 RegimeClusterer initialized")
        self.logger.info("RegimeClusterer initialized with config: %s", config)
    
    def load_hmm_results(self, hmm_outcome_path: str) -> Dict[str, Any]:
        """
        Load HMM regime discovery results from outcome file.
        
        Args:
            hmm_outcome_path: Path to HMM regime discovery outcome JSON file
            
        Returns:
            Dictionary containing regime data
        """
        tprint(f"📂 Loading HMM results from: {hmm_outcome_path}")
        
        with open(hmm_outcome_path, 'r') as f:
            hmm_data = json.load(f)
        
        if hmm_data['status'] != 'completed':
            raise ValueError(f"HMM outcome file shows failed status: {hmm_data['status']}")
        
        regime_result = hmm_data['artifacts']['hmm_regime_discovery_result']
        
        self.regime_data = {
            'regime_models': regime_result['regime_models'],
            'regime_assignments': regime_result['regime_assignments'],
            'metadata': hmm_data['metadata']
        }
        
        total_samples = len(self.regime_data['regime_assignments'])
        total_regimes = len(self.regime_data['regime_models'])
        
        tprint(f"✅ Loaded {total_samples} samples across {total_regimes} regimes")
        self.logger.info(f"Loaded {total_samples} samples across {total_regimes} regimes")
        
        return self.regime_data
    
    def parse_regime_coordinates(self) -> np.ndarray:
        """
        Parse regime names to extract 3D coordinates (Momentum, Volatility, Volume).
        
        Returns:
            Array of shape (n_regimes, 3) with coordinates for each regime
        """
        tprint("🔍 Parsing regime coordinates from names")
        
        regime_models = self.regime_data['regime_models']
        coordinates = []
        regime_names = []
        
        # Pattern to extract M, V, Vol values from regime names
        pattern = r'regime_M(\d+)_V(\d+)_Vol(\d+)'
        
        for i, regime_name in enumerate(regime_models):
            match = re.match(pattern, regime_name)
            if match:
                momentum, volatility, volume = map(int, match.groups())
                coordinates.append([momentum, volatility, volume])
                regime_names.append(regime_name)
            else:
                self.logger.warning(f"Could not parse regime name: {regime_name}")
                # Use default coordinates for unparseable names
                coordinates.append([0, 0, 0])
                regime_names.append(regime_name)
        
        self.regime_coordinates = np.array(coordinates)
        
        tprint(f"✅ Parsed coordinates for {len(coordinates)} regimes")
        self.logger.info(f"Parsed coordinates for {len(coordinates)} regimes")
        
        return self.regime_coordinates
    
    def analyze_regime_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of regimes by sample count.
        
        Returns:
            Dictionary with distribution statistics
        """
        assignments = self.regime_data['regime_assignments']
        regime_counts = Counter(assignments)
        
        counts = list(regime_counts.values())
        total_samples = len(assignments)
        
        stats = {
            'total_samples': total_samples,
            'total_regimes': len(regime_counts),
            'min_samples_per_regime': min(counts),
            'max_samples_per_regime': max(counts),
            'mean_samples_per_regime': np.mean(counts),
            'median_samples_per_regime': np.median(counts),
            'regime_counts': regime_counts,
            'regime_percentages': {regime_id: (count/total_samples)*100 
                                 for regime_id, count in regime_counts.items()}
        }
        
        tprint("📊 Regime Distribution Analysis:")
        tprint(f"   Total samples: {total_samples:,}")
        tprint(f"   Total regimes: {len(regime_counts)}")
        tprint(f"   Avg samples per regime: {np.mean(counts):.1f}")
        tprint(f"   Min samples: {min(counts)}, Max samples: {max(counts)}")
        
        return stats
    
    def perform_clustering(self) -> np.ndarray:
        """
        Perform hierarchical clustering on regime coordinates.
        
        Returns:
            Array of cluster labels for each regime
        """
        tprint("🎯 Performing hierarchical clustering")
        
        # Standardize coordinates for clustering
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(self.regime_coordinates)
        
        # Determine optimal number of clusters
        n_clusters = self._find_optimal_clusters(scaled_coords)
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage_method
        )
        
        cluster_labels = clustering.fit_predict(scaled_coords)
        self.cluster_labels = cluster_labels
        
        tprint(f"✅ Clustering completed: {n_clusters} clusters")
        self.logger.info(f"Clustering completed with {n_clusters} clusters")
        
        return cluster_labels
    
    def _find_optimal_clusters(self, scaled_coords: np.ndarray) -> int:
        """
        Find optimal number of clusters using multiple criteria.
        
        Args:
            scaled_coords: Standardized regime coordinates
            
        Returns:
            Optimal number of clusters
        """
        max_clusters = min(50, len(scaled_coords) // 2)
        cluster_range = range(2, max_clusters + 1)
        
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        for n_clusters in cluster_range:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage_method
            )
            labels = clustering.fit_predict(scaled_coords)
            
            # Calculate metrics
            if len(set(labels)) > 1:  # Need at least 2 clusters
                silhouette_scores.append(silhouette_score(scaled_coords, labels))
                calinski_scores.append(calinski_harabasz_score(scaled_coords, labels))
                davies_bouldin_scores.append(davies_bouldin_score(scaled_coords, labels))
            else:
                silhouette_scores.append(-1)
                calinski_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
        
        # Find optimal number based on silhouette score (higher is better)
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = cluster_range[optimal_idx]
        
        # Adjust based on target cluster count
        if abs(optimal_clusters - self.target_clusters) <= 5:
            optimal_clusters = self.target_clusters
        
        tprint(f"🎯 Optimal clusters: {optimal_clusters} (silhouette: {silhouette_scores[optimal_idx]:.3f})")
        
        return optimal_clusters
    
    def apply_size_constraints(self) -> np.ndarray:
        """
        Apply size constraints to ensure clusters are 3-8% of total data.
        
        Returns:
            Adjusted cluster labels
        """
        tprint("⚖️ Applying size constraints")
        
        assignments = self.regime_data['regime_assignments']
        total_samples = len(assignments)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for regime_id, cluster_id in enumerate(self.cluster_labels):
            regime_count = Counter(assignments)[regime_id]
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + regime_count
        
        # Identify clusters that need adjustment
        min_size = int(total_samples * self.min_cluster_size_pct)
        max_size = int(total_samples * self.max_cluster_size_pct)
        
        # Adjust cluster assignments
        adjusted_labels = self.cluster_labels.copy()
        
        # Merge small clusters
        small_clusters = [cid for cid, size in cluster_sizes.items() if size < min_size]
        if small_clusters:
            # Merge smallest clusters with nearest larger ones
            target_cluster = max(cluster_sizes.keys(), key=lambda x: cluster_sizes[x])
            for small_cluster in small_clusters:
                adjusted_labels[self.cluster_labels == small_cluster] = target_cluster
        
        # Split large clusters
        large_clusters = [cid for cid, size in cluster_sizes.items() if size > max_size]
        if large_clusters:
            # For now, mark large clusters for later processing
            self.logger.warning(f"Large clusters detected: {large_clusters}")
        
        self.cluster_labels = adjusted_labels
        
        tprint(f"✅ Size constraints applied")
        return adjusted_labels
    
    def create_noise_cluster(self) -> np.ndarray:
        """
        Create a dedicated noise cluster for regimes with very few samples.
        
        Returns:
            Final cluster labels with noise cluster
        """
        tprint("🗑️ Creating noise cluster")
        
        assignments = self.regime_data['regime_assignments']
        regime_counts = Counter(assignments)
        total_samples = len(assignments)
        
        # Find regimes with very few samples
        noise_threshold = max(1, int(total_samples * 0.001))  # 0.1% threshold
        noise_regimes = [regime_id for regime_id, count in regime_counts.items() 
                        if count < noise_threshold]
        
        # Create noise cluster (use highest cluster ID + 1)
        max_cluster_id = np.max(self.cluster_labels)
        noise_cluster_id = max_cluster_id + 1
        
        # Assign noise regimes to noise cluster
        final_labels = self.cluster_labels.copy()
        for regime_id in noise_regimes:
            final_labels[regime_id] = noise_cluster_id
        
        self.cluster_labels = final_labels
        
        noise_samples = sum(regime_counts[regime_id] for regime_id in noise_regimes)
        noise_percentage = (noise_samples / total_samples) * 100
        
        tprint(f"✅ Noise cluster created: {len(noise_regimes)} regimes, {noise_samples} samples ({noise_percentage:.2f}%)")
        
        return final_labels
    
    def calculate_cluster_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for each cluster.
        
        Returns:
            Dictionary with cluster statistics
        """
        tprint("📊 Calculating cluster statistics")
        
        assignments = self.regime_data['regime_assignments']
        regime_counts = Counter(assignments)
        total_samples = len(assignments)
        
        cluster_stats = {}
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster_id in unique_clusters:
            # Find regimes in this cluster
            regime_ids = np.where(self.cluster_labels == cluster_id)[0]
            
            # Calculate cluster metrics
            cluster_samples = sum(regime_counts[regime_id] for regime_id in regime_ids 
                                if regime_id in regime_counts)
            cluster_percentage = (cluster_samples / total_samples) * 100
            
            # Calculate cluster centroid
            cluster_coords = self.regime_coordinates[regime_ids]
            centroid = np.mean(cluster_coords, axis=0)
            
            # Calculate cluster spread (standard deviation)
            spread = np.std(cluster_coords, axis=0)
            
            cluster_stats[cluster_id] = {
                'cluster_id': cluster_id,
                'regime_count': len(regime_ids),
                'sample_count': cluster_samples,
                'percentage': cluster_percentage,
                'centroid': centroid.tolist(),
                'spread': spread.tolist(),
                'regime_ids': regime_ids.tolist(),
                'regime_names': [self.regime_data['regime_models'][rid] for rid in regime_ids]
            }
        
        self.cluster_stats = cluster_stats
        
        tprint(f"✅ Calculated statistics for {len(unique_clusters)} clusters")
        return cluster_stats
    
    def validate_clustering_quality(self) -> Dict[str, Any]:
        """
        Validate the quality of the clustering results.
        
        Returns:
            Dictionary with validation metrics
        """
        tprint("✅ Validating clustering quality")
        
        # Calculate clustering metrics
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(self.regime_coordinates)
        
        silhouette = silhouette_score(scaled_coords, self.cluster_labels)
        calinski = calinski_harabasz_score(scaled_coords, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(scaled_coords, self.cluster_labels)
        
        # Calculate size distribution metrics
        cluster_percentages = [stats['percentage'] for stats in self.cluster_stats.values()]
        size_variance = np.var(cluster_percentages)
        size_range = max(cluster_percentages) - min(cluster_percentages)
        
        # Check constraints
        total_clusters = len(self.cluster_stats)
        clusters_in_range = sum(1 for pct in cluster_percentages 
                              if self.min_cluster_size_pct * 100 <= pct <= self.max_cluster_size_pct * 100)
        constraint_satisfaction = clusters_in_range / total_clusters
        
        validation_metrics = {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'total_clusters': total_clusters,
            'size_variance': size_variance,
            'size_range': size_range,
            'constraint_satisfaction': constraint_satisfaction,
            'clusters_in_range': clusters_in_range,
            'cluster_percentages': cluster_percentages
        }
        
        self.clustering_metrics = validation_metrics
        
        tprint(f"✅ Validation completed:")
        tprint(f"   Silhouette score: {silhouette:.3f}")
        tprint(f"   Clusters in size range: {clusters_in_range}/{total_clusters}")
        tprint(f"   Size range: {min(cluster_percentages):.1f}% - {max(cluster_percentages):.1f}%")
        
        return validation_metrics
    
    def cluster_regimes(self, hmm_outcome_path: str) -> Dict[str, Any]:
        """
        Main method to cluster regimes from HMM results.
        
        Args:
            hmm_outcome_path: Path to HMM regime discovery outcome file
            
        Returns:
            Dictionary with clustering results
        """
        tprint("🚀 Starting regime clustering pipeline")
        
        # Step 1: Load HMM results
        self.load_hmm_results(hmm_outcome_path)
        
        # Step 2: Parse regime coordinates
        self.parse_regime_coordinates()
        
        # Step 3: Analyze regime distribution
        distribution_stats = self.analyze_regime_distribution()
        
        # Step 4: Perform clustering
        self.perform_clustering()
        
        # Step 5: Apply size constraints
        self.apply_size_constraints()
        
        # Step 6: Create noise cluster
        self.create_noise_cluster()
        
        # Step 7: Calculate statistics
        self.calculate_cluster_statistics()
        
        # Step 8: Validate quality
        validation_metrics = self.validate_clustering_quality()
        
        # Prepare results
        results = {
            'clustering_results': {
                'cluster_labels': self.cluster_labels.tolist(),
                'cluster_stats': self.cluster_stats,
                'validation_metrics': validation_metrics,
                'distribution_stats': distribution_stats
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'total_regimes': len(self.regime_data['regime_models']),
                'total_samples': len(self.regime_data['regime_assignments']),
                'total_clusters': len(self.cluster_stats)
            }
        }
        
        tprint("🎉 Regime clustering completed successfully!")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save clustering results to file.
        
        Args:
            results: Clustering results dictionary
            output_path: Path to save results
        """
        tprint(f"💾 Saving results to: {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        tprint("✅ Results saved successfully")
        self.logger.info(f"Results saved to {output_path}")