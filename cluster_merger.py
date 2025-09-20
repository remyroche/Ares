"""
Micro-Cluster Merging for HMM Market Regime Clustering

This module provides intelligent merging of micro-clusters based on similarity,
size constraints, and economic interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass

@dataclass
class MergeCandidate:
    """Represents a potential cluster merge"""
    cluster1: int
    cluster2: int
    similarity: float
    combined_size: int
    merge_benefit: float
    economic_coherence: float = 0.0

class ClusterMerger:
    """
    Intelligent cluster merging system for market regime clustering
    """
    
    def __init__(self, 
                 min_cluster_size: int = 50,
                 max_cluster_size: int = 1000,
                 similarity_threshold: float = 0.8,
                 economic_weight: float = 0.3):
        """
        Initialize cluster merger
        
        Args:
            min_cluster_size: Minimum samples per cluster
            max_cluster_size: Maximum samples per cluster
            similarity_threshold: Minimum similarity for merging
            economic_weight: Weight for economic interpretability in merge decisions
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.similarity_threshold = similarity_threshold
        self.economic_weight = economic_weight
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.merge_history = []
        self.cluster_statistics = {}
    
    def merge_micro_clusters(self, 
                           data: np.ndarray,
                           cluster_labels: np.ndarray,
                           regime_features: Optional[Dict] = None,
                           preserve_large_clusters: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Main function to merge micro-clusters
        
        Args:
            data: Feature matrix (n_samples, n_features)
            cluster_labels: Original cluster assignments
            regime_features: Additional regime characteristics
            preserve_large_clusters: Whether to avoid merging large clusters
            
        Returns:
            Tuple of (new_cluster_labels, merge_report)
        """
        
        self.logger.info("Starting micro-cluster merging process...")
        
        # Analyze current cluster distribution
        cluster_stats = self._analyze_cluster_distribution(data, cluster_labels)
        self.cluster_statistics = cluster_stats
        
        # Identify merge candidates
        merge_candidates = self._identify_merge_candidates(
            data, cluster_labels, regime_features, preserve_large_clusters
        )
        
        # Execute merges
        new_labels, merge_report = self._execute_merges(
            data, cluster_labels, merge_candidates, regime_features
        )
        
        # Validate merge results
        final_stats = self._analyze_cluster_distribution(data, new_labels)
        merge_report['before_stats'] = cluster_stats
        merge_report['after_stats'] = final_stats
        merge_report['improvement_metrics'] = self._calculate_improvement_metrics(
            cluster_stats, final_stats
        )
        
        self.logger.info(f"Merging complete: {len(np.unique(cluster_labels))} → {len(np.unique(new_labels))} clusters")
        
        return new_labels, merge_report
    
    def _analyze_cluster_distribution(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """Analyze current cluster size distribution and characteristics"""
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Categorize clusters by size
        micro_clusters = []  # < min_cluster_size
        small_clusters = []  # min_cluster_size to 2*min_cluster_size
        medium_clusters = []  # 2*min_cluster_size to max_cluster_size/2
        large_clusters = []  # > max_cluster_size/2
        
        for label, count in zip(unique_labels, counts):
            if count < self.min_cluster_size:
                micro_clusters.append(int(label))
            elif count < 2 * self.min_cluster_size:
                small_clusters.append(int(label))
            elif count < self.max_cluster_size // 2:
                medium_clusters.append(int(label))
            else:
                large_clusters.append(int(label))
        
        stats = {
            'total_clusters': len(unique_labels),
            'cluster_sizes': dict(zip(unique_labels.astype(int), counts.astype(int))),
            'micro_clusters': micro_clusters,
            'small_clusters': small_clusters,
            'medium_clusters': medium_clusters,
            'large_clusters': large_clusters,
            'size_statistics': {
                'min': int(np.min(counts)),
                'max': int(np.max(counts)),
                'mean': float(np.mean(counts)),
                'median': float(np.median(counts)),
                'std': float(np.std(counts))
            },
            'samples_in_micro_clusters': sum(counts[np.isin(unique_labels, micro_clusters)]),
            'micro_cluster_ratio': len(micro_clusters) / len(unique_labels)
        }
        
        return stats
    
    def _identify_merge_candidates(self, 
                                 data: np.ndarray,
                                 labels: np.ndarray,
                                 regime_features: Optional[Dict] = None,
                                 preserve_large_clusters: bool = True) -> List[MergeCandidate]:
        """Identify potential cluster merges"""
        
        unique_labels = np.unique(labels)
        merge_candidates = []
        
        # Calculate cluster centroids and characteristics
        cluster_centroids = {}
        cluster_characteristics = {}
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_centroids[label] = np.mean(cluster_data, axis=0)
                cluster_characteristics[label] = {
                    'size': len(cluster_data),
                    'std': np.std(cluster_data, axis=0),
                    'variance': np.var(cluster_data, axis=0)
                }
        
        # Identify merge candidates
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels[i+1:], i+1):
                
                size1 = cluster_characteristics[label1]['size']
                size2 = cluster_characteristics[label2]['size']
                combined_size = size1 + size2
                
                # Skip if combined cluster would be too large
                if combined_size > self.max_cluster_size:
                    continue
                
                # Skip if both clusters are large and we want to preserve them
                if preserve_large_clusters and size1 > self.max_cluster_size//2 and size2 > self.max_cluster_size//2:
                    continue
                
                # Calculate similarity between clusters
                similarity = self._calculate_cluster_similarity(
                    cluster_centroids[label1], 
                    cluster_centroids[label2],
                    cluster_characteristics[label1],
                    cluster_characteristics[label2]
                )
                
                # Calculate merge benefit (higher for micro-clusters)
                merge_benefit = self._calculate_merge_benefit(size1, size2, similarity)
                
                # Economic coherence (if regime features available)
                economic_coherence = 0.0
                if regime_features is not None:
                    economic_coherence = self._calculate_economic_coherence(
                        label1, label2, labels, regime_features
                    )
                
                # Create merge candidate if above threshold
                if similarity >= self.similarity_threshold or (size1 < self.min_cluster_size or size2 < self.min_cluster_size):
                    candidate = MergeCandidate(
                        cluster1=int(label1),
                        cluster2=int(label2),
                        similarity=similarity,
                        combined_size=combined_size,
                        merge_benefit=merge_benefit,
                        economic_coherence=economic_coherence
                    )
                    merge_candidates.append(candidate)
        
        # Sort candidates by merge benefit (descending)
        merge_candidates.sort(key=lambda x: x.merge_benefit, reverse=True)
        
        self.logger.info(f"Identified {len(merge_candidates)} merge candidates")
        
        return merge_candidates
    
    def _calculate_cluster_similarity(self, 
                                    centroid1: np.ndarray, 
                                    centroid2: np.ndarray,
                                    chars1: Dict,
                                    chars2: Dict) -> float:
        """Calculate similarity between two clusters"""
        
        # Centroid similarity (cosine similarity)
        centroid_sim = cosine_similarity([centroid1], [centroid2])[0, 0]
        
        # Shape similarity (based on standard deviations)
        std1 = chars1['std']
        std2 = chars2['std']
        
        # Avoid division by zero
        std1 = np.where(std1 == 0, 1e-8, std1)
        std2 = np.where(std2 == 0, 1e-8, std2)
        
        # Calculate shape similarity
        shape_sim = np.mean(1.0 - np.abs(std1 - std2) / (std1 + std2 + 1e-8))
        
        # Combined similarity
        similarity = 0.7 * centroid_sim + 0.3 * shape_sim
        
        return float(similarity)
    
    def _calculate_merge_benefit(self, size1: int, size2: int, similarity: float) -> float:
        """Calculate benefit of merging two clusters"""
        
        # Size benefit (higher for micro-clusters)
        size_benefit = 0.0
        if size1 < self.min_cluster_size:
            size_benefit += (self.min_cluster_size - size1) / self.min_cluster_size
        if size2 < self.min_cluster_size:
            size_benefit += (self.min_cluster_size - size2) / self.min_cluster_size
        
        # Similarity benefit
        similarity_benefit = similarity
        
        # Combined benefit
        total_benefit = 0.6 * size_benefit + 0.4 * similarity_benefit
        
        return float(total_benefit)
    
    def _calculate_economic_coherence(self, 
                                    label1: int, 
                                    label2: int,
                                    labels: np.ndarray,
                                    regime_features: Dict) -> float:
        """Calculate economic coherence between two clusters"""
        
        # This is a placeholder - implement based on your regime features
        # For example, check if clusters represent similar market conditions
        
        try:
            # Example: Compare average returns, volatility, momentum
            cluster1_mask = labels == label1
            cluster2_mask = labels == label2
            
            # This would depend on your regime_features structure
            # coherence = compare_regime_characteristics(
            #     regime_features[cluster1_mask], 
            #     regime_features[cluster2_mask]
            # )
            
            # Placeholder return
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Could not calculate economic coherence: {e}")
            return 0.0
    
    def _execute_merges(self, 
                       data: np.ndarray,
                       original_labels: np.ndarray,
                       merge_candidates: List[MergeCandidate],
                       regime_features: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Execute the selected merges"""
        
        new_labels = original_labels.copy()
        executed_merges = []
        cluster_mapping = {}  # old_label -> new_label
        
        # Initialize mapping
        for label in np.unique(original_labels):
            cluster_mapping[label] = label
        
        # Track which clusters have been merged
        merged_clusters = set()
        
        for candidate in merge_candidates:
            cluster1 = candidate.cluster1
            cluster2 = candidate.cluster2
            
            # Skip if either cluster has already been merged
            if cluster1 in merged_clusters or cluster2 in merged_clusters:
                continue
            
            # Get current labels (after previous merges)
            current_label1 = cluster_mapping[cluster1]
            current_label2 = cluster_mapping[cluster2]
            
            # Skip if they're already the same cluster (due to previous merges)
            if current_label1 == current_label2:
                continue
            
            # Check if merge is still beneficial after previous merges
            current_size1 = np.sum(new_labels == current_label1)
            current_size2 = np.sum(new_labels == current_label2)
            combined_size = current_size1 + current_size2
            
            if combined_size > self.max_cluster_size:
                continue
            
            # Execute merge: assign all samples from cluster2 to cluster1
            merge_mask = new_labels == current_label2
            new_labels[merge_mask] = current_label1
            
            # Update mapping
            for old_label, new_label in cluster_mapping.items():
                if new_label == current_label2:
                    cluster_mapping[old_label] = current_label1
            
            # Record merge
            merge_info = {
                'original_clusters': [cluster1, cluster2],
                'current_clusters': [current_label1, current_label2],
                'new_cluster': current_label1,
                'similarity': candidate.similarity,
                'combined_size': combined_size,
                'merge_benefit': candidate.merge_benefit
            }
            executed_merges.append(merge_info)
            
            # Mark clusters as merged
            merged_clusters.add(cluster1)
            merged_clusters.add(cluster2)
            
            self.logger.debug(f"Merged clusters {cluster1} and {cluster2} -> {current_label1}")
        
        # Relabel clusters to be consecutive
        new_labels = self._relabel_clusters(new_labels)
        
        merge_report = {
            'executed_merges': executed_merges,
            'total_merges': len(executed_merges),
            'clusters_before': len(np.unique(original_labels)),
            'clusters_after': len(np.unique(new_labels)),
            'cluster_reduction': len(np.unique(original_labels)) - len(np.unique(new_labels)),
            'cluster_mapping': cluster_mapping
        }
        
        return new_labels, merge_report
    
    def _relabel_clusters(self, labels: np.ndarray) -> np.ndarray:
        """Relabel clusters to be consecutive integers starting from 0"""
        
        unique_labels = np.unique(labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        new_labels = np.array([label_mapping[label] for label in labels])
        
        return new_labels
    
    def _calculate_improvement_metrics(self, before_stats: Dict, after_stats: Dict) -> Dict:
        """Calculate improvement metrics after merging"""
        
        improvements = {
            'cluster_reduction': before_stats['total_clusters'] - after_stats['total_clusters'],
            'micro_cluster_reduction': len(before_stats['micro_clusters']) - len(after_stats['micro_clusters']),
            'micro_cluster_ratio_improvement': before_stats['micro_cluster_ratio'] - after_stats['micro_cluster_ratio'],
            'size_distribution_improvement': {
                'mean_size_increase': after_stats['size_statistics']['mean'] - before_stats['size_statistics']['mean'],
                'std_reduction': before_stats['size_statistics']['std'] - after_stats['size_statistics']['std'],
                'min_size_increase': after_stats['size_statistics']['min'] - before_stats['size_statistics']['min']
            }
        }
        
        return improvements
    
    def hierarchical_merge(self, 
                          data: np.ndarray,
                          cluster_labels: np.ndarray,
                          target_clusters: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Alternative: Hierarchical merging approach
        
        Uses hierarchical clustering to merge similar clusters
        """
        
        unique_labels = np.unique(cluster_labels)
        
        # Calculate cluster centroids
        centroids = []
        cluster_sizes = []
        
        for label in unique_labels:
            cluster_data = data[cluster_labels == label]
            centroids.append(np.mean(cluster_data, axis=0))
            cluster_sizes.append(len(cluster_data))
        
        centroids = np.array(centroids)
        cluster_sizes = np.array(cluster_sizes)
        
        # Perform hierarchical clustering on centroids
        linkage_matrix = linkage(centroids, method='ward')
        
        # Determine number of clusters
        if target_clusters is None:
            # Use elbow method or set based on minimum cluster size
            target_clusters = max(10, len(unique_labels) // 2)
        
        # Get cluster assignments
        hierarchical_labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')
        
        # Map back to original data
        new_labels = np.zeros_like(cluster_labels)
        for i, old_label in enumerate(unique_labels):
            mask = cluster_labels == old_label
            new_labels[mask] = hierarchical_labels[i] - 1  # Convert to 0-based
        
        merge_report = {
            'method': 'hierarchical',
            'clusters_before': len(unique_labels),
            'clusters_after': target_clusters,
            'linkage_matrix': linkage_matrix
        }
        
        return new_labels, merge_report
    
    def print_merge_report(self, merge_report: Dict) -> None:
        """Print a comprehensive merge report"""
        
        print("=" * 60)
        print("CLUSTER MERGING REPORT")
        print("=" * 60)
        
        print(f"\nCLUSTER REDUCTION:")
        print(f"  Before: {merge_report['clusters_before']} clusters")
        print(f"  After: {merge_report['clusters_after']} clusters")
        print(f"  Reduction: {merge_report['cluster_reduction']} clusters")
        
        if 'executed_merges' in merge_report:
            print(f"\nMERGES EXECUTED: {merge_report['total_merges']}")
            
            if merge_report['executed_merges']:
                print("\nTop 5 merges by benefit:")
                for i, merge in enumerate(merge_report['executed_merges'][:5]):
                    print(f"  {i+1}. Clusters {merge['original_clusters']} → {merge['new_cluster']} "
                          f"(similarity: {merge['similarity']:.3f}, size: {merge['combined_size']})")
        
        if 'improvement_metrics' in merge_report:
            improvements = merge_report['improvement_metrics']
            print(f"\nIMPROVEMENTS:")
            print(f"  Micro-cluster reduction: {improvements['micro_cluster_reduction']}")
            print(f"  Micro-cluster ratio improvement: {improvements['micro_cluster_ratio_improvement']:.3f}")
            print(f"  Mean cluster size increase: {improvements['size_distribution_improvement']['mean_size_increase']:.1f}")
        
        print("=" * 60)


# Integration function to use both validation and merging
def validate_and_merge_clusters(data: np.ndarray, 
                              cluster_labels: np.ndarray,
                              regime_features: Optional[Dict] = None,
                              min_cluster_size: int = 50,
                              similarity_threshold: float = 0.8) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Complete workflow: validate clustering and merge micro-clusters if needed
    
    Args:
        data: Feature matrix
        cluster_labels: Original cluster assignments
        regime_features: Additional regime characteristics
        min_cluster_size: Minimum samples per cluster
        similarity_threshold: Similarity threshold for merging
        
    Returns:
        Tuple of (final_labels, validation_results, merge_report)
    """
    
    # Step 1: Validate original clustering
    from cluster_validation import ClusterValidator
    
    validator = ClusterValidator(min_cluster_size=min_cluster_size)
    validation_results = validator.validate_clustering(data, cluster_labels, regime_features)
    
    print("Initial validation results:")
    validator.print_validation_report()
    
    # Step 2: Merge micro-clusters if validation suggests it
    final_labels = cluster_labels
    merge_report = {'status': 'no_merging_needed'}
    
    if not validation_results['validation_passed']:
        print("\nValidation failed. Attempting cluster merging...")
        
        merger = ClusterMerger(
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold
        )
        
        final_labels, merge_report = merger.merge_micro_clusters(
            data, cluster_labels, regime_features
        )
        
        print("\nMerge results:")
        merger.print_merge_report(merge_report)
        
        # Step 3: Re-validate after merging
        print("\nRe-validating after merging...")
        final_validation = validator.validate_clustering(data, final_labels, regime_features)
        validator.print_validation_report()
        
        merge_report['final_validation'] = final_validation
    
    return final_labels, validation_results, merge_report


# Example usage
def example_usage():
    """Example of how to use the cluster merger"""
    
    # Generate example data
    np.random.seed(42)
    n_samples = 5000
    n_features = 15
    data = np.random.randn(n_samples, n_features)
    
    # Generate problematic cluster labels (many micro-clusters)
    cluster_labels = np.random.choice(range(100), n_samples)  # 100 clusters
    
    # Run complete validation and merging workflow
    final_labels, validation_results, merge_report = validate_and_merge_clusters(
        data, cluster_labels, min_cluster_size=30, similarity_threshold=0.75
    )
    
    print(f"\nFinal result: {len(np.unique(cluster_labels))} → {len(np.unique(final_labels))} clusters")
    
    return final_labels, validation_results, merge_report

if __name__ == "__main__":
    example_usage()