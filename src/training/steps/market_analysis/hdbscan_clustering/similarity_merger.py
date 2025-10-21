"""
Data-Driven Similarity Merger

This module provides regime merging capabilities for HDBSCAN-based
regime discovery with data-driven threshold optimization, including 
similarity-based merging, cluster consolidation, and regime optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wasserstein_distance
import warnings

# Import data-driven optimization
try:
    from .optimization.data_driven_merging_thresholds import (
        DataDrivenMergingThresholdOptimizer, RegimeMergingThresholdResult
    )
    from .config.data_driven_config import (
        RegimeMergingThresholdConfig, OptimizationStrategy
    )
    DATA_DRIVEN_AVAILABLE = True
except ImportError:
    DATA_DRIVEN_AVAILABLE = False
    logging.warning("Data-driven merging optimization not available")

logger = logging.getLogger(__name__)

@dataclass
class DataDrivenSimilarityMergerConfig:
    """Configuration for data-driven similarity-based merging."""
    # Enable data-driven optimization
    enable_data_driven_optimization: bool = True
    
    # Fallback parameters (used if data-driven optimization fails)
    similarity_threshold: float = 0.8
    distance_threshold: float = 0.2
    statistical_threshold: float = 0.05
    
    # Merging parameters
    enable_merging: bool = True
    merging_method: str = 'similarity'  # 'similarity', 'hierarchical', 'statistical'
    
    # Similarity metrics
    similarity_metric: str = 'cosine'  # 'cosine', 'euclidean', 'manhattan', 'wasserstein'
    use_feature_weights: bool = True
    feature_weights: Optional[List[float]] = None
    
    # Hierarchical clustering
    linkage_method: str = 'ward'  # 'ward', 'complete', 'average', 'single'
    n_clusters: Optional[int] = None
    distance_threshold_hierarchical: Optional[float] = None
    
    # Statistical merging
    statistical_test: str = 'ks'  # 'ks', 'ttest', 'mannwhitney'
    p_value_threshold: float = 0.05
    multiple_testing_correction: str = 'bonferroni'  # 'bonferroni', 'fdr', 'none'
    
    # Quality constraints
    min_merge_improvement: float = 0.01
    preserve_cluster_count: bool = False
    max_merge_iterations: int = 5
    
    # Validation
    validate_merging: bool = True
    min_cluster_size_after_merge: int = 5
    max_clusters_after_merge: Optional[int] = None

class DataDrivenSimilarityMerger:
    """
    Data-driven similarity-based regime merger for regime discovery.
    
    Provides intelligent merging of similar regimes based on data-driven
    threshold optimization and various similarity metrics.
    """
    
    def __init__(self, config: Optional[DataDrivenSimilarityMergerConfig] = None):
        """
        Initialize data-driven similarity merger.
        
        Args:
            config: Configuration for data-driven similarity merging
        """
        self.config = config or DataDrivenSimilarityMergerConfig()
        self.merging_stats = {}
        self.original_labels = None
        self.merged_labels = None
        self.similarity_matrix = None
        self.optimization_result = None
        
        # Initialize data-driven optimizer if available
        if self.config.enable_data_driven_optimization and DATA_DRIVEN_AVAILABLE:
            self.threshold_optimizer = DataDrivenMergingThresholdOptimizer(
                RegimeMergingThresholdConfig()
            )
        else:
            self.threshold_optimizer = None
        
    def merge_regimes(self, 
                     cluster_labels: np.ndarray,
                     features: np.ndarray,
                     target_metric: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Merge similar regimes based on data-driven threshold optimization.
        
        Args:
            cluster_labels: Cluster labels to merge
            features: Feature matrix
            target_metric: Target quality metric (optional)
            
        Returns:
            Tuple of (merged_labels, merging_info)
        """
        try:
            if not self.config.enable_merging:
                logger.info("Regime merging disabled")
                return cluster_labels, {'merging_performed': False}
            
            logger.info("🔗 Starting data-driven regime merging...")
            
            # Store original data
            self.original_labels = cluster_labels.copy()
            
            # Validate input
            if self.config.validate_merging:
                cluster_labels, features = self._validate_input(cluster_labels, features)
            
            # Optimize thresholds if data-driven optimization is enabled
            if self.config.enable_data_driven_optimization and self.threshold_optimizer:
                try:
                    logger.info("🔍 Optimizing merging thresholds using data-driven methods...")
                    
                    # Create merging function for optimization
                    def merging_func(labels, features, thresholds):
                        return self._merge_with_thresholds(labels, features, thresholds, target_metric)
                    
                    # Optimize thresholds
                    self.optimization_result = self.threshold_optimizer.optimize_thresholds(
                        cluster_labels, features, merging_func
                    )
                    
                    # Use optimized thresholds
                    optimal_thresholds = self.optimization_result.optimal_thresholds
                    logger.info(f"✅ Optimized thresholds: {optimal_thresholds}")
                    
                except Exception as e:
                    logger.warning(f"Data-driven optimization failed: {e}, using fallback thresholds")
                    optimal_thresholds = {
                        'similarity_threshold': self.config.similarity_threshold,
                        'distance_threshold': self.config.distance_threshold,
                        'p_value_threshold': self.config.statistical_threshold
                    }
            else:
                # Use fallback thresholds
                optimal_thresholds = {
                    'similarity_threshold': self.config.similarity_threshold,
                    'distance_threshold': self.config.distance_threshold,
                    'p_value_threshold': self.config.statistical_threshold
                }
            
            # Perform merging with optimized/fallback thresholds
            merged_labels = self._merge_with_thresholds(cluster_labels, features, optimal_thresholds, target_metric)
            
            # Calculate merging statistics
            merging_info = self._calculate_merging_stats(
                cluster_labels, merged_labels, features, optimal_thresholds
            )
            
            # Add optimization results if available
            if self.optimization_result:
                merging_info['optimization_results'] = {
                    'optimization_score': self.optimization_result.optimization_score,
                    'validation_scores': self.optimization_result.validation_scores,
                    'merging_statistics': self.optimization_result.merging_statistics
                }
            
            self.merging_stats = merging_info
            self.merged_labels = merged_labels
            
            logger.info(f"✅ Data-driven regime merging completed. Quality change: {merging_info.get('quality_change', 0):.4f}")
            
            return merged_labels, merging_info
            
        except Exception as e:
            logger.error(f"❌ Data-driven regime merging failed: {e}")
            return cluster_labels, {'error': str(e)}
    
    def _merge_with_thresholds(self, 
                              cluster_labels: np.ndarray,
                              features: np.ndarray,
                              thresholds: Dict[str, float],
                              target_metric: Optional[str]) -> np.ndarray:
        """Merge regimes using specified thresholds."""
        try:
            # Update config with optimized thresholds
            original_sim_thresh = self.config.similarity_threshold
            original_dist_thresh = self.config.distance_threshold
            original_stat_thresh = self.config.statistical_threshold
            
            self.config.similarity_threshold = thresholds.get('similarity_threshold', original_sim_thresh)
            self.config.distance_threshold = thresholds.get('distance_threshold', original_dist_thresh)
            self.config.statistical_threshold = thresholds.get('p_value_threshold', original_stat_thresh)
            
            # Perform merging based on method
            if self.config.merging_method == 'similarity':
                merged_labels = self._merge_by_similarity(cluster_labels, features, target_metric)
            elif self.config.merging_method == 'hierarchical':
                merged_labels = self._merge_by_hierarchical(cluster_labels, features, target_metric)
            elif self.config.merging_method == 'statistical':
                merged_labels = self._merge_by_statistical(cluster_labels, features, target_metric)
            else:
                logger.warning(f"⚠️ Unknown merging method: {self.config.merging_method}")
                merged_labels = self._merge_by_similarity(cluster_labels, features, target_metric)
            
            # Restore original thresholds
            self.config.similarity_threshold = original_sim_thresh
            self.config.distance_threshold = original_dist_thresh
            self.config.statistical_threshold = original_stat_thresh
            
            return merged_labels
            
        except Exception as e:
            logger.error(f"❌ Merging with thresholds failed: {e}")
            return cluster_labels
    
    def _validate_input(self, cluster_labels: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input data for merging."""
        try:
            # Check for sufficient samples
            if len(cluster_labels) < 10:
                logger.warning("⚠️ Insufficient samples for merging")
                return cluster_labels, features
            
            # Check for sufficient clusters
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            if len(unique_labels) < 2:
                logger.warning("⚠️ Insufficient clusters for merging")
                return cluster_labels, features
            
            # Check for NaN or infinite values
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("⚠️ Found NaN or infinite values, cleaning data")
                features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            return cluster_labels, features
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return cluster_labels, features
    
    def _merge_by_similarity(self, 
                           cluster_labels: np.ndarray, 
                           features: np.ndarray,
                           target_metric: Optional[str] = None) -> np.ndarray:
        """Merge regimes based on similarity metrics."""
        try:
            # Get unique clusters
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) < 2:
                return cluster_labels
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(cluster_labels, features, unique_labels)
            self.similarity_matrix = similarity_matrix
            
            # Find similar pairs
            similar_pairs = self._find_similar_pairs(similarity_matrix, unique_labels)
            
            if len(similar_pairs) == 0:
                logger.info("No similar regimes found for merging")
                return cluster_labels
            
            # Merge similar pairs iteratively
            merged_labels = cluster_labels.copy()
            merge_count = 0
            
            for pair in similar_pairs:
                if merge_count >= self.config.max_merge_iterations:
                    break
                
                # Check if merging improves quality
                temp_labels = self._merge_pair(merged_labels, pair)
                if self._should_merge(merged_labels, temp_labels, features, target_metric):
                    merged_labels = temp_labels
                    merge_count += 1
                    logger.debug(f"Merged regimes {pair[0]} and {pair[1]}")
            
            logger.info(f"✅ Merged {merge_count} regime pairs")
            return merged_labels
            
        except Exception as e:
            logger.error(f"❌ Similarity-based merging failed: {e}")
            return cluster_labels
    
    def _merge_by_hierarchical(self, 
                             cluster_labels: np.ndarray, 
                             features: np.ndarray,
                             target_metric: Optional[str] = None) -> np.ndarray:
        """Merge regimes using hierarchical clustering."""
        try:
            # Get unique clusters
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) < 2:
                return cluster_labels
            
            # Calculate cluster centers
            cluster_centers = self._calculate_cluster_centers(cluster_labels, features, unique_labels)
            
            # Perform hierarchical clustering
            if self.config.distance_threshold_hierarchical is not None:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.config.distance_threshold_hierarchical,
                    linkage=self.config.linkage_method
                )
            else:
                n_clusters = self.config.n_clusters or max(2, len(unique_labels) // 2)
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=self.config.linkage_method
                )
            
            cluster_assignments = clustering.fit_predict(cluster_centers)
            
            # Create mapping from old to new labels
            label_mapping = dict(zip(unique_labels, cluster_assignments))
            
            # Apply mapping to cluster labels
            merged_labels = cluster_labels.copy()
            for old_label, new_label in label_mapping.items():
                merged_labels[cluster_labels == old_label] = new_label
            
            logger.info(f"✅ Hierarchical merging completed. {len(unique_labels)} -> {len(set(cluster_assignments))} regimes")
            return merged_labels
            
        except Exception as e:
            logger.error(f"❌ Hierarchical merging failed: {e}")
            return cluster_labels
    
    def _merge_by_statistical(self, 
                            cluster_labels: np.ndarray, 
                            features: np.ndarray,
                            target_metric: Optional[str] = None) -> np.ndarray:
        """Merge regimes based on statistical tests."""
        try:
            # Get unique clusters
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) < 2:
                return cluster_labels
            
            # Perform statistical tests between all pairs
            similar_pairs = self._find_statistically_similar_pairs(cluster_labels, features, unique_labels)
            
            if len(similar_pairs) == 0:
                logger.info("No statistically similar regimes found for merging")
                return cluster_labels
            
            # Merge similar pairs
            merged_labels = cluster_labels.copy()
            merge_count = 0
            
            for pair in similar_pairs:
                if merge_count >= self.config.max_merge_iterations:
                    break
                
                # Check if merging improves quality
                temp_labels = self._merge_pair(merged_labels, pair)
                if self._should_merge(merged_labels, temp_labels, features, target_metric):
                    merged_labels = temp_labels
                    merge_count += 1
                    logger.debug(f"Merged regimes {pair[0]} and {pair[1]} (statistical)")
            
            logger.info(f"✅ Statistically merged {merge_count} regime pairs")
            return merged_labels
            
        except Exception as e:
            logger.error(f"❌ Statistical merging failed: {e}")
            return cluster_labels
    
    def _calculate_similarity_matrix(self, 
                                   cluster_labels: np.ndarray, 
                                   features: np.ndarray,
                                   unique_labels: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between regimes."""
        try:
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)  # Initialize with identity matrix
            
            # Calculate cluster centers
            cluster_centers = self._calculate_cluster_centers(cluster_labels, features, unique_labels)
            
            # Calculate pairwise similarities
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    similarity = self._calculate_similarity(
                        cluster_centers[i], cluster_centers[j], features, 
                        cluster_labels, unique_labels[i], unique_labels[j]
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"❌ Similarity matrix calculation failed: {e}")
            return np.eye(len(unique_labels))
    
    def _calculate_similarity(self, 
                            center1: np.ndarray, 
                            center2: np.ndarray,
                            features: np.ndarray,
                            cluster_labels: np.ndarray,
                            label1: int,
                            label2: int) -> float:
        """Calculate similarity between two regimes."""
        try:
            if self.config.similarity_metric == 'cosine':
                # Cosine similarity
                dot_product = np.dot(center1, center2)
                norm1 = np.linalg.norm(center1)
                norm2 = np.linalg.norm(center2)
                return dot_product / (norm1 * norm2 + 1e-10)
            
            elif self.config.similarity_metric == 'euclidean':
                # Euclidean distance (convert to similarity)
                distance = np.linalg.norm(center1 - center2)
                return 1.0 / (1.0 + distance)
            
            elif self.config.similarity_metric == 'manhattan':
                # Manhattan distance (convert to similarity)
                distance = np.sum(np.abs(center1 - center2))
                return 1.0 / (1.0 + distance)
            
            elif self.config.similarity_metric == 'wasserstein':
                # Wasserstein distance (convert to similarity)
                # Get samples from each cluster
                samples1 = features[cluster_labels == label1]
                samples2 = features[cluster_labels == label2]
                
                if len(samples1) == 0 or len(samples2) == 0:
                    return 0.0
                
                # Calculate Wasserstein distance for each feature
                distances = []
                for i in range(features.shape[1]):
                    dist = wasserstein_distance(samples1[:, i], samples2[:, i])
                    distances.append(dist)
                
                avg_distance = np.mean(distances)
                return 1.0 / (1.0 + avg_distance)
            
            else:
                # Default to cosine similarity
                dot_product = np.dot(center1, center2)
                norm1 = np.linalg.norm(center1)
                norm2 = np.linalg.norm(center2)
                return dot_product / (norm1 * norm2 + 1e-10)
                
        except Exception as e:
            logger.debug(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_cluster_centers(self, 
                                 cluster_labels: np.ndarray, 
                                 features: np.ndarray,
                                 unique_labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        try:
            centers = []
            for label in unique_labels:
                mask = cluster_labels == label
                if mask.sum() > 0:
                    center = np.mean(features[mask], axis=0)
                    centers.append(center)
                else:
                    centers.append(np.zeros(features.shape[1]))
            
            return np.array(centers)
            
        except Exception as e:
            logger.error(f"❌ Cluster center calculation failed: {e}")
            return np.array([])
    
    def _find_similar_pairs(self, 
                          similarity_matrix: np.ndarray, 
                          unique_labels: np.ndarray) -> List[Tuple[int, int]]:
        """Find pairs of similar regimes."""
        try:
            similar_pairs = []
            n_clusters = len(unique_labels)
            
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    similarity = similarity_matrix[i, j]
                    
                    if similarity > self.config.similarity_threshold:
                        similar_pairs.append((unique_labels[i], unique_labels[j]))
            
            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: similarity_matrix[
                np.where(unique_labels == x[0])[0][0], 
                np.where(unique_labels == x[1])[0][0]
            ], reverse=True)
            
            return similar_pairs
            
        except Exception as e:
            logger.error(f"❌ Similar pair finding failed: {e}")
            return []
    
    def _find_statistically_similar_pairs(self, 
                                        cluster_labels: np.ndarray, 
                                        features: np.ndarray,
                                        unique_labels: np.ndarray) -> List[Tuple[int, int]]:
        """Find statistically similar regime pairs."""
        try:
            similar_pairs = []
            n_clusters = len(unique_labels)
            
            # Apply multiple testing correction
            alpha = self.config.statistical_threshold
            if self.config.multiple_testing_correction == 'bonferroni':
                alpha = alpha / (n_clusters * (n_clusters - 1) / 2)
            elif self.config.multiple_testing_correction == 'fdr':
                # FDR correction will be applied later
                pass
            
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    label1, label2 = unique_labels[i], unique_labels[j]
                    
                    # Get samples from each cluster
                    samples1 = features[cluster_labels == label1]
                    samples2 = features[cluster_labels == label2]
                    
                    if len(samples1) < 2 or len(samples2) < 2:
                        continue
                    
                    # Perform statistical test
                    p_value = self._perform_statistical_test(samples1, samples2)
                    
                    if p_value > alpha:  # Not significantly different
                        similar_pairs.append((label1, label2))
            
            return similar_pairs
            
        except Exception as e:
            logger.error(f"❌ Statistical pair finding failed: {e}")
            return []
    
    def _perform_statistical_test(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """Perform statistical test between two samples."""
        try:
            if self.config.statistical_test == 'ks':
                # Kolmogorov-Smirnov test
                from scipy.stats import ks_2samp
                # Test each feature and take minimum p-value
                p_values = []
                for i in range(samples1.shape[1]):
                    _, p_value = ks_2samp(samples1[:, i], samples2[:, i])
                    p_values.append(p_value)
                return min(p_values)
            
            elif self.config.statistical_test == 'ttest':
                # Two-sample t-test
                from scipy.stats import ttest_ind
                # Test each feature and take minimum p-value
                p_values = []
                for i in range(samples1.shape[1]):
                    _, p_value = ttest_ind(samples1[:, i], samples2[:, i])
                    p_values.append(p_value)
                return min(p_values)
            
            elif self.config.statistical_test == 'mannwhitney':
                # Mann-Whitney U test
                from scipy.stats import mannwhitneyu
                # Test each feature and take minimum p-value
                p_values = []
                for i in range(samples1.shape[1]):
                    _, p_value = mannwhitneyu(samples1[:, i], samples2[:, i], alternative='two-sided')
                    p_values.append(p_value)
                return min(p_values)
            
            else:
                # Default to KS test
                from scipy.stats import ks_2samp
                p_values = []
                for i in range(samples1.shape[1]):
                    _, p_value = ks_2samp(samples1[:, i], samples2[:, i])
                    p_values.append(p_value)
                return min(p_values)
                
        except Exception as e:
            logger.debug(f"Statistical test failed: {e}")
            return 1.0  # Return high p-value (not significant)
    
    def _merge_pair(self, cluster_labels: np.ndarray, pair: Tuple[int, int]) -> np.ndarray:
        """Merge a pair of regimes."""
        try:
            merged_labels = cluster_labels.copy()
            label1, label2 = pair
            
            # Merge label2 into label1
            merged_labels[cluster_labels == label2] = label1
            
            return merged_labels
            
        except Exception as e:
            logger.error(f"❌ Pair merging failed: {e}")
            return cluster_labels
    
    def _should_merge(self, 
                     original_labels: np.ndarray, 
                     merged_labels: np.ndarray,
                     features: np.ndarray,
                     target_metric: Optional[str] = None) -> bool:
        """Determine if merging should be performed based on quality improvement."""
        try:
            # Calculate quality metrics
            original_quality = self._calculate_quality_metric(original_labels, features, target_metric)
            merged_quality = self._calculate_quality_metric(merged_labels, features, target_metric)
            
            # Check for improvement
            improvement = merged_quality - original_quality
            
            # Check minimum improvement threshold
            if improvement < self.config.min_merge_improvement:
                return False
            
            # Check cluster size constraints
            if self.config.min_cluster_size_after_merge > 0:
                unique_labels = np.unique(merged_labels)
                unique_labels = unique_labels[unique_labels != -1]
                
                for label in unique_labels:
                    cluster_size = np.sum(merged_labels == label)
                    if cluster_size < self.config.min_cluster_size_after_merge:
                        return False
            
            # Check maximum cluster count
            if self.config.max_clusters_after_merge is not None:
                unique_labels = np.unique(merged_labels)
                unique_labels = unique_labels[unique_labels != -1]
                if len(unique_labels) > self.config.max_clusters_after_merge:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Merge decision failed: {e}")
            return False
    
    def _calculate_quality_metric(self, 
                                cluster_labels: np.ndarray, 
                                features: np.ndarray,
                                target_metric: Optional[str] = None) -> float:
        """Calculate clustering quality metric."""
        try:
            metric = target_metric or 'silhouette'
            
            # Remove noise points for validation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate metric
            if metric == 'silhouette':
                return silhouette_score(valid_features, valid_labels)
            elif metric == 'calinski_harabasz':
                return calinski_harabasz_score(valid_features, valid_labels)
            elif metric == 'davies_bouldin':
                return -davies_bouldin_score(valid_features, valid_labels)  # Negative because lower is better
            else:
                return silhouette_score(valid_features, valid_labels)
            
        except Exception as e:
            logger.debug(f"Quality metric calculation failed: {e}")
            return -np.inf
    
    def _calculate_merging_stats(self, 
                               original_labels: np.ndarray,
                               merged_labels: np.ndarray,
                               features: np.ndarray,
                               thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Calculate merging statistics."""
        try:
            # Basic statistics
            n_samples = len(original_labels)
            
            # Regime count changes
            original_regimes = len(set(original_labels)) - (1 if -1 in original_labels else 0)
            merged_regimes = len(set(merged_labels)) - (1 if -1 in merged_labels else 0)
            
            # Quality changes
            original_quality = self._calculate_quality_metric(original_labels, features)
            merged_quality = self._calculate_quality_metric(merged_labels, features)
            quality_change = merged_quality - original_quality
            
            # Regime size changes
            original_sizes = self._calculate_regime_sizes(original_labels)
            merged_sizes = self._calculate_regime_sizes(merged_labels)
            
            stats = {
                'merging_performed': True,
                'n_samples': n_samples,
                'original_regimes': original_regimes,
                'merged_regimes': merged_regimes,
                'regime_reduction': original_regimes - merged_regimes,
                'original_quality': original_quality,
                'merged_quality': merged_quality,
                'quality_change': quality_change,
                'original_regime_sizes': original_sizes,
                'merged_regime_sizes': merged_sizes,
                'similarity_matrix': self.similarity_matrix,
                'thresholds_used': thresholds
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Merging stats calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """Calculate regime sizes."""
        try:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            return dict(zip(unique_labels, counts))
        except Exception as e:
            logger.debug(f"Regime size calculation failed: {e}")
            return {}
    
    def get_merging_stats(self) -> Dict[str, Any]:
        """Get merging statistics."""
        return self.merging_stats.copy()
    
    def get_original_labels(self) -> Optional[np.ndarray]:
        """Get original cluster labels."""
        return self.original_labels.copy() if self.original_labels is not None else None
    
    def get_merged_labels(self) -> Optional[np.ndarray]:
        """Get merged cluster labels."""
        return self.merged_labels.copy() if self.merged_labels is not None else None
    
    def get_similarity_matrix(self) -> Optional[np.ndarray]:
        """Get similarity matrix."""
        return self.similarity_matrix.copy() if self.similarity_matrix is not None else None
    
    def get_optimization_results(self) -> Optional[RegimeMergingThresholdResult]:
        """Get data-driven optimization results."""
        return self.optimization_result


# Alias for backward compatibility
SimilarityMerger = DataDrivenSimilarityMerger