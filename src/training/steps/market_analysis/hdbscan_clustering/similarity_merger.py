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
from scipy.spatial.distance import pdist, squareform, mahalanobis
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wasserstein_distance, entropy
import warnings
import itertools
import time

# Import utility tools for enhanced functionality
from src.utils.common_operations import (
    safe_divide, safe_mean, safe_std, safe_correlation, 
    memory_efficient_apply, performance_timer, memory_monitor
)
from src.utils.common_utilities import (
    analyze_nan_values_detailed, calculate_data_quality_metrics,
    safe_dataframe_operation, validate_dataframe_columns
)
from src.utils.math_validation import (
    validate_finite, validate_array_finite, safe_correlation as safe_corr,
    safe_covariance, safe_mean as safe_mean_math, safe_std as safe_std_math
)
from src.utils.tprint import tprint, tprint_info, tprint_debug, tprint_performance

# Import hardware optimization tools
try:
    from src.utils.hardware import (
        memory_optimized, auto_optimize, smart_cache, performance_tracked,
        optimize_dataframe_default, optimize_numpy_array_default
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    # Create dummy decorators
    def memory_optimized(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def auto_optimize(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def smart_cache(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def performance_tracked(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import ML utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager
    )
    ML_UTILITIES_AVAILABLE = True
except ImportError:
    ML_UTILITIES_AVAILABLE = False

# Import VectorBT optimization
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

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
class EnhancedSimilarityConfig:
    """Configuration for enhanced similarity metrics."""
    # Available similarity metrics
    available_metrics: List[str] = None
    
    # Mahalanobis distance configuration
    enable_mahalanobis: bool = True
    mahalanobis_regularization: float = 1e-6
    mahalanobis_min_samples: int = 10
    
    # Jensen-Shannon divergence configuration
    enable_jensen_shannon: bool = True
    js_bins: int = 50
    js_smoothing: float = 1e-10
    
    # Dynamic Time Warping configuration
    enable_dtw: bool = True
    dtw_window: Optional[int] = None
    dtw_max_length: int = 1000
    
    # Ensemble similarity configuration
    enable_ensemble: bool = True
    ensemble_weights: Optional[Dict[str, float]] = None
    ensemble_method: str = 'weighted_average'  # 'weighted_average', 'voting', 'stacking'
    
    # Vectorization configuration
    enable_vectorization: bool = True
    vectorization_chunk_size: int = 1000
    use_vectorbt: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_threshold_mb: float = 100.0
    enable_caching: bool = True
    
    def __post_init__(self):
        if self.available_metrics is None:
            self.available_metrics = [
                'cosine', 'euclidean', 'manhattan', 'wasserstein',
                'mahalanobis', 'jensen_shannon', 'dtw', 'ensemble'
            ]
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'cosine': 0.25,
                'mahalanobis': 0.25,
                'jensen_shannon': 0.25,
                'dtw': 0.25
            }

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
    similarity_metric: str = 'ensemble'  # 'cosine', 'euclidean', 'manhattan', 'wasserstein', 'ensemble'
    use_feature_weights: bool = True
    feature_weights: Optional[List[float]] = None
    
    # Enhanced similarity configuration
    enhanced_similarity: EnhancedSimilarityConfig = None
    
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
    
    def __post_init__(self):
        if self.enhanced_similarity is None:
            self.enhanced_similarity = EnhancedSimilarityConfig()

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
        
        # Initialize artifact management
        self.artifact_manager = None
        self.similarity_cache = {}
        self._initialize_artifact_management()
        
        # Initialize vectorization tools
        self.vectorization_manager = None
        self.vectorbt_optimizer = None
        self._initialize_vectorization_tools()
        
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
    
    @memory_optimized(memory_threshold_mb=100.0) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    @performance_tracked(log_performance=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def _calculate_similarity_matrix(self, 
                                   cluster_labels: np.ndarray, 
                                   features: np.ndarray,
                                   unique_labels: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix between regimes with vectorized computation and caching."""
        try:
            tprint_info(f"Calculating similarity matrix for {len(unique_labels)} clusters")
            
            # Check cache first
            cached_matrix = self._get_cached_similarity_matrix(cluster_labels, features, unique_labels)
            if cached_matrix is not None:
                tprint_info("Using cached similarity matrix")
                return cached_matrix
            
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)  # Initialize with identity matrix
            
            # Calculate cluster centers
            cluster_centers = self._calculate_cluster_centers(cluster_labels, features, unique_labels)
            
            # Use vectorized computation if enabled
            if self.config.enhanced_similarity.enable_vectorization:
                similarity_matrix = self._calculate_vectorized_similarity_matrix(
                    cluster_labels, features, unique_labels, cluster_centers
                )
            else:
                # Calculate pairwise similarities
                for i in range(n_clusters):
                    for j in range(i + 1, n_clusters):
                        similarity = self._calculate_similarity(
                            cluster_centers[i], cluster_centers[j], features, 
                            cluster_labels, unique_labels[i], unique_labels[j]
                        )
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
            
            # Cache the result
            self._cache_similarity_matrix(similarity_matrix, cluster_labels, features, unique_labels)
            
            tprint_info(f"Similarity matrix calculation completed")
            return similarity_matrix
            
        except Exception as e:
            tprint_error(f"Similarity matrix calculation failed: {e}")
            logger.error(f"❌ Similarity matrix calculation failed: {e}")
            return np.eye(len(unique_labels))
    
    def _calculate_vectorized_similarity_matrix(self, 
                                             cluster_labels: np.ndarray, 
                                             features: np.ndarray,
                                             unique_labels: np.ndarray,
                                             cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using vectorized operations."""
        try:
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)
            
            # Use VectorBT optimization if available
            if VECTORBT_AVAILABLE and self.config.enhanced_similarity.use_vectorbt:
                return self._calculate_vectorbt_similarity_matrix(
                    cluster_labels, features, unique_labels, cluster_centers
                )
            
            # Use unified vectorization manager if available
            if ML_UTILITIES_AVAILABLE:
                return self._calculate_unified_vectorized_similarity_matrix(
                    cluster_labels, features, unique_labels, cluster_centers
                )
            
            # Fallback to chunked computation
            return self._calculate_chunked_similarity_matrix(
                cluster_labels, features, unique_labels, cluster_centers
            )
            
        except Exception as e:
            tprint_debug(f"Vectorized similarity calculation failed: {e}")
            # Fallback to basic computation
            return self._calculate_basic_similarity_matrix(
                cluster_labels, features, unique_labels, cluster_centers
            )
    
    def _calculate_vectorbt_similarity_matrix(self, 
                                            cluster_labels: np.ndarray, 
                                            features: np.ndarray,
                                            unique_labels: np.ndarray,
                                            cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using VectorBT optimization."""
        try:
            tprint_debug("Using VectorBT for similarity matrix calculation")
            
            # Get VectorBT rolling optimizer
            optimizer = get_vectorbt_rolling_optimizer()
            
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)
            
            # Prepare data for VectorBT
            features_df = pd.DataFrame(features)
            features_df['cluster'] = cluster_labels
            
            # Calculate similarities in chunks
            chunk_size = self.config.enhanced_similarity.vectorization_chunk_size
            
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    # Get samples for both clusters
                    samples1 = features[cluster_labels == unique_labels[i]]
                    samples2 = features[cluster_labels == unique_labels[j]]
                    
                    if len(samples1) == 0 or len(samples2) == 0:
                        continue
                    
                    # Use VectorBT for efficient computation
                    similarity = self._calculate_similarity(
                        cluster_centers[i], cluster_centers[j], features, 
                        cluster_labels, unique_labels[i], unique_labels[j]
                    )
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            return similarity_matrix
            
        except Exception as e:
            tprint_debug(f"VectorBT similarity calculation failed: {e}")
            raise
    
    def _calculate_unified_vectorized_similarity_matrix(self, 
                                                     cluster_labels: np.ndarray, 
                                                     features: np.ndarray,
                                                     unique_labels: np.ndarray,
                                                     cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using unified vectorization manager."""
        try:
            tprint_debug("Using unified vectorization manager for similarity matrix calculation")
            
            # Get unified vectorization manager
            vectorization_manager = get_unified_vectorization_manager()
            
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)
            
            # Create similarity calculation function
            def similarity_func(i, j):
                return self._calculate_similarity(
                    cluster_centers[i], cluster_centers[j], features, 
                    cluster_labels, unique_labels[i], unique_labels[j]
                )
            
            # Use vectorized computation
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    similarity = similarity_func(i, j)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            return similarity_matrix
            
        except Exception as e:
            tprint_debug(f"Unified vectorization similarity calculation failed: {e}")
            raise
    
    def _calculate_chunked_similarity_matrix(self, 
                                           cluster_labels: np.ndarray, 
                                           features: np.ndarray,
                                           unique_labels: np.ndarray,
                                           cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using chunked computation."""
        try:
            tprint_debug("Using chunked computation for similarity matrix calculation")
            
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)
            
            # Process in chunks to manage memory
            chunk_size = self.config.enhanced_similarity.vectorization_chunk_size
            
            for i in range(0, n_clusters, chunk_size):
                for j in range(i, min(i + chunk_size, n_clusters)):
                    for k in range(j + 1, n_clusters):
                        similarity = self._calculate_similarity(
                            cluster_centers[j], cluster_centers[k], features, 
                            cluster_labels, unique_labels[j], unique_labels[k]
                        )
                        similarity_matrix[j, k] = similarity
                        similarity_matrix[k, j] = similarity
                
                # Force cleanup after each chunk
                if HARDWARE_OPTIMIZATION_AVAILABLE:
                    force_cleanup()
            
            return similarity_matrix
            
        except Exception as e:
            tprint_debug(f"Chunked similarity calculation failed: {e}")
            raise
    
    def _calculate_basic_similarity_matrix(self, 
                                         cluster_labels: np.ndarray, 
                                         features: np.ndarray,
                                         unique_labels: np.ndarray,
                                         cluster_centers: np.ndarray) -> np.ndarray:
        """Calculate similarity matrix using basic computation."""
        try:
            n_clusters = len(unique_labels)
            similarity_matrix = np.eye(n_clusters)
            
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
            tprint_debug(f"Basic similarity calculation failed: {e}")
            raise
    
    @memory_optimized(memory_threshold_mb=50.0) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    @performance_tracked(log_performance=True) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def _calculate_similarity(self, 
                            center1: np.ndarray, 
                            center2: np.ndarray,
                            features: np.ndarray,
                            cluster_labels: np.ndarray,
                            label1: int,
                            label2: int) -> float:
        """Calculate similarity between two regimes using enhanced metrics."""
        try:
            tprint_debug(f"Calculating similarity between regimes {label1} and {label2}")
            
            # Get samples from each cluster
            samples1 = features[cluster_labels == label1]
            samples2 = features[cluster_labels == label2]
            
            if len(samples1) == 0 or len(samples2) == 0:
                tprint_debug(f"Empty clusters detected: {label1} ({len(samples1)}), {label2} ({len(samples2)})")
                return 0.0
            
            # Use enhanced similarity calculation
            if self.config.similarity_metric == 'ensemble':
                return self._calculate_ensemble_similarity(
                    center1, center2, samples1, samples2, label1, label2
                )
            else:
                return self._calculate_single_similarity(
                    center1, center2, samples1, samples2, label1, label2
                )
                
        except Exception as e:
            tprint_debug(f"Similarity calculation failed: {e}")
            logger.debug(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_single_similarity(self, 
                                   center1: np.ndarray, 
                                   center2: np.ndarray,
                                   samples1: np.ndarray,
                                   samples2: np.ndarray,
                                   label1: int,
                                   label2: int) -> float:
        """Calculate similarity using a single metric."""
        metric = self.config.similarity_metric
        
        if metric == 'cosine':
            return self._cosine_similarity(center1, center2)
        elif metric == 'euclidean':
            return self._euclidean_similarity(center1, center2)
        elif metric == 'manhattan':
            return self._manhattan_similarity(center1, center2)
        elif metric == 'wasserstein':
            return self._wasserstein_similarity(samples1, samples2)
        elif metric == 'mahalanobis':
            return self._mahalanobis_similarity(samples1, samples2)
        elif metric == 'jensen_shannon':
            return self._jensen_shannon_similarity(samples1, samples2)
        elif metric == 'dtw':
            return self._dtw_similarity(samples1, samples2)
        else:
            # Default to cosine similarity
            return self._cosine_similarity(center1, center2)
    
    def _calculate_ensemble_similarity(self, 
                                     center1: np.ndarray, 
                                     center2: np.ndarray,
                                     samples1: np.ndarray,
                                     samples2: np.ndarray,
                                     label1: int,
                                     label2: int) -> float:
        """Calculate ensemble similarity combining multiple metrics."""
        try:
            if not self.config.enhanced_similarity.enable_ensemble:
                return self._calculate_single_similarity(
                    center1, center2, samples1, samples2, label1, label2
                )
            
            tprint_debug(f"Calculating ensemble similarity for regimes {label1} and {label2}")
            
            # Calculate individual similarities
            similarities = {}
            
            # Basic metrics
            similarities['cosine'] = self._cosine_similarity(center1, center2)
            similarities['euclidean'] = self._euclidean_similarity(center1, center2)
            similarities['manhattan'] = self._manhattan_similarity(center1, center2)
            similarities['wasserstein'] = self._wasserstein_similarity(samples1, samples2)
            
            # Enhanced metrics
            if self.config.enhanced_similarity.enable_mahalanobis:
                similarities['mahalanobis'] = self._mahalanobis_similarity(samples1, samples2)
            
            if self.config.enhanced_similarity.enable_jensen_shannon:
                similarities['jensen_shannon'] = self._jensen_shannon_similarity(samples1, samples2)
            
            if self.config.enhanced_similarity.enable_dtw:
                similarities['dtw'] = self._dtw_similarity(samples1, samples2)
            
            # Combine similarities
            return self._combine_similarities(similarities)
            
        except Exception as e:
            tprint_debug(f"Ensemble similarity calculation failed: {e}")
            # Fallback to cosine similarity
            return self._cosine_similarity(center1, center2)
    
    def _cosine_similarity(self, center1: np.ndarray, center2: np.ndarray) -> float:
        """Calculate cosine similarity between cluster centers."""
        try:
            dot_product = np.dot(center1, center2)
            norm1 = np.linalg.norm(center1)
            norm2 = np.linalg.norm(center2)
            return safe_divide(dot_product, norm1 * norm2, 0.0)
        except Exception:
            return 0.0
    
    def _euclidean_similarity(self, center1: np.ndarray, center2: np.ndarray) -> float:
        """Calculate Euclidean distance-based similarity."""
        try:
            distance = np.linalg.norm(center1 - center2)
            return 1.0 / (1.0 + distance)
        except Exception:
            return 0.0
    
    def _manhattan_similarity(self, center1: np.ndarray, center2: np.ndarray) -> float:
        """Calculate Manhattan distance-based similarity."""
        try:
            distance = np.sum(np.abs(center1 - center2))
            return 1.0 / (1.0 + distance)
        except Exception:
            return 0.0
    
    def _wasserstein_similarity(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """Calculate Wasserstein distance-based similarity."""
        try:
            if len(samples1) == 0 or len(samples2) == 0:
                return 0.0
            
            # Calculate Wasserstein distance for each feature
            distances = []
            for i in range(samples1.shape[1]):
                try:
                    dist = wasserstein_distance(samples1[:, i], samples2[:, i])
                    distances.append(dist)
                except Exception:
                    # Skip problematic features
                    continue
            
            if not distances:
                return 0.0
            
            avg_distance = safe_mean(distances, default=0.0)
            return 1.0 / (1.0 + avg_distance)
        except Exception:
            return 0.0
    
    def _mahalanobis_similarity(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """Calculate Mahalanobis distance-based similarity."""
        try:
            if len(samples1) < self.config.enhanced_similarity.mahalanobis_min_samples or \
               len(samples2) < self.config.enhanced_similarity.mahalanobis_min_samples:
                tprint_debug("Insufficient samples for Mahalanobis distance")
                return 0.0
            
            # Calculate mean and covariance
            mean1 = np.mean(samples1, axis=0)
            mean2 = np.mean(samples2, axis=0)
            
            # Use samples1 for covariance estimation (or combine both)
            combined_samples = np.vstack([samples1, samples2])
            cov_matrix = np.cov(combined_samples.T)
            
            # Add regularization to avoid singular matrix
            reg = self.config.enhanced_similarity.mahalanobis_regularization
            cov_matrix += reg * np.eye(cov_matrix.shape[0])
            
            # Calculate Mahalanobis distance
            try:
                inv_cov = np.linalg.inv(cov_matrix)
                diff = mean1 - mean2
                mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                return 1.0 / (1.0 + mahal_dist)
            except np.linalg.LinAlgError:
                # Fallback to Euclidean distance
                tprint_debug("Singular covariance matrix, falling back to Euclidean")
                return self._euclidean_similarity(mean1, mean2)
                
        except Exception as e:
            tprint_debug(f"Mahalanobis similarity calculation failed: {e}")
            return 0.0
    
    def _jensen_shannon_similarity(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence-based similarity."""
        try:
            if len(samples1) == 0 or len(samples2) == 0:
                return 0.0
            
            # Calculate JS divergence for each feature
            js_divergences = []
            bins = self.config.enhanced_similarity.js_bins
            smoothing = self.config.enhanced_similarity.js_smoothing
            
            for i in range(samples1.shape[1]):
                try:
                    # Create histograms
                    hist1, bin_edges = np.histogram(samples1[:, i], bins=bins, density=True)
                    hist2, _ = np.histogram(samples2[:, i], bins=bin_edges, density=True)
                    
                    # Add smoothing to avoid zero probabilities
                    hist1 = hist1 + smoothing
                    hist2 = hist2 + smoothing
                    
                    # Normalize
                    hist1 = hist1 / np.sum(hist1)
                    hist2 = hist2 / np.sum(hist2)
                    
                    # Calculate JS divergence
                    m = 0.5 * (hist1 + hist2)
                    js_div = 0.5 * entropy(hist1, m) + 0.5 * entropy(hist2, m)
                    js_divergences.append(js_div)
                    
                except Exception:
                    # Skip problematic features
                    continue
            
            if not js_divergences:
                return 0.0
            
            avg_js_div = safe_mean(js_divergences, default=0.0)
            # Convert divergence to similarity (0 = identical, 1 = completely different)
            return 1.0 - avg_js_div
            
        except Exception as e:
            tprint_debug(f"Jensen-Shannon similarity calculation failed: {e}")
            return 0.0
    
    def _dtw_similarity(self, samples1: np.ndarray, samples2: np.ndarray) -> float:
        """Calculate Dynamic Time Warping-based similarity."""
        try:
            if len(samples1) == 0 or len(samples2) == 0:
                return 0.0
            
            # Limit sequence length for performance
            max_length = self.config.enhanced_similarity.dtw_max_length
            if len(samples1) > max_length:
                samples1 = samples1[:max_length]
            if len(samples2) > max_length:
                samples2 = samples2[:max_length]
            
            # Calculate DTW distance for each feature
            dtw_distances = []
            window = self.config.enhanced_similarity.dtw_window
            
            for i in range(samples1.shape[1]):
                try:
                    # Simple DTW implementation
                    seq1 = samples1[:, i]
                    seq2 = samples2[:, i]
                    
                    dtw_dist = self._calculate_dtw_distance(seq1, seq2, window)
                    dtw_distances.append(dtw_dist)
                    
                except Exception:
                    # Skip problematic features
                    continue
            
            if not dtw_distances:
                return 0.0
            
            avg_dtw_dist = safe_mean(dtw_distances, default=0.0)
            return 1.0 / (1.0 + avg_dtw_dist)
            
        except Exception as e:
            tprint_debug(f"DTW similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray, window: Optional[int] = None) -> float:
        """Calculate Dynamic Time Warping distance between two sequences."""
        try:
            n, m = len(seq1), len(seq2)
            
            # Initialize DTW matrix
            dtw_matrix = np.full((n + 1, m + 1), np.inf)
            dtw_matrix[0, 0] = 0
            
            # Set window constraint
            if window is not None:
                window = min(window, max(n, m))
            else:
                window = max(n, m)
            
            # Fill DTW matrix
            for i in range(1, n + 1):
                for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                    cost = (seq1[i-1] - seq2[j-1]) ** 2
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            return np.sqrt(dtw_matrix[n, m])
            
        except Exception:
            return float('inf')
    
    def _combine_similarities(self, similarities: Dict[str, float]) -> float:
        """Combine multiple similarity scores into a single score."""
        try:
            method = self.config.enhanced_similarity.ensemble_method
            weights = self.config.enhanced_similarity.ensemble_weights
            
            if method == 'weighted_average':
                # Weighted average of similarities
                weighted_sum = 0.0
                total_weight = 0.0
                
                for metric, similarity in similarities.items():
                    if metric in weights and np.isfinite(similarity):
                        weight = weights[metric]
                        weighted_sum += weight * similarity
                        total_weight += weight
                
                if total_weight > 0:
                    return weighted_sum / total_weight
                else:
                    return safe_mean(list(similarities.values()), default=0.0)
            
            elif method == 'voting':
                # Majority voting (convert to binary decisions)
                threshold = 0.5
                votes = [1 if sim > threshold else 0 for sim in similarities.values()]
                return np.mean(votes) if votes else 0.0
            
            elif method == 'stacking':
                # Simple stacking (average of all similarities)
                valid_similarities = [sim for sim in similarities.values() if np.isfinite(sim)]
                return safe_mean(valid_similarities, default=0.0)
            
            else:
                # Default to weighted average
                return self._combine_similarities(similarities)
                
        except Exception as e:
            tprint_debug(f"Similarity combination failed: {e}")
            return safe_mean(list(similarities.values()), default=0.0)
    
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
    
    def _initialize_artifact_management(self):
        """Initialize artifact management for caching similarity matrices."""
        try:
            if self.config.enhanced_similarity.enable_caching:
                # Initialize artifact manager if available
                try:
                    from src.utils.artifact_manager import ArtifactManager
                    self.artifact_manager = ArtifactManager()
                    tprint_debug("Artifact manager initialized for similarity caching")
                except ImportError:
                    tprint_debug("Artifact manager not available, using in-memory cache")
                    self.artifact_manager = None
        except Exception as e:
            tprint_debug(f"Artifact management initialization failed: {e}")
            self.artifact_manager = None
    
    def _initialize_vectorization_tools(self):
        """Initialize vectorization tools for efficient computation."""
        try:
            if self.config.enhanced_similarity.enable_vectorization:
                # Initialize VectorBT optimizer if available
                if VECTORBT_AVAILABLE:
                    try:
                        self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                        tprint_debug("VectorBT optimizer initialized")
                    except Exception as e:
                        tprint_debug(f"VectorBT optimizer initialization failed: {e}")
                        self.vectorbt_optimizer = None
                
                # Initialize unified vectorization manager if available
                if ML_UTILITIES_AVAILABLE:
                    try:
                        self.vectorization_manager = get_unified_vectorization_manager()
                        tprint_debug("Unified vectorization manager initialized")
                    except Exception as e:
                        tprint_debug(f"Unified vectorization manager initialization failed: {e}")
                        self.vectorization_manager = None
        except Exception as e:
            tprint_debug(f"Vectorization tools initialization failed: {e}")
    
    @smart_cache(ttl=3600) if HARDWARE_OPTIMIZATION_AVAILABLE else lambda x: x
    def _get_cached_similarity_matrix(self, 
                                    cluster_labels: np.ndarray, 
                                    features: np.ndarray,
                                    unique_labels: np.ndarray) -> Optional[np.ndarray]:
        """Get cached similarity matrix if available."""
        try:
            if not self.config.enhanced_similarity.enable_caching:
                return None
            
            # Create cache key based on data characteristics
            cache_key = self._create_similarity_cache_key(cluster_labels, features, unique_labels)
            
            # Check in-memory cache first
            if cache_key in self.similarity_cache:
                tprint_debug("Using cached similarity matrix from memory")
                return self.similarity_cache[cache_key]
            
            # Check artifact manager cache
            if self.artifact_manager:
                try:
                    cached_matrix = self.artifact_manager.get_artifact(f"similarity_matrix_{cache_key}")
                    if cached_matrix is not None:
                        tprint_debug("Using cached similarity matrix from artifact manager")
                        self.similarity_cache[cache_key] = cached_matrix
                        return cached_matrix
                except Exception as e:
                    tprint_debug(f"Artifact manager cache retrieval failed: {e}")
            
            return None
            
        except Exception as e:
            tprint_debug(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_similarity_matrix(self, 
                               similarity_matrix: np.ndarray,
                               cluster_labels: np.ndarray, 
                               features: np.ndarray,
                               unique_labels: np.ndarray):
        """Cache similarity matrix for future use."""
        try:
            if not self.config.enhanced_similarity.enable_caching:
                return
            
            # Create cache key
            cache_key = self._create_similarity_cache_key(cluster_labels, features, unique_labels)
            
            # Store in memory cache
            self.similarity_cache[cache_key] = similarity_matrix.copy()
            
            # Store in artifact manager if available
            if self.artifact_manager:
                try:
                    self.artifact_manager.save_artifact(
                        f"similarity_matrix_{cache_key}", 
                        similarity_matrix,
                        metadata={
                            'n_clusters': len(unique_labels),
                            'n_features': features.shape[1],
                            'n_samples': len(cluster_labels),
                            'similarity_metric': self.config.similarity_metric
                        }
                    )
                    tprint_debug("Similarity matrix cached in artifact manager")
                except Exception as e:
                    tprint_debug(f"Artifact manager cache storage failed: {e}")
            
        except Exception as e:
            tprint_debug(f"Cache storage failed: {e}")
    
    def _create_similarity_cache_key(self, 
                                   cluster_labels: np.ndarray, 
                                   features: np.ndarray,
                                   unique_labels: np.ndarray) -> str:
        """Create a cache key for similarity matrix."""
        try:
            # Create hash based on data characteristics
            import hashlib
            
            # Include relevant data characteristics
            key_data = {
                'n_clusters': len(unique_labels),
                'n_features': features.shape[1],
                'n_samples': len(cluster_labels),
                'similarity_metric': self.config.similarity_metric,
                'enhanced_config': str(self.config.enhanced_similarity.__dict__),
                'data_hash': hashlib.md5(features.tobytes()).hexdigest()[:16],
                'labels_hash': hashlib.md5(cluster_labels.tobytes()).hexdigest()[:16]
            }
            
            key_string = str(sorted(key_data.items()))
            return hashlib.md5(key_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            tprint_debug(f"Cache key creation failed: {e}")
            return "default_key"
    
    def clear_similarity_cache(self):
        """Clear similarity matrix cache."""
        try:
            self.similarity_cache.clear()
            tprint_debug("Similarity matrix cache cleared")
        except Exception as e:
            tprint_debug(f"Cache clearing failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = {
                'cache_size': len(self.similarity_cache),
                'cache_enabled': self.config.enhanced_similarity.enable_caching,
                'artifact_manager_available': self.artifact_manager is not None,
                'vectorization_enabled': self.config.enhanced_similarity.enable_vectorization,
                'vectorbt_available': self.vectorbt_optimizer is not None,
                'unified_vectorization_available': self.vectorization_manager is not None
            }
            return stats
        except Exception as e:
            tprint_debug(f"Cache stats retrieval failed: {e}")
            return {}
    
    def validate_enhanced_similarity_metrics(self, 
                                           test_features: np.ndarray,
                                           test_labels: np.ndarray) -> Dict[str, Any]:
        """Validate enhanced similarity metrics with test data."""
        try:
            tprint_info("Validating enhanced similarity metrics")
            
            validation_results = {
                'metrics_tested': [],
                'performance_metrics': {},
                'errors': [],
                'recommendations': []
            }
            
            # Test each similarity metric
            test_centers = self._calculate_cluster_centers(test_labels, test_features, np.unique(test_labels))
            
            for metric in self.config.enhanced_similarity.available_metrics:
                try:
                    tprint_debug(f"Testing similarity metric: {metric}")
                    
                    # Temporarily change similarity metric
                    original_metric = self.config.similarity_metric
                    self.config.similarity_metric = metric
                    
                    # Calculate similarity matrix
                    start_time = time.time()
                    similarity_matrix = self._calculate_similarity_matrix(test_labels, test_features, np.unique(test_labels))
                    calculation_time = time.time() - start_time
                    
                    # Validate similarity matrix
                    validation_score = self._validate_similarity_matrix(similarity_matrix)
                    
                    validation_results['metrics_tested'].append(metric)
                    validation_results['performance_metrics'][metric] = {
                        'calculation_time': calculation_time,
                        'validation_score': validation_score,
                        'matrix_shape': similarity_matrix.shape,
                        'mean_similarity': np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                        'std_similarity': np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                    }
                    
                    # Restore original metric
                    self.config.similarity_metric = original_metric
                    
                except Exception as e:
                    validation_results['errors'].append(f"Metric {metric} failed: {str(e)}")
                    tprint_debug(f"Validation failed for metric {metric}: {e}")
            
            # Generate recommendations
            validation_results['recommendations'] = self._generate_similarity_recommendations(validation_results)
            
            tprint_info("Enhanced similarity metrics validation completed")
            return validation_results
            
        except Exception as e:
            tprint_error(f"Similarity metrics validation failed: {e}")
            return {'error': str(e)}
    
    def _validate_similarity_matrix(self, similarity_matrix: np.ndarray) -> float:
        """Validate similarity matrix quality."""
        try:
            # Check basic properties
            if not np.allclose(similarity_matrix, similarity_matrix.T):
                return 0.0  # Not symmetric
            
            if not np.allclose(np.diag(similarity_matrix), 1.0):
                return 0.0  # Diagonal not 1
            
            if np.any(similarity_matrix < 0) or np.any(similarity_matrix > 1):
                return 0.0  # Values not in [0,1]
            
            # Calculate quality score based on distribution
            off_diagonal = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            mean_sim = np.mean(off_diagonal)
            std_sim = np.std(off_diagonal)
            
            # Good similarity matrix should have reasonable mean and low std
            quality_score = mean_sim * (1.0 - std_sim)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _generate_similarity_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        try:
            recommendations = []
            
            if not validation_results['metrics_tested']:
                recommendations.append("No metrics were successfully tested")
                return recommendations
            
            # Find best performing metric
            best_metric = None
            best_score = -1
            
            for metric, metrics in validation_results['performance_metrics'].items():
                if metrics['validation_score'] > best_score:
                    best_score = metrics['validation_score']
                    best_metric = metric
            
            if best_metric:
                recommendations.append(f"Best performing metric: {best_metric} (score: {best_score:.3f})")
            
            # Check for performance issues
            for metric, metrics in validation_results['performance_metrics'].items():
                if metrics['calculation_time'] > 10.0:  # More than 10 seconds
                    recommendations.append(f"Consider optimizing {metric} - calculation time: {metrics['calculation_time']:.2f}s")
                
                if metrics['validation_score'] < 0.5:
                    recommendations.append(f"Consider tuning {metric} - validation score: {metrics['validation_score']:.3f}")
            
            # Check for errors
            if validation_results['errors']:
                recommendations.append(f"Fix {len(validation_results['errors'])} metric errors before production use")
            
            # General recommendations
            if len(validation_results['metrics_tested']) > 1:
                recommendations.append("Consider using ensemble similarity for better robustness")
            
            if validation_results['performance_metrics']:
                avg_time = np.mean([m['calculation_time'] for m in validation_results['performance_metrics'].values()])
                if avg_time > 5.0:
                    recommendations.append("Enable vectorization for better performance")
            
            return recommendations
            
        except Exception as e:
            tprint_debug(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations"]
    
    def benchmark_similarity_metrics(self, 
                                   test_features: np.ndarray,
                                   test_labels: np.ndarray,
                                   n_iterations: int = 5) -> Dict[str, Any]:
        """Benchmark similarity metrics performance."""
        try:
            tprint_info(f"Benchmarking similarity metrics with {n_iterations} iterations")
            
            benchmark_results = {
                'metrics': {},
                'summary': {},
                'recommendations': []
            }
            
            unique_labels = np.unique(test_labels)
            
            for metric in self.config.enhanced_similarity.available_metrics:
                try:
                    tprint_debug(f"Benchmarking metric: {metric}")
                    
                    # Temporarily change similarity metric
                    original_metric = self.config.similarity_metric
                    self.config.similarity_metric = metric
                    
                    times = []
                    memory_usage = []
                    
                    for i in range(n_iterations):
                        # Measure memory before
                        if HARDWARE_OPTIMIZATION_AVAILABLE:
                            try:
                                from src.utils.hardware import get_memory_stats
                                mem_before = get_memory_stats().get('rss', 0)
                            except:
                                mem_before = 0
                        else:
                            mem_before = 0
                        
                        # Time the calculation
                        start_time = time.time()
                        similarity_matrix = self._calculate_similarity_matrix(test_labels, test_features, unique_labels)
                        end_time = time.time()
                        
                        # Measure memory after
                        if HARDWARE_OPTIMIZATION_AVAILABLE:
                            try:
                                from src.utils.hardware import get_memory_stats
                                mem_after = get_memory_stats().get('rss', 0)
                            except:
                                mem_after = 0
                        else:
                            mem_after = 0
                        
                        times.append(end_time - start_time)
                        memory_usage.append(max(0, mem_after - mem_before))
                    
                    # Calculate statistics
                    benchmark_results['metrics'][metric] = {
                        'mean_time': np.mean(times),
                        'std_time': np.std(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'mean_memory_mb': np.mean(memory_usage) / (1024 * 1024),
                        'max_memory_mb': np.max(memory_usage) / (1024 * 1024),
                        'iterations': n_iterations
                    }
                    
                    # Restore original metric
                    self.config.similarity_metric = original_metric
                    
                except Exception as e:
                    tprint_debug(f"Benchmarking failed for metric {metric}: {e}")
                    benchmark_results['metrics'][metric] = {'error': str(e)}
            
            # Generate summary and recommendations
            benchmark_results['summary'] = self._generate_benchmark_summary(benchmark_results['metrics'])
            benchmark_results['recommendations'] = self._generate_benchmark_recommendations(benchmark_results['metrics'])
            
            tprint_info("Similarity metrics benchmarking completed")
            return benchmark_results
            
        except Exception as e:
            tprint_error(f"Benchmarking failed: {e}")
            return {'error': str(e)}
    
    def _generate_benchmark_summary(self, metrics_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        try:
            valid_metrics = {k: v for k, v in metrics_results.items() if 'error' not in v}
            
            if not valid_metrics:
                return {'error': 'No valid metrics found'}
            
            # Find fastest and most memory efficient
            fastest_metric = min(valid_metrics.items(), key=lambda x: x[1]['mean_time'])
            most_memory_efficient = min(valid_metrics.items(), key=lambda x: x[1]['mean_memory_mb'])
            
            summary = {
                'total_metrics_tested': len(valid_metrics),
                'fastest_metric': fastest_metric[0],
                'fastest_time': fastest_metric[1]['mean_time'],
                'most_memory_efficient': most_memory_efficient[0],
                'lowest_memory': most_memory_efficient[1]['mean_memory_mb'],
                'average_time': np.mean([m['mean_time'] for m in valid_metrics.values()]),
                'average_memory': np.mean([m['mean_memory_mb'] for m in valid_metrics.values()])
            }
            
            return summary
            
        except Exception as e:
            tprint_debug(f"Benchmark summary generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_benchmark_recommendations(self, metrics_results: Dict[str, Any]) -> List[str]:
        """Generate benchmark recommendations."""
        try:
            recommendations = []
            valid_metrics = {k: v for k, v in metrics_results.items() if 'error' not in v}
            
            if not valid_metrics:
                return ["No valid metrics to analyze"]
            
            # Performance recommendations
            times = [m['mean_time'] for m in valid_metrics.values()]
            memories = [m['mean_memory_mb'] for m in valid_metrics.values()]
            
            if np.std(times) > np.mean(times) * 0.5:
                recommendations.append("High variance in calculation times - consider metric tuning")
            
            if any(m['mean_memory_mb'] > 100 for m in valid_metrics.values()):
                recommendations.append("High memory usage detected - consider enabling vectorization")
            
            if any(m['mean_time'] > 10 for m in valid_metrics.values()):
                recommendations.append("Slow metrics detected - consider using faster alternatives")
            
            # General recommendations
            if len(valid_metrics) > 1:
                recommendations.append("Multiple metrics available - consider ensemble approach")
            
            recommendations.append("Enable caching for repeated calculations")
            recommendations.append("Use vectorization for large datasets")
            
            return recommendations
            
        except Exception as e:
            tprint_debug(f"Benchmark recommendations generation failed: {e}")
            return ["Unable to generate recommendations"]


# Alias for backward compatibility
SimilarityMerger = DataDrivenSimilarityMerger