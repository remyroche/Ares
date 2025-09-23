"""
Optimal Regime Clustering Configuration

This configuration file defines parameters for creating optimal clusters from HMM regime discovery output.

GOALS:
- Target: 90-95% coverage (cluster data) with <5% noise
- Alternative: 0% noise (all data clustered) if regime patterns are subtle
- Focus: Capture market regime patterns with minimal noise classification

CURRENT STATUS:
- Using maximum permissive parameters for low noise regime clustering
- Prioritizing noise reduction over strict quality metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class OptimalClusteringConfig:
    """Configuration for optimal regime clustering."""

    # Target cluster parameters - More inclusive clustering
    target_n_clusters: int = 20
    force_n_clusters: bool = True  # Force exactly 20 clusters to be created
    target_coverage_pct: float = 0.85  # 85% coverage (more inclusive)
    max_noise_pct: float = 0.15  # <15% noise (more inclusive)
    min_cluster_size_pct: float = 0.02  # 2% minimum (more inclusive)
    max_cluster_size_pct: float = 0.25  # 25% maximum (much more inclusive)

    # Clustering algorithm parameters - CV-OPTIMIZED PERFECT DISTRIBUTION APPROACH
    clustering_method: str = "centroid_based"  # "hdbscan", "dbscan", "kmeans", "hybrid", "centroid_based"
    enable_aggressive_splitting: bool = False  # Disable aggressive cluster splitting (preserve more clusters)
    enable_weighted_splitting: bool = False  # Disable weighted cluster splitting (preserve more clusters)
    enable_dynamic_merging: bool = False  # Disable dynamic size-constrained merging (preserve more clusters)
    force_exact_constraints: bool = False  # Don't force exact constraints (preserve more clusters)
    max_cluster_splitting_iterations: int = 20  # Maximum iterations for cluster splitting
    splitting_aggressiveness: float = 1.0  # How aggressively to split oversized clusters (less aggressive)
    min_samples: int = 1  # Minimum samples for HDBSCAN/DBSCAN (more inclusive)
    min_cluster_size: int = 3  # Minimum cluster size for HDBSCAN (more inclusive)
    cluster_selection_epsilon: float = 0.025  # DBSCAN epsilon parameter (more inclusive)
    max_iter: int = 500  # Maximum iterations for iterative algorithms
    random_state: int = 42  # Random state for reproducibility

    # Stable KMeans parameters
    kmeans_n_init: int = 50  # Increase restarts for stability (50–100)
    kmeans_max_iter: int = 1000  # Higher max_iter for convergence
    kmeans_num_seeds: int = 10  # Multiple seeds; pick best by size-penalized objective
    size_penalty_weight: float = 0.25  # Weight for size imbalance penalty in selection
    stable_selection_objective: str = "inertia_size_cv"  # Selection objective type

    # Overcluster-then-merge parameters
    overcluster_enabled: bool = True
    overcluster_k_min: int = 30
    overcluster_k_max: int = 40
    overcluster_merge_linkage: str = "centroid"  # centroid or average

    # Constrained KMeans parameters
    constrained_kmeans_enabled: bool = True
    constraint_backend: str = "kmeans_constrained"  # "kmeans_constrained" | "assignment"

    # Split/Merge post-processing with percentile-based gating
    split_merge_enabled: bool = True
    split_merge_max_iters: int = 10
    merge_similarity_metric: str = "cosine"  # cosine or euclidean
    easy_merge_top_percentile: float = 0.75  # Pairs >= P75 similarity merge with relaxed criteria
    strict_split_bottom_percentile: float = 0.25  # Sub-splits <= P25 similarity favored
    use_internal_cv_for_split: bool = True  # Use per-cluster CV to prioritize splits

    # Optional robust clustering candidate
    use_kmedoids_candidate: bool = False  # Try K-Medoids as a candidate during stability selection

    # Advanced clustering parameters - CV-OPTIMIZED FOR PERFECT DISTRIBUTION
    weighted_4d_mapping: bool = True  # Use weighted 4D feature mapping
    equidistant_centroids: bool = True  # Find equidistant centroids with quality scoring
    size_constrained_merging: bool = True  # Enable size-constrained merging
    cv_based_similarity: bool = True  # Use CV for similarity calculations
    cv_optimized_splitting: bool = True  # Enable CV-optimized cluster splitting
    enhanced_redistribution: bool = True  # Enable enhanced redistribution with multiple rounds
    iterative_refinement: bool = True  # Enable iterative refinement passes
    adaptive_targets: bool = True  # Enable adaptive target adjustment
    smart_cluster_transfer: bool = True  # Enable smart transfer between adjacent clusters

    # Pareto merge selection and adjacency gating
    enable_pareto_merging: bool = True  # Use Pareto-based candidate selection when merging to target K
    knn_adjacency_k: int = 3  # k-NN adjacency (on 4D/weighted centroids) for candidate merge gating
    epsilon_cv_increase: float = 0.05  # Block merges that increase pooled CV beyond this fraction
    enable_pareto_feature_weighting: bool = False  # Apply Pareto-aware feature weighting in weighted 4D map

    # Quality metrics thresholds - ULTRA-LENIENT FOR PERFECT DISTRIBUTION
    min_silhouette_score: float = 0.01  # Ultra-low minimum for regime clustering
    min_calinski_harabasz_score: float = 10.0  # Ultra-low minimum for regime clustering
    min_davies_bouldin_score: float = 5.0  # Ultra-high maximum (ultra-lenient)
    target_coherence_score: float = 0.2  # Very low target for maximum flexibility

    # CV-Optimized distribution parameters
    perfect_distribution_threshold: float = 0.98  # Require 98% of clusters in 3-8% range
    outlier_redistribution_rounds: int = 8  # Number of redistribution rounds
    smart_transfer_percentage: float = 0.25  # Percentage of points to transfer in smart transfer
    refinement_passes: int = 6  # Number of iterative refinement passes
    cv_split_optimization: bool = True  # Enable CV-based split optimization
    min_cluster_split_size: int = 20  # Minimum size for CV-optimized splitting

    # Feature dimensions (4D: volume, volatility, momentum, trend)
    feature_dimensions: List[str] = field(default_factory=lambda: [
        'volume', 'volatility', 'momentum', 'trend'
    ])

    stability_threshold: float = 0.8  # Minimum stability score

    # Output parameters
    save_intermediate_results: bool = False  # Disabled for faster execution
    generate_cluster_reports: bool = True
    save_cluster_visualizations: bool = False  # Disabled for faster execution

    # Memory optimization
    chunk_size: int = 100000  # Process data in chunks (increased for better efficiency)
    use_memory_optimization: bool = True

    # Advanced parameters
    adaptive_clustering: bool = True  # Adapt clustering based on data characteristics
    multi_stage_clustering: bool = True  # Use multi-stage approach
    outlier_detection_method: str = "none"  # "isolation_forest", "local_outlier_factor", "none"

    # HMM integration parameters
    hmm_min_states: int = 3  # Minimum HMM states to consider
    hmm_max_states: int = 8  # Maximum HMM states to consider
    hmm_state_prob_threshold: float = 0.6  # Minimum state probability threshold

    # ===== ENHANCED CLUSTERING PARAMETERS =====

    # Enhanced 4D frontier optimization
    enable_enhanced_clustering: bool = True  # Enable enhanced clustering with 4D frontiers
    enable_frontier_optimization: bool = True  # Enable 4D frontier establishment
    enable_regime_transfer_optimization: bool = True  # Enable CV-based regime transfers
    frontier_optimization_iterations: int = 5  # Number of frontier optimization iterations

    # Enhanced quality thresholds
    enhanced_min_silhouette_score: float = 0.35  # Enhanced Silhouette threshold
    enhanced_min_calinski_harabasz_score: float = 200.0  # Enhanced CH threshold
    enhanced_min_davies_bouldin_score: float = 1.3  # Enhanced DB threshold (lower = better)

    # Enhanced CV optimization
    enhanced_cv_optimization_enabled: bool = True  # Enable enhanced CV optimization
    enhanced_cv_outlier_mitigation: bool = True  # Use MAD for outlier mitigation
    enhanced_cv_similarity_threshold: float = 0.1  # Minimum CV similarity benefit for transfers

    # Enhanced cluster size targeting (5% average)
    enhanced_target_avg_cluster_pct: float = 0.05  # 5% average cluster size
    enhanced_size_constraint_ratio: float = 1.5  # 50% size difference limit for transfers

    # Enhanced frontier types
    enhanced_frontier_types: List[str] = field(default_factory=lambda: [
        'volume_volatility', 'momentum_trend', 'volume_momentum',
        'volatility_trend', 'cross_dimensional'
    ])

    # Enhanced transfer optimization
    enhanced_transfer_batch_size: float = 0.1  # 10% batch processing
    enhanced_transfer_benefit_threshold: float = 0.1  # Minimum benefit for transfers
    enhanced_convergence_threshold: int = 0  # Stop if no transfers in iteration

    # Enhanced matrix operations
    enhanced_matrix_operations_enabled: bool = True  # Use enhanced matrix operations
    enhanced_memory_optimization: bool = True  # Enable memory optimization for large datasets
    enhanced_gpu_acceleration: bool = False  # Enable GPU acceleration if available

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base_dict = {
            'target_n_clusters': self.target_n_clusters,
            'target_coverage_pct': self.target_coverage_pct,
            'max_noise_pct': self.max_noise_pct,
            'min_cluster_size_pct': self.min_cluster_size_pct,
            'max_cluster_size_pct': self.max_cluster_size_pct,
            'clustering_method': self.clustering_method,
            'feature_dimensions': self.feature_dimensions,
            'enable_pareto_merging': self.enable_pareto_merging,
            'knn_adjacency_k': self.knn_adjacency_k,
            'epsilon_cv_increase': self.epsilon_cv_increase,
            'quality_metrics': {
                'min_silhouette_score': self.min_silhouette_score,
                'min_calinski_harabasz_score': self.min_calinski_harabasz_score,
                'min_davies_bouldin_score': self.min_davies_bouldin_score,
                'target_coherence_score': self.target_coherence_score
            },
            'validation': {
                'stability_threshold': self.stability_threshold
            }
        }

        # Add enhanced clustering parameters
        enhanced_dict = {
            'enable_enhanced_clustering': self.enable_enhanced_clustering,
            'enable_frontier_optimization': self.enable_frontier_optimization,
            'enable_regime_transfer_optimization': self.enable_regime_transfer_optimization,
            'frontier_optimization_iterations': self.frontier_optimization_iterations,
            'enhanced_quality_metrics': {
                'min_silhouette_score': self.enhanced_min_silhouette_score,
                'min_calinski_harabasz_score': self.enhanced_min_calinski_harabasz_score,
                'min_davies_bouldin_score': self.enhanced_min_davies_bouldin_score
            },
            'enhanced_cv_optimization': {
                'enabled': self.enhanced_cv_optimization_enabled,
                'outlier_mitigation': self.enhanced_cv_outlier_mitigation,
                'similarity_threshold': self.enhanced_cv_similarity_threshold
            },
            'enhanced_cluster_sizing': {
                'target_avg_cluster_pct': self.enhanced_target_avg_cluster_pct,
                'size_constraint_ratio': self.enhanced_size_constraint_ratio
            },
            'enhanced_frontier_config': {
                'frontier_types': self.enhanced_frontier_types,
                'transfer_batch_size': self.enhanced_transfer_batch_size,
                'transfer_benefit_threshold': self.enhanced_transfer_benefit_threshold,
                'convergence_threshold': self.enhanced_convergence_threshold
            },
            'enhanced_matrix_operations': {
                'enabled': self.enhanced_matrix_operations_enabled,
                'memory_optimization': self.enhanced_memory_optimization,
                'gpu_acceleration': self.enhanced_gpu_acceleration
            }
        }

        return {**base_dict, **enhanced_dict}

    @classmethod
    def create_default(cls) -> 'OptimalClusteringConfig':
        """Create default configuration."""
        return cls()

    @classmethod
    def create_high_quality(cls) -> 'OptimalClusteringConfig':
        """Create high-quality clustering configuration."""
        config = cls()
        config.min_silhouette_score = 0.4
        config.min_calinski_harabasz_score = 150.0
        config.min_davies_bouldin_score = 1.2
        config.target_coherence_score = 0.8
        config.min_samples = 100
        config.min_cluster_size = 200
        return config

    @classmethod
    def create_fast_processing(cls) -> 'OptimalClusteringConfig':
        """Create fast processing configuration."""
        config = cls()
        config.chunk_size = 100000
        config.use_memory_optimization = True
        config.max_iter = 100
        # REMOVED: bootstrap_iterations - removed for performance
        return config

    @classmethod
    def create_enhanced_clustering(cls) -> 'OptimalClusteringConfig':
        """Create enhanced clustering configuration with 4D frontier optimization."""
        config = cls()

        # Enable all enhanced features
        config.enable_enhanced_clustering = True
        config.enable_frontier_optimization = True
        config.enable_regime_transfer_optimization = True
        config.frontier_optimization_iterations = 5

        # Enhanced quality thresholds
        config.enhanced_min_silhouette_score = 0.35
        config.enhanced_min_calinski_harabasz_score = 200.0
        config.enhanced_min_davies_bouldin_score = 1.3

        # Enhanced CV optimization
        config.enhanced_cv_optimization_enabled = True
        config.enhanced_cv_outlier_mitigation = True
        config.enhanced_cv_similarity_threshold = 0.1

        # Enhanced cluster sizing (5% average)
        config.enhanced_target_avg_cluster_pct = 0.05
        config.enhanced_size_constraint_ratio = 1.5

        # Enhanced frontier configuration
        config.enhanced_frontier_types = [
            'volume_volatility', 'momentum_trend', 'volume_momentum',
            'volatility_trend', 'cross_dimensional'
        ]
        config.enhanced_transfer_batch_size = 0.1
        config.enhanced_transfer_benefit_threshold = 0.1
        config.enhanced_convergence_threshold = 0

        # Enhanced matrix operations
        config.enhanced_matrix_operations_enabled = True
        config.enhanced_memory_optimization = True
        config.enhanced_gpu_acceleration = False

        return config

def get_clustering_config(mode: str = "default") -> OptimalClusteringConfig:
    """Get clustering configuration based on mode."""
    if mode == "high_quality":
        return OptimalClusteringConfig.create_high_quality()
    elif mode == "fast_processing":
        return OptimalClusteringConfig.create_fast_processing()
    elif mode == "enhanced":
        return OptimalClusteringConfig.create_enhanced_clustering()
    else:
        return OptimalClusteringConfig.create_default()

# Default configuration instance
DEFAULT_CONFIG = get_clustering_config("default")
HIGH_QUALITY_CONFIG = get_clustering_config("high_quality")
FAST_CONFIG = get_clustering_config("fast_processing")
ENHANCED_CONFIG = get_clustering_config("enhanced")