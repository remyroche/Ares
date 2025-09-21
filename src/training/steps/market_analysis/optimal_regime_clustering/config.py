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

    # Target cluster parameters - Force creation of 20 clusters
    target_n_clusters: int = 20
    force_n_clusters: bool = True  # Force exactly 20 clusters to be created
    target_coverage_pct: float = 0.95  # 90-95% coverage
    max_noise_pct: float = 0.05  # <5% noise
    min_cluster_size_pct: float = 0.03  # 3% minimum (strict constraint)
    max_cluster_size_pct: float = 0.08  # 8% maximum (strict constraint)

    # Clustering algorithm parameters - CV-OPTIMIZED PERFECT DISTRIBUTION APPROACH
    clustering_method: str = "centroid_based"  # "hdbscan", "dbscan", "kmeans", "hybrid", "centroid_based"
    enable_aggressive_splitting: bool = True  # Enable aggressive cluster splitting
    enable_weighted_splitting: bool = True  # Enable weighted cluster splitting
    enable_dynamic_merging: bool = True  # Enable dynamic size-constrained merging
    force_exact_constraints: bool = True  # Force clusters to meet exact size constraints
    max_cluster_splitting_iterations: int = 20  # Maximum iterations for cluster splitting
    splitting_aggressiveness: float = 3.0  # How aggressively to split oversized clusters
    min_samples: int = 3  # Minimum samples for HDBSCAN/DBSCAN (ultra-permissive)
    min_cluster_size: int = 5  # Minimum cluster size for HDBSCAN (ultra-permissive)
    cluster_selection_epsilon: float = 0.015  # DBSCAN epsilon parameter (ultra-permissive)
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

    # Validation parameters
    validation_splits: int = 5  # Cross-validation splits
    bootstrap_iterations: int = 100  # Bootstrap iterations for stability
    stability_threshold: float = 0.8  # Minimum stability score

    # Output parameters
    save_intermediate_results: bool = True
    generate_cluster_reports: bool = True
    save_cluster_visualizations: bool = True

    # Memory optimization
    chunk_size: int = 50000  # Process data in chunks
    use_memory_optimization: bool = True

    # Advanced parameters
    adaptive_clustering: bool = True  # Adapt clustering based on data characteristics
    multi_stage_clustering: bool = True  # Use multi-stage approach
    outlier_detection_method: str = "none"  # "isolation_forest", "local_outlier_factor", "none"

    # HMM integration parameters
    hmm_min_states: int = 3  # Minimum HMM states to consider
    hmm_max_states: int = 8  # Maximum HMM states to consider
    hmm_state_prob_threshold: float = 0.6  # Minimum state probability threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'target_n_clusters': self.target_n_clusters,
            'target_coverage_pct': self.target_coverage_pct,
            'max_noise_pct': self.max_noise_pct,
            'min_cluster_size_pct': self.min_cluster_size_pct,
            'max_cluster_size_pct': self.max_cluster_size_pct,
            'clustering_method': self.clustering_method,
            'feature_dimensions': self.feature_dimensions,
            'quality_metrics': {
                'min_silhouette_score': self.min_silhouette_score,
                'min_calinski_harabasz_score': self.min_calinski_harabasz_score,
                'min_davies_bouldin_score': self.min_davies_bouldin_score,
                'target_coherence_score': self.target_coherence_score
            },
            'validation': {
                'validation_splits': self.validation_splits,
                'bootstrap_iterations': self.bootstrap_iterations,
                'stability_threshold': self.stability_threshold
            }
        }

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
        config.bootstrap_iterations = 50
        return config

def get_clustering_config(mode: str = "default") -> OptimalClusteringConfig:
    """Get clustering configuration based on mode."""
    if mode == "high_quality":
        return OptimalClusteringConfig.create_high_quality()
    elif mode == "fast_processing":
        return OptimalClusteringConfig.create_fast_processing()
    else:
        return OptimalClusteringConfig.create_default()

# Default configuration instance
DEFAULT_CONFIG = get_clustering_config("default")
HIGH_QUALITY_CONFIG = get_clustering_config("high_quality")
FAST_CONFIG = get_clustering_config("fast_processing")