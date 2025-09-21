"""
Optimal Regime Clustering Configuration

This configuration file defines parameters for creating 20 optimal clusters from HMM regime discovery output.
The goal is to capture 90-95% of the total distribution with 3-8% cluster sizes and <5% noise.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class OptimalClusteringConfig:
    """Configuration for optimal regime clustering."""

    # Target cluster parameters
    target_n_clusters: int = 20
    target_coverage_pct: float = 0.95  # 90-95% coverage
    max_noise_pct: float = 0.05  # <5% noise
    min_cluster_size_pct: float = 0.03  # 3% minimum
    max_cluster_size_pct: float = 0.08  # 8% maximum

    # Clustering algorithm parameters
    clustering_method: str = "hybrid"  # "hdbscan", "dbscan", "kmeans", "hybrid"
    min_samples: int = 50  # Minimum samples for HDBSCAN/DBSCAN
    min_cluster_size: int = 100  # Minimum cluster size for HDBSCAN
    cluster_selection_epsilon: float = 0.1  # DBSCAN epsilon parameter
    max_iter: int = 300  # Maximum iterations for iterative algorithms
    random_state: int = 42  # Random state for reproducibility

    # Quality metrics thresholds
    min_silhouette_score: float = 0.3  # Minimum acceptable silhouette score
    min_calinski_harabasz_score: float = 100.0  # Minimum CH score
    min_davies_bouldin_score: float = 1.5  # Maximum DB score (lower is better)
    target_coherence_score: float = 0.7  # Target cluster coherence

    # Feature dimensions (4D: volume, volatility, momentum, trend)
    feature_dimensions: List[str] = [
        'volume', 'volatility', 'momentum', 'trend'
    ]

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
    outlier_detection_method: str = "isolation_forest"  # "isolation_forest", "local_outlier_factor"

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