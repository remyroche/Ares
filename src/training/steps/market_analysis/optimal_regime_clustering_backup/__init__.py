"""
Optimal Regime Clustering Module

This module provides tools for creating optimal clusters from HMM regime discovery output.
It creates 20-ish clusters that capture 90-95% of the distribution with 3-8% cluster sizes
and <5% noise for ML model training.

Main components:
- config: Configuration parameters for clustering
- clustering: Core clustering algorithms
- orchestrator: Pipeline orchestration
- utils: Utility functions for analysis and validation

Example usage:
    from optimal_regime_clustering import run_optimal_clustering

    results = run_optimal_clustering(
        data_path="hmm_cluster_data.parquet",
        output_dir="optimal_clusters/",
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="1h"
    )
"""

from .config import OptimalClusteringConfig, get_clustering_config, DEFAULT_CONFIG, HIGH_QUALITY_CONFIG, FAST_CONFIG
from .clustering import OptimalRegimeClusterer, ClusteringResult, create_optimal_clusterer
from .optimized_clustering import MatrixOptimizedClusterer, OptimizedClusteringResult, create_matrix_optimized_clusterer, cluster_regimes_optimized
from .orchestrator import OptimalRegimeClusteringOrchestrator, run_optimal_clustering, run_high_quality_clustering, run_fast_clustering, run_matrix_optimized_clustering
from .utils import (
    load_regime_data, prepare_clustering_features, calculate_cluster_statistics,
    calculate_cluster_quality_metrics, validate_cluster_quality, create_cluster_summary_report,
    detect_outliers, optimize_cluster_parameters,
    ClusterStatistics, ClusterValidationResult, ClusterQualityMetric
)

__version__ = "2.0.0"
__author__ = "Optimal Regime Clustering System"
__description__ = "Create optimal clusters from HMM regime discovery for ML training"

__all__ = [
    # Configuration
    "OptimalClusteringConfig",
    "get_clustering_config",
    "DEFAULT_CONFIG",
    "HIGH_QUALITY_CONFIG",
    "FAST_CONFIG",

    # Core clustering
    "OptimalRegimeClusterer",
    "ClusteringResult",
    "create_optimal_clusterer",

    # Matrix-optimized clustering
    "MatrixOptimizedClusterer",
    "OptimizedClusteringResult",
    "create_matrix_optimized_clusterer",
    "cluster_regimes_optimized",

    # Orchestration
    "OptimalRegimeClusteringOrchestrator",
    "run_optimal_clustering",
    "run_high_quality_clustering",
    "run_fast_clustering",
    "run_matrix_optimized_clustering",

    # Utilities
    "load_hmm_regime_data",
    "prepare_clustering_features",
    "calculate_cluster_statistics",
    "calculate_cluster_quality_metrics",
    "validate_cluster_quality",
    "create_cluster_summary_report",
    "detect_outliers",
    "optimize_cluster_parameters",
    "ClusterStatistics",
    "ClusterValidationResult",
    "ClusterQualityMetric"
]
