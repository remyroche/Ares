#!/usr/bin/env python3
"""HMM Clustering Package for Enhanced Regime Discovery.

This package contains all the enhanced components for HMM regime discovery:
- Optimized Bayesian parameter optimization
- Enhanced regime discovery features
- Economic significance validation
- Ensemble clustering (HMM + K-means + DBSCAN)
- Enhanced ML transition detection (Random Forest + LGBM)
- Memory optimization and streaming processing
- Hierarchical multi-scale detection
- Dynamic regime count optimization
- Regime persistence and forecasting
- Microservices architecture
- Real-time streaming pipeline
- Common utilities integration
"""

# Import the main functions and classes from the respective modules
from .step03_hmm_regime_discovery import (
    run_step as run_enhanced_step,
    HMMRegimeDiscoveryStep,
    EnhancedFeatureEngineer
)
from .step03_5_final_regime_clustering import (
    run_step as run_final_clustering_step,
    FinalRegimeClusteringStep
)

# Import enhanced HMM clustering modules with common utilities integration
from .hmm_executor import (
    create_hmm_dependencies,
    train_hmm_optimized,
    train_hmm_gpu_optimized,
    train_hmm_cpu_optimized,
    save_hmm_results,
    validate_hmm_model,
    HMMDependencies
)
from .hmm_utils import (
    HMMCommonUtilities,
    TechnicalIndicators,
    create_fallback_logger,
    safe_json_dump
)
from .clustering_executor import (
    create_clustering_dependencies,
    kmeans_standard,
    kmeans_minibatch,
    save_clustering_results,
    ClusteringDependencies
)

# from .parameter_optimization import ParameterOptimizer  # Temporarily disabled due to syntax errors
# Note: Removed ensemble_optimization import as it contained outdated clustering metrics

__all__ = [
    # Main entry points
    'run_enhanced_step',
    'run_final_clustering_step',
    
    # Core classes
    'HMMRegimeDiscoveryStep',
    'FinalRegimeClusteringStep',
    'EnhancedFeatureEngineer',
    'ParameterOptimizer',
    
    # Enhanced HMM clustering with common utilities
    'create_hmm_dependencies',
    'train_hmm_optimized',
    'train_hmm_gpu_optimized',
    'train_hmm_cpu_optimized',
    'save_hmm_results',
    'validate_hmm_model',
    'HMMDependencies',
    
    # HMM utilities
    'HMMCommonUtilities',
    'TechnicalIndicators',
    'create_fallback_logger',
    'safe_json_dump',
    
    # Clustering utilities
    'create_clustering_dependencies',
    'kmeans_standard',
    'kmeans_minibatch',
    'save_clustering_results',
    'ClusteringDependencies',
    
    # Note: Removed EnsembleWeightOptimizer as it contained outdated clustering metrics
]