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
from .parameter_optimization import ParameterOptimizer
from .ensemble_optimization import EnsembleWeightOptimizer

__all__ = [
    # Main entry points
    'run_enhanced_step',
    'run_final_clustering_step',
    
    # Core classes
    'HMMRegimeDiscoveryStep',
    'FinalRegimeClusteringStep',
    'EnhancedFeatureEngineer',
    'ParameterOptimizer',
    'EnsembleWeightOptimizer'
]