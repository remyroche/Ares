"""
Advanced Clustering Algorithms for Statsmodels

This module provides comprehensive clustering algorithms including temporal modeling,
static clustering, hybrid approaches, and ensemble methods.

Key Features:
- Temporal clustering using HMM/Markov-switching
- Static clustering (hierarchical, spectral, community detection)
- Hybrid clustering combining static + temporal
- Ensemble methods with consensus clustering
- Churn regularization with stickiness priors
"""

from .temporal_clustering import TemporalClusteringEngine
from .static_clustering import StaticClusteringEngine
from .hybrid_clustering import HybridClusteringEngine
from .ensemble_clustering import EnsembleClusteringEngine
from .churn_regularization import ChurnRegularizer

__all__ = [
    'TemporalClusteringEngine',
    'StaticClusteringEngine', 
    'HybridClusteringEngine',
    'EnsembleClusteringEngine',
    'ChurnRegularizer'
]