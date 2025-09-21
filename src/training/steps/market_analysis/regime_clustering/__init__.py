"""
Regime Clustering Package for HMM Regime Consolidation.

This package provides functionality to cluster small HMM regimes into coherent,
larger clusters suitable for ML model training.

Main Components:
- RegimeClusterer: Main clustering orchestrator
- ClusterValidator: Quality validation for clusters
- ClusterAnalyzer: Analysis and reporting tools
"""

from .regime_clusterer import RegimeClusterer
from .cluster_validator import ClusterValidator
from .cluster_analyzer import ClusterAnalyzer

__all__ = [
    'RegimeClusterer',
    'ClusterValidator', 
    'ClusterAnalyzer'
]