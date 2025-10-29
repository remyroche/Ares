"""
GMM Clustering Module

This module provides Gaussian Mixture Model-based regime discovery as an alternative
to HDBSCAN clustering, with correlation-based feature reduction to remove redundant
volatility features.
"""

from .gmm_regime_discovery_step import GMMRegimeDiscoveryStep, create_gmm_regime_discovery_step

__all__ = [
    'GMMRegimeDiscoveryStep',
    'create_gmm_regime_discovery_step'
]
