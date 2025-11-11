"""
Rolling HMM Clustering Module

This module implements a comprehensive rolling HMM clustering system with sticky priors
for market regime discovery and analysis. The implementation is optimized for Mac M1
and includes extensive feature engineering, hyperparameter optimization, and quality assessment.

Components:
- feature_engineering: EWMA-based rolling feature generation
- sticky_hmm_model: Sticky HMM implementation with regularization
- hpo_config: Hierarchical parameter optimization configuration
- rolling_hmm_regime_discovery_step: Main step implementation
"""

from src.training.steps.market_analysis.rolling_hmm_clustering.rolling_hmm_regime_discovery_step import (
    RollingHMMRegimeDiscoveryStep
)

__all__ = [
    'RollingHMMRegimeDiscoveryStep'
]

__version__ = '1.0.0'
