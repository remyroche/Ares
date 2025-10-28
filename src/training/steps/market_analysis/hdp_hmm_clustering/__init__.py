"""
HDP-HMM Clustering Module

This module provides Hierarchical Dirichlet Process Hidden Markov Model
clustering for regime discovery.
"""

from .hdp_hmm_clusterer import (
    HDPHMMClusterer,
    HDPHMMConfig,
    HDPHMMResult,
    create_hdp_hmm_clusterer,
    HMM_AVAILABLE,
    HMM_LIBRARY
)

__all__ = [
    'HDPHMMClusterer',
    'HDPHMMConfig',
    'HDPHMMResult',
    'create_hdp_hmm_clusterer',
    'HMM_AVAILABLE',
    'HMM_LIBRARY'
]
