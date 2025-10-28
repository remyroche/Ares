"""
MS-DR Clustering Module

This module provides Markov-Switching Dynamic Regression clustering
for regime discovery.
"""

from .ms_dr_clusterer import (
    MSDRClusterer,
    MSDRConfig,
    MSDRResult,
    create_ms_dr_clusterer,
    MS_AVAILABLE,
    MS_LIBRARY
)

__all__ = [
    'MSDRClusterer',
    'MSDRConfig',
    'MSDRResult',
    'create_ms_dr_clusterer',
    'MS_AVAILABLE',
    'MS_LIBRARY'
]
