"""
HDBSCAN-Based Regime Discovery System

A comprehensive regime discovery system that replaces NAS/TAS clustering
with HDBSCAN while preserving sophisticated post-clustering optimization.

Key Features:
- 5 feature families (Returns, Volatility, Volume/Flow, Entropy, Spectral)
- Multi-mode dimensionality reduction (PCA/UMAP/densMAP)
- HDBSCAN clustering with tree export
- Post-clustering optimization with change budget
- Economic validation and profiling
- Temporal stabilization with causal/acausal modes
- Deterministic reproducibility
- Hardware optimization for M1 systems
"""

from .main_regime_discovery import HDBSCANRegimeDiscovery, RegimeResult
from .config.regime_discovery_config import RegimeDiscoveryConfig
from .hdbscan_regime_discovery_step import HDBSCANRegimeDiscoveryStep

__all__ = [
    'HDBSCANRegimeDiscovery',
    'RegimeResult', 
    'RegimeDiscoveryConfig',
    'HDBSCANRegimeDiscoveryStep'
]

__version__ = '1.0.0'
__author__ = 'Ares Trading System'
__description__ = 'HDBSCAN-based regime discovery with economic validation'
