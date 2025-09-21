"""
Regime Clustering Module

This module provides regime consolidation capabilities that take HMM discovery outputs
and create balanced, coherent clusters suitable for ML model training.

Key Features:
- Complete coverage regime consolidation (100% distribution accounted for)
- Similarity-based regime merging (preserves market information)
- Balanced cluster sizes (3-8% each)
- Top 20 clusters capture 90-95% of market states
- ML-ready output formats
"""

from .regime_consolidator import RegimeConsolidator
from .hmm_integration import HMMDiscoveryIntegration
from .ml_output_generator import MLOutputGenerator

__all__ = [
    'RegimeConsolidator',
    'HMMDiscoveryIntegration', 
    'MLOutputGenerator'
]

__version__ = "1.0.0"