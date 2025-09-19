"""
Pure Price Action Pattern Research Framework

This module focuses exclusively on price action patterns - what prices actually do,
not the underlying causes. All patterns are defined mathematically using only
price movements, without reference to volume, fundamentals, or market structure.

Main Components:
- core_patterns.py: Basic mathematical pattern definitions
- advanced_patterns.py: Sophisticated price action patterns  
- ml_discovery.py: ML-based pattern discovery methods
- gradient_targets.py: Gradient-based pattern intensity measurement
- lstm_discovery.py: LSTM autoencoder pattern discovery
- matrix_profile_discovery.py: Matrix profile motif discovery
"""

from .core_patterns import (
    PurePricePatternOrchestrator,
    PurePatternResult,
    PurePricePattern
)

from .gradient_targets import (
    GradientPatternTargetGenerator,
    PatternIntensityMeasurement
)

__all__ = [
    'PurePricePatternOrchestrator',
    'PurePatternResult', 
    'PurePricePattern',
    'GradientPatternTargetGenerator',
    'PatternIntensityMeasurement'
]