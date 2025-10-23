"""
Price Patterns Research Framework

This module focuses on mathematical discovery and definition of price patterns.
All patterns are defined precisely using mathematical formulas and provide
both binary labels and gradient-based intensity measurements for ML training.

Main Components:
- core_patterns.py: Fundamental mathematical pattern definitions
- gradient_targets.py: Binary + gradient intensity targets for ML
- lstm_discovery.py: LSTM autoencoder pattern discovery
- matrix_profile_discovery.py: Matrix profile motif discovery
- pattern_discovery_framework.py: Complete pattern discovery framework
- advanced_pattern_definitions.py: Sophisticated pattern definitions
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

from .lstm_discovery import (
    LSTMPricePatternDiscovery,
    LSTMDiscoveredPattern
)

from .matrix_profile_discovery import (
    MatrixProfileOrchestrator,
    MatrixProfilePattern
)

__all__ = [
    'PurePricePatternOrchestrator',
    'PurePatternResult',
    'PurePricePattern',
    'GradientPatternTargetGenerator',
    'PatternIntensityMeasurement',
    'LSTMPricePatternDiscovery',
    'LSTMDiscoveredPattern',
    'MatrixProfileOrchestrator',
    'MatrixProfilePattern'
]
