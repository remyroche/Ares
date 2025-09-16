"""
Triple Barrier Labeling for Market Analysis

This module provides comprehensive triple barrier labeling functionality for market analysis,
integrating with the existing utility infrastructure and ML common tools.

Key Features:
- Multiple triple barrier implementations
- Regime-aware labeling
- Quality assessment and validation
- Cross-validation integration
- M1 hardware optimization
- GPU acceleration support
"""

from .core import TripleBarrierLabeler, TripleBarrierConfig
from .regime_aware import RegimeAwareLabeler, RegimeAwareConfig
from .quality_assessment import LabelQualityAssessment
from .cross_validation import LabelCrossValidator
from .utils import MarketAnalysisUtils
from .optimized_labeler import OptimizedTripleBarrierLabeler, OptimizedBarrierParams, RegimeTradingMetrics

__version__ = "1.0.0"
__author__ = "Market Analysis Team"

__all__ = [
    "TripleBarrierLabeler",
    "TripleBarrierConfig", 
    "RegimeAwareLabeler",
    "RegimeAwareConfig",
    "LabelQualityAssessment",
    "LabelCrossValidator",
    "MarketAnalysisUtils",
    "OptimizedTripleBarrierLabeler",
    "OptimizedBarrierParams",
    "RegimeTradingMetrics"
]