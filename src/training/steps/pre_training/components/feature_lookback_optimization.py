"""
Feature Lookback Optimization Component Import.

This module provides the FeatureLookbackOptimizationComponent for the market analysis pipeline.
"""

# Import tprint for enhanced logging
try:
    from src.utils.tprint import tprint, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

from ..feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimizationComponent

# Log component import
if TPRINT_AVAILABLE:
    tprint_debug("🔧 Feature Lookback Optimization Component imported successfully")

__all__ = ['FeatureLookbackOptimizationComponent']
