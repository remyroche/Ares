"""
Lookback Optimization System

This module provides the lookback optimization system for feature generation,
allowing automatic optimization of lookback periods based on data-driven analysis.
"""

from .lookback_optimizer import (
    LookbackOptimizer,
    FeatureOptimizationConfig,
    FeatureOptimizationResult,
    optimize_feature_lookbacks,
    get_optimization_config
)

__all__ = [
    "LookbackOptimizer",
    "FeatureOptimizationConfig",
    "FeatureOptimizationResult",
    "optimize_feature_lookbacks",
    "get_optimization_config"
]