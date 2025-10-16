"""
Feature Engineering Optimization Package

Unified optimization system for feature generation parameters.
"""

from .unified_optimizer import (
    FeatureGenerationOptimizer,
    FeatureOptimizationConfig,
    FeatureOptimizationResult,
    OptimizationMethod,
    ValidationLevel,
    OptimizationConfigManager,
    get_feature_optimizer,
    optimize_feature_lookback,
    get_optimization_config,
    get_default_config,

    # Backward compatibility aliases
    LookbackOptimizer,
    OptimizationSystemConfig
)

__all__ = [
    'FeatureGenerationOptimizer',
    'FeatureOptimizationConfig',
    'FeatureOptimizationResult',
    'OptimizationMethod',
    'ValidationLevel',
    'OptimizationConfigManager',
    'get_feature_optimizer',
    'optimize_feature_lookback',
    'get_optimization_config',
    'get_default_config',
    'LookbackOptimizer',
    'OptimizationSystemConfig'
]
