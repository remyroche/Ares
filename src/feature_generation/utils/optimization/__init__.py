"""
Feature Optimization Package

This package provides various optimization strategies for feature generation,
including the new complementary lookback optimization for tactician training.
"""

from .complementary_lookback_optimizer import (
    ComplementaryLookbackOptimizer,
    ComplementaryOptimizationConfig,
    ComplementaryOptimizationResult,
    ComplementaryOptimizationMethod,
    optimize_complementary_lookbacks,
    get_complementary_optimization_config
)

from .tactician_feature_optimization import (
    TacticianFeatureOptimizer,
    optimize_tactician_features,
    get_tactician_optimization_config
)

# Legacy imports for backward compatibility
try:
    from .lookback_optimizer import (
        LookbackOptimizer,
        FeatureOptimizationConfig,
        FeatureOptimizationResult,
        OptimizationMethod,
        optimize_feature_lookbacks,
        get_optimization_config
    )
except ImportError:
    # Legacy optimizer not available
    pass

__all__ = [
    # Complementary optimization (new approach)
    'ComplementaryLookbackOptimizer',
    'ComplementaryOptimizationConfig', 
    'ComplementaryOptimizationResult',
    'ComplementaryOptimizationMethod',
    'optimize_complementary_lookbacks',
    'get_complementary_optimization_config',
    
    # Tactician-specific optimization
    'TacticianFeatureOptimizer',
    'optimize_tactician_features',
    'get_tactician_optimization_config',
    
    # Legacy optimization (for backward compatibility)
    'LookbackOptimizer',
    'FeatureOptimizationConfig',
    'FeatureOptimizationResult', 
    'OptimizationMethod',
    'optimize_feature_lookbacks',
    'get_optimization_config'
]