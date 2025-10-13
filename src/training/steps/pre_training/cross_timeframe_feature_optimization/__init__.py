"""
Cross-Timeframe Feature Optimization Module

This module provides comprehensive cross-timeframe feature optimization
following the pattern of FeatureLookbackOptimizationComponent.

Key Components:
- CrossTimeframeFeatureOptimizationComponent: Main optimization component
- CrossTimeframeOptimizer: Core feature optimization logic
- FeatureBacktester: Economic significance evaluation
- FeatureSelector: Feature selection and ranking
"""

from .cross_timeframe_feature_optimization import (
    CrossTimeframeFeatureOptimizationComponent,
    CrossTimeframeOptimizationConfig,
    CrossTimeframeOptimizationResult
)

from .core.cross_timeframe_optimizer import (
    CrossTimeframeOptimizer,
    CrossTimeframeOptimizationConfig as OptimizerConfig,
    OptimizationResult,
    OptimizationMethod
)

from .core.feature_backtester import (
    FeatureBacktester,
    BacktestConfig,
    BacktestResult
)

from .core.feature_selector import (
    FeatureSelector,
    SelectionConfig,
    SelectionResult
)

__all__ = [
    # Main component
    'CrossTimeframeFeatureOptimizationComponent',
    'CrossTimeframeOptimizationConfig',
    'CrossTimeframeOptimizationResult',
    
    # Core modules
    'CrossTimeframeOptimizer',
    'OptimizerConfig',
    'OptimizationResult',
    'OptimizationMethod',
    
    'FeatureBacktester',
    'BacktestConfig',
    'BacktestResult',
    
    'FeatureSelector',
    'SelectionConfig',
    'SelectionResult'
]