"""
Architecture Evaluation Framework

This module provides comprehensive evaluation and validation tools for neural architectures
in regime detection tasks, including multi-objective optimization and performance metrics.
"""

from .architecture_evaluator import (
    ArchitectureEvaluator,
    RegimeDetectionEvaluator,
    MultiObjectiveEvaluator,
    PerformanceMetrics
)

from .regime_metrics import (
    RegimeStabilityMetrics,
    EconomicSignificanceMetrics,
    TradingViabilityMetrics,
    RegimeTransitionMetrics,
    MicroRegimeMetrics
)

from .multi_objective import (
    MultiObjectiveOptimizer,
    ParetoFrontier,
    NSGAIIOptimizer,
    WeightedSumOptimizer,
    ObjectiveFunction
)

__all__ = [
    # Architecture evaluation
    'ArchitectureEvaluator',
    'RegimeDetectionEvaluator',
    'MultiObjectiveEvaluator',
    'PerformanceMetrics',
    
    # Regime metrics
    'RegimeStabilityMetrics',
    'EconomicSignificanceMetrics',
    'TradingViabilityMetrics',
    'RegimeTransitionMetrics',
    'MicroRegimeMetrics',
    
    # Multi-objective optimization
    'MultiObjectiveOptimizer',
    'ParetoFrontier',
    'NSGAIIOptimizer',
    'WeightedSumOptimizer',
    'ObjectiveFunction'
]