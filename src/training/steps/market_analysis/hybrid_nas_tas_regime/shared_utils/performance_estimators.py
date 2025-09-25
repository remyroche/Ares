"""
Performance Estimators - Updated to use Unified Implementation

This module redirects to the unified multi-objective optimization framework.
"""

# All functionality moved to src.utils.nas_tas.unified_multi_objective

# Import unified multi-objective optimizer
try:
    from src.utils.nas_tas import (
        UnifiedMultiObjectiveOptimizer,
        PerformanceEstimator,
        ArchitectureFeatures,
        PerformancePrediction,
        PerformanceMetric,
        EstimatorType,
        OptimizationConfig,
        MultiObjectiveResult
    )
    UNIFIED_MULTI_OBJECTIVE_AVAILABLE = True
except ImportError:
    UNIFIED_MULTI_OBJECTIVE_AVAILABLE = False

# Maintain backward compatibility
__all__ = [
    'UnifiedMultiObjectiveOptimizer',
    'PerformanceEstimator',
    'ArchitectureFeatures',
    'PerformancePrediction',
    'PerformanceMetric',
    'EstimatorType',
    'OptimizationConfig',
    'MultiObjectiveResult'
]