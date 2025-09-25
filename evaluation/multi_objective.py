"""
Multi-Objective Optimization - Updated to use Unified Implementation

This module provides NAS-specific multi-objective optimization using
the unified multi-objective optimization framework.
"""

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

# NAS-specific wrapper for unified multi-objective optimizer
class NASMultiObjectiveOptimizer:
    """NAS-specific multi-objective optimizer using unified implementation."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize NAS multi-objective optimizer."""
        if not UNIFIED_MULTI_OBJECTIVE_AVAILABLE:
            raise ImportError("Unified multi-objective optimizer not available")
        self.unified_optimizer = UnifiedMultiObjectiveOptimizer(config)
    
    def __getattr__(self, name):
        """Delegate to unified optimizer."""
        return getattr(self.unified_optimizer, name)

# Maintain backward compatibility
__all__ = [
    'NASMultiObjectiveOptimizer',
    'UnifiedMultiObjectiveOptimizer',
    'PerformanceEstimator',
    'ArchitectureFeatures',
    'PerformancePrediction',
    'PerformanceMetric',
    'EstimatorType',
    'OptimizationConfig',
    'MultiObjectiveResult'
]