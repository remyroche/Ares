"""
MOEA Optimization Framework for UnifiedDataDrivenPipeline

Provides robust multi-objective evolutionary algorithm optimization with
comprehensive convergence criteria and performance monitoring.

Key Features:
- Robust convergence criteria
- Anytime stopping conditions
- Hypervolume and ε-progress monitoring
- Time-boxed optimization
- Performance tracking
"""

from .robust_moea_optimizer import (
    RobustMOEAOptimizer,
    ConvergenceConfig,
    ConvergenceResult,
    ConvergenceCriterion
)

from .convergence_monitor import (
    ConvergenceMonitor,
    ConvergenceMonitorConfig,
    ConvergenceStatus,
    ProgressMetric
)

from .hypervolume_calculator import (
    HypervolumeCalculator,
    HypervolumeConfig,
    HypervolumeResult
)

from .performance_tracker import (
    MOEAPerformanceTracker,
    PerformanceConfig,
    PerformanceMetrics
)

__all__ = [
    # Robust MOEA optimizer
    'RobustMOEAOptimizer',
    'ConvergenceConfig',
    'ConvergenceResult',
    'ConvergenceCriterion',
    
    # Convergence monitor
    'ConvergenceMonitor',
    'ConvergenceMonitorConfig',
    'ConvergenceStatus',
    'ProgressMetric',
    
    # Hypervolume calculator
    'HypervolumeCalculator',
    'HypervolumeConfig',
    'HypervolumeResult',
    
    # Performance tracker
    'MOEAPerformanceTracker',
    'PerformanceConfig',
    'PerformanceMetrics'
]