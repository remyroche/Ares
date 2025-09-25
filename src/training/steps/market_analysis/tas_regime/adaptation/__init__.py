"""
Real-time Adaptation for TAS

Advanced adaptation capabilities for tree architecture search including:
- Real-time performance monitoring
- Dynamic architecture adaptation
- Incremental learning and updates
- Performance tracking and metrics collection
- Adaptive search strategies
"""

from .real_time_adaptation import TreeRealTimeAdapter, TreePerformanceMonitor, TreeAdaptiveSearch
from .dynamic_optimization import TreeDynamicOptimizer, TreeIncrementalLearner, TreeOnlineOptimizer
from src.utils.nas_tas.performance_tracker import AdvancedPerformanceTracker, MetricsCollector, PerformanceAnalytics

# Backward compatibility aliases
TreePerformanceTracker = AdvancedPerformanceTracker
TreeMetricsCollector = MetricsCollector
TreeAnalytics = PerformanceAnalytics

__all__ = [
    'TreeRealTimeAdapter', 'TreePerformanceMonitor', 'TreeAdaptiveSearch',
    'TreeDynamicOptimizer', 'TreeIncrementalLearner', 'TreeOnlineOptimizer',
    'TreePerformanceTracker', 'TreeMetricsCollector', 'TreeAnalytics'
]