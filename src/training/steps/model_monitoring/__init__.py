"""
Model Performance Monitoring Step

This package provides comprehensive model performance monitoring and tracking
for all trained models, ensuring continuous model health and performance.
"""

__version__ = "1.0.0"
__author__ = "Model Monitoring Framework"

# Import main monitoring components
try:
    from .model_performance_monitor import ModelPerformanceMonitor
    from .drift_detector import DriftDetector
    from .performance_tracker import PerformanceTracker
    from .model_health_checker import ModelHealthChecker
    from .monitoring_pipeline import ModelMonitoringPipeline
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

__all__ = [
    'ModelPerformanceMonitor',
    'DriftDetector',
    'PerformanceTracker', 
    'ModelHealthChecker',
    'ModelMonitoringPipeline',
    'MONITORING_AVAILABLE'
]