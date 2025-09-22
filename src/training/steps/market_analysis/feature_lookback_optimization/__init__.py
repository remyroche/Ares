"""
Feature Lookback Optimization Module.

This module provides comprehensive feature lookback optimization capabilities
with robust validation, detailed reporting, and performance monitoring.

Main Components:
- FeatureLookbackOptimizationComponent: Main component (legacy)
- Modular components in submodules for better architecture
"""

from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .feature_lookback_optimization_modular import FeatureLookbackOptimizationComponent as ModularFeatureLookbackOptimizationComponent
from .optimization_reporter import OptimizationReporter
from src.utils.validation.unified_framework import FeatureLookbackValidationFramework, ValidationLevel, ValidationStatus
from .dependency_manager import DependencyManager, get_dependency, is_dependency_available
from .monitoring_metrics import MonitoringMetrics, MetricType, MetricLevel

# Import modular components
from .core.optimizer import CoreOptimizer, OptimizationMethod, OptimizationResult
from .validation.validator import InputValidator
from .error_handling.error_handler import StandardizedErrorHandler
from .performance.monitor import PerformanceMonitor

__all__ = [
    # Main components
    'FeatureLookbackOptimizationComponent',
    'ModularFeatureLookbackOptimizationComponent',

    # Modular components
    'CoreOptimizer',
    'InputValidator',
    'StandardizedErrorHandler',
    'PerformanceMonitor',

    # Types and enums
    'OptimizationMethod',
    'OptimizationResult',
    'ValidationLevel',
    'ValidationStatus',

    # Utilities
    'OptimizationReporter',
    'FeatureLookbackValidationFramework',
    'DependencyManager',
    'get_dependency',
    'is_dependency_available',
    'MonitoringMetrics',
    'MetricType',
    'MetricLevel'
]