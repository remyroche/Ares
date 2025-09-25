"""
Unified TAS-NAS Regime Detection System

This module provides a unified regime detection system that combines the best aspects
of both TAS (Tree Architecture Search) and NAS (Neural Architecture Search) regime
detection with enhanced economic significance and trading viability evaluation.
"""

from .unified_regime_config import (
    UnifiedRegimeConfig,
    RegimeDetectionMethod,
    OptimizationStrategy,
    EconomicEvaluationMode
)

from .unified_regime_detector import (
    UnifiedRegimeDetector,
    UnifiedRegimeResult
)

from .performance_optimizer import (
    PerformanceOptimizer,
    PerformanceCache,
    GPUAccelerator,
    MemoryOptimizer,
    optimize_performance,
    get_performance_optimizer
)

from .real_time_monitor import (
    RealTimeRegimeMonitor,
    RegimeChangeEvent,
    RealTimeMetrics,
    DataStreamProcessor,
    RegimeChangeDetector,
    PerformanceMonitor,
    create_real_time_monitor
)

__all__ = [
    'UnifiedRegimeConfig',
    'RegimeDetectionMethod',
    'OptimizationStrategy',
    'EconomicEvaluationMode',
    'UnifiedRegimeDetector',
    'UnifiedRegimeResult',
    'PerformanceOptimizer',
    'PerformanceCache',
    'GPUAccelerator',
    'MemoryOptimizer',
    'optimize_performance',
    'get_performance_optimizer',
    'RealTimeRegimeMonitor',
    'RegimeChangeEvent',
    'RealTimeMetrics',
    'DataStreamProcessor',
    'RegimeChangeDetector',
    'PerformanceMonitor',
    'create_real_time_monitor'
]

__version__ = "1.0.0"
__author__ = "Unified Regime Detection System"