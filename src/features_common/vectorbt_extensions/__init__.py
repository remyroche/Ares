"""
Unified VectorBT management system.

This module provides a unified interface for VectorBT optimization
with automatic fallback, performance monitoring, and intelligent
optimization selection.
"""

from .unified_manager import UnifiedVectorBTManager, get_unified_vectorbt_manager
from .optimization_engine import VectorBTOptimizationEngine, get_optimization_engine
from .gpu_accelerator import GPUAccelerator, get_gpu_accelerator
from .performance_monitor import VectorBTPerformanceMonitor, get_performance_monitor

__all__ = [
    'UnifiedVectorBTManager',
    'get_unified_vectorbt_manager',
    'VectorBTOptimizationEngine',
    'get_optimization_engine',
    'GPUAccelerator',
    'get_gpu_accelerator',
    'VectorBTPerformanceMonitor',
    'get_performance_monitor'
]
