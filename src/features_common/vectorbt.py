"""
VectorBT compatibility module for features_common.

This module provides backward compatibility for VectorBT imports
and re-exports key functions from vectorbt_extensions.
"""

from .vectorbt_extensions import (
    get_unified_vectorbt_manager,
    UnifiedVectorBTManager,
    VectorBTOptimizationEngine,
    get_optimization_engine,
    GPUAccelerator,
    get_gpu_accelerator,
    VectorBTPerformanceMonitor,
    get_performance_monitor
)

__all__ = [
    'get_unified_vectorbt_manager',
    'UnifiedVectorBTManager',
    'VectorBTOptimizationEngine',
    'get_optimization_engine',
    'GPUAccelerator',
    'get_gpu_accelerator',
    'VectorBTPerformanceMonitor',
    'get_performance_monitor'
]
