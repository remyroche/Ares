"""
Hardware modules for NAS-TAS regime detection.

This package contains hardware management and optimization components for
market analysis pipeline steps, including GPU acceleration, memory
optimization, and M1 chip optimization.
"""

from .hardware_manager import HardwareManager, HardwareConfig, HardwareCapabilities
from .gpu_accelerator import GPUAccelerator, GPUConfig
from .memory_optimizer import MemoryOptimizer, MemoryConfig
from .m1_optimizer import M1Optimizer, M1Config
from .performance_monitor import PerformanceMonitor, PerformanceConfig

__all__ = [
    'HardwareManager',
    'HardwareConfig',
    'HardwareCapabilities',
    'GPUAccelerator',
    'GPUConfig',
    'MemoryOptimizer',
    'MemoryConfig',
    'M1Optimizer',
    'M1Config',
    'PerformanceMonitor',
    'PerformanceConfig'
]