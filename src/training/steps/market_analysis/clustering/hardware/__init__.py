"""
Hardware optimization and management.

This module provides hardware-specific optimizations for M1 chips, memory management,
and GPU utilization.
"""

from .m1_optimizer import M1Optimizer
from .memory_manager import MemoryManager
from .gpu_manager import GPUManager

__all__ = [
    'M1Optimizer',
    'MemoryManager',
    'GPUManager'
]
