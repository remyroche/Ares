"""
Hardware optimization and management.

This module provides hardware-specific optimizations for M1 chips, memory management,
and GPU utilization.

Imports from the existing clusters directory where the hardware components are implemented.
"""

# Import from existing clusters directory
from ...clusters.m1_optimizer import M1Optimizer
from ...clusters.memory_manager import MemoryManager
from ...clusters.gpu_manager import GPUManager

__all__ = [
    'M1Optimizer',
    'MemoryManager',
    'GPUManager'
]
