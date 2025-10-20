"""
ML Common - Optimization Memory Optimization Module

This module now re-exports memory optimization from the hardware optimization system
to maintain compatibility while using the new hardware-aware optimizations.
"""

from ..hardware_optimized_parallel_processor import HardwareOptimizedMLProcessor
from ..gpu_acceleration_utils import GPUAccelerationUtils

# Compatibility alias
MemoryEfficientTraining = HardwareOptimizedMLProcessor

__all__ = ['MemoryEfficientTraining', 'HardwareOptimizedMLProcessor', 'GPUAccelerationUtils']
