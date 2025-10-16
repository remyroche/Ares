"""
ML Common - Optimization Memory Optimization Module

Compatibility shim that re-exports MemoryEfficientTraining from utils.memory_optimization
to avoid duplication and keep existing imports working.
"""

from ..utils.memory_optimization import MemoryEfficientTraining

__all__ = ['MemoryEfficientTraining']
