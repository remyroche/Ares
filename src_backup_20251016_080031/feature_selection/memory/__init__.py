"""
Feature Selection Memory Management Module

This module provides memory-efficient feature selection operations using
hardware optimization tools for large datasets.
"""

from .memory_efficient_selector import (
    MemoryEfficientFeatureSelector,
    ChunkedFeatureProcessor,
    SparseFeatureSelector,
    create_memory_efficient_selector
)

__all__ = [
    'MemoryEfficientFeatureSelector',
    'ChunkedFeatureProcessor', 
    'SparseFeatureSelector',
    'create_memory_efficient_selector'
]