"""
Feature Selection Optimizations Module

This module provides optimized implementations of feature selection operations
including vectorized operations and algorithm-specific optimizations.
"""

from .vectorized_operations import (
    VectorizedFeatureSelector,
    OptimizedCorrelationFilter,
    OptimizedVarianceFilter,
    create_vectorized_selector
)

__all__ = [
    'VectorizedFeatureSelector',
    'OptimizedCorrelationFilter',
    'OptimizedVarianceFilter',
    'create_vectorized_selector'
]
