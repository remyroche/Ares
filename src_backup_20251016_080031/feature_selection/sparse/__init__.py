"""
Feature Selection Sparse Matrix Support Module

This module provides optimized feature selection operations for sparse matrices
with memory-efficient algorithms.
"""

from .sparse_feature_selector import (
    SparseFeatureSelector,
    SparseMatrixProcessor,
    create_sparse_selector
)

__all__ = [
    'SparseFeatureSelector',
    'SparseMatrixProcessor',
    'create_sparse_selector'
]