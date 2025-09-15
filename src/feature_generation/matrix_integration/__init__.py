"""
Matrix Operations Integration

This module provides integration with the matrix operations framework
for optimized feature computation using vectorized operations and GPU acceleration.
"""

from .matrix_processor import (
    MatrixFeatureProcessor,
    VectorizedFeatureGenerator,
    get_matrix_processor,
    enable_matrix_acceleration
)

__all__ = [
    "MatrixFeatureProcessor",
    "VectorizedFeatureGenerator",
    "get_matrix_processor",
    "enable_matrix_acceleration"
]