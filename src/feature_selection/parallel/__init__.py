"""
Feature Selection Parallel Processing Module

This module provides parallel processing capabilities for feature selection
operations using hardware optimization tools.
"""

from .parallel_feature_selector import (
    ParallelFeatureSelector,
    ParallelSelectionManager,
    create_parallel_selector
)

__all__ = [
    'ParallelFeatureSelector',
    'ParallelSelectionManager',
    'create_parallel_selector'
]
