"""
ML Common - Optimization Parallel Processing Module (consolidated)

This module re-exports the canonical ParallelProcessingCoordinator from
`src.utils.ml_common.utils.parallel_processing` to eliminate duplicate
implementations and keep a single source of truth.
"""

from ..utils.parallel_processing import ParallelProcessingCoordinator  # noqa: F401

__all__ = ['ParallelProcessingCoordinator']
