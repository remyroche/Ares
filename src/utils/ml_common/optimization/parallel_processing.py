"""
ML Common - Optimization Parallel Processing Module

This module re-exports the rich ParallelProcessingCoordinator from
utils.parallel_processing to avoid duplication and ensure a single
source of truth.
"""

from ..utils.parallel_processing import (
    ParallelProcessingCoordinator as _UtilsParallelProcessingCoordinator,
)


# Re-export under the optimization namespace for backwards compatibility
ParallelProcessingCoordinator = _UtilsParallelProcessingCoordinator


__all__ = ['ParallelProcessingCoordinator']
