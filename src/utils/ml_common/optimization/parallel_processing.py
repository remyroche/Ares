"""
ML Common - Optimization Parallel Processing Module

This module re-exports the rich ParallelProcessingCoordinator from
utils.parallel_processing to avoid duplication and ensure a single
source of truth.
"""

# Best-effort import from utils; if unavailable, define a minimal fallback
try:
    from ..utils.parallel_processing import (
        ParallelProcessingCoordinator as _UtilsParallelProcessingCoordinator,
    )
    ParallelProcessingCoordinator = _UtilsParallelProcessingCoordinator
except Exception:
    class ParallelProcessingCoordinator:  # type: ignore
        """Minimal fallback coordinator to avoid import errors.

        Provides a compatible interface subset used by HPO utilities.
        """

        def __init__(self, config: Optional[dict] = None):
            self.config = config or {}

        def map(self, fn, iterable, max_workers: int = 1):
            return [fn(x) for x in iterable]


# Re-export under the optimization namespace for backwards compatibility


__all__ = ['ParallelProcessingCoordinator']

