from __future__ import annotations

"""Core training API exports.

This package exposes the simplified training manager facade used by other
components. Legacy pipeline internals (e.g., pipeline_base, stage_registry)
are intentionally not imported here to avoid hard dependencies that are not
required by the current codebase.
"""

from .training_manager import TrainingManager, create_training_manager

__all__ = [
    "TrainingManager",
    "create_training_manager",
]
