from __future__ import annotations

"""Core training pipeline framework for Ares trading bot.

This module provides the foundational classes and interfaces for the modular
training pipeline architecture.
"""

from src.utils.warning_symbols import (
    connection_error,
    critical,
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    missing,
    problem,
    timeout,
    validation_error,
    warning,
)

from .pipeline_base import PipelineStage, StageContext
from .stage_registry import StageRegistry
from .checkpoint_manager import CheckpointManager

__all__ = [
    "PipelineStage",
    "StageContext",
    "StageRegistry",
    "CheckpointManager",
]
