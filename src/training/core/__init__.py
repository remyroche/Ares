"""
Core training pipeline framework for Ares trading bot.

This module provides the foundational classes and interfaces for the modular
training pipeline architecture.
"""

from .checkpoint_manager import CheckpointManager
from .pipeline_base import PipelineStage, StageContext
from .pipeline_orchestrator import PipelineOrchestrator
from .stage_registry import StageRegistry

__all__ = [
    "PipelineStage",
    "StageContext",
    "PipelineOrchestrator",
    "StageRegistry",
    "CheckpointManager",
]
