"""Core training pipeline framework for Ares trading bot.

This module provides the foundational classes and interfaces for the modular
training pipeline architecture.
"""

from .pipeline_orchestrator import PipelineOrchestrator
from .stage_context import StageContext
from .stage_registry import StageRegistry

__all__ = [
    "PipelineOrchestrator",
    "StageContext",
    "StageRegistry",
]
