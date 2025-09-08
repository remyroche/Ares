"""Compatibility shim for step05_labeling.

This module re-exports `LabelingStep` so both `step5_labeling` (tests)
and `step05_labeling` (orchestrator) names resolve.
"""
from .step5_labeling import LabelingStep  # noqa: F401

__all__ = ["LabelingStep"]
