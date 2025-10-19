"""
Data-Driven Regime-to-Model Mapping System

This package provides a completely data-driven approach to map market regimes to optimal models
without any heuristics or hardcoded choices. It automatically discovers the best model for each
regime through performance analysis and continuous learning.

Key Components:
- DataDrivenModelSelector: Core model selection engine
"""

from .data_driven_model_selector import (
    DataDrivenModelSelector,
    ModelSelectorConfig,
    ModelPerformanceMetrics,
    RegimeModelMapping
)

__all__ = [
    'DataDrivenModelSelector',
    'ModelSelectorConfig',
    'ModelPerformanceMetrics',
    'RegimeModelMapping'
]
