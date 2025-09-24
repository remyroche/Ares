"""
Data-Driven Regime-to-Model Mapping System

This package provides a completely data-driven approach to map market regimes to optimal models
without any heuristics or hardcoded choices. It automatically discovers the best model for each
regime through performance analysis and continuous learning.

Key Components:
- DataDrivenModelSelector: Core model selection engine
- NASModelSelector: Integration with NAS regime detection
- TASModelSelector: Integration with TAS regime detection
"""

from .data_driven_model_selector import (
    DataDrivenModelSelector,
    ModelSelectorConfig,
    ModelPerformanceMetrics,
    RegimeModelMapping
)

from .nas_integration import NASModelSelector
from .tas_integration import TASModelSelector

__all__ = [
    'DataDrivenModelSelector',
    'ModelSelectorConfig', 
    'ModelPerformanceMetrics',
    'RegimeModelMapping',
    'NASModelSelector',
    'TASModelSelector'
]