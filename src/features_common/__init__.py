"""
Common utilities shared between feature_generation and feature_engineering_roadmap.

This module provides base classes and shared functionality to reduce duplication
between the two feature systems.
"""

__version__ = "1.0.0"

from .transforms.base_scaler import BaseScaler
from .optimization.cv_base import BaseCVSplitter
from .registry.base_registry import BaseFeatureRegistry

__all__ = [
    'BaseScaler',
    'BaseCVSplitter',
    'BaseFeatureRegistry',
]
