"""
Components Package

This package provides various components for the pre-training pipeline.
"""

# Set up path first before any imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from .final_feature_selection import (
    FinalFeatureSelectionComponent,
    FinalFeatureSelectionConfig
)

# Import the component factory and registry
from .component_factory import BaseComponent, ComponentFactory
from .base_component import ComponentConfig
from .component_registry import ComponentRegistry

__all__ = [
    "FinalFeatureSelectionComponent",
    "FinalFeatureSelectionConfig",
    "ComponentFactory",
    "ComponentConfig", 
    "BaseComponent",
    "ComponentRegistry"
]