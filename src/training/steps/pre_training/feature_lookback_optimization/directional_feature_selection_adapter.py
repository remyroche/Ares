"""
Directional Feature Selection Adapter

DEPRECATED: This module has been moved to src.feature_selection.specialized.directional_selector
Please update your imports to use the new location.

This compatibility shim will be removed in a future version.
"""

import warnings
from src.feature_selection.specialized.directional_selector import (
    DirectionalFeatureSelectionConfig,
    DirectionalFeatureSelectionResult,
)

# Issue deprecation warning
warnings.warn(
    "Importing from 'src.training.steps.pre_training.feature_lookback_optimization.directional_feature_selection_adapter' is deprecated. "
    "Please use 'from src.feature_selection.specialized.directional_selector import ...' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Import all public members from the new location
from src.feature_selection.specialized.directional_selector import *

__all__ = [
    'DirectionalFeatureSelectionConfig',
    'DirectionalFeatureSelectionResult',
]