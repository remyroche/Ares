"""
Adaptive Feature Selection for Small Sample SR Level Analysis

DEPRECATED: This module has been moved to src.feature_selection.specialized.adaptive_selector
Please update your imports to use the new location.

This compatibility shim will be removed in a future version.
"""

import warnings
from src.feature_selection.specialized.adaptive_selector import (
    AdaptiveFeatureSelectionConfig,
    FeatureSelectionResult,
    AdaptiveFeatureSelector,
)

# Issue deprecation warning
warnings.warn(
    "Importing from 'src.utils.sr_clustering.adaptive_feature_selection' is deprecated. "
    "Please use 'from src.feature_selection.specialized.adaptive_selector import ...' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'AdaptiveFeatureSelectionConfig',
    'FeatureSelectionResult',
    'AdaptiveFeatureSelector',
]
