"""
Feature Selection and Regularization Module

DEPRECATED: This module has been moved to src.feature_selection.methods.regularization
Please update your imports to use the new location.

This compatibility shim will be removed in a future version.
"""

import warnings
from src.feature_selection.methods.regularization import (
    FeatureRegularizationConfig,
    FeatureRegularizationSelector,
    create_feature_regularization_selector,
)

# Issue deprecation warning
warnings.warn(
    "Importing from 'src.utils.feature_selection_regularization' is deprecated. "
    "Please use 'from src.feature_selection.methods.regularization import ...' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'FeatureRegularizationConfig',
    'FeatureRegularizationSelector',
    'create_feature_regularization_selector',
]
