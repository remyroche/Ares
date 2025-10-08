"""
Feature Selection Utilities

DEPRECATED: This module has been moved to src.feature_selection
Please update your imports to use the new location.

This compatibility shim will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from 'src.utils.feature_selection' is deprecated. "
    "Please use 'from src.feature_selection import ...' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new locations for backward compatibility
from src.feature_selection.core.framework import (
    get_feature_selection_framework,
    select_features,
    run_comprehensive_feature_selection,
    lasso_feature_selection,
    cross_validated_feature_selection,
    hierarchical_feature_selection,
    comprehensive_feature_selection,
)
from src.feature_selection.methods import (
    MRMRSelector,
    ElasticNetStabilitySelector,
    RecursiveFeatureEliminator,
    FeatureImportanceRanker,
    StabilityAnalyzer,
)
from src.feature_selection.dimensionality import (
    PCAModule,
    create_pca_module,
    VIFModule,
    create_vif_module,
)

__all__ = [
    'get_feature_selection_framework',
    'select_features',
    'run_comprehensive_feature_selection',
    'lasso_feature_selection',
    'cross_validated_feature_selection',
    'hierarchical_feature_selection',
    'comprehensive_feature_selection',
    'MRMRSelector',
    'ElasticNetStabilitySelector',
    'RecursiveFeatureEliminator',
    'FeatureImportanceRanker',
    'StabilityAnalyzer',
    'PCAModule',
    'create_pca_module',
    'VIFModule',
    'create_vif_module',
]