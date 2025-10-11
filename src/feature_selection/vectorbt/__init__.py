"""
VectorBT-Enhanced Feature Selection

This module provides VectorBT-optimized feature selection methods that offer
significant performance improvements over standard implementations.

Key Features:
- 10-100x performance improvements with VectorBT vectorized operations
- Memory-efficient processing for large datasets
- Parallel processing capabilities
- Financial data optimization
- Unified API across all feature selection methods
"""

from .vectorbt_feature_selector import VectorBTFeatureSelector
from .vectorbt_correlation_filter import VectorBTCorrelationFilter
from .vectorbt_mutual_information import VectorBTMutualInformation
from .vectorbt_stability_selection import VectorBTStabilitySelection
from .vectorbt_mrmr_selector import VectorBTMRMRSelector
from .vectorbt_regularization import VectorBTRegularizationSelector
from .vectorbt_rfe_selector import VectorBTRFESelector
from .vectorbt_config import VectorBTFeatureSelectionConfig

__all__ = [
    'VectorBTFeatureSelector',
    'VectorBTCorrelationFilter', 
    'VectorBTMutualInformation',
    'VectorBTStabilitySelection',
    'VectorBTMRMRSelector',
    'VectorBTRegularizationSelector',
    'VectorBTRFESelector',
    'VectorBTFeatureSelectionConfig'
]