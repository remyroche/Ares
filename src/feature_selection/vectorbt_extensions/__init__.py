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

from .vectorbt_unified_framework import VectorBTUnifiedFramework, create_vectorbt_unified_framework
from .vectorbt_feature_selector import VectorBTFeatureSelector
from .vectorbt_correlation_filter import VectorBTCorrelationFilter, create_vectorbt_correlation_filter
from .vectorbt_mutual_information import VectorBTMutualInformation, create_vectorbt_mutual_information
from .vectorbt_stability_selection import VectorBTStabilitySelection
from .vectorbt_mrmr_selector import VectorBTMRMRSelector, create_vectorbt_mrmr_selector
from .vectorbt_regularization import VectorBTRegularizationSelector
from .vectorbt_rfe_selector import VectorBTRFESelector
from .vectorbt_rolling_operations import VectorBTRollingOperations, create_vectorbt_rolling_operations
from .vectorbt_config import VectorBTFeatureSelectionConfig

__all__ = [
    'VectorBTUnifiedFramework',
    'create_vectorbt_unified_framework',
    'VectorBTFeatureSelector',
    'VectorBTCorrelationFilter',
    'create_vectorbt_correlation_filter',
    'VectorBTMutualInformation',
    'create_vectorbt_mutual_information',
    'VectorBTStabilitySelection',
    'VectorBTMRMRSelector',
    'create_vectorbt_mrmr_selector',
    'VectorBTRegularizationSelector',
    'VectorBTRFESelector',
    'VectorBTRollingOperations',
    'create_vectorbt_rolling_operations',
    'VectorBTFeatureSelectionConfig'
]
