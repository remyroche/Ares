"""
Feature Selection Module

This module provides a comprehensive suite of feature selection tools organized into:
- Core: Main framework and selection functions
- Methods: Various selection algorithms (filter, wrapper, embedded, stability)
- Specialized: Domain-specific selectors (adaptive, directional, entropy-based)
- Dimensionality: PCA, VIF, and correlation handling
- Analysis: Feature importance, stability, temporal, and causal analysis

Usage Examples:
    # Basic feature selection
    from src.feature_selection import select_features
    result = select_features(X, y, method='comprehensive')
    
    # Adaptive selection for small samples
    from src.feature_selection.specialized import AdaptiveFeatureSelector
    selector = AdaptiveFeatureSelector()
    result = selector.select_features(X, y)
    
    # Regularization-based selection
    from src.feature_selection.methods import FeatureRegularizationSelector
    selector = FeatureRegularizationSelector()
    selector.fit(X, y)
    selected_features = selector.get_selected_features()
"""

# Core framework
from .core import (
    get_feature_selection_framework,
    select_features,
    run_comprehensive_feature_selection,
)

# Selection methods
from .methods import (
    MRMRSelector,
    ElasticNetStabilitySelector,
    StabilityAnalyzer,
    RecursiveFeatureEliminator,
    FeatureImportanceRanker,
    FeatureRegularizationSelector,
    FeatureRegularizationConfig,
    create_feature_regularization_selector,
)

# Specialized selectors
from .specialized import (
    EntropyBalancerConfig,
    EntropyFilterResult,
    EntropyStabilityFilter,
    AdaptiveFeatureSelector,
    AdaptiveFeatureSelectionConfig,
    AdaptiveFeatureSelectionResult,
    DirectionalFeatureSelectionConfig,
    DirectionalFeatureSelectionResult,
)

# Dimensionality reduction
from .dimensionality import (
    PCAModule,
    create_pca_module,
    VIFModule,
    create_vif_module,
)

__all__ = [
    # Core framework
    'get_feature_selection_framework',
    'select_features',
    'run_comprehensive_feature_selection',
    
    # Selection methods
    'MRMRSelector',
    'ElasticNetStabilitySelector',
    'StabilityAnalyzer',
    'RecursiveFeatureEliminator',
    'FeatureImportanceRanker',
    'FeatureRegularizationSelector',
    'FeatureRegularizationConfig',
    'create_feature_regularization_selector',
    
    # Specialized selectors
    'EntropyBalancerConfig',
    'EntropyFilterResult',
    'EntropyStabilityFilter',
    'AdaptiveFeatureSelector',
    'AdaptiveFeatureSelectionConfig',
    'AdaptiveFeatureSelectionResult',
    'DirectionalFeatureSelectionConfig',
    'DirectionalFeatureSelectionResult',
    
    # Dimensionality reduction
    'PCAModule',
    'create_pca_module',
    'VIFModule',
    'create_vif_module',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Ares Team'