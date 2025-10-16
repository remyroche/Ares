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

# Core framework - VectorBT based
from .core.framework import (
    get_feature_selection_framework,
    select_features,
    benchmark_methods,
    get_performance_stats,
    reset_framework,
    # Legacy compatibility
    get_enhanced_framework,
    enhanced_select_features,
)

# VectorBT framework
from .vectorbt import (
    VectorBTUnifiedFramework,
    create_vectorbt_unified_framework,
    VectorBTFeatureSelector,
    VectorBTCorrelationFilter,
    VectorBTMutualInformation,
    VectorBTStabilitySelection,
    VectorBTMRMRSelector,
    VectorBTRegularizationSelector,
    VectorBTRFESelector,
    VectorBTMemoryOptimizer,
    VectorBTRollingOperations,
    VectorBTFeatureSelectionConfig,
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

# Enhanced capabilities
from .caching import (
    IntelligentFeatureCache,
    FeatureSelectionCacheManager,
    cached_feature_selection,
    create_feature_cache,
)

from .error_handling import (
    FeatureSelectionError,
    InsufficientDataError,
    SelectionConvergenceError,
    ConfigurationError,
    EnhancedErrorHandler,
    robust_feature_selection,
    create_error_handler,
)

from .memory import (
    MemoryEfficientFeatureSelector,
    ChunkedFeatureProcessor,
    SparseFeatureSelector,
    create_memory_efficient_selector,
)

from .parallel import (
    ParallelFeatureSelector,
    ParallelSelectionManager,
    create_parallel_selector,
)

from .optimizations import (
    VectorizedFeatureSelector,
    OptimizedCorrelationFilter,
    OptimizedVarianceFilter,
    create_vectorized_selector,
)

from .sparse import (
    SparseFeatureSelector as SparseMatrixSelector,
    SparseMatrixProcessor,
    create_sparse_selector,
)

from .chunked import (
    ChunkedFeatureProcessor as ChunkedProcessor,
    AdaptiveChunkProcessor,
    create_chunked_processor,
)

# Advanced selection methods
from .advanced import (
    AdvancedFeatureSelector,
    LASSOFeatureSelector,
    RandomForestFeatureSelector,
    LightGBMFeatureSelector,
    EnsembleAdvancedSelector,
    create_advanced_selector,

    FeatureSelectionValidator,
    CrossValidationFramework,
    RegressionTestFramework,
    ValidationMetrics,
    create_validation_framework,

    PermutationImportanceCalculator,
    PermutationConfig,
    create_permutation_calculator,
)

# Enhanced advanced methods
from .advanced import (
    EnhancedEnsembleAdvancedSelector,
    EnhancedAdvancedFeatureSelector,
    EnhancedEnsembleConfig,
    EnhancedAdvancedConfig,
    create_enhanced_ensemble_selector,
    create_enhanced_advanced_selector,
)

# Pre-filtering and improved mRMR
from .advanced import (
    MRMRSpearmanPreFilter,
    create_mrmr_spearman_prefilter,
    ImprovedMRMR,
    create_improved_mrmr,
    EnhancedMultiStageRFE,
    PlateauDetector,
    create_enhanced_multi_stage_rfe,
)

__all__ = [
    # Core framework - VectorBT based
    'get_feature_selection_framework',
    'select_features',
    'benchmark_methods',
    'get_performance_stats',
    'reset_framework',
    'get_enhanced_framework',
    'enhanced_select_features',

    # VectorBT framework
    'VectorBTUnifiedFramework',
    'create_vectorbt_unified_framework',
    'VectorBTFeatureSelector',
    'VectorBTCorrelationFilter',
    'VectorBTMutualInformation',
    'VectorBTStabilitySelection',
    'VectorBTMRMRSelector',
    'VectorBTRegularizationSelector',
    'VectorBTRFESelector',
    'VectorBTMemoryOptimizer',
    'VectorBTRollingOperations',
    'VectorBTFeatureSelectionConfig',

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

    # Enhanced capabilities
    'IntelligentFeatureCache',
    'FeatureSelectionCacheManager',
    'cached_feature_selection',
    'create_feature_cache',

    'FeatureSelectionError',
    'InsufficientDataError',
    'SelectionConvergenceError',
    'ConfigurationError',
    'EnhancedErrorHandler',
    'robust_feature_selection',
    'create_error_handler',

    'MemoryEfficientFeatureSelector',
    'ChunkedFeatureProcessor',
    'SparseFeatureSelector',
    'create_memory_efficient_selector',

    'ParallelFeatureSelector',
    'ParallelSelectionManager',
    'create_parallel_selector',

    'VectorizedFeatureSelector',
    'OptimizedCorrelationFilter',
    'OptimizedVarianceFilter',
    'create_vectorized_selector',

    'SparseMatrixSelector',
    'SparseMatrixProcessor',
    'create_sparse_selector',

    'ChunkedProcessor',
    'AdaptiveChunkProcessor',
    'create_chunked_processor',

    # Advanced selection methods
    'AdvancedFeatureSelector',
    'LASSOFeatureSelector',
    'RandomForestFeatureSelector',
    'LightGBMFeatureSelector',
    'EnsembleAdvancedSelector',
    'create_advanced_selector',

    'FeatureSelectionValidator',
    'CrossValidationFramework',
    'RegressionTestFramework',
    'ValidationMetrics',
    'create_validation_framework',

    'PermutationImportanceCalculator',
    'PermutationConfig',
    'create_permutation_calculator',

    # Enhanced advanced methods
    'EnhancedEnsembleAdvancedSelector',
    'EnhancedAdvancedFeatureSelector',
    'EnhancedEnsembleConfig',
    'EnhancedAdvancedConfig',
    'create_enhanced_ensemble_selector',
    'create_enhanced_advanced_selector',

    # Pre-filtering and improved mRMR
    'MRMRSpearmanPreFilter',
    'create_mrmr_spearman_prefilter',
    'ImprovedMRMR',
    'create_improved_mrmr',

    # Enhanced multi-stage RFE
    'EnhancedMultiStageRFE',
    'PlateauDetector',
    'create_enhanced_multi_stage_rfe',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Ares Team'
