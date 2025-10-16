"""
Advanced Feature Selection Methods

This module provides advanced feature selection methods using LASSO, RandomForest,
and LightGBM with permutation importance and comprehensive validation framework.
"""

from .advanced_selector import (
    AdvancedFeatureSelector,
    LASSOFeatureSelector,
    RandomForestFeatureSelector,
    LightGBMFeatureSelector,
    EnsembleAdvancedSelector,
    create_advanced_selector
)

# Enhanced components
from .enhanced_config import (
    EnhancedEnsembleConfig,
    EnhancedAdvancedConfig,
    AdaptiveWeightingConfig,
    ConfidenceScoringConfig,
    NativeValidationConfig,
    DynamicFeatureSelectionConfig,
    ElbowMethodConfig,
    StatisticalThresholdingConfig
)

from .enhanced_ensemble_selector import (
    EnhancedEnsembleAdvancedSelector
)

from .enhanced_advanced_selector import (
    EnhancedAdvancedFeatureSelector
)

from .adaptive_weighting import (
    AdaptiveWeightingSystem
)

from .confidence_scoring import (
    ConfidenceScoringSystem
)

from .native_validation import (
    NativeValidationFramework
)

from .dynamic_selection import (
    DynamicFeatureSelector
)

from .prefiltering import (
    MRMRSpearmanPreFilter,
    create_mrmr_spearman_prefilter
)

from .improved_mrmr import (
    ImprovedMRMR,
    create_improved_mrmr
)

from .enhanced_multi_stage_rfe import (
    EnhancedMultiStageRFE,
    PlateauDetector,
    create_enhanced_multi_stage_rfe
)

# Factory functions
from typing import Optional

def create_enhanced_ensemble_selector(config: Optional[EnhancedEnsembleConfig] = None) -> EnhancedEnsembleAdvancedSelector:
    """Create an enhanced ensemble advanced selector."""
    return EnhancedEnsembleAdvancedSelector(config)

def create_enhanced_advanced_selector(config: Optional[EnhancedAdvancedConfig] = None) -> EnhancedAdvancedFeatureSelector:
    """Create an enhanced advanced feature selector."""
    return EnhancedAdvancedFeatureSelector(config)

from .validation_framework import (
    FeatureSelectionValidator,
    CrossValidationFramework,
    RegressionTestFramework,
    ValidationMetrics,
    create_validation_framework
)

from .permutation_importance import (
    PermutationImportanceCalculator,
    PermutationConfig,
    create_permutation_calculator
)

__all__ = [
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

    # Enhanced components
    'EnhancedEnsembleConfig',
    'EnhancedAdvancedConfig',
    'AdaptiveWeightingConfig',
    'ConfidenceScoringConfig',
    'NativeValidationConfig',
    'DynamicFeatureSelectionConfig',
    'ElbowMethodConfig',
    'StatisticalThresholdingConfig',

    'EnhancedEnsembleAdvancedSelector',
    'EnhancedAdvancedFeatureSelector',
    'AdaptiveWeightingSystem',
    'ConfidenceScoringSystem',
    'NativeValidationFramework',
    'DynamicFeatureSelector',

    # Factory functions
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
    'create_enhanced_multi_stage_rfe'
]
