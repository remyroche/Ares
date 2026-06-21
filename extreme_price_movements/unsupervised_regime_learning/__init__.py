"""Utilities for unsupervised regime feature selection and transforms."""

from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    UNSUPERVISED_REGIME_LEARNING_DEFAULTS,
    UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
)
from extreme_price_movements.unsupervised_regime_learning.regime_models import (
    AdvancedRegimeLearningArtifact,
    AdvancedRegimeLearningConfig,
    fit_advanced_regime_learning,
    load_advanced_regime_learning_artifact,
    save_advanced_regime_learning_artifact,
)
from extreme_price_movements.unsupervised_regime_learning.regime_hpo import (
    RegimeHPOConfig,
    RegimeHPOResult,
    run_advanced_regime_learning_hpo,
)
from extreme_price_movements.unsupervised_regime_learning.context_features import (
    RegimeContextFeatureConfig,
    build_regime_context_feature_frame,
    build_regime_context_features_from_artifact,
    cross_sectional_regime_residuals,
    generate_signal_regime_interaction_features,
    market_regime_aggregate_features,
)
from extreme_price_movements.unsupervised_regime_learning.lgbm_feature_filter import (
    RegimeFeatureLGBMFilterConfig,
    RegimeFeatureLGBMFilterResult,
    extract_lgbm_reuse_contract,
    select_regime_lgbm_addon_features,
)
from extreme_price_movements.unsupervised_regime_learning.validation import (
    RegimePipelineValidationConfig,
    regime_pipeline_validation_summary,
    validate_regime_learning_artifact,
)

__all__ = [
    "AdvancedRegimeLearningArtifact",
    "AdvancedRegimeLearningConfig",
    "RegimeHPOConfig",
    "RegimeHPOResult",
    "RegimeContextFeatureConfig",
    "RegimeFeatureLGBMFilterConfig",
    "RegimeFeatureLGBMFilterResult",
    "RegimePipelineValidationConfig",
    "UNSUPERVISED_REGIME_LEARNING_DEFAULTS",
    "UNSUPERVISED_REGIME_PRIMITIVE_FEATURES",
    "build_regime_context_feature_frame",
    "build_regime_context_features_from_artifact",
    "cross_sectional_regime_residuals",
    "extract_lgbm_reuse_contract",
    "fit_advanced_regime_learning",
    "generate_signal_regime_interaction_features",
    "load_advanced_regime_learning_artifact",
    "market_regime_aggregate_features",
    "regime_pipeline_validation_summary",
    "run_advanced_regime_learning_hpo",
    "save_advanced_regime_learning_artifact",
    "select_regime_lgbm_addon_features",
    "validate_regime_learning_artifact",
]
