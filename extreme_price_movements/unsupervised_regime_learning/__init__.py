"""Utilities for unsupervised regime feature selection and transforms.

The package root intentionally keeps imports lazy. Live inference often imports
only ``feature_registry`` through this package; eagerly importing regime models
would also import optional UMAP/TensorFlow dependencies and inflate memory.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from extreme_price_movements.unsupervised_regime_learning.feature_registry import (
    UNSUPERVISED_REGIME_LEARNING_DEFAULTS,
    UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
)

_LAZY_EXPORTS = {
    "AdvancedRegimeLearningArtifact": ".regime_models",
    "AdvancedRegimeLearningConfig": ".regime_models",
    "fit_advanced_regime_learning": ".regime_models",
    "load_advanced_regime_learning_artifact": ".regime_models",
    "save_advanced_regime_learning_artifact": ".regime_models",
    "RegimeHPOConfig": ".regime_hpo",
    "RegimeHPOResult": ".regime_hpo",
    "run_advanced_regime_learning_hpo": ".regime_hpo",
    "RegimeContextFeatureConfig": ".context_features",
    "build_regime_context_feature_frame": ".context_features",
    "build_regime_context_features_from_artifact": ".context_features",
    "cross_sectional_regime_residuals": ".context_features",
    "generate_signal_regime_interaction_features": ".context_features",
    "market_regime_aggregate_features": ".context_features",
    "BadRegimeArchetypeFeatureConfig": ".bad_regime_archetypes",
    "build_bad_regime_archetype_feature_frame": ".bad_regime_archetypes",
    "load_bad_regime_archetype_definitions": ".bad_regime_archetypes",
    "RegimeFeatureLGBMFilterConfig": ".lgbm_feature_filter",
    "RegimeFeatureLGBMFilterResult": ".lgbm_feature_filter",
    "extract_lgbm_reuse_contract": ".lgbm_feature_filter",
    "select_regime_lgbm_addon_features": ".lgbm_feature_filter",
    "RegimePipelineValidationConfig": ".validation",
    "regime_pipeline_validation_summary": ".validation",
    "validate_regime_learning_artifact": ".validation",
}

__all__ = [
    "AdvancedRegimeLearningArtifact",
    "AdvancedRegimeLearningConfig",
    "RegimeHPOConfig",
    "RegimeHPOResult",
    "RegimeContextFeatureConfig",
    "BadRegimeArchetypeFeatureConfig",
    "RegimeFeatureLGBMFilterConfig",
    "RegimeFeatureLGBMFilterResult",
    "RegimePipelineValidationConfig",
    "UNSUPERVISED_REGIME_LEARNING_DEFAULTS",
    "UNSUPERVISED_REGIME_PRIMITIVE_FEATURES",
    "build_regime_context_feature_frame",
    "build_regime_context_features_from_artifact",
    "build_bad_regime_archetype_feature_frame",
    "cross_sectional_regime_residuals",
    "extract_lgbm_reuse_contract",
    "fit_advanced_regime_learning",
    "generate_signal_regime_interaction_features",
    "load_advanced_regime_learning_artifact",
    "load_bad_regime_archetype_definitions",
    "market_regime_aggregate_features",
    "regime_pipeline_validation_summary",
    "run_advanced_regime_learning_hpo",
    "save_advanced_regime_learning_artifact",
    "select_regime_lgbm_addon_features",
    "validate_regime_learning_artifact",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
