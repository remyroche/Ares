"""Lazy and error-aware access to :mod:`src.utils.ml_common` symbols."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from .dependency_management import dependency_manager

__all__ = [
    "ML_COMMON_AVAILABLE",
    "ConfigurationValidator",
    "FeatureSelectionConfig",
    "FeatureSelector",
    "ParetoFront",
    "ParetoFrontAnalyzer",
    "ParetoOptimizer",
    "RegimeSpecificTPSLOptimizer",
    "StabilityAnalyzer",
    "TemporalCrossValidator",
    "UnifiedCrossValidator",
    "nested_cross_validation",
    "perform_cross_validation",
    "temporal_cross_validation",
    "PurgedKFold",
    "EnhancedLearningCurveAnalyzer",
    "LearningCurveAnalysisResult",
    "ModelConfig",
    "ModelType",
    "BayesianTPEConfig",
    "BayesianTPEOptimizer",
    "optimize_with_bayesian_tpe",
    "build_coarse_grid_from_search_space",
    "build_fine_grid_around_best",
    "HyperparameterOptimization",
    "HyperparameterOptimizer",
    "ValidationIntegrationConfig",
    "get_validation_integrator",
    "LookaheadProtection",
    "get_validation_framework",
    "CrossValidationManager",
    "DataLeakageConfig",
    "DataLeakagePrevention",
    "LeakageReport",
    "UniversalOverfittingDetector",
]

logger = logging.getLogger(__name__)

_BASE_MODULE = "src.utils.ml_common"

_MODULE_HINTS: Dict[str, str] = {
    _BASE_MODULE: "Include src/utils/ml_common in the runtime environment",
    "src.utils.ml_common.evaluation.enhanced_learning_curve_analysis": "Install ML common evaluation extras",
    "src.utils.ml_common.models.model_factory": "Install ML common model extras",
    "src.utils.ml_common.optimization.bayesian_tpe_optimizer": "Install ML common optimization extras",
    "src.utils.ml_common.optimization.grid_utils": "Install ML common optimization extras",
    "src.utils.ml_common.optimization.hpo_utils": "Install ML common optimization extras",
    "src.utils.ml_common.training.universal_validation_integration": "Install ML common training extras",
    "src.utils.ml_common.utils.lookahead_protection": "Install ML common utility extras",
    "src.utils.ml_common.validation": "Install ML common validation extras",
    "src.utils.ml_common.validation.cv": "Install ML common validation extras",
    "src.utils.ml_common.validation.data_leakage_prevention": "Install ML common validation extras",
    "src.utils.ml_common.validation.enhanced_overfitting_detection": "Install ML common validation extras",
}

# Track which modules failed to load so we only log once.
_MISSING_MODULES: set[str] = set()


def _log_missing(module: str, name: str) -> None:
    if module not in _MISSING_MODULES:
        hint = _MODULE_HINTS.get(module)
        if hint:
            logger.warning("Optional ML common module '%s' missing – %s.", module, hint)
        else:
            logger.warning("Optional ML common module '%s' missing.", module)
        _MISSING_MODULES.add(module)
    logger.error("Cannot access '%s' because its source module '%s' is unavailable.", name, module)


def _load_module(module: str):
    return dependency_manager.import_optional(module, install_hint=_MODULE_HINTS.get(module))


def _missing_callable(module: str, name: str) -> Callable[..., Any]:
    def _raiser(*_args: Any, **_kwargs: Any) -> Any:
        _log_missing(module, name)
        raise RuntimeError(
            f"Optional ML common dependency '{module}.{name}' is unavailable. "
            "Install the ML common extras to enable this functionality."
        )

    _raiser.__name__ = name
    return _raiser


def _missing_class(module: str, name: str):
    class _Missing:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _log_missing(module, name)
            raise RuntimeError(
                f"Optional ML common dependency '{module}.{name}' is unavailable. "
                "Install the ML common extras to enable this functionality."
            )

        def __repr__(self) -> str:  # pragma: no cover - represent missing dependency
            return f"<Unavailable ML common class {module}.{name}>"

    _Missing.__name__ = name
    return _Missing


def _export(module: str, name: str, kind: str):
    mod = _load_module(module)
    if mod is not None:
        attr = getattr(mod, name, None)
        if attr is not None:
            return attr
    return _missing_class(module, name) if kind == "class" else _missing_callable(module, name)


ML_COMMON_AVAILABLE = dependency_manager.import_optional(_BASE_MODULE, install_hint=_MODULE_HINTS[_BASE_MODULE]) is not None

ConfigurationValidator = _export(_BASE_MODULE, "ConfigurationValidator", "class")
FeatureSelectionConfig = _export(_BASE_MODULE, "FeatureSelectionConfig", "class")
FeatureSelector = _export(_BASE_MODULE, "FeatureSelector", "class")
ParetoFront = _export(_BASE_MODULE, "ParetoFront", "class")
ParetoFrontAnalyzer = _export(_BASE_MODULE, "ParetoFrontAnalyzer", "class")
ParetoOptimizer = _export(_BASE_MODULE, "ParetoOptimizer", "class")
RegimeSpecificTPSLOptimizer = _export(_BASE_MODULE, "RegimeSpecificTPSLOptimizer", "class")
StabilityAnalyzer = _export(_BASE_MODULE, "StabilityAnalyzer", "class")
TemporalCrossValidator = _export(_BASE_MODULE, "TemporalCrossValidator", "class")
UnifiedCrossValidator = _export(_BASE_MODULE, "UnifiedCrossValidator", "class")
nested_cross_validation = _export(_BASE_MODULE, "nested_cross_validation", "function")
perform_cross_validation = _export(_BASE_MODULE, "perform_cross_validation", "function")
temporal_cross_validation = _export(_BASE_MODULE, "temporal_cross_validation", "function")
PurgedKFold = _export(_BASE_MODULE, "PurgedKFold", "class")

EnhancedLearningCurveAnalyzer = _export(
    "src.utils.ml_common.evaluation.enhanced_learning_curve_analysis",
    "EnhancedLearningCurveAnalyzer",
    "class",
)
LearningCurveAnalysisResult = _export(
    "src.utils.ml_common.evaluation.enhanced_learning_curve_analysis",
    "LearningCurveAnalysisResult",
    "class",
)

ModelConfig = _export("src.utils.ml_common.models.model_factory", "ModelConfig", "class")
ModelType = _export("src.utils.ml_common.models.model_factory", "ModelType", "class")

BayesianTPEConfig = _export(
    "src.utils.ml_common.optimization.bayesian_tpe_optimizer",
    "BayesianTPEConfig",
    "class",
)
BayesianTPEOptimizer = _export(
    "src.utils.ml_common.optimization.bayesian_tpe_optimizer",
    "BayesianTPEOptimizer",
    "class",
)
optimize_with_bayesian_tpe = _export(
    "src.utils.ml_common.optimization.bayesian_tpe_optimizer",
    "optimize_with_bayesian_tpe",
    "function",
)

build_coarse_grid_from_search_space = _export(
    "src.utils.ml_common.optimization.grid_utils",
    "build_coarse_grid_from_search_space",
    "function",
)
build_fine_grid_around_best = _export(
    "src.utils.ml_common.optimization.grid_utils",
    "build_fine_grid_around_best",
    "function",
)

HyperparameterOptimization = _export(
    "src.utils.ml_common.optimization.hpo_utils",
    "HyperparameterOptimization",
    "class",
)
HyperparameterOptimizer = _export(
    "src.utils.ml_common.optimization.hpo_utils",
    "HyperparameterOptimizer",
    "class",
)

ValidationIntegrationConfig = _export(
    "src.utils.ml_common.training.universal_validation_integration",
    "ValidationIntegrationConfig",
    "class",
)
get_validation_integrator = _export(
    "src.utils.ml_common.training.universal_validation_integration",
    "get_validation_integrator",
    "function",
)

LookaheadProtection = _export(
    "src.utils.ml_common.utils.lookahead_protection",
    "LookaheadProtection",
    "class",
)

get_validation_framework = _export(
    "src.utils.ml_common.validation",
    "get_validation_framework",
    "function",
)

CrossValidationManager = _export("src.utils.ml_common.validation.cv", "CrossValidationManager", "class")

DataLeakageConfig = _export(
    "src.utils.ml_common.validation.data_leakage_prevention",
    "DataLeakageConfig",
    "class",
)
DataLeakagePrevention = _export(
    "src.utils.ml_common.validation.data_leakage_prevention",
    "DataLeakagePrevention",
    "class",
)
LeakageReport = _export(
    "src.utils.ml_common.validation.data_leakage_prevention",
    "LeakageReport",
    "class",
)

UniversalOverfittingDetector = _export(
    "src.utils.ml_common.validation.enhanced_overfitting_detection",
    "UniversalOverfittingDetector",
    "class",
)
