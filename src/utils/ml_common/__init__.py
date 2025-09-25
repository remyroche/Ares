"""Curated exports for the :mod:`src.utils.ml_common` package.

The previous version of this module attempted to import almost every
submodule eagerly during package initialisation. That behaviour carried a
heavy import-time cost, masked import errors, and listed names in
``__all__`` that were never actually bound. The implementation below
keeps the public surface area explicit while deferring all heavy imports
until the exported symbols are first accessed.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any, Dict, Iterable, Tuple

from .logger import get_ml_logger

_LOGGER = get_ml_logger('package')

_MODELS = "src.utils.ml_common.models"
_ENSEMBLES = "src.utils.ml_common.ensembles"
_EXPLAINABILITY = "src.utils.ml_common.explainability"
_OPTIMIZATION = "src.utils.ml_common.optimization"
_DATA_PROCESSING = "src.utils.ml_common.data_processing"
_VALIDATION = "src.utils.ml_common.validation"
_UTILS = "src.utils.ml_common.utils"
_FS_COMPAT = "src.utils.ml_common.feature_selection_backwards_compat"
_CONFIDENCE = "src.utils.ml_common.confidence_metrics"
_PIPELINE = "src.utils.ml_common.pipeline_orchestrator"
_FEATURE_IMPORTANCE = "src.utils.feature_selection.feature_importance_analyzer"
_DATA_DRIFT = "src.utils.ml_common.data_drift_detector"
_MATRIX_OPERATIONS = "src.utils.matrix_operations"
_HMM = "src.utils.ml_common.hmm_regime_detection"

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {}


def _register(module: str, names: Iterable[str] | Dict[str, str]) -> None:
    if isinstance(names, dict):
        for export, attr in names.items():
            _EXPORT_MAP[export] = (module, attr)
    else:
        for name in names:
            _EXPORT_MAP[name] = (module, name)


_register(
    _MODELS,
    [
        'EnhancedModelFactory',
        'ModelType',
        'ModelConfig',
        'create_model_factory',
        'MultiOutputConfig',
        'MultiOutputModel',
        'MultiOutputStackingModel',
        'MultiOutputResult',
        'prepare_multi_output_targets',
        'create_analyst_outputs',
        'create_tactician_outputs',
        'create_multi_output_stacking_model',
        'EnhancedModelTrainer',
        'train_model_with_confidence_metrics',
        'ModelEvaluator',
        'ModelRegistry',
    ],
)

_register(
    _ENSEMBLES,
    [
        'EnsembleManager',
        'EnsembleType',
        'EnsembleConfig',
        'StackingEnsembleManager',
        'StackingEnsembleConfig',
        'StackingEnsembleResult',
        'create_analyst_ensemble',
        'create_tactician_ensemble',
        'StackingConfidenceCalibrator',
        'StackingCalibrationConfig',
        'StackingCalibrationResult',
        'create_analyst_calibrator',
        'create_tactician_calibrator',
    ],
)

_register(
    _EXPLAINABILITY,
    [
        'ModelExplainer',
        'ModelInterpretabilityEngine',
        'ExplanationResult',
    ],
)

_register(
    _OPTIMIZATION,
    [
        'ParetoOptimizer',
        'ParetoFront',
        'ParetoFrontAnalyzer',
        'RegimeSpecificTPSLOptimizer',
    ],
)

_register(
    _DATA_PROCESSING,
    {
        'EnhancedDataLabelerGetter': 'get_enhanced_data_labeler',
        'LabelingConfigGetter': 'get_labeling_config',
    },
)

_register(
    _VALIDATION,
    [
        'ConfigurationValidator',
        'TemporalCrossValidator',
        'PurgedKFold',
        'CrossValidationUtilities',
        'PurgedSplitConfig',
        'UnifiedCrossValidator',
        'UnifiedCVResult',
        'perform_cross_validation',
        'temporal_cross_validation',
        'nested_cross_validation',
        'StabilityAnalyzer',
        'feature_selection_stability',
        'aggregate_time_blocks',
        'optimize_threshold',
        'calibrate_probabilities',
    ],
)

_register(
    _UTILS,
    [
        'setup_logger',
        'get_logger',
        'MemoryOptimizer',
        'MemoryIntegrator',
        'ParallelProcessor',
        'UnifiedCache',
        'get_unified_cache',
        'cached',
        'SharedMLCache',
        'limit_blas_threads',
        'get_thread_info',
        'validate_thread_environment',
        'LookaheadProtection',
        'MLTrainingSafeguards',
        'RobustErrorHandler',
    ],
)

_register(
    _FS_COMPAT,
    {
        'FeatureSelector': 'FeatureSelector',
        'FeatureSelectionConfig': 'FeatureSelectionConfig',
        'LegacyFeatureSelector': 'FeatureSelector',
    },
)

_register(
    _HMM,
    [
        'HMMRegimeDetector',
        'RegimeConfig',
    ],
)

_register(
    _CONFIDENCE,
    [
        'calculate_confidence_metrics',
        'calculate_calibration_metrics',
    ],
)

_register(
    _MATRIX_OPERATIONS,
    [
        'M1EnhancedMatrixOperations',
        'get_enhanced_matrix_operations',
    ],
)

_register(
    _PIPELINE,
    {'PipelineOrchestrator': 'MLPipelineOrchestrator'},
)

_register(
    _FEATURE_IMPORTANCE,
    [
        'FeatureImportanceAnalyzer',
        'FeatureImportanceConfig',
        'FeatureImportanceResult',
        'ImportanceMethod',
        'analyze_feature_importance',
        'get_important_features',
    ],
)

_register(
    _DATA_DRIFT,
    [
        'DataDriftDetector',
        'DriftDetectionConfig',
        'DriftReport',
        'DriftResult',
        'DriftType',
        'DriftMethod',
        'DriftSeverity',
        'detect_data_drift',
        'get_drifted_features',
    ],
)


def available_exports() -> Tuple[str, ...]:
    """Return a tuple of export names provided by this package."""

    return tuple(sorted(_EXPORT_MAP))


def load_export(name: str) -> Any:
    """Explicit helper for loading a lazily exported symbol."""

    if name not in _EXPORT_MAP:
        raise AttributeError(f"{name!r} is not a recognised export")
    return getattr(sys.modules[__name__], name)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORT_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    try:
        module = import_module(module_name)
        value = getattr(module, attribute)
    except Exception as exc:  # pragma: no cover - defensive path
        _LOGGER.error("Failed to load %s from %s", attribute, module_name, exc_info=exc)
        raise

    globals()[name] = value
    return value


def __dir__() -> Iterable[str]:
    return sorted(set(globals()) | set(_EXPORT_MAP))


__all__ = tuple(sorted(_EXPORT_MAP)) + ('available_exports', 'load_export', 'get_ml_logger')
