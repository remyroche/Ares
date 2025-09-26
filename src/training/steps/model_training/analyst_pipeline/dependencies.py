"""Dependency loading helpers for the analyst training pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, Iterable, MutableMapping, Optional

from ..import_helpers import (
    DependencyNotFoundError,
    ensure_dependencies,
    import_module_safely,
    load_module_attributes,
)

logger = logging.getLogger(__name__)

COMMON_OPERATION_NAMES: tuple[str, ...] = (
    "get_m1_gpu_manager",
    "get_m1_memory_optimizer",
    "get_m1_cpu_optimizer",
    "integrate_with_m1_optimizers",
    "cleanup_m1_optimizers",
    "safe_divide",
    "safe_log",
    "safe_sqrt",
    "safe_power",
    "safe_mean",
    "safe_std",
    "validate_finite",
    "validate_positive",
    "validate_range",
    "safe_kelly_calculation",
    "safe_weighted_average",
    "safe_percentage_change",
    "ensure_directory",
    "safe_file_exists",
    "safe_json_dump",
    "safe_json_load",
    "create_empty_dataframe",
    "validate_dataframe",
    "validate_dataframe_columns",
    "safe_dataframe_operation",
    "safe_fillna",
    "safe_convert_dtypes",
    "safe_merge_dataframes",
    "safe_drop_columns",
    "safe_rename_columns",
    "validate_timestamp_column",
    "safe_timestamp_conversion",
    "optimize_dataframe_dtypes",
    "calculate_data_quality_metrics",
    "get_dataframe_info",
    "create_data_quality_report",
    "safe_rolling",
    "safe_groupby_operation",
    "safe_apply_function",
    "safe_filter_dataframe",
    "create_summary_statistics",
    "safe_to_parquet",
    "safe_read_parquet",
    "list_parquet_files",
    "safe_copy",
    "validate_dataframe_schema",
    "validate_file_size",
    "guard_dataframe_nulls",
    "secure_file_path",
    "with_tracing_span",
    "sanitize_string",
    "memory_checkpoint",
    "gpu_context",
    "optimize_memory",
    "get_memory_usage",
    "validate_file_path",
    "get_file_size",
    "check_disk_space",
    "timed_operation",
    "format_bytes",
    "chunked_iterable",
    "parallel_map",
    "validate_correlation_matrix",
    "safe_matrix_inverse",
    "math_safe",
    "MathValidationError",
)

TPRINT_NAMES: tuple[str, ...] = (
    "tprint",
    "tprint_info",
    "tprint_warning",
    "tprint_error",
    "tprint_success",
    "tprint_debug",
    "tprint_progress",
    "tprint_performance",
    "tprint_structured",
    "tprint_timer",
    "tprint_logged",
    "LogLevel",
)

MATH_VALIDATION_NAMES: tuple[str, ...] = (
    "safe_divide",
    "safe_log",
    "safe_sqrt",
    "safe_power",
    "validate_finite",
    "validate_positive",
    "validate_range",
    "safe_kelly_calculation",
    "safe_weighted_average",
    "safe_percentage_change",
    "safe_correlation",
    "safe_covariance",
    "safe_mean",
    "safe_std",
    "safe_percentile",
    "validate_correlation_matrix",
    "safe_matrix_inverse",
    "math_safe",
    "MathValidation",
    "MathValidationError",
)

SERIALIZATION_NAMES: tuple[str, ...] = (
    "JSONSerializer",
    "PickleSerializer",
    "ParquetSerializer",
    "UniversalSerializer",
)

HARDWARE_NAMES: tuple[str, ...] = (
    "get_m1_gpu_manager",
    "is_m1_available",
    "is_mps_available",
    "optimize_dataframe_for_m1",
    "create_m1_optimized_array",
    "m1_backtesting_simulate",
    "m1_monte_carlo_simulate",
    "get_m1_memory_optimizer",
    "get_m1_cpu_optimizer",
)

ML_VALIDATION_NAMES: tuple[str, ...] = (
    "validate_input_data",
    "validate_model_config",
    "validate_training_data",
)

ML_HPO_NAMES: tuple[str, ...] = (
    "optimize_hyperparameters",
    "create_search_space",
    "validate_hpo_config",
)

ML_EVALUATION_NAMES: tuple[str, ...] = (
    "calculate_metrics",
    "evaluate_model_performance",
    "create_evaluation_report",
)

ML_REPORTING_NAMES: tuple[str, ...] = (
    "ReportGenerator",
    "ReportManager",
    "create_training_report",
)

ERROR_MONITOR_NAMES: tuple[str, ...] = (
    "EnhancedErrorDetector",
    "ErrorDetector",
    "ErrorHandler",
    "ErrorReporter",
)

ENHANCED_HPO_NAMES: tuple[str, ...] = (
    "enhance_existing_hpo_pipeline",
    "EnhancedCVStrategies",
    "RegimeType",
    "RegimeCharacteristics",
)

ENHANCED_TRAINING_NAMES: tuple[str, ...] = (
    "EnhancedTrainingUtils",
    "EarlyStoppingConfig",
    "PurgedCVConfig",
    "OverfittingMonitorConfig",
    "RegularizationConfig",
)

TRAINING_INTEGRATION_NAMES: tuple[str, ...] = (
    "TrainingStepEnhancer",
    "TrainingIntegrationConfig",
)


class _UnavailableDependency:
    """Placeholder object that raises informative errors when used."""

    def __init__(self, name: str, module_name: str):
        self._name = name
        self._module_name = module_name

    def __call__(self, *args, **kwargs):
        raise DependencyNotFoundError(
            f"'{self._name}' requires module '{self._module_name}' which is not installed."
        )

    def __getattr__(self, item: str):
        raise DependencyNotFoundError(
            f"'{self._name}.{item}' requires module '{self._module_name}' which is not installed."
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<UnavailableDependency {self._name} from {self._module_name}>"


@dataclass
class AnalystDependencyBundle:
    """Container for dependency availability state."""

    numpy: ModuleType
    pandas: ModuleType
    psutil: ModuleType
    required: Dict[str, bool] = field(default_factory=dict)
    availability: Dict[str, bool] = field(default_factory=dict)

    def ensure_critical(self) -> None:
        """Raise an informative error if required dependencies are missing."""

        ensure_dependencies(self.required, error_message="Critical dependencies missing")


def _inject_required(
    namespace: MutableMapping[str, object],
    module_name: str,
    attr_names: Iterable[str],
    *,
    package_hint: Optional[str] = None,
) -> ModuleType:
    module = import_module_safely(module_name, required=True, package_hint=package_hint)
    attributes, missing = load_module_attributes(module, attr_names, module_name)
    if missing:
        details = ", ".join(sorted(missing))
        raise DependencyNotFoundError(
            f"Module '{module_name}' is missing required attributes: {details}"
        )
    namespace.update(attributes)
    return module


def _inject_optional(
    namespace: MutableMapping[str, object],
    module_name: str,
    attr_names: Iterable[str],
    availability: Dict[str, bool],
    key: str,
    *,
    package_hint: Optional[str] = None,
) -> Optional[ModuleType]:
    module = import_module_safely(module_name, required=False, package_hint=package_hint)
    if module is None:
        availability[key] = False
        for name in attr_names:
            namespace[name] = _UnavailableDependency(name, module_name)
        logger.warning("Optional dependency '%s' is not available", module_name)
        return None

    attributes, missing = load_module_attributes(module, attr_names, module_name)
    if missing:
        details = ", ".join(sorted(missing))
        raise DependencyNotFoundError(
            f"Module '{module_name}' is missing required attributes: {details}"
        )

    namespace.update(attributes)
    availability[key] = True
    return module


def load_dependency_bundle(
    namespace: MutableMapping[str, object],
    *,
    log: Optional[logging.Logger] = None,
) -> AnalystDependencyBundle:
    """Load the dependencies required by the analyst training pipeline."""

    logger_to_use = log or logger

    numpy_module = import_module_safely("numpy", required=True, package_hint="pip install numpy")
    pandas_module = import_module_safely("pandas", required=True, package_hint="pip install pandas")
    psutil_module = import_module_safely("psutil", required=True, package_hint="pip install psutil")

    required = {
        "numpy": numpy_module is not None,
        "pandas": pandas_module is not None,
        "psutil": psutil_module is not None,
    }

    availability: Dict[str, bool] = {}

    _inject_required(namespace, "src.utils.tprint", TPRINT_NAMES)
    _inject_required(namespace, "src.utils.common_operations", COMMON_OPERATION_NAMES)
    _inject_required(namespace, "src.utils.math_validation", MATH_VALIDATION_NAMES)
    _inject_required(namespace, "src.utils.serialization_utils", SERIALIZATION_NAMES)
    _inject_required(namespace, "src.utils.hardware.m1_gpu_utils", HARDWARE_NAMES)
    _inject_required(namespace, "src.utils.ml_common.validation.validation_utils", ML_VALIDATION_NAMES)
    _inject_required(namespace, "src.utils.ml_common.optimization.hpo_utils", ML_HPO_NAMES)
    _inject_required(namespace, "src.utils.ml_common.evaluation.evaluation_utils", ML_EVALUATION_NAMES)
    _inject_required(namespace, "src.utils.ml_common.reporting.enhanced_reporting_system", ML_REPORTING_NAMES)
    _inject_optional(
        namespace,
        "src.utils.ml_common.monitoring.enhanced_error_detector",
        ERROR_MONITOR_NAMES,
        availability,
        key="error_monitoring",
    )

    if availability.get("error_monitoring"):
        logger_to_use.debug("Enhanced error monitoring utilities available")

    enhanced_hpo_module = _inject_optional(
        namespace,
        "src.training.steps.model_training.enhanced_regime_aware_hpo",
        ENHANCED_HPO_NAMES,
        availability,
        key="enhanced_hpo",
    )

    enhanced_training_module = _inject_optional(
        namespace,
        "src.utils.ml_common.training.enhanced_training_utils",
        ENHANCED_TRAINING_NAMES,
        availability,
        key="enhanced_training",
    )

    training_integration_module = _inject_optional(
        namespace,
        "src.utils.ml_common.training.training_integration",
        TRAINING_INTEGRATION_NAMES,
        availability,
        key="training_integration",
    )

    if enhanced_training_module is None or training_integration_module is None:
        availability["enhanced_training"] = False
        availability["training_integration"] = False

    availability.setdefault("enhanced_hpo", enhanced_hpo_module is not None)
    availability.setdefault("common_utilities", True)
    availability.setdefault("math_validation", True)
    availability.setdefault("serialization_utilities", True)
    availability.setdefault("hardware_utilities", True)
    availability.setdefault("ml_utilities", True)

    bundle = AnalystDependencyBundle(
        numpy=numpy_module,
        pandas=pandas_module,
        psutil=psutil_module,
        required=required,
        availability=availability,
    )

    bundle.ensure_critical()
    return bundle


__all__ = ["AnalystDependencyBundle", "load_dependency_bundle"]
