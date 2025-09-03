from __future__ import annotations

"""
Domain-specific decorators for the trading system.

This module provides all domain-specific decorators built on top of
the core decorator system. It combines decorators from multiple modules
for easy importing.
"""

# Additional composite decorators
from src.core.decorators import cached, compose, handles_errors, traced, validates

# Import all decorators from domain_decorators module
from .decorators import (  # Enums; Data Quality; Monitoring and Performance; Security and Processing; Pipeline Management; Utilities
    PerformanceLevel,
    ValidationLevel,
    create_step_decorator,
    ensure_data_integrity,
    monitor_step_execution,
    prevent_data_leakage,
    quality_gate,
    secure_data_processing,
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection,
    validate_klines_data_quality,
    validate_multi_timeframe_data_quality,
    validate_pipeline_step,
)

# Import all decorators from domain_decorators_extended module
from .decorators_extended import (  # OHLCV and specialized data validation; Step-specific validators; Memory and processing; Feature engineering; Security and monitoring; Artifacts and reproducibility; Caching
    artifact_versioning,
    deterministic_seed,
    monitor_feature_engineering,
    monitor_pipeline_performance,
    optimize_memory_usage,
    secure_step_execution,
    smart_validation_cache,
    validate_feature_engineering_pipeline,
    validate_hmm_data_requirements,
    validate_hmm_regime_discovery,
    validate_ohlcv_data_quality,
    validate_step2_operation,
    validate_step3_5_comprehensive,
    validate_step3_comprehensive,
    validate_step4_comprehensive,
    validate_step5_comprehensive,
    validate_step6_comprehensive,
    validate_step_comprehensive,
    validate_wavelet_data_quality,
)


def comprehensive_validation(
    data_quality: bool = True,
    feature_engineering: bool = True,
    performance_monitoring: bool = True,
    **kwargs,
) -> callable:
    """Apply comprehensive validation combining multiple validators."""
    decorators = []

    if data_quality:
        decorators.append(validate_data_quality(**kwargs))

    if feature_engineering:
        decorators.append(validate_feature_engineering_pipeline())

    if performance_monitoring:
        decorators.append(monitor_step_execution("comprehensive_validation"))

    return compose(*decorators)


def idempotent_step(
    check_existing: bool = True,
    force_rerun: bool = False,
    cache_results: bool = True,
) -> callable:
    """Make a step idempotent with result caching."""
    decorators = [handles_errors()]

    if cache_results:
        decorators.append(cached(ttl=7200))  # 2 hour cache

    decorators.append(traced(name="idempotent_step"))

    return compose(*decorators)


def time_budget_watchdog(
    max_seconds: int = 300,
    warning_threshold: float = 0.8,
    fail_on_timeout: bool = True,
) -> callable:
    """Monitor execution time against a budget."""
    from src.core.decorators import timeout

    decorators = []

    if fail_on_timeout:
        decorators.append(timeout(seconds=max_seconds))

    decorators.append(
        monitor_step_execution("time_budget", performance_level=PerformanceLevel.HIGH)
    )

    return compose(*decorators)


def enforce_ndarray(
    arg_positions: list = None,
    kwargs_names: list = None,
    output: bool = True,
) -> callable:
    """Enforce numpy array types for specified arguments."""
    from functools import wraps

    import numpy as np

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert positional arguments
            if arg_positions:
                args = list(args)
                for pos in arg_positions:
                    if pos < len(args) and not isinstance(args[pos], np.ndarray):
                        args[pos] = np.asarray(args[pos])
                args = tuple(args)

            # Convert keyword arguments
            if kwargs_names:
                for name in kwargs_names:
                    if name in kwargs and not isinstance(kwargs[name], np.ndarray):
                        kwargs[name] = np.asarray(kwargs[name])

            # Execute function
            result = func(*args, **kwargs)

            # Convert output if needed
            if output and result is not None and not isinstance(result, np.ndarray):
                try:
                    result = np.asarray(result)
                except:
                    pass  # If conversion fails, return as is

            return result

        return wrapper

    return decorator


# Classes for pipeline stage management
class PipelineStage:
    """Pipeline stage enumeration."""

    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"


class PipelineValidationLevel:
    """Pipeline validation level enumeration."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    STRICT = "strict"


def validate_pipeline_input(
    stage: str = None,
    level: str = PipelineValidationLevel.STANDARD,
    required_keys: list = None,
) -> callable:
    """Validate pipeline input data."""
    return validate_pipeline_step(
        prerequisites=required_keys,
        stage=stage,
    )


def monitor_pipeline_step(
    stage: str,
    metrics_to_track: list = None,
    alert_on_anomaly: bool = True,
) -> callable:
    """Monitor a specific pipeline step."""
    return monitor_step_execution(
        step_name=f"pipeline.{stage}",
        performance_level=PerformanceLevel.HIGH,
        log_memory=True,
    )


# Aliases for backward compatibility
comprehensive_data_validation = validate_data_quality
validate_constant_features = lambda: validate_data_quality(check_constant=True)
validate_low_variance_features = lambda: validate_data_quality(
    check_constant=True, min_unique_values=3
)
validate_data_completeness = lambda: validate_data_quality(max_null_ratio=0.0)
validate_datetime_index = lambda: validate_data_quality(check_timestamps=True)
validate_memory_optimized_data_quality = lambda: compose(
    validate_data_quality(), optimize_memory_usage()
)
validate_multi_timeframe_alignment = validate_multi_timeframe_data_quality
validate_multi_timeframe_processing = validate_multi_timeframe_data_quality

# File operation validators
validate_file_operation = secure_data_processing
validate_dataframe_operation = validate_data_quality
validate_file_size = lambda max_size_mb=100: validates

# Additional security decorators
secure_file_path = secure_data_processing
sanitize_string = secure_data_processing

# Export all decorators
__all__ = [
    # Enums and Classes
    "ValidationLevel",
    "PerformanceLevel",
    "PipelineStage",
    "PipelineValidationLevel",
    # Data Quality
    "validate_data_quality",
    "validate_feature_engineering_with_lookahead_bias_detection",
    "validate_klines_data_quality",
    "validate_multi_timeframe_data_quality",
    "validate_ohlcv_data_quality",
    "validate_wavelet_data_quality",
    "validate_hmm_data_requirements",
    "validate_hmm_regime_discovery",
    # Step-specific validators
    "validate_step_comprehensive",
    "validate_step2_operation",
    "validate_step3_comprehensive",
    "validate_step3_5_comprehensive",
    "validate_step4_comprehensive",
    "validate_step5_comprehensive",
    "validate_step6_comprehensive",
    # Monitoring and Performance
    "monitor_step_execution",
    "monitor_feature_engineering",
    "monitor_pipeline_performance",
    "monitor_pipeline_step",
    "quality_gate",
    # Security and Processing
    "secure_data_processing",
    "prevent_data_leakage",
    "ensure_data_integrity",
    "secure_step_execution",
    # Pipeline Management
    "validate_pipeline_step",
    "validate_pipeline_input",
    # Memory and Processing
    "optimize_memory_usage",
    "enforce_ndarray",
    # Feature Engineering
    "validate_feature_engineering_pipeline",
    # Artifacts and Reproducibility
    "artifact_versioning",
    "deterministic_seed",
    "idempotent_step",
    # Caching
    "smart_validation_cache",
    # Utilities
    "create_step_decorator",
    "comprehensive_validation",
    "time_budget_watchdog",
    # Backward compatibility aliases
    "comprehensive_data_validation",
    "validate_constant_features",
    "validate_low_variance_features",
    "validate_data_completeness",
    "validate_datetime_index",
    "validate_memory_optimized_data_quality",
    "validate_multi_timeframe_alignment",
    "validate_multi_timeframe_processing",
    "validate_file_operation",
    "validate_dataframe_operation",
    "validate_file_size",
    "secure_file_path",
    "sanitize_string",
]
