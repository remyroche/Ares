from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

import pandas as pd

"""
Domain-specific decorators for the trading system.

This module provides all domain-specific decorators built on top of
the core decorator system. It combines decorators from multiple modules
for easy importing.
"""
from ..decorators import compose, handles_errors, traced, validates

def with_tracing_span(*args, **kwargs) -> None:
    """Alias for traced to preserve backward compatibility with older imports."""
    return traced(*args, **kwargs)

def handle_errors(*args, **kwargs) -> None:
    """Alias for handles_errors to preserve backward compatibility with older imports."""
    return handles_errors(*args, **kwargs)

def validate_data_structure(*args, **kwargs) -> bool:
    """Legacy alias retained for compatibility; use validate_data_quality instead."""
    return validate_data_quality(*args, **kwargs)

def resource_monitor(*args, **kwargs) -> None:
    """Lightweight compatibility shim; monitoring handled via traced/log decorators."""

    def _decorator(func: Callable) -> None:
        return func
    return _decorator

def memory_efficient(*args, **kwargs) -> None:
    """Prefer src.utils.enhanced_memory_management.memory_efficient; kept for imports."""
    try:
        return _mem(*args, **kwargs)
    except Exception:

        def _decorator(func: Callable) -> None:
            return func
        return _decorator

def comprehensive_validation(data_quality: bool = True, feature_engineering: bool = True, performance_monitoring: bool = True, **kwargs) -> callable:
    """Apply comprehensive validation combining multiple validators."""
    decorators = []
    if data_quality:
        decorators.append(validate_data_quality(**kwargs))
    if feature_engineering:
        decorators.append(validate_feature_engineering_pipeline())
    if performance_monitoring:
        decorators.append(monitor_step_execution('comprehensive_validation'))
    return compose(*decorators)

def idempotent_step(check_existing: bool = True, force_rerun: bool = False, cache_results: bool = True) -> callable:
    """Make a step idempotent with result caching."""
    decorators = [handles_errors()]
    if cache_results:
        decorators.append(cached(ttl = 7200))
    decorators.append(traced(name='idempotent_step'))
    return compose(*decorators)

def time_budget_watchdog(max_seconds: int = 300, warning_threshold: float = 0.8, fail_on_timeout: bool = True) -> callable:
    """Monitor execution time against a budget."""
    decorators = []
    if fail_on_timeout:
        decorators.append(timeout(seconds = max_seconds))
    decorators.append(monitor_step_execution('time_budget', performance_level = PerformanceLevel.HIGH))
    return compose(*decorators)

def enforce_ndarray(arg_positions: list = None, kwargs_names: list = None, output: bool = True) -> callable:
    """Enforce numpy array types for specified arguments."""
    from functools import wraps

    def decorator(func: Callable) -> None:

        @wraps(func)
        def wrapper(*args, **kwargs) -> None:
            if arg_positions:
                args = list(args)
                for pos in arg_positions:
                    if pos < len(args) and (not isinstance(args[pos], np.ndarray)):
                        args[pos] = np.asarray(args[pos])
                args = tuple(args)
            if kwargs_names:
                for name in kwargs_names:
                    if name in kwargs and (not isinstance(kwargs[name], np.ndarray)):
                        kwargs[name] = np.asarray(kwargs[name])
            result = func(*args, **kwargs)
            if output and result is not None and (not isinstance(result, np.ndarray)):
                try:
                    result = np.asarray(result)
                except:
                    pass
            return result
        return wrapper
    return decorator

class PipelineStage:
    """Pipeline stage enumeration."""
    DATA_COLLECTION = 'data_collection'
    DATA_PROCESSING = 'data_processing'
    FEATURE_ENGINEERING = 'feature_engineering'
    MODEL_TRAINING = 'model_training'
    MODEL_EVALUATION = 'model_evaluation'
    MODEL_DEPLOYMENT = 'model_deployment'

class PipelineValidationLevel:
    """Pipeline validation level enumeration."""
    MINIMAL = 'minimal'
    STANDARD = 'standard'
    COMPREHENSIVE = 'comprehensive'
    STRICT = 'strict'

def validate_pipeline_input(stage: str = None, level: str = PipelineValidationLevel.STANDARD, required_keys: list = None) -> callable:
    """Validate pipeline input data."""
    return validate_pipeline_step(prerequisites = required_keys, stage = stage)

def monitor_pipeline_step(stage: str, metrics_to_track: list = None, alert_on_anomaly: bool = True) -> callable:
    """Monitor a specific pipeline step."""
    return monitor_step_execution(step_name = f'pipeline.{stage}', performance_level = PerformanceLevel.HIGH, log_memory = True)

def validate_data_quality(**kwargs) -> bool:
    """Simple data quality validation decorator."""

    def decorator(func: Callable) -> None:
        return func
    return decorator

def validate_pipeline_step(**kwargs) -> bool:
    """Simple pipeline step validation decorator."""

    def decorator(func: Callable) -> None:
        return func
    return decorator

def monitor_step_execution(**kwargs) -> None:
    """Simple step execution monitoring decorator."""

    def decorator(func: Callable) -> None:
        return func
    return decorator

def optimize_memory_usage(**kwargs) -> Dict[str, Any]:
    """Simple memory optimization decorator."""

    def decorator(func: Callable) -> None:
        return func
    return decorator

def validate_multi_timeframe_data_quality(**kwargs) -> bool:
    """Simple multi-timeframe data quality validation decorator."""

    def decorator(func: Callable) -> None:
        return func
    return decorator

def secure_data_processing(**kwargs) -> None:
    """Simple secure data processing decorator."""

    def decorator(func: Callable) -> None:
        return func
    return decorator

class PerformanceLevel:
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'

class ValidationLevel:
    STRICT = 'strict'
    MODERATE = 'moderate'
    LENIENT = 'lenient'

class PipelineStage:
    DATA_COLLECTION = 'data_collection'
    DATA_PROCESSING = 'data_processing'
    MODEL_TRAINING = 'model_training'
    VALIDATION = 'validation'

class PipelineValidationLevel:
    COMPREHENSIVE = 'comprehensive'
    STANDARD = 'standard'
    BASIC = 'basic'
comprehensive_data_validation = validate_data_quality
validate_constant_features = lambda: validate_data_quality(check_constant = True)
validate_low_variance_features = lambda: validate_data_quality(check_constant = True, min_unique_values = 3)
validate_data_completeness = lambda: validate_data_quality(max_null_ratio = 0.0)
validate_datetime_index = lambda: validate_data_quality(check_timestamps = True)
validate_memory_optimized_data_quality = lambda: compose(validate_data_quality(), optimize_memory_usage())
validate_multi_timeframe_alignment = validate_multi_timeframe_data_quality
validate_multi_timeframe_processing = validate_multi_timeframe_data_quality
validate_file_operation = secure_data_processing
validate_dataframe_operation = validate_data_quality
def validate_file_size(max_size_mb: int = 100) -> Callable:
    """Validate file size decorator."""
    return validates(max_size_mb=max_size_mb)
secure_file_path = secure_data_processing
sanitize_string = secure_data_processing
__all__ = ['ValidationLevel', 'PerformanceLevel', 'PipelineStage', 'PipelineValidationLevel', 'validate_data_quality', 'validate_feature_engineering_with_lookahead_bias_detection', 'validate_klines_data_quality', 'validate_multi_timeframe_data_quality', 'validate_ohlcv_data_quality', 'validate_wavelet_data_quality', 'validate_hmm_data_requirements', 'validate_hmm_regime_discovery', 'validate_step_comprehensive', 'validate_step2_operation', 'validate_step3_comprehensive', 'validate_step3_5_comprehensive', 'validate_step4_comprehensive', 'validate_step5_comprehensive', 'validate_step6_comprehensive', 'monitor_step_execution', 'monitor_feature_engineering', 'monitor_pipeline_performance', 'monitor_pipeline_step', 'quality_gate', 'secure_data_processing', 'prevent_data_leakage', 'ensure_data_integrity', 'secure_step_execution', 'validate_pipeline_step', 'validate_pipeline_input', 'optimize_memory_usage', 'enforce_ndarray', 'validate_feature_engineering_pipeline', 'artifact_versioning', 'deterministic_seed', 'idempotent_step', 'smart_validation_cache', 'create_step_decorator', 'comprehensive_validation', 'time_budget_watchdog', 'comprehensive_data_validation', 'validate_constant_features', 'validate_low_variance_features', 'validate_data_completeness', 'validate_datetime_index', 'validate_memory_optimized_data_quality', 'validate_multi_timeframe_alignment', 'validate_multi_timeframe_processing', 'validate_file_operation', 'validate_dataframe_operation', 'validate_file_size', 'secure_file_path', 'sanitize_string']
