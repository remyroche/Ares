from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
'\nDomain-specific decorators for the trading system.\n\nThis module provides all domain-specific decorators built on top of\nthe core decorator system. It combines decorators from multiple modules\nfor easy importing.\n'
from src.core.decorators import cached as cached_src_core_decorators
from src.core.decorators import compose, handles_errors, traced, validates
from .decorators import PerformanceLevel, ValidationLevel, create_step_decorator, ensure_data_integrity, monitor_step_execution, prevent_data_leakage, quality_gate, secure_data_processing, validate_data_quality, validate_feature_engineering_with_lookahead_bias_detection, validate_klines_data_quality, validate_multi_timeframe_data_quality, validate_pipeline_step
from .decorators_extended import artifact_versioning, deterministic_seed, monitor_feature_engineering, monitor_pipeline_performance, optimize_memory_usage, secure_step_execution, smart_validation_cache, validate_feature_engineering_pipeline, validate_hmm_data_requirements, validate_hmm_regime_discovery, validate_ohlcv_data_quality, validate_step2_operation, validate_step3_5_comprehensive, validate_step3_comprehensive, validate_step4_comprehensive, validate_step5_comprehensive, validate_step6_comprehensive, validate_step_comprehensive, validate_wavelet_data_quality

# Backward-compatibility aliases for older code references
def with_tracing_span(*args, **kwargs):
    """Alias for traced to preserve backward compatibility with older imports."""
    return traced(*args, **kwargs)

def handle_errors(*args, **kwargs):
    """Alias for handles_errors to preserve backward compatibility with older imports."""
    return handles_errors(*args, **kwargs)

# No-ops or thin wrappers for legacy names referenced in some modules
def validate_data_structure(*args, **kwargs):
    """Legacy alias retained for compatibility; use validate_data_quality instead."""
    return validate_data_quality(*args, **kwargs)

def resource_monitor(*args, **kwargs):  # pragma: no cover - compatibility shim
    """Lightweight compatibility shim; monitoring handled via traced/log decorators."""
    def _decorator(func):
        return func
    return _decorator

def memory_efficient(*args, **kwargs):  # pragma: no cover - compatibility shim
    """Prefer src.utils.enhanced_memory_management.memory_efficient; kept for imports."""
    try:
        from src.utils.enhanced_memory_management import memory_efficient as _mem
        return _mem(*args, **kwargs)
    except Exception:
        def _decorator(func):
            return func
        return _decorator

def comprehensive_validation(data_quality: bool=True, feature_engineering: bool=True, performance_monitoring: bool=True, **kwargs) -> callable:
    """Apply comprehensive validation combining multiple validators."""
    decorators = []
    if data_quality:
        decorators.append(validate_data_quality(**kwargs))
    if feature_engineering:
        decorators.append(validate_feature_engineering_pipeline())
    if performance_monitoring:
        decorators.append(monitor_step_execution('comprehensive_validation'))
    return compose(*decorators)

def idempotent_step(check_existing: bool=True, force_rerun: bool=False, cache_results: bool=True) -> callable:
    """Make a step idempotent with result caching."""
    decorators = [handles_errors()]
    if cache_results:
        decorators.append(cached(ttl=7200))
    decorators.append(traced(name='idempotent_step'))
    return compose(*decorators)

def time_budget_watchdog(max_seconds: int=300, warning_threshold: float=0.8, fail_on_timeout: bool=True) -> callable:
    """Monitor execution time against a budget."""
    from src.core.decorators import timeout as timeout_src_core_decorators
    decorators = []
    if fail_on_timeout:
        decorators.append(timeout(seconds=max_seconds))
    decorators.append(monitor_step_execution('time_budget', performance_level=PerformanceLevel.HIGH))
    return compose(*decorators)

def enforce_ndarray(arg_positions: list=None, kwargs_names: list=None, output: bool=True) -> callable:
    """Enforce numpy array types for specified arguments."""
    from functools import wraps
    import numpy as np

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

def validate_pipeline_input(stage: str=None, level: str=PipelineValidationLevel.STANDARD, required_keys: list=None) -> callable:
    """Validate pipeline input data."""
    return validate_pipeline_step(prerequisites=required_keys, stage=stage)

def monitor_pipeline_step(stage: str, metrics_to_track: list=None, alert_on_anomaly: bool=True) -> callable:
    """Monitor a specific pipeline step."""
    return monitor_step_execution(step_name=f'pipeline.{stage}', performance_level=PerformanceLevel.HIGH, log_memory=True)
comprehensive_data_validation = validate_data_quality
validate_constant_features = lambda: validate_data_quality(check_constant=True)
validate_low_variance_features = lambda: validate_data_quality(check_constant=True, min_unique_values=3)
validate_data_completeness = lambda: validate_data_quality(max_null_ratio=0.0)
validate_datetime_index = lambda: validate_data_quality(check_timestamps=True)
validate_memory_optimized_data_quality = lambda: compose(validate_data_quality(), optimize_memory_usage())
validate_multi_timeframe_alignment = validate_multi_timeframe_data_quality
validate_multi_timeframe_processing = validate_multi_timeframe_data_quality
validate_file_operation = secure_data_processing
validate_dataframe_operation = validate_data_quality
validate_file_size = lambda max_size_mb=100: validates
secure_file_path = secure_data_processing
sanitize_string = secure_data_processing
__all__ = ['ValidationLevel', 'PerformanceLevel', 'PipelineStage', 'PipelineValidationLevel', 'validate_data_quality', 'validate_feature_engineering_with_lookahead_bias_detection', 'validate_klines_data_quality', 'validate_multi_timeframe_data_quality', 'validate_ohlcv_data_quality', 'validate_wavelet_data_quality', 'validate_hmm_data_requirements', 'validate_hmm_regime_discovery', 'validate_step_comprehensive', 'validate_step2_operation', 'validate_step3_comprehensive', 'validate_step3_5_comprehensive', 'validate_step4_comprehensive', 'validate_step5_comprehensive', 'validate_step6_comprehensive', 'monitor_step_execution', 'monitor_feature_engineering', 'monitor_pipeline_performance', 'monitor_pipeline_step', 'quality_gate', 'secure_data_processing', 'prevent_data_leakage', 'ensure_data_integrity', 'secure_step_execution', 'validate_pipeline_step', 'validate_pipeline_input', 'optimize_memory_usage', 'enforce_ndarray', 'validate_feature_engineering_pipeline', 'artifact_versioning', 'deterministic_seed', 'idempotent_step', 'smart_validation_cache', 'create_step_decorator', 'comprehensive_validation', 'time_budget_watchdog', 'comprehensive_data_validation', 'validate_constant_features', 'validate_low_variance_features', 'validate_data_completeness', 'validate_datetime_index', 'validate_memory_optimized_data_quality', 'validate_multi_timeframe_alignment', 'validate_multi_timeframe_processing', 'validate_file_operation', 'validate_dataframe_operation', 'validate_file_size', 'secure_file_path', 'sanitize_string']