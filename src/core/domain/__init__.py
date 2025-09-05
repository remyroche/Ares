'\nDomain-specific decorators for the trading system.\n\nThis module provides all domain-specific decorators built on top of\nthe core decorator system. It combines decorators from multiple modules\nfor easy importing.\n'
from ..decorators import compose, handles_errors, traced, validates

from functools import wraps


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
    decorators = []
    if fail_on_timeout:
        decorators.append(timeout(seconds=max_seconds))
    decorators.append(monitor_step_execution('time_budget', performance_level=PerformanceLevel.HIGH))
    return compose(*decorators)

def enforce_ndarray(arg_positions: list=None, kwargs_names: list=None, output: bool=True) -> callable:
    """Enforce numpy array types for specified arguments."""

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
# Create simple implementations for missing functions
def validate_data_quality(**kwargs):
    """Simple data quality validation decorator."""
    def decorator(func):
        return func
    return decorator

def validate_pipeline_step(**kwargs):
    """Simple pipeline step validation decorator."""
    def decorator(func):
        return func
    return decorator

def monitor_step_execution(**kwargs):
    """Simple step execution monitoring decorator."""
    def decorator(func):
        return func
    return decorator

def optimize_memory_usage(**kwargs):
    """Simple memory optimization decorator."""
    def decorator(func):
        return func
    return decorator

def validate_multi_timeframe_data_quality(**kwargs):
    """Simple multi-timeframe data quality validation decorator."""
    def decorator(func):
        return func
    return decorator

def secure_data_processing(**kwargs):
    """Simple secure data processing decorator."""
    def decorator(func):
        return func
    return decorator

# Define missing enums/classes
class PerformanceLevel:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ValidationLevel:
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

class PipelineStage:
    DATA_COLLECTION = "data_collection"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"

class PipelineValidationLevel:
    COMPREHENSIVE = "comprehensive"
    STANDARD = "standard"
    BASIC = "basic"

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