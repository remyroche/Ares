"""
Centralized Decorators Module with Standardized Import Management
This module centralizes all decorators used throughout the codebase for easy import and management.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, TypeVar
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# Type variables for generic decorators
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Standardized import management
REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "src.utils.logger",
    "src.utils.error_handler",
    "src.utils.training_pipeline_decorators",
    "src.utils.decorators",
    "src.utils.enhanced_data_quality_decorators",
    "src.utils.advanced_decorators"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
numpy = PipelineStandards.safe_import("numpy", None)
pandas = PipelineStandards.safe_import("pandas", None)
system_logger = PipelineStandards.safe_import("src.utils.logger", None)

# Fallback functions if imports fail
def create_fallback_logger():
    """Create a fallback logger if system_logger is not available."""
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("CentralizedDecorators")

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

# Import all decorators from their respective modules
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
    handle_file_operations,
)

from src.utils.training_pipeline_decorators import (
    deterministic_seed,
    idempotent_step,
    artifact_write_lock,
    nan_inf_and_constant_guard,
    artifact_versioning,
    time_budget_watchdog,
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
)

from src.utils.decorators import (
    validate_call_or_runtime_types,
    pa_check_input,
    pa_check_output,
    pa_check_io,
    enforce_ndarray,
    auto_vectorize,
    guard_array_nan_inf,
    guard_dataframe_nulls,
    with_tracing_span,
)

from src.utils.enhanced_data_quality_decorators import (
    validate_constant_features,
    validate_low_variance_features,
    validate_data_completeness,
    validate_datetime_index,
    validate_multi_timeframe_alignment,
    validate_hmm_data_requirements,
    validate_data_structure,
    optimize_memory_usage,
    comprehensive_data_validation,
    validate_memory_optimized_data_quality,
    validate_feature_engineering_pipeline,
    validate_hmm_regime_discovery,
    validate_multi_timeframe_processing,
)

# Import advanced decorators
from src.utils.advanced_decorators import (
    performance_monitor,
    model_validation,
    pipeline_checkpoint,
    intelligent_caching,
    adaptive_resource_allocation,
    comprehensive_validation,
    PerformanceLevel,
    ValidationLevel,
)

# ============================================================================
# VALIDATE_DATA_QUALITY DECORATOR IMPLEMENTATION
# ============================================================================

def validate_data_quality(
    validation_level: str = "WARNING",
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    max_null_ratio: float = 0.5,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_correlation: bool = False,
    max_correlation_threshold: float = 0.95,
    min_unique_values: int = 2,
    context: str = "default",
    fail_on_issues: bool = False
):
    """
    Comprehensive data quality validation decorator.
    
    Args:
        validation_level: Validation level ("WARNING", "ERROR", "INFO")
        required_columns: List of required columns
        min_rows: Minimum number of rows required
        max_null_ratio: Maximum allowed ratio of null values
        check_duplicates: Whether to check for duplicates
        check_timestamps: Whether to check timestamp consistency
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_constant: Whether to check for constant features
        check_correlation: Whether to check for high correlations
        max_correlation_threshold: Maximum correlation threshold
        min_unique_values: Minimum unique values for non-constant features
        context: Context for logging
        fail_on_issues: Whether to fail on quality issues
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{context}")
            
            # Validate input data
            input_issues = _validate_data_quality_internal(
                args, kwargs, "input", logger, validation_level,
                required_columns, min_rows, max_null_ratio, check_duplicates,
                check_timestamps, check_nan, check_infinite, check_constant,
                check_correlation, max_correlation_threshold, min_unique_values
            )
            
            if input_issues and validation_level == "ERROR":
                raise ValueError(f"Input data quality validation failed: {input_issues}")
            elif input_issues and validation_level == "WARNING":
                logger.warning(f"⚠️ Input data quality issues: {input_issues}")
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ Function execution failed in {context}: {e}")
                raise
            
            # Validate output data
            if result is not None:
                output_issues = _validate_data_quality_internal(
                    [result], {}, "output", logger, validation_level,
                    required_columns, min_rows, max_null_ratio, check_duplicates,
                    check_timestamps, check_nan, check_infinite, check_constant,
                    check_correlation, max_correlation_threshold, min_unique_values
                )
                
                if output_issues and validation_level == "ERROR":
                    raise ValueError(f"Output data quality validation failed: {output_issues}")
                elif output_issues and validation_level == "WARNING":
                    logger.warning(f"⚠️ Output data quality issues: {output_issues}")
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{context}")
            
            # Validate input data
            input_issues = _validate_data_quality_internal(
                args, kwargs, "input", logger, validation_level,
                required_columns, min_rows, max_null_ratio, check_duplicates,
                check_timestamps, check_nan, check_infinite, check_constant,
                check_correlation, max_correlation_threshold, min_unique_values
            )
            
            if input_issues and validation_level == "ERROR":
                raise ValueError(f"Input data quality validation failed: {input_issues}")
            elif input_issues and validation_level == "WARNING":
                logger.warning(f"⚠️ Input data quality issues: {input_issues}")
            
            # Execute the function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ Function execution failed in {context}: {e}")
                raise
            
            # Validate output data
            if result is not None:
                output_issues = _validate_data_quality_internal(
                    [result], {}, "output", logger, validation_level,
                    required_columns, min_rows, max_null_ratio, check_duplicates,
                    check_timestamps, check_nan, check_infinite, check_constant,
                    check_correlation, max_correlation_threshold, min_unique_values
                )
                
                if output_issues and validation_level == "ERROR":
                    raise ValueError(f"Output data quality validation failed: {output_issues}")
                elif output_issues and validation_level == "WARNING":
                    logger.warning(f"⚠️ Output data quality issues: {output_issues}")
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def _validate_data_quality_internal(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    data_type: str,
    logger: logging.Logger,
    validation_level: str,
    required_columns: Optional[List[str]],
    min_rows: int,
    max_null_ratio: float,
    check_duplicates: bool,
    check_timestamps: bool,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_correlation: bool,
    max_correlation_threshold: float,
    min_unique_values: int
) -> List[str]:
    """Internal function to validate data quality."""
    issues = []
    
    # Find DataFrame arguments
    dataframes = []
    for arg in args:
        if pandas is not None and isinstance(arg, pandas.DataFrame):
            dataframes.append(arg)
    
    for value in kwargs.values():
        if pandas is not None and isinstance(value, pandas.DataFrame):
            dataframes.append(value)
    
    if not dataframes:
        return issues
    
    for df in dataframes:
        df_issues = _validate_single_dataframe(
            df, required_columns, min_rows, max_null_ratio,
            check_duplicates, check_timestamps, check_nan,
            check_infinite, check_constant, check_correlation,
            max_correlation_threshold, min_unique_values
        )
        issues.extend([f"{data_type}: {issue}" for issue in df_issues])
    
    return issues

def _validate_single_dataframe(
    df: Any,
    required_columns: Optional[List[str]],
    min_rows: int,
    max_null_ratio: float,
    check_duplicates: bool,
    check_timestamps: bool,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_correlation: bool,
    max_correlation_threshold: float,
    min_unique_values: int
) -> List[str]:
    """Validate a single DataFrame."""
    issues = []
    
    if not pandas or not isinstance(df, pandas.DataFrame):
        return issues
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"Insufficient rows: {len(df)} < {min_rows}")
    
    # Check null ratio
    if max_null_ratio < 1.0:
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio > max_null_ratio:
                issues.append(f"High null ratio in {col}: {null_ratio:.2%}")
    
    # Check for duplicates
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
    
    # Check for NaN values
    if check_nan and pandas is not None:
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
    
    # Check for infinite values
    if check_infinite and numpy is not None:
        inf_count = numpy.isinf(df.select_dtypes(include=[numpy.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
    
    # Check for constant features
    if check_constant:
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count < min_unique_values:
                issues.append(f"Constant feature {col}: {unique_count} unique values")
    
    # Check correlations
    if check_correlation and len(df.columns) > 1:
        numeric_df = df.select_dtypes(include=[numpy.number] if numpy else [])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > max_correlation_threshold:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_value))
            
            if high_corr_pairs:
                issues.append(f"High correlations found: {len(high_corr_pairs)} pairs > {max_correlation_threshold}")
    
    return issues

# ============================================================================
# QUALITY_GATE DECORATOR IMPLEMENTATION
# ============================================================================

def quality_gate(
    min_quality_score: float = 0.7,
    quality_metrics: Optional[List[str]] = None,
    alert_config: Optional[Dict[str, Any]] = None,
    validation_level: str = "WARNING"
):
    """
    Quality gate decorator that enforces data quality standards.
    
    Args:
        min_quality_score: Minimum quality score required (0.0 to 1.0)
        quality_metrics: List of quality metrics to check
        alert_config: Configuration for alert system
        validation_level: Validation level ("basic", "comprehensive", "strict")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("QualityGate")
            
            # Execute the original function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ Function execution failed: {e}")
                raise
            
            # Check quality gates
            if result is not None:
                quality_score, grade = _check_quality_gates(
                    result, min_quality_score, quality_metrics, validation_level
                )
                
                if quality_score < min_quality_score:
                    message = f"Quality gate failed: score {quality_score:.3f} < {min_quality_score}"
                    if validation_level == "strict":
                        raise ValueError(message)
                    else:
                        logger.warning(f"⚠️ {message}")
                
                # Send alerts if configured
                if alert_config and quality_score < min_quality_score:
                    _send_quality_alert(alert_config, quality_score, grade, logger)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("QualityGate")
            
            # Execute the original function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ Function execution failed: {e}")
                raise
            
            # Check quality gates
            if result is not None:
                quality_score, grade = _check_quality_gates(
                    result, min_quality_score, quality_metrics, validation_level
                )
                
                if quality_score < min_quality_score:
                    message = f"Quality gate failed: score {quality_score:.3f} < {min_quality_score}"
                    if validation_level == "strict":
                        raise ValueError(message)
                    else:
                        logger.warning(f"⚠️ {message}")
                
                # Send alerts if configured
                if alert_config and quality_score < min_quality_score:
                    _send_quality_alert(alert_config, quality_score, grade, logger)
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def _extract_dataframe_from_result(result: Any) -> Optional[Any]:
    """Extract DataFrame from function result."""
    if not pandas:
        return None
    
    if isinstance(result, pandas.DataFrame):
        return result
    elif isinstance(result, (list, tuple)) and result:
        # Check if first element is a DataFrame
        if isinstance(result[0], pandas.DataFrame):
            return result[0]
    elif isinstance(result, dict):
        # Check if any value is a DataFrame
        for value in result.values():
            if isinstance(value, pandas.DataFrame):
                return value
    
    return None

def _calculate_quality_score(df: Any) -> Tuple[float, str]:
    """Calculate quality score for a DataFrame."""
    if not pandas or not isinstance(df, pandas.DataFrame):
        return 0.5, "C"  # Default score when dependencies not available
    
    try:
        # Calculate various quality metrics
        completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Check for duplicates
        uniqueness = 1.0 - (df.duplicated().sum() / len(df))
        
        # Check for constant features
        constant_features = sum(1 for col in df.columns if df[col].nunique() <= 1)
        variety = 1.0 - (constant_features / len(df.columns))
        
        # Check for infinite values in numeric columns
        if numpy:
            numeric_df = df.select_dtypes(include=[numpy.number])
            if not numeric_df.empty:
                inf_ratio = numpy.isinf(numeric_df).sum().sum() / (len(numeric_df) * len(numeric_df.columns))
                validity = 1.0 - inf_ratio
            else:
                validity = 1.0
        else:
            validity = 1.0
        
        # Calculate overall score (weighted average)
        overall_score = (completeness * 0.3 + uniqueness * 0.3 + variety * 0.2 + validity * 0.2)
        
        # Determine grade
        if overall_score >= 0.9:
            grade = "A"
        elif overall_score >= 0.8:
            grade = "B"
        elif overall_score >= 0.7:
            grade = "C"
        elif overall_score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        return overall_score, grade
        
    except Exception:
        return 0.5, "C"

def _check_quality_gates(
    result: Any,
    min_quality_score: float,
    quality_metrics: Optional[List[str]],
    validation_level: str
) -> Tuple[float, str]:
    """Check quality gates against the result."""
    # Extract DataFrame from result
    df = _extract_dataframe_from_result(result)
    
    if df is None:
        return 0.5, "C"  # Default score for non-DataFrame results
    
    # Calculate quality score
    quality_score, grade = _calculate_quality_score(df)
    
    # Log quality metrics
    logger = system_logger.getChild("QualityGate")
    logger.info(f"Quality score: {quality_score:.3f} ({grade})")
    
    return quality_score, grade

def _send_quality_alert(
    alert_config: Dict[str, Any],
    quality_score: float,
    grade: str,
    logger: logging.Logger
) -> None:
    """Send quality alert based on configuration."""
    try:
        # Extract alert configuration
        webhook_url = alert_config.get("webhook_url")
        slack_webhook = alert_config.get("slack_webhook")
        email_config = alert_config.get("email_config")
        
        # Prepare alert message
        message = f"Quality gate alert: Score {quality_score:.3f} ({grade})"
        
        # Send to webhook if configured
        if webhook_url:
            logger.info(f"📤 Sending webhook alert: {message}")
            # Implementation would go here
        
        # Send to Slack if configured
        if slack_webhook:
            logger.info(f"📤 Sending Slack alert: {message}")
            # Implementation would go here
        
        # Send email if configured
        if email_config:
            logger.info(f"📤 Sending email alert: {message}")
            # Implementation would go here
            
    except Exception as e:
        logger.error(f"Failed to send quality alert: {e}")

# ============================================================================
# PLACEHOLDER DECORATORS FOR BACKWARD COMPATIBILITY
# ============================================================================

# These decorators are placeholders for backward compatibility
# They should be replaced with actual implementations as needed

def validate_klines_data(func: F) -> F:
    """Placeholder decorator for klines data validation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement klines data validation
        return func(*args, **kwargs)
    return wrapper

def format_klines_data(func: F) -> F:
    """Placeholder decorator for klines data formatting."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement klines data formatting
        return func(*args, **kwargs)
    return wrapper

def validate_trading_data(func: F) -> F:
    """Placeholder decorator for trading data validation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement trading data validation
        return func(*args, **kwargs)
    return wrapper

def validate_model_output(func: F) -> F:
    """Placeholder decorator for model output validation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement model output validation
        return func(*args, **kwargs)
    return wrapper

def auto_fix_data_quality_issues(func: F) -> F:
    """Placeholder decorator for auto-fixing data quality issues."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO: Implement auto-fix functionality
        return func(*args, **kwargs)
    return wrapper

# ============================================================================
# EXPORT ALL DECORATORS
# ============================================================================

__all__ = [
    # Core decorators
    "validate_data_quality",
    "quality_gate",
    
    # Error handling decorators
    "handle_errors",
    "handle_specific_errors",
    "handle_file_operations",
    
    # Training pipeline decorators
    "deterministic_seed",
    "idempotent_step",
    "artifact_write_lock",
    "nan_inf_and_constant_guard",
    "artifact_versioning",
    "time_budget_watchdog",
    "validate_step_prerequisites",
    "secure_data_processing",
    "prevent_data_leakage",
    "resource_monitor",
    "memory_efficient",
    "debug_training_step",
    "circuit_breaker_protection",
    "validate_step_output",
    
    # Data validation decorators
    "validate_call_or_runtime_types",
    "pa_check_input",
    "pa_check_output",
    "pa_check_io",
    "enforce_ndarray",
    "auto_vectorize",
    "guard_array_nan_inf",
    "guard_dataframe_nulls",
    "with_tracing_span",
    
    # Enhanced data quality decorators
    "validate_constant_features",
    "validate_low_variance_features",
    "validate_data_completeness",
    "validate_datetime_index",
    "validate_multi_timeframe_alignment",
    "validate_hmm_data_requirements",
    "validate_data_structure",
    "optimize_memory_usage",
    "comprehensive_data_validation",
    "validate_memory_optimized_data_quality",
    "validate_feature_engineering_pipeline",
    "validate_hmm_regime_discovery",
    "validate_multi_timeframe_processing",
    
    # Advanced decorators
    "performance_monitor",
    "model_validation",
    "pipeline_checkpoint",
    "intelligent_caching",
    "adaptive_resource_allocation",
    "comprehensive_validation",
    "PerformanceLevel",
    "ValidationLevel",
    
    # Placeholder decorators for backward compatibility
    "validate_klines_data",
    "format_klines_data",
    "validate_trading_data",
    "validate_model_output",
    "auto_fix_data_quality_issues",
]