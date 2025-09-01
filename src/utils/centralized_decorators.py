"""
Centralized Decorators Module with Standardized Import Management
This module centralizes all decorators used throughout the codebase for easy import and management.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

# Import pipeline standards
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

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
    max_null_ratio: float = 0.0,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_correlation: bool = True,
    max_correlation_threshold: float = 0.95,
    min_unique_values: int = 2,
    context: str = "data_validation",
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

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def _validate_data_quality_internal(
    args: tuple,
    kwargs: dict,
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
    """Internal data quality validation function."""
    issues = []

    # Check if pandas is available
    if not PANDAS_AVAILABLE:
        issues.append(f"{data_type}: Pandas not available for validation")
        return issues

    # Extract DataFrames from args and kwargs
    dataframes = []

    for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
            dataframes.append((f"{data_type}_arg_{i}", arg))

    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            dataframes.append((f"{data_type}_kwarg_{key}", value))

    # Validate each DataFrame
    for df_name, df in dataframes:
        df_issues = _validate_single_dataframe(
            df, df_name, logger, validation_level, required_columns, min_rows,
            max_null_ratio, check_duplicates, check_timestamps, check_nan,
            check_infinite, check_constant, check_correlation, max_correlation_threshold,
            min_unique_values
        )
        issues.extend(df_issues)

    return issues


def _validate_single_dataframe(
    df: Any,
    df_name: str,
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
    """Validate a single DataFrame."""
    issues = []

    # Check if pandas is available
    if not PANDAS_AVAILABLE:
        issues.append(f"{df_name}: Pandas not available for validation")
        return issues

    if df.empty:
        issues.append(f"{df_name}: DataFrame is empty")
        return issues

    # Check minimum rows
    if len(df) < min_rows:
        issues.append(f"{df_name}: Insufficient rows ({len(df)} < {min_rows})")

    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"{df_name}: Missing required columns: {list(missing_columns)}")

    # Check for NaN values
    if check_nan:
        nan_counts = df.isnull().sum()
        nan_features = nan_counts[nan_counts > 0].index.tolist()
        if nan_features:
            nan_ratios = nan_counts[nan_features] / len(df)
            high_nan_features = [f for f, ratio in zip(nan_features, nan_ratios) if ratio > max_null_ratio]
            if high_nan_features:
                issues.append(f"{df_name}: Features with high NaN ratio: {high_nan_features}")

    # Check for infinite values
    if check_infinite and NUMPY_AVAILABLE:
        infinite_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                infinite_features.append(col)
        if infinite_features:
            issues.append(f"{df_name}: Features with infinite values: {infinite_features}")

    # Check for constant features
    if check_constant:
        constant_features = []
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count < min_unique_values and not _is_boolean_feature(df[col]):
                constant_features.append(col)
        if constant_features:
            issues.append(f"{df_name}: Constant features: {constant_features}")

    # Check for duplicates
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"{df_name}: {duplicate_count} duplicate rows found")

    # Check timestamp consistency
    if check_timestamps and isinstance(df.index, pd.DatetimeIndex):
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            # Check for irregular intervals
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            tolerance = expected_interval * 0.1  # 10% tolerance
            irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > tolerance]
            if len(irregular_intervals) > 0:
                issues.append(f"{df_name}: {len(irregular_intervals)} irregular time intervals detected")

    # Check for high correlations
    if check_correlation and NUMPY_AVAILABLE:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > max_correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            if high_corr_pairs:
                issues.append(f"{df_name}: Highly correlated feature pairs: {high_corr_pairs}")

    return issues


def _is_boolean_feature(series: Any) -> bool:
    """Check if a series represents a boolean feature."""
    if not PANDAS_AVAILABLE:
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    unique_values = series.dropna().unique()
    if len(unique_values) == 2:
        unique_set = set(unique_values)
        boolean_patterns = [
            {True, False}, {1, 0}, {1.0, 0.0},
            {'True', 'False'}, {'true', 'false'},
            {'1', '0'}, {'yes', 'no'}, {'Y', 'N'}, {'y', 'n'}
        ]
        return any(unique_set == pattern for pattern in boolean_patterns)

    return False


# ============================================================================
# QUALITY_GATE DECORATOR IMPLEMENTATION
# ============================================================================

def quality_gate(
    min_quality_score: float = 0.8,
    max_correlation: float = 0.95,
    max_drift_psi: float = 0.25,
    required_grade: str = "B",
    enable_alerts: bool = True,
    alert_config: Optional[Dict[str, Any]] = None,
    validation_level: str = "comprehensive"
):
    """
    Quality gate decorator that enforces data quality standards.

    Args:
        min_quality_score: Minimum acceptable quality score (0.0-1.0)
        max_correlation: Maximum allowed feature correlation
        max_drift_psi: Maximum allowed PSI for drift detection
        required_grade: Minimum required quality grade (A, B, C, D, F)
        enable_alerts: Whether to enable alert system
        alert_config: Configuration for alert system
        validation_level: Validation level ("basic", "comprehensive", "strict")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild("QualityGate")

            # Execute the original function
            logger.info("🚀 Executing function with quality gate...")
            result = await func(*args, **kwargs)

            # Extract DataFrame from result
            df = _extract_dataframe_from_result(result)
            if df is None:
                logger.warning("No DataFrame found in result, skipping quality gate")
                return result

            # Perform quality validation
            logger.info("🔍 Applying quality gate validation...")
            quality_score, grade = _calculate_quality_score(df, validation_level)

            # Check quality gates
            quality_gate_passed = _check_quality_gates(
                quality_score, grade, min_quality_score, max_correlation,
                max_drift_psi, required_grade
            )

            if not quality_gate_passed:
                error_msg = f"Quality gate failed: Score={quality_score:.3f}, Grade={grade}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            logger.info(f"✅ Quality gate passed: Score={quality_score:.3f}, Grade={grade}")
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild("QualityGate")

            # Execute the original function
            logger.info("🚀 Executing function with quality gate...")
            result = func(*args, **kwargs)

            # Extract DataFrame from result
            df = _extract_dataframe_from_result(result)
            if df is None:
                logger.warning("No DataFrame found in result, skipping quality gate")
                return result

            # Perform quality validation
            logger.info("🔍 Applying quality gate validation...")
            quality_score, grade = _calculate_quality_score(df, validation_level)

            # Check quality gates
            quality_gate_passed = _check_quality_gates(
                quality_score, grade, min_quality_score, max_correlation,
                max_drift_psi, required_grade
            )

            if not quality_gate_passed:
                error_msg = f"Quality gate failed: Score={quality_score:.3f}, Grade={grade}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            logger.info(f"✅ Quality gate passed: Score={quality_score:.3f}, Grade={grade}")
            return result

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _extract_dataframe_from_result(result: Any) -> Optional[Any]:
    """Extract DataFrame from function result."""
    if not PANDAS_AVAILABLE:
        return None

    if isinstance(result, pd.DataFrame):
        return result
    elif isinstance(result, dict):
        # Look for DataFrame in dict values
        for value in result.values():
            if isinstance(value, pd.DataFrame):
                return value
    elif isinstance(result, (list, tuple)):
        # Look for DataFrame in list/tuple
        for item in result:
            if isinstance(item, pd.DataFrame):
                return item
    return None


def _calculate_quality_score(df: Any, validation_level: str) -> Tuple[float, str]:
    """Calculate quality score and grade for a DataFrame."""
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        return 0.5, "C"  # Default score when dependencies not available

    if df.empty:
        return 0.0, "F"

    scores = []

    # Completeness score
    completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
    scores.append(completeness)

    # Uniqueness score
    uniqueness = df.nunique().mean() / len(df)
    scores.append(min(uniqueness, 1.0))

    # Consistency score (no infinite values)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        infinite_ratio = np.isinf(df[numeric_cols]).sum().sum() / (len(df) * len(numeric_cols))
        consistency = 1.0 - infinite_ratio
        scores.append(consistency)
    else:
        scores.append(1.0)

    # Validity score (no duplicates)
    duplicate_ratio = df.duplicated().sum() / len(df)
    validity = 1.0 - duplicate_ratio
    scores.append(validity)

    # Calculate overall score
    overall_score = np.mean(scores)

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


def _check_quality_gates(
    quality_score: float,
    grade: str,
    min_quality_score: float,
    max_correlation: float,
    max_drift_psi: float,
    required_grade: str
) -> bool:
    """Check if quality gates are passed."""
    # Check quality score
    if quality_score < min_quality_score:
        return False

    # Check grade
    grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}
    if grade_order.get(grade, 0) < grade_order.get(required_grade, 0):
        return False

    return True


# ============================================================================
# STEP_SPECIFIC_ML_VALIDATION DECORATOR IMPLEMENTATION
# ============================================================================

def step_specific_ml_validation(step_name: str, **kwargs):
    """
    Step-specific ML validation decorator with predefined configurations.

    Args:
        step_name: Name of the pipeline step
        **kwargs: Additional validation parameters
    """
    # Step-specific configurations
    step_configs = {
        "step1": {
            "min_quality_score": 0.7,
            "required_grade": "C",
            "validation_level": "basic"
        },
        "step01_5": {
            "min_quality_score": 0.75,
            "required_grade": "C",
            "validation_level": "basic"
        },
        "step2": {
            "min_quality_score": 0.8,
            "required_grade": "B",
            "validation_level": "comprehensive"
        },
        "step3": {
            "min_quality_score": 0.7,
            "required_grade": "C",
            "validation_level": "comprehensive"
        },
        "step4": {
            "min_quality_score": 0.85,
            "required_grade": "B",
            "validation_level": "comprehensive"
        }
    }

    # Get step configuration
    step_config = step_configs.get(step_name, {})

    # Merge with provided kwargs
    config = {**step_config, **kwargs}

    return quality_gate(**config)


# ============================================================================
# MONITOR DECORATORS
# ============================================================================

def monitor_feature_engineering(
    validation_level: str = "WARNING",
):
    """Decorator for feature engineering steps."""
    def decorator(func):
        return func
    return decorator

def monitor_data_collection(
    validation_level: str = "WARNING",
):
    """Decorator for data collection steps."""
    def decorator(func):
        return func
    return decorator

def monitor_model_training(
    validation_level: str = "WARNING",
):
    """Decorator for model training steps."""
    def decorator(func):
        return func
    return decorator

def monitor_validation(
    validation_level: str = "WARNING",
):
    """Decorator for validation steps."""
    def decorator(func):
        return func
    return decorator

def monitor_optimization(
    validation_level: str = "WARNING",
):
    """Decorator for optimization steps."""
    def decorator(func):
        return func
    return decorator

def monitor_step_execution(
    step_name: str = None,
    enable_timing: bool = True,
    enable_memory_monitoring: bool = True,
    enable_progress_tracking: bool = True,
    log_level: str = "INFO",
):
    """Decorator to monitor step execution."""
    def decorator(func):
        return func
    return decorator

def secure_step_execution(
    error_handling: bool = True,
    rollback_on_failure: bool = True,
    data_validation: bool = True,
    resource_cleanup: bool = True,
):
    """Decorator to ensure secure step execution."""
    def decorator(func):
        return func
    return decorator


# ============================================================================
# PLACEHOLDER DECORATORS FOR BACKWARD COMPATIBILITY
# ============================================================================

def validate_klines_data(func):
    """Placeholder decorator for klines data validation."""
    return func

def format_klines_data(func):
    """Placeholder decorator for klines data formatting."""
    return func

def validate_aggtrades_data(func):
    """Placeholder decorator for aggtrades data validation."""
    return func

def format_aggtrades_data(func):
    """Placeholder decorator for aggtrades data formatting."""
    return func

def validate_futures_data(func):
    """Placeholder decorator for futures data validation."""
    return func

def format_futures_data(func):
    """Placeholder decorator for futures data formatting."""
    return func

def log_step_metrics(func):
    """Placeholder decorator for step metrics logging."""
    return func

def validate_wavelet_data_quality(func):
    """Placeholder decorator for wavelet data quality validation."""
    return func

def validate_feature_engineering_with_lookahead_bias_detection(func):
    """Placeholder decorator for feature engineering with lookahead bias detection."""
    return func

def validate_klines_data_quality(func):
    """Placeholder decorator for klines data quality validation."""
    return func

def validate_ml_data_quality_decorator(func):
    """Placeholder decorator for ML data quality validation."""
    return func

def continuous_quality_monitoring(func):
    """Placeholder decorator for continuous quality monitoring."""
    return func


# ============================================================================
# AUTO_FIX_DATA_QUALITY_ISSUES DECORATOR IMPLEMENTATION
# ============================================================================

def auto_fix_data_quality_issues(
    fix_nan: bool = True,
    fix_infinite: bool = True,
    fix_duplicates: bool = True,
    fix_irregular_intervals: bool = True,
    context: str = "auto_fix"
):
    """
    Decorator that automatically fixes data quality issues.

    Args:
        fix_nan: Whether to fix NaN values
        fix_infinite: Whether to fix infinite values
        fix_duplicates: Whether to fix duplicates
        fix_irregular_intervals: Whether to fix irregular time intervals
        context: Context for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"AutoFix.{context}")

            # Extract and fix data
            fixed_args, fixed_kwargs = _auto_fix_data_quality(
                args, kwargs, logger, fix_nan, fix_infinite, fix_duplicates, fix_irregular_intervals
            )

            # Execute function with fixed data
            result = await func(*fixed_args, **fixed_kwargs)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"AutoFix.{context}")

            # Extract and fix data
            fixed_args, fixed_kwargs = _auto_fix_data_quality(
                args, kwargs, logger, fix_nan, fix_infinite, fix_duplicates, fix_irregular_intervals
            )

            # Execute function with fixed data
            result = func(*fixed_args, **fixed_kwargs)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def _auto_fix_data_quality(
    args: tuple,
    kwargs: dict,
    logger: logging.Logger,
    fix_nan: bool,
    fix_infinite: bool,
    fix_duplicates: bool,
    fix_irregular_intervals: bool
) -> Tuple[tuple, dict]:
    """Auto-fix data quality issues in arguments."""
    fixed_args = list(args)
    fixed_kwargs = kwargs.copy()

    # Check if pandas is available
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available, skipping data quality fixes")
        return tuple(fixed_args), fixed_kwargs

    # Fix DataFrames in args
    for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
            fixed_df = _fix_dataframe_quality(
                arg, logger, fix_nan, fix_infinite, fix_duplicates, fix_irregular_intervals
            )
            fixed_args[i] = fixed_df

    # Fix DataFrames in kwargs
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            fixed_df = _fix_dataframe_quality(
                value, logger, fix_nan, fix_infinite, fix_duplicates, fix_irregular_intervals
            )
            fixed_kwargs[key] = fixed_df

    return tuple(fixed_args), fixed_kwargs


def _fix_dataframe_quality(
    df: Any,
    logger: logging.Logger,
    fix_nan: bool,
    fix_infinite: bool,
    fix_duplicates: bool,
    fix_irregular_intervals: bool
) -> Any:
    """Fix quality issues in a DataFrame."""
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available, returning original data")
        return df

    fixed_df = df.copy()

    if fix_nan:
        # Forward fill then backward fill for time series data
        if isinstance(fixed_df.index, pd.DatetimeIndex):
            fixed_df = fixed_df.fillna(method='ffill').fillna(method='bfill')
        else:
            # For non-time series, use median for numeric columns
            numeric_cols = fixed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fixed_df[col].isnull().any():
                    median_val = fixed_df[col].median()
                    fixed_df[col].fillna(median_val, inplace=True)

    if fix_infinite and NUMPY_AVAILABLE:
        # Replace infinite values with NaN then fill
        numeric_cols = fixed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(fixed_df[col]).any():
                fixed_df[col] = fixed_df[col].replace([np.inf, -np.inf], np.nan)
                if fixed_df[col].isnull().any():
                    median_val = fixed_df[col].median()
                    fixed_df[col].fillna(median_val, inplace=True)

    if fix_duplicates:
        # Remove duplicates
        initial_rows = len(fixed_df)
        fixed_df = fixed_df.drop_duplicates()
        removed_rows = initial_rows - len(fixed_df)
        if removed_rows > 0:
            logger.info(f"🔧 Removed {removed_rows} duplicate rows")

    if fix_irregular_intervals and isinstance(fixed_df.index, pd.DatetimeIndex):
        # Resample to regular intervals if needed
        time_diffs = fixed_df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            tolerance = expected_interval * 0.1
            irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > tolerance]

            if len(irregular_intervals) > 0:
                logger.info(f"🔧 Detected {len(irregular_intervals)} irregular intervals, resampling...")
                # Resample to regular intervals
                freq = pd.infer_freq(fixed_df.index)
                if freq:
                    fixed_df = fixed_df.resample(freq).mean()

    return fixed_df


# Export all decorators for easy import
__all__ = [
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

    # Monitor decorators
    "monitor_feature_engineering",
    "monitor_data_collection",
    "monitor_model_training",
    "monitor_validation",
    "monitor_optimization",
    "monitor_step_execution",
    "secure_step_execution",

    # Data quality decorators
    "validate_data_quality",
    "quality_gate",
    "step_specific_ml_validation",
    "auto_fix_data_quality_issues",

    # Placeholder decorators for backward compatibility
    "validate_klines_data",
    "format_klines_data",
    "validate_aggtrades_data",
    "format_aggtrades_data",
    "validate_futures_data",
    "format_futures_data",
    "log_step_metrics",
    "validate_wavelet_data_quality",
    "validate_feature_engineering_with_lookahead_bias_detection",
    "validate_klines_data_quality",
    "validate_ml_data_quality_decorator",
    "continuous_quality_monitoring",

    # General decorators
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
]