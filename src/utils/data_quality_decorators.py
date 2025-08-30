"""
Data Quality Decorators

This module provides decorators for automatic data quality validation
at each pipeline step, with special attention to NaN, infinite, and constant values.

ENHANCED FEATURES:
- Integration with enhanced decorator system
- Intelligent caching for validation results
- Performance monitoring and metrics
- Better error handling and recovery
- Centralized configuration support
"""

import asyncio
import functools
import logging
import time
import inspect
from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import pandas as pd

try:
    from src.utils.logger import system_logger
except ImportError:
    system_logger = logging.getLogger("DataQualityDecorators")

# Import enhanced system components (optional to avoid circular imports)
try:
    from .decorator_config import global_config
    from .decorator_registry import decorator_registry, register_decorator
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEM_AVAILABLE = False
    global_config = None
    decorator_registry = None

# --------------------------
# Enhanced helper functions
# --------------------------

def _get_enhanced_config(key: str, default: Any = None) -> Any:
    """Get configuration from enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and global_config:
        return getattr(global_config, key, default)
    return default

def _should_enable_caching() -> bool:
    """Check if caching should be enabled based on configuration."""
    return _get_enhanced_config('cache_enabled', False)

def _should_enable_performance_monitoring() -> bool:
    """Check if performance monitoring should be enabled."""
    return _get_enhanced_config('enable_performance_monitoring', False)

def _get_cache_settings() -> tuple[int, int]:
    """Get cache settings from configuration."""
    cache_size = _get_enhanced_config('cache_size', 128)
    cache_ttl = _get_enhanced_config('cache_ttl', 3600)
    return cache_size, cache_ttl

def _register_decorator_if_available(name: str, decorator: Callable, **kwargs):
    """Register decorator in enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and decorator_registry:
        try:
            decorator_registry.register(name=name, decorator=decorator, **kwargs)
        except Exception as e:
            logging.debug(f"Could not register decorator {name}: {e}")

def _create_cache_key(func: Callable, args: tuple, kwargs: dict) -> int:
    """Create a cache key for function calls."""
    try:
        # Create a hash of function signature and arguments
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        key_data = f"{func.__name__}:{sorted(bound.arguments.items())}"
        return hash(key_data)  # Use hash for faster key generation
    except Exception:
        # Fallback to simpler key generation
        key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hash(key_data)

def _apply_caching(wrapper_func: Callable, cache_size: int, ttl_seconds: int) -> Callable:
    """Apply caching to a wrapper function."""
    if not _should_enable_caching():
        return wrapper_func
    
    cache = {}
    
    @functools.wraps(wrapper_func)
    def cached_wrapper(*args, **kwargs):
        cache_key = _create_cache_key(wrapper_func, args, kwargs)
        current_time = time.time()
        
        # Check cache
        if cache_key in cache:
            cache_entry = cache[cache_key]
            if current_time - cache_entry['timestamp'] < ttl_seconds:
                logging.debug(f"Cache hit for {wrapper_func.__name__}")
                return cache_entry['result']
        
        # Execute and cache
        result = wrapper_func(*args, **kwargs)
        cache[cache_key] = {
            'result': result,
            'timestamp': current_time
        }
        
        # Maintain cache size
        if len(cache) > cache_size:
            oldest_key = min(cache.keys(), key=lambda k: cache[k]['timestamp'])
            del cache[oldest_key]
        
        logging.debug(f"Cached result for {wrapper_func.__name__}")
        return result
    
    return cached_wrapper

def _apply_performance_monitoring(wrapper_func: Callable, level: str = "basic") -> Callable:
    """Apply performance monitoring to a wrapper function."""
    if not _should_enable_performance_monitoring():
        return wrapper_func
    
    @functools.wraps(wrapper_func)
    def monitored_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage() if level in ["detailed", "profiling"] else 0
        
        try:
            result = wrapper_func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            metrics = {
                'function': wrapper_func.__name__,
                'execution_time': execution_time,
                'timestamp': time.time()
            }
            
            if level in ["detailed", "profiling"]:
                end_memory = _get_memory_usage()
                metrics['memory_delta_mb'] = end_memory - start_memory
                metrics['peak_memory_mb'] = end_memory
            
            _log_performance_metrics(metrics, level)
    
    return monitored_wrapper

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

def _log_performance_metrics(metrics: Dict[str, Any], level: str):
    """Log performance metrics based on level."""
    if level == "basic":
        logging.info(f"Performance: {metrics['function']} took {metrics['execution_time']:.3f}s")
    elif level == "detailed":
        logging.info(f"Performance details for {metrics['function']}: {metrics}")
    elif level == "profiling":
        logging.debug(f"Performance profiling for {metrics['function']}: {metrics}")

# --------------------------
# Enhanced Data Quality Decorators
# --------------------------

@_register_decorator_if_available(
    name="validate_data_quality",
    version="2.0",
    description="Enhanced data quality validation with caching and performance monitoring",
    tags=["validation", "data-quality", "enhanced"]
)
def validate_data_quality(
    required_columns: Optional[list] = None,
    min_rows: int = 1,
    max_null_ratio: float = 0.0,
    check_duplicates: bool = True,
    check_timestamps: bool = True,
    context: str = "data validation"
):
    """
    Enhanced decorator to validate data quality with specific parameters.
    
    ENHANCED FEATURES:
    - Intelligent caching for validation results
    - Performance monitoring and metrics
    - Better error handling and recovery
    - Integration with enhanced configuration system
    
    Args:
        required_columns: List of required columns (None for no validation)
        min_rows: Minimum number of rows required
        max_null_ratio: Maximum allowed ratio of null values
        check_duplicates: Whether to check for duplicates
        check_timestamps: Whether to check timestamp consistency
        context: Context for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{context}")
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ Function execution failed in {context}: {e}")
                raise
            
            return result
        
        # Apply enhanced features
        cache_size, ttl_seconds = _get_cache_settings()
        enhanced_wrapper = _apply_caching(wrapper, cache_size, ttl_seconds)
        enhanced_wrapper = _apply_performance_monitoring(enhanced_wrapper, "basic")
        
        return enhanced_wrapper
    return decorator


async def _validate_and_execute(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    validation_level: Any,
    validate_input: bool,
    validate_output: bool
) -> Any:
    """
    Enhanced internal function to validate and execute pipeline functions.
    This is used by the training pipeline decorators.
    
    ENHANCED FEATURES:
    - Performance monitoring for validation operations
    - Better error handling and recovery
    - Integration with enhanced configuration system
    """
    logger = system_logger.getChild("ValidateAndExecute")
    
    # Execute the function
    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(self, *args, **kwargs)
        else:
            result = func(self, *args, **kwargs)
        
        logger.info("✅ Function executed successfully")
        return result
        
    except Exception as e:
        logger.error(f"❌ Function execution failed: {e}")
        raise


@_register_decorator_if_available(
    name="validate_data_quality_at_step",
    version="2.0",
    description="Enhanced step-based data quality validation with intelligent caching",
    tags=["validation", "data-quality", "step-based", "enhanced"]
)
def validate_data_quality_at_step(
    step_name: str,
    validate_input: bool = True,
    validate_output: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_correlation: bool = True,
    max_nan_ratio: float = 0.0,    # 0% NaN (zero tolerance)
    max_infinite_count: int = 0,   # 0 infinite values (zero tolerance)
    min_unique_values: int = 2,
    max_correlation_threshold: float = 0.95,
    fail_on_issues: bool = False,
    log_issues: bool = True
):
    """
    Decorator to validate data quality at each pipeline step.
    
    Args:
        step_name: Name of the step for logging
        validate_input: Whether to validate input data
        validate_output: Whether to validate output data
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_constant: Whether to check for constant features
        check_correlation: Whether to check for high correlations
        max_nan_ratio: Maximum allowed ratio of NaN values
        max_infinite_ratio: Maximum allowed ratio of infinite values
        min_unique_values: Minimum unique values for non-constant features
        max_correlation_threshold: Maximum correlation threshold
        fail_on_issues: Whether to fail the step on quality issues
        log_issues: Whether to log quality issues
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{step_name}")
            
            # Validate input data if requested
            if validate_input:
                logger.info(f"🔍 Validating input data quality for {step_name}...")
                input_issues = _validate_data_quality(
                    args, kwargs, "input", logger,
                    check_nan, check_infinite, check_constant, check_correlation,
                    max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
                )
                
                if input_issues and log_issues:
                    logger.warning(f"⚠️ Input data quality issues found in {step_name}:")
                    for issue in input_issues[:5]:  # Show first 5 issues
                        logger.warning(f"   - {issue}")
                    if len(input_issues) > 5:
                        logger.warning(f"   ... and {len(input_issues) - 5} more issues")
                
                if input_issues and fail_on_issues:
                    raise ValueError(f"Input data quality validation failed for {step_name}: {input_issues}")
            
            # Execute the function
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ {step_name} execution failed: {e}")
                raise
            
            # Validate output data if requested
            if validate_output and result is not None:
                logger.info(f"🔍 Validating output data quality for {step_name}...")
                output_issues = _validate_data_quality(
                    [result], {}, "output", logger,
                    check_nan, check_infinite, check_constant, check_correlation,
                    max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
                )
                
                if output_issues and log_issues:
                    logger.warning(f"⚠️ Output data quality issues found in {step_name}:")
                    for issue in output_issues[:5]:  # Show first 5 issues
                        logger.warning(f"   - {issue}")
                    if len(output_issues) > 5:
                        logger.warning(f"   ... and {len(output_issues) - 5} more issues")
                
                if output_issues and fail_on_issues:
                    raise ValueError(f"Output data quality validation failed for {step_name}: {output_issues}")
            
            return result
        
        return wrapper
    return decorator


def _validate_data_quality(
    args: tuple,
    kwargs: dict,
    data_type: str,
    logger: logging.Logger,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_correlation: bool,
    max_nan_ratio: float,
    max_infinite_count: int,
    min_unique_values: int,
    max_correlation_threshold: float
) -> list[str]:
    """
    Validate data quality for given arguments and keyword arguments.
    
    Returns:
        List of quality issues found
    """
    issues = []
    
    # Check all arguments for DataFrames
    for i, arg in enumerate(args):
        if isinstance(arg, pd.DataFrame):
            df_issues = _validate_dataframe_quality(
                arg, f"{data_type}_arg_{i}", logger,
                check_nan, check_infinite, check_constant, check_correlation,
                max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
            )
            issues.extend(df_issues)
    
    # Check all keyword arguments for DataFrames
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            df_issues = _validate_dataframe_quality(
                value, f"{data_type}_kwarg_{key}", logger,
                check_nan, check_infinite, check_constant, check_correlation,
                max_nan_ratio, max_infinite_count, min_unique_values, max_correlation_threshold
            )
            issues.extend(df_issues)
    
    return issues


def _validate_dataframe_quality(
    df: pd.DataFrame,
    df_name: str,
    logger: logging.Logger,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_correlation: bool,
    max_nan_ratio: float,
    max_infinite_count: int,
    min_unique_values: int,
    max_correlation_threshold: float
) -> list[str]:
    """
    Validate DataFrame quality with specific checks.
    
    Returns:
        List of quality issues found
    """
    issues = []
    
    if df.empty:
        issues.append(f"{df_name}: DataFrame is empty")
        return issues
    
    # Check for NaN values (zero tolerance)
    if check_nan:
        nan_counts = df.isnull().sum()
        nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
        if nan_features:
            issues.append(f"{df_name}: Features with NaN values (zero tolerance): {nan_features}")
    
    # Check for infinite values (zero tolerance)
    if check_infinite:
        infinite_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:  # Any infinite values
                infinite_features.append(col)
        
        if infinite_features:
            issues.append(f"{df_name}: Features with infinite values (zero tolerance): {infinite_features}")
    
    # Check for constant features (2+ unique values, except boolean)
    if check_constant:
        constant_features = []
        for col in df.columns:
            unique_count = df[col].nunique()
            # Allow boolean features (2 unique values) and binary features
            if unique_count < min_unique_values and not _is_boolean_feature(df[col]):
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"{df_name}: Constant features found: {constant_features}")
    
    # Check for high correlations
    if check_correlation:
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


def _is_boolean_feature(series: pd.Series) -> bool:
    """
    Check if a series represents a boolean feature.
    
    Args:
        series: Pandas series to check
        
    Returns:
        True if the series represents a boolean feature
    """
    # Check if it's already boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return True
    
    # Check if it has exactly 2 unique values that could be boolean
    unique_values = series.dropna().unique()
    if len(unique_values) == 2:
        # Check if values are typical boolean patterns
        unique_set = set(unique_values)
        boolean_patterns = [
            {True, False},
            {1, 0},
            {1.0, 0.0},
            {'True', 'False'},
            {'true', 'false'},
            {'1', '0'},
            {'yes', 'no'},
            {'Y', 'N'},
            {'y', 'n'}
        ]
        
        for pattern in boolean_patterns:
            if unique_set == pattern:
                return True
    
    return False


def validate_step1_quality(func: Callable) -> Callable:
    """Decorator specifically for Step1 data quality validation."""
    return validate_data_quality_at_step(
        "step1_data_collection",
        validate_input=True,
        validate_output=True,
        check_nan=True,
        check_infinite=True,
        check_constant=False,  # Raw data can be constant
        check_correlation=False,  # Raw data correlation is not relevant
        fail_on_issues=False,
        log_issues=True
    )(func)


def validate_step1_5_quality(func: Callable) -> Callable:
    """Decorator specifically for Step1.5 data quality validation."""
    return validate_data_quality_at_step(
        "step1_5_data_converter",
        validate_input=True,
        validate_output=True,
        check_nan=True,
        check_infinite=True,
        check_constant=False,  # Unified data can be constant
        check_correlation=False,  # Unified data correlation is not relevant
        fail_on_issues=False,
        log_issues=True
    )(func)


def validate_step2_quality(func: Callable) -> Callable:
    """Decorator specifically for Step2 data quality validation with special attention to features."""
    return validate_data_quality_at_step(
        "step2_feature_engineering",
        validate_input=True,
        validate_output=True,
        check_nan=True,
        check_infinite=True,
        check_constant=True,  # Features should not be constant
        check_correlation=True,  # Feature correlation is important
        max_nan_ratio=0.0,  # 0% NaN (zero tolerance)
        max_infinite_count=0,  # 0 infinite values (zero tolerance)
        min_unique_values=2,  # 2+ unique values (except boolean)
        max_correlation_threshold=0.95,
        fail_on_issues=False,
        log_issues=True
    )(func)


def log_feature_quality_issues(df: pd.DataFrame, df_name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log detailed feature quality issues for a DataFrame.
    
    Args:
        df: DataFrame to check
        df_name: Name of the DataFrame for logging
        logger: Logger to use (defaults to system_logger)
    """
    if logger is None:
        logger = system_logger.getChild("FeatureQualityLogger")
    
    logger.info(f"🔍 Checking feature quality for {df_name}...")
    
    # Check for NaN values (zero tolerance) with detailed information
    nan_counts = df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
    if nan_features:
        logger.warning(f"⚠️ {df_name}: Features with NaN values (zero tolerance) ({len(nan_features)}):")
        logger.warning("📊 Detailed NaN Analysis:")
        for feature in nan_features[:10]:  # Show first 10
            nan_count = nan_counts[feature]
            nan_ratio = nan_count / len(df) * 100
            logger.warning(f"   • {feature}: {nan_count} NaN values ({nan_ratio:.3f}%)")
            # Log sample of problematic indices
            nan_indices = df[df[feature].isnull()].index[:5]  # First 5 NaN indices
            if len(nan_indices) > 0:
                logger.warning(f"     Sample NaN indices: {list(nan_indices)}")
        if len(nan_features) > 10:
            logger.warning(f"   ... and {len(nan_features) - 10} more features with NaN values")
    
    # Check for infinite values (zero tolerance) with detailed information
    infinite_features = []
    for col in df.select_dtypes(include=[np.number]).columns:
        infinite_count = np.isinf(df[col]).sum()
        if infinite_count > 0:  # Any infinite values
            infinite_features.append((col, infinite_count))
    
    if infinite_features:
        logger.warning(f"⚠️ {df_name}: Features with infinite values (zero tolerance) ({len(infinite_features)}):")
        logger.warning("📊 Detailed Infinite Value Analysis:")
        for feature, count in infinite_features[:10]:  # Show first 10
            infinite_ratio = count / len(df) * 100
            logger.warning(f"   • {feature}: {count} infinite values ({infinite_ratio:.3f}%)")
            # Log sample of problematic indices
            infinite_indices = df[np.isinf(df[feature])].index[:5]  # First 5 infinite indices
            if len(infinite_indices) > 0:
                logger.warning(f"     Sample infinite indices: {list(infinite_indices)}")
        if len(infinite_features) > 10:
            logger.warning(f"   ... and {len(infinite_features) - 10} more features with infinite values")
    
    # Check for constant features (2+ unique values, except boolean)
    constant_features = []
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 2 and not _is_boolean_feature(df[col]):
            constant_features.append((col, unique_count))
    
    if constant_features:
        logger.warning(f"⚠️ {df_name}: Constant or near-constant features ({len(constant_features)}):")
        logger.warning("📊 Detailed Constant Feature Analysis:")
        for feature, unique_count in constant_features[:10]:  # Show first 10
            unique_values = df[feature].dropna().unique()
            logger.warning(f"   • {feature}: {unique_count} unique values: {unique_values}")
            # Log value distribution
            value_counts = df[feature].value_counts()
            logger.warning(f"     Value distribution: {dict(value_counts.head(3))}")
        if len(constant_features) > 10:
            logger.warning(f"   ... and {len(constant_features) - 10} more constant features")
    
    # Check for high correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
            if high_corr_pairs:
                logger.warning(f"⚠️ {df_name}: Highly correlated feature pairs ({len(high_corr_pairs)}):")
            logger.warning("📊 Detailed Correlation Analysis:")
            for feat1, feat2, corr_value in high_corr_pairs[:5]:  # Show first 5
                logger.warning(f"   • {feat1} ↔ {feat2}: correlation = {corr_value:.3f}")
                # Log sample of values to show the relationship
                sample_size = min(5, len(df))
                sample_df = df[[feat1, feat2]].head(sample_size)
                logger.warning(f"     Sample values: {feat1}={list(sample_df[feat1])}, {feat2}={list(sample_df[feat2])}")
            if len(high_corr_pairs) > 5:
                logger.warning(f"   ... and {len(high_corr_pairs) - 5} more highly correlated pairs")
    
    # Summary with detailed breakdown
    total_issues = len(nan_features) + len(infinite_features) + len(constant_features) + len(high_corr_pairs)
    if total_issues == 0:
        logger.info(f"✅ {df_name}: No feature quality issues detected")
    else:
        logger.warning(f"⚠️ {df_name}: Total feature quality issues: {total_issues}")
        logger.warning("📋 Issue Breakdown:")
        logger.warning(f"   • NaN features: {len(nan_features)}")
        logger.warning(f"   • Infinite features: {len(infinite_features)}")
        logger.warning(f"   • Constant features: {len(constant_features)}")
        logger.warning(f"   • High correlation pairs: {len(high_corr_pairs)}")
        logger.warning("💡 For detailed information about each problematic value, check the validation results above")


# Convenience function for quick validation
def quick_validate_features(df: pd.DataFrame, df_name: str = "DataFrame") -> Dict[str, Any]:
    """
    Quick validation of features with summary statistics.
    
    Args:
        df: DataFrame to validate
        df_name: Name of the DataFrame
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "df_name": df_name,
        "shape": df.shape,
        "total_features": len(df.columns),
        "total_samples": len(df),
        "issues": {
            "nan_features": [],
            "infinite_features": [],
            "constant_features": [],
            "high_correlation_pairs": []
        },
        "summary": {
            "nan_count": 0,
            "infinite_count": 0,
            "constant_count": 0,
            "high_correlation_count": 0
        },
        "details": {
            "nan_details": {},
            "infinite_details": {},
            "constant_details": {},
            "correlation_details": {}
        }
    }
    
    # Check for NaN values (zero tolerance) with detailed information
    nan_counts = df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0].index.tolist()  # Any NaN values
    results["issues"]["nan_features"] = nan_features
    results["summary"]["nan_count"] = len(nan_features)
    
    # Add detailed NaN information
    nan_details = {}
    for feature in nan_features:
        nan_count = nan_counts[feature]
        nan_ratio = nan_count / len(df) * 100
        nan_details[feature] = {
            "count": int(nan_count),
            "percentage": float(nan_ratio),
            "sample_indices": df[df[feature].isnull()].index[:10].tolist()  # First 10 NaN indices
        }
    results["details"]["nan_details"] = nan_details
    
    # Check for infinite values (zero tolerance) with detailed information
    infinite_features = []
    infinite_details = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        infinite_count = np.isinf(df[col]).sum()
        if infinite_count > 0:  # Any infinite values
            infinite_features.append(col)
            infinite_ratio = infinite_count / len(df) * 100
            infinite_details[col] = {
                "count": int(infinite_count),
                "percentage": float(infinite_ratio),
                "sample_indices": df[np.isinf(df[col])].index[:10].tolist()  # First 10 infinite indices
            }
    results["issues"]["infinite_features"] = infinite_features
    results["summary"]["infinite_count"] = len(infinite_features)
    results["details"]["infinite_details"] = infinite_details
    
    # Check for constant features (2+ unique values, except boolean) with detailed information
    constant_features = []
    constant_details = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 2 and not _is_boolean_feature(df[col]):
            constant_features.append(col)
            unique_values = df[col].dropna().unique()
            value_counts = df[col].value_counts()
            constant_details[col] = {
                "unique_count": int(unique_count),
                "unique_values": unique_values.tolist(),
                "value_distribution": value_counts.to_dict(),
                "is_boolean": _is_boolean_feature(df[col])
            }
    results["issues"]["constant_features"] = constant_features
    results["summary"]["constant_count"] = len(constant_features)
    results["details"]["constant_details"] = constant_details
    
    # Check for high correlations with detailed information
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    high_corr_pairs = []
    correlation_details = {}
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.95:
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    pair_key = f"{feat1}↔{feat2}"
                    high_corr_pairs.append((feat1, feat2, corr_value))
                    
                    # Add detailed correlation information
                    correlation_details[pair_key] = {
                        "feature1": feat1,
                        "feature2": feat2,
                        "correlation": float(corr_value),
                        "sample_values": {
                            feat1: df[feat1].head(5).tolist(),
                            feat2: df[feat2].head(5).tolist()
                        }
                    }
    results["issues"]["high_correlation_pairs"] = high_corr_pairs
    results["summary"]["high_correlation_count"] = len(high_corr_pairs)
    results["details"]["correlation_details"] = correlation_details
    
    return results
