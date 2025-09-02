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
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
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
    """Get configuration value from enhanced system if available."""
    if ENHANCED_SYSTEM_AVAILABLE and global_config:
        return getattr(global_config, key, default)
    return default

def _should_enable_caching() -> bool:
    """Check if caching should be enabled."""
    return _get_enhanced_config('cache_enabled', False)

def _should_enable_performance_monitoring() -> bool:
    """Check if performance monitoring should be enabled."""
    return _get_enhanced_config('enable_performance_monitoring', False)

def _get_cache_settings() -> tuple:
    """Get cache size and TTL settings."""
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
    """Create a cache key for function arguments."""
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

def _apply_caching(wrapper_func: Callable, cache_size: int = 128, ttl_seconds: int = 3600) -> Callable:
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
        
        # Manage cache size
        if len(cache) >= cache_size:
            # Remove oldest entries
            oldest_key = min(cache.keys(), key=lambda k: cache[k]['timestamp'])
            del cache[oldest_key]
        
        cache[cache_key] = {
            'result': result,
            'timestamp': current_time
        }
        
        return result

    return cached_wrapper

def _apply_performance_monitoring(wrapper_func: Callable) -> Callable:
    """Apply performance monitoring to a wrapper function."""
    if not _should_enable_performance_monitoring():
        return wrapper_func

    @functools.wraps(wrapper_func)
    def monitored_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = wrapper_func(*args, **kwargs)
            execution_time = time.time() - start_time
            end_memory = _get_memory_usage()
            memory_delta = end_memory - start_memory
            
            logging.info(f"Performance: {wrapper_func.__name__} took {execution_time:.4f}s, memory: {memory_delta:+d} bytes")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Error in {wrapper_func.__name__} after {execution_time:.4f}s: {e}")
            raise

    return monitored_wrapper

def _get_memory_usage() -> int:
    """Get current memory usage in bytes."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    except ImportError:
        return 0

# --------------------------
# Main decorators
# --------------------------

def validate_data_quality(
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_timestamps: bool = False,
    context: str = "default"
):
    """
    Enhanced decorator to validate data quality with specific parameters.
    
    Args:
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_constant: Whether to check for constant columns
        check_timestamps: Whether to check timestamp consistency
        context: Context for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{context}")
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Validate result
            if result is not None:
                validation_result = await _validate_and_execute(
                    result, check_nan, check_infinite, check_constant, check_timestamps, logger
                )
                if not validation_result['is_valid']:
                    logger.warning(f"Data quality issues found: {validation_result['issues']}")
            
            return result

        # Apply enhancements
        enhanced_wrapper = _apply_performance_monitoring(wrapper)
        enhanced_wrapper = _apply_caching(enhanced_wrapper)
        
        # Register if available
        _register_decorator_if_available(
            "validate_data_quality",
            enhanced_wrapper,
            description="Enhanced data quality validation with caching and performance monitoring",
            tags=["validation", "data-quality", "enhanced"]
        )
        
        return enhanced_wrapper
    return decorator

async def _validate_and_execute(
    data: Any,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_timestamps: bool,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Validate data and return validation results."""
    logger = system_logger.getChild("ValidateAndExecute")
    
    issues = []
    
    if isinstance(data, pd.DataFrame):
        issues.extend(await _validate_dataframe_quality(
            data, check_nan, check_infinite, check_constant, check_timestamps
        ))
    elif isinstance(data, pd.Series):
        issues.extend(await _validate_series_quality(
            data, check_nan, check_infinite, check_constant
        ))
    elif isinstance(data, np.ndarray):
        issues.extend(await _validate_array_quality(
            data, check_nan, check_infinite, check_constant
        ))
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'total_issues': len(issues)
    }

def validate_data_quality_at_step(
    step_name: str,
    validate_input: bool = True,
    validate_output: bool = True,
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_timestamps: bool = False,
    fail_on_issues: bool = False,
    log_issues: bool = True
):
    """
    Enhanced step-based data quality validation with intelligent caching.
    
    Args:
        step_name: Name of the pipeline step
        validate_input: Whether to validate input data
        validate_output: Whether to validate output data
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_constant: Whether to check for constant columns
        check_timestamps: Whether to check timestamp consistency
        fail_on_issues: Whether to fail the step on quality issues
        log_issues: Whether to log quality issues
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{step_name}")
            
            # Validate input data if requested
            if validate_input and args:
                input_data = args[0] if args else None
                if input_data is not None:
                    input_validation = await _validate_data_quality(
                        input_data, check_nan, check_infinite, check_constant, check_timestamps, logger
                    )
                    if not input_validation['is_valid']:
                        if log_issues:
                            logger.warning(f"Input validation failed for {step_name}: {input_validation['issues']}")
                        if fail_on_issues:
                            raise ValueError(f"Input validation failed for {step_name}")
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Validate output data if requested
            if validate_output and result is not None:
                output_validation = await _validate_data_quality(
                    result, check_nan, check_infinite, check_constant, check_timestamps, logger
                )
                if not output_validation['is_valid']:
                    if log_issues:
                        logger.warning(f"Output validation failed for {step_name}: {output_validation['issues']}")
                    if fail_on_issues:
                        raise ValueError(f"Output validation failed for {step_name}")
            
            return result

        # Apply enhancements
        enhanced_wrapper = _apply_performance_monitoring(wrapper)
        enhanced_wrapper = _apply_caching(enhanced_wrapper)
        
        # Register if available
        _register_decorator_if_available(
            f"validate_data_quality_at_step_{step_name}",
            enhanced_wrapper,
            description=f"Step-based data quality validation for {step_name}",
            tags=["validation", "data-quality", "step-based", "enhanced"]
        )
        
        return enhanced_wrapper
    return decorator

async def _validate_data_quality(
    data: Any,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_timestamps: bool,
    logger: logging.Logger
) -> list:
    """Validate data and return list of issues."""
    issues = []
    
    if isinstance(data, pd.DataFrame):
        issues.extend(await _validate_dataframe_quality(
            data, check_nan, check_infinite, check_constant, check_timestamps
        ))
    elif isinstance(data, pd.Series):
        issues.extend(await _validate_series_quality(
            data, check_nan, check_infinite, check_constant
        ))
    elif isinstance(data, np.ndarray):
        issues.extend(await _validate_array_quality(
            data, check_nan, check_infinite, check_constant
        ))
    
    return issues

async def _validate_dataframe_quality(
    df: pd.DataFrame,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool,
    check_timestamps: bool
) -> list:
    """Validate DataFrame quality and return list of issues."""
    issues = []
    
    if check_nan:
        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]
        for col, count in nan_columns.items():
            issues.append(f"Column '{col}' has {count} NaN values")
    
    if check_infinite:
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        inf_columns = inf_counts[inf_counts > 0]
        for col, count in inf_columns.items():
            issues.append(f"Column '{col}' has {count} infinite values")
    
    if check_constant:
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        if constant_columns:
            issues.append(f"Constant columns detected: {constant_columns}")
    
    if check_timestamps:
        timestamp_columns = df.select_dtypes(include=['datetime64']).columns
        for col in timestamp_columns:
            if df[col].is_monotonic_increasing:
                issues.append(f"Timestamp column '{col}' is not monotonically increasing")
    
    return issues

async def _validate_series_quality(
    series: pd.Series,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool
) -> list:
    """Validate Series quality and return list of issues."""
    issues = []
    
    if check_nan:
        nan_count = series.isna().sum()
        if nan_count > 0:
            issues.append(f"Series has {nan_count} NaN values")
    
    if check_infinite and pd.api.types.is_numeric_dtype(series):
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            issues.append(f"Series has {inf_count} infinite values")
    
    if check_constant:
        if series.nunique() <= 1:
            issues.append("Series is constant")
    
    return issues

async def _validate_array_quality(
    array: np.ndarray,
    check_nan: bool,
    check_infinite: bool,
    check_constant: bool
) -> list:
    """Validate numpy array quality and return list of issues."""
    issues = []
    
    if check_nan:
        nan_count = np.isnan(array).sum()
        if nan_count > 0:
            issues.append(f"Array has {nan_count} NaN values")
    
    if check_infinite:
        inf_count = np.isinf(array).sum()
        if inf_count > 0:
            issues.append(f"Array has {inf_count} infinite values")
    
    if check_constant:
        if array.size > 0 and np.all(array == array.flat[0]):
            issues.append("Array is constant")
    
    return issues

def _is_boolean_feature(series: pd.Series) -> bool:
    """Check if a series represents a boolean feature."""
    # Check if it's already boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return True
    
    # Check if it has only 2 unique values
    unique_values = series.dropna().unique()
    if len(unique_values) == 2:
        return True
    
    # Check if it's numeric with only 0 and 1
    if pd.api.types.is_numeric_dtype(series):
        unique_numeric = series.dropna().unique()
        if len(unique_numeric) == 2 and 0 in unique_numeric and 1 in unique_numeric:
            return True
    
    return False

# --------------------------
# Utility decorators
# --------------------------

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = system_logger.getChild("ExecutionTime")
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    return wrapper

def validate_input_types(*expected_types: type):
    """Decorator to validate input parameter types."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args, expected_types)):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} must be of type {expected_type}, got {type(arg)}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# --------------------------
# Export main decorators
# --------------------------

__all__ = [
    'validate_data_quality',
    'validate_data_quality_at_step',
    'log_execution_time',
    'validate_input_types'
]
