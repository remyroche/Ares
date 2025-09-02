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
from typing import Any, Callable, Dict, Optional, Union, List
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
    passend_memory, _get_memory_usage()
metrics['memory_delta_mb'] = end_memory - start_memory
metrics['peak_memory_mb'] = end_memory

_log_performance_metrics(metrics, level)

return monitored_wrapper

def _get_memory_usage(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
import psutil
process, psutil.Process()
return process.memory_info().rss / 1024 / 1024
except ImportError:
    passpassreturn 0.0

def _log_performance_metrics(...):
    passdef _log_performance_metrics(...):
    passdef _log_performance_metrics(...):
    passdef _log_performance_metrics(...):
    pass"""Log performance metrics based on level."""
if level == "basic":
    passlogging.info(f"Performance: {metrics['function']} took {metrics['execution_time']:.3f}s")
elif level == "detailed":
    passpasslogging.info(f"Performance details for {metrics['function']}: {metrics}")
elif level == "profiling":
    passpasslogging.debug(f"Performance profiling for {metrics['function']}: {metrics}")


# --------------------------
# Enhanced Data Quality Decorators
# --------------------------

def validate_data_quality(
    check_nan: bool = True,
    check_infinite: bool = True,
    check_constant: bool = True,
    check_timestamps: bool = False,
    context: str = "default",
    cache_enabled: bool = True,
    performance_monitoring: str = "basic"
):
    """
    Enhanced decorator to validate data quality with specific parameters.
    
    Args:
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        check_constant: Whether to check for constant columns
        check_timestamps: Whether to check timestamp consistency
        context: Context for logging
        cache_enabled: Whether to enable caching
        performance_monitoring: Level of performance monitoring
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{context}")
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Validate output data quality
            if result is not None:
                issues = await _validate_data_quality(
                    result, 
                    check_nan=check_nan,
                    check_infinite=check_infinite,
                    check_constant=check_constant,
                    check_timestamps=check_timestamps
                )
                
                if issues:
                    logger.warning(f"Data quality issues detected: {len(issues)} issues")
                    for issue in issues[:5]:  # Log first 5 issues
                        logger.warning(f"  - {issue}")
                else:
                    logger.info("Data quality validation passed")
            
            return result

        # Apply enhancements
        enhanced_wrapper = wrapper
        
        if cache_enabled:
            enhanced_wrapper = _apply_caching(enhanced_wrapper)
        
        if performance_monitoring != "none":
            enhanced_wrapper = _apply_performance_monitoring(enhanced_wrapper, performance_monitoring)
        
        # Register decorator if enhanced system is available
        _register_decorator_if_available(
            f"validate_data_quality_{context}",
            enhanced_wrapper,
            description="Enhanced data quality validation with caching and performance monitoring",
            tags=["validation", "data-quality", "enhanced"]
        )
        
        return enhanced_wrapper
    return decorator

async def _validate_and_execute(func: Callable, args: tuple, kwargs: dict, validation_config: dict) -> Any:
    """Validate data and execute function."""
    logger = system_logger.getChild("ValidateAndExecute")
    
    # Validate input if requested
    if validation_config.get('validate_input', True):
        input_issues = await _validate_data_quality(args[0] if args else None)
        if input_issues:
            logger.warning(f"Input validation issues: {len(input_issues)}")
    
    # Execute function
    result = await func(*args, **kwargs)
    
    # Validate output if requested
    if validation_config.get('validate_output', True):
        output_issues = await _validate_data_quality(result)
        if output_issues:
            logger.warning(f"Output validation issues: {len(output_issues)}")
    
    return result

def validate_data_quality_at_step(
    step_name: str,
    validate_input: bool = True,
    validate_output: bool = True,
    fail_on_issues: bool = False,
    log_issues: bool = True,
    cache_enabled: bool = True,
    performance_monitoring: str = "basic"
):
    """
    Enhanced step-based data quality validation with intelligent caching.
    
    Args:
        step_name: Name of the pipeline step
        validate_input: Whether to validate input data
        validate_output: Whether to validate output data
        fail_on_issues: Whether to fail the step on quality issues
        log_issues: Whether to log quality issues
        cache_enabled: Whether to enable caching
        performance_monitoring: Level of performance monitoring
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = system_logger.getChild(f"DataQuality.{step_name}")
            
            # Validate input data if requested
            if validate_input and args:
                input_data = args[0]
                if input_data is not None:
                    input_issues = await _validate_data_quality(input_data)
                    if input_issues:
                        logger.warning(f"Input validation issues for {step_name}: {len(input_issues)}")
                        if fail_on_issues:
                            raise ValueError(f"Data quality validation failed for {step_name} input")
            
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Validate output data if requested
            if validate_output and result is not None:
                output_issues = await _validate_data_quality(result)
                if output_issues:
                    logger.warning(f"Output validation issues for {step_name}: {len(output_issues)}")
                    if log_issues:
                        for issue in output_issues[:5]:
                            logger.warning(f"  - {issue}")
                    if fail_on_issues:
                        raise ValueError(f"Data quality validation failed for {step_name} output")
                else:
                    logger.info(f"Data quality validation passed for {step_name}")
            
            return result

        # Apply enhancements
        enhanced_wrapper = wrapper
        
        if cache_enabled:
            enhanced_wrapper = _apply_caching(enhanced_wrapper)
        
        if performance_monitoring != "none":
            enhanced_wrapper = _apply_performance_monitoring(enhanced_wrapper, performance_monitoring)
        
        # Register decorator if enhanced system is available
        _register_decorator_if_available(
            f"validate_data_quality_at_step_{step_name}",
            enhanced_wrapper,
            description=f"Step-based data quality validation for {step_name}",
            tags=["validation", "data-quality", "step-based", "enhanced"]
        )
        
        return enhanced_wrapper
    return decorator

# --------------------------
# Data Quality Validation Functions
# --------------------------

async def _validate_data_quality(data: Any, **kwargs) -> List[str]:
    """Validate data quality and return list of issues."""
    issues = []
    
    if data is None:
        return issues
    
    if isinstance(data, pd.DataFrame):
        issues.extend(await _validate_dataframe_quality(data, **kwargs))
    elif isinstance(data, pd.Series):
        issues.extend(await _validate_series_quality(data, **kwargs))
    elif isinstance(data, np.ndarray):
        issues.extend(await _validate_array_quality(data, **kwargs))
    elif isinstance(data, (list, tuple)):
        issues.extend(await _validate_sequence_quality(data, **kwargs))
    
    return issues

async def _validate_dataframe_quality(df: pd.DataFrame, **kwargs) -> List[str]:
    """Validate DataFrame quality."""
    issues = []
    
    # Check for empty DataFrame
    if df.empty:
        issues.append("DataFrame is empty")
        return issues
    
    # Check for NaN values
    if kwargs.get('check_nan', True):
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            high_nan_cols = nan_counts[nan_counts > 0]
            for col, count in high_nan_cols.items():
                ratio = count / len(df)
                if ratio > 0.5:
                    issues.append(f"Column '{col}' has {ratio:.1%} NaN values")
    
    # Check for infinite values
    if kwargs.get('check_infinite', True):
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            high_inf_cols = inf_counts[inf_counts > 0]
            for col, count in high_inf_cols.items():
                issues.append(f"Column '{col}' has {count} infinite values")
    
    # Check for constant columns
    if kwargs.get('check_constant', True):
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"Column '{col}' is constant (single value)")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"DataFrame has {duplicate_count} duplicate rows")
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types in object columns
            unique_types = df[col].apply(type).nunique()
            if unique_types > 1:
                issues.append(f"Column '{col}' has mixed data types")
    
    return issues

async def _validate_series_quality(series: pd.Series, **kwargs) -> List[str]:
    """Validate Series quality."""
    issues = []
    
    if series.empty:
        issues.append("Series is empty")
        return issues
    
    # Check for NaN values
    if kwargs.get('check_nan', True):
        nan_count = series.isna().sum()
        if nan_count > 0:
            ratio = nan_count / len(series)
            if ratio > 0.5:
                issues.append(f"Series has {ratio:.1%} NaN values")
    
    # Check for infinite values (numeric series only)
    if kwargs.get('check_infinite', True) and pd.api.types.is_numeric_dtype(series):
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            issues.append(f"Series has {inf_count} infinite values")
    
    # Check for constant values
    if kwargs.get('check_constant', True):
        if series.nunique() == 1:
            issues.append("Series is constant (single value)")
    
    # Check for duplicate values
    duplicate_count = series.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Series has {duplicate_count} duplicate values")
    
    return issues

async def _validate_array_quality(arr: np.ndarray, **kwargs) -> List[str]:
    """Validate numpy array quality."""
    issues = []
    
    if arr.size == 0:
        issues.append("Array is empty")
        return issues
    
    # Check for NaN values
    if kwargs.get('check_nan', True):
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            ratio = nan_count / arr.size
            if ratio > 0.5:
                issues.append(f"Array has {ratio:.1%} NaN values")
    
    # Check for infinite values
    if kwargs.get('check_infinite', True):
        inf_count = np.isinf(arr).sum()
        if inf_count > 0:
            issues.append(f"Array has {inf_count} infinite values")
    
    # Check for constant values
    if kwargs.get('check_constant', True):
        if np.all(arr == arr.flat[0]):
            issues.append("Array is constant (single value)")
    
    return issues

async def _validate_sequence_quality(seq: Union[List, Tuple], **kwargs) -> List[str]:
    """Validate sequence quality."""
    issues = []
    
    if len(seq) == 0:
        issues.append("Sequence is empty")
        return issues
    
    # Check for None values
    if kwargs.get('check_nan', True):
        none_count = sum(1 for item in seq if item is None)
        if none_count > 0:
            ratio = none_count / len(seq)
            if ratio > 0.5:
                issues.append(f"Sequence has {ratio:.1%} None values")
    
    # Check for constant values
    if kwargs.get('check_constant', True):
        if len(set(seq)) == 1:
            issues.append("Sequence is constant (single value)")
    
    # Check for duplicate values
    duplicate_count = len(seq) - len(set(seq))
    if duplicate_count > 0:
        issues.append(f"Sequence has {duplicate_count} duplicate values")
    
    return issues

# --------------------------
# Utility Functions
# --------------------------

def is_boolean_feature(series: pd.Series) -> bool:
    """Check if a series represents a boolean feature."""
    # Check if it's already boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return True
    
    # Check if it has only 2 unique values
    unique_values = series.dropna().unique()
    if len(unique_values) == 2:
        # Check if values are boolean-like
        bool_like = all(str(val).lower() in ['true', 'false', '1', '0', 'yes', 'no'] 
                       for val in unique_values)
        if bool_like:
            return True
    
    return False

def get_data_quality_score(issues: List[str]) -> float:
    """Calculate a data quality score based on issues."""
    if not issues:
        return 1.0
    
    # Weight different types of issues
    critical_issues = sum(1 for issue in issues if 'empty' in issue.lower() or 'constant' in issue.lower())
    warning_issues = len(issues) - critical_issues
    
    # Calculate score (0.0 to 1.0)
    score = 1.0 - (critical_issues * 0.3 + warning_issues * 0.1)
    return max(0.0, min(1.0, score))

def format_quality_report(issues: List[str], data_shape: tuple = None) -> str:
    """Format a data quality report."""
    if not issues:
        return "✅ Data quality validation passed - no issues found"
    
    report = f"⚠️  Data quality validation found {len(issues)} issues:\n"
    
    if data_shape:
        report += f"Data shape: {data_shape}\n\n"
    
    for i, issue in enumerate(issues, 1):
        report += f"{i}. {issue}\n"
    
    return report

# --------------------------
# Decorator Registration
# --------------------------

# Register main decorators if enhanced system is available
if ENHANCED_SYSTEM_AVAILABLE:
    _register_decorator_if_available(
        "validate_data_quality",
        validate_data_quality,
        description="Enhanced data quality validation decorator",
        tags=["validation", "data-quality", "enhanced"]
    )
    
    _register_decorator_if_available(
        "validate_data_quality_at_step",
        validate_data_quality_at_step,
        description="Step-based data quality validation decorator",
        tags=["validation", "data-quality", "step-based", "enhanced"]
    )
