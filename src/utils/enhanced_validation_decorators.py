"""
Enhanced Validation Decorators for Comprehensive Pipeline Validation

This module provides enhanced decorators that integrate with BaseValidator and
provide comprehensive validation capabilities with better performance, error handling,
and consistency across all training steps.
"""

import asyncio
import functools
import inspect
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.utils.base_validator import BaseValidator
from src.utils.comprehensive_file_validation import ComprehensiveFileValidator
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards


class ValidationContext:
    """Context for validation operations with caching and performance tracking."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.validation_cache = {}
        self.performance_metrics = {}
        self.start_time = None

    def start_validation(self):
        """Start timing validation operation."""
        self.start_time = time.time()

    def end_validation(self, validation_type: str):
        """End timing and record performance."""
        if self.start_time:
            duration = time.time() - self.start_time
            if validation_type not in self.performance_metrics:
                self.performance_metrics[validation_type] = []
            self.performance_metrics[validation_type].append(duration)
            self.start_time = None


def comprehensive_step_validation(
    step_name: str,
    validate_prerequisites: bool = True,
    validate_inputs: bool = True,
    validate_outputs: bool = True,
    validate_data_quality: bool = True,
    cache_validation: bool = True,
    log_level: str = "INFO",
):
    """
    Comprehensive decorator for step validation that integrates with BaseValidator.

    Args:
        step_name: Name of the step for context
        validate_prerequisites: Whether to validate step prerequisites
        validate_inputs: Whether to validate input files/data
        validate_outputs: Whether to validate output files/data
        validate_data_quality: Whether to perform data quality checks
        cache_validation: Whether to cache validation results for performance
        log_level: Logging level for validation messages
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ValidationContext(step_name)
            logger = system_logger.getChild(f"EnhancedValidation.{step_name}")

            try:
                # Extract validator instance if available
                validator = _extract_validator_instance(args, kwargs)

                # Validate prerequisites
                if validate_prerequisites and validator:
                    context.start_validation()
                    prereq_result = await _validate_prerequisites_async(validator, args, kwargs, context)
                    context.end_validation("prerequisites")

                    if not prereq_result["validation_passed"]:
                        logger.error(f"❌ Prerequisites validation failed: {prereq_result['errors']}")
                        return await _handle_validation_failure(func, args, kwargs, prereq_result)

                # Validate inputs
                if validate_inputs and validator:
                    context.start_validation()
                    input_result = await _validate_inputs_async(validator, args, kwargs, context)
                    context.end_validation("inputs")

                    if not input_result["validation_passed"]:
                        logger.warning(f"⚠️ Input validation issues: {input_result['warnings']}")

                # Execute the function
                result = await func(*args, **kwargs)

                # Validate outputs
                if validate_outputs and validator:
                    context.start_validation()
                    output_result = await _validate_outputs_async(validator, result, context)
                    context.end_validation("outputs")

                    if not output_result["validation_passed"]:
                        logger.error(f"❌ Output validation failed: {output_result['errors']}")
                        return await _handle_validation_failure(func, args, kwargs, output_result)

                # Validate data quality
                if validate_data_quality and validator:
                    context.start_validation()
                    quality_result = await _validate_data_quality_async(validator, result, context)
                    context.end_validation("data_quality")

                    if not quality_result["validation_passed"]:
                        logger.warning(f"⚠️ Data quality issues: {quality_result['warnings']}")

                # Log performance metrics
                _log_validation_performance(context, logger, log_level)

                return result

            except Exception as e:
                logger.exception(f"❌ Validation error in {step_name}: {e}")
                # Fall back to original function execution
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            context = ValidationContext(step_name)
            logger = system_logger.getChild(f"EnhancedValidation.{step_name}")

            try:
                # Extract validator instance if available
                validator = _extract_validator_instance(args, kwargs)

                # Validate prerequisites
                if validate_prerequisites and validator:
                    context.start_validation()
                    prereq_result = _validate_prerequisites_sync(validator, args, kwargs, context)
                    context.end_validation("prerequisites")

                    if not prereq_result["validation_passed"]:
                        logger.error(f"❌ Prerequisites validation failed: {prereq_result['errors']}")
                        return _handle_validation_failure_sync(func, args, kwargs, prereq_result)

                # Validate inputs
                if validate_inputs and validator:
                    context.start_validation()
                    input_result = _validate_inputs_sync(validator, args, kwargs, context)
                    context.end_validation("inputs")

                    if not input_result["validation_passed"]:
                        logger.warning(f"⚠️ Input validation issues: {input_result['warnings']}")

                # Execute the function
                result = func(*args, **kwargs)

                # Validate outputs
                if validate_outputs and validator:
                    context.start_validation()
                    output_result = _validate_outputs_sync(validator, result, context)
                    context.end_validation("outputs")

                    if not output_result["validation_passed"]:
                        logger.error(f"❌ Output validation failed: {output_result['errors']}")
                        return _handle_validation_failure_sync(func, args, kwargs, output_result)

                # Validate data quality
                if validate_data_quality and validator:
                    context.start_validation()
                    quality_result = _validate_data_quality_sync(validator, result, context)
                    context.end_validation("data_quality")

                    if not quality_result["validation_passed"]:
                        logger.warning(f"⚠️ Data quality issues: {quality_result['warnings']}")

                # Log performance metrics
                _log_validation_performance(context, logger, log_level)

                return result

            except Exception as e:
                logger.exception(f"❌ Validation error in {step_name}: {e}")
                # Fall back to original function execution
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_with_base_validator(
    validator_class: type, validation_method: str = "validate", fallback_to_original: bool = True
):
    """
    Decorator that uses a specific BaseValidator class for validation.

    Args:
        validator_class: The BaseValidator class to use
        validation_method: The method name to call for validation
        fallback_to_original: Whether to fall back to original function if validation fails
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Create validator instance
                config = kwargs.get("config", {})
                validator = validator_class(config)

                # Run validation
                if hasattr(validator, validation_method):
                    validation_method_func = getattr(validator, validation_method)
                    validation_result = await validation_method_func(*args, **kwargs)

                    if not validation_result:
                        system_logger.warning(f"⚠️ Validation failed for {func.__name__}")
                        if not fallback_to_original:
                            raise ValueError(f"Validation failed for {func.__name__}")

                # Execute original function
                return await func(*args, **kwargs)

            except Exception as e:
                system_logger.exception(f"❌ Validation error in {func.__name__}: {e}")
                if fallback_to_original:
                    return await func(*args, **kwargs)
                else:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                # Create validator instance
                config = kwargs.get("config", {})
                validator = validator_class(config)

                # Run validation
                if hasattr(validator, validation_method):
                    validation_method_func = getattr(validator, validation_method)
                    validation_result = validation_method_func(*args, **kwargs)

                    if not validation_result:
                        system_logger.warning(f"⚠️ Validation failed for {func.__name__}")
                        if not fallback_to_original:
                            raise ValueError(f"Validation failed for {func.__name__}")

                # Execute original function
                return func(*args, **kwargs)

            except Exception as e:
                system_logger.exception(f"❌ Validation error in {func.__name__}: {e}")
                if fallback_to_original:
                    return func(*args, **kwargs)
                else:
                    raise

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def smart_validation_cache(
    cache_key_func: Optional[Callable] = None, ttl_seconds: int = 300, max_cache_size: int = 1000  # 5 minutes default
):
    """
    Smart caching decorator for validation results to improve performance.

    Args:
        cache_key_func: Function to generate cache key from function arguments
        ttl_seconds: Time to live for cache entries in seconds
        max_cache_size: Maximum number of cache entries
    """

    def decorator(func: Callable) -> Callable:
        # Initialize cache
        cache = {}
        cache_timestamps = {}

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = str(hash(str(args) + str(sorted(kwargs.items()))))

            # Check cache
            current_time = time.time()
            if cache_key in cache and cache_key in cache_timestamps:
                if current_time - cache_timestamps[cache_key] < ttl_seconds:
                    return cache[cache_key]
                else:
                    # Expired entry
                    del cache[cache_key]
                    del cache_timestamps[cache_key]

            # Execute function and cache result
            result = await func(*args, **kwargs)

            # Manage cache size
            if len(cache) >= max_cache_size:
                # Remove oldest entries
                oldest_key = min(cache_timestamps.keys(), key=lambda k: cache_timestamps[k])
                del cache[oldest_key]
                del cache_timestamps[oldest_key]

            # Cache result
            cache[cache_key] = result
            cache_timestamps[cache_key] = current_time

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = str(hash(str(args) + str(sorted(kwargs.items()))))

            # Check cache
            current_time = time.time()
            if cache_key in cache and cache_key in cache_timestamps:
                if current_time - cache_timestamps[cache_key] < ttl_seconds:
                    return cache[cache_key]
                else:
                    # Expired entry
                    del cache[cache_key]
                    del cache_timestamps[cache_key]

            # Execute function and cache result
            result = func(*args, **kwargs)

            # Manage cache size
            if len(cache) >= max_cache_size:
                # Remove oldest entries
                oldest_key = min(cache_timestamps.keys(), key=lambda k: cache_timestamps[k])
                del cache[oldest_key]
                del cache_timestamps[oldest_key]

            # Cache result
            cache[cache_key] = result
            cache_timestamps[cache_key] = current_time

            return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience decorators for specific steps
def validate_step1_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 1: Data Collection."""
    return comprehensive_step_validation(
        "step1_data_collection",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step1_5_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 1.5: Data Converter."""
    return comprehensive_step_validation(
        "step1_5_data_converter",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step2_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 2: Data Reading."""
    return comprehensive_step_validation(
        "step2_data_reading",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step3_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 3: HMM Regime Discovery."""
    return comprehensive_step_validation(
        "step3_hmm_regime_discovery",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step4_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 4: Regime Data Splitting."""
    return comprehensive_step_validation(
        "step4_regime_data_splitting",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step5_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 5: Labeling."""
    return comprehensive_step_validation(
        "step5_labeling",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step6_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 6: Feature Engineering."""
    return comprehensive_step_validation(
        "step6_feature_engineering",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


def validate_step7_comprehensive(func: Callable) -> Callable:
    """Comprehensive validation for Step 7: Enhanced Matrix Operations."""
    return comprehensive_step_validation(
        "step7_enhanced_matrix_operations",
        validate_prerequisites=True,
        validate_inputs=True,
        validate_outputs=True,
        validate_data_quality=True,
    )(func)


# Helper functions for validation decorators


def _extract_validator_instance(args: tuple, kwargs: dict) -> Optional[BaseValidator]:
    """Extract BaseValidator instance from function arguments."""
    # Look for validator in self parameter (for class methods)
    if args and hasattr(args[0], "__class__"):
        if issubclass(args[0].__class__, BaseValidator):
            return args[0]

    # Look for validator in keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, BaseValidator):
            return value

    return None


async def _validate_prerequisites_async(
    validator: BaseValidator, args: tuple, kwargs: dict, context: ValidationContext
) -> Dict[str, Any]:
    """Validate prerequisites asynchronously."""
    try:
        if hasattr(validator, "validate_step_prerequisites"):
            # Extract common parameters
            symbol = kwargs.get("symbol", "ETHUSDT")
            exchange = kwargs.get("exchange", "BINANCE")
            timeframe = kwargs.get("timeframe", "1m")

            return validator.validate_step_prerequisites(symbol, exchange, timeframe)
        else:
            return {"validation_passed": True, "warnings": [], "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


def _validate_prerequisites_sync(
    validator: BaseValidator, args: tuple, kwargs: dict, context: ValidationContext
) -> Dict[str, Any]:
    """Validate prerequisites synchronously."""
    try:
        if hasattr(validator, "validate_step_prerequisites"):
            # Extract common parameters
            symbol = kwargs.get("symbol", "ETHUSDT")
            exchange = kwargs.get("exchange", "BINANCE")
            timeframe = kwargs.get("timeframe", "1m")

            return validator.validate_step_prerequisites(symbol, exchange, timeframe)
        else:
            return {"validation_passed": True, "warnings": [], "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


async def _validate_inputs_async(
    validator: BaseValidator, args: tuple, kwargs: dict, context: ValidationContext
) -> Dict[str, Any]:
    """Validate inputs asynchronously."""
    try:
        # Extract file paths and validate
        file_paths = _extract_file_paths_from_args(args, kwargs)

        validation_results = []
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                file_validator = ComprehensiveFileValidator()
                result = file_validator.validate_file_format(file_path, None, validator.step_name)
                validation_results.append(result)

        # Aggregate results
        all_valid = all(r.is_valid for r in validation_results)
        warnings = []
        for result in validation_results:
            if not result.is_valid:
                warnings.extend([f"{issue.description}" for issue in result.issues])

        return {"validation_passed": all_valid, "warnings": warnings, "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


def _validate_inputs_sync(
    validator: BaseValidator, args: tuple, kwargs: dict, context: ValidationContext
) -> Dict[str, Any]:
    """Validate inputs synchronously."""
    try:
        # Extract file paths and validate
        file_paths = _extract_file_paths_from_args(args, kwargs)

        validation_results = []
        for file_path in file_paths:
            if file_path and Path(file_path).exists():
                file_validator = ComprehensiveFileValidator()
                result = file_validator.validate_file_format(file_path, None, validator.step_name)
                validation_results.append(result)

        # Aggregate results
        all_valid = all(r.is_valid for r in validation_results)
        warnings = []
        for result in validation_results:
            if not result.is_valid:
                warnings.extend([f"{issue.description}" for issue in result.issues])

        return {"validation_passed": all_valid, "warnings": warnings, "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


async def _validate_outputs_async(validator: BaseValidator, result: Any, context: ValidationContext) -> Dict[str, Any]:
    """Validate outputs asynchronously."""
    try:
        if hasattr(validator, "validate_step_output"):
            # Extract common parameters from context
            symbol = getattr(validator, "symbol", "ETHUSDT")
            exchange = getattr(validator, "exchange", "BINANCE")
            timeframe = getattr(validator, "timeframe", "1m")

            return validator.validate_step_output(symbol, exchange, timeframe)
        else:
            return {"validation_passed": True, "warnings": [], "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


def _validate_outputs_sync(validator: BaseValidator, result: Any, context: ValidationContext) -> Dict[str, Any]:
    """Validate outputs synchronously."""
    try:
        if hasattr(validator, "validate_step_output"):
            # Extract common parameters from context
            symbol = getattr(validator, "symbol", "symbol", "ETHUSDT")
            exchange = getattr(validator, "exchange", "BINANCE")
            timeframe = getattr(validator, "timeframe", "1m")

            return validator.validate_step_output(symbol, exchange, timeframe)
        else:
            return {"validation_passed": True, "warnings": [], "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


async def _validate_data_quality_async(
    validator: BaseValidator, result: Any, context: ValidationContext
) -> Dict[str, Any]:
    """Validate data quality asynchronously."""
    try:
        # Check if result contains DataFrames
        dataframes = _extract_dataframes_from_result(result)

        validation_results = []
        for df in dataframes:
            if hasattr(validator, "validate_dataframe_quality"):
                quality_result = validator.validate_dataframe_quality(
                    df,
                    min_rows=100,
                    required_columns=None,
                    check_data_types=True,
                    check_value_ranges=True,
                    check_duplicates=True,
                    check_temporal_consistency=True,
                )
                validation_results.append(quality_result)

        # Aggregate results
        all_valid = all(r[0] for r in validation_results)
        warnings = []
        for passed, metrics in validation_results:
            if not passed and "critical_issues" in metrics:
                warnings.extend(metrics["critical_issues"])

        return {"validation_passed": all_valid, "warnings": warnings, "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


def _validate_data_quality_sync(validator: BaseValidator, result: Any, context: ValidationContext) -> Dict[str, Any]:
    """Validate data quality synchronously."""
    try:
        # Check if result contains DataFrames
        dataframes = _extract_dataframes_from_result(result)

        validation_results = []
        for df in dataframes:
            if hasattr(validator, "validate_dataframe_quality"):
                quality_result = validator.validate_dataframe_quality(
                    df,
                    min_rows=100,
                    required_columns=None,
                    check_data_types=True,
                    check_value_ranges=True,
                    check_duplicates=True,
                    check_temporal_consistency=True,
                )
                validation_results.append(quality_result)

        # Aggregate results
        all_valid = all(r[0] for r in validation_results)
        warnings = []
        for passed, metrics in validation_results:
            if not passed and "critical_issues" in metrics:
                warnings.extend(metrics["critical_issues"])

        return {"validation_passed": all_valid, "warnings": warnings, "errors": []}
    except Exception as e:
        return {"validation_passed": False, "warnings": [], "errors": [str(e)]}


async def _handle_validation_failure(
    func: Callable, args: tuple, kwargs: dict, validation_result: Dict[str, Any]
) -> Any:
    """Handle validation failure for async functions."""
    # For now, log the failure and continue with original function
    # In production, you might want to raise an exception or take other action
    system_logger.warning(f"Validation failed but continuing with {func.__name__}: {validation_result}")
    return await func(*args, **kwargs)


def _handle_validation_failure_sync(
    func: Callable, args: tuple, kwargs: dict, validation_result: Dict[str, Any]
) -> Any:
    """Handle validation failure for sync functions."""
    # For now, log the failure and continue with original function
    # In production, you might want to raise an exception or take other action
    system_logger.warning(f"Validation failed but continuing with {func.__name__}: {validation_result}")
    return func(*args, **kwargs)


def _log_validation_performance(context: ValidationContext, logger: Any, log_level: str):
    """Log validation performance metrics."""
    if log_level.upper() == "DEBUG":
        for validation_type, times in context.performance_metrics.items():
            if times:
                avg_time = sum(times) / len(times)
                logger.debug(f"📊 {validation_type} validation: avg={avg_time:.3f}s, count={len(times)}")


def _extract_file_paths_from_args(args: tuple, kwargs: dict) -> List[str]:
    """Extract file paths from function arguments."""
    file_paths = []

    # Look for file paths in arguments
    for arg in args:
        if isinstance(arg, str) and _looks_like_file_path(arg):
            file_paths.append(arg)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, str) and _looks_like_file_path(item):
                    file_paths.append(item)

    # Look for file paths in keyword arguments
    file_keywords = ["file_path", "filepath", "path", "file", "filename", "data_dir", "output_dir"]
    for key, value in kwargs.items():
        if any(file_key in key.lower() for file_key in file_keywords):
            if isinstance(value, str) and _looks_like_file_path(value):
                file_paths.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and _looks_like_file_path(item):
                        file_paths.append(item)

    return file_paths


def _extract_dataframes_from_result(result: Any) -> List[Any]:
    """Extract DataFrames from function result."""
    dataframes = []

    if hasattr(result, "shape"):  # Single DataFrame result
        dataframes.append(result)
    elif isinstance(result, dict):
        for key, value in result.items():
            if hasattr(value, "shape"):  # DataFrame in dict
                dataframes.append(value)
    elif isinstance(result, (list, tuple)):
        for item in result:
            if hasattr(item, "shape"):  # DataFrame in list/tuple
                dataframes.append(item)

    return dataframes


def _looks_like_file_path(path: str) -> bool:
    """Check if a string looks like a file path."""
    if not isinstance(path, str):
        return False

    # Check for common file extensions
    file_extensions = [".parquet", ".csv", ".json", ".pkl", ".pickle", ".h5", ".hdf5"]
    return any(path.lower().endswith(ext) for ext in file_extensions) or "/" in path or "\\" in path
