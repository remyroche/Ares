#!/usr/bin/env python3

"""
Fast Failing Validation Utility

This module provides comprehensive validation utilities that fail fast on errors
and provide detailed error reporting with extensive logging using tprint.

Key Features:
- Fast-failing validation with detailed error messages
- Comprehensive input validation
- Data quality checks with immediate failure
- Configuration validation
- Performance monitoring with validation
- Integration with tprint for extensive logging
- Silent failure detection and prevention
- Memory and resource validation
"""

import numpy as np
import pandas as pd
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import warnings
from functools import wraps
import inspect
import gc
import psutil
import sys

# Enhanced logging with tprint
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_performance, tprint_timer, tprint_exception
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions for when tprint is not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)
    def tprint_timer(*args, **kwargs): print("TIMER:", *args, **kwargs)
    def tprint_exception(*args, **kwargs): print("EXCEPTION:", *args, **kwargs)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for different strictness."""
    STRICT = "strict"      # Fail on any issue
    MODERATE = "moderate"  # Fail on significant issues
    LENIENT = "lenient"    # Only fail on critical issues
    DEBUG = "debug"        # Verbose logging, fail on any issue


class ValidationError(Exception):
    """Custom exception for validation failures."""
    
    def __init__(self, message: str, validation_type: str = "general", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.validation_type = validation_type
        self.details = details or {}
        self.timestamp = time.time()
        
        # Log the error with tprint
        tprint_error(f"ValidationError [{validation_type}]: {message}")
        if details:
            tprint_debug(f"Validation details: {details}")


class SilentFailureError(Exception):
    """Exception for silent failures that should be caught."""
    
    def __init__(self, message: str, operation: str = "unknown"):
        super().__init__(message)
        self.operation = operation
        self.timestamp = time.time()
        
        # Log the silent failure
        tprint_error(f"SilentFailureError in {operation}: {message}")


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    is_valid: bool
    message: str
    validation_type: str
    details: Dict[str, Any]
    timestamp: float
    duration: float
    
    def __post_init__(self):
        """Log validation result."""
        if self.is_valid:
            tprint_success(f"✅ Validation passed [{self.validation_type}]: {self.message}")
        else:
            tprint_error(f"❌ Validation failed [{self.validation_type}]: {self.message}")


@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""
    
    level: ValidationLevel = ValidationLevel.MODERATE
    enable_memory_checks: bool = True
    enable_performance_checks: bool = True
    enable_data_quality_checks: bool = True
    enable_configuration_checks: bool = True
    max_memory_usage_gb: float = 8.0
    max_validation_time_seconds: float = 30.0
    enable_silent_failure_detection: bool = True
    fail_fast: bool = True
    detailed_error_messages: bool = True
    log_validation_steps: bool = True


class FastFailingValidator:
    """
    Fast-failing validator with comprehensive validation capabilities.
    
    This validator provides extensive validation with immediate failure
    on errors and detailed logging using tprint.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the fast-failing validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.validation_history: List[ValidationResult] = []
        self.silent_failures: List[SilentFailureError] = []
        
        tprint_info("🚀 Initializing FastFailingValidator with enhanced logging")
        tprint_debug(f"Validation config: {self.config}")
        
        # Initialize performance tracking
        self._start_time = time.time()
        self._validation_count = 0
        
        tprint_success("✅ FastFailingValidator initialized")
    
    def validate_dataframe(self, df: Any, name: str = "dataframe", 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 1, max_rows: Optional[int] = None,
                          allow_nulls: bool = True) -> ValidationResult:
        """
        Validate a pandas DataFrame with fast-failing checks.
        
        Args:
            df: DataFrame to validate
            name: Name for logging purposes
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed
            allow_nulls: Whether null values are allowed
            
        Returns:
            ValidationResult with validation details
            
        Raises:
            ValidationError: If validation fails
        """
        tprint_debug(f"🔍 Validating {name} DataFrame")
        start_time = time.time()
        
        try:
            # Check if it's a DataFrame
            if not isinstance(df, pd.DataFrame):
                raise ValidationError(
                    f"{name} is not a pandas DataFrame, got {type(df)}",
                    "dataframe_type",
                    {"expected_type": "pandas.DataFrame", "actual_type": str(type(df))}
                )
            
            # Check if DataFrame is empty
            if df.empty:
                if min_rows > 0:
                    raise ValidationError(
                        f"{name} is empty but minimum {min_rows} rows required",
                        "empty_dataframe",
                        {"min_rows": min_rows, "actual_rows": 0}
                    )
                else:
                    tprint_warning(f"⚠️ {name} is empty but allowed")
            
            # Check row count
            actual_rows = len(df)
            if actual_rows < min_rows:
                raise ValidationError(
                    f"{name} has {actual_rows} rows but minimum {min_rows} required",
                    "insufficient_rows",
                    {"min_rows": min_rows, "actual_rows": actual_rows}
                )
            
            if max_rows and actual_rows > max_rows:
                raise ValidationError(
                    f"{name} has {actual_rows} rows but maximum {max_rows} allowed",
                    "excessive_rows",
                    {"max_rows": max_rows, "actual_rows": actual_rows}
                )
            
            # Check required columns
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise ValidationError(
                        f"{name} missing required columns: {missing_columns}",
                        "missing_columns",
                        {"required_columns": required_columns, "missing": list(missing_columns)}
                    )
            
            # Check for null values if not allowed
            if not allow_nulls:
                null_columns = df.columns[df.isnull().any()].tolist()
                if null_columns:
                    raise ValidationError(
                        f"{name} contains null values in columns: {null_columns}",
                        "null_values",
                        {"null_columns": null_columns}
                    )
            
            # Memory usage check
            if self.config.enable_memory_checks:
                memory_usage = df.memory_usage(deep=True).sum() / (1024**3)  # GB
                if memory_usage > self.config.max_memory_usage_gb:
                    raise ValidationError(
                        f"{name} memory usage {memory_usage:.2f}GB exceeds limit {self.config.max_memory_usage_gb}GB",
                        "memory_usage",
                        {"memory_usage_gb": memory_usage, "limit_gb": self.config.max_memory_usage_gb}
                    )
            
            duration = time.time() - start_time
            result = ValidationResult(
                is_valid=True,
                message=f"{name} validation passed",
                validation_type="dataframe",
                details={
                    "rows": actual_rows,
                    "columns": len(df.columns),
                    "memory_usage_gb": memory_usage if self.config.enable_memory_checks else None,
                    "duration_seconds": duration
                },
                timestamp=time.time(),
                duration=duration
            )
            
            self.validation_history.append(result)
            self._validation_count += 1
            
            tprint_success(f"✅ {name} DataFrame validation passed ({actual_rows} rows, {len(df.columns)} columns)")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error validating {name}: {str(e)}"
            tprint_exception(e, f"DataFrame validation error for {name}")
            
            raise ValidationError(
                error_msg,
                "unexpected_error",
                {"error_type": type(e).__name__, "error_message": str(e), "duration_seconds": duration}
            )
    
    def validate_numpy_array(self, arr: Any, name: str = "array",
                            expected_shape: Optional[Tuple[int, ...]] = None,
                            expected_dtype: Optional[Type] = None,
                            allow_empty: bool = False) -> ValidationResult:
        """
        Validate a numpy array with fast-failing checks.
        
        Args:
            arr: Array to validate
            name: Name for logging purposes
            expected_shape: Expected shape tuple
            expected_dtype: Expected data type
            allow_empty: Whether empty arrays are allowed
            
        Returns:
            ValidationResult with validation details
            
        Raises:
            ValidationError: If validation fails
        """
        tprint_debug(f"🔍 Validating {name} numpy array")
        start_time = time.time()
        
        try:
            # Check if it's a numpy array
            if not isinstance(arr, np.ndarray):
                raise ValidationError(
                    f"{name} is not a numpy array, got {type(arr)}",
                    "array_type",
                    {"expected_type": "numpy.ndarray", "actual_type": str(type(arr))}
                )
            
            # Check if array is empty
            if arr.size == 0:
                if not allow_empty:
                    raise ValidationError(
                        f"{name} is empty but empty arrays not allowed",
                        "empty_array",
                        {"allow_empty": allow_empty, "size": arr.size}
                    )
                else:
                    tprint_warning(f"⚠️ {name} is empty but allowed")
            
            # Check shape
            if expected_shape and arr.shape != expected_shape:
                raise ValidationError(
                    f"{name} shape {arr.shape} does not match expected {expected_shape}",
                    "shape_mismatch",
                    {"expected_shape": expected_shape, "actual_shape": arr.shape}
                )
            
            # Check dtype
            if expected_dtype and not np.issubdtype(arr.dtype, expected_dtype):
                raise ValidationError(
                    f"{name} dtype {arr.dtype} does not match expected {expected_dtype}",
                    "dtype_mismatch",
                    {"expected_dtype": expected_dtype, "actual_dtype": arr.dtype}
                )
            
            # Check for NaN values
            if np.any(np.isnan(arr)):
                nan_count = np.sum(np.isnan(arr))
                raise ValidationError(
                    f"{name} contains {nan_count} NaN values",
                    "nan_values",
                    {"nan_count": int(nan_count), "total_elements": arr.size}
                )
            
            # Check for infinite values
            if np.any(np.isinf(arr)):
                inf_count = np.sum(np.isinf(arr))
                raise ValidationError(
                    f"{name} contains {inf_count} infinite values",
                    "infinite_values",
                    {"inf_count": int(inf_count), "total_elements": arr.size}
                )
            
            duration = time.time() - start_time
            result = ValidationResult(
                is_valid=True,
                message=f"{name} array validation passed",
                validation_type="numpy_array",
                details={
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                    "size": arr.size,
                    "duration_seconds": duration
                },
                timestamp=time.time(),
                duration=duration
            )
            
            self.validation_history.append(result)
            self._validation_count += 1
            
            tprint_success(f"✅ {name} array validation passed (shape: {arr.shape}, dtype: {arr.dtype})")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error validating {name}: {str(e)}"
            tprint_exception(e, f"Array validation error for {name}")
            
            raise ValidationError(
                error_msg,
                "unexpected_error",
                {"error_type": type(e).__name__, "error_message": str(e), "duration_seconds": duration}
            )
    
    def validate_configuration(self, config: Any, name: str = "config",
                             required_fields: Optional[List[str]] = None,
                             field_types: Optional[Dict[str, Type]] = None) -> ValidationResult:
        """
        Validate a configuration object with fast-failing checks.
        
        Args:
            config: Configuration object to validate
            name: Name for logging purposes
            required_fields: List of required field names
            field_types: Dictionary mapping field names to expected types
            
        Returns:
            ValidationResult with validation details
            
        Raises:
            ValidationError: If validation fails
        """
        tprint_debug(f"🔍 Validating {name} configuration")
        start_time = time.time()
        
        try:
            # Check if config is not None
            if config is None:
                raise ValidationError(
                    f"{name} configuration is None",
                    "null_config",
                    {"config_type": "None"}
                )
            
            # Check required fields
            if required_fields:
                missing_fields = []
                for field in required_fields:
                    if not hasattr(config, field):
                        missing_fields.append(field)
                
                if missing_fields:
                    raise ValidationError(
                        f"{name} missing required fields: {missing_fields}",
                        "missing_fields",
                        {"required_fields": required_fields, "missing": missing_fields}
                    )
            
            # Check field types
            if field_types:
                type_errors = []
                for field, expected_type in field_types.items():
                    if hasattr(config, field):
                        actual_value = getattr(config, field)
                        if not isinstance(actual_value, expected_type):
                            type_errors.append(f"{field}: expected {expected_type.__name__}, got {type(actual_value).__name__}")
                
                if type_errors:
                    raise ValidationError(
                        f"{name} type errors: {'; '.join(type_errors)}",
                        "type_errors",
                        {"type_errors": type_errors}
                    )
            
            duration = time.time() - start_time
            result = ValidationResult(
                is_valid=True,
                message=f"{name} configuration validation passed",
                validation_type="configuration",
                details={
                    "config_type": type(config).__name__,
                    "required_fields_checked": len(required_fields) if required_fields else 0,
                    "field_types_checked": len(field_types) if field_types else 0,
                    "duration_seconds": duration
                },
                timestamp=time.time(),
                duration=duration
            )
            
            self.validation_history.append(result)
            self._validation_count += 1
            
            tprint_success(f"✅ {name} configuration validation passed")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error validating {name}: {str(e)}"
            tprint_exception(e, f"Configuration validation error for {name}")
            
            raise ValidationError(
                error_msg,
                "unexpected_error",
                {"error_type": type(e).__name__, "error_message": str(e), "duration_seconds": duration}
            )
    
    def validate_function_call(self, func: Callable, args: tuple = (), kwargs: dict = None,
                             expected_return_type: Optional[Type] = None,
                             max_execution_time: Optional[float] = None) -> ValidationResult:
        """
        Validate a function call with fast-failing checks.
        
        Args:
            func: Function to call and validate
            args: Positional arguments
            kwargs: Keyword arguments
            expected_return_type: Expected return type
            max_execution_time: Maximum execution time in seconds
            
        Returns:
            ValidationResult with validation details
            
        Raises:
            ValidationError: If validation fails
        """
        tprint_debug(f"🔍 Validating function call: {func.__name__}")
        start_time = time.time()
        
        try:
            # Check if function is callable
            if not callable(func):
                raise ValidationError(
                    f"Object {func} is not callable",
                    "not_callable",
                    {"object_type": type(func).__name__}
                )
            
            # Execute function with timeout
            kwargs = kwargs or {}
            if max_execution_time:
                # Simple timeout implementation
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Function {func.__name__} exceeded {max_execution_time}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(max_execution_time))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                result = func(*args, **kwargs)
            
            # Check return type
            if expected_return_type and not isinstance(result, expected_return_type):
                raise ValidationError(
                    f"Function {func.__name__} returned {type(result)} but expected {expected_return_type}",
                    "return_type_mismatch",
                    {"expected_type": expected_return_type.__name__, "actual_type": type(result).__name__}
                )
            
            duration = time.time() - start_time
            result_data = ValidationResult(
                is_valid=True,
                message=f"Function {func.__name__} call validation passed",
                validation_type="function_call",
                details={
                    "function_name": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "execution_time_seconds": duration,
                    "return_type": type(result).__name__
                },
                timestamp=time.time(),
                duration=duration
            )
            
            self.validation_history.append(result_data)
            self._validation_count += 1
            
            tprint_success(f"✅ Function {func.__name__} call validation passed ({duration:.3f}s)")
            return result_data
            
        except ValidationError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Function {func.__name__} call failed: {str(e)}"
            tprint_exception(e, f"Function call validation error for {func.__name__}")
            
            raise ValidationError(
                error_msg,
                "function_call_error",
                {"function_name": func.__name__, "error_type": type(e).__name__, 
                 "error_message": str(e), "duration_seconds": duration}
            )
    
    def detect_silent_failures(self, operation: str, expected_result: Any = None,
                             check_return_value: bool = True) -> bool:
        """
        Detect silent failures in operations.
        
        Args:
            operation: Description of the operation
            expected_result: Expected result (if known)
            check_return_value: Whether to check if return value is None/empty
            
        Returns:
            True if silent failure detected, False otherwise
        """
        tprint_debug(f"🔍 Detecting silent failures in: {operation}")
        
        try:
            # Check for None return values
            if check_return_value and expected_result is None:
                tprint_warning(f"⚠️ Potential silent failure: {operation} returned None")
                silent_failure = SilentFailureError(f"Operation returned None", operation)
                self.silent_failures.append(silent_failure)
                return True
            
            # Check for empty results
            if hasattr(expected_result, '__len__') and len(expected_result) == 0:
                tprint_warning(f"⚠️ Potential silent failure: {operation} returned empty result")
                silent_failure = SilentFailureError(f"Operation returned empty result", operation)
                self.silent_failures.append(silent_failure)
                return True
            
            # Check for NaN results
            if isinstance(expected_result, (np.ndarray, pd.Series, pd.DataFrame)):
                if isinstance(expected_result, np.ndarray):
                    has_nan = np.any(np.isnan(expected_result))
                else:
                    has_nan = expected_result.isnull().any().any()
                
                if has_nan:
                    tprint_warning(f"⚠️ Potential silent failure: {operation} contains NaN values")
                    silent_failure = SilentFailureError(f"Operation contains NaN values", operation)
                    self.silent_failures.append(silent_failure)
                    return True
            
            tprint_debug(f"✅ No silent failures detected in: {operation}")
            return False
            
        except Exception as e:
            tprint_error(f"Error detecting silent failures in {operation}: {e}")
            return False
    
    def validate_memory_usage(self, operation: str = "operation") -> ValidationResult:
        """
        Validate current memory usage.
        
        Args:
            operation: Name of the operation for logging
            
        Returns:
            ValidationResult with memory usage details
            
        Raises:
            ValidationError: If memory usage exceeds limits
        """
        tprint_debug(f"🔍 Validating memory usage for: {operation}")
        start_time = time.time()
        
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024**3)
            
            # Check against limit
            if memory_usage_gb > self.config.max_memory_usage_gb:
                raise ValidationError(
                    f"Memory usage {memory_usage_gb:.2f}GB exceeds limit {self.config.max_memory_usage_gb}GB",
                    "memory_limit_exceeded",
                    {"memory_usage_gb": memory_usage_gb, "limit_gb": self.config.max_memory_usage_gb}
                )
            
            duration = time.time() - start_time
            result = ValidationResult(
                is_valid=True,
                message=f"Memory usage validation passed for {operation}",
                validation_type="memory_usage",
                details={
                    "memory_usage_gb": memory_usage_gb,
                    "limit_gb": self.config.max_memory_usage_gb,
                    "operation": operation,
                    "duration_seconds": duration
                },
                timestamp=time.time(),
                duration=duration
            )
            
            self.validation_history.append(result)
            self._validation_count += 1
            
            tprint_success(f"✅ Memory usage validation passed: {memory_usage_gb:.2f}GB")
            return result
            
        except ValidationError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Memory usage validation failed for {operation}: {str(e)}"
            tprint_exception(e, f"Memory validation error for {operation}")
            
            raise ValidationError(
                error_msg,
                "memory_validation_error",
                {"operation": operation, "error_type": type(e).__name__, 
                 "error_message": str(e), "duration_seconds": duration}
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validations performed.
        
        Returns:
            Dictionary with validation summary
        """
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v.is_valid)
        failed_validations = total_validations - successful_validations
        
        total_duration = sum(v.duration for v in self.validation_history)
        avg_duration = total_duration / total_validations if total_validations > 0 else 0
        
        validation_types = {}
        for v in self.validation_history:
            validation_types[v.validation_type] = validation_types.get(v.validation_type, 0) + 1
        
        summary = {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "validation_types": validation_types,
            "silent_failures_detected": len(self.silent_failures),
            "config": {
                "level": self.config.level.value,
                "fail_fast": self.config.fail_fast,
                "memory_checks": self.config.enable_memory_checks,
                "performance_checks": self.config.enable_performance_checks
            }
        }
        
        tprint_info(f"📊 Validation Summary: {successful_validations}/{total_validations} passed, {len(self.silent_failures)} silent failures")
        return summary
    
    def reset_validation_history(self):
        """Reset validation history and counters."""
        tprint_info("🔄 Resetting validation history")
        self.validation_history.clear()
        self.silent_failures.clear()
        self._validation_count = 0
        self._start_time = time.time()
        tprint_success("✅ Validation history reset")


# Global validator instance
_global_validator = None


def get_fast_failing_validator(config: Optional[ValidationConfig] = None) -> FastFailingValidator:
    """
    Get or create the global fast-failing validator.
    
    Args:
        config: Validation configuration
        
    Returns:
        FastFailingValidator instance
    """
    global _global_validator
    if _global_validator is None or config is not None:
        _global_validator = FastFailingValidator(config)
    return _global_validator


def validate_dataframe_fast_fail(df: Any, name: str = "dataframe", **kwargs) -> ValidationResult:
    """
    Quick validation function for DataFrames.
    
    Args:
        df: DataFrame to validate
        name: Name for logging
        **kwargs: Additional validation parameters
        
    Returns:
        ValidationResult
    """
    validator = get_fast_failing_validator()
    return validator.validate_dataframe(df, name, **kwargs)


def validate_array_fast_fail(arr: Any, name: str = "array", **kwargs) -> ValidationResult:
    """
    Quick validation function for numpy arrays.
    
    Args:
        arr: Array to validate
        name: Name for logging
        **kwargs: Additional validation parameters
        
    Returns:
        ValidationResult
    """
    validator = get_fast_failing_validator()
    return validator.validate_numpy_array(arr, name, **kwargs)


def validate_config_fast_fail(config: Any, name: str = "config", **kwargs) -> ValidationResult:
    """
    Quick validation function for configurations.
    
    Args:
        config: Configuration to validate
        name: Name for logging
        **kwargs: Additional validation parameters
        
    Returns:
        ValidationResult
    """
    validator = get_fast_failing_validator()
    return validator.validate_configuration(config, name, **kwargs)


def fast_fail_decorator(validation_type: str = "general", **validation_kwargs):
    """
    Decorator for fast-failing validation of function calls.
    
    Args:
        validation_type: Type of validation
        **validation_kwargs: Validation parameters
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tprint_debug(f"🔍 Fast-fail validation for {func.__name__}")
            
            validator = get_fast_failing_validator()
            
            try:
                # Validate inputs if specified
                if 'validate_inputs' in validation_kwargs and validation_kwargs['validate_inputs']:
                    for i, arg in enumerate(args):
                        if isinstance(arg, pd.DataFrame):
                            validator.validate_dataframe(arg, f"arg_{i}")
                        elif isinstance(arg, np.ndarray):
                            validator.validate_numpy_array(arg, f"arg_{i}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Validate output if specified
                if 'validate_output' in validation_kwargs and validation_kwargs['validate_output']:
                    if isinstance(result, pd.DataFrame):
                        validator.validate_dataframe(result, "result")
                    elif isinstance(result, np.ndarray):
                        validator.validate_numpy_array(result, "result")
                
                # Detect silent failures
                if 'detect_silent_failures' in validation_kwargs and validation_kwargs['detect_silent_failures']:
                    validator.detect_silent_failures(func.__name__, result)
                
                tprint_success(f"✅ Fast-fail validation passed for {func.__name__}")
                return result
                
            except Exception as e:
                tprint_exception(e, f"Fast-fail validation failed for {func.__name__}")
                raise
        
        return wrapper
    return decorator


# Export all functions and classes
__all__ = [
    'FastFailingValidator',
    'ValidationConfig',
    'ValidationResult',
    'ValidationError',
    'SilentFailureError',
    'ValidationLevel',
    'get_fast_failing_validator',
    'validate_dataframe_fast_fail',
    'validate_array_fast_fail',
    'validate_config_fast_fail',
    'fast_fail_decorator'
]