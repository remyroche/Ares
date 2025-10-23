"""
Comprehensive error handling and validation system.

This module provides robust error handling to ensure no silent failures
and comprehensive validation with detailed logging.
"""

import logging
from typing import Any, Optional, Union, List, Dict, Callable, Tuple
from functools import wraps
import traceback

# Import tprint utilities
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs):
        print(*args)

from .logging_config import get_logger

logger = logging.getLogger(__name__)

class FeaturesCommonError(Exception):
    """Base exception for features_common errors."""
    pass

class ValidationError(FeaturesCommonError):
    """Exception raised when data validation fails."""
    pass

class OptimizationError(FeaturesCommonError):
    """Exception raised when optimization fails."""
    pass

class VectorBTError(FeaturesCommonError):
    """Exception raised when VectorBT operations fail."""
    pass

class ConfigurationError(FeaturesCommonError):
    """Exception raised when configuration is invalid."""
    pass

class SilentFailureError(FeaturesCommonError):
    """Exception raised when a silent failure is detected."""
    pass

def ensure_no_silent_failures(func: Callable) -> Callable:
    """
    Decorator to ensure no silent failures in functions.

    This decorator wraps functions to catch and log all exceptions,
    ensuring that no failures go unnoticed.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger_instance = get_logger()
        func_name = func.__name__

        try:
            if TPRINT_AVAILABLE:
                tprint(f"🔧 [ErrorHandler] Executing {func_name}", color="cyan")

            result = func(*args, **kwargs)

            if TPRINT_AVAILABLE:
                tprint(f"✅ [ErrorHandler] {func_name} completed successfully", color="green")

            return result

        except Exception as e:
            error_msg = f"Silent failure detected in {func_name}: {e}"

            if TPRINT_AVAILABLE:
                tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
                tprint(f"   Traceback: {traceback.format_exc()}", color="red")
            else:
                logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")

            logger_instance.log_error(error_msg, "ErrorHandler")

            # Re-raise as SilentFailureError to ensure it's not ignored
            raise SilentFailureError(error_msg) from e

    return wrapper

def validate_input_data(data: Any, data_name: str = "data",
                       required_type: Optional[type] = None,
                       allow_empty: bool = False) -> Tuple[bool, List[str]]:
    """
    Comprehensive input data validation with detailed logging.

    Args:
        data: Data to validate
        data_name: Name of the data for error messages
        required_type: Required type for the data
        allow_empty: Whether empty data is allowed

    Returns:
        Tuple of (is_valid, list_of_warnings)

    Raises:
        ValidationError: If critical validation fails
    """
    logger_instance = get_logger()
    warnings = []
    is_valid = True

    if TPRINT_AVAILABLE:
        tprint(f"🔍 [ErrorHandler] Validating {data_name}", color="cyan")

    try:
        # Check if data is None
        if data is None:
            error_msg = f"{data_name} is None"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
            logger_instance.log_error(error_msg, "Validation")
            raise ValidationError(error_msg)

        # Check if data is empty
        if hasattr(data, '__len__') and len(data) == 0:
            if not allow_empty:
                error_msg = f"{data_name} is empty"
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
                logger_instance.log_error(error_msg, "Validation")
                raise ValidationError(error_msg)
            else:
                warning_msg = f"{data_name} is empty (allowed)"
                warnings.append(warning_msg)
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [ErrorHandler] {warning_msg}", color="yellow")
                logger_instance.log_warning(warning_msg, "Validation")

        # Check required type
        if required_type is not None and not isinstance(data, required_type):
            error_msg = f"{data_name} must be of type {required_type.__name__}, got {type(data).__name__}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
            logger_instance.log_error(error_msg, "Validation")
            raise ValidationError(error_msg)

        # Additional validation for pandas objects
        if hasattr(data, 'isna'):
            import pandas as pd
            if isinstance(data, (pd.Series, pd.DataFrame)):
                na_count = data.isna().sum()
                if hasattr(na_count, 'sum'):
                    na_count = na_count.sum()

                if na_count > 0:
                    warning_msg = f"{data_name} contains {na_count} NaN values"
                    warnings.append(warning_msg)
                    if TPRINT_AVAILABLE:
                        tprint(f"⚠️  [ErrorHandler] {warning_msg}", color="yellow")
                    logger_instance.log_warning(warning_msg, "Validation")

        if TPRINT_AVAILABLE:
            tprint(f"✅ [ErrorHandler] {data_name} validation passed", color="green")

        logger_instance.log_validation(data_name, is_valid, warnings)
        return is_valid, warnings

    except ValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        error_msg = f"Validation failed for {data_name}: {e}"
        if TPRINT_AVAILABLE:
            tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
        logger_instance.log_error(error_msg, "Validation")
        raise ValidationError(error_msg) from e

def safe_execute(operation: Callable, *args, **kwargs) -> Tuple[Any, bool, Optional[str]]:
    """
    Safely execute an operation with comprehensive error handling.

    Args:
        operation: Function to execute
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation

    Returns:
        Tuple of (result, success, error_message)
    """
    logger_instance = get_logger()
    operation_name = getattr(operation, '__name__', 'unknown_operation')

    try:
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [ErrorHandler] Executing {operation_name}", color="cyan")

        result = operation(*args, **kwargs)

        if TPRINT_AVAILABLE:
            tprint(f"✅ [ErrorHandler] {operation_name} completed successfully", color="green")

        logger_instance.log_operation_success(operation_name)
        return result, True, None

    except Exception as e:
        error_msg = f"{operation_name} failed: {e}"

        if TPRINT_AVAILABLE:
            tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
            tprint(f"   Traceback: {traceback.format_exc()}", color="red")
        else:
            logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")

        logger_instance.log_operation_failure(operation_name, e)
        return None, False, error_msg

def validate_configuration(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate configuration dictionary with required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required configuration keys

    Raises:
        ConfigurationError: If configuration is invalid
    """
    logger_instance = get_logger()

    if TPRINT_AVAILABLE:
        tprint("🔍 [ErrorHandler] Validating configuration", color="cyan")

    try:
        if not isinstance(config, dict):
            error_msg = f"Configuration must be a dictionary, got {type(config).__name__}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
            logger_instance.log_error(error_msg, "Configuration")
            raise ConfigurationError(error_msg)

        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            error_msg = f"Configuration missing required keys: {missing_keys}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
            logger_instance.log_error(error_msg, "Configuration")
            raise ConfigurationError(error_msg)

        if TPRINT_AVAILABLE:
            tprint("✅ [ErrorHandler] Configuration validation passed", color="green")

        logger_instance.log_info("Configuration validation passed", "Configuration")

    except ConfigurationError:
        # Re-raise configuration errors
        raise
    except Exception as e:
        error_msg = f"Configuration validation failed: {e}"
        if TPRINT_AVAILABLE:
            tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
        logger_instance.log_error(error_msg, "Configuration")
        raise ConfigurationError(error_msg) from e

def check_system_health() -> Dict[str, Any]:
    """
    Check system health and report any issues.

    Returns:
        Dictionary containing health status and issues
    """
    logger_instance = get_logger()
    health_status = {
        'overall_health': 'healthy',
        'issues': [],
        'warnings': [],
        'recommendations': []
    }

    if TPRINT_AVAILABLE:
        tprint("🔍 [ErrorHandler] Checking system health", color="cyan")

    try:
        # Check Python version
        import sys
        if sys.version_info < (3, 7):
            health_status['issues'].append("Python version < 3.7 may cause compatibility issues")
            health_status['overall_health'] = 'warning'

        # Check required packages
        required_packages = ['pandas', 'numpy']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                health_status['issues'].append(f"Required package {package} not available")
                health_status['overall_health'] = 'critical'

        # Check VectorBT availability
        try:
            import vectorbt
            health_status['recommendations'].append("VectorBT is available for optimization")
        except ImportError:
            health_status['warnings'].append("VectorBT not available - using pandas fallback")
            if health_status['overall_health'] == 'healthy':
                health_status['overall_health'] = 'warning'

        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_status['warnings'].append(f"High memory usage: {memory.percent:.1f}%")
                if health_status['overall_health'] == 'healthy':
                    health_status['overall_health'] = 'warning'
        except ImportError:
            health_status['warnings'].append("psutil not available - cannot monitor memory usage")

        if TPRINT_AVAILABLE:
            tprint(f"✅ [ErrorHandler] System health check completed: {health_status['overall_health']}", color="green")

        logger_instance.log_info(f"System health: {health_status['overall_health']}", "HealthCheck")

        return health_status

    except Exception as e:
        error_msg = f"Health check failed: {e}"
        if TPRINT_AVAILABLE:
            tprint(f"❌ [ErrorHandler] {error_msg}", color="red")
        logger_instance.log_error(error_msg, "HealthCheck")

        health_status['overall_health'] = 'error'
        health_status['issues'].append(error_msg)
        return health_status

def report_silent_failures() -> Dict[str, Any]:
    """
    Report any detected silent failures.

    Returns:
        Dictionary containing silent failure statistics
    """
    logger_instance = get_logger()
    stats = logger_instance.get_stats()

    if TPRINT_AVAILABLE:
        tprint("📊 [ErrorHandler] Reporting silent failure statistics", color="cyan")
        tprint(f"   Total operations: {stats['total_operations']}", color="blue")
        tprint(f"   Successful operations: {stats['successful_operations']}", color="green")
        tprint(f"   Failed operations: {stats['failed_operations']}", color="red")
        tprint(f"   Warnings issued: {stats['warnings_issued']}", color="yellow")
        tprint(f"   Errors handled: {stats['errors_handled']}", color="red")

    return stats
