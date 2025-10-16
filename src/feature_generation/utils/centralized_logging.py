"""
Centralized Logging Utility for Feature Generation

This module provides a centralized logging system that replaces the scattered
tprint imports throughout the feature generation codebase. It provides consistent
logging with proper error handling and fast-fail mechanisms.
"""

import logging
import sys
from typing import Any, Optional, Union
from functools import wraps
import traceback

# Try to import tprint, fallback to custom implementation
try:
    from tprint import tprint as _tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    _tprint = None

# Configure logging
logger = logging.getLogger(__name__)

class FeatureGenerationLogger:
    """
    Centralized logger for feature generation with fast-fail error handling.
    """

    def __init__(self, name: str = "feature_generation"):
        self.logger = logging.getLogger(name)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)

    def info(self, message: str, **kwargs):
        """Log info message with tprint if available."""
        self.logger.info(message)
        if TPRINT_AVAILABLE and _tprint:
            _tprint(f"INFO: {message}", **kwargs)
        else:
            print(f"INFO: {message}")

    def debug(self, message: str, **kwargs):
        """Log debug message with tprint if available."""
        self.logger.debug(message)
        if TPRINT_AVAILABLE and _tprint:
            _tprint(f"DEBUG: {message}", **kwargs)
        else:
            print(f"DEBUG: {message}")

    def warning(self, message: str, **kwargs):
        """Log warning message with tprint if available."""
        self.logger.warning(message)
        if TPRINT_AVAILABLE and _tprint:
            _tprint(f"WARNING: {message}", **kwargs)
        else:
            print(f"WARNING: {message}")

    def error(self, message: str, **kwargs):
        """Log error message with tprint if available."""
        self.logger.error(message)
        if TPRINT_AVAILABLE and _tprint:
            _tprint(f"ERROR: {message}", **kwargs)
        else:
            print(f"ERROR: {message}")

    def critical(self, message: str, **kwargs):
        """Log critical message with tprint if available."""
        self.logger.critical(message)
        if TPRINT_AVAILABLE and _tprint:
            _tprint(f"CRITICAL: {message}", **kwargs)
        else:
            print(f"CRITICAL: {message}")

    def exception(self, message: str, exc_info: bool = True, **kwargs):
        """Log exception with full traceback."""
        self.logger.exception(message, exc_info=exc_info)
        if TPRINT_AVAILABLE and _tprint:
            _tprint(f"EXCEPTION: {message}", **kwargs)
        else:
            print(f"EXCEPTION: {message}")
            if exc_info:
                traceback.print_exc()

# Global logger instance
_feature_logger = FeatureGenerationLogger()

def tprint(message: str, level: str = "info", **kwargs):
    """
    Centralized tprint function that replaces scattered tprint imports.

    Args:
        message: Message to log
        level: Log level (info, debug, warning, error, critical)
        **kwargs: Additional arguments for tprint
    """
    # Check if the message should be printed based on log level
    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, 
                  "error": logging.ERROR, "critical": logging.CRITICAL}
    message_level = log_levels.get(level, logging.INFO)
    
    # Only print if the message level is >= current log level
    if message_level >= _feature_logger.logger.level:
        if level == "info":
            _feature_logger.info(message, **kwargs)
        elif level == "debug":
            _feature_logger.debug(message, **kwargs)
        elif level == "warning":
            _feature_logger.warning(message, **kwargs)
        elif level == "error":
            _feature_logger.error(message, **kwargs)
        elif level == "critical":
            _feature_logger.critical(message, **kwargs)
        else:
            _feature_logger.info(message, **kwargs)

def log_function_call(func_name: str, **kwargs):
    """Log function call with parameters."""
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    tprint(f"Calling {func_name}({params})", level="debug")

def log_function_result(func_name: str, success: bool, duration: float = None, **kwargs):
    """Log function result."""
    status = "SUCCESS" if success else "FAILED"
    duration_str = f" in {duration:.3f}s" if duration else ""
    tprint(f"{func_name} {status}{duration_str}", level="info" if success else "error")

def fast_fail_error(message: str, exception_class: type = ValueError, **kwargs):
    """
    Fast fail with proper error handling instead of silent failures.

    Args:
        message: Error message
        exception_class: Exception class to raise
        **kwargs: Additional context
    """
    error_msg = f"FAST FAIL: {message}"
    if kwargs:
        error_msg += f" Context: {kwargs}"

    tprint(error_msg, level="error")
    raise exception_class(error_msg)

def safe_execute(func, *args, error_message: str = None, **kwargs):
    """
    Safely execute a function with proper error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        error_message: Custom error message
        **kwargs: Function keyword arguments

    Returns:
        Function result or None if failed
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Function {func.__name__} failed"
        tprint(f"{msg}: {str(e)}", level="error")
        return None

def validate_dataframe(data, required_columns: list = None, min_rows: int = 1):
    """
    Validate DataFrame with fast fail on errors.

    Args:
        data: DataFrame to validate
        required_columns: List of required columns
        min_rows: Minimum number of rows required

    Raises:
        ValueError: If validation fails
    """
    if data is None:
        fast_fail_error("DataFrame is None")

    if len(data) == 0:
        fast_fail_error("DataFrame is empty")

    if hasattr(data, 'shape') and data.shape[0] < min_rows:
        fast_fail_error(f"DataFrame has insufficient rows: {data.shape[0]} < {min_rows}")

    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            fast_fail_error(f"Missing required columns: {missing_columns}")

def log_performance(func_name: str, data_shape: tuple = None, duration: float = None):
    """Log performance metrics."""
    shape_str = f" (shape: {data_shape})" if data_shape else ""
    duration_str = f" in {duration:.3f}s" if duration else ""
    tprint(f"Performance: {func_name}{shape_str}{duration_str}", level="debug")

# Decorator for automatic function logging
def log_function_execution(level: str = "debug"):
    """Decorator to automatically log function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_function_call(func.__name__, **kwargs)
            start_time = None
            if level in ["debug", "info"]:
                import time
                start_time = time.time()

            try:
                result = func(*args, **kwargs)
                if start_time:
                    duration = time.time() - start_time
                    log_function_result(func.__name__, True, duration)
                return result
            except Exception as e:
                if start_time:
                    duration = time.time() - start_time
                    log_function_result(func.__name__, False, duration)
                tprint(f"Function {func.__name__} failed: {str(e)}", level="error")
                raise
        return wrapper
    return decorator

# Export the main functions
__all__ = [
    'tprint',
    'log_function_call',
    'log_function_result',
    'fast_fail_error',
    'safe_execute',
    'validate_dataframe',
    'log_performance',
    'log_function_execution',
    'FeatureGenerationLogger'
]
