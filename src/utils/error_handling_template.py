"""
Error Handling Template and Utilities

This module provides standardized error handling patterns to replace
bare except clauses and silent failures throughout the codebase.
"""

import logging
import functools
from typing import Callable, Any, Optional, TypeVar, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')

def with_error_logging(
    fallback_value: Any = None,
    log_level: str = "warning",
    context: str = "",
    raise_on_error: bool = False
):
    """
    Decorator to add standardized error logging to functions.

    Args:
        fallback_value: Value to return if function fails
        log_level: Logging level ('debug', 'info', 'warning', 'error')
        context: Additional context for the error message
        raise_on_error: Whether to re-raise the exception after logging

    Usage:
        @with_error_logging(fallback_value=None, context="Data processing")
        def risky_operation(data):
            # ... risky code ...
            return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context_msg = f" in {context}" if context else ""
                logger_func = getattr(logger, log_level, logger.warning)
                logger_func(f"⚠️ {func.__name__} failed{context_msg}: {e}")

                if raise_on_error:
                    raise
                return fallback_value
        return wrapper
    return decorator

@contextmanager
def safe_operation_context(operation_name: str, log_level: str = "warning"):
    """
    Context manager for safe operations with error logging.

    Args:
        operation_name: Name of the operation for logging
        log_level: Logging level for errors

    Usage:
        with safe_operation_context("Data processing"):
            # ... risky code ...
    """
    try:
        logger.debug(f"🔄 Starting {operation_name}")
        yield
        logger.debug(f"✅ {operation_name} completed successfully")
    except Exception as e:
        logger_func = getattr(logger, log_level, logger.warning)
        logger_func(f"❌ {operation_name} failed: {e}")
        raise

def safe_call(
    func: Callable[..., T],
    *args,
    fallback_value: Any = None,
    log_error: bool = True,
    error_context: str = "",
    **kwargs
) -> Union[T, Any]:
    """
    Safely call a function with error handling.

    Args:
        func: Function to call
        *args: Positional arguments for the function
        fallback_value: Value to return if function fails
        log_error: Whether to log the error
        error_context: Additional context for the error message
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or fallback_value if it fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            context_msg = f" in {error_context}" if error_context else ""
            logger.warning(f"⚠️ {func.__name__}{context_msg} failed: {e}")
        return fallback_value

def replace_bare_except(
    original_function: Callable,
    error_message: str = "Operation failed",
    log_level: str = "warning"
) -> Callable:
    """
    Replace bare except clauses with proper error handling.

    Args:
        original_function: Function containing bare except clauses
        error_message: Custom error message
        log_level: Logging level for errors

    Returns:
        Function with improved error handling
    """
    @functools.wraps(original_function)
    def wrapper(*args, **kwargs):
        try:
            return original_function(*args, **kwargs)
        except Exception as e:
            logger_func = getattr(logger, log_level, logger.warning)
            logger_func(f"⚠️ {error_message}: {e}")
            raise  # Re-raise to maintain original behavior
    return wrapper

# Common error handling patterns
class ErrorHandlingPatterns:
    """Collection of common error handling patterns."""

    @staticmethod
    def data_processing_error_handler(func):
        """Error handler for data processing operations."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ Data processing failed: {e}")
                return None
        return wrapper

    @staticmethod
    def metrics_calculation_error_handler(func):
        """Error handler for metrics calculations."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ Metrics calculation failed: {e}")
                return None
        return wrapper

    @staticmethod
    def file_operation_error_handler(func):
        """Error handler for file operations."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"❌ File operation failed: {e}")
                return False
        return wrapper
