"""
Error handling utilities for the Ares trading bot.

This module provides centralized error handling patterns, including
decorators for consistent error handling, retry logic, and safe
operation wrappers.
"""

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any
from functools import wraps

import numpy as np
import pandas as pd


# Lazy import to prevent circular imports
def get_system_logger():
    """Get system logger with lazy import to prevent circular dependencies."""
    try:
        from src.utils.logger import system_logger

        return system_logger
    except ImportError:
        # Fallback to basic logger if circular import occurs
        logger = logging.getLogger("System")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class ErrorHandler:
    """Simplified error handler class to reduce complexity."""

    def __init__(self, logger=None, context: str = ""):
        self.logger = logger
        self.context = context

    def handle_generic_errors(
        self,
        exceptions: tuple = (Exception,),
        default_return: Any = None,
    ):
        """Handle generic errors with logging."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    self._log_error(func.__name__, e)
                    return default_return

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    self._log_error(func.__name__, e)
                    return default_return

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def handle_specific_errors(self, error_handlers: dict, default_return: Any = None):
        """Handle specific error types."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return self._handle_specific_error(
                        e,
                        error_handlers,
                        default_return,
                    )

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return self._handle_specific_error(
                        e,
                        error_handlers,
                        default_return,
                    )

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def _log_error(self, func_name: str, error: Exception) -> None:
        """Log error with context."""
        if self.logger:
            self.logger.exception(f"Error in {self.context}.{func_name}: {error}")
        else:
            print(f"Error in {self.context}.{func_name}: {error}")

    def _handle_specific_error(
        self,
        error: Exception,
        handlers: dict,
        default_return: Any,
    ) -> Any:
        """Handle specific error types."""
        error_type = type(error)
        if error_type in handlers:
            return_value, message = handlers[error_type]
            self._log_error("function", error)
            return return_value
        self._log_error("function", error)
        return default_return


# Simplified decorator functions that use the ErrorHandler class
def handle_errors(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    default_return: Any = None,
    context: str = "",
    *,
    log_errors: bool = True,
    reraise: bool = False,
    retry_count: int = 0,
    retry_delay: float = 1.0,
):
    """Simplified error handling decorator."""
    handler = ErrorHandler(context=context)
    return handler.handle_generic_errors(exceptions, default_return)


def handle_specific_errors(
    error_handlers: dict[type[Exception], tuple[Any, str]] = None,
    default_return: Any = None,
    context: str = "",
    *,
    log_errors: bool = True,
):
    """Simplified specific error handling decorator."""
    if error_handlers is None:
        error_handlers = {}

    handler = ErrorHandler(context=context)
    return handler.handle_specific_errors(error_handlers, default_return)


def _log_success_simple(
    func_name: str,
    attempt: int,
    max_retries: int,
    attempt_start_time: float,
    start_time: float,
    result: Any,
) -> None:
    """Simple success logging without logger dependency."""
    if max_retries > 0:
        print(
            f"SUCCESS: {func_name} completed on attempt {attempt + 1}/{max_retries + 1}",
        )
    else:
        print(f"SUCCESS: {func_name} completed")


def _log_retry_attempt_simple(
    func_name: str,
    attempt: int,
    max_retries: int,
    attempt_start_time: float,
    start_time: float,
    error: Exception,
) -> None:
    """Simple retry attempt logging without logger dependency."""
    print(
        f"ERROR: {func_name} failed on attempt {attempt + 1}/{max_retries + 1}: {error}",
    )


def handle_network_operations(
    max_retries: int = 3,
    default_return: Any = None,
):
    """
    Decorator for network operations with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        default_return: Value to return on failure

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_with_retries(
                func,
                args,
                kwargs,
                max_retries,
                default_return,
                is_async=True,
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_with_retries(
                func,
                args,
                kwargs,
                max_retries,
                default_return,
                is_async=False,
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def _execute_with_retries(
    func: Callable,
    args: tuple,
    kwargs: dict,
    max_retries: int,
    default_return: Any,
    is_async: bool,
) -> Any:
    """Execute function with retry logic."""
    start_time = time.time()

    for attempt in range(max_retries + 1):
        attempt_start_time = time.time()

        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            _log_success_simple(
                func.__name__,
                attempt,
                max_retries,
                attempt_start_time,
                start_time,
                result,
            )
            return result

        except Exception as e:
            _log_retry_attempt_simple(
                func.__name__,
                attempt,
                max_retries,
                attempt_start_time,
                start_time,
                e,
            )

            if attempt < max_retries:
                wait_time = 2**attempt
                print(f"WARNING: Retrying {func.__name__} in {wait_time} seconds...")
                if is_async:
                    await asyncio.sleep(wait_time)
                else:
                    time.sleep(wait_time)
            else:
                system_logger = get_system_logger()
                system_logger.exception(
                    f"Max retries ({max_retries}) reached. "
                    f"Returning default value.",
                )
                return default_return

    return default_return


def _log_success(
    logger: logging.Logger,
    attempt: int,
    max_retries: int,
    attempt_start_time: float,
    start_time: float,
    result: Any,
) -> None:
    """Log successful operation."""
    attempt_duration = time.time() - attempt_start_time
    total_duration = time.time() - start_time

    logger.info("✅ Network operation successful:")
    logger.info(f"   Attempt: {attempt + 1}/{max_retries + 1}")
    logger.info(f"   Attempt duration: {attempt_duration:.2f} seconds")
    logger.info(f"   Total duration: {total_duration:.2f} seconds")
    logger.info(f"   Result type: {type(result)}")


def _log_retry_attempt(
    logger: logging.Logger,
    attempt: int,
    max_retries: int,
    attempt_start_time: float,
    start_time: float,
    error: Exception,
) -> None:
    """Log retry attempt details."""
    attempt_duration = time.time() - attempt_start_time
    total_duration = time.time() - start_time

    logger.exception("💥 Network operation failed:")
    logger.exception(f"   Attempt: {attempt + 1}/{max_retries + 1}")
    logger.exception(f"   Attempt duration: {attempt_duration:.2f} seconds")
    logger.exception(f"   Total duration: {total_duration:.2f} seconds")
    logger.exception(f"   Error type: {type(error).__name__}")
    logger.exception("Full traceback:")
    logger.exception(traceback.format_exc())


def handle_data_processing_errors(
    default_return: Any = None,
    context: str = "",
):
    """
    Decorator for data processing operations with NaN/inf handling.

    Args:
        default_return: Value to return on error
        context: Context string for logging

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return _clean_data_result(result)
            except Exception as e:
                context_str = f" ({context})" if context else ""
                system_logger = get_system_logger()
                system_logger.exception(f"DataFrame operation failed{context_str}: {e}")
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return _clean_data_result(result)
            except Exception as e:
                context_str = f" ({context})" if context else ""
                system_logger = get_system_logger()
                system_logger.exception(f"DataFrame operation failed{context_str}: {e}")
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _clean_data_result(result: Any) -> Any:
    """Clean and validate data processing results."""
    if result is None:
        return result

    # Handle NaN values in result
    if isinstance(result, pd.DataFrame | pd.Series):
        result = result.fillna(0)
    elif isinstance(result, np.ndarray):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def handle_file_operations(
    default_return: Any = None,
    context: str = "",
):
    """
    Decorator for file operations with comprehensive error handling.

    Args:
        default_return: Value to return on error
        context: Context string for logging

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except OSError as e:
                context_str = f" ({context})" if context else ""
                system_logger = get_system_logger()
                system_logger.exception(
                    f"OS error during file operation{context_str}: {e}",
                )
                return default_return
            except Exception as e:
                context_str = f" ({context})" if context else ""
                system_logger = get_system_logger()
                system_logger.exception(
                    f"Unexpected error in file operation{context_str}: {e}",
                )
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except OSError as e:
                context_str = f" ({context})" if context else ""
                system_logger = get_system_logger()
                system_logger.exception(
                    f"OS error during file operation{context_str}: {e}",
                )
                return default_return
            except Exception as e:
                context_str = f" ({context})" if context else ""
                system_logger = get_system_logger()
                system_logger.exception(
                    f"Unexpected error in file operation{context_str}: {e}",
                )
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def handle_type_conversions(
    default_return: Any = None,
    *,
    log_errors: bool = True,
):
    """
    Decorator for type conversion operations.

    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return _clean_numeric_result(result)
            except (ValueError, TypeError) as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.warning(
                        f"Type conversion error in {func.__name__}: {e}",
                    )
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.exception(f"Unexpected error in {func.__name__}: {e}")
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return _clean_numeric_result(result)
            except (ValueError, TypeError) as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.warning(
                        f"Type conversion error in {func.__name__}: {e}",
                    )
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.exception(f"Unexpected error in {func.__name__}: {e}")
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _clean_numeric_result(result: Any) -> Any:
    """Clean and validate numeric results."""
    if result is None:
        return result

    # Handle special numeric values
    if isinstance(result, int | float):
        if np.isnan(result) or np.isinf(result):
            return 0.0
    elif isinstance(result, np.ndarray):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


async def safe_network_operation(
    operation: Callable,
    max_retries: int = 3,
    *args,
    **kwargs,
) -> Any:
    """
    Safe wrapper for network operations with retry logic.

    Args:
        operation: Function to execute
        max_retries: Maximum number of retry attempts
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        Operation result or None on failure
    """
    try:
        import aiohttp

        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    system_logger = get_system_logger()
                    system_logger.warning(
                        f"Network error (attempt {attempt + 1}/{max_retries}): "
                        f"{e}. Retrying in {wait_time}s...",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    system_logger = get_system_logger()
                    system_logger.exception(
                        f"Network operation failed after {max_retries} attempts: {e}",
                    )
                    return None
            except Exception as e:
                system_logger = get_system_logger()
                system_logger.exception(f"Unexpected error in network operation: {e}")
                return None
        return None
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.exception(f"Error in safe network operation: {e}")
        return None


def safe_database_operation(operation: Callable, *args, **kwargs) -> Any:
    """
    Safe wrapper for database operations.

    Args:
        operation: Function to execute
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        Operation result or None on failure
    """
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.exception(f"Database operation failed: {e}")
        return None


def safe_dataframe_operation(operation: Callable, *args, **kwargs) -> Any:
    """
    Safe wrapper for DataFrame operations with NaN handling.

    Args:
        operation: Function to execute
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        Operation result or None on failure
    """
    try:
        result = operation(*args, **kwargs)
        return _clean_data_result(result)
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.exception(f"DataFrame operation failed: {e}")
        return None


def safe_numeric_operation(operation: Callable, *args, **kwargs) -> Any:
    """
    Wrapper for safe numeric operations with division by zero and overflow handling.

    Args:
        operation: Function to execute
        *args: Positional arguments for operation
        **kwargs: Keyword arguments for operation

    Returns:
        Operation result or 0.0 on failure
    """
    try:
        result = operation(*args, **kwargs)
        return _clean_numeric_result(result)
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as e:
        system_logger = get_system_logger()
        system_logger.exception(f"Numeric operation failed: {e}")
        return 0.0
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.exception(f"Unexpected error in numeric operation: {e}")
        return 0.0


def safe_dict_access(data: dict, key: str, default: Any = None) -> Any:
    """
    Safe dictionary access with default value.

    Args:
        data: Dictionary to access
        key: Key to access
        default: Default value if key doesn't exist

    Returns:
        Value or default
    """
    try:
        return data.get(key, default)
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.warning(f"Error accessing dictionary key '{key}': {e}")
        return default


def safe_dataframe_access(df: pd.DataFrame, column: str, default: Any = None) -> Any:
    """
    Safe DataFrame column access with default value.

    Args:
        df: DataFrame to access
        column: Column name to access
        default: Default value if column doesn't exist

    Returns:
        Column data or default
    """
    try:
        if column in df.columns:
            return df[column]
        return default
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.warning(f"Error accessing DataFrame column '{column}': {e}")
        return default


class ErrorRecoveryStrategies:
    """Utility class for error recovery strategies."""

    @staticmethod
    def retry_with_backoff(
        operation: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        *args,
        **kwargs,
    ) -> Any:
        """
        Retry operation with exponential backoff.

        Args:
            operation: Function to retry
            max_retries: Maximum number of retries
            base_delay: Base delay between retries
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result or None on failure
        """
        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    system_logger = get_system_logger()
                    system_logger.exception(
                        f"Operation failed after {max_retries} retries: {e}",
                    )
                    return None

                delay = base_delay * (2**attempt)
                system_logger = get_system_logger()
                system_logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{e}. Retrying in {delay}s...",
                )
                time.sleep(delay)

        return None

    @staticmethod
    def fallback_chain(operations: list[Callable], *args, **kwargs) -> Any:
        """
        Execute operations in fallback chain.

        Args:
            operations: List of operations to try
            *args: Positional arguments for operations
            **kwargs: Keyword arguments for operations

        Returns:
            First successful result or None
        """
        for i, operation in enumerate(operations):
            try:
                result = operation(*args, **kwargs)
                system_logger = get_system_logger()
                system_logger.info(f"Fallback operation {i + 1} succeeded")
                return result
            except Exception as e:
                system_logger = get_system_logger()
                system_logger.warning(f"Fallback operation {i + 1} failed: {e}")
                if i == len(operations) - 1:
                    system_logger = get_system_logger()
                    system_logger.exception("All fallback operations failed")
                    return None

        return None


class ErrorContext:
    """
    Context manager for error handling.

    This context manager provides a way to handle errors within a code block
    and optionally execute cleanup code.
    """

    def __init__(
        self,
        error_handler: Callable | None = None,
        cleanup_handler: Callable | None = None,
        *,
        reraise: bool = True,
    ):
        """
        Initialize error context.

        Args:
            error_handler: Function to call on error
            cleanup_handler: Function to call for cleanup
            reraise: Whether to reraise exceptions
        """
        self.error_handler = error_handler
        self.cleanup_handler = cleanup_handler
        self.reraise = reraise
        self.exception = None

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and handle any exceptions."""
        if exc_type is not None:
            self.exception = exc_val

            if self.error_handler:
                try:
                    self.error_handler(exc_type, exc_val, exc_tb)
                except Exception as e:
                    system_logger = get_system_logger()
                    system_logger.exception(f"Error in error handler: {e}")

            if self.cleanup_handler:
                try:
                    self.cleanup_handler()
                except Exception as e:
                    system_logger = get_system_logger()
                    system_logger.exception(f"Error in cleanup handler: {e}")

            return not self.reraise

        return False


def handle_assertion_errors(
    default_return: Any = None,
    context: str = "",
    *,
    log_errors: bool = True,
):
    """
    Decorator for handling assertion errors with proper message formatting.
    
    This decorator addresses EM101/EM102 and TRY003 issues by:
    - Assigning exception messages to variables before raising
    - Using proper exception message formatting
    - Providing context-aware error handling
    
    Args:
        default_return: Value to return on error
        context: Context string for logging
        log_errors: Whether to log errors
        
    Returns:
        Decorated function
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AssertionError as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.error(f"Assertion error in {context}.{func.__name__}: {e}")
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.exception(f"Unexpected error in {context}.{func.__name__}: {e}")
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AssertionError as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.error(f"Assertion error in {context}.{func.__name__}: {e}")
                return default_return
            except Exception as e:
                if log_errors:
                    system_logger = get_system_logger()
                    system_logger.exception(f"Unexpected error in {context}.{func.__name__}: {e}")
                return default_return

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def safe_assertion(
    condition: bool,
    message: str,
    *,
    context: str = "",
    error_type: type[Exception] = AssertionError,
    log_errors: bool = True,
) -> None:
    """
    Safe assertion function that properly formats error messages.
    
    This function addresses EM101/EM102 issues by:
    - Assigning the message to a variable before raising
    - Providing context-aware error messages
    - Using proper exception message formatting
    
    Args:
        condition: Condition to assert
        message: Error message (assigned to variable before raising)
        context: Context string for logging
        error_type: Type of exception to raise
        log_errors: Whether to log errors
        
    Raises:
        error_type: If condition is False
    """
    if not condition:
        # Assign message to variable to address EM101/EM102
        error_message = f"{context}: {message}" if context else message
        
        if log_errors:
            system_logger = get_system_logger()
            system_logger.error(f"Assertion failed: {error_message}")
            
        raise error_type(error_message)


def format_assertion_message(
    expected: Any,
    actual: Any,
    context: str = "",
    *,
    message_template: str = "Expected {expected}, got {actual}",
) -> str:
    """
    Format assertion messages properly to address EM101/EM102 issues.
    
    Args:
        expected: Expected value
        actual: Actual value
        context: Context string
        message_template: Template for the message
        
    Returns:
        Formatted message string
    """
    # Assign formatted message to variable to address EM101/EM102
    formatted_message = message_template.format(expected=expected, actual=actual)
    
    if context:
        return f"{context}: {formatted_message}"
    return formatted_message


def handle_nan_issues(func: Callable) -> Callable:
    """
    Decorator for data processing operations with comprehensive NaN handling.
    
    This decorator:
    1. Replaces infinite values with NaN
    2. Fills NaN values with appropriate defaults based on data type
    3. Handles division by zero
    4. Provides detailed logging of NaN issues
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Handle DataFrame results
            if isinstance(result, pd.DataFrame):
                initial_shape = result.shape
                
                # Replace infinite values
                result = result.replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN values with appropriate defaults
                for col in result.columns:
                    if result[col].dtype in ['float64', 'float32']:
                        # For numeric columns, use 0 as default
                        result[col] = result[col].fillna(0)
                    elif result[col].dtype in ['int64', 'int32']:
                        # For integer columns, use 0 as default
                        result[col] = result[col].fillna(0)
                    else:
                        # For other types, use forward fill then backward fill
                        result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
                
                # Log any remaining NaN issues
                nan_counts = result.isnull().sum()
                if nan_counts.sum() > 0:
                    system_logger = get_system_logger()
                    system_logger.warning(f"⚠️ NaN handling completed. Remaining NaN counts: {nan_counts[nan_counts > 0].to_dict()}")
                
                final_shape = result.shape
                if initial_shape != final_shape:
                    system_logger = get_system_logger()
                    system_logger.warning(f"⚠️ DataFrame shape changed from {initial_shape} to {final_shape}")
                
                return result
            
            # Handle Series results
            elif isinstance(result, pd.Series):
                # Replace infinite values
                result = result.replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN based on data type
                if result.dtype in ['float64', 'float32']:
                    result = result.fillna(0)
                elif result.dtype in ['int64', 'int32']:
                    result = result.fillna(0)
                else:
                    result = result.fillna(method='ffill').fillna(method='bfill')
                
                return result
            
            # Handle numpy arrays
            elif isinstance(result, np.ndarray):
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return result
            
            # Handle scalar values
            elif isinstance(result, (int, float)):
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return result
            
            return result
            
        except Exception as e:
            system_logger = get_system_logger()
            system_logger.error(f"Error in NaN handling for {func.__name__}: {e}")
            # Return safe default based on function signature
            return None
    
    return wrapper


def safe_division(numerator, denominator, default=0.0):
    """
    Safe division function that handles division by zero and NaN values.
    
    Args:
        numerator: Numerator value or array
        denominator: Denominator value or array
        default: Default value to return when division fails
        
    Returns:
        Result of safe division
    """
    try:
        if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
            # Handle pandas Series
            result = numerator / denominator.replace(0, np.nan)
            result = result.fillna(default)
            return result
        elif isinstance(numerator, (pd.Series, np.ndarray)) and isinstance(denominator, (int, float)):
            # Handle Series/array divided by scalar
            if denominator == 0:
                return pd.Series(default, index=numerator.index) if isinstance(numerator, pd.Series) else np.full_like(numerator, default)
            result = numerator / denominator
            result = result.fillna(default) if isinstance(result, pd.Series) else np.nan_to_num(result, nan=default)
            return result
        else:
            # Handle scalar division
            if denominator == 0:
                return default
            result = numerator / denominator
            return result if not (np.isnan(result) or np.isinf(result)) else default
    except Exception as e:
        system_logger = get_system_logger()
        system_logger.warning(f"Error in safe division: {e}")
        return default


def clean_dataframe(df: pd.DataFrame, critical_columns: list = None) -> pd.DataFrame:
    """
    Comprehensive DataFrame cleaning function.
    
    Args:
        df: DataFrame to clean
        critical_columns: List of critical columns that must not have NaN values
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    initial_shape = df.shape
    system_logger = get_system_logger()
    
    # Remove rows with NaN in critical columns
    if critical_columns:
        critical_cols = [col for col in critical_columns if col in df.columns]
        if critical_cols:
            df = df.dropna(subset=critical_cols)
            system_logger.info(f"Removed rows with NaN in critical columns: {critical_cols}")
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values based on data type
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].fillna(0)
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    final_shape = df.shape
    if initial_shape != final_shape:
        system_logger.warning(f"DataFrame shape changed from {initial_shape} to {final_shape}")
    
    return df
