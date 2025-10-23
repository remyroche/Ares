"""
Decorators package for utils.

This module provides decorators for error handling, tracing, and validation.
"""

import functools
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

logger = logging.getLogger(__name__)

def handles_errors(default_return=None, log_errors=True, reraise=False):
    """
    Decorator to handle errors gracefully.

    Args:
        default_return: Value to return if an error occurs
        log_errors: Whether to log errors
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

def traced(log_args=False, log_result=False):
    """
    Decorator to trace function calls.

    Args:
        log_args: Whether to log function arguments
        log_result: Whether to log function result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__}")
            if log_args:
                logger.debug(f"Args: {args}, Kwargs: {kwargs}")

            result = func(*args, **kwargs)

            if log_result:
                logger.debug(f"Result: {result}")

            return result
        return wrapper
    return decorator

def validates(validator_func=None, error_message="Validation failed"):
    """
    Decorator to validate function inputs/outputs.

    Args:
        validator_func: Function to validate the result
        error_message: Error message if validation fails
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if validator_func and not validator_func(result):
                raise ValueError(error_message)

            return result
        return wrapper
    return decorator

__all__ = ['handles_errors', 'traced', 'validates']
