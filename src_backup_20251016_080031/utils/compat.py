"""
Compatibility utilities for error handling and other cross-module functionality.

This module provides compatibility functions that may be needed across different
parts of the codebase to ensure consistent error handling and other utilities.
"""

from typing import Any, Dict, Callable, Tuple, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def handle_specific_errors(error_handlers: Optional[Dict[type, Tuple[Any, str]]] = None,
                          default_return: Any = None,
                          context: str = "operation") -> Callable:
    """
    Decorator for handling specific errors with custom responses.

    Args:
        error_handlers: Dict mapping exception types to (return_value, log_message) tuples
        default_return: Default return value if no specific handler matches
        context: Context string for logging

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check for specific error handlers
                if error_handlers:
                    for error_type, (return_value, log_message) in error_handlers.items():
                        if isinstance(e, error_type):
                            logger.warning(f"⚠️ {context}: {log_message} - {e}")
                            return return_value

                # Default error handling
                logger.error(f"❌ {context}: Unexpected error - {e}")
                return default_return

        return wrapper
    return decorator
