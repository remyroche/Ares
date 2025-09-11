"""
Network Operations Utilities

This module provides network-related utilities and decorators for handling
network operations with retry logic and error handling.
"""

import asyncio
import logging
from typing import Any, Callable, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

def handle_network_operations(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for handling network operations with retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff factor for exponential wait

    Returns:
        Decorated function with network operation handling
    """
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=backoff_factor),
            reraise=True
        )
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Network operation failed in {func.__name__}: {e}")
                raise

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=backoff_factor),
            reraise=True
        )
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Network operation failed in {func.__name__}: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# For backward compatibility
network_retry = handle_network_operations
