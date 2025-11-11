"""
Retry mechanism for trading operations with exponential backoff.
"""

import asyncio
import time
from typing import Callable, Type, Union, Tuple, Optional, Any
from functools import wraps

from src.utils.tprint import tprint
from .error_handling import TradingError, TradingErrorSeverity, NetworkError, RateLimitError
from .constants import (
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_RETRY_EXPONENT
)

def retry_on_error(
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    exponent: float = DEFAULT_RETRY_EXPONENT,
    retry_on: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    exponential_backoff: bool = True,
    reraise: bool = True
):
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponent: Backoff exponent
        retry_on: Exception types to retry on
        exponential_backoff: Whether to use exponential backoff
        reraise: Whether to re-raise the exception after all retries fail

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tprint(f"[RETRY] async_wrapper: func={func.__name__}, max_attempts={max_attempts}, base_delay={base_delay}")
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        tprint(f"[RETRY] async_wrapper: Attempt {attempt + 1}/{max_attempts} for {func.__name__}")
                        result = await func(*args, **kwargs)
                        tprint(f"[RETRY] async_wrapper -> success on attempt {attempt + 1}")
                        return result
                    except retry_on as e:
                        last_exception = e
                        tprint(f"[RETRY] async_wrapper: Attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}")
                        if attempt < max_attempts - 1:
                            if exponential_backoff:
                                delay = min(base_delay * (exponent ** attempt), max_delay)
                            else:
                                delay = base_delay

                            tprint(f"[RETRY] async_wrapper: Waiting {delay:.2f}s before retry")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            tprint(f"[RETRY] async_wrapper: All attempts exhausted")
                            break

                if reraise and last_exception:
                    tprint(f"[RETRY] async_wrapper: Re-raising exception after {max_attempts} attempts")
                    # Wrap non-TradingError exceptions
                    if not isinstance(last_exception, TradingError):
                        raise NetworkError(
                            f"Operation failed after {max_attempts} attempts: {str(last_exception)}",
                            original_exception=last_exception,
                            context={'max_attempts': max_attempts, 'attempts': attempt + 1}
                        )
                    raise last_exception
                tprint(f"[RETRY] async_wrapper -> None (failed after all attempts)")
                return None

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                tprint(f"[RETRY] sync_wrapper: func={func.__name__}, max_attempts={max_attempts}, base_delay={base_delay}")
                last_exception = None
                for attempt in range(max_attempts):
                    try:
                        tprint(f"[RETRY] sync_wrapper: Attempt {attempt + 1}/{max_attempts} for {func.__name__}")
                        result = func(*args, **kwargs)
                        tprint(f"[RETRY] sync_wrapper -> success on attempt {attempt + 1}")
                        return result
                    except retry_on as e:
                        last_exception = e
                        tprint(f"[RETRY] sync_wrapper: Attempt {attempt + 1} failed with {type(e).__name__}: {str(e)}")
                        if attempt < max_attempts - 1:
                            if exponential_backoff:
                                delay = min(base_delay * (exponent ** attempt), max_delay)
                            else:
                                delay = base_delay

                            tprint(f"[RETRY] sync_wrapper: Waiting {delay:.2f}s before retry")
                            time.sleep(delay)
                            continue
                        else:
                            tprint(f"[RETRY] sync_wrapper: All attempts exhausted")
                            break

                if reraise and last_exception:
                    tprint(f"[RETRY] sync_wrapper: Re-raising exception after {max_attempts} attempts")
                    # Wrap non-TradingError exceptions
                    if not isinstance(last_exception, TradingError):
                        raise NetworkError(
                            f"Operation failed after {max_attempts} attempts: {str(last_exception)}",
                            original_exception=last_exception,
                            context={'max_attempts': max_attempts, 'attempts': attempt + 1}
                        )
                    raise last_exception
                tprint(f"[RETRY] sync_wrapper -> None (failed after all attempts)")
                return None

            return sync_wrapper

    return decorator

def retry_on_rate_limit(
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY
):
    """
    Decorator specifically for rate limit errors with longer delays.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds (default: 5s for rate limits)
        max_delay: Maximum delay in seconds

    Returns:
        Decorated function
    """
    return retry_on_error(
        max_attempts=max_attempts,
        base_delay=max(base_delay, 5.0),  # At least 5 seconds for rate limits
        max_delay=max_delay,
        retry_on=RateLimitError,
        exponential_backoff=True
    )

def retry_on_network_error(
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY
):
    """
    Decorator specifically for network errors.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Decorated function
    """
    return retry_on_error(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=(NetworkError, ConnectionError, TimeoutError),
        exponential_backoff=True
    )
