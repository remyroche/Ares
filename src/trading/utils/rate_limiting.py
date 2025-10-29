"""
Rate limiting utilities for trading operations.
"""

import time
from threading import Lock
from typing import Dict, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta

from .error_handling import RateLimitError
from .constants import DEFAULT_RATE_LIMIT_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW

class RateLimiter:
    """
    Token bucket rate limiter implementation.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW,
        name: str = "rate_limiter"
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            name: Name for logging
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.name = name

        self.requests: list = []  # List of request timestamps
        self.lock = Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from the rate limiter.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        with self.lock:
            now = time.time()
            # Remove requests outside the window
            cutoff = now - self.window_seconds
            self.requests = [req_time for req_time in self.requests if req_time > cutoff]

            # Check if we can make the request
            if len(self.requests) + tokens <= self.max_requests:
                self.requests.extend([now] * tokens)
                return True
            return False

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit is exceeded, return wait time.

        Args:
            None

        Returns:
            Seconds waited (0 if no wait needed)

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        if not self.acquire():
            # Calculate wait time
            if self.requests:
                oldest_request = min(self.requests)
                wait_time = self.window_seconds - (time.time() - oldest_request)
                if wait_time > 0:
                    time.sleep(wait_time)
                    return wait_time
            
            # If we still can't acquire, raise error
            raise RateLimitError(
                f"Rate limit exceeded for {self.name}: {self.max_requests} requests per {self.window_seconds}s",
                context={
                    'max_requests': self.max_requests,
                    'window_seconds': self.window_seconds,
                    'current_requests': len(self.requests)
                }
            )
        return 0.0

    async def wait_if_needed_async(self) -> float:
        """
        Async version of wait_if_needed.

        Returns:
            Seconds waited (0 if no wait needed)

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        import asyncio
        
        if not self.acquire():
            # Calculate wait time
            if self.requests:
                oldest_request = min(self.requests)
                wait_time = self.window_seconds - (time.time() - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return wait_time
            
            # If we still can't acquire, raise error
            raise RateLimitError(
                f"Rate limit exceeded for {self.name}: {self.max_requests} requests per {self.window_seconds}s",
                context={
                    'max_requests': self.max_requests,
                    'window_seconds': self.window_seconds,
                    'current_requests': len(self.requests)
                }
            )
        return 0.0

    def reset(self) -> None:
        """Reset rate limiter."""
        with self.lock:
            self.requests.clear()

    def get_status(self) -> Dict:
        """Get current rate limiter status."""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            current_requests = [req_time for req_time in self.requests if req_time > cutoff]
            
            return {
                'max_requests': self.max_requests,
                'window_seconds': self.window_seconds,
                'current_requests': len(current_requests),
                'remaining': max(0, self.max_requests - len(current_requests))
            }

# Global rate limiters by name
_rate_limiters: Dict[str, RateLimiter] = {}
_rate_limiters_lock = Lock()

def get_rate_limiter(
    name: str,
    max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
    window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW
) -> RateLimiter:
    """
    Get or create a rate limiter by name.

    Args:
        name: Name of the rate limiter
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds

    Returns:
        RateLimiter instance
    """
    with _rate_limiters_lock:
        if name not in _rate_limiters:
            _rate_limiters[name] = RateLimiter(
                max_requests=max_requests,
                window_seconds=window_seconds,
                name=name
            )
        return _rate_limiters[name]

def rate_limit(
    max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
    window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW,
    name: Optional[str] = None
):
    """
    Decorator that adds rate limiting to a function.

    Args:
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds
        name: Name for the rate limiter (defaults to function name)

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        limiter_name = name or f"{func.__module__}.{func.__name__}"
        limiter = get_rate_limiter(
            name=limiter_name,
            max_requests=max_requests,
            window_seconds=window_seconds
        )

        import asyncio
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                await limiter.wait_if_needed_async()
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                limiter.wait_if_needed()
                return func(*args, **kwargs)
            return sync_wrapper

    return decorator
