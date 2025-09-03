from __future__ import annotations
"""
Enhanced Error Handling Utilities

This module provides enhanced error handling capabilities including retry mechanisms,
circuit breakers, and error categorization for the training pipeline.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from src.utils.logger import system_logger
    from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
except ImportError:
    system_logger = logging.getLogger("EnhancedErrorHandling")


class ErrorType(Enum):
    """Types of errors for categorization."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    CRITICAL = "critical"


class RetryableError(Exception):
    """Error that can be retried."""



class NonRetryableError(Exception):
    """Error that should not be retried."""



class CircuitBreakerError(Exception):
    """Error raised when circuit breaker is open."""



@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type[Exception] = Exception
    monitor_interval: float = 10.0


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = system_logger.getChild("CircuitBreaker")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                msg = "Circuit breaker is OPEN"
                raise CircuitBreakerError(msg)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker transitioning to CLOSED")
        self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def retry_with_backoff(config: RetryConfig | None = None):
    """Decorator for retrying operations with exponential backoff."""
    if config is None:
        # Fallback implementation for config
        # Fallback implementation for config
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    return func(*args, **kwargs)
                except RetryableError as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        wait_time = _calculate_backoff_delay(attempt, config)
                        logging.warning(f"Retryable error on attempt {attempt + 1}: {e}. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.exception(f"Max retries ({config.max_retries}) exceeded. Last error: {e}")
                        raise
                except NonRetryableError as e:
                    logging.exception(f"Non-retryable error: {e}")
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        wait_time = _calculate_backoff_delay(attempt, config)
                        logging.warning(f"Unexpected error on attempt {attempt + 1}: {e}. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.exception(f"Max retries ({config.max_retries}) exceeded. Last error: {e}")
                        raise

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RetryableError as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        wait_time = _calculate_backoff_delay(attempt, config)
                        logging.warning(f"Retryable error on attempt {attempt + 1}: {e}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.exception(f"Max retries ({config.max_retries}) exceeded. Last error: {e}")
                        raise
                except NonRetryableError as e:
                    logging.exception(f"Non-retryable error: {e}")
                    raise
                except Exception as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        wait_time = _calculate_backoff_delay(attempt, config)
                        logging.warning(f"Unexpected error on attempt {attempt + 1}: {e}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logging.exception(f"Max retries ({config.max_retries}) exceeded. Last error: {e}")
                        raise

            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _calculate_backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate backoff delay with optional jitter."""
    delay = min(config.initial_delay * (config.backoff_factor**attempt), config.max_delay)

    if config.jitter:
        import random

        delay *= 0.5 + random.random() * 0.5  # Add 50% jitter

    return delay


def circuit_breaker(config: CircuitBreakerConfig | None = None):
    """Decorator for circuit breaker pattern."""
    if config is None:
        # Fallback implementation for config
        # Fallback implementation for config
        config = CircuitBreakerConfig()

    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(config)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return breaker.call(lambda: asyncio.create_task(func(*args, **kwargs)), *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def categorize_errors(error_mapping: dict[type[Exception], ErrorType]):
    """Decorator for categorizing errors into retryable/non-retryable."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            except Exception as e:
                error_type = _get_error_type(e, error_mapping)
                if error_type == ErrorType.RETRYABLE:
                    msg = f"Retryable error: {e}"
                    raise RetryableError(msg) from e
                if error_type == ErrorType.NON_RETRYABLE:
                    msg = f"Non-retryable error: {e}"
                    raise NonRetryableError(msg) from e
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = _get_error_type(e, error_mapping)
                if error_type == ErrorType.RETRYABLE:
                    msg = f"Retryable error: {e}"
                    raise RetryableError(msg) from e
                if error_type == ErrorType.NON_RETRYABLE:
                    msg = f"Non-retryable error: {e}"
                    raise NonRetryableError(msg) from e
                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _get_error_type(exception: Exception, error_mapping: dict[type[Exception], ErrorType]) -> ErrorType:
    """Get the error type for an exception based on the mapping."""
    for error_class, error_type in error_mapping.items():
        if isinstance(exception, error_class):
            return error_type
    return ErrorType.CRITICAL


# Common error mappings for data operations
DATA_OPERATION_ERRORS = {
    ConnectionError: ErrorType.RETRYABLE,
    TimeoutError: ErrorType.RETRYABLE,
    OSError: ErrorType.RETRYABLE,
    ValueError: ErrorType.NON_RETRYABLE,
    TypeError: ErrorType.NON_RETRYABLE,
    KeyError: ErrorType.NON_RETRYABLE,
    IndexError: ErrorType.NON_RETRYABLE,
}


# Convenience decorators
def retry_data_operation(max_retries: int = 3, backoff_factor: float = 2.0):
    """Convenience decorator for data operations with retry."""
    config = RetryConfig(max_retries=max_retries, backoff_factor=backoff_factor)
    return retry_with_backoff(config)


def circuit_breaker_data_operation(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """Convenience decorator for data operations with circuit breaker."""
    config = CircuitBreakerConfig(failure_threshold=failure_threshold, recovery_timeout=recovery_timeout)
    return circuit_breaker(config)
