"""
Retry, timeout, and circuit breaker decorators.

Provides resilience patterns for handling transient failures,
timeouts, and cascading failures.
"""

import asyncio
import functools
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Type, Union

from ..errors.base import TimeoutError as AppTimeoutError, ServiceUnavailableError
from .compose import uniform_wrapper, P, R


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._state_change_callbacks: List[Callable] = []
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise ServiceUnavailableError(
                    "Circuit breaker is OPEN",
                    service_name=func.__name__,
                    details={
                        "failures": self.stats.failure_count,
                        "last_failure": self.stats.last_failure_time,
                    }
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Call async function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise ServiceUnavailableError(
                    "Circuit breaker is OPEN",
                    service_name=func.__name__,
                    details={
                        "failures": self.stats.failure_count,
                        "last_failure": self.stats.last_failure_time,
                    }
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        return (
            self.stats.last_failure_time is not None and
            time.time() - self.stats.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.stats.success_count += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.success_threshold:
                self._transition_to_closed()
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.stats.failure_count += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif (self.state == CircuitState.CLOSED and 
              self.stats.consecutive_failures >= self.failure_threshold):
            self._transition_to_open()
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self._notify_state_change()
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self._notify_state_change()
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.stats.consecutive_successes = 0
        self.stats.consecutive_failures = 0
        self._notify_state_change()
    
    def _notify_state_change(self) -> None:
        """Notify callbacks of state change."""
        for callback in self._state_change_callbacks:
            try:
                callback(self.state)
            except Exception:
                pass  # Ignore callback errors
    
    def on_state_change(self, callback: Callable[[CircuitState], None]) -> None:
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def retry(
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        max_delay: Maximum delay between retries
        jitter: Add random jitter to delays
        exceptions: Exceptions to retry on
    
    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data(url: str) -> dict:
            response = requests.get(url)
            return response.json()
    """
    def calculate_delay(attempt: int) -> float:
        """Calculate delay for attempt with backoff and jitter."""
        delay_time = min(delay * (backoff ** (attempt - 1)), max_delay)
        if jitter:
            # Add random jitter between 0% and 25% of delay
            delay_time *= (1 + random.random() * 0.25)
        return delay_time
    
    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt < max_attempts:
                    sleep_time = calculate_delay(attempt)
                    time.sleep(sleep_time)
                
        # All attempts failed
        raise ServiceUnavailableError(
            f"Failed after {max_attempts} attempts",
            service_name=func.__name__,
            cause=last_exception,
            details={
                "attempts": max_attempts,
                "last_error": str(last_exception)
            }
        )
    
    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt < max_attempts:
                    sleep_time = calculate_delay(attempt)
                    await asyncio.sleep(sleep_time)
        
        # All attempts failed
        raise ServiceUnavailableError(
            f"Failed after {max_attempts} attempts",
            service_name=func.__name__,
            cause=last_exception,
            details={
                "attempts": max_attempts,
                "last_error": str(last_exception)
            }
        )
    
    return uniform_wrapper(
        f"retry(max_attempts={max_attempts})",
        sync_handler,
        async_handler
    )


def timeout(seconds: float) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Timeout decorator that raises TimeoutError if execution exceeds limit.
    
    Args:
        seconds: Timeout in seconds
    
    Example:
        @timeout(30.0)
        async def slow_operation() -> str:
            await asyncio.sleep(60)  # Will timeout
            return "done"
    """
    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # For sync functions, we can't easily implement timeout without threads
        # Log a warning and execute normally
        import logging
        logging.getLogger(__name__).warning(
            f"Timeout decorator on sync function {func.__name__} has no effect. "
            "Consider making the function async."
        )
        return func(*args, **kwargs)
    
    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        except asyncio.TimeoutError as e:
            raise AppTimeoutError(
                f"Operation timed out after {seconds} seconds",
                timeout_seconds=seconds,
                details={"function": func.__name__}
            ) from e
    
    return uniform_wrapper(
        f"timeout({seconds}s)",
        sync_handler,
        async_handler
    )


def circuit_breaker(
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    expected_exception: Type[Exception] = Exception,
    breaker_name: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Circuit breaker decorator to prevent cascading failures.
    
    Args:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Time before attempting recovery
        success_threshold: Successes needed to close circuit
        expected_exception: Exception types that trigger the breaker
        breaker_name: Name for the breaker (defaults to function name)
    
    Example:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def call_external_service() -> dict:
            return external_api.get_data()
    """
    def get_or_create_breaker(name: str) -> CircuitBreaker:
        """Get existing breaker or create new one."""
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
                expected_exception=expected_exception,
            )
        return _circuit_breakers[name]
    
    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        name = breaker_name or func.__name__
        breaker = get_or_create_breaker(name)
        return breaker.call(func, *args, **kwargs)
    
    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        name = breaker_name or func.__name__
        breaker = get_or_create_breaker(name)
        return await breaker.async_call(func, *args, **kwargs)
    
    return uniform_wrapper(
        f"circuit_breaker({breaker_name or 'auto'})",
        sync_handler,
        async_handler
    )


def retry_with_circuit_breaker(
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Combine retry and circuit breaker patterns.
    
    This decorator first applies circuit breaker logic, then retry logic.
    If the circuit is open, it fails fast without retrying.
    
    Example:
        @retry_with_circuit_breaker(
            max_attempts=3,
            failure_threshold=10,
            recovery_timeout=120
        )
        async def resilient_api_call() -> dict:
            return await external_api.fetch()
    """
    # Compose the decorators
    return compose(
        retry(
            max_attempts=max_attempts,
            delay=delay,
            backoff=backoff,
            exceptions=exceptions
        ),
        circuit_breaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=exceptions[0] if exceptions else Exception
        )
    )


def fallback(
    fallback_value: Union[Any, Callable[[], Any]],
    exceptions: tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Provide fallback value on failure.
    
    Args:
        fallback_value: Value to return on failure (or callable that returns it)
        exceptions: Exceptions to catch
    
    Example:
        @fallback(fallback_value={"status": "unavailable"})
        def get_service_status() -> dict:
            return external_service.get_status()
    """
    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except exceptions:
            if callable(fallback_value):
                return fallback_value()
            return fallback_value
    
    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await func(*args, **kwargs)
        except exceptions:
            if callable(fallback_value):
                if asyncio.iscoroutinefunction(fallback_value):
                    return await fallback_value()
                return fallback_value()
            return fallback_value
    
    return uniform_wrapper("fallback", sync_handler, async_handler)


# Import compose for the retry_with_circuit_breaker decorator
from .compose import compose