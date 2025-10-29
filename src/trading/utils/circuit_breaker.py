"""
Circuit breaker pattern for trading operations.
"""

import time
from enum import Enum
from typing import Callable, Optional, Dict, Any
from functools import wraps
from threading import Lock
from datetime import datetime, timedelta

from .error_handling import TradingError, TradingErrorSeverity
from .constants import (
    DEFAULT_CB_FAILURE_THRESHOLD,
    DEFAULT_CB_RECOVERY_TIMEOUT,
    DEFAULT_CB_HALF_OPEN_MAX_CALLS
)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD,
        recovery_timeout: float = DEFAULT_CB_RECOVERY_TIMEOUT,
        half_open_max_calls: int = DEFAULT_CB_HALF_OPEN_MAX_CALLS,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls in half-open state before deciding
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.lock = Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TradingError: If circuit is open
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.failure_count = 0
                else:
                    raise TradingError(
                        f"Circuit breaker {self.name} is OPEN. Service unavailable.",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        severity=TradingErrorSeverity.HIGH,
                        context={
                            'state': self.state.value,
                            'failure_count': self.failure_count,
                            'last_failure_time': self.last_failure_time
                        }
                    )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            TradingError: If circuit is open
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.failure_count = 0
                else:
                    raise TradingError(
                        f"Circuit breaker {self.name} is OPEN. Service unavailable.",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        severity=TradingErrorSeverity.HIGH,
                        context={
                            'state': self.state.value,
                            'failure_count': self.failure_count,
                            'last_failure_time': self.last_failure_time
                        }
                    )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery should be attempted."""
        if self.last_failure_time is None:
            return False
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        with self.lock:
            self.last_success_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_max_calls:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self.lock:
            self.last_failure_time = time.time()
            self.failure_count += 1

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout
            }

def circuit_breaker(
    failure_threshold: int = DEFAULT_CB_FAILURE_THRESHOLD,
    recovery_timeout: float = DEFAULT_CB_RECOVERY_TIMEOUT,
    half_open_max_calls: int = DEFAULT_CB_HALF_OPEN_MAX_CALLS,
    name: Optional[str] = None
):
    """
    Decorator that wraps a function with circuit breaker protection.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_max_calls: Max calls in half-open state
        name: Name for the circuit breaker

    Returns:
        Decorated function
    """
    cb = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        name=name or "circuit_breaker"
    )

    def decorator(func: Callable):
        import asyncio
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await cb.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            return sync_wrapper

    return decorator
