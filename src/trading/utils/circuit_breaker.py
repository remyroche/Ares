"""
Circuit breaker pattern for trading operations.
"""

import time
from enum import Enum
from typing import Callable, Optional, Dict, Any
from functools import wraps
from threading import Lock
from datetime import datetime, timedelta

from src.printing import tprint
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
        tprint(f"[CIRCUIT_BREAKER] __init__: name={name}, failure_threshold={failure_threshold}, recovery_timeout={recovery_timeout}, half_open_max_calls={half_open_max_calls}")

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

        tprint(f"[CIRCUIT_BREAKER] __init__ -> initialized in CLOSED state")

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
        tprint(f"[CIRCUIT_BREAKER] call: name={self.name}, func={func.__name__}, state={self.state.value}")

        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    tprint(f"[CIRCUIT_BREAKER] call: Transitioning from OPEN to HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.failure_count = 0
                else:
                    tprint(f"[CIRCUIT_BREAKER] call: Circuit is OPEN, rejecting call")
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
            tprint(f"[CIRCUIT_BREAKER] call -> success")
            return result
        except Exception as e:
            tprint(f"[CIRCUIT_BREAKER] call: Function failed with {type(e).__name__}: {str(e)}")
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
        tprint(f"[CIRCUIT_BREAKER] call_async: name={self.name}, func={func.__name__}, state={self.state.value}")

        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    tprint(f"[CIRCUIT_BREAKER] call_async: Transitioning from OPEN to HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.failure_count = 0
                else:
                    tprint(f"[CIRCUIT_BREAKER] call_async: Circuit is OPEN, rejecting call")
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
            tprint(f"[CIRCUIT_BREAKER] call_async -> success")
            return result
        except Exception as e:
            tprint(f"[CIRCUIT_BREAKER] call_async: Function failed with {type(e).__name__}: {str(e)}")
            self._record_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery should be attempted."""
        if self.last_failure_time is None:
            tprint(f"[CIRCUIT_BREAKER] _should_attempt_recovery -> False (no last failure)")
            return False
        elapsed = time.time() - self.last_failure_time
        should_recover = elapsed >= self.recovery_timeout
        tprint(f"[CIRCUIT_BREAKER] _should_attempt_recovery: elapsed={elapsed:.2f}s, timeout={self.recovery_timeout}s -> {should_recover}")
        return should_recover

    def _record_success(self) -> None:
        """Record a successful call."""
        with self.lock:
            self.last_success_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                tprint(f"[CIRCUIT_BREAKER] _record_success: HALF_OPEN state, success_count={self.success_count}/{self.half_open_max_calls}")
                if self.success_count >= self.half_open_max_calls:
                    tprint(f"[CIRCUIT_BREAKER] _record_success: Transitioning HALF_OPEN -> CLOSED")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
                tprint(f"[CIRCUIT_BREAKER] _record_success: CLOSED state, reset failure_count")

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self.lock:
            self.last_failure_time = time.time()
            self.failure_count += 1
            tprint(f"[CIRCUIT_BREAKER] _record_failure: failure_count={self.failure_count}/{self.failure_threshold}, state={self.state.value}")

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                tprint(f"[CIRCUIT_BREAKER] _record_failure: Transitioning HALF_OPEN -> OPEN")
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    tprint(f"[CIRCUIT_BREAKER] _record_failure: Transitioning CLOSED -> OPEN (threshold reached)")
                    self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        tprint(f"[CIRCUIT_BREAKER] reset: Resetting circuit breaker {self.name}")
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.last_success_time = None
        tprint(f"[CIRCUIT_BREAKER] reset -> reset complete")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        tprint(f"[CIRCUIT_BREAKER] get_state: name={self.name}")
        with self.lock:
            state_dict = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout
            }
        tprint(f"[CIRCUIT_BREAKER] get_state -> state={self.state.value}, failures={self.failure_count}")
        return state_dict

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
