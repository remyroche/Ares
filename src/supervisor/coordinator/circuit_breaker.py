"""
Circuit Breaker Module.

This module implements the circuit breaker pattern for handling failures
in external services and preventing cascading failures.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class CircuitBreaker:
    """Circuit breaker pattern for external services."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = system_logger.getChild("CircuitBreaker")

    @handles_errors(
        exceptions=(ValueError, TypeError, AttributeError, RuntimeError),
        default_return=None,
    )
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of function execution
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                msg = "Circuit breaker is OPEN"
                raise Exception(msg)

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker closed successfully")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self.logger.info("Circuit breaker reset")

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "timeout": self.timeout,
        }