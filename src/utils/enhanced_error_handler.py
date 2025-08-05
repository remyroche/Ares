"""
Enhanced Error Handling and Recovery Strategies for Ares Trading Bot.

This module provides advanced error handling patterns, automatic recovery strategies,
circuit breaker pattern, and comprehensive type safety with 100% type hint coverage.
"""

import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TypeVar, cast

import numpy as np

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, reject requests
    HALF_OPEN = auto()  # Testing if service is recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type[Exception] = Exception
    monitor_interval: float = 10.0


@dataclass
class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> Any | None:
        """Execute the recovery strategy."""

    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this strategy can handle the given error."""


@dataclass
class RetryStrategy(RecoveryStrategy):
    """Retry strategy with exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True

    async def execute(self, context: dict[str, Any]) -> Any | None:
        """Execute retry strategy."""
        operation = context.get("operation")
        args = context.get("args", ())
        kwargs = context.get("kwargs", {})

        if not operation:
            return None

        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise e

                delay = min(
                    self.base_delay * (self.backoff_factor**attempt),
                    self.max_delay,
                )

                if self.jitter:
                    delay *= 0.5 + np.random.random() * 0.5

                await asyncio.sleep(delay)

        return None

    def can_handle(self, error: Exception) -> bool:
        """Retry can handle any exception."""
        return True


@dataclass
class FallbackStrategy(RecoveryStrategy):
    """Fallback strategy with multiple fallback operations."""

    fallback_operations: list[Callable[..., Any]] = field(default_factory=list)

    async def execute(self, context: dict[str, Any]) -> Any | None:
        """Execute fallback strategy."""
        args = context.get("args", ())
        kwargs = context.get("kwargs", {})

        for i, operation in enumerate(self.fallback_operations):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except Exception as e:
                if i == len(self.fallback_operations) - 1:
                    raise e
                continue

        return None

    def can_handle(self, error: Exception) -> bool:
        """Fallback can handle any exception."""
        return True


@dataclass
class GracefulDegradationStrategy(RecoveryStrategy):
    """Graceful degradation strategy."""

    default_return: Any = None
    error_types: list[type[Exception]] = field(default_factory=list)

    async def execute(self, context: dict[str, Any]) -> Any | None:
        """Execute graceful degradation."""
        return self.default_return

    def can_handle(self, error: Exception) -> bool:
        """Check if this strategy can handle the error."""
        if not self.error_types:
            return True
        return any(isinstance(error, error_type) for error_type in self.error_types)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")

    async def call(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        """Execute operation with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                self.logger.warning("Circuit breaker is OPEN, rejecting request")
                return None

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker recovered, transitioning to CLOSED")

            return result

        except self.config.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures",
                )

            raise e


class ErrorRecoveryManager:
    """Manages automatic error recovery strategies."""

    def __init__(self) -> None:
        self.strategies: list[RecoveryStrategy] = []
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(f"{__name__}.ErrorRecoveryManager")

    def add_strategy(self, strategy: RecoveryStrategy) -> None:
        """Add a recovery strategy."""
        self.strategies.append(strategy)

    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> None:
        """Add a circuit breaker."""
        self.circuit_breakers[name] = CircuitBreaker(config)

    async def execute_with_recovery(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        """Execute operation with automatic recovery."""
        try:
            return await self._execute_operation(operation, *args, **kwargs)
        except Exception as e:
            return await self._attempt_recovery(e, operation, *args, **kwargs)

    async def _execute_operation(
        self,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute the operation."""
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        return operation(*args, **kwargs)

    async def _attempt_recovery(
        self,
        error: Exception,
        operation: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        """Attempt recovery using available strategies."""
        context = {
            "operation": operation,
            "args": args,
            "kwargs": kwargs,
            "error": error,
        }

        for strategy in self.strategies:
            if strategy.can_handle(error):
                try:
                    self.logger.info(
                        f"Attempting recovery with {type(strategy).__name__}",
                    )
                    result = await strategy.execute(context)
                    if result is not None:
                        self.logger.info(
                            f"Recovery successful with {type(strategy).__name__}",
                        )
                        return result
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
                    continue

        self.logger.error(f"All recovery strategies failed for error: {error}")
        return None


class EnhancedErrorHandler:
    """Enhanced error handler with comprehensive type safety."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_manager = ErrorRecoveryManager()

    def handle_errors(
        self,
        exceptions: tuple[type[Exception], ...] = (Exception,),
        default_return: T | None = None,
        context: str = "",
        *,
        log_errors: bool = True,
        reraise: bool = False,
        recovery_strategies: list[RecoveryStrategy] | None = None,
    ) -> Callable[[F], F]:
        """Enhanced error handling decorator."""

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T | None:
                try:
                    result = await func(*args, **kwargs)
                    return cast(T | None, result)
                except exceptions as e:
                    if log_errors:
                        self.logger.exception(
                            f"Error in {context}.{func.__name__}: {e}",
                        )

                    if recovery_strategies:
                        for strategy in recovery_strategies:
                            if strategy.can_handle(e):
                                try:
                                    recovery_result = await strategy.execute(
                                        {
                                            "operation": func,
                                            "args": args,
                                            "kwargs": kwargs,
                                            "error": e,
                                        },
                                    )
                                    if recovery_result is not None:
                                        return cast(T | None, recovery_result)
                                except Exception as recovery_error:
                                    self.logger.error(
                                        f"Recovery failed: {recovery_error}",
                                    )

                    if reraise:
                        raise e
                    return default_return

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T | None:
                try:
                    result = func(*args, **kwargs)
                    return cast(T | None, result)
                except exceptions as e:
                    if log_errors:
                        self.logger.exception(
                            f"Error in {context}.{func.__name__}: {e}",
                        )

                    if recovery_strategies:
                        for strategy in recovery_strategies:
                            if strategy.can_handle(e):
                                try:
                                    # For sync functions, we need to handle recovery differently
                                    if hasattr(strategy, "execute"):
                                        # Create async context for sync recovery
                                        async def run_recovery() -> Any | None:
                                            return await strategy.execute(
                                                {
                                                    "operation": func,
                                                    "args": args,
                                                    "kwargs": kwargs,
                                                    "error": e,
                                                },
                                            )

                                        loop = asyncio.get_event_loop()
                                        recovery_result = loop.run_until_complete(
                                            run_recovery(),
                                        )
                                        if recovery_result is not None:
                                            return cast(T | None, recovery_result)
                                except Exception as recovery_error:
                                    self.logger.error(
                                        f"Recovery failed: {recovery_error}",
                                    )

                    if reraise:
                        raise e
                    return default_return

            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)

        return decorator

    def handle_specific_errors(
        self,
        error_handlers: dict[type[Exception], tuple[Any, str]],
        default_return: T | None = None,
        context: str = "",
        *,
        log_errors: bool = True,
    ) -> Callable[[F], F]:
        """Handle specific error types with type safety."""

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T | None:
                try:
                    result = await func(*args, **kwargs)
                    return cast(T | None, result)
                except Exception as e:
                    error_type = type(e)
                    if error_type in error_handlers:
                        return_value, message = error_handlers[error_type]
                        if log_errors:
                            self.logger.error(
                                f"{message} in {context}.{func.__name__}: {e}",
                            )
                        return cast(T | None, return_value)

                    if log_errors:
                        self.logger.exception(
                            f"Unexpected error in {context}.{func.__name__}: {e}",
                        )
                    return default_return

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T | None:
                try:
                    result = func(*args, **kwargs)
                    return cast(T | None, result)
                except Exception as e:
                    error_type = type(e)
                    if error_type in error_handlers:
                        return_value, message = error_handlers[error_type]
                        if log_errors:
                            self.logger.error(
                                f"{message} in {context}.{func.__name__}: {e}",
                            )
                        return cast(T | None, return_value)

                    if log_errors:
                        self.logger.exception(
                            f"Unexpected error in {context}.{func.__name__}: {e}",
                        )
                    return default_return

            if asyncio.iscoroutinefunction(func):
                return cast(F, async_wrapper)
            return cast(F, sync_wrapper)

        return decorator


# Global enhanced error handler instance
enhanced_error_handler = EnhancedErrorHandler()


# Convenience functions with full type hints
def handle_enhanced_errors(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    default_return: T | None = None,
    context: str = "",
    *,
    log_errors: bool = True,
    reraise: bool = False,
    recovery_strategies: list[RecoveryStrategy] | None = None,
) -> Callable[[F], F]:
    """Enhanced error handling decorator with recovery strategies."""
    return enhanced_error_handler.handle_errors(
        exceptions=exceptions,
        default_return=default_return,
        context=context,
        log_errors=log_errors,
        reraise=reraise,
        recovery_strategies=recovery_strategies,
    )


def handle_enhanced_specific_errors(
    error_handlers: dict[type[Exception], tuple[Any, str]],
    default_return: T | None = None,
    context: str = "",
    *,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """Enhanced specific error handling decorator."""
    return enhanced_error_handler.handle_specific_errors(
        error_handlers=error_handlers,
        default_return=default_return,
        context=context,
        log_errors=log_errors,
    )


# Type-safe utility functions
def safe_operation(
    operation: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """Execute operation safely with type hints."""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Operation failed: {e}")
        return None


async def safe_async_operation(
    operation: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """Execute async operation safely with type hints."""
    try:
        return await operation(*args, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Async operation failed: {e}")
        return None


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception,
) -> CircuitBreaker:
    """Create a circuit breaker with type hints."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
    )
    return CircuitBreaker(config)


def create_retry_strategy(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> RetryStrategy:
    """Create a retry strategy with type hints."""
    return RetryStrategy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
    )


def create_fallback_strategy(
    fallback_operations: list[Callable[..., Any]],
) -> FallbackStrategy:
    """Create a fallback strategy with type hints."""
    return FallbackStrategy(fallback_operations=fallback_operations)


def create_graceful_degradation_strategy(
    default_return: Any = None,
    error_types: list[type[Exception]] | None = None,
) -> GracefulDegradationStrategy:
    """Create a graceful degradation strategy with type hints."""
    return GracefulDegradationStrategy(
        default_return=default_return,
        error_types=error_types or [],
    )
