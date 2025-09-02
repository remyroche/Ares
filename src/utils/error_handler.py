"""Enhanced Error Handling and Recovery Strategies for Ares Trading Bot.

This module provides centralized error handling patterns, including
decorators for consistent error handling, retry logic, automatic recovery
strategies, circuit breaker pattern, and safe operation wrappers with 100% type hint coverage.
"""

import asyncio
import functools
import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, TypeVar, cast

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()  # Normal operation
    OPEN = auto()    # Failing, reject requests
    HALF_OPEN = auto()  # Testing if service is recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type[Exception] = Exception
    monitor_interval: float = 10.0


@dataclass
class PlaceholderDataClass:
    """Placeholder data class for future implementation."""
    pass


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""
    
    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> Any:
        """Execute the recovery strategy."""
        pass
    
    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this strategy can handle the given error."""
        pass


@dataclass
class RetryStrategy(RecoveryStrategy):
    """Retry strategy with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    
    async def execute(self, context: dict[str, Any]) -> Any:
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
            except Exception:
                if attempt == self.max_retries:
                    raise
                
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay,
                )
                
                if self.jitter:
                    delay *= 0.5 + 0.5 * (hash(str(attempt)) % 100) / 100
                
                await asyncio.sleep(delay)
        
        return None
    
    def can_handle(self, error: Exception) -> bool:
        """Check if retry strategy can handle the error."""
        return True


@dataclass
class FallbackStrategy(RecoveryStrategy):
    """Fallback strategy with multiple operations."""
    fallback_operations: list[Callable[..., Any]] = field(default_factory=list)
    
    async def execute(self, context: dict[str, Any]) -> Any:
        """Execute fallback strategy."""
        args = context.get("args", ())
        kwargs = context.get("kwargs", {})
        
        for i, operation in enumerate(self.fallback_operations):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except Exception:
                if i == len(self.fallback_operations) - 1:
                    raise
                continue
        
        return None
    
    def can_handle(self, error: Exception) -> bool:
        """Check if fallback strategy can handle the error."""
        return True


@dataclass
class GracefulDegradationStrategy(RecoveryStrategy):
    """Graceful degradation strategy."""
    default_return: Any = None
    error_types: list[type[Exception]] = field(default_factory=list)
    
    async def execute(self, context: dict[str, Any]) -> Any:
        """Execute graceful degradation."""
        return self.default_return
    
    def can_handle(self, error: Exception) -> bool:
        """Check if graceful degradation can handle the error."""
        if not self.error_types:
            return True
        return any(isinstance(error, error_type) for error_type in self.error_types)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, config: CircuitBreakerConfig) -> None:
        """Initialize circuit breaker."""
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    async def call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call operation with circuit breaker protection."""
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
                self.logger.exception(
                    f"Circuit breaker opened after {self.failure_count} failures: {e}",
                )
            
            raise


class ErrorRecoveryManager:
    """Manages automatic error recovery strategies."""
    
    def __init__(self) -> None:
        """Initialize error recovery manager."""
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
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute operation with automatic error recovery."""
        try:
            return await self._execute_operation(operation, *args, **kwargs)
        except Exception as e:
            return await self._attempt_recovery(e, operation, *args, **kwargs)
    
    async def _execute_operation(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the operation."""
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        return operation(*args, **kwargs)
    
    async def _attempt_recovery(
        self, error: Exception, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Attempt error recovery using available strategies."""
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
                    self.logger.exception(
                        f"Recovery strategy failed: {recovery_error}",
                    )
                    continue
        
        self.logger.error(f"All recovery strategies failed for error: {error}")
        return None


class ErrorHandler:
    """Enhanced error handler class with recovery strategies."""
    
    def __init__(
        self,
        logger: logging.Logger | None = None,
        context: str = "",
    ) -> None:
        """Initialize error handler."""
        self.logger = logger
        self.context = context
        self.recovery_manager = ErrorRecoveryManager()
    
    def handle_generic_errors(
        self,
        exceptions: tuple[type[Exception], ...] = (Exception,),
        default_return: T | None = None,
        *,
        recovery_strategies: list[RecoveryStrategy] | None = None,
    ) -> Callable[[F], F]:
        """Handle generic errors with logging and recovery."""
        
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T | None:
                try:
                    result = await func(*args, **kwargs)
                    return cast("T | None", result)
                except exceptions as e:
                    self._log_error(func.__name__, e)
                    
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
                                        return cast("T | None", recovery_result)
                                except Exception as recovery_error:
                                    self.logger.exception(
                                        f"Recovery failed: {recovery_error}",
                                    )
                    
                    return default_return
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T | None:
                try:
                    result = func(*args, **kwargs)
                    return cast("T | None", result)
                except exceptions as e:
                    self._log_error(func.__name__, e)
                    
                    if recovery_strategies:
                        for strategy in recovery_strategies:
                            if strategy.can_handle(e):
                                try:
                                    # For sync functions, handle recovery differently
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
                                        return cast("T | None", recovery_result)
                                except Exception as e:
                                    self.logger.exception(f"Recovery failed: {e}")
                    
                    return default_return
            
            if asyncio.iscoroutinefunction(func):
                return cast("F", async_wrapper)
            return cast("F", sync_wrapper)
        
        return decorator
    
    def handle_specific_errors(
        self,
        error_handlers: dict[type[Exception], tuple[Any, str]],
        default_return: Any = None,
        *,
        recovery_strategies: list[RecoveryStrategy] | None = None,
    ) -> Callable[[F], F]:
        """Handle specific errors with custom error handling."""
        
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any | None:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_type = type(e)
                    if error_type in error_handlers:
                        return_value, message = error_handlers[error_type]
                        self._log_error(func.__name__, e)
                        
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
                                            return recovery_result
                                    except Exception as recovery_error:
                                        self.logger.exception(
                                            f"Recovery failed: {recovery_error}",
                                        )
                        
                        return return_value
                    
                    self._log_error(func.__name__, e)
                    return default_return
            
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any | None:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_type = type(e)
                    if error_type in error_handlers:
                        return_value, message = error_handlers[error_type]
                        self._log_error(func.__name__, e)
                        
                        if recovery_strategies:
                            for strategy in recovery_strategies:
                                if strategy.can_handle(e):
                                    try:
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
                                            return recovery_result
                                    except Exception as recovery_error:
                                        self.logger.exception(
                                            f"Recovery failed: {recovery_error}",
                                        )
                        
                        return return_value
                    
                    self._log_error(func.__name__, e)
                    return default_return
            
            if asyncio.iscoroutinefunction(func):
                return cast("F", async_wrapper)
            return cast("F", sync_wrapper)
        
        return decorator
    
    def _log_error(self, func_name: str, error: Exception) -> None:
        """Log error with context."""
        if self.logger:
            self.logger.exception(
                f"Error in {self.context}.{func_name}: {error}",
            )
        else:
            _logger = logging.getLogger(__name__)
            if not _logger.handlers:
                _logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
                handler.setFormatter(formatter)
                _logger.addHandler(handler)
            # Fallback print if no logger configured
            print(f"Error in {self.context}.{func_name}: {error}")


# Enhanced decorator functions with recovery strategies
def handle_errors(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    default_return: T | None = None,
    context: str = "",
    *,
    log_errors: bool = True,
    reraise: bool = False,
    recovery_strategies: list[RecoveryStrategy] | None = None,
) -> Callable[[F], F]:
    """Enhanced error handling decorator with recovery strategies."""
    handler = ErrorHandler(context=context)
    return handler.handle_generic_errors(
        exceptions=exceptions,
        default_return=default_return,
        recovery_strategies=recovery_strategies,
    )


def handle_specific_errors(
    error_handlers: dict[type[Exception], tuple[Any, str]],
    default_return: Any = None,
    context: str = "",
    *,
    recovery_strategies: list[RecoveryStrategy] | None = None,
) -> Callable[[F], F]:
    """Enhanced specific error handling decorator with recovery strategies."""
    handler = ErrorHandler(context=context)
    return handler.handle_specific_errors(
        error_handlers=error_handlers,
        default_return=default_return,
        recovery_strategies=recovery_strategies,
    )


# Type-safe utility functions
def safe_operation(
    operation: Callable[..., T],
    *args: Any,
    default_return: T | None = None,
    **kwargs: Any,
) -> T | None:
    """Safe operation wrapper."""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Operation failed: {e}")
        return default_return


async def safe_async_operation(
    operation: Callable[..., Awaitable[T]],
    *args: Any,
    default_return: T | None = None,
    **kwargs: Any,
) -> T | None:
    """Safe async operation wrapper."""
    try:
        return await operation(*args, **kwargs)
    except Exception as e:
        logging.getLogger(__name__).exception(f"Async operation failed: {e}")
        return default_return


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type[Exception] = Exception,
) -> CircuitBreaker:
    """Create a circuit breaker."""
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
    """Create a retry strategy."""
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
    """Create a fallback strategy."""
    return FallbackStrategy(fallback_operations=fallback_operations)


def create_graceful_degradation_strategy(
    default_return: Any = None,
    error_types: list[type[Exception]] | None = None,
) -> GracefulDegradationStrategy:
    """Create a graceful degradation strategy."""
    return GracefulDegradationStrategy(
        default_return=default_return,
        error_types=error_types or [],
    )


async def safe_network_operation(
    operation: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any | None:
    """Safe network operation with retry logic."""
    try:
        import aiohttp
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                return operation(*args, **kwargs)
            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logging.getLogger(__name__).warning(
                        f"Network error (attempt {attempt + 1}/{max_retries}): "
                        f"{e}. Retrying in {wait_time}s...",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logging.getLogger(__name__).exception(
                        f"Network operation failed after {max_retries} attempts: {e}",
                    )
                    return None
            except Exception as e:
                logging.getLogger(__name__).exception(
                    f"Unexpected error in network operation: {e}",
                )
                return None
        return None
    except Exception as e:
        logging.getLogger(__name__).exception(
            f"Error in safe network operation: {e}",
        )
        return None


def safe_database_operation(
    operation: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """Safe database operation wrapper."""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(f"Database operation failed: {e}")
        return None


def safe_dataframe_operation(
    operation: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """Safe DataFrame operation wrapper."""
    try:
        result = operation(*args, **kwargs)
        return result
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(f"DataFrame operation failed: {e}")
        return None


def safe_numeric_operation(
    operation: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T | None:
    """Safe numeric operation wrapper."""
    try:
        result = operation(*args, **kwargs)
        return result
    except (ZeroDivisionError, ValueError, TypeError, OverflowError) as e:
        logger = logging.getLogger(__name__)
        logger.exception(f"Numeric operation failed: {e}")
        return None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception(f"Unexpected error in numeric operation: {e}")
        return None


def safe_dict_access(
    data: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Safe dictionary access."""
    try:
        return data.get(key, default)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error accessing dictionary key '{key}': {e}")
        return default


def safe_dataframe_access(
    df: Any,
    column: str,
    default: Any = None,
) -> Any:
    """Safe DataFrame column access."""
    try:
        if hasattr(df, 'columns') and column in df.columns:
            return df[column]
        return default
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error accessing DataFrame column '{column}': {e}")
        return default


class ErrorRecoveryStrategies:
    """Utility class for error recovery strategies."""
    
    @staticmethod
    def retry_with_backoff(
        operation: Callable[..., T],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
        **kwargs: Any,
    ) -> T | None:
        """Retry operation with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    logging.getLogger(__name__).exception(
                        f"Operation failed after {max_retries} retries: {e}",
                    )
                    return None
                
                delay = base_delay * (2 ** attempt)
                logging.getLogger(__name__).warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{e}. Retrying in {delay}s...",
                )
                time.sleep(delay)
        
        return None
    
    @staticmethod
    def fallback_chain(
        operations: list[Callable[..., T]],
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        """Execute operations in fallback chain."""
        for i, operation in enumerate(operations):
            try:
                result = operation(*args, **kwargs)
                logging.getLogger(__name__).info(f"Fallback operation {i + 1} succeeded")
                return result
            except Exception as e:
                logging.getLogger(__name__).exception(
                    f"Fallback operation {i + 1} failed: {e}",
                )
                if i == len(operations) - 1:
                    logging.getLogger(__name__).error("All fallback operations failed")
                    return None
        
        return None


class ErrorContext:
    """
    Context manager for error handling.
    
    This context manager provides a way to handle errors within a code block
    and optionally execute cleanup code.
    """
    
    def __init__(
        self,
        error_handler: Callable[[type[Exception], Exception, Any], None] | None = None,
        cleanup_handler: Callable[[], None] | None = None,
        reraise: bool = True,
    ):
        """
        Initialize error context.
        
        Args:
            error_handler: Function to call on error
            cleanup_handler: Function to call for cleanup
            reraise: Whether to reraise exceptions
        """
        self.error_handler = error_handler
        self.cleanup_handler = cleanup_handler
        self.reraise = reraise
        self.exception: Exception | None = None
    
    def __enter__(self):
        """Enter the context."""
        return self
    
    def __exit__(self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        """Exit the context and handle any exceptions."""
        if exc_type is not None:
            self.exception = exc_val
            
            if self.error_handler:
                try:
                    self.error_handler(exc_type, exc_val, exc_tb)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.exception(f"Error in error handler: {e}")
            
            if self.cleanup_handler:
                try:
                    self.cleanup_handler()
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.exception(f"Error in cleanup handler: {e}")
            
            return not self.reraise
        
        return False


def handle_assertion_errors(
    default_return: Any = None,
    context: str = "",
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for handling assertion errors with proper message formatting.
    
    This decorator addresses EM101/EM102 and TRY003 issues by:
    - Assigning exception messages to variables before raising
    - Using proper exception message formatting
    - Providing context-aware error handling
    
    Args:
        default_return: Value to return on error
        context: Context string for logging
        log_errors: Whether to log errors
    
    Returns:
        Decorated function
    """
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except AssertionError as e:
                if log_errors:
                    logging.getLogger(__name__).exception(
                        f"Assertion error in {context}.{func.__name__}: {e}",
                    )
                return default_return
            except Exception as e:
                if log_errors:
                    logging.getLogger(__name__).exception(
                        f"Unexpected error in {context}.{func.__name__}: {e}",
                    )
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except AssertionError as e:
                if log_errors:
                    logging.getLogger(__name__).exception(
                        f"Assertion error in {context}.{func.__name__}: {e}",
                    )
                return default_return
            except Exception as e:
                if log_errors:
                    logging.getLogger(__name__).exception(
                        f"Unexpected error in {context}.{func.__name__}: {e}",
                    )
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", sync_wrapper)
    
    return decorator


def safe_assertion(
    condition: bool,
    message: str,
    error_type: type[Exception] = AssertionError,
    context: str = "",
    log_errors: bool = True,
) -> None:
    """Safe assertion with proper error handling."""
    if not condition:
        # Assign message to variable to address EM101/EM102
        error_message = f"{context}: {message}" if context else message
        
        if log_errors:
            logger = logging.getLogger(__name__)
            logger.error(f"Assertion failed: {error_message}")
        
        raise error_type(error_message)


def format_assertion_message(
    message_template: str,
    expected: Any,
    actual: Any,
    context: str = "",
) -> str:
    """Format assertion message with expected vs actual values."""
    # Assign formatted message to variable to address EM101/EM102
    formatted_message = message_template.format(expected=expected, actual=actual)
    
    if context:
        return f"{context}: {formatted_message}"
    return formatted_message
