"""
Error Handling and Retry Logic

Comprehensive error handling system for the trading system.
Provides retry mechanisms, error categorization, and recovery strategies.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
from contextlib import asynccontextmanager

T = TypeVar('T')

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    API = "api"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_ORDER = "invalid_order"
    SYSTEM = "system"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    context: Dict[str, Any]
    retry_count: int = 0
    last_retry: Optional[datetime] = None
    resolved: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.NETWORK,
        ErrorCategory.API,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMIT
    ])


class ErrorHandler:
    """Centralized error handling and retry management"""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)

        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.active_errors: Dict[str, ErrorRecord] = {}
        self.error_counts: Dict[str, int] = {}

        # Recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {
            ErrorCategory.NETWORK: self._network_recovery,
            ErrorCategory.API: self._api_recovery,
            ErrorCategory.RATE_LIMIT: self._rate_limit_recovery,
            ErrorCategory.AUTHENTICATION: self._auth_recovery,
            ErrorCategory.INSUFFICIENT_FUNDS: self._funds_recovery,
            ErrorCategory.INVALID_ORDER: self._order_recovery,
            ErrorCategory.SYSTEM: self._system_recovery,
            ErrorCategory.TIMEOUT: self._timeout_recovery,
        }

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an exception into an error category"""
        error_str = str(error).lower()

        if any(keyword in error_str for keyword in ["network", "connection", "timeout", "dns", "socket"]):
            return ErrorCategory.NETWORK
        elif any(keyword in error_str for keyword in ["rate limit", "too many requests", "429"]):
            return ErrorCategory.RATE_LIMIT
        elif any(keyword in error_str for keyword in ["auth", "api key", "signature", "401", "403"]):
            return ErrorCategory.AUTHENTICATION
        elif any(keyword in error_str for keyword in ["insufficient", "balance", "funds"]):
            return ErrorCategory.INSUFFICIENT_FUNDS
        elif any(keyword in error_str for keyword in ["invalid", "order", "symbol", "quantity"]):
            return ErrorCategory.INVALID_ORDER
        elif any(keyword in error_str for keyword in ["system", "internal", "server", "500", "502", "503"]):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN

    def assess_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Assess the severity of an error"""
        category = self.categorize_error(error)

        # Critical errors that require immediate attention
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.SYSTEM]:
            return ErrorSeverity.CRITICAL
        elif category == ErrorCategory.RATE_LIMIT and context.get("retry_count", 0) > 5:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.NETWORK and context.get("retry_count", 0) > 10:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM

    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        retry_func: Optional[Callable] = None
    ) -> Union[T, None]:
        """Handle an error with retry logic"""
        category = self.categorize_error(error)
        severity = self.assess_severity(error, context)

        # Create error record
        error_record = ErrorRecord(
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            context=context,
            retry_count=context.get("retry_count", 0)
        )

        # Record the error
        self._record_error(error_record)

        # Check if we should retry
        should_retry = (
            category in self.config.retry_on and
            context.get("retry_count", 0) < self.config.max_retries
        )

        if should_retry and retry_func:
            return await self._retry_with_backoff(error_record, retry_func)
        else:
            # Execute recovery strategy
            await self._execute_recovery_strategy(error_record)

            # Log error
            self.logger.error(
                f"Error {error_record.error_type}: {error_record.error_message} "
                f"({category.value}, {severity.value})"
            )

            return None

    async def _retry_with_backoff(
        self,
        error_record: ErrorRecord,
        retry_func: Callable[[], Awaitable[T]]
    ) -> T:
        """Execute function with exponential backoff retry"""
        retry_count = error_record.retry_count

        # Calculate delay
        delay = min(
            self.config.base_delay * (self.config.backoff_factor ** retry_count),
            self.config.max_delay
        )

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay *= (0.5 + 0.5 * (retry_count / 10))  # 50-100% of calculated delay

        # Update error record
        error_record.retry_count = retry_count + 1
        error_record.last_retry = datetime.now()

        self.logger.warning(
            f"Retrying {error_record.error_type} in {delay".1f"}s "
            f"(attempt {error_record.retry_count}/{self.config.max_retries})"
        )

        # Wait before retry
        await asyncio.sleep(delay)

        try:
            # Execute the retry function
            result = await retry_func()

            # Mark error as resolved
            error_record.resolved = True
            self.logger.info(f"Retry successful for {error_record.error_type}")

            return result

        except Exception as retry_error:
            # Update error record with retry failure
            error_record.error_message = str(retry_error)

            # Recursive retry if still within limits
            if error_record.retry_count < self.config.max_retries:
                return await self._retry_with_backoff(error_record, retry_func)
            else:
                # Max retries exceeded
                self.logger.error(f"Max retries exceeded for {error_record.error_type}")
                await self._execute_recovery_strategy(error_record)
                raise retry_error

    async def _execute_recovery_strategy(self, error_record: ErrorRecord) -> None:
        """Execute recovery strategy for error category"""
        strategy = self.recovery_strategies.get(error_record.category)
        if strategy:
            try:
                await strategy(error_record)
            except Exception as e:
                self.logger.error(f"Recovery strategy failed: {e}")

    async def _network_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for network errors"""
        self.logger.info("Executing network recovery strategy")
        # Could implement connection pooling, circuit breaker, etc.

    async def _api_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for API errors"""
        self.logger.info("Executing API recovery strategy")
        # Could implement API failover, request queuing, etc.

    async def _rate_limit_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for rate limiting"""
        self.logger.info("Executing rate limit recovery strategy")
        # Could implement request throttling, priority queuing, etc.

    async def _auth_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for authentication errors"""
        self.logger.warning("Authentication error detected - requires manual intervention")
        # Could implement token refresh, re-authentication, etc.

    async def _funds_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for insufficient funds"""
        self.logger.warning("Insufficient funds error - order cannot be executed")
        # Could implement position sizing checks, balance monitoring, etc.

    async def _order_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for invalid orders"""
        self.logger.info("Executing order validation recovery strategy")
        # Could implement order parameter validation, symbol checking, etc.

    async def _system_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for system errors"""
        self.logger.warning("System error detected - monitoring required")
        # Could implement health checks, failover, etc.

    async def _timeout_recovery(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for timeout errors"""
        self.logger.info("Executing timeout recovery strategy")
        # Could implement request timeout adjustment, connection pooling, etc.

    def _record_error(self, error_record: ErrorRecord) -> None:
        """Record an error for tracking and analysis"""
        self.error_history.append(error_record)
        self.active_errors[error_record.error_type] = error_record

        # Update error counts
        key = f"{error_record.category.value}_{error_record.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        # Limit error history size
        if len(self.error_history) > 10000:
            self.error_history = self.error_history[-5000:]

    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_errors = len(self.error_history)
        resolved_errors = len([e for e in self.error_history if e.resolved])
        unresolved_errors = total_errors - resolved_errors

        # Category breakdown
        category_counts = {}
        for error in self.error_history:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # Severity breakdown
        severity_counts = {}
        for error in self.error_history:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_errors": total_errors,
            "resolved_errors": resolved_errors,
            "unresolved_errors": unresolved_errors,
            "resolution_rate": (resolved_errors / total_errors * 100) if total_errors > 0 else 0,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "error_counts": self.error_counts,
            "recent_errors": [e.__dict__ for e in self.error_history[-100:]],  # Last 100 errors
            "timestamp": datetime.now().isoformat()
        }

    async def clear_error_history(self) -> None:
        """Clear error history and reset statistics"""
        self.error_history.clear()
        self.active_errors.clear()
        self.error_counts.clear()
        self.logger.info("Error history cleared")


# Decorator for error handling
def with_error_handling(
    error_handler: ErrorHandler,
    retry_on_failure: bool = True,
    context: Optional[Dict[str, Any]] = None
):
    """Decorator to add error handling to async functions"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                call_context = context or {}
                call_context.update({
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "retry_count": kwargs.get("retry_count", 0)
                })

                if retry_on_failure:
                    return await error_handler.handle_error(e, call_context)
                else:
                    error_handler._record_error(ErrorRecord(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        category=error_handler.categorize_error(e),
                        severity=error_handler.assess_severity(e, call_context),
                        timestamp=datetime.now(),
                        context=call_context
                    ))
                    raise e

        return wrapper
    return decorator


# Async context manager for error handling
@asynccontextmanager
async def error_handling_context(error_handler: ErrorHandler, context: Dict[str, Any]):
    """Async context manager for error handling"""
    try:
        yield
    except Exception as e:
        await error_handler.handle_error(e, context)
        raise


# Utility function to create default error handler
def create_default_error_handler() -> ErrorHandler:
    """Create a default error handler with sensible defaults"""
    return ErrorHandler(RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
        retry_on=[
            ErrorCategory.NETWORK,
            ErrorCategory.API,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT
        ]
    ))