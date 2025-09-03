from __future__ import annotations
"""
Structured logging decorators with correlation IDs.

Provides decorators for consistent, structured logging with
automatic correlation ID propagation and sensitive data masking.
"""

import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from .compose import P, R, uniform_wrapper

# Context variable for correlation ID
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Sensitive field names to mask
SENSITIVE_FIELDS = {
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey",
    "access_key", "private_key", "auth", "authorization", "credit_card",
    "card_number", "cvv", "ssn", "social_security", "tax_id",
}


def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID from current context."""
    correlation_id_var.set(None)


def mask_sensitive_data(data: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively mask sensitive data in various data structures.

    Args:
        data: Data to mask
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Data with sensitive fields masked
    """
    if depth > max_depth:
        return "***MAX_DEPTH***"

    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            if any(sensitive in str(key).lower() for sensitive in SENSITIVE_FIELDS):
                masked[key] = "***MASKED***"
            else:
                masked[key] = mask_sensitive_data(value, depth + 1, max_depth)
        return masked

    if isinstance(data, list | tuple):
        return type(data)(mask_sensitive_data(item, depth + 1, max_depth) for item in data)

    if isinstance(data, str):
        # Check if the string itself looks like sensitive data
        lower_data = data.lower()
        if any(field in lower_data for field in ["bearer ", "basic ", "token "]):
            return "***MASKED***"
        return data

    return data


def log_call(
    *,
    level: str = "INFO",
    log_args: bool = True,
    log_result: bool = True,
    log_duration: bool = True,
    mask_sensitive: bool = True,
    include_metadata: bool = True,
    logger_name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Log function calls with structured data.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log execution duration
        mask_sensitive: Whether to mask sensitive data
        include_metadata: Whether to include metadata (correlation ID, etc.)
        logger_name: Custom logger name (defaults to function module)

    Example:
        @log_call(level="INFO", log_args=True, mask_sensitive=True)
        def create_user(username: str, password: str) -> dict:
            # password will be masked in logs
            return {"id": 123, "username": username}
    """
    def get_logger(func: Callable) -> logging.Logger:
        """Get appropriate logger for function."""
        if logger_name:
            return logging.getLogger(logger_name)
        return logging.getLogger(func.__module__)

    def prepare_log_data(
        func: Callable,
        args: tuple,
        kwargs: dict,
        result: Any = None,
        error: Exception = None,
        duration: float = None,
    ) -> dict[str, Any]:
        """Prepare structured log data."""
        log_data = {
            "function": func.__name__,
            "module": func.__module__,
        }

        if include_metadata:
            log_data["correlation_id"] = get_correlation_id()

        if log_args:
            # Mask sensitive data if needed
            masked_args = mask_sensitive_data(args) if mask_sensitive else args
            masked_kwargs = mask_sensitive_data(kwargs) if mask_sensitive else kwargs

            log_data["args"] = masked_args
            log_data["kwargs"] = masked_kwargs

        if result is not None and log_result:
            masked_result = mask_sensitive_data(result) if mask_sensitive else result
            log_data["result"] = masked_result

        if error is not None:
            log_data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }

        if duration is not None and log_duration:
            log_data["duration_ms"] = round(duration * 1000, 2)

        return log_data

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger(func)
        log_method = getattr(logger, level.lower(), logger.info)

        start_time = time.time() if log_duration else None

        # Log function call
        log_data = prepare_log_data(func, args, kwargs)
        log_method(f"Calling {func.__name__}", extra=log_data)

        try:
            result = func(*args, **kwargs)

            # Log successful completion
            duration = time.time() - start_time if start_time else None
            log_data = prepare_log_data(func, args, kwargs, result=result, duration=duration)
            log_method(f"Completed {func.__name__}", extra=log_data)

            return result

        except Exception as e:
            # Log error
            duration = time.time() - start_time if start_time else None
            log_data = prepare_log_data(func, args, kwargs, error=e, duration=duration)
            logger.error(f"Failed {func.__name__}", extra=log_data, exc_info=True)
            raise

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger(func)
        log_method = getattr(logger, level.lower(), logger.info)

        start_time = time.time() if log_duration else None

        # Log function call
        log_data = prepare_log_data(func, args, kwargs)
        log_method(f"Calling {func.__name__}", extra=log_data)

        try:
            result = await func(*args, **kwargs)

            # Log successful completion
            duration = time.time() - start_time if start_time else None
            log_data = prepare_log_data(func, args, kwargs, result=result, duration=duration)
            log_method(f"Completed {func.__name__}", extra=log_data)

            return result

        except Exception as e:
            # Log error
            duration = time.time() - start_time if start_time else None
            log_data = prepare_log_data(func, args, kwargs, error=e, duration=duration)
            logger.error(f"Failed {func.__name__}", extra=log_data, exc_info=True)
            raise

    return uniform_wrapper(f"log_call({level})", sync_handler, async_handler)


def log_execution_time(
    *,
    threshold_ms: float | None = None,
    logger_name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Log execution time of functions.

    Args:
        threshold_ms: Only log if execution time exceeds this threshold
        logger_name: Custom logger name

    Example:
        @log_execution_time(threshold_ms=100)
        def slow_operation():
            time.sleep(0.2)  # Will be logged
    """
    def get_logger(func: Callable) -> logging.Logger:
        if logger_name:
            return logging.getLogger(logger_name)
        return logging.getLogger(func.__module__)

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()

        try:
            return func(*args, **kwargs)
        finally:
            duration_ms = (time.time() - start_time) * 1000

            if threshold_ms is None or duration_ms >= threshold_ms:
                logger = get_logger(func)
                logger.info(
                    f"{func.__name__} executed in {duration_ms:.2f}ms",
                    extra={
                        "function": func.__name__,
                        "duration_ms": duration_ms,
                        "correlation_id": get_correlation_id(),
                    },
                )

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.time()

        try:
            return await func(*args, **kwargs)
        finally:
            duration_ms = (time.time() - start_time) * 1000

            if threshold_ms is None or duration_ms >= threshold_ms:
                logger = get_logger(func)
                logger.info(
                    f"{func.__name__} executed in {duration_ms:.2f}ms",
                    extra={
                        "function": func.__name__,
                        "duration_ms": duration_ms,
                        "correlation_id": get_correlation_id(),
                    },
                )

    return uniform_wrapper("log_execution_time", sync_handler, async_handler)


def audit_log(
    *,
    action: str | None = None,
    resource_type: str | None = None,
    include_user: bool = True,
    include_ip: bool = True,
    logger_name: str = "audit",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Create audit log entries for sensitive operations.

    Args:
        action: Action being performed (defaults to function name)
        resource_type: Type of resource being accessed
        include_user: Whether to include user information
        include_ip: Whether to include IP address
        logger_name: Logger name for audit logs

    Example:
        @audit_log(action="delete_user", resource_type="user")
        def delete_user(user_id: str) -> bool:
            # This will create an audit log entry
            return database.delete_user(user_id)
    """
    def get_context_data() -> dict[str, Any]:
        """Get context data for audit log."""
        context = {
            "correlation_id": get_correlation_id(),
            "timestamp": time.time(),
        }

        # In a real implementation, you would get these from the request context
        if include_user:
            # context["user_id"] = get_current_user_id()
            context["user_id"] = "system"  # Placeholder

        if include_ip:
            # context["ip_address"] = get_client_ip()
            context["ip_address"] = "127.0.0.1"  # Placeholder

        return context

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        logger = logging.getLogger(logger_name)

        audit_action = action or func.__name__
        context = get_context_data()

        # Log audit entry
        logger.info(
            f"AUDIT: {audit_action}",
            extra={
                "audit": True,
                "action": audit_action,
                "resource_type": resource_type,
                "function": func.__name__,
                **context,
                "args": mask_sensitive_data(args),
                "kwargs": mask_sensitive_data(kwargs),
            },
        )

        try:
            result = func(*args, **kwargs)

            # Log success
            logger.info(
                f"AUDIT: {audit_action} completed",
                extra={
                    "audit": True,
                    "action": audit_action,
                    "resource_type": resource_type,
                    "function": func.__name__,
                    "status": "success",
                    **context,
                },
            )

            return result

        except Exception as e:
            # Log failure
            logger.exception(
                f"AUDIT: {audit_action} failed",
                extra={
                    "audit": True,
                    "action": audit_action,
                    "resource_type": resource_type,
                    "function": func.__name__,
                    "status": "failure",
                    "error": str(e),
                    **context,
                },
            )
            raise

    async def async_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        logger = logging.getLogger(logger_name)

        audit_action = action or func.__name__
        context = get_context_data()

        # Log audit entry
        logger.info(
            f"AUDIT: {audit_action}",
            extra={
                "audit": True,
                "action": audit_action,
                "resource_type": resource_type,
                "function": func.__name__,
                **context,
                "args": mask_sensitive_data(args),
                "kwargs": mask_sensitive_data(kwargs),
            },
        )

        try:
            result = await func(*args, **kwargs)

            # Log success
            logger.info(
                f"AUDIT: {audit_action} completed",
                extra={
                    "audit": True,
                    "action": audit_action,
                    "resource_type": resource_type,
                    "function": func.__name__,
                    "status": "success",
                    **context,
                },
            )

            return result

        except Exception as e:
            # Log failure
            logger.exception(
                f"AUDIT: {audit_action} failed",
                extra={
                    "audit": True,
                    "action": audit_action,
                    "resource_type": resource_type,
                    "function": func.__name__,
                    "status": "failure",
                    "error": str(e),
                    **context,
                },
            )
            raise

    return uniform_wrapper(f"audit_log({action or 'auto'})", sync_handler, async_handler)
