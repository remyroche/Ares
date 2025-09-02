"""
Structured logging utilities for the Ares trading bot.

This module provides structured logging capabilities including:
- Correlation ID management for request tracing
- JSON formatting for log output
- Context management for correlation IDs
"""

import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Any, Dict

if TYPE_CHECKING:
    from fastapi import Request

try:
    # Optional: only needed when JSON format is enabled
    from pythonjsonlogger import jsonlogger  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    jsonlogger = None  # type: ignore

# Context variables for correlation across logs
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="-",
)

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id",
    default="-",
)


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the current correlation ID."""
    correlation_id_var.set(correlation_id)


def ensure_correlation_id() -> str:
    """Ensure a correlation ID exists, generating one if needed."""
    current = correlation_id_var.get()
    if not current or current == "-":
        new_id = generate_correlation_id()
        correlation_id_var.set(new_id)
        return new_id
    return current


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return uuid.uuid4().hex


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager that sets a correlation ID for the duration of the block."""
    token = None
    cid = correlation_id or generate_correlation_id()
    try:
        token = correlation_id_var.set(cid)
        yield cid
    finally:
        if token is not None:
            correlation_id_var.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Filter that adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        record.correlation_id = get_correlation_id()
        record.session_id = session_id_var.get()
        return True


def get_json_formatter() -> logging.Formatter:
    """Get a JSON formatter for structured logging."""
    if jsonlogger is not None:
        return jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="%",
        )
    else:
        # Fallback to standard formatter if JSON logger is not available
        return logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_standard_formatter() -> logging.Formatter:
    """Get a standard text formatter with correlation ID."""
    return logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_correlation_logging(logger: logging.Logger) -> None:
    """Setup correlation ID filtering for a logger."""
    correlation_filter = CorrelationIdFilter()
    logger.addFilter(correlation_filter)
    
    # Also add to all existing handlers
    for handler in logger.handlers:
        handler.addFilter(correlation_filter)


def log_with_correlation(
    logger: logging.Logger,
    level: int,
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log a message with correlation ID."""
    if correlation_id:
        with correlation_context(correlation_id):
            logger.log(level, message, **kwargs)
    else:
        logger.log(level, message, **kwargs)


def log_request_start(logger: logging.Logger, request: "Request", correlation_id: Optional[str] = None) -> str:
    """Log the start of a request with correlation ID."""
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    with correlation_context(correlation_id):
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
            }
        )
    
    return correlation_id


def log_request_end(logger: logging.Logger, request: "Request", correlation_id: str, status_code: int, duration: float) -> None:
    """Log the end of a request with correlation ID."""
    with correlation_context(correlation_id):
        logger.info(
            f"Request completed: {request.method} {request.url.path} - {status_code} ({duration:.3f}s)",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration": duration,
            }
        )


def log_error_with_correlation(
    logger: logging.Logger,
    message: str,
    exc_info: Optional[Exception] = None,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log an error with correlation ID."""
    if correlation_id:
        with correlation_context(correlation_id):
            logger.error(message, exc_info=exc_info, **kwargs)
    else:
        logger.error(message, exc_info=exc_info, **kwargs)


def log_warning_with_correlation(
    logger: logging.Logger,
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log a warning with correlation ID."""
    if correlation_id:
        with correlation_context(correlation_id):
            logger.warning(message, **kwargs)
    else:
        logger.warning(message, **kwargs)


def log_info_with_correlation(
    logger: logging.Logger,
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log an info message with correlation ID."""
    if correlation_id:
        with correlation_context(correlation_id):
            logger.info(message, **kwargs)
    else:
        logger.info(message, **kwargs)


def log_debug_with_correlation(
    logger: logging.Logger,
    message: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log a debug message with correlation ID."""
    if correlation_id:
        with correlation_context(correlation_id):
            logger.debug(message, **kwargs)
    else:
        logger.debug(message, **kwargs)


# Convenience functions for common logging patterns
def log_function_entry(logger: logging.Logger, function_name: str, **kwargs: Any) -> str:
    """Log function entry with correlation ID."""
    correlation_id = generate_correlation_id()
    with correlation_context(correlation_id):
        logger.debug(f"Entering function: {function_name}", extra=kwargs)
    return correlation_id


def log_function_exit(logger: logging.Logger, function_name: str, correlation_id: str, result: Any = None) -> None:
    """Log function exit with correlation ID."""
    with correlation_context(correlation_id):
        if result is not None:
            logger.debug(f"Exiting function: {function_name}", extra={"result": str(result)})
        else:
            logger.debug(f"Exiting function: {function_name}")


def log_database_operation(
    logger: logging.Logger,
    operation: str,
    table: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log database operations with correlation ID."""
    message = f"Database operation: {operation} on table {table}"
    log_info_with_correlation(logger, message, correlation_id, extra=kwargs)


def log_external_api_call(
    logger: logging.Logger,
    method: str,
    url: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any
) -> None:
    """Log external API calls with correlation ID."""
    message = f"External API call: {method} {url}"
    log_info_with_correlation(logger, message, correlation_id, extra=kwargs)
