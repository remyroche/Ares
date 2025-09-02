"""
Structured logging utilities for correlation tracking.
"""

import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import Optional

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
    """Logging filter that adds correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record."""
        record.correlation_id = get_correlation_id()
        return True


def setup_correlation_logging(logger: logging.Logger) -> None:
    """Set up correlation ID logging for a logger."""
    correlation_filter = CorrelationIdFilter()
    logger.addFilter(correlation_filter)
    
    # Add correlation ID to log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
    )
    
    for handler in logger.handlers:
        handler.setFormatter(formatter)
