"""
Structured Logging Framework

This module provides structured logging capabilities with correlation IDs,
session management, and FastAPI middleware integration.
"""

import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import Request

try:
    # Optional: only needed when JSON format is enabled
    from pythonjsonlogger import jsonlogger  # type: ignore
except Exception:  # pragma: no cover - optional dependency
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


def ensure_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Ensure a correlation ID exists, generating one if needed."""
    current = correlation_id_var.get()
    if not current or current == "-":
        new_id = correlation_id or generate_correlation_id()
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
    """Logging filter that adds correlation ID and session ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - filter is required API
        """Add correlation and session IDs to the log record."""
        try:
            record.correlation_id = get_correlation_id()
            record.session_id = session_id_var.get()
        except Exception:
            # Best-effort enrichment should not break logging
            pass
        return True


def get_json_formatter(datefmt: Optional[str] = None):
    """Get a JSON formatter for structured logging."""
    fmt = (
        "%(asctime)s %(levelname)s %(name)s %(message)s "
        "%(correlation_id)s %(session_id)s"
    )
    
    if jsonlogger is None:
        # Fallback implementation for jsonlogger
        return logging.Formatter(fmt=fmt, datefmt=datefmt)
    
    return jsonlogger.JsonFormatter(
        fmt=fmt,
        timestamp=True,
        json_ensure_ascii=False,
        json_indent=None,
        datefmt=datefmt,
    )


# FastAPI middleware utilities (optional import to avoid hard dependency)
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class CorrelationIdMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
        """Middleware that extracts or generates X-Request-ID and sets it in context."""
        
        def __init__(self, app, header_name: str = "X-Request-ID"):
            super().__init__(app)
            self.header_name = header_name
        
        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            """Process the request and add correlation ID."""
            incoming_id = request.headers.get(self.header_name)
            cid = incoming_id or generate_correlation_id()
            
            # Bind to context for downstream code
            token = correlation_id_var.set(cid)
            try:
                response = await call_next(request)
                response.headers[self.header_name] = cid
                return response
            finally:
                correlation_id_var.reset(token)
                
except Exception:
    # FastAPI is optional; if not present, users can still use logging utils
    pass


def setup_structured_logging(
    logger_name: str,
    level: int = logging.INFO,
    use_json: bool = True,
    datefmt: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with structured logging capabilities."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    console_handler.addFilter(correlation_filter)
    
    # Set formatter
    if use_json and jsonlogger is not None:
        formatter = get_json_formatter(datefmt)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - "
            "correlation_id=%(correlation_id)s session_id=%(session_id)s",
            datefmt=datefmt
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_structured_logger(
    name: str,
    level: int = logging.INFO,
    use_json: bool = True
) -> logging.Logger:
    """Get a structured logger instance."""
    return setup_structured_logging(name, level, use_json)


# Convenience function for quick setup
def quick_setup(
    name: str = "structured_logger",
    level: int = logging.INFO
) -> logging.Logger:
    """Quick setup for a structured logger."""
    return get_structured_logger(name, level)


# Example usage and testing
if __name__ == "__main__":
    # Set up a test logger
    test_logger = quick_setup("test_logger", logging.DEBUG)
    
    # Test basic logging
    test_logger.info("Basic log message")
    
    # Test with correlation context
    with correlation_context("test-correlation-123"):
        test_logger.info("Log message with correlation ID")
        test_logger.warning("Warning message in correlation context")
    
    # Test without correlation context
    test_logger.error("Error message without correlation context")
    
    print("Structured logging test completed!")
