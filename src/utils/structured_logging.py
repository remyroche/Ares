from __future__ import annotations
import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
if TYPE_CHECKING:
    from fastapi import Request
try:
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='-')
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('session_id', default='-')

def get_correlation_id() -> str:
    """Return current correlation ID or '-' if unset."""
    return correlation_id_var.get()

def set_correlation_id(correlation_id: str) -> None:
    correlation_id_var.set(correlation_id)

def ensure_correlation_id() -> str:
    """Ensure a correlation ID exists in context and return it."""
    current = correlation_id_var.get()
    if not current or current == '-':
        new_id = generate_correlation_id()
        correlation_id_var.set(new_id)
        return new_id
    return current

def generate_correlation_id() -> str:
    return uuid.uuid4().hex

@contextmanager
def correlation_context(correlation_id: str | None=None) -> None:
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
    """Logging filter that injects correlation_id and session_id into records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.correlation_id = get_correlation_id()
            record.session_id = session_id_var.get()
        except Exception:
            pass
        return True

def get_json_formatter(datefmt: str | None=None) -> logging.Formatter:
    """Return a JSON formatter with standard fields plus correlation IDs.

    Falls back to a plain formatter if python-json-logger is unavailable.
    """
    fmt = '%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s %(session_id)s'
    if jsonlogger is None:
        return logging.Formatter(fmt=fmt, datefmt=datefmt)
    return jsonlogger.JsonFormatter(fmt=fmt, timestamp=True, json_ensure_ascii=False, json_indent=None, datefmt=datefmt)
try:
    from starlette.middleware.base import BaseHTTPMiddleware

    class CorrelationIdMiddleware(BaseHTTPMiddleware):
        """Middleware that extracts or generates X-Request-ID and sets it in context."""

        def __init__(self, app: Any, header_name: str='X-Request-ID') -> None:
            super().__init__(app)
            self.header_name = header_name

        async def dispatch(self, request: Request, call_next: Any) -> None:
            incoming_id = request.headers.get(self.header_name)
            cid = incoming_id or generate_correlation_id()
            token = correlation_id_var.set(cid)
            try:
                response = await call_next(request)
                response.headers[self.header_name] = cid
                return response
            finally:
                correlation_id_var.reset(token)
except Exception:
    pass