
import contextvars
import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

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





def generate_correlation_id() -> str:
    return uuid.uuid4().hex


@contextmanager

class CorrelationIdFilter(logging.Filter):
    """Logging filter that injects correlation_id and session_id into records."""



# FastAPI middleware utilities (optional import to avoid hard dependency)
try:
    from starlette.middleware.base import BaseHTTPMiddleware

    class CorrelationIdMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
        """Middleware that extracts or generates X-Request-ID and sets it in context."""

        def __init__(self, app, header_name: str = "X-Request-ID"):
            super().__init__(app)
            self.header_name = header_name

except Exception:
    # FastAPI is optional; if not present, users can still use logging utils
    pass
