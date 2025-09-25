"""Shared logging utilities for NAS/TAS modules.

This module centralises the import of the rich ``tprint`` logging helpers used
throughout the NAS and TAS code paths.  Several legacy modules implemented their
own ``try/except`` wrappers around ``src.utils.tprint`` in order to provide
fallback console logging when the dependency was unavailable.  The duplicated
logic made it harder to keep behaviour consistent.

The helpers exposed here provide a single place where the graceful degradation
is handled.  Any consumer can simply import the symbols from this module and be
confident that they resolve to the fully featured ``tprint`` functions when the
package is installed, or to lightweight timestamped fallbacks otherwise.  This
keeps the calling code clean while guaranteeing identical semantics across
modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

TPRINT_AVAILABLE = False

try:  # pragma: no cover - passthrough import
    from src.utils.tprint import (  # type: ignore
        tprint,
        tprint_debug,
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
        tprint_progress,
        tprint_performance,
        tprint_timer,
        tprint_structured,
        configure_tprint,
        TPrintConfig,
        LogLevel,
    )
    TPRINT_AVAILABLE = True
except Exception:  # pragma: no cover - executed only when dependency missing
    # Provide minimal timestamped fallbacks that mirror the signature of the
    # original helpers.  Keeping the same callable surface allows modules to
    # adopt the shared import without additional guard clauses.

    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _wrap(prefix: str) -> Callable[..., None]:
        def _printer(*args, **kwargs) -> None:
            print(f"[{_timestamp()}] {prefix}", *args, **kwargs)

        return _printer

    tprint = _wrap("LOG:")
    tprint_debug = _wrap("DEBUG:")
    tprint_info = _wrap("INFO:")
    tprint_warning = _wrap("WARNING:")
    tprint_error = _wrap("ERROR:")
    tprint_success = _wrap("SUCCESS:")
    tprint_structured = _wrap("STRUCTURED:")

    def tprint_progress(step: int, total: int, message: str = "", **kwargs) -> None:
        percentage = (step / total) * 100 if total else 0
        print(f"[{_timestamp()}] PROGRESS: {step}/{total} ({percentage:.1f}%) {message}")

    def tprint_performance(operation: str, duration: float, **kwargs) -> None:
        print(f"[{_timestamp()}] PERFORMANCE: {operation} took {duration:.3f}s")

    def tprint_timer(operation: str, level: str | None = None):  # pragma: no cover - context manager
        from contextlib import contextmanager

        @contextmanager
        def timer():
            start = datetime.now()
            try:
                yield
            finally:
                duration = (datetime.now() - start).total_seconds()
                tprint_performance(operation, duration)

        return timer()

    # Lightweight stand-ins used only when the real module is unavailable.
    def configure_tprint(*_, **__) -> None:  # pragma: no cover - noop fallback
        return None

    class TPrintConfig:  # pragma: no cover - simple namespace
        def __init__(self, *_, **__):
            pass

    class LogLevel:  # pragma: no cover - provides enum-like compatibility
        INFO = "INFO"
        DEBUG = "DEBUG"
        WARNING = "WARNING"
        ERROR = "ERROR"

__all__ = [
    "TPRINT_AVAILABLE",
    "tprint",
    "tprint_debug",
    "tprint_info",
    "tprint_warning",
    "tprint_error",
    "tprint_success",
    "tprint_progress",
    "tprint_performance",
    "tprint_timer",
    "tprint_structured",
    "configure_tprint",
    "TPrintConfig",
    "LogLevel",
]
