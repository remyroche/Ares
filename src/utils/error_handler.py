"""Typed, safe error-handling decorators.

Provides:
- handle_errors: generic error wrapper for sync/async functions
- handle_specific_errors: like handle_errors for a subset of exceptions
- handle_file_operations: specialized wrapper for IO operations
- handle_data_processing_errors: specialized wrapper for data processing
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Optional, Tuple, TypeVar

from .logger import get_logger

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _log_exception(context: Optional[str], exc: BaseException) -> None:
    logger = get_logger("ErrorHandler")
    prefix = f"[{context}] " if context else ""
    logger.exception(f"{prefix}Unhandled exception: {exc}")


def handle_errors(
    *,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
    default_return: Optional[T] = None,
    context: Optional[str] = None,
) -> Callable[[F], F]:
    """Generic error-handling decorator.

    - Catches `exceptions`
    - Logs the error with context
    - Returns `default_return` if provided, otherwise re-raises
    - Works for sync and async callables
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):  # type: ignore[arg-type]

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except exceptions as exc:  # type: ignore[misc]
                    _log_exception(context or func.__name__, exc)
                    if default_return is not None:
                        return default_return
                    raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as exc:  # type: ignore[misc]
                _log_exception(context or func.__name__, exc)
                if default_return is not None:
                    return default_return
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def handle_specific_errors(
    exceptions: Tuple[type[BaseException], ...],
    *,
    default_return: Optional[T] = None,
    context: Optional[str] = None,
) -> Callable[[F], F]:
    """Specialization of handle_errors for a limited set of exceptions."""

    return handle_errors(
        exceptions=exceptions,
        default_return=default_return,
        context=context,
    )


def handle_file_operations(
    *, default_return: Optional[T] = None, context: Optional[str] = None
) -> Callable[[F], F]:
    """Wrapper for file IO operations. Currently delegates to handle_errors."""

    return handle_errors(
        exceptions=(OSError, IOError, FileNotFoundError, PermissionError),
        default_return=default_return,
        context=context,
    )


def handle_data_processing_errors(
    *, default_return: Optional[T] = None, context: Optional[str] = None
) -> Callable[[F], F]:
    """Wrapper for data processing pipelines. Delegates to handle_errors."""

    return handle_errors(
        exceptions=(ValueError, TypeError, RuntimeError, KeyError, IndexError),
        default_return=default_return,
        context=context,
    )
