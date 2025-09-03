from __future__ import annotations

"""
Error boundary decorator for consistent error handling.

Provides a decorator that creates error boundaries with mapping,
logging, and optional recovery strategies.
"""

import logging

from src.core.errors.base import AppError
from src.core.errors.mapping import error_mapper

from .compose import P, R, uniform_wrapper
import asyncio

logger = logging.getLogger(__name__)


def handles_errors(
    *error_types: type[Exception],
    fallback: Any | None = None,
    map_to: type[AppError] | None = None,
    log_level: str = "ERROR",
    include_traceback: bool = True,
    propagate: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Create an error boundary that handles specific exceptions.

    Args:
        *error_types: Exception types to handle (defaults to Exception)
        fallback: Value to return on error (None means re-raise)
        map_to: AppError type to map exceptions to
        log_level: Logging level for errors
        include_traceback: Whether to include traceback in logs
        propagate: Whether to propagate mapped errors

    Example:
        @handles_errors(ValueError, TypeError, map_to=ValidationError)
        def process_data(data: dict) -> dict:
            return {"result": data["value"] * 2}
    """
    # Default to catching all exceptions if none specified
    exceptions = error_types or (Exception,)
    log_method = getattr(logger, log_level.lower(), logger.error)

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except exceptions as exc:
            # Map the exception
            app_error = error_mapper.map_exception(exc)

            # Override with specific mapping if provided
            if map_to and not isinstance(exc, AppError):
                app_error = map_to(
                    str(exc),
                    cause=exc,
                    details={"original_type": type(exc).__name__},
                )

            # Log the error
            log_method(
                f"Error in {func.__name__}: {app_error.message}",
                exc_info=include_traceback,
                extra={
                    "error_code": app_error.code.value,
                    "error_details": app_error.details,
                    "function": func.__name__,
                },
            )

            # Return fallback or propagate
            if fallback is not None:
                return fallback
            if propagate:
                raise app_error from exc
            raise

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        try:
            return await func(*args, **kwargs)
        except exceptions as exc:
            # Map the exception
            app_error = error_mapper.map_exception(exc)

            # Override with specific mapping if provided
            if map_to and not isinstance(exc, AppError):
                app_error = map_to(
                    str(exc),
                    cause=exc,
                    details={"original_type": type(exc).__name__},
                )

            # Log the error
            log_method(
                f"Error in {func.__name__}: {app_error.message}",
                exc_info=include_traceback,
                extra={
                    "error_code": app_error.code.value,
                    "error_details": app_error.details,
                    "function": func.__name__,
                },
            )

            # Return fallback or propagate
            if fallback is not None:
                return fallback
            if propagate:
                raise app_error from exc
            raise

    return uniform_wrapper(
        f"handles_errors({', '.join(e.__name__ for e in exceptions)})",
        sync_handler,
        async_handler,
    )


def error_boundary(
    name: str | None = None,
    log_errors: bool = True,
    capture_all: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Simple error boundary for containing errors.

    Args:
        name: Boundary name for logging
        log_errors: Whether to log errors
        capture_all: Whether to capture all exceptions (dangerous!)

    Example:
        @error_boundary(name="data_processing")
        def process_batch(items: List[dict]) -> List[dict]:
            return [transform(item) for item in items]
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        boundary_name = name or func.__name__
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if log_errors:
                logger.error(
                    f"Error in boundary '{boundary_name}': {exc}",
                    exc_info=True,
                    extra={"boundary": boundary_name},
                )

            # Only suppress if capture_all is True and it's not an AppError
            if capture_all and not isinstance(exc, AppError):
                return None
            raise

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        boundary_name = name or func.__name__
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            if log_errors:
                logger.error(
                    f"Error in boundary '{boundary_name}': {exc}",
                    exc_info=True,
                    extra={"boundary": boundary_name},
                )

            # Only suppress if capture_all is True and it's not an AppError
            if capture_all and not isinstance(exc, AppError):
                return None
            raise

    return uniform_wrapper(
        f"error_boundary({name or 'unnamed'})",
        sync_handler,
        async_handler,
    )


def converts_errors(
    mapping: dict[type[Exception], type[AppError]],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Convert specific exceptions to AppError types.

    Args:
        mapping: Dictionary mapping exception types to AppError types

    Example:
        @converts_errors({
            KeyError: NotFoundError,
            ValueError: ValidationError,
        })
        def get_user(user_id: str) -> dict:
            return users[user_id]  # KeyError -> NotFoundError
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # Check if we have a mapping for this exception
            for exc_type, error_type in mapping.items():
                if isinstance(exc, exc_type):
                    raise error_type(str(exc), cause=exc) from exc
            # No mapping found, raise original
            raise

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            # Check if we have a mapping for this exception
            for exc_type, error_type in mapping.items():
                if isinstance(exc, exc_type):
                    raise error_type(str(exc), cause=exc) from exc
            # No mapping found, raise original
            raise

    return uniform_wrapper(
        "converts_errors",
        sync_handler,
        async_handler,
    )
