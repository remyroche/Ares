from __future__ import annotations

"""
Decorator composition utilities with a uniform wrapper.

Provides a tiny, typed utility that wraps sync/async functions consistently,
preserving signature and metadata.
"""

import asyncio
import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")

# Marker to track wrapped functions
WRAPPER_MARKER = "_decorator_wrapped"


def is_wrapped(func: Callable[..., Any]) -> bool:
    """Check if a function has been wrapped by our decorator system."""
    return hasattr(func, WRAPPER_MARKER)


def mark_wrapped(func: Callable[..., Any]) -> None:
    """Mark a function as wrapped to prevent double-application."""
    setattr(func, WRAPPER_MARKER, True)


def uniform_wrapper(
    decorator_name: str,
    sync_handler: Callable[[Callable[P, R], P.args, P.kwargs], R],
    async_handler: Callable[[Callable[P, R], P.args, P.kwargs], R] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Create a uniform wrapper that handles both sync and async functions.

    This is the single wrapper pattern that all decorators should use to ensure
    consistency, type preservation, and proper metadata handling.

    Args:
        decorator_name: Name for debugging/logging purposes
        sync_handler: Handler for synchronous functions
        async_handler: Handler for async functions (defaults to sync_handler)

    Returns:
        A decorator that preserves types and handles sync/async uniformly
    """
    if async_handler is None:
        async_handler = sync_handler

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Prevent double-wrapping
        if is_wrapped(func):
            return func

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return await async_handler(func, *args, **kwargs)

            mark_wrapped(async_wrapper)
            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return sync_handler(func, *args, **kwargs)

        mark_wrapped(sync_wrapper)
        return cast("Callable[P, R]", sync_wrapper)

    # Store decorator metadata
    decorator.__name__ = decorator_name
    decorator.__qualname__ = decorator_name

    return decorator


def compose(
    *decorators: Callable[[Callable[P, R]], Callable[P, R]]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Compose multiple decorators into a single decorator.

    Decorators are applied in reverse order (bottom-up), matching Python's
    decorator stacking behavior.

    Example:
        @compose(decorator1, decorator2, decorator3)
        def func(): ...

        # Equivalent to:
        @decorator1
        @decorator2
        @decorator3
        def func(): ...
    """

    def composed_decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Apply decorators in reverse order
        result = func
        for decorator in reversed(decorators):
            result = decorator(result)
        return result

    # Create a meaningful name
    names = [getattr(d, "__name__", "unknown") for d in decorators]
    composed_decorator.__name__ = f"compose({', '.join(names)})"

    return composed_decorator


def ensure_async(func: Callable[P, R]) -> Callable[P, R | Callable[..., R]]:
    """
    Ensure a function is async, wrapping sync functions in an async wrapper.

    This is useful for decorators that need to work with async operations
    but want to support sync functions.
    """
    if asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    async def async_wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        # Run sync function in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )

    return cast("Callable[P, R]", async_wrapped)


def ensure_sync(func: Callable[P, R]) -> Callable[P, R]:
    """
    Ensure a function is sync, creating a sync wrapper for async functions.

    Note: This should be used carefully as it can block the event loop.
    """
    if not asyncio.iscoroutinefunction(func):
        return func

    @functools.wraps(func)
    def sync_wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        # Try to get existing event loop
        try:
            asyncio.get_running_loop()
            # We're in an async context, but need sync
            # This is generally not recommended
            msg = f"Cannot call async function {func.__name__} synchronously from async context"
            raise RuntimeError(
                msg,
            )
        except RuntimeError:
            # No running loop, safe to create one
            return asyncio.run(func(*args, **kwargs))

    return cast("Callable[P, R]", sync_wrapped)


# Decorator metadata helpers
def get_decorator_metadata(
    func: Callable[..., Any], key: str, default: Any = None
) -> Any:
    """Get metadata stored by decorators on a function."""
    return getattr(func, f"_decorator_meta_{key}", default)


def set_decorator_metadata(func: Callable[..., Any], key: str, value: Any) -> None:
    """Set metadata on a function for decorators to use."""
    setattr(func, f"_decorator_meta_{key}", value)


def copy_decorator_metadata(
    source: Callable[..., Any], target: Callable[..., Any]
) -> None:
    """Copy all decorator metadata from source to target function."""
    for attr in dir(source):
        if attr.startswith("_decorator_meta_"):
            setattr(target, attr, getattr(source, attr))
