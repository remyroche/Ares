"""Error handling decorators.

Provides a tolerant `handles_errors` decorator compatible with both sync and async
functions, supporting keyword options commonly used across steps:

- exceptions: tuple[type[BaseException], ...] to catch (default: (Exception,))
- default_return / fallback: value to return on error
- context: optional string for log context
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Tuple, Type
import asyncio


def handles_errors(*decorator_args: Any, **decorator_kwargs: Any):
    exceptions: Tuple[Type[BaseException], ...] = decorator_kwargs.get("exceptions") or (Exception,)
    fallback = (
        decorator_kwargs.get("default_return")
        if "default_return" in decorator_kwargs
        else decorator_kwargs.get("fallback")
    )
    context: Optional[str] = decorator_kwargs.get("context")

    def _decorator(func: Callable[..., Any] | Callable[..., Awaitable[Any]]):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:  # type: ignore[misc]
                    prefix = f"[{context}] " if context else ""
                    print(f"Error in {func.__name__}: {prefix}{e}")
                    return fallback

            return _async_wrapper

        @wraps(func)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:  # type: ignore[misc]
                prefix = f"[{context}] " if context else ""
                print(f"Error in {func.__name__}: {prefix}{e}")
                return fallback

        return _sync_wrapper

    # Support bare @handles_errors usage without parentheses
    if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
        return _decorator(decorator_args[0])

    return _decorator

