"""Error handling decorators with async support and flexible signature."""

from typing import Any, Callable, Iterable
import asyncio


def handles_errors(
    exceptions: Any = Exception,
    fallback: Any | None = None,
    context: str | None = None,
    default_return: Any | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to catch exceptions and return a fallback value instead.

    Supports both async and sync functions.

    Args:
        exceptions: Exception type or tuple of types to catch (default: Exception)
        fallback: Value to return when an exception is caught
        context: Optional string for logging context (ignored if not provided)
        default_return: Alias for fallback for backward compatibility
    """

    # Normalize exceptions to a tuple for isinstance checks
    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)  # type: ignore[assignment]

    # Backward compatibility alias
    if fallback is None and default_return is not None:
        fallback = default_return

    def _log_error(func: Callable[..., Any], error: Exception) -> None:
        try:
            prefix = f"[{context}] " if context else ""
            print(f"{prefix}Error in {func.__name__}: {error}")
        except Exception:
            # Best-effort logging only
            pass

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*f_args: Any, **f_kwargs: Any) -> Any:
                try:
                    return await func(*f_args, **f_kwargs)
                except exceptions as e:  # type: ignore[misc]
                    _log_error(func, e)
                    return fallback

            return async_wrapper

        def sync_wrapper(*f_args: Any, **f_kwargs: Any) -> Any:
            try:
                return func(*f_args, **f_kwargs)
            except exceptions as e:  # type: ignore[misc]
                _log_error(func, e)
                return fallback

        return sync_wrapper

    return decorator


# Placeholders for compatibility with code expecting these decorators
def converts_errors(*d_args: Any, **d_kwargs: Any):
    """No-op decorator placeholder for compatibility."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _decorator


def error_boundary(*d_args: Any, **d_kwargs: Any):
    """No-op decorator placeholder for compatibility."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn

    return _decorator
