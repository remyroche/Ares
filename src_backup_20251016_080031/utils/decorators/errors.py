from src.utils.tprint import tprint


"""Error handling decorators with async/sync support and flexible signature.

This module provides a robust `handles_errors` decorator compatible with various
call sites across the codebase. It supports both synchronous and asynchronous
functions, optional exception filtering, default/fallback return values, and
per-exception handlers.
"""
from ..logger import system_logger

import inspect
import traceback
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Tuple, Type

def _resolve_default_return(default_return: Any, *args: Any, **kwargs: Any) -> Any:
    """Return default value, calling if it's a zero-arg or flexible callable."""
    try:
        if callable(default_return):
            try:
                return default_return(*args, **kwargs)
            except TypeError:
                return default_return()
        return default_return
    except Exception:
        return None

def handles_errors(func: Optional[Callable[..., Any]]=None, *, exceptions: Optional[Iterable[Type[BaseException]]]=None, exception_types: Optional[Iterable[Type[BaseException]]]=None, default_return: Any = None, fallback: Any = None, context: Optional[str]=None, error_handlers: Optional[Dict[Type[BaseException], Tuple[Any, str] | Any]]=None) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Awaitable[Any]]:
    """Decorator to handle errors consistently across sync/async functions.

    Parameters
    - exceptions / exception_types: Iterable of exception classes to catch. Defaults to (Exception,).
    - default_return / fallback: Value (or callable) to return on error if no specific handler is found.
    - context: Optional string describing the operation, used for logging.
    - error_handlers: Mapping of Exception type to either a return value or a (return, message) tuple.

    Notes
    - Works with or without parentheses: @handles_errors or @handles_errors(...)
    - If wrapping an async function, the wrapper is async and awaits the function
    - If default_return/fallback is callable, it will be invoked to produce the value
    """
    catch_exceptions = tuple(exceptions or exception_types or (BaseException,))
    if catch_exceptions == (BaseException,):
        catch_exceptions = (Exception,)
    default_value = default_return if default_return is not None else fallback
    handlers = error_handlers or {}

    def _log_error(err: BaseException, fn_name: str) -> None:
        import logging
        
        try:
            if system_logger is not None:
                prefix = f'Error in {fn_name}'
                if context:
                    prefix += f' (context: {context})'
                system_logger.exception(f'{prefix}: {err}')
                return
        except Exception:
            pass
        ctx = f' (context: {context})' if context else ''
        tprint(f'Error in {fn_name}{ctx}: {err}\n{traceback.format_exc()}')

    def _handle_with_mapping(err: BaseException, *args: Any, **kwargs: Any) -> Any:
        for exc_type, ret in handlers.items():
            try:
                if isinstance(err, exc_type):
                    if isinstance(ret, tuple) and len(ret) >= 1:
                        return ret[0]
                    return ret
            except Exception:
                continue
        return _resolve_default_return(default_value, *args, **kwargs)

    def _decorate(f: Callable[..., Any]) -> Callable[..., Any] | Callable[..., Awaitable[Any]]:
        if inspect.iscoroutinefunction(f):

            async def _async_wrapper(*a: Any, **kw: Any) -> Any:
                try:
                    return await f(*a, **kw)
                except catch_exceptions as err:
                    _log_error(err, f.__name__)
                    return _handle_with_mapping(err, *a, **kw)
            _async_wrapper.__name__ = getattr(f, '__name__', 'wrapped_async')
            _async_wrapper.__doc__ = getattr(f, '__doc__', None)
            return _async_wrapper

        def _sync_wrapper(*a: Any, **kw: Any) -> Any:
            try:
                return f(*a, **kw)
            except catch_exceptions as err:
                _log_error(err, f.__name__)
                return _handle_with_mapping(err, *a, **kw)
        _sync_wrapper.__name__ = getattr(f, '__name__', 'wrapped')
        _sync_wrapper.__doc__ = getattr(f, '__doc__', None)
        return _sync_wrapper
    if callable(func):
        return _decorate(func)
    return _decorate

def converts_errors(*_args: Any, **_kwargs: Any) -> None:
    """No-op decorator placeholder for compatibility."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn
    return _decorator

def error_boundary(*_args: Any, **_kwargs: Any) -> None:
    """No-op decorator placeholder for compatibility."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn
    return _decorator