from src.utils.tprint import tprint
from src.utils.initialization_guard import init_guard

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import functools
"""Core decorators for the Ares project."""

if init_guard.mark_initialized("core.decorators"):
    tprint("DEBUG: decorators.py module starting to load...")

def handles_errors(*args, **kwargs) -> Callable:
    """Enhanced decorator for handling errors in functions."""
    tprint("DEBUG: handles_errors decorator function defined")
    import inspect

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Try to import error classes for better handling
                    try:
                        from .errors.base import AppError, ValidationError
                        from .error_classes import initialization_error, execution_error

                        # Safe logger import with fallback
                        def get_safe_logger():
                            try:
                                from src.utils.logger import system_logger
                                return system_logger
                            except (ImportError, AttributeError, Exception):
                                # Fallback to basic logging
                                import logging
                                return logging.getLogger('AresFallback')

                        system_logger = get_safe_logger()

                        # If it's already an AppError, re-raise it
                        if isinstance(e, AppError):
                            system_logger.error(f"AppError in {func.__name__}: {e.message}")
                            raise

                        # Convert generic exceptions to appropriate AppError types
                        if 'validation' in str(e).lower() or 'invalid' in str(e).lower():
                            error = ValidationError(f"Validation failed in {func.__name__}: {str(e)}")
                        elif 'init' in str(e).lower() or 'setup' in str(e).lower():
                            error = initialization_error(f"Initialization failed in {func.__name__}: {str(e)}")
                        else:
                            error = execution_error(f"Execution failed in {func.__name__}: {str(e)}")

                        system_logger.error(f"Error in {func.__name__}: {error.message}")
                        raise error

                    except (ImportError, AttributeError):
                        # Fallback to simple logging if error classes not available or circular import
                        tprint(f'Error in {func.__name__}: {e}')
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Try to import error classes for better handling
                    try:

                        # Safe logger import with fallback
                        def get_safe_logger():
                            try:
                                return system_logger
                            except (ImportError, AttributeError, Exception):
                                # Fallback to basic logging
                                return logging.getLogger('AresFallback')

                        system_logger = get_safe_logger()

                        # If it's already an AppError, re-raise it
                        if isinstance(e, AppError):
                            system_logger.error(f"AppError in {func.__name__}: {e.message}")
                            raise

                        # Convert generic exceptions to appropriate AppError types
                        if 'validation' in str(e).lower() or 'invalid' in str(e).lower():
                            error = ValidationError(f"Validation failed in {func.__name__}: {str(e)}")
                        elif 'init' in str(e).lower() or 'setup' in str(e).lower():
                            error = initialization_error(f"Initialization failed in {func.__name__}: {str(e)}")
                        else:
                            error = execution_error(f"Execution failed in {func.__name__}: {str(e)}")

                        system_logger.error(f"Error in {func.__name__}: {error.message}")
                        raise error

                    except (ImportError, AttributeError):
                        # Fallback to simple logging if error classes not available or circular import
                        tprint(f'Error in {func.__name__}: {e}')
                        raise
            return sync_wrapper
    return decorator

def traced(*args, **kwargs) -> Callable:
    """Tracing decorator that accepts optional parameters."""
    tprint("DEBUG: traced decorator function defined")

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validates(*args, **kwargs) -> Callable:
    """Enhanced validation decorator that works with error classes."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to import validation error for better handling
                try:

                    # Safe logger import with fallback
                    def get_safe_logger():
                        try:
                            return system_logger
                        except (ImportError, AttributeError, Exception):
                            # Fallback to basic logging
                            return logging.getLogger('AresFallback')

                    system_logger = get_safe_logger()

                    # Convert to ValidationError if not already
                    if not isinstance(e, (ValidationError, AppError)):
                        error = ValidationError(f"Validation failed in {func.__name__}: {str(e)}")
                        system_logger.error(f"Validation error in {func.__name__}: {error.message}")
                        raise error
                    else:
                        system_logger.error(f"Validation error in {func.__name__}: {e.message}")
                        raise

                except (ImportError, AttributeError):
                    # Fallback to simple error handling if imports fail or circular dependency
                    tprint(f'Validation error in {func.__name__}: {e}')
                    raise
        return wrapper

    # If called as @validates (without parentheses), args[0] will be the function
    if args and callable(args[0]):
        return decorator(args[0])
    # If called as @validates() or @validates(param=value), return the decorator
    else:
        return decorator

def cached(*args, **kwargs) -> None:
    """Caching decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution_time(func_or_callable=None, **kwargs):
    """
    Log execution time decorator.

    Can be used as:
    @log_execution_time
    @log_execution_time()
    @log_execution_time(param=value)
    """
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*func_args, **func_kwargs):
            start_time = time.time()
            try:
                result = func(*func_args, **func_kwargs)
                duration = time.time() - start_time
                # Try to get a logger, fallback to print if not available
                try:
                    logger = logging.getLogger(func.__module__)
                    logger.info(f"{func.__name__} executed in {duration:.2f}s")
                except:
                    tprint(f"{func.__name__} executed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                try:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
                except:
                    tprint(f"{func.__name__} failed after {duration:.2f}s: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*func_args, **func_kwargs):
            start_time = time.time()
            try:
                result = await func(*func_args, **func_kwargs)
                duration = time.time() - start_time
                # Try to get a logger, fallback to print if not available
                try:
                    logger = logging.getLogger(func.__module__)
                    logger.info(f"{func.__name__} executed in {duration:.2f}s")
                except:
                    tprint(f"{func.__name__} executed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                try:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
                except:
                    tprint(f"{func.__name__} failed after {duration:.2f}s: {e}")
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    # If called as @log_execution_time (no parentheses)
    if func_or_callable is not None and callable(func_or_callable):
        return decorator(func_or_callable)

    # If called as @log_execution_time() or @log_execution_time(param=value)
    return decorator

def log_call(*args, **kwargs) -> None:
    """Log call decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def circuit_breaker(*args, **kwargs) -> None:
    """Circuit breaker decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def span_event(*args, **kwargs) -> None:
    """Span event decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Export all decorators for external use
__all__ = [
    'handles_errors',
    'traced',
    'validates',
    'cached',
    'log_execution_time',
    'log_call',
    'span_event'
]

if init_guard.is_initialized("core.decorators"):
    tprint("DEBUG: decorators.py module loaded successfully!")
