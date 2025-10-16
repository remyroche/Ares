from src.utils.tprint import tprint

"""Training core decorators for the Ares project."""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import functools
import time
import inspect

def handles_errors(exceptions: tuple = (Exception,), default_return: Any = None, fallback: str = None, context: str = None) -> Callable:
    """Enhanced decorator for handling errors in training functions."""
    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    # Try to import error classes for better handling
                    try:
                        from src.utils.logger import system_logger

                        error_msg = f"Error in {func.__name__}"
                        if context:
                            error_msg += f" (context: {context})"
                        error_msg += f": {str(e)}"

                        if fallback:
                            system_logger.error(f"{fallback}: {error_msg}")
                        else:
                            system_logger.error(error_msg)

                        if default_return is not None:
                            return default_return
                        raise

                    except ImportError:
                        # Fallback to simple logging if error classes not available
                        tprint(f'Error in {func.__name__}: {e}')
                        if default_return is not None:
                            return default_return
                        raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Try to import error classes for better handling
                    try:
                        import logging

                        error_msg = f"Error in {func.__name__}"
                        if context:
                            error_msg += f" (context: {context})"
                        error_msg += f": {str(e)}"

                        if fallback:
                            system_logger.error(f"{fallback}: {error_msg}")
                        else:
                            system_logger.error(error_msg)

                        if default_return is not None:
                            return default_return
                        raise

                    except ImportError:
                        # Fallback to simple logging if error classes not available
                        tprint(f'Error in {func.__name__}: {e}')
                        if default_return is not None:
                            return default_return
                        raise
            return sync_wrapper
    return decorator

def traced(span_name: str = None) -> Callable:
    """Tracing decorator that accepts optional parameters."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                name = span_name or func.__name__
                system_logger.debug(f"Starting {name}")
                start_time = time.time()

                result = func(*args, **kwargs)

                end_time = time.time()
                duration = end_time - start_time
                system_logger.debug(f"Completed {name} in {duration:.3f}s")

                return result
            except ImportError:
                # Fallback if logger not available
                return func(*args, **kwargs)

        return wrapper
    return decorator

def validates(validation_func: Callable = None) -> Callable:
    """Enhanced validation decorator that works with training validation."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Run validation if provided
                if validation_func:
                    validation_func(*args, **kwargs)

                return func(*args, **kwargs)
            except Exception as e:
                # Try to import validation error for better handling
                try:

                    error_msg = f"Validation failed in {func.__name__}: {str(e)}"
                    system_logger.error(error_msg)
                    raise

                except ImportError:
                    # Fallback to simple error handling
                    tprint(f'Validation error in {func.__name__}: {e}')
                    raise
        return wrapper
    return decorator

def cached(cache_key_prefix: str = None, ttl: int = None) -> Callable:
    """Caching decorator for training operations."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Simple caching implementation
            cache_key = cache_key_prefix or func.__name__
            # For now, just call the function - can be enhanced with actual caching
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution_time(log_level: str = "info") -> Callable:
    """Log execution time decorator."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                start_time = time.time()

                result = func(*args, **kwargs)

                end_time = time.time()
                duration = end_time - start_time

                log_method = getattr(system_logger, log_level, system_logger.info)
                log_method(f"{func.__name__} executed in {duration:.3f}s")

                return result
            except ImportError:
                # Fallback if logger not available
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                tprint(f"{func.__name__} executed in {end_time - start_time:.3f}s")
                return result
        return wrapper
    return decorator

def log_call(log_level: str = "debug") -> Callable:
    """Log function call decorator."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                log_method = getattr(system_logger, log_level, system_logger.debug)
                log_method(f"Calling {func.__name__}")
                return func(*args, **kwargs)
            except ImportError:
                # Fallback if logger not available
                tprint(f"Calling {func.__name__}")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60) -> Callable:
    """Circuit breaker decorator for training operations."""

    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = 0
        circuit_open = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal failure_count, last_failure_time, circuit_open

            current_time = time.time()

            # Check if circuit should be closed
            if circuit_open and (current_time - last_failure_time) > recovery_timeout:
                circuit_open = False
                failure_count = 0

            if circuit_open:
                try:
                    system_logger.warning(f"Circuit breaker open for {func.__name__}")
                except ImportError:
                    tprint(f"Circuit breaker open for {func.__name__}")
                raise Exception(f"Circuit breaker is open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                failure_count = 0  # Reset on success
                return result
            except Exception as e:
                failure_count += 1
                last_failure_time = current_time

                if failure_count >= failure_threshold:
                    circuit_open = True
                    try:
                        system_logger.warning(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")
                    except ImportError:
                        tprint(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")

                raise
        return wrapper
    return decorator

def span_event(event_name: str = None) -> Callable:
    """Span event decorator for monitoring."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                name = event_name or func.__name__
                system_logger.debug(f"Span event: {name}")
                return func(*args, **kwargs)
            except ImportError:
                # Fallback if logger not available
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
    'circuit_breaker',
    'span_event'
]
