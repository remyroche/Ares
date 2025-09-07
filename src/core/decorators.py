from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import functools
"""Core decorators for the Ares project."""

def handles_errors(*args, **kwargs) -> Callable:
    """Enhanced decorator for handling errors in functions."""
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
                        from src.utils.logger import system_logger
                        
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
                        
                    except ImportError:
                        # Fallback to simple logging if error classes not available
                        print(f'Error in {func.__name__}: {e}')
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
                        from .errors.base import AppError, ValidationError
                        from .error_classes import initialization_error, execution_error
                        from src.utils.logger import system_logger
                        
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
                        
                    except ImportError:
                        # Fallback to simple logging if error classes not available
                        print(f'Error in {func.__name__}: {e}')
                        raise
            return sync_wrapper
    return decorator

def traced(*args, **kwargs) -> Callable:
    """Tracing decorator that accepts optional parameters."""

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
                    from .errors.base import AppError, ValidationError
                    from src.utils.logger import system_logger
                    
                    # Convert to ValidationError if not already
                    if not isinstance(e, (ValidationError, AppError)):
                        error = ValidationError(f"Validation failed in {func.__name__}: {str(e)}")
                        system_logger.error(f"Validation error in {func.__name__}: {error.message}")
                        raise error
                    else:
                        system_logger.error(f"Validation error in {func.__name__}: {e.message}")
                        raise
                        
                except ImportError:
                    # Fallback to simple error handling
                    print(f'Validation error in {func.__name__}: {e}')
                    raise
        return wrapper
    return decorator

def cached(*args, **kwargs) -> None:
    """Caching decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution_time(*args, **kwargs) -> None:
    """Log execution time decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
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