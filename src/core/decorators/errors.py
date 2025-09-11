from src.utils.tprint import tprint

# handles_errors is defined in this file
"""Error handling decorators."""
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
# handles_errors is defined in this file

def handles_errors(*args, **kwargs) -> Callable:
    """Decorator for handling errors in functions.

    Usage:
        @handles_errors  # No arguments
        @handles_errors(Exception)  # Exception type only
        @handles_errors(fallback=False)  # Fallback value
        @handles_errors(Exception, fallback=False)  # Both
    """

    # Handle the case where decorator is used without parentheses
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return handles_errors()(args[0])

    def decorator(func: Callable) -> Callable:
        def wrapper(*func_args, **func_kwargs) -> Any:
            try:
                return func(*func_args, **func_kwargs)
            except Exception as e:
                tprint(f'Error in {func.__name__}: {e}')
                # Check if a fallback is specified in kwargs
                if 'fallback' in kwargs:
                    return kwargs['fallback']
                return None
        return wrapper
    return decorator

def converts_errors(*args, **kwargs) -> None:
    """Decorator for converting errors."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tprint(f'Error in {func.__name__}: {e}')
                return None
        return wrapper
    return decorator

def error_boundary(*args, **kwargs) -> None:
    """Decorator for error boundary."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tprint(f'Error in {func.__name__}: {e}')
                return None
        return wrapper
    return decorator