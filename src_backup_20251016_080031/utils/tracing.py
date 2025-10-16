from typing import Dict, List, Optional, Union, Any, Tuple
"""Tracing utilities."""

def with_tracing_span(*args, **kwargs) -> None:
    """Tracing span decorator."""

    def decorator(func: Callable) -> None:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator