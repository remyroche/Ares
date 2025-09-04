"""Tracing utilities."""

def with_tracing_span(*args, **kwargs):
    """Tracing span decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
