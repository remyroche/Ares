"""Performance monitoring utilities."""

def performance_monitor(*args, **kwargs):
    """Performance monitoring decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
