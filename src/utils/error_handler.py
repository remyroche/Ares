"""Error handling utilities for the Ares project."""

def handles_errors(fallback=True, *args, **kwargs):
    """Decorator for handling errors in functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if fallback:
                    print(f"Error in {func.__name__}: {e}")
                    return None
                else:
                    raise
        return wrapper
    return decorator

def handle_errors(*args, **kwargs):
    """Alternative name for handles_errors."""
    return handles_errors(*args, **kwargs)