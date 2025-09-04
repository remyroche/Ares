"""Error handling decorators."""

def handles_errors(*args, **kwargs):
    """Decorator for handling errors in functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator

