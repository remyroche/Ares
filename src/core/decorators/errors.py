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

def converts_errors(*args, **kwargs):
    """Decorator for converting errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator

def error_boundary(*args, **kwargs):
    """Decorator for error boundary."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator