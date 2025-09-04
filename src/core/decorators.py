"""Core decorators for the Ares project."""

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

def traced(*args, **kwargs):
    """Tracing decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validates(*args, **kwargs):
    """Validation decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cached(*args, **kwargs):
    """Caching decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution_time(*args, **kwargs):
    """Log execution time decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_call(*args, **kwargs):
    """Log call decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def circuit_breaker(*args, **kwargs):
    """Circuit breaker decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def span_event(*args, **kwargs):
    """Span event decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
