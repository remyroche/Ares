"""Validation decorators for data operations."""

def validate_dataframe_operation(*args, **kwargs):
    """Validate dataframe operation decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
