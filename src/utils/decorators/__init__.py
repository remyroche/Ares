"""Decorators package for utils."""

from .errors import handles_errors

# Import placeholder decorators from parent module
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from decorators import traced, validates
except ImportError:
    # Fallback implementations
    def traced(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def validates(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

__all__ = ['handles_errors', 'traced', 'validates']
