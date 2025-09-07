"""Decorators package for utils."""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Import from the actual decorators module
try:
    from .errors import handles_errors
    # Create fallback decorators for traced and validates
    def traced(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def validates(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
except (ImportError, ModuleNotFoundError):
    # Fallback decorators
    def handles_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def traced(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def validates(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

__all__ = ['handles_errors', 'traced', 'validates']