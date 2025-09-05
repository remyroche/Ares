"""Decorators package for utils."""
from .errors import handles_errors
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
try:
    from decorators import traced, validates
except ImportError:

    def traced(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def validates(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator
__all__ = ['handles_errors', 'traced', 'validates']