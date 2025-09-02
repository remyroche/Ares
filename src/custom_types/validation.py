# src/types/validation.py

"""
Runtime type validation utilities for critical paths.
"""

from collections.abc import Callable
import logging
from typing import Any, TypeVar, Union, get_origin, get_args
import types
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RuntimeTypeError(Exception):
    """Exception raised when type validation fails."""
    
    def __init__(self, expected_type: type, actual_value: Any, context: str):
        """Initialize RuntimeTypeError."""
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.context = context
        super().__init__(
            f"Type validation failed in {context}: expected {expected_type}, got {type(actual_value)}",
        )


class TypeValidator:
    """Runtime type validation utilities."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, context: str) -> Any:
        """Validate that a value matches the expected type."""
        if not TypeValidator._check_type(value, expected_type):
            raise RuntimeTypeError(expected_type, value, context)
        return value
    
    @staticmethod
    def _check_type(value: Any, expected_type: type) -> bool:
        """Check if a value matches the expected type."""
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        
        # Handle Union types (including | syntax)
        if origin in (Union, types.UnionType):
            return any(TypeValidator._check_type(value, arg) for arg in args)
        
        # Handle List types
        if origin is list:
            if not isinstance(value, list):
                return False
            if args and value:  # Check element types if specified and list not empty
                return all(TypeValidator._check_type(item, args[0]) for item in value)
            return True
        
        # Handle Dict types
        if origin is dict:
            if not isinstance(value, dict):
                return False
            if args and value:  # Check key/value types if specified and dict not empty
                key_type, value_type = args
                return all(
                    TypeValidator._check_type(k, key_type) and 
                    TypeValidator._check_type(v, value_type)
                    for k, v in value.items()
                )
            return True
        
        # Handle basic types
        return isinstance(value, expected_type)


def validate_market_data(data: dict) -> dict:
    """Validate market data structure."""
    required_fields = ["symbol", "price", "timestamp"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    return data


def validate_model_input(data: dict) -> dict:
    """Validate ML model input data."""
    if "features" not in data:
        raise ValueError("Missing required field: features")
    if not isinstance(data["features"], (list, dict)):
        raise ValueError("Features must be a list or dict")
    return data


def handle_errors(
    exceptions: tuple = (Exception,),
    default_return: Any = None,
    context: str = "unknown",
):
    """Decorator to handle errors gracefully."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error(f"Error in {context}: {e}")
                return default_return
        return wrapper
    return decorator
