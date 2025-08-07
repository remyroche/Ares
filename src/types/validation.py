# src/types/validation.py

"""
Runtime type validation utilities for critical paths.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin

from .base_types import Price, Symbol, Timestamp, Volume
from .config_types import ConfigDict
from .data_types import MarketDataDict, OHLCVData
from .ml_types import FeatureDict, ModelInput

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RuntimeTypeError(Exception):
    """Exception raised when runtime type validation fails."""
    
    def __init__(self, expected_type: Type, actual_value: Any, context: str = ""):
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.context = context
        super().__init__(f"Type validation failed in {context}: expected {expected_type}, got {type(actual_value)}")


class TypeValidator:
    """Runtime type validation utilities."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type[T], context: str = "") -> T:
        """
        Validate that a value matches the expected type.
        
        Args:
            value: The value to validate
            expected_type: The expected type
            context: Context for error messages
            
        Returns:
            The validated value
            
        Raises:
            RuntimeTypeError: If validation fails
        """
        if not TypeValidator._check_type(value, expected_type):
            raise RuntimeTypeError(expected_type, value, context)
        return value
    
    @staticmethod
    def _check_type(value: Any, expected_type: Type) -> bool:
        """Check if value matches expected type."""
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        
        # Handle Union types
        if origin is Union:
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
            if args and len(args) == 2 and value:  # Check key/value types if specified
                key_type, value_type = args
                return all(
                    TypeValidator._check_type(k, key_type) and TypeValidator._check_type(v, value_type)
                    for k, v in value.items()
                )
            return True
        
        # Handle Optional types (Union[T, None])
        if origin is Union and len(args) == 2 and type(None) in args:
            if value is None:
                return True
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return TypeValidator._check_type(value, non_none_type)
        
        # Handle basic types
        if expected_type in (int, float, str, bool):
            return isinstance(value, expected_type)
        
        # Handle NewType instances (like Symbol, Price, etc.)
        if hasattr(expected_type, '__supertype__'):
            return isinstance(value, expected_type.__supertype__)
        
        # Default isinstance check
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # Fallback for complex types
            return True


def validate_config(config: Any) -> ConfigDict:
    """Validate configuration dictionary."""
    return TypeValidator.validate_type(config, ConfigDict, "configuration")


def validate_market_data(data: Any) -> MarketDataDict:
    """Validate market data structure."""
    return TypeValidator.validate_type(data, MarketDataDict, "market_data")


def validate_model_input(input_data: Any) -> ModelInput:
    """Validate ML model input structure."""
    return TypeValidator.validate_type(input_data, ModelInput, "model_input")


def validate_ohlcv_data(data: Any) -> OHLCVData:
    """Validate OHLCV data structure."""
    return TypeValidator.validate_type(data, OHLCVData, "ohlcv_data")


def type_safe(func: Callable) -> Callable:
    """
    Decorator for type-safe function execution.
    Validates inputs and outputs based on type hints.
    """
    
    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        
        # Validate input arguments
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation and param.annotation != inspect.Parameter.empty:
                try:
                    TypeValidator.validate_type(
                        param_value, 
                        param.annotation, 
                        f"{func.__name__}.{param_name}"
                    )
                except RuntimeTypeError as e:
                    logger.warning(f"Type validation warning: {e}")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Validate return value
        if sig.return_annotation and sig.return_annotation != inspect.Parameter.empty:
            try:
                TypeValidator.validate_type(
                    result, 
                    sig.return_annotation, 
                    f"{func.__name__} return value"
                )
            except RuntimeTypeError as e:
                logger.warning(f"Return type validation warning: {e}")
        
        return result
    
    return wrapper


def validate_critical_path(validator_func: Callable[[Any], T]) -> Callable[[Callable], Callable]:
    """
    Decorator for critical path type validation.
    Used for functions where type safety is crucial.
    """
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate result
            validated_result = validator_func(result)
            
            return validated_result
        
        return wrapper
    
    return decorator


# Specific validators for common types
def validate_symbol(value: Any) -> Symbol:
    """Validate symbol type."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeTypeError(Symbol, value, "symbol")
    return Symbol(value.upper())


def validate_price(value: Any) -> Price:
    """Validate price type."""
    if not isinstance(value, (int, float)) or value < 0:
        raise RuntimeTypeError(Price, value, "price")
    return Price(float(value))


def validate_volume(value: Any) -> Volume:
    """Validate volume type."""
    if not isinstance(value, (int, float)) or value < 0:
        raise RuntimeTypeError(Volume, value, "volume")
    return Volume(float(value))