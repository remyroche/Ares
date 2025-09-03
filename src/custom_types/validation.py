from __future__ import annotations
'\nRuntime type validation utilities for critical paths.\n'
import inspect
import logging
import types
from functools import wraps
from typing import TypeVar, Union, get_args, get_origin
from .base_types import Price, Symbol, Volume
from .config_types import ConfigDict
from .data_types import MarketDataDict, OHLCVData
from .ml_types import ModelInput
logger = logging.getLogger(__name__)
T = TypeVar('T')

class RuntimeTypeError(Exception):
    """Exception raised when runtime type validation fails."""

    def __init__(self, expected_type: Any, actual_value: Any, context: str='') -> None:
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.context = context
        super().__init__(f'Type validation failed in {context}: expected {expected_type}, got {type(actual_value)}')

class TypeValidator:
    """Runtime type validation utilities."""

    @staticmethod
    def validate_type(value: Any, expected_type: Any, context: str='') -> T:
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
    def _check_type(value: Any, expected_type: Any) -> bool:
        """Check if value matches expected type."""
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        if origin in (Union, types.UnionType):
            return any((TypeValidator._check_type(value, arg) for arg in args))
        if origin is list:
            if not isinstance(value, list):
                return False
            if args and value:
                return all((TypeValidator._check_type(item, args[0]) for item in value))
            return True
        if origin is dict:
            if not isinstance(value, dict):
                return False
            if args and len(args) == 2 and value:
                key_type, value_type = args
                return all((TypeValidator._check_type(k, key_type) and TypeValidator._check_type(v, value_type) for k, v in value.items()))
            return True
        if origin in (Union, types.UnionType) and len(args) == 2 and (type(None) in args):
            if value is None:
                return True
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return TypeValidator._check_type(value, non_none_type)
        if expected_type in (int, float, str, bool):
            return isinstance(value, expected_type)
        if hasattr(expected_type, '__supertype__'):
            return isinstance(value, expected_type.__supertype__)
        try:
            return isinstance(value, expected_type)
        except TypeError:
            return True

def validate_config(config: Any) -> ConfigDict:
    """Validate configuration dictionary."""
    return TypeValidator.validate_type(config, ConfigDict, 'configuration')

def validate_market_data(data: Any) -> MarketDataDict:
    """Validate market data structure."""
    return TypeValidator.validate_type(data, MarketDataDict, 'market_data')

def validate_model_input(input_data: Any) -> ModelInput:
    """Validate ML model input structure."""
    return TypeValidator.validate_type(input_data, ModelInput, 'model_input')

def validate_ohlcv_data(data: Any) -> OHLCVData:
    """Validate OHLCV data structure."""
    return TypeValidator.validate_type(data, OHLCVData, 'ohlcv_data')

def type_safe(func: Callable) -> Callable:
    """
    Decorator for type-safe function execution.
    Validates inputs and outputs based on type hints.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation and param.annotation != inspect.Parameter.empty:
                try:
                    TypeValidator.validate_type(param_value, param.annotation, f'{func.__name__}.{param_name}')
                except RuntimeTypeError as e:
                    logger.warning(validation_error(f'Type validation warning: {e}'))
        result = func(*args, **kwargs)
        if sig.return_annotation and sig.return_annotation != inspect.Parameter.empty:
            try:
                TypeValidator.validate_type(result, sig.return_annotation, f'{func.__name__} return value')
            except RuntimeTypeError as e:
                logger.warning(validation_error(f'Return type validation warning: {e}'))
        return result
    return wrapper

def validate_critical_path(validator_func: Callable[[Any], T]) -> Callable[[Callable], Callable]:
    """
    Decorator for critical path type validation.
    Used for functions where type safety is crucial.
    """

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            try:
                return validator_func(result)
            except RuntimeTypeError as e:
                logger.exception(validation_error(f'Critical path type validation failed: {e}'))
                raise
        return wrapper
    return decorator

def validate_symbol(value: Any) -> Symbol:
    """Validate symbol type."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeTypeError(Symbol, value, 'symbol')
    return Symbol(value.upper())

def validate_price(value: Any) -> Price:
    """Validate price type."""
    if not isinstance(value, int | float) or value < 0:
        raise RuntimeTypeError(Price, value, 'price')
    return Price(float(value))

def validate_volume(value: Any) -> Volume:
    """Validate volume type."""
    if not isinstance(value, int | float) or value < 0:
        raise RuntimeTypeError(Volume, value, 'volume')
    return Volume(float(value))

def validate_type(value: Any, expected_type: type[T], context: str='') -> T:
    """Validate that a value matches the expected type."""
    return TypeValidator.validate_type(value, expected_type, context)