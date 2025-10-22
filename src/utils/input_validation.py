"""
Comprehensive Input Validation Module

This module provides robust input validation for all public utility functions
with detailed error messages and type checking.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import inspect
import warnings

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationResult:
    """Result of input validation."""
    
    def __init__(self, is_valid: bool = True, errors: List[str] = None, 
                 warnings: List[str] = None, info: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.info = info or []
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str):
        """Add an info message."""
        self.info.append(message)
    
    def __bool__(self) -> bool:
        return self.is_valid

@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    strict_mode: bool = True
    allow_none: bool = False
    allow_empty: bool = False
    max_size_mb: Optional[float] = None
    required_columns: Optional[List[str]] = None
    allowed_types: Optional[List[Type]] = None
    custom_validators: Optional[List[Callable]] = None

class InputValidator:
    """Comprehensive input validator for utility functions."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logger
    
    def validate_dataframe(self, df: Any, name: str = "dataframe", 
                          config: Optional[ValidationConfig] = None) -> ValidationResult:
        """Validate DataFrame input."""
        result = ValidationResult()
        config = config or self.config
        
        # Check if None
        if df is None:
            if config.allow_none:
                result.add_info(f"{name} is None (allowed)")
                return result
            else:
                result.add_error(f"{name} cannot be None")
                return result
        
        # Check type
        if not isinstance(df, (pd.DataFrame, np.ndarray)):
            result.add_error(f"{name} must be a DataFrame or ndarray, got {type(df)}")
            return result
        
        # Convert ndarray to DataFrame for validation
        if isinstance(df, np.ndarray):
            if df.ndim == 1:
                df = pd.DataFrame(df.reshape(1, -1))
            elif df.ndim == 2:
                df = pd.DataFrame(df)
            else:
                result.add_error(f"{name} ndarray must be 1D or 2D, got {df.ndim}D")
                return result
        
        # Check if empty
        if df.empty:
            if config.allow_empty:
                result.add_warning(f"{name} is empty")
            else:
                result.add_error(f"{name} cannot be empty")
                return result
        
        # Check size
        if config.max_size_mb:
            size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            if size_mb > config.max_size_mb:
                result.add_error(f"{name} size ({size_mb:.2f}MB) exceeds limit ({config.max_size_mb}MB)")
                return result
        
        # Check required columns
        if config.required_columns:
            missing_cols = set(config.required_columns) - set(df.columns)
            if missing_cols:
                result.add_error(f"{name} missing required columns: {missing_cols}")
                return result
        
        # Check for infinite values
        if df.select_dtypes(include=[np.number]).applymap(np.isinf).any().any():
            result.add_warning(f"{name} contains infinite values")
        
        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            result.add_info(f"{name} contains {nan_count} NaN values")
        
        return result
    
    def validate_array(self, arr: Any, name: str = "array", 
                      config: Optional[ValidationConfig] = None) -> ValidationResult:
        """Validate array input."""
        result = ValidationResult()
        config = config or self.config
        
        # Check if None
        if arr is None:
            if config.allow_none:
                result.add_info(f"{name} is None (allowed)")
                return result
            else:
                result.add_error(f"{name} cannot be None")
                return result
        
        # Check type
        if not isinstance(arr, (np.ndarray, list, tuple)):
            result.add_error(f"{name} must be an array-like object, got {type(arr)}")
            return result
        
        # Convert to numpy array
        try:
            arr = np.asarray(arr)
        except Exception as e:
            result.add_error(f"{name} cannot be converted to array: {e}")
            return result
        
        # Check if empty
        if arr.size == 0:
            if config.allow_empty:
                result.add_warning(f"{name} is empty")
            else:
                result.add_error(f"{name} cannot be empty")
                return result
        
        # Check size
        if config.max_size_mb:
            size_mb = arr.nbytes / 1024 / 1024
            if size_mb > config.max_size_mb:
                result.add_error(f"{name} size ({size_mb:.2f}MB) exceeds limit ({config.max_size_mb}MB)")
                return result
        
        # Check for finite values
        if np.issubdtype(arr.dtype, np.number):
            if not np.isfinite(arr).all():
                result.add_warning(f"{name} contains non-finite values")
        
        return result
    
    def validate_path(self, path: Any, name: str = "path", 
                     must_exist: bool = False, must_be_file: bool = False,
                     must_be_dir: bool = False) -> ValidationResult:
        """Validate path input."""
        result = ValidationResult()
        
        # Check if None
        if path is None:
            result.add_error(f"{name} cannot be None")
            return result
        
        # Convert to Path
        try:
            path = Path(path)
        except Exception as e:
            result.add_error(f"{name} is not a valid path: {e}")
            return result
        
        # Check if exists
        if must_exist and not path.exists():
            result.add_error(f"{name} does not exist: {path}")
            return result
        
        # Check if file
        if must_be_file and not path.is_file():
            result.add_error(f"{name} is not a file: {path}")
            return result
        
        # Check if directory
        if must_be_dir and not path.is_dir():
            result.add_error(f"{name} is not a directory: {path}")
            return result
        
        return result
    
    def validate_function(self, func: Any, name: str = "function") -> ValidationResult:
        """Validate function input."""
        result = ValidationResult()
        
        # Check if None
        if func is None:
            result.add_error(f"{name} cannot be None")
            return result
        
        # Check if callable
        if not callable(func):
            result.add_error(f"{name} must be callable, got {type(func)}")
            return result
        
        # Check if it's a function (not a method or class)
        if not inspect.isfunction(func):
            result.add_warning(f"{name} is not a regular function: {type(func)}")
        
        return result
    
    def validate_numeric(self, value: Any, name: str = "value", 
                        min_val: Optional[float] = None, max_val: Optional[float] = None,
                        allow_inf: bool = False, allow_nan: bool = False) -> ValidationResult:
        """Validate numeric input."""
        result = ValidationResult()
        
        # Check if None
        if value is None:
            result.add_error(f"{name} cannot be None")
            return result
        
        # Check if numeric
        try:
            value = float(value)
        except (ValueError, TypeError):
            result.add_error(f"{name} must be numeric, got {type(value)}")
            return result
        
        # Check for infinity
        if not allow_inf and not np.isfinite(value):
            result.add_error(f"{name} must be finite, got {value}")
            return result
        
        # Check for NaN
        if not allow_nan and np.isnan(value):
            result.add_error(f"{name} cannot be NaN")
            return result
        
        # Check range
        if min_val is not None and value < min_val:
            result.add_error(f"{name} must be >= {min_val}, got {value}")
            return result
        
        if max_val is not None and value > max_val:
            result.add_error(f"{name} must be <= {max_val}, got {value}")
            return result
        
        return result
    
    def validate_string(self, value: Any, name: str = "string", 
                       min_length: Optional[int] = None, max_length: Optional[int] = None,
                       allow_empty: bool = True) -> ValidationResult:
        """Validate string input."""
        result = ValidationResult()
        
        # Check if None
        if value is None:
            result.add_error(f"{name} cannot be None")
            return result
        
        # Check type
        if not isinstance(value, str):
            result.add_error(f"{name} must be a string, got {type(value)}")
            return result
        
        # Check length
        if not allow_empty and len(value) == 0:
            result.add_error(f"{name} cannot be empty")
            return result
        
        if min_length is not None and len(value) < min_length:
            result.add_error(f"{name} must be at least {min_length} characters, got {len(value)}")
            return result
        
        if max_length is not None and len(value) > max_length:
            result.add_error(f"{name} must be at most {max_length} characters, got {len(value)}")
            return result
        
        return result
    
    def validate_list(self, value: Any, name: str = "list", 
                     min_length: Optional[int] = None, max_length: Optional[int] = None,
                     element_type: Optional[Type] = None) -> ValidationResult:
        """Validate list input."""
        result = ValidationResult()
        
        # Check if None
        if value is None:
            result.add_error(f"{name} cannot be None")
            return result
        
        # Check type
        if not isinstance(value, (list, tuple)):
            result.add_error(f"{name} must be a list or tuple, got {type(value)}")
            return result
        
        # Check length
        if min_length is not None and len(value) < min_length:
            result.add_error(f"{name} must have at least {min_length} elements, got {len(value)}")
            return result
        
        if max_length is not None and len(value) > max_length:
            result.add_error(f"{name} must have at most {max_length} elements, got {len(value)}")
            return result
        
        # Check element types
        if element_type is not None:
            for i, element in enumerate(value):
                if not isinstance(element, element_type):
                    result.add_error(f"{name}[{i}] must be {element_type.__name__}, got {type(element)}")
                    return result
        
        return result

# Global validator instance
_global_validator = InputValidator()

def validate_inputs(*validators: Callable):
    """Decorator to validate function inputs."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Apply validators
            for i, validator in enumerate(validators):
                if i < len(args):
                    result = validator(args[i], f"arg_{i}")
                    if not result.is_valid:
                        error_msg = f"Validation failed for {func.__name__}: {', '.join(result.errors)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_dataframe_input(required_columns: Optional[List[str]] = None,
                           max_size_mb: Optional[float] = None,
                           allow_empty: bool = False):
    """Decorator to validate DataFrame inputs."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, value in bound_args.arguments.items():
                if isinstance(value, (pd.DataFrame, np.ndarray)):
                    config = ValidationConfig(
                        required_columns=required_columns,
                        max_size_mb=max_size_mb,
                        allow_empty=allow_empty
                    )
                    result = _global_validator.validate_dataframe(value, param_name, config)
                    if not result.is_valid:
                        error_msg = f"DataFrame validation failed for {func.__name__}.{param_name}: {', '.join(result.errors)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_numeric_input(min_val: Optional[float] = None, max_val: Optional[float] = None):
    """Decorator to validate numeric inputs."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find numeric arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, value in bound_args.arguments.items():
                if isinstance(value, (int, float, np.number)):
                    result = _global_validator.validate_numeric(
                        value, param_name, min_val, max_val
                    )
                    if not result.is_valid:
                        error_msg = f"Numeric validation failed for {func.__name__}.{param_name}: {', '.join(result.errors)}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Convenience functions
def validate_dataframe(df: Any, name: str = "dataframe", **kwargs) -> ValidationResult:
    """Validate a DataFrame with the global validator."""
    config = ValidationConfig(**kwargs)
    return _global_validator.validate_dataframe(df, name, config)

def validate_array(arr: Any, name: str = "array", **kwargs) -> ValidationResult:
    """Validate an array with the global validator."""
    config = ValidationConfig(**kwargs)
    return _global_validator.validate_array(arr, name, config)

def validate_path(path: Any, name: str = "path", **kwargs) -> ValidationResult:
    """Validate a path with the global validator."""
    return _global_validator.validate_path(path, name, **kwargs)

def validate_function(func: Any, name: str = "function") -> ValidationResult:
    """Validate a function with the global validator."""
    return _global_validator.validate_function(func, name)

def validate_numeric(value: Any, name: str = "value", **kwargs) -> ValidationResult:
    """Validate a numeric value with the global validator."""
    return _global_validator.validate_numeric(value, name, **kwargs)

def validate_string(value: Any, name: str = "string", **kwargs) -> ValidationResult:
    """Validate a string with the global validator."""
    return _global_validator.validate_string(value, name, **kwargs)

def validate_list(value: Any, name: str = "list", **kwargs) -> ValidationResult:
    """Validate a list with the global validator."""
    return _global_validator.validate_list(value, name, **kwargs)

# Export main classes and functions
__all__ = [
    'ValidationSeverity', 'ValidationResult', 'ValidationConfig', 'InputValidator',
    'validate_inputs', 'validate_dataframe_input', 'validate_numeric_input',
    'validate_dataframe', 'validate_array', 'validate_path', 'validate_function',
    'validate_numeric', 'validate_string', 'validate_list'
]