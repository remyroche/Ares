"""
Base Utilities - Core Functions Without Dependencies

This module provides basic utility functions that don't have circular dependencies.
These are the fundamental utilities that other modules can safely import.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
from datetime import datetime, date

# Import input validation (avoid circular import by using lazy import)
def _get_input_validator():
    """Lazy import of input validator to avoid circular dependencies."""
    try:
        from .input_validation import validate_path, validate_numeric, validate_string
        return validate_path, validate_numeric, validate_string
    except ImportError:
        # Return dummy functions if validation not available
        def dummy_validate(*args, **kwargs):
            from .input_validation import ValidationResult
            return ValidationResult(True)
        return dummy_validate, dummy_validate, dummy_validate


def validate_file_path(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file path exists and is a file with comprehensive validation.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if file exists and is a file, False otherwise
        
    Raises:
        ValueError: If validation fails
    """
    # Get input validator
    validate_path, _, _ = _get_input_validator()
    
    # Validate input
    result = validate_path(file_path, "file_path", must_exist=True, must_be_file=True)
    if not result.is_valid:
        raise ValueError(f"File path validation failed: {', '.join(result.errors)}")
    
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except (TypeError, ValueError):
        return False


def validate_directory_path(dir_path: Union[str, Path]) -> bool:
    """
    Validate if a directory path exists and is a directory.
    
    Args:
        dir_path: Path to validate
        
    Returns:
        True if directory exists and is a directory, False otherwise
    """
    try:
        path = Path(dir_path)
        return path.exists() and path.is_dir()
    except (TypeError, ValueError):
        return False


def create_directory_safe(dir_path: Union[str, Path], parents: bool = True) -> bool:
    """
    Safely create a directory.
    
    Args:
        dir_path: Directory path to create
        parents: Whether to create parent directories
        
    Returns:
        True if directory was created or already exists, False otherwise
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=parents, exist_ok=True)
        return True
    except (OSError, PermissionError):
        return False


def safe_read_parquet(file_path: Union[str, Path]) -> Optional['pd.DataFrame']:
    """
    Safely read a parquet file.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        if not PANDAS_AVAILABLE:
            return None
        if not validate_file_path(file_path):
            return None
        return pd.read_parquet(file_path)
    except Exception:
        return None


def safe_write_parquet(df: 'pd.DataFrame', file_path: Union[str, Path], **kwargs) -> bool:
    """
    Safely write a DataFrame to parquet.
    
    Args:
        df: DataFrame to write
        file_path: Output file path
        **kwargs: Additional arguments for to_parquet
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not PANDAS_AVAILABLE:
            return False
        # Ensure directory exists
        output_path = Path(file_path)
        create_directory_safe(output_path.parent)
        
        df.to_parquet(file_path, **kwargs)
        return True
    except Exception:
        return False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def is_dataframe_valid(df: 'pd.DataFrame') -> bool:
    """
    Check if a DataFrame is valid (not None and not empty).
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if valid, False otherwise
    """
    if not PANDAS_AVAILABLE:
        return False
    return df is not None and not df.empty


def safe_get_shape(data: Any) -> tuple:
    """
    Safely get the shape of data.
    
    Args:
        data: Data to get shape from
        
    Returns:
        Shape tuple or (0, 0) if not available
    """
    try:
        if hasattr(data, 'shape'):
            return data.shape
        elif hasattr(data, '__len__'):
            return (len(data),)
        else:
            return (0, 0)
    except Exception:
        return (0, 0)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with comprehensive validation.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
        
    Raises:
        ValueError: If validation fails
    """
    # Get input validator
    _, validate_numeric, _ = _get_input_validator()
    
    # Validate inputs
    num_result = validate_numeric(numerator, "numerator", allow_inf=False, allow_nan=False)
    if not num_result.is_valid:
        raise ValueError(f"Numerator validation failed: {', '.join(num_result.errors)}")
    
    den_result = validate_numeric(denominator, "denominator", allow_inf=False, allow_nan=False)
    if not den_result.is_valid:
        raise ValueError(f"Denominator validation failed: {', '.join(den_result.errors)}")
    
    try:
        if not NUMPY_AVAILABLE:
            return numerator / denominator if denominator != 0 else default
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
    except Exception:
        return default


def validate_finite(value: Any) -> bool:
    """
    Check if a value is finite.
    
    Args:
        value: Value to check
        
    Returns:
        True if finite, False otherwise
    """
    try:
        if not NUMPY_AVAILABLE:
            return True  # Assume finite if numpy not available
        if isinstance(value, (int, float)):
            return np.isfinite(value)
        elif hasattr(value, '__iter__'):
            return all(np.isfinite(v) for v in value if isinstance(v, (int, float)))
        return True
    except Exception:
        return False


def safe_percentage_change(current: float, previous: float, default: float = 0.0) -> float:
    """
    Calculate safe percentage change.
    
    Args:
        current: Current value
        previous: Previous value
        default: Default value if calculation fails
        
    Returns:
        Percentage change or default
    """
    try:
        if not NUMPY_AVAILABLE:
            if previous == 0:
                return default
            return (current - previous) / previous * 100
        
        if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
            return default
        
        change = (current - previous) / previous * 100
        return change if np.isfinite(change) else default
    except Exception:
        return default


def create_fallback_logger(name: str) -> logging.Logger:
    """
    Create a fallback logger with basic configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def create_fallback_decorator(func):
    """
    Create a fallback decorator that doesn't depend on external modules.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = get_logger(func.__module__)
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
