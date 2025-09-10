"""
Math validation utilities for safe mathematical operations.

This module provides safe mathematical operations and validation functions
to prevent errors in mathematical calculations.
"""

import logging
import numpy as np
from typing import Any, List, Callable

# Setup logging
logger = logging.getLogger(__name__)

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        return a / b if b != 0 else default
    except Exception:
        return default

def safe_log(x: float, default: float = 0.0) -> float:
    """Safely calculate logarithm."""
    try:
        return np.log(x) if x > 0 else default
    except Exception:
        return default

def safe_sqrt(x: float, default: float = 0.0) -> float:
    """Safely calculate square root."""
    try:
        return np.sqrt(x) if x >= 0 else default
    except Exception:
        return default

def safe_power(x: float, y: float, default: float = 0.0) -> float:
    """Safely calculate power."""
    try:
        return x ** y
    except Exception:
        return default

def validate_finite(value: Any, name: str = "value") -> float:
    """Validate that a value is finite."""
    try:
        val = float(value)
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        return val
    except Exception as e:
        raise ValueError(f"Invalid {name}: {e}")

def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

def validate_range(value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
    """Validate that a value is in range."""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    return value

def safe_kelly_calculation(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Safely calculate Kelly criterion."""
    try:
        if avg_loss <= 0:
            return 0.0
        return (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
    except Exception:
        return 0.0

def safe_weighted_average(values: List[float], weights: List[float]) -> float:
    """Safely calculate weighted average."""
    try:
        if not values or not weights or len(values) != len(weights):
            return 0.0
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / total_weight
    except Exception:
        return 0.0

def safe_percentage_change(old_value: float, new_value: float) -> float:
    """Safely calculate percentage change."""
    try:
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    except Exception:
        return 0.0

def validate_correlation_matrix(corr_matrix: np.ndarray) -> bool:
    """Validate correlation matrix."""
    try:
        if not isinstance(corr_matrix, np.ndarray):
            return False
        if corr_matrix.ndim != 2:
            return False
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            return False
        # Check if all values are between -1 and 1
        return np.all((corr_matrix >= -1) & (corr_matrix <= 1))
    except Exception:
        return False

def safe_matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Safely calculate matrix inverse."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if regular inverse fails
        return np.linalg.pinv(matrix)
    except Exception:
        return np.eye(matrix.shape[0])

def math_safe(func: Callable, *args, default: Any = 0.0, **kwargs) -> Any:
    """Safely execute math function."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default

class MathValidationError(Exception):
    """Exception raised for math validation errors."""
    pass
