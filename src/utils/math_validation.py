"""
Math validation utilities for safe mathematical operations.

This module provides safe mathematical operations and validation functions
to prevent errors in mathematical calculations.
"""

import logging
import numpy as np
from typing import Any, List, Callable, Optional

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

def validate_finite(value: Any, name: str = "value") -> Any:
    """Validate that a value is finite."""
    try:
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError(f"{name} cannot be empty")
            # Check for non-finite values using explicit boolean array handling
            finite_mask = np.isfinite(value)
            has_non_finite = not finite_mask.all()
            if has_non_finite:
                non_finite_count = np.sum(~finite_mask)
                raise ValueError(f"{name} contains {non_finite_count} non-finite values (NaN or inf)")
            return value

        # Handle scalar values - check if it's a single-element array first
        if hasattr(value, '__len__') and len(value) == 1:
            # Single-element array or list
            val = float(value[0])
        elif hasattr(value, '__len__') and len(value) > 1:
            # Multi-element array - convert to numpy array for validation
            val_array = np.array(value)
            finite_mask = np.isfinite(val_array)
            if not finite_mask.all():
                raise ValueError(f"{name} contains non-finite values")
            return val_array
        else:
            # Scalar value
            val = float(value)

        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        return val
    except Exception as e:
        raise ValueError(f"Invalid {name}: {e}")

def validate_array_finite(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate that an array contains only finite values."""
    if array is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")

    # Check for non-finite values
    finite_mask = np.isfinite(array)
    if not finite_mask.all():
        non_finite_count = np.sum(~finite_mask)
        raise ValueError(f"{name} contains {non_finite_count} non-finite values (NaN or inf)")

    return array

def validate_matrix_finite(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Validate that a matrix contains only finite values."""
    if matrix is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(matrix)}")

    if matrix.size == 0:
        raise ValueError(f"{name} cannot be empty")

    # Check for non-finite values
    finite_mask = np.isfinite(matrix)
    if not finite_mask.all():
        non_finite_count = np.sum(~finite_mask)
        raise ValueError(f"{name} contains {non_finite_count} non-finite values (NaN or inf)")

    return matrix

def validate_positive(value, name: str = "value") -> Optional[float]:
    """Validate that a value is positive."""
    if value is None:
        return None
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


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate that a value is a valid probability (between 0 and 1)."""
    try:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
        return float(value)
    except TypeError:
        raise ValueError(f"{name} must be numeric")


def validate_numeric_array(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Validate that an array contains only numeric values and is finite."""
    if array is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array)}")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")

    # Check for non-finite values
    finite_mask = np.isfinite(array)
    if not finite_mask.all():
        non_finite_count = np.sum(~finite_mask)
        raise ValueError(f"{name} contains {non_finite_count} non-finite values (NaN or inf)")

    # Check if array contains numeric data
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric data, got dtype {array.dtype}")

    return array

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

def safe_correlation(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    """Safely calculate correlation coefficient between two arrays."""
    try:
        if x is None or y is None:
            return default
        if len(x) != len(y):
            return default
        if len(x) <= 1:
            return default

        # Remove NaN and infinite values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid_mask):
            return default

        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) <= 1:
            return default

        # Calculate correlation coefficient
        corr_matrix = np.corrcoef(x_clean, y_clean)
        if corr_matrix.shape != (2, 2):
            return default

        corr = corr_matrix[0, 1]

        # Ensure result is valid
        if not np.isfinite(corr):
            return default

        return corr

    except Exception:
        return default

def safe_covariance(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
    """Safely calculate covariance between two arrays."""
    try:
        if x is None or y is None:
            return default
        if len(x) != len(y):
            return default
        if len(x) <= 1:
            return default

        # Remove NaN and infinite values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid_mask):
            return default

        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) <= 1:
            return default

        # Calculate covariance
        cov = np.cov(x_clean, y_clean)[0, 1]

        # Ensure result is valid
        if not np.isfinite(cov):
            return default

        return cov

    except Exception:
        return default

def safe_mean(x: np.ndarray, axis: Optional[int] = None, default: float = 0.0) -> float:
    """Safely calculate mean of array."""
    try:
        if x is None or len(x) == 0:
            return default

        # Remove NaN and infinite values
        valid_mask = np.isfinite(x)
        if not np.any(valid_mask):
            return default

        x_clean = x[valid_mask] if axis is None else x[valid_mask]
        if len(x_clean) == 0:
            return default

        mean_val = np.mean(x_clean, axis=axis)

        # Ensure result is valid
        if not np.isfinite(mean_val):
            return default

        return mean_val

    except Exception:
        return default

def safe_std(x: np.ndarray, axis: Optional[int] = None, default: float = 0.0) -> float:
    """Safely calculate standard deviation of array."""
    try:
        if x is None or len(x) <= 1:
            return default

        # Remove NaN and infinite values
        valid_mask = np.isfinite(x)
        if not np.any(valid_mask):
            return default

        x_clean = x[valid_mask] if axis is None else x[valid_mask]
        if len(x_clean) <= 1:
            return default

        std_val = np.std(x_clean, axis=axis, ddof=1)  # Use ddof=1 for sample standard deviation

        # Ensure result is valid
        if not np.isfinite(std_val):
            return default

        return std_val

    except Exception:
        return default

def safe_percentile(x: np.ndarray, percentile: float = 50.0, default: float = 0.0) -> float:
    """Safely calculate percentile of array."""
    try:
        if x is None or len(x) == 0:
            return default
        if not (0 <= percentile <= 100):
            return default

        # Remove NaN and infinite values
        valid_mask = np.isfinite(x)
        if not np.any(valid_mask):
            return default

        x_clean = x[valid_mask]
        if len(x_clean) == 0:
            return default

        percentile_val = np.percentile(x_clean, percentile)

        # Ensure result is valid
        if not np.isfinite(percentile_val):
            return default

        return percentile_val

    except Exception:
        return default

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

def check_for_inf_nan(data, name="data"):
    """Check for infinite or NaN values in data."""
    import pandas as pd

    if isinstance(data, (pd.DataFrame, pd.Series)):
        has_inf = np.isinf(data).any().any() if isinstance(data, pd.DataFrame) else np.isinf(data).any()
        has_nan = data.isna().any().any() if isinstance(data, pd.DataFrame) else data.isna().any()
    else:
        has_inf = np.isinf(data).any()
        has_nan = np.isnan(data).any()

    if has_inf:
        logger.warning(f"{name} contains infinite values")
    if has_nan:
        logger.warning(f"{name} contains NaN values")

    return not (has_inf or has_nan)


def check_for_nans(data, name="data"):
    """Check for NaN values in data."""
    import pandas as pd

    if isinstance(data, (pd.DataFrame, pd.Series)):
        has_nan = data.isna().any().any() if isinstance(data, pd.DataFrame) else data.isna().any()
    else:
        has_nan = np.isnan(data).any()

    if has_nan:
        logger.warning(f"{name} contains NaN values")

    return not has_nan


def check_for_infs(data, name="data"):
    """Check for infinite values in data."""
    import pandas as pd

    if isinstance(data, (pd.DataFrame, pd.Series)):
        has_inf = np.isinf(data).any().any() if isinstance(data, pd.DataFrame) else np.isinf(data).any()
    else:
        has_inf = np.isinf(data).any()

    if has_inf:
        logger.warning(f"{name} contains infinite values")

    return not has_inf

def is_valid_number(value):
    """Check if a value is a valid number (not NaN or infinite)."""
    try:
        return np.isfinite(float(value))
    except (ValueError, TypeError):
        return False

def math_safe(func: Callable, *args, default: Any = 0.0, **kwargs) -> Any:
    """Safely execute math function."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default

class MathValidation:
    """Math validation wrapper class for safe mathematical operations."""

    def __init__(self):
        """Initialize math validation."""
        pass

    def validate_finite(self, value: Any, name: str = "value") -> float:
        """Validate that a value is finite."""
        return validate_finite(value, name)

    def validate_positive(self, value: float, name: str = "value") -> float:
        """Validate that a value is positive."""
        return validate_positive(value, name)

    def validate_range(self, value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
        """Validate that a value is in range."""
        return validate_range(value, min_val, max_val, name)

    def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safely divide two numbers."""
        return safe_divide(a, b, default)

    def safe_log(self, x: float, default: float = 0.0) -> float:
        """Safely calculate logarithm."""
        return safe_log(x, default)

    def safe_sqrt(self, x: float, default: float = 0.0) -> float:
        """Safely calculate square root."""
        return safe_sqrt(x, default)

    def safe_power(self, x: float, y: float, default: float = 0.0) -> float:
        """Safely calculate power."""
        return safe_power(x, y, default)

class MathValidationError(Exception):
    """Exception raised for math validation errors."""
    pass
