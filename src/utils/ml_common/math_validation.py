"""
ML Common - Math Validation Module

This module provides mathematical validation utilities for ML operations.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class MathValidationError(Exception):
    """Exception raised for math validation errors."""
    pass


class ValidationLevel(Enum):
    """Validation strictness levels."""
    LAX = "lax"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class MathValidator:
    """Mathematical validation utilities for ML operations."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level

    def validate_numeric_array(self, array: np.ndarray, name: str = "array") -> ValidationResult:
        """Validate a numeric array for mathematical operations."""
        errors = []
        warnings = []
        details = {}

        try:
            # Check if it's a numpy array
            if not isinstance(array, np.ndarray):
                array = np.array(array)

            # Check for NaN values
            nan_count = np.isnan(array).sum()
            if nan_count > 0:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Array '{name}' contains {nan_count} NaN values")
                else:
                    warnings.append(f"Array '{name}' contains {nan_count} NaN values")

            # Check for infinite values
            inf_count = np.isinf(array).sum()
            if inf_count > 0:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Array '{name}' contains {inf_count} infinite values")
                else:
                    warnings.append(f"Array '{name}' contains {inf_count} infinite values")

            # Check for very large values
            if array.size > 0:
                max_val = np.max(np.abs(array[np.isfinite(array)]))
                if max_val > 1e10:
                    warnings.append(f"Array '{name}' contains very large values (max: {max_val})")

            # Check for very small values (near zero)
            finite_array = array[np.isfinite(array)]
            if finite_array.size > 0:
                non_zero = finite_array[finite_array != 0]
                if non_zero.size > 0:
                    min_abs_val = np.min(np.abs(non_zero))
                    if min_abs_val < 1e-10:
                        warnings.append(f"Array '{name}' contains very small values (min abs: {min_abs_val})")

            details.update({
                'shape': array.shape,
                'dtype': str(array.dtype),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'finite_count': int(np.isfinite(array).sum()),
                'size': int(array.size)
            })

        except Exception as e:
            errors.append(f"Failed to validate array '{name}': {str(e)}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            details=details
        )

    def validate_matrix(self, matrix: np.ndarray, name: str = "matrix") -> ValidationResult:
        """Validate a matrix for mathematical operations."""
        errors = []
        warnings = []
        details = {}

        try:
            # First validate as array
            array_result = self.validate_numeric_array(matrix, name)
            errors.extend(array_result.errors)
            warnings.extend(array_result.warnings)

            # Additional matrix-specific checks
            if matrix.ndim != 2:
                errors.append(f"Matrix '{name}' must be 2-dimensional, got {matrix.ndim}D")
            elif matrix.shape[0] == 0 or matrix.shape[1] == 0:
                errors.append(f"Matrix '{name}' has zero size in one dimension: {matrix.shape}")

            # Check condition number for square matrices
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] > 0:
                try:
                    # Only compute condition number for well-conditioned matrices
                    if np.all(np.isfinite(matrix)) and np.linalg.det(matrix) != 0:
                        cond_num = np.linalg.cond(matrix)
                        if cond_num > 1e12:
                            warnings.append(f"Matrix '{name}' is very ill-conditioned (cond: {cond_num:.2e})")
                        details['condition_number'] = float(cond_num)
                except Exception:
                    # Skip condition number calculation if it fails
                    pass

            details.update(array_result.details)
            details.update({
                'is_matrix': matrix.ndim == 2,
                'matrix_shape': matrix.shape if matrix.ndim == 2 else None,
            })

        except Exception as e:
            errors.append(f"Failed to validate matrix '{name}': {str(e)}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            details=details
        )

    def safe_divide(self, numerator: Union[float, np.ndarray],
                   denominator: Union[float, np.ndarray],
                   default_value: float = 0.0) -> Union[float, np.ndarray]:
        """Safely divide two values, handling division by zero."""
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = numerator / denominator
                # Replace inf and nan with default value
                result = np.where(np.isfinite(result), result, default_value)
                return result
        except Exception:
            return default_value

    def safe_log(self, x: Union[float, np.ndarray],
                default_value: float = 0.0) -> Union[float, np.ndarray]:
        """Safely compute logarithm, handling negative and zero values."""
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.log(x)
                # Replace inf and nan with default value
                result = np.where(np.isfinite(result), result, default_value)
                return result
        except Exception:
            return default_value

    def validate_positive(self, values: np.ndarray, name: str = "values") -> ValidationResult:
        """Validate that all values are positive."""
        errors = []
        warnings = []
        details = {}

        try:
            negative_count = np.sum(values < 0)
            zero_count = np.sum(values == 0)

            if negative_count > 0:
                errors.append(f"'{name}' contains {negative_count} negative values")
            if zero_count > 0:
                warnings.append(f"'{name}' contains {zero_count} zero values")

            details.update({
                'total_count': len(values),
                'negative_count': int(negative_count),
                'zero_count': int(zero_count),
                'positive_count': int(np.sum(values > 0))
            })

        except Exception as e:
            errors.append(f"Failed to validate positive values for '{name}': {str(e)}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            details=details
        )


# Convenience functions
def safe_divide(numerator: Union[float, np.ndarray],
               denominator: Union[float, np.ndarray],
               default_value: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide two values."""
    validator = MathValidator()
    return validator.safe_divide(numerator, denominator, default_value)


def safe_log(x: Union[float, np.ndarray], default_value: float = 0.0) -> Union[float, np.ndarray]:
    """Safely compute logarithm."""
    validator = MathValidator()
    return validator.safe_log(x, default_value)


# Additional safe math functions
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


def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate_range(value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
    """Validate that a value is within a specified range."""
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    return value


__all__ = [
    'MathValidator',
    'MathValidationError',
    'ValidationLevel',
    'ValidationResult',
    'safe_divide',
    'safe_log',
    'safe_sqrt',
    'validate_positive',
    'validate_range'
]
