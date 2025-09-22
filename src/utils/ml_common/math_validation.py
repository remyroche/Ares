"""
ML Common - Math Validation Module (compatibility layer)

This module re-exports the canonical math validation utilities from
`src.utils.math_validation` to remove duplication while preserving the
existing ML Common import path. It also keeps a lightweight `MathValidator`
wrapper for code that expects an object-oriented API.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Canonical implementations (single source of truth)
from src.utils.math_validation import (
    safe_divide as _safe_divide_scalar,
    safe_log as _safe_log_scalar,
    safe_sqrt,
    validate_positive as _validate_positive_scalar,
    validate_range as _validate_range_scalar,
    validate_numeric_array as _validate_numeric_array,
    validate_finite as _validate_finite,
    MathValidationError,
)


class ValidationLevel(Enum):
    LAX = "lax"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class MathValidator:
    """Object-oriented facade delegating to canonical safe math helpers."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level

    def safe_divide(self, numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], default_value: float = 0.0) -> Union[float, np.ndarray]:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = numerator / denominator
                return np.where(np.isfinite(result), result, default_value)
        except Exception:
            return default_value

    def safe_log(self, x: Union[float, np.ndarray], default_value: float = 0.0) -> Union[float, np.ndarray]:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.log(x)
                return np.where(np.isfinite(result), result, default_value)
        except Exception:
            return default_value

    def validate_numeric_array(self, array: np.ndarray, name: str = "array") -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}
        try:
            array = _validate_numeric_array(array, name)
            nan_count = int(np.isnan(array).sum())
            inf_count = int(np.isinf(array).sum())
            if nan_count > 0:
                (errors if self.validation_level == ValidationLevel.STRICT else warnings).append(
                    f"Array '{name}' contains {nan_count} NaN values"
                )
            if inf_count > 0:
                (errors if self.validation_level == ValidationLevel.STRICT else warnings).append(
                    f"Array '{name}' contains {inf_count} infinite values"
                )
            details.update({'shape': array.shape, 'nan_count': nan_count, 'inf_count': inf_count})
        except Exception as e:
            errors.append(str(e))
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings, details=details)

    def validate_positive(self, values: np.ndarray, name: str = "values") -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}
        try:
            negative_count = int(np.sum(values < 0))
            zero_count = int(np.sum(values == 0))
            if negative_count > 0:
                errors.append(f"'{name}' contains {negative_count} negative values")
            if zero_count > 0:
                warnings.append(f"'{name}' contains {zero_count} zero values")
            details.update({'total_count': int(values.size), 'negative_count': negative_count, 'zero_count': zero_count})
        except Exception as e:
            errors.append(str(e))
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings, details=details)


# Re-export scalar helpers for compatibility
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return _safe_divide_scalar(a, b, default)


def safe_log(x: float, default: float = 0.0) -> float:
    return _safe_log_scalar(x, default)


def validate_positive(value: float, name: str = "value") -> float:
    return _validate_positive_scalar(value, name)


def validate_range(value: float, min_val: float = None, max_val: float = None, name: str = "value") -> float:
    return _validate_range_scalar(value, min_val, max_val, name)


def validate_numeric_array(array: np.ndarray, name: str = "array") -> np.ndarray:
    return _validate_numeric_array(array, name)


def validate_finite(value: float, name: str = "value") -> float:
    return _validate_finite(value, name)


__all__ = [
    'MathValidator',
    'MathValidationError',
    'ValidationLevel',
    'ValidationResult',
    'safe_divide',
    'safe_log',
    'safe_sqrt',
    'validate_positive',
    'validate_range',
    'validate_numeric_array',
    'validate_finite',
]
