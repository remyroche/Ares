"""Fallback-aware access to :mod:`src.utils.math_validation`."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .dependency_management import dependency_manager
from ..fallback_utilities import FallbackConfig, FallbackMathUtils

__all__ = [
    "MATH_VALIDATION_AVAILABLE",
    "MathValidation",
    "MathValidationError",
    "math_safe",
    "safe_correlation",
    "safe_covariance",
    "safe_divide",
    "safe_kelly_calculation",
    "safe_log",
    "safe_matrix_inverse",
    "safe_mean",
    "safe_percentage_change",
    "safe_percentile",
    "safe_power",
    "safe_sqrt",
    "safe_std",
    "safe_weighted_average",
    "validate_correlation_matrix",
    "validate_finite",
    "validate_numeric_array",
    "validate_positive",
    "validate_range",
]

logger = logging.getLogger(__name__)

_MATH_MODULE = dependency_manager.import_optional(
    "src.utils.math_validation",
    install_hint="pip install nas-tas-commons extras or include src/utils/math_validation.py",
)
MATH_VALIDATION_AVAILABLE = _MATH_MODULE is not None

if MATH_VALIDATION_AVAILABLE:
    # Expose concrete implementations directly when available.
    safe_divide = getattr(_MATH_MODULE, "safe_divide")
    safe_log = getattr(_MATH_MODULE, "safe_log")
    safe_sqrt = getattr(_MATH_MODULE, "safe_sqrt")
    safe_power = getattr(_MATH_MODULE, "safe_power")
    safe_mean = getattr(_MATH_MODULE, "safe_mean")
    safe_std = getattr(_MATH_MODULE, "safe_std")
    safe_correlation = getattr(_MATH_MODULE, "safe_correlation")
    safe_covariance = getattr(_MATH_MODULE, "safe_covariance")
    safe_percentile = getattr(_MATH_MODULE, "safe_percentile")
    safe_weighted_average = getattr(_MATH_MODULE, "safe_weighted_average")
    safe_percentage_change = getattr(_MATH_MODULE, "safe_percentage_change")
    safe_matrix_inverse = getattr(_MATH_MODULE, "safe_matrix_inverse")
    safe_kelly_calculation = getattr(_MATH_MODULE, "safe_kelly_calculation")
    math_safe = getattr(_MATH_MODULE, "math_safe")
    validate_numeric_array = getattr(_MATH_MODULE, "validate_numeric_array")
    validate_positive = getattr(_MATH_MODULE, "validate_positive")
    validate_range = getattr(_MATH_MODULE, "validate_range")
    validate_finite = getattr(_MATH_MODULE, "validate_finite")
    validate_correlation_matrix = getattr(_MATH_MODULE, "validate_correlation_matrix")
    MathValidation = getattr(_MATH_MODULE, "MathValidation")
    MathValidationError = getattr(_MATH_MODULE, "MathValidationError")
else:
    _math = FallbackMathUtils(FallbackConfig(enable_logging=True))
    _logged_fallbacks: set[str] = set()

    def _log_fallback(name: str) -> None:
        if name in _logged_fallbacks:
            return
        logger.warning(
            "Using fallback implementation for src.utils.math_validation.%s because the optional module is unavailable.",
            name,
        )
        _logged_fallbacks.add(name)

    def _wrap(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
        def _inner(*args: Any, **kwargs: Any) -> Any:
            _log_fallback(name)
            return func(*args, **kwargs)

        return _inner

    safe_divide = _wrap("safe_divide", _math.safe_divide)
    safe_log = _wrap("safe_log", _math.safe_log)
    safe_sqrt = _wrap("safe_sqrt", _math.safe_sqrt)
    safe_power = _wrap("safe_power", _math.safe_power)
    safe_mean = _wrap("safe_mean", _math.safe_mean)
    safe_std = _wrap("safe_std", _math.safe_std)
    safe_correlation = _wrap("safe_correlation", _math.safe_correlation)
    safe_covariance = _wrap("safe_covariance", _math.safe_covariance)
    safe_percentile = _wrap("safe_percentile", _math.safe_percentile)
    safe_weighted_average = _wrap("safe_weighted_average", _math.safe_weighted_average)
    safe_percentage_change = _wrap("safe_percentage_change", _math.safe_percentage_change)
    safe_matrix_inverse = _wrap("safe_matrix_inverse", _math.safe_matrix_inverse)
    safe_kelly_calculation = _wrap("safe_kelly_calculation", _math.safe_kelly_calculation)

    def math_safe(func: Callable[..., Any], *args: Any, default: float = 0.0, **kwargs: Any) -> float:
        _log_fallback("math_safe")
        try:
            return float(func(*args, **kwargs))
        except Exception:  # noqa: BLE001
            logger.warning("math_safe fallback returning default value", exc_info=True)
            return default

    def validate_numeric_array(array: Any, name: str = "array") -> np.ndarray:
        _log_fallback("validate_numeric_array")
        arr = np.asarray(array)
        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"{name} must contain numeric data, got dtype {arr.dtype}")
        mask = np.isfinite(arr)
        if not np.all(mask):
            raise ValueError(f"{name} contains {np.sum(~mask)} non-finite values")
        return arr

    def validate_positive(value: float, name: str = "value") -> float:
        _log_fallback("validate_positive")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def validate_range(
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value",
    ) -> float:
        _log_fallback("validate_range")
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        return value

    def validate_finite(value: Any, name: str = "value") -> float:
        _log_fallback("validate_finite")
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name}: {exc}") from exc
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val}")
        return val

    def validate_correlation_matrix(matrix: Any, tol: float = 1e-8) -> bool:
        _log_fallback("validate_correlation_matrix")
        try:
            arr = np.asarray(matrix, dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to coerce matrix to numpy array: %s", exc)
            return False
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            logger.error("Correlation matrix must be square")
            return False
        if np.any(np.isnan(arr)):
            logger.error("Correlation matrix contains NaN entries")
            return False
        if not np.allclose(arr, arr.T, atol=tol):
            logger.error("Correlation matrix must be symmetric")
            return False
        if np.any(np.abs(arr) > 1 + tol):
            logger.error("Correlation matrix entries must be within [-1, 1]")
            return False
        return True

    class MathValidationError(Exception):
        """Fallback math validation error."""

    class MathValidation:
        """Minimal fallback validator using :class:`FallbackMathUtils`."""

        def __init__(self) -> None:
            self._math = _math

        def validate(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise MathValidationError(f"Validation function failed: {exc}") from exc
            return float(result)

        def ensure_numeric(self, array: Iterable[float], name: str = "array") -> np.ndarray:
            return validate_numeric_array(array, name=name)

        def ensure_positive(self, value: float, name: str = "value") -> float:
            return validate_positive(value, name=name)

        def ensure_finite(self, value: float, name: str = "value") -> float:
            return validate_finite(value, name=name)

        def ensure_range(
            self,
            value: float,
            min_val: Optional[float] = None,
            max_val: Optional[float] = None,
            name: str = "value",
        ) -> float:
            return validate_range(value, min_val=min_val, max_val=max_val, name=name)

