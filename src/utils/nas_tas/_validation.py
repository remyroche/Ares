"""Common validation helpers for NAS-TAS configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import MutableMapping, Sequence, Tuple, TypeVar

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigValidationError",
    "ValidationIssue",
    "ValidationReport",
    "ensure_between",
    "ensure_non_empty_string",
    "ensure_positive",
    "ensure_probability",
    "ensure_sequence_not_empty",
    "ensure_min_less_than_max",
    "normalize_weights",
]


class ConfigValidationError(ValueError):
    """Exception raised when configuration validation fails."""


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single validation issue discovered during checks."""

    field: str
    message: str
    level: str = "error"


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation output that can include warnings and errors."""

    issues: Tuple[ValidationIssue, ...]

    @property
    def has_errors(self) -> bool:
        """Return ``True`` when at least one error-level issue is present."""

        return any(issue.level == "error" for issue in self.issues)

    def log(self, *, prefix: str = "") -> None:
        """Emit all issues to the module logger for diagnostics."""

        for issue in self.issues:
            message = f"{prefix}{issue.field}: {issue.message}"
            if issue.level == "warning":
                logger.warning(message)
            else:
                logger.error(message)


T = TypeVar("T", bound=float)


def ensure_between(name: str, value: T, *, minimum: float, maximum: float) -> T:
    """Ensure ``value`` falls within ``[minimum, maximum]``."""

    if not (minimum <= value <= maximum):
        raise ConfigValidationError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def ensure_probability(name: str, value: T) -> T:
    """Ensure ``value`` is a valid probability (between 0 and 1 inclusive)."""

    return ensure_between(name, value, minimum=0.0, maximum=1.0)


def ensure_positive(name: str, value: T, *, allow_zero: bool = False) -> T:
    """Ensure ``value`` is strictly positive unless ``allow_zero`` is set."""

    if allow_zero and value == 0:
        return value
    if value > 0:
        return value
    comparator = ">=" if allow_zero else ">"
    raise ConfigValidationError(f"{name} must be {comparator} 0, got {value}")


def ensure_non_empty_string(name: str, value: str) -> str:
    """Ensure ``value`` is a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{name} must be a non-empty string")
    return value


def ensure_sequence_not_empty(name: str, value: Sequence[object]) -> Sequence[object]:
    """Ensure ``value`` is a non-empty sequence."""

    if not value:
        raise ConfigValidationError(f"{name} must contain at least one item")
    return value


def ensure_min_less_than_max(
    min_name: str,
    min_value: float,
    max_name: str,
    max_value: float,
) -> Tuple[float, float]:
    """Ensure ``min_value`` is strictly less than ``max_value``."""

    if min_value >= max_value:
        raise ConfigValidationError(
            f"{min_name} ({min_value}) must be less than {max_name} ({max_value})"
        )
    return min_value, max_value


def normalize_weights(
    weights: MutableMapping[object, float],
) -> MutableMapping[object, float]:
    """Normalize weight mappings so their values sum to one."""

    total = sum(weights.values())
    if total <= 0:
        raise ConfigValidationError("Weight totals must be positive to normalize")

    for key in list(weights.keys()):
        weights[key] = weights[key] / total
    return weights


def build_report(*issues: ValidationIssue) -> ValidationReport:
    """Create a :class:`ValidationReport` from ``issues`` and log details."""

    report = ValidationReport(issues=tuple(issues))
    report.log(prefix="Validation issue - ")
    return report


__all__.append("build_report")
