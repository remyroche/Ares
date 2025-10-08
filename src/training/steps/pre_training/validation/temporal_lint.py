"""Backward compatible shim for :mod:`temporal_leakage`.

This module provides explicit re-exports for backward compatibility.
Consider importing directly from .temporal_leakage instead.
"""

# Explicit re-exports instead of star imports for better maintainability
from .temporal_leakage import (
    TemporalLintError,
    TemporalLintViolation,
    lint_for_temporal_leakage,
    main,
    run_temporal_linting,
)

__all__ = [
    "TemporalLintError",
    "TemporalLintViolation",
    "lint_for_temporal_leakage",
    "main",
    "run_temporal_linting",
]
