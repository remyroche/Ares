"""Backward compatible shim for :mod:`temporal_leakage`."""

from .temporal_leakage import *  # noqa: F401,F403

__all__ = [
    "TemporalLintError",
    "TemporalLintViolation",
    "lint_for_temporal_leakage",
    "run_temporal_linting",
    "main",
]
