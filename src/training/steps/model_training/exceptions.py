"""Common exceptions for the model training package."""

from __future__ import annotations


class TrainingError(Exception):
    """Base exception for errors raised within the model training package."""


class ConfigurationError(TrainingError):
    """Raised when invalid configuration values are supplied to a training step."""


class MissingDependencyError(TrainingError, ImportError):
    """Raised when a required runtime dependency for training is not available."""


class DataQualityError(TrainingError):
    """Raised when input datasets fail validation or quality checks."""
