"""
Custom exceptions for ML optimization utilities.

This module provides a comprehensive exception hierarchy for the optimization
system, enabling better error handling and debugging.
"""

from typing import Optional, Any, Dict


class OptimizationError(Exception):
    """Base exception for all optimization-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ConfigurationError(OptimizationError):
    """Raised when configuration is invalid or missing required parameters."""
    pass


class ModelEvaluationError(OptimizationError):
    """Raised when model evaluation fails."""
    pass


class HardwareOptimizationError(OptimizationError):
    """Raised when hardware optimization fails."""
    pass


class PruningError(OptimizationError):
    """Raised when trial pruning fails."""
    pass


class SearchSpaceError(OptimizationError):
    """Raised when search space definition is invalid."""
    pass


class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge."""
    pass


class TimeoutError(OptimizationError):
    """Raised when optimization times out."""
    pass


class ValidationError(OptimizationError):
    """Raised when data validation fails."""
    pass


class CacheError(OptimizationError):
    """Raised when caching operations fail."""
    pass


class MonitoringError(OptimizationError):
    """Raised when monitoring operations fail."""
    pass


class VectorBTError(OptimizationError):
    """Raised when VectorBT operations fail."""
    pass


class AresModeError(OptimizationError):
    """Raised when Ares execution mode operations fail."""
    pass


# Convenience functions for common error patterns
def raise_config_error(message: str, param_name: str = None, value: Any = None) -> None:
    """Raise a configuration error with context."""
    context = {}
    if param_name:
        context['parameter'] = param_name
    if value is not None:
        context['value'] = value
    raise ConfigurationError(message, context)


def raise_model_evaluation_error(message: str, model_type: str = None, trial_number: int = None) -> None:
    """Raise a model evaluation error with context."""
    context = {}
    if model_type:
        context['model_type'] = model_type
    if trial_number is not None:
        context['trial_number'] = trial_number
    raise ModelEvaluationError(message, context)


def raise_hardware_error(message: str, component: str = None, operation: str = None) -> None:
    """Raise a hardware optimization error with context."""
    context = {}
    if component:
        context['component'] = component
    if operation:
        context['operation'] = operation
    raise HardwareOptimizationError(message, context)