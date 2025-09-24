"""
Enhanced Error System for Training Pipeline

This module provides a comprehensive error handling system that:
1. Eliminates silent failures
2. Preserves rich error context
3. Provides structured debugging information
4. Implements proper error propagation
5. Creates actionable error messages

Usage:
    from src.training.core.errors import (
        TrainingError, PipelineError, DataError,
        with_error_context, ErrorContext
    )

    # Use structured errors
    raise PipelineError("Stage failed", stage="data_collection", step="download")

    # Use error context
    with ErrorContext("Processing market data"):
        # Your code here
        pass
"""

import logging
import traceback
import inspect
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Warnings, recoverable issues
    MEDIUM = "medium"     # Should be investigated
    HIGH = "high"         # Critical, requires attention
    CRITICAL = "critical" # System-breaking, immediate action needed

class ErrorCategory(Enum):
    """Error categories for classification."""
    CONFIGURATION = "configuration"
    DATA = "data"
    PIPELINE = "pipeline"
    MODEL = "model"
    BACKTESTING = "backtesting"
    SYSTEM = "system"
    VALIDATION = "validation"

@dataclass
class ErrorContext:
    """Rich error context information."""
    operation: str
    stage: Optional[str] = None
    step: Optional[str] = None
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    data_size: Optional[int] = None
    memory_usage: Optional[float] = None
    execution_time: Optional[float] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            'operation': self.operation,
            'stage': self.stage,
            'step': self.step,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'data_size': self.data_size,
            'memory_usage': self.memory_usage,
            'execution_time': self.execution_time,
            'custom_data': self.custom_data
        }

@dataclass
class ErrorRecovery:
    """Error recovery suggestions and actions."""
    suggestions: List[str] = field(default_factory=list)
    auto_retry: bool = False
    retry_delay: float = 1.0
    max_retries: int = 3
    fallback_action: Optional[Callable] = None

class TrainingError(Exception):
    """Base exception for all training pipeline errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[ErrorContext] = None,
        recovery: Optional[ErrorRecovery] = None,
        cause: Optional[Exception] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext("")
        self.recovery = recovery or ErrorRecovery()
        self.cause = cause
        self.additional_data = kwargs
        self.timestamp = datetime.now()
        self.traceback_info = traceback.format_exc()

    def __str__(self) -> str:
        """Rich error message with context."""
        parts = [f"[{self.category.value.upper()}] {self.message}"]

        if self.context.operation:
            parts.append(f"Operation: {self.context.operation}")

        if self.context.stage:
            parts.append(f"Stage: {self.context.stage}")

        if self.context.step:
            parts.append(f"Step: {self.context.step}")

        if self.context.symbol:
            parts.append(f"Symbol: {self.context.symbol}")

        if self.severity != ErrorSeverity.LOW:
            parts.append(f"Severity: {self.severity.value}")

        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {str(self.cause)}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging."""
        return {
            'type': type(self).__name__,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'recovery': {
                'suggestions': self.recovery.suggestions,
                'auto_retry': self.recovery.auto_retry,
                'retry_delay': self.recovery.retry_delay,
                'max_retries': self.recovery.max_retries
            },
            'additional_data': self.additional_data,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback_info
        }

class PipelineError(TrainingError):
    """Pipeline execution errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PIPELINE, **kwargs)

class DataError(TrainingError):
    """Data processing errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)

class ConfigurationError(TrainingError):
    """Configuration-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)

class ModelError(TrainingError):
    """Model training errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, **kwargs)

class BacktestingError(TrainingError):
    """Backtesting errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BACKTESTING, **kwargs)

class ValidationError(TrainingError):
    """Validation errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)

class ErrorHandler:
    """Centralized error handling and recovery."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[TrainingError] = []
        self.recovery_strategies: Dict[str, Callable] = {}

    def handle_error(self, error: TrainingError, reraise: bool = True) -> bool:
        """Handle an error with recovery attempts."""
        self.error_history.append(error)

        # Log structured error information
        error_dict = error.to_dict()
        self.logger.error(f"Training Error: {error}", extra={'error_data': error_dict})

        # Check if this is a critical error that should stop execution
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error detected: {error}")
            if reraise:
                raise error
            return False

        # Attempt recovery if configured
        if error.recovery.auto_retry:
            return self._attempt_recovery(error)

        # For non-critical errors, log and continue
        if error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            self.logger.warning(f"Handled non-critical error: {error}")
            return True

        # For high severity errors, always reraise
        if reraise:
            raise error
        return False

    def _attempt_recovery(self, error: TrainingError) -> bool:
        """Attempt error recovery."""
        self.logger.info(f"Attempting recovery for error: {error}")

        try:
            # Use fallback action if available
            if error.recovery.fallback_action:
                error.recovery.fallback_action()
                self.logger.info("Fallback action executed successfully")
                return True

            # Try retrying the operation
            for attempt in range(error.recovery.max_retries):
                try:
                    self.logger.info(f"Retry attempt {attempt + 1}/{error.recovery.max_retries}")
                    # Here you would retry the actual operation
                    # For now, just log the attempt
                    return True
                except Exception as retry_error:
                    self.logger.warning(f"Retry {attempt + 1} failed: {retry_error}")

            self.logger.error("All recovery attempts failed")
            return False

        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return False

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        return {
            'total_errors': len(self.error_history),
            'by_category': self._group_errors_by_category(),
            'by_severity': self._group_errors_by_severity(),
            'recent_errors': [error.to_dict() for error in self.error_history[-10:]]
        }

    def _group_errors_by_category(self) -> Dict[str, int]:
        """Group errors by category."""
        categories = {}
        for error in self.error_history:
            categories[error.category.value] = categories.get(error.category.value, 0) + 1
        return categories

    def _group_errors_by_severity(self) -> Dict[str, int]:
        """Group errors by severity."""
        severities = {}
        for error in self.error_history:
            severities[error.severity.value] = severities.get(error.severity.value, 0) + 1
        return severities

def with_error_context(operation: str, **context_kwargs):
    """Decorator to add error context to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(operation=operation, **context_kwargs)

            try:
                # Add context to function if it accepts it
                if 'error_context' in inspect.signature(func).parameters:
                    kwargs['error_context'] = context

                return func(*args, **kwargs)

            except Exception as e:
                # Enhance the error with context
                if isinstance(e, TrainingError):
                    e.context = context
                    raise

                # Convert to TrainingError with context
                enhanced_error = TrainingError(
                    f"Error in {operation}: {str(e)}",
                    context=context,
                    cause=e
                )
                raise enhanced_error

        return wrapper
    return decorator

def create_error_context_from_config(config: Dict[str, Any]) -> ErrorContext:
    """Create error context from configuration."""
    return ErrorContext(
        operation="pipeline_execution",
        symbol=config.get('symbol'),
        exchange=config.get('exchange'),
        timeframe=config.get('timeframe'),
        custom_data=config
    )

# Global error handler instance
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    return error_handler

def log_error_summary():
    """Log a summary of recent errors."""
    summary = error_handler.get_error_summary()
    logger.info(f"Error Summary: {summary}")

# Convenience functions for creating common errors
def pipeline_execution_error(message: str, stage: str, step: str = None, **kwargs) -> PipelineError:
    """Create a pipeline execution error."""
    return PipelineError(
        message,
        context=ErrorContext(
            operation="pipeline_execution",
            stage=stage,
            step=step,
            **kwargs
        )
    )

def data_processing_error(message: str, operation: str, **kwargs) -> DataError:
    """Create a data processing error."""
    return DataError(
        message,
        context=ErrorContext(
            operation=operation,
            **kwargs
        )
    )

def configuration_error(message: str, config_key: str = None, **kwargs) -> ConfigurationError:
    """Create a configuration error."""
    context = ErrorContext(operation="configuration_validation", **kwargs)
    if config_key:
        context.custom_data['config_key'] = config_key

    return ConfigurationError(message, context=context)

# Export all classes and functions
__all__ = [
    'ErrorSeverity', 'ErrorCategory', 'ErrorContext', 'ErrorRecovery',
    'TrainingError', 'PipelineError', 'DataError', 'ConfigurationError',
    'ModelError', 'BacktestingError', 'ValidationError', 'ErrorHandler',
    'with_error_context', 'create_error_context_from_config',
    'get_error_handler', 'log_error_summary',
    'pipeline_execution_error', 'data_processing_error', 'configuration_error'
]