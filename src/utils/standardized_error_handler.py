"""
Standardized Error Handler

This module provides unified error handling patterns across all steps including:
- Centralized error categorization
- Error recovery strategies
- Error reporting and logging
- Error context tracking
"""

import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

from .pipeline_standards import PipelineStandards, pipeline_standards
from .logger import system_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_QUALITY = "data_quality"
    MODEL_TRAINING = "model_training"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class ErrorContext:
    """Error context information."""

    def __init__(self, step_name: str, operation: str, **kwargs):
        self.step_name = step_name
        self.operation = operation
        self.timestamp = datetime.now().isoformat()
        self.data_context = kwargs.get('data_context', {})
        self.config_context = kwargs.get('config_context', {})
        self.user_context = kwargs.get('user_context', {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'step_name': self.step_name,
            'operation': self.operation,
            'timestamp': self.timestamp,
            'data_context': self.data_context,
            'config_context': self.config_context,
            'user_context': self.user_context
        }


class ErrorRecord:
    """Error record with full context."""

    def __init__(self, error: Exception, context: ErrorContext,
                 severity: ErrorSeverity = ErrorSeverity.ERROR):
        self.error = error
        self.context = context
        self.severity = severity
        self.category = self._categorize_error(error)
        self.traceback = traceback.format_exc()
        self.recovery_strategy = self._get_recovery_strategy()

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize the error based on its type and message."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Data quality errors
        if any(keyword in error_message for keyword in ['data', 'dataframe', 'nan', 'null', 'missing']):
            return ErrorCategory.DATA_QUALITY

        # Model training errors
        if any(keyword in error_message for keyword in ['model', 'training', 'fit', 'predict', 'loss']):
            return ErrorCategory.MODEL_TRAINING

        # Configuration errors
        if any(keyword in error_message for keyword in ['config', 'parameter', 'setting', 'option']):
            return ErrorCategory.CONFIGURATION

        # Dependency errors
        if any(keyword in error_message for keyword in ['import', 'module', 'package', 'dependency']):
            return ErrorCategory.DEPENDENCY

        # Resource errors
        if any(keyword in error_message for keyword in ['memory', 'disk', 'cpu', 'gpu', 'resource']):
            return ErrorCategory.RESOURCE

        # Network errors
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK

        # Validation errors
        if any(keyword in error_message for keyword in ['validation', 'schema', 'format', 'type']):
            return ErrorCategory.VALIDATION

        return ErrorCategory.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert error record to dictionary."""
        return {
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.to_dict(),
            'traceback': self.traceback,
            'recovery_strategy': self.recovery_strategy
        }


class StandardizedErrorHandler:
    """Centralized error handling system."""

    def __init__(self):
        """Initialize the error handler."""
        self.standards = pipeline_standards
        self.logger = system_logger
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = 1000

    def handle_step_error(
        self,
        error: Exception,
        step_name: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR
    ) -> ErrorRecord:
        """Handle an error in a pipeline step.

        Args:
            error: The exception that occurred
            step_name: Name of the step where error occurred
            context: Additional context information
            severity: Error severity level

        Returns:
            ErrorRecord: Record of the error with context
        """
        # Create error context
        error_context = ErrorContext(
            step_name=step_name,
            operation=context.get('operation', 'unknown') if context else 'unknown',
            data_context=context.get('data_context', {}) if context else {},
            config_context=context.get('config_context', {}) if context else {},
            user_context=context.get('user_context', {}) if context else {}
        )

        # Create error record
        error_record = ErrorRecord(error, error_context, severity)

        # Log the error
        self._log_error_with_context(error_record)

        # Add to history
        self._add_to_history(error_record)

        return error_record

    def _log_error_with_context(self, error_record: ErrorRecord) -> None:
        """Log error with full context."""
        log_message = f"""
Error in {error_record.context.step_name}:
  Type: {type(error_record.error).__name__}
  Message: {str(error_record.error)}
  Category: {error_record.category.value}
  Severity: {error_record.severity.value}
  Operation: {error_record.context.operation}
  Recovery: {error_record.recovery_strategy['description']}
"""

        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _add_to_history(self, error_record: ErrorRecord) -> None:
        """Add error record to history."""
        self.error_history.append(error_record)

        # Maintain history size
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)


# Global instance
standardized_error_handler = StandardizedErrorHandler()