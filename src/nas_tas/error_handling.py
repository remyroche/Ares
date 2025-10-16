"""
Unified Error Handling for NAS/TAS Systems

This module provides consistent error handling and recovery mechanisms
across both NAS and TAS implementations.
"""

import logging
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import functools
import asyncio

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    DATA_ERROR = "data_error"
    CONFIG_ERROR = "config_error"
    TRAINING_ERROR = "training_error"
    EVALUATION_ERROR = "evaluation_error"
    SYSTEM_ERROR = "system_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    FILE_ERROR = "file_error"
    VALIDATION_ERROR = "validation_error"

@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    action_type: str  # retry, fallback, skip, abort, custom
    action_params: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    delay_seconds: float = 1.0
    custom_handler: Optional[Callable] = None

@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str = ""
    component: str = ""
    data_shape: Optional[tuple] = None
    config_hash: str = ""
    execution_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = field(default_factory=dict)

class UnifiedError(Exception):
    """Unified error class for NAS/TAS systems."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': {
                'operation': self.context.operation,
                'component': self.context.component,
                'data_shape': self.context.data_shape,
                'config_hash': self.context.config_hash,
                'execution_id': self.context.execution_id,
                'timestamp': self.context.timestamp.isoformat(),
                'additional_info': self.context.additional_info
            },
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None,
            'traceback': self.traceback
        }

class UnifiedErrorHandler:
    """
    Unified error handler for NAS/TAS systems.

    This class consolidates error handling logic that was previously
    scattered across NAS and TAS implementations, providing consistent
    error management and recovery capabilities.
    """

    def __init__(self, enable_recovery: bool = True, log_errors: bool = True):
        """
        Initialize unified error handler.

        Args:
            enable_recovery: Whether to enable automatic error recovery
            log_errors: Whether to log errors
        """
        self.enable_recovery = enable_recovery
        self.log_errors = log_errors
        self.logger = logging.getLogger(self.__class__.__name__)

        # Error tracking
        self.error_history: List[UnifiedError] = []
        self.recovery_attempts: Dict[str, int] = {}

        # Recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, RecoveryAction] = {
            ErrorCategory.DATA_ERROR: RecoveryAction("retry", {"max_attempts": 2}),
            ErrorCategory.CONFIG_ERROR: RecoveryAction("fallback", {"use_default": True}),
            ErrorCategory.TRAINING_ERROR: RecoveryAction("retry", {"max_attempts": 1}),
            ErrorCategory.EVALUATION_ERROR: RecoveryAction("skip", {}),
            ErrorCategory.MEMORY_ERROR: RecoveryAction("fallback", {"reduce_batch_size": True}),
            ErrorCategory.SYSTEM_ERROR: RecoveryAction("retry", {"max_attempts": 3}),
            ErrorCategory.FILE_ERROR: RecoveryAction("retry", {"max_attempts": 2}),
            ErrorCategory.VALIDATION_ERROR: RecoveryAction("fallback", {"use_relaxed_validation": True})
        }

        # Error callbacks
        self.error_callbacks: List[Callable[[UnifiedError], None]] = []

        tprint_info("Unified error handler initialized")

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> RecoveryAction:
        """
        Handle an error and determine recovery action.

        Args:
            error: The exception that occurred
            context: Error context information
            category: Error category (auto-detected if None)
            severity: Error severity (auto-detected if None)

        Returns:
            RecoveryAction to take
        """
        # Auto-detect category and severity if not provided
        if category is None:
            category = self._detect_error_category(error)

        if severity is None:
            severity = self._detect_error_severity(error, context)

        # Create unified error
        unified_error = UnifiedError(
            message=str(error),
            category=category,
            severity=severity,
            context=context,
            original_error=error
        )

        # Log error
        if self.log_errors:
            self._log_error(unified_error)

        # Store error
        self.error_history.append(unified_error)

        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(unified_error)
            except Exception as e:
                tprint_error(f"Error in error callback: {e}")

        # Determine recovery action
        recovery_action = self._determine_recovery_action(unified_error)

        tprint_warning(f"Error handled: {category.value} ({severity.value}) - Action: {recovery_action.action_type}")

        return recovery_action

    def handle_training_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> RecoveryAction:
        """
        Handle training-specific errors.

        Args:
            error: Training error
            context: Training context

        Returns:
            RecoveryAction for training error
        """
        error_context = ErrorContext(
            operation="training",
            component=context.get("component", "training"),
            data_shape=context.get("data_shape"),
            config_hash=context.get("config_hash", ""),
            execution_id=context.get("execution_id", ""),
            additional_info=context
        )

        return self.handle_error(error, error_context, ErrorCategory.TRAINING_ERROR)

    def handle_evaluation_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> RecoveryAction:
        """
        Handle evaluation-specific errors.

        Args:
            error: Evaluation error
            context: Evaluation context

        Returns:
            RecoveryAction for evaluation error
        """
        error_context = ErrorContext(
            operation="evaluation",
            component=context.get("component", "evaluation"),
            data_shape=context.get("data_shape"),
            config_hash=context.get("config_hash", ""),
            execution_id=context.get("execution_id", ""),
            additional_info=context
        )

        return self.handle_error(error, error_context, ErrorCategory.EVALUATION_ERROR)

    def handle_data_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> RecoveryAction:
        """
        Handle data-specific errors.

        Args:
            error: Data error
            context: Data context

        Returns:
            RecoveryAction for data error
        """
        error_context = ErrorContext(
            operation="data_processing",
            component=context.get("component", "data"),
            data_shape=context.get("data_shape"),
            config_hash=context.get("config_hash", ""),
            execution_id=context.get("execution_id", ""),
            additional_info=context
        )

        return self.handle_error(error, error_context, ErrorCategory.DATA_ERROR)

    def execute_recovery_action(
        self,
        action: RecoveryAction,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute recovery action.

        Args:
            action: Recovery action to execute
            operation: Operation to retry or modify
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Result of operation execution
        """
        if not self.enable_recovery:
            raise RuntimeError("Recovery is disabled")

        if action.action_type == "retry":
            return self._retry_operation(action, operation, *args, **kwargs)
        elif action.action_type == "fallback":
            return self._fallback_operation(action, operation, *args, **kwargs)
        elif action.action_type == "skip":
            tprint_info("Skipping operation due to error recovery")
            return None
        elif action.action_type == "abort":
            raise RuntimeError("Operation aborted due to error recovery")
        elif action.action_type == "custom" and action.custom_handler:
            return action.custom_handler(operation, *args, **kwargs)
        else:
            raise ValueError(f"Unknown recovery action: {action.action_type}")

    async def execute_recovery_action_async(
        self,
        action: RecoveryAction,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute recovery action asynchronously.

        Args:
            action: Recovery action to execute
            operation: Async operation to retry or modify
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Result of operation execution
        """
        if not self.enable_recovery:
            raise RuntimeError("Recovery is disabled")

        if action.action_type == "retry":
            return await self._retry_operation_async(action, operation, *args, **kwargs)
        elif action.action_type == "fallback":
            return await self._fallback_operation_async(action, operation, *args, **kwargs)
        elif action.action_type == "skip":
            tprint_info("Skipping operation due to error recovery")
            return None
        elif action.action_type == "abort":
            raise RuntimeError("Operation aborted due to error recovery")
        elif action.action_type == "custom" and action.custom_handler:
            return await action.custom_handler(operation, *args, **kwargs)
        else:
            raise ValueError(f"Unknown recovery action: {action.action_type}")

    def add_error_callback(self, callback: Callable[[UnifiedError], None]):
        """Add error callback."""
        self.error_callbacks.append(callback)

    def set_recovery_strategy(
        self,
        category: ErrorCategory,
        action: RecoveryAction
    ):
        """Set recovery strategy for error category."""
        self.recovery_strategies[category] = action

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {}

        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        return {
            'total_errors': len(self.error_history),
            'category_counts': category_counts,
            'severity_counts': severity_counts,
            'recovery_attempts': dict(self.recovery_attempts),
            'latest_error': self.error_history[-1].to_dict() if self.error_history else None
        }

    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.recovery_attempts.clear()

    def _detect_error_category(self, error: Exception) -> ErrorCategory:
        """Auto-detect error category."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        if any(keyword in error_message for keyword in ['data', 'dataset', 'file', 'csv', 'parquet']):
            return ErrorCategory.DATA_ERROR
        elif any(keyword in error_message for keyword in ['config', 'configuration', 'parameter']):
            return ErrorCategory.CONFIG_ERROR
        elif any(keyword in error_message for keyword in ['training', 'train', 'model', 'fit']):
            return ErrorCategory.TRAINING_ERROR
        elif any(keyword in error_message for keyword in ['evaluation', 'evaluate', 'score', 'metric']):
            return ErrorCategory.EVALUATION_ERROR
        elif any(keyword in error_message for keyword in ['memory', 'ram', 'out of memory']):
            return ErrorCategory.MEMORY_ERROR
        elif any(keyword in error_message for keyword in ['network', 'connection', 'timeout']):
            return ErrorCategory.NETWORK_ERROR
        elif any(keyword in error_message for keyword in ['file', 'directory', 'path', 'permission']):
            return ErrorCategory.FILE_ERROR
        elif any(keyword in error_message for keyword in ['validation', 'validate', 'check']):
            return ErrorCategory.VALIDATION_ERROR
        else:
            return ErrorCategory.SYSTEM_ERROR

    def _detect_error_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Auto-detect error severity."""
        error_message = str(error).lower()

        # Critical errors
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'abort', 'terminate']):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if any(keyword in error_message for keyword in ['memory', 'disk space', 'permission denied']):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if any(keyword in error_message for keyword in ['timeout', 'connection', 'network']):
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _determine_recovery_action(self, error: UnifiedError) -> RecoveryAction:
        """Determine recovery action for error."""
        # Check if we've exceeded max attempts for this error
        error_key = f"{error.category.value}_{error.context.operation}"
        attempts = self.recovery_attempts.get(error_key, 0)

        if attempts >= 3:  # Max attempts reached
            return RecoveryAction("abort")

        # Get strategy for error category
        strategy = self.recovery_strategies.get(error.category, RecoveryAction("abort"))

        # Update attempt count
        self.recovery_attempts[error_key] = attempts + 1

        return strategy

    def _log_error(self, error: UnifiedError):
        """Log error with appropriate level."""
        log_message = f"Error in {error.context.operation}: {error.message}"

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _retry_operation(
        self,
        action: RecoveryAction,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry operation with specified parameters."""
        for attempt in range(action.max_attempts):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == action.max_attempts - 1:
                    raise e

                tprint_info(f"Retry attempt {attempt + 1}/{action.max_attempts} failed, retrying...")
                if action.delay_seconds > 0:
                    import time
                    time.sleep(action.delay_seconds)

        raise RuntimeError("Max retry attempts exceeded")

    async def _retry_operation_async(
        self,
        action: RecoveryAction,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry async operation with specified parameters."""
        for attempt in range(action.max_attempts):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                if attempt == action.max_attempts - 1:
                    raise e

                tprint_info(f"Retry attempt {attempt + 1}/{action.max_attempts} failed, retrying...")
                if action.delay_seconds > 0:
                    await asyncio.sleep(action.delay_seconds)

        raise RuntimeError("Max retry attempts exceeded")

    def _fallback_operation(
        self,
        action: RecoveryAction,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback operation."""
        tprint_info(f"Executing fallback operation: {action.action_params}")

        # Modify kwargs based on fallback parameters
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.update(action.action_params)

        return operation(*args, **fallback_kwargs)

    async def _fallback_operation_async(
        self,
        action: RecoveryAction,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute async fallback operation."""
        tprint_info(f"Executing fallback operation: {action.action_params}")

        # Modify kwargs based on fallback parameters
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.update(action.action_params)

        return await operation(*args, **fallback_kwargs)

def error_handler_decorator(
    category: Optional[ErrorCategory] = None,
    severity: Optional[ErrorSeverity] = None,
    recovery_enabled: bool = True
):
    """
    Decorator for automatic error handling.

    Args:
        category: Error category (auto-detected if None)
        severity: Error severity (auto-detected if None)
        recovery_enabled: Whether to enable recovery
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get error handler from instance or create new one
            error_handler = None
            for arg in args:
                if hasattr(arg, 'error_handler'):
                    error_handler = arg.error_handler
                    break

            if error_handler is None:
                error_handler = UnifiedErrorHandler(enable_recovery=recovery_enabled)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=func.__name__,
                    component=func.__module__,
                    additional_info={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                )

                recovery_action = error_handler.handle_error(e, context, category, severity)

                if recovery_enabled and recovery_action.action_type != "abort":
                    return error_handler.execute_recovery_action(recovery_action, func, *args, **kwargs)
                else:
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get error handler from instance or create new one
            error_handler = None
            for arg in args:
                if hasattr(arg, 'error_handler'):
                    error_handler = arg.error_handler
                    break

            if error_handler is None:
                error_handler = UnifiedErrorHandler(enable_recovery=recovery_enabled)

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=func.__name__,
                    component=func.__module__,
                    additional_info={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                )

                recovery_action = error_handler.handle_error(e, context, category, severity)

                if recovery_enabled and recovery_action.action_type != "abort":
                    return await error_handler.execute_recovery_action_async(recovery_action, func, *args, **kwargs)
                else:
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator
