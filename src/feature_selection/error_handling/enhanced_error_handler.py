"""
Enhanced Error Handling for Feature Selection

This module provides comprehensive error handling and recovery strategies for
feature selection operations with enhanced logging using tprint.
"""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.tprint import (
    tprint, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_info, tprint_performance
)

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    FALLBACK_METHOD = "fallback_method"
    REDUCE_FEATURES = "reduce_features"
    SIMPLIFY_PARAMETERS = "simplify_parameters"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    SKIP_OPERATION = "skip_operation"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    method: str
    data_shape: Tuple[int, int]
    parameters: Dict[str, Any]
    error_type: str
    error_message: str
    severity: ErrorSeverity
    suggested_recovery: RecoveryStrategy
    additional_info: Optional[Dict[str, Any]] = None

class FeatureSelectionError(Exception):
    """Base exception for feature selection errors."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context
        self.timestamp = time.time()

        # Log error with tprint
        tprint_error(f"❌ FeatureSelectionError: {message}")
        if context:
            tprint_error(f"   Operation: {context.operation}")
            tprint_error(f"   Method: {context.method}")
            tprint_error(f"   Data shape: {context.data_shape}")

class InsufficientDataError(FeatureSelectionError):
    """Raised when data is insufficient for feature selection."""

    def __init__(self, message: str, min_required: int, actual: int, context: Optional[ErrorContext] = None):
        super().__init__(message, context)
        self.min_required = min_required
        self.actual = actual

        tprint_error(f"📊 Insufficient data: {actual} samples, need at least {min_required}")

class SelectionConvergenceError(FeatureSelectionError):
    """Raised when selection algorithm fails to converge."""

    def __init__(self, message: str, max_iterations: int, actual_iterations: int, context: Optional[ErrorContext] = None):
        super().__init__(message, context)
        self.max_iterations = max_iterations
        self.actual_iterations = actual_iterations

        tprint_error(f"🔄 Convergence failed: {actual_iterations}/{max_iterations} iterations")

class ConfigurationError(FeatureSelectionError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, invalid_param: str, suggested_value: Any, context: Optional[ErrorContext] = None):
        super().__init__(message, context)
        self.invalid_param = invalid_param
        self.suggested_value = suggested_value

        tprint_error(f"⚙️ Configuration error: {invalid_param} = {suggested_value}")

class EnhancedErrorHandler:
    """Enhanced error handler for feature selection operations."""

    def __init__(self, enable_recovery: bool = True, log_level: str = "INFO"):
        """Initialize the error handler."""
        self.enable_recovery = enable_recovery
        self.log_level = log_level
        self.error_history: List[ErrorContext] = []
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0
        }

        tprint_success("🛡️ EnhancedErrorHandler initialized")

    def _analyze_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Analyze error and determine recovery strategy."""
        error_type = type(error).__name__
        operation = context.get('operation', 'unknown')
        method = context.get('method', 'unknown')
        data_shape = context.get('data_shape', (0, 0))
        parameters = context.get('parameters', {})

        # Determine severity and recovery strategy
        if isinstance(error, InsufficientDataError):
            severity = ErrorSeverity.HIGH
            recovery = RecoveryStrategy.REDUCE_FEATURES
        elif isinstance(error, SelectionConvergenceError):
            severity = ErrorSeverity.MEDIUM
            recovery = RecoveryStrategy.SIMPLIFY_PARAMETERS
        elif isinstance(error, ConfigurationError):
            severity = ErrorSeverity.LOW
            recovery = RecoveryStrategy.SIMPLIFY_PARAMETERS
        else:
            severity = ErrorSeverity.MEDIUM
            recovery = RecoveryStrategy.FALLBACK_METHOD

        return ErrorContext(
            operation=operation,
            method=method,
            data_shape=data_shape,
            parameters=parameters,
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            suggested_recovery=recovery
        )

    def _log_error(self, error_context: ErrorContext, error: Exception):
        """Log error with appropriate level and formatting."""
        severity_colors = {
            ErrorSeverity.LOW: "🔵",
            ErrorSeverity.MEDIUM: "🟡",
            ErrorSeverity.HIGH: "🟠",
            ErrorSeverity.CRITICAL: "🔴"
        }

        color = severity_colors.get(error_context.severity, "⚪")

        tprint_error(f"{color} {error_context.severity.value.upper()}: {error_context.error_type}")
        tprint_error(f"   Operation: {error_context.operation}")
        tprint_error(f"   Method: {error_context.method}")
        tprint_error(f"   Data: {error_context.data_shape}")
        tprint_error(f"   Message: {error_context.error_message}")
        tprint_error(f"   Recovery: {error_context.suggested_recovery.value}")

        # Log stack trace for critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            tprint_debug(f"Stack trace:\n{traceback.format_exc()}")

    def _attempt_recovery(self, error_context: ErrorContext,
                         original_func: Callable,
                         *args, **kwargs) -> Optional[Any]:
        """Attempt to recover from error using appropriate strategy."""
        if not self.enable_recovery:
            return None

        tprint_info(f"🔄 Attempting recovery: {error_context.suggested_recovery.value}")

        try:
            if error_context.suggested_recovery == RecoveryStrategy.FALLBACK_METHOD:
                return self._fallback_method_recovery(original_func, *args, **kwargs)
            elif error_context.suggested_recovery == RecoveryStrategy.REDUCE_FEATURES:
                return self._reduce_features_recovery(original_func, *args, **kwargs)
            elif error_context.suggested_recovery == RecoveryStrategy.SIMPLIFY_PARAMETERS:
                return self._simplify_parameters_recovery(original_func, *args, **kwargs)
            elif error_context.suggested_recovery == RecoveryStrategy.RETRY_WITH_BACKOFF:
                return self._retry_with_backoff_recovery(original_func, *args, **kwargs)
            else:
                tprint_warning("⚠️ No recovery strategy available")
                return None

        except Exception as recovery_error:
            tprint_error(f"❌ Recovery failed: {recovery_error}")
            self.recovery_stats['failed_recoveries'] += 1
            return None

    def _fallback_method_recovery(self, original_func: Callable, *args, **kwargs) -> Any:
        """Recovery using fallback method."""
        tprint_info("🔄 Using fallback method: basic selection")

        # Use basic selection as fallback
        kwargs['method'] = 'basic'
        kwargs['max_features'] = min(kwargs.get('max_features', 50), 20)

        return original_func(*args, **kwargs)

    def _reduce_features_recovery(self, original_func: Callable, *args, **kwargs) -> Any:
        """Recovery by reducing feature count."""
        tprint_info("🔄 Reducing feature count for recovery")

        # Reduce max_features significantly
        current_max = kwargs.get('max_features', 50)
        kwargs['max_features'] = max(5, current_max // 4)

        return original_func(*args, **kwargs)

    def _simplify_parameters_recovery(self, original_func: Callable, *args, **kwargs) -> Any:
        """Recovery by simplifying parameters."""
        tprint_info("🔄 Simplifying parameters for recovery")

        # Remove complex parameters
        simplified_kwargs = {
            'method': 'basic',
            'max_features': min(kwargs.get('max_features', 50), 20)
        }

        return original_func(*args, **simplified_kwargs)

    def _retry_with_backoff_recovery(self, original_func: Callable, *args, **kwargs) -> Any:
        """Recovery by retrying with exponential backoff."""
        import time

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                tprint_info(f"🔄 Retry attempt {attempt + 1}/{max_retries}")
                return original_func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    tprint_info(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    raise e

    def handle_error(self, error: Exception, context: Dict[str, Any],
                    original_func: Optional[Callable] = None,
                    *args, **kwargs) -> Optional[Any]:
        """Handle error with analysis and recovery."""
        # Analyze error
        error_context = self._analyze_error(error, context)
        self.error_history.append(error_context)
        self.recovery_stats['total_errors'] += 1

        # Log error
        self._log_error(error_context, error)

        # Attempt recovery if function provided
        if original_func and self.enable_recovery:
            try:
                result = self._attempt_recovery(error_context, original_func, *args, **kwargs)
                if result is not None:
                    self.recovery_stats['recovered_errors'] += 1
                    tprint_success("✅ Recovery successful")
                    return result
            except Exception as recovery_error:
                tprint_error(f"❌ Recovery failed: {recovery_error}")

        # If no recovery possible, re-raise original error
        raise error

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error handling statistics."""
        total_errors = self.recovery_stats['total_errors']
        recovered = self.recovery_stats['recovered_errors']
        failed = self.recovery_stats['failed_recoveries']

        recovery_rate = recovered / max(1, total_errors)

        summary = {
            'total_errors': total_errors,
            'recovered_errors': recovered,
            'failed_recoveries': failed,
            'recovery_rate': recovery_rate,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

        tprint_performance(f"📊 Error Summary: {recovery_rate:.1%} recovery rate "
                         f"({recovered}/{total_errors} errors)")

        return summary

def robust_feature_selection(selection_func: Callable,
                           error_handler: Optional[EnhancedErrorHandler] = None):
    """Decorator for robust feature selection with error handling."""
    if error_handler is None:
        error_handler = EnhancedErrorHandler()

    def wrapper(X, y, method='comprehensive', **kwargs):
        context = {
            'operation': 'feature_selection',
            'method': method,
            'data_shape': X.shape if hasattr(X, 'shape') else (len(X), 0),
            'parameters': kwargs
        }

        try:
            tprint_info(f"🔍 Starting {method} feature selection")
            result = selection_func(X, y, method=method, **kwargs)
            tprint_success(f"✅ Feature selection completed: {len(result.get('selected_features', []))} features")
            return result

        except Exception as error:
            tprint_error(f"❌ Feature selection failed: {error}")
            return error_handler.handle_error(error, context, selection_func, X, y, method=method, **kwargs)

    return wrapper

def create_error_handler(enable_recovery: bool = True, log_level: str = "INFO") -> EnhancedErrorHandler:
    """Create an enhanced error handler."""
    return EnhancedErrorHandler(enable_recovery, log_level)
