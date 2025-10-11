"""
Error Handling Framework for Feature Lookback Optimization.

This module provides standardized error handling with graceful degradation,
detailed logging, and recovery mechanisms.
"""

import logging
import traceback
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

# Import utility modules
from src.utils.common_utilities import CommonUtilities
from src.utils.serialization_utils import UniversalSerializer
from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    DATA_PROCESSING = "data_processing"
    FILE_IO = "file_io"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorDetails:
    """Detailed error information."""
    error: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    timestamp: str
    context: Dict[str, Any]
    stack_trace: str
    recoverable: bool = False


# Removed ErrorRecoveryResult - not needed for fast failing


class StandardizedErrorHandler:
    """
    Standardized error handler with fast failing.

    Provides consistent error handling across the feature lookback optimization
    component with immediate failure propagation and detailed logging.
    """

    def __init__(self, logger=None, component_name: str = "FeatureLookbackOptimization"):
        """Initialize the error handler."""
        self.logger = logger or logging.getLogger(__name__)
        self.component_name = component_name
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        # Error tracking
        self.error_counts = {}
        self.recent_errors = []

    def handle_error(
        self,
        error: Exception,
        operation: str,
        return_value: Any = None,
        reraise: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Handle errors with configurable behavior.

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            return_value: Value to return if reraise=False
            reraise: Whether to re-raise the exception (default: True for fast failing)
            context: Additional context information

        Returns:
            return_value if reraise=False, otherwise raises the error
        """
        try:
            # Create error details
            error_details = self._create_error_details(error, operation, context)

            # Log the error with tprint
            self._log_error(error_details)
            tprint_error(f"❌ Error in {operation}: {str(error_details.error)}")
            if error_details.severity == ErrorSeverity.CRITICAL:
                tprint_error(f"🚨 Critical error - failing immediately: {operation}")

            # Track error statistics
            self._track_error(error_details)

            # Update recent errors
            self.recent_errors.append(error_details)
            if len(self.recent_errors) > 100:  # Keep only recent errors
                self.recent_errors = self.recent_errors[-100:]

            # Respect the reraise parameter
            if reraise:
                raise error
            else:
                tprint_warning(f"⚠️ Error suppressed in {operation}, returning fallback value")
                return return_value

        except Exception as e:
            self.logger.critical(f"Error handler failed: {e}")
            tprint_error(f"🚨 Error handler itself failed: {e}")
            # Always re-raise the original error if handler fails
            raise error

    def handle_warning(self, warning_msg: str, operation: str, context: Optional[Dict[str, Any]] = None):
        """Handle warnings in a standardized way."""
        try:
            self.logger.warning(f"[{self.component_name}] {operation}: {warning_msg}")
            if context:
                self.logger.debug(f"Warning context: {context}")
        except Exception as e:
            self.logger.error(f"Failed to handle warning: {e}")

    def handle_info(self, info_msg: str, operation: str, context: Optional[Dict[str, Any]] = None):
        """Handle info messages in a standardized way."""
        try:
            self.logger.info(f"[{self.component_name}] {operation}: {info_msg}")
            if context:
                self.logger.debug(f"Info context: {context}")
        except Exception as e:
            self.logger.error(f"Failed to handle info message: {e}")

    def _create_error_details(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorDetails:
        """Create detailed error information."""
        import datetime

        # Determine severity and category
        severity = self._classify_error_severity(error, operation)
        category = self._classify_error_category(error, operation)

        # Determine if error is recoverable
        recoverable = self._is_error_recoverable(error, operation, category)

        return ErrorDetails(
            error=error,
            severity=severity,
            category=category,
            operation=operation,
            timestamp=datetime.datetime.now().isoformat(),
            context=context or {},
            stack_trace=traceback.format_exc(),
            recoverable=recoverable
        )

    def _classify_error_severity(self, error: Exception, operation: str) -> ErrorSeverity:
        """Classify error severity based on error type and operation."""
        error_type = type(error).__name__

        # Critical errors
        critical_operations = ['execute', 'optimize', 'validate_data']
        if operation in critical_operations:
            return ErrorSeverity.CRITICAL

        # High severity errors
        high_severity_types = ['ValueError', 'TypeError', 'KeyError']
        if error_type in high_severity_types:
            return ErrorSeverity.HIGH

        # Medium severity errors
        medium_severity_types = ['AttributeError', 'IndexError']
        if error_type in medium_severity_types:
            return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW

    def _classify_error_category(self, error: Exception, operation: str) -> ErrorCategory:
        """Classify error category based on error type and operation."""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Validation errors
        if 'validation' in operation.lower() or 'validate' in operation.lower():
            return ErrorCategory.VALIDATION

        # Data processing errors
        if any(term in error_msg for term in ['data', 'column', 'dataframe', 'series']):
            return ErrorCategory.DATA_PROCESSING

        # Memory errors
        if any(term in error_msg for term in ['memory', 'out of memory', 'allocation']):
            return ErrorCategory.MEMORY

        # File I/O errors
        if any(term in error_msg for term in ['file', 'io', 'read', 'write', 'path']):
            return ErrorCategory.FILE_IO

        # Configuration errors
        if any(term in error_msg for term in ['config', 'parameter', 'setting']):
            return ErrorCategory.CONFIGURATION

        # Network errors
        if any(term in error_msg for term in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK

        # Optimization errors
        if any(term in operation.lower() for term in ['optimize', 'lookback', 'feature']):
            return ErrorCategory.OPTIMIZATION

        # Default to unknown
        return ErrorCategory.UNKNOWN

    def _is_error_recoverable(self, error: Exception, operation: str, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable - always False for fast failing."""
        # Fast failing: all errors are non-recoverable
        return False

    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_message = f"[{self.component_name}] {error_details.operation}: {error_details.error}"

        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.debug(f"Stack trace: {error_details.stack_trace}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
            self.logger.debug(f"Context: {error_details.context}")
        else:
            self.logger.info(log_message)

    def _track_error(self, error_details: ErrorDetails):
        """Track error statistics."""
        # Track by category
        category_key = f"category_{error_details.category.value}"
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1

        # Track by severity
        severity_key = f"severity_{error_details.severity.value}"
        self.error_counts[severity_key] = self.error_counts.get(severity_key, 0) + 1

        # Track by operation
        operation_key = f"operation_{error_details.operation}"
        self.error_counts[operation_key] = self.error_counts.get(operation_key, 0) + 1

    # Removed all recovery methods - fast failing doesn't need them

    def get_error_statistics(self) -> Dict[str, int]:
        """Get error statistics."""
        return self.error_counts.copy()

    def get_recent_errors(self, limit: int = 10) -> List[ErrorDetails]:
        """Get recent errors."""
        return self.recent_errors[-limit:].copy()

    def reset_error_tracking(self):
        """Reset error tracking statistics."""
        self.error_counts.clear()
        self.recent_errors.clear()
