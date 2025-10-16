"""
Enhanced Error Handler - Compatibility layer for error handling across the codebase.

This module provides enhanced error handling functionality that's compatible
with the existing error handling patterns used throughout the Ares project.
"""

import logging
import traceback
from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
from enum import Enum
import functools
import time

# Import from existing error_handler for consistency
from .error_handler import (
    UnifiedErrorHandler,
    ValidationError,
    DataQualityError,
    ConfigurationError,
    ProcessingError,
    MathValidationError,
    handles_errors,
    safe_execution,
    get_unified_error_handler
)

# =============================================================================
# ENHANCED ERROR CLASSES
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    DATA_QUALITY = "data_quality"
    CONFIGURATION = "configuration"
    PROCESSING = "processing"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class ErrorContext:
    """Context information for errors."""

    def __init__(self, operation: str = "", component: str = "",
                 user_id: Optional[str] = None, session_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.component = component
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = time.time()
        self.metadata = metadata or {}

class ErrorRecord:
    """Structured error record for tracking and analysis."""

    def __init__(self, error: Exception, context: ErrorContext,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 recovery_action: Optional[str] = None):
        self.error = error
        self.context = context
        self.severity = severity
        self.category = category
        self.recovery_action = recovery_action
        self.traceback = traceback.format_exc()
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'severity': self.severity.value,
            'category': self.category.value,
            'context': {
                'operation': self.context.operation,
                'component': self.context.component,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'metadata': self.context.metadata
            },
            'recovery_action': self.recovery_action,
            'traceback': self.traceback,
            'timestamp': self.timestamp
        }

# =============================================================================
# ENHANCED ERROR HANDLER
# =============================================================================

class EnhancedErrorHandler(UnifiedErrorHandler):
    """Enhanced error handler with additional tracking and recovery features."""

    def __init__(self, logger: logging.Logger = None):
        super().__init__(logger)
        self.error_records: List[ErrorRecord] = []
        self.recovery_actions: Dict[str, Callable] = {}

    def handle_error_with_tracking(self, error: Exception, context: ErrorContext,
                                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                                  category: ErrorCategory = ErrorCategory.UNKNOWN,
                                  reraise: bool = True) -> Any:
        """Handle error with enhanced tracking."""
        # Create error record
        record = ErrorRecord(error, context, severity, category)
        self.error_records.append(record)

        # Log with enhanced information
        log_message = (f"❌ [{severity.value.upper()}] {category.value}: "
                      f"{context.operation} in {context.component} - {type(error).__name__}: {error}")

        self.logger.error(log_message, exc_info=True)

        # Try recovery if available
        if category.value in self.recovery_actions:
            try:
                recovery_result = self.recovery_actions[category.value](error, context)
                if recovery_result is not None:
                    self.logger.info(f"✅ Recovery successful for {category.value}")
                    return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"❌ Recovery failed: {recovery_error}")

        if reraise:
            raise error

        return None

    def register_recovery_action(self, category: str, action: Callable):
        """Register a recovery action for a specific error category."""
        self.recovery_actions[category] = action

    def get_error_summary_enhanced(self) -> Dict[str, Any]:
        """Get enhanced error summary with severity and category breakdown."""
        summary = super().get_error_summary()

        # Add enhanced metrics
        severity_counts = {}
        category_counts = {}

        for record in self.error_records[-100:]:  # Last 100 errors
            severity_counts[record.severity.value] = severity_counts.get(record.severity.value, 0) + 1
            category_counts[record.category.value] = category_counts.get(record.category.value, 0) + 1

        summary.update({
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'recent_records': [record.to_dict() for record in self.error_records[-10:]]
        })

        return summary

# =============================================================================
# COMPATIBILITY FUNCTIONS
# =============================================================================

def handle_errors_with_tracking(error_handlers: Optional[Dict[Type[Exception], Tuple[Any, str]]] = None,
                               default_return: Any = None,
                               context: str = "operation",
                               severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                               category: ErrorCategory = ErrorCategory.UNKNOWN) -> Callable:
    """
    Enhanced error handling decorator compatible with existing patterns.

    Args:
        error_handlers: Dict mapping exception types to (return_value, log_message) tuples
        default_return: Default return value if no specific handler matches
        context: Context string for logging
        severity: Error severity level
        category: Error category

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check for specific error handlers
                if error_handlers:
                    for error_type, (return_value, log_message) in error_handlers.items():
                        if isinstance(e, error_type):
                            logger = logging.getLogger(__name__)
                            logger.warning(f"⚠️ {context}: {log_message} - {e}")
                            return return_value

                # Enhanced error handling
                error_context = ErrorContext(
                    operation=context,
                    component=func.__name__,
                    metadata={'function': func.__name__, 'args_count': len(args)}
                )

                handler = get_enhanced_error_handler()
                return handler.handle_error_with_tracking(
                    e, error_context, severity, category, reraise=False
                ) or default_return

        return wrapper
    return decorator

def handle_errors_basic(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """
    Basic error handling function for simple cases.

    Args:
        func: Function to execute
        *args: Arguments for the function
        default_return: Default return value on error
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error in {func.__name__}: {e}")
        return default_return

# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_enhanced_error_handler: Optional[EnhancedErrorHandler] = None

def get_enhanced_error_handler() -> EnhancedErrorHandler:
    """Get the global enhanced error handler."""
    global _enhanced_error_handler
    if _enhanced_error_handler is None:
        _enhanced_error_handler = EnhancedErrorHandler()
    return _enhanced_error_handler

def setup_enhanced_error_handling(logger: logging.Logger = None) -> EnhancedErrorHandler:
    """Setup enhanced error handling."""
    global _enhanced_error_handler
    _enhanced_error_handler = EnhancedErrorHandler(logger)
    return _enhanced_error_handler

# Initialize by default
if _enhanced_error_handler is None:
    setup_enhanced_error_handling()
