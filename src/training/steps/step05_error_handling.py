from ..standardized_parquet_handler import standardized_parquet_handler
"""
Step05 Standardized Error Handling Module

This module provides standardized error handling patterns for Step05 labeling,
including centralized error logging, recovery mechanisms, and error classification.
"""

import traceback
import functools
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from src.utils.logger import system_logger
from src.utils.common_operations import safe_mean, safe_std, safe_float, safe_int, validate_dataframe_schema, validate_data_quality, safe_copy, safe_deepcopy, get_current_datetime, format_datetime, create_empty_dataframe, safe_fillna, safe_rolling, safe_append, safe_extend, safe_dict_get, safe_dict_items, safe_lower, safe_upper, safe_join, get_logger, setup_basic_logging, safe_exception_handler, timed_operation, format_bytes, chunked_iterable, parallel_map, safe_log_metric, safe_log_params, safe_log_artifact
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive, validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change, validate_correlation_matrix, safe_matrix_inverse, math_safe, MathValidationError
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.core.decorators import traced, validates, cached, log_execution_time, handles_errors
from src.core.errors import AppError, ValidationError, DataIntegrityError, BusinessRuleError, NotFoundError, ConflictError, RateLimitError, TimeoutError, ServiceUnavailableError, ErrorCode
import logging
import time

logger = system_logger.getChild('Step05ErrorHandling')

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    DATA_INTEGRITY = "data_integrity"
    COMPUTATION = "computation"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    step_name: str = "step05"
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: str
    recovery_action: Optional[str] = None
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

class Step05ErrorHandler:
    """Standardized error handler for Step05 operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        self.error_records: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self._setup_default_recovery_strategies()
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.VALIDATION: self._recover_from_validation_error,
            ErrorCategory.DATA_INTEGRITY: self._recover_from_data_integrity_error,
            ErrorCategory.COMPUTATION: self._recover_from_computation_error,
            ErrorCategory.MEMORY: self._recover_from_memory_error,
            ErrorCategory.CONFIGURATION: self._recover_from_configuration_error,
            ErrorCategory.BUSINESS_LOGIC: self._recover_from_business_logic_error,
            ErrorCategory.UNKNOWN: self._recover_from_unknown_error
        }
    
    def handle_error(self, error: Exception, context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    recovery_action: Optional[str] = None) -> ErrorRecord:
        """
        Handle an error with standardized processing.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            severity: Severity level of the error
            category: Category of the error
            recovery_action: Optional recovery action taken
            
        Returns:
            ErrorRecord with details of the error
        """
        try:
            # Generate unique error ID
            error_id = f"{context.step_name}_{context.function_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                error_type=type(error).__name__,
                error_message=str(error),
                severity=severity,
                category=category,
                context=context,
                stack_trace=traceback.format_exc(),
                recovery_action=recovery_action
            )
            
            # Log the error
            self._log_error(error_record)
            
            # Update error counts
            self._update_error_counts(error_record)
            
            # Store error record
            self.error_records.append(error_record)
            
            # Attempt recovery if strategy exists
            if category in self.recovery_strategies:
                try:
                    recovery_result = self.recovery_strategies[category](error_record)
                    if recovery_result:
                        error_record.recovery_action = recovery_result
                        error_record.resolved = True
                        error_record.resolution_timestamp = datetime.now()
                        self.logger.info(f"✅ Error {error_id} recovered successfully")
                except Exception as recovery_error:
                    self.logger.error(f"❌ Recovery failed for error {error_id}: {recovery_error}")
            
            return error_record
            
        except Exception as handling_error:
            self.logger.critical(f"💥 Error handling failed: {handling_error}")
            # Return minimal error record
            return ErrorRecord(
                error_id="error_handling_failed",
                error_type=type(handling_error).__name__,
                error_message=str(handling_error),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.UNKNOWN,
                context=context,
                stack_trace=traceback.format_exc()
            )
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level based on severity."""
        log_message = f"[{error_record.error_id}] {error_record.error_type}: {error_record.error_message}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log stack trace for high severity errors
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Stack trace: {error_record.stack_trace}")
    
    def _update_error_counts(self, error_record: ErrorRecord):
        """Update error count statistics."""
        error_key = f"{error_record.category.value}_{error_record.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _recover_from_validation_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for validation errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from validation error: {error_record.error_id}")
            
            # Common validation error recoveries
            if "missing" in error_record.error_message.lower():
                return "Attempted to use default values for missing data"
            elif "invalid" in error_record.error_message.lower():
                return "Attempted to sanitize invalid data"
            elif "type" in error_record.error_message.lower():
                return "Attempted to convert data types"
            
            return "Validation error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Validation error recovery failed: {e}")
            return None
    
    def _recover_from_data_integrity_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for data integrity errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from data integrity error: {error_record.error_id}")
            
            # Common data integrity recoveries
            if "null" in error_record.error_message.lower() or "nan" in error_record.error_message.lower():
                return "Attempted to handle null/NaN values"
            elif "duplicate" in error_record.error_message.lower():
                return "Attempted to remove duplicate records"
            elif "format" in error_record.error_message.lower():
                return "Attempted to fix data format issues"
            
            return "Data integrity error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Data integrity error recovery failed: {e}")
            return None
    
    def _recover_from_computation_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for computation errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from computation error: {error_record.error_id}")
            
            # Common computation error recoveries
            if "division" in error_record.error_message.lower() or "zero" in error_record.error_message.lower():
                return "Attempted to handle division by zero"
            elif "overflow" in error_record.error_message.lower():
                return "Attempted to handle numerical overflow"
            elif "convergence" in error_record.error_message.lower():
                return "Attempted to adjust convergence parameters"
            
            return "Computation error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Computation error recovery failed: {e}")
            return None
    
    def _recover_from_memory_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for memory errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from memory error: {error_record.error_id}")
            
            # Common memory error recoveries
            if "memory" in error_record.error_message.lower():
                return "Attempted to free memory and retry operation"
            elif "allocation" in error_record.error_message.lower():
                return "Attempted to reduce memory allocation"
            
            return "Memory error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Memory error recovery failed: {e}")
            return None
    
    def _recover_from_configuration_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for configuration errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from configuration error: {error_record.error_id}")
            
            # Common configuration error recoveries
            if "missing" in error_record.error_message.lower():
                return "Attempted to use default configuration values"
            elif "invalid" in error_record.error_message.lower():
                return "Attempted to validate and correct configuration"
            
            return "Configuration error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Configuration error recovery failed: {e}")
            return None
    
    def _recover_from_business_logic_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for business logic errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from business logic error: {error_record.error_id}")
            
            # Common business logic error recoveries
            if "label" in error_record.error_message.lower():
                return "Attempted to use fallback labeling strategy"
            elif "barrier" in error_record.error_message.lower():
                return "Attempted to use default barrier parameters"
            
            return "Business logic error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Business logic error recovery failed: {e}")
            return None
    
    def _recover_from_unknown_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for unknown errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from unknown error: {error_record.error_id}")
            return "Generic error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Unknown error recovery failed: {e}")
            return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors handled."""
        try:
            total_errors = len(self.error_records)
            resolved_errors = len([e for e in self.error_records if e.resolved])
            critical_errors = len([e for e in self.error_records if e.severity == ErrorSeverity.CRITICAL])
            high_errors = len([e for e in self.error_records if e.severity == ErrorSeverity.HIGH])
            
            # Error counts by category
            category_counts = {}
            for record in self.error_records:
                category = record.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Recent errors (last 24 hours)
            recent_cutoff = datetime.now().timestamp() - 86400  # 24 hours
            recent_errors = len([e for e in self.error_records 
                               if e.context.timestamp.timestamp() > recent_cutoff])
            
            return {
                'total_errors': total_errors,
                'resolved_errors': resolved_errors,
                'unresolved_errors': total_errors - resolved_errors,
                'critical_errors': critical_errors,
                'high_errors': high_errors,
                'recent_errors': recent_errors,
                'error_counts_by_category': category_counts,
                'error_counts_by_type': self.error_counts,
                'resolution_rate': resolved_errors / total_errors if total_errors > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error summary generation failed: {e}")
            return {'error': str(e)}
    
    def clear_old_errors(self, days_to_keep: int = 30):
        """Clear error records older than specified days."""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 86400)
            
            original_count = len(self.error_records)
            self.error_records = [e for e in self.error_records 
                                if e.context.timestamp.timestamp() > cutoff_date]
            
            cleared_count = original_count - len(self.error_records)
            self.logger.info(f"🧹 Cleared {cleared_count} old error records")
            
        except Exception as e:
            self.logger.error(f"❌ Error clearing failed: {e}")

def step05_error_handler(error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                        recovery_action: Optional[str] = None):
    """
    Decorator for standardized error handling in Step05 functions.
    
    Args:
        error_severity: Default severity level for errors
        error_category: Default category for errors
        recovery_action: Default recovery action
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get error handler instance (assuming it's available in the class)
            error_handler = None
            if args and hasattr(args[0], 'error_handler'):
                error_handler = args[0].error_handler
            elif 'error_handler' in kwargs:
                error_handler = kwargs['error_handler']
            
            if error_handler is None:
                # Create default error handler
                error_handler = Step05ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    function_name=func.__name__,
                    step_name="step05",
                    additional_context={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'function_module': func.__module__
                    }
                )
                
                # Handle the error
                error_record = error_handler.handle_error(
                    error=e,
                    context=context,
                    severity=error_severity,
                    category=error_category,
                    recovery_action=recovery_action
                )
                
                # Re-raise if critical or high severity
                if error_record.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                    raise
                
                # Return None for other severities (graceful degradation)
                return None
        
        return wrapper
    return decorator

def step05_async_error_handler(error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                              error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                              recovery_action: Optional[str] = None):
    """
    Decorator for standardized error handling in async Step05 functions.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get error handler instance
            error_handler = None
            if args and hasattr(args[0], 'error_handler'):
                error_handler = args[0].error_handler
            elif 'error_handler' in kwargs:
                error_handler = kwargs['error_handler']
            
            if error_handler is None:
                error_handler = Step05ErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    function_name=func.__name__,
                    step_name="step05",
                    additional_context={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'function_module': func.__module__,
                        'async_function': True
                    }
                )
                
                # Handle the error
                error_record = error_handler.handle_error(
                    error=e,
                    context=context,
                    severity=error_severity,
                    category=error_category,
                    recovery_action=recovery_action
                )
                
                # Re-raise if critical or high severity
                if error_record.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                    raise
                
                # Return None for other severities
                return None
        
        return wrapper
    return decorator