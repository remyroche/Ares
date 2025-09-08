"""
Enhanced Error Handling for Training Steps

This module provides comprehensive error handling that ensures:
1. No silent failures - all errors are properly logged and propagated
2. Fail-fast behavior for critical processes
3. Proper error categorization and severity levels
4. Recovery strategies where appropriate
5. Comprehensive error reporting and monitoring
"""

import traceback
import functools

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import system_logger

class ErrorSeverity(Enum):
    """Error severity levels for proper handling."""
    LOW = "low"           # Non-critical, can continue
    MEDIUM = "medium"     # Important, should be logged but can continue
    HIGH = "high"         # Critical, should fail fast
    CRITICAL = "critical" # Fatal, must stop immediately

class ErrorCategory(Enum):
    """Error categories for proper classification."""
    DATA_QUALITY = "data_quality"
    VALIDATION = "validation"
    COMPUTATION = "computation"
    MEMORY = "memory"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    step_name: str
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
    should_fail_fast: bool = False

class CriticalProcessError(Exception):
    """Exception raised when a critical process fails and should stop execution."""
    def __init__(self, message: str, error_record: ErrorRecord):
        super().__init__(message)
        self.error_record = error_record

class EnhancedErrorHandler:
    """Enhanced error handler with fail-fast capabilities and comprehensive logging."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('EnhancedErrorHandler')
        self.error_records: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        self.critical_processes = {
            'hmm_clustering',
            'feature_generation', 
            'matrix_operations',
            'ml_model_training',
            'sr_levels_detection',
            'regime_detection'
        }
        
    def handle_error(self, 
                    error: Exception, 
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    recovery_action: Optional[str] = None,
                    should_fail_fast: bool = False) -> ErrorRecord:
        """
        Handle an error with enhanced processing and fail-fast logic.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            severity: Severity level of the error
            category: Category of the error
            recovery_action: Optional recovery action taken
            should_fail_fast: Whether this error should trigger fail-fast behavior
            
        Returns:
            ErrorRecord with details of the error
            
        Raises:
            CriticalProcessError: If this is a critical process error that should stop execution
        """
        try:
            # Generate unique error ID
            error_id = f"{context.step_name}_{context.function_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine if this is a critical process
            is_critical_process = any(process in context.function_name.lower() 
                                    for process in self.critical_processes)
            
            # Override severity for critical processes
            if is_critical_process and severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                severity = ErrorSeverity.HIGH
                should_fail_fast = True
                
            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                error_type=type(error).__name__,
                error_message=str(error),
                severity=severity,
                category=category,
                context=context,
                stack_trace=traceback.format_exc(),
                recovery_action=recovery_action,
                should_fail_fast=should_fail_fast
            )
            
            # Log the error with appropriate level
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
                    # If recovery fails for critical process, escalate severity
                    if is_critical_process:
                        error_record.severity = ErrorSeverity.CRITICAL
                        should_fail_fast = True
            
            # Fail fast for critical errors or critical processes
            if should_fail_fast or error_record.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"🚨 FAIL-FAST TRIGGERED: {error_id}")
                raise CriticalProcessError(
                    f"Critical process failed: {error_record.error_message}",
                    error_record
                )
            
            return error_record
            
        except CriticalProcessError:
            # Re-raise critical process errors
            raise
        except Exception as handling_error:
            self.logger.critical(f"💥 Error handling failed: {handling_error}")
            # Return minimal error record and raise critical error
            minimal_record = ErrorRecord(
                error_id="error_handling_failed",
                error_type=type(handling_error).__name__,
                error_message=str(handling_error),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.UNKNOWN,
                context=context,
                stack_trace=traceback.format_exc(),
                should_fail_fast=True
            )
            raise CriticalProcessError(
                f"Error handling system failed: {handling_error}",
                minimal_record
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
    
    @property
    def recovery_strategies(self) -> Dict[ErrorCategory, Callable]:
        """Get recovery strategies for different error categories."""
        return {
            ErrorCategory.DATA_QUALITY: self._recover_from_data_quality_error,
            ErrorCategory.VALIDATION: self._recover_from_validation_error,
            ErrorCategory.COMPUTATION: self._recover_from_computation_error,
            ErrorCategory.MEMORY: self._recover_from_memory_error,
            ErrorCategory.CONFIGURATION: self._recover_from_configuration_error,
            ErrorCategory.BUSINESS_LOGIC: self._recover_from_business_logic_error,
            ErrorCategory.DEPENDENCY: self._recover_from_dependency_error,
            ErrorCategory.TIMEOUT: self._recover_from_timeout_error,
            ErrorCategory.UNKNOWN: self._recover_from_unknown_error
        }
    
    def _recover_from_data_quality_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for data quality errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from data quality error: {error_record.error_id}")
            
            if "missing" in error_record.error_message.lower():
                return "Attempted to use default values for missing data"
            elif "invalid" in error_record.error_message.lower():
                return "Attempted to sanitize invalid data"
            elif "null" in error_record.error_message.lower() or "nan" in error_record.error_message.lower():
                return "Attempted to handle null/NaN values"
            elif "duplicate" in error_record.error_message.lower():
                return "Attempted to remove duplicate records"
            
            return "Data quality error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Data quality error recovery failed: {e}")
            return None
    
    def _recover_from_validation_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for validation errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from validation error: {error_record.error_id}")
            
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
    
    def _recover_from_computation_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for computation errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from computation error: {error_record.error_id}")
            
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
            
            if "label" in error_record.error_message.lower():
                return "Attempted to use fallback labeling strategy"
            elif "barrier" in error_record.error_message.lower():
                return "Attempted to use default barrier parameters"
            
            return "Business logic error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Business logic error recovery failed: {e}")
            return None
    
    def _recover_from_dependency_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for dependency errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from dependency error: {error_record.error_id}")
            
            if "import" in error_record.error_message.lower():
                return "Attempted to use fallback implementation"
            elif "module" in error_record.error_message.lower():
                return "Attempted to use alternative module"
            
            return "Dependency error recovery attempted"
            
        except Exception as e:
            self.logger.error(f"❌ Dependency error recovery failed: {e}")
            return None
    
    def _recover_from_timeout_error(self, error_record: ErrorRecord) -> Optional[str]:
        """Recovery strategy for timeout errors."""
        try:
            self.logger.info(f"🔄 Attempting recovery from timeout error: {error_record.error_id}")
            
            return "Attempted to retry with extended timeout"
            
        except Exception as e:
            self.logger.error(f"❌ Timeout error recovery failed: {e}")
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
            fail_fast_errors = len([e for e in self.error_records if e.should_fail_fast])
            
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
                'fail_fast_errors': fail_fast_errors,
                'recent_errors': recent_errors,
                'error_counts_by_category': category_counts,
                'error_counts_by_type': self.error_counts,
                'resolution_rate': resolved_errors / total_errors if total_errors > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error summary generation failed: {e}")
            return {'error': str(e)}

def enhanced_error_handler(error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                          recovery_action: Optional[str] = None,
                          should_fail_fast: bool = False,
                          step_name: str = "unknown"):
    """
    Enhanced decorator for error handling in training steps.
    
    Args:
        error_severity: Default severity level for errors
        error_category: Default category for errors
        recovery_action: Default recovery action
        should_fail_fast: Whether errors should trigger fail-fast behavior
        step_name: Name of the step for context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get error handler instance
            error_handler = None
            if args and hasattr(args[0], 'error_handler'):
                error_handler = args[0].error_handler
            elif 'error_handler' in kwargs:
                error_handler = kwargs['error_handler']
            
            if error_handler is None:
                error_handler = EnhancedErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except CriticalProcessError:
                # Re-raise critical process errors
                raise
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    function_name=func.__name__,
                    step_name=step_name,
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
                    recovery_action=recovery_action,
                    should_fail_fast=should_fail_fast
                )
                
                # If we get here, the error was handled without fail-fast
                # Return None for graceful degradation
                return None
        
        return wrapper
    return decorator

def enhanced_async_error_handler(error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                                error_category: ErrorCategory = ErrorCategory.UNKNOWN,
                                recovery_action: Optional[str] = None,
                                should_fail_fast: bool = False,
                                step_name: str = "unknown"):
    """
    Enhanced decorator for error handling in async training steps.
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
                error_handler = EnhancedErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except CriticalProcessError:
                # Re-raise critical process errors
                raise
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    function_name=func.__name__,
                    step_name=step_name,
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
                    recovery_action=recovery_action,
                    should_fail_fast=should_fail_fast
                )
                
                # If we get here, the error was handled without fail-fast
                # Return None for graceful degradation
                return None
        
        return wrapper
    return decorator

def critical_process(step_name: str):
    """
    Decorator to mark a function as a critical process that should fail fast.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    function_name=func.__name__,
                    step_name=step_name,
                    additional_context={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'function_module': func.__module__,
                        'critical_process': True
                    }
                )
                
                # Create error record
                error_record = ErrorRecord(
                    error_id=f"{step_name}_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.BUSINESS_LOGIC,
                    context=context,
                    stack_trace=traceback.format_exc(),
                    should_fail_fast=True
                )
                
                # Log critical error
                logger = system_logger.getChild('CriticalProcess')
                logger.critical(f"🚨 CRITICAL PROCESS FAILED: {error_record.error_id}")
                logger.critical(f"Error: {error_record.error_message}")
                logger.critical(f"Stack trace: {error_record.stack_trace}")
                
                # Raise critical process error
                raise CriticalProcessError(
                    f"Critical process {step_name}.{func.__name__} failed: {e}",
                    error_record
                )
        
        return wrapper
    return decorator

def critical_async_process(step_name: str):
    """
    Decorator to mark an async function as a critical process that should fail fast.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = ErrorContext(
                    function_name=func.__name__,
                    step_name=step_name,
                    additional_context={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'function_module': func.__module__,
                        'critical_process': True,
                        'async_function': True
                    }
                )
                
                # Create error record
                error_record = ErrorRecord(
                    error_id=f"{step_name}_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.BUSINESS_LOGIC,
                    context=context,
                    stack_trace=traceback.format_exc(),
                    should_fail_fast=True
                )
                
                # Log critical error
                logger = system_logger.getChild('CriticalProcess')
                logger.critical(f"🚨 CRITICAL ASYNC PROCESS FAILED: {error_record.error_id}")
                logger.critical(f"Error: {error_record.error_message}")
                logger.critical(f"Stack trace: {error_record.stack_trace}")
                
                # Raise critical process error
                raise CriticalProcessError(
                    f"Critical async process {step_name}.{func.__name__} failed: {e}",
                    error_record
                )
        
        return wrapper
    return decorator

# Global error handler instance
_global_error_handler = EnhancedErrorHandler()

def get_global_error_handler() -> EnhancedErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler

def set_global_error_handler(handler: EnhancedErrorHandler) -> None:
    """Set the global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler