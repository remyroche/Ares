"""
Comprehensive Error Handling System for Data Qualification Pipeline

This module provides centralized error handling, recovery mechanisms, and fallback strategies
for all data qualification steps, ensuring robust operation even when utilities fail.

Key Features:
- Centralized error handling with detailed error classification
- Automatic fallback mechanisms for utility failures
- Retry strategies with exponential backoff
- Error recovery and graceful degradation
- Comprehensive error logging and monitoring
- Error analytics and reporting
- Circuit breaker pattern for failing services
"""

import time
import logging
import traceback
from typing import Dict, Any, Optional, List, Callable, Type, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import functools
import asyncio
from datetime import datetime, timedelta
import json
from pathlib import Path

# Initialize logger
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    IMPORT_ERROR = "import_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_ERROR = "data_error"
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    FILE_ERROR = "file_error"
    VALIDATION_ERROR = "validation_error"
    UTILITY_ERROR = "utility_error"
    UNKNOWN_ERROR = "unknown_error"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    step_name: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorInfo:
    """Detailed error information."""
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    recovery_strategy: RecoveryStrategy
    retry_count: int = 0
    max_retries: int = 3
    fallback_used: bool = False
    recovery_successful: bool = False
    error_message: str = ""
    stack_trace: str = ""

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 2

class CircuitBreaker:
    """Circuit breaker implementation for failing services."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logger.getChild('CircuitBreaker')
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.failure_count = 0
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class DataQualificationErrorHandler:
    """
    Centralized error handler for data qualification pipeline.
    
    Provides comprehensive error handling with automatic recovery,
    fallback mechanisms, and detailed error tracking.
    
    Example:
        >>> handler = DataQualificationErrorHandler()
        >>> result = handler.handle_utility_failure(
        ...     step_name="sr_optimization",
        ...     utility_name="ml_common",
        ...     error=ImportError("Module not found"),
        ...     fallback_func=get_legacy_utilities
        ... )
    """
    
    def __init__(self, enable_circuit_breaker: bool = True, log_errors: bool = True):
        """
        Initialize the error handler.
        
        Args:
            enable_circuit_breaker: Whether to enable circuit breaker pattern
            log_errors: Whether to log errors
        """
        self.enable_circuit_breaker = enable_circuit_breaker
        self.log_errors = log_errors
        self.logger = logger.getChild('ErrorHandler')
        self.error_history: List[ErrorInfo] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_registry: Dict[str, Callable] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        
        # Initialize default retry configurations
        self._initialize_default_retry_configs()
        
        if self.log_errors:
            self.logger.info("🚀 Data Qualification Error Handler initialized")
    
    def _initialize_default_retry_configs(self):
        """Initialize default retry configurations for different error types."""
        self.retry_configs = {
            ErrorCategory.IMPORT_ERROR.value: RetryConfig(
                max_retries=2,
                base_delay=0.5,
                max_delay=5.0,
                retry_on_exceptions=(ImportError, ModuleNotFoundError)
            ),
            ErrorCategory.NETWORK_ERROR.value: RetryConfig(
                max_retries=5,
                base_delay=1.0,
                max_delay=30.0,
                retry_on_exceptions=(ConnectionError, TimeoutError, OSError)
            ),
            ErrorCategory.MEMORY_ERROR.value: RetryConfig(
                max_retries=2,
                base_delay=2.0,
                max_delay=10.0,
                retry_on_exceptions=(MemoryError, OSError)
            ),
            ErrorCategory.FILE_ERROR.value: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=15.0,
                retry_on_exceptions=(FileNotFoundError, PermissionError, OSError)
            ),
            ErrorCategory.COMPUTATION_ERROR.value: RetryConfig(
                max_retries=2,
                base_delay=1.0,
                max_delay=10.0,
                retry_on_exceptions=(ValueError, RuntimeError, ArithmeticError)
            )
        }
    
    def classify_error(self, error: Exception, context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify error by category and severity.
        
        Args:
            error: The exception to classify
            context: Error context information
            
        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        error_type = type(error)
        error_message = str(error).lower()
        
        # Classification logic
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            category = ErrorCategory.IMPORT_ERROR
            severity = ErrorSeverity.MEDIUM
        elif isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            category = ErrorCategory.FILE_ERROR
            severity = ErrorSeverity.MEDIUM
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.NETWORK_ERROR
            severity = ErrorSeverity.HIGH
        elif isinstance(error, MemoryError):
            category = ErrorCategory.MEMORY_ERROR
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, KeyError)):
            category = ErrorCategory.VALIDATION_ERROR
            severity = ErrorSeverity.MEDIUM
        elif isinstance(error, (RuntimeError, ArithmeticError)):
            category = ErrorCategory.COMPUTATION_ERROR
            severity = ErrorSeverity.MEDIUM
        elif "data" in error_message or "dataframe" in error_message:
            category = ErrorCategory.DATA_ERROR
            severity = ErrorSeverity.MEDIUM
        elif "config" in error_message or "configuration" in error_message:
            category = ErrorCategory.CONFIGURATION_ERROR
            severity = ErrorSeverity.HIGH
        else:
            category = ErrorCategory.UNKNOWN_ERROR
            severity = ErrorSeverity.MEDIUM
        
        # Adjust severity based on context
        if context.step_name in ["sr_optimization", "hmm_regime_discovery"]:
            if severity == ErrorSeverity.MEDIUM:
                severity = ErrorSeverity.HIGH
        
        return category, severity
    
    def determine_recovery_strategy(
        self, 
        category: ErrorCategory, 
        severity: ErrorSeverity,
        retry_count: int
    ) -> RecoveryStrategy:
        """
        Determine the appropriate recovery strategy.
        
        Args:
            category: Error category
            severity: Error severity
            retry_count: Number of retries already attempted
            
        Returns:
            RecoveryStrategy to use
        """
        # Critical errors should abort
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT
        
        # Import errors should use fallback
        if category == ErrorCategory.IMPORT_ERROR:
            return RecoveryStrategy.FALLBACK
        
        # Network errors should retry with circuit breaker
        if category == ErrorCategory.NETWORK_ERROR:
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        # Memory errors should skip or abort
        if category == ErrorCategory.MEMORY_ERROR:
            return RecoveryStrategy.SKIP if retry_count < 2 else RecoveryStrategy.ABORT
        
        # Configuration errors should abort
        if category == ErrorCategory.CONFIGURATION_ERROR:
            return RecoveryStrategy.ABORT
        
        # Default to retry for other errors
        return RecoveryStrategy.RETRY
    
    def handle_utility_failure(
        self,
        step_name: str,
        utility_name: str,
        error: Exception,
        fallback_func: Optional[Callable] = None,
        context: Optional[ErrorContext] = None
    ) -> Any:
        """
        Handle utility failure with appropriate recovery strategy.
        
        Args:
            step_name: Name of the step where error occurred
            utility_name: Name of the utility that failed
            error: The exception that occurred
            fallback_func: Optional fallback function
            context: Optional error context
            
        Returns:
            Result from fallback function or raises exception
            
        Example:
            >>> handler = DataQualificationErrorHandler()
            >>> result = handler.handle_utility_failure(
            ...     step_name="sr_optimization",
            ...     utility_name="ml_common",
            ...     error=ImportError("Module not found"),
            ...     fallback_func=get_legacy_utilities
            ... )
        """
        if context is None:
            context = ErrorContext(step_name=step_name, operation=f"utility_{utility_name}")
        
        # Classify error
        category, severity = self.classify_error(error, context)
        
        # Create error info
        error_info = ErrorInfo(
            error=error,
            category=category,
            severity=severity,
            context=context,
            recovery_strategy=RecoveryStrategy.FALLBACK,  # Will be determined below
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )
        
        # Determine recovery strategy
        recovery_strategy = self.determine_recovery_strategy(category, severity, 0)
        error_info.recovery_strategy = recovery_strategy
        
        # Log error
        if self.log_errors:
            self._log_error(error_info)
        
        # Store error info
        self.error_history.append(error_info)
        
        # Execute recovery strategy
        try:
            if recovery_strategy == RecoveryStrategy.FALLBACK:
                return self._execute_fallback(utility_name, fallback_func, error_info)
            elif recovery_strategy == RecoveryStrategy.RETRY:
                return self._execute_retry(utility_name, fallback_func, error_info)
            elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._execute_circuit_breaker(utility_name, fallback_func, error_info)
            elif recovery_strategy == RecoveryStrategy.SKIP:
                return self._execute_skip(utility_name, error_info)
            else:  # ABORT
                return self._execute_abort(error_info)
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed for {utility_name}: {recovery_error}")
            raise error  # Re-raise original error
    
    def _execute_fallback(self, utility_name: str, fallback_func: Optional[Callable], error_info: ErrorInfo) -> Any:
        """Execute fallback strategy."""
        if fallback_func is None:
            fallback_func = self.fallback_registry.get(utility_name)
        
        if fallback_func is None:
            self.logger.error(f"No fallback available for {utility_name}")
            raise error_info.error
        
        try:
            self.logger.warning(f"Using fallback for {utility_name}")
            result = fallback_func()
            error_info.fallback_used = True
            error_info.recovery_successful = True
            return result
        except Exception as e:
            self.logger.error(f"Fallback failed for {utility_name}: {e}")
            raise error_info.error
    
    def _execute_retry(self, utility_name: str, fallback_func: Optional[Callable], error_info: ErrorInfo) -> Any:
        """Execute retry strategy."""
        retry_config = self.retry_configs.get(error_info.category.value)
        if retry_config is None:
            retry_config = RetryConfig()
        
        for attempt in range(retry_config.max_retries):
            try:
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt, retry_config)
                    self.logger.info(f"Retrying {utility_name} in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                
                # Try the original operation or fallback
                if fallback_func and attempt > 0:
                    result = fallback_func()
                else:
                    raise error_info.error  # Re-raise to simulate retry
                
                error_info.retry_count = attempt + 1
                error_info.recovery_successful = True
                return result
                
            except retry_config.retry_on_exceptions as e:
                if attempt == retry_config.max_retries - 1:
                    self.logger.error(f"Retry failed for {utility_name} after {retry_config.max_retries} attempts")
                    raise error_info.error
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error during retry for {utility_name}: {e}")
                raise error_info.error
    
    def _execute_circuit_breaker(self, utility_name: str, fallback_func: Optional[Callable], error_info: ErrorInfo) -> Any:
        """Execute circuit breaker strategy."""
        if not self.enable_circuit_breaker:
            return self._execute_retry(utility_name, fallback_func, error_info)
        
        # Get or create circuit breaker
        if utility_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[utility_name] = CircuitBreaker(config)
        
        circuit_breaker = self.circuit_breakers[utility_name]
        
        try:
            if fallback_func:
                result = circuit_breaker.call(fallback_func)
            else:
                raise error_info.error
            
            error_info.recovery_successful = True
            return result
        except Exception as e:
            self.logger.error(f"Circuit breaker failed for {utility_name}: {e}")
            raise error_info.error
    
    def _execute_skip(self, utility_name: str, error_info: ErrorInfo) -> Any:
        """Execute skip strategy."""
        self.logger.warning(f"Skipping {utility_name} due to error")
        error_info.recovery_successful = True
        return None
    
    def _execute_abort(self, error_info: ErrorInfo) -> Any:
        """Execute abort strategy."""
        self.logger.error(f"Aborting due to critical error: {error_info.error_message}")
        raise error_info.error
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        delay = min(
            config.base_delay * (config.exponential_base ** attempt),
            config.max_delay
        )
        
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information."""
        log_message = (
            f"Error in {error_info.context.step_name}:{error_info.context.operation} - "
            f"{error_info.category.value} ({error_info.severity.value}) - "
            f"{error_info.error_message}"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def register_fallback(self, utility_name: str, fallback_func: Callable):
        """Register a fallback function for a utility."""
        self.fallback_registry[utility_name] = fallback_func
        self.logger.info(f"Registered fallback for {utility_name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and analytics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        errors_by_category = {}
        errors_by_severity = {}
        errors_by_step = {}
        recovery_success_rate = 0
        
        for error_info in self.error_history:
            # Count by category
            category = error_info.category.value
            errors_by_category[category] = errors_by_category.get(category, 0) + 1
            
            # Count by severity
            severity = error_info.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
            
            # Count by step
            step = error_info.context.step_name
            errors_by_step[step] = errors_by_step.get(step, 0) + 1
            
            # Calculate recovery success rate
            if error_info.recovery_successful:
                recovery_success_rate += 1
        
        recovery_success_rate = recovery_success_rate / total_errors if total_errors > 0 else 0
        
        return {
            "total_errors": total_errors,
            "errors_by_category": errors_by_category,
            "errors_by_severity": errors_by_severity,
            "errors_by_step": errors_by_step,
            "recovery_success_rate": recovery_success_rate,
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        self.logger.info("Error history cleared")
    
    def export_error_report(self, file_path: str):
        """Export error report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "error_history": [
                {
                    "timestamp": error_info.context.timestamp.isoformat(),
                    "step_name": error_info.context.step_name,
                    "operation": error_info.context.operation,
                    "category": error_info.category.value,
                    "severity": error_info.severity.value,
                    "error_message": error_info.error_message,
                    "recovery_strategy": error_info.recovery_strategy.value,
                    "recovery_successful": error_info.recovery_successful,
                    "fallback_used": error_info.fallback_used,
                    "retry_count": error_info.retry_count
                }
                for error_info in self.error_history
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Error report exported to {file_path}")

# Decorators for error handling
def handle_errors(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    fallback_func: Optional[Callable] = None,
    retry_count: int = 0,
    log_level: str = "ERROR"
):
    """
    Decorator for automatic error handling.
    
    Args:
        exceptions: Tuple of exception types to catch
        fallback_func: Optional fallback function
        retry_count: Number of retries
        log_level: Log level for errors
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = DataQualificationErrorHandler()
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retry_count:
                        if fallback_func:
                            try:
                                return fallback_func(*args, **kwargs)
                            except Exception as fallback_error:
                                logger.error(f"Fallback failed: {fallback_error}")
                        raise e
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
            
            return None
        
        return wrapper
    return decorator

def with_error_recovery(
    step_name: str,
    utility_name: str,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator for error recovery with fallback.
    
    Args:
        step_name: Name of the step
        utility_name: Name of the utility
        fallback_func: Optional fallback function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = DataQualificationErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_utility_failure(
                    step_name=step_name,
                    utility_name=utility_name,
                    error=e,
                    fallback_func=fallback_func
                )
        
        return wrapper
    return decorator

@contextmanager
def error_context(step_name: str, operation: str, **kwargs):
    """
    Context manager for error handling with context.
    
    Args:
        step_name: Name of the step
        operation: Name of the operation
        **kwargs: Additional context data
    """
    context = ErrorContext(
        step_name=step_name,
        operation=operation,
        additional_data=kwargs
    )
    
    try:
        yield context
    except Exception as e:
        error_handler = DataQualificationErrorHandler()
        category, severity = error_handler.classify_error(e, context)
        
        error_info = ErrorInfo(
            error=e,
            category=category,
            severity=severity,
            context=context,
            recovery_strategy=RecoveryStrategy.ABORT,
            error_message=str(e),
            stack_trace=traceback.format_exc()
        )
        
        error_handler._log_error(error_info)
        error_handler.error_history.append(error_info)
        
        raise e

# Global error handler instance
_error_handler: Optional[DataQualificationErrorHandler] = None

def get_error_handler() -> DataQualificationErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = DataQualificationErrorHandler()
    return _error_handler

# Convenience functions
def handle_utility_failure(
    step_name: str,
    utility_name: str,
    error: Exception,
    fallback_func: Optional[Callable] = None
) -> Any:
    """Handle utility failure using global error handler."""
    return get_error_handler().handle_utility_failure(
        step_name=step_name,
        utility_name=utility_name,
        error=error,
        fallback_func=fallback_func
    )

def register_fallback(utility_name: str, fallback_func: Callable):
    """Register fallback function using global error handler."""
    get_error_handler().register_fallback(utility_name, fallback_func)

def get_error_statistics() -> Dict[str, Any]:
    """Get error statistics from global error handler."""
    return get_error_handler().get_error_statistics()

# Export main classes and functions
__all__ = [
    'DataQualificationErrorHandler',
    'ErrorInfo',
    'ErrorContext',
    'ErrorCategory',
    'ErrorSeverity',
    'RecoveryStrategy',
    'RetryConfig',
    'CircuitBreakerConfig',
    'CircuitBreaker',
    'handle_errors',
    'with_error_recovery',
    'error_context',
    'get_error_handler',
    'handle_utility_failure',
    'register_fallback',
    'get_error_statistics'
]