#!/usr/bin/env python3
"""
Comprehensive Error Handling System for NAS Components

This module provides a unified error handling system with specific exception types,
proper error logging, and circuit breaker patterns for critical operations.
"""

import logging
import traceback
import functools
import time
from typing import Any, Dict, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import threading
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization."""
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"
    COMPUTATION = "computation"
    MEMORY = "memory"
    THREADING = "threading"
    SERIALIZATION = "serialization"
    HARDWARE = "hardware"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    component: str
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class NASBaseException(Exception):
    """Base exception for all NAS-related errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.original_exception = original_exception
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context.__dict__ if self.context else None,
            'original_exception': str(self.original_exception) if self.original_exception else None,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc()
        }


class NASValidationError(NASBaseException):
    """Validation-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.HIGH, context, original_exception)


class NASConfigurationError(NASBaseException):
    """Configuration-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, context, original_exception)


class NASResourceError(NASBaseException):
    """Resource-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL, context, original_exception)


class NASMemoryError(NASBaseException):
    """Memory-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.MEMORY, ErrorSeverity.CRITICAL, context, original_exception)


class NASThreadingError(NASBaseException):
    """Threading-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.THREADING, ErrorSeverity.CRITICAL, context, original_exception)


class NASSerializationError(NASBaseException):
    """Serialization-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.SERIALIZATION, ErrorSeverity.HIGH, context, original_exception)


class NASHardwareError(NASBaseException):
    """Hardware-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.HARDWARE, ErrorSeverity.HIGH, context, original_exception)


class NASComputationError(NASBaseException):
    """Computation-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, original_exception: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM, context, original_exception)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    name: str = "default"


class CircuitBreaker:
    """Circuit breaker implementation for critical operations."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise NASResourceError(
                        f"Circuit breaker {self.config.name} is OPEN",
                        ErrorContext("circuit_breaker", "error_handling")
                    )
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except self.config.expected_exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = "OPEN"
                
                raise NASResourceError(
                    f"Circuit breaker {self.config.name} failure: {str(e)}",
                    ErrorContext("circuit_breaker", "error_handling"),
                    e
                )


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True,
        log_level: int = logging.ERROR
    ) -> Optional[NASBaseException]:
        """Handle an error with proper logging and categorization."""
        try:
            # Convert to NAS exception if needed
            if not isinstance(error, NASBaseException):
                nas_error = self._convert_to_nas_error(error, context)
            else:
                nas_error = error
            
            # Log the error
            self._log_error(nas_error, log_level)
            
            # Update error counts
            self._update_error_counts(nas_error)
            
            # Check for circuit breaker triggers
            self._check_circuit_breakers(nas_error)
            
            if reraise:
                raise nas_error
            
            return nas_error
            
        except Exception as logging_error:
            # Fallback logging if error handling fails
            self.logger.critical(f"Error in error handling: {logging_error}")
            if reraise:
                raise
    
    def _convert_to_nas_error(self, error: Exception, context: Optional[ErrorContext]) -> NASBaseException:
        """Convert generic exception to NAS exception."""
        error_type = type(error).__name__
        message = str(error)
        
        if "memory" in error_type.lower() or "MemoryError" in error_type:
            return NASMemoryError(message, context, error)
        elif "validation" in error_type.lower() or "ValueError" in error_type:
            return NASValidationError(message, context, error)
        elif "config" in error_type.lower() or "ConfigurationError" in error_type:
            return NASConfigurationError(message, context, error)
        elif "thread" in error_type.lower() or "ThreadingError" in error_type:
            return NASThreadingError(message, context, error)
        elif "serialization" in error_type.lower() or "pickle" in error_type.lower():
            return NASSerializationError(message, context, error)
        elif "hardware" in error_type.lower() or "gpu" in error_type.lower():
            return NASHardwareError(message, context, error)
        else:
            return NASComputationError(message, context, error)
    
    def _log_error(self, error: NASBaseException, log_level: int):
        """Log error with appropriate level and context."""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {error.message}", extra=error_dict)
    
    def _update_error_counts(self, error: NASBaseException):
        """Update error counts for monitoring."""
        with self._lock:
            key = f"{error.category.value}_{error.severity.value}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def _check_circuit_breakers(self, error: NASBaseException):
        """Check if error should trigger circuit breakers."""
        # Implementation for circuit breaker triggers
        pass
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig(name=name)
            self.circuit_breakers[name] = CircuitBreaker(config)
        return self.circuit_breakers[name]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            return {
                'error_counts': self.error_counts.copy(),
                'circuit_breakers': {
                    name: {
                        'state': cb.state,
                        'failure_count': cb.failure_count,
                        'last_failure_time': cb.last_failure_time
                    }
                    for name, cb in self.circuit_breakers.items()
                }
            }


# Global error handler instance
_global_error_handler = ErrorHandler()


def handle_errors(
    context: Optional[ErrorContext] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR
):
    """Decorator for error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return _global_error_handler.handle_error(e, context, reraise, log_level)
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, component: str, **kwargs):
    """Context manager for error handling with context."""
    context = ErrorContext(operation=operation, component=component, additional_data=kwargs)
    try:
        yield context
    except Exception as e:
        _global_error_handler.handle_error(e, context, reraise=True)


def safe_execute(
    func: Callable,
    *args,
    context: Optional[ErrorContext] = None,
    default_return: Any = None,
    reraise: bool = False,
    **kwargs
) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        _global_error_handler.handle_error(e, context, reraise=reraise)
        return default_return


def validate_not_none(value: Any, name: str, context: Optional[ErrorContext] = None) -> Any:
    """Validate that value is not None."""
    if value is None:
        raise NASValidationError(f"{name} cannot be None", context)
    return value


def validate_positive(value: Union[int, float], name: str, context: Optional[ErrorContext] = None) -> Union[int, float]:
    """Validate that value is positive."""
    if value <= 0:
        raise NASValidationError(f"{name} must be positive, got {value}", context)
    return value


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str,
    context: Optional[ErrorContext] = None
) -> Union[int, float]:
    """Validate that value is within range."""
    if not (min_val <= value <= max_val):
        raise NASValidationError(f"{name} must be between {min_val} and {max_val}, got {value}", context)
    return value


def validate_list_not_empty(value: List[Any], name: str, context: Optional[ErrorContext] = None) -> List[Any]:
    """Validate that list is not empty."""
    if not value:
        raise NASValidationError(f"{name} cannot be empty", context)
    return value


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


# Export main classes and functions
__all__ = [
    'ErrorSeverity',
    'ErrorCategory', 
    'ErrorContext',
    'NASBaseException',
    'NASValidationError',
    'NASConfigurationError',
    'NASResourceError',
    'NASMemoryError',
    'NASThreadingError',
    'NASSerializationError',
    'NASHardwareError',
    'NASComputationError',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'ErrorHandler',
    'handle_errors',
    'error_context',
    'safe_execute',
    'validate_not_none',
    'validate_positive',
    'validate_range',
    'validate_list_not_empty',
    'get_error_handler'
]