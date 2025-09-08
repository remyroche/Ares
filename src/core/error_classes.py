"""
Comprehensive error classes and handlers for the Ares project.
Integrates with existing error infrastructure.
"""

from typing import Any, Dict, Optional, Callable
import logging

# Import existing error infrastructure
import datetime

try:
    from .errors.base import AppError as BaseAppError, ErrorCode

    # Use existing AppError as base
    AppError = BaseAppError
except ImportError:
    # Fallback if base errors not available
    class AppError(Exception):
        """Base application error."""
        def __init__(self, message: str, status_code: int = 500, **kwargs):
            super().__init__(message)
            self.message = message
            self.status_code = status_code
            self.details = kwargs
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'error': self.__class__.__name__,
                'message': self.message,
                'status_code': self.status_code,
                'details': self.details
            }

class ValidationError(AppError):
    """Validation error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=400, **kwargs)

class AuthenticationError(AppError):
    """Authentication error."""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)

class AuthorizationError(AppError):
    """Authorization error."""
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, status_code=403, **kwargs)

class ServiceUnavailableError(AppError):
    """Service unavailable error."""
    def __init__(self, message: str = "Service unavailable", **kwargs):
        super().__init__(message, status_code=503, **kwargs)

class AppTimeoutError(AppError):
    """Application timeout error."""
    def __init__(self, message: str = "Operation timed out", **kwargs):
        super().__init__(message, status_code=408, **kwargs)

class RetryableError(AppError):
    """Retryable error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=429, **kwargs)

class NonRetryableError(AppError):
    """Non-retryable error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, status_code=400, **kwargs)

# Error recovery strategies
class ErrorRecoveryStrategies:
    """Error recovery strategies."""
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_retries: int = 3) -> Any:
        """Retry function with exponential backoff."""
        import time
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return None

# Error constants and mappings
DATA_OPERATION_ERRORS = {
    'read_error': 'Failed to read data',
    'write_error': 'Failed to write data',
    'validation_error': 'Data validation failed',
    'format_error': 'Invalid data format'
}

EXCEPTION_TYPES = {
    'validation': ValidationError,
    'authentication': AuthenticationError,
    'authorization': AuthorizationError,
    'service_unavailable': ServiceUnavailableError,
    'timeout': AppTimeoutError,
    'retryable': RetryableError,
    'non_retryable': NonRetryableError
}

# Error handler functions
def handles_errors(*args, **kwargs):
    """Error handling decorator."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def core_handles_errors(*args, **kwargs):
    """Core error handling decorator."""
    return handles_errors(*args, **kwargs)

def _handles_errors(*args, **kwargs):
    """Private error handling decorator."""
    return handles_errors(*args, **kwargs)

# Error context and mapping functions
def error_context(error: Exception) -> Dict[str, Any]:
    """Get error context."""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'module': getattr(error, '__module__', 'unknown')
    }

def error_mapper(error: Exception) -> str:
    """Map error to standardized message."""
    error_type = type(error).__name__
    return DATA_OPERATION_ERRORS.get(error_type.lower(), str(error))

def categorize_errors(errors: list) -> Dict[str, list]:
    """Categorize errors by type."""
    categories = {}
    for error in errors:
        error_type = type(error).__name__
        if error_type not in categories:
            categories[error_type] = []
        categories[error_type].append(error)
    return categories

# Specific error functions
def initialization_error(message: str = "Initialization failed") -> AppError:
    """Create initialization error."""
    return AppError(message, status_code=500)

def execution_error(message: str = "Execution failed") -> AppError:
    """Create execution error."""
    return AppError(message, status_code=500)

def validation_error(message: str = "Validation failed") -> ValidationError:
    """Create validation error."""
    return ValidationError(message)

def artifact_error(message: str = "Artifact operation failed") -> AppError:
    """Create artifact error."""
    return AppError(message, status_code=500)

def cleanup_error(message: str = "Cleanup failed") -> AppError:
    """Create cleanup error."""
    return AppError(message, status_code=500)

def handler_error(message: str = "Handler error") -> AppError:
    """Create handler error."""
    return AppError(message, status_code=500)

def mapping_error(message: str = "Mapping error") -> AppError:
    """Create mapping error."""
    return AppError(message, status_code=500)

def metrics_error(message: str = "Metrics error") -> AppError:
    """Create metrics error."""
    return AppError(message, status_code=500)

def prob_error(message: str = "Probability error") -> AppError:
    """Create probability error."""
    return AppError(message, status_code=500)

def recovery_error(message: str = "Recovery error") -> AppError:
    """Create recovery error."""
    return AppError(message, status_code=500)

def retry_error(message: str = "Retry error") -> AppError:
    """Create retry error."""
    return AppError(message, status_code=500)

def save_error(message: str = "Save error") -> AppError:
    """Create save error."""
    return AppError(message, status_code=500)

def fix_error(message: str = "Fix error") -> AppError:
    """Create fix error."""
    return AppError(message, status_code=500)

# Error boundary and exception handling
def error_boundary(func: Callable) -> Callable:
    """Error boundary decorator."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error boundary caught: {e}")
            return None
    return wrapper

def return_exceptions(func: Callable) -> Callable:
    """Return exceptions instead of raising them."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
    return wrapper

def expected_exception(exception_type: type) -> bool:
    """Check if exception is expected."""
    return issubclass(exception_type, AppError)

# Cache and exception handling
def cache_exceptions(func: Callable) -> Callable:
    """Cache exceptions decorator."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Cache the exception for later analysis
            logger = logging.getLogger(func.__module__)
            logger.warning(f"Cached exception in {func.__name__}: {e}")
            raise
    return wrapper

# Standardized error handler
def standardized_error_handler(error: Exception) -> Dict[str, Any]:
    """Standardized error handler."""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'handled': True
    }

# Export all error classes and functions
__all__ = [
    # Error classes
    'AppError', 'ValidationError', 'AuthenticationError', 'AuthorizationError',
    'ServiceUnavailableError', 'AppTimeoutError', 'RetryableError', 'NonRetryableError',
    
    # Error handlers
    'EnhancedErrorHandler', 'ErrorRecoveryStrategies',
    
    # Error constants
    'DATA_OPERATION_ERRORS', 'EXCEPTION_TYPES',
    
    # Error functions
    'handles_errors', 'core_handles_errors', '_handles_errors',
    'error_context', 'error_mapper', 'categorize_errors',
    'initialization_error', 'execution_error', 'validation_error',
    'artifact_error', 'cleanup_error', 'handler_error', 'mapping_error',
    'metrics_error', 'prob_error', 'recovery_error', 'retry_error',
    'save_error', 'fix_error', 'error_boundary', 'return_exceptions',
    'expected_exception', 'cache_exceptions', 'standardized_error_handler'
]
