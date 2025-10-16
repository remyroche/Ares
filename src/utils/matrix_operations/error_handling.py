"""
Error Handling and Recovery - Unified Implementation

This module provides comprehensive error handling and recovery mechanisms
for matrix operations with backwards compatibility.
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import deque
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

# Comprehensive Error Handling Framework
class OptimizationError(Exception):
    """Base exception for optimization-related errors."""
    def __init__(self, message: str, operation: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.operation = operation
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'operation': self.operation,
            'details': self.details,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc()
        }

class GPUError(OptimizationError):
    """GPU-related errors."""
    pass

class MemoryError(OptimizationError):
    """Memory-related errors."""
    pass

class MatrixOperationError(OptimizationError):
    """Matrix operation errors."""
    pass

class DataProcessingError(OptimizationError):
    """Data processing errors."""
    pass

class ConfigurationError(OptimizationError):
    """Configuration-related errors."""
    pass

class DataValidationError(OptimizationError):
    """Data validation errors."""
    pass

@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    fallback_used: bool
    recovery_method: str
    execution_time: float
    error_details: Dict[str, Any] = field(default_factory=dict)

class ErrorHandler:
    """Comprehensive error handler with recovery mechanisms."""

    def __init__(self, enable_logging: bool = True, enable_recovery: bool = True):
        self.enable_logging = enable_logging
        self.enable_recovery = enable_recovery
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_strategies = self._init_recovery_strategies()
        self.logger = logging.getLogger(f"{__name__}.ErrorHandler")

    def _init_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            'gpu_memory_error': self._recover_gpu_memory_error,
            'cpu_memory_error': self._recover_cpu_memory_error,
            'matrix_singular_error': self._recover_matrix_singular_error,
            'file_io_error': self._recover_file_io_error,
            'network_error': self._recover_network_error,
            'timeout_error': self._recover_timeout_error
        }

    def handle_error(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> ErrorRecoveryResult:
        """Handle an error with appropriate recovery mechanism."""
        error_info = {
            'error_type': error.__class__.__name__,
            'error_message': str(error),
            'operation': operation,
            'context': context or {},
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }

        # Log error if enabled
        if self.enable_logging:
            self.logger.error(f"🚨 {operation} failed: {error}")
            self.error_history.append(error_info)

        # Attempt recovery if enabled
        if self.enable_recovery:
            recovery_result = self._attempt_recovery(error, operation, context)
            if recovery_result.success:
                self.logger.info(f"✅ Recovered from {operation} error using {recovery_result.recovery_method}")
                return recovery_result

        # Return failure result
        return ErrorRecoveryResult(
            success=False,
            fallback_used=False,
            recovery_method='none',
            execution_time=0.0,
            error_details=error_info
        )

    def _attempt_recovery(self, error: Exception, operation: str, context: Dict[str, Any]) -> ErrorRecoveryResult:
        """Attempt to recover from an error."""
        error_type = self._classify_error(error)

        if error_type in self.recovery_strategies:
            try:
                start_time = time.time()
                result = self.recovery_strategies[error_type](error, operation, context)
                execution_time = time.time() - start_time

                if result:
                    return ErrorRecoveryResult(
                        success=True,
                        fallback_used=True,
                        recovery_method=error_type,
                        execution_time=execution_time
                    )
            except Exception as recovery_error:
                self.logger.warning(f"Recovery attempt failed: {recovery_error}")

        return ErrorRecoveryResult(
            success=False,
            fallback_used=False,
            recovery_method='failed',
            execution_time=0.0
        )

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate recovery strategy."""
        error_str = str(error).lower()
        error_type = error.__class__.__name__.lower()

        # GPU-related errors
        if any(keyword in error_str for keyword in ['cuda', 'gpu', 'mps', 'memory']):
            return 'gpu_memory_error'

        # CPU memory errors
        if any(keyword in error_str for keyword in ['memory', 'allocation', 'out of memory']):
            return 'cpu_memory_error'

        # Matrix operation errors
        if any(keyword in error_str for keyword in ['singular', 'not positive definite', 'linear dependence']):
            return 'matrix_singular_error'

        # File I/O errors
        if any(keyword in error_type for keyword in ['file', 'io', 'permission']):
            return 'file_io_error'

        # Network errors
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network']):
            return 'network_error'

        # Timeout errors
        if 'timeout' in error_str:
            return 'timeout_error'

        return 'unknown_error'

    def _recover_gpu_memory_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from GPU memory errors."""
        try:
            # Clear GPU caches
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass

            # Try with smaller batch size if available
            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                self.logger.info(f"Reduced batch size to {context['batch_size']} for recovery")

            return True
        except Exception:
            return False

    def _recover_cpu_memory_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from CPU memory errors."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Clear any large caches
            if hasattr(context, 'clear_cache'):
                context['clear_cache']()

            return True
        except Exception:
            return False

    def _recover_matrix_singular_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from matrix singularity errors."""
        try:
            # Add regularization if matrix operation
            if 'matrix' in context:
                matrix = context['matrix']
                # Add small diagonal regularization
                if hasattr(matrix, 'shape') and len(matrix.shape) == 2:
                    try:
                        import numpy as np
                        regularization = np.eye(matrix.shape[0]) * 1e-8
                        context['matrix'] = matrix + regularization
                        return True
                    except ImportError:
                        return False
            return False
        except Exception:
            return False

    def _recover_file_io_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from file I/O errors."""
        try:
            # Try alternative file paths or formats
            if 'filepath' in context:
                filepath = context['filepath']

                # Try with .tmp extension first
                if not filepath.endswith('.tmp'):
                    context['filepath'] = filepath + '.tmp'
                    return True

                # Try in temp directory
                import tempfile
                import os
                temp_dir = tempfile.gettempdir()
                filename = os.path.basename(filepath)
                context['filepath'] = os.path.join(temp_dir, filename)
                return True

            return False
        except Exception:
            return False

    def _recover_network_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from network errors."""
        try:
            # Implement exponential backoff
            if 'retry_count' not in context:
                context['retry_count'] = 0

            context['retry_count'] += 1
            if context['retry_count'] <= 3:
                # Wait with exponential backoff
                wait_time = 2 ** context['retry_count']
                time.sleep(wait_time)
                return True

            return False
        except Exception:
            return False

    def _recover_timeout_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors."""
        try:
            # Increase timeout or reduce operation complexity
            if 'timeout' in context:
                context['timeout'] *= 1.5  # Increase timeout by 50%
                return True

            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                return True

            return False
        except Exception:
            return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {'total_errors': 0}

        total_errors = len(self.error_history)
        error_types = {}

        for error in self.error_history:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Calculate error rate over time
        recent_errors = [e for e in self.error_history if time.time() - e['timestamp'] < 3600]  # Last hour
        error_rate_per_hour = len(recent_errors)

        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'error_rate_per_hour': error_rate_per_hour,
            'most_common_error': max(error_types, key=error_types.get) if error_types else 'none',
            'recovery_success_rate': self._calculate_recovery_rate()
        }

    def _calculate_recovery_rate(self) -> float:
        """Calculate recovery success rate."""
        if not self.error_history:
            return 0.0

        recovery_attempts = [e for e in self.error_history if 'recovery_attempted' in e]
        successful_recoveries = [e for e in recovery_attempts if e.get('recovery_success', False)]

        return len(successful_recoveries) / len(recovery_attempts) if recovery_attempts else 0.0

# Error handling decorators
def with_error_handling(operation_name: str = None, enable_recovery: bool = True,
                       log_errors: bool = True, reraise: bool = True):
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get operation name
            op_name = operation_name or func.__name__

            # Create error handler
            error_handler = ErrorHandler(enable_logging=log_errors, enable_recovery=enable_recovery)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the error
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'module': func.__module__
                }

                recovery_result = error_handler.handle_error(e, op_name, context)

                if recovery_result.success:
                    # Try to re-execute with recovery context
                    try:
                        # Update kwargs with recovery context
                        recovery_kwargs = kwargs.copy()
                        recovery_kwargs.update(context)
                        return func(*args, **recovery_kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry after recovery failed: {retry_error}")

                if reraise:
                    if isinstance(e, OptimizationError):
                        raise
                    else:
                        # Wrap in appropriate error type
                        raise OptimizationError(
                            f"{op_name} failed: {str(e)}",
                            operation=op_name,
                            details={'original_error': str(e), 'context': context}
                        ) from e
                else:
                    logger.warning(f"Error in {op_name} suppressed: {e}")
                    return None

        return wrapper
    return decorator

def with_gpu_fallback(operation_name: str = None):
    """Decorator that provides GPU fallback to CPU."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['cuda', 'gpu', 'mps', 'memory']):
                    logger.warning(f"GPU operation failed, falling back to CPU: {e}")

                    # Try with CPU fallback
                    cpu_kwargs = kwargs.copy()
                    cpu_kwargs['use_gpu'] = False
                    cpu_kwargs['device'] = 'cpu'

                    try:
                        return func(*args, **cpu_kwargs)
                    except Exception as cpu_error:
                        logger.error(f"CPU fallback also failed: {cpu_error}")
                        raise OptimizationError(
                            f"Both GPU and CPU operations failed for {operation_name or func.__name__}",
                            operation=operation_name or func.__name__,
                            details={'gpu_error': str(e), 'cpu_error': str(cpu_error)}
                        ) from cpu_error
                else:
                    # Not a GPU error, re-raise
                    raise

        return wrapper
    return decorator

def with_memory_optimization(operation_name: str = None, max_retries: int = 3):
    """Decorator that optimizes memory usage and handles memory errors."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            last_exception = None

            for attempt in range(max_retries):
                try:
                    # Memory cleanup before execution
                    if attempt > 0:
                        gc.collect()

                        # Clear caches if available
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                        except ImportError:
                            pass

                        # Reduce batch size if specified
                        if 'batch_size' in kwargs:
                            kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                            logger.info(f"Reduced batch size to {kwargs['batch_size']} for retry")

                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    if any(keyword in error_str for keyword in ['memory', 'allocation', 'out of memory']):
                        if attempt < max_retries - 1:
                            logger.warning(f"Memory error in {op_name} (attempt {attempt + 1}/{max_retries}): {e}")
                            continue
                        else:
                            logger.error(f"Memory error in {op_name} persisted after {max_retries} attempts: {e}")
                    else:
                        # Not a memory error, don't retry
                        break

            # If we get here, all retries failed or it wasn't a memory error
            raise OptimizationError(
                f"{op_name} failed after {max_retries} attempts",
                operation=op_name,
                details={'last_error': str(last_exception), 'attempts': max_retries}
            ) from last_exception

        return wrapper
    return decorator

# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def safe_operation(operation_name: str, default_value: Any = None, log_level: str = "warning"):
    """
    Decorator for safe operations with error handling and fallback to default values.

    Args:
        operation_name: Name of the operation for logging
        default_value: Default value to return if operation fails
        log_level: Logging level for errors ("debug", "info", "warning", "error")

    Returns:
        Decorated function that handles errors gracefully
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                logger.log(
                    getattr(logging, log_level.upper(), logging.WARNING),
                    f"🚨 Operation '{operation_name}' failed: {e}"
                )

                # Return default value if provided
                return default_value

        return wrapper
    return decorator
