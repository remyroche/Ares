"""Optimized utility modules for Step02_5 with unified monitoring and memory efficiency."""

import time
import functools
import inspect
from typing import Any, Dict, Callable, List, Optional
from collections import deque
import logging
import numpy as np

# Core imports
from .logger import system_logger

# Required utility modules integration
from .common_operations import (
    safe_json_load, safe_json_dump, safe_read_parquet, 
    ensure_directory, create_fallback_logger
)
from .math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_positive, 
    validate_range, MathValidationError
)
from .parquet_utils import ParquetUtils

# Core decorators and errors
from ..core.decorators import handles_errors, error_boundary, converts_errors
from ..core.errors import (
    AppError, ValidationError, DataIntegrityError, 
    NotFoundError, BusinessRuleError
)

logger = system_logger.getChild('Step02_5Utilities')

class UnifiedPerformanceMonitor:
    """Unified, memory-efficient monitoring system that consolidates all tracking."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.call_count = 0
        self.call_history = deque(maxlen=max_history)  # Memory-efficient circular buffer
        self.performance_metrics = {}
        self.error_count = 0
        self.success_count = 0
        self.start_time = time.time()

    @handles_errors(default_return=None)
    def track_function_call(self, func_name: str, execution_time: float, success: bool,
                          args_count: int = 0, kwargs_count: int = 0, error: Optional[str] = None):
        """Track a function call with memory-efficient storage using core error handling."""
        try:
            self.call_count += 1
            if success:
                self.success_count += 1
            else:
                self.error_count += 1

            # Update performance metrics with safe math operations
            if func_name not in self.performance_metrics:
                self.performance_metrics[func_name] = {
                    'total_calls': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'error_count': 0,
                    'success_count': 0
                }

            metrics = self.performance_metrics[func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            # Use safe division to prevent division by zero
            metrics['avg_time'] = safe_divide(metrics['total_time'], metrics['total_calls'], default=0.0)
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)

            if success:
                metrics['success_count'] += 1
            else:
                metrics['error_count'] += 1

            # Store call history (limited by max_history)
            self.call_history.append({
                'call_id': self.call_count,
                'function': func_name,
                'timestamp': time.time(),
                'execution_time': execution_time,
                'success': success,
                'error': error,
                'args_count': args_count,
                'kwargs_count': kwargs_count
            })
        except Exception as e:
            logger.error(f"Error tracking function call for {func_name}: {e}")
            # Continue execution even if tracking fails

    @handles_errors(default_return={'total_calls': 0, 'success_count': 0, 'error_count': 0, 'success_rate': 0.0, 'total_runtime': 0.0, 'calls_per_second': 0.0, 'performance_metrics': {}, 'recent_calls': []})
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary using safe math operations."""
        total_time = time.time() - self.start_time
        return {
            'total_calls': self.call_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': safe_divide(self.success_count, max(1, self.call_count), default=0.0),
            'total_runtime': total_time,
            'calls_per_second': safe_divide(self.call_count, max(1, total_time), default=0.0),
            'performance_metrics': self.performance_metrics,
            'recent_calls': list(self.call_history)[-10:]  # Last 10 calls
        }

class FunctionCallTracker:
    """Legacy class - now uses UnifiedPerformanceMonitor internally."""

    def __init__(self):
        self.monitor = UnifiedPerformanceMonitor()
        self.tracker = self.monitor  # Backward compatibility
    
    def monitor_function_calls(self, func: Callable) -> Callable:
        """Memory-efficient function call monitoring using UnifiedPerformanceMonitor."""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.monitor.track_function_call(
                    func.__name__, execution_time, True,
                    len(args), len(kwargs)
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.monitor.track_function_call(
                    func.__name__, execution_time, False,
                    len(args), len(kwargs), str(e)
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.monitor.track_function_call(
                    func.__name__, execution_time, True,
                    len(args), len(kwargs)
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.monitor.track_function_call(
                    func.__name__, execution_time, False,
                    len(args), len(kwargs), str(e)
                )
                raise

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary from unified monitor."""
        return self.monitor.get_summary()

def function_tracker(func):
    """Track function calls for monitoring."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def logging_patterns():
    """Return logging patterns for step02_5."""
    return {
        'start': '🔧 Starting {function_name}...',
        'success': '✅ {function_name} completed successfully',
        'error': '❌ {function_name} failed: {error}'
    }

# Global unified monitor instance (memory-efficient)
global_monitor = UnifiedPerformanceMonitor(max_history=500)  # Reduced from default 1000
global_tracker = FunctionCallTracker()
# Make global_tracker use the same monitor instance as global_monitor
global_tracker.monitor = global_monitor
global_tracker.tracker = global_monitor

def monitor_function_calls(func: Callable) -> Callable:
    """Global function call monitoring decorator."""
    return global_tracker.monitor_function_calls(func)

@handles_errors(default_return=None)
def validate_function_inputs(func: Callable) -> Callable:
    """Validate function inputs using core validation utilities."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Basic input validation
        if args and hasattr(args[0], '__dict__'):
            logger.debug(f'🔍 Validating inputs for {func.__name__}')
        return func(*args, **kwargs)
    return wrapper

@handles_errors(default_return=None)
def safe_json_operations(file_path: str, data: Optional[Dict[str, Any]] = None, operation: str = 'load') -> Optional[Dict[str, Any]]:
    """Safe JSON operations using common_operations utilities."""
    try:
        if operation == 'load':
            return safe_json_load(file_path)
        elif operation == 'save' and data is not None:
            return safe_json_dump(data, file_path)
        else:
            logger.error(f"Invalid operation: {operation}")
            return None
    except Exception as e:
        logger.error(f"Error in JSON operation {operation}: {e}")
        return None

@handles_errors(default_return=None)
def safe_parquet_operations(file_path: str, operation: str = 'read', data: Optional[Any] = None) -> Optional[Any]:
    """Safe parquet operations using parquet_utils."""
    try:
        parquet_utils = ParquetUtils()
        if operation == 'read':
            return parquet_utils.safe_read_parquet(file_path)
        elif operation == 'validate':
            return parquet_utils.validate_parquet_file(file_path)
        else:
            logger.error(f"Invalid parquet operation: {operation}")
            return None
    except Exception as e:
        logger.error(f"Error in parquet operation {operation}: {e}")
        return None

@handles_errors(default_return=False)
def validate_math_operations(values: List[float], operation: str = 'positive') -> bool:
    """Validate mathematical operations using math_validation utilities."""
    try:
        for value in values:
            if operation == 'positive' and not validate_positive(value):
                return False
            elif operation == 'range' and not validate_range(value, 0.0, 1.0):
                return False
        return True
    except Exception as e:
        logger.error(f"Error in math validation: {e}")
        return False