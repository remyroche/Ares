"""
Monitoring Utilities

This module provides comprehensive monitoring and tracking utilities
to replace the functionality from step02_5_utilities.py.
"""

import time
import functools
import inspect
import logging
from typing import Any, Dict, Callable, List, Optional
from collections import deque
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

@dataclass
class FunctionCall:
    """Represents a function call with metadata."""
    function_name: str
    execution_time: float
    timestamp: float
    args_count: int = 0
    kwargs_count: int = 0
    success: bool = True
    error_message: Optional[str] = None
    memory_usage: float = 0.0

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
        self._lock = threading.Lock()

    def track_function_call(self, func_name: str, execution_time: float, success: bool,
                          args_count: int = 0, kwargs_count: int = 0, error: Optional[str] = None,
                          memory_usage: float = 0.0):
        """Track a function call with memory-efficient storage."""
        try:
            with self._lock:
                self.call_count += 1
                
                call_record = FunctionCall(
                    function_name=func_name,
                    execution_time=execution_time,
                    timestamp=time.time(),
                    args_count=args_count,
                    kwargs_count=kwargs_count,
                    success=success,
                    error_message=error,
                    memory_usage=memory_usage
                )
                
                self.call_history.append(call_record)
                
                if success:
                    self.success_count += 1
                else:
                    self.error_count += 1
                
                # Update performance metrics
                if func_name not in self.performance_metrics:
                    self.performance_metrics[func_name] = {
                        'call_count': 0,
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'error_count': 0,
                        'success_count': 0
                    }
                
                metrics = self.performance_metrics[func_name]
                metrics['call_count'] += 1
                metrics['total_time'] += execution_time
                metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
                metrics['min_time'] = min(metrics['min_time'], execution_time)
                metrics['max_time'] = max(metrics['max_time'], execution_time)
                
                if success:
                    metrics['success_count'] += 1
                else:
                    metrics['error_count'] += 1
                
                logger.debug(f"Tracked call to {func_name}: {execution_time:.4f}s")
                
        except Exception as e:
            logger.error(f"Error tracking function call: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            uptime = time.time() - self.start_time

            return {
                'total_calls': self.call_count,
                'successful_calls': self.success_count,
                'error_calls': self.error_count,
                'success_rate': self.success_count / max(self.call_count, 1),
                'uptime_seconds': uptime,
                'calls_per_second': self.call_count / max(uptime, 1),
                'function_metrics': dict(self.performance_metrics),
                'recent_calls': list(self.call_history)[-10:] if self.call_history else []
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary compatible with PerformanceMonitor interface."""
        with self._lock:
            # Calculate total execution time from all function metrics
            total_execution_time = sum(
                metrics.get('total_time', 0)
                for metrics in self.performance_metrics.values()
            )

            # Calculate memory usage (approximate from recent calls)
            total_memory_usage = sum(
                call.memory_usage for call in self.call_history
                if call.memory_usage > 0
            )

            return {
                'total_execution_time': total_execution_time,
                'total_memory_delta_mb': total_memory_usage,
                'successful_operations': self.success_count,
                'failed_operations': self.error_count,
                'success_rate': self.success_count / max(self.call_count, 1),
                'operations': list(self.performance_metrics.keys())
            }

    def get_function_metrics(self, function_name: str) -> Dict[str, Any]:
        """Get metrics for a specific function."""
        with self._lock:
            return self.performance_metrics.get(function_name, {})

    def clear_history(self) -> None:
        """Clear monitoring history."""
        with self._lock:
            self.call_history.clear()
            self.performance_metrics.clear()
            self.call_count = 0
            self.error_count = 0
            self.success_count = 0
            self.start_time = time.time()

class FunctionTracker:
    """Function call tracking decorator."""
    
    def __init__(self, monitor: Optional[UnifiedPerformanceMonitor] = None):
        self.monitor = monitor or global_monitor
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                
                self.monitor.track_function_call(
                    func_name=func.__name__,
                    execution_time=execution_time,
                    success=success,
                    args_count=len(args),
                    kwargs_count=len(kwargs),
                    error=error_message,
                    memory_usage=memory_usage
                )
        
        return wrapper
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

class LoggingPatterns:
    """Standardized logging patterns."""
    
    @staticmethod
    def log_function_start(func_name: str, args: tuple = (), kwargs: dict = None) -> None:
        """Log function start."""
        kwargs = kwargs or {}
        logger.debug(f"🚀 Starting {func_name} with {len(args)} args, {len(kwargs)} kwargs")
    
    @staticmethod
    def log_function_end(func_name: str, execution_time: float, success: bool = True) -> None:
        """Log function end."""
        status = "✅" if success else "❌"
        logger.debug(f"{status} Completed {func_name} in {execution_time:.4f}s")
    
    @staticmethod
    def log_function_error(func_name: str, error: Exception) -> None:
        """Log function error."""
        logger.error(f"❌ Error in {func_name}: {error}")
    
    @staticmethod
    def log_performance_metrics(metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        logger.info(f"📊 Performance Summary: {metrics['total_calls']} calls, "
                   f"{metrics['success_rate']:.1%} success rate, "
                   f"{metrics['calls_per_second']:.2f} calls/sec")

# Global monitor instance
global_monitor = UnifiedPerformanceMonitor()

# Convenience decorators
def track_function(monitor: Optional[UnifiedPerformanceMonitor] = None) -> Callable:
    """Decorator to track function calls."""
    tracker = FunctionTracker(monitor)
    return tracker

def monitor_function_calls(func: Callable) -> Callable:
    """Decorator to monitor function calls using global monitor."""
    return track_function(global_monitor)(func)

# Legacy compatibility functions
def function_tracker(func: Callable) -> Callable:
    """Legacy function tracker decorator."""
    return monitor_function_calls(func)

def logging_patterns() -> LoggingPatterns:
    """Get logging patterns instance."""
    return LoggingPatterns()

# Global tracker for backward compatibility
global_tracker = global_monitor

__all__ = [
    'FunctionCall',
    'UnifiedPerformanceMonitor',
    'FunctionTracker',
    'LoggingPatterns',
    'global_monitor',
    'global_tracker',
    'track_function',
    'monitor_function_calls',
    'function_tracker',
    'logging_patterns'
]