"""
Performance Utilities

This module provides comprehensive performance monitoring, timing, and profiling utilities
with memory management and hardware optimization.
"""

import time
import functools
import logging
import psutil
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union, Generator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

# Import memory management
try:
    from .memory_management import memory_managed, MemoryStrategy, force_cleanup
except ImportError:
    # Create dummy decorator if memory management not available
    def memory_managed(strategy=None):
        def decorator(func):
            return func
        return decorator
    def force_cleanup():
        pass

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: float
    args_count: int = 0
    kwargs_count: int = 0
    success: bool = True
    error_message: Optional[str] = None

class PerformanceMonitor:
    """Performance monitoring and profiling utility with memory management."""

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.

        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.function_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'error_count': 0,
            'success_count': 0
        })
        self._lock = threading.Lock()
        self._cleanup_interval = 100  # Cleanup every 100 operations
        self._operation_count = 0

    @memory_managed(MemoryStrategy.MODERATE)
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics with memory management."""
        with self._lock:
            self.metrics_history.append(metrics)
            self._operation_count += 1

            # Update function statistics
            stats = self.function_stats[metrics.function_name]
            stats['call_count'] += 1
            stats['total_time'] += metrics.execution_time
            stats['min_time'] = min(stats['min_time'], metrics.execution_time)
            stats['max_time'] = max(stats['max_time'], metrics.execution_time)
            stats['avg_time'] = stats['total_time'] / stats['call_count']

            if metrics.success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
            
            # Periodic cleanup to prevent memory leaks
            if self._operation_count % self._cleanup_interval == 0:
                self._cleanup_old_data()

    def get_function_stats(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for a function or all functions."""
        with self._lock:
            if function_name:
                return self.function_stats.get(function_name, {})
            return dict(self.function_stats)

    def get_recent_metrics(self, count: int = 100) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        with self._lock:
            return list(self.metrics_history)[-count:]

    def clear_history(self) -> None:
        """Clear performance history with memory cleanup."""
        with self._lock:
            self.metrics_history.clear()
            self.function_stats.clear()
            self._operation_count = 0
            
        # Force cleanup to free memory
        force_cleanup()
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old data to prevent memory leaks."""
        # Keep only recent metrics if we're approaching the limit
        if len(self.metrics_history) > self.max_history * 0.8:
            # Remove oldest 20% of metrics
            remove_count = int(self.max_history * 0.2)
            for _ in range(remove_count):
                if self.metrics_history:
                    self.metrics_history.popleft()
        
        # Cleanup function stats for functions that haven't been called recently
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour ago
        
        functions_to_remove = []
        for func_name, stats in self.function_stats.items():
            if stats.get('last_call_time', 0) < cutoff_time:
                functions_to_remove.append(func_name)
        
        for func_name in functions_to_remove:
            del self.function_stats[func_name]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {}

            recent_metrics = list(self.metrics_history)
            total_calls = len(recent_metrics)
            successful_calls = sum(1 for m in recent_metrics if m.success)

            execution_times = [m.execution_time for m in recent_metrics]
            memory_usage = [m.memory_usage for m in recent_metrics]
            cpu_usage = [m.cpu_usage for m in recent_metrics]

            return {
                'total_calls': total_calls,
                'successful_calls': successful_calls,
                'error_calls': total_calls - successful_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
                'execution_time': {
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'avg': np.mean(execution_times),
                    'std': np.std(execution_times)
                },
                'memory_usage': {
                    'min': min(memory_usage),
                    'max': max(memory_usage),
                    'avg': np.mean(memory_usage),
                    'std': np.std(memory_usage)
                },
                'cpu_usage': {
                    'min': min(cpu_usage),
                    'max': max(cpu_usage),
                    'avg': np.mean(cpu_usage),
                    'std': np.std(cpu_usage)
                }
            }

# Global performance monitor
global_monitor = PerformanceMonitor()

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except Exception:
        return 0.0

def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    try:
        return psutil.cpu_percent()
    except Exception:
        return 0.0

@contextmanager
def timer(name: str = "operation") -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for timing operations.

    Args:
        name: Name of the operation being timed

    Yields:
        Dictionary with timing information
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    start_cpu = get_cpu_usage()

    timing_info = {
        'name': name,
        'start_time': start_time,
        'start_memory': start_memory,
        'start_cpu': start_cpu
    }

    try:
        yield timing_info
    finally:
        end_time = time.time()
        end_memory = get_memory_usage()
        end_cpu = get_cpu_usage()

        timing_info.update({
            'end_time': end_time,
            'end_memory': end_memory,
            'end_cpu': end_cpu,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'cpu_delta': end_cpu - start_cpu
        })

        logger.debug(f"Operation '{name}' took {timing_info['execution_time']:.4f}s")

def profile_function(monitor: Optional[PerformanceMonitor] = None,
                    log_result: bool = False) -> Callable:
    """
    Decorator to profile function performance.

    Args:
        monitor: Performance monitor instance (uses global if None)
        log_result: Whether to log the result

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            perf_monitor = monitor or global_monitor
            start_time = time.time()
            start_memory = get_memory_usage()
            start_cpu = get_cpu_usage()

            success = True
            error_message = None
            result = None

            try:
                result = func(*args, **kwargs)
                if log_result:
                    logger.debug(f"Function {func.__name__} result: {result}")
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = get_memory_usage()
                end_cpu = get_cpu_usage()

                metrics = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time=end_time - start_time,
                    memory_usage=end_memory - start_memory,
                    cpu_usage=end_cpu - start_cpu,
                    timestamp=end_time,
                    args_count=len(args),
                    kwargs_count=len(kwargs),
                    success=success,
                    error_message=error_message
                )

                perf_monitor.record_metrics(metrics)

        return wrapper
    return decorator

class MemoryProfiler:
    """Memory usage profiler."""

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []

    def take_snapshot(self, name: str) -> Dict[str, Any]:
        """Take a memory snapshot."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            snapshot = {
                'name': name,
                'timestamp': time.time(),
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'percent': memory_percent,
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024
            }

            self.snapshots.append(snapshot)
            return snapshot

        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return {}

    def get_memory_growth(self) -> List[Dict[str, Any]]:
        """Get memory growth between snapshots."""
        if len(self.snapshots) < 2:
            return []

        growth = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]
            curr = self.snapshots[i]

            growth.append({
                'from': prev['name'],
                'to': curr['name'],
                'rss_growth': curr['rss'] - prev['rss'],
                'vms_growth': curr['vms'] - prev['vms'],
                'rss_growth_mb': (curr['rss'] - prev['rss']) / 1024 / 1024,
                'vms_growth_mb': (curr['vms'] - prev['vms']) / 1024 / 1024,
                'time_delta': curr['timestamp'] - prev['timestamp']
            })

        return growth

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()

class SystemMonitor:
    """System resource monitoring."""

    def __init__(self, interval: float = 1.0):
        """
        Initialize system monitor.

        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()

    def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        self._stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join()

        logger.info("System monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics.append(metrics)

                # Keep only recent metrics (last 1000)
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            self._stop_event.wait(self.interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                }
            }
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {'timestamp': time.time(), 'error': str(e)}

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return self._collect_metrics()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics:
            return {}

        cpu_values = [m.get('cpu_percent', 0) for m in self.metrics if 'cpu_percent' in m]
        memory_values = [m.get('memory', {}).get('percent', 0) for m in self.metrics if 'memory' in m]

        return {
            'total_samples': len(self.metrics),
            'cpu': {
                'min': min(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'avg': np.mean(cpu_values) if cpu_values else 0
            },
            'memory': {
                'min': min(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'avg': np.mean(memory_values) if memory_values else 0
            }
        }

# Convenience functions
def time_function(func: Callable) -> Callable:
    """Convenience decorator to time function execution."""
    return profile_function()(func)

def benchmark_function(func: Callable, iterations: int = 100) -> Dict[str, Any]:
    """
    Benchmark a function by running it multiple times.

    Args:
        func: Function to benchmark
        iterations: Number of iterations

    Returns:
        Benchmark results
    """
    times = []

    for _ in range(iterations):
        start_time = time.time()
        try:
            func()
            times.append(time.time() - start_time)
        except Exception as e:
            logger.error(f"Benchmark iteration failed: {e}")

    if not times:
        return {'error': 'No successful iterations'}

    return {
        'iterations': len(times),
        'total_time': sum(times),
        'avg_time': np.mean(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': np.std(times),
        'ops_per_second': 1.0 / np.mean(times)
    }

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    try:
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'boot_time': psutil.boot_time(),
            'platform': {
                'system': psutil.sys.platform,
                'python_version': psutil.sys.version
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {'error': str(e)}

def performance_timer(func: Callable) -> Callable:
    """
    Decorator to time function performance and log results.

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = get_memory_usage()

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = get_memory_usage()

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.debug(f"Function {func.__name__} took {execution_time:.4f}s, memory delta: {memory_delta:.2f}MB")

            # Record in global monitor
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage=memory_delta,
                cpu_usage=get_cpu_usage(),
                timestamp=end_time,
                args_count=len(args),
                kwargs_count=len(kwargs),
                success=True
            )
            global_monitor.record_metrics(metrics)

            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            logger.error(f"Function {func.__name__} failed after {execution_time:.4f}s: {e}")

            # Record failure
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage=0.0,
                cpu_usage=get_cpu_usage(),
                timestamp=end_time,
                args_count=len(args),
                kwargs_count=len(kwargs),
                success=False,
                error_message=str(e)
            )
            global_monitor.record_metrics(metrics)

            raise

    return wrapper

def memory_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of function execution.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with memory monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_memory = get_memory_usage()

        try:
            result = func(*args, **kwargs)
            end_memory = get_memory_usage()
            memory_delta = end_memory - start_memory

            logger.debug(f"Function {func.__name__} memory usage: {memory_delta:.2f}MB (start: {start_memory:.2f}MB, end: {end_memory:.2f}MB)")

            return result

        except Exception as e:
            end_memory = get_memory_usage()
            memory_delta = end_memory - start_memory

            logger.error(f"Function {func.__name__} failed with memory usage: {memory_delta:.2f}MB: {e}")
            raise

    return wrapper

__all__ = [
    'PerformanceMetrics',
    'PerformanceMonitor',
    'MemoryProfiler',
    'SystemMonitor',
    'global_monitor',
    'get_memory_usage',
    'get_cpu_usage',
    'timer',
    'profile_function',
    'time_function',
    'benchmark_function',
    'get_system_info',
    'performance_timer',
    'memory_monitor'
]
