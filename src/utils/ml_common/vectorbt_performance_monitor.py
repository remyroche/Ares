"""
VectorBT Performance Monitor

This module provides comprehensive performance monitoring for VectorBT operations
to track execution times, memory usage, and optimization effectiveness.
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import threading
from contextlib import contextmanager

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    enable_timing: bool = True
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_detailed_logging: bool = False
    max_history_size: int = 1000
    log_frequency: int = 10  # Log every N operations
    enable_threading: bool = True
    save_stats_to_file: bool = False
    stats_file_path: str = "vectorbt_performance_stats.json"

@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    duration: float
    memory_used_gb: float
    memory_peak_gb: float
    gpu_used: bool
    gpu_memory_used_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    thread_count: int = 1
    cache_hit: bool = False
    cache_miss: bool = False
    error_occurred: bool = False
    error_message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class VectorBTPerformanceMonitor:
    """
    Comprehensive performance monitoring for VectorBT operations.

    This class provides:
    - Operation timing and profiling
    - Memory usage tracking
    - GPU utilization monitoring
    - Performance statistics and analysis
    - Automatic optimization recommendations
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize VectorBT performance monitor."""
        self.config = config or PerformanceConfig()
        self.logger = logger.getChild('VectorBTPerformanceMonitor')

        # Performance tracking
        self.operation_history: deque = deque(maxlen=self.config.max_history_size)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.total_operations = 0

        # Performance statistics
        self.stats = {
            'total_operations': 0,
            'total_duration': 0.0,
            'total_memory_used_gb': 0.0,
            'peak_memory_used_gb': 0.0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'average_duration': 0.0,
            'average_memory_used_gb': 0.0,
            'fastest_operation': None,
            'slowest_operation': None,
            'most_memory_intensive': None
        }

        # Thread safety
        self._lock = threading.Lock() if self.config.enable_threading else None

        # Cache for performance optimization
        self._cache: Dict[str, Any] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0
        }

        self.logger.info("✅ VectorBT Performance Monitor initialized")
        self.logger.info(f"📊 Monitoring enabled: timing={self.config.enable_timing}, "
                        f"memory={self.config.enable_memory_tracking}, "
                        f"gpu={self.config.enable_gpu_tracking}")

    def _thread_safe(self, func: Callable) -> Callable:
        """Apply thread safety to a function if threading is enabled."""
        if self.config.enable_threading and self._lock:
            def wrapper(*args, **kwargs):
                with self._lock:
                    return func(*args, **kwargs)
            return wrapper
        return func

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start monitoring an operation.

        Args:
            operation_name: Name of the operation
            metadata: Additional metadata

        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        # Store operation start info
        operation_info = {
            'id': operation_id,
            'name': operation_name,
            'start_time': time.time(),
            'start_memory_gb': self._get_current_memory_usage(),
            'start_cpu_percent': psutil.cpu_percent(),
            'metadata': metadata or {}
        }

        # Store in thread-local storage or global cache
        if not hasattr(self, '_current_operations'):
            self._current_operations = {}
        self._current_operations[operation_id] = operation_info

        self.logger.debug(f"🚀 Started operation: {operation_name} (ID: {operation_id})")
        return operation_id

    def end_operation(self, operation_id: str, gpu_used: bool = False,
                     gpu_memory_used_gb: float = 0.0, error: Optional[Exception] = None) -> OperationMetrics:
        """
        End monitoring an operation and record metrics.

        Args:
            operation_id: Operation ID from start_operation
            gpu_used: Whether GPU was used
            gpu_memory_used_gb: GPU memory used in GB
            error: Exception if operation failed

        Returns:
            Operation metrics
        """
        if not hasattr(self, '_current_operations') or operation_id not in self._current_operations:
            self.logger.warning(f"⚠️ Operation ID {operation_id} not found")
            return None

        # Get operation info
        operation_info = self._current_operations[operation_id]
        del self._current_operations[operation_id]

        # Calculate metrics
        end_time = time.time()
        duration = end_time - operation_info['start_time']

        current_memory_gb = self._get_current_memory_usage()
        memory_used_gb = current_memory_gb - operation_info['start_memory_gb']
        memory_peak_gb = max(0, memory_used_gb)  # Peak is positive change

        current_cpu_percent = psutil.cpu_percent()
        cpu_usage_percent = (current_cpu_percent + operation_info['start_cpu_percent']) / 2

        # Create metrics
        metrics = OperationMetrics(
            operation_name=operation_info['name'],
            duration=duration,
            memory_used_gb=memory_used_gb,
            memory_peak_gb=memory_peak_gb,
            gpu_used=gpu_used,
            gpu_memory_used_gb=gpu_memory_used_gb,
            cpu_usage_percent=cpu_usage_percent,
            thread_count=psutil.cpu_count(),
            error_occurred=error is not None,
            error_message=str(error) if error else "",
            timestamp=end_time,
            metadata=operation_info['metadata']
        )

        # Record metrics
        self._record_metrics(metrics)

        # Log operation completion
        if error:
            self.logger.error(f"❌ Operation failed: {operation_info['name']} - {error}")
        else:
            self.logger.debug(f"✅ Completed operation: {operation_info['name']} "
                            f"({duration:.3f}s, {memory_used_gb:.2f}GB)")

        return metrics

    def _record_metrics(self, metrics: OperationMetrics):
        """Record operation metrics and update statistics."""
        # Add to history
        self.operation_history.append(metrics)

        # Update counts
        self.operation_counts[metrics.operation_name] += 1
        self.total_operations += 1

        # Update statistics
        self.stats['total_operations'] += 1
        self.stats['total_duration'] += metrics.duration
        self.stats['total_memory_used_gb'] += metrics.memory_used_gb
        self.stats['peak_memory_used_gb'] = max(
            self.stats['peak_memory_used_gb'],
            metrics.memory_peak_gb
        )

        if metrics.gpu_used:
            self.stats['gpu_operations'] += 1
        else:
            self.stats['cpu_operations'] += 1

        if metrics.cache_hit:
            self.stats['cache_hits'] += 1
        if metrics.cache_miss:
            self.stats['cache_misses'] += 1

        if metrics.error_occurred:
            self.stats['errors'] += 1

        # Update averages
        if self.stats['total_operations'] > 0:
            self.stats['average_duration'] = (
                self.stats['total_duration'] / self.stats['total_operations']
            )
            self.stats['average_memory_used_gb'] = (
                self.stats['total_memory_used_gb'] / self.stats['total_operations']
            )

        # Update fastest/slowest operations
        if (self.stats['fastest_operation'] is None or
            metrics.duration < self.stats['fastest_operation']['duration']):
            self.stats['fastest_operation'] = {
                'name': metrics.operation_name,
                'duration': metrics.duration,
                'timestamp': metrics.timestamp
            }

        if (self.stats['slowest_operation'] is None or
            metrics.duration > self.stats['slowest_operation']['duration']):
            self.stats['slowest_operation'] = {
                'name': metrics.operation_name,
                'duration': metrics.duration,
                'timestamp': metrics.timestamp
            }

        # Update most memory intensive
        if (self.stats['most_memory_intensive'] is None or
            metrics.memory_used_gb > self.stats['most_memory_intensive']['memory_used_gb']):
            self.stats['most_memory_intensive'] = {
                'name': metrics.operation_name,
                'memory_used_gb': metrics.memory_used_gb,
                'timestamp': metrics.timestamp
            }

        # Log performance periodically
        if self.total_operations % self.config.log_frequency == 0:
            self._log_performance_summary()

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if not self.config.enable_memory_tracking:
            return 0.0

        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception:
            return 0.0

    def _log_performance_summary(self):
        """Log performance summary."""
        if not self.config.enable_detailed_logging:
            return

        self.logger.info(f"📊 Performance Summary ({self.total_operations} operations):")
        self.logger.info(f"   Average duration: {self.stats['average_duration']:.3f}s")
        self.logger.info(f"   Average memory: {self.stats['average_memory_used_gb']:.2f}GB")
        self.logger.info(f"   Peak memory: {self.stats['peak_memory_used_gb']:.2f}GB")
        self.logger.info(f"   GPU operations: {self.stats['gpu_operations']}")
        self.logger.info(f"   Cache hit rate: {self._get_cache_hit_rate():.2%}")
        self.logger.info(f"   Error rate: {self._get_error_rate():.2%}")

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation type."""
        operations = [op for op in self.operation_history if op.operation_name == operation_name]

        if not operations:
            return {}

        durations = [op.duration for op in operations]
        memory_usage = [op.memory_used_gb for op in operations]

        return {
            'operation_name': operation_name,
            'count': len(operations),
            'total_duration': sum(durations),
            'average_duration': np.mean(durations) if NUMPY_AVAILABLE else sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_memory_used_gb': sum(memory_usage),
            'average_memory_used_gb': np.mean(memory_usage) if NUMPY_AVAILABLE else sum(memory_usage) / len(memory_usage),
            'gpu_operations': sum(1 for op in operations if op.gpu_used),
            'error_count': sum(1 for op in operations if op.error_occurred)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.stats.copy()

        # Add additional metrics
        summary.update({
            'cache_hit_rate': self._get_cache_hit_rate(),
            'error_rate': self._get_error_rate(),
            'gpu_utilization_rate': self._get_gpu_utilization_rate(),
            'operation_types': len(self.operation_counts),
            'most_frequent_operation': self._get_most_frequent_operation(),
            'recent_performance': self._get_recent_performance(),
            'memory_efficiency': self._get_memory_efficiency(),
            'optimization_recommendations': self._get_optimization_recommendations()
        })

        return summary

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_operations = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_cache_operations == 0:
            return 0.0
        return self.stats['cache_hits'] / total_cache_operations

    def _get_error_rate(self) -> float:
        """Calculate error rate."""
        if self.stats['total_operations'] == 0:
            return 0.0
        return self.stats['errors'] / self.stats['total_operations']

    def _get_gpu_utilization_rate(self) -> float:
        """Calculate GPU utilization rate."""
        if self.stats['total_operations'] == 0:
            return 0.0
        return self.stats['gpu_operations'] / self.stats['total_operations']

    def _get_most_frequent_operation(self) -> str:
        """Get the most frequently performed operation."""
        if not self.operation_counts:
            return "None"
        return max(self.operation_counts.items(), key=lambda x: x[1])[0]

    def _get_recent_performance(self, n_operations: int = 10) -> Dict[str, float]:
        """Get performance metrics for recent operations."""
        recent_operations = list(self.operation_history)[-n_operations:]

        if not recent_operations:
            return {}

        durations = [op.duration for op in recent_operations]
        memory_usage = [op.memory_used_gb for op in recent_operations]

        return {
            'recent_avg_duration': np.mean(durations) if NUMPY_AVAILABLE else sum(durations) / len(durations),
            'recent_avg_memory': np.mean(memory_usage) if NUMPY_AVAILABLE else sum(memory_usage) / len(memory_usage),
            'recent_operations': len(recent_operations)
        }

    def _get_memory_efficiency(self) -> Dict[str, Any]:
        """Calculate memory efficiency metrics."""
        if self.stats['total_operations'] == 0:
            return {}

        return {
            'avg_memory_per_operation': self.stats['average_memory_used_gb'],
            'peak_memory_usage': self.stats['peak_memory_used_gb'],
            'memory_utilization': self.stats['peak_memory_used_gb'] / psutil.virtual_memory().total * (1024**3),
            'memory_growth_rate': self._calculate_memory_growth_rate()
        }

    def _calculate_memory_growth_rate(self) -> float:
        """Calculate memory growth rate over time."""
        if len(self.operation_history) < 10:
            return 0.0

        # Get memory usage over time
        memory_usage = [op.memory_used_gb for op in self.operation_history]
        timestamps = [op.timestamp for op in self.operation_history]

        if len(memory_usage) < 2:
            return 0.0

        # Simple linear regression to find growth rate
        if NUMPY_AVAILABLE:
            try:
                coeffs = np.polyfit(timestamps, memory_usage, 1)
                return coeffs[0]  # Slope
            except:
                return 0.0
        else:
            return 0.0

    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance data."""
        recommendations = []

        # Check for slow operations
        if self.stats['average_duration'] > 10.0:
            recommendations.append("Consider optimizing slow operations - average duration is high")

        # Check for high memory usage
        if self.stats['peak_memory_used_gb'] > 4.0:
            recommendations.append("High memory usage detected - consider chunking or data type optimization")

        # Check for low GPU utilization
        if self.stats['gpu_operations'] > 0 and self._get_gpu_utilization_rate() < 0.3:
            recommendations.append("Low GPU utilization - consider more GPU-accelerated operations")

        # Check for high error rate
        if self._get_error_rate() > 0.1:
            recommendations.append("High error rate detected - check operation stability")

        # Check for cache efficiency
        if self._get_cache_hit_rate() < 0.5 and self.stats['cache_hits'] > 0:
            recommendations.append("Low cache hit rate - consider improving cache strategy")

        return recommendations

    @contextmanager
    def monitor_operation(self, operation_name: str, gpu_used: bool = False,
                         gpu_memory_used_gb: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for monitoring operations.

        Args:
            operation_name: Name of the operation
            gpu_used: Whether GPU will be used
            gpu_memory_used_gb: Expected GPU memory usage
            metadata: Additional metadata
        """
        operation_id = self.start_operation(operation_name, metadata)
        error = None

        try:
            yield operation_id
        except Exception as e:
            error = e
            raise
        finally:
            self.end_operation(operation_id, gpu_used, gpu_memory_used_gb, error)

    def save_stats_to_file(self, file_path: Optional[str] = None):
        """Save performance statistics to file."""
        if not self.config.save_stats_to_file:
            return

        file_path = file_path or self.config.stats_file_path

        try:
            stats_data = {
                'summary': self.get_performance_summary(),
                'operation_stats': {
                    name: self.get_operation_stats(name)
                    for name in self.operation_counts.keys()
                },
                'timestamp': time.time()
            }

            with open(file_path, 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)

            self.logger.info(f"📊 Performance stats saved to {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save stats to file: {e}")

    def reset_stats(self):
        """Reset all performance statistics."""
        self.operation_history.clear()
        self.operation_counts.clear()
        self.total_operations = 0
        self.stats = {
            'total_operations': 0,
            'total_duration': 0.0,
            'total_memory_used_gb': 0.0,
            'peak_memory_used_gb': 0.0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'average_duration': 0.0,
            'average_memory_used_gb': 0.0,
            'fastest_operation': None,
            'slowest_operation': None,
            'most_memory_intensive': None
        }

        self.logger.info("🔄 Performance statistics reset")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> VectorBTPerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = VectorBTPerformanceMonitor()
    return _performance_monitor

@contextmanager
def monitor_operation(operation_name: str, gpu_used: bool = False,
                     gpu_memory_used_gb: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
    """Convenience context manager for monitoring operations."""
    monitor = get_performance_monitor()
    with monitor.monitor_operation(operation_name, gpu_used, gpu_memory_used_gb, metadata):
        yield
