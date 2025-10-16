from src.utils.tprint import tprint

"""
Performance Monitoring Component

This module provides comprehensive performance monitoring and optimization
utilities for the feature selection framework.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
import psutil
import gc
from collections import defaultdict, deque

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.PerformanceMonitoring")
    tprint("✅ Custom logger available for FeatureSelection.PerformanceMonitoring")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("FeatureSelection.PerformanceMonitoring")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

class PerformanceMonitor:
    """Comprehensive performance monitoring for feature selection operations."""

    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor."""
        self.max_history = max_history
        self.logger = logger.getChild('PerformanceMonitor')

        # Performance tracking
        self.execution_times = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)

        # System monitoring
        self.system_stats = deque(maxlen=max_history)
        self.gc_stats = deque(maxlen=max_history)

        # Performance thresholds
        self.slow_operation_threshold = 10.0  # seconds
        self.high_memory_threshold = 1024  # MB

        _LOGGER.info("📊 PerformanceMonitor initialized")
        _LOGGER.info(f"⚙️ Max history: {max_history}")
        _LOGGER.info(f"⚙️ Slow operation threshold: {self.slow_operation_threshold}s")
        _LOGGER.info(f"⚙️ High memory threshold: {self.high_memory_threshold}MB")

    def monitor(self, operation_name: str):
        """Context manager for monitoring operations."""
        return PerformanceContext(self, operation_name)

    def record_execution(self, operation_name: str, execution_time: float, memory_usage: float = 0.0):
        """Record execution statistics."""
        try:
            # Record execution time
            self.execution_times[operation_name].append(execution_time)
            if len(self.execution_times[operation_name]) > self.max_history:
                self.execution_times[operation_name] = self.execution_times[operation_name][-self.max_history:]

            # Record memory usage
            if memory_usage > 0:
                self.memory_usage[operation_name].append(memory_usage)
                if len(self.memory_usage[operation_name]) > self.max_history:
                    self.memory_usage[operation_name] = self.memory_usage[operation_name][-self.max_history:]

            # Update operation count
            self.operation_counts[operation_name] += 1

            # Check for performance issues
            if execution_time > self.slow_operation_threshold:
                _LOGGER.warning(f"⚠️ Slow operation detected: {operation_name} took {execution_time:.3f}s")

            if memory_usage > self.high_memory_threshold:
                _LOGGER.warning(f"⚠️ High memory usage detected: {operation_name} used {memory_usage:.2f}MB")

            _LOGGER.debug(f"📊 Recorded execution: {operation_name} - {execution_time:.3f}s, {memory_usage:.2f}MB")

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to record execution stats: {e}")

    def record_error(self, operation_name: str, error: str):
        """Record error occurrence."""
        try:
            self.error_counts[operation_name] += 1
            _LOGGER.warning(f"❌ Error recorded for {operation_name}: {error}")
        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to record error: {e}")

    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'process_memory_mb': self.get_current_memory_usage(),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }

            # Record system stats
            self.system_stats.append(stats)

            return stats
        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to get system stats: {e}")
            return {}

    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'gc_counts': list(gc.get_count()),
                'gc_thresholds': list(gc.get_threshold()),
                'gc_stats': gc.get_stats()
            }

            # Record GC stats
            self.gc_stats.append(stats)

            return stats
        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to get GC stats: {e}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            stats = {
                'execution_times': {},
                'memory_usage': {},
                'operation_counts': dict(self.operation_counts),
                'error_counts': dict(self.error_counts),
                'system_stats': list(self.system_stats)[-10:],  # Last 10 system stats
                'gc_stats': list(self.gc_stats)[-10:],  # Last 10 GC stats
                'performance_summary': self._get_performance_summary()
            }

            # Calculate execution time statistics
            for operation, times in self.execution_times.items():
                if times:
                    stats['execution_times'][operation] = {
                        'count': len(times),
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'total': np.sum(times)
                    }

            # Calculate memory usage statistics
            for operation, memory in self.memory_usage.items():
                if memory:
                    stats['memory_usage'][operation] = {
                        'count': len(memory),
                        'mean': np.mean(memory),
                        'std': np.std(memory),
                        'min': np.min(memory),
                        'max': np.max(memory),
                        'total': np.sum(memory)
                    }

            return stats

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to get performance stats: {e}")
            return {'error': str(e)}

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            summary = {
                'total_operations': sum(self.operation_counts.values()),
                'total_errors': sum(self.error_counts.values()),
                'error_rate': 0.0,
                'slowest_operation': None,
                'most_memory_intensive': None,
                'most_frequent_operation': None
            }

            # Calculate error rate
            if summary['total_operations'] > 0:
                summary['error_rate'] = summary['total_errors'] / summary['total_operations']

            # Find slowest operation
            if self.execution_times:
                slowest_ops = []
                for operation, times in self.execution_times.items():
                    if times:
                        slowest_ops.append((operation, np.mean(times)))
                if slowest_ops:
                    summary['slowest_operation'] = max(slowest_ops, key=lambda x: x[1])

            # Find most memory intensive operation
            if self.memory_usage:
                memory_ops = []
                for operation, memory in self.memory_usage.items():
                    if memory:
                        memory_ops.append((operation, np.mean(memory)))
                if memory_ops:
                    summary['most_memory_intensive'] = max(memory_ops, key=lambda x: x[1])

            # Find most frequent operation
            if self.operation_counts:
                summary['most_frequent_operation'] = max(self.operation_counts.items(), key=lambda x: x[1])

            return summary

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to get performance summary: {e}")
            return {}

    def optimize_memory(self):
        """Trigger memory optimization."""
        try:
            _LOGGER.info("🧠 Triggering memory optimization...")

            # Force garbage collection
            collected = gc.collect()
            _LOGGER.info(f"🗑️ Garbage collected {collected} objects")

            # Get memory stats after cleanup
            memory_after = self.get_current_memory_usage()
            _LOGGER.info(f"💾 Memory usage after cleanup: {memory_after:.2f}MB")

            return {
                'objects_collected': collected,
                'memory_after_mb': memory_after,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            _LOGGER.warning(f"⚠️ Memory optimization failed: {e}")
            return {'error': str(e)}

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        try:
            stats = self.get_stats()
            summary = stats.get('performance_summary', {})

            report = f"""
=== Feature Selection Performance Report ===
Generated: {datetime.now().isoformat()}

=== Summary ===
Total Operations: {summary.get('total_operations', 0)}
Total Errors: {summary.get('total_errors', 0)}
Error Rate: {summary.get('error_rate', 0):.2%}

=== Performance Metrics ===
Slowest Operation: {summary.get('slowest_operation', 'N/A')}
Most Memory Intensive: {summary.get('most_memory_intensive', 'N/A')}
Most Frequent Operation: {summary.get('most_frequent_operation', 'N/A')}

=== Operation Details ===
"""

            # Add execution time details
            for operation, times in stats.get('execution_times', {}).items():
                report += f"""
{operation}:
  Count: {times['count']}
  Mean Time: {times['mean']:.3f}s
  Std Time: {times['std']:.3f}s
  Min Time: {times['min']:.3f}s
  Max Time: {times['max']:.3f}s
  Total Time: {times['total']:.3f}s
"""

            # Add memory usage details
            for operation, memory in stats.get('memory_usage', {}).items():
                report += f"""
{operation} Memory:
  Count: {memory['count']}
  Mean Memory: {memory['mean']:.2f}MB
  Max Memory: {memory['max']:.2f}MB
  Total Memory: {memory['total']:.2f}MB
"""

            return report

        except Exception as e:
            _LOGGER.error(f"❌ Failed to generate performance report: {e}")
            return f"Error generating performance report: {e}"

    def reset_stats(self):
        """Reset all performance statistics."""
        try:
            self.execution_times.clear()
            self.memory_usage.clear()
            self.operation_counts.clear()
            self.error_counts.clear()
            self.system_stats.clear()
            self.gc_stats.clear()

            _LOGGER.info("🔄 Performance statistics reset")

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to reset stats: {e}")

class PerformanceContext:
    """Context manager for performance monitoring."""

    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        """Initialize performance context."""
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        """Enter performance monitoring context."""
        self.start_time = time.time()
        self.start_memory = self.monitor.get_current_memory_usage()
        _LOGGER.debug(f"📊 Starting performance monitoring: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit performance monitoring context."""
        try:
            end_time = time.time()
            end_memory = self.monitor.get_current_memory_usage()

            execution_time = end_time - self.start_time
            memory_delta = end_memory - self.start_memory

            # Record performance
            self.monitor.record_execution(self.operation_name, execution_time, memory_delta)

            # Record error if exception occurred
            if exc_type is not None:
                self.monitor.record_error(self.operation_name, str(exc_val))

            _LOGGER.debug(f"📊 Completed performance monitoring: {self.operation_name} - "
                         f"{execution_time:.3f}s, {memory_delta:.2f}MB")

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to record performance context: {e}")

class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory optimizer."""
        self.config = config or {}
        self.logger = logger.getChild('MemoryOptimizer')

        self.memory_threshold = self.config.get('memory_threshold', 0.8)  # 80% of available memory
        self.chunk_size = self.config.get('chunk_size', 10000)
        self.gc_frequency = self.config.get('gc_frequency', 100)

        self.operation_count = 0

        _LOGGER.info("🧠 MemoryOptimizer initialized")
        _LOGGER.info(f"⚙️ Memory threshold: {self.memory_threshold}")
        _LOGGER.info(f"⚙️ Chunk size: {self.chunk_size}")

    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return status."""
        try:
            memory_info = psutil.virtual_memory()
            process = psutil.Process()

            memory_status = {
                'total_memory_gb': memory_info.total / (1024**3),
                'available_memory_gb': memory_info.available / (1024**3),
                'used_memory_gb': memory_info.used / (1024**3),
                'memory_percent': memory_info.percent,
                'process_memory_mb': process.memory_info().rss / (1024**2),
                'memory_pressure': memory_info.percent / 100 > self.memory_threshold
            }

            return memory_status

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to check memory usage: {e}")
            return {'error': str(e)}

    def optimize_memory_usage(self):
        """Optimize memory usage."""
        try:
            self.operation_count += 1

            # Check if we need to optimize
            memory_status = self.check_memory_usage()
            if memory_status.get('memory_pressure', False):
                _LOGGER.info("🧠 Memory pressure detected, optimizing...")

                # Force garbage collection
                collected = gc.collect()
                _LOGGER.info(f"🗑️ Collected {collected} objects")

                # Check memory after cleanup
                memory_after = self.check_memory_usage()
                _LOGGER.info(f"💾 Memory after optimization: {memory_after.get('memory_percent', 0):.1f}%")

                return {
                    'optimization_performed': True,
                    'objects_collected': collected,
                    'memory_before_percent': memory_status.get('memory_percent', 0),
                    'memory_after_percent': memory_after.get('memory_percent', 0)
                }
            else:
                return {'optimization_performed': False, 'reason': 'No memory pressure'}

        except Exception as e:
            _LOGGER.warning(f"⚠️ Memory optimization failed: {e}")
            return {'error': str(e)}

    def process_in_chunks(self, data: np.ndarray, operation: callable,
                         chunk_size: int = None) -> np.ndarray:
        """Process data in memory-efficient chunks."""
        if chunk_size is None:
            chunk_size = self.chunk_size

        try:
            n_samples = data.shape[0]
            results = []

            _LOGGER.info(f"🔄 Processing {n_samples} samples in chunks of {chunk_size}")

            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                chunk = data[i:end_idx]

                # Process chunk
                chunk_result = operation(chunk)
                results.append(chunk_result)

                # Optimize memory periodically
                if self.operation_count % self.gc_frequency == 0:
                    self.optimize_memory_usage()

                _LOGGER.debug(f"📊 Processed chunk {i//chunk_size + 1}/{(n_samples-1)//chunk_size + 1}")

            # Combine results
            if results:
                if isinstance(results[0], np.ndarray):
                    return np.concatenate(results, axis=0)
                else:
                    return results
            else:
                return np.array([])

        except Exception as e:
            _LOGGER.error(f"❌ Chunk processing failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get memory optimizer statistics."""
        try:
            memory_status = self.check_memory_usage()

            return {
                'operation_count': self.operation_count,
                'memory_threshold': self.memory_threshold,
                'chunk_size': self.chunk_size,
                'gc_frequency': self.gc_frequency,
                'current_memory_status': memory_status
            }

        except Exception as e:
            _LOGGER.warning(f"⚠️ Failed to get memory optimizer stats: {e}")
            return {'error': str(e)}
