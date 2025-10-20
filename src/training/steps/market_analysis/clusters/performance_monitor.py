"""
Performance Monitoring Module

This module provides comprehensive performance monitoring for clustering operations,
tracking runtime, memory usage, GPU utilization, and providing detailed performance reports.
"""

import logging
import time
import psutil
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
from functools import wraps
import os

# Import utility modules
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    format_bytes, chunked_iterable, parallel_map, timed_operation,
    get_current_datetime, format_datetime, parse_datetime,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    get_logger, integrate_with_m1_optimizers, cleanup_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    is_m1_available, is_mps_available, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, validate_file_path, get_file_size,
    check_disk_space, safe_rolling, safe_groupby_operation, safe_apply_function,
    safe_filter_dataframe, create_summary_statistics, safe_resample,
    align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
    sanitize_string, math_safe, validate_correlation_matrix, safe_matrix_inverse,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

from src.utils.math_validation import (
    MathValidationError, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array as math_validate_numeric_array
)

# Import hardware utilities
try:
    from src.utils.hardware import get_integrated_hardware_manager
     as hw_get_m1_memory_optimizer
     as hw_get_m1_cpu_optimizer
except ImportError:
    hw_is_m1_available = lambda: False
    hw_is_mps_available = lambda: False
    hw_get_m1_memory_optimizer = lambda: None
    hw_get_m1_cpu_optimizer = lambda: None

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.validation.validation_utils import (
        ConfigurationValidator, DataValidator as MLDataValidator, ResourceValidator,
        ExecutionValidator, ResultValidator
    )
    # CVLSA PerformanceAnalytics removed - no longer available
    PerformanceAnalytics = None
    from src.utils.ml_common.optimization.shared_utils.integration_verification import SharedUtilsIntegrationVerifier
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
except ImportError:
    BayesianTPEOptimizer = None
    ConfigurationValidator = None
    MLDataValidator = None
    ResourceValidator = None
    ExecutionValidator = None
    ResultValidator = None
    PerformanceAnalytics = None
    SharedUtilsIntegrationVerifier = None

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class PerformanceMonitor:
    """Comprehensive performance monitoring system for clustering operations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize performance monitor."""
        self.logger = logger or logging.getLogger(__name__)
        self.performance_history: List[Dict[str, Any]] = []
        self.function_performance_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_thresholds: Dict[str, float] = {}
        self.memory_usage_history: List[Dict[str, Any]] = []
        self.cpu_usage_history: List[Dict[str, Any]] = []
        self.gpu_usage_history: List[Dict[str, Any]] = []
        self.throughput_history: List[Dict[str, Any]] = []

        # Monitoring intervals
        self.system_monitor_interval = 1.0  # seconds
        self.throughput_window = 10  # seconds

        # Auto-monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None

    def start_performance_monitoring(self, function_name: str, call_id: str) -> Dict[str, Any]:
        """Start performance monitoring for a function call."""
        try:
            # Get initial system metrics
            initial_metrics = self._get_system_metrics()
            initial_throughput = self._get_throughput_metrics()

            performance_record = {
                'function_name': function_name,
                'call_id': call_id,
                'start_time': datetime.now(),
                'start_metrics': initial_metrics,
                'start_throughput': initial_throughput,
                'end_time': None,
                'end_metrics': None,
                'end_throughput': None,
                'execution_time': 0.0,
                'memory_delta_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'gpu_usage_percent': 0.0,
                'throughput_samples_per_sec': 0.0,
                'performance_score': 0.0,
                'bottlenecks': [],
                'optimization_suggestions': [],
                'memory_snapshots': [],
                'cpu_snapshots': [],
                'gpu_snapshots': [],
                'throughput_snapshots': []
            }

            return performance_record

        except Exception as e:
            self.logger.error(f"Failed to start performance monitoring: {e}")
            return {}

    def end_performance_monitoring(self, performance_record: Dict[str, Any]) -> Dict[str, Any]:
        """End performance monitoring and calculate metrics."""
        try:
            if not performance_record:
                return {}

            # Get final system metrics
            final_metrics = self._get_system_metrics()
            final_throughput = self._get_throughput_metrics()

            performance_record['end_time'] = datetime.now()
            performance_record['end_metrics'] = final_metrics
            performance_record['end_throughput'] = final_throughput

            # Calculate execution time
            if performance_record['start_time'] and performance_record['end_time']:
                execution_time = (
                    performance_record['end_time'] - performance_record['start_time']
                ).total_seconds()
                performance_record['execution_time'] = execution_time

            # Calculate resource deltas
            self._calculate_resource_deltas(performance_record)

            # Calculate performance score
            performance_record['performance_score'] = self._calculate_performance_score(performance_record)

            # Identify bottlenecks
            performance_record['bottlenecks'] = self._identify_bottlenecks(performance_record)

            # Generate optimization suggestions
            performance_record['optimization_suggestions'] = self._generate_optimization_suggestions(
                performance_record
            )

            # Update function performance stats
            self._update_function_performance_stats(performance_record)

            # Add to history
            self.performance_history.append(performance_record)

            return performance_record

        except Exception as e:
            self.logger.error(f"Failed to end performance monitoring: {e}")
            return performance_record

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            tprint_debug("Collecting system metrics")
            metrics = {}

            # Memory usage using common operations utility
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['memory_mb'] = safe_divide(memory_info.rss, 1024 * 1024, 0.0)  # Convert to MB
            metrics['memory_percent'] = safe_float(process.memory_percent(), 0.0)

            # CPU usage
            metrics['cpu_percent'] = safe_float(psutil.cpu_percent(interval=None), 0.0)

            # System load
            if hasattr(psutil, 'getloadavg'):
                try:
                    load_avg = psutil.getloadavg()
                    metrics['load_average'] = safe_float(load_avg[0], 0.0) if load_avg else None
                except:
                    metrics['load_average'] = None

            # GPU metrics
            gpu_metrics = self._get_gpu_metrics()
            metrics.update(gpu_metrics)

            # Use hardware utilities for M1-specific metrics
            if hw_is_m1_available():
                tprint_debug("Using M1 hardware utilities for enhanced metrics")
                try:
                    m1_memory_opt = hw_get_integrated_hardware_manager()
                    if m1_memory_opt:
                        m1_metrics = m1_memory_opt.get_memory_info()
                        metrics.update(m1_metrics)
                except Exception as e:
                    tprint_warning(f"Failed to get M1 memory metrics: {e}")

            tprint_debug(f"Collected {len(metrics)} system metrics")
            return metrics

        except Exception as e:
            tprint_warning(f"Failed to get system metrics: {e}")
            self.logger.warning(f"Failed to get system metrics: {e}")
            return {}

    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available."""
        try:
            if not NVML_AVAILABLE:
                return {}

            metrics = {}
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(min(device_count, 2)):  # Monitor up to 2 GPUs
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f'gpu_{i}_memory_mb'] = mem_info.used / 1024 / 1024
                metrics[f'gpu_{i}_memory_percent'] = (mem_info.used / mem_info.total) * 100

                # Utilization
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f'gpu_{i}_utilization_percent'] = util_info.gpu

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics[f'gpu_{i}_temperature_c'] = temp
                except:
                    pass

            return metrics

        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
            return {}

    def _get_throughput_metrics(self) -> Dict[str, Any]:
        """Get throughput metrics."""
        try:
            # This is a simplified version - in practice, you'd track actual data processing rates
            return {
                'samples_processed': 0,
                'bytes_processed': 0,
                'throughput_samples_per_sec': 0.0,
                'throughput_mb_per_sec': 0.0
            }
        except Exception as e:
            self.logger.warning(f"Failed to get throughput metrics: {e}")
            return {}

    def _calculate_resource_deltas(self, performance_record: Dict[str, Any]) -> None:
        """Calculate resource usage deltas."""
        try:
            start_metrics = performance_record.get('start_metrics', {})
            end_metrics = performance_record.get('end_metrics', {})

            # Memory delta
            if ('memory_mb' in start_metrics and 'memory_mb' in end_metrics):
                performance_record['memory_delta_mb'] = (
                    end_metrics['memory_mb'] - start_metrics['memory_mb']
                )

            # CPU usage (average over execution time)
            if ('cpu_percent' in start_metrics and 'cpu_percent' in end_metrics):
                performance_record['cpu_usage_percent'] = (
                    end_metrics['cpu_percent'] + start_metrics['cpu_percent']
                ) / 2

            # GPU usage
            gpu_usage = 0.0
            gpu_count = 0
            for key in end_metrics.keys():
                if key.startswith('gpu_') and key.endswith('_utilization_percent'):
                    gpu_usage += end_metrics[key]
                    gpu_count += 1

            if gpu_count > 0:
                performance_record['gpu_usage_percent'] = gpu_usage / gpu_count

            # Throughput
            start_throughput = performance_record.get('start_throughput', {})
            end_throughput = performance_record.get('end_throughput', {})
            if (start_throughput and end_throughput and
                'samples_processed' in start_throughput and 'samples_processed' in end_throughput):

                samples_delta = end_throughput['samples_processed'] - start_throughput['samples_processed']
                execution_time = performance_record.get('execution_time', 1.0)

                if execution_time > 0:
                    performance_record['throughput_samples_per_sec'] = samples_delta / execution_time

        except Exception as e:
            self.logger.warning(f"Failed to calculate resource deltas: {e}")

    def _calculate_performance_score(self, performance_record: Dict[str, Any]) -> float:
        """Calculate performance score based on multiple metrics."""
        try:
            score = 100.0  # Start with perfect score

            # Execution time penalty (0-30 points)
            execution_time = performance_record.get('execution_time', 0)
            if execution_time > 60:  # More than 1 minute
                score -= min(30, (execution_time - 60) * 0.5)
            elif execution_time > 10:  # More than 10 seconds
                score -= min(20, (execution_time - 10) * 2)

            # Memory usage penalty (0-25 points)
            memory_delta = abs(performance_record.get('memory_delta_mb', 0))
            if memory_delta > 1000:  # More than 1GB
                score -= min(25, (memory_delta - 1000) * 0.025)
            elif memory_delta > 100:  # More than 100MB
                score -= min(15, (memory_delta - 100) * 0.15)

            # CPU usage penalty (0-20 points)
            cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
            if cpu_usage > 80:  # More than 80% CPU
                score -= min(20, (cpu_usage - 80) * 0.5)
            elif cpu_usage > 50:  # More than 50% CPU
                score -= min(10, (cpu_usage - 50) * 0.33)

            # GPU usage penalty (0-15 points)
            gpu_usage = abs(performance_record.get('gpu_usage_percent', 0))
            if gpu_usage > 90:  # More than 90% GPU
                score -= min(15, (gpu_usage - 90) * 0.75)
            elif gpu_usage > 70:  # More than 70% GPU
                score -= min(10, (gpu_usage - 70) * 0.5)

            # Throughput bonus (0-10 points)
            throughput = performance_record.get('throughput_samples_per_sec', 0)
            if throughput > 1000:  # High throughput
                score += min(10, (throughput - 1000) * 0.01)
            elif throughput > 100:  # Moderate throughput
                score += min(5, (throughput - 100) * 0.05)

            return max(0, min(100, score))  # Ensure score is between 0-100

        except Exception as e:
            self.logger.warning(f"Failed to calculate performance score: {e}")
            return 50.0  # Default score

    def _identify_bottlenecks(self, performance_record: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        execution_time = performance_record.get('execution_time', 0)
        memory_delta = abs(performance_record.get('memory_delta_mb', 0))
        cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
        gpu_usage = abs(performance_record.get('gpu_usage_percent', 0))
        throughput = performance_record.get('throughput_samples_per_sec', 0)

        if execution_time > 60:
            bottlenecks.append("Long execution time (>60s)")
        elif execution_time > 10:
            bottlenecks.append("Moderate execution time (>10s)")

        if memory_delta > 1000:
            bottlenecks.append("High memory usage (>1GB)")
        elif memory_delta > 100:
            bottlenecks.append("Moderate memory usage (>100MB)")

        if cpu_usage > 80:
            bottlenecks.append("High CPU usage (>80%)")
        elif cpu_usage > 50:
            bottlenecks.append("Moderate CPU usage (>50%)")

        if gpu_usage > 90:
            bottlenecks.append("High GPU usage (>90%)")
        elif gpu_usage > 70:
            bottlenecks.append("Moderate GPU usage (>70%)")

        if throughput < 10 and execution_time > 5:
            bottlenecks.append("Low throughput for execution time")

        return bottlenecks

    def _generate_optimization_suggestions(self, performance_record: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on performance metrics."""
        suggestions = []

        execution_time = performance_record.get('execution_time', 0)
        memory_delta = abs(performance_record.get('memory_delta_mb', 0))
        cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
        gpu_usage = abs(performance_record.get('gpu_usage_percent', 0))
        function_name = performance_record.get('function_name', '')

        if execution_time > 30:
            suggestions.extend([
                "Consider breaking down the function into smaller, more manageable parts",
                "Implement caching for repeated computations",
                "Use vectorized operations instead of loops where possible",
                "Consider parallel processing for independent operations"
            ])

        if memory_delta > 500:
            suggestions.extend([
                "Process data in smaller chunks to reduce memory footprint",
                "Use memory-efficient data types (e.g., float32 instead of float64)",
                "Clear unused variables and objects explicitly",
                "Consider using memory mapping for large datasets"
            ])

        if cpu_usage > 70:
            suggestions.extend([
                "Consider parallel processing for independent operations",
                "Optimize algorithms for better time complexity",
                "Use more efficient data structures",
                "Consider using NumPy/SciPy optimized functions"
            ])

        if gpu_usage > 70 and TORCH_AVAILABLE:
            suggestions.extend([
                "Optimize GPU memory usage with torch.cuda.empty_cache()",
                "Consider using mixed precision training (float16)",
                "Batch operations to improve GPU utilization",
                "Monitor GPU memory fragmentation"
            ])

        # Function-specific suggestions
        if 'clustering' in function_name.lower():
            suggestions.extend([
                "Consider using approximate nearest neighbor algorithms for large datasets",
                "Use incremental clustering algorithms for streaming data",
                "Pre-compute distance matrices when possible",
                "Consider dimensionality reduction before clustering"
            ])
        elif 'knn' in function_name.lower():
            suggestions.extend([
                "Use KD-trees or ball trees for efficient nearest neighbor search",
                "Consider approximate nearest neighbor methods for large datasets",
                "Cache distance computations when possible"
            ])

        return list(set(suggestions))  # Remove duplicates

    def _update_function_performance_stats(self, performance_record: Dict[str, Any]) -> None:
        """Update function performance statistics."""
        try:
            function_name = performance_record['function_name']

            if function_name not in self.function_performance_stats:
                self.function_performance_stats[function_name] = {
                    'total_calls': 0,
                    'total_execution_time': 0.0,
                    'total_memory_usage': 0.0,
                    'total_cpu_usage': 0.0,
                    'total_gpu_usage': 0.0,
                    'execution_times': [],
                    'memory_usages': [],
                    'cpu_usages': [],
                    'gpu_usages': [],
                    'performance_scores': [],
                    'bottlenecks': defaultdict(int),
                    'optimization_suggestions': set()
                }

            stats = self.function_performance_stats[function_name]
            stats['total_calls'] += 1
            stats['total_execution_time'] += performance_record.get('execution_time', 0)
            stats['total_memory_usage'] += abs(performance_record.get('memory_delta_mb', 0))
            stats['total_cpu_usage'] += abs(performance_record.get('cpu_usage_percent', 0))
            stats['total_gpu_usage'] += abs(performance_record.get('gpu_usage_percent', 0))

            stats['execution_times'].append(performance_record.get('execution_time', 0))
            stats['memory_usages'].append(abs(performance_record.get('memory_delta_mb', 0)))
            stats['cpu_usages'].append(abs(performance_record.get('cpu_usage_percent', 0)))
            stats['gpu_usages'].append(abs(performance_record.get('gpu_usage_percent', 0)))
            stats['performance_scores'].append(performance_record.get('performance_score', 0))

            # Update bottlenecks
            for bottleneck in performance_record.get('bottlenecks', []):
                stats['bottlenecks'][bottleneck] += 1

            # Update optimization suggestions
            for suggestion in performance_record.get('optimization_suggestions', []):
                stats['optimization_suggestions'].add(suggestion)

        except Exception as e:
            self.logger.error(f"Failed to update function performance stats: {e}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            if not self.performance_history:
                return {'total_monitored_calls': 0, 'message': 'No performance data recorded'}

            # Overall statistics
            total_calls = len(self.performance_history)
            total_execution_time = sum(record.get('execution_time', 0) for record in self.performance_history)
            total_memory_usage = sum(abs(record.get('memory_delta_mb', 0)) for record in self.performance_history)
            total_cpu_usage = sum(abs(record.get('cpu_usage_percent', 0)) for record in self.performance_history)
            total_gpu_usage = sum(abs(record.get('gpu_usage_percent', 0)) for record in self.performance_history)

            # Performance scores
            performance_scores = [record.get('performance_score', 0) for record in self.performance_history]
            avg_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0

            # Identify worst performers
            worst_performers = sorted(
                self.performance_history,
                key=lambda x: x.get('performance_score', 0)
            )[:5]

            # Function-specific analysis
            function_analysis = {}
            for function_name, stats in self.function_performance_stats.items():
                if stats['total_calls'] > 0:
                    function_analysis[function_name] = {
                        'total_calls': stats['total_calls'],
                        'average_execution_time': stats['total_execution_time'] / stats['total_calls'],
                        'average_memory_usage': stats['total_memory_usage'] / stats['total_calls'],
                        'average_cpu_usage': stats['total_cpu_usage'] / stats['total_calls'],
                        'average_gpu_usage': stats['total_gpu_usage'] / stats['total_calls'],
                        'average_performance_score': sum(stats['performance_scores']) / len(stats['performance_scores']),
                        'most_common_bottlenecks': sorted(
                            stats['bottlenecks'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:3],
                        'optimization_suggestions': list(stats['optimization_suggestions'])[:5]
                    }

            return {
                'total_monitored_calls': total_calls,
                'overall_statistics': {
                    'total_execution_time': total_execution_time,
                    'total_memory_usage': total_memory_usage,
                    'total_cpu_usage': total_cpu_usage,
                    'total_gpu_usage': total_gpu_usage,
                    'average_performance_score': avg_performance_score
                },
                'worst_performers': [
                    {
                        'function_name': record['function_name'],
                        'call_id': record['call_id'],
                        'performance_score': record.get('performance_score', 0),
                        'execution_time': record.get('execution_time', 0),
                        'bottlenecks': record.get('bottlenecks', [])
                    }
                    for record in worst_performers
                ],
                'function_analysis': function_analysis,
                'performance_trends': self._analyze_performance_trends()
            }

        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {}

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if len(self.performance_history) < 2:
                return {'trend': 'insufficient_data'}

            # Sort by start time
            sorted_history = sorted(self.performance_history, key=lambda x: x['start_time'])

            # Calculate trend for execution time
            execution_times = [record.get('execution_time', 0) for record in sorted_history]
            if len(execution_times) > 1:
                time_trend = 'improving' if execution_times[-1] < execution_times[0] else 'degrading'
            else:
                time_trend = 'stable'

            # Calculate trend for performance scores
            performance_scores = [record.get('performance_score', 0) for record in sorted_history]
            if len(performance_scores) > 1:
                score_trend = 'improving' if performance_scores[-1] > performance_scores[0] else 'degrading'
            else:
                score_trend = 'stable'

            return {
                'execution_time_trend': time_trend,
                'performance_score_trend': score_trend,
                'data_points': len(sorted_history)
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
            return {}

    def log_performance_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive performance report."""
        try:
            if report.get('total_monitored_calls', 0) == 0:
                self.logger.info("📊 No performance data recorded")
                return

            self.logger.info("📊 CLUSTERING PERFORMANCE MONITORING REPORT")
            self.logger.info("=" * 60)
            self.logger.info(f"Total Monitored Calls: {report['total_monitored_calls']}")

            # Overall statistics
            overall_stats = report.get('overall_statistics', {})
            if overall_stats:
                self.logger.info("📈 OVERALL STATISTICS:")
                self.logger.info(f"   Total Execution Time: {overall_stats.get('total_execution_time', 0):.3f}s")
                self.logger.info(f"   Total Memory Usage: {overall_stats.get('total_memory_usage', 0):.1f}MB")
                self.logger.info(f"   Total CPU Usage: {overall_stats.get('total_cpu_usage', 0):.1f}%")
                self.logger.info(f"   Total GPU Usage: {overall_stats.get('total_gpu_usage', 0):.1f}%")
                self.logger.info(f"   Average Performance Score: {overall_stats.get('average_performance_score', 0):.1f}/100")

            # Worst performers
            worst_performers = report.get('worst_performers', [])
            if worst_performers:
                self.logger.info("⚠️ WORST PERFORMERS:")
                for i, performer in enumerate(worst_performers, 1):
                    self.logger.info(f"   {i}. {performer['function_name']} (Score: {performer['performance_score']:.1f})")
                    self.logger.info(f"      Execution Time: {performer['execution_time']:.3f}s")
                    if performer['bottlenecks']:
                        self.logger.info(f"      Bottlenecks: {', '.join(performer['bottlenecks'])}")

            # Function analysis
            function_analysis = report.get('function_analysis', {})
            if function_analysis:
                self.logger.info("🔍 FUNCTION ANALYSIS:")
                for function_name, analysis in function_analysis.items():
                    self.logger.info(f"   {function_name}:")
                    self.logger.info(f"     Calls: {analysis['total_calls']}")
                    self.logger.info(f"     Avg Execution Time: {analysis['average_execution_time']:.3f}s")
                    self.logger.info(f"     Avg Memory Usage: {analysis['average_memory_usage']:.1f}MB")
                    self.logger.info(f"     Avg Performance Score: {analysis['average_performance_score']:.1f}/100")

                    if analysis['most_common_bottlenecks']:
                        self.logger.info(f"     Common Bottlenecks: {', '.join([b[0] for b in analysis['most_common_bottlenecks']])}")

        except Exception as e:
            self.logger.error(f"Failed to log performance report: {e}")

def performance_monitor(monitor: 'PerformanceMonitor'):
    """Decorator for performance monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate call ID
            call_id = f"{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Start performance monitoring
            perf_record = monitor.start_performance_monitoring(func.__name__, call_id)

            try:
                result = func(*args, **kwargs)
                # End performance monitoring
                monitor.end_performance_monitoring(perf_record)
                return result
            except Exception as e:
                # End performance monitoring even on error
                monitor.end_performance_monitoring(perf_record)
                raise

        return wrapper
    return decorator
