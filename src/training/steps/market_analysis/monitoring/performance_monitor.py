from src.utils.tprint import tprint

from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

"""Performance Monitoring System.

This module provides comprehensive performance monitoring for function calls.
"""
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List

from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls,
    log_internal_call, log_step_progress, log_data_operation
)

import asyncio

try:
    import psutil
    import time
except ImportError:
    psutil = None

class PerformanceMonitor:
    """Comprehensive performance monitoring system for function calls."""

    @log_important_calls
    def __init__(self, logger: Any = None):
        # 🖨️ THOROUGH PRINTING: Performance Monitor Initialization
        tprint("🔧 INITIALIZING PERFORMANCE MONITOR")
        tprint(f"   📊 Logger provided: {logger is not None}")

        self.logger = logger or logging.getLogger(__name__)
        self.performance_history: List[Dict[str, Any]] = []
        self.function_performance_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_thresholds: Dict[str, float] = {}
        self.memory_usage_history: List[Dict[str, Any]] = []
        self.cpu_usage_history: List[Dict[str, Any]] = []

        tprint("   ✅ Performance history initialized")
        tprint("   ✅ Function performance stats initialized")
        tprint("   ✅ Performance thresholds initialized")
        tprint("   ✅ Memory usage history initialized")
        tprint("   ✅ CPU usage history initialized")
        tprint("   🎉 Performance monitor initialization complete")

    def start_performance_monitoring(self, function_name: str, call_id: str) -> Dict[str, Any]:
        """Start performance monitoring for a function call."""
        tprint(f"🚀 STARTING PERFORMANCE MONITORING")
        tprint(f"   📋 Function name: {function_name}")
        tprint(f"   🔗 Call ID: {call_id}")

        try:
            # Get initial system metrics
            tprint("   📊 Getting initial system metrics...")
            initial_metrics = self._get_system_metrics()
            tprint(f"   ✅ Initial metrics obtained: {bool(initial_metrics)}")

            performance_record = {
                'function_name': function_name,
                'call_id': call_id,
                'start_time': datetime.now(),
                'start_metrics': initial_metrics,
                'end_time': None,
                'end_metrics': None,
                'execution_time': 0.0,
                'memory_delta_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'performance_score': 0.0,
                'bottlenecks': [],
                'optimization_suggestions': []
            }

            return performance_record

        except Exception as e:
            self.logger.error(f"❌ Failed to start performance monitoring: {e}")
            return {}

    def end_performance_monitoring(self, performance_record: Dict[str, Any]) -> Dict[str, Any]:
        """End performance monitoring and calculate metrics."""
        try:
            if not performance_record:
                return {}

            # Get final system metrics
            final_metrics = self._get_system_metrics()
            performance_record['end_time'] = datetime.now()
            performance_record['end_metrics'] = final_metrics

            # Calculate execution time
            if performance_record['start_time'] and performance_record['end_time']:
                performance_record['execution_time'] = (
                    performance_record['end_time'] - performance_record['start_time']
                ).total_seconds()

            # Calculate memory delta
            if (performance_record['start_metrics'] and performance_record['end_metrics'] and
                'memory_mb' in performance_record['start_metrics'] and
                'memory_mb' in performance_record['end_metrics']):
                performance_record['memory_delta_mb'] = (
                    performance_record['end_metrics']['memory_mb'] -
                    performance_record['start_metrics']['memory_mb']
                )

            # Calculate CPU usage
            if (performance_record['start_metrics'] and performance_record['end_metrics'] and
                'cpu_percent' in performance_record['start_metrics'] and
                'cpu_percent' in performance_record['end_metrics']):
                performance_record['cpu_usage_percent'] = (
                    performance_record['end_metrics']['cpu_percent'] -
                    performance_record['start_metrics']['cpu_percent']
                )

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
            self.logger.error(f"❌ Failed to end performance monitoring: {e}")
            return performance_record

    @log_all_calls
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            metrics = {}

            # Memory usage
            if psutil:
                process = psutil.Process()
                memory_info = process.memory_info()
                metrics['memory_mb'] = memory_info.rss / 1024 / 1024  # Convert to MB
                metrics['memory_percent'] = process.memory_percent()

            # CPU usage
            if psutil:
                metrics['cpu_percent'] = psutil.cpu_percent()

            # System load
            if psutil:
                metrics['load_average'] = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get system metrics: {e}")
            return {}

    @log_all_calls
    def _calculate_performance_score(self, performance_record: Dict[str, Any]) -> float:
        """Calculate performance score based on execution time, memory usage, and CPU usage."""
        try:
            score = 100.0  # Start with perfect score

            # Execution time penalty
            execution_time = performance_record.get('execution_time', 0)
            if execution_time > 60:  # More than 1 minute
                score -= min(30, (execution_time - 60) * 0.5)
            elif execution_time > 10:  # More than 10 seconds
                score -= min(20, (execution_time - 10) * 2)

            # Memory usage penalty
            memory_delta = abs(performance_record.get('memory_delta_mb', 0))
            if memory_delta > 1000:  # More than 1GB
                score -= min(25, (memory_delta - 1000) * 0.025)
            elif memory_delta > 100:  # More than 100MB
                score -= min(15, (memory_delta - 100) * 0.15)

            # CPU usage penalty
            cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
            if cpu_usage > 80:  # More than 80% CPU
                score -= min(20, (cpu_usage - 80) * 0.5)
            elif cpu_usage > 50:  # More than 50% CPU
                score -= min(10, (cpu_usage - 50) * 0.33)

            return max(0, score)  # Ensure score doesn't go below 0

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate performance score: {e}")
            return 50.0  # Default score

    @log_all_calls
    def _identify_bottlenecks(self, performance_record: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        execution_time = performance_record.get('execution_time', 0)
        memory_delta = abs(performance_record.get('memory_delta_mb', 0))
        cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))

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

        return bottlenecks

    @log_all_calls
    def _generate_optimization_suggestions(self, performance_record: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on performance metrics."""
        suggestions = []

        execution_time = performance_record.get('execution_time', 0)
        memory_delta = abs(performance_record.get('memory_delta_mb', 0))
        cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
        function_name = performance_record.get('function_name', '')

        if execution_time > 30:
            suggestions.extend([
                "Consider breaking down the function into smaller, more manageable parts",
                "Implement caching for repeated computations",
                "Use vectorized operations instead of loops where possible"
            ])

        if memory_delta > 500:
            suggestions.extend([
                "Process data in smaller chunks to reduce memory footprint",
                "Use memory-efficient data types (e.g., float32 instead of float64)",
                "Clear unused variables and objects explicitly"
            ])

        if cpu_usage > 70:
            suggestions.extend([
                "Consider parallel processing for independent operations",
                "Optimize algorithms for better time complexity",
                "Use more efficient data structures"
            ])

        # Function-specific suggestions
        if 'labeling' in function_name.lower():
            suggestions.extend([
                "Consider using vectorized labeling operations",
                "Implement early termination for labeling loops",
                "Use efficient data structures for label storage"
            ])
        elif 'regime' in function_name.lower():
            suggestions.extend([
                "Cache regime detection results",
                "Use efficient regime transition algorithms",
                "Optimize regime-specific computations"
            ])

        return suggestions

    @log_all_calls
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
                    'execution_times': [],
                    'memory_usages': [],
                    'cpu_usages': [],
                    'performance_scores': [],
                    'bottlenecks': {},
                    'optimization_suggestions': set()
                }

            stats = self.function_performance_stats[function_name]
            stats['total_calls'] += 1
            stats['total_execution_time'] += performance_record.get('execution_time', 0)
            stats['total_memory_usage'] += abs(performance_record.get('memory_delta_mb', 0))
            stats['total_cpu_usage'] += abs(performance_record.get('cpu_usage_percent', 0))

            stats['execution_times'].append(performance_record.get('execution_time', 0))
            stats['memory_usages'].append(abs(performance_record.get('memory_delta_mb', 0)))
            stats['cpu_usages'].append(abs(performance_record.get('cpu_usage_percent', 0)))
            stats['performance_scores'].append(performance_record.get('performance_score', 0))

            # Update bottlenecks
            for bottleneck in performance_record.get('bottlenecks', []):
                stats['bottlenecks'][bottleneck] = stats['bottlenecks'].get(bottleneck, 0) + 1

            # Update optimization suggestions
            for suggestion in performance_record.get('optimization_suggestions', []):
                stats['optimization_suggestions'].add(suggestion)

        except Exception as e:
            self.logger.error(f"❌ Failed to update function performance stats: {e}")

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

            # Performance scores
            performance_scores = [record.get('performance_score', 0) for record in self.performance_history]
            avg_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0

            # Identify worst performers
            worst_performers = sorted(
                self.performance_history,
                key = lambda x: x.get('performance_score', 0)
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
                        'average_performance_score': sum(stats['performance_scores']) / len(stats['performance_scores']),
                        'most_common_bottlenecks': sorted(
                            stats['bottlenecks'].items(),
                            key = lambda x: x[1],
                            reverse = True
                        )[:3],
                        'optimization_suggestions': list(stats['optimization_suggestions'])[:5]
                    }

            return {
                'total_monitored_calls': total_calls,
                'overall_statistics': {
                    'total_execution_time': total_execution_time,
                    'total_memory_usage': total_memory_usage,
                    'total_cpu_usage': total_cpu_usage,
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
            self.logger.error(f"❌ Failed to generate performance report: {e}")
            return {}

    @log_all_calls
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if len(self.performance_history) < 2:
                return {'trend': 'insufficient_data'}

            # Sort by start time
            sorted_history = sorted(self.performance_history, key = lambda x: x['start_time'])

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
            self.logger.error(f"❌ Failed to analyze performance trends: {e}")
            return {}

    def log_performance_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive performance report."""
        try:
            if report.get('total_monitored_calls', 0) == 0:
                self.logger.info("📊 No performance data recorded")
                return

            self.logger.info("📊 PERFORMANCE MONITORING REPORT")
            self.logger.info("=" * 50)
            self.logger.info(f"Total Monitored Calls: {report['total_monitored_calls']}")

            # Overall statistics
            overall_stats = report.get('overall_statistics', {})
            if overall_stats:
                self.logger.info(f"\n📈 OVERALL STATISTICS:")
                self.logger.info(f"   Total Execution Time: {overall_stats.get('total_execution_time', 0):.3f}s")
                self.logger.info(f"   Total Memory Usage: {overall_stats.get('total_memory_usage', 0):.1f}MB")
                self.logger.info(f"   Total CPU Usage: {overall_stats.get('total_cpu_usage', 0):.1f}%")
                self.logger.info(f"   Average Performance Score: {overall_stats.get('average_performance_score', 0):.1f}/100")

            # Worst performers
            worst_performers = report.get('worst_performers', [])
            if worst_performers:
                self.logger.info(f"\n⚠️ WORST PERFORMERS:")
                for i, performer in enumerate(worst_performers, 1):
                    self.logger.info(f"   {i}. {performer['function_name']} (Score: {performer['performance_score']:.1f})")
                    self.logger.info(f"      Execution Time: {performer['execution_time']:.3f}s")
                    if performer['bottlenecks']:
                        self.logger.info(f"      Bottlenecks: {', '.join(performer['bottlenecks'])}")

            # Function analysis
            function_analysis = report.get('function_analysis', {})
            if function_analysis:
                self.logger.info(f"\n🔍 FUNCTION ANALYSIS:")
                for function_name, analysis in function_analysis.items():
                    self.logger.info(f"   {function_name}:")
                    self.logger.info(f"     Calls: {analysis['total_calls']}")
                    self.logger.info(f"     Avg Execution Time: {analysis['average_execution_time']:.3f}s")
                    self.logger.info(f"     Avg Memory Usage: {analysis['average_memory_usage']:.1f}MB")
                    self.logger.info(f"     Avg Performance Score: {analysis['average_performance_score']:.1f}/100")

                    if analysis['most_common_bottlenecks']:
                        self.logger.info(f"     Common Bottlenecks: {', '.join([b[0] for b in analysis['most_common_bottlenecks']])}")

            # Performance trends
            trends = report.get('performance_trends', {})
            if trends:
                self.logger.info(f"\n📊 PERFORMANCE TRENDS:")
                self.logger.info(f"   Execution Time Trend: {trends.get('execution_time_trend', 'unknown')}")
                self.logger.info(f"   Performance Score Trend: {trends.get('performance_score_trend', 'unknown')}")

        except Exception as e:
            self.logger.error(f"❌ Failed to log performance report: {e}")

def performance_monitor(monitor: PerformanceMonitor):
    """Decorator for performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate call ID
            call_id = f"{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Start performance monitoring
            perf_record = monitor.start_performance_monitoring(func.__name__, call_id)

            try:
                result = await func(*args, **kwargs)
                # End performance monitoring
                monitor.end_performance_monitoring(perf_record)
                return result
            except Exception as e:
                # End performance monitoring even on error
                monitor.end_performance_monitoring(perf_record)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
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

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
