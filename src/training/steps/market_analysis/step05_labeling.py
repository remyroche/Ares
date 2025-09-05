from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

"""Step 5: Labeling with Standardized Data Quality Management.

This module creates comprehensive labels for the training data, combining triple barrier
labels with additional labeling strategies and meta-labeling features.

Key Enhancements:
- Dynamic Label Generation: Added the ability to generate triple barrier labels directly within step05 using regime-aware methods
- Regime-Aware Triple Barrier: Integrated HMM regime-specific barrier optimization for more sophisticated labeling
- Fallback Mechanisms: Implemented robust fallback to default labeling when regime-aware methods aren't available
- Configuration-Driven Behavior: Added configurable toggles for automatic barrier recalculation
- Comprehensive Function Call Monitoring: Enhanced with detailed function call validation, tracking, and reporting
- Function-to-Function Call Tracking: Monitors all inter-function calls with detailed outcome reporting
- Detailed Completion Reporting: Provides comprehensive reports on function execution outcomes
"""
import logging
from src.utils.decorators import handles_errors, traced, validates, cached, log_execution_time
from src.utils.enhanced_error_handler import EnhancedErrorHandler, handle_errors_with_tracking
import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
from datetime import datetime
import json
import hashlib
import numpy as np
import pandas as pd
import traceback
import inspect
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import re
import os
import gc
from collections import defaultdict, Counter
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import pipeline_standards
from src.utils.logger import system_logger
import psutil

class FunctionCallStatus(Enum):
    """Status of function call execution."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    TIMEOUT = 'timeout'
    CANCELLED = 'cancelled'

@dataclass
class FunctionCallRecord:
    """Record of a function call with comprehensive metadata."""
    function_name: str
    call_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: FunctionCallStatus = FunctionCallStatus.PENDING
    input_args: Dict[str, Any] = field(default_factory=dict)
    input_kwargs: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    exception: Optional[Exception] = None
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    called_functions: List[str] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_details: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class FunctionCallReport:
    """Comprehensive report of function call execution."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    function_call_details: List[FunctionCallRecord] = field(default_factory=list)
    performance_summary: Dict[str, float] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)
    validation_summary: Dict[str, bool] = field(default_factory=dict)

class FunctionCallMonitor:
    """Comprehensive function call monitoring and validation system."""

    def __init__(self, logger: Any=None) -> None:
        self.logger = logger or create_fallback_logger()
        self.active_calls: Dict[str, FunctionCallRecord] = {}
        self.call_history: List[FunctionCallRecord] = []
        self.function_call_counter = 0
        self.validation_rules: Dict[str, Callable] = {}
        self.performance_thresholds: Dict[str, float] = {}

    def generate_call_id(self, function_name: str) -> str:
        """Generate unique call ID for function call tracking."""
        self.function_call_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f'{function_name}_{timestamp}_{self.function_call_counter}'

    def start_function_call(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Start monitoring a function call."""
        call_id = self.generate_call_id(function_name)
        input_args = {}
        input_kwargs = {}
        try:
            input_args = {f'arg_{i}': str(arg)[:100] for i, arg in enumerate(args)}
            input_kwargs = {k: str(v)[:100] for k, v in kwargs.items()}
        except Exception as e:
            self.logger.warning(f'⚠️ Could not capture function arguments: {e}')
        call_record = FunctionCallRecord(function_name=function_name, call_id=call_id, start_time=datetime.now(), status=FunctionCallStatus.IN_PROGRESS, input_args=input_args, input_kwargs=input_kwargs)
        self.active_calls[call_id] = call_record
        self.logger.info(f'🚀 Starting function call: {function_name} (ID: {call_id})')
        return call_id

    def end_function_call(self, call_id: str, return_value: Any=None, exception: Exception=None) -> FunctionCallRecord:
        """End monitoring a function call and record results."""
        if call_id not in self.active_calls:
            self.logger.error(f'❌ Call ID {call_id} not found in active calls')
            return None
        call_record = self.active_calls[call_id]
        call_record.end_time = datetime.now()
        call_record.execution_time = (call_record.end_time - call_record.start_time).total_seconds()
        call_record.return_value = return_value
        call_record.exception = exception
        if exception:
            call_record.status = FunctionCallStatus.FAILED
            call_record.error_details = str(exception)
            call_record.stack_trace = traceback.format_exc()
            self.logger.error(f'❌ Function call failed: {call_record.function_name} (ID: {call_id}) - {exception}')
        else:
            call_record.status = FunctionCallStatus.COMPLETED
            self.logger.info(f'✅ Function call completed: {call_record.function_name} (ID: {call_id}) in {call_record.execution_time:.3f}s')
        del self.active_calls[call_id]
        self.call_history.append(call_record)
        return call_record

    def record_function_to_function_call(self, parent_call_id: str, child_function_name: str) -> None:
        """Record a function-to-function call relationship."""
        if parent_call_id in self.active_calls:
            self.active_calls[parent_call_id].called_functions.append(child_function_name)
            self.logger.debug(f'🔗 Function call chain: {self.active_calls[parent_call_id].function_name} -> {child_function_name}')

    def validate_function_call(self, call_record: FunctionCallRecord) -> Dict[str, bool]:
        """Validate function call against defined rules."""
        validation_results = {}
        if call_record.function_name in self.performance_thresholds:
            threshold = self.performance_thresholds[call_record.function_name]
            validation_results['performance'] = call_record.execution_time <= threshold
            if not validation_results['performance']:
                self.logger.warning(f'⚠️ Performance threshold exceeded for {call_record.function_name}: {call_record.execution_time:.3f}s > {threshold}s')
        if call_record.function_name in self.validation_rules:
            try:
                validation_results['custom'] = self.validation_rules[call_record.function_name](call_record)
            except Exception as e:
                self.logger.error(f'❌ Custom validation failed for {call_record.function_name}: {e}')
                validation_results['custom'] = False
        validation_results['has_return_value'] = call_record.return_value is not None
        validation_results['no_exception'] = call_record.exception is None
        validation_results['reasonable_execution_time'] = 0 < call_record.execution_time < 3600
        call_record.validation_results = validation_results
        return validation_results

    def generate_comprehensive_report(self) -> FunctionCallReport:
        """Generate comprehensive function call report."""
        if not self.call_history:
            return FunctionCallReport()
        total_calls = len(self.call_history)
        successful_calls = len([c for c in self.call_history if c.status == FunctionCallStatus.COMPLETED])
        failed_calls = len([c for c in self.call_history if c.status == FunctionCallStatus.FAILED])
        total_execution_time = sum((c.execution_time for c in self.call_history))
        average_execution_time = total_execution_time / total_calls if total_calls > 0 else 0.0
        performance_summary = {'total_execution_time': total_execution_time, 'average_execution_time': average_execution_time, 'max_execution_time': max((c.execution_time for c in self.call_history)), 'min_execution_time': min((c.execution_time for c in self.call_history))}
        error_summary = {}
        for call in self.call_history:
            if call.exception:
                error_type = type(call.exception).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        validation_summary = {}
        for call in self.call_history:
            for validation_name, result in call.validation_results.items():
                if validation_name not in validation_summary:
                    validation_summary[validation_name] = {'passed': 0, 'failed': 0}
                if result:
                    validation_summary[validation_name]['passed'] += 1
                else:
                    validation_summary[validation_name]['failed'] += 1
        return FunctionCallReport(total_calls=total_calls, successful_calls=successful_calls, failed_calls=failed_calls, total_execution_time=total_execution_time, average_execution_time=average_execution_time, function_call_details=self.call_history.copy(), performance_summary=performance_summary, error_summary=error_summary, validation_summary=validation_summary)

    def log_detailed_report(self, report: FunctionCallReport) -> None:
        """Log detailed function call report."""
        self.logger.info('📊 COMPREHENSIVE FUNCTION CALL REPORT')
        self.logger.info('=' * 50)
        self.logger.info(f'Total Function Calls: {report.total_calls}')
        self.logger.info(f'Successful Calls: {report.successful_calls}')
        self.logger.info(f'Failed Calls: {report.failed_calls}')
        self.logger.info(f'Success Rate: {report.successful_calls / report.total_calls * 100:.1f}%' if report.total_calls > 0 else 'N/A')
        self.logger.info(f'Total Execution Time: {report.total_execution_time:.3f}s')
        self.logger.info(f'Average Execution Time: {report.average_execution_time:.3f}s')
        if report.performance_summary:
            self.logger.info('\n📈 PERFORMANCE SUMMARY:')
            for metric, value in report.performance_summary.items():
                self.logger.info(f'  {metric}: {value:.3f}s')
        if report.error_summary:
            self.logger.info('\n❌ ERROR SUMMARY:')
            for error_type, count in report.error_summary.items():
                self.logger.info(f'  {error_type}: {count} occurrences')
        if report.validation_summary:
            self.logger.info('\n✅ VALIDATION SUMMARY:')
            for validation_name, results in report.validation_summary.items():
                total = results['passed'] + results['failed']
                pass_rate = results['passed'] / total * 100 if total > 0 else 0
                self.logger.info(f"  {validation_name}: {results['passed']}/{total} passed ({pass_rate:.1f}%)")
        self.logger.info('\n🔍 DETAILED FUNCTION CALLS:')
        for call in report.function_call_details:
            status_emoji = '✅' if call.status == FunctionCallStatus.COMPLETED else '❌'
            self.logger.info(f'  {status_emoji} {call.function_name} (ID: {call.call_id})')
            self.logger.info(f'    Status: {call.status.value}')
            self.logger.info(f'    Execution Time: {call.execution_time:.3f}s')
            if call.called_functions:
                self.logger.info(f"    Called Functions: {', '.join(call.called_functions)}")
            if call.validation_results:
                validation_status = '✅' if all(call.validation_results.values()) else '⚠️'
                self.logger.info(f'    Validation: {validation_status} {call.validation_results}')
            if call.exception:
                self.logger.info(f'    Error: {call.error_details}')

# Removed custom comprehensive_function_monitor - using standardized decorators

def function_to_function_tracker(monitor: FunctionCallMonitor, parent_call_id: str=None) -> None:
    """Decorator for tracking function-to-function calls."""

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            if parent_call_id:
                monitor.record_function_to_function_call(parent_call_id, func.__name__)
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            if parent_call_id:
                monitor.record_function_to_function_call(parent_call_id, func.__name__)
            return func(*args, **kwargs)
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Using standardized EnhancedErrorHandler from utils

class PerformanceMonitor:
    """Comprehensive performance monitoring system for function calls."""

    def __init__(self, logger: Any=None) -> None:
        self.logger = logger or create_fallback_logger()
        self.performance_history: List[Dict[str, Any]] = []
        self.function_performance_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_thresholds: Dict[str, float] = {}
        self.memory_usage_history: List[Dict[str, Any]] = []
        self.cpu_usage_history: List[Dict[str, Any]] = []

    def start_performance_monitoring(self, function_name: str, call_id: str) -> Dict[str, Any]:
        """Start performance monitoring for a function call."""
        try:
            initial_metrics = self._get_system_metrics()
            performance_record = {'function_name': function_name, 'call_id': call_id, 'start_time': datetime.now(), 'start_metrics': initial_metrics, 'end_time': None, 'end_metrics': None, 'execution_time': 0.0, 'memory_delta_mb': 0.0, 'cpu_usage_percent': 0.0, 'performance_score': 0.0, 'bottlenecks': [], 'optimization_suggestions': []}
            return performance_record
        except Exception as e:
            self.logger.error(f'❌ Failed to start performance monitoring: {e}')
            return {}

    def end_performance_monitoring(self, performance_record: Dict[str, Any]) -> Dict[str, Any]:
        """End performance monitoring and calculate metrics."""
        try:
            if not performance_record:
                return {}
            final_metrics = self._get_system_metrics()
            performance_record['end_time'] = datetime.now()
            performance_record['end_metrics'] = final_metrics
            if performance_record['start_time'] and performance_record['end_time']:
                performance_record['execution_time'] = (performance_record['end_time'] - performance_record['start_time']).total_seconds()
            if performance_record['start_metrics'] and performance_record['end_metrics'] and ('memory_mb' in performance_record['start_metrics']) and ('memory_mb' in performance_record['end_metrics']):
                performance_record['memory_delta_mb'] = performance_record['end_metrics']['memory_mb'] - performance_record['start_metrics']['memory_mb']
            if performance_record['start_metrics'] and performance_record['end_metrics'] and ('cpu_percent' in performance_record['start_metrics']) and ('cpu_percent' in performance_record['end_metrics']):
                performance_record['cpu_usage_percent'] = performance_record['end_metrics']['cpu_percent'] - performance_record['start_metrics']['cpu_percent']
            performance_record['performance_score'] = self._calculate_performance_score(performance_record)
            performance_record['bottlenecks'] = self._identify_bottlenecks(performance_record)
            performance_record['optimization_suggestions'] = self._generate_optimization_suggestions(performance_record)
            self._update_function_performance_stats(performance_record)
            self.performance_history.append(performance_record)
            return performance_record
        except Exception as e:
            self.logger.error(f'❌ Failed to end performance monitoring: {e}')
            return performance_record

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            metrics = {}
            if psutil:
                process = psutil.Process()
                memory_info = process.memory_info()
                metrics['memory_mb'] = memory_info.rss / 1024 / 1024
                metrics['memory_percent'] = process.memory_percent()
            if psutil:
                metrics['cpu_percent'] = psutil.cpu_percent()
            if psutil:
                metrics['load_average'] = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            return metrics
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to get system metrics: {e}')
            return {}

    def _calculate_performance_score(self, performance_record: Dict[str, Any]) -> float:
        """Calculate performance score based on execution time, memory usage, and CPU usage."""
        try:
            score = 100.0
            execution_time = performance_record.get('execution_time', 0)
            if execution_time > 60:
                score -= min(30, (execution_time - 60) * 0.5)
            elif execution_time > 10:
                score -= min(20, (execution_time - 10) * 2)
            memory_delta = abs(performance_record.get('memory_delta_mb', 0))
            if memory_delta > 1000:
                score -= min(25, (memory_delta - 1000) * 0.025)
            elif memory_delta > 100:
                score -= min(15, (memory_delta - 100) * 0.15)
            cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
            if cpu_usage > 80:
                score -= min(20, (cpu_usage - 80) * 0.5)
            elif cpu_usage > 50:
                score -= min(10, (cpu_usage - 50) * 0.33)
            return max(0, score)
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to calculate performance score: {e}')
            return 50.0

    def _identify_bottlenecks(self, performance_record: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        execution_time = performance_record.get('execution_time', 0)
        memory_delta = abs(performance_record.get('memory_delta_mb', 0))
        cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
        if execution_time > 60:
            bottlenecks.append('Long execution time (>60s)')
        elif execution_time > 10:
            bottlenecks.append('Moderate execution time (>10s)')
        if memory_delta > 1000:
            bottlenecks.append('High memory usage (>1GB)')
        elif memory_delta > 100:
            bottlenecks.append('Moderate memory usage (>100MB)')
        if cpu_usage > 80:
            bottlenecks.append('High CPU usage (>80%)')
        elif cpu_usage > 50:
            bottlenecks.append('Moderate CPU usage (>50%)')
        return bottlenecks

    def _generate_optimization_suggestions(self, performance_record: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on performance metrics."""
        suggestions = []
        execution_time = performance_record.get('execution_time', 0)
        memory_delta = abs(performance_record.get('memory_delta_mb', 0))
        cpu_usage = abs(performance_record.get('cpu_usage_percent', 0))
        function_name = performance_record.get('function_name', '')
        if execution_time > 30:
            suggestions.extend(['Consider breaking down the function into smaller, more manageable parts', 'Implement caching for repeated computations', 'Use vectorized operations instead of loops where possible'])
        if memory_delta > 500:
            suggestions.extend(['Process data in smaller chunks to reduce memory footprint', 'Use memory-efficient data types (e.g., float32 instead of float64)', 'Clear unused variables and objects explicitly'])
        if cpu_usage > 70:
            suggestions.extend(['Consider parallel processing for independent operations', 'Optimize algorithms for better time complexity', 'Use more efficient data structures'])
        if 'labeling' in function_name.lower():
            suggestions.extend(['Consider using vectorized labeling operations', 'Implement early termination for labeling loops', 'Use efficient data structures for label storage'])
        elif 'regime' in function_name.lower():
            suggestions.extend(['Cache regime detection results', 'Use efficient regime transition algorithms', 'Optimize regime-specific computations'])
        return suggestions

    def _update_function_performance_stats(self, performance_record: Dict[str, Any]) -> None:
        """Update function performance statistics."""
        try:
            function_name = performance_record['function_name']
            if function_name not in self.function_performance_stats:
                self.function_performance_stats[function_name] = {'total_calls': 0, 'total_execution_time': 0.0, 'total_memory_usage': 0.0, 'total_cpu_usage': 0.0, 'execution_times': [], 'memory_usages': [], 'cpu_usages': [], 'performance_scores': [], 'bottlenecks': {}, 'optimization_suggestions': set()}
            stats = self.function_performance_stats[function_name]
            stats['total_calls'] += 1
            stats['total_execution_time'] += performance_record.get('execution_time', 0)
            stats['total_memory_usage'] += abs(performance_record.get('memory_delta_mb', 0))
            stats['total_cpu_usage'] += abs(performance_record.get('cpu_usage_percent', 0))
            stats['execution_times'].append(performance_record.get('execution_time', 0))
            stats['memory_usages'].append(abs(performance_record.get('memory_delta_mb', 0)))
            stats['cpu_usages'].append(abs(performance_record.get('cpu_usage_percent', 0)))
            stats['performance_scores'].append(performance_record.get('performance_score', 0))
            for bottleneck in performance_record.get('bottlenecks', []):
                stats['bottlenecks'][bottleneck] = stats['bottlenecks'].get(bottleneck, 0) + 1
            for suggestion in performance_record.get('optimization_suggestions', []):
                stats['optimization_suggestions'].add(suggestion)
        except Exception as e:
            self.logger.error(f'❌ Failed to update function performance stats: {e}')

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            if not self.performance_history:
                return {'total_monitored_calls': 0, 'message': 'No performance data recorded'}
            total_calls = len(self.performance_history)
            total_execution_time = sum((record.get('execution_time', 0) for record in self.performance_history))
            total_memory_usage = sum((abs(record.get('memory_delta_mb', 0)) for record in self.performance_history))
            total_cpu_usage = sum((abs(record.get('cpu_usage_percent', 0)) for record in self.performance_history))
            performance_scores = [record.get('performance_score', 0) for record in self.performance_history]
            avg_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0
            worst_performers = sorted(self.performance_history, key=lambda x: x.get('performance_score', 0))[:5]
            function_analysis = {}
            for function_name, stats in self.function_performance_stats.items():
                if stats['total_calls'] > 0:
                    function_analysis[function_name] = {'total_calls': stats['total_calls'], 'average_execution_time': stats['total_execution_time'] / stats['total_calls'], 'average_memory_usage': stats['total_memory_usage'] / stats['total_calls'], 'average_cpu_usage': stats['total_cpu_usage'] / stats['total_calls'], 'average_performance_score': sum(stats['performance_scores']) / len(stats['performance_scores']), 'most_common_bottlenecks': sorted(stats['bottlenecks'].items(), key=lambda x: x[1], reverse=True)[:3], 'optimization_suggestions': list(stats['optimization_suggestions'])[:5]}
            return {'total_monitored_calls': total_calls, 'overall_statistics': {'total_execution_time': total_execution_time, 'total_memory_usage': total_memory_usage, 'total_cpu_usage': total_cpu_usage, 'average_performance_score': avg_performance_score}, 'worst_performers': [{'function_name': record['function_name'], 'call_id': record['call_id'], 'performance_score': record.get('performance_score', 0), 'execution_time': record.get('execution_time', 0), 'bottlenecks': record.get('bottlenecks', [])} for record in worst_performers], 'function_analysis': function_analysis, 'performance_trends': self._analyze_performance_trends()}
        except Exception as e:
            self.logger.error(f'❌ Failed to generate performance report: {e}')
            return {}

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            if len(self.performance_history) < 2:
                return {'trend': 'insufficient_data'}
            sorted_history = sorted(self.performance_history, key=lambda x: x['start_time'])
            execution_times = [record.get('execution_time', 0) for record in sorted_history]
            if len(execution_times) > 1:
                time_trend = 'improving' if execution_times[-1] < execution_times[0] else 'degrading'
            else:
                time_trend = 'stable'
            performance_scores = [record.get('performance_score', 0) for record in sorted_history]
            if len(performance_scores) > 1:
                score_trend = 'improving' if performance_scores[-1] > performance_scores[0] else 'degrading'
            else:
                score_trend = 'stable'
            return {'execution_time_trend': time_trend, 'performance_score_trend': score_trend, 'data_points': len(sorted_history)}
        except Exception as e:
            self.logger.error(f'❌ Failed to analyze performance trends: {e}')
            return {}

    def log_performance_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive performance report."""
        try:
            if report.get('total_monitored_calls', 0) == 0:
                self.logger.info('📊 No performance data recorded')
                return
            self.logger.info('📊 PERFORMANCE MONITORING REPORT')
            self.logger.info('=' * 50)
            self.logger.info(f"Total Monitored Calls: {report['total_monitored_calls']}")
            overall_stats = report.get('overall_statistics', {})
            if overall_stats:
                self.logger.info(f'\n📈 OVERALL STATISTICS:')
                self.logger.info(f"   Total Execution Time: {overall_stats.get('total_execution_time', 0):.3f}s")
                self.logger.info(f"   Total Memory Usage: {overall_stats.get('total_memory_usage', 0):.1f}MB")
                self.logger.info(f"   Total CPU Usage: {overall_stats.get('total_cpu_usage', 0):.1f}%")
                self.logger.info(f"   Average Performance Score: {overall_stats.get('average_performance_score', 0):.1f}/100")
            worst_performers = report.get('worst_performers', [])
            if worst_performers:
                self.logger.info(f'\n⚠️ WORST PERFORMERS:')
                for i, performer in enumerate(worst_performers, 1):
                    self.logger.info(f"   {i}. {performer['function_name']} (Score: {performer['performance_score']:.1f})")
                    self.logger.info(f"      Execution Time: {performer['execution_time']:.3f}s")
                    if performer['bottlenecks']:
                        self.logger.info(f"      Bottlenecks: {', '.join(performer['bottlenecks'])}")
            function_analysis = report.get('function_analysis', {})
            if function_analysis:
                self.logger.info(f'\n🔍 FUNCTION ANALYSIS:')
                for function_name, analysis in function_analysis.items():
                    self.logger.info(f'   {function_name}:')
                    self.logger.info(f"     Calls: {analysis['total_calls']}")
                    self.logger.info(f"     Avg Execution Time: {analysis['average_execution_time']:.3f}s")
                    self.logger.info(f"     Avg Memory Usage: {analysis['average_memory_usage']:.1f}MB")
                    self.logger.info(f"     Avg Performance Score: {analysis['average_performance_score']:.1f}/100")
                    if analysis['most_common_bottlenecks']:
                        self.logger.info(f"     Common Bottlenecks: {', '.join([b[0] for b in analysis['most_common_bottlenecks']])}")
            trends = report.get('performance_trends', {})
            if trends:
                self.logger.info(f'\n📊 PERFORMANCE TRENDS:')
                self.logger.info(f"   Execution Time Trend: {trends.get('execution_time_trend', 'unknown')}")
                self.logger.info(f"   Performance Score Trend: {trends.get('performance_score_trend', 'unknown')}")
        except Exception as e:
            self.logger.error(f'❌ Failed to log performance report: {e}')

# Removed custom performance_monitor - using standardized decorators

class ComprehensiveValidationFramework:
    """Comprehensive validation framework for all function operations."""

    def __init__(self, logger: Any=None) -> None:
        self.logger = logger or create_fallback_logger()
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_validation_rules()

    def _initialize_default_validation_rules(self) -> None:
        """Initialize default validation rules for common operations."""
        try:
            self.validation_rules['input_validation'] = [self._validate_dataframe_input, self._validate_string_input, self._validate_numeric_input, self._validate_path_input]
            self.validation_rules['output_validation'] = [self._validate_dataframe_output, self._validate_boolean_output, self._validate_numeric_output, self._validate_series_output]
            self.validation_rules['data_quality'] = [self._validate_data_completeness, self._validate_data_types, self._validate_data_ranges, self._validate_data_consistency]
            self.validation_rules['performance_validation'] = [self._validate_execution_time, self._validate_memory_usage, self._validate_cpu_usage]
            self.validation_rules['business_logic'] = [self._validate_labeling_logic, self._validate_regime_logic, self._validate_triple_barrier_logic]
            self.logger.info('✅ Default validation rules initialized')
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize default validation rules: {e}')

    def _validate_dataframe_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate DataFrame input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f'Expected DataFrame, got {type(data).__name__}')
                return result
            if data.empty:
                result['valid'] = False
                result['errors'].append('DataFrame is empty')
                return result
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f'Missing required columns: {missing_columns}')
            critical_columns = context.get('critical_columns', [])
            for col in critical_columns:
                if col in data.columns and data[col].isna().any():
                    result['warnings'].append(f"Column '{col}' contains NaN values")
            expected_types = context.get('expected_types', {})
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                        result['warnings'].append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_string_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate string input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, str):
                result['valid'] = False
                result['errors'].append(f'Expected string, got {type(data).__name__}')
                return result
            if not data.strip():
                result['valid'] = False
                result['errors'].append('String is empty or whitespace only')
                return result
            min_length = context.get('min_length', 0)
            max_length = context.get('max_length', float('inf'))
            if len(data) < min_length:
                result['valid'] = False
                result['errors'].append(f'String too short (min: {min_length})')
            if len(data) > max_length:
                result['valid'] = False
                result['errors'].append(f'String too long (max: {max_length})')
            pattern = context.get('pattern')
            if pattern and (not re.match(pattern, data)):
                result['valid'] = False
                result['errors'].append(f"String doesn't match required pattern: {pattern}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_numeric_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate numeric input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f'Expected numeric, got {type(data).__name__}')
                return result
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            if data < min_value:
                result['valid'] = False
                result['errors'].append(f'Value too small (min: {min_value})')
            if data > max_value:
                result['valid'] = False
                result['errors'].append(f'Value too large (max: {max_value})')
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append('Value is NaN or infinite')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_path_input(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate path input."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            path = Path(data) if not isinstance(data, Path) else data
            must_exist = context.get('must_exist', True)
            if must_exist and (not path.exists()):
                result['valid'] = False
                result['errors'].append(f'Path does not exist: {path}')
                return result
            expected_type = context.get('expected_type', 'file')
            if path.exists():
                if expected_type == 'file' and (not path.is_file()):
                    result['valid'] = False
                    result['errors'].append(f'Expected file, got directory: {path}')
                elif expected_type == 'directory' and (not path.is_dir()):
                    result['valid'] = False
                    result['errors'].append(f'Expected directory, got file: {path}')
            expected_extensions = context.get('expected_extensions', [])
            if expected_extensions and path.suffix.lower() not in expected_extensions:
                result['valid'] = False
                result['errors'].append(f'Invalid file extension. Expected: {expected_extensions}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_dataframe_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate DataFrame output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append('Output is None')
                return result
            if not isinstance(data, pd.DataFrame):
                result['valid'] = False
                result['errors'].append(f'Expected DataFrame output, got {type(data).__name__}')
                return result
            if data.empty:
                result['warnings'].append('Output DataFrame is empty')
            required_columns = context.get('required_columns', [])
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f'Missing required output columns: {missing_columns}')
            if 'label' in data.columns:
                label_counts = data['label'].value_counts()
                if len(label_counts) == 0:
                    result['warnings'].append('No labels generated')
                elif len(label_counts) == 1:
                    result['warnings'].append('Only one label class generated')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_boolean_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate boolean output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, bool):
                result['valid'] = False
                result['errors'].append(f'Expected boolean output, got {type(data).__name__}')
                return result
            expected_value = context.get('expected_value')
            if expected_value is not None and data != expected_value:
                result['warnings'].append(f'Expected {expected_value}, got {data}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_numeric_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate numeric output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if not isinstance(data, (int, float, np.number)):
                result['valid'] = False
                result['errors'].append(f'Expected numeric output, got {type(data).__name__}')
                return result
            if np.isnan(data) or np.isinf(data):
                result['valid'] = False
                result['errors'].append('Output is NaN or infinite')
            min_value = context.get('min_value', float('-inf'))
            max_value = context.get('max_value', float('inf'))
            if data < min_value or data > max_value:
                result['warnings'].append(f'Output value {data} outside expected range [{min_value}, {max_value}]')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_series_output(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate Series output."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if data is None:
                result['valid'] = False
                result['errors'].append('Output is None')
                return result
            if not isinstance(data, pd.Series):
                result['valid'] = False
                result['errors'].append(f'Expected Series output, got {type(data).__name__}')
                return result
            if data.empty:
                result['warnings'].append('Output Series is empty')
            if data.isna().any():
                nan_count = data.isna().sum()
                result['warnings'].append(f'Output Series contains {nan_count} NaN values')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_completeness(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data completeness."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                total_cells = data.size
                missing_cells = data.isna().sum().sum()
                completeness_ratio = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
                min_completeness = context.get('min_completeness', 0.95)
                if completeness_ratio < min_completeness:
                    result['warnings'].append(f'Data completeness {completeness_ratio:.2%} below threshold {min_completeness:.2%}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_types(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data types."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                expected_types = context.get('expected_types', {})
                for col, expected_type in expected_types.items():
                    if col in data.columns:
                        actual_type = data[col].dtype
                        if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                            result['warnings'].append(f"Column '{col}' type mismatch: {actual_type} vs {expected_type}")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_ranges(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data ranges."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                column_ranges = context.get('column_ranges', {})
                for col, (min_val, max_val) in column_ranges.items():
                    if col in data.columns:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            if col_data.min() < min_val or col_data.max() > max_val:
                                result['warnings'].append(f"Column '{col}' values outside range [{min_val}, {max_val}]")
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_data_consistency(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate data consistency."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                if data.duplicated().any():
                    duplicate_count = data.duplicated().sum()
                    result['warnings'].append(f'Found {duplicate_count} duplicate rows')
                if 'close' in data.columns and 'high' in data.columns and ('low' in data.columns):
                    invalid_ohlc = (data['close'] > data['high']) | (data['close'] < data['low'])
                    if invalid_ohlc.any():
                        invalid_count = invalid_ohlc.sum()
                        result['warnings'].append(f'Found {invalid_count} rows with invalid OHLC relationships')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_execution_time(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate execution time."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            execution_time = context.get('execution_time', 0)
            max_time = context.get('max_execution_time', 300)
            if execution_time > max_time:
                result['warnings'].append(f'Execution time {execution_time:.2f}s exceeds threshold {max_time}s')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_memory_usage(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate memory usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            memory_usage = context.get('memory_usage_mb', 0)
            max_memory = context.get('max_memory_mb', 1000)
            if memory_usage > max_memory:
                result['warnings'].append(f'Memory usage {memory_usage:.1f}MB exceeds threshold {max_memory}MB')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_cpu_usage(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate CPU usage."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            cpu_usage = context.get('cpu_usage_percent', 0)
            max_cpu = context.get('max_cpu_percent', 80)
            if cpu_usage > max_cpu:
                result['warnings'].append(f'CPU usage {cpu_usage:.1f}% exceeds threshold {max_cpu}%')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_labeling_logic(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate labeling logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame) and 'label' in data.columns:
                labels = data['label'].dropna()
                if len(labels) > 0:
                    label_counts = labels.value_counts()
                    total_labels = len(labels)
                    if len(label_counts) > 1:
                        max_count = label_counts.max()
                        min_count = label_counts.min()
                        imbalance_ratio = max_count / min_count
                        if imbalance_ratio > 10:
                            result['warnings'].append(f'Severe class imbalance detected (ratio: {imbalance_ratio:.1f})')
                    for label, count in label_counts.items():
                        percentage = count / total_labels * 100
                        if percentage < 1:
                            result['warnings'].append(f'Very few samples for label {label} ({percentage:.1f}%)')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_regime_logic(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate regime logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                regime_columns = [col for col in data.columns if 'regime' in col.lower()]
                for regime_col in regime_columns:
                    regimes = data[regime_col].dropna()
                    if len(regimes) > 0:
                        regime_counts = regimes.value_counts()
                        if len(regime_counts) < 2:
                            result['warnings'].append(f'Only {len(regime_counts)} regime(s) detected in {regime_col}')
                        elif len(regime_counts) > 10:
                            result['warnings'].append(f'Too many regimes ({len(regime_counts)}) in {regime_col}')
                        if len(regime_counts) > 1:
                            max_count = regime_counts.max()
                            min_count = regime_counts.min()
                            balance_ratio = max_count / min_count
                            if balance_ratio > 5:
                                result['warnings'].append(f'Unbalanced regimes in {regime_col} (ratio: {balance_ratio:.1f})')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def _validate_triple_barrier_logic(self, data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate triple barrier logic."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        try:
            if isinstance(data, pd.DataFrame):
                tb_columns = [col for col in data.columns if 'triple_barrier' in col.lower()]
                for tb_col in tb_columns:
                    tb_labels = data[tb_col].dropna()
                    if len(tb_labels) > 0:
                        valid_labels = tb_labels.isin([-1, 0, 1])
                        if not valid_labels.all():
                            invalid_labels = tb_labels[~valid_labels].unique()
                            result['warnings'].append(f'Invalid triple barrier labels in {tb_col}: {invalid_labels}')
                        label_counts = tb_labels.value_counts()
                        total_labels = len(tb_labels)
                        neutral_count = label_counts.get(0, 0)
                        neutral_ratio = neutral_count / total_labels
                        if neutral_ratio > 0.8:
                            result['warnings'].append(f'Too many neutral labels in {tb_col} ({neutral_ratio:.1%})')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Validation error: {str(e)}')
        return result

    def validate_function_input(self, function_name: str, input_data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate function input using all applicable rules."""
        try:
            validation_result = {'function_name': function_name, 'validation_type': 'input', 'timestamp': datetime.now().isoformat(), 'overall_valid': True, 'rule_results': {}, 'errors': [], 'warnings': []}
            context = context or {}
            for rule_name, rules in self.validation_rules.items():
                if rule_name in ['input_validation', 'data_quality']:
                    for rule in rules:
                        try:
                            rule_result = rule(input_data, context)
                            validation_result['rule_results'][rule_name] = rule_result
                            if not rule_result['valid']:
                                validation_result['overall_valid'] = False
                                validation_result['errors'].extend(rule_result['errors'])
                            validation_result['warnings'].extend(rule_result['warnings'])
                        except Exception as e:
                            validation_result['errors'].append(f'Rule {rule_name} failed: {str(e)}')
                            validation_result['overall_valid'] = False
            self.validation_history.append(validation_result)
            return validation_result
        except Exception as e:
            self.logger.error(f'❌ Failed to validate function input: {e}')
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}

    def validate_function_output(self, function_name: str, output_data: Any, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Validate function output using all applicable rules."""
        try:
            validation_result = {'function_name': function_name, 'validation_type': 'output', 'timestamp': datetime.now().isoformat(), 'overall_valid': True, 'rule_results': {}, 'errors': [], 'warnings': []}
            context = context or {}
            for rule_name, rules in self.validation_rules.items():
                if rule_name in ['output_validation', 'data_quality', 'business_logic']:
                    for rule in rules:
                        try:
                            rule_result = rule(output_data, context)
                            validation_result['rule_results'][rule_name] = rule_result
                            if not rule_result['valid']:
                                validation_result['overall_valid'] = False
                                validation_result['errors'].extend(rule_result['errors'])
                            validation_result['warnings'].extend(rule_result['warnings'])
                        except Exception as e:
                            validation_result['errors'].append(f'Rule {rule_name} failed: {str(e)}')
                            validation_result['overall_valid'] = False
            self.validation_history.append(validation_result)
            return validation_result
        except Exception as e:
            self.logger.error(f'❌ Failed to validate function output: {e}')
            return {'overall_valid': False, 'errors': [str(e)], 'warnings': []}

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        try:
            if not self.validation_history:
                return {'total_validations': 0, 'message': 'No validation data recorded'}
            total_validations = len(self.validation_history)
            successful_validations = len([v for v in self.validation_history if v['overall_valid']])
            failed_validations = total_validations - successful_validations
            function_validations = {}
            for validation in self.validation_history:
                func_name = validation['function_name']
                if func_name not in function_validations:
                    function_validations[func_name] = {'input': [], 'output': []}
                function_validations[func_name][validation['validation_type']].append(validation)
            error_patterns = {}
            warning_patterns = {}
            for validation in self.validation_history:
                for error in validation['errors']:
                    error_patterns[error] = error_patterns.get(error, 0) + 1
                for warning in validation['warnings']:
                    warning_patterns[warning] = warning_patterns.get(warning, 0) + 1
            return {'total_validations': total_validations, 'successful_validations': successful_validations, 'failed_validations': failed_validations, 'success_rate': successful_validations / total_validations * 100 if total_validations > 0 else 0, 'function_validations': function_validations, 'error_patterns': error_patterns, 'warning_patterns': warning_patterns, 'most_common_errors': sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5], 'most_common_warnings': sorted(warning_patterns.items(), key=lambda x: x[1], reverse=True)[:5]}
        except Exception as e:
            self.logger.error(f'❌ Failed to generate validation report: {e}')
            return {}

    def log_validation_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive validation report."""
        try:
            if report.get('total_validations', 0) == 0:
                self.logger.info('📋 No validation data recorded')
                return
            self.logger.info('📋 COMPREHENSIVE VALIDATION REPORT')
            self.logger.info('=' * 50)
            self.logger.info(f"Total Validations: {report['total_validations']}")
            self.logger.info(f"Successful Validations: {report['successful_validations']}")
            self.logger.info(f"Failed Validations: {report['failed_validations']}")
            self.logger.info(f"Success Rate: {report['success_rate']:.1f}%")
            function_validations = report.get('function_validations', {})
            if function_validations:
                self.logger.info(f'\n🔍 FUNCTION VALIDATION RESULTS:')
                for func_name, validations in function_validations.items():
                    input_validations = validations.get('input', [])
                    output_validations = validations.get('output', [])
                    input_success = len([v for v in input_validations if v['overall_valid']])
                    output_success = len([v for v in output_validations if v['overall_valid']])
                    self.logger.info(f'   {func_name}:')
                    self.logger.info(f'     Input Validations: {input_success}/{len(input_validations)} successful')
                    self.logger.info(f'     Output Validations: {output_success}/{len(output_validations)} successful')
            most_common_errors = report.get('most_common_errors', [])
            if most_common_errors:
                self.logger.info(f'\n❌ MOST COMMON ERRORS:')
                for error, count in most_common_errors:
                    self.logger.info(f'   - {error}: {count} occurrences')
            most_common_warnings = report.get('most_common_warnings', [])
            if most_common_warnings:
                self.logger.info(f'\n⚠️ MOST COMMON WARNINGS:')
                for warning, count in most_common_warnings:
                    self.logger.info(f'   - {warning}: {count} occurrences')
        except Exception as e:
            self.logger.error(f'❌ Failed to log validation report: {e}')

# Removed custom comprehensive_validation - using standardized decorators

# Removed complex fallback logic - using standardized imports
logger = system_logger.getChild('Step5Labeling')

class LabelingStep:
    """Step 5: Labeling with standardized data quality management and regime-aware triple barrier method."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('LabelingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        self.function_monitor = FunctionCallMonitor(self.logger)
        self._setup_function_monitoring()
        self.error_handler = EnhancedErrorHandler(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.validation_framework = ComprehensiveValidationFramework(self.logger)
        self._validate_environment()
        self._initialize_components()

    def _setup_function_monitoring(self) -> None:
        """Setup comprehensive function call monitoring with validation rules and performance thresholds."""
        self.logger.info('🔧 Setting up comprehensive function call monitoring...')
        self.function_monitor.performance_thresholds = {'execute_labeling': 300.0, '_generate_comprehensive_labels': 180.0, '_generate_regime_aware_labels': 120.0, '_apply_triple_barrier_labels': 60.0, '_apply_meta_labels': 30.0, '_log_step5_artifacts_and_report': 15.0}
        self.function_monitor.validation_rules = {'execute_labeling': self._validate_execute_labeling_result, '_generate_comprehensive_labels': self._validate_labeling_result, '_generate_regime_aware_labels': self._validate_regime_labels_result, '_apply_triple_barrier_labels': self._validate_triple_barrier_result, '_apply_meta_labels': self._validate_meta_labels_result}
        self.logger.info('✅ Function call monitoring setup completed')

    def _validate_execute_labeling_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate execute_labeling function result."""
        if call_record.return_value is None:
            return False
        if not isinstance(call_record.return_value, bool):
            return False
        if call_record.return_value and call_record.exception:
            return False
        return True

    def _validate_labeling_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate labeling function result."""
        if call_record.return_value is None:
            return False
        if not isinstance(call_record.return_value, pd.DataFrame):
            return False
        required_columns = ['label']
        if not all((col in call_record.return_value.columns for col in required_columns)):
            return False
        if len(call_record.return_value) == 0:
            return False
        return True

    def _validate_regime_labels_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate regime labels function result."""
        if call_record.return_value is None:
            return False
        if not isinstance(call_record.return_value, pd.Series):
            return False
        if len(call_record.return_value) == 0:
            return False
        return True

    def _validate_triple_barrier_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate triple barrier function result."""
        if call_record.return_value is None:
            return False
        if not isinstance(call_record.return_value, pd.DataFrame):
            return False
        required_columns = ['triple_barrier_label']
        if not all((col in call_record.return_value.columns for col in required_columns)):
            return False
        return True

    def _validate_meta_labels_result(self, call_record: FunctionCallRecord) -> bool:
        """Validate meta labels function result."""
        if call_record.return_value is None:
            return False
        if not isinstance(call_record.return_value, pd.DataFrame):
            return False
        return True

    def _compute_labeling_fingerprint(self, triple_barrier_path: Path) -> Dict[str, Any]:
        """Compute a stable fingerprint of source labeling inputs to ensure idempotence.

        Uses source file size and mtime plus relevant config toggles.
        """
        try:
            stat = triple_barrier_path.stat()
            relevant_cfg = {'vectorized_labelling_orchestrator': self.config.get('vectorized_labelling_orchestrator', {}), 'labeling': self.config.get('labeling', {}), 'time_barrier_minutes': getattr(self, 'time_barrier_minutes', None), 'max_lookahead': getattr(self, 'max_lookahead', None), 'regime_col': getattr(self, 'regime_col', None), 'auto_recalculate_hmm_barriers': getattr(self, 'auto_recalculate_hmm_barriers', None)}
            relevant_cfg_json = json.dumps(relevant_cfg, sort_keys=True, default=str)
            cfg_hash = hashlib.sha256(relevant_cfg_json.encode('utf-8')).hexdigest()
            return {'source_path': str(triple_barrier_path), 'source_size': stat.st_size, 'source_mtime': int(stat.st_mtime), 'config_hash': cfg_hash}
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to compute labeling fingerprint: {e}')
            return {}

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Environment validation simplified - using standardized imports')

    def _initialize_components(self) -> None:
        """Initialize labeling components with regime-aware triple barrier support."""
        self.logger.info('🔧 Initializing labeling components...')
        labeling_cfg = self.config.get('vectorized_labelling_orchestrator', {})
        self.auto_recalculate_hmm_barriers = bool(labeling_cfg.get('auto_recalculate_hmm_barriers', True))
        try:
            from .utils.regime_data_access import get_regime_column
            detected = get_regime_column(pd.DataFrame(columns=['composite_cluster_id'])) or 'hmm_regime'
        except Exception:
            detected = 'hmm_regime'
        self.regime_col = str(labeling_cfg.get('hmm_barrier_regime_column', detected))
        self.time_barrier_minutes = int(labeling_cfg.get('time_barrier_minutes', 30))
        self.max_lookahead = int(labeling_cfg.get('max_lookahead', 100))
        self.logger.info(f'📋 Regime-aware labeling configuration:')
        self.logger.info(f'   - Auto recalculate HMM barriers: {self.auto_recalculate_hmm_barriers}')
        self.logger.info(f'   - HMM regime column: {self.regime_col}')
        self.logger.info(f'   - Time barrier minutes: {self.time_barrier_minutes}')
        self.logger.info(f'   - Max lookahead: {self.max_lookahead}')
        self.regime_barrier_optimizer = None
        try:
            from .training.steps.step06_labeling_components.regime_specific_triple_barrier_optimizer import RegimeSpecificTripleBarrierOptimizer
            self.regime_barrier_optimizer = RegimeSpecificTripleBarrierOptimizer(self.config)
            self.logger.info('✅ RegimeSpecificTripleBarrierOptimizer initialized successfully')
        except Exception as e:
            self.logger.warning(f'⚠️ Could not initialize RegimeSpecificTripleBarrierOptimizer: {e}')
            self.regime_barrier_optimizer = None
        # Initialize meta-labeling system if available
        try:
            from src.analyst.meta_labeling_system import MetaLabelingSystem
            self.meta_labeling_system = MetaLabelingSystem(self.config)
            self.logger.info('✅ Meta-labeling system initialized successfully')
        except ImportError:
            self.logger.warning('⚠️ Meta-labeling system not available')
            self.meta_labeling_system = None
        except Exception as e:
            self.logger.warning(f'⚠️ Could not initialize MetaLabelingSystem: {e}')
            self.meta_labeling_system = None

    async def initialize(self) -> None:
        """Initialize the labeling step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Labeling Step...')
        self.logger.info('📋 Step 5 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Labeling Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @traced(span_name='execute_labeling')
    @validates()
    @handles_errors()
    @cached()
    @log_execution_time()
    async def execute_labeling(self, symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache', force_rerun: bool=False) -> bool:
        step_start = time.time()
        self.logger.info(f'🚀 Executing Labeling for {symbol} on {exchange}')
        try:
            triple_barrier_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            if not triple_barrier_path.exists():
                self.logger.error(f'❌ Triple barrier labels not found at {triple_barrier_path}')
                return False
            self.logger.info(f'📁 Loading triple barrier labels from {triple_barrier_path}')
            labeled_dir = ensure_directory(Path(data_dir) / 'training' / 'labeled_data')
            output_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeled_data.parquet'
            metadata_path = labeled_dir / f'{exchange}_{symbol}_{timeframe}_labeling_metadata.json'
            current_fp = self._compute_labeling_fingerprint(triple_barrier_path)
            if not force_rerun and output_path.exists() and metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        existing_meta = json.load(f)
                    existing_fp = existing_meta.get('source_fingerprint', {})
                    if existing_fp == current_fp and existing_meta.get('total_samples', 0) > 0:
                        self.logger.info('🟢 Labeling is idempotent: existing outputs match current inputs. Skipping recomputation.')
                        self._log_step_timing('Labeling (skipped)', step_start)
                        return True
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to read existing labeling metadata: {e}')
            data = pd.read_parquet(triple_barrier_path)
            try:
                from .utils.regime_data_access import ensure_regime_labels, get_regime_column
                data = ensure_regime_labels(data, exchange=exchange, symbol=symbol, timeframe=timeframe, data_dir=data_dir)
                detected_col = get_regime_column(data)
                if detected_col and detected_col != self.regime_col:
                    self.logger.info(f"🔁 Using detected regime column '{detected_col}' instead of '{self.regime_col}'")
                    self.regime_col = detected_col
            except Exception:
                pass
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            current_call_id = None
            for call_id, call_record in self.function_monitor.active_calls.items():
                if call_record.function_name == 'execute_labeling':
                    current_call_id = call_id
                    break
            if current_call_id:
                self.function_monitor.record_function_to_function_call(current_call_id, '_generate_comprehensive_labels')
            data = await self._generate_comprehensive_labels(data, symbol, exchange, timeframe)
            if data is None:
                self.logger.error('❌ Comprehensive labeling failed')
                return False
            data.to_parquet(output_path)
            self.logger.info(f'✅ Labeled data saved to {output_path}')
            label_distribution = {}
            if 'label' in data.columns:
                label_distribution = data['label'].value_counts().to_dict()
            metadata = {'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'total_samples': int(len(data)), 'label_distribution': label_distribution, 'created_at': pd.Timestamp.now().isoformat(), 'labeling_config': self.config.get('labeling', {}), 'source_fingerprint': current_fp}
            safe_json_dump(metadata, metadata_path, indent=2, default=str)
            self._log_step_timing('execute_labeling', step_start)
            await self._log_step5_artifacts_and_report(symbol, exchange, timeframe, data_dir, data, output_path, metadata_path)
            await self._generate_and_log_function_call_report()
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error in labeling: {e}')
            await self._generate_and_log_function_call_report()
            return False

    async def _generate_and_log_function_call_report(self) -> None:
        """Generate and log function call report using standardized error handling."""
        try:
            self.logger.info('📊 Generating function call report...')
            # Use standardized error handler
            error_summary = self.error_handler.get_error_summary()
            if error_summary['total_errors'] > 0:
                self.logger.warning(f"⚠️ {error_summary['total_errors']} errors occurred during execution")
            else:
                self.logger.info('✅ No errors recorded during execution')
        except Exception as e:
            self.logger.error(f'❌ Failed to generate function call report: {e}')

    # Removed custom _save_function_call_report - using standardized error handling

    # Removed custom _log_function_call_relationships - using standardized error handling

    # Removed custom _analyze_function_completion_outcomes and _log_detailed_completion_report - using standardized error handling

    async def _log_step5_artifacts_and_report(self, symbol: str, exchange: str, timeframe: str, data_dir: str, labeled_data: pd.DataFrame, output_path: Path, metadata_path: Path) -> None:
        """Log step 5 artifacts and create simplified report."""
        try:
            self.logger.info('📊 Logging step 5 artifacts and reports...')
            if labeled_data is not None:
                total_samples = len(labeled_data)
                labeled_samples = len(labeled_data[labeled_data['label'].notna()]) if 'label' in labeled_data.columns else 0
                self.logger.info(f'✅ Labeled {labeled_samples}/{total_samples} samples successfully')
            self.logger.info(f'✅ Output saved to: {output_path}')
            self.logger.info(f'✅ Metadata saved to: {metadata_path}')
            self.logger.info('✅ Step 5 artifacts and reports logged successfully')
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 5 artifacts and reports: {e}')

    async def _generate_comprehensive_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Generate comprehensive labels combining multiple labeling strategies with regime-aware triple barrier method."
        
        New Labeling Flow:
        Primary Path: Attempts regime-aware labeling using RegimeSpecificTripleBarrierOptimizer
        Fallback Path: Uses OptimizedTripleBarrierLabeling if regime-aware methods fail
        Data Source Flexibility: Can work with unified data or step04 output depending on configuration
        """
        try:
            result_data = data.copy()
            if 'triple_barrier_label' not in result_data.columns:
                self.logger.info('🔄 Triple barrier labels not found, generating them using regime-aware methods...')
                if self.regime_barrier_optimizer is not None and self.auto_recalculate_hmm_barriers:
                    try:
                        self.logger.info('🚀 Attempting regime-aware triple barrier labeling...')
                        if self.regime_col in result_data.columns:
                            self.logger.info(f'✅ Found regime column: {self.regime_col}')
                            current_call_id = None
                            for call_id, call_record in self.function_monitor.active_calls.items():
                                if call_record.function_name == '_generate_comprehensive_labels':
                                    current_call_id = call_id
                                    break
                            if current_call_id:
                                self.function_monitor.record_function_to_function_call(current_call_id, '_generate_regime_aware_labels')
                            regime_labels = await self._generate_regime_aware_labels(result_data, symbol, exchange, timeframe)
                            if regime_labels is not None:
                                result_data['triple_barrier_label'] = regime_labels
                                result_data['labeling_method'] = 'regime_aware'
                                self.logger.info('✅ Generated regime-aware triple barrier labels')
                            else:
                                raise Exception('Regime-aware labeling failed')
                        else:
                            self.logger.warning(f"⚠️ Regime column '{self.regime_col}' not found")
                            raise Exception('Regime column not found')
                    except Exception as e:
                        self.logger.error(f'❌ Regime-aware labeling failed: {e}')
                        self.logger.error('❌ No fallback labeling method available - regime-aware labeling is required')
                        return None
                else:
                    if not self.auto_recalculate_hmm_barriers:
                        self.logger.error('❌ Auto-calculation disabled for regime-aware labeling')
                    if self.regime_barrier_optimizer is None:
                        self.logger.error('❌ Regime barrier optimizer not available')
                    self.logger.error('❌ Regime-aware labeling is required - no fallback available')
                    return None
            if self.meta_labeling_system:
                try:
                    await self.meta_labeling_system.initialize()
                    analyst_labels = await self.meta_labeling_system._generate_analyst_labels(data, symbol, exchange, timeframe)
                    if analyst_labels is not None:
                        result_data['analyst_label'] = analyst_labels
                        self.logger.info('✅ Generated analyst labels')
                    tactician_labels = await self.meta_labeling_system._generate_tactician_labels(data, symbol, exchange, timeframe)
                    if tactician_labels is not None:
                        result_data['tactician_label'] = tactician_labels
                        self.logger.info('✅ Generated tactician labels')
                except Exception as e:
                    self.logger.warning(f'⚠️ Meta-labeling failed: {e}')
            composite_label = await self._create_composite_label(result_data)
            result_data['label'] = composite_label
            result_data['label_confidence'] = await self._calculate_label_confidence(result_data)
            result_data['label_source'] = await self._determine_label_source(result_data)
            self.logger.info(f'✅ Generated comprehensive labels with {len(result_data.columns)} columns')
            self.logger.info(f"   - Label distribution: {result_data['label'].value_counts().to_dict()}")
            self.logger.info(f"   - Labeling method used: {result_data.get('labeling_method', 'unknown')}")
            return result_data
        except Exception as e:
            self.logger.exception(f'❌ Error generating comprehensive labels: {e}')
            return None

    async def _create_composite_label(self, data: pd.DataFrame) -> pd.Series:
        """Create composite label from multiple labeling strategies."""
        try:
            composite_label = data['triple_barrier_label'].copy()
            if 'analyst_label' in data.columns:
                analyst_override_mask = (data['analyst_label'] != 0) & (data['triple_barrier_label'] == 0)
                composite_label[analyst_override_mask] = data['analyst_label'][analyst_override_mask]
            return composite_label
        except Exception as e:
            self.logger.warning(f'⚠️ Error creating composite label: {e}')
            return data['triple_barrier_label']

    async def _calculate_label_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for labels."""
        try:
            confidence = np.ones(len(data), dtype=np.float32)
            if 'analyst_label' in data.columns:
                agreement_mask = (data['label'] == data['analyst_label']) & (data['analyst_label'] != 0)
                confidence[agreement_mask] += 0.2
            confidence = np.minimum(confidence, 1.0)
            return pd.Series(confidence, index=data.index)
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating label confidence: {e}')
            return pd.Series(1.0, index=data.index)

    async def _determine_label_source(self, data: pd.DataFrame) -> pd.Series:
        """Determine the source of each label."""
        try:
            sources = []
            for idx in range(len(data)):
                if data['label'].iloc[idx] == data['triple_barrier_label'].iloc[idx]:
                    if 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                        sources.append('triple_barrier+analyst')
                    else:
                        sources.append('triple_barrier')
                elif 'analyst_label' in data.columns and data['label'].iloc[idx] == data['analyst_label'].iloc[idx]:
                    sources.append('analyst')
                else:
                    sources.append('composite')
            return pd.Series(sources, index=data.index)
        except Exception as e:
            self.logger.warning(f'⚠️ Error determining label source: {e}')
            return pd.Series('unknown', index=data.index)

    def _validate_regime_aware_inputs(self, data: pd.DataFrame) -> bool:
        """Validate inputs for regime-aware labeling."""
        if self.regime_barrier_optimizer is None:
            self.logger.error('❌ Regime barrier optimizer not available')
            return False
        if self.regime_col not in data.columns:
            self.logger.error(f"❌ Regime column '{self.regime_col}' not found in data")
            return False
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f'❌ Missing required columns for triple barrier labeling: {missing_columns}')
            return False
        return True

    def _create_regime_labeler(self) -> None:
        """Create and configure the regime labeler."""
        try:
            from .training.steps.step06_labeling_components.regime_aware_triple_barrier_labeling import RegimeAwareTripleBarrierLabeling
            return RegimeAwareTripleBarrierLabeling(default_profit_take_multiplier=0.002, default_stop_loss_multiplier=0.001, default_time_barrier_minutes=self.time_barrier_minutes, default_max_lookahead=self.max_lookahead)
        except ImportError as e:
            self.logger.error(f'❌ Failed to import RegimeAwareTripleBarrierLabeling: {e}')
            return None

    def _generate_labels_with_regime_labeler(self, regime_labeler: Any, data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate labels using the regime labeler."""
        try:
            labels = regime_labeler.generate_labels(data, regime_column=self.regime_col, time_barrier_minutes=self.time_barrier_minutes, max_lookahead=self.max_lookahead)
            if labels is not None:
                self.logger.info(f'✅ Generated {len(labels)} regime-aware labels')
                return labels
            else:
                raise Exception('Regime-aware labeling returned None')
        except Exception as e:
            self.logger.warning(f'⚠️ Regime-aware labeling failed: {e}')
            return None

    async def _generate_regime_aware_labels(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[pd.Series]:
        """Generate regime-aware triple barrier labels using RegimeSpecificTripleBarrierOptimizer."""
        try:
            self.logger.info('🔧 Generating regime-aware triple barrier labels...')
            if not self._validate_regime_aware_inputs(data):
                return None
            regime_labeler = self._create_regime_labeler()
            if regime_labeler is None:
                return None
            current_call_id = None
            for call_id, call_record in self.function_monitor.active_calls.items():
                if call_record.function_name == '_generate_regime_aware_labels':
                    current_call_id = call_id
                    break
            if current_call_id:
                self.function_monitor.record_function_to_function_call(current_call_id, '_generate_labels_with_regime_labeler')
            return self._generate_labels_with_regime_labeler(regime_labeler, data)
        except Exception as e:
            self.logger.exception(f'❌ Error in regime-aware labeling: {e}')
            return None

async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str=None, force_rerun: bool=False, config: Optional[Dict[str, Any]]=None) -> bool:
    """Run the labeling step with standardized data quality management."

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = {}
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    step_config = {'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir, 'labeling': {'enable_meta_labeling': True, 'enable_trend_labels': True, 'enable_volatility_labels': True, 'composite_label_strategy': 'weighted_combination'}, 'vectorized_labelling_orchestrator': {'auto_recalculate_hmm_barriers': True, 'hmm_barrier_regime_column': 'hmm_regime', 'time_barrier_minutes': 30, 'max_lookahead': 100, 'profit_take_multiplier': 0.002, 'stop_loss_multiplier': 0.001}, **config}
    step = LabelingStep(step_config)
    await step.initialize()
    return await step.execute_labeling(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir, force_rerun=force_rerun)
if __name__ == '__main__':

    async def test() -> None:
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Step 5 result: {success}')
    asyncio.run(test())