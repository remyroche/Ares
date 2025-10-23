"""
Comprehensive Function Call Monitoring Decorator.

This module provides a comprehensive monitoring system that tracks:
1. Function entry with parameter validation
2. Function-to-function calls within the monitored function
3. Function exit with detailed outcome reporting
4. Performance metrics (timing, memory, CPU)
5. Error handling and recovery
6. Detailed execution reports
"""
import asyncio
import functools
import inspect
import logging
# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

import time

import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from .compose import P, R, uniform_wrapper
# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

function_call_stack: ContextVar[List[str]] = ContextVar('function_call_stack', default=[])
execution_report: ContextVar[Dict[str, Any]] = ContextVar('execution_report', default={})

@dataclass
class FunctionCallMetrics:
    """Metrics for a single function call."""
    function_name: str
    module_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    cpu_percent_before: Optional[float] = None
    cpu_percent_after: Optional[float] = None
    cpu_delta: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    exception: Optional[Exception] = None
    success: bool = True
    nested_calls: List['FunctionCallMetrics'] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    performance_warnings: List[str] = field(default_factory=list)

@dataclass
class ExecutionReport:
    """Comprehensive execution report for a function call."""
    execution_id: str
    root_function: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    total_memory_used: Optional[float] = None
    total_cpu_used: Optional[float] = None
    function_calls: List[FunctionCallMetrics] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_issues: List[str] = field(default_factory=list)
    validation_failures: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    summary: Dict[str, Any] = field(default_factory=dict)

class FunctionCallMonitor:
    """Comprehensive function call monitoring system."""

    def __init__(self, enable_performance_monitoring: bool=True, enable_memory_monitoring: bool=True, enable_cpu_monitoring: bool=True, enable_parameter_validation: bool=True, enable_nested_call_tracking: bool=True, log_level: str='INFO', generate_detailed_report: bool=True, report_file_path: Optional[str]=None) -> None:
        """
        Initialize the function call monitor.

        Args:
            enable_performance_monitoring: Track timing and performance metrics
            enable_memory_monitoring: Track memory usage
            enable_cpu_monitoring: Track CPU usage
            enable_parameter_validation: Validate function parameters
            enable_nested_call_tracking: Track nested function calls
            log_level: Logging level for monitoring output
            generate_detailed_report: Generate detailed execution report
            report_file_path: Path to save detailed reports
        """
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_cpu_monitoring = enable_cpu_monitoring
        self.enable_parameter_validation = enable_parameter_validation
        self.enable_nested_call_tracking = enable_nested_call_tracking
        self.log_level = log_level
        self.generate_detailed_report = generate_detailed_report
        self.report_file_path = report_file_path
        self.logger = logging.getLogger(f'{__name__}.FunctionCallMonitor')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.performance_thresholds = {'max_duration_seconds': 30.0, 'max_memory_mb': 1000.0, 'max_cpu_percent': 80.0, 'max_nested_calls': 50}
        self.active_sessions: Dict[str, ExecutionReport] = {}

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except Exception:
            return 0.0

    def _validate_parameters(self, func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Validate function parameters."""
        validation_results = {'valid': True, 'issues': [], 'parameter_types': {}, 'parameter_values': {}}
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            for param_name, param_value in bound_args.arguments.items():
                param_info = sig.parameters[param_name]
                validation_results['parameter_types'][param_name] = type(param_value).__name__
                validation_results['parameter_values'][param_name] = str(param_value)[:100]
                if param_info.annotation != inspect.Parameter.empty:
                    expected_type = param_info.annotation
                    if not isinstance(param_value, expected_type):
                        validation_results['issues'].append(f"Parameter '{param_name}' expected {expected_type.__name__}, got {type(param_value).__name__}")
                        validation_results['valid'] = False
                if param_value is None and param_info.default is inspect.Parameter.empty:
                    validation_results['issues'].append(f"Parameter '{param_name}' is None but no default provided")
                    validation_results['valid'] = False
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f'Parameter validation error: {str(e)}')
        return validation_results

    def _check_performance_thresholds(self, metrics: FunctionCallMetrics) -> List[str]:
        """Check performance thresholds and return warnings."""
        warnings = []
        if metrics.duration and metrics.duration > self.performance_thresholds['max_duration_seconds']:
            warnings.append(f"Function execution time ({metrics.duration:.2f}s) exceeds threshold ({self.performance_thresholds['max_duration_seconds']}s)")
        if metrics.memory_delta and metrics.memory_delta > self.performance_thresholds['max_memory_mb']:
            warnings.append(f"Memory usage increase ({metrics.memory_delta:.2f}MB) exceeds threshold ({self.performance_thresholds['max_memory_mb']}MB)")
        if metrics.cpu_delta and metrics.cpu_delta > self.performance_thresholds['max_cpu_percent']:
            warnings.append(f"CPU usage increase ({metrics.cpu_delta:.2f}%) exceeds threshold ({self.performance_thresholds['max_cpu_percent']}%)")
        return warnings

    def _log_function_entry(self, func: Callable, args: tuple, kwargs: dict, metrics: FunctionCallMetrics) -> None:
        """Log function entry with detailed information."""
        self.logger.info(f'🚀 ENTERING {func.__name__}')
        self.logger.info(f'   📍 Module: {func.__module__}')
        self.logger.info(f"   ⏰ Start time: {datetime.fromtimestamp(metrics.start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        if self.enable_parameter_validation:
            self.logger.info(f"   🔍 Parameter validation: {('✅ PASSED' if metrics.validation_results['valid'] else '❌ FAILED')}")
            if metrics.validation_results['issues']:
                for issue in metrics.validation_results['issues']:
                    self.logger.warning(f'      ⚠️ {issue}')
        if self.enable_memory_monitoring and metrics.memory_before:
            self.logger.info(f'   💾 Memory before: {metrics.memory_before:.2f} MB')
        if self.enable_cpu_monitoring and metrics.cpu_percent_before:
            self.logger.info(f'   🖥️ CPU before: {metrics.cpu_percent_before:.2f}%')
        if metrics.parameters:
            self.logger.info(f'   📋 Parameters ({len(metrics.parameters)}):')
            for param_name, param_value in list(metrics.parameters.items())[:5]:
                self.logger.info(f'      - {param_name}: {str(param_value)[:50]}...')
            if len(metrics.parameters) > 5:
                self.logger.info(f'      ... and {len(metrics.parameters) - 5} more parameters')

    def _log_function_exit(self, func: Callable, metrics: FunctionCallMetrics, success: bool) -> None:
        """Log function exit with detailed outcome information."""
        status_emoji = '✅' if success else '❌'
        status_text = 'COMPLETED' if success else 'FAILED'
        self.logger.info(f'{status_emoji} EXITING {func.__name__} - {status_text}')
        self.logger.info(f"   ⏰ End time: {datetime.fromtimestamp(metrics.end_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        self.logger.info(f'   ⏱️ Duration: {metrics.duration:.4f} seconds')
        if self.enable_memory_monitoring:
            self.logger.info(f'   💾 Memory after: {metrics.memory_after:.2f} MB')
            if metrics.memory_delta:
                delta_emoji = '📈' if metrics.memory_delta > 0 else '📉'
                self.logger.info(f'   {delta_emoji} Memory delta: {metrics.memory_delta:+.2f} MB')
        if self.enable_cpu_monitoring:
            self.logger.info(f'   🖥️ CPU after: {metrics.cpu_percent_after:.2f}%')
            if metrics.cpu_delta:
                delta_emoji = '📈' if metrics.cpu_delta > 0 else '📉'
                self.logger.info(f'   {delta_emoji} CPU delta: {metrics.cpu_delta:+.2f}%')
        if metrics.nested_calls:
            self.logger.info(f'   🔗 Nested calls: {len(metrics.nested_calls)}')
            for nested_call in metrics.nested_calls[:3]:
                nested_status = '✅' if nested_call.success else '❌'
                self.logger.info(f'      {nested_status} {nested_call.function_name} ({nested_call.duration:.3f}s)')
            if len(metrics.nested_calls) > 3:
                self.logger.info(f'      ... and {len(metrics.nested_calls) - 3} more nested calls')
        if metrics.performance_warnings:
            self.logger.warning(f'   ⚠️ Performance warnings ({len(metrics.performance_warnings)}):')
            for warning in metrics.performance_warnings:
                self.logger.warning(f'      - {warning}')
        if success and metrics.return_value is not None:
            return_type = type(metrics.return_value).__name__
            return_str = str(metrics.return_value)[:100]
            self.logger.info(f'   📤 Return value: {return_type} - {return_str}...')
        if not success and metrics.exception:
            self.logger.error(f'   💥 Exception: {type(metrics.exception).__name__}: {str(metrics.exception)}')
            self.logger.error(f'   📍 Exception location: {metrics.exception.__traceback__.tb_frame.f_code.co_filename}:{metrics.exception.__traceback__.tb_lineno}')

    def _generate_detailed_report(self, execution_id: str, report: ExecutionReport) -> None:
        """Generate detailed execution report."""
        if not self.generate_detailed_report:
            return
        try:
            report_data = {'execution_id': execution_id, 'root_function': report.root_function, 'start_time': report.start_time.isoformat(), 'end_time': report.end_time.isoformat() if report.end_time else None, 'total_duration': report.total_duration, 'total_memory_used': report.total_memory_used, 'total_cpu_used': report.total_cpu_used, 'success': report.success, 'function_calls_count': len(report.function_calls), 'errors_count': len(report.errors), 'warnings_count': len(report.warnings), 'performance_issues_count': len(report.performance_issues), 'validation_failures_count': len(report.validation_failures), 'function_calls': [{'function_name': call.function_name, 'module_name': call.module_name, 'duration': call.duration, 'memory_delta': call.memory_delta, 'cpu_delta': call.cpu_delta, 'success': call.success, 'nested_calls_count': len(call.nested_calls), 'performance_warnings': call.performance_warnings, 'validation_results': call.validation_results} for call in report.function_calls], 'errors': report.errors, 'warnings': report.warnings, 'performance_issues': report.performance_issues, 'validation_failures': report.validation_failures, 'summary': report.summary}
            if self.report_file_path:
                report_file = Path(self.report_file_path) / f'execution_report_{execution_id}.json'
                report_file.parent.mkdir(parents=True, exist_ok=True)
                import json

                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                self.logger.info(f'📊 Detailed execution report saved to: {report_file}')
            self.logger.info('📊 EXECUTION REPORT SUMMARY')
            self.logger.info(f'   🆔 Execution ID: {execution_id}')
            self.logger.info(f'   🎯 Root function: {report.root_function}')
            self.logger.info(f'   ⏱️ Total duration: {report.total_duration:.4f} seconds')
            self.logger.info(f'   🔗 Function calls: {len(report.function_calls)}')
            self.logger.info(f'   ✅ Successful calls: {sum((1 for call in report.function_calls if call.success))}')
            self.logger.info(f'   ❌ Failed calls: {sum((1 for call in report.function_calls if not call.success))}')
            self.logger.info(f'   ⚠️ Warnings: {len(report.warnings)}')
            self.logger.info(f'   💥 Errors: {len(report.errors)}')
            self.logger.info(f'   🚨 Performance issues: {len(report.performance_issues)}')
        except Exception as e:
            self.logger.error(f'Failed to generate detailed report: {e}')

    def monitor_function(self, func: Callable[P, R]) -> Callable[P, R]:
        """Main decorator for comprehensive function monitoring."""

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return self._monitor_sync_function(func, args, kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await self._monitor_async_function(func, args, kwargs)
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def _monitor_sync_function(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Monitor synchronous function execution."""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        metrics = FunctionCallMetrics(function_name=func.__name__, module_name=func.__module__, start_time=start_time)
        current_stack = function_call_stack.get()
        new_stack = current_stack + [func.__name__]
        function_call_stack.set(new_stack)
        if self.enable_memory_monitoring:
            metrics.memory_before = self._get_memory_usage()
        if self.enable_cpu_monitoring:
            metrics.cpu_percent_before = self._get_cpu_percent()
        if self.enable_parameter_validation:
            metrics.validation_results = self._validate_parameters(func, args, kwargs)
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            metrics.parameters = dict(bound_args.arguments)
        except Exception:
            metrics.parameters = {'args': str(args), 'kwargs': str(kwargs)}
        self._log_function_entry(func, args, kwargs, metrics)
        try:
            result = func(*args, **kwargs)
            metrics.return_value = result
            metrics.success = True
        except Exception as e:
            metrics.exception = e
            metrics.success = False
            raise
        finally:
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            if self.enable_memory_monitoring:
                metrics.memory_after = self._get_memory_usage()
                if metrics.memory_before:
                    metrics.memory_delta = metrics.memory_after - metrics.memory_before
            if self.enable_cpu_monitoring:
                metrics.cpu_percent_after = self._get_cpu_percent()
                if metrics.cpu_percent_before:
                    metrics.cpu_delta = metrics.cpu_percent_after - metrics.cpu_percent_before
            metrics.performance_warnings = self._check_performance_thresholds(metrics)
            self._log_function_exit(func, metrics, metrics.success)
            function_call_stack.set(current_stack)
        return metrics.return_value

    async def _monitor_async_function(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Monitor asynchronous function execution."""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        metrics = FunctionCallMetrics(function_name=func.__name__, module_name=func.__module__, start_time=start_time)
        current_stack = function_call_stack.get()
        new_stack = current_stack + [func.__name__]
        function_call_stack.set(new_stack)
        if self.enable_memory_monitoring:
            metrics.memory_before = self._get_memory_usage()
        if self.enable_cpu_monitoring:
            metrics.cpu_percent_before = self._get_cpu_percent()
        if self.enable_parameter_validation:
            metrics.validation_results = self._validate_parameters(func, args, kwargs)
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            metrics.parameters = dict(bound_args.arguments)
        except Exception:
            metrics.parameters = {'args': str(args), 'kwargs': str(kwargs)}
        self._log_function_entry(func, args, kwargs, metrics)
        try:
            result = await func(*args, **kwargs)
            metrics.return_value = result
            metrics.success = True
        except Exception as e:
            metrics.exception = e
            metrics.success = False
            raise
        finally:
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            if self.enable_memory_monitoring:
                metrics.memory_after = self._get_memory_usage()
                if metrics.memory_before:
                    metrics.memory_delta = metrics.memory_after - metrics.memory_before
            if self.enable_cpu_monitoring:
                metrics.cpu_percent_after = self._get_cpu_percent()
                if metrics.cpu_percent_before:
                    metrics.cpu_delta = metrics.cpu_percent_after - metrics.cpu_percent_before
            metrics.performance_warnings = self._check_performance_thresholds(metrics)
            self._log_function_exit(func, metrics, metrics.success)
            function_call_stack.set(current_stack)
        return metrics.return_value

def monitor_function_calls(enable_performance_monitoring: bool=True, enable_memory_monitoring: bool=True, enable_cpu_monitoring: bool=True, enable_parameter_validation: bool=True, enable_nested_call_tracking: bool=True, log_level: str='INFO', generate_detailed_report: bool=True, report_file_path: Optional[str]=None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory for comprehensive function call monitoring.

    Args:
        enable_performance_monitoring: Track timing and performance metrics
        enable_memory_monitoring: Track memory usage
        enable_cpu_monitoring: Track CPU usage
        enable_parameter_validation: Validate function parameters
        enable_nested_call_tracking: Track nested function calls
        log_level: Logging level for monitoring output
        generate_detailed_report: Generate detailed execution report
        report_file_path: Path to save detailed reports

    Returns:
        Decorator function for monitoring function calls
    """
    monitor = FunctionCallMonitor(enable_performance_monitoring=enable_performance_monitoring, enable_memory_monitoring=enable_memory_monitoring, enable_cpu_monitoring=enable_cpu_monitoring, enable_parameter_validation=enable_parameter_validation, enable_nested_call_tracking=enable_nested_call_tracking, log_level=log_level, generate_detailed_report=generate_detailed_report, report_file_path=report_file_path)
    return monitor.monitor_function

def monitor_step03_functions(func: Callable[P, R]) -> Callable[P, R]:
    """Specialized decorator for step03 functions with comprehensive monitoring."""
    return monitor_function_calls(enable_performance_monitoring=True, enable_memory_monitoring=True, enable_cpu_monitoring=True, enable_parameter_validation=True, enable_nested_call_tracking=True, log_level='INFO', generate_detailed_report=True, report_file_path='logs/step03_monitoring')(func)

def monitor_critical_functions(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for critical functions with maximum monitoring."""
    return monitor_function_calls(enable_performance_monitoring=True, enable_memory_monitoring=True, enable_cpu_monitoring=True, enable_parameter_validation=True, enable_nested_call_tracking=True, log_level='DEBUG', generate_detailed_report=True, report_file_path='logs/critical_monitoring')(func)

def monitor_performance_only(func: Callable[P, R]) -> Callable[P, R]:
    """Lightweight decorator for performance monitoring only."""
    return monitor_function_calls(enable_performance_monitoring=True, enable_memory_monitoring=False, enable_cpu_monitoring=False, enable_parameter_validation=False, enable_nested_call_tracking=False, log_level='INFO', generate_detailed_report=False)(func)
