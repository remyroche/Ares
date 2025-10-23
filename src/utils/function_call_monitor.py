"""
Comprehensive Function Call Monitoring System for Step01

This module provides detailed function call monitoring including:
- Function entry validation and parameter checking
- Inter-function call tracking and dependency monitoring
- Function completion reporting with outcome analysis
- Performance monitoring with timing and resource usage
- Comprehensive error handling and recovery
- Structured logging with detailed reports
- Memory management and cleanup
"""
import asyncio
import functools
import inspect
import logging
import time
import traceback
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from contextlib import contextmanager
import collections

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

class FunctionCallStatus(Enum):
    """Status of function call monitoring."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    TIMEOUT = 'timeout'

class ValidationLevel(Enum):
    """Level of validation to perform."""
    BASIC = 'basic'
    STANDARD = 'standard'
    COMPREHENSIVE = 'comprehensive'

@dataclass
class FunctionCallMetrics:
    """Metrics for a function call."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    cpu_percent: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    call_depth: int = 0
    child_calls: int = 0
    status: FunctionCallStatus = FunctionCallStatus.PENDING
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

@dataclass
class FunctionCallReport:
    """Comprehensive report for a function call."""
    function_name: str
    module_name: str
    call_id: str
    timestamp: datetime
    metrics: FunctionCallMetrics
    parameters: Dict[str, Any] = field(default_factory = dict)
    return_value: Any = None
    validation_results: Dict[str, Any] = field(default_factory = dict)
    dependencies: List[str] = field(default_factory = list)
    side_effects: List[str] = field(default_factory = list)
    warnings: List[str] = field(default_factory = list)
    recommendations: List[str] = field(default_factory = list)

class FunctionCallMonitor:
    """Main function call monitoring system with memory management."""

    def __init__(self, logger: Optional[logging.Logger]=None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.active_calls: Dict[str, FunctionCallReport] = {}
        self.call_history: List[FunctionCallReport] = []
        self.call_stack: List[str] = []
        self._lock = threading.Lock()
        self._call_counter = 0
        self._max_history = 10000  # Limit history to prevent memory leaks
        self._cleanup_interval = 1000  # Cleanup every 1000 calls

    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        with self._lock:
            self._call_counter += 1
            return f'call_{self._call_counter}_{int(time.time() * 1000)}'

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if psutil is None:
                return 0.0
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            if psutil is None:
                return 0.0
            return psutil.cpu_percent()
        except Exception:
            return 0.0

    def _validate_parameters(self, func: Callable, args: tuple, kwargs: dict, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Validate function parameters."""
        validation_results = {'parameter_count_valid': True, 'required_parameters_present': True, 'parameter_types_valid': True, 'parameter_values_valid': True, 'warnings': [], 'errors': []}
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            if len(bound_args.arguments) != len(sig.parameters):
                validation_results['parameter_count_valid'] = False
                validation_results['warnings'].append(f'Parameter count mismatch: expected {len(sig.parameters)}, got {len(bound_args.arguments)}')
            for param_name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty and param_name not in bound_args.arguments:
                    validation_results['required_parameters_present'] = False
                    validation_results['errors'].append(f"Required parameter '{param_name}' missing")
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                for param_name, value in bound_args.arguments.items():
                    param = sig.parameters[param_name]
                    if param.annotation != inspect.Parameter.empty:
                        if not isinstance(value, param.annotation):
                            validation_results['parameter_types_valid'] = False
                            validation_results['warnings'].append(f"Parameter '{param_name}' type mismatch: expected {param.annotation}, got {type(value)}")
            if validation_level == ValidationLevel.COMPREHENSIVE:
                for param_name, value in bound_args.arguments.items():
                    if value is None and param_name in ['symbol', 'exchange', 'data_dir']:
                        validation_results['parameter_values_valid'] = False
                        validation_results['errors'].append(f"Critical parameter '{param_name}' is None")
                    if isinstance(value, str) and value.strip() == '':
                        validation_results['parameter_values_valid'] = False
                        validation_results['warnings'].append(f"Parameter '{param_name}' is empty string")
        except Exception as e:
            validation_results['errors'].append(f'Parameter validation error: {str(e)}')
        return validation_results

    def _validate_return_value(self, func: Callable, return_value: Any, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Validate function return value."""
        validation_results = {'return_type_valid': True, 'return_value_valid': True, 'warnings': [], 'errors': []}
        try:
            if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
                expected_type = func.__annotations__['return']
                if not isinstance(return_value, expected_type):
                    validation_results['return_type_valid'] = False
                    validation_results['warnings'].append(f'Return type mismatch: expected {expected_type}, got {type(return_value)}')
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                if return_value is None and func.__name__ in ['execute', 'run_step', 'initialize']:
                    validation_results['return_value_valid'] = False
                    validation_results['errors'].append('Critical function returned None')
            if validation_level == ValidationLevel.COMPREHENSIVE:
                if isinstance(return_value, (list, dict)) and len(return_value) == 0:
                    validation_results['warnings'].append('Function returned empty collection')
        except Exception as e:
            validation_results['errors'].append(f'Return value validation error: {str(e)}')
        return validation_results

    def _detect_side_effects(self, func: Callable, args: tuple, kwargs: dict, return_value: Any) -> List[str]:
        """Detect potential side effects of function call."""
        side_effects = []
        try:
            if any(('data_dir' in str(arg) for arg in args)):
                side_effects.append('File system access detected')
            if any(('path' in str(kwarg) for kwarg in kwargs.values())):
                side_effects.append('File path manipulation detected')
            if any(('download' in str(arg).lower() for arg in args)):
                side_effects.append('Network operation detected')
            if func.__name__ in ['execute', 'initialize', 'run_step']:
                side_effects.append('Pipeline state modification')
            if 'log' in func.__name__.lower():
                side_effects.append('Logging operation')
        except Exception:
            pass
        return side_effects

    def _generate_recommendations(self, report: FunctionCallReport) -> List[str]:
        """Generate performance and quality recommendations."""
        recommendations = []
        try:
            if report.metrics.duration and report.metrics.duration > 10.0:
                recommendations.append('Consider optimizing function performance - execution time > 10s')
            if report.metrics.memory_peak and report.metrics.memory_peak > 1000:
                recommendations.append('High memory usage detected - consider memory optimization')
            if report.validation_results.get('errors'):
                recommendations.append('Address parameter validation errors')
            if report.warnings:
                recommendations.append('Review and address function warnings')
            if len(report.dependencies) > 5:
                recommendations.append('High dependency count - consider refactoring')
            if len(report.side_effects) > 3:
                recommendations.append('Multiple side effects detected - consider pure function design')
        except Exception:
            pass
        return recommendations

    @contextmanager
    def monitor_call(self, func: Callable, args: tuple, kwargs: dict, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> None:
        """Context manager for monitoring a function call."""
        call_id = self._generate_call_id()
        module_name = func.__module__ if hasattr(func, '__module__') else 'unknown'
        report = FunctionCallReport(function_name = func.__name__, module_name = module_name, call_id = call_id, timestamp = datetime.now(), metrics = FunctionCallMetrics(start_time = time.time(), memory_before = self._get_memory_usage(), call_depth = len(self.call_stack)))
        report.validation_results = self._validate_parameters(func, args, kwargs, validation_level)
        with self._lock:
            self.active_calls[call_id] = report
            self.call_stack.append(call_id)
        self.logger.info(f'🔍 FUNCTION CALL ENTRY: {func.__name__} (ID: {call_id})')
        self.logger.info(f'   📍 Module: {module_name}')
        self.logger.info(f'   📊 Parameters: {len(args)} args, {len(kwargs)} kwargs')
        self.logger.info(f'   🧠 Memory before: {report.metrics.memory_before:.2f} MB')
        self.logger.info(f'   📏 Call depth: {report.metrics.call_depth}')
        if report.validation_results.get('warnings'):
            for warning in report.validation_results['warnings']:
                self.logger.warning(f'   ⚠️ Parameter warning: {warning}')
        if report.validation_results.get('errors'):
            for error in report.validation_results['errors']:
                self.logger.error(f'   ❌ Parameter error: {error}')
        try:
            report.metrics.status = FunctionCallStatus.IN_PROGRESS
            start_time = time.time()
            yield report
            report.metrics.end_time = time.time()
            report.metrics.duration = report.metrics.end_time - report.metrics.start_time
            report.metrics.memory_after = self._get_memory_usage()
            report.metrics.cpu_percent = self._get_cpu_percent()
            report.metrics.status = FunctionCallStatus.COMPLETED
            return_validation = self._validate_return_value(func, report.return_value, validation_level)
            report.validation_results.update(return_validation)
            report.side_effects = self._detect_side_effects(func, args, kwargs, report.return_value)
            report.recommendations = self._generate_recommendations(report)
            self.logger.info(f'✅ FUNCTION CALL COMPLETED: {func.__name__} (ID: {call_id})')
            self.logger.info(f'   ⏱️ Duration: {report.metrics.duration:.3f}s')
            self.logger.info(f'   🧠 Memory after: {report.metrics.memory_after:.2f} MB')
            self.logger.info(f'   📈 Memory delta: {report.metrics.memory_after - report.metrics.memory_before:.2f} MB')
            self.logger.info(f'   🔄 Child calls: {report.metrics.child_calls}')
            self.logger.info(f'   📊 Side effects: {len(report.side_effects)}')
            if report.validation_results.get('warnings'):
                for warning in report.validation_results['warnings']:
                    self.logger.warning(f'   ⚠️ Return warning: {warning}')
            if report.recommendations:
                for rec in report.recommendations:
                    self.logger.info(f'   💡 Recommendation: {rec}')
        except Exception as e:
            report.metrics.end_time = time.time()
            report.metrics.duration = report.metrics.end_time - report.metrics.start_time
            report.metrics.memory_after = self._get_memory_usage()
            report.metrics.status = FunctionCallStatus.FAILED
            report.metrics.error_message = str(e)
            report.metrics.stack_trace = traceback.format_exc()
            self.logger.error(f'❌ FUNCTION CALL FAILED: {func.__name__} (ID: {call_id})')
            self.logger.error(f'   ⏱️ Duration: {report.metrics.duration:.3f}s')
            self.logger.error(f'   🧠 Memory after: {report.metrics.memory_after:.2f} MB')
            self.logger.error(f'   ❌ Error: {report.metrics.error_message}')
            raise
        finally:
            with self._lock:
                if call_id in self.active_calls:
                    del self.active_calls[call_id]
                if call_id in self.call_stack:
                    self.call_stack.remove(call_id)
                self.call_history.append(report)
                
                # Periodic cleanup to prevent memory leaks
                if len(self.call_history) > self._max_history:
                    self._cleanup_old_history()

    def get_call_summary(self) -> Dict[str, Any]:
        """Get summary of all function calls."""
        with self._lock:
            total_calls = len(self.call_history)
            successful_calls = sum((1 for call in self.call_history if call.metrics.status == FunctionCallStatus.COMPLETED))
            failed_calls = sum((1 for call in self.call_history if call.metrics.status == FunctionCallStatus.FAILED))
            avg_duration = 0.0
            if total_calls > 0:
                durations = [call.metrics.duration for call in self.call_history if call.metrics.duration]
                avg_duration = sum(durations) / len(durations) if durations else 0.0
            return {'total_calls': total_calls, 'successful_calls': successful_calls, 'failed_calls': failed_calls, 'success_rate': successful_calls / total_calls * 100 if total_calls > 0 else 0.0, 'average_duration': avg_duration, 'active_calls': len(self.active_calls), 'call_stack_depth': len(self.call_stack)}

    @memory_managed(MemoryStrategy.MODERATE)
    def export_detailed_report(self, filepath: str) -> None:
        """Export detailed function call report to file with memory management."""
        try:
            report_data = {'summary': self.get_call_summary(), 'call_history': [{'function_name': call.function_name, 'module_name': call.module_name, 'call_id': call.call_id, 'timestamp': call.timestamp.isoformat(), 'metrics': {'duration': call.metrics.duration, 'memory_before': call.metrics.memory_before, 'memory_after': call.metrics.memory_after, 'memory_peak': call.metrics.memory_peak, 'cpu_percent': call.metrics.cpu_percent, 'call_depth': call.metrics.call_depth, 'child_calls': call.metrics.child_calls, 'status': call.metrics.status.value, 'error_message': call.metrics.error_message}, 'validation_results': call.validation_results, 'dependencies': call.dependencies, 'side_effects': call.side_effects, 'warnings': call.warnings, 'recommendations': call.recommendations} for call in self.call_history]}
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent = 2)
            self.logger.info(f'📊 Detailed function call report exported to: {filepath}')
        except Exception as e:
            self.logger.error(f'❌ Failed to export detailed report: {e}')
    
    def _cleanup_old_history(self) -> None:
        """Cleanup old call history to prevent memory leaks."""
        # Keep only the most recent 80% of history
        keep_count = int(self._max_history * 0.8)
        if len(self.call_history) > keep_count:
            self.call_history = self.call_history[-keep_count:]
            self.logger.info(f"Cleaned up call history, keeping {keep_count} most recent calls")
        
        # Force cleanup to free memory
        force_cleanup()
_global_monitor = FunctionCallMonitor()

def monitor_function_calls(validation_level: ValidationLevel = ValidationLevel.STANDARD) -> None:
    """Decorator for monitoring function calls with comprehensive reporting."""

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            with _global_monitor.monitor_call(func, args, kwargs, validation_level) as report:
                result = await func(*args, **kwargs)
                report.return_value = result
                return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            with _global_monitor.monitor_call(func, args, kwargs, validation_level) as report:
                result = func(*args, **kwargs)
                report.return_value = result
                return result
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def get_function_call_monitor() -> FunctionCallMonitor:
    """Get the global function call monitor instance."""
    return _global_monitor

def log_function_call_summary(logger: logging.Logger) -> None:
    """Log a summary of function calls."""
    summary = _global_monitor.get_call_summary()
    logger.info('📊 FUNCTION CALL SUMMARY:')
    logger.info(f"   Total calls: {summary['total_calls']}")
    logger.info(f"   Successful: {summary['successful_calls']}")
    logger.info(f"   Failed: {summary['failed_calls']}")
    logger.info(f"   Success rate: {summary['success_rate']:.1f}%")
    logger.info(f"   Average duration: {summary['average_duration']:.3f}s")
    logger.info(f"   Active calls: {summary['active_calls']}")
    logger.info(f"   Call stack depth: {summary['call_stack_depth']}")

def monitor_basic(func: Callable) -> Callable:
    """Monitor function with basic validation."""
    return monitor_function_calls(ValidationLevel.BASIC)(func)

def monitor_standard(func: Callable) -> Callable:
    """Monitor function with standard validation."""
    return monitor_function_calls(ValidationLevel.STANDARD)(func)

def monitor_comprehensive(func: Callable) -> Callable:
    """Monitor function with comprehensive validation."""
    return monitor_function_calls(ValidationLevel.COMPREHENSIVE)(func)
