"""Enhanced Validator for Step 1.5: Data Converter with Comprehensive Function Call Monitoring.

This module provides comprehensive function call monitoring, detailed outcome reporting,
and health check mechanisms for the Step 1.5 data converter validator.

Features:
- Function call entry/exit logging with parameter validation
- Nested function call stack tracking with depth monitoring
- Comprehensive performance monitoring (execution time, memory usage)
- Detailed outcome reporting with success/failure metrics
- Enhanced error handling with context preservation
- Complete audit trail system for all function calls
- Health check mechanisms for system and data integrity

Dependencies:
=============
Standard Library:
- asyncio: Asynchronous programming support
- functools: Function utilities (wraps decorator)
- glob: File path pattern matching
- inspect: Runtime introspection of objects
- json: JSON data handling
- os: Operating system interface
- sys: System-specific parameters and functions
- threading: Thread-based parallelism
- time: Time-related functions
- traceback: Print or retrieve a stack traceback
- datetime: Date and time handling
- pathlib: Object-oriented filesystem paths
- typing: Type hints support

Third-Party:
- pandas: Data manipulation and analysis
- psutil: System and process utilities

Local Dependencies:
- .config: Configuration management
- .utils.base_validator: Base validator class
- .utils.common_operations: Common utility functions
- .utils.logger: Logging system

Installation Requirements:
=========================
pip install pandas psutil

Optional Dependencies (for enhanced functionality):
- numpy: Numerical computing (used by pandas)
- pyarrow: Fast columnar data processing (for parquet files)

Version Requirements:
====================
- Python >= 3.8
- pandas >= 1.3.0
- psutil >= 5.8.0
"""
import asyncio
import functools
import glob
import inspect
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import psutil
import numpy as np
import pandas as pd
import warnings

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from .config import CONFIG
from .utils.base_validator import BaseValidator
from .utils.common_operations import safe_json_load
from .utils.logger import system_logger

def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    missing_deps = []
    optional_deps = []
    required_modules = {'pandas': 'Data manipulation and analysis', 'psutil': 'System and process utilities'}
    for module, description in required_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(f'{module} ({description})')
    optional_modules = {'numpy': 'Numerical computing (used by pandas)', 'pyarrow': 'Fast parquet file processing', 'fastparquet': 'Alternative parquet engine'}
    for module, description in optional_modules.items():
        try:
            __import__(module)
        except ImportError:
            optional_deps.append(f'{module} ({description})')
    return (missing_deps, optional_deps)

def validate_environment() -> bool:
    """Validate the environment and dependencies."""
    missing_deps, optional_deps = check_dependencies()
    if missing_deps:
        error_msg = 'Missing required dependencies:\n' + '\n'.join((f'  - {dep}' for dep in missing_deps))
        error_msg += '\n\nInstall with: pip install pandas psutil'
        raise ImportError(error_msg)
    if optional_deps:

        warning_msg = 'Optional dependencies not available (functionality may be limited):\n'
        warning_msg += '\n'.join((f'  - {dep}' for dep in optional_deps))
        warning_msg += '\n\nInstall with: pip install pyarrow fastparquet'
        warnings.warn(warning_msg, UserWarning)
    return True
try:
    validate_environment()
except ImportError as e:
    print(f'❌ Environment validation failed: {e}')
    raise

class FunctionCallMonitor:
    """Comprehensive function call monitoring and reporting system."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.call_stack = []
        self.call_history = []
        self.performance_metrics = {}
        self.error_tracking = []
        self._lock = threading.Lock()

    def monitor_function(self, func: Callable) -> Callable:
        """Decorator to monitor function calls with comprehensive tracking."""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            return await self._monitor_call(func, args, kwargs, is_async=True)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            return self._monitor_call(func, args, kwargs, is_async=False)
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    async def _monitor_call(self, func: Callable, args: tuple, kwargs: dict, is_async: bool=False) -> None:
        """Monitor a function call with comprehensive tracking."""
        call_id = f'{func.__name__}_{id(func)}_{int(time.time() * 1000000)}'
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        call_context = {'call_id': call_id, 'function_name': func.__name__, 'module': func.__module__, 'start_time': start_time, 'start_memory_mb': start_memory, 'args_count': len(args), 'kwargs_count': len(kwargs), 'call_depth': len(self.call_stack), 'is_async': is_async, 'thread_id': threading.get_ident(), 'stack_trace': traceback.format_stack()[:-1]}
        self.logger.info(f'🔍 ENTRY: {func.__name__} (ID: {call_id})')
        self.logger.info(f'   📊 Args: {len(args)}, Kwargs: {len(kwargs)}')
        self.logger.info(f"   📈 Depth: {call_context['call_depth']}, Memory: {start_memory:.2f}MB")
        validation_result = self._validate_function_inputs(func, args, kwargs)
        if not validation_result['valid']:
            self.logger.warning(f"⚠️ INPUT VALIDATION FAILED: {validation_result['errors']}")
            call_context['input_validation_failed'] = True
            call_context['validation_errors'] = validation_result['errors']
        with self._lock:
            self.call_stack.append(call_context)
            self.call_history.append(call_context.copy())
        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            call_context.update({'end_time': end_time, 'end_memory_mb': end_memory, 'execution_time_seconds': execution_time, 'memory_delta_mb': memory_delta, 'success': True, 'result_type': type(result).__name__, 'result_size': self._estimate_result_size(result)})
            self.logger.info(f'✅ EXIT: {func.__name__} (ID: {call_id})')
            self.logger.info(f'   ⏱️ Execution time: {execution_time:.4f}s')
            self.logger.info(f'   💾 Memory delta: {memory_delta:+.2f}MB')
            self.logger.info(f'   📤 Result type: {type(result).__name__}')
            output_validation = self._validate_function_output(func, result)
            if not output_validation['valid']:
                self.logger.warning(f"⚠️ OUTPUT VALIDATION FAILED: {output_validation['errors']}")
                call_context['output_validation_failed'] = True
                call_context['output_validation_errors'] = output_validation['errors']
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            call_context.update({'end_time': end_time, 'execution_time_seconds': execution_time, 'success': False, 'exception_type': type(e).__name__, 'exception_message': str(e), 'exception_traceback': traceback.format_exc()})
            self.logger.error(f'❌ ERROR: {func.__name__} (ID: {call_id})')
            self.logger.error(f'   ⏱️ Execution time: {execution_time:.4f}s')
            self.logger.error(f'   🚨 Exception: {type(e).__name__}: {str(e)}')
            self.logger.error(f'   📍 Traceback: {traceback.format_exc()}')
            with self._lock:
                self.error_tracking.append({'call_id': call_id, 'function_name': func.__name__, 'exception_type': type(e).__name__, 'exception_message': str(e), 'timestamp': datetime.now().isoformat(), 'call_stack_depth': len(self.call_stack)})
            raise
        finally:
            with self._lock:
                if self.call_stack and self.call_stack[-1]['call_id'] == call_id:
                    self.call_stack.pop()
            self._update_performance_metrics(call_context)

    def _validate_function_inputs(self, func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Validate function input parameters."""
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            errors = []
            for param_name, param_value in bound_args.arguments.items():
                param = sig.parameters[param_name]
                if param_value is None and param.default is inspect.Parameter.empty:
                    errors.append(f"Parameter '{param_name}' is None but required")
                if isinstance(param_value, (list, dict, str)) and len(param_value) == 0:
                    if param_name in ['symbol', 'exchange', 'timeframe']:
                        errors.append(f"Parameter '{param_name}' is empty")
            return {'valid': len(errors) == 0, 'errors': errors}
        except Exception as e:
            return {'valid': False, 'errors': [f'Input validation error: {str(e)}']}

    def _validate_function_output(self, func: Callable, result: Any) -> Dict[str, Any]:
        """Validate function output."""
        try:
            errors = []
            if result is None and func.__name__ not in ['_validate_unified_config']:
                errors.append('Function returned None unexpectedly')
            if isinstance(result, (list, dict)) and len(result) == 0:
                if func.__name__ in ['_check_unified_data_structure']:
                    pass
                else:
                    errors.append('Function returned empty collection')
            if func.__name__.startswith('_validate_') and (not isinstance(result, bool)):
                errors.append(f'Validation function should return bool, got {type(result).__name__}')
            return {'valid': len(errors) == 0, 'errors': errors}
        except Exception as e:
            return {'valid': False, 'errors': [f'Output validation error: {str(e)}']}

    def _estimate_result_size(self, result: Any) -> str:
        """Estimate the size of the result."""
        try:
            if result is None:
                return 'None'
            elif isinstance(result, (str, int, float, bool)):
                return f'{type(result).__name__}'
            elif isinstance(result, (list, dict)):
                return f'{type(result).__name__}[{len(result)}]'
            elif hasattr(result, '__len__'):
                return f'{type(result).__name__}[{len(result)}]'
            else:
                return f'{type(result).__name__}'
        except:
            return 'Unknown'

    def _update_performance_metrics(self, call_context: Dict[str, Any]) -> None:
        """Update performance metrics for the function."""
        func_name = call_context['function_name']
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = {'call_count': 0, 'total_time': 0.0, 'avg_time': 0.0, 'min_time': float('inf'), 'max_time': 0.0, 'success_count': 0, 'error_count': 0, 'total_memory_delta': 0.0}
        metrics = self.performance_metrics[func_name]
        metrics['call_count'] += 1
        metrics['total_time'] += call_context['execution_time_seconds']
        metrics['avg_time'] = metrics['total_time'] / metrics['call_count']
        metrics['min_time'] = min(metrics['min_time'], call_context['execution_time_seconds'])
        metrics['max_time'] = max(metrics['max_time'], call_context['execution_time_seconds'])
        if call_context['success']:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1
        if 'memory_delta_mb' in call_context:
            metrics['total_memory_delta'] += call_context['memory_delta_mb']

    def get_call_summary(self) -> Dict[str, Any]:
        """Get comprehensive call summary."""
        with self._lock:
            return {'total_calls': len(self.call_history), 'active_calls': len(self.call_stack), 'error_count': len(self.error_tracking), 'performance_metrics': self.performance_metrics.copy(), 'recent_errors': self.error_tracking[-10:] if self.error_tracking else [], 'call_stack_depth': max([c['call_depth'] for c in self.call_history]) if self.call_history else 0}

    def log_comprehensive_report(self) -> None:
        """Log a comprehensive function call report."""
        summary = self.get_call_summary()
        self.logger.info('📊 COMPREHENSIVE FUNCTION CALL REPORT')
        self.logger.info('=' * 60)
        self.logger.info(f"📈 Total function calls: {summary['total_calls']}")
        self.logger.info(f"🔄 Active calls: {summary['active_calls']}")
        self.logger.info(f"❌ Total errors: {summary['error_count']}")
        self.logger.info(f"📊 Max call depth: {summary['call_stack_depth']}")
        if summary['performance_metrics']:
            self.logger.info('\n🎯 PERFORMANCE METRICS:')
            for func_name, metrics in summary['performance_metrics'].items():
                self.logger.info(f'   {func_name}:')
                self.logger.info(f"     📞 Calls: {metrics['call_count']}")
                self.logger.info(f"     ⏱️ Avg time: {metrics['avg_time']:.4f}s")
                self.logger.info(f"     📊 Min/Max: {metrics['min_time']:.4f}s / {metrics['max_time']:.4f}s")
                self.logger.info(f"     ✅ Success rate: {metrics['success_count']}/{metrics['call_count']}")
                if metrics['total_memory_delta'] != 0:
                    self.logger.info(f"     💾 Memory delta: {metrics['total_memory_delta']:+.2f}MB")
        if summary['recent_errors']:
            self.logger.info('\n🚨 RECENT ERRORS:')
            for error in summary['recent_errors']:
                self.logger.info(f"   {error['function_name']}: {error['exception_type']} - {error['exception_message']}")

class HealthCheckSystem:
    """Comprehensive health check system for data integrity, system state, and component availability."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.health_checks = {'system_resources': self._check_system_resources, 'data_integrity': self._check_data_integrity, 'component_availability': self._check_component_availability, 'file_system_health': self._check_file_system_health, 'memory_health': self._check_memory_health, 'disk_space': self._check_disk_space}
        self.health_status = {}
        self.last_health_check = None

    async def run_comprehensive_health_check(self, context: Dict[str, Any]=None) -> Dict[str, Any]:
        """Run comprehensive health checks and return detailed status."""
        health_check_start = time.time()
        health_check_id = f'health_check_{int(time.time() * 1000000)}'
        self.logger.info('🏥 STARTING COMPREHENSIVE HEALTH CHECK')
        self.logger.info('=' * 60)
        self.logger.info(f'🆔 Health Check ID: {health_check_id}')
        health_results = {'health_check_id': health_check_id, 'start_time': health_check_start, 'overall_status': 'HEALTHY', 'checks_performed': [], 'checks_passed': [], 'checks_failed': [], 'warnings': [], 'critical_issues': [], 'system_metrics': {}, 'recommendations': []}
        try:
            for check_name, check_function in self.health_checks.items():
                self.logger.info(f'🔍 Running {check_name} check...')
                health_results['checks_performed'].append(check_name)
                try:
                    check_result = await check_function(context or {})
                    health_results['checks_passed'].append(check_name)
                    health_results[check_name] = check_result
                    if check_result.get('status') == 'WARNING':
                        health_results['warnings'].append(f"{check_name}: {check_result.get('message', 'Warning detected')}")
                    elif check_result.get('status') == 'CRITICAL':
                        health_results['critical_issues'].append(f"{check_name}: {check_result.get('message', 'Critical issue detected')}")
                        health_results['overall_status'] = 'CRITICAL'
                    elif check_result.get('status') == 'DEGRADED':
                        if health_results['overall_status'] == 'HEALTHY':
                            health_results['overall_status'] = 'DEGRADED'
                    self.logger.info(f"✅ {check_name} check completed: {check_result.get('status', 'UNKNOWN')}")
                except Exception as e:
                    health_results['checks_failed'].append(check_name)
                    health_results[check_name] = {'status': 'ERROR', 'message': f'Health check failed: {str(e)}', 'error': str(e)}
                    health_results['critical_issues'].append(f'{check_name}: Health check error - {str(e)}')
                    health_results['overall_status'] = 'CRITICAL'
                    self.logger.error(f'❌ {check_name} check failed: {str(e)}')
            health_results['recommendations'] = self._generate_health_recommendations(health_results)
            health_results['end_time'] = time.time()
            health_results['duration'] = health_results['end_time'] - health_check_start
            self.logger.info(f'\n🏥 HEALTH CHECK SUMMARY')
            self.logger.info(f"   🎯 Overall Status: {health_results['overall_status']}")
            self.logger.info(f"   ✅ Checks Passed: {len(health_results['checks_passed'])}")
            self.logger.info(f"   ❌ Checks Failed: {len(health_results['checks_failed'])}")
            self.logger.info(f"   ⚠️ Warnings: {len(health_results['warnings'])}")
            self.logger.info(f"   🚨 Critical Issues: {len(health_results['critical_issues'])}")
            self.logger.info(f"   ⏱️ Duration: {health_results['duration']:.4f}s")
            if health_results['recommendations']:
                self.logger.info(f'\n💡 RECOMMENDATIONS:')
                for i, rec in enumerate(health_results['recommendations'], 1):
                    self.logger.info(f'   {i}. {rec}')
            self.logger.info('=' * 60)
            self.last_health_check = health_results
            self.health_status = health_results
            return health_results
        except Exception as e:
            self.logger.exception(f'❌ Critical error during health check: {e}')
            health_results['overall_status'] = 'CRITICAL'
            health_results['critical_issues'].append(f'Health check system error: {str(e)}')
            return health_results

    async def _check_system_resources(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system resource availability and usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            status = 'HEALTHY'
            message = 'System resources are within normal limits'
            warnings = []
            if cpu_percent > 90:
                status = 'CRITICAL'
                message = f'High CPU usage: {cpu_percent:.1f}%'
            elif cpu_percent > 80:
                status = 'WARNING'
                warnings.append(f'Elevated CPU usage: {cpu_percent:.1f}%')
            if memory.percent > 95:
                status = 'CRITICAL'
                message = f'Critical memory usage: {memory.percent:.1f}%'
            elif memory.percent > 85:
                if status == 'HEALTHY':
                    status = 'WARNING'
                warnings.append(f'High memory usage: {memory.percent:.1f}%')
            if disk.percent > 95:
                status = 'CRITICAL'
                message = f'Critical disk usage: {disk.percent:.1f}%'
            elif disk.percent > 85:
                if status == 'HEALTHY':
                    status = 'WARNING'
                warnings.append(f'High disk usage: {disk.percent:.1f}%')
            return {'status': status, 'message': message, 'warnings': warnings, 'metrics': {'cpu_percent': cpu_percent, 'memory_percent': memory.percent, 'memory_available_gb': memory.available / 1024 ** 3, 'disk_percent': disk.percent, 'disk_free_gb': disk.free / 1024 ** 3}}
        except Exception as e:
            return {'status': 'ERROR', 'message': f'System resource check failed: {str(e)}', 'error': str(e)}

    async def _check_data_integrity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check data integrity and consistency."""
        try:
            data_dir = context.get('data_dir', 'data_cache')
            unified_dir = os.path.join(data_dir, 'unified')
            status = 'HEALTHY'
            message = 'Data integrity checks passed'
            issues = []
            if not os.path.exists(data_dir):
                status = 'CRITICAL'
                issues.append(f'Data directory does not exist: {data_dir}')
            if not os.path.exists(unified_dir):
                status = 'WARNING'
                issues.append(f'Unified data directory does not exist: {unified_dir}')
            if os.path.exists(unified_dir):
                try:
                    for root, dirs, files in os.walk(unified_dir):
                        for file in files:
                            if file.endswith('.parquet'):
                                file_path = os.path.join(root, file)
                                try:
                                    df = pd.read_parquet(file_path, nrows=1)
                                    if df.empty:
                                        issues.append(f'Empty parquet file: {file_path}')
                                except Exception as e:
                                    status = 'CRITICAL'
                                    issues.append(f'Corrupted parquet file {file_path}: {str(e)}')
                except Exception as e:
                    issues.append(f'Error scanning unified directory: {str(e)}')
            if issues:
                message = f"Data integrity issues found: {'; '.join(issues[:3])}"
                if len(issues) > 3:
                    message += f' and {len(issues) - 3} more'
            return {'status': status, 'message': message, 'issues': issues, 'data_directory_exists': os.path.exists(data_dir), 'unified_directory_exists': os.path.exists(unified_dir)}
        except Exception as e:
            return {'status': 'ERROR', 'message': f'Data integrity check failed: {str(e)}', 'error': str(e)}

    async def _check_component_availability(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check availability of required components and dependencies."""
        try:
            status = 'HEALTHY'
            message = 'All required components are available'
            unavailable_components = []
            required_modules = ['pandas', 'numpy', 'psutil', 'asyncio']
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    unavailable_components.append(f'Python module: {module}')
                    status = 'CRITICAL'
            data_dir = context.get('data_dir', 'data_cache')
            try:
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir, exist_ok=True)
                test_file = os.path.join(data_dir, 'health_check_test.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                unavailable_components.append(f'File system access: {str(e)}')
                status = 'CRITICAL'
            if unavailable_components:
                message = f"Unavailable components: {'; '.join(unavailable_components)}"
            return {'status': status, 'message': message, 'unavailable_components': unavailable_components, 'required_modules_available': len(unavailable_components) == 0}
        except Exception as e:
            return {'status': 'ERROR', 'message': f'Component availability check failed: {str(e)}', 'error': str(e)}

    async def _check_file_system_health(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check file system health and performance."""
        try:
            status = 'HEALTHY'
            message = 'File system is healthy'
            issues = []
            data_dir = context.get('data_dir', 'data_cache')
            if os.path.exists(data_dir):
                test_file = os.path.join(data_dir, 'io_test.tmp')
                test_data = 'x' * 1024
                start_time = time.time()
                with open(test_file, 'w') as f:
                    f.write(test_data)
                write_time = time.time() - start_time
                start_time = time.time()
                with open(test_file, 'r') as f:
                    _ = f.read()
                read_time = time.time() - start_time
                os.remove(test_file)
                if write_time > 0.1:
                    issues.append(f'Slow write performance: {write_time:.3f}s')
                    status = 'WARNING'
                if read_time > 0.05:
                    issues.append(f'Slow read performance: {read_time:.3f}s')
                    if status == 'HEALTHY':
                        status = 'WARNING'
                if issues:
                    message = f"File system performance issues: {'; '.join(issues)}"
                return {'status': status, 'message': message, 'issues': issues, 'performance_metrics': {'write_time_seconds': write_time, 'read_time_seconds': read_time, 'write_speed_mb_per_sec': 0.001 / write_time if write_time > 0 else 0, 'read_speed_mb_per_sec': 0.001 / read_time if read_time > 0 else 0}}
            else:
                return {'status': 'WARNING', 'message': f'Data directory does not exist: {data_dir}', 'issues': [f'Missing data directory: {data_dir}']}
        except Exception as e:
            return {'status': 'ERROR', 'message': f'File system health check failed: {str(e)}', 'error': str(e)}

    async def _check_memory_health(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check memory health and potential memory leaks."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            status = 'HEALTHY'
            message = 'Memory usage is within normal limits'
            if memory_percent > 90:
                status = 'CRITICAL'
                message = f'Critical memory usage: {memory_percent:.1f}%'
            elif memory_percent > 80:
                status = 'WARNING'
                message = f'High memory usage: {memory_percent:.1f}%'
            rss_mb = memory_info.rss / (1024 * 1024)
            vms_mb = memory_info.vms / (1024 * 1024)
            if vms_mb > rss_mb * 2:
                if status == 'HEALTHY':
                    status = 'WARNING'
                message += f' (Potential memory fragmentation: VMS={vms_mb:.1f}MB, RSS={rss_mb:.1f}MB)'
            return {'status': status, 'message': message, 'metrics': {'memory_percent': memory_percent, 'rss_mb': rss_mb, 'vms_mb': vms_mb, 'memory_fragmentation_ratio': vms_mb / rss_mb if rss_mb > 0 else 0}}
        except Exception as e:
            return {'status': 'ERROR', 'message': f'Memory health check failed: {str(e)}', 'error': str(e)}

    async def _check_disk_space(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check disk space availability and fragmentation."""
        try:
            data_dir = context.get('data_dir', 'data_cache')
            if os.path.exists(data_dir):
                disk_usage = psutil.disk_usage(data_dir)
            else:
                disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / 1024 ** 3
            total_gb = disk_usage.total / 1024 ** 3
            used_percent = disk_usage.used / disk_usage.total * 100
            status = 'HEALTHY'
            message = f'Sufficient disk space available: {free_gb:.1f}GB free'
            if free_gb < 1:
                status = 'CRITICAL'
                message = f'Critical: Less than 1GB free space ({free_gb:.1f}GB)'
            elif free_gb < 5:
                status = 'WARNING'
                message = f'Warning: Low disk space ({free_gb:.1f}GB free)'
            elif used_percent > 95:
                status = 'CRITICAL'
                message = f'Critical: Disk usage at {used_percent:.1f}%'
            elif used_percent > 85:
                if status == 'HEALTHY':
                    status = 'WARNING'
                message = f'Warning: High disk usage ({used_percent:.1f}%)'
            return {'status': status, 'message': message, 'metrics': {'free_gb': free_gb, 'total_gb': total_gb, 'used_percent': used_percent, 'used_gb': disk_usage.used / 1024 ** 3}}
        except Exception as e:
            return {'status': 'ERROR', 'message': f'Disk space check failed: {str(e)}', 'error': str(e)}

    def _generate_health_recommendations(self, health_results: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on check results."""
        recommendations = []
        if 'system_resources' in health_results:
            sys_res = health_results['system_resources']
            if sys_res.get('status') in ['WARNING', 'CRITICAL']:
                metrics = sys_res.get('metrics', {})
                if metrics.get('cpu_percent', 0) > 80:
                    recommendations.append('Consider reducing concurrent operations to lower CPU usage')
                if metrics.get('memory_percent', 0) > 80:
                    recommendations.append('Monitor memory usage and consider increasing available RAM')
                if metrics.get('disk_percent', 0) > 85:
                    recommendations.append('Clean up old data files or increase disk space')
        if 'data_integrity' in health_results:
            data_int = health_results['data_integrity']
            if data_int.get('status') in ['WARNING', 'CRITICAL']:
                if not data_int.get('data_directory_exists', True):
                    recommendations.append('Create the required data directory structure')
                if not data_int.get('unified_directory_exists', True):
                    recommendations.append('Run data conversion to create unified data structure')
        if 'component_availability' in health_results:
            comp_avail = health_results['component_availability']
            if comp_avail.get('status') == 'CRITICAL':
                recommendations.append('Install missing Python dependencies')
                recommendations.append('Check file system permissions for data directory')
        if 'memory_health' in health_results:
            mem_health = health_results['memory_health']
            if mem_health.get('status') in ['WARNING', 'CRITICAL']:
                recommendations.append('Monitor for memory leaks and consider restarting the process')
                recommendations.append('Consider optimizing data processing to reduce memory usage')
        if 'disk_space' in health_results:
            disk_space = health_results['disk_space']
            if disk_space.get('status') in ['WARNING', 'CRITICAL']:
                recommendations.append('Clean up temporary files and old data')
                recommendations.append('Consider archiving or compressing old data files')
        return recommendations

class Step1_5DataConverterValidator(BaseValidator):
    """Enhanced Validator for Step 1.5: Data Converter with Comprehensive Function Call Monitoring."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__('step01_5_data_converter', config)
        self.logger = system_logger.getChild('Validator.Step1_5')
        self.call_monitor = FunctionCallMonitor(self.logger)
        self.min_records: int = 500
        self.min_files: int = 1
        self.required_columns: list[str] = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.validation_artifacts = {'function_calls': [], 'data_quality_metrics': {}, 'performance_metrics': {}, 'error_summary': {}, 'validation_timeline': []}
        self._apply_function_monitoring()
        self.health_checker = HealthCheckSystem(self.logger)

    def _apply_function_monitoring(self) -> None:
        """Apply function call monitoring to all validator methods."""
        methods_to_monitor = ['validate', '_check_unified_data_structure', '_validate_unified_files', '_validate_single_unified_file', '_validate_unified_config']
        for method_name in methods_to_monitor:
            if hasattr(self, method_name):
                original_method = getattr(self, method_name)
                monitored_method = self.call_monitor.monitor_function(original_method)
                setattr(self, method_name, monitored_method)
                self.logger.info(f'🔍 Applied monitoring to method: {method_name}')

    async def validate(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> bool:
        """Enhanced validation with comprehensive function call monitoring and detailed reporting.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        validation_start_time = time.time()
        validation_id = f'validation_{int(time.time() * 1000000)}'
        self.logger.info('🚀 STARTING ENHANCED STEP 1.5 VALIDATION')
        self.logger.info('=' * 80)
        self.logger.info(f'🆔 Validation ID: {validation_id}')
        self.logger.info(f'⏰ Start time: {datetime.now().isoformat()}')
        symbol: str = str(training_input.get('symbol', 'ETHUSDT'))
        exchange: str = str(training_input.get('exchange', 'BINANCE'))
        timeframe: str = str(training_input.get('timeframe', '1m'))
        data_dir: str = str(training_input.get('data_dir', 'data_cache'))
        self.logger.info(f'🎯 Target: {exchange} {symbol} {timeframe}')
        self.logger.info(f'📁 Data directory: {data_dir}')
        validation_artifacts = {'validation_id': validation_id, 'start_time': validation_start_time, 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'steps_completed': [], 'validation_results': {}, 'performance_metrics': {}, 'errors_encountered': [], 'warnings_issued': []}
        try:
            self.logger.info('\n📋 STEP 0: Running comprehensive health check...')
            health_context = {'data_dir': data_dir, 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
            health_results = await self.health_checker.run_comprehensive_health_check(health_context)
            validation_artifacts['steps_completed'].append('health_check')
            validation_artifacts['validation_results']['health_check'] = health_results
            if health_results['overall_status'] == 'CRITICAL':
                self.logger.error('❌ Critical health issues detected - validation aborted')
                validation_artifacts['errors_encountered'].append('Critical health issues detected')
                await self._generate_comprehensive_validation_report(validation_artifacts, False)
                return False
            elif health_results['overall_status'] == 'DEGRADED':
                self.logger.warning('⚠️ System health degraded - proceeding with caution')
                validation_artifacts['warnings_issued'].append('System health degraded')
            self.logger.info('\n📋 STEP 1: Checking pipeline state...')
            unified_data = pipeline_state.get('unified_data') or {}
            if isinstance(unified_data, dict) and unified_data.get('status') == 'SUCCESS':
                self.logger.info('✅ Unified data present in pipeline state')
                validation_artifacts['steps_completed'].append('pipeline_state_check')
                validation_artifacts['validation_results']['pipeline_state'] = True
                await self._generate_comprehensive_validation_report(validation_artifacts, True)
                return True
            self.logger.info('\n📋 STEP 2: Checking unified data structure...')
            unified_structure = await self._check_unified_data_structure(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir)
            validation_artifacts['steps_completed'].append('unified_structure_check')
            validation_artifacts['validation_results']['unified_structure'] = unified_structure
            if unified_structure['found']:
                self.logger.info(f"✅ Found unified data structure: {unified_structure['base_path']}")
                self.logger.info(f"📊 File count: {unified_structure['file_count']}")
                self.logger.info('\n📋 STEP 3: Validating unified data files...')
                files_validation = await self._validate_unified_files(unified_structure['base_path'], symbol, exchange, timeframe)
                validation_artifacts['steps_completed'].append('files_validation')
                validation_artifacts['validation_results']['files_validation'] = files_validation
                self.logger.info('\n📋 STEP 4: Validating configuration file...')
                config_validation = await self._validate_unified_config(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir)
                validation_artifacts['steps_completed'].append('config_validation')
                validation_artifacts['validation_results']['config_validation'] = config_validation
                overall_success = files_validation and config_validation
                if overall_success:
                    self.logger.info('✅ Unified data validation passed')
                    validation_artifacts['validation_results']['overall_success'] = True
                else:
                    self.logger.warning('⚠️ Unified data validation issues detected')
                    validation_artifacts['validation_results']['overall_success'] = False
                    validation_artifacts['warnings_issued'].append('Validation issues detected in files or config')
                await self._generate_comprehensive_validation_report(validation_artifacts, overall_success)
                return overall_success
            else:
                self.logger.error('❌ No unified data structure found')
                validation_artifacts['validation_results']['overall_success'] = False
                validation_artifacts['errors_encountered'].append('No unified data structure found')
                await self._generate_comprehensive_validation_report(validation_artifacts, False)
                return False
        except Exception as e:
            self.logger.exception(f'❌ CRITICAL ERROR during validation: {e}')
            validation_artifacts['errors_encountered'].append(f'Critical error: {str(e)}')
            validation_artifacts['validation_results']['overall_success'] = False
            await self._generate_comprehensive_validation_report(validation_artifacts, False)
            return False
        finally:
            self.call_monitor.log_comprehensive_report()
            validation_end_time = time.time()
            validation_artifacts['end_time'] = validation_end_time
            validation_artifacts['total_duration'] = validation_end_time - validation_start_time
            validation_artifacts['performance_metrics'] = self.call_monitor.get_call_summary()
            self.logger.info('🏁 ENHANCED STEP 1.5 VALIDATION COMPLETED')
            self.logger.info('=' * 80)

    async def _generate_comprehensive_validation_report(self, validation_artifacts: Dict[str, Any], success: bool) -> None:
        """Generate comprehensive validation report with detailed outcomes."""
        try:
            self.logger.info('\n📊 COMPREHENSIVE VALIDATION REPORT')
            self.logger.info('=' * 80)
            self.logger.info(f"🆔 Validation ID: {validation_artifacts['validation_id']}")
            self.logger.info(f"🎯 Target: {validation_artifacts['exchange']} {validation_artifacts['symbol']} {validation_artifacts['timeframe']}")
            self.logger.info(f"📁 Data Directory: {validation_artifacts['data_dir']}")
            self.logger.info(f'✅ Overall Success: {success}')
            self.logger.info(f"\n📋 Steps Completed ({len(validation_artifacts['steps_completed'])}):")
            for i, step in enumerate(validation_artifacts['steps_completed'], 1):
                self.logger.info(f'   {i}. {step}')
            self.logger.info(f'\n🔍 Validation Results:')
            for key, value in validation_artifacts['validation_results'].items():
                if isinstance(value, bool):
                    status = '✅ PASS' if value else '❌ FAIL'
                    self.logger.info(f'   {key}: {status}')
                elif isinstance(value, dict):
                    self.logger.info(f'   {key}: {value}')
                else:
                    self.logger.info(f'   {key}: {value}')
            if 'health_check' in validation_artifacts['validation_results']:
                health = validation_artifacts['validation_results']['health_check']
                self.logger.info(f'\n🏥 Health Check Results:')
                self.logger.info(f"   🎯 Overall Status: {health.get('overall_status', 'UNKNOWN')}")
                self.logger.info(f"   ✅ Checks Passed: {len(health.get('checks_passed', []))}")
                self.logger.info(f"   ❌ Checks Failed: {len(health.get('checks_failed', []))}")
                self.logger.info(f"   ⚠️ Warnings: {len(health.get('warnings', []))}")
                self.logger.info(f"   🚨 Critical Issues: {len(health.get('critical_issues', []))}")
                if health.get('recommendations'):
                    self.logger.info(f"   💡 Recommendations: {len(health['recommendations'])}")
            if validation_artifacts.get('performance_metrics'):
                perf = validation_artifacts['performance_metrics']
                self.logger.info(f'\n⚡ Performance Metrics:')
                self.logger.info(f"   📞 Total function calls: {perf.get('total_calls', 0)}")
                self.logger.info(f"   🔄 Active calls: {perf.get('active_calls', 0)}")
                self.logger.info(f"   ❌ Total errors: {perf.get('error_count', 0)}")
                self.logger.info(f"   📊 Max call depth: {perf.get('call_stack_depth', 0)}")
                if perf.get('performance_metrics'):
                    self.logger.info(f'   🎯 Function Performance:')
                    for func_name, metrics in perf['performance_metrics'].items():
                        self.logger.info(f'     {func_name}:')
                        self.logger.info(f"       📞 Calls: {metrics['call_count']}")
                        self.logger.info(f"       ⏱️ Avg time: {metrics['avg_time']:.4f}s")
                        self.logger.info(f"       ✅ Success rate: {metrics['success_count']}/{metrics['call_count']}")
            if validation_artifacts['errors_encountered']:
                self.logger.info(f"\n🚨 Errors Encountered ({len(validation_artifacts['errors_encountered'])}):")
                for i, error in enumerate(validation_artifacts['errors_encountered'], 1):
                    self.logger.info(f'   {i}. {error}')
            if validation_artifacts['warnings_issued']:
                self.logger.info(f"\n⚠️ Warnings Issued ({len(validation_artifacts['warnings_issued'])}):")
                for i, warning in enumerate(validation_artifacts['warnings_issued'], 1):
                    self.logger.info(f'   {i}. {warning}')
            if 'total_duration' in validation_artifacts:
                self.logger.info(f"\n⏱️ Total Duration: {validation_artifacts['total_duration']:.4f} seconds")
            if success:
                self.logger.info(f'\n🎉 VALIDATION SUCCESSFUL - All checks passed!')
            else:
                self.logger.info(f'\n❌ VALIDATION FAILED - Issues detected that require attention')
            self.logger.info('=' * 80)
        except Exception as e:
            self.logger.exception(f'❌ Error generating comprehensive validation report: {e}')

    async def _check_unified_data_structure(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> dict[str, Any]:
        """Check for unified data structure in the data directory."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            Dictionary with structure information
        """
        unified_base = os.path.join(data_dir, 'unified', exchange.lower(), symbol, timeframe)
        if os.path.exists(unified_base) and os.path.isdir(unified_base):
            parquet_files = glob.glob(os.path.join(unified_base, '*.parquet'), recursive=True)
            return {'found': True, 'base_path': unified_base, 'parquet_files': parquet_files, 'file_count': len(parquet_files)}
        return {'found': False, 'base_path': unified_base, 'parquet_files': [], 'file_count': 0}

    async def _validate_unified_files(self, base_path: str, symbol: str, exchange: str, timeframe: str) -> bool:
        """Validate the unified data files."

        Args:
            base_path: Base path to unified data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            bool: True if validation passed
        """
        try:
            parquet_files = glob.glob(os.path.join(base_path, '*.parquet'), recursive=True)
            if not parquet_files:
                self.logger.error(f'❌ No parquet files found in {base_path}')
                return False
            self.logger.info(f'📊 Found {len(parquet_files)} parquet files')
            valid_files = 0
            total_records = 0
            for file_path in parquet_files:
                file_validation = await self._validate_single_unified_file(file_path=file_path, symbol=symbol, exchange=exchange, timeframe=timeframe)
                if file_validation['valid']:
                    valid_files += 1
                    total_records += file_validation['records']
                    self.logger.info(f"✅ {os.path.basename(file_path)}: {file_validation['records']} records")
                else:
                    self.logger.warning(f"⚠️ {os.path.basename(file_path)}: {file_validation['error']}")
            if valid_files < self.min_files:
                self.logger.error(f'❌ Insufficient valid files: {valid_files} (minimum: {self.min_files})')
                return False
            if total_records < self.min_records:
                self.logger.warning(f'⚠️ Low total records: {total_records} (minimum: {self.min_records})')
            self.logger.info(f'✅ Unified files validation: {valid_files}/{len(parquet_files)} files, {total_records} total records')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error validating unified files: {e}')
            return False

    async def _validate_single_unified_file(self, file_path: str, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Enhanced validation of a single unified data file with comprehensive checks and detailed reporting.

        Args:
            file_path: Path to the parquet file
            symbol: Expected symbol
            exchange: Expected exchange
            timeframe: Expected timeframe

        Returns:
            Dictionary with comprehensive validation results
        """
        file_validation_start = time.time()
        file_id = f'file_{os.path.basename(file_path)}_{int(time.time() * 1000000)}'
        self.logger.info(f'🔍 ENHANCED FILE VALIDATION: {os.path.basename(file_path)}')
        self.logger.info(f'   🆔 File ID: {file_id}')
        self.logger.info(f'   📁 Path: {file_path}')
        validation_results = {'file_id': file_id, 'file_path': file_path, 'file_name': os.path.basename(file_path), 'validation_start_time': file_validation_start, 'checks_performed': [], 'checks_passed': [], 'checks_failed': [], 'warnings': [], 'data_quality_metrics': {}, 'schema_validation': {}, 'content_validation': {}, 'metadata_validation': {}}
        try:
            self.logger.info(f'   📋 CHECK 1: File existence and accessibility')
            if not os.path.exists(file_path):
                validation_results['checks_failed'].append('file_not_found')
                return {'valid': False, 'records': 0, 'error': f'File not found: {file_path}', 'validation_details': validation_results}
            file_size = os.path.getsize(file_path)
            validation_results['data_quality_metrics']['file_size_bytes'] = file_size
            validation_results['checks_passed'].append('file_exists')
            self.logger.info(f'   ✅ File exists, size: {file_size:,} bytes')
            self.logger.info(f'   📋 CHECK 2: Loading parquet file')
            try:
                df = pd.read_parquet(file_path)
                validation_results['checks_passed'].append('file_load_success')
                validation_results['data_quality_metrics']['total_records'] = len(df)
                validation_results['data_quality_metrics']['total_columns'] = len(df.columns)
                self.logger.info(f'   ✅ File loaded successfully: {len(df):,} records, {len(df.columns)} columns')
            except Exception as e:
                validation_results['checks_failed'].append('file_load_failed')
                return {'valid': False, 'records': 0, 'error': f'File load error: {str(e)}', 'validation_details': validation_results}
            self.logger.info(f'   📋 CHECK 3: Minimum records validation')
            if len(df) < self.min_records:
                validation_results['checks_failed'].append('insufficient_records')
                validation_results['data_quality_metrics']['records_validation'] = {'actual': len(df), 'minimum_required': self.min_records, 'deficit': self.min_records - len(df)}
                return {'valid': False, 'records': len(df), 'error': f'Insufficient records: {len(df)} (minimum: {self.min_records})', 'validation_details': validation_results}
            validation_results['checks_passed'].append('sufficient_records')
            self.logger.info(f'   ✅ Sufficient records: {len(df):,} >= {self.min_records}')
            self.logger.info(f'   📋 CHECK 4: Required columns validation')
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            unified_required = ['exchange', 'symbol', 'timeframe']
            missing_unified = [col for col in unified_required if col not in df.columns]
            validation_results['schema_validation']['required_columns'] = {'present': [col for col in self.required_columns if col in df.columns], 'missing': missing_columns}
            validation_results['schema_validation']['unified_columns'] = {'present': [col for col in unified_required if col in df.columns], 'missing': missing_unified}
            if missing_unified:
                validation_results['checks_failed'].append('missing_unified_columns')
                return {'valid': False, 'records': len(df), 'error': f'Missing unified schema columns: {missing_unified}', 'validation_details': validation_results}
            if missing_columns:
                validation_results['checks_failed'].append('missing_required_columns')
                return {'valid': False, 'records': len(df), 'error': f'Missing columns: {missing_columns}', 'validation_details': validation_results}
            validation_results['checks_passed'].append('all_required_columns_present')
            self.logger.info(f'   ✅ All required columns present')
            self.logger.info(f'   📋 CHECK 5: Timestamp validation')
            if 'timestamp' not in df.columns:
                validation_results['checks_failed'].append('missing_timestamp_column')
                return {'valid': False, 'records': len(df), 'error': 'Missing timestamp column', 'validation_details': validation_results}
            ts_is_datetime = pd.api.types.is_datetime64_any_dtype(df['timestamp'])
            ts_is_numeric = pd.api.types.is_integer_dtype(df['timestamp']) or pd.api.types.is_float_dtype(df['timestamp'])
            validation_results['schema_validation']['timestamp_type'] = {'is_datetime': ts_is_datetime, 'is_numeric': ts_is_numeric, 'dtype': str(df['timestamp'].dtype)}
            if not (ts_is_datetime or ts_is_numeric):
                validation_results['checks_failed'].append('invalid_timestamp_type')
                return {'valid': False, 'records': len(df), 'error': 'Timestamp column must be datetime64 or numeric (ms)', 'validation_details': validation_results}
            validation_results['checks_passed'].append('valid_timestamp_type')
            self.logger.info(f"   ✅ Valid timestamp type: {df['timestamp'].dtype}")
            self.logger.info(f'   📋 CHECK 6: Metadata validation')
            metadata_validation = {}
            try:
                if 'exchange' in df.columns:
                    df_exchange = str(df['exchange'].dropna().iloc[0]).upper()
                    metadata_validation['exchange'] = {'expected': exchange.upper(), 'actual': df_exchange, 'match': df_exchange == exchange.upper()}
                    if df_exchange != exchange.upper():
                        validation_results['checks_failed'].append('exchange_mismatch')
                        return {'valid': False, 'records': len(df), 'error': f'Exchange mismatch in data: {df_exchange} != {exchange}', 'validation_details': validation_results}
                if 'symbol' in df.columns:
                    df_symbol = str(df['symbol'].dropna().iloc[0])
                    metadata_validation['symbol'] = {'expected': symbol, 'actual': df_symbol, 'match': df_symbol == symbol}
                    if df_symbol != symbol:
                        validation_results['checks_failed'].append('symbol_mismatch')
                        return {'valid': False, 'records': len(df), 'error': f'Symbol mismatch in data: {df_symbol} != {symbol}', 'validation_details': validation_results}
                if 'timeframe' in df.columns:
                    df_timeframe = str(df['timeframe'].dropna().iloc[0])
                    metadata_validation['timeframe'] = {'expected': timeframe, 'actual': df_timeframe, 'match': df_timeframe == timeframe}
                    if df_timeframe != timeframe:
                        validation_results['checks_failed'].append('timeframe_mismatch')
                        return {'valid': False, 'records': len(df), 'error': f'Timeframe mismatch in data: {df_timeframe} != {timeframe}', 'validation_details': validation_results}
                validation_results['metadata_validation'] = metadata_validation
                validation_results['checks_passed'].append('metadata_matches')
                self.logger.info(f'   ✅ Metadata validation passed')
            except Exception as e:
                validation_results['warnings'].append(f'Metadata validation warning: {str(e)}')
                self.logger.warning(f'   ⚠️ Metadata validation warning: {str(e)}')
            self.logger.info(f'   📋 CHECK 7: Data quality validation')
            quality_issues = []
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        negative_count = (df[col] < 0).sum()
                        if negative_count > 0:
                            quality_issues.append(f'Negative prices in {col}: {negative_count} records')
                            validation_results['checks_failed'].append(f'negative_prices_{col}')
                        else:
                            validation_results['checks_passed'].append(f'valid_prices_{col}')
            if 'volume' in df.columns and pd.api.types.is_numeric_dtype(df['volume']):
                negative_volume_count = (df['volume'] < 0).sum()
                if negative_volume_count > 0:
                    quality_issues.append(f'Negative volumes: {negative_volume_count} records')
                    validation_results['checks_failed'].append('negative_volumes')
                else:
                    validation_results['checks_passed'].append('valid_volumes')
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            validation_results['data_quality_metrics']['missing_values'] = {'total_missing': int(total_missing), 'missing_by_column': missing_data.to_dict(), 'missing_percentage': total_missing / (len(df) * len(df.columns)) * 100}
            if total_missing > 0:
                validation_results['warnings'].append(f'Missing values detected: {total_missing} total')
                self.logger.warning(f'   ⚠️ Missing values: {total_missing} total')
            if quality_issues:
                validation_results['checks_failed'].extend(['data_quality_issues'])
                return {'valid': False, 'records': len(df), 'error': f"Data quality issues: {'; '.join(quality_issues)}", 'validation_details': validation_results}
            validation_results['checks_passed'].append('data_quality_passed')
            self.logger.info(f'   ✅ Data quality validation passed')
            validation_results['validation_end_time'] = time.time()
            validation_results['validation_duration'] = validation_results['validation_end_time'] - file_validation_start
            validation_results['total_checks'] = len(validation_results['checks_passed']) + len(validation_results['checks_failed'])
            validation_results['success_rate'] = len(validation_results['checks_passed']) / validation_results['total_checks'] if validation_results['total_checks'] > 0 else 0
            self.logger.info(f'   🎉 FILE VALIDATION SUCCESSFUL')
            self.logger.info(f"   📊 Checks passed: {len(validation_results['checks_passed'])}/{validation_results['total_checks']}")
            self.logger.info(f"   ⏱️ Validation duration: {validation_results['validation_duration']:.4f}s")
            return {'valid': True, 'records': len(df), 'error': None, 'validation_details': validation_results}
        except Exception as e:
            validation_results['checks_failed'].append('critical_error')
            validation_results['validation_end_time'] = time.time()
            validation_results['validation_duration'] = validation_results['validation_end_time'] - file_validation_start
            validation_results['critical_error'] = str(e)
            self.logger.error(f'   ❌ CRITICAL ERROR: {str(e)}')
            return {'valid': False, 'records': 0, 'error': f'File validation error: {str(e)}', 'validation_details': validation_results}

    async def _validate_unified_config(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate the unified data configuration file."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            bool: True if validation passed
        """
        try:
            config_path = os.path.join(data_dir, 'unified', f'{exchange.lower()}_{symbol}_{timeframe}_config.json')
            if not os.path.exists(config_path):
                self.logger.warning(f'⚠️ Config file not found: {config_path}')
                return False
            config: Dict[str, Any] = safe_json_load(config_path)
            required_fields = ['symbol', 'exchange', 'timeframe', 'data_path', 'created_at']
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                self.logger.warning(f'⚠️ Missing config fields: {missing_fields}')
                return False
            if str(config.get('symbol')) != symbol:
                self.logger.warning(f"⚠️ Symbol mismatch in config: {config.get('symbol')} != {symbol}")
                return False
            if str(config.get('exchange')).upper() != exchange.upper():
                self.logger.warning(f"⚠️ Exchange mismatch in config: {config.get('exchange')} != {exchange}")
                return False
            if str(config.get('timeframe')) != timeframe:
                self.logger.warning(f"⚠️ Timeframe mismatch in config: {config.get('timeframe')} != {timeframe}")
                return False
            expected_base = os.path.join(data_dir, 'unified', exchange.lower(), symbol, timeframe)
            cfg_path = str(config.get('data_path', ''))
            if not cfg_path:
                self.logger.warning('⚠️ Config missing data_path field')
                return False
            if os.path.abspath(cfg_path) != os.path.abspath(expected_base):
                self.logger.warning(f'⚠️ Config data_path mismatch: {cfg_path} != {expected_base}')
                return False
            if not os.path.isdir(cfg_path):
                self.logger.warning(f'⚠️ Config data_path does not exist: {cfg_path}')
                return False
            self.logger.info(f'✅ Config validation passed: {config_path}')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error validating config: {e}')
            return False

async def run_validator(training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
    """Enhanced Step 1.5 Data Converter validator with comprehensive function call monitoring and detailed reporting.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing comprehensive validation results with detailed function call metrics
    """
    execution_start_time = time.time()
    execution_id = f'execution_{int(time.time() * 1000000)}'
    logger = system_logger.getChild('Step1_5ValidatorRunner')
    logger.info('🚀 STARTING ENHANCED STEP 1.5 VALIDATOR EXECUTION')
    logger.info('=' * 80)
    logger.info(f'🆔 Execution ID: {execution_id}')
    logger.info(f'⏰ Start time: {datetime.now().isoformat()}')
    validator = Step1_5DataConverterValidator(CONFIG)
    execution_metrics = {'execution_id': execution_id, 'start_time': execution_start_time, 'validator_initialization_time': 0, 'validation_execution_time': 0, 'total_execution_time': 0, 'memory_usage_start': psutil.Process().memory_info().rss / 1024 / 1024, 'memory_usage_end': 0, 'memory_delta': 0, 'function_call_summary': {}, 'validation_artifacts': {}, 'errors_encountered': [], 'warnings_issued': []}
    try:
        init_start = time.time()
        logger.info('🔧 Initializing enhanced validator...')
        execution_metrics['validator_initialization_time'] = time.time() - init_start
        logger.info(f"✅ Validator initialized in {execution_metrics['validator_initialization_time']:.4f}s")
        validation_start = time.time()
        logger.info('🔍 Executing enhanced validation...')
        validation_passed = await validator.validate(training_input, pipeline_state)
        execution_metrics['validation_execution_time'] = time.time() - validation_start
        logger.info(f"✅ Validation completed in {execution_metrics['validation_execution_time']:.4f}s")
        execution_metrics['function_call_summary'] = validator.call_monitor.get_call_summary()
        execution_end_time = time.time()
        execution_metrics['total_execution_time'] = execution_end_time - execution_start_time
        execution_metrics['memory_usage_end'] = psutil.Process().memory_info().rss / 1024 / 1024
        execution_metrics['memory_delta'] = execution_metrics['memory_usage_end'] - execution_metrics['memory_usage_start']
        logger.info('\n📊 COMPREHENSIVE EXECUTION REPORT')
        logger.info('=' * 80)
        logger.info(f'🆔 Execution ID: {execution_id}')
        logger.info(f"✅ Validation Result: {('PASSED' if validation_passed else 'FAILED')}")
        logger.info(f"⏱️ Total Execution Time: {execution_metrics['total_execution_time']:.4f}s")
        logger.info(f"🔧 Initialization Time: {execution_metrics['validator_initialization_time']:.4f}s")
        logger.info(f"🔍 Validation Time: {execution_metrics['validation_execution_time']:.4f}s")
        logger.info(f"💾 Memory Delta: {execution_metrics['memory_delta']:+.2f}MB")
        call_summary = execution_metrics['function_call_summary']
        logger.info(f'\n📞 Function Call Metrics:')
        logger.info(f"   📈 Total function calls: {call_summary.get('total_calls', 0)}")
        logger.info(f"   🔄 Active calls: {call_summary.get('active_calls', 0)}")
        logger.info(f"   ❌ Total errors: {call_summary.get('error_count', 0)}")
        logger.info(f"   📊 Max call depth: {call_summary.get('call_stack_depth', 0)}")
        if call_summary.get('performance_metrics'):
            logger.info(f'\n🎯 Function Performance Details:')
            for func_name, metrics in call_summary['performance_metrics'].items():
                logger.info(f'   {func_name}:')
                logger.info(f"     📞 Calls: {metrics['call_count']}")
                logger.info(f"     ⏱️ Avg time: {metrics['avg_time']:.4f}s")
                logger.info(f"     📊 Min/Max: {metrics['min_time']:.4f}s / {metrics['max_time']:.4f}s")
                logger.info(f"     ✅ Success rate: {metrics['success_count']}/{metrics['call_count']}")
                if metrics['total_memory_delta'] != 0:
                    logger.info(f"     💾 Memory delta: {metrics['total_memory_delta']:+.2f}MB")
        if call_summary.get('recent_errors'):
            logger.info(f'\n🚨 Recent Errors:')
            for error in call_summary['recent_errors']:
                logger.info(f"   {error['function_name']}: {error['exception_type']} - {error['exception_message']}")
        logger.info('=' * 80)
        result = {'step_name': 'step01_5_data_converter_enhanced', 'execution_id': execution_id, 'validation_passed': validation_passed, 'validation_results': validator.validation_results, 'execution_metrics': execution_metrics, 'function_call_summary': execution_metrics['function_call_summary'], 'duration': execution_metrics['total_execution_time'], 'timestamp': execution_start_time, 'memory_usage': {'start_mb': execution_metrics['memory_usage_start'], 'end_mb': execution_metrics['memory_usage_end'], 'delta_mb': execution_metrics['memory_delta']}, 'performance_breakdown': {'initialization_time': execution_metrics['validator_initialization_time'], 'validation_time': execution_metrics['validation_execution_time'], 'total_time': execution_metrics['total_execution_time']}}
        if validation_passed:
            logger.info('🎉 ENHANCED STEP 1.5 VALIDATION EXECUTION SUCCESSFUL')
        else:
            logger.warning('⚠️ ENHANCED STEP 1.5 VALIDATION EXECUTION COMPLETED WITH ISSUES')
        return result
    except Exception as e:
        execution_end_time = time.time()
        execution_metrics['total_execution_time'] = execution_end_time - execution_start_time
        execution_metrics['errors_encountered'].append(f'Critical execution error: {str(e)}')
        logger.exception(f'❌ CRITICAL ERROR during validator execution: {e}')
        return {'step_name': 'step01_5_data_converter_enhanced', 'execution_id': execution_id, 'validation_passed': False, 'validation_results': {}, 'execution_metrics': execution_metrics, 'function_call_summary': validator.call_monitor.get_call_summary() if 'validator' in locals() else {}, 'duration': execution_metrics['total_execution_time'], 'timestamp': execution_start_time, 'error': str(e), 'error_type': type(e).__name__, 'critical_failure': True}
if __name__ == '__main__':

    async def test_enhanced_validator() -> None:
        """Test the enhanced validator with comprehensive function call monitoring."""
        print('🚀 TESTING ENHANCED STEP 1.5 VALIDATOR')
        print('=' * 80)
        print('\n📋 TEST CASE 1: Successful validation with existing pipeline state')
        training_input = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m', 'data_dir': 'data_cache'}
        pipeline_state = {'unified_data': {'status': 'SUCCESS', 'duration': 45.2}}
        result1 = await run_validator(training_input, pipeline_state)
        print(f"✅ Test Case 1 Result: {('PASSED' if result1['validation_passed'] else 'FAILED')}")
        print(f"📊 Function calls: {result1['function_call_summary'].get('total_calls', 0)}")
        print(f"⏱️ Duration: {result1['duration']:.4f}s")
        print('\n📋 TEST CASE 2: Validation without pipeline state')
        training_input2 = {'symbol': 'BTCUSDT', 'exchange': 'BINANCE', 'timeframe': '5m', 'data_dir': 'data_cache'}
        pipeline_state2 = {}
        result2 = await run_validator(training_input2, pipeline_state2)
        print(f"✅ Test Case 2 Result: {('PASSED' if result2['validation_passed'] else 'FAILED')}")
        print(f"📊 Function calls: {result2['function_call_summary'].get('total_calls', 0)}")
        print(f"⏱️ Duration: {result2['duration']:.4f}s")
        print('\n📋 TEST CASE 3: Invalid input parameters')
        training_input3 = {'symbol': '', 'exchange': 'INVALID_EXCHANGE', 'timeframe': 'invalid_timeframe', 'data_dir': 'nonexistent_directory'}
        pipeline_state3 = {}
        result3 = await run_validator(training_input3, pipeline_state3)
        print(f"✅ Test Case 3 Result: {('PASSED' if result3['validation_passed'] else 'FAILED')}")
        print(f"📊 Function calls: {result3['function_call_summary'].get('total_calls', 0)}")
        print(f"⏱️ Duration: {result3['duration']:.4f}s")
        print('\n📊 TEST SUMMARY')
        print('=' * 80)
        total_calls = result1['function_call_summary'].get('total_calls', 0) + result2['function_call_summary'].get('total_calls', 0) + result3['function_call_summary'].get('total_calls', 0)
        total_duration = result1['duration'] + result2['duration'] + result3['duration']
        print(f'📈 Total function calls across all tests: {total_calls}')
        print(f'⏱️ Total execution time: {total_duration:.4f}s')
        print(f'🎯 Average calls per test: {total_calls / 3:.1f}')
        print(f'⚡ Average duration per test: {total_duration / 3:.4f}s')
        print(f'\n🎯 PERFORMANCE ANALYSIS:')
        for i, result in enumerate([result1, result2, result3], 1):
            perf = result.get('performance_breakdown', {})
            print(f"   Test {i}: Init={perf.get('initialization_time', 0):.4f}s, Validation={perf.get('validation_time', 0):.4f}s, Total={perf.get('total_time', 0):.4f}s")
        print('\n🎉 ENHANCED VALIDATOR TESTING COMPLETED')
        print('=' * 80)
    if __name__ == '__main__':
        asyncio.run(test_enhanced_validator())