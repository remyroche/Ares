"""
Standalone Test for Enhanced Step02 Function Monitoring

This script tests the core function monitoring mechanisms in isolation.
"""
import asyncio
import sys
import time
import traceback
import inspect
import contextvars
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import functools
import numpy as np

class FunctionCallStatus(Enum):
    """Status of function calls."""
    PENDING = 'PENDING'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    TIMEOUT = 'TIMEOUT'
    CANCELLED = 'CANCELLED'

@dataclass
class FunctionCallContext:
    """Context for function call monitoring."""
    function_name: str
    module_name: str
    call_id: str
    start_time: float
    end_time: Optional[float] = None
    status: FunctionCallStatus = FunctionCallStatus.PENDING
    input_args: Dict[str, Any] = field(default_factory=dict)
    input_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_result: Any = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    called_functions: List[str] = field(default_factory=list)
    parent_call_id: Optional[str] = None
    child_calls: List[str] = field(default_factory=list)

@dataclass
class FunctionInteractionReport:
    """Report of function interactions and outcomes."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    function_call_details: List[FunctionCallContext] = field(default_factory=list)
    call_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)

class FunctionCallMonitor:
    """Comprehensive function call monitoring system with performance tracking."""

    def __init__(self) -> None:
        self.active_calls: Dict[str, FunctionCallContext] = {}
        self.completed_calls: List[FunctionCallContext] = []
        self.call_counter = 0
        self.logger = None
        self.performance_metrics = {'memory_usage': [], 'cpu_usage': [], 'execution_times': [], 'error_rates': []}
        self._setup_logger()
        self._setup_performance_monitoring()

    def _setup_logger(self) -> None:
        """Setup logger for function monitoring."""
        import logging
        self.logger = logging.getLogger(f'{__name__}.FunctionCallMonitor')
        logging.basicConfig(level=logging.INFO)

    def _setup_performance_monitoring(self) -> None:
        """Setup performance monitoring capabilities."""
        try:
            import psutil

            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            self.logger.warning('⚠️ psutil not available - performance monitoring limited')

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.psutil_available:
            try:
                memory_info = self.process.memory_info()
                return memory_info.rss / 1024 / 1024
            except Exception:
                return 0.0
        return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if self.psutil_available:
            try:
                return self.process.cpu_percent()
            except Exception:
                return 0.0
        return 0.0

    def _generate_call_id(self, function_name: str) -> str:
        """Generate unique call ID."""
        self.call_counter += 1
        return f'{function_name}_{self.call_counter}_{int(time.time() * 1000)}'

    def start_function_call(self, func: Callable, args: tuple, kwargs: dict, parent_call_id: Optional[str]=None) -> str:
        """Start monitoring a function call with enhanced interaction tracking."""
        call_id = self._generate_call_id(func.__name__)
        input_args = {}
        for i, arg in enumerate(args):
            arg_type = type(arg).__name__
            arg_value = str(arg)[:100] if arg is not None else 'None'
            input_args[f'arg_{i}'] = {'type': arg_type, 'value': arg_value, 'size': len(str(arg)) if hasattr(arg, '__len__') else 0}
        input_kwargs = {}
        for k, v in kwargs.items():
            val_type = type(v).__name__
            val_value = str(v)[:100] if v is not None else 'None'
            input_kwargs[k] = {'type': val_type, 'value': val_value, 'size': len(str(v)) if hasattr(v, '__len__') else 0}
        initial_memory = self._get_memory_usage()
        initial_cpu = self._get_cpu_usage()
        context = FunctionCallContext(function_name=func.__name__, module_name=func.__module__, call_id=call_id, start_time=time.time(), status=FunctionCallStatus.IN_PROGRESS, input_args=input_args, input_kwargs=input_kwargs, parent_call_id=parent_call_id, memory_usage=initial_memory, cpu_usage=initial_cpu)
        self.active_calls[call_id] = context
        if parent_call_id and parent_call_id in self.active_calls:
            self.active_calls[parent_call_id].child_calls.append(call_id)
            parent_function = self.active_calls[parent_call_id].function_name
            if parent_function not in context.called_functions:
                context.called_functions.append(parent_function)
        self.logger.info(f'🔍 Function call started: {func.__name__} (ID: {call_id})')
        self.logger.info(f'   - Module: {func.__module__}')
        self.logger.info(f"   - Parent call: {(parent_call_id if parent_call_id else 'None')}")
        self.logger.info(f'   - Input args: {len(input_args)} arguments')
        self.logger.info(f'   - Input kwargs: {len(input_kwargs)} keyword arguments')
        return call_id

    def complete_function_call(self, call_id: str, result: Any=None, error: Optional[Exception]=None) -> None:
        """Complete monitoring a function call with detailed outcome analysis."""
        if call_id not in self.active_calls:
            self.logger.warning(f'⚠️ Unknown call ID: {call_id}')
            return
        context = self.active_calls[call_id]
        context.end_time = time.time()
        context.execution_time = context.end_time - context.start_time
        final_memory = self._get_memory_usage()
        final_cpu = self._get_cpu_usage()
        memory_delta = final_memory - (context.memory_usage or 0)
        cpu_delta = final_cpu - (context.cpu_usage or 0)
        self.performance_metrics['memory_usage'].append(memory_delta)
        self.performance_metrics['cpu_usage'].append(cpu_delta)
        self.performance_metrics['execution_times'].append(context.execution_time)
        if result is not None:
            result_type = type(result).__name__
            result_size = len(str(result)) if hasattr(result, '__len__') else 0
            context.output_result = {'type': result_type, 'value': str(result)[:200], 'size': result_size, 'is_dataframe': hasattr(result, 'shape') and hasattr(result, 'columns'), 'is_dict': isinstance(result, dict), 'is_list': isinstance(result, list)}
        else:
            context.output_result = {'type': 'NoneType', 'value': 'None', 'size': 0, 'is_dataframe': False, 'is_dict': False, 'is_list': False}
        if error:
            context.status = FunctionCallStatus.FAILED
            context.error_details = {'error_type': type(error).__name__, 'error_message': str(error), 'traceback': traceback.format_exc(), 'error_location': f'{error.__class__.__module__}.{error.__class__.__name__}', 'error_severity': 'HIGH' if isinstance(error, (ValueError, TypeError, AttributeError)) else 'MEDIUM'}
            self.logger.error(f'❌ Function call failed: {context.function_name} (ID: {call_id})')
            self.logger.error(f'   - Error type: {type(error).__name__}')
            self.logger.error(f'   - Error message: {str(error)}')
            self.logger.error(f'   - Execution time: {context.execution_time:.3f}s')
            self.logger.error(f'   - Child calls: {len(context.child_calls)}')
        else:
            context.status = FunctionCallStatus.COMPLETED
            self.logger.info(f'✅ Function call completed: {context.function_name} (ID: {call_id})')
            self.logger.info(f'   - Execution time: {context.execution_time:.3f}s')
            self.logger.info(f"   - Result type: {context.output_result['type']}")
            self.logger.info(f"   - Result size: {context.output_result['size']}")
            self.logger.info(f'   - Child calls: {len(context.child_calls)}')
            if context.execution_time > 1.0:
                self.logger.warning(f'⚠️ Slow function execution: {context.function_name} took {context.execution_time:.3f}s')
            elif context.execution_time < 0.001:
                self.logger.info(f'⚡ Fast function execution: {context.function_name} took {context.execution_time:.3f}s')
        self.completed_calls.append(context)
        del self.active_calls[call_id]

    def get_function_interaction_report(self) -> FunctionInteractionReport:
        """Generate comprehensive function interaction report."""
        total_calls = len(self.completed_calls)
        successful_calls = len([c for c in self.completed_calls if c.status == FunctionCallStatus.COMPLETED])
        failed_calls = len([c for c in self.completed_calls if c.status == FunctionCallStatus.FAILED])
        total_execution_time = sum((c.execution_time or 0 for c in self.completed_calls))
        average_execution_time = total_execution_time / total_calls if total_calls > 0 else 0.0
        call_hierarchy = {}
        for call in self.completed_calls:
            if call.parent_call_id:
                if call.parent_call_id not in call_hierarchy:
                    call_hierarchy[call.parent_call_id] = []
                call_hierarchy[call.parent_call_id].append(call.call_id)
        error_summary = {}
        for call in self.completed_calls:
            if call.error_details:
                error_type = call.error_details.get('error_type', 'Unknown')
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        if self.completed_calls:
            fastest_call = min(self.completed_calls, key=lambda c: c.execution_time or float('inf'))
            slowest_call = max(self.completed_calls, key=lambda c: c.execution_time or 0)
            function_frequency = {}
            for call in self.completed_calls:
                function_frequency[call.function_name] = function_frequency.get(call.function_name, 0) + 1
            most_called_function = max(function_frequency.items(), key=lambda x: x[1])[0] if function_frequency else None
            execution_times = [c.execution_time for c in self.completed_calls if c.execution_time is not None]
            median_execution_time = sorted(execution_times)[len(execution_times) // 2] if execution_times else 0.0
            max_depth = 0
            for call in self.completed_calls:
                depth = self._calculate_call_depth(call.call_id)
                max_depth = max(max_depth, depth)
            dataframe_calls = len([c for c in self.completed_calls if c.output_result and c.output_result.get('is_dataframe', False)])
            dict_calls = len([c for c in self.completed_calls if c.output_result and c.output_result.get('is_dict', False)])
            list_calls = len([c for c in self.completed_calls if c.output_result and c.output_result.get('is_list', False)])
            performance_metrics = {'fastest_call': fastest_call.function_name, 'fastest_call_time': fastest_call.execution_time, 'slowest_call': slowest_call.function_name, 'slowest_call_time': slowest_call.execution_time, 'most_called_function': most_called_function, 'most_called_count': function_frequency.get(most_called_function, 0) if most_called_function else 0, 'success_rate': successful_calls / total_calls * 100 if total_calls > 0 else 0.0, 'median_execution_time': median_execution_time, 'max_call_depth': max_depth, 'dataframe_operations': dataframe_calls, 'dict_operations': dict_calls, 'list_operations': list_calls, 'function_frequency': function_frequency}
        else:
            performance_metrics = {'fastest_call': None, 'fastest_call_time': 0.0, 'slowest_call': None, 'slowest_call_time': 0.0, 'most_called_function': None, 'most_called_count': 0, 'success_rate': 0.0, 'median_execution_time': 0.0, 'max_call_depth': 0, 'dataframe_operations': 0, 'dict_operations': 0, 'list_operations': 0, 'function_frequency': {}}
        return FunctionInteractionReport(total_calls=total_calls, successful_calls=successful_calls, failed_calls=failed_calls, total_execution_time=total_execution_time, average_execution_time=average_execution_time, function_call_details=self.completed_calls.copy(), call_hierarchy=call_hierarchy, performance_metrics=performance_metrics, error_summary=error_summary)

    def _calculate_call_depth(self, call_id: str) -> int:
        """Calculate the depth of a function call in the hierarchy."""
        depth = 0
        current_call_id = call_id
        current_call = None
        for call in self.completed_calls:
            if call.call_id == current_call_id:
                current_call = call
                break
        if not current_call:
            return 0
        while current_call and current_call.parent_call_id:
            depth += 1
            parent_call_id = current_call.parent_call_id
            current_call = None
            for call in self.completed_calls:
                if call.call_id == parent_call_id:
                    current_call = call
                    break
        return depth
function_monitor = FunctionCallMonitor()
current_call_context = contextvars.ContextVar('current_call_id', default=None)

def comprehensive_function_monitoring(validate_inputs: bool=True, validate_outputs: bool=True, track_performance: bool=True, track_memory: bool=True, timeout_seconds: Optional[int]=None, retry_attempts: int=0) -> None:
    """Comprehensive decorator for function call monitoring and validation."""

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            parent_call_id = current_call_context.get()
            call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            token = current_call_context.set(call_id)
            try:
                if validate_inputs:
                    await _validate_function_inputs(func, args, kwargs)
                if timeout_seconds:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                else:
                    result = await func(*args, **kwargs)
                if validate_outputs:
                    await _validate_function_outputs(result)
                function_monitor.complete_function_call(call_id, result)
                return result
            except Exception as e:
                function_monitor.complete_function_call(call_id, error=e)
                if retry_attempts > 0:
                    return await _retry_function_call(func, args, kwargs, retry_attempts, call_id)
                raise
            finally:
                current_call_context.reset(token)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            parent_call_id = current_call_context.get()
            call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            token = current_call_context.set(call_id)
            try:
                if validate_inputs:
                    _validate_function_inputs_sync(func, args, kwargs)
                result = func(*args, **kwargs)
                if validate_outputs:
                    _validate_function_outputs_sync(result)
                function_monitor.complete_function_call(call_id, result)
                return result
            except Exception as e:
                function_monitor.complete_function_call(call_id, error=e)
                if retry_attempts > 0:
                    return _retry_function_call_sync(func, args, kwargs, retry_attempts, call_id)
                raise
            finally:
                current_call_context.reset(token)
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

async def _validate_function_inputs(func: Callable, args: tuple, kwargs: dict) -> None:
    """Validate function inputs with comprehensive error handling."""
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param_name, param_value in bound_args.arguments.items():
            param_annotation = sig.parameters[param_name].annotation
            if param_annotation == str and (not isinstance(param_value, str)):
                raise TypeError(f"Parameter '{param_name}' must be a string, got {type(param_value).__name__}")
            if 'path' in param_name.lower() or 'dir' in param_name.lower():
                if param_value and (not isinstance(param_value, (str, Path))):
                    raise TypeError(f"Parameter '{param_name}' must be a string or Path, got {type(param_value).__name__}")
        pass
    except Exception as e:
        function_monitor.logger.error(f'❌ Input validation failed for {func.__name__}: {e}')
        raise ValueError(f'Input validation failed: {e}') from e

def _validate_function_inputs_sync(func: Callable, args: tuple, kwargs: dict) -> None:
    """Validate function inputs (sync version) with comprehensive error handling."""
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param_name, param_value in bound_args.arguments.items():
            param_annotation = sig.parameters[param_name].annotation
            if param_annotation == str and (not isinstance(param_value, str)):
                raise TypeError(f"Parameter '{param_name}' must be a string, got {type(param_value).__name__}")
            if 'path' in param_name.lower() or 'dir' in param_name.lower():
                if param_value and (not isinstance(param_value, (str, Path))):
                    raise TypeError(f"Parameter '{param_name}' must be a string or Path, got {type(param_value).__name__}")
        pass
    except Exception as e:
        function_monitor.logger.error(f'❌ Input validation failed for {func.__name__}: {e}')
        raise ValueError(f'Input validation failed: {e}') from e

async def _validate_function_outputs(result: Any) -> None:
    """Validate function outputs with comprehensive error handling."""
    try:
        if result is None:
            raise ValueError('Function returned None')
        if hasattr(result, 'shape') and hasattr(result, 'columns'):
            if result.shape[0] == 0:
                raise ValueError('Function returned empty DataFrame')
            if result.shape[1] == 0:
                raise ValueError('Function returned DataFrame with no columns')
        elif isinstance(result, dict):
            if not result:
                raise ValueError('Function returned empty dictionary')
        elif isinstance(result, list):
            if not result:
                raise ValueError('Function returned empty list')
        pass
    except Exception as e:
        function_monitor.logger.error(f'❌ Output validation failed: {e}')
        raise ValueError(f'Output validation failed: {e}') from e

def _validate_function_outputs_sync(result: Any) -> None:
    """Validate function outputs (sync version) with comprehensive error handling."""
    try:
        if result is None:
            raise ValueError('Function returned None')
        if hasattr(result, 'shape') and hasattr(result, 'columns'):
            if result.shape[0] == 0:
                raise ValueError('Function returned empty DataFrame')
            if result.shape[1] == 0:
                raise ValueError('Function returned DataFrame with no columns')
        elif isinstance(result, dict):
            if not result:
                raise ValueError('Function returned empty dictionary')
        elif isinstance(result, list):
            if not result:
                raise ValueError('Function returned empty list')
        pass
    except Exception as e:
        function_monitor.logger.error(f'❌ Output validation failed: {e}')
        raise ValueError(f'Output validation failed: {e}') from e

async def _retry_function_call(func: Callable, args: tuple, kwargs: dict, retry_attempts: int, original_call_id: str) -> Any:
    """Retry function call with monitoring."""
    for attempt in range(retry_attempts):
        try:
            parent_call_id = current_call_context.get()
            retry_call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            token = current_call_context.set(retry_call_id)
            try:
                result = await func(*args, **kwargs)
                function_monitor.complete_function_call(retry_call_id, result)
                return result
            finally:
                current_call_context.reset(token)
        except Exception as e:
            function_monitor.complete_function_call(retry_call_id, error=e)
            if attempt == retry_attempts - 1:
                raise
            await asyncio.sleep(0.1 * 2 ** attempt)

def _retry_function_call_sync(func: Callable, args: tuple, kwargs: dict, retry_attempts: int, original_call_id: str) -> Any:
    """Retry function call with monitoring (sync version)."""
    for attempt in range(retry_attempts):
        try:
            parent_call_id = current_call_context.get()
            retry_call_id = function_monitor.start_function_call(func, args, kwargs, parent_call_id)
            token = current_call_context.set(retry_call_id)
            try:
                result = func(*args, **kwargs)
                function_monitor.complete_function_call(retry_call_id, result)
                return result
            finally:
                current_call_context.reset(token)
        except Exception as e:
            function_monitor.complete_function_call(retry_call_id, error=e)
            if attempt == retry_attempts - 1:
                raise
            time.sleep(0.1 * 2 ** attempt)

class StandaloneStep02Tester:
    """Standalone test suite for Step02 function monitoring."""

    def __init__(self) -> None:
        self.test_results = {}

    async def test_basic_function_monitoring(self) -> Dict[str, Any]:
        """Test basic function call monitoring."""
        print('\n🔍 Testing Basic Function Call Monitoring...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def test_function(x: int) -> int:
                """Simple test function."""
                await asyncio.sleep(0.01)
                return x * 2
            result = await test_function(5)
            report = function_monitor.get_function_interaction_report()
            test_passed = result == 10 and report.total_calls == 1 and (report.successful_calls == 1) and (report.failed_calls == 0)
            print(f'   - Function result: {result}')
            print(f'   - Total calls: {report.total_calls}')
            print(f'   - Successful calls: {report.successful_calls}')
            print(f'   - Failed calls: {report.failed_calls}')
            print(f"   - Success rate: {report.performance_metrics.get('success_rate', 0):.1f}%")
            return {'test_name': 'basic_function_monitoring', 'passed': test_passed, 'result': result, 'report': report}
        except Exception as e:
            print(f'❌ Basic function monitoring test failed: {e}')
            return {'test_name': 'basic_function_monitoring', 'passed': False, 'error': str(e)}

    async def test_function_interaction_tracking(self) -> Dict[str, Any]:
        """Test function-to-function call tracking."""
        print('\n🔗 Testing Function Interaction Tracking...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def parent_function(x: int) -> Dict[str, Any]:
                """Parent function."""
                child1_result = await child_function_1(x)
                child2_result = await child_function_2(x)
                return {'parent': x, 'child1': child1_result, 'child2': child2_result}

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def child_function_1(x: int) -> int:
                """Child function 1."""
                await asyncio.sleep(0.01)
                return x + 1

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def child_function_2(x: int) -> int:
                """Child function 2."""
                await asyncio.sleep(0.01)
                return x * 2
            result = await parent_function(5)
            report = function_monitor.get_function_interaction_report()
            test_passed = report.total_calls >= 3 and len(report.call_hierarchy) > 0 and (report.performance_metrics.get('max_call_depth', 0) > 0)
            print(f'   - Total calls: {report.total_calls}')
            print(f'   - Call hierarchy entries: {len(report.call_hierarchy)}')
            print(f"   - Max call depth: {report.performance_metrics.get('max_call_depth', 0)}")
            print(f"   - Function frequency: {report.performance_metrics.get('function_frequency', {})}")
            return {'test_name': 'function_interaction_tracking', 'passed': test_passed, 'result': result, 'report': report}
        except Exception as e:
            print(f'❌ Function interaction tracking test failed: {e}')
            return {'test_name': 'function_interaction_tracking', 'passed': False, 'error': str(e)}

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        print('\n⚠️ Testing Error Handling...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def failing_function(x: int) -> int:
                """Function that fails with negative input."""
                if x < 0:
                    raise ValueError('Negative value not allowed')
                return x * 2
            result1 = await failing_function(5)
            try:
                result2 = await failing_function(-1)
                error_handled = False
            except ValueError:
                error_handled = True
            report = function_monitor.get_function_interaction_report()
            test_passed = result1 == 10 and error_handled and (report.failed_calls > 0) and (len(report.error_summary) > 0)
            print(f'   - Successful call result: {result1}')
            print(f'   - Error handled correctly: {error_handled}')
            print(f'   - Failed calls: {report.failed_calls}')
            print(f'   - Error summary: {report.error_summary}')
            return {'test_name': 'error_handling', 'passed': test_passed, 'result1': result1, 'error_handled': error_handled, 'report': report}
        except Exception as e:
            print(f'❌ Error handling test failed: {e}')
            return {'test_name': 'error_handling', 'passed': False, 'error': str(e)}

    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring."""
        print('\n⚡ Testing Performance Monitoring...')
        try:
            function_monitor.active_calls.clear()
            function_monitor.completed_calls.clear()
            function_monitor.call_counter = 0

            @comprehensive_function_monitoring(validate_inputs=True, validate_outputs=True, track_performance=True, timeout_seconds=30, retry_attempts=1)
            async def performance_function(duration: float) -> Dict[str, Any]:
                """Function with controllable duration."""
                await asyncio.sleep(duration)
                return {'duration': duration, 'timestamp': time.time()}
            await performance_function(0.01)
            await performance_function(0.05)
            await performance_function(0.02)
            report = function_monitor.get_function_interaction_report()
            test_passed = report.total_calls == 3 and report.performance_metrics.get('fastest_call') is not None and (report.performance_metrics.get('slowest_call') is not None) and (report.performance_metrics.get('median_execution_time', 0) > 0)
            print(f'   - Total calls: {report.total_calls}')
            print(f"   - Fastest call: {report.performance_metrics.get('fastest_call')}")
            print(f"   - Slowest call: {report.performance_metrics.get('slowest_call')}")
            print(f"   - Median execution time: {report.performance_metrics.get('median_execution_time', 0):.3f}s")
            print(f'   - Average execution time: {report.average_execution_time:.3f}s')
            return {'test_name': 'performance_monitoring', 'passed': test_passed, 'report': report}
        except Exception as e:
            print(f'❌ Performance monitoring test failed: {e}')
            return {'test_name': 'performance_monitoring', 'passed': False, 'error': str(e)}

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        print('🚀 Starting Standalone Step02 Function Monitoring Tests...')
        print('=' * 60)
        test_methods = [self.test_basic_function_monitoring, self.test_function_interaction_tracking, self.test_error_handling, self.test_performance_monitoring]
        all_results = {}
        passed_tests = 0
        total_tests = len(test_methods)
        for test_method in test_methods:
            try:
                result = await test_method()
                all_results[result['test_name']] = result
                if result['passed']:
                    passed_tests += 1
                    print(f"✅ {result['test_name']}: PASSED")
                else:
                    print(f"❌ {result['test_name']}: FAILED")
                    if 'error' in result:
                        print(f"   Error: {result['error']}")
            except Exception as e:
                print(f'❌ {test_method.__name__}: EXCEPTION - {e}')
                all_results[test_method.__name__] = {'test_name': test_method.__name__, 'passed': False, 'error': str(e)}
        print('\n' + '=' * 60)
        print('📊 TEST RESULTS SUMMARY')
        print('=' * 60)
        print(f'Total Tests: {total_tests}')
        print(f'Passed: {passed_tests}')
        print(f'Failed: {total_tests - passed_tests}')
        print(f'Success Rate: {passed_tests / total_tests * 100:.1f}%')
        return {'total_tests': total_tests, 'passed_tests': passed_tests, 'failed_tests': total_tests - passed_tests, 'success_rate': passed_tests / total_tests * 100, 'detailed_results': all_results}

async def main() -> None:
    """Main test execution."""
    tester = StandaloneStep02Tester()
    try:
        results = await tester.run_all_tests()
        print('\n' + '=' * 60)
        if results['success_rate'] == 100:
            print('🎉 ALL TESTS PASSED! Step02 Function Monitoring is working perfectly!')
        elif results['success_rate'] >= 75:
            print('✅ MOSTLY SUCCESSFUL! Step02 Function Monitoring is working well.')
        else:
            print('⚠️ SOME ISSUES DETECTED! Step02 Function Monitoring needs attention.')
        print(f"Overall Success Rate: {results['success_rate']:.1f}%")
        print('=' * 60)
        return results
    except Exception as e:
        print(f'❌ Test execution failed: {e}')
        return {'success_rate': 0, 'error': str(e)}
if __name__ == '__main__':
    results = asyncio.run(main())
    if results.get('success_rate', 0) == 100:
        sys.exit(0)
    else:
        sys.exit(1)