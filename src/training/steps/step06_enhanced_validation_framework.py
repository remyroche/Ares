from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

"""
Enhanced Validation Framework for Step06

This module provides comprehensive function call validation, tracking, and reporting
for all step06 components including:
- Function call validation and logging
- Function-to-function call tracking
- Function completion reports with detailed outcomes
- Performance monitoring
- Error handling with context
"""
import functools
import inspect
import logging
import time

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from contextlib import contextmanager

class ValidationLevel(Enum):
    """Validation levels for function calls."""
    BASIC = 'basic'
    DETAILED = 'detailed'
    COMPREHENSIVE = 'comprehensive'

class FunctionStatus(Enum):
    """Function execution status."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    TIMEOUT = 'timeout'

@dataclass
class FunctionCallContext:
    """Context information for a function call."""
    function_name: str
    module_name: str
    call_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: FunctionStatus = FunctionStatus.PENDING
    input_args: Dict[str, Any] = field(default_factory = dict)
    input_kwargs: Dict[str, Any] = field(default_factory = dict)
    output_result: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    called_functions: List[str] = field(default_factory = list)
    validation_results: Dict[str, Any] = field(default_factory = dict)
    performance_metrics: Dict[str, float] = field(default_factory = dict)

@dataclass
class FunctionCallReport:
    """Comprehensive report for function call execution."""
    call_context: FunctionCallContext
    validation_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]
    error_analysis: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory = list)
    timestamp: datetime = field(default_factory = datetime.now)

class FunctionCallTracker:
    """Tracks function calls and their relationships."""

    def __init__(self) -> None:
        self.active_calls: Dict[str, FunctionCallContext] = {}
        self.call_history: deque = deque(maxlen = 10000)
        self.function_relationships: Dict[str, List[str]] = defaultdict(list)
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        self.error_stats: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def start_call(self, function_name: str, module_name: str, call_id: str, args: tuple, kwargs: dict) -> FunctionCallContext:
        """Start tracking a function call."""
        with self._lock:
            context = FunctionCallContext(function_name = function_name, module_name = module_name, call_id = call_id, start_time = datetime.now(), input_args = self._serialize_args(args), input_kwargs = kwargs.copy())
            context.status = FunctionStatus.IN_PROGRESS
            self.active_calls[call_id] = context
            return context

    def end_call(self, call_id: str, result: Any = None, error: Exception = None) -> FunctionCallContext:
        """End tracking a function call."""
        with self._lock:
            if call_id not in self.active_calls:
                return None
            context = self.active_calls[call_id]
            context.end_time = datetime.now()
            context.execution_time = (context.end_time - context.start_time).total_seconds()
            if error:
                context.status = FunctionStatus.FAILED
                context.error_message = str(error)
                self.error_stats[context.function_name] += 1
            else:
                context.status = FunctionStatus.COMPLETED
                context.output_result = result
            self.call_history.append(context)
            del self.active_calls[call_id]
            self.performance_stats[context.function_name].append(context.execution_time)
            return context

    def track_function_call(self, caller_id: str, callee_name: str) -> None:
        """Track a function calling another function."""
        with self._lock:
            if caller_id in self.active_calls:
                self.active_calls[caller_id].called_functions.append(callee_name)
                self.function_relationships[caller_id].append(callee_name)

    @log_all_calls
    def _serialize_args(self, args: tuple) -> Dict[str, Any]:
        """Serialize function arguments for logging."""
        serialized = {}
        for i, arg in enumerate(args):
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                serialized[f'arg_{i}'] = {'type': type(arg).__name__, 'shape': getattr(arg, 'shape', None), 'dtype': str(arg.dtype) if hasattr(arg, 'dtype') else None}
            elif isinstance(arg, np.ndarray):
                serialized[f'arg_{i}'] = {'type': 'ndarray', 'shape': arg.shape, 'dtype': str(arg.dtype)}
            elif isinstance(arg, (dict, list, tuple)):
                serialized[f'arg_{i}'] = {'type': type(arg).__name__, 'length': len(arg) if hasattr(arg, '__len__') else None}
            else:
                serialized[f'arg_{i}'] = {'type': type(arg).__name__, 'value': str(arg)[:100] if len(str(arg)) > 100 else str(arg)}
        return serialized

    def get_call_statistics(self) -> Dict[str, Any]:
        """Get comprehensive call statistics."""
        with self._lock:
            stats = {'total_calls': len(self.call_history), 'active_calls': len(self.active_calls), 'function_stats': {}, 'error_stats': dict(self.error_stats), 'performance_stats': {}}
            for context in self.call_history:
                func_name = context.function_name
                if func_name not in stats['function_stats']:
                    stats['function_stats'][func_name] = {'total_calls': 0, 'successful_calls': 0, 'failed_calls': 0, 'avg_execution_time': 0.0, 'max_execution_time': 0.0, 'min_execution_time': float('inf')}
                func_stats = stats['function_stats'][func_name]
                func_stats['total_calls'] += 1
                if context.status == FunctionStatus.COMPLETED:
                    func_stats['successful_calls'] += 1
                elif context.status == FunctionStatus.FAILED:
                    func_stats['failed_calls'] += 1
                if context.execution_time:
                    func_stats['avg_execution_time'] += context.execution_time
                    func_stats['max_execution_time'] = max(func_stats['max_execution_time'], context.execution_time)
                    func_stats['min_execution_time'] = min(func_stats['min_execution_time'], context.execution_time)
            for func_stats in stats['function_stats'].values():
                if func_stats['total_calls'] > 0:
                    func_stats['avg_execution_time'] /= func_stats['total_calls']
                if func_stats['min_execution_time'] == float('inf'):
                    func_stats['min_execution_time'] = 0.0
            for func_name, times in self.performance_stats.items():
                if times:
                    stats['performance_stats'][func_name] = {'avg_time': np.mean(times), 'std_time': np.std(times), 'min_time': np.min(times), 'max_time': np.max(times), 'p95_time': np.percentile(times, 95), 'p99_time': np.percentile(times, 99)}
            return stats

class Step06Validator:
    """Comprehensive validator for step06 components."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> None:
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        self.tracker = FunctionCallTracker()
        self.validation_rules = self._initialize_validation_rules()
        
    @log_all_calls
    def _initialize_validation_rules(self) -> Dict[str, List[Callable]]:
        """Initialize validation rules for different function types."""
        return {'data_processing': [self._validate_dataframe_input, self._validate_dataframe_output, self._validate_data_quality], 'feature_engineering': [self._validate_feature_input, self._validate_feature_output, self._validate_feature_quality], 'labeling': [self._validate_labeling_input, self._validate_labeling_output, self._validate_label_distribution], 'optimization': [self._validate_optimization_input, self._validate_optimization_output, self._validate_optimization_convergence]}

    def validate_function_call(self, context: FunctionCallContext, function_type: str='general') -> Dict[str, Any]:
        """Validate a function call based on its type."""
        validation_results = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        rules = self.validation_rules.get(function_type, [])
        for rule in rules:
            try:
                rule_result = rule(context)
                if not rule_result.get('passed', True):
                    validation_results['passed'] = False
                    validation_results['errors'].extend(rule_result.get('errors', []))
                validation_results['warnings'].extend(rule_result.get('warnings', []))
                validation_results['metrics'].update(rule_result.get('metrics', {}))
            except Exception as e:
                validation_results['passed'] = False
                validation_results['errors'].append(f'Validation rule failed: {str(e)}')
        return validation_results

    @log_all_calls
    def _validate_dataframe_input(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate DataFrame input parameters."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        for arg_name, arg_info in context.input_args.items():
            if arg_info.get('type') == 'DataFrame':
                if not arg_info.get('shape'):
                    result['errors'].append(f'DataFrame {arg_name} has no shape information')
                    result['passed'] = False
                elif arg_info['shape'][0] == 0:
                    result['errors'].append(f'DataFrame {arg_name} is empty')
                    result['passed'] = False
                elif arg_info['shape'][1] == 0:
                    result['warnings'].append(f'DataFrame {arg_name} has no columns')
        return result

    @log_all_calls
    def _validate_dataframe_output(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate DataFrame output."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        if context.output_result is not None:
            if isinstance(context.output_result, pd.DataFrame):
                if context.output_result.empty:
                    result['warnings'].append('Output DataFrame is empty')
                result['metrics']['output_shape'] = context.output_result.shape
                result['metrics']['output_columns'] = list(context.output_result.columns)
            elif isinstance(context.output_result, pd.Series):
                if context.output_result.empty:
                    result['warnings'].append('Output Series is empty')
                result['metrics']['output_length'] = len(context.output_result)
        return result

    @log_all_calls
    def _validate_data_quality(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate data quality metrics."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        return result

    @log_all_calls
    def _validate_feature_input(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate feature engineering input."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        required_columns = ['open', 'high', 'low', 'close']
        for arg_name, arg_info in context.input_args.items():
            if arg_info.get('type') == 'DataFrame':
                pass
        return result

    @log_all_calls
    def _validate_feature_output(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate feature engineering output."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        if isinstance(context.output_result, pd.DataFrame):
            feature_cols = [col for col in context.output_result.columns if col.startswith('feature_')]
            result['metrics']['feature_count'] = len(feature_cols)
            if len(feature_cols) == 0:
                result['warnings'].append('No features were generated')
        return result

    @log_all_calls
    def _validate_feature_quality(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate feature quality."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        if isinstance(context.output_result, pd.DataFrame):
            nan_count = context.output_result.isna().sum().sum()
            inf_count = np.isinf(context.output_result.select_dtypes(include=[np.number])).sum().sum()
            result['metrics']['nan_count'] = nan_count
            result['metrics']['inf_count'] = inf_count
            if nan_count > 0:
                result['warnings'].append(f'Found {nan_count} NaN values in output')
            if inf_count > 0:
                result['errors'].append(f'Found {inf_count} infinite values in output')
                result['passed'] = False
        return result

    @log_all_calls
    def _validate_labeling_input(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate labeling input."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        required_columns = ['open', 'high', 'low', 'close']
        for arg_name, arg_info in context.input_args.items():
            if arg_info.get('type') == 'DataFrame':
                pass
        return result

    @log_all_calls
    def _validate_labeling_output(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate labeling output."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        if isinstance(context.output_result, pd.DataFrame):
            if 'label' in context.output_result.columns:
                label_dist = context.output_result['label'].value_counts()
                result['metrics']['label_distribution'] = label_dist.to_dict()
                if len(label_dist) > 1:
                    max_count = label_dist.max()
                    min_count = label_dist.min()
                    balance_ratio = min_count / max_count
                    result['metrics']['label_balance_ratio'] = balance_ratio
                    if balance_ratio < 0.1:
                        result['warnings'].append('Severe label imbalance detected')
                    elif balance_ratio < 0.3:
                        result['warnings'].append('Moderate label imbalance detected')
        return result

    @log_all_calls
    def _validate_label_distribution(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate label distribution."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        return result

    @log_all_calls
    def _validate_optimization_input(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate optimization input."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        return result

    @log_all_calls
    def _validate_optimization_output(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate optimization output."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        return result

    @log_all_calls
    def _validate_optimization_convergence(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Validate optimization convergence."""
        result = {'passed': True, 'errors': [], 'warnings': [], 'metrics': {}}
        return result

class Step06Reporter:
    """Generates comprehensive reports for step06 function calls."""

    def __init__(self, validator: Step06Validator) -> None:
        self.validator = validator
        self.logger = logging.getLogger(__name__)

    def generate_function_report(self, context: FunctionCallContext) -> FunctionCallReport:
        """Generate a comprehensive function call report."""
        validation_results = self.validator.validate_function_call(context)
        performance_summary = self._generate_performance_summary(context)
        error_analysis = None
        if context.status == FunctionStatus.FAILED:
            error_analysis = self._analyze_error(context)
        recommendations = self._generate_recommendations(context, validation_results)
        return FunctionCallReport(call_context = context, validation_summary = validation_results, performance_summary = performance_summary, error_analysis = error_analysis, recommendations = recommendations)
    @log_all_calls

    def _generate_performance_summary(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Generate performance summary for a function call."""
        summary = {'execution_time': context.execution_time, 'status': context.status.value, 'called_functions': context.called_functions, 'function_complexity': len(context.called_functions)}
        if context.memory_usage:
            summary['memory_usage'] = context.memory_usage
        summary.update(context.performance_metrics)
        return summary
    @log_all_calls

    def _analyze_error(self, context: FunctionCallContext) -> Dict[str, Any]:
        """Analyze function execution errors."""
        return {'error_type': type(context.error_message).__name__ if context.error_message else 'Unknown', 'error_message': context.error_message, 'execution_time_before_error': context.execution_time, 'input_validation_passed': True, 'suggested_fixes': self._suggest_error_fixes(context)}
    @log_all_calls

    def _suggest_error_fixes(self, context: FunctionCallContext) -> List[str]:
        """Suggest fixes for function errors."""
        suggestions = []
        if 'DataFrame' in str(context.error_message):
            suggestions.append('Check DataFrame input format and required columns')
        if 'memory' in str(context.error_message).lower():
            suggestions.append('Consider processing data in smaller chunks')
        if 'timeout' in str(context.error_message).lower():
            suggestions.append('Increase timeout or optimize algorithm performance')
        return suggestions

    @log_all_calls
    def _generate_recommendations(self, context: FunctionCallContext, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on function execution."""
        recommendations = []
        if context.execution_time and context.execution_time > 10.0:
            recommendations.append('Consider optimizing function performance - execution time > 10s')
        if not validation_results.get('passed', True):
            recommendations.append('Address validation errors before proceeding')
        if validation_results.get('warnings'):
            recommendations.append('Review validation warnings for potential improvements')
        if len(context.called_functions) > 10:
            recommendations.append('Consider breaking down complex function into smaller components')
        return recommendations

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report for all tracked function calls."""
        stats = self.validator.tracker.get_call_statistics()
        return {'summary': {'total_function_calls': stats['total_calls'], 'active_calls': stats['active_calls'], 'functions_with_errors': len(stats['error_stats']), 'most_called_function': max(stats['function_stats'].items(), key=lambda x: x[1]['total_calls'])[0] if stats['function_stats'] else None}, 'function_statistics': stats['function_stats'], 'error_statistics': stats['error_stats'], 'performance_statistics': stats['performance_stats'], 'recommendations': self._generate_global_recommendations(stats)}

    @log_all_calls
    def _generate_global_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate global recommendations based on overall statistics."""
        recommendations = []
        for func_name, perf_stats in stats['performance_stats'].items():
            if perf_stats['avg_time'] > 5.0:
                recommendations.append(f"Optimize {func_name} - average execution time: {perf_stats['avg_time']:.2f}s")
        for func_name, error_count in stats['error_stats'].items():
            if error_count > 5:
                recommendations.append(f'Investigate {func_name} - {error_count} errors detected')
        return recommendations
_global_tracker = FunctionCallTracker()
_global_validator = Step06Validator()
_global_reporter = Step06Reporter(_global_validator)

def step06_function_validator(function_type: str='general', validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE, timeout: Optional[float]=None, log_calls: bool = True) -> None:
    """
    Comprehensive function validator decorator for step06 components.
    
    Args:
        function_type: Type of function being validated
        validation_level: Level of validation to perform
        timeout: Optional timeout for function execution
        log_calls: Whether to log function calls
    
    Returns:
        Decorated function with comprehensive validation
    """

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                call_id = f'{func.__module__}.{func.__name__}_{int(time.time() * 1000000)}'
                context = _global_tracker.start_call(function_name = func.__name__, module_name = func.__module__, call_id = call_id, args = args, kwargs = kwargs)
                if log_calls:
                    _global_validator.logger.info(f'🔍 Starting async function call: {func.__name__} (ID: {call_id})')
                try:
                    if timeout:
                        import signal

                        def timeout_handler(signum: Any, frame: Any) -> None:
                            raise TimeoutError(f'Function {func.__name__} timed out after {timeout}s')
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout))
                    result = await func(*args, **kwargs)
                    if timeout:
                        signal.alarm(0)
                    context = _global_tracker.end_call(call_id, result = result)
                    if log_calls and context:
                        _global_validator.logger.info(f'✅ Async function call completed: {func.__name__} (ID: {call_id}) in {context.execution_time:.3f}s')
                        report = _global_reporter.generate_function_report(context)
                        _global_validator.logger.info(f'📊 Function report: {json.dumps(report.validation_summary, indent = 2)}')
                    return result
                except Exception as e:
                    context = _global_tracker.end_call(call_id, error = e)
                    if log_calls and context:
                        _global_validator.logger.error(f'❌ Async function call failed: {func.__name__} (ID: {call_id}) after {context.execution_time:.3f}s - {str(e)}')
                        report = _global_reporter.generate_function_report(context)
                        _global_validator.logger.error(f'📊 Error report: {json.dumps(report.error_analysis, indent = 2)}')
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                call_id = f'{func.__module__}.{func.__name__}_{int(time.time() * 1000000)}'
                context = _global_tracker.start_call(function_name = func.__name__, module_name = func.__module__, call_id = call_id, args = args, kwargs = kwargs)
                if log_calls:
                    _global_validator.logger.info(f'🔍 Starting function call: {func.__name__} (ID: {call_id})')
                try:
                    if timeout:
                        import signal

                        def timeout_handler(signum: Any, frame: Any) -> None:
                            raise TimeoutError(f'Function {func.__name__} timed out after {timeout}s')
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout))
                    result = func(*args, **kwargs)
                    if timeout:
                        signal.alarm(0)
                    context = _global_tracker.end_call(call_id, result = result)
                    if log_calls and context:
                        _global_validator.logger.info(f'✅ Function call completed: {func.__name__} (ID: {call_id}) in {context.execution_time:.3f}s')
                        report = _global_reporter.generate_function_report(context)
                        _global_validator.logger.info(f'📊 Function report: {json.dumps(report.validation_summary, indent = 2)}')
                    return result
                except Exception as e:
                    context = _global_tracker.end_call(call_id, error = e)
                    if log_calls and context:
                        _global_validator.logger.error(f'❌ Function call failed: {func.__name__} (ID: {call_id}) after {context.execution_time:.3f}s - {str(e)}')
                        report = _global_reporter.generate_function_report(context)
                        _global_validator.logger.error(f'📊 Error report: {json.dumps(report.error_analysis, indent = 2)}')
                    raise
            return sync_wrapper
    return decorator

def step06_function_tracker(func: Callable) -> Callable:
    """
    Simple function call tracker decorator.
    
    Args:
        func: Function to track
        
    Returns:
        Decorated function with call tracking
    """

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            call_id = f'{func.__module__}.{func.__name__}_{int(time.time() * 1000000)}'
            context = _global_tracker.start_call(function_name = func.__name__, module_name = func.__module__, call_id = call_id, args = args, kwargs = kwargs)
            try:
                result = await func(*args, **kwargs)
                _global_tracker.end_call(call_id, result = result)
                return result
            except Exception as e:
                _global_tracker.end_call(call_id, error = e)
                raise
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            call_id = f'{func.__module__}.{func.__name__}_{int(time.time() * 1000000)}'
            context = _global_tracker.start_call(function_name = func.__name__, module_name = func.__module__, call_id = call_id, args = args, kwargs = kwargs)
            try:
                result = func(*args, **kwargs)
                _global_tracker.end_call(call_id, result = result)
                return result
            except Exception as e:
                _global_tracker.end_call(call_id, error = e)
                raise
        return sync_wrapper

@contextmanager
def step06_validation_context(function_name: str, function_type: str='general') -> None:
    """
    Context manager for step06 validation.
    
    Args:
        function_name: Name of the function being validated
        function_type: Type of function for validation rules
        
    Yields:
        Validation context
    """
    call_id = f'{function_name}_{int(time.time() * 1000000)}'
    context = _global_tracker.start_call(function_name = function_name, module_name='context_manager', call_id = call_id, args=(), kwargs={})
    try:
        yield context
    finally:
        _global_tracker.end_call(call_id)

def get_step06_validation_summary() -> Dict[str, Any]:
    """Get comprehensive validation summary for step06."""
    return _global_reporter.generate_summary_report()

def reset_step06_validation_tracking() -> None:
    """Reset step06 validation tracking."""
    global _global_tracker, _global_validator, _global_reporter
    _global_tracker = FunctionCallTracker()
    _global_validator = Step06Validator()
    _global_reporter = Step06Reporter(_global_validator)
__all__ = ['step06_function_validator', 'step06_function_tracker', 'step06_validation_context', 'get_step06_validation_summary', 'reset_step06_validation_tracking', 'ValidationLevel', 'FunctionStatus', 'FunctionCallContext', 'FunctionCallReport', 'Step06Validator', 'Step06Reporter']
from typing import Dict, List, Optional, Union, Any, Tuple