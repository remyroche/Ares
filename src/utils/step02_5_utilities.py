"""Utility modules for Step02_5 to reduce code duplication and improve maintainability."""

import time
import functools
import traceback
import inspect
from typing import Any, Dict, Callable, List
from src.utils.logger import system_logger

logger = system_logger.getChild('Step02_5Utilities')

class FunctionCallTracker:
    """Centralized function call tracking to reduce duplication."""
    
    def __init__(self):
        self.tracker = {
            'call_count': 0,
            'call_history': [],
            'performance_metrics': {},
            'error_count': 0,
            'success_count': 0
        }
    
    def monitor_function_calls(self, func: Callable) -> Callable:
        """Comprehensive function call monitoring decorator."""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_with_monitoring(func, *args, **kwargs, is_async=True)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._execute_with_monitoring(func, *args, **kwargs, is_async=False)
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    def _execute_with_monitoring(self, func: Callable, *args, **kwargs, is_async: bool = False):
        """Execute function with monitoring."""
        self.tracker['call_count'] += 1
        call_id = self.tracker['call_count']
        func_name = func.__name__
        module_name = func.__module__
        start_time = time.time()
        
        logger.info(f'🔵 FUNCTION ENTRY [{call_id}] - {module_name}.{func_name}')
        logger.info(f'📥 Parameters: args={len(args)}, kwargs={list(kwargs.keys())}')
        
        call_info = {
            'call_id': call_id,
            'function_name': func_name,
            'module_name': module_name,
            'start_time': start_time,
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys()),
            'status': 'running'
        }
        self.tracker['call_history'].append(call_info)
        
        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            call_info.update({
                'status': 'success',
                'execution_time': execution_time,
                'result_type': type(result).__name__,
                'result_size': len(str(result)) if hasattr(result, '__len__') else 1
            })
            
            self._update_metrics(func_name, execution_time, success=True)
            self.tracker['success_count'] += 1
            
            logger.info(f'🟢 FUNCTION EXIT [{call_id}] - {module_name}.{func_name}')
            logger.info(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.info(f'📤 Result type: {type(result).__name__}')
            logger.info(f'✅ Status: SUCCESS')
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            call_info.update({
                'status': 'error',
                'execution_time': execution_time,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            })
            
            self._update_metrics(func_name, execution_time, success=False)
            self.tracker['error_count'] += 1
            
            logger.error(f'🔴 FUNCTION ERROR [{call_id}] - {module_name}.{func_name}')
            logger.error(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.error(f'❌ Error type: {type(e).__name__}')
            logger.error(f'💥 Error message: {str(e)}')
            logger.error(f'📋 Traceback: {traceback.format_exc()}')
            
            raise
    
    def _update_metrics(self, func_name: str, execution_time: float, success: bool):
        """Update performance metrics for a function."""
        if func_name not in self.tracker['performance_metrics']:
            self.tracker['performance_metrics'][func_name] = {
                'total_calls': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'success_count': 0,
                'error_count': 0
            }
        
        metrics = self.tracker['performance_metrics'][func_name]
        metrics['total_calls'] += 1
        metrics['total_time'] += execution_time
        metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
        metrics['min_time'] = min(metrics['min_time'], execution_time)
        metrics['max_time'] = max(metrics['max_time'], execution_time)
        
        if success:
            metrics['success_count'] += 1
        else:
            metrics['error_count'] += 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive function call report."""
        total_calls = self.tracker['call_count']
        success_rate = self.tracker['success_count'] / total_calls * 100 if total_calls > 0 else 0
        
        return {
            'summary': {
                'total_function_calls': total_calls,
                'successful_calls': self.tracker['success_count'],
                'failed_calls': self.tracker['error_count'],
                'success_rate_percent': round(success_rate, 2),
                'report_generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'performance_metrics': self.tracker['performance_metrics'],
            'call_history': self.tracker['call_history'][-50:],
            'top_performing_functions': sorted(
                self.tracker['performance_metrics'].items(),
                key=lambda x: x[1]['avg_time']
            )[:10],
            'most_called_functions': sorted(
                self.tracker['performance_metrics'].items(),
                key=lambda x: x[1]['total_calls'],
                reverse=True
            )[:10]
        }

class InputValidator:
    """Centralized input validation to reduce duplication."""
    
    @staticmethod
    def validate_function_inputs(func: Callable) -> Callable:
        """Validate function inputs and outputs."""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await InputValidator._validate_and_execute(func, *args, **kwargs, is_async=True)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return InputValidator._validate_and_execute(func, *args, **kwargs, is_async=False)
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    
    @staticmethod
    def _validate_and_execute(func: Callable, *args, **kwargs, is_async: bool = False):
        """Validate inputs and execute function."""
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        logger.info(f'🔍 INPUT VALIDATION - {func.__name__}')
        for param_name, param_value in bound_args.arguments.items():
            param_type = sig.parameters[param_name].annotation
            logger.info(f'  📋 {param_name}: {type(param_value).__name__} = {str(param_value)[:100]}...')
            
            if param_type != inspect.Parameter.empty and not isinstance(param_value, param_type):
                logger.warning(f'  ⚠️ Type mismatch for {param_name}: expected {param_type}, got {type(param_value)}')
        
        if is_async:
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        logger.info(f'🔍 OUTPUT VALIDATION - {func.__name__}')
        logger.info(f'  📤 Result type: {type(result).__name__}')
        logger.info(f"  📊 Result size: {(len(str(result)) if hasattr(result, '__len__') else 1)}")
        
        return result

class ErrorHandler:
    """Centralized error handling patterns."""
    
    @staticmethod
    def handles_errors(exceptions: tuple = (Exception,), fallback: Any = None, context: str = ""):
        """Centralized error handling decorator."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.error(f'❌ Error in {func.__name__} ({context}): {e}')
                    logger.error(f'📋 Traceback: {traceback.format_exc()}')
                    return fallback
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.error(f'❌ Error in {func.__name__} ({context}): {e}')
                    logger.error(f'📋 Traceback: {traceback.format_exc()}')
                    return fallback
            
            return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        return decorator

class LoggingPatterns:
    """Centralized logging patterns to reduce duplication."""
    
    @staticmethod
    def log_step_start(step_name: str, details: Dict[str, Any] = None):
        """Log step start with consistent format."""
        logger.info(f'🎯 Starting {step_name}...')
        if details:
            for key, value in details.items():
                logger.info(f'  📋 {key}: {value}')
    
    @staticmethod
    def log_step_completion(step_name: str, execution_time: float, metrics: Dict[str, Any] = None):
        """Log step completion with consistent format."""
        logger.info(f'✅ {step_name} completed in {execution_time:.4f}s')
        if metrics:
            for key, value in metrics.items():
                logger.info(f'  📊 {key}: {value}')
    
    @staticmethod
    def log_data_info(data, data_name: str = "Data"):
        """Log data information with consistent format."""
        logger.info(f'📊 {data_name} info:')
        if hasattr(data, 'shape'):
            logger.info(f'  📋 Shape: {data.shape}')
        if hasattr(data, 'columns'):
            logger.info(f'  📋 Columns: {list(data.columns)}')
        if hasattr(data, 'dtypes'):
            logger.info(f'  📋 Types: {data.dtypes.to_dict()}')

# Global instances for easy access
function_tracker = FunctionCallTracker()
input_validator = InputValidator()
error_handler = ErrorHandler()
logging_patterns = LoggingPatterns()

# Convenience decorators
monitor_function_calls = function_tracker.monitor_function_calls
validate_function_inputs = input_validator.validate_function_inputs
handles_errors = error_handler.handles_errors