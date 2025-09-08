"""Function Call Tracking System for Step 7 Enhanced Matrix Operations.

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This module provides comprehensive function call tracking, validation, and reporting
capabilities to monitor execution flow and performance.
"""
import time
import traceback
import functools
import inspect
from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

class FunctionCallTracker:
    """Comprehensive function call tracking and validation system."""

    def __init__(self, logger):
        self.logger = logger
        self.call_stack = []
        self.function_calls = {}
        self.function_to_function_calls = {}
        self.completion_reports = {}
        self.start_time = time.time()
    
    def track_function_call(self, func_name: str, args: tuple, kwargs: dict, caller: str = None):
        """Track function call initiation."""
        call_id = f"{func_name}_{len(self.call_stack)}_{int(time.time() * 1000)}"
        call_info = {
            'call_id': call_id,
            'function_name': func_name,
            'caller': caller,
            'args_count': len(args),
            'kwargs_count': len(kwargs),
            'start_time': time.time(),
            'args_types': [type(arg).__name__ for arg in args],
            'kwargs_keys': list(kwargs.keys())
        }
        
        self.call_stack.append(call_id)
        self.function_calls[call_id] = call_info
        
        # Track function-to-function calls
        if caller:
            if caller not in self.function_to_function_calls:
                self.function_to_function_calls[caller] = []
            self.function_to_function_calls[caller].append({
                'called_function': func_name,
                'call_id': call_id,
                'timestamp': time.time()
            })
        
        self.logger.debug(f"🔍 Function call initiated: {func_name} (ID: {call_id})")
        return call_id
    
    def track_function_completion(self, call_id: str, result: Any = None, error: Exception = None):
        """Track function call completion with detailed outcome."""
        if call_id not in self.function_calls:
            self.logger.warning(f"⚠️ Unknown call ID: {call_id}")
            return
        
        call_info = self.function_calls[call_id]
        end_time = time.time()
        duration = end_time - call_info['start_time']
        
        completion_report = {
            'call_id': call_id,
            'function_name': call_info['function_name'],
            'caller': call_info['caller'],
            'duration_seconds': duration,
            'success': error is None,
            'error': str(error) if error else None,
            'error_type': type(error).__name__ if error else None,
            'result_type': type(result).__name__ if result is not None else None,
            'result_size': self._get_result_size(result),
            'end_time': end_time,
            'stack_depth': len(self.call_stack)
        }
        
        self.completion_reports[call_id] = completion_report
        
        # Remove from call stack
        if call_id in self.call_stack:
            self.call_stack.remove(call_id)
        
        # Log completion
        status = "✅" if error is None else "❌"
        self.logger.info(f"{status} Function completed: {call_info['function_name']} "
                        f"(ID: {call_id}, Duration: {duration:.3f}s)")
        
        if error:
            self.logger.error(f"❌ Function error: {call_info['function_name']} - {error}")
            self.logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        return completion_report

    def _get_result_size(self, result: Any) -> str:
        """Get human-readable size of result."""
        if result is None:
            return "None"
        elif isinstance(result, (list, tuple)):
            return f"len={len(result)}"
        elif isinstance(result, dict):
            return f"keys={len(result)}"
        else:
            # Handle numpy arrays and pandas DataFrames
            try:
                if isinstance(result, np.ndarray):
                    return f"shape={result.shape}"
                elif isinstance(result, pd.DataFrame):
                    return f"shape={result.shape}"
            except ImportError:
                pass
            return f"type={type(result).__name__}"
    
    def get_call_summary(self) -> Dict[str, Any]:
        """Get comprehensive call summary."""
        total_calls = len(self.function_calls)
        successful_calls = len([r for r in self.completion_reports.values() if r['success']])
        failed_calls = total_calls - successful_calls
        
        total_duration = sum(r['duration_seconds'] for r in self.completion_reports.values())
        
        return {
            'total_function_calls': total_calls,
            'successful_calls': successful_calls,
            'failed_calls': failed_calls,
            'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
            'total_duration_seconds': total_duration,
            'average_duration_seconds': total_duration / total_calls if total_calls > 0 else 0,
            'function_to_function_calls': len(self.function_to_function_calls),
            'max_stack_depth': max((r['stack_depth'] for r in self.completion_reports.values()), default = 0),
            'session_duration_seconds': time.time() - self.start_time
        }

def comprehensive_function_tracker(logger):
    """Decorator for comprehensive function call tracking."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get caller information
            frame = inspect.currentframe().f_back
            caller_name = frame.f_code.co_name if frame else "unknown"
            
            # Get tracker from self if available
            tracker = None
            if args and hasattr(args[0], 'call_tracker'):
                tracker = args[0].call_tracker
            
            if tracker is None:
                # Create temporary tracker
                tracker = FunctionCallTracker(logger)
            
            call_id = tracker.track_function_call(
                func.__name__, 
                args, 
                kwargs, 
                caller_name
            )
            
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                tracker.track_function_completion(call_id, result)
                return result
                
            except Exception as e:
                tracker.track_function_completion(call_id, error = e)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get caller information
            frame = inspect.currentframe().f_back
            caller_name = frame.f_code.co_name if frame else "unknown"
            
            # Get tracker from self if available
            tracker = None
            if args and hasattr(args[0], 'call_tracker'):
                tracker = args[0].call_tracker
            
            if tracker is None:
                # Create temporary tracker
                tracker = FunctionCallTracker(logger)
            
            call_id = tracker.track_function_call(
                func.__name__, 
                args, 
                kwargs, 
                caller_name
            )
            
            try:
                result = func(*args, **kwargs)
                tracker.track_function_completion(call_id, result)
                return result
                
            except Exception as e:
                tracker.track_function_completion(call_id, error = e)
                raise
        
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

__all__ = ['FunctionCallTracker', 'comprehensive_function_tracker']