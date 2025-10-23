"""
Comprehensive Function Call Logging System for Ares Training Steps.

This module provides enhanced logging decorators that ensure all functions
log their entry, exit, internal calls, and completion with descriptive messages.
"""

import functools
import time
import inspect

import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from contextvars import ContextVar
import uuid

from src.utils.logger import get_logger
import numpy as np

# Import global monitor for function call tracking
try:
    from .monitoring_utils import global_monitor
    GLOBAL_MONITOR_AVAILABLE = True
except ImportError:
    global_monitor = None
    GLOBAL_MONITOR_AVAILABLE = False

import logging

# Context variable for tracking function call depth
call_depth_var: ContextVar[int] = ContextVar("call_depth", default=0)
call_id_var: ContextVar[str] = ContextVar("call_id", default="")

def get_call_depth() -> int:
    """Get current function call depth."""
    return call_depth_var.get()

def increment_call_depth() -> int:
    """Increment and return new call depth."""
    current_depth = get_call_depth()
    new_depth = current_depth + 1
    call_depth_var.set(new_depth)
    return new_depth

def decrement_call_depth() -> int:
    """Decrement and return new call depth."""
    current_depth = get_call_depth()
    new_depth = max(0, current_depth - 1)
    call_depth_var.set(new_depth)
    return new_depth

def get_call_id() -> str:
    """Get or generate call ID."""
    call_id = call_id_var.get()
    if not call_id:
        call_id = str(uuid.uuid4())[:8]
        call_id_var.set(call_id)
    return call_id

def log_function_call(
    func_name: str,
    module_name: str,
    call_type: str,
    message: str,
    level: str = "INFO",
    **kwargs
) -> None:
    """Log function call with consistent formatting."""
    depth = get_call_depth()
    call_id = get_call_id()
    indent = "  " * depth

    # Create descriptive log message
    log_message = f"{indent}🔵 {call_type} [{call_id}] {module_name}.{func_name}: {message}"

    # Add additional context if provided
    if kwargs:
        context_parts = []
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)) and len(value) > 3:
                context_parts.append(f"{key}=[{len(value)} items]")
            elif isinstance(value, dict) and len(value) > 3:
                context_parts.append(f"{key}={{keys: {list(value.keys())[:3]}}}")
            else:
                context_parts.append(f"{key}={value}")

        if context_parts:
            log_message += f" | {', '.join(context_parts)}"

    # Log with appropriate level
    logger = get_logger('FunctionLogger')
    if level.upper() == "DEBUG":
        logger.debug(log_message)
    elif level.upper() == "WARNING":
        logger.warning(log_message)
    elif level.upper() == "ERROR":
        logger.error(log_message)
    else:
        logger.info(log_message)

def comprehensive_logging(
    log_internal_calls: bool = True,
    log_execution_time: bool = True,
    log_parameters: bool = True,
    log_return_value: bool = True,
    log_errors: bool = True,
    min_execution_time: float = 0.001
) -> Callable:
    """
    Comprehensive logging decorator for all functions.

    Args:
        log_internal_calls: Whether to log internal function calls
        log_execution_time: Whether to log execution time
        log_parameters: Whether to log function parameters
        log_return_value: Whether to log return values
        log_errors: Whether to log errors
        min_execution_time: Minimum execution time to log (seconds)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_with_comprehensive_logging(
                func, args, kwargs, is_async=True,
                log_internal_calls=log_internal_calls,
                log_execution_time=log_execution_time,
                log_parameters=log_parameters,
                log_return_value=log_return_value,
                log_errors=log_errors,
                min_execution_time=min_execution_time
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_with_comprehensive_logging_sync(
                func, args, kwargs, is_async=False,
                log_internal_calls=log_internal_calls,
                log_execution_time=log_execution_time,
                log_parameters=log_parameters,
                log_return_value=log_return_value,
                log_errors=log_errors,
                min_execution_time=min_execution_time
            )

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

async def _execute_with_comprehensive_logging(
    func: Callable,
    args: tuple,
    kwargs: dict,
    is_async: bool = False,
    log_internal_calls: bool = True,
    log_execution_time: bool = True,
    log_parameters: bool = True,
    log_return_value: bool = True,
    log_errors: bool = True,
    min_execution_time: float = 0.001
) -> Any:
    """Execute function with comprehensive logging."""
    func_name = func.__name__
    module_name = func.__module__
    depth = increment_call_depth()
    start_time = time.time()

    # Log function entry
    entry_message = "Function called"
    if log_parameters:
        param_info = _format_parameters(args, kwargs)
        entry_message += f" with {param_info}"

    log_function_call(
        func_name, module_name, "ENTRY", entry_message,
        args_count=len(args), kwargs_count=len(kwargs)
    )

    try:
        # Execute function
        if is_async:
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log function exit
        exit_message = "Function completed successfully"
        if log_execution_time and execution_time >= min_execution_time:
            exit_message += f" in {execution_time:.4f}s"

        if log_return_value:
            return_info = _format_return_value(result)
            exit_message += f" returning {return_info}"

        log_function_call(
            func_name, module_name, "EXIT", exit_message,
            execution_time=execution_time if log_execution_time else None
        )

        # Track function call in global monitor if available
        if GLOBAL_MONITOR_AVAILABLE and global_monitor:
            global_monitor.track_function_call(
                func_name, execution_time, True,
                len(args), len(kwargs)
            )

        return result

    except Exception as e:
        execution_time = time.time() - start_time

        if log_errors:
            error_message = f"Function failed with {type(e).__name__}: {str(e)}"
            if log_execution_time:
                error_message += f" after {execution_time:.4f}s"

            log_function_call(
                func_name, module_name, "ERROR", error_message,
                level="ERROR",
                error_type=type(e).__name__,
                execution_time=execution_time if log_execution_time else None
            )

            # Log full traceback for debugging
            get_logger('FunctionLogger').debug(
                f"Full traceback for {module_name}.{func_name}:\n{traceback.format_exc()}"
            )

        # Track failed function call in global monitor if available
        if GLOBAL_MONITOR_AVAILABLE and global_monitor:
            global_monitor.track_function_call(
                func_name, execution_time, False,
                len(args), len(kwargs), str(e)
            )

        raise

    finally:
        decrement_call_depth()

def _execute_with_comprehensive_logging_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    is_async: bool = False,
    log_internal_calls: bool = True,
    log_execution_time: bool = True,
    log_parameters: bool = True,
    log_return_value: bool = True,
    log_errors: bool = True,
    min_execution_time: float = 0.001
) -> Any:
    """Execute function with comprehensive logging (sync version)."""
    func_name = func.__name__
    module_name = func.__module__
    depth = increment_call_depth()
    start_time = time.time()

    # Log function entry
    entry_message = "Function called"
    if log_parameters:
        param_info = _format_parameters(args, kwargs)
        entry_message += f" with {param_info}"

    log_function_call(
        func_name, module_name, "ENTRY", entry_message,
        args_count=len(args), kwargs_count=len(kwargs)
    )

    try:
        # Execute function (sync only)
        result = func(*args, **kwargs)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log function exit
        exit_message = "Function completed successfully"
        if log_execution_time and execution_time >= min_execution_time:
            exit_message += f" in {execution_time:.4f}s"

        if log_return_value:
            return_info = _format_return_value(result)
            exit_message += f" returning {return_info}"

        log_function_call(
            func_name, module_name, "EXIT", exit_message,
            execution_time=execution_time if log_execution_time else None
        )

        # Track function call in global monitor if available
        if GLOBAL_MONITOR_AVAILABLE and global_monitor:
            global_monitor.track_function_call(
                func_name, execution_time, True,
                len(args), len(kwargs)
            )

        return result

    except Exception as e:
        execution_time = time.time() - start_time

        if log_errors:
            error_message = f"Function failed with {type(e).__name__}: {str(e)}"
            if log_execution_time:
                error_message += f" after {execution_time:.4f}s"

            log_function_call(
                func_name, module_name, "ERROR", error_message,
                level="ERROR",
                error_type=type(e).__name__,
                execution_time=execution_time if log_execution_time else None
            )

            # Log full traceback for debugging
            get_logger('FunctionLogger').debug(
                f"Full traceback for {module_name}.{func_name}:\n{traceback.format_exc()}"
            )

        # Track failed function call in global monitor if available
        if GLOBAL_MONITOR_AVAILABLE and global_monitor:
            global_monitor.track_function_call(
                func_name, execution_time, False,
                len(args), len(kwargs), str(e)
            )

        raise

    finally:
        decrement_call_depth()

def _format_parameters(args: tuple, kwargs: dict) -> str:
    """Format function parameters for logging."""
    parts = []

    if args:
        # Format positional arguments
        arg_strs = []
        for i, arg in enumerate(args):
            if hasattr(arg, '__len__') and not isinstance(arg, str):
                if len(arg) > 3:
                    arg_strs.append(f"arg{i}=[{len(arg)} items]")
                else:
                    arg_strs.append(f"arg{i}={arg}")
            else:
                arg_strs.append(f"arg{i}={arg}")
        parts.append(f"args=[{', '.join(arg_strs)}]")

    if kwargs:
        # Format keyword arguments
        kwarg_strs = []
        for key, value in kwargs.items():
            if hasattr(value, '__len__') and not isinstance(value, str):
                if len(value) > 3:
                    kwarg_strs.append(f"{key}=[{len(value)} items]")
                else:
                    kwarg_strs.append(f"{key}={value}")
            else:
                kwarg_strs.append(f"{key}={value}")
        parts.append(f"kwargs=[{', '.join(kwarg_strs)}]")

    return ", ".join(parts)

def _format_return_value(result: Any) -> str:
    """Format return value for logging."""
    if result is None:
        return "None"
    elif isinstance(result, (list, tuple)):
        return f"{type(result).__name__}[{len(result)}]"
    elif isinstance(result, dict):
        return f"dict[{len(result)} keys]"
    elif hasattr(result, '__len__') and not isinstance(result, str):
        return f"{type(result).__name__}[{len(result)}]"
    else:
        return f"{type(result).__name__}({str(result)[:50]}{'...' if len(str(result)) > 50 else ''})"

def log_internal_call(caller_func: str, called_func: str, message: str = "", **kwargs) -> None:
    """Log internal function calls."""
    depth = get_call_depth()
    call_id = get_call_id()
    indent = "  " * depth

    log_message = f"{indent}🔄 INTERNAL CALL [{call_id}] {caller_func} -> {called_func}"
    if message:
        log_message += f": {message}"

    if kwargs:
        context_parts = []
        for key, value in kwargs.items():
            context_parts.append(f"{key}={value}")
        if context_parts:
            log_message += f" | {', '.join(context_parts)}"

    get_logger('FunctionLogger').info(log_message)

def log_step_progress(step_name: str, progress_message: str, **kwargs) -> None:
    """Log step progress with consistent formatting."""
    call_id = get_call_id()
    log_message = f"📊 STEP PROGRESS [{call_id}] {step_name}: {progress_message}"

    if kwargs:
        context_parts = []
        for key, value in kwargs.items():
            context_parts.append(f"{key}={value}")
        if context_parts:
            log_message += f" | {', '.join(context_parts)}"

    get_logger('StepLogger').info(log_message)

def log_data_operation(operation: str, data_info: str, **kwargs) -> None:
    """Log data operations with consistent formatting."""
    call_id = get_call_id()
    log_message = f"📈 DATA OP [{call_id}] {operation}: {data_info}"

    if kwargs:
        context_parts = []
        for key, value in kwargs.items():
            context_parts.append(f"{key}={value}")
        if context_parts:
            log_message += f" | {', '.join(context_parts)}"

    get_logger('DataLogger').info(log_message)

# Convenience decorators for different logging levels
def log_all_calls(func: Callable) -> Callable:
    """Decorator to log all function calls with full detail."""
    return comprehensive_logging(
        log_internal_calls=True,
        log_execution_time=True,
        log_parameters=True,
        log_return_value=True,
        log_errors=True,
        min_execution_time=0.001
    )(func)

def log_important_calls(func: Callable) -> Callable:
    """Decorator to log important function calls with moderate detail."""
    return comprehensive_logging(
        log_internal_calls=False,
        log_execution_time=True,
        log_parameters=True,
        log_return_value=False,
        log_errors=True,
        min_execution_time=0.01
    )(func)

def log_step_functions(func: Callable) -> Callable:
    """Decorator specifically for step functions with enhanced logging."""
    return comprehensive_logging(
        log_internal_calls=True,
        log_execution_time=True,
        log_parameters=True,
        log_return_value=True,
        log_errors=True,
        min_execution_time=0.001
    )(func)
