"""
Enhanced Error Handling System with Detailed Function-Level Tracking
"""
import asyncio
import functools
import inspect
import logging
import traceback
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import collections

class ErrorSeverity(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class ErrorCategory(Enum):
    VALIDATION_ERROR = 'validation_error'
    RUNTIME_ERROR = 'runtime_error'
    SYSTEM_ERROR = 'system_error'
    NETWORK_ERROR = 'network_error'
    DATA_ERROR = 'data_error'
    UNKNOWN_ERROR = 'unknown_error'

@dataclass
class ErrorContext:
    function_name: str
    module_name: str
    line_number: int
    timestamp: datetime
    call_stack: List[str]
    local_variables: Dict[str, Any] = field(default_factory = dict)

@dataclass
class ErrorRecord:
    error_id: str
    error_type: str
    error_message: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False
    stack_trace: str = ''

class EnhancedErrorHandler:

    def __init__(self, logger: Optional[logging.Logger]=None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.error_records: List[ErrorRecord] = []
        self._lock = threading.Lock()
        self._error_counter = 0

    def _generate_error_id(self) -> str:
        with self._lock:
            self._error_counter += 1
            return f'error_{self._error_counter}_{int(time.time() * 1000)}'

    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        if isinstance(exception, (ValueError, TypeError, AttributeError)):
            return ErrorCategory.VALIDATION_ERROR
        elif isinstance(exception, (RuntimeError, NotImplementedError)):
            return ErrorCategory.RUNTIME_ERROR
        elif isinstance(exception, (OSError, IOError, FileNotFoundError)):
            return ErrorCategory.SYSTEM_ERROR
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK_ERROR
        elif isinstance(exception, (KeyError, IndexError)):
            return ErrorCategory.DATA_ERROR
        else:
            return ErrorCategory.UNKNOWN_ERROR

    def _assess_error_severity(self, exception: Exception, context: ErrorContext) -> ErrorSeverity:
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif context.function_name in ['execute', 'run_step', 'initialize']:
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _create_error_context(self, func: Callable, exception: Exception) -> ErrorContext:
        try:
            frame = sys.exc_info()[2].tb_frame
            while frame and frame.f_code.co_filename != func.__code__.co_filename:
                frame = frame.f_back
            if frame:
                line_number = frame.f_lineno
                local_vars = {k: str(v)[:100] for k, v in frame.f_locals.items()}
            else:
                line_number = 0
                local_vars = {}
            call_stack = []
            tb = sys.exc_info()[2]
            while tb:
                call_stack.append(f'{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno} in {tb.tb_frame.f_code.co_name}')
                tb = tb.tb_next
            return ErrorContext(function_name = func.__name__, module_name = func.__module__ if hasattr(func, '__module__') else 'unknown', line_number = line_number, timestamp = datetime.now(), call_stack = call_stack, local_variables = local_vars)
        except Exception:
            return ErrorContext(function_name = func.__name__, module_name = func.__module__ if hasattr(func, '__module__') else 'unknown', line_number = 0, timestamp = datetime.now(), call_stack=[])

    def handle_error(self, func: Callable, exception: Exception, args: tuple, kwargs: dict) -> Tuple[bool, Any]:
        error_id = self._generate_error_id()
        context = self._create_error_context(func, exception)
        category = self._categorize_error(exception)
        severity = self._assess_error_severity(exception, context)
        error_record = ErrorRecord(error_id = error_id, error_type = type(exception).__name__, error_message = str(exception), error_category = category, error_severity = severity, context = context, stack_trace = traceback.format_exc())
        self._log_error_details(error_record)
        with self._lock:
            self.error_records.append(error_record)
        return (False, None)

    def _log_error_details(self, error_record: ErrorRecord) -> None:
        self.logger.error(f'❌ ERROR OCCURRED: {error_record.error_id}')
        self.logger.error(f'   Function: {error_record.context.function_name}')
        self.logger.error(f'   Module: {error_record.context.module_name}')
        self.logger.error(f'   Line: {error_record.context.line_number}')
        self.logger.error(f'   Type: {error_record.error_type}')
        self.logger.error(f'   Category: {error_record.error_category.value}')
        self.logger.error(f'   Severity: {error_record.error_severity.value}')
        self.logger.error(f'   Message: {error_record.error_message}')
        if error_record.context.call_stack:
            self.logger.error('   Call Stack:')
            for frame in error_record.context.call_stack[:5]:
                self.logger.error(f'     {frame}')

    def get_error_summary(self) -> Dict[str, Any]:
        total_errors = len(self.error_records)
        return {'total_errors': total_errors, 'recent_errors': [{'error_id': record.error_id, 'function_name': record.context.function_name, 'error_type': record.error_type, 'error_message': record.error_message, 'category': record.error_category.value, 'severity': record.error_severity.value, 'timestamp': record.context.timestamp.isoformat()} for record in self.error_records[-10:]]}
_global_error_handler = EnhancedErrorHandler()

def handle_errors_with_tracking(fallback: bool = True) -> None:

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                recovery_success, recovery_result = _global_error_handler.handle_error(func, e, args, kwargs)
                if recovery_success and recovery_result is not None:
                    return recovery_result
                elif fallback:
                    return None
                else:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery_success, recovery_result = _global_error_handler.handle_error(func, e, args, kwargs)
                if recovery_success and recovery_result is not None:
                    return recovery_result
                elif fallback:
                    return None
                else:
                    raise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def get_error_handler() -> EnhancedErrorHandler:
    return _global_error_handler

def log_error_summary(logger: logging.Logger) -> None:
    summary = _global_error_handler.get_error_summary()
    logger.info('📊 ERROR SUMMARY:')
    logger.info(f"   Total errors: {summary['total_errors']}")

def handle_errors_basic(func: Callable) -> Callable:
    return handle_errors_with_tracking(fallback = True)(func)

def handle_errors_strict(func: Callable) -> Callable:
    return handle_errors_with_tracking(fallback = False)(func)