from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Function Call Monitoring System.

This module provides comprehensive function call monitoring and validation.
"""
import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
import collections
import numpy as np
import time

class FunctionCallStatus(Enum):
    """Status of function call execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

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
    @log_important_calls

    def __init__(self, logger: Any = None):
        self.logger = logger or logging.getLogger(__name__)
        self.active_calls: Dict[str, FunctionCallRecord] = {}
        self.call_history: List[FunctionCallRecord] = []
        self.function_call_counter = 0
        self.validation_rules: Dict[str, Callable] = {}
        self.performance_thresholds: Dict[str, float] = {}

    def generate_call_id(self, function_name: str) -> str:
        """Generate unique call ID for function call tracking."""
        self.function_call_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{function_name}_{timestamp}_{self.function_call_counter}"

    def start_function_call(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Start monitoring a function call."""
        call_id = self.generate_call_id(function_name)

        # Capture input arguments (sanitized for logging)
        input_args = {}
        input_kwargs = {}

        try:
            input_args = {f"arg_{i}": str(arg)[:100] for i, arg in enumerate(args)}
            input_kwargs = {k: str(v)[:100] for k, v in kwargs.items()}
        except Exception as e:
            self.logger.warning(f"⚠️ Could not capture function arguments: {e}")

        call_record = FunctionCallRecord(
            function_name=function_name,
            call_id=call_id,
            start_time=datetime.now(),
            status=FunctionCallStatus.IN_PROGRESS,
            input_args=input_args,
            input_kwargs=input_kwargs
        )

        self.active_calls[call_id] = call_record
        self.logger.info(f"🚀 Starting function call: {function_name} (ID: {call_id})")

        return call_id

    def end_function_call(self, call_id: str, return_value: Any = None, exception: Exception = None) -> FunctionCallRecord:
        """End monitoring a function call and record results."""
        if call_id not in self.active_calls:
            self.logger.error(f"❌ Call ID {call_id} not found in active calls")
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
            self.logger.error(f"❌ Function call failed: {call_record.function_name} (ID: {call_id}) - {exception}")
        else:
            call_record.status = FunctionCallStatus.COMPLETED
            self.logger.info(f"✅ Function call completed: {call_record.function_name} (ID: {call_id}) in {call_record.execution_time:.3f}s")

        # Move from active to history
        del self.active_calls[call_id]
        self.call_history.append(call_record)

        return call_record

    def record_function_to_function_call(self, parent_call_id: str, child_function_name: str) -> None:
        """Record a function-to-function call relationship."""
        if parent_call_id in self.active_calls:
            self.active_calls[parent_call_id].called_functions.append(child_function_name)
            self.logger.debug(f"🔗 Function call chain: {self.active_calls[parent_call_id].function_name} -> {child_function_name}")

    def validate_function_call(self, call_record: FunctionCallRecord) -> Dict[str, bool]:
        """Validate function call against defined rules."""
        validation_results = {}

        # Performance validation
        if call_record.function_name in self.performance_thresholds:
            threshold = self.performance_thresholds[call_record.function_name]
            validation_results['performance'] = call_record.execution_time <= threshold
            if not validation_results['performance']:
                self.logger.warning(f"⚠️ Performance threshold exceeded for {call_record.function_name}: {call_record.execution_time:.3f}s > {threshold}s")

        # Custom validation rules
        if call_record.function_name in self.validation_rules:
            try:
                validation_results['custom'] = self.validation_rules[call_record.function_name](call_record)
            except Exception as e:
                self.logger.error(f"❌ Custom validation failed for {call_record.function_name}: {e}")
                validation_results['custom'] = False

        # Basic validation rules
        validation_results['has_return_value'] = call_record.return_value is not None
        validation_results['no_exception'] = call_record.exception is None
        validation_results['reasonable_execution_time'] = 0 < call_record.execution_time < 3600  # Less than 1 hour

        call_record.validation_results = validation_results
        return validation_results

    def generate_comprehensive_report(self) -> FunctionCallReport:
        """Generate comprehensive function call report."""
        if not self.call_history:
            return FunctionCallReport()

        total_calls = len(self.call_history)
        successful_calls = len([c for c in self.call_history if c.status == FunctionCallStatus.COMPLETED])
        failed_calls = len([c for c in self.call_history if c.status == FunctionCallStatus.FAILED])
        total_execution_time = sum(c.execution_time for c in self.call_history)
        average_execution_time = total_execution_time / total_calls if total_calls > 0 else 0.0

        # Performance summary
        performance_summary = {
            'total_execution_time': total_execution_time,
            'average_execution_time': average_execution_time,
            'max_execution_time': max(c.execution_time for c in self.call_history),
            'min_execution_time': min(c.execution_time for c in self.call_history)
        }

        # Error summary
        error_summary = {}
        for call in self.call_history:
            if call.exception:
                error_type = type(call.exception).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

        # Validation summary
        validation_summary = {}
        for call in self.call_history:
            for validation_name, result in call.validation_results.items():
                if validation_name not in validation_summary:
                    validation_summary[validation_name] = {'passed': 0, 'failed': 0}
                if result:
                    validation_summary[validation_name]['passed'] += 1
                else:
                    validation_summary[validation_name]['failed'] += 1

        return FunctionCallReport(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_execution_time=total_execution_time,
            average_execution_time=average_execution_time,
            function_call_details=self.call_history.copy(),
            performance_summary=performance_summary,
            error_summary=error_summary,
            validation_summary=validation_summary
        )

    def log_detailed_report(self, report: FunctionCallReport) -> None:
        """Log detailed function call report."""
        self.logger.info("📊 COMPREHENSIVE FUNCTION CALL REPORT")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Function Calls: {report.total_calls}")
        self.logger.info(f"Successful Calls: {report.successful_calls}")
        self.logger.info(f"Failed Calls: {report.failed_calls}")
        self.logger.info(f"Success Rate: {report.successful_calls/report.total_calls*100:.1f}%" if report.total_calls > 0 else "N/A")
        self.logger.info(f"Total Execution Time: {report.total_execution_time:.3f}s")
        self.logger.info(f"Average Execution Time: {report.average_execution_time:.3f}s")

        if report.performance_summary:
            self.logger.info("\n📈 PERFORMANCE SUMMARY:")
            for metric, value in report.performance_summary.items():
                self.logger.info(f"  {metric}: {value:.3f}s")

        if report.error_summary:
            self.logger.info("\n❌ ERROR SUMMARY:")
            for error_type, count in report.error_summary.items():
                self.logger.info(f"  {error_type}: {count} occurrences")

        if report.validation_summary:
            self.logger.info("\n✅ VALIDATION SUMMARY:")
            for validation_name, results in report.validation_summary.items():
                total = results['passed'] + results['failed']
                pass_rate = results['passed'] / total * 100 if total > 0 else 0
                self.logger.info(f"  {validation_name}: {results['passed']}/{total} passed ({pass_rate:.1f}%)")

        self.logger.info("\n🔍 DETAILED FUNCTION CALLS:")
        for call in report.function_call_details:
            status_emoji = "✅" if call.status == FunctionCallStatus.COMPLETED else "❌"
            self.logger.info(f"  {status_emoji} {call.function_name} (ID: {call.call_id})")
            self.logger.info(f"    Status: {call.status.value}")
            self.logger.info(f"    Execution Time: {call.execution_time:.3f}s")
            if call.called_functions:
                self.logger.info(f"    Called Functions: {', '.join(call.called_functions)}")
            if call.validation_results:
                validation_status = "✅" if all(call.validation_results.values()) else "⚠️"
                self.logger.info(f"    Validation: {validation_status} {call.validation_results}")

def comprehensive_function_monitor(monitor: FunctionCallMonitor):
    """Decorator for comprehensive function call monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            call_id = monitor.start_function_call(func.__name__, args, kwargs)
            try:
                result = await func(*args, **kwargs)
                call_record = monitor.end_function_call(call_id, result)
                if call_record:
                    monitor.validate_function_call(call_record)
                return result
            except Exception as e:
                call_record = monitor.end_function_call(call_id, exception=e)
                if call_record:
                    monitor.validate_function_call(call_record)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            call_id = monitor.start_function_call(func.__name__, args, kwargs)
            try:
                result = func(*args, **kwargs)
                call_record = monitor.end_function_call(call_id, result)
                if call_record:
                    monitor.validate_function_call(call_record)
                return result
            except Exception as e:
                call_record = monitor.end_function_call(call_id, exception=e)
                if call_record:
                    monitor.validate_function_call(call_record)
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def function_to_function_tracker(monitor: FunctionCallMonitor, parent_call_id: str = None):
    """Decorator for tracking function-to-function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if parent_call_id:
                monitor.record_function_to_function_call(parent_call_id, func.__name__)
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if parent_call_id:
                monitor.record_function_to_function_call(parent_call_id, func.__name__)
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
