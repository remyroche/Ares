from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

"""
Monitoring utilities for data collection steps.

This module provides comprehensive function monitoring capabilities
for step validation and performance tracking.
"""

import functools
import inspect
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    import psutil
    import collections
except ImportError:
    psutil = None

class FunctionCallStatus(Enum):
    """Status of function call monitoring."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    TIMEOUT = 'timeout'

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
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class FunctionInteractionReport:
    """Report summarizing function interactions."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)
    call_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    function_call_details: List[FunctionCallReport] = field(default_factory=list)

class FunctionCallMonitor:
    """Main function call monitoring system."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.active_calls: Dict[str, FunctionCallReport] = {}
        self.call_history: List[FunctionCallReport] = []
        self.call_stack: List[str] = []
        self._call_counter = 0
        self._start_time = time.time()

    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        self._call_counter += 1
        return f'call_{self._call_counter}_{int(time.time() * 1000)}'

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if psutil is None:
                return 0.0
            import os
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

    def start_call(self, func: Callable, args: tuple, kwargs: dict) -> FunctionCallReport:
        """Start monitoring a function call."""
        call_id = self._generate_call_id()
        module_name = getattr(func, '__module__', 'unknown')

        report = FunctionCallReport(
            function_name=func.__name__,
            module_name=module_name,
            call_id=call_id,
            timestamp=datetime.now(),
            metrics=FunctionCallMetrics(
                start_time=time.time(),
                memory_before=self._get_memory_usage(),
                call_depth=len(self.call_stack),
                status=FunctionCallStatus.IN_PROGRESS
            )
        )

        self.active_calls[call_id] = report
        self.call_stack.append(call_id)

        return report

    def end_call(self, report: FunctionCallReport, return_value: Any = None, error: Exception = None) -> None:
        """End monitoring a function call."""
        report.metrics.end_time = time.time()
        report.metrics.duration = report.metrics.end_time - report.metrics.start_time
        report.metrics.memory_after = self._get_memory_usage()
        report.metrics.cpu_percent = self._get_cpu_percent()
        report.return_value = return_value

        if error:
            report.metrics.status = FunctionCallStatus.FAILED
            report.metrics.error_message = str(error)
            report.metrics.stack_trace = traceback.format_exc()
        else:
            report.metrics.status = FunctionCallStatus.COMPLETED

        # Clean up
        call_id = report.call_id
        if call_id in self.active_calls:
            del self.active_calls[call_id]
        if call_id in self.call_stack:
            self.call_stack.remove(call_id)

        self.call_history.append(report)

    def get_function_interaction_report(self) -> FunctionInteractionReport:
        """Get a comprehensive report of function interactions."""
        total_calls = len(self.call_history)
        successful_calls = sum(1 for call in self.call_history
                              if call.metrics.status == FunctionCallStatus.COMPLETED)
        failed_calls = total_calls - successful_calls

        total_execution_time = sum(call.metrics.duration or 0 for call in self.call_history)
        average_execution_time = total_execution_time / total_calls if total_calls > 0 else 0

        # Performance metrics
        performance_metrics = {
            'success_rate': (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': average_execution_time,
        }

        # Find fastest and slowest calls
        if self.call_history:
            durations = [(call.metrics.duration, call.function_name) for call in self.call_history
                        if call.metrics.duration is not None]
            if durations:
                fastest_duration, fastest_call = min(durations, key=lambda x: x[0])
                slowest_duration, slowest_call = max(durations, key=lambda x: x[0])

                performance_metrics.update({
                    'fastest_call': fastest_call,
                    'fastest_call_time': fastest_duration,
                    'slowest_call': slowest_call,
                    'slowest_call_time': slowest_duration,
                })

        # Error summary
        error_summary = {}
        for call in self.call_history:
            if call.metrics.error_message:
                error_type = type(call.metrics.error_message).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

        # Call hierarchy (simplified)
        call_hierarchy = {}
        for call in self.call_history:
            if call.metrics.call_depth > 0 and self.call_stack:
                parent_call_id = self.call_stack[-1] if len(self.call_stack) > call.metrics.call_depth else None
                if parent_call_id:
                    call_hierarchy.setdefault(parent_call_id, []).append(call.call_id)

        return FunctionInteractionReport(
            total_calls=total_calls,
            successful_calls=successful_calls,
            failed_calls=failed_calls,
            total_execution_time=total_execution_time,
            average_execution_time=average_execution_time,
            performance_metrics=performance_metrics,
            error_summary=error_summary,
            call_hierarchy=call_hierarchy,
            function_call_details=self.call_history.copy()
        )

# Global function monitor instance
function_monitor = FunctionCallMonitor()

def comprehensive_function_monitoring(
    validate_inputs: bool = True,
    validate_outputs: bool = True,
    track_performance: bool = True,
    timeout_seconds: Optional[int] = None,
    retry_attempts: int = 0
) -> Callable:
    """
    Decorator for comprehensive function monitoring.

    Args:
        validate_inputs: Whether to validate function inputs
        validate_outputs: Whether to validate function outputs
        track_performance: Whether to track performance metrics
        timeout_seconds: Timeout for function execution
        retry_attempts: Number of retry attempts on failure

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__ or __name__)

            for attempt in range(retry_attempts + 1):
                try:
                    # Start monitoring
                    if track_performance:
                        report = function_monitor.start_call(func, args, kwargs)

                    # Execute function
                    if timeout_seconds:
                        import asyncio
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                    else:
                        result = await func(*args, **kwargs)

                    # End monitoring
                    if track_performance:
                        function_monitor.end_call(report, result)

                    return result

                except Exception as e:
                    logger.warning(f"Function {func.__name__} attempt {attempt + 1} failed: {e}")

                    if track_performance and 'report' in locals():
                        function_monitor.end_call(report, error=e)

                    if attempt == retry_attempts:
                        raise

                    # Wait before retry
                    await asyncio.sleep(0.1 * (attempt + 1))

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__ or __name__)

            for attempt in range(retry_attempts + 1):
                try:
                    # Start monitoring
                    if track_performance:
                        report = function_monitor.start_call(func, args, kwargs)

                    # Execute function
                    if timeout_seconds:
                        import signal
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout_seconds)

                        try:
                            result = func(*args, **kwargs)
                        finally:
                            signal.alarm(0)
                    else:
                        result = func(*args, **kwargs)

                    # End monitoring
                    if track_performance:
                        function_monitor.end_call(report, result)

                    return result

                except Exception as e:
                    logger.warning(f"Function {func.__name__} attempt {attempt + 1} failed: {e}")

                    if track_performance and 'report' in locals():
                        function_monitor.end_call(report, error=e)

                    if attempt == retry_attempts:
                        raise

                    # Wait before retry
                    time.sleep(0.1 * (attempt + 1))

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Export the key components
__all__ = [
    'comprehensive_function_monitoring',
    'function_monitor',
    'FunctionCallMonitor',
    'FunctionInteractionReport',
    'FunctionCallStatus',
    'FunctionCallReport',
    'FunctionCallMetrics'
]
