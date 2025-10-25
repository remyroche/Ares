"""
Shared logging utilities for HDBSCAN regime detection.

This module provides common logging functionality that eliminates redundancy
between NAS and TAS components, including execution tracking, performance monitoring,
and standardized logging patterns.
"""

import time
import psutil
import logging
import functools
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager
from datetime import datetime
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

class LoggingContext:
    """Context manager for standardized logging across HDBSCAN components."""

    def __init__(
        self,
        component_name: str,
        operation_name: str,
        verbose: bool = True,
        track_performance: bool = True,
        track_memory: bool = True
    ):
        """Initialize logging context.

        Args:
            component_name: Name of the component (e.g., 'NAS', 'TAS', 'Hybrid')
            operation_name: Name of the operation being performed
            verbose: Whether to enable verbose logging
            track_performance: Whether to track execution time
            track_memory: Whether to track memory usage
        """
        self.component_name = component_name
        self.operation_name = operation_name
        self.verbose = verbose
        self.track_performance = track_performance
        self.track_memory = track_memory

        self.start_time = None
        self.start_memory = None
        self.logger = None

    def __enter__(self):
        """Enter the logging context."""
        self.start_time = time.time()
        if self.track_memory:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if self.verbose:
            tprint(f"🚀 [{self.component_name}] Starting {self.operation_name}", color="cyan", bold=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the logging context."""
        if self.track_performance or self.track_memory:
            execution_time = time.time() - self.start_time

            if self.track_memory and self.start_memory is not None:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = current_memory - self.start_memory
            else:
                memory_used = None

            if exc_type is None:
                # Success
                if self.verbose:
                    if memory_used is not None:
                        tprint_success(f"✅ [{self.component_name}] {self.operation_name} completed in {execution_time:.3f}s, memory: {memory_used:+.1f}MB")
                    else:
                        tprint_success(f"✅ [{self.component_name}] {self.operation_name} completed in {execution_time:.3f}s")
            else:
                # Error
                if self.verbose:
                    if memory_used is not None:
                        tprint_error(f"❌ [{self.component_name}] {self.operation_name} failed after {execution_time:.3f}s, memory: {memory_used:+.1f}MB")
                    else:
                        tprint_error(f"❌ [{self.component_name}] {self.operation_name} failed after {execution_time:.3f}s")

def log_execution(
    component_name: str,
    operation_name: str,
    verbose: bool = True,
    track_performance: bool = True,
    track_memory: bool = True
):
    """
    Decorator for logging function execution with performance and memory tracking.

    Args:
        component_name: Name of the component
        operation_name: Name of the operation
        verbose: Whether to enable verbose logging
        track_performance: Whether to track execution time
        track_memory: Whether to track memory usage
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with LoggingContext(
                component_name=component_name,
                operation_name=operation_name,
                verbose=verbose,
                track_performance=track_performance,
                track_memory=track_memory
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def log_performance(
    component_name: str,
    operation_name: str,
    verbose: bool = True
):
    """
    Decorator for logging function performance metrics.

    Args:
        component_name: Name of the component
        operation_name: Name of the operation
        verbose: Whether to enable verbose logging
    """
    return log_execution(
        component_name=component_name,
        operation_name=operation_name,
        verbose=verbose,
        track_performance=True,
        track_memory=True
    )

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a standardized logger for HDBSCAN components.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger

def log_info(message: str, component_name: Optional[str] = None):
    """Log an info message with optional component context."""
    if component_name:
        tprint_info(f"[{component_name}] {message}")
    else:
        tprint_info(message)

def log_warning(message: str, component_name: Optional[str] = None):
    """Log a warning message with optional component context."""
    if component_name:
        tprint_warning(f"[{component_name}] {message}")
    else:
        tprint_warning(message)

def log_error(message: str, component_name: Optional[str] = None):
    """Log an error message with optional component context."""
    if component_name:
        tprint_error(f"[{component_name}] {message}")
    else:
        tprint_error(message)

def log_success(message: str, component_name: Optional[str] = None):
    """Log a success message with optional component context."""
    if component_name:
        tprint_success(f"[{component_name}] {message}")
    else:
        tprint_success(message)

def log_debug(message: str, component_name: Optional[str] = None):
    """Log a debug message with optional component context."""
    if component_name:
        tprint_debug(f"[{component_name}] {message}")
    else:
        tprint_debug(message)

@contextmanager
def log_data_info(data_name: str, data: Any, context: str = ""):
    """
    Context manager for logging data information.

    Args:
        data_name: Name of the data being processed
        data: Data object to log information about
        context: Additional context information
    """
    try:
        # Log data information
        if hasattr(data, 'shape'):
            log_info(f"Processing {data_name}: shape={data.shape}, context={context}")
        elif hasattr(data, '__len__'):
            log_info(f"Processing {data_name}: length={len(data)}, context={context}")
        else:
            log_info(f"Processing {data_name}: type={type(data)}, context={context}")

        yield

        log_success(f"Successfully processed {data_name}")

    except Exception as e:
        log_error(f"Failed to process {data_name}: {e}")
        raise

@contextmanager
def log_validation_result(validation_name: str, result: bool, details: str = ""):
    """
    Context manager for logging validation results.

    Args:
        validation_name: Name of the validation being performed
        result: Validation result (True/False)
        details: Additional details about the validation
    """
    try:
        if result:
            log_success(f"{validation_name} validation passed: {details}")
        else:
            log_error(f"{validation_name} validation failed: {details}")

        yield result

    except Exception as e:
        log_error(f"{validation_name} validation error: {e}")
        raise

@contextmanager
def log_step_progress(step_name: str, total_steps: int, current_step: int):
    """
    Context manager for logging step progress.

    Args:
        step_name: Name of the step
        total_steps: Total number of steps
        current_step: Current step number (1-based)
    """
    try:
        progress_percent = (current_step / total_steps) * 100
        log_info(f"Step {current_step}/{total_steps} ({progress_percent:.1f}%): {step_name}")

        yield

        log_success(f"Step {current_step}/{total_steps} completed: {step_name}")

    except Exception as e:
        log_error(f"Step {current_step}/{total_steps} failed: {step_name} - {e}")
        raise

class PerformanceTracker:
    """Performance tracking utility for HDBSCAN components."""

    def __init__(self, component_name: str):
        """Initialize performance tracker.

        Args:
            component_name: Name of the component being tracked
        """
        self.component_name = component_name
        self.metrics = {}
        self.start_time = None
        self.start_memory = None

    def start_tracking(self):
        """Start performance tracking."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def stop_tracking(self, operation_name: str = "operation"):
        """Stop performance tracking and record metrics.

        Args:
            operation_name: Name of the operation being tracked
        """
        if self.start_time is None:
            log_warning(f"Performance tracking not started for {operation_name}")
            return

        execution_time = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.start_memory

        self.metrics[operation_name] = {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'timestamp': datetime.now().isoformat()
        }

        log_performance(f"Performance tracked for {operation_name}: {execution_time:.3f}s, {memory_used:+.1f}MB")

        # Reset for next tracking
        self.start_time = None
        self.start_memory = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics."""
        return self.metrics.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {}

        total_time = sum(metric['execution_time'] for metric in self.metrics.values())
        total_memory = sum(metric['memory_used'] for metric in self.metrics.values())

        return {
            'total_operations': len(self.metrics),
            'total_execution_time': total_time,
            'total_memory_used': total_memory,
            'average_execution_time': total_time / len(self.metrics),
            'average_memory_used': total_memory / len(self.metrics)
        }

class LoggingManager:
    """Centralized logging manager for HDBSCAN components."""

    def __init__(self, component_name: str, verbose: bool = True):
        """Initialize logging manager.

        Args:
            component_name: Name of the component
            verbose: Whether to enable verbose logging
        """
        self.component_name = component_name
        self.verbose = verbose
        self.logger = get_logger(component_name)
        self.performance_tracker = PerformanceTracker(component_name)

    def log_operation_start(self, operation_name: str):
        """Log the start of an operation."""
        if self.verbose:
            log_info(f"Starting {operation_name}", self.component_name)
        self.performance_tracker.start_tracking()

    def log_operation_end(self, operation_name: str, success: bool = True):
        """Log the end of an operation."""
        if self.verbose:
            if success:
                log_success(f"Completed {operation_name}", self.component_name)
            else:
                log_error(f"Failed {operation_name}", self.component_name)
        self.performance_tracker.stop_tracking(operation_name)

    def log_data_info(self, data_name: str, data: Any, context: str = ""):
        """Log data information."""
        if self.verbose:
            with log_data_info(data_name, data, context):
                pass

    def log_validation_result(self, validation_name: str, result: bool, details: str = ""):
        """Log validation result."""
        if self.verbose:
            with log_validation_result(validation_name, result, details):
                pass

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_tracker.get_summary()

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return self.performance_tracker.get_metrics()
