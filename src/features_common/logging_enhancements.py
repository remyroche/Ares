"""
Backward-compatible logging enhancements.

This module provides optional extensive logging that can be enabled
without breaking existing code or changing the core behavior.
"""

import logging
from typing import Any, Callable, Optional, Union
from functools import wraps

# Import tprint utilities
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)

class LoggingEnhancements:
    """
    Backward-compatible logging enhancements that can be optionally enabled.
    """

    def __init__(self, enable_verbose_logging: bool = True):
        """
        Initialize logging enhancements.

        Args:
            enable_verbose_logging: Whether to enable verbose tprint logging
        """
        self.enable_verbose_logging = enable_verbose_logging
        self._original_methods = {}

    def enhance_method(self, method_name: str, component_name: str = "Unknown"):
        """
        Decorator to enhance a method with optional verbose logging.

        Args:
            method_name: Name of the method being enhanced
            component_name: Name of the component for logging
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # Check if verbose logging is enabled
                if hasattr(self, 'enable_verbose_logging') and self.enable_verbose_logging:
                    if TPRINT_AVAILABLE:
                        tprint(f"🔧 [{component_name}] Starting {method_name}", color="cyan")

                try:
                    result = func(self, *args, **kwargs)

                    if hasattr(self, 'enable_verbose_logging') and self.enable_verbose_logging:
                        if TPRINT_AVAILABLE:
                            tprint(f"✅ [{component_name}] {method_name} completed successfully", color="green")

                    return result

                except Exception as e:
                    if hasattr(self, 'enable_verbose_logging') and self.enable_verbose_logging:
                        if TPRINT_AVAILABLE:
                            tprint(f"❌ [{component_name}] {method_name} failed: {e}", color="red")
                    raise

            return wrapper
        return decorator

    def enable_verbose_logging_for_instance(self, instance, enable: bool = True):
        """
        Enable or disable verbose logging for a specific instance.

        Args:
            instance: The instance to enable logging for
            enable: Whether to enable verbose logging
        """
        instance.enable_verbose_logging = enable

    def log_operation_start(self, operation_name: str, component_name: str = "Unknown", **kwargs):
        """Log the start of an operation."""
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [{component_name}] Starting {operation_name}", color="cyan")
            if kwargs:
                tprint(f"   Parameters: {kwargs}", color="blue")

    def log_operation_success(self, operation_name: str, component_name: str = "Unknown", result_info: str = None):
        """Log successful operation completion."""
        if TPRINT_AVAILABLE:
            tprint(f"✅ [{component_name}] {operation_name} completed successfully", color="green")
            if result_info:
                tprint(f"   Result: {result_info}", color="green")

    def log_operation_failure(self, operation_name: str, error: Exception, component_name: str = "Unknown"):
        """Log operation failure."""
        if TPRINT_AVAILABLE:
            tprint(f"❌ [{component_name}] {operation_name} failed: {error}", color="red")

    def log_warning(self, message: str, component_name: str = "Unknown"):
        """Log a warning message."""
        if TPRINT_AVAILABLE:
            tprint(f"⚠️  [{component_name}] {message}", color="yellow")

    def log_info(self, message: str, component_name: str = "Unknown"):
        """Log an info message."""
        if TPRINT_AVAILABLE:
            tprint(f"ℹ️  [{component_name}] {message}", color="blue")

    def log_debug(self, message: str, component_name: str = "Unknown"):
        """Log a debug message."""
        if TPRINT_AVAILABLE:
            tprint(f"🔍 [{component_name}] {message}", color="magenta")

# Global logging enhancements instance
_logging_enhancements = LoggingEnhancements()

def enable_verbose_logging(enable: bool = True):
    """
    Enable or disable verbose logging globally.

    Args:
        enable: Whether to enable verbose logging
    """
    _logging_enhancements.enable_verbose_logging = enable

def get_logging_enhancements() -> LoggingEnhancements:
    """Get the global logging enhancements instance."""
    return _logging_enhancements

def enhance_with_logging(method_name: str, component_name: str = "Unknown"):
    """
    Decorator to enhance a method with optional verbose logging.

    Args:
        method_name: Name of the method being enhanced
        component_name: Name of the component for logging
    """
    return _logging_enhancements.enhance_method(method_name, component_name)

def log_operation(operation_name: str, component_name: str = "Unknown", **kwargs):
    """Log an operation with optional verbose logging."""
    if _logging_enhancements.enable_verbose_logging:
        _logging_enhancements.log_operation_start(operation_name, component_name, **kwargs)

def log_success(operation_name: str, component_name: str = "Unknown", result_info: str = None):
    """Log operation success with optional verbose logging."""
    if _logging_enhancements.enable_verbose_logging:
        _logging_enhancements.log_operation_success(operation_name, component_name, result_info)

def log_failure(operation_name: str, error: Exception, component_name: str = "Unknown"):
    """Log operation failure with optional verbose logging."""
    if _logging_enhancements.enable_verbose_logging:
        _logging_enhancements.log_operation_failure(operation_name, error, component_name)

def log_warning(message: str, component_name: str = "Unknown"):
    """Log a warning with optional verbose logging."""
    if _logging_enhancements.enable_verbose_logging:
        _logging_enhancements.log_warning(message, component_name)

def log_info(message: str, component_name: str = "Unknown"):
    """Log info with optional verbose logging."""
    if _logging_enhancements.enable_verbose_logging:
        _logging_enhancements.log_info(message, component_name)

def log_debug(message: str, component_name: str = "Unknown"):
    """Log debug with optional verbose logging."""
    if _logging_enhancements.enable_verbose_logging:
        _logging_enhancements.log_debug(message, component_name)
