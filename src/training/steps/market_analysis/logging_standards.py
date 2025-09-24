"""
Market Analysis Logging Standards

This module defines consistent logging patterns using tprint for all market analysis components.
Ensures uniform logging format and behavior across all modules.

Usage:
    from .logging_standards import get_logger, log_info, log_warning, log_error, log_success, log_debug

    # Use standardized logging functions
    log_info("Starting market analysis pipeline")
    log_warning("Data quality issues detected")
    log_error("Critical failure occurred")
    log_success("Pipeline completed successfully")
    log_debug("Detailed debugging information")
"""

import logging
import sys
from typing import Any, Optional
from pathlib import Path

# Import tprint with fallback
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_error, tprint_warning,
        tprint_success, tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    # Fallback implementations
    TPRINT_AVAILABLE = False

    def tprint(*args, **kwargs):
        """Fallback tprint function."""
        timestamp = "[{:02d}:{:02d}:{:02d}]".format(
            *map(int, str(__import__('datetime').datetime.now().time()).split(':'))
        )
        print(f"{timestamp} {' '.join(map(str, args))}")

    def tprint_info(*args, **kwargs):
        """Fallback info print."""
        tprint("ℹ️", *args)

    def tprint_error(*args, **kwargs):
        """Fallback error print."""
        tprint("❌", *args)

    def tprint_warning(*args, **kwargs):
        """Fallback warning print."""
        tprint("⚠️", *args)

    def tprint_success(*args, **kwargs):
        """Fallback success print."""
        tprint("✅", *args)

    def tprint_debug(*args, **kwargs):
        """Fallback debug print."""
        tprint("🔍", *args)

    def tprint_performance(*args, **kwargs):
        """Fallback performance print."""
        tprint("⚡", *args)

# Import system logger for integration
try:
    from src.utils.logger import system_logger
    SYSTEM_LOGGER_AVAILABLE = True
except ImportError:
    SYSTEM_LOGGER_AVAILABLE = False
    system_logger = None

def get_logger(component_name: str):
    """
    Get a standardized logger for a component.

    Args:
        component_name: Name of the component (e.g., 'MarketAnalysisSubPipeline')

    Returns:
        Logger instance with consistent configuration
    """
    if SYSTEM_LOGGER_AVAILABLE and system_logger:
        return system_logger.getChild(component_name)
    else:
        # Create a basic logger
        logger = logging.getLogger(component_name)
        logger.setLevel(logging.INFO)

        # Add handler if not already added
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(f'[{component_name}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

# Standardized logging functions using tprint
def log_info(message: str, *args: Any, **kwargs: Any) -> None:
    """Log informational message."""
    if TPRINT_AVAILABLE:
        tprint_info(message, *args, **kwargs)
    else:
        tprint("ℹ️", message, *args, **kwargs)

def log_warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Log warning message."""
    if TPRINT_AVAILABLE:
        tprint_warning(message, *args, **kwargs)
    else:
        tprint("⚠️", message, *args, **kwargs)

def log_error(message: str, *args: Any, **kwargs: Any) -> None:
    """Log error message."""
    if TPRINT_AVAILABLE:
        tprint_error(message, *args, **kwargs)
    else:
        tprint("❌", message, *args, **kwargs)

def log_success(message: str, *args: Any, **kwargs: Any) -> None:
    """Log success message."""
    if TPRINT_AVAILABLE:
        tprint_success(message, *args, **kwargs)
    else:
        tprint("✅", message, *args, **kwargs)

def log_debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Log debug message."""
    if TPRINT_AVAILABLE:
        tprint_debug(message, *args, **kwargs)
    else:
        tprint("🔍", message, *args, **kwargs)

def log_performance(message: str, *args: Any, **kwargs: Any) -> None:
    """Log performance message."""
    if TPRINT_AVAILABLE:
        tprint_performance(message, *args, **kwargs)
    else:
        tprint("⚡", message, *args, **kwargs)

# Context managers for structured logging
class LoggingContext:
    """Context manager for structured logging."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = __import__('time').time()
        log_info(f"🚀 Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = __import__('time').time() - self.start_time
        if exc_type is None:
            log_success(f"✅ {self.operation_name} completed in {duration:.3f}s")
        else:
            log_error(f"❌ {self.operation_name} failed after {duration:.3f}s: {exc_val}")

# Utility functions for common logging patterns
def log_step_progress(step_number: int, total_steps: int, step_name: str, *args: Any) -> None:
    """Log step progress in a pipeline."""
    percentage = (step_number / total_steps) * 100
    log_info(f"📊 [{step_number}/{total_steps}] ({percentage:.1f}%) {step_name}", *args)

def log_data_info(data_name: str, data_shape: Any, additional_info: str = "") -> None:
    """Log data information in a standardized format."""
    if hasattr(data_shape, 'shape'):
        shape_info = f"{data_shape.shape[0]} rows × {data_shape.shape[1]} columns"
    else:
        shape_info = str(data_shape)

    message = f"📊 {data_name}: {shape_info}"
    if additional_info:
        message += f" ({additional_info})"

    log_info(message)

def log_validation_result(validation_name: str, passed: bool, details: str = "") -> None:
    """Log validation results in a standardized format."""
    if passed:
        log_success(f"✅ {validation_name} validation passed")
        if details:
            log_debug(f"   Details: {details}")
    else:
        log_error(f"❌ {validation_name} validation failed")
        if details:
            log_info(f"   Details: {details}")

# Configuration for logging standards
LOGGING_CONFIG = {
    'use_tprint': TPRINT_AVAILABLE,
    'show_timestamps': True,
    'show_emojis': True,
    'show_component_prefix': True,
    'log_level': 'INFO'
}

# Export all functions
__all__ = [
    'get_logger',
    'log_info', 'log_warning', 'log_error', 'log_success', 'log_debug', 'log_performance',
    'LoggingContext', 'log_step_progress', 'log_data_info', 'log_validation_result',
    'LOGGING_CONFIG', 'TPRINT_AVAILABLE', 'SYSTEM_LOGGER_AVAILABLE'
]
