#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Numba-Friendly Timestamp Utilities

This module provides timestamp functionality that works within numba-compiled functions.
It uses numba's objmode context manager to handle datetime operations that aren't
natively supported by numba's nopython mode.

Key Features:
- Numba-compatible timestamp generation
- Console printing with timestamps in numba functions
- Performance-optimized timestamp formatting
- Fallback mechanisms for non-numba environments
"""

import time
from typing import Optional, Union
from datetime import datetime

try:
    import numba
    from numba import njit, objmode
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators for non-numba environments
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    objmode = None

# Global timestamp format configuration
TIMESTAMP_FORMAT = "%H:%M:%S"
DETAILED_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
SIMPLE_TIMESTAMP_FORMAT = "%H:%M:%S.%f"

class NumbaTimestampFormatter:
    """Numba-compatible timestamp formatter."""

    def __init__(self, format_string: str = TIMESTAMP_FORMAT):
        self.format_string = format_string

    def get_timestamp(self) -> str:
        """Get current timestamp as string."""
        return datetime.now().strftime(self.format_string)

    def get_timestamp_with_microseconds(self) -> str:
        """Get current timestamp with microseconds."""
        return datetime.now().strftime(SIMPLE_TIMESTAMP_FORMAT)[:-3]  # Remove last 3 digits for milliseconds

# Global formatter instance
_timestamp_formatter = NumbaTimestampFormatter()

def get_numba_timestamp() -> str:
    """Get a numba-compatible timestamp string."""
    return _timestamp_formatter.get_timestamp()

def get_detailed_timestamp() -> str:
    """Get a detailed timestamp string."""
    return datetime.now().strftime(DETAILED_TIMESTAMP_FORMAT)[:-3]  # Remove last 3 digits for milliseconds

def get_simple_timestamp() -> str:
    """Get a simple timestamp string."""
    return _timestamp_formatter.get_timestamp_with_microseconds()

# Numba-compatible print functions
if NUMBA_AVAILABLE:

    @njit
    def numba_print_with_timestamp(message: str) -> None:
        """Print with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            tprint(f"[{timestamp}] {message}")

    @njit
    def numba_print_detailed(message: str) -> None:
        """Print with detailed timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(DETAILED_TIMESTAMP_FORMAT)[:-3]
            tprint(f"[{timestamp}] {message}")

    @njit
    def numba_print_simple(message: str) -> None:
        """Print with simple timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(SIMPLE_TIMESTAMP_FORMAT)[:-3]
            tprint(f"[{timestamp}] {message}")

    @njit
    def numba_print_progress(step: int, total: int, message: str) -> None:
        """Print progress with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            progress = (step / total) * 100 if total > 0 else 0
            tprint(f"[{timestamp}] Progress: {step}/{total} ({progress:.1f}%) - {message}")

    @njit
    def numba_print_performance(operation: str, duration: float) -> None:
        """Print performance metrics with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            tprint(f"[{timestamp}] Performance: {operation} took {duration:.3f}s")

    @njit
    def numba_print_error(error_msg: str) -> None:
        """Print error with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            tprint(f"[{timestamp}] ERROR: {error_msg}")

    @njit
    def numba_print_warning(warning_msg: str) -> None:
        """Print warning with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            tprint(f"[{timestamp}] WARNING: {warning_msg}")

    @njit
    def numba_print_info(info_msg: str) -> None:
        """Print info with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            tprint(f"[{timestamp}] INFO: {info_msg}")

    @njit
    def numba_print_debug(debug_msg: str) -> None:
        """Print debug with timestamp in numba nopython mode."""
        with objmode():
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            tprint(f"[{timestamp}] DEBUG: {debug_msg}")

else:
    # Fallback functions for non-numba environments
    def numba_print_with_timestamp(message: str) -> None:
        """Fallback print with timestamp."""
        timestamp = get_numba_timestamp()
        tprint(f"[{timestamp}] {message}")

    def numba_print_detailed(message: str) -> None:
        """Fallback detailed print with timestamp."""
        timestamp = get_detailed_timestamp()
        tprint(f"[{timestamp}] {message}")

    def numba_print_simple(message: str) -> None:
        """Fallback simple print with timestamp."""
        timestamp = get_simple_timestamp()
        tprint(f"[{timestamp}] {message}")

    def numba_print_progress(step: int, total: int, message: str) -> None:
        """Fallback progress print with timestamp."""
        timestamp = get_numba_timestamp()
        progress = (step / total) * 100 if total > 0 else 0
        tprint(f"[{timestamp}] Progress: {step}/{total} ({progress:.1f}%) - {message}")

    def numba_print_performance(operation: str, duration: float) -> None:
        """Fallback performance print with timestamp."""
        timestamp = get_numba_timestamp()
        tprint(f"[{timestamp}] Performance: {operation} took {duration:.3f}s")

    def numba_print_error(error_msg: str) -> None:
        """Fallback error print with timestamp."""
        timestamp = get_numba_timestamp()
        tprint(f"[{timestamp}] ERROR: {error_msg}")

    def numba_print_warning(warning_msg: str) -> None:
        """Fallback warning print with timestamp."""
        timestamp = get_numba_timestamp()
        tprint(f"[{timestamp}] WARNING: {warning_msg}")

    def numba_print_info(info_msg: str) -> None:
        """Fallback info print with timestamp."""
        timestamp = get_numba_timestamp()
        tprint(f"[{timestamp}] INFO: {info_msg}")

    def numba_print_debug(debug_msg: str) -> None:
        """Fallback debug print with timestamp."""
        timestamp = get_numba_timestamp()
        tprint(f"[{timestamp}] DEBUG: {debug_msg}")

# Utility functions for getting timestamps in numba functions
if NUMBA_AVAILABLE:

    @njit
    def get_numba_timestamp_string() -> str:
        """Get timestamp string in numba nopython mode."""
        with objmode(timestamp=str):
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        return timestamp

    @njit
    def get_numba_detailed_timestamp_string() -> str:
        """Get detailed timestamp string in numba nopython mode."""
        with objmode(timestamp=str):
            timestamp = datetime.now().strftime(DETAILED_TIMESTAMP_FORMAT)[:-3]
        return timestamp

    @njit
    def get_numba_simple_timestamp_string() -> str:
        """Get simple timestamp string in numba nopython mode."""
        with objmode(timestamp=str):
            timestamp = datetime.now().strftime(SIMPLE_TIMESTAMP_FORMAT)[:-3]
        return timestamp

else:
    # Fallback functions
    def get_numba_timestamp_string() -> str:
        """Fallback timestamp string."""
        return get_numba_timestamp()

    def get_numba_detailed_timestamp_string() -> str:
        """Fallback detailed timestamp string."""
        return get_detailed_timestamp()

    def get_numba_simple_timestamp_string() -> str:
        """Fallback simple timestamp string."""
        return get_simple_timestamp()

# Performance monitoring utilities for numba
if NUMBA_AVAILABLE:

    @njit
    def numba_timer_start() -> float:
        """Start a timer in numba nopython mode."""
        return time.perf_counter()

    @njit
    def numba_timer_elapsed(start_time: float) -> float:
        """Get elapsed time in numba nopython mode."""
        return time.perf_counter() - start_time

    @njit
    def numba_print_timing(operation: str, start_time: float) -> None:
        """Print timing information in numba nopython mode."""
        elapsed = numba_timer_elapsed(start_time)
        numba_print_performance(operation, elapsed)

else:
    # Fallback functions
    def numba_timer_start() -> float:
        """Fallback timer start."""
        return time.perf_counter()

    def numba_timer_elapsed(start_time: float) -> float:
        """Fallback timer elapsed."""
        return time.perf_counter() - start_time

    def numba_print_timing(operation: str, start_time: float) -> None:
        """Fallback timing print."""
        elapsed = numba_timer_elapsed(start_time)
        numba_print_performance(operation, elapsed)

# Example usage functions
def example_numba_function_with_timestamps():
    """Example of how to use timestamps in numba functions."""

    @njit
    def process_data_with_timestamps(data: list) -> list:
        """Example numba function with timestamp logging."""
        numba_print_info("Starting data processing")

        start_time = numba_timer_start()

        # Process data
        result = []
        for i, item in enumerate(data):
            if i % 1000 == 0:  # Log every 1000 items
                numba_print_progress(i, len(data), f"Processing item {i}")

            # Some processing
            processed_item = item * 2
            result.append(processed_item)

        numba_print_timing("Data processing", start_time)
        numba_print_info(f"Completed processing {len(data)} items")

        return result

    return process_data_with_timestamps

# Export all public functions
__all__ = [
    'numba_print_with_timestamp',
    'numba_print_detailed',
    'numba_print_simple',
    'numba_print_progress',
    'numba_print_performance',
    'numba_print_error',
    'numba_print_warning',
    'numba_print_info',
    'numba_print_debug',
    'get_numba_timestamp_string',
    'get_numba_detailed_timestamp_string',
    'get_numba_simple_timestamp_string',
    'numba_timer_start',
    'numba_timer_elapsed',
    'numba_print_timing',
    'get_numba_timestamp',
    'get_detailed_timestamp',
    'get_simple_timestamp',
    'NumbaTimestampFormatter',
    'NUMBA_AVAILABLE'
]
