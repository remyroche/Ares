"""
Timestamped print utility - Safe and explicit approach.

This module provides a tprint function that adds timestamps to print statements
without modifying global state or interfering with numba compilation.
"""

import sys
from datetime import datetime
from typing import Any, Optional


def tprint(*args, **kwargs) -> None:
    """
    Print with timestamp - Safe and explicit approach.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint("User logged in")  # [2025-09-11 06:30:15] User logged in
        tprint("Value:", 42)      # [2025-09-11 06:30:15] Value: 42
    """
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create timestamped message
    if args:
        # Add timestamp to the first argument
        first_arg = str(args[0])
        timestamped_args = (f"[{timestamp}] {first_arg}",) + args[1:]
    else:
        # No arguments, just timestamp
        timestamped_args = (f"[{timestamp}]",)
    
    # Print with timestamp
    print(*timestamped_args, **kwargs)


def tprint_debug(*args, **kwargs) -> None:
    """
    Print with timestamp and DEBUG prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_debug("Processing data")  # [2025-09-11 06:30:15] DEBUG: Processing data
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if args:
        first_arg = str(args[0])
        timestamped_args = (f"[{timestamp}] DEBUG: {first_arg}",) + args[1:]
    else:
        timestamped_args = (f"[{timestamp}] DEBUG:",)
    
    print(*timestamped_args, **kwargs)


def tprint_info(*args, **kwargs) -> None:
    """
    Print with timestamp and INFO prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_info("Operation completed")  # [2025-09-11 06:30:15] INFO: Operation completed
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if args:
        first_arg = str(args[0])
        timestamped_args = (f"[{timestamp}] INFO: {first_arg}",) + args[1:]
    else:
        timestamped_args = (f"[{timestamp}] INFO:",)
    
    print(*timestamped_args, **kwargs)


def tprint_warning(*args, **kwargs) -> None:
    """
    Print with timestamp and WARNING prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_warning("Low memory")  # [2025-09-11 06:30:15] WARNING: Low memory
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if args:
        first_arg = str(args[0])
        timestamped_args = (f"[{timestamp}] WARNING: {first_arg}",) + args[1:]
    else:
        timestamped_args = (f"[{timestamp}] WARNING:",)
    
    print(*timestamped_args, **kwargs)


def tprint_error(*args, **kwargs) -> None:
    """
    Print with timestamp and ERROR prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_error("Connection failed")  # [2025-09-11 06:30:15] ERROR: Connection failed
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if args:
        first_arg = str(args[0])
        timestamped_args = (f"[{timestamp}] ERROR: {first_arg}",) + args[1:]
    else:
        timestamped_args = (f"[{timestamp}] ERROR:",)
    
    print(*timestamped_args, **kwargs)


def tprint_success(*args, **kwargs) -> None:
    """
    Print with timestamp and SUCCESS prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_success("Data saved")  # [2025-09-11 06:30:15] SUCCESS: Data saved
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if args:
        first_arg = str(args[0])
        timestamped_args = (f"[{timestamp}] SUCCESS: {first_arg}",) + args[1:]
    else:
        timestamped_args = (f"[{timestamp}] SUCCESS:",)
    
    print(*timestamped_args, **kwargs)


def tprint_progress(step: int, total: int, message: str = "", **kwargs) -> None:
    """
    Print progress with timestamp.
    
    Args:
        step: Current step number
        total: Total number of steps
        message: Optional message
        **kwargs: Additional keyword arguments for print function
    
    Example:
        tprint_progress(3, 10, "Processing data")  # [2025-09-11 06:30:15] PROGRESS: 3/10 Processing data
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    percentage = (step / total) * 100 if total > 0 else 0
    
    progress_msg = f"[{timestamp}] PROGRESS: {step}/{total} ({percentage:.1f}%)"
    if message:
        progress_msg += f" {message}"
    
    print(progress_msg, **kwargs)


def tprint_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Print performance metrics with timestamp.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional keyword arguments for print function
    
    Example:
        tprint_performance("Data processing", 2.5)  # [2025-09-11 06:30:15] PERFORMANCE: Data processing took 2.5s
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    performance_msg = f"[{timestamp}] PERFORMANCE: {operation} took {duration:.3f}s"
    print(performance_msg, **kwargs)


# Convenience function for backward compatibility
def timestamped_print(*args, **kwargs) -> None:
    """
    Alias for tprint - backward compatibility.
    """
    tprint(*args, **kwargs)


# Export all functions
__all__ = [
    'tprint',
    'tprint_debug', 
    'tprint_info',
    'tprint_warning',
    'tprint_error',
    'tprint_success',
    'tprint_progress',
    'tprint_performance',
    'timestamped_print'
]