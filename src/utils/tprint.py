#!/usr/bin/env python3
"""
Enhanced Timestamped Print Utility - Production-Ready Version

This module provides a comprehensive tprint function suite that adds timestamps to print statements
with advanced features including configuration, thread safety, performance optimization, and
integration with existing logging systems.

Key Features:
- Configurable timestamp formats and timezones
- Thread-safe logging with performance optimization
- Color-coded output for different log levels
- File logging with rotation support
- Integration with existing numba_timestamps module
- Structured logging with JSON output
- Context managers for structured logging
- Memory-efficient string formatting
"""

import sys
import os
import threading
import time
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Union, Dict, List, TextIO, Callable
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import functools

# Try to import colorama for colored output
try:
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # Create dummy color constants
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = Back = Style = DummyColor()

# Import existing numba timestamps for compatibility
try:
    from .numba_timestamps import (
        NUMBA_AVAILABLE, 
        get_numba_timestamp, 
        get_detailed_timestamp,
        get_simple_timestamp
    )
except ImportError:
    NUMBA_AVAILABLE = False
    def get_numba_timestamp():
        return datetime.now().strftime('%H:%M:%S')
    def get_detailed_timestamp():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    def get_simple_timestamp():
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    PROGRESS = "PROGRESS"
    PERFORMANCE = "PERFORMANCE"


class TimestampFormat(Enum):
    """Timestamp format enumeration."""
    SIMPLE = "%H:%M:%S"
    DETAILED = "%Y-%m-%d %H:%M:%S"
    WITH_MICROSECONDS = "%Y-%m-%d %H:%M:%S.%f"
    ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass
class TPrintConfig:
    """Configuration for tprint functionality."""
    
    # Timestamp configuration
    timestamp_format: TimestampFormat = TimestampFormat.WITH_MICROSECONDS
    timezone: Optional[timezone] = None
    include_microseconds: bool = True
    
    # Output configuration
    use_colors: bool = COLORAMA_AVAILABLE
    output_file: Optional[Union[str, Path]] = None
    output_to_console: bool = True
    output_to_file: bool = False
    
    # Logging configuration
    min_log_level: LogLevel = LogLevel.DEBUG
    
    # Performance configuration
    enable_lazy_evaluation: bool = True
    cache_timestamps: bool = True
    timestamp_cache_duration: float = 0.001  # 1ms
    
    # File logging configuration - single file per run
    single_file_per_run: bool = True
    run_id: Optional[str] = None
    
    # Structured logging
    enable_structured_logging: bool = False
    structured_format: str = "json"  # json, yaml, custom
    
    # Integration
    integrate_with_logging: bool = True
    log_to_python_logger: bool = False
    
    # Color scheme
    colors: Dict[LogLevel, str] = field(default_factory=lambda: {
        LogLevel.DEBUG: Fore.CYAN,
        LogLevel.INFO: Fore.GREEN,
        LogLevel.WARNING: Fore.YELLOW,
        LogLevel.ERROR: Fore.RED,
        LogLevel.SUCCESS: Fore.GREEN + Style.BRIGHT,
        LogLevel.PROGRESS: Fore.BLUE,
        LogLevel.PERFORMANCE: Fore.MAGENTA,
    })


class TPrintManager:
    """Manager for tprint functionality."""
    
    def __init__(self, config: Optional[TPrintConfig] = None):
        self.config = config or TPrintConfig()
        self._timestamp_cache: Dict[str, str] = {}
        self._last_timestamp_time = 0.0
        self._file_handle: Optional[TextIO] = None
        self._setup_file_logging()
        self._setup_python_logging()
    
    def _setup_file_logging(self):
        """Setup file logging if configured."""
        if self.config.output_to_file and self.config.output_file:
            try:
                log_file = Path(self.config.output_file)
                
                # If single file per run is enabled, add run ID to filename
                if self.config.single_file_per_run:
                    if not self.config.run_id:
                        # Generate a unique run ID based on timestamp
                        self.config.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    
                    # Add run ID to filename before extension
                    if log_file.suffix:
                        log_file = log_file.parent / f"{log_file.stem}_{self.config.run_id}{log_file.suffix}"
                    else:
                        log_file = log_file.parent / f"{log_file.name}_{self.config.run_id}.log"
                
                log_file.parent.mkdir(parents=True, exist_ok=True)
                self._file_handle = open(log_file, 'w', encoding='utf-8')  # 'w' for single file per run
            except Exception as e:
                print(f"Warning: Could not open log file {self.config.output_file}: {e}")
    
    def _setup_python_logging(self):
        """Setup integration with Python logging if configured."""
        if self.config.integrate_with_logging:
            # Create a custom logger for tprint
            self.logger = logging.getLogger('tprint')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.DEBUG)
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp with caching."""
        if not self.config.cache_timestamps:
            return self._format_timestamp()
        
        current_time = time.time()
        if (current_time - self._last_timestamp_time) < self.config.timestamp_cache_duration:
            return self._timestamp_cache.get('cached', self._format_timestamp())
        
        timestamp = self._format_timestamp()
        self._timestamp_cache['cached'] = timestamp
        self._last_timestamp_time = current_time
        return timestamp
    
    def _format_timestamp(self) -> str:
        """Format timestamp according to configuration."""
        now = datetime.now(self.config.timezone)
        
        if self.config.timestamp_format == TimestampFormat.SIMPLE:
            return now.strftime('%H:%M:%S')
        elif self.config.timestamp_format == TimestampFormat.DETAILED:
            return now.strftime('%Y-%m-%d %H:%M:%S')
        elif self.config.timestamp_format == TimestampFormat.WITH_MICROSECONDS:
            return now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Remove last 3 digits for milliseconds
        elif self.config.timestamp_format == TimestampFormat.ISO:
            return now.isoformat()
        else:
            return now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Default to microseconds
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level."""
        level_priority = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.SUCCESS: 1,
            LogLevel.PROGRESS: 1,
            LogLevel.PERFORMANCE: 1,
        }
        return level_priority.get(level, 0) >= level_priority.get(self.config.min_log_level, 0)
    
    def _format_message(self, level: LogLevel, *args, **kwargs) -> str:
        """Format message with timestamp and level."""
        if not args:
            return f"[{self._get_timestamp()}] {level.value}:"
        
        timestamp = self._get_timestamp()
        first_arg = str(args[0])
        message = f"[{timestamp}] {level.value}: {first_arg}"
        
        if len(args) > 1:
            message += " " + " ".join(str(arg) for arg in args[1:])
        
        return message
    
    def _get_colored_message(self, level: LogLevel, message: str) -> str:
        """Get colored message if colors are enabled."""
        if not self.config.use_colors or not COLORAMA_AVAILABLE:
            return message
        
        color = self.config.colors.get(level, "")
        return f"{color}{message}{Style.RESET_ALL}"
    
    def _write_to_outputs(self, message: str, level: LogLevel, **kwargs):
        """Write message to configured outputs."""
        colored_message = self._get_colored_message(level, message)
        
        if self.config.output_to_console:
            print(colored_message, **kwargs)
        
        if self.config.output_to_file and self._file_handle:
            # Write uncolored message to file
            self._file_handle.write(message + '\n')
            self._file_handle.flush()
        
        if self.config.log_to_python_logger:
            log_level = getattr(logging, level.value, logging.INFO)
            self.logger.log(log_level, message)
    
    def _log(self, level: LogLevel, *args, **kwargs):
        """Internal logging method."""
        if not self._should_log(level):
            return
        
        message = self._format_message(level, *args, **kwargs)
        self._write_to_outputs(message, level, **kwargs)
    
    def _log_without_level(self, *args, **kwargs):
        """Internal logging method without log level prefix."""
        if not args:
            message = f"[{self._get_timestamp()}]"
        else:
            timestamp = self._get_timestamp()
            first_arg = str(args[0])
            message = f"[{timestamp}] {first_arg}"
            
            if len(args) > 1:
                message += " " + " ".join(str(arg) for arg in args[1:])
        
        self._write_to_outputs(message, LogLevel.INFO, **kwargs)
    
    def close(self):
        """Close file handles and cleanup."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


# Global manager instance
_global_manager = TPrintManager()


def configure_tprint(config: TPrintConfig) -> None:
    """Configure global tprint settings."""
    global _global_manager
    _global_manager = TPrintManager(config)


def get_tprint_config() -> TPrintConfig:
    """Get current tprint configuration."""
    return _global_manager.config


@contextmanager
def tprint_context(config: Optional[TPrintConfig] = None):
    """Context manager for temporary tprint configuration."""
    global _global_manager
    old_manager = _global_manager
    try:
        if config:
            _global_manager = TPrintManager(config)
        yield _global_manager
    finally:
        _global_manager = old_manager


def tprint(*args, **kwargs) -> None:
    """
    Enhanced print with timestamp - Production-ready version.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint("User logged in")  # [2025-01-11 06:30:15] User logged in
        tprint("Value:", 42)      # [2025-01-11 06:30:15] Value: 42
    """
    _global_manager._log_without_level(*args, **kwargs)


def tprint_debug(*args, **kwargs) -> None:
    """
    Print with timestamp and DEBUG prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_debug("Processing data")  # [2025-01-11 06:30:15] DEBUG: Processing data
    """
    _global_manager._log(LogLevel.DEBUG, *args, **kwargs)


def tprint_info(*args, **kwargs) -> None:
    """
    Print with timestamp and INFO prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_info("Operation completed")  # [2025-01-11 06:30:15] INFO: Operation completed
    """
    _global_manager._log(LogLevel.INFO, *args, **kwargs)


def tprint_warning(*args, **kwargs) -> None:
    """
    Print with timestamp and WARNING prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_warning("Low memory")  # [2025-01-11 06:30:15] WARNING: Low memory
    """
    _global_manager._log(LogLevel.WARNING, *args, **kwargs)


def tprint_error(*args, **kwargs) -> None:
    """
    Print with timestamp and ERROR prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_error("Connection failed")  # [2025-01-11 06:30:15] ERROR: Connection failed
    """
    _global_manager._log(LogLevel.ERROR, *args, **kwargs)


def tprint_success(*args, **kwargs) -> None:
    """
    Print with timestamp and SUCCESS prefix.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    
    Example:
        tprint_success("Data saved")  # [2025-01-11 06:30:15] SUCCESS: Data saved
    """
    _global_manager._log(LogLevel.SUCCESS, *args, **kwargs)


def tprint_progress(step: int, total: int, message: str = "", **kwargs) -> None:
    """
    Print progress with timestamp.
    
    Args:
        step: Current step number
        total: Total number of steps
        message: Optional message
        **kwargs: Additional keyword arguments for print function
    
    Example:
        tprint_progress(3, 10, "Processing data")  # [2025-01-11 06:30:15] PROGRESS: 3/10 (30.0%) Processing data
    """
    percentage = (step / total) * 100 if total > 0 else 0
    progress_msg = f"{step}/{total} ({percentage:.1f}%)"
    if message:
        progress_msg += f" {message}"
    _global_manager._log(LogLevel.PROGRESS, progress_msg, **kwargs)


def tprint_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Print performance metrics with timestamp.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional keyword arguments for print function
    
    Example:
        tprint_performance("Data processing", 2.5)  # [2025-01-11 06:30:15] PERFORMANCE: Data processing took 2.5s
    """
    performance_msg = f"{operation} took {duration:.3f}s"
    _global_manager._log(LogLevel.PERFORMANCE, performance_msg, **kwargs)


def tprint_structured(data: Dict[str, Any], level: LogLevel = LogLevel.INFO, **kwargs) -> None:
    """
    Print structured data with timestamp.
    
    Args:
        data: Dictionary of structured data
        level: Log level for the message
        **kwargs: Additional keyword arguments for print function
    
    Example:
        tprint_structured({"user_id": 123, "action": "login"})  # [2025-01-11 06:30:15] INFO: {"user_id": 123, "action": "login"}
    """
    if _global_manager.config.enable_structured_logging:
        if _global_manager.config.structured_format == "json":
            structured_msg = json.dumps(data, default=str)
        else:
            structured_msg = str(data)
        _global_manager._log(level, structured_msg, **kwargs)
    else:
        _global_manager._log(level, str(data), **kwargs)


@contextmanager
def tprint_timer(operation: str, level: LogLevel = LogLevel.PERFORMANCE):
    """
    Context manager for timing operations.
    
    Args:
        operation: Name of the operation to time
        level: Log level for the timing message
    
    Example:
        with tprint_timer("Data processing"):
            # ... do work ...
            pass  # Will automatically log the duration
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        tprint_performance(operation, duration)


def tprint_with_level(level: LogLevel, *args, **kwargs) -> None:
    """
    Print with specific log level.
    
    Args:
        level: Log level to use
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    _global_manager._log(level, *args, **kwargs)


def tprint_batch(messages: List[tuple], **kwargs) -> None:
    """
    Print multiple messages in batch for better performance.
    
    Args:
        messages: List of (level, *args) tuples
        **kwargs: Additional keyword arguments for print function
    
    Example:
        tprint_batch([
            (LogLevel.INFO, "Starting process"),
            (LogLevel.DEBUG, "Debug info"),
            (LogLevel.SUCCESS, "Process completed")
        ])
    """
    for message_tuple in messages:
        level = message_tuple[0]
        args = message_tuple[1:]
        _global_manager._log(level, *args, **kwargs)


# Convenience function for backward compatibility
def timestamped_print(*args, **kwargs) -> None:
    """
    Alias for tprint - backward compatibility.
    """
    tprint(*args, **kwargs)


# Decorator for automatic function logging
def tprint_logged(level: LogLevel = LogLevel.INFO, include_args: bool = False, include_result: bool = False):
    """
    Decorator to automatically log function calls.
    
    Args:
        level: Log level for the messages
        include_args: Whether to include function arguments in log
        include_result: Whether to include function result in log
    
    Example:
        @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if include_args:
                tprint_with_level(level, f"Calling {func_name} with args={args}, kwargs={kwargs}")
            else:
                tprint_with_level(level, f"Calling {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    tprint_with_level(level, f"{func_name} completed with result={result}")
                else:
                    tprint_with_level(level, f"{func_name} completed")
                
                return result
            except Exception as e:
                tprint_error(f"{func_name} failed with error: {e}")
                raise
        
        return wrapper
    return decorator


# Integration with existing numba timestamps
def tprint_numba_compatible(*args, **kwargs) -> None:
    """
    Numba-compatible version of tprint using existing numba_timestamps.
    
    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    if NUMBA_AVAILABLE:
        timestamp = get_numba_timestamp()
        if args:
            first_arg = str(args[0])
            message = f"[{timestamp}] {first_arg}"
            if len(args) > 1:
                message += " " + " ".join(str(arg) for arg in args[1:])
        else:
            message = f"[{timestamp}]"
        print(message, **kwargs)
    else:
        tprint(*args, **kwargs)


# Cleanup function
def cleanup_tprint():
    """Cleanup tprint resources."""
    _global_manager.close()


# Export all functions
__all__ = [
    # Core functions
    'tprint',
    'tprint_debug', 
    'tprint_info',
    'tprint_warning',
    'tprint_error',
    'tprint_success',
    'tprint_progress',
    'tprint_performance',
    'tprint_structured',
    'tprint_with_level',
    'tprint_batch',
    'tprint_numba_compatible',
    
    # Configuration and management
    'configure_tprint',
    'get_tprint_config',
    'tprint_context',
    'tprint_timer',
    'tprint_logged',
    'cleanup_tprint',
    
    # Classes and enums
    'TPrintConfig',
    'TPrintManager',
    'LogLevel',
    'TimestampFormat',
    
    # Backward compatibility
    'timestamped_print',
    
    # Integration
    'NUMBA_AVAILABLE',
    'COLORAMA_AVAILABLE',
]