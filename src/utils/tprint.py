#!/usr/bin/env python3

"""
Enhanced Timestamped Print Utility - Production-Ready Version with Auto-Logging

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
- Automatic logging of all print statements to both tprint and Python logging
- Custom print function that integrates with logging systems
"""

import sys
import os
import threading
import time
import json
import logging
import inspect
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Union, Dict, List, TextIO, Callable
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import functools
import io

# Import enhanced hardware optimization tools (optional to avoid circular imports)
try:
    from .hardware import (
        memory_optimized, m1_optimized, memory_efficient_function,
        gc_optimized_function, force_cleanup, get_memory_stats
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    # Create dummy functions
    def memory_optimized(*args, **kwargs): 
        def decorator(f):
            return f
        return decorator
    def m1_optimized(*args, **kwargs): 
        def decorator(f):
            return f
        return decorator
    def memory_efficient_function(func=None, *args, **kwargs): 
        def decorator(f):
            return f
        if func is None:
            return decorator
        return decorator(func)
    def gc_optimized_function(func=None, *args, **kwargs): 
        def decorator(f):
            return f
        if func is None:
            return decorator
        return decorator(func)
    def force_cleanup():
        """Force garbage collection and memory cleanup."""
        import gc
        gc.collect()
        try:
            # Try to import and use hardware-specific cleanup if available
            from src.utils.hardware import force_cleanup as hw_force_cleanup
            hw_force_cleanup()
        except ImportError:
            pass
    def get_memory_stats(): return {}

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
        get_simple_timestamp,
        numba_print_with_timestamp,
        numba_print_detailed,
        numba_print_simple,
        numba_print_progress,
        numba_print_performance,
        numba_print_error,
        numba_print_warning,
        numba_print_info,
        numba_print_debug,
        get_numba_timestamp_string,
        get_numba_detailed_timestamp_string,
        get_numba_simple_timestamp_string,
        numba_timer_start,
        numba_timer_elapsed,
        numba_print_timing,
        NumbaTimestampFormatter
    )
except ImportError:
    NUMBA_AVAILABLE = False
    def get_numba_timestamp():
        return datetime.now().strftime('%H:%M:%S')
    def get_detailed_timestamp():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    def get_simple_timestamp():
        return datetime.now().strftime('%H:%M:%S.%f')[:-3]

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

    def get_numba_timestamp_string() -> str:
        """Fallback timestamp string."""
        return get_numba_timestamp()

    def get_numba_detailed_timestamp_string() -> str:
        """Fallback detailed timestamp string."""
        return get_detailed_timestamp()

    def get_numba_simple_timestamp_string() -> str:
        """Fallback simple timestamp string."""
        return get_simple_timestamp()

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

    class NumbaTimestampFormatter:
        """Fallback Numba-compatible timestamp formatter."""
        def __init__(self, format_string: str = '%H:%M:%S'):
            self.format_string = format_string

        def get_timestamp(self) -> str:
            """Get current timestamp as string."""
            return datetime.now().strftime(self.format_string)

        def get_timestamp_with_microseconds(self) -> str:
            """Get current timestamp with microseconds."""
            return datetime.now().strftime('%H:%M:%S.%f')[:-3]  # Remove last 3 digits for milliseconds

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

    # Auto-logging configuration
    auto_log_prints: bool = True
    print_log_level: LogLevel = LogLevel.INFO
    capture_print_to_tprint: bool = True
    print_capture_level: LogLevel = LogLevel.INFO
    auto_replace_print: bool = False  # Set to True to auto-replace built-in print on import

    # Traceback configuration
    include_traceback: bool = True  # Include traceback in error messages
    traceback_depth: int = 10  # Maximum number of stack frames to show (0 = all)
    show_locals: bool = False  # Include local variables in traceback
    compact_traceback: bool = False  # Use compact traceback format

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
    """Manager for tprint functionality with hardware optimization."""

    @memory_optimized(optimization_level='balanced')
    def __init__(self, config: Optional[TPrintConfig] = None):
        self.config = config or TPrintConfig()
        # Minimal mode to reduce overhead in hot loops: set env TPRINT_MINIMAL=1
        try:
            self.minimal_mode = bool(int(os.getenv('TPRINT_MINIMAL', '0')))
        except Exception:
            self.minimal_mode = False
        if self.minimal_mode:
            # Disable colors, file logging, and python logger integration
            self.config.use_colors = False
            self.config.output_to_file = False
            self.config.log_to_python_logger = False
            self.config.integrate_with_logging = False
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
            # Try to use the system logger if available
            try:
                from .logger import system_logger
                self.logger = system_logger.getChild('tprint')
                # Use the system logger's existing handlers and configuration
            except ImportError:
                # Fallback to creating a custom logger for tprint
                self.logger = logging.getLogger('tprint')
                if not self.logger.handlers:
                    # Try to use the structured logging formatter if available
                    try:
                        from .structured_logging import get_json_formatter
                        formatter = get_json_formatter()
                        handler = logging.StreamHandler()
                        handler.setFormatter(formatter)
                        self.logger.addHandler(handler)
                    except ImportError:
                        # Fallback to basic formatter
                        handler = logging.StreamHandler()
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        handler.setFormatter(formatter)
                        self.logger.addHandler(handler)

                    # Also add the correlation ID filter if available
                    try:
                        from .structured_logging import CorrelationIdFilter
                        self.logger.addFilter(CorrelationIdFilter())
                    except ImportError:
                        pass

                    self.logger.setLevel(logging.DEBUG)

    @memory_efficient_function
    def _get_timestamp(self) -> str:
        """Get formatted timestamp with caching and memory optimization."""
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

    @memory_efficient_function
    def _write_to_outputs(self, message: str, level: LogLevel, **kwargs):
        """Write message to configured outputs with memory optimization."""
        # Fast path for minimal mode
        if getattr(self, 'minimal_mode', False):
            try:
                _original_print(message)
            except BrokenPipeError:
                sys.stdout.close()
                sys.exit(0)
            return

        colored_message = self._get_colored_message(level, message)

        # Filter out tprint-specific parameters that print() doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['color', 'bold']}

        if self.config.output_to_console:
            try:
                # Use original print to avoid recursion with captured stdout
                _original_print(colored_message, **filtered_kwargs)
            except BrokenPipeError:
                # Handle broken pipe gracefully (e.g., when piping output)
                sys.stdout.close()
                sys.exit(0)

        if self.config.output_to_file and self._file_handle:
            try:
                # Write uncolored message to file
                self._file_handle.write(message + '\n')
                self._file_handle.flush()
            except Exception:
                # Silently handle file write errors
                pass

        if self.config.log_to_python_logger:
            try:
                log_level = getattr(logging, level.value, logging.INFO)
                self.logger.log(log_level, message)
            except Exception:
                # Silently handle logger errors
                pass

    def _format_traceback(self, exc: Optional[BaseException] = None, include_locals: bool = None, depth: int = None) -> str:
        """Format traceback for error messages.

        Args:
            exc: Exception to format traceback for (if None, uses current exception)
            include_locals: Whether to include local variables (overrides config)
            depth: Maximum depth to show (overrides config, 0 = all)

        Returns:
            Formatted traceback string
        """
        if not self.config.include_traceback:
            return ""

        include_locals = include_locals if include_locals is not None else self.config.show_locals
        depth = depth if depth is not None else self.config.traceback_depth

        try:
            if exc is None:
                # Get current exception info
                exc_type, exc_value, exc_tb = sys.exc_info()
                if exc_type is None:
                    # No exception available, get stack trace instead
                    stack_frames = traceback.extract_stack()[:-1]  # Exclude this function
                    if depth > 0 and len(stack_frames) > depth:
                        stack_frames = stack_frames[-depth:]
                    tb_lines = ['Stack trace (most recent call last):']
                    for frame in stack_frames:
                        tb_lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
                        if frame.line:
                            tb_lines.append(f'    {frame.line}')
                    return '\n' + '\n'.join(tb_lines)
            else:
                exc_type = type(exc)
                exc_value = exc
                exc_tb = exc.__traceback__

            if exc_tb is None:
                return f"\n{exc_type.__name__}: {exc_value}"

            # Format the traceback
            if self.config.compact_traceback:
                # Compact format: just file, line, function
                tb_frames = traceback.extract_tb(exc_tb)
                if depth > 0 and len(tb_frames) > depth:
                    tb_frames = tb_frames[-depth:]
                tb_lines = ['\nTraceback (most recent call last):']
                for frame in tb_frames:
                    tb_lines.append(f'  {frame.filename}:{frame.lineno} in {frame.name}')
                tb_lines.append(f'{exc_type.__name__}: {exc_value}')
                return '\n'.join(tb_lines)
            else:
                # Full format
                tb_lines = ['', 'Traceback (most recent call last):']
                tb_frames = traceback.extract_tb(exc_tb)

                if depth > 0 and len(tb_frames) > depth:
                    tb_frames = tb_frames[-depth:]
                    tb_lines.append(f'  ... ({len(traceback.extract_tb(exc_tb)) - depth} frames omitted)')

                for frame in tb_frames:
                    tb_lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}')
                    if frame.line:
                        tb_lines.append(f'    {frame.line}')

                    # Add local variables if requested
                    if include_locals and exc_tb is not None:
                        # Get locals from the frame
                        try:
                            local_vars = exc_tb.tb_frame.f_locals
                            if local_vars:
                                tb_lines.append('    Local variables:')
                                for var_name, var_value in list(local_vars.items())[:5]:  # Limit to 5 vars
                                    try:
                                        var_repr = repr(var_value)
                                        if len(var_repr) > 100:
                                            var_repr = var_repr[:97] + '...'
                                        tb_lines.append(f'      {var_name} = {var_repr}')
                                    except Exception:
                                        tb_lines.append(f'      {var_name} = <unavailable>')
                        except Exception:
                            pass

                # Add the exception message
                tb_lines.append(f'{exc_type.__name__}: {exc_value}')

                # Check for exception chain (__cause__ or __context__)
                if hasattr(exc, '__cause__') and exc.__cause__ is not None:
                    tb_lines.append('')
                    tb_lines.append('The above exception was the direct cause of the following exception:')
                    tb_lines.append(self._format_traceback(exc.__cause__, include_locals, depth))
                elif hasattr(exc, '__context__') and exc.__context__ is not None and not exc.__suppress_context__:
                    tb_lines.append('')
                    tb_lines.append('During handling of the above exception, another exception occurred:')
                    tb_lines.append(self._format_traceback(exc.__context__, include_locals, depth))

                return '\n'.join(tb_lines)

        except Exception as format_error:
            return f"\n[Error formatting traceback: {format_error}]"

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

    def _log_print_statement(self, *args, **kwargs):
        """Log print statements to tprint and Python logging."""
        if not self.config.auto_log_prints:
            return

        # Prevent recursion when we're already capturing stdout
        if hasattr(kwargs, '_recursion_guard') and kwargs.get('_recursion_guard'):
            return

        # Format the message
        if not args:
            message = ""
        else:
            message = " ".join(str(arg) for arg in args)

        # Filter out recursion guard from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != '_recursion_guard'}

        # Log to tprint if enabled
        if self.config.capture_print_to_tprint:
            self._write_to_outputs(f"[PRINT] {message}", self.config.print_capture_level, **filtered_kwargs)

        # Log to Python logging if enabled
        if self.config.log_to_python_logger:
            try:
                log_level = getattr(logging, self.config.print_log_level.value, logging.INFO)
                self.logger.log(log_level, f"[PRINT] {message}")
            except Exception:
                # Silently handle logger errors
                pass

    @gc_optimized_function
    def close(self):
        """Close file handles and cleanup with garbage collection optimization."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        
        # Force cleanup of cached data
        self._timestamp_cache.clear()
        force_cleanup()

# Global manager instance
_global_manager = TPrintManager()

def configure_tprint(config: TPrintConfig) -> None:
    """Configure global tprint settings."""
    global _global_manager
    _global_manager = TPrintManager(config)

    # Auto-replace print if configured
    if config.auto_replace_print:
        replace_builtin_print()

def configure_tprint_with_system_logger(enable_logging: bool = True, enable_file_output: bool = True) -> None:
    """Configure tprint to integrate with the system logger."""
    config = TPrintConfig(
        integrate_with_logging=enable_logging,
        log_to_python_logger=enable_logging,
        output_to_file=enable_file_output,
        output_to_console=True,
        use_colors=True,
        min_log_level=LogLevel.INFO
    )
    configure_tprint(config)

def get_tprint_config() -> TPrintConfig:
    """Get current tprint configuration."""
    return _global_manager.config

def enable_auto_print_logging(enable: bool = True):
    """Enable or disable automatic logging of print statements.

    Args:
        enable: Whether to enable automatic print logging
    """
    config = _global_manager.config
    config.auto_log_prints = enable
    config.auto_replace_print = enable
    configure_tprint(config)

def set_print_log_level(level: LogLevel):
    """Set the log level for print statement logging.

    Args:
        level: Log level to use for print statements
    """
    config = _global_manager.config
    config.print_log_level = level
    config.print_capture_level = level
    configure_tprint(config)

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
        **kwargs: Keyword arguments for print function (including color for backward compatibility)
            - include_traceback: bool = True - Include traceback if error message detected
            - traceback_depth: int = None - Override default traceback depth
            - show_locals: bool = None - Override show_locals config

    Example:
        tprint("User logged in")  # [2025-01-11 06:30:15] User logged in
        tprint("Value:", 42)      # [2025-01-11 06:30:15] Value: 42
        tprint("Message", color="blue")  # [2025-01-11 06:30:15] Message (with blue color)
        tprint("Error occurred")  # [2025-01-11 06:30:15] Error occurred + traceback if in exception context
    """
    # Extract traceback-related kwargs
    include_traceback = kwargs.pop('include_traceback', True)
    traceback_depth = kwargs.pop('traceback_depth', None)
    show_locals = kwargs.pop('show_locals', None)
    
    # Handle color parameter for backward compatibility
    color = kwargs.pop('color', None)
    bold = kwargs.pop('bold', False)

    # Check if this looks like an error message
    is_error_message = False
    if args:
        message_str = str(args[0]).lower()
        error_indicators = ['error', 'failed', 'exception', 'traceback', 'crash', 'fatal', 'critical', '❌', '🚨']
        is_error_message = any(indicator in message_str for indicator in error_indicators)

    if color:
        # Map color names to log levels for backward compatibility
        color_to_level = {
            'red': LogLevel.ERROR,
            'green': LogLevel.SUCCESS,
            'yellow': LogLevel.WARNING,
            'blue': LogLevel.INFO,
            'cyan': LogLevel.DEBUG,
            'magenta': LogLevel.PERFORMANCE,
        }
        level = color_to_level.get(color, LogLevel.INFO)
        _global_manager._log(level, *args, **kwargs)
    else:
        _global_manager._log_without_level(*args, **kwargs)

    # Add traceback if this is an error message and we're in an exception context
    if is_error_message and include_traceback and _global_manager.config.include_traceback:
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            tb_str = _global_manager._format_traceback(exc_value, show_locals, traceback_depth)
            if tb_str:
                _global_manager._write_to_outputs(tb_str, LogLevel.ERROR)

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
            - include_traceback: bool = True - Include traceback if available
            - traceback_depth: int = None - Override default traceback depth
            - show_locals: bool = None - Override show_locals config

    Example:
        tprint_error("Connection failed")  # [2025-01-11 06:30:15] ERROR: Connection failed
        tprint_error("Error occurred", include_traceback=True)  # Includes traceback if in exception context
    """
    # Extract traceback-related kwargs
    include_traceback = kwargs.pop('include_traceback', True)
    traceback_depth = kwargs.pop('traceback_depth', None)
    show_locals = kwargs.pop('show_locals', None)

    # Log the error message
    _global_manager._log(LogLevel.ERROR, *args, **kwargs)

    # Add traceback if we're in an exception context
    if include_traceback and _global_manager.config.include_traceback:
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            tb_str = _global_manager._format_traceback(exc_value, show_locals, traceback_depth)
            if tb_str:
                _global_manager._write_to_outputs(tb_str, LogLevel.ERROR)

def tprint_exception(exc: Optional[BaseException] = None, message: str = "", **kwargs) -> None:
    """
    Print exception with full traceback details.

    Args:
        exc: Exception to log (if None, uses current exception from sys.exc_info())
        message: Optional message to include with the exception
        **kwargs: Additional keyword arguments
            - include_locals: bool - Include local variables in traceback
            - traceback_depth: int - Maximum depth of traceback to show (0 = all)
            - compact: bool - Use compact traceback format

    Example:
        try:
            # ... code that raises exception ...
        except Exception as e:
            tprint_exception(e, "Failed to process data")

        # Or use without argument in except block:
        try:
            # ... code ...
        except:
            tprint_exception(message="Unexpected error occurred")
    """
    # Extract configuration overrides
    include_locals = kwargs.pop('include_locals', None)
    traceback_depth = kwargs.pop('traceback_depth', None)
    compact = kwargs.pop('compact', None)

    # Get exception if not provided
    if exc is None:
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_type is not None:
            exc = exc_value
        else:
            tprint_error("tprint_exception called but no exception is active")
            return

    # Build the error message
    exc_type_name = type(exc).__name__
    exc_message = str(exc)

    if message:
        error_msg = f"{message}: {exc_type_name}: {exc_message}"
    else:
        error_msg = f"{exc_type_name}: {exc_message}"

    # Log the error message
    _global_manager._log(LogLevel.ERROR, error_msg, **kwargs)

    # Temporarily override compact setting if requested
    if compact is not None:
        old_compact = _global_manager.config.compact_traceback
        _global_manager.config.compact_traceback = compact

    # Format and log the traceback
    tb_str = _global_manager._format_traceback(exc, include_locals, traceback_depth)
    if tb_str:
        _global_manager._write_to_outputs(tb_str, LogLevel.ERROR)

    # Restore compact setting if it was overridden
    if compact is not None:
        _global_manager.config.compact_traceback = old_compact

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
def tprint_logged(level: LogLevel = LogLevel.INFO, include_args: bool = False, include_result: bool = False,
                  include_traceback: bool = True, traceback_depth: int = None):
    """
    Decorator to automatically log function calls with enhanced error tracking.

    Args:
        level: Log level for the messages
        include_args: Whether to include function arguments in log
        include_result: Whether to include function result in log
        include_traceback: Whether to include traceback on errors (default: True)
        traceback_depth: Maximum traceback depth to show (None = use config default)

    Example:
        @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
        def my_function(x, y):
            return x + y

        @tprint_logged(LogLevel.DEBUG, include_traceback=True, traceback_depth=5)
        def risky_function():
            # Will show detailed traceback if it fails
            pass
    """
    def decorator(func):
        func_name = func.__name__
        func_module = func.__module__
        is_coroutine = inspect.iscoroutinefunction(func)

        if is_coroutine:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get caller information
                frame = inspect.currentframe()
                caller_frame = frame.f_back if frame else None
                caller_info = ""
                if caller_frame:
                    caller_info = f" (called from {caller_frame.f_code.co_filename}:{caller_frame.f_lineno})"

                if include_args:
                    # Truncate long args for readability
                    args_repr = str(args)[:200] + ('...' if len(str(args)) > 200 else '')
                    kwargs_repr = str(kwargs)[:200] + ('...' if len(str(kwargs)) > 200 else '')
                    tprint_with_level(level, f"Calling {func_module}.{func_name} with args={args_repr}, kwargs={kwargs_repr}{caller_info}")
                else:
                    tprint_with_level(level, f"Calling {func_module}.{func_name}{caller_info}")

                try:
                    result = await func(*args, **kwargs)

                    if include_result:
                        result_repr = str(result)[:200] + ('...' if len(str(result)) > 200 else '')
                        tprint_with_level(level, f"{func_module}.{func_name} completed with result={result_repr}")
                    else:
                        tprint_with_level(level, f"{func_module}.{func_name} completed")

                    return result
                except Exception as e:
                    if include_traceback:
                        tprint_exception(e, f"{func_module}.{func_name} failed", traceback_depth=traceback_depth)
                    else:
                        tprint_error(f"{func_module}.{func_name} failed with error: {e}", include_traceback=False)
                    raise

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get caller information
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None
            caller_info = ""
            if caller_frame:
                caller_info = f" (called from {caller_frame.f_code.co_filename}:{caller_frame.f_lineno})"

            if include_args:
                # Truncate long args for readability
                args_repr = str(args)[:200] + ('...' if len(str(args)) > 200 else '')
                kwargs_repr = str(kwargs)[:200] + ('...' if len(str(kwargs)) > 200 else '')
                tprint_with_level(level, f"Calling {func_module}.{func_name} with args={args_repr}, kwargs={kwargs_repr}{caller_info}")
            else:
                tprint_with_level(level, f"Calling {func_module}.{func_name}{caller_info}")

            try:
                result = func(*args, **kwargs)

                if include_result:
                    result_repr = str(result)[:200] + ('...' if len(str(result)) > 200 else '')
                    tprint_with_level(level, f"{func_module}.{func_name} completed with result={result_repr}")
                else:
                    tprint_with_level(level, f"{func_module}.{func_name} completed")

                return result
            except Exception as e:
                if include_traceback:
                    tprint_exception(e, f"{func_module}.{func_name} failed", traceback_depth=traceback_depth)
                else:
                    tprint_error(f"{func_module}.{func_name} failed with error: {e}", include_traceback=False)
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
        tprint(message, **kwargs)
    else:
        tprint(*args, **kwargs)

# Custom print function that automatically logs to tprint and logging
def tprint_print(*args, **kwargs):
    """
    Custom print function that automatically logs to tprint and Python logging.
    This replaces the built-in print function to ensure all output is captured.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    # Call the original print function
    print(*args, **kwargs)

    # Also log to tprint and logging if enabled
    _global_manager._log_print_statement(*args, **kwargs)

# Store original print function to avoid recursion
import builtins
_original_print = builtins.print

# Enhanced print function with auto-logging
def enhanced_print(*args, **kwargs):
    """
    Enhanced print function that automatically logs all print statements.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments for print function
    """
    # Call the original print function to avoid recursion
    _original_print(*args, **kwargs)

    # Log to tprint and logging if enabled
    _global_manager._log_print_statement(*args, **kwargs)

# Function to replace built-in print with enhanced version
def replace_builtin_print():
    """Replace the built-in print function with our enhanced version."""
    builtins.print = enhanced_print

# Function to restore original print function
def restore_builtin_print():
    """Restore the original built-in print function."""
    builtins.print = _original_print

# Context manager for automatic print capture

@contextmanager
def capture_print_to_tprint():
    """
    Context manager that captures all print statements and redirects them to tprint.

    Usage:
        with capture_print_to_tprint():
            print("This will be captured and logged to tprint")
    """
    # Store original stdout
    original_stdout = sys.stdout

    # Create a custom stdout that captures print statements
    class CaptureStdout(io.StringIO):
        def write(self, s):
            if s.strip():  # Only log non-empty strings
                # Check if we're already in a logging context to prevent recursion
                if not getattr(self, '_in_logging_context', False):
                    self._in_logging_context = True
                    try:
                        _global_manager._log_print_statement(s.strip(), _recursion_guard=True)
                    finally:
                        self._in_logging_context = False
            return super().write(s)

        def flush(self):
            return super().flush()

    # Replace stdout temporarily
    captured_stdout = CaptureStdout()
    sys.stdout = captured_stdout

    try:
        yield captured_stdout
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

# Auto-replacement on import if configured
def _auto_setup_print_capture():
    """Auto-setup print capture if configured."""
    config = _global_manager.config
    if config.auto_log_prints and hasattr(config, 'auto_replace_print') and config.auto_replace_print:
        replace_builtin_print()

# Cleanup function
def cleanup_tprint():
    """Cleanup tprint resources."""
    _global_manager.close()

# Helper functions for traceback configuration
def enable_traceback(enabled: bool = True, depth: int = None, show_locals: bool = None, compact: bool = None):
    """Enable or disable traceback in error messages.

    Args:
        enabled: Whether to enable traceback
        depth: Maximum traceback depth (None = use current, 0 = all)
        show_locals: Whether to show local variables
        compact: Whether to use compact format

    Example:
        enable_traceback(True, depth=5, show_locals=True)
    """
    _global_manager.config.include_traceback = enabled
    if depth is not None:
        _global_manager.config.traceback_depth = depth
    if show_locals is not None:
        _global_manager.config.show_locals = show_locals
    if compact is not None:
        _global_manager.config.compact_traceback = compact

def get_traceback_config() -> Dict[str, Any]:
    """Get current traceback configuration.

    Returns:
        Dictionary with traceback configuration settings
    """
    return {
        'include_traceback': _global_manager.config.include_traceback,
        'traceback_depth': _global_manager.config.traceback_depth,
        'show_locals': _global_manager.config.show_locals,
        'compact_traceback': _global_manager.config.compact_traceback,
    }

@contextmanager
def enhanced_traceback(depth: int = 0, show_locals: bool = True, compact: bool = False):
    """Context manager for temporarily enabling enhanced traceback.

    Args:
        depth: Maximum traceback depth (0 = all)
        show_locals: Whether to show local variables
        compact: Whether to use compact format

    Example:
        with enhanced_traceback(depth=0, show_locals=True):
            # Code with enhanced error reporting
            risky_operation()
    """
    old_include = _global_manager.config.include_traceback
    old_depth = _global_manager.config.traceback_depth
    old_locals = _global_manager.config.show_locals
    old_compact = _global_manager.config.compact_traceback

    try:
        _global_manager.config.include_traceback = True
        _global_manager.config.traceback_depth = depth
        _global_manager.config.show_locals = show_locals
        _global_manager.config.compact_traceback = compact
        yield
    finally:
        _global_manager.config.include_traceback = old_include
        _global_manager.config.traceback_depth = old_depth
        _global_manager.config.show_locals = old_locals
        _global_manager.config.compact_traceback = old_compact

@dataclass
class DataFormatConfig:
    """Configuration for data format checking."""
    max_cols: int = 10
    max_rows: int = 5
    max_keys: int = 10
    max_preview_chars: int = 100
    max_stat_items: int = 1000
    timeout_seconds: float = 1.0
    include_values: bool = True
    include_memory: bool = True
    include_semantics: bool = True
    safe_sampling: bool = True
    sample_size: int = 1000

def _get_caller_chain(max_depth: int = 3) -> str:
    """Get a more detailed caller chain for better debugging."""
    frame = None
    current_frame = None
    try:
        import inspect
        frame = inspect.currentframe()
        chain = []
        current_frame = frame
        
        for _ in range(max_depth):
            if current_frame is None:
                break
            current_frame = current_frame.f_back
            if current_frame is None:
                break
                
            filename = current_frame.f_code.co_filename
            lineno = current_frame.f_lineno
            function = current_frame.f_code.co_name
            
            # Extract just the filename from the full path
            caller_filename = filename.split('/')[-1] if '/' in filename else filename.split('\\')[-1]
            chain.append(f"{caller_filename}:{lineno} in {function}")
        
        if chain:
            return f" (called from {' -> '.join(reversed(chain))})"
        return ""
    except Exception:
        return ""
    finally:
        # Clean up both frame references to avoid memory leaks
        if current_frame is not None:
            del current_frame
        if frame is not None:
            del frame

def _timeout_guard(timeout_seconds: float, operation_name: str = "operation"):
    """Context manager for timeout enforcement."""
    class TimeoutGuard:
        def __init__(self, timeout_seconds, operation_name):
            self.timeout_seconds = timeout_seconds
            self.operation_name = operation_name
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed > self.timeout_seconds:
                    # Use proper logging instead of direct print
                    try:
                        tprint_with_level(LogLevel.WARNING, 
                            f"⚠️  {self.operation_name} timed out after {elapsed:.2f}s (limit: {self.timeout_seconds}s)")
                    except Exception:
                        print(f"⚠️  {self.operation_name} timed out after {elapsed:.2f}s (limit: {self.timeout_seconds}s)")
            return False
    
    return TimeoutGuard(timeout_seconds, operation_name)

def _safe_repr(obj: Any, max_chars: int = 100) -> str:
    """Safe representation that won't explode on large objects."""
    try:
        import reprlib
        import textwrap
        
        # Create a custom reprlib.Repr with tuned limits
        repr_obj = reprlib.Repr()
        repr_obj.maxstring = max_chars
        repr_obj.maxother = max_chars
        repr_obj.maxlist = 10
        repr_obj.maxdict = 10
        repr_obj.maxset = 10
        repr_obj.maxtuple = 10
        
        # Use the custom repr
        safe_repr = repr_obj.repr(obj)
        
        # Further truncate if still too long
        if len(safe_repr) > max_chars:
            safe_repr = textwrap.shorten(safe_repr, width=max_chars, placeholder="...")
        return safe_repr
    except Exception:
        # Fallback to basic repr with truncation
        try:
            basic_repr = repr(obj)
            if len(basic_repr) > max_chars:
                return basic_repr[:max_chars-3] + "..."
            return basic_repr
        except Exception:
            return f"<{type(obj).__name__} (repr failed)>"

def _check_pandas_dataframe(data, name: str, config: DataFormatConfig, caller_info: str, level: LogLevel) -> Dict[str, Any]:
    """Comprehensive pandas DataFrame analysis."""
    summary = {"type": "DataFrame", "shape": data.shape}
    
    # Initialize defaults for variables that might not be set
    null_pct = None
    object_cols = []
    categorical_cols = []
    
    try:
        import pandas as pd
        import numpy as np
        
        # Basic info
        tprint_with_level(level, f"🔍 {name} format{caller_info}:")
        tprint_with_level(level, f"  Type: DataFrame")
        tprint_with_level(level, f"  Shape: {data.shape}")
        
        # Dtypes summary
        dtypes = data.dtypes
        dtype_counts = dtypes.value_counts().to_dict()
        # Cast to strings for JSON compatibility
        dtypes_dict = {str(k): str(v) for k, v in dict(dtypes).items()}
        dtype_counts_str = {str(k): v for k, v in dtype_counts.items()}
        tprint_with_level(level, f"  Dtypes: {dtypes_dict}")
        tprint_with_level(level, f"  Dtype counts: {dtype_counts_str}")
        
        # Index info with diagnostics
        idx = data.index
        info = {
            "class": type(idx).__name__,
            "dtype": str(getattr(idx, "dtype", "")),
            "is_unique": bool(getattr(idx, "is_unique", False)),
            "is_monotonic": bool(getattr(idx, "is_monotonic_increasing", False)),
        }
        tprint_with_level(level, f"  Index info: {info}")
        summary["index"] = info
        
        if config.include_semantics:
            # Null analysis
            null_counts = data.isnull().sum()
            total_nulls = null_counts.sum()
            # Zero-division guard
            total_cells = data.shape[0] * data.shape[1]
            null_pct = (total_nulls / total_cells) * 100 if total_cells > 0 else 0.0
            tprint_with_level(level, f"  Nulls: {total_nulls} total ({null_pct:.1f}%)")
            
            # Column analysis
            object_cols = data.select_dtypes(include=['object']).columns
            categorical_cols = data.select_dtypes(include=['category']).columns
            tprint_with_level(level, f"  Object cols: {len(object_cols)}, Categorical: {len(categorical_cols)}")
            
            # Suspicious columns
            high_null_cols = null_counts[null_counts > data.shape[0] * 0.5].index.tolist()
            if high_null_cols:
                tprint_with_level(level, f"  ⚠️  High null cols: {high_null_cols[:5]}")
        
        if config.include_memory:
            try:
                with _timeout_guard(config.timeout_seconds, "DataFrame memory calculation"):
                    memory_mb = data.memory_usage(deep=True).sum() / 1024**2
                    tprint_with_level(level, f"  Memory: {memory_mb:.2f} MB")
                    summary["memory_mb"] = memory_mb
            except Exception:
                tprint_with_level(level, f"  Memory: (unable to calculate)")
        
        # Sample columns
        sample_cols = data.columns[:config.max_cols].tolist()
        if len(data.columns) > config.max_cols:
            sample_cols.append(f"... and {len(data.columns) - config.max_cols} more")
        tprint_with_level(level, f"  Columns: {sample_cols}")
        
        summary.update({
            "dtypes": dtypes_dict,
            "dtype_counts": dtype_counts_str,
            "null_pct": null_pct,
            "object_cols": len(object_cols),
            "categorical_cols": len(categorical_cols)
        })
        
    except Exception as e:
        tprint_with_level(level, f"  ⚠️  DataFrame analysis error: {e}")
        summary["error"] = str(e)
    
    return summary

def _check_pandas_series(data, name: str, config: DataFormatConfig, caller_info: str, level: LogLevel) -> Dict[str, Any]:
    """Comprehensive pandas Series analysis."""
    summary = {"type": "Series", "length": len(data)}
    
    try:
        import pandas as pd
        import numpy as np
        
        tprint_with_level(level, f"🔍 {name} format{caller_info}:")
        tprint_with_level(level, f"  Type: Series")
        tprint_with_level(level, f"  Length: {len(data)}")
        tprint_with_level(level, f"  Dtype: {data.dtype}")
        tprint_with_level(level, f"  Index: {type(data.index).__name__}")
        
        if config.include_semantics:
            # Null analysis
            null_count = data.isnull().sum()
            # Zero-division guard
            null_pct = (null_count / len(data)) * 100 if len(data) > 0 else 0.0
            tprint_with_level(level, f"  Nulls: {null_count} ({null_pct:.1f}%)")
            
            # Uniqueness
            unique_count = data.nunique()
            # Zero-division guard
            unique_pct = (unique_count / len(data)) * 100 if len(data) > 0 else 0.0
            tprint_with_level(level, f"  Unique: {unique_count} ({unique_pct:.1f}%)")
            
            # Monotonicity using proper pandas API
            try:
                from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
                if is_numeric_dtype(data) or is_datetime64_any_dtype(data):
                    is_monotonic = data.is_monotonic_increasing or data.is_monotonic_decreasing
                    tprint_with_level(level, f"  Monotonic: {is_monotonic}")
                    summary["is_monotonic"] = is_monotonic
            except ImportError:
                # Fallback for older pandas versions
                if str(data.dtype) in ['int64', 'float64', 'datetime64[ns]']:
                    is_monotonic = data.is_monotonic_increasing or data.is_monotonic_decreasing
                    tprint_with_level(level, f"  Monotonic: {is_monotonic}")
                    summary["is_monotonic"] = is_monotonic
        
        if config.include_memory:
            try:
                with _timeout_guard(config.timeout_seconds, "Series memory calculation"):
                    memory_mb = data.memory_usage(deep=True) / 1024**2
                    tprint_with_level(level, f"  Memory: {memory_mb:.2f} MB")
                    summary["memory_mb"] = memory_mb
            except Exception:
                tprint_with_level(level, f"  Memory: (unable to calculate)")
        
        summary.update({
            "dtype": str(data.dtype),
            "null_pct": null_pct if config.include_semantics else None,
            "unique_pct": unique_pct if config.include_semantics else None
        })
        
    except Exception as e:
        tprint_with_level(level, f"  ⚠️  Series analysis error: {e}")
        summary["error"] = str(e)
    
    return summary

def _check_numpy_array(data, name: str, config: DataFormatConfig, caller_info: str, level: LogLevel) -> Dict[str, Any]:
    """Comprehensive numpy array analysis."""
    summary = {"type": "ndarray", "shape": data.shape, "dtype": str(data.dtype)}
    
    try:
        import numpy as np
        
        tprint_with_level(level, f"🔍 {name} format{caller_info}:")
        tprint_with_level(level, f"  Type: ndarray")
        tprint_with_level(level, f"  Shape: {data.shape}")
        tprint_with_level(level, f"  Dtype: {data.dtype}")
        tprint_with_level(level, f"  Size: {data.size}")
        tprint_with_level(level, f"  Itemsize: {data.itemsize} bytes")
        tprint_with_level(level, f"  Strides: {data.strides}")
        tprint_with_level(level, f"  Contiguous: C={data.flags.c_contiguous}, F={data.flags.f_contiguous}")
        
        if config.include_semantics and data.size > 0:
            # Safe sampling for large arrays
            if config.safe_sampling and data.size > config.sample_size:
                sample_indices = np.random.choice(data.size, min(config.sample_size, data.size), replace=False)
                sample_data = np.asarray(data.flat[sample_indices])
            else:
                sample_data = np.asarray(data.flat)
            
            # Limit statistics calculation based on max_stat_items
            max_stats = min(config.max_stat_items, len(sample_data))
            
            # Numeric analysis
            if np.issubdtype(data.dtype, np.number):
                try:
                    # Actually use max_stat_items
                    finite_mask = np.isfinite(sample_data[:max_stats])
                    finite_pct = float(np.mean(finite_mask) * 100.0)
                    tprint_with_level(level, f"  Finite: {finite_pct:.1f}%")
                    
                    if np.any(finite_mask):
                        finite_data = sample_data[:max_stats][finite_mask]
                        vmin = float(np.min(finite_data))
                        vmax = float(np.max(finite_data))
                        tprint_with_level(level, f"  Range: [{vmin:.3f}, {vmax:.3f}]")
                        summary.update({"finite_pct": finite_pct, "min_val": vmin, "max_val": vmax})
                except Exception:
                    pass
        
        if config.include_memory:
            try:
                memory_mb = data.nbytes / 1024**2
                tprint_with_level(level, f"  Memory: {memory_mb:.2f} MB")
                summary["memory_mb"] = memory_mb
            except Exception:
                tprint_with_level(level, f"  Memory: (unable to calculate)")
        
    except Exception as e:
        tprint_with_level(level, f"  ⚠️  Array analysis error: {e}")
        summary["error"] = str(e)
    
    return summary

def tprint_data_format(data: Any, name: str = "data", level: LogLevel = LogLevel.DEBUG, 
                      config: Optional[DataFormatConfig] = None, return_summary: bool = False) -> Optional[Dict[str, Any]]:
    """
    Universal data format checker - comprehensive version for fast troubleshooting.
    
    Provides detailed analysis of data formatting including:
    - Data types (int64 vs int32, dict, string, etc.)
    - Shapes and dimensions
    - Memory usage and performance characteristics
    - Data quality indicators (nulls, uniqueness, etc.)
    - Schema analysis and potential issues
    
    Args:
        data: Data to check format for
        name: Name/description of the data
        level: Log level for the output
        config: Configuration for analysis parameters
        return_summary: If True, returns a dict summary in addition to printing
    
    Returns:
        Optional dict with summary information if return_summary=True
    
    Example:
        tprint_data_format(my_dataframe, "training_data")  # Quick format check
        summary = tprint_data_format(42, "my_int", return_summary=True)  # Get summary
    """
    if config is None:
        config = DataFormatConfig()
    
    caller_info = _get_caller_chain()
    summary = {"name": name, "type": type(data).__name__}
    
    try:
        # Import libraries with individual tracking
        PANDAS_AVAILABLE = False
        NUMPY_AVAILABLE = False
        PYARROW_AVAILABLE = False
        SCIPY_AVAILABLE = False
        
        try:
            import pandas as pd
            PANDAS_AVAILABLE = True
        except ImportError:
            pass
        
        try:
            import numpy as np
            NUMPY_AVAILABLE = True
        except ImportError:
            pass
        
        try:
            import pyarrow as pa
            PYARROW_AVAILABLE = True
        except ImportError:
            pass
        
        try:
            import scipy.sparse as sp
            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False
        
        # Handle None first
        if data is None:
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: NoneType")
            tprint_with_level(level, f"  Value: None")
            summary["type"] = "NoneType"
            return summary if return_summary else None
        
        # Handle booleans before numbers (bool is subclass of int)
        elif isinstance(data, bool):
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: bool")
            tprint_with_level(level, f"  Value: {data}")
            summary["value"] = data
            return summary if return_summary else None
        
        # --- Parquet path (must be BEFORE the plain string branch) ---
        elif PYARROW_AVAILABLE and isinstance(data, (str, os.PathLike)) and str(data).lower().endswith('.parquet'):
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(data)
                tprint_with_level(level, f"🔍 {name} format{caller_info}:")
                tprint_with_level(level, "  Type: Parquet file")
                tprint_with_level(level, f"  Shape: {pf.metadata.num_rows} rows × {len(pf.schema_arrow)} cols")
                tprint_with_level(level, f"  Row groups: {pf.num_row_groups}")
                field_info = [(pf.schema_arrow.names[i], str(pf.schema_arrow.types[i])) for i in range(len(pf.schema_arrow))]
                tprint_with_level(level, f"  Schema: {field_info}")
                summary.update({"type": "Parquet file", "shape": (pf.metadata.num_rows, len(pf.schema_arrow)),
                                "row_groups": pf.num_row_groups, "schema": field_info})
            except Exception as err:
                tprint_with_level(level, f"🔍 {name} format{caller_info}: Parquet file (error: {err})")
                summary["error"] = str(err)
            return summary if return_summary else None
        
        # Handle strings
        elif isinstance(data, str):
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: str")
            tprint_with_level(level, f"  Length: {len(data)}")
            
            if config.include_values:
                preview = data[:config.max_preview_chars]
                tprint_with_level(level, f"  Preview: {_safe_repr(preview, config.max_preview_chars)}")
            
            if config.include_semantics:
                is_ascii = data.isascii()
                printable_ratio = sum(1 for c in data if c.isprintable()) / len(data) if data else 0
                tprint_with_level(level, f"  ASCII: {is_ascii}, Printable: {printable_ratio:.1%}")
                summary["is_ascii"] = is_ascii
                summary["printable_ratio"] = printable_ratio
            
            summary["length"] = len(data)
            return summary if return_summary else None
        
        # Handle bytes/bytearray/memoryview
        elif isinstance(data, (bytes, bytearray, memoryview)):
            # Convert memoryview to bytes for safe operations
            buf = bytes(data) if isinstance(data, memoryview) else data
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: {type(data).__name__}")
            tprint_with_level(level, f"  Length: {len(buf)} bytes")
            
            if config.include_values and len(buf) > 0:
                # Show first 16 bytes as hex
                preview_bytes = buf[:16]
                hex_preview = ' '.join(f'{b:02x}' for b in preview_bytes)
                tprint_with_level(level, f"  Hex preview: {hex_preview}{'...' if len(buf) > 16 else ''}")
                
                # Try to decode as UTF-8 for text preview
                try:
                    text_preview = buf[:config.max_preview_chars].decode('utf-8', errors='ignore')
                    tprint_with_level(level, f"  Text preview: {_safe_repr(text_preview, config.max_preview_chars)}")
                except Exception:
                    pass
            
            summary["length"] = len(buf)
            return summary if return_summary else None
        
        # Handle numbers (after bool check)
        elif isinstance(data, (int, float, complex)):
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: {type(data).__name__}")
            tprint_with_level(level, f"  Value: {data}")
            
            if isinstance(data, int):
                tprint_with_level(level, f"  Bits: {data.bit_length()}")
                summary["bits"] = data.bit_length()
            elif isinstance(data, float):
                if config.include_semantics:
                    import math
                    is_finite = math.isfinite(data)
                    tprint_with_level(level, f"  Finite: {is_finite}")
                    summary["is_finite"] = is_finite
            
            summary["value"] = data
            return summary if return_summary else None
        
        # CRITICAL: Handle library-specific types BEFORE generic iterable/sequence checks
        # This prevents pandas DataFrames/Series from being caught by generic sequence logic
        
        # Handle pandas DataFrames
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            summary = _check_pandas_dataframe(data, name, config, caller_info, level)
            return summary if return_summary else None
        
        # Handle pandas Series
        elif PANDAS_AVAILABLE and isinstance(data, pd.Series):
            summary = _check_pandas_series(data, name, config, caller_info, level)
            return summary if return_summary else None
        
        # Handle numpy arrays
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            summary = _check_numpy_array(data, name, config, caller_info, level)
            return summary if return_summary else None
        
        # Handle scipy sparse matrices
        elif SCIPY_AVAILABLE and sp.issparse(data):
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: {type(data).__name__} (sparse)")
            tprint_with_level(level, f"  Shape: {data.shape}")
            tprint_with_level(level, f"  Dtype: {data.dtype}")
            tprint_with_level(level, f"  NNZ: {data.nnz}")
            
            # Correct density calculation: nnz / (rows * cols), not nnz / size
            density = data.nnz / (data.shape[0] * data.shape[1]) if data.shape[0] > 0 and data.shape[1] > 0 else 0.0
            tprint_with_level(level, f"  Density: {density:.2%}")
            
            summary.update({
                "type": f"{type(data).__name__} (sparse)",
                "shape": data.shape,
                "dtype": str(data.dtype),
                "nnz": data.nnz,
                "density": density
            })
            return summary if return_summary else None
        
        # Handle PyArrow Tables
        elif PYARROW_AVAILABLE and isinstance(data, pa.Table):
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: Arrow Table")
            tprint_with_level(level, f"  Shape: {data.num_rows} rows × {data.num_columns} cols")
            
            # Schema details
            field_info = [(field.name, str(field.type)) for field in data.schema]
            tprint_with_level(level, f"  Schema: {field_info}")
            
            if config.include_semantics:
                # Null counts per column (use index-based access)
                null_counts = {data.schema[i].name: data.column(i).null_count for i in range(data.num_columns)}
                tprint_with_level(level, f"  Null counts: {null_counts}")
                summary["null_counts"] = null_counts
            
            summary.update({
                "type": "Arrow Table",
                "shape": (data.num_rows, data.num_columns),
                "schema": field_info
            })
            return summary if return_summary else None
        
        
        # Handle sets specifically
        elif isinstance(data, set):
            try:
                length = len(data)
                tprint_with_level(level, f"🔍 {name} format{caller_info}:")
                tprint_with_level(level, f"  Type: set")
                tprint_with_level(level, f"  Length: {length}")
                
                if length > 0 and config.include_values:
                    sample_size = min(config.max_rows, length)
                    sample_items = list(data)[:sample_size]
                    element_types = [type(item).__name__ for item in sample_items]
                    tprint_with_level(level, f"  Element types: {element_types}{'...' if length > sample_size else ''}")
                    summary["element_types"] = element_types
                
                summary["length"] = length
                return summary if return_summary else None
            except Exception as e:
                tprint_with_level(level, f"🔍 {name} format{caller_info}: set (error: {e})")
                summary["error"] = str(e)
                return summary if return_summary else None
        
        # Handle mappings (dict, etc.) - BEFORE sequences to prevent KeyError
        elif hasattr(data, 'keys') and hasattr(data, '__getitem__'):
            try:
                length = len(data)
                tprint_with_level(level, f"🔍 {name} format{caller_info}:")
                tprint_with_level(level, f"  Type: {type(data).__name__}")
                tprint_with_level(level, f"  Keys: {length}")
                
                if length > 0 and config.include_values:
                    sample_keys = list(data.keys())[:config.max_keys]
                    tprint_with_level(level, f"  Sample keys: {sample_keys}{'...' if length > config.max_keys else ''}")
                    
                    # Show value types for sample keys
                    try:
                        value_types = {k: type(data[k]).__name__ for k in sample_keys}
                        tprint_with_level(level, f"  Value types: {value_types}")
                        summary["sample_keys"] = sample_keys
                        summary["value_types"] = value_types
                    except Exception as e:
                        tprint_with_level(level, f"  Value types: (error accessing: {e})")
                
                summary["length"] = length
                return summary if return_summary else None
            except Exception as e:
                tprint_with_level(level, f"🔍 {name} format{caller_info}: {type(data).__name__} (error: {e})")
                summary["error"] = str(e)
                return summary if return_summary else None
        
        # Handle iterators/generators - BEFORE generic sequence handling
        elif hasattr(data, '__next__') and not isinstance(data, (str, bytes, bytearray)):
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: {type(data).__name__} (iterator)")
            tprint_with_level(level, f"  Note: Iterator not consumed")
            summary["type"] = f"{type(data).__name__} (iterator)"
            return summary if return_summary else None
        
        # Handle collections.abc types
        elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes, bytearray)):
            # Handle sequences - AFTER mappings
            if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
                try:
                    length = len(data)
                    tprint_with_level(level, f"🔍 {name} format{caller_info}:")
                    tprint_with_level(level, f"  Type: {type(data).__name__}")
                    tprint_with_level(level, f"  Length: {length}")
                    
                    if length > 0 and config.include_values:
                        sample_size = min(config.max_rows, length)
                        try:
                            # Use itertools.islice for safer sampling
                            import itertools
                            sample_items = list(itertools.islice(data, sample_size))
                            element_types = [type(item).__name__ for item in sample_items]
                            tprint_with_level(level, f"  Element types: {element_types}{'...' if length > sample_size else ''}")
                            summary["element_types"] = element_types
                        except Exception as e:
                            tprint_with_level(level, f"  Element types: (error sampling: {e})")
                    
                    summary["length"] = length
                    return summary if return_summary else None
                except Exception as e:
                    tprint_with_level(level, f"🔍 {name} format{caller_info}: {type(data).__name__} (error: {e})")
                    summary["error"] = str(e)
                    return summary if return_summary else None
        
        # Handle other objects
        else:
            tprint_with_level(level, f"🔍 {name} format{caller_info}:")
            tprint_with_level(level, f"  Type: {type(data).__name__}")
            
            if hasattr(data, '__len__'):
                try:
                    length = len(data)
                    tprint_with_level(level, f"  Length: {length}")
                    summary["length"] = length
                except Exception:
                    pass
            
            if config.include_values:
                preview = _safe_repr(data, config.max_preview_chars)
                tprint_with_level(level, f"  Preview: {preview}")
                summary["preview"] = preview
            
            return summary if return_summary else None
    
    except Exception as e:
        tprint_with_level(level, f"🔍 {name} format (error){caller_info}: {e}")
        summary["error"] = str(e)
        return summary if return_summary else None

def tprint_data_preview(data: Any, name: str = "data", max_rows: int = None, 
                       max_cols: int = None, level: LogLevel = LogLevel.DEBUG, 
                       include_metadata: bool = True, force_log: bool = False) -> None:
    """
    Smart data preview with performance optimization.
    
    Args:
        data: Data to preview (DataFrame, array, or any data structure)
        name: Name/description of the data
        max_rows: Maximum rows to show (default: from config or 5)
        max_cols: Maximum columns to show (default: from config or 10)
        level: Log level for the preview
        include_metadata: Whether to include metadata (default: True)
        force_log: Force logging even for large datasets (default: False)
    """
    # Get caller information for debugging
    try:
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        caller_info = ""
        if caller_frame:
            caller_file = caller_frame.f_code.co_filename
            caller_line = caller_frame.f_lineno
            caller_function = caller_frame.f_code.co_name
            # Extract just the filename from the full path
            caller_filename = caller_file.split('/')[-1] if '/' in caller_file else caller_file.split('\\')[-1]
            caller_info = f" (called from {caller_filename}:{caller_line} in {caller_function})"
    except Exception:
        caller_info = ""
    
    # Check if data preview is enabled
    try:
        import os
        if not os.getenv('ENABLE_DATA_PREVIEW', 'true').lower() == 'true':
            return
    except:
        pass
    
    # Load configuration defaults
    try:
        import os
        DATA_PREVIEW_CONFIG = {
            'max_rows': int(os.getenv('DATA_PREVIEW_MAX_ROWS', '5')),
            'max_cols': int(os.getenv('DATA_PREVIEW_MAX_COLS', '10')),
            'large_dataset_threshold': int(os.getenv('DATA_PREVIEW_LARGE_THRESHOLD', '10000'))
        }
    except:
        DATA_PREVIEW_CONFIG = {
            'max_rows': 5,
            'max_cols': 10,
            'large_dataset_threshold': 10000
        }
    
    # Use config defaults if not provided
    if max_rows is None:
        max_rows = DATA_PREVIEW_CONFIG['max_rows']
    if max_cols is None:
        max_cols = DATA_PREVIEW_CONFIG['max_cols']
    
    # Convert string log level to enum safely
    if isinstance(level, str):
        try:
            level = LogLevel[level.upper()]
        except KeyError:
            level = LogLevel.DEBUG
    
    # Check for large datasets to prevent log pollution (with improved __len__ check)
    try:
        if not force_log and hasattr(data, '__len__'):
            try:
                data_len = len(data)
                if data_len > DATA_PREVIEW_CONFIG['large_dataset_threshold']:
                    tprint_with_level(level, f"📊 {name} preview{caller_info}: Large dataset ({data_len} items) - use force_log=True to preview")
                    return
            except TypeError:
                # Handle 0-D arrays or other types that don't support len()
                pass
    except:
        pass
    
    try:
        # Try to import required libraries for type checking
        try:
            import pandas as pd
            import numpy as np
            PANDAS_AVAILABLE = True
            NUMPY_AVAILABLE = True
        except ImportError:
            PANDAS_AVAILABLE = False
            NUMPY_AVAILABLE = False
        
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            PYARROW_AVAILABLE = True
        except ImportError:
            PYARROW_AVAILABLE = False
        
        # Handle PyArrow Tables
        if PYARROW_AVAILABLE and isinstance(data, pa.Table):
            tprint_with_level(level, f"📊 {name} preview (Arrow Table){caller_info}:")
            tprint_with_level(level, f"  Schema: {data.schema}")
            tprint_with_level(level, f"  Num rows: {data.num_rows}, Num columns: {data.num_columns}")
            
            if include_metadata:
                try:
                    size_bytes = data.nbytes
                    tprint_with_level(level, f"  Estimated memory: {size_bytes / 1024**2:.2f} MB")
                except Exception as mem_err:
                    tprint_with_level(level, f"  Memory: (Could not measure: {mem_err})")
            
            # Preview first few rows
            try:
                if PANDAS_AVAILABLE:
                    preview_df = data.slice(0, max_rows).to_pandas()
                    tprint_with_level(level, f"  Sample data (first {max_rows} rows):")
                    tprint_with_level(level, f"  {preview_df}")
                else:
                    # Fallback to Arrow table slice
                    preview_table = data.slice(0, max_rows)
                    tprint_with_level(level, f"  Sample data (first {max_rows} rows):")
                    tprint_with_level(level, f"  {preview_table}")
            except Exception as err:
                tprint_with_level(level, f"  ⚠️  Could not preview rows: {err}")
        
        # Handle Parquet files by path
        elif PYARROW_AVAILABLE and isinstance(data, str) and data.endswith('.parquet'):
            try:
                pf = pq.ParquetFile(data)
                tprint_with_level(level, f"📊 {name} preview (Parquet file){caller_info}:")
                tprint_with_level(level, f"  Path: {data}")
                tprint_with_level(level, f"  Schema: {pf.schema_arrow}")
                tprint_with_level(level, f"  Num rows: {pf.metadata.num_rows}, Num row groups: {pf.num_row_groups}")
                
                if include_metadata:
                    try:
                        import os
                        file_size = os.path.getsize(data)
                        tprint_with_level(level, f"  File size: {file_size / 1024**2:.2f} MB")
                    except Exception as size_err:
                        tprint_with_level(level, f"  File size: (Could not measure: {size_err})")
                
                # Read first few rows (efficiently)
                if pf.num_row_groups > 0:
                    table = pf.read_row_group(0).slice(0, max_rows)
                    if PANDAS_AVAILABLE:
                        preview_df = table.to_pandas()
                        tprint_with_level(level, f"  Sample data (first {max_rows} rows from first row group):")
                        tprint_with_level(level, f"  {preview_df}")
                    else:
                        tprint_with_level(level, f"  Sample data (first {max_rows} rows from first row group):")
                        tprint_with_level(level, f"  {table}")
                else:
                    tprint_with_level(level, f"  No row groups available")
            
            except Exception as err:
                tprint_with_level(level, f"  ⚠️  Could not preview Parquet file: {err}")
        
        # Handle pandas DataFrames
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            tprint_with_level(level, f"📊 {name} preview{caller_info}:")
            tprint_with_level(level, f"  Shape: {data.shape}")
            tprint_with_level(level, f"  Dtypes: {dict(data.dtypes)}")
            
            if include_metadata:
                try:
                    memory_mb = data.memory_usage(deep=True).sum() / 1024**2
                    tprint_with_level(level, f"  Memory: {memory_mb:.2f} MB")
                except Exception as mem_err:
                    tprint_with_level(level, f"  Memory: (Could not measure: {mem_err})")
                
                # Check for common data quality issues
                try:
                    null_count = data.isnull().sum().sum()
                    if null_count > 0:
                        tprint_with_level(level, f"  ⚠️  Null values: {null_count}")
                except Exception as null_err:
                    tprint_with_level(level, f"  ⚠️  Could not check null values: {null_err}")
                
                # Check for infinite values
                try:
                    if NUMPY_AVAILABLE:
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            inf_count = np.isinf(data[numeric_cols]).sum().sum()
                            if inf_count > 0:
                                tprint_with_level(level, f"  ⚠️  Infinite values: {inf_count}")
                except Exception as inf_err:
                    tprint_with_level(level, f"  ⚠️  Could not check infinite values: {inf_err}")
            
            # Show sample data with smart truncation
            if len(data) > 0:
                preview_data = data.head(max_rows)
                if len(data.columns) > max_cols:
                    preview_data = preview_data.iloc[:, :max_cols]
                    tprint_with_level(level, f"  Sample data (first {max_rows} rows, first {max_cols} cols):")
                else:
                    tprint_with_level(level, f"  Sample data (first {max_rows} rows):")
                tprint_with_level(level, f"  {preview_data}")
            else:
                tprint_with_level(level, f"  Empty dataset")
        
        # Handle numpy arrays
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            tprint_with_level(level, f"📊 {name} preview{caller_info}:")
            tprint_with_level(level, f"  Shape: {data.shape}")
            tprint_with_level(level, f"  Dtype: {data.dtype}")
            
            if include_metadata:
                try:
                    memory_mb = data.nbytes / 1024**2
                    tprint_with_level(level, f"  Memory: {memory_mb:.2f} MB")
                except Exception as mem_err:
                    tprint_with_level(level, f"  Memory: (Could not measure: {mem_err})")
                
                # Check for data quality issues - simplified
                try:
                    if np.issubdtype(data.dtype, np.floating):
                        null_count = np.isnan(data).sum()
                        inf_count = np.isinf(data).sum()
                        
                        if null_count > 0:
                            tprint_with_level(level, f"  ⚠️  NaN values: {null_count}")
                        if inf_count > 0:
                            tprint_with_level(level, f"  ⚠️  Infinite values: {inf_count}")
                except Exception as quality_err:
                    tprint_with_level(level, f"  ⚠️  Could not check data quality: {quality_err}")
            
            # Show sample data
            if data.size > 0:
                if data.ndim == 1:
                    sample_size = min(max_rows, len(data))
                    tprint_with_level(level, f"  Sample data (first {sample_size} values):")
                    tprint_with_level(level, f"  {data[:sample_size]}")
                elif data.ndim == 2:
                    sample_rows = min(max_rows, data.shape[0])
                    sample_cols = min(max_cols, data.shape[1])
                    tprint_with_level(level, f"  Sample data (first {sample_rows} rows, first {sample_cols} cols):")
                    tprint_with_level(level, f"  {data[:sample_rows, :sample_cols]}")
                else:
                    tprint_with_level(level, f"  Array shape: {data.shape}")
            else:
                tprint_with_level(level, f"  Empty array")
        
        # Handle dictionaries with deep preview
        elif isinstance(data, dict):
            tprint_with_level(level, f"📊 {name} preview{caller_info}:")
            tprint_with_level(level, f"  Type: dict")
            tprint_with_level(level, f"  Keys: {len(data)}")
            
            if include_metadata:
                try:
                    import sys
                    size_bytes = sys.getsizeof(data)
                    tprint_with_level(level, f"  Memory: {size_bytes / 1024**2:.2f} MB")
                except Exception as mem_err:
                    tprint_with_level(level, f"  Memory: (Could not measure: {mem_err})")
            
            # Show sample keys and values
            sample_keys = list(data.keys())[:max_rows]
            tprint_with_level(level, f"  Sample keys: {sample_keys}")
            
            # Try to pretty print if it's JSON-serializable
            try:
                import json
                json_str = json.dumps(data, indent=2, default=str)
                if len(json_str) > 1000:
                    json_str = json_str[:1000] + "..."
                tprint_with_level(level, f"  JSON preview:\n{json_str}")
            except Exception:
                # Fallback to string representation
                data_str = str(data)
                if len(data_str) > 500:
                    data_str = data_str[:500] + "..."
                tprint_with_level(level, f"  Preview: {data_str}")
        
        # Handle other data types
        else:
            tprint_with_level(level, f"📊 {name} preview{caller_info}:")
            tprint_with_level(level, f"  Type: {type(data).__name__}")
            
            if hasattr(data, '__len__'):
                try:
                    tprint_with_level(level, f"  Length: {len(data)}")
                except Exception:
                    pass
            
            if include_metadata:
                try:
                    import sys
                    size_bytes = sys.getsizeof(data)
                    tprint_with_level(level, f"  Memory: {size_bytes / 1024**2:.2f} MB")
                except Exception as mem_err:
                    tprint_with_level(level, f"  Memory: (Could not measure: {mem_err})")
            
            # Show string representation (truncated)
            try:
                data_str = str(data)
                if len(data_str) > 500:
                    data_str = data_str[:500] + "..."
                tprint_with_level(level, f"  Preview: {data_str}")
            except Exception as str_err:
                tprint_with_level(level, f"  Preview: (Could not convert to string: {str_err})")
    
    except Exception as e:
        tprint_with_level(level, f"📊 {name} preview (error){caller_info}: {e}")


# Export all functions
__all__ = [
    # Core functions
    'tprint',
    'tprint_debug',
    'tprint_info',
    'tprint_warning',
    'tprint_error',
    'tprint_exception',
    'tprint_success',
    'tprint_progress',
    'tprint_performance',
    'tprint_structured',
    'tprint_with_level',
    'tprint_batch',
    'tprint_numba_compatible',
    'tprint_data_preview',
    'tprint_data_format',
    'DataFormatConfig',

    # Enhanced print functions
    'enhanced_print',
    'tprint_print',
    'replace_builtin_print',
    'restore_builtin_print',
    'capture_print_to_tprint',

    # Configuration and management
    'configure_tprint',
    'configure_tprint_with_system_logger',
    'get_tprint_config',
    'tprint_context',
    'tprint_timer',
    'tprint_logged',
    'cleanup_tprint',
    'enable_auto_print_logging',
    'set_print_log_level',
    '_auto_setup_print_capture',

    # Traceback configuration
    'enable_traceback',
    'get_traceback_config',
    'enhanced_traceback',

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

    # Numba compatibility exports
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
    'NumbaTimestampFormatter',
]

# Auto-setup print capture if configured
_auto_setup_print_capture()
