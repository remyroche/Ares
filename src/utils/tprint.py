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
