"""
Enhanced timestamped print utility - Advanced features and configurations.

This module provides a comprehensive tprint system with advanced features
including colors, file output, threading safety, performance tracking,
context management, and much more.
"""

import sys
import os
import threading
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log levels for tprint system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"


class Color(Enum):
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


@dataclass
class TPrintConfig:
    """Configuration for tprint system."""
    # Timestamp format
    timestamp_format: str = '%Y-%m-%d %H:%M:%S'
    include_microseconds: bool = False
    include_timezone: bool = False
    
    # Output options
    enable_colors: bool = True
    enable_file_output: bool = False
    log_file_path: Optional[str] = None
    log_file_rotation: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Threading
    thread_safe: bool = True
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_threshold: float = 1.0  # seconds
    
    # Context tracking
    enable_context_tracking: bool = True
    max_context_depth: int = 10
    
    # Filtering
    min_log_level: LogLevel = LogLevel.DEBUG
    enable_filtering: bool = False
    allowed_modules: Optional[List[str]] = None
    blocked_modules: Optional[List[str]] = None


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    operation: str
    duration: float
    timestamp: datetime
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TPrintManager:
    """Advanced tprint manager with configuration and state management."""
    
    def __init__(self, config: Optional[TPrintConfig] = None):
        self.config = config or TPrintConfig()
        self._lock = threading.Lock() if self.config.thread_safe else None
        self._performance_metrics: List[PerformanceMetric] = []
        self._context_stack: List[str] = []
        self._start_time = datetime.now()
        self._log_file_handle = None
        
        # Initialize log file if enabled
        if self.config.enable_file_output and self.config.log_file_path:
            self._setup_log_file()
    
    def _acquire_lock(self):
        """Acquire thread lock if threading is enabled."""
        if self._lock:
            self._lock.acquire()
    
    def _release_lock(self):
        """Release thread lock if threading is enabled."""
        if self._lock:
            self._lock.release()
    
    def _setup_log_file(self):
        """Setup log file with rotation."""
        log_path = Path(self.config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotate if file is too large
        if log_path.exists() and log_path.stat().st_size > self.config.max_log_file_size:
            backup_path = log_path.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            log_path.rename(backup_path)
        
        self._log_file_handle = open(log_path, 'a', encoding='utf-8')
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        now = datetime.now()
        
        if self.config.include_microseconds:
            timestamp = now.strftime(f'{self.config.timestamp_format}.%f')
        else:
            timestamp = now.strftime(self.config.timestamp_format)
        
        if self.config.include_timezone:
            timestamp += f' {now.strftime("%z")}'
        
        return timestamp
    
    def _get_color(self, level: LogLevel) -> str:
        """Get color code for log level."""
        if not self.config.enable_colors:
            return ""
        
        color_map = {
            LogLevel.DEBUG: Color.CYAN,
            LogLevel.INFO: Color.BLUE,
            LogLevel.WARNING: Color.YELLOW,
            LogLevel.ERROR: Color.RED,
            LogLevel.CRITICAL: Color.BG_RED + Color.WHITE,
            LogLevel.SUCCESS: Color.GREEN,
        }
        
        return color_map.get(level, Color.RESET).value
    
    def _format_message(self, level: LogLevel, message: str, context: Optional[str] = None) -> str:
        """Format message with timestamp, level, and context."""
        timestamp = self._get_timestamp()
        color = self._get_color(level)
        reset = Color.RESET.value if self.config.enable_colors else ""
        
        # Build context string
        context_str = ""
        if context:
            context_str = f" [{context}]"
        elif self._context_stack:
            context_str = f" [{'.'.join(self._context_stack)}]"
        
        return f"{color}[{timestamp}] {level.value}:{context_str} {message}{reset}"
    
    def _should_log(self, level: LogLevel, module: Optional[str] = None) -> bool:
        """Check if message should be logged based on filters."""
        if not self.config.enable_filtering:
            return True
        
        # Check log level
        if level.value not in [l.value for l in LogLevel]:
            return False
        
        # Check module filters
        if module:
            if self.config.blocked_modules and module in self.config.blocked_modules:
                return False
            if self.config.allowed_modules and module not in self.config.allowed_modules:
                return False
        
        return True
    
    def _write_to_file(self, message: str):
        """Write message to log file."""
        if self._log_file_handle:
            try:
                self._log_file_handle.write(message + '\n')
                self._log_file_handle.flush()
            except Exception:
                pass  # Fail silently to avoid breaking the main flow
    
    def tprint(self, *args, level: LogLevel = LogLevel.INFO, context: Optional[str] = None, 
               module: Optional[str] = None, **kwargs) -> None:
        """Enhanced tprint with advanced features."""
        self._acquire_lock()
        try:
            # Check if we should log this message
            if not self._should_log(level, module):
                return
            
            # Format message
            message = ' '.join(str(arg) for arg in args) if args else ""
            formatted_message = self._format_message(level, message, context)
            
            # Print to console
            print(formatted_message, **kwargs)
            
            # Write to file if enabled
            if self.config.enable_file_output:
                self._write_to_file(formatted_message)
                
        finally:
            self._release_lock()
    
    def tprint_debug(self, *args, **kwargs) -> None:
        """Print debug message."""
        self.tprint(*args, level=LogLevel.DEBUG, **kwargs)
    
    def tprint_info(self, *args, **kwargs) -> None:
        """Print info message."""
        self.tprint(*args, level=LogLevel.INFO, **kwargs)
    
    def tprint_warning(self, *args, **kwargs) -> None:
        """Print warning message."""
        self.tprint(*args, level=LogLevel.WARNING, **kwargs)
    
    def tprint_error(self, *args, **kwargs) -> None:
        """Print error message."""
        self.tprint(*args, level=LogLevel.ERROR, **kwargs)
    
    def tprint_critical(self, *args, **kwargs) -> None:
        """Print critical message."""
        self.tprint(*args, level=LogLevel.CRITICAL, **kwargs)
    
    def tprint_success(self, *args, **kwargs) -> None:
        """Print success message."""
        self.tprint(*args, level=LogLevel.SUCCESS, **kwargs)
    
    def tprint_progress(self, step: int, total: int, message: str = "", 
                       show_bar: bool = True, **kwargs) -> None:
        """Enhanced progress printing with progress bar."""
        percentage = (step / total) * 100 if total > 0 else 0
        
        if show_bar:
            bar_length = 20
            filled_length = int(bar_length * step // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            progress_msg = f"{message} [{bar}] {step}/{total} ({percentage:.1f}%)"
        else:
            progress_msg = f"{message} {step}/{total} ({percentage:.1f}%)"
        
        self.tprint(progress_msg, level=LogLevel.INFO, **kwargs)
    
    def tprint_performance(self, operation: str, duration: float, 
                          context: Optional[str] = None, **kwargs) -> None:
        """Enhanced performance tracking with metrics storage."""
        # Store performance metric
        if self.config.enable_performance_tracking:
            metric = PerformanceMetric(
                operation=operation,
                duration=duration,
                timestamp=datetime.now(),
                context=context,
                metadata=kwargs.get('metadata')
            )
            self._performance_metrics.append(metric)
        
        # Print performance message
        level = LogLevel.WARNING if duration > self.config.performance_threshold else LogLevel.INFO
        perf_msg = f"{operation} took {duration:.3f}s"
        self.tprint(perf_msg, level=level, context=context, **kwargs)
    
    @contextmanager
    def context(self, name: str):
        """Context manager for nested logging."""
        self._context_stack.append(name)
        try:
            yield
        finally:
            if self._context_stack:
                self._context_stack.pop()
    
    @contextmanager
    def performance_timer(self, operation: str, **kwargs):
        """Context manager for performance timing."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.tprint_performance(operation, duration, **kwargs)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self._performance_metrics:
            return {"message": "No performance metrics recorded"}
        
        total_operations = len(self._performance_metrics)
        total_duration = sum(m.duration for m in self._performance_metrics)
        avg_duration = total_duration / total_operations
        max_duration = max(m.duration for m in self._performance_metrics)
        min_duration = min(m.duration for m in self._performance_metrics)
        
        # Group by operation
        operation_stats = {}
        for metric in self._performance_metrics:
            if metric.operation not in operation_stats:
                operation_stats[metric.operation] = []
            operation_stats[metric.operation].append(metric.duration)
        
        # Calculate stats per operation
        operation_summary = {}
        for op, durations in operation_stats.items():
            operation_summary[op] = {
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations),
                "max": max(durations),
                "min": min(durations)
            }
        
        return {
            "total_operations": total_operations,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "operation_breakdown": operation_summary,
            "session_duration": (datetime.now() - self._start_time).total_seconds()
        }
    
    def export_performance_metrics(self, filepath: str) -> None:
        """Export performance metrics to JSON file."""
        metrics_data = []
        for metric in self._performance_metrics:
            metric_dict = asdict(metric)
            metric_dict['timestamp'] = metric.timestamp.isoformat()
            metrics_data.append(metric_dict)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        self._performance_metrics.clear()
    
    def close(self) -> None:
        """Close log file and cleanup."""
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None


# Global instance
_global_manager = TPrintManager()


# Convenience functions that use the global manager
def tprint(*args, **kwargs) -> None:
    """Enhanced tprint with global manager."""
    _global_manager.tprint(*args, **kwargs)


def tprint_debug(*args, **kwargs) -> None:
    """Print debug message."""
    _global_manager.tprint_debug(*args, **kwargs)


def tprint_info(*args, **kwargs) -> None:
    """Print info message."""
    _global_manager.tprint_info(*args, **kwargs)


def tprint_warning(*args, **kwargs) -> None:
    """Print warning message."""
    _global_manager.tprint_warning(*args, **kwargs)


def tprint_error(*args, **kwargs) -> None:
    """Print error message."""
    _global_manager.tprint_error(*args, **kwargs)


def tprint_critical(*args, **kwargs) -> None:
    """Print critical message."""
    _global_manager.tprint_critical(*args, **kwargs)


def tprint_success(*args, **kwargs) -> None:
    """Print success message."""
    _global_manager.tprint_success(*args, **kwargs)


def tprint_progress(step: int, total: int, message: str = "", show_bar: bool = True, **kwargs) -> None:
    """Enhanced progress printing."""
    _global_manager.tprint_progress(step, total, message, show_bar, **kwargs)


def tprint_performance(operation: str, duration: float, **kwargs) -> None:
    """Enhanced performance tracking."""
    _global_manager.tprint_performance(operation, duration, **kwargs)


@contextmanager
def tprint_context(name: str):
    """Context manager for nested logging."""
    with _global_manager.context(name):
        yield


@contextmanager
def tprint_timer(operation: str, **kwargs):
    """Context manager for performance timing."""
    with _global_manager.performance_timer(operation, **kwargs):
        yield


def configure_tprint(config: TPrintConfig) -> None:
    """Configure the global tprint manager."""
    global _global_manager
    _global_manager.close()  # Close existing manager
    _global_manager = TPrintManager(config)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance metrics summary."""
    return _global_manager.get_performance_summary()


def export_performance_metrics(filepath: str) -> None:
    """Export performance metrics to JSON file."""
    _global_manager.export_performance_metrics(filepath)


def clear_performance_metrics() -> None:
    """Clear all performance metrics."""
    _global_manager.clear_performance_metrics()


# Backward compatibility
def timestamped_print(*args, **kwargs) -> None:
    """Alias for tprint - backward compatibility."""
    tprint(*args, **kwargs)


# Export all functions and classes
__all__ = [
    # Core functions
    'tprint', 'tprint_debug', 'tprint_info', 'tprint_warning', 
    'tprint_error', 'tprint_critical', 'tprint_success',
    'tprint_progress', 'tprint_performance', 'timestamped_print',
    
    # Context managers
    'tprint_context', 'tprint_timer',
    
    # Configuration and management
    'configure_tprint', 'get_performance_summary', 
    'export_performance_metrics', 'clear_performance_metrics',
    
    # Classes and enums
    'TPrintManager', 'TPrintConfig', 'LogLevel', 'Color', 'PerformanceMetric'
]