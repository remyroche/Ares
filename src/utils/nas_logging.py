#!/usr/bin/env python3
"""
Standardized Logging System for NAS Components

This module provides a comprehensive logging system with structured logging,
performance monitoring, and centralized log management for NAS operations.
"""

import logging
import logging.handlers
import json
import time
import threading
import queue
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
import sys
import traceback
import psutil
import weakref

from .nas_error_handling import (
    ErrorContext, error_context, safe_execute, get_error_handler
)
from .nas_threading import ThreadSafeCounter, ThreadSafeQueue
from .nas_resource_manager import ResourceType, managed_resource


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Categories for organizing logs."""
    GENERAL = "general"
    PERFORMANCE = "performance"
    ERROR = "error"
    SECURITY = "security"
    AUDIT = "audit"
    DEBUG = "debug"
    TRAINING = "training"
    INFERENCE = "inference"
    DATA = "data"
    MODEL = "model"
    OPTIMIZATION = "optimization"
    RESOURCE = "resource"
    THREADING = "threading"
    NETWORK = "network"
    STORAGE = "storage"
    CONFIGURATION = "configuration"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None
    stack_trace: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_metadata: bool = True):
        super().__init__()
        self.include_metadata = include_metadata
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        try:
            log_entry = LogEntry(
                timestamp=record.created,
                level=LogLevel(record.levelno),
                category=getattr(record, 'category', LogCategory.GENERAL),
                message=record.getMessage(),
                component=getattr(record, 'component', 'unknown'),
                operation=getattr(record, 'operation', 'unknown'),
                user_id=getattr(record, 'user_id', None),
                session_id=getattr(record, 'session_id', None),
                request_id=getattr(record, 'request_id', None),
                duration=getattr(record, 'duration', None),
                memory_usage=getattr(record, 'memory_usage', None),
                cpu_usage=getattr(record, 'cpu_usage', None),
                metadata=getattr(record, 'metadata', {}),
                exception_info=getattr(record, 'exception_info', None),
                stack_trace=getattr(record, 'stack_trace', None)
            )
            
            # Convert to dictionary
            log_dict = {
                'timestamp': log_entry.timestamp,
                'level': log_entry.level.name,
                'category': log_entry.category.value,
                'message': log_entry.message,
                'component': log_entry.component,
                'operation': log_entry.operation
            }
            
            # Add optional fields
            if log_entry.user_id:
                log_dict['user_id'] = log_entry.user_id
            if log_entry.session_id:
                log_dict['session_id'] = log_entry.session_id
            if log_entry.request_id:
                log_dict['request_id'] = log_entry.request_id
            if log_entry.duration is not None:
                log_dict['duration'] = log_entry.duration
            if log_entry.memory_usage is not None:
                log_dict['memory_usage'] = log_entry.memory_usage
            if log_entry.cpu_usage is not None:
                log_dict['cpu_usage'] = log_entry.cpu_usage
            if log_entry.metadata:
                log_dict['metadata'] = log_entry.metadata
            if log_entry.exception_info:
                log_dict['exception_info'] = log_entry.exception_info
            if log_entry.stack_trace:
                log_dict['stack_trace'] = log_entry.stack_trace
            
            return json.dumps(log_dict, default=str)
            
        except Exception as e:
            # Fallback to simple format
            return f"{record.levelname}: {record.getMessage()}"


class PerformanceLoggingHandler(logging.Handler):
    """Custom handler for performance logging."""
    
    def __init__(self, performance_logger: 'PerformanceLogger'):
        super().__init__()
        self.performance_logger = performance_logger
    
    def emit(self, record: logging.LogRecord):
        """Emit performance log record."""
        try:
            if hasattr(record, 'performance_metrics'):
                self.performance_logger.log_performance(record.performance_metrics)
        except Exception:
            pass


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._performance_data: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        with self._lock:
            try:
                self._performance_data.append(metrics)
                
                # Keep only recent history
                if len(self._performance_data) > self.max_history:
                    self._performance_data = self._performance_data[-self.max_history:]
                    
            except Exception as e:
                context = ErrorContext("log_performance", "performance_logger")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self._performance_data:
                return {}
            
            # Calculate statistics
            durations = [m.get('duration', 0) for m in self._performance_data]
            memory_usage = [m.get('memory_usage', 0) for m in self._performance_data]
            cpu_usage = [m.get('cpu_usage', 0) for m in self._performance_data]
            
            return {
                'total_operations': len(self._performance_data),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage),
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_usage': max(cpu_usage)
            }
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self._lock:
            operation_data = [
                m for m in self._performance_data 
                if m.get('operation') == operation_name
            ]
            
            if not operation_data:
                return {}
            
            durations = [m.get('duration', 0) for m in operation_data]
            memory_usage = [m.get('memory_usage', 0) for m in operation_data]
            cpu_usage = [m.get('cpu_usage', 0) for m in operation_data]
            
            return {
                'operation': operation_name,
                'total_calls': len(operation_data),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage),
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_usage': max(cpu_usage)
            }


class LogManager:
    """Centralized log management system."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: Dict[str, logging.Handler] = {}
        self._performance_logger = PerformanceLogger()
        self._error_handler = get_error_handler()
        self._lock = threading.RLock()
        
        # Setup default logging configuration
        self._setup_default_logging()
    
    def _setup_default_logging(self) -> None:
        """Setup default logging configuration."""
        try:
            # Create main logger
            main_logger = self.get_logger("nas_main")
            
            # Add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter())
            main_logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "nas_main.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(StructuredFormatter())
            main_logger.addHandler(file_handler)
            
            # Add performance handler
            performance_handler = PerformanceLoggingHandler(self._performance_logger)
            main_logger.addHandler(performance_handler)
            
        except Exception as e:
            context = ErrorContext("setup_default_logging", "log_manager")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def get_logger(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        category: LogCategory = LogCategory.GENERAL
    ) -> logging.Logger:
        """Get or create a logger."""
        with self._lock:
            try:
                if name not in self._loggers:
                    logger = logging.getLogger(name)
                    logger.setLevel(level.value)
                    
                    # Add category filter
                    category_filter = CategoryFilter(category)
                    logger.addFilter(category_filter)
                    
                    self._loggers[name] = logger
                
                return self._loggers[name]
                
            except Exception as e:
                context = ErrorContext("get_logger", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)
                return logging.getLogger(name)
    
    def add_file_handler(
        self,
        logger_name: str,
        file_path: str,
        level: LogLevel = LogLevel.INFO,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ) -> None:
        """Add file handler to logger."""
        with self._lock:
            try:
                logger = self.get_logger(logger_name)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                file_handler.setFormatter(StructuredFormatter())
                file_handler.setLevel(level.value)
                
                logger.addHandler(file_handler)
                self._handlers[f"{logger_name}_file"] = file_handler
                
            except Exception as e:
                context = ErrorContext("add_file_handler", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def add_console_handler(
        self,
        logger_name: str,
        level: LogLevel = LogLevel.INFO
    ) -> None:
        """Add console handler to logger."""
        with self._lock:
            try:
                logger = self.get_logger(logger_name)
                
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(StructuredFormatter())
                console_handler.setLevel(level.value)
                
                logger.addHandler(console_handler)
                self._handlers[f"{logger_name}_console"] = console_handler
                
            except Exception as e:
                context = ErrorContext("add_console_handler", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def add_network_handler(
        self,
        logger_name: str,
        host: str,
        port: int,
        level: LogLevel = LogLevel.INFO
    ) -> None:
        """Add network handler to logger."""
        with self._lock:
            try:
                logger = self.get_logger(logger_name)
                
                network_handler = logging.handlers.SocketHandler(host, port)
                network_handler.setFormatter(StructuredFormatter())
                network_handler.setLevel(level.value)
                
                logger.addHandler(network_handler)
                self._handlers[f"{logger_name}_network"] = network_handler
                
            except Exception as e:
                context = ErrorContext("add_network_handler", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def remove_handler(self, logger_name: str, handler_type: str) -> None:
        """Remove handler from logger."""
        with self._lock:
            try:
                logger = self.get_logger(logger_name)
                handler_key = f"{logger_name}_{handler_type}"
                
                if handler_key in self._handlers:
                    handler = self._handlers[handler_key]
                    logger.removeHandler(handler)
                    del self._handlers[handler_key]
                    
            except Exception as e:
                context = ErrorContext("remove_handler", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def set_log_level(self, logger_name: str, level: LogLevel) -> None:
        """Set log level for logger."""
        with self._lock:
            try:
                logger = self.get_logger(logger_name)
                logger.setLevel(level.value)
                
            except Exception as e:
                context = ErrorContext("set_log_level", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._performance_logger.get_performance_stats()
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        return self._performance_logger.get_operation_stats(operation_name)
    
    def cleanup(self) -> None:
        """Clean up log manager resources."""
        with self._lock:
            try:
                # Close all handlers
                for handler in self._handlers.values():
                    handler.close()
                
                # Clear handlers and loggers
                self._handlers.clear()
                self._loggers.clear()
                
            except Exception as e:
                context = ErrorContext("cleanup", "log_manager")
                self._error_handler.handle_error(e, context, reraise=False)


class CategoryFilter(logging.Filter):
    """Filter logs by category."""
    
    def __init__(self, category: LogCategory):
        super().__init__()
        self.category = category
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record by category."""
        record_category = getattr(record, 'category', LogCategory.GENERAL)
        return record_category == self.category


class LoggingContext:
    """Context manager for logging with additional context."""
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        component: str,
        category: LogCategory = LogCategory.GENERAL,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        self.logger = logger
        self.operation = operation
        self.component = component
        self.category = category
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.start_cpu = psutil.cpu_percent()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def __enter__(self):
        """Enter logging context."""
        self.logger.info(
            f"Starting operation: {self.operation}",
            extra={
                'category': self.category,
                'component': self.component,
                'operation': self.operation,
                'user_id': self.user_id,
                'session_id': self.session_id,
                'request_id': self.request_id
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context."""
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()
        
        duration = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        cpu_avg = (self.start_cpu + end_cpu) / 2
        
        if exc_type is None:
            self.logger.info(
                f"Completed operation: {self.operation}",
                extra={
                    'category': self.category,
                    'component': self.component,
                    'operation': self.operation,
                    'user_id': self.user_id,
                    'session_id': self.session_id,
                    'request_id': self.request_id,
                    'duration': duration,
                    'memory_usage': memory_delta,
                    'cpu_usage': cpu_avg
                }
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation} - {exc_val}",
                extra={
                    'category': LogCategory.ERROR,
                    'component': self.component,
                    'operation': self.operation,
                    'user_id': self.user_id,
                    'session_id': self.session_id,
                    'request_id': self.request_id,
                    'duration': duration,
                    'memory_usage': memory_delta,
                    'cpu_usage': cpu_avg,
                    'exception_info': str(exc_val),
                    'stack_trace': traceback.format_exc()
                }
            )


class LoggingDecorator:
    """Decorator for automatic logging of function calls."""
    
    def __init__(
        self,
        logger: logging.Logger,
        operation: str = None,
        component: str = None,
        category: LogCategory = LogCategory.GENERAL,
        log_args: bool = False,
        log_result: bool = False
    ):
        self.logger = logger
        self.operation = operation
        self.component = component
        self.category = category
        self.log_args = log_args
        self.log_result = log_result
    
    def __call__(self, func: Callable) -> Callable:
        """Apply logging decorator to function."""
        operation = self.operation or func.__name__
        component = self.component or func.__module__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with LoggingContext(
                self.logger, operation, component, self.category
            ) as log_context:
                try:
                    if self.log_args:
                        log_context.logger.debug(
                            f"Function arguments: args={args}, kwargs={kwargs}",
                            extra={'category': LogCategory.DEBUG}
                        )
                    
                    result = func(*args, **kwargs)
                    
                    if self.log_result:
                        log_context.logger.debug(
                            f"Function result: {result}",
                            extra={'category': LogCategory.DEBUG}
                        )
                    
                    return result
                    
                except Exception as e:
                    log_context.logger.error(
                        f"Function failed: {e}",
                        extra={
                            'category': LogCategory.ERROR,
                            'exception_info': str(e),
                            'stack_trace': traceback.format_exc()
                        }
                    )
                    raise
        
        return wrapper


# Global log manager
_global_log_manager = LogManager()


def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    category: LogCategory = LogCategory.GENERAL
) -> logging.Logger:
    """Get a logger instance."""
    return _global_log_manager.get_logger(name, level, category)


def get_performance_logger() -> PerformanceLogger:
    """Get the performance logger."""
    return _global_log_manager._performance_logger


def log_performance(metrics: Dict[str, Any]) -> None:
    """Log performance metrics."""
    _global_log_manager._performance_logger.log_performance(metrics)


def log_operation(
    logger: logging.Logger,
    operation: str,
    component: str,
    category: LogCategory = LogCategory.GENERAL,
    level: LogLevel = LogLevel.INFO,
    message: str = None,
    **kwargs
) -> None:
    """Log an operation."""
    log_message = message or f"Operation: {operation}"
    
    logger.log(
        level.value,
        log_message,
        extra={
            'category': category,
            'component': component,
            'operation': operation,
            **kwargs
        }
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    component: str,
    **kwargs
) -> None:
    """Log an error."""
    logger.error(
        f"Error in {operation}: {error}",
        extra={
            'category': LogCategory.ERROR,
            'component': component,
            'operation': operation,
            'exception_info': str(error),
            'stack_trace': traceback.format_exc(),
            **kwargs
        }
    )


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    component: str,
    duration: float,
    memory_usage: float = None,
    cpu_usage: float = None,
    **kwargs
) -> None:
    """Log performance metrics."""
    metrics = {
        'operation': operation,
        'component': component,
        'duration': duration,
        'memory_usage': memory_usage,
        'cpu_usage': cpu_usage,
        **kwargs
    }
    
    logger.info(
        f"Performance metrics for {operation}",
        extra={
            'category': LogCategory.PERFORMANCE,
            'component': component,
            'operation': operation,
            'performance_metrics': metrics,
            **kwargs
        }
    )


@contextmanager
def logging_context(
    logger: logging.Logger,
    operation: str,
    component: str,
    category: LogCategory = LogCategory.GENERAL,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
):
    """Context manager for logging operations."""
    with LoggingContext(
        logger, operation, component, category, user_id, session_id, request_id
    ) as log_context:
        yield log_context


def logging_decorator(
    logger: logging.Logger,
    operation: str = None,
    component: str = None,
    category: LogCategory = LogCategory.GENERAL,
    log_args: bool = False,
    log_result: bool = False
):
    """Decorator for automatic logging of function calls."""
    return LoggingDecorator(
        logger, operation, component, category, log_args, log_result
    )


def setup_logging(
    log_dir: str = "logs",
    level: LogLevel = LogLevel.INFO,
    console_output: bool = True,
    file_output: bool = True,
    network_output: bool = False,
    network_host: str = "localhost",
    network_port: int = 9020
) -> LogManager:
    """Setup logging configuration."""
    global _global_log_manager
    
    _global_log_manager = LogManager(log_dir)
    
    if console_output:
        _global_log_manager.add_console_handler("nas_main", level)
    
    if file_output:
        _global_log_manager.add_file_handler("nas_main", str(Path(log_dir) / "nas_main.log"), level)
    
    if network_output:
        _global_log_manager.add_network_handler("nas_main", network_host, network_port, level)
    
    return _global_log_manager


def cleanup_logging() -> None:
    """Clean up logging resources."""
    _global_log_manager.cleanup()


# Export main classes and functions
__all__ = [
    'LogLevel',
    'LogCategory',
    'LogEntry',
    'StructuredFormatter',
    'PerformanceLoggingHandler',
    'PerformanceLogger',
    'LogManager',
    'CategoryFilter',
    'LoggingContext',
    'LoggingDecorator',
    'get_logger',
    'get_performance_logger',
    'log_performance',
    'log_operation',
    'log_error',
    'log_performance_metrics',
    'logging_context',
    'logging_decorator',
    'setup_logging',
    'cleanup_logging'
]