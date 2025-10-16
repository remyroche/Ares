"""
Unified Logger - Enhanced Logging System

This module provides a comprehensive, unified logging system that consolidates
functionality from logger.py, comprehensive_logger.py, and other logging utilities.
"""

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

class LoggingConfig:
    """Configuration for unified logging system."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 console_output: bool = True,
                 file_output: bool = True,
                 json_format: bool = False,
                 include_emojis: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.console_output = console_output
        self.file_output = file_output
        self.json_format = json_format
        self.include_emojis = include_emojis
        self.max_file_size = max_file_size
        self.backup_count = backup_count

# =============================================================================
# FORMATTERS
# =============================================================================

class EmojiFormatter(logging.Formatter):
    """Formatter that adds emojis to log messages."""
    
    LEVEL_EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        if hasattr(record, 'emoji'):
            emoji = record.emoji
        else:
            emoji = self.LEVEL_EMOJIS.get(record.levelname, '📝')

        # Avoid mutating record.msg permanently (which can leak to other handlers)
        original_msg = record.msg
        try:
            if not record.getMessage().startswith(emoji):
                record.msg = f"{emoji} {original_msg}"
            return super().format(record)
        finally:
            record.msg = original_msg

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter with relative time information."""
    
    def __init__(self, fmt=None, datefmt=None, include_relative=True):
        super().__init__(fmt, datefmt)
        self.include_relative = include_relative
        self.start_time = datetime.now()
    
    def formatTime(self, record, datefmt=None):
        """Format timestamp with relative time."""
        try:
            ct = datetime.fromtimestamp(record.created)
            
            if datefmt:
                timestamp = ct.strftime(datefmt)
            else:
                timestamp = ct.strftime('%b %d, %Y %H:%M:%S')
            
            if self.include_relative:
                relative_time = ct - self.start_time
                hours, remainder = divmod(relative_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                relative_str = f" (+{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d})"
                return f"{timestamp}{relative_str}"
            
            return timestamp
        except Exception:
            return super().formatTime(record, datefmt)

# =============================================================================
# SAFE STREAM HANDLER
# =============================================================================

class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that gracefully handles broken pipes."""

    def emit(self, record):
        """Emit a record, handling broken pipe errors gracefully."""
        try:
            super().emit(record)
        except (BrokenPipeError, OSError) as e:
            # Handle broken pipe by silently suppressing the error
            # This prevents crashes during module initialization when stdout is broken
            if hasattr(e, 'errno') and e.errno == 32:  # Broken pipe
                # Silently ignore broken pipe errors during logging
                pass
            else:
                # Re-raise other OSError exceptions
                raise

# =============================================================================
# UNIFIED LOGGER CLASS
# =============================================================================

class UnifiedLogger:
    """Unified logger that consolidates all logging functionality."""
    
    def __init__(self, config: LoggingConfig = None):
        self.config = config or LoggingConfig()
        self.loggers = {}
        self.handlers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup the logging system."""
        # Ensure log directory exists
        if self.config.file_output:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup console handler with error handling for broken pipes
        if self.config.console_output:
            console_handler = SafeStreamHandler(sys.stdout)
            console_handler.setLevel(self.config.log_level)

            if self.config.json_format:
                console_handler.setFormatter(JSONFormatter())
            elif self.config.include_emojis:
                console_handler.setFormatter(EmojiFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            else:
                console_handler.setFormatter(HumanReadableFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))

            root_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler
        
        # Setup file handler
        if self.config.file_output:
            try:
                from logging.handlers import RotatingFileHandler
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = self.config.log_dir / f'unified_{timestamp}.log'
                
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=self.config.max_file_size,
                    backupCount=self.config.backup_count
                )
                file_handler.setLevel(logging.DEBUG)
                
                if self.config.json_format:
                    file_handler.setFormatter(JSONFormatter())
                else:
                    file_handler.setFormatter(HumanReadableFormatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    ))
                
                root_logger.addHandler(file_handler)
                self.handlers['file'] = file_handler
                
            except ImportError:
                # Fallback to basic FileHandler
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = self.config.log_dir / f'unified_{timestamp}.log'
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(HumanReadableFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                
                root_logger.addHandler(file_handler)
                self.handlers['file'] = file_handler
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger instance."""
        if name is None:
            name = 'UnifiedLogger'
        
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        
        return self.loggers[name]
    
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """Get a component-specific logger."""
        return self.get_logger(f'Component.{component_name}')
    
    def get_system_logger(self) -> logging.Logger:
        """Get the system logger."""
        return self.get_logger('System')
    
    def get_data_logger(self) -> logging.Logger:
        """Get the data processing logger."""
        return self.get_logger('Data')
    
    def get_ml_logger(self) -> logging.Logger:
        """Get the ML processing logger."""
        return self.get_logger('ML')
    
    def get_trading_logger(self) -> logging.Logger:
        """Get the trading logger."""
        return self.get_logger('Trading')
    
    def log_metric(self, name: str, value: float, logger_name: str = 'Metrics'):
        """Log a metric with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"📊 {name}: {value}")
    
    def log_parameter(self, name: str, value: Any, logger_name: str = 'Parameters'):
        """Log a parameter with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"⚙️ {name}: {value}")
    
    def log_artifact(self, name: str, path: str, logger_name: str = 'Artifacts'):
        """Log an artifact with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"📁 {name}: {path}")
    
    def log_step_start(self, step_name: str, logger_name: str = 'Pipeline'):
        """Log step start with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"🚀 Starting step: {step_name}")
    
    def log_step_end(self, step_name: str, success: bool = True, logger_name: str = 'Pipeline'):
        """Log step end with emoji."""
        logger = self.get_logger(logger_name)
        emoji = "✅" if success else "❌"
        status = "completed" if success else "failed"
        logger.info(f"{emoji} Step {step_name} {status}")
    
    def log_data_operation(self, operation: str, details: str = "", logger_name: str = 'Data'):
        """Log data operation with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"🔄 {operation}: {details}")
    
    def log_error(self, message: str, exc_info: bool = False, logger_name: str = 'Error'):
        """Log an error with emoji."""
        logger = self.get_logger(logger_name)
        logger.error(f"❌ {message}", exc_info=exc_info)
    
    def log_warning(self, message: str, logger_name: str = 'Warning'):
        """Log a warning with emoji."""
        logger = self.get_logger(logger_name)
        logger.warning(f"⚠️ {message}")
    
    def log_info(self, message: str, logger_name: str = 'Info'):
        """Log info with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"ℹ️ {message}")
    
    def log_debug(self, message: str, logger_name: str = 'Debug'):
        """Log debug with emoji."""
        logger = self.get_logger(logger_name)
        logger.debug(f"🔍 {message}")
    
    def log_success(self, message: str, logger_name: str = 'Success'):
        """Log success with emoji."""
        logger = self.get_logger(logger_name)
        logger.info(f"✅ {message}")
    
    def log_launcher_start(self, command: str, symbol: str = None, exchange: str = None):
        """Log launcher start."""
        logger = self.get_system_logger()
        logger.info(f"🚀 Launcher started: {command}")
        if symbol and exchange:
            logger.info(f"📊 Symbol: {symbol}, Exchange: {exchange}")
    
    def log_launcher_end(self, exit_code: int):
        """Log launcher end."""
        logger = self.get_system_logger()
        status = 'SUCCESS' if exit_code == 0 else 'FAILED'
        emoji = "✅" if exit_code == 0 else "❌"
        logger.info(f"{emoji} Launcher ended: {status} (exit code: {exit_code})")

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_unified_logger: Optional[UnifiedLogger] = None

def setup_unified_logging(config: LoggingConfig = None) -> UnifiedLogger:
    """Setup unified logging system."""
    global _unified_logger
    _unified_logger = UnifiedLogger(config)
    return _unified_logger

def get_unified_logger() -> Optional[UnifiedLogger]:
    """Get the unified logger instance."""
    return _unified_logger

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance from the unified system."""
    if _unified_logger is None:
        setup_unified_logging()
    return _unified_logger.get_logger(name)

def get_system_logger() -> logging.Logger:
    """Get the system logger."""
    if _unified_logger is None:
        setup_unified_logging()
    return _unified_logger.get_system_logger()

def get_component_logger(component_name: str) -> logging.Logger:
    """Get a component-specific logger."""
    if _unified_logger is None:
        setup_unified_logging()
    return _unified_logger.get_component_logger(component_name)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def log_metric(name: str, value: float):
    """Log a metric."""
    if _unified_logger:
        _unified_logger.log_metric(name, value)

def log_parameter(name: str, value: Any):
    """Log a parameter."""
    if _unified_logger:
        _unified_logger.log_parameter(name, value)

def log_artifact(name: str, path: str):
    """Log an artifact."""
    if _unified_logger:
        _unified_logger.log_artifact(name, path)

def log_step_start(step_name: str):
    """Log step start."""
    if _unified_logger:
        _unified_logger.log_step_start(step_name)

def log_step_end(step_name: str, success: bool = True):
    """Log step end."""
    if _unified_logger:
        _unified_logger.log_step_end(step_name, success)

def log_data_operation(operation: str, details: str = ""):
    """Log data operation."""
    if _unified_logger:
        _unified_logger.log_data_operation(operation, details)

def log_error(message: str, exc_info: bool = False):
    """Log an error."""
    if _unified_logger:
        _unified_logger.log_error(message, exc_info)

def log_warning(message: str):
    """Log a warning."""
    if _unified_logger:
        _unified_logger.log_warning(message)

def log_info(message: str):
    """Log info."""
    if _unified_logger:
        _unified_logger.log_info(message)

def log_debug(message: str):
    """Log debug."""
    if _unified_logger:
        _unified_logger.log_debug(message)

def log_success(message: str):
    """Log success."""
    if _unified_logger:
        _unified_logger.log_success(message)

# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def log_execution_time(operation_name: str, logger_name: str = 'Timing'):
    """Context manager to log execution time."""
    start_time = time.time()
    logger = get_logger(logger_name)
    logger.info(f"⏱️ Starting {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"⏱️ {operation_name} completed in {duration:.2f} seconds")

@contextmanager
def log_step_execution(step_name: str):
    """Context manager to log step execution."""
    log_step_start(step_name)
    try:
        yield
        log_step_end(step_name, success=True)
    except Exception as e:
        log_step_end(step_name, success=False)
        log_error(f"Step {step_name} failed: {str(e)}", exc_info=True)
        raise

# =============================================================================
# DECORATORS
# =============================================================================

def log_function_calls(func: Callable) -> Callable:
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger('FunctionCalls')
        logger.debug(f"🔧 Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} failed: {str(e)}")
            raise
    return wrapper

def log_important_calls(func: Callable) -> Callable:
    """Decorator to log important function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger('ImportantCalls')
        logger.info(f"⭐ Important call: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} failed: {str(e)}")
            raise
    return wrapper

def log_all_calls(func: Callable) -> Callable:
    """Decorator to log all function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger('AllCalls')
        logger.debug(f"🔧 {func.__name__} called")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"✅ {func.__name__} returned")
            return result
        except Exception as e:
            logger.error(f"❌ {func.__name__} raised exception: {str(e)}")
            raise
    return wrapper

def log_internal_call(func: Callable) -> Callable:
    """Decorator to log internal function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger('InternalCalls')
        logger.debug(f"🔧 Internal call: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"✅ Internal call {func.__name__} completed")
            return result
        except Exception as e:
            logger.error(f"❌ Internal call {func.__name__} failed: {str(e)}")
            raise
    return wrapper

def log_step_progress(func: Callable) -> Callable:
    """Decorator to log step progress."""
    def wrapper(*args, **kwargs):
        logger = get_logger('StepProgress')
        logger.info(f"📈 Step progress: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ Step {func.__name__} completed")
            return result
        except Exception as e:
            logger.error(f"❌ Step {func.__name__} failed: {str(e)}")
            raise
    return wrapper

def log_data_operation(func: Callable) -> Callable:
    """Decorator to log data operations."""
    def wrapper(*args, **kwargs):
        logger = get_logger('DataOperations')
        logger.info(f"🔄 Data operation: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ Data operation {func.__name__} completed")
            return result
        except Exception as e:
            logger.error(f"❌ Data operation {func.__name__} failed: {str(e)}")
            raise
    return wrapper

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize the unified logger by default
if _unified_logger is None:
    setup_unified_logging()

# Create a system logger instance for backward compatibility
system_logger = get_system_logger()