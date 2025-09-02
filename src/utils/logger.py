"""Enhanced logger module with both mock and production logging capabilities.

This module provides a MockLogger for testing purposes and a production Logger
class that integrates with Python's standard logging module.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime


class MockLogger:
    """Mock logger that provides basic logging functionality for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.handlers = []
        self.level = logging.INFO
    
    def getChild(self, name: str) -> 'MockLogger':
        """Get a child logger."""
        return MockLogger(f"{self.name}.{name}")
    
    def info(self, message: str) -> None:
        """Log info message."""
        print(f"[INFO] {self.name}: {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        print(f"[ERROR] {self.name}: {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        print(f"[WARNING] {self.name}: {message}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        print(f"[DEBUG] {self.name}: {message}")
    
    def exception(self, message: str) -> None:
        """Log exception message."""
        print(f"[EXCEPTION] {self.name}: {message}")
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        print(f"[CRITICAL] {self.name}: {message}")


class Logger:
    """Production-ready logger with enhanced features."""
    
    def __init__(self, name: str, level: int = logging.INFO, 
                 format_string: Optional[str] = None, 
                 log_to_file: bool = False, 
                 log_file_path: Optional[str] = None):
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level
            format_string: Custom format string for log messages
            log_to_file: Whether to log to file
            log_file_path: Path to log file
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(format_string, log_to_file, log_file_path)
    
    def _setup_handlers(self, format_string: Optional[str], 
                        log_to_file: bool, log_file_path: Optional[str]):
        """Set up logging handlers."""
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if requested
        if log_to_file and log_file_path:
            try:
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not create file handler: {e}")
    
    def getChild(self, name: str) -> 'Logger':
        """Get a child logger."""
        return Logger(f"{self.name}.{name}")
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception message."""
        self.logger.exception(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def setLevel(self, level: int) -> None:
        """Set logging level."""
        self.logger.setLevel(level)
    
    def addHandler(self, handler: logging.Handler) -> None:
        """Add a logging handler."""
        self.logger.addHandler(handler)
    
    def removeHandler(self, handler: logging.Handler) -> None:
        """Remove a logging handler."""
        self.logger.removeHandler(handler)


class StructuredLogger(Logger):
    """Logger that supports structured logging with JSON-like output."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Bind context variables to this logger."""
        new_logger = StructuredLogger(self.name)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def _format_structured_message(self, message: str, level: str, **kwargs) -> str:
        """Format message with structured data."""
        structured_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'context': self._context,
            **kwargs
        }
        
        # Simple JSON-like formatting (for production, use json.dumps)
        parts = []
        for key, value in structured_data.items():
            if value is not None:
                parts.append(f'{key}="{value}"')
        
        return f"[{', '.join(parts)}]"
    
    def info(self, message: str, **kwargs) -> None:
        """Log structured info message."""
        formatted_message = self._format_structured_message(message, 'INFO', **kwargs)
        super().info(formatted_message)
    
    def error(self, message: str, **kwargs) -> None:
        """Log structured error message."""
        formatted_message = self._format_structured_message(message, 'ERROR', **kwargs)
        super().error(formatted_message)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log structured warning message."""
        formatted_message = self._format_structured_message(message, 'WARNING', **kwargs)
        super().warning(formatted_message)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log structured debug message."""
        formatted_message = self._format_structured_message(message, 'DEBUG', **kwargs)
        super().debug(formatted_message)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log structured exception message."""
        formatted_message = self._format_structured_message(message, 'EXCEPTION', **kwargs)
        super().exception(formatted_message)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log structured critical message."""
        formatted_message = self._format_structured_message(message, 'CRITICAL', **kwargs)
        super().critical(formatted_message)


# Create default logger instances
system_logger = MockLogger("System")
production_logger = Logger("System")
structured_logger = StructuredLogger("System")

# Convenience function to get a logger
def get_logger(name: str, logger_type: str = "standard") -> Any:
    """Get a logger instance.
    
    Args:
        name: Logger name
        logger_type: Type of logger ("mock", "standard", or "structured")
    
    Returns:
        Logger instance
    """
    if logger_type == "mock":
        return MockLogger(name)
    elif logger_type == "structured":
        return StructuredLogger(name)
    else:
        return Logger(name)