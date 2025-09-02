"""
Simple Error Handler Decorators for Core Training Modules.

This module provides basic error handling decorators that work without
external dependencies, allowing the core modules to function properly.
"""

import functools
import logging
import sys
from typing import Any, Callable, Type, Union, Tuple, Optional
from collections.abc import Awaitable


class SimpleLogger:
    """Simple logger implementation for core modules."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(handler)
    
    def getChild(self, name: str) -> 'SimpleLogger':
        """Get a child logger."""
        return SimpleLogger(f"{self.logger.name}.{name}")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str) -> None:
        """Log exception message with traceback."""
        self.logger.exception(message)


# Create a global system logger
system_logger = SimpleLogger("SystemLogger")


def handle_errors(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    default_return: Any = None,
    context: str = "unknown"
) -> Callable:
    """
    Decorator to handle exceptions with a default return value.
    
    Args:
        exceptions: Exception type(s) to catch
        default_return: Value to return if exception occurs
        context: Context for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                system_logger.error(f"Error in {context}: {e}")
                return default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                system_logger.error(f"Error in {context}: {e}")
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def handle_specific_errors(
    error_handlers: dict[Type[Exception], Tuple[Any, str]],
    default_return: Any = None,
    context: str = "unknown"
) -> Callable:
    """
    Decorator to handle specific exceptions with custom return values and messages.
    
    Args:
        error_handlers: Dict mapping exception types to (return_value, message)
        default_return: Default value to return if exception not in handlers
        context: Context for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we have a specific handler for this exception
                for exc_type, (return_value, message) in error_handlers.items():
                    if isinstance(e, exc_type):
                        system_logger.error(f"{message} in {context}: {e}")
                        return return_value
                
                # Use default handler
                system_logger.error(f"Unexpected error in {context}: {e}")
                return default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if we have a specific handler for this exception
                for exc_type, (return_value, message) in error_handlers.items():
                    if isinstance(e, exc_type):
                        system_logger.error(f"{message} in {context}: {e}")
                        return return_value
                
                # Use default handler
                system_logger.error(f"Unexpected error in {context}: {e}")
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


# Import asyncio for async function detection
import asyncio