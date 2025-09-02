# src/utils/simple_decorators.py

import asyncio
import functools
import time
import logging
from enum import Enum
from typing import Any, Callable, Optional

from src.utils.logger import system_logger


class PerformanceLevel(Enum):
    """Performance monitoring levels."""
    BASIC = "basic"
    MEDIUM = "medium"
    HIGH = "high"
    DETAILED = "detailed"


def performance_monitor(level: PerformanceLevel = PerformanceLevel.BASIC):
    """Simple performance monitoring decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = system_logger.getChild(f"PerformanceMonitor.{func.__name__}")
            
            try:
                if level in [PerformanceLevel.HIGH, PerformanceLevel.DETAILED]:
                    logger.debug(f"🚀 Starting {func.__name__} (level: {level.value})")
                
                result = await func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                if level in [PerformanceLevel.MEDIUM, PerformanceLevel.HIGH, PerformanceLevel.DETAILED]:
                    logger.debug(f"✅ {func.__name__} completed in {execution_time:.4f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = system_logger.getChild(f"PerformanceMonitor.{func.__name__}")
            
            try:
                if level in [PerformanceLevel.HIGH, PerformanceLevel.DETAILED]:
                    logger.debug(f"🚀 Starting {func.__name__} (level: {level.value})")
                
                result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                if level in [PerformanceLevel.MEDIUM, PerformanceLevel.HIGH, PerformanceLevel.DETAILED]:
                    logger.debug(f"✅ {func.__name__} completed in {execution_time:.4f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_errors(exceptions: tuple = (Exception,), default_return: Any = None, context: str = ""):
    """Simple error handling decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                logger = system_logger.getChild(f"ErrorHandler.{func.__name__}")
                logger.error(f"Error in {context or func.__name__}: {e}")
                return default_return
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger = system_logger.getChild(f"ErrorHandler.{func.__name__}")
                logger.error(f"Error in {context or func.__name__}: {e}")
                return default_return
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_specific_errors(error_handlers: dict, default_return: Any = None, context: str = ""):
    """Specific error handling decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger = system_logger.getChild(f"ErrorHandler.{func.__name__}")
                
                # Check for specific error handlers
                for exception_type, (return_value, message) in error_handlers.items():
                    if isinstance(e, exception_type):
                        logger.error(f"{message} in {context or func.__name__}: {e}")
                        return return_value
                
                # Default error handling
                logger.error(f"Unexpected error in {context or func.__name__}: {e}")
                return default_return
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = system_logger.getChild(f"ErrorHandler.{func.__name__}")
                
                # Check for specific error handlers
                for exception_type, (return_value, message) in error_handlers.items():
                    if isinstance(e, exception_type):
                        logger.error(f"{message} in {context or func.__name__}: {e}")
                        return return_value
                
                # Default error handling
                logger.error(f"Unexpected error in {context or func.__name__}: {e}")
                return default_return
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator