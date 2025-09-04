#!/usr/bin/env python3
"""S/R Error Handlers.

This module provides specialized error handling for S/R detection and optimization operations.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Tuple
from functools import wraps
import asyncio
from datetime import datetime

from src.utils.logger import system_logger


class SRError(Exception):
    """Base exception for S/R related errors."""
    pass


class SRConfigurationError(SRError):
    """Configuration-related S/R errors."""
    pass


class SRDataError(SRError):
    """Data-related S/R errors."""
    pass


class SROptimizationError(SRError):
    """Optimization-related S/R errors."""
    pass


class SRValidationError(SRError):
    """Validation-related S/R errors."""
    pass


class SRCacheError(SRError):
    """Cache-related S/R errors."""
    pass


class SRErrorHandler:
    """Specialized error handler for S/R operations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler."""
        self.logger = logger or system_logger.getChild("SRErrorHandler")
        self.error_counts: Dict[str, int] = {}
        self.max_consecutive_errors = 5
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
    
    def handle_error(
        self,
        error: Exception,
        context: str = "S/R operation",
        default_return: Any = None,
        reraise: bool = False
    ) -> Any:
        """Handle S/R related errors with context-aware logging."""
        try:
            # Update error tracking
            self._update_error_tracking(error, context)
            
            # Log error with context
            self._log_error(error, context)
            
            # Check if we should continue or stop
            if self._should_stop_processing():
                self.logger.critical(f"Too many consecutive errors in {context}, stopping processing")
                if reraise:
                    raise error
                return default_return
            
            # Return default value or reraise
            if reraise:
                raise error
            return default_return
            
        except Exception as handler_error:
            self.logger.critical(f"Error handler itself failed: {handler_error}")
            if reraise:
                raise error
            return default_return
    
    def _update_error_tracking(self, error: Exception, context: str) -> None:
        """Update error tracking statistics."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Track consecutive errors
        current_time = datetime.now()
        if (self.last_error_time is None or 
            (current_time - self.last_error_time).total_seconds() > 60):
            self.consecutive_errors = 1
        else:
            self.consecutive_errors += 1
        
        self.last_error_time = current_time
    
    def _log_error(self, error: Exception, context: str) -> None:
        """Log error with appropriate level and context."""
        error_type = type(error).__name__
        
        # Determine log level based on error type and frequency
        if isinstance(error, (SRConfigurationError, SRDataError)):
            log_level = logging.ERROR
        elif isinstance(error, (SROptimizationError, SRValidationError)):
            log_level = logging.WARNING
        elif isinstance(error, SRCacheError):
            log_level = logging.INFO
        else:
            log_level = logging.ERROR
        
        # Create detailed error message
        error_msg = f"S/R Error in {context}: {error_type}: {str(error)}"
        
        # Add context information
        if hasattr(error, '__traceback__'):
            tb_lines = traceback.format_tb(error.__traceback__)
            if tb_lines:
                error_msg += f"\nTraceback: {''.join(tb_lines[-3:])}"  # Last 3 lines
        
        # Log with appropriate level
        self.logger.log(log_level, error_msg)
        
        # Log error statistics
        if self.error_counts[error_type] % 10 == 0:  # Every 10th occurrence
            self.logger.warning(f"Error statistics - {error_type}: {self.error_counts[error_type]} occurrences")
    
    def _should_stop_processing(self) -> bool:
        """Determine if processing should stop due to too many errors."""
        return self.consecutive_errors >= self.max_consecutive_errors
    
    def reset_error_tracking(self) -> None:
        """Reset error tracking statistics."""
        self.error_counts.clear()
        self.consecutive_errors = 0
        self.last_error_time = None
        self.logger.info("S/R error tracking reset")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "consecutive_errors": self.consecutive_errors,
            "last_error_time": self.last_error_time,
            "total_errors": sum(self.error_counts.values())
        }


def sr_error_handler(
    exceptions: Tuple[type, ...] = (Exception,),
    default_return: Any = None,
    context: str = "S/R operation",
    reraise: bool = False,
    max_retries: int = 0,
    retry_delay: float = 1.0
):
    """Decorator for S/R error handling with retry logic."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = SRErrorHandler()
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        error_handler.logger.warning(
                            f"Attempt {attempt + 1} failed in {context}, retrying in {retry_delay}s: {e}"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        return error_handler.handle_error(e, context, default_return, reraise)
            
            # This should never be reached, but just in case
            if last_error:
                return error_handler.handle_error(last_error, context, default_return, reraise)
            return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            error_handler = SRErrorHandler()
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        error_handler.logger.warning(
                            f"Attempt {attempt + 1} failed in {context}, retrying in {retry_delay}s: {e}"
                        )
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        return error_handler.handle_error(e, context, default_return, reraise)
            
            # This should never be reached, but just in case
            if last_error:
                return error_handler.handle_error(last_error, context, default_return, reraise)
            return default_return
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_sr_data(data: Any, required_columns: Optional[list] = None) -> None:
    """Validate S/R data with detailed error messages."""
    if data is None:
        raise SRDataError("S/R data is None")
    
    if not hasattr(data, '__len__'):
        raise SRDataError("S/R data must be iterable")
    
    if len(data) == 0:
        raise SRDataError("S/R data is empty")
    
    if hasattr(data, 'columns') and required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise SRDataError(f"Missing required columns: {missing_columns}")
    
    if hasattr(data, 'isnull'):
        null_count = data.isnull().sum().sum()
        if null_count > 0:
            raise SRDataError(f"S/R data contains {null_count} null values")


def validate_sr_parameters(params: Dict[str, Any], required_params: Optional[list] = None) -> None:
    """Validate S/R parameters with detailed error messages."""
    if not isinstance(params, dict):
        raise SRConfigurationError("S/R parameters must be a dictionary")
    
    if required_params:
        missing_params = set(required_params) - set(params.keys())
        if missing_params:
            raise SRConfigurationError(f"Missing required parameters: {missing_params}")
    
    # Validate parameter ranges
    for param_name, param_value in params.items():
        if param_name.endswith('_threshold') and not (0 <= param_value <= 1):
            raise SRConfigurationError(f"Threshold parameter {param_name} must be between 0 and 1, got {param_value}")
        
        if param_name.endswith('_weight') and not (0 <= param_value <= 1):
            raise SRConfigurationError(f"Weight parameter {param_name} must be between 0 and 1, got {param_value}")
        
        if param_name.endswith('_period') and not (param_value > 0 and isinstance(param_value, int)):
            raise SRConfigurationError(f"Period parameter {param_name} must be a positive integer, got {param_value}")


def validate_sr_levels(levels: list, min_levels: int = 1) -> None:
    """Validate S/R levels with detailed error messages."""
    if not isinstance(levels, list):
        raise SRValidationError("S/R levels must be a list")
    
    if len(levels) < min_levels:
        raise SRValidationError(f"Expected at least {min_levels} S/R levels, got {len(levels)}")
    
    for i, level in enumerate(levels):
        if not isinstance(level, dict):
            raise SRValidationError(f"S/R level {i} must be a dictionary")
        
        required_fields = ['price', 'type', 'strength']
        missing_fields = set(required_fields) - set(level.keys())
        if missing_fields:
            raise SRValidationError(f"S/R level {i} missing required fields: {missing_fields}")
        
        if not isinstance(level['price'], (int, float)) or level['price'] <= 0:
            raise SRValidationError(f"S/R level {i} price must be a positive number, got {level['price']}")
        
        if level['type'] not in ['support', 'resistance']:
            raise SRValidationError(f"S/R level {i} type must be 'support' or 'resistance', got {level['type']}")
        
        if not isinstance(level['strength'], (int, float)) or not (0 <= level['strength'] <= 1):
            raise SRValidationError(f"S/R level {i} strength must be between 0 and 1, got {level['strength']}")


# Global error handler instance
_global_error_handler: Optional[SRErrorHandler] = None


def get_global_sr_error_handler() -> SRErrorHandler:
    """Get global S/R error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = SRErrorHandler()
    return _global_error_handler


def handle_sr_error(
    error: Exception,
    context: str = "S/R operation",
    default_return: Any = None,
    reraise: bool = False
) -> Any:
    """Handle S/R error using global error handler."""
    handler = get_global_sr_error_handler()
    return handler.handle_error(error, context, default_return, reraise)