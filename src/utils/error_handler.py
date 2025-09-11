"""
Unified Error Handler - Enhanced Error Management System

This module provides comprehensive error handling, validation, and error prevention
functionality consolidated from multiple error handling utilities.
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Type
from contextlib import contextmanager
import warnings

# =============================================================================
# ERROR CLASSES
# =============================================================================

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class DataQualityError(Exception):
    """Exception raised for data quality issues."""
    pass

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

class ProcessingError(Exception):
    """Exception raised for processing errors."""
    pass

class MathValidationError(Exception):
    """Exception raised for math validation errors."""
    pass

# =============================================================================
# UNIFIED ERROR HANDLER
# =============================================================================

class UnifiedErrorHandler:
    """Unified error handler that consolidates all error handling functionality."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.error_history = []
    
    def handle_error(self, error: Exception, context: str = "", 
                    reraise: bool = True, log_level: int = logging.ERROR) -> Any:
        """Handle an error with comprehensive logging and tracking."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to error history
        self.error_history.append({
            'type': error_type,
            'message': error_message,
            'context': context,
            'traceback': traceback.format_exc()
        })
        
        # Log the error
        log_message = f"❌ Error in {context}: {error_type} - {error_message}"
        self.logger.log(log_level, log_message, exc_info=True)
        
        # Reraise if requested
        if reraise:
            raise error
        
        return None
    
    def safe_execute(self, func: Callable, *args, default: Any = None, 
                    context: str = "", **kwargs) -> Any:
        """Safely execute a function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context, reraise=False)
            return default
    
    def validate_not_none(self, value: Any, name: str = "value") -> Any:
        """Validate that a value is not None."""
        if value is None:
            error = ValidationError(f"{name} cannot be None")
            self.handle_error(error, f"Validation: {name}")
        return value
    
    def validate_not_empty(self, value: Union[str, List, Dict], name: str = "value") -> Any:
        """Validate that a value is not empty."""
        if not value:
            error = ValidationError(f"{name} cannot be empty")
            self.handle_error(error, f"Validation: {name}")
        return value
    
    def validate_range(self, value: float, min_val: float = None, max_val: float = None, 
                      name: str = "value") -> float:
        """Validate that a value is in range."""
        if min_val is not None and value < min_val:
            error = ValidationError(f"{name} must be >= {min_val}, got {value}")
            self.handle_error(error, f"Validation: {name}")
        if max_val is not None and value > max_val:
            error = ValidationError(f"{name} must be <= {max_val}, got {value}")
            self.handle_error(error, f"Validation: {name}")
        return value
    
    def validate_positive(self, value: float, name: str = "value") -> float:
        """Validate that a value is positive."""
        if value <= 0:
            error = ValidationError(f"{name} must be positive, got {value}")
            self.handle_error(error, f"Validation: {name}")
        return value
    
    def validate_finite(self, value: Any, name: str = "value") -> float:
        """Validate that a value is finite."""
        try:
            val = float(value)
            if not (val == val and val != float('inf') and val != float('-inf')):
                error = MathValidationError(f"{name} must be finite, got {val}")
                self.handle_error(error, f"Validation: {name}")
            return val
        except (ValueError, TypeError) as e:
            error = MathValidationError(f"Invalid {name}: {e}")
            self.handle_error(error, f"Validation: {name}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of errors encountered."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_error_history(self):
        """Clear error history and counts."""
        self.error_counts.clear()
        self.error_history.clear()

# =============================================================================
# DECORATORS
# =============================================================================

def handles_errors(default_return: Any = None, context: str = "", 
                  log_level: int = logging.ERROR):
    """Decorator to handle errors in functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = UnifiedErrorHandler()
                handler.handle_error(e, context or func.__name__, reraise=False, log_level=log_level)
                return default_return
        return wrapper
    return decorator

def validates_inputs(*validators: Callable):
    """Decorator to validate function inputs."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validators to args
            for i, validator in enumerate(validators):
                if i < len(args):
                    validator(args[i])
            return func(*args, **kwargs)
        return wrapper
    return decorator

def safe_execution(default_return: Any = None, context: str = ""):
    """Decorator for safe execution with error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = UnifiedErrorHandler()
            return handler.safe_execute(func, *args, default=default_return, 
                                      context=context or func.__name__, **kwargs)
        return wrapper
    return decorator

def error_prevention(func: Callable) -> Callable:
    """Decorator to prevent errors through validation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Basic validation
            if not args:
                raise ValidationError("Function requires at least one argument")
            
            return func(*args, **kwargs)
        except Exception as e:
            handler = UnifiedErrorHandler()
            handler.handle_error(e, f"Error prevention in {func.__name__}", reraise=True)
    return wrapper

# =============================================================================
# CONTEXT MANAGERS
# =============================================================================

@contextmanager
def error_context(context_name: str, handler: UnifiedErrorHandler = None):
    """Context manager for error handling."""
    if handler is None:
        handler = UnifiedErrorHandler()
    
    try:
        yield handler
    except Exception as e:
        handler.handle_error(e, context_name, reraise=True)

@contextmanager
def safe_context(default_return: Any = None, context_name: str = ""):
    """Context manager for safe execution."""
    handler = UnifiedErrorHandler()
    try:
        yield handler
    except Exception as e:
        handler.handle_error(e, context_name, reraise=False)
        return default_return

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class DataValidator:
    """Data validation utilities."""
    
    def __init__(self, handler: UnifiedErrorHandler = None):
        self.handler = handler or UnifiedErrorHandler()
    
    def validate_dataframe(self, df, required_columns: List[str] = None, 
                          min_rows: int = 0) -> bool:
        """Validate DataFrame structure and content."""
        try:
            if df is None:
                raise DataQualityError("DataFrame cannot be None")
            
            if df.empty:
                raise DataQualityError("DataFrame cannot be empty")
            
            if min_rows > 0 and len(df) < min_rows:
                raise DataQualityError(f"DataFrame must have at least {min_rows} rows")
            
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise DataQualityError(f"Missing required columns: {missing_columns}")
            
            return True
        except Exception as e:
            self.handler.handle_error(e, "DataFrame validation")
            return False
    
    def validate_numeric_data(self, data, name: str = "data") -> bool:
        """Validate numeric data."""
        try:
            import numpy as np
            if not np.isfinite(data).all():
                raise DataQualityError(f"{name} contains non-finite values")
            return True
        except Exception as e:
            self.handler.handle_error(e, f"Numeric data validation: {name}")
            return False
    
    def validate_timestamp_data(self, timestamps, name: str = "timestamps") -> bool:
        """Validate timestamp data."""
        try:
            import pandas as pd
            pd.to_datetime(timestamps)
            return True
        except Exception as e:
            self.handler.handle_error(e, f"Timestamp validation: {name}")
            return False

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_unified_error_handler: Optional[UnifiedErrorHandler] = None

def get_unified_error_handler() -> UnifiedErrorHandler:
    """Get the global unified error handler."""
    global _unified_error_handler
    if _unified_error_handler is None:
        _unified_error_handler = UnifiedErrorHandler()
    return _unified_error_handler

def setup_unified_error_handling(logger: logging.Logger = None) -> UnifiedErrorHandler:
    """Setup unified error handling."""
    global _unified_error_handler
    _unified_error_handler = UnifiedErrorHandler(logger)
    return _unified_error_handler

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def safe_execute(func: Callable, *args, default: Any = None, 
                context: str = "", **kwargs) -> Any:
    """Safely execute a function."""
    handler = get_unified_error_handler()
    return handler.safe_execute(func, *args, default=default, 
                              context=context, **kwargs)

def validate_not_none(value: Any, name: str = "value") -> Any:
    """Validate that a value is not None."""
    handler = get_unified_error_handler()
    return handler.validate_not_none(value, name)

def validate_not_empty(value: Union[str, List, Dict], name: str = "value") -> Any:
    """Validate that a value is not empty."""
    handler = get_unified_error_handler()
    return handler.validate_not_empty(value, name)

def validate_range(value: float, min_val: float = None, max_val: float = None, 
                  name: str = "value") -> float:
    """Validate that a value is in range."""
    handler = get_unified_error_handler()
    return handler.validate_range(value, min_val, max_val, name)

def validate_positive(value: float, name: str = "value") -> float:
    """Validate that a value is positive."""
    handler = get_unified_error_handler()
    return handler.validate_positive(value, name)

def validate_finite(value: Any, name: str = "value") -> float:
    """Validate that a value is finite."""
    handler = get_unified_error_handler()
    return handler.validate_finite(value, name)

# =============================================================================
# INITIALIZATION
# =============================================================================

# Initialize the unified error handler by default
if _unified_error_handler is None:
    setup_unified_error_handling()