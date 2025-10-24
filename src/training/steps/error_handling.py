"""
Comprehensive error handling utilities for training steps.

This module provides standardized error handling, logging, and recovery mechanisms
for all training step components.
"""

import traceback
from typing import Any, Dict, List, Optional, Callable, Type, Union
from functools import wraps
from datetime import datetime

from src.utils.tprint import (
    tprint_info, tprint_success, tprint_error, tprint_warning, tprint_debug,
    tprint_exception, LogLevel
)

from .step_types import (
    ExecutionResult, ValidationResult, MetricsDict,
    TrainingStepError, ValidationError, DataLoadError, ModelTrainingError,
    FeatureSelectionError, ConfigurationError, ArtifactError,
    create_error_result, create_success_result
)

# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def handle_errors(
    error_type: Type[Exception] = TrainingStepError,
    fallback_result: Any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """
    Decorator for comprehensive error handling in training steps.
    
    Args:
        error_type: Type of exception to catch and convert
        fallback_result: Result to return on error (if not reraise)
        log_errors: Whether to log errors
        reraise: Whether to reraise the exception after logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    tprint_error(f"❌ Error in {func.__name__}: {e}")
                    tprint_exception(e, f"Context: {func.__name__}")
                
                if reraise:
                    raise error_type(f"{func.__name__} failed: {e}") from e
                else:
                    return fallback_result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    tprint_error(f"❌ Error in {func.__name__}: {e}")
                    tprint_exception(e, f"Context: {func.__name__}")
                
                if reraise:
                    raise error_type(f"{func.__name__} failed: {e}") from e
                else:
                    return fallback_result
        
        # Return appropriate wrapper based on function type
        if func.__name__.startswith('_') or 'async' in str(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def validate_inputs(*validators: Callable[[Any], bool]):
    """
    Decorator to validate function inputs.
    
    Args:
        *validators: Validation functions that return True if valid
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            for i, validator in enumerate(validators):
                if i < len(args) and not validator(args[i]):
                    raise ValidationError(f"Input validation failed for argument {i}: {args[i]}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        tprint_warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        tprint_error(f"❌ All {max_retries + 1} attempts failed. Last error: {e}")
                        raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        tprint_warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s...")
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        tprint_error(f"❌ All {max_retries + 1} attempts failed. Last error: {e}")
                        raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if func.__name__.startswith('_') or 'async' in str(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ============================================================================
# ERROR RECOVERY UTILITIES
# ============================================================================

class ErrorRecoveryManager:
    """Manages error recovery strategies for training steps."""
    
    def __init__(self):
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.fallback_strategies: List[Callable] = []
    
    def register_recovery_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for a specific exception type."""
        self.recovery_strategies[exception_type] = strategy
        tprint_info(f"📝 Registered recovery strategy for {exception_type.__name__}")
    
    def register_fallback_strategy(self, strategy: Callable):
        """Register a fallback strategy for any unhandled exceptions."""
        self.fallback_strategies.append(strategy)
        tprint_info(f"📝 Registered fallback strategy: {strategy.__name__}")
    
    async def attempt_recovery(self, error: Exception, context: str = "") -> Any:
        """Attempt to recover from an error using registered strategies."""
        tprint_info(f"🔧 Attempting error recovery for {type(error).__name__} in {context}")
        
        # Try specific recovery strategies
        for exception_type, strategy in self.recovery_strategies.items():
            if isinstance(error, exception_type):
                try:
                    tprint_info(f"🔄 Trying recovery strategy for {exception_type.__name__}")
                    result = await strategy(error, context) if callable(strategy) else strategy
                    tprint_success(f"✅ Recovery successful for {exception_type.__name__}")
                    return result
                except Exception as recovery_error:
                    tprint_warning(f"⚠️ Recovery strategy failed: {recovery_error}")
                    continue
        
        # Try fallback strategies
        for strategy in self.fallback_strategies:
            try:
                tprint_info(f"🔄 Trying fallback strategy: {strategy.__name__}")
                result = await strategy(error, context) if callable(strategy) else strategy
                tprint_success(f"✅ Fallback strategy successful")
                return result
            except Exception as fallback_error:
                tprint_warning(f"⚠️ Fallback strategy failed: {fallback_error}")
                continue
        
        tprint_error(f"❌ No recovery strategies succeeded for {type(error).__name__}")
        return None

# Global error recovery manager
error_recovery_manager = ErrorRecoveryManager()

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dataframe(data: Any, required_columns: List[str] = None) -> bool:
    """Validate that data is a DataFrame with required columns."""
    try:
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            return False
        if data.empty:
            return False
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False
        return True
    except Exception:
        return False

def validate_series(data: Any) -> bool:
    """Validate that data is a Series."""
    try:
        import pandas as pd
        return isinstance(data, pd.Series) and not data.empty
    except Exception:
        return False

def validate_config_dict(config: Any) -> bool:
    """Validate that config is a valid configuration dictionary."""
    return isinstance(config, dict) and 'symbol' in config and 'exchange' in config

def validate_positive_number(value: Any) -> bool:
    """Validate that value is a positive number."""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False

# ============================================================================
# ERROR LOGGING UTILITIES
# ============================================================================

def log_error_with_context(
    error: Exception,
    context: str,
    additional_info: Dict[str, Any] = None,
    level: LogLevel = LogLevel.ERROR
):
    """Log an error with comprehensive context information."""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'timestamp': datetime.now().isoformat(),
        'traceback': traceback.format_exc()
    }
    
    if additional_info:
        error_info.update(additional_info)
    
    tprint_exception(error, f"Context: {context}")
    tprint_debug(f"Error details: {error_info}")

def create_error_metrics(error: Exception, context: str) -> MetricsDict:
    """Create metrics dictionary for error reporting."""
    return {
        'error_type': type(error).__name__,
        'error_context': context,
        'error_timestamp': datetime.now().isoformat(),
        'success': False
    }

# ============================================================================
# SAFE EXECUTION UTILITIES
# ============================================================================

async def safe_execute_async(
    func: Callable,
    *args,
    error_handler: Callable = None,
    fallback_result: Any = None,
    **kwargs
) -> Any:
    """Safely execute an async function with error handling."""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            return await error_handler(e, *args, **kwargs)
        else:
            tprint_error(f"❌ Safe execution failed: {e}")
            return fallback_result

def safe_execute_sync(
    func: Callable,
    *args,
    error_handler: Callable = None,
    fallback_result: Any = None,
    **kwargs
) -> Any:
    """Safely execute a sync function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            return error_handler(e, *args, **kwargs)
        else:
            tprint_error(f"❌ Safe execution failed: {e}")
            return fallback_result

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Decorators
    'handle_errors', 'validate_inputs', 'retry_on_failure',
    
    # Error recovery
    'ErrorRecoveryManager', 'error_recovery_manager',
    
    # Validation utilities
    'validate_dataframe', 'validate_series', 'validate_config_dict', 'validate_positive_number',
    
    # Error logging
    'log_error_with_context', 'create_error_metrics',
    
    # Safe execution
    'safe_execute_async', 'safe_execute_sync'
]