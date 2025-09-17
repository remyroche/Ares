"""
Utility functions for HMM Models Training

Standardized utilities to reduce code duplication and improve consistency.
"""

import functools
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from .constants import LoggingConstants

# Type variable for generic functions
T = TypeVar('T')

logger = system_logger.getChild('HMMTrainingUtils')


class StandardizedLogger:
    """Standardized logging utility for consistent messaging."""
    
    @staticmethod
    def log_initialization(component_name: str, success: bool, details: str = "") -> None:
        """Log component initialization."""
        if success:
            tprint(f"{LoggingConstants.SUCCESS_INDICATOR} {component_name} {LoggingConstants.INITIALIZATION_SUCCESS}")
            if details:
                logger.info(f"{component_name} initialized: {details}")
        else:
            tprint(f"{LoggingConstants.ERROR_INDICATOR} {component_name} {LoggingConstants.INITIALIZATION_FAILED}")
            if details:
                logger.error(f"{component_name} initialization failed: {details}")
    
    @staticmethod
    def log_validation(operation: str, success: bool, message: str = "") -> None:
        """Log validation results."""
        if success:
            tprint(f"{LoggingConstants.SUCCESS_INDICATOR} {operation} {LoggingConstants.VALIDATION_PASSED}")
            if message:
                logger.info(f"{operation}: {message}")
        else:
            tprint(f"{LoggingConstants.ERROR_INDICATOR} {operation} {LoggingConstants.VALIDATION_FAILED}")
            if message:
                logger.error(f"{operation}: {message}")
    
    @staticmethod
    def log_training_progress(model_name: str, stage: str, success: bool = True, details: str = "") -> None:
        """Log training progress."""
        if success:
            indicator = LoggingConstants.PROGRESS_INDICATOR if stage == "started" else LoggingConstants.SUCCESS_INDICATOR
            message = f"{indicator} {model_name} training {stage}"
            tprint(message)
            if details:
                logger.info(f"{model_name}: {details}")
        else:
            tprint(f"{LoggingConstants.ERROR_INDICATOR} {model_name} training failed at {stage}")
            if details:
                logger.error(f"{model_name}: {details}")
    
    @staticmethod
    def log_warning(component: str, message: str) -> None:
        """Log warning message."""
        tprint(f"{LoggingConstants.WARNING_INDICATOR} {component}: {message}")
        logger.warning(f"{component}: {message}")
    
    @staticmethod
    def log_error(component: str, error: Union[str, Exception], include_traceback: bool = False) -> None:
        """Log error message."""
        error_msg = str(error)
        tprint(f"{LoggingConstants.ERROR_INDICATOR} {component}: {error_msg}")
        
        if include_traceback and isinstance(error, Exception):
            tb = traceback.format_exc()
            logger.error(f"{component}: {error_msg}\nTraceback: {tb}")
        else:
            logger.error(f"{component}: {error_msg}")
    
    @staticmethod
    def log_info(component: str, message: str) -> None:
        """Log info message."""
        tprint(f"{LoggingConstants.INFO_INDICATOR} {component}: {message}")
        logger.info(f"{component}: {message}")


def safe_execute(func: Callable[..., T], *args, default_return: T = None, 
                 component_name: str = "operation", log_errors: bool = True, **kwargs) -> T:
    """
    Safely execute a function with standardized error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        component_name: Name for logging
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            StandardizedLogger.log_error(component_name, e, include_traceback=True)
        return default_return


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    exponential_backoff: bool = True):
    """
    Decorator for retrying operations on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        exponential_backoff: Whether to use exponential backoff
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        StandardizedLogger.log_warning(
                            func.__name__, 
                            f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        StandardizedLogger.log_error(
                            func.__name__, 
                            f"All {max_attempts} attempts failed. Last error: {e}",
                            include_traceback=True
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


@contextmanager
def performance_monitor(operation_name: str, log_results: bool = True):
    """
    Context manager for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation
        log_results: Whether to log performance results
    """
    start_time = time.time()
    start_memory = None
    
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass
    
    try:
        yield
        success = True
    except Exception as e:
        success = False
        raise
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        
        if log_results:
            memory_info = ""
            if start_memory:
                try:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                    memory_info = f", Memory: {memory_delta:+.1f}MB"
                except:
                    pass
            
            status = "completed" if success else "failed"
            StandardizedLogger.log_info(
                operation_name, 
                f"Operation {status} in {execution_time:.2f}s{memory_info}"
            )


def validate_and_convert_types(data: Dict[str, Any], expected_types: Dict[str, type], 
                              component_name: str = "data_validation") -> Dict[str, Any]:
    """
    Validate and convert data types with standardized error handling.
    
    Args:
        data: Data to validate
        expected_types: Expected types for each key
        component_name: Component name for logging
        
    Returns:
        Validated and converted data
    """
    validated_data = {}
    errors = []
    
    for key, expected_type in expected_types.items():
        if key not in data:
            errors.append(f"Missing required key: {key}")
            continue
        
        value = data[key]
        
        try:
            if expected_type == int:
                validated_data[key] = int(float(value))  # Handle string numbers
            elif expected_type == float:
                validated_data[key] = float(value)
            elif expected_type == str:
                validated_data[key] = str(value)
            elif expected_type == bool:
                if isinstance(value, str):
                    validated_data[key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    validated_data[key] = bool(value)
            else:
                if not isinstance(value, expected_type):
                    errors.append(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}")
                else:
                    validated_data[key] = value
        except (ValueError, TypeError) as e:
            errors.append(f"Type conversion failed for {key}: {e}")
    
    if errors:
        error_msg = f"Validation failed: {'; '.join(errors)}"
        StandardizedLogger.log_error(component_name, error_msg)
        raise ValueError(error_msg)
    
    return validated_data


def safe_dictionary_access(data: Dict[str, Any], keys: List[str], 
                          default: Any = None, component_name: str = "dict_access") -> Any:
    """
    Safely access nested dictionary values.
    
    Args:
        data: Dictionary to access
        keys: List of keys for nested access
        default: Default value if access fails
        component_name: Component name for logging
        
    Returns:
        Value from dictionary or default
    """
    try:
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except Exception as e:
        StandardizedLogger.log_warning(component_name, f"Dictionary access failed: {e}")
        return default


def batch_process(items: List[Any], processor: Callable[[Any], Any], 
                 batch_size: int = 100, component_name: str = "batch_processing") -> List[Any]:
    """
    Process items in batches with error handling.
    
    Args:
        items: Items to process
        processor: Processing function
        batch_size: Size of each batch
        component_name: Component name for logging
        
    Returns:
        List of processed results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch_num = i // batch_size + 1
        batch = items[i:i + batch_size]
        
        try:
            with performance_monitor(f"{component_name}_batch_{batch_num}", log_results=False):
                batch_results = [processor(item) for item in batch]
                results.extend(batch_results)
                
            StandardizedLogger.log_info(
                component_name, 
                f"Processed batch {batch_num}/{total_batches} ({len(batch)} items)"
            )
            
        except Exception as e:
            StandardizedLogger.log_error(
                component_name, 
                f"Batch {batch_num} processing failed: {e}",
                include_traceback=True
            )
            # Continue with next batch instead of failing completely
            results.extend([None] * len(batch))
    
    return results


class ConfigurationValidator:
    """Standardized configuration validation utility."""
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Union[int, float], 
                             max_val: Union[int, float], param_name: str) -> bool:
        """Validate numeric parameter is within range."""
        try:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{param_name} must be numeric, got {type(value).__name__}")
            
            if value < min_val or value > max_val:
                raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")
            
            return True
        except Exception as e:
            StandardizedLogger.log_error("ConfigValidator", f"Range validation failed for {param_name}: {e}")
            return False
    
    @staticmethod
    def validate_choices(value: Any, valid_choices: List[Any], param_name: str) -> bool:
        """Validate parameter is one of valid choices."""
        try:
            if value not in valid_choices:
                raise ValueError(f"{param_name} must be one of {valid_choices}, got {value}")
            return True
        except Exception as e:
            StandardizedLogger.log_error("ConfigValidator", f"Choice validation failed for {param_name}: {e}")
            return False
    
    @staticmethod
    def validate_list_not_empty(value: List[Any], param_name: str) -> bool:
        """Validate list parameter is not empty."""
        try:
            if not isinstance(value, list):
                raise ValueError(f"{param_name} must be a list, got {type(value).__name__}")
            
            if len(value) == 0:
                raise ValueError(f"{param_name} cannot be empty")
            
            return True
        except Exception as e:
            StandardizedLogger.log_error("ConfigValidator", f"List validation failed for {param_name}: {e}")
            return False