"""
Consistent Error Handling for ML Model Trainer

This module provides standardized error handling, logging, and graceful degradation
for the ML model training pipeline.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import numpy as np
import pandas as pd

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_exception, tprint_data_format, tprint_data_preview, LogLevel
)

logger = logging.getLogger(__name__)

class MLModelTrainerError(Exception):
    """Base exception for ML Model Trainer errors."""
    pass

class ConfigurationError(MLModelTrainerError):
    """Configuration-related errors."""
    pass

class DataValidationError(MLModelTrainerError):
    """Data validation errors."""
    pass

class ModelTrainingError(MLModelTrainerError):
    """Model training errors."""
    pass

class PredictionError(MLModelTrainerError):
    """Prediction errors."""
    pass

class ResourceError(MLModelTrainerError):
    """Resource-related errors (memory, GPU, etc.)."""
    pass

def handle_errors(
    error_type: type = MLModelTrainerError,
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = False
):
    """
    Decorator for consistent error handling.
    
    Args:
        error_type: Type of exception to catch
        default_return: Value to return on error
        log_level: Logging level for errors
        reraise: Whether to reraise the exception after logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                tprint_error(f"Error in {func.__name__}: {str(e)}")
                tprint_debug(f"Full traceback: {traceback.format_exc()}")
                logger.log(log_level, f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                return default_return
            except Exception as e:
                tprint_exception(e, f"Unexpected error in {func.__name__}")
                tprint_debug(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise MLModelTrainerError(f"Unexpected error in {func.__name__}: {str(e)}")
                return default_return
        return wrapper
    return decorator

def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, raises ConfigurationError if not
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        tprint_error(f"Missing required configuration keys: {missing_keys}")
        tprint_data_format(config, "Configuration being validated", LogLevel.DEBUG)
        raise ConfigurationError(f"Missing required configuration keys: {missing_keys}")
    
    tprint_debug(f"Configuration validation passed for keys: {required_keys}")
    return True

def validate_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    min_samples: int = 1,
    min_features: int = 1
) -> bool:
    """
    Validate input data.
    
    Args:
        X: Feature data
        y: Target data (optional)
        min_samples: Minimum number of samples
        min_features: Minimum number of features
        
    Returns:
        True if valid, raises DataValidationError if not
    """
    if X is None:
        tprint_error("Feature data X cannot be None")
        raise DataValidationError("Feature data X cannot be None")
    
    # Log data format information
    tprint_data_format(X, "Input features X", LogLevel.DEBUG)
    if y is not None:
        tprint_data_format(y, "Target data y", LogLevel.DEBUG)
    
    if hasattr(X, 'shape'):
        if X.shape[0] < min_samples:
            tprint_error(f"Insufficient samples: {X.shape[0]} < {min_samples}")
            raise DataValidationError(f"Insufficient samples: {X.shape[0]} < {min_samples}")
        if X.shape[1] < min_features:
            tprint_error(f"Insufficient features: {X.shape[1]} < {min_features}")
            raise DataValidationError(f"Insufficient features: {X.shape[1]} < {min_features}")
    
    if y is not None:
        if hasattr(y, 'shape') and len(y) != len(X):
            tprint_error(f"Target length {len(y)} doesn't match feature length {len(X)}")
            raise DataValidationError(f"Target length {len(y)} doesn't match feature length {len(X)}")
    
    tprint_success(f"Data validation passed: {X.shape[0]} samples, {X.shape[1]} features")
    return True

def safe_import(module_name: str, package_name: str = None, fallback: Any = None):
    """
    Safely import a module with fallback.
    
    Args:
        module_name: Name of module to import
        package_name: Package name for pip install suggestion
        fallback: Fallback value if import fails
        
    Returns:
        Imported module or fallback value
    """
    try:
        tprint_debug(f"Importing module: {module_name}")
        return __import__(module_name)
    except ImportError as e:
        tprint_warning(f"Failed to import {module_name}: {e}")
        if package_name:
            tprint_info(f"Install with: pip install {package_name}")
        logger.warning(f"Failed to import {module_name}: {e}")
        if package_name:
            logger.info(f"Install with: pip install {package_name}")
        return fallback

def check_memory_usage(threshold_gb: float = 8.0) -> bool:
    """
    Check if memory usage is within threshold.
    
    Args:
        threshold_gb: Memory threshold in GB
        
    Returns:
        True if within threshold, raises ResourceError if not
    """
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        used_gb = memory_info.used / (1024**3)
        
        tprint_data_format({
            "used_gb": used_gb,
            "threshold_gb": threshold_gb,
            "available_gb": memory_info.available / (1024**3),
            "total_gb": memory_info.total / (1024**3)
        }, "Memory usage check", LogLevel.DEBUG)
        
        if used_gb > threshold_gb:
            tprint_error(f"Memory usage {used_gb:.2f}GB exceeds threshold {threshold_gb}GB")
            raise ResourceError(f"Memory usage {used_gb:.2f}GB exceeds threshold {threshold_gb}GB")
        
        tprint_success(f"Memory usage {used_gb:.2f}GB within threshold {threshold_gb}GB")
        return True
    except ImportError:
        tprint_warning("psutil not available, skipping memory check")
        logger.warning("psutil not available, skipping memory check")
        return True

def graceful_degradation(
    primary_func: Callable,
    fallback_func: Callable,
    error_types: tuple = (Exception,)
):
    """
    Implement graceful degradation with fallback function.
    
    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        error_types: Types of errors to catch
        
    Returns:
        Result from primary or fallback function
    """
    try:
        tprint_debug("Attempting primary function")
        result = primary_func()
        tprint_success("Primary function completed successfully")
        return result
    except error_types as e:
        tprint_warning(f"Primary function failed: {e}, using fallback")
        logger.warning(f"Primary function failed: {e}, using fallback")
        tprint_debug("Attempting fallback function")
        result = fallback_func()
        tprint_success("Fallback function completed successfully")
        return result

class ErrorContext:
    """Context manager for error handling with cleanup."""
    
    def __init__(self, operation_name: str, cleanup_func: Optional[Callable] = None):
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        tprint_debug(f"Starting {self.operation_name}")
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            tprint_error(f"Error in {self.operation_name}: {exc_val}")
            logger.error(f"Error in {self.operation_name}: {exc_val}")
            if self.cleanup_func:
                try:
                    tprint_debug(f"Running cleanup for {self.operation_name}")
                    self.cleanup_func()
                    tprint_success(f"Cleanup completed for {self.operation_name}")
                except Exception as cleanup_error:
                    tprint_error(f"Cleanup failed: {cleanup_error}")
                    logger.error(f"Cleanup failed: {cleanup_error}")
        else:
            duration = time.time() - self.start_time
            tprint_success(f"Completed {self.operation_name} in {duration:.2f}s")
            logger.debug(f"Completed {self.operation_name} in {duration:.2f}s")

# Import time for ErrorContext
import time