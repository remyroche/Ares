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
                logger.log(log_level, f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                return default_return
            except Exception as e:
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
        raise ConfigurationError(f"Missing required configuration keys: {missing_keys}")
    
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
        raise DataValidationError("Feature data X cannot be None")
    
    if hasattr(X, 'shape'):
        if X.shape[0] < min_samples:
            raise DataValidationError(f"Insufficient samples: {X.shape[0]} < {min_samples}")
        if X.shape[1] < min_features:
            raise DataValidationError(f"Insufficient features: {X.shape[1]} < {min_features}")
    
    if y is not None:
        if hasattr(y, 'shape') and len(y) != len(X):
            raise DataValidationError(f"Target length {len(y)} doesn't match feature length {len(X)}")
    
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
        return __import__(module_name)
    except ImportError as e:
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
        
        if used_gb > threshold_gb:
            raise ResourceError(f"Memory usage {used_gb:.2f}GB exceeds threshold {threshold_gb}GB")
        
        return True
    except ImportError:
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
        return primary_func()
    except error_types as e:
        logger.warning(f"Primary function failed: {e}, using fallback")
        return fallback_func()

class ErrorContext:
    """Context manager for error handling with cleanup."""
    
    def __init__(self, operation_name: str, cleanup_func: Optional[Callable] = None):
        self.operation_name = operation_name
        self.cleanup_func = cleanup_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error in {self.operation_name}: {exc_val}")
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {cleanup_error}")
        else:
            duration = time.time() - self.start_time
            logger.debug(f"Completed {self.operation_name} in {duration:.2f}s")

# Import time for ErrorContext
import time