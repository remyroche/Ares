"""
Standardized error handling decorators for SR detection methods.

This module provides consistent error handling patterns across all SR detection
methods to ensure proper error reporting and graceful degradation.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from src.utils.logger import system_logger

@dataclass
class SRErrorContext:
    """Context information for SR detection errors."""
    method_name: str
    data_size: int
    execution_time: float
    error_type: str
    error_message: str
    fallback_used: bool = False

class SRErrorHandler:
    """Centralized error handler for SR detection methods."""
    
    def __init__(self):
        self.logger = system_logger.getChild('SRErrorHandler')
        self.error_stats = {}
        self.performance_thresholds = {
            'fractal': 5.0,
            'pivot': 3.0,
            'volume': 2.0,
            'statistical': 1.0,
            'psychological': 0.5,
            'fibonacci': 4.0,
            'trendline': 10.0,
            'channel': 8.0,
            'volume_profile': 6.0,
            'market_structure': 7.0
        }
    
    def log_error_context(self, context: SRErrorContext):
        """Log detailed error context for debugging."""
        self.logger.error(
            f"SR Detection Error - Method: {context.method_name}, "
            f"Data Size: {context.data_size}, "
            f"Execution Time: {context.execution_time:.2f}s, "
            f"Error: {context.error_type}: {context.error_message}"
        )
        
        # Track error statistics
        if context.method_name not in self.error_stats:
            self.error_stats[context.method_name] = {
                'total_errors': 0,
                'error_types': {},
                'avg_execution_time': 0.0,
                'fallback_usage': 0
            }
        
        stats = self.error_stats[context.method_name]
        stats['total_errors'] += 1
        stats['error_types'][context.error_type] = stats['error_types'].get(context.error_type, 0) + 1
        stats['avg_execution_time'] = (stats['avg_execution_time'] + context.execution_time) / 2
        if context.fallback_used:
            stats['fallback_usage'] += 1
    
    def should_use_fallback(self, method_name: str, error_type: str) -> bool:
        """Determine if fallback method should be used based on error type."""
        fallback_errors = {
            'ValueError', 'AttributeError', 'KeyError', 'IndexError',
            'TypeError', 'MemoryError'
        }
        return error_type in fallback_errors
    
    def get_fallback_method(self, method_name: str) -> Optional[str]:
        """Get fallback method name for a given detection method."""
        fallback_map = {
            'fractal': 'basic_fractal',
            'pivot': 'basic_pivot',
            'volume': 'basic_volume',
            'statistical': 'basic_statistical',
            'psychological': 'basic_psychological',
            'fibonacci': 'basic_fibonacci',
            'trendline': 'basic_trendline',
            'channel': 'basic_channel',
            'volume_profile': 'basic_volume_profile',
            'market_structure': 'basic_market_structure'
        }
        return fallback_map.get(method_name)

# Global error handler instance
error_handler = SRErrorHandler()

def handles_sr_detection_errors(
    exceptions: tuple = (Exception,),
    default_return: Any = None,
    context: str = 'SR detection',
    use_fallback: bool = True,
    log_performance: bool = True
):
    """
    Decorator for standardized error handling in SR detection methods.
    
    Args:
        exceptions: Tuple of exception types to catch
        default_return: Default return value on error
        context: Context string for logging
        use_fallback: Whether to attempt fallback method
        log_performance: Whether to log performance metrics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, data, *args, **kwargs):
            method_name = func.__name__.replace('_detect_', '').replace('_levels', '')
            start_time = time.time()
            
            try:
                # Validate input data
                if data is None or len(data) == 0:
                    raise ValueError(f"Invalid data for {method_name} detection: data is None or empty")
                
                # Execute the method
                result = func(self, data, *args, **kwargs)
                
                # Log performance if enabled
                if log_performance:
                    execution_time = time.time() - start_time
                    threshold = error_handler.performance_thresholds.get(method_name, 5.0)
                    if execution_time > threshold:
                        error_handler.logger.warning(
                            f"Slow {method_name} detection: {execution_time:.2f}s (threshold: {threshold}s)"
                        )
                
                return result
                
            except exceptions as e:
                execution_time = time.time() - start_time
                error_type = type(e).__name__
                
                # Create error context
                error_context = SRErrorContext(
                    method_name=method_name,
                    data_size=len(data) if data is not None else 0,
                    execution_time=execution_time,
                    error_type=error_type,
                    error_message=str(e)
                )
                
                # Log error context
                error_handler.log_error_context(error_context)
                
                # Try fallback method if enabled
                if use_fallback and error_handler.should_use_fallback(method_name, error_type):
                    fallback_method_name = error_handler.get_fallback_method(method_name)
                    if fallback_method_name and hasattr(self, fallback_method_name):
                        try:
                            error_handler.logger.info(f"Attempting fallback method: {fallback_method_name}")
                            fallback_method = getattr(self, fallback_method_name)
                            result = fallback_method(data, *args, **kwargs)
                            error_context.fallback_used = True
                            error_handler.log_error_context(error_context)
                            return result
                        except Exception as fallback_error:
                            error_handler.logger.error(f"Fallback method {fallback_method_name} also failed: {fallback_error}")
                
                # Return default value
                if default_return is not None:
                    error_handler.logger.warning(f"Returning default value for {method_name}: {default_return}")
                    return default_return
                
                # Re-raise if no default return specified
                raise
                
        return wrapper
    return decorator

def handles_sr_data_validation(
    required_columns: List[str] = None,
    min_rows: int = 10,
    max_rows: int = 100000
):
    """
    Decorator for data validation in SR detection methods.
    
    Args:
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        max_rows: Maximum number of rows allowed
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, data, *args, **kwargs):
            # Validate data existence
            if data is None:
                raise ValueError("Data cannot be None")
            
            # Validate data type
            if not hasattr(data, 'shape') or not hasattr(data, 'columns'):
                raise ValueError("Data must be a pandas DataFrame")
            
            # Validate data size
            if len(data) < min_rows:
                raise ValueError(f"Insufficient data: {len(data)} rows, minimum {min_rows} required")
            
            if len(data) > max_rows:
                raise ValueError(f"Data too large: {len(data)} rows, maximum {max_rows} allowed")
            
            # Validate required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate data quality - drop columns with all null values
            null_cols = data.columns[data.isnull().all()].tolist()
            if null_cols:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ Dropping {len(null_cols)} columns with all null values: {null_cols}")
                # Create a copy with null columns dropped
                data = data.drop(columns=null_cols).copy()
            
            # Pass the cleaned data to the function
            return func(self, data, *args, **kwargs)
            
        return wrapper
    return decorator

def monitors_sr_performance(
    method_name: str = None,
    threshold_seconds: float = 5.0
):
    """
    Decorator for performance monitoring in SR detection methods.
    
    Args:
        method_name: Name of the method for logging
        threshold_seconds: Performance threshold in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, data, *args, **kwargs):
            start_time = time.time()
            start_memory = 0
            
            # Get memory usage if psutil is available
            try:
                import psutil
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            except ImportError:
                pass
            
            # Execute method
            result = func(self, data, *args, **kwargs)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            end_memory = 0
            memory_delta = 0
            
            try:
                import psutil
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
            except ImportError:
                pass
            
            # Log performance metrics
            method = method_name or func.__name__
            logger = getattr(self, 'logger', system_logger.getChild('SRPerformance'))
            
            logger.info(
                f"Performance - {method}: "
                f"Time: {execution_time:.2f}s, "
                f"Memory: {memory_delta:+.1f}MB, "
                f"Data: {len(data)} rows"
            )
            
            # Warn if performance threshold exceeded
            if execution_time > threshold_seconds:
                logger.warning(
                    f"Slow performance - {method}: {execution_time:.2f}s "
                    f"(threshold: {threshold_seconds}s)"
                )
            
            return result
            
        return wrapper
    return decorator

def validates_sr_output(
    expected_type: Type = list,
    min_items: int = 0,
    max_items: int = 1000
):
    """
    Decorator for output validation in SR detection methods.
    
    Args:
        expected_type: Expected return type
        min_items: Minimum number of items expected
        max_items: Maximum number of items allowed
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, data, *args, **kwargs):
            result = func(self, data, *args, **kwargs)
            
            # Validate result type
            if not isinstance(result, expected_type):
                raise TypeError(f"Expected {expected_type.__name__}, got {type(result).__name__}")
            
            # Validate result size
            if hasattr(result, '__len__'):
                if len(result) < min_items:
                    logger = getattr(self, 'logger', system_logger.getChild('SROutput'))
                    logger.warning(f"Few results: {len(result)} items (minimum: {min_items})")
                
                if len(result) > max_items:
                    logger = getattr(self, 'logger', system_logger.getChild('SROutput'))
                    logger.warning(f"Many results: {len(result)} items (maximum: {max_items})")
            
            return result
            
        return wrapper
    return decorator