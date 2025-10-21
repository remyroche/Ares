"""
Common utilities for clustering operations.

This module provides shared utility functions and decorators to eliminate
code duplication across the clustering codebase.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import time
import gc
import numpy as np
import pandas as pd
from contextlib import contextmanager
from functools import wraps
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error


class ClusteringCommonUtils:
    """Common utilities for clustering operations."""
    
    @staticmethod
    def safe_execute_with_cleanup(func: Callable, 
                                 cleanup_funcs: List[Callable] = None,
                                 error_message: str = "Operation failed",
                                 verbose: bool = True) -> Any:
        """
        Safely execute function with automatic cleanup.
        
        Args:
            func: Function to execute
            cleanup_funcs: List of cleanup functions to call on error
            error_message: Error message prefix
            verbose: Whether to print error messages
            
        Returns:
            Function result or raises exception
        """
        try:
            return func()
        except Exception as e:
            if verbose:
                tprint_error(f"{error_message}: {e}")
            
            # Execute cleanup functions
            if cleanup_funcs:
                for cleanup_func in cleanup_funcs:
                    try:
                        cleanup_func()
                    except Exception as cleanup_error:
                        if verbose:
                            tprint_warning(f"Cleanup failed: {cleanup_error}")
            
            raise
    
    @staticmethod
    def memory_cleanup(*arrays):
        """
        Safely clean up memory by deleting arrays.
        
        Args:
            *arrays: Arrays to clean up
        """
        try:
            for arr in arrays:
                if arr is not None:
                    del arr
            gc.collect()
        except Exception as e:
            tprint_warning(f"Memory cleanup warning: {e}")
    
    @staticmethod
    def performance_timer(operation_name: str):
        """
        Context manager for performance timing.
        
        Args:
            operation_name: Name of the operation for logging
            
        Returns:
            Context manager for timing
        """
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                tprint_info(f"⏱️ {operation_name}: {duration:.2f}s")
        return timer()
    
    @staticmethod
    def validate_config(config: Any, required_attrs: List[str]) -> bool:
        """
        Validate configuration object has required attributes.
        
        Args:
            config: Configuration object to validate
            required_attrs: List of required attribute names
            
        Returns:
            True if valid, False otherwise
        """
        if config is None:
            tprint_error("Configuration is None")
            return False
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            tprint_error(f"Missing required config attributes: {missing_attrs}")
            return False
        
        return True
    
    @staticmethod
    def get_safe_config_value(config: Any, attr: str, default: Any = None) -> Any:
        """
        Safely get configuration value with default.
        
        Args:
            config: Configuration object
            attr: Attribute name
            default: Default value if attribute doesn't exist
            
        Returns:
            Configuration value or default
        """
        return getattr(config, attr, default)
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if division by zero
            
        Returns:
            Division result or default
        """
        try:
            if abs(denominator) < 1e-10:
                return default
            return numerator / denominator
        except (ZeroDivisionError, OverflowError):
            return default
    
    @staticmethod
    def safe_log(value: float, default: float = 0.0) -> float:
        """
        Safely compute logarithm, returning default if invalid.
        
        Args:
            value: Value to take logarithm of
            default: Default value if logarithm is invalid
            
        Returns:
            Logarithm result or default
        """
        try:
            if value <= 0:
                return default
            return np.log(value)
        except (ValueError, OverflowError):
            return default
    
    @staticmethod
    def safe_sqrt(value: float, default: float = 0.0) -> float:
        """
        Safely compute square root, returning default if invalid.
        
        Args:
            value: Value to take square root of
            default: Default value if square root is invalid
            
        Returns:
            Square root result or default
        """
        try:
            if value < 0:
                return default
            return np.sqrt(value)
        except (ValueError, OverflowError):
            return default
    
    @staticmethod
    def chunked_processing(data: Union[np.ndarray, pd.DataFrame], 
                          chunk_size: int,
                          process_func: Callable,
                          **kwargs) -> List[Any]:
        """
        Process data in chunks to manage memory usage.
        
        Args:
            data: Data to process
            chunk_size: Size of each chunk
            process_func: Function to process each chunk
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results from processing each chunk
        """
        results = []
        n_samples = len(data)
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = data[i:end_idx]
            
            try:
                result = process_func(chunk, **kwargs)
                results.append(result)
            except Exception as e:
                tprint_warning(f"Chunk processing failed for range {i}:{end_idx}: {e}")
                results.append(None)
        
        return results
    
    @staticmethod
    def get_memory_usage_mb(data: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Get memory usage of data structure in MB.
        
        Args:
            data: Data structure to measure
            
        Returns:
            Memory usage in MB
        """
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes / (1024 * 1024)
            elif isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def format_memory_size(size_mb: float) -> str:
        """
        Format memory size in human-readable format.
        
        Args:
            size_mb: Size in MB
            
        Returns:
            Formatted size string
        """
        if size_mb < 1:
            return f"{size_mb * 1024:.1f} KB"
        elif size_mb < 1024:
            return f"{size_mb:.1f} MB"
        else:
            return f"{size_mb / 1024:.1f} GB"


# Decorator for automatic error handling and logging
def clustering_operation(operation_name: str, 
                        cleanup_funcs: List[Callable] = None,
                        verbose: bool = True):
    """
    Decorator for clustering operations with error handling.
    
    Args:
        operation_name: Name of the operation for logging
        cleanup_funcs: List of cleanup functions to call on error
        verbose: Whether to print status messages
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if verbose:
                    tprint_info(f"Starting {operation_name}...")
                result = func(*args, **kwargs)
                if verbose:
                    tprint_info(f"Completed {operation_name}")
                return result
            except Exception as e:
                if verbose:
                    tprint_error(f"{operation_name} failed: {e}")
                
                # Execute cleanup functions
                if cleanup_funcs:
                    for cleanup_func in cleanup_funcs:
                        try:
                            cleanup_func()
                        except Exception as cleanup_error:
                            if verbose:
                                tprint_warning(f"Cleanup failed: {cleanup_error}")
                
                raise
        return wrapper
    return decorator


# Memory optimization decorator
def memory_optimized(level: str = "moderate"):
    """
    Decorator for memory-optimized operations.
    
    Args:
        level: Optimization level ("light", "moderate", "aggressive")
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-execution memory optimization
            if level == "aggressive":
                gc.collect()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Post-execution cleanup
                if level in ["moderate", "aggressive"]:
                    gc.collect()
        return wrapper
    return decorator


# Performance tracking decorator
def performance_tracked(operation_name: str = None):
    """
    Decorator for performance tracking.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with ClusteringCommonUtils.performance_timer(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Safe execution decorator
def safe_execution(error_message: str = "Operation failed", 
                  cleanup_funcs: List[Callable] = None,
                  verbose: bool = True):
    """
    Decorator for safe execution with error handling.
    
    Args:
        error_message: Error message prefix
        cleanup_funcs: List of cleanup functions to call on error
        verbose: Whether to print error messages
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return ClusteringCommonUtils.safe_execute_with_cleanup(
                lambda: func(*args, **kwargs),
                cleanup_funcs=cleanup_funcs,
                error_message=error_message,
                verbose=verbose
            )
        return wrapper
    return decorator


# Convenience functions for common patterns
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Convenience function for safe division."""
    return ClusteringCommonUtils.safe_divide(numerator, denominator, default)


def safe_log(value: float, default: float = 0.0) -> float:
    """Convenience function for safe logarithm."""
    return ClusteringCommonUtils.safe_log(value, default)


def safe_sqrt(value: float, default: float = 0.0) -> float:
    """Convenience function for safe square root."""
    return ClusteringCommonUtils.safe_sqrt(value, default)


def memory_cleanup(*arrays):
    """Convenience function for memory cleanup."""
    ClusteringCommonUtils.memory_cleanup(*arrays)


def get_memory_usage_mb(data: Union[np.ndarray, pd.DataFrame]) -> float:
    """Convenience function for memory usage."""
    return ClusteringCommonUtils.get_memory_usage_mb(data)