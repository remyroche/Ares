"""
Memory Leak Prevention Decorators and Utilities

This module provides decorators and utilities to prevent memory leaks
in the pre-training pipeline.
"""

import functools
import gc
import logging
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

# Import memory management utilities
from .memory_management_utils import get_memory_manager, MemoryLeakDetector

logger = logging.getLogger(__name__)


def prevent_memory_leaks(cleanup_after: bool = True, 
                        monitor_memory: bool = True,
                        max_memory_mb: Optional[float] = None):
    """
    Decorator to prevent memory leaks in functions.
    
    Args:
        cleanup_after: Whether to cleanup memory after function execution
        monitor_memory: Whether to monitor memory usage during execution
        max_memory_mb: Maximum memory usage in MB before cleanup
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager = get_memory_manager()
            leak_detector = MemoryLeakDetector()
            
            # Set baseline if monitoring
            if monitor_memory:
                leak_detector.set_baseline()
            
            try:
                # Check memory pressure before execution
                if memory_manager.is_memory_pressure():
                    logger.warning("High memory pressure detected, cleaning up before execution")
                    memory_manager.cleanup_memory()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Check for memory leaks
                if monitor_memory and leak_detector.check_for_leaks():
                    logger.warning(f"Potential memory leak detected in {func.__name__}")
                
                # Cleanup after execution if requested
                if cleanup_after:
                    memory_manager.cleanup_memory()
                
                return result
                
            except Exception as e:
                # Cleanup on error
                if cleanup_after:
                    memory_manager.cleanup_memory()
                raise
        
        return wrapper
    return decorator


def safe_dataframe_operation(operation_name: str = "dataframe_operation"):
    """
    Decorator for safe DataFrame operations with memory management.
    
    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager = get_memory_manager()
            
            # Find DataFrame arguments
            dataframes = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    dataframes.append(arg)
            
            for value in kwargs.values():
                if isinstance(value, pd.DataFrame):
                    dataframes.append(value)
            
            # Optimize DataFrames before operation
            optimized_dataframes = []
            for df in dataframes:
                optimized_df = memory_manager.optimize_dataframe(df)
                optimized_dataframes.append(optimized_df)
            
            # Replace DataFrame arguments with optimized versions
            new_args = list(args)
            df_index = 0
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    new_args[i] = optimized_dataframes[df_index]
                    df_index += 1
            
            new_kwargs = kwargs.copy()
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    new_kwargs[key] = optimized_dataframes[df_index]
                    df_index += 1
            
            # Execute with memory monitoring
            return memory_manager.monitor_memory_usage(
                operation_name, func, *new_args, **new_kwargs
            )
        
        return wrapper
    return decorator


def chunk_large_operations(chunk_size: int = 10000, 
                          process_chunks: bool = True):
    """
    Decorator to process large DataFrames in chunks to prevent memory issues.
    
    Args:
        chunk_size: Size of each chunk
        process_chunks: Whether to process chunks individually
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager = get_memory_manager()
            
            # Find DataFrame arguments
            dataframe_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame) and len(arg) > chunk_size:
                    dataframe_args.append((i, arg))
            
            if not dataframe_args:
                # No large DataFrames, execute normally
                return func(*args, **kwargs)
            
            # Process large DataFrames in chunks
            results = []
            for arg_index, df in dataframe_args:
                chunks = memory_manager.chunk_dataframe(df, chunk_size)
                
                if process_chunks:
                    chunk_results = []
                    for chunk in chunks:
                        # Create new args with chunk
                        new_args = list(args)
                        new_args[arg_index] = chunk
                        
                        # Execute function on chunk
                        chunk_result = func(*new_args, **kwargs)
                        chunk_results.append(chunk_result)
                    
                    # Combine results
                    if chunk_results and isinstance(chunk_results[0], pd.DataFrame):
                        results.append(pd.concat(chunk_results, ignore_index=True))
                    else:
                        results.extend(chunk_results)
                else:
                    # Process all chunks at once
                    new_args = list(args)
                    new_args[arg_index] = pd.concat(chunks, ignore_index=True)
                    return func(*new_args, **kwargs)
            
            return results[0] if len(results) == 1 else results
        
        return wrapper
    return decorator


def cleanup_resources(*resource_names: str):
    """
    Decorator to cleanup specific resources after function execution.
    
    Args:
        *resource_names: Names of resources to cleanup
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Cleanup resources
                for resource_name in resource_names:
                    try:
                        if resource_name == 'memory':
                            get_memory_manager().cleanup_memory()
                        elif resource_name == 'gc':
                            gc.collect()
                        elif resource_name == 'tracemalloc':
                            if tracemalloc.is_tracing():
                                tracemalloc.stop()
                                tracemalloc.start()
                        else:
                            logger.warning(f"Unknown resource: {resource_name}")
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {resource_name}: {e}")
        
        return wrapper
    return decorator


class MemoryAwareDataFrame:
    """Memory-aware DataFrame wrapper with automatic cleanup."""
    
    def __init__(self, df: pd.DataFrame, auto_cleanup: bool = True):
        """
        Initialize memory-aware DataFrame.
        
        Args:
            df: DataFrame to wrap
            auto_cleanup: Whether to automatically cleanup on deletion
        """
        self.df = df
        self.auto_cleanup = auto_cleanup
        self.memory_manager = get_memory_manager()
        self._optimized = False
    
    def optimize(self) -> 'MemoryAwareDataFrame':
        """Optimize DataFrame memory usage."""
        if not self._optimized:
            self.df = self.memory_manager.optimize_dataframe(self.df)
            self._optimized = True
        return self
    
    def chunk(self, chunk_size: int = 10000) -> List['MemoryAwareDataFrame']:
        """Split DataFrame into memory-efficient chunks."""
        chunks = self.memory_manager.chunk_dataframe(self.df, chunk_size)
        return [MemoryAwareDataFrame(chunk, self.auto_cleanup) for chunk in chunks]
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying DataFrame."""
        return getattr(self.df, name)
    
    def __getitem__(self, key):
        """Delegate item access to underlying DataFrame."""
        return self.df[key]
    
    def __setitem__(self, key, value):
        """Delegate item assignment to underlying DataFrame."""
        self.df[key] = value
    
    def __len__(self):
        """Return length of underlying DataFrame."""
        return len(self.df)
    
    def __repr__(self):
        """Return string representation."""
        return f"MemoryAwareDataFrame({self.df.__repr__()})"
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.auto_cleanup:
            try:
                self.memory_manager.cleanup_memory()
            except Exception:
                pass


def create_memory_aware_dataframe(df: pd.DataFrame, 
                                 auto_cleanup: bool = True) -> MemoryAwareDataFrame:
    """Create a memory-aware DataFrame wrapper."""
    return MemoryAwareDataFrame(df, auto_cleanup)


# Utility functions for common memory leak patterns
def safe_merge_dataframes(left: pd.DataFrame, right: pd.DataFrame, 
                         **kwargs) -> pd.DataFrame:
    """Safely merge DataFrames with memory management."""
    memory_manager = get_memory_manager()
    
    with memory_manager.memory_checkpoint("dataframe_merge"):
        # Optimize DataFrames before merge
        left_opt = memory_manager.optimize_dataframe(left)
        right_opt = memory_manager.optimize_dataframe(right)
        
        # Perform merge
        result = pd.merge(left_opt, right_opt, **kwargs)
        
        # Cleanup original DataFrames if they're large
        if left.memory_usage(deep=True).sum() > 100 * 1024 * 1024:  # 100MB
            del left_opt
        if right.memory_usage(deep=True).sum() > 100 * 1024 * 1024:  # 100MB
            del right_opt
        
        return result


def safe_concat_dataframes(dataframes: List[pd.DataFrame], 
                          **kwargs) -> pd.DataFrame:
    """Safely concatenate DataFrames with memory management."""
    memory_manager = get_memory_manager()
    
    with memory_manager.memory_checkpoint("dataframe_concat"):
        # Optimize DataFrames before concatenation
        optimized_dfs = [memory_manager.optimize_dataframe(df) for df in dataframes]
        
        # Perform concatenation
        result = pd.concat(optimized_dfs, **kwargs)
        
        # Cleanup if result is large
        if result.memory_usage(deep=True).sum() > 500 * 1024 * 1024:  # 500MB
            memory_manager.cleanup_memory()
        
        return result


def safe_groupby_operation(df: pd.DataFrame, by: Union[str, List[str]], 
                          operation: Callable, **kwargs) -> pd.DataFrame:
    """Safely perform groupby operation with memory management."""
    memory_manager = get_memory_manager()
    
    with memory_manager.memory_checkpoint("groupby_operation"):
        # Optimize DataFrame before groupby
        optimized_df = memory_manager.optimize_dataframe(df)
        
        # Perform groupby operation
        result = optimized_df.groupby(by).apply(operation, **kwargs)
        
        return result


# Export main utilities
__all__ = [
    'prevent_memory_leaks',
    'safe_dataframe_operation',
    'chunk_large_operations',
    'cleanup_resources',
    'MemoryAwareDataFrame',
    'create_memory_aware_dataframe',
    'safe_merge_dataframes',
    'safe_concat_dataframes',
    'safe_groupby_operation'
]