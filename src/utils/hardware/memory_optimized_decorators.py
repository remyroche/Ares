"""
Memory-Optimized Decorators with Garbage Collection and Chunking.

This module provides decorators that automatically apply memory optimization,
garbage collection, and chunking strategies to functions throughout the codebase.
"""

import logging
import time
import functools
import gc
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .advanced_memory_manager import (
    get_advanced_memory_manager, memory_efficient_processing, 
    chunked_processing, track_memory_usage, MemoryConfig
)
from .enhanced_caching_system import get_global_cache, CacheConfig
from .optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    OptimizationConfig, OptimizationLevel as DecoratorOptimizationLevel
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class MemoryOptimizationLevel(Enum):
    """Memory optimization levels."""
    NONE = "none"
    LIGHT = "light"          # Basic GC and cleanup
    MODERATE = "moderate"    # Chunking and memory pools
    AGGRESSIVE = "aggressive" # Full optimization with weak references
    MAXIMUM = "maximum"      # All optimizations including tracing

class ChunkingMode(Enum):
    """Chunking modes for data processing."""
    DISABLED = "disabled"
    AUTO = "auto"            # Automatic chunking based on data size
    FIXED = "fixed"          # Fixed chunk size
    MEMORY_AWARE = "memory_aware"  # Based on available memory
    STREAMING = "streaming"  # Streaming processing

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization decorators."""
    # Memory optimization level
    optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.AGGRESSIVE
    
    # Garbage collection
    enable_aggressive_gc: bool = True
    gc_after_each_chunk: bool = True
    gc_after_function: bool = True
    
    # Chunking
    enable_chunking: bool = True
    chunking_mode: ChunkingMode = ChunkingMode.MEMORY_AWARE
    chunk_size_mb: Optional[float] = None
    max_chunk_size_mb: float = 100.0
    min_chunk_size_mb: float = 1.0
    
    # Memory pools
    enable_memory_pools: bool = True
    pool_size_mb: float = 50.0
    
    # Weak references
    enable_weak_references: bool = True
    
    # Memory monitoring
    enable_memory_monitoring: bool = True
    log_memory_usage: bool = False
    memory_threshold_mb: float = 200.0
    
    # Performance tracking
    enable_performance_tracking: bool = True
    track_memory_delta: bool = True
    track_gc_performance: bool = True

def memory_optimized(
    optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.AGGRESSIVE,
    enable_chunking: bool = True,
    chunking_mode: ChunkingMode = ChunkingMode.MEMORY_AWARE,
    chunk_size_mb: Optional[float] = None,
    enable_aggressive_gc: bool = True,
    log_memory_usage: bool = False
):
    """
    Memory-optimized decorator with garbage collection and chunking.
    
    Args:
        optimization_level: Level of memory optimization to apply
        enable_chunking: Whether to enable chunking for large data
        chunking_mode: Mode for chunking strategy
        chunk_size_mb: Fixed chunk size in MB (if using FIXED mode)
        enable_aggressive_gc: Whether to enable aggressive garbage collection
        log_memory_usage: Whether to log memory usage
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory manager
            memory_config = MemoryConfig(
                enable_aggressive_gc=enable_aggressive_gc,
                enable_chunking=enable_chunking,
                default_chunk_size_mb=chunk_size_mb or 50.0,
                max_chunk_size_mb=200.0,
                enable_memory_pools=True,
                enable_weak_references=optimization_level in [MemoryOptimizationLevel.AGGRESSIVE, MemoryOptimizationLevel.MAXIMUM]
            )
            memory_manager = get_advanced_memory_manager(memory_config)
            
            # Track memory usage if enabled
            if log_memory_usage:
                start_stats = memory_manager.get_memory_stats()
                tprint_debug(f"Starting {func.__name__} - Memory: {start_stats.used_memory_mb:.1f}MB")
            
            # Use memory context for automatic cleanup
            with memory_manager.memory_context(func.__name__):
                # Check if we need chunking
                if enable_chunking and len(args) > 0:
                    first_arg = args[0]
                    
                    # Determine if data is large enough for chunking
                    should_chunk = False
                    if isinstance(first_arg, pd.DataFrame):
                        should_chunk = first_arg.memory_usage(deep=True).sum() > chunk_size_mb * 1024 * 1024 if chunk_size_mb else True
                    elif isinstance(first_arg, np.ndarray):
                        should_chunk = first_arg.nbytes > chunk_size_mb * 1024 * 1024 if chunk_size_mb else True
                    elif isinstance(first_arg, dict):
                        should_chunk = len(str(first_arg)) > chunk_size_mb * 1024 * 1024 if chunk_size_mb else True
                    
                    if should_chunk:
                        # Process in chunks
                        return _process_with_chunking(func, args, kwargs, memory_manager, chunking_mode)
                
                # Process normally
                result = func(*args, **kwargs)
                
                # Force GC after function if enabled
                if enable_aggressive_gc:
                    memory_manager._force_gc_all_generations()
                
                # Log memory usage if enabled
                if log_memory_usage:
                    end_stats = memory_manager.get_memory_stats()
                    memory_delta = end_stats.used_memory_mb - start_stats.used_memory_mb
                    tprint_debug(f"Completed {func.__name__} - Memory delta: {memory_delta:+.1f}MB")
                
                return result
        
        return wrapper
    return decorator

def _process_with_chunking(func: Callable, args: Tuple, kwargs: Dict, 
                          memory_manager, chunking_mode: ChunkingMode) -> Any:
    """Process function with chunking."""
    first_arg = args[0]
    remaining_args = args[1:]
    
    # Determine chunk size
    if chunking_mode == ChunkingMode.FIXED:
        chunk_size_bytes = int(50 * 1024 * 1024)  # 50MB default
    else:
        chunk_size_bytes = None
    
    # Process in chunks
    if isinstance(first_arg, pd.DataFrame):
        results = []
        for chunk in memory_manager.chunk_data(first_arg, chunk_size_bytes):
            chunk_result = func(chunk, *remaining_args, **kwargs)
            results.append(chunk_result)
            
            # Force GC after each chunk
            if memory_manager.config.enable_aggressive_gc:
                gc.collect()
        
        # Combine results
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        else:
            return results
    
    elif isinstance(first_arg, np.ndarray):
        results = []
        for chunk in memory_manager.chunk_data(first_arg, chunk_size_bytes):
            chunk_result = func(chunk, *remaining_args, **kwargs)
            results.append(chunk_result)
            
            # Force GC after each chunk
            if memory_manager.config.enable_aggressive_gc:
                gc.collect()
        
        # Combine results
        if results and isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        else:
            return results
    
    else:
        # For other types, process normally
        return func(*args, **kwargs)

def gc_optimized(
    gc_after_function: bool = True,
    gc_after_chunks: bool = True,
    gc_generation: int = 2
):
    """
    Garbage collection optimized decorator.
    
    Args:
        gc_after_function: Whether to force GC after function execution
        gc_after_chunks: Whether to force GC after each chunk (if chunking)
        gc_generation: GC generation to collect (0, 1, or 2)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Track GC stats
            start_gc_count = sum(gc.get_count())
            
            try:
                result = func(*args, **kwargs)
                
                # Force GC after function if enabled
                if gc_after_function:
                    if gc_generation == 2:
                        collected = gc.collect()
                    else:
                        collected = gc.collect(gc_generation)
                    
                    if collected > 0:
                        tprint_debug(f"GC collected {collected} objects after {func.__name__}")
                
                return result
                
            except Exception as e:
                # Force GC even on error
                if gc_after_function:
                    gc.collect()
                raise
        
        return wrapper
    return decorator

def chunked_processing_auto(
    chunk_size_mb: Optional[float] = None,
    chunking_mode: ChunkingMode = ChunkingMode.MEMORY_AWARE,
    combine_results: bool = True
):
    """
    Automatic chunked processing decorator.
    
    Args:
        chunk_size_mb: Chunk size in MB (None for automatic)
        chunking_mode: Chunking strategy
        combine_results: Whether to combine chunk results
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return func(*args, **kwargs)
            
            first_arg = args[0]
            remaining_args = args[1:]
            
            # Check if data is large enough for chunking
            should_chunk = False
            if isinstance(first_arg, pd.DataFrame):
                data_size_mb = first_arg.memory_usage(deep=True).sum() / (1024 * 1024)
                should_chunk = data_size_mb > (chunk_size_mb or 50.0)
            elif isinstance(first_arg, np.ndarray):
                data_size_mb = first_arg.nbytes / (1024 * 1024)
                should_chunk = data_size_mb > (chunk_size_mb or 50.0)
            elif isinstance(first_arg, dict):
                data_size_mb = len(str(first_arg)) / (1024 * 1024)
                should_chunk = data_size_mb > (chunk_size_mb or 50.0)
            
            if not should_chunk:
                return func(*args, **kwargs)
            
            # Get memory manager
            memory_config = MemoryConfig(
                enable_chunking=True,
                default_chunk_size_mb=chunk_size_mb or 50.0,
                chunking_strategy=chunking_mode.value
            )
            memory_manager = get_advanced_memory_manager(memory_config)
            
            # Process in chunks
            results = []
            for i, chunk in enumerate(memory_manager.chunk_data(first_arg, 
                                                               int((chunk_size_mb or 50.0) * 1024 * 1024))):
                tprint_debug(f"Processing chunk {i+1}")
                
                chunk_result = func(chunk, *remaining_args, **kwargs)
                results.append(chunk_result)
                
                # Force GC after each chunk
                gc.collect()
            
            # Combine results if requested
            if combine_results and results:
                if isinstance(results[0], pd.DataFrame):
                    return pd.concat(results, ignore_index=True)
                elif isinstance(results[0], np.ndarray):
                    return np.concatenate(results)
                elif isinstance(results[0], dict):
                    # Merge dictionaries
                    combined = {}
                    for result in results:
                        combined.update(result)
                    return combined
                else:
                    return results
            else:
                return results
        
        return wrapper
    return decorator

def memory_pool_optimized(
    pool_size_mb: float = 50.0,
    enable_object_reuse: bool = True
):
    """
    Memory pool optimized decorator for object reuse.
    
    Args:
        pool_size_mb: Size of memory pool in MB
        enable_object_reuse: Whether to enable object reuse
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory manager with pool
            memory_config = MemoryConfig(
                enable_memory_pools=True,
                pool_size_mb=pool_size_mb
            )
            memory_manager = get_advanced_memory_manager(memory_config)
            
            # Get memory pool
            memory_pool = memory_manager.get_memory_pool()
            
            if memory_pool and enable_object_reuse:
                # Try to reuse objects from pool
                # This is a simplified example - in practice, you'd need to
                # modify the function to use the pool
                pass
            
            result = func(*args, **kwargs)
            
            # Return objects to pool if possible
            if memory_pool and enable_object_reuse:
                # This would need to be implemented based on the specific function
                pass
            
            return result
        
        return wrapper
    return decorator

def weak_reference_managed(
    enable_weak_refs: bool = True,
    cleanup_interval: float = 60.0
):
    """
    Weak reference managed decorator for large objects.
    
    Args:
        enable_weak_refs: Whether to enable weak reference management
        cleanup_interval: Interval for cleaning up dead references
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory manager with weak references
            memory_config = MemoryConfig(
                enable_weak_references=enable_weak_refs,
                weak_ref_cleanup_interval=cleanup_interval
            )
            memory_manager = get_advanced_memory_manager(memory_config)
            
            # Track large objects with weak references
            tracked_objects = []
            
            try:
                result = func(*args, **kwargs)
                
                # Track result if it's large
                if enable_weak_refs and hasattr(result, '__sizeof__'):
                    size_mb = result.__sizeof__() / (1024 * 1024)
                    if size_mb > 10:  # Track objects larger than 10MB
                        weak_ref = memory_manager.track_object(result)
                        tracked_objects.append(weak_ref)
                
                return result
                
            finally:
                # Clean up tracked objects
                for weak_ref in tracked_objects:
                    if weak_ref() is None:
                        tracked_objects.remove(weak_ref)
        
        return wrapper
    return decorator

def comprehensive_memory_optimization(
    optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.MAXIMUM,
    enable_caching: bool = True,
    enable_chunking: bool = True,
    enable_gc: bool = True,
    enable_pools: bool = True,
    enable_weak_refs: bool = True
):
    """
    Comprehensive memory optimization decorator combining all strategies.
    
    Args:
        optimization_level: Level of memory optimization
        enable_caching: Whether to enable caching
        enable_chunking: Whether to enable chunking
        enable_gc: Whether to enable garbage collection
        enable_pools: Whether to enable memory pools
        enable_weak_refs: Whether to enable weak references
    """
    def decorator(func: Callable) -> Callable:
        # Apply all optimizations
        optimized_func = func
        
        if enable_caching:
            optimized_func = smart_cache()(optimized_func)
        
        if enable_chunking:
            optimized_func = chunked_processing_auto()(optimized_func)
        
        if enable_gc:
            optimized_func = gc_optimized()(optimized_func)
        
        if enable_pools:
            optimized_func = memory_pool_optimized()(optimized_func)
        
        if enable_weak_refs:
            optimized_func = weak_reference_managed()(optimized_func)
        
        # Apply memory optimization
        optimized_func = memory_optimized(
            optimization_level=optimization_level,
            enable_chunking=enable_chunking,
            enable_aggressive_gc=enable_gc,
            log_memory_usage=True
        )(optimized_func)
        
        return optimized_func
    
    return decorator

# Convenience decorators for common use cases
def optimize_large_dataframes():
    """Optimize functions that process large DataFrames."""
    return comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True
    )

def optimize_large_arrays():
    """Optimize functions that process large NumPy arrays."""
    return comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True
    )

def optimize_memory_intensive():
    """Optimize memory-intensive functions."""
    return comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.MAXIMUM,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True,
        enable_weak_refs=True
    )

def optimize_streaming_processing():
    """Optimize functions for streaming data processing."""
    return memory_optimized(
        optimization_level=MemoryOptimizationLevel.MODERATE,
        enable_chunking=True,
        chunking_mode=ChunkingMode.STREAMING,
        enable_aggressive_gc=True,
        log_memory_usage=True
    )

# Global memory optimization functions
def force_garbage_collection():
    """Force garbage collection on all generations."""
    memory_manager = get_advanced_memory_manager()
    memory_manager._force_gc_all_generations()

def cleanup_all_memory():
    """Perform comprehensive memory cleanup."""
    memory_manager = get_advanced_memory_manager()
    memory_manager.cleanup_all()

def get_memory_optimization_stats() -> Dict[str, Any]:
    """Get memory optimization statistics."""
    memory_manager = get_advanced_memory_manager()
    return memory_manager.get_detailed_memory_info()

def optimize_dataframe_with_gc(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame with garbage collection."""
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def _optimize_df(data):
        from .enhanced_caching_system import optimize_dataframe_default
        return optimize_dataframe_default(data)
    
    return _optimize_df(df)

def optimize_array_with_gc(arr: np.ndarray) -> np.ndarray:
    """Optimize NumPy array with garbage collection."""
    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    def _optimize_arr(data):
        from .enhanced_caching_system import optimize_numpy_array_default
        return optimize_numpy_array_default(data)
    
    return _optimize_arr(arr)