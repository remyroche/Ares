"""
Optimization Decorators for Automatic Caching and Data Type Optimization.

This module provides decorators that automatically apply caching, data type optimization,
and memory efficiency improvements to functions throughout the codebase.
"""

import logging
import time
import functools
import hashlib
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class DummyModule:
        def __getattr__(self, name):
            return None
    np = DummyModule()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    class DummyModule:
        def __getattr__(self, name):
            return None
    pd = DummyModule()

from .enhanced_caching_system import (
    get_global_cache, CacheConfig, DataTypeOptimization, 
    optimize_dataframe_default, optimize_numpy_array_default
)
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, tprint_timer, LogLevel
)

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization levels for decorators."""
    NONE = "none"
    BASIC = "basic"          # Basic caching and data type optimization
    BALANCED = "balanced"    # Balanced optimization
    AGGRESSIVE = "aggressive"  # Full optimization with compression
    MAXIMUM = "maximum"      # Maximum optimization with all features

@dataclass
class OptimizationConfig:
    """Configuration for optimization decorators."""
    # Caching
    enable_caching: bool = True
    cache_ttl: Optional[float] = None  # None = use default TTL
    cache_key_func: Optional[Callable] = None
    
    # Data type optimization
    enable_dtype_optimization: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_threshold_mb: float = 100.0
    
    # Performance tracking
    enable_performance_tracking: bool = True
    log_performance: bool = False
    
    # Input/output optimization
    optimize_inputs: bool = True
    optimize_outputs: bool = True
    auto_convert_dataframes: bool = True
    auto_convert_arrays: bool = True
    
    # Compression
    enable_compression: bool = False

def smart_cache(
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    cache_config: Optional[CacheConfig] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    auto_detect_context: bool = True
):
    """
    Smart caching decorator with automatic data type optimization.
    
    Args:
        ttl: Time to live for cached items (seconds)
        key_func: Custom function to generate cache keys
        cache_config: Cache configuration
        optimization_config: Optimization configuration
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get optimization config
            config = optimization_config or OptimizationConfig()
            
            # Auto-detect context if enabled
            if auto_detect_context:
                # Detect if this function should be cached based on function name and data size
                func_name = func.__name__.lower()
                should_cache = any(keyword in func_name for keyword in [
                    'select', 'compute', 'calculate', 'process', 'correlation', 
                    'mutual', 'variance', 'stability', 'ensemble'
                ])
                
                # Auto-adjust TTL based on function type
                # Shorter TTLs to prevent stale data and memory bloat
                current_ttl = ttl
                if current_ttl is None:
                    if any(keyword in func_name for keyword in ['correlation', 'mutual', 'variance']):
                        current_ttl = 300   # 5 minutes for statistical calculations (stable results)
                    elif any(keyword in func_name for keyword in ['select', 'ensemble']):
                        current_ttl = 180   # 3 minutes for selection methods (may change with data)
                    else:
                        current_ttl = 60    # 1 minute for general functions (short-lived)
                
                # Enable caching if function should be cached
                if should_cache:
                    config.enable_caching = True
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_default_cache_key(func, args, kwargs)
            
            # Get cache instance
            cache = get_global_cache(cache_config)
            
            # Try to get from cache
            if config.enable_caching:
                result = cache.get(cache_key)
                if result is not None:
                    tprint_debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Optimize inputs if enabled
            if config.optimize_inputs:
                optimized_args, optimized_kwargs = _optimize_inputs(args, kwargs, config)
            else:
                optimized_args, optimized_kwargs = args, kwargs
            
            # Track performance if enabled
            start_time = time.perf_counter() if config.enable_performance_tracking else None
            
            # Execute function
            try:
                result = func(*optimized_args, **optimized_kwargs)
            except Exception as e:
                tprint_error(f"Function {func.__name__} failed: {e}")
                raise
            
            # Track performance
            if start_time and config.enable_performance_tracking:
                execution_time = time.perf_counter() - start_time
                if config.log_performance:
                    tprint_performance(f"{func.__name__} executed in {execution_time:.3f}s")
            
            # Optimize outputs if enabled
            if config.optimize_outputs:
                result = _optimize_output(result, config)
            
            # Cache result if enabled
            if config.enable_caching:
                cache.put(cache_key, result, current_ttl)
                tprint_debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

# @auto_optimize functionality is now merged into @memory_efficient

def memory_efficient(
    memory_threshold_mb: float = None,  # Auto-detect if None
    enable_compression: bool = True,
    auto_cleanup: bool = True,
    optimization_level: OptimizationLevel = None,  # Auto-detect if None
    optimize_inputs: bool = True,
    optimize_outputs: bool = True,
    auto_detect_context: bool = True,
    ttl: int = None  # Time-to-live for caching
):
    """
    Memory-efficient decorator with automatic cleanup, optimization, and data type optimization.
    
    Args:
        memory_threshold_mb: Memory threshold for triggering optimizations
        enable_compression: Whether to enable compression for large data
        auto_cleanup: Whether to automatically clean up memory
        optimization_level: Level of data type optimization to apply
        optimize_inputs: Whether to optimize function inputs
        optimize_outputs: Whether to optimize function outputs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache for memory monitoring
            cache = get_global_cache()
            
            # Auto-detect context if enabled
            if auto_detect_context:
                # Detect data size from first argument
                data_size_mb = 0
                if args and hasattr(args[0], 'nbytes'):
                    data_size_mb = args[0].nbytes / (1024 * 1024)
                elif args and hasattr(args[0], 'shape'):
                    # Estimate size for numpy arrays
                    data_size_mb = np.prod(args[0].shape) * 8 / (1024 * 1024)  # Assume float64
                
                # Auto-adjust memory threshold based on data size
                local_memory_threshold_mb = memory_threshold_mb
                local_optimization_level = optimization_level
                if local_memory_threshold_mb is None:
                    if data_size_mb > 1000:  # Large dataset
                        local_memory_threshold_mb = 2000.0
                        local_optimization_level = local_optimization_level or OptimizationLevel.AGGRESSIVE
                    elif data_size_mb > 100:  # Medium dataset
                        local_memory_threshold_mb = 500.0
                        local_optimization_level = local_optimization_level or OptimizationLevel.BALANCED
                    else:  # Small dataset
                        local_memory_threshold_mb = 100.0
                        local_optimization_level = local_optimization_level or OptimizationLevel.MINIMAL
                
                # Auto-adjust optimization level based on function name
                if local_optimization_level is None:
                    func_name = func.__name__.lower()
                    if any(keyword in func_name for keyword in ['ensemble', 'comprehensive', 'stability']):
                        optimization_level = OptimizationLevel.AGGRESSIVE
                    elif any(keyword in func_name for keyword in ['correlation', 'mutual', 'variance']):
                        local_optimization_level = OptimizationLevel.BALANCED
                    else:
                        optimization_level = OptimizationLevel.MINIMAL
            
            # Use defaults if not auto-detected
            local_memory_threshold_mb = local_memory_threshold_mb or 100.0
            optimization_level = optimization_level or OptimizationLevel.BALANCED
            
            # Check memory usage before execution
            initial_memory = cache._get_current_memory_usage()
            initial_memory_mb = initial_memory / (1024 * 1024)
            
            if initial_memory_mb > local_memory_threshold_mb and auto_cleanup:
                tprint_warning(f"High memory usage detected: {initial_memory_mb:.1f}MB, cleaning up...")
                cache._aggressive_cleanup()
            
            # Optimize inputs if enabled
            if optimize_inputs:
                # Create optimization config
                config = OptimizationConfig(
                    enable_caching=False,  # This decorator handles memory, not caching
                    enable_dtype_optimization=True,
                    optimization_level=optimization_level,
                    optimize_inputs=optimize_inputs,
                    optimize_outputs=optimize_outputs
                )
                optimized_args, optimized_kwargs = _optimize_inputs(args, kwargs, config)
            else:
                optimized_args, optimized_kwargs = args, kwargs
            
            # Execute function with memory monitoring
            try:
                result = func(*optimized_args, **optimized_kwargs)
                
                # Optimize outputs if enabled
                if optimize_outputs:
                    config = OptimizationConfig(
                        enable_caching=False,
                        enable_dtype_optimization=True,
                        optimization_level=optimization_level,
                        optimize_inputs=optimize_inputs,
                        optimize_outputs=optimize_outputs
                    )
                    result = _optimize_output(result, config)
                
                # Check memory usage after execution
                final_memory = cache._get_current_memory_usage()
                final_memory_mb = final_memory / (1024 * 1024)
                memory_delta = final_memory_mb - initial_memory_mb
                
                if memory_delta > local_memory_threshold_mb:
                    tprint_warning(f"Function {func.__name__} used {memory_delta:.1f}MB additional memory")
                
                # Auto cleanup if enabled
                if auto_cleanup and final_memory_mb > local_memory_threshold_mb * 1.5:
                    cache._evict_items(0.2)  # Evict 20% of items
                
                return result
                
            except Exception as e:
                tprint_error(f"Function {func.__name__} failed: {e}")
                raise
        
        return wrapper
    return decorator

def performance_tracked(
    log_performance: bool = None,  # Auto-detect if None
    track_memory: bool = True,
    track_cache_hits: bool = True,
    auto_detect_context: bool = True
):
    """
    Performance tracking decorator.
    
    Args:
        log_performance: Whether to log performance metrics
        track_memory: Whether to track memory usage
        track_cache_hits: Whether to track cache hit rates
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Auto-detect context if enabled
            if auto_detect_context and log_performance is None:
                # Detect if this is a performance-critical function
                func_name = func.__name__.lower()
                if any(keyword in func_name for keyword in ['select', 'process', 'compute', 'calculate']):
                    log_performance = True
                else:
                    log_performance = False
            
            # Use default if not auto-detected
            log_performance = log_performance if log_performance is not None else True
            
            # Performance tracking
            # Enhanced performance tracking
            start_time = time.perf_counter()
            initial_memory = 0
            cache_hits_before = 0
            cpu_usage_before = 0
            
            # Get system metrics before execution
            try:
                import psutil
                cpu_usage_before = psutil.cpu_percent()
            except ImportError:
                pass
            
            if track_memory:
                try:
                    from .enhanced_unified_memory_manager import get_enhanced_unified_memory_manager
                    memory_manager = get_enhanced_unified_memory_manager()
                    initial_memory = memory_manager.get_enhanced_memory_stats().get('current_usage_mb', 0) * 1024 * 1024
                except ImportError:
                    cache = get_global_cache()
                    initial_memory = cache._get_current_memory_usage()
            
            if track_cache_hits:
                try:
                    from .enhanced_caching_system import get_global_cache as get_enhanced_cache
                    cache = get_enhanced_cache()
                    stats = cache.get_statistics()
                    cache_hits_before = stats['hits']
                except ImportError:
                    cache = get_global_cache()
                    stats = cache.get_statistics()
                    cache_hits_before = stats['hits']
            
            # Execute function with enhanced monitoring
            try:
                # Try to get adaptive optimization recommendations
                try:
                    from .adaptive_optimization_engine import get_adaptive_optimization_engine, WorkloadCategory
                    adaptive_engine = get_adaptive_optimization_engine()
                    
                    # Record performance for learning
                    data_size_mb = sum(arg.nbytes for arg in args if hasattr(arg, 'nbytes')) / (1024 * 1024) if any(hasattr(arg, 'nbytes') for arg in args) else 100.0
                    
                    # Get optimization recommendations
                    optimization = adaptive_engine.optimize_operation(
                        operation_type=func.__name__,
                        workload_category=WorkloadCategory.DATA_PROCESSING,
                        data_size_mb=data_size_mb
                    )
                except ImportError:
                    optimization = None
                
                result = func(*args, **kwargs)
                
                # Calculate enhanced metrics
                execution_time = time.perf_counter() - start_time
                
                # Get system metrics after execution
                try:
                    cpu_usage_after = psutil.cpu_percent()
                    cpu_utilization = cpu_usage_after - cpu_usage_before
                except ImportError:
                    cpu_utilization = 0
                
                metrics = {
                    'function_name': func.__name__,
                    'execution_time': execution_time,
                    'success': True,
                    'cpu_utilization': cpu_utilization,
                    'optimization_applied': optimization.strategy.value if optimization else 'none',
                    'performance_improvement': optimization.performance_improvement if optimization else 1.0
                }
                
                if track_memory:
                    try:
                        from .enhanced_unified_memory_manager import get_enhanced_unified_memory_manager
                        memory_manager = get_enhanced_unified_memory_manager()
                        final_memory = memory_manager.get_enhanced_memory_stats().get('current_usage_mb', 0) * 1024 * 1024
                        metrics['memory_delta_mb'] = (final_memory - initial_memory) / (1024 * 1024)
                    except ImportError:
                        cache = get_global_cache()
                        final_memory = cache._get_current_memory_usage()
                        metrics['memory_delta_mb'] = (final_memory - initial_memory) / (1024 * 1024)
                
                if track_cache_hits:
                    try:
                        from .enhanced_caching_system import get_global_cache as get_enhanced_cache
                        cache = get_enhanced_cache()
                        stats = cache.get_statistics()
                        metrics['cache_hits_delta'] = stats['hits'] - cache_hits_before
                    except ImportError:
                        cache = get_global_cache()
                        stats = cache.get_statistics()
                        metrics['cache_hits_delta'] = stats['hits'] - cache_hits_before
                
                # Log enhanced performance metrics
                if log_performance:
                    tprint_performance(f"{func.__name__}: {execution_time:.3f}s")
                    if track_memory and 'memory_delta_mb' in metrics:
                        tprint_performance(f"Memory delta: {metrics['memory_delta_mb']:.2f}MB")
                    if 'cpu_utilization' in metrics:
                        tprint_performance(f"CPU utilization: {metrics['cpu_utilization']:.1f}%")
                    if optimization:
                        tprint_performance(f"Optimization: {optimization.strategy.value} ({optimization.performance_improvement:.2f}x)")
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                tprint_error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator

def _generate_default_cache_key(func: Callable, args: Tuple, kwargs: Dict) -> str:
    """Generate default cache key for function call."""
    # Create a string representation of the function call
    key_parts = [func.__name__]
    
    # Add args (skip self if it's a method)
    if args and hasattr(args[0], '__class__') and inspect.ismethod(func):
        key_parts.append(str(args[1:]))  # Skip self
    else:
        key_parts.append(str(args))
    
    # Add kwargs
    key_parts.append(str(sorted(kwargs.items())))
    
    # Create hash
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def _optimize_inputs(args: Tuple, kwargs: Dict, config: OptimizationConfig) -> Tuple[Tuple, Dict]:
    """Optimize function inputs based on configuration."""
    optimized_args = []
    optimized_kwargs = {}
    
    # Optimize positional arguments
    for arg in args:
        optimized_arg = _optimize_value(arg, config)
        optimized_args.append(optimized_arg)
    
    # Optimize keyword arguments
    for key, value in kwargs.items():
        optimized_value = _optimize_value(value, config)
        optimized_kwargs[key] = optimized_value
    
    return tuple(optimized_args), optimized_kwargs

def _optimize_output(result: Any, config: OptimizationConfig) -> Any:
    """Optimize function output based on configuration."""
    return _optimize_value(result, config)

def _optimize_value(value: Any, config: OptimizationConfig) -> Any:
    """Optimize a single value based on configuration."""
    if not config.enable_dtype_optimization:
        return value
    
    try:
        if isinstance(value, pd.DataFrame) and config.auto_convert_dataframes:
            return optimize_dataframe_default(value)
        elif isinstance(value, np.ndarray) and config.auto_convert_arrays:
            return optimize_numpy_array_default(value)
        elif isinstance(value, dict):
            # Recursively optimize dictionary values
            return {k: _optimize_value(v, config) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Optimize list/tuple elements
            optimized_items = [_optimize_value(item, config) for item in value]
            return type(value)(optimized_items)
        else:
            return value
    except Exception as e:
        logger.warning(f"Value optimization failed: {e}")
        return value

# Convenience decorators for common use cases
def cache_dataframe_result(ttl: Optional[float] = None):
    """Decorator specifically for DataFrame processing functions."""
    return smart_cache(
        ttl=ttl,
        optimization_config=OptimizationConfig(
            enable_caching=True,
            enable_dtype_optimization=True,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            auto_convert_dataframes=True,
            auto_convert_arrays=True
        )
    )

def cache_numpy_result(ttl: Optional[float] = None):
    """Decorator specifically for NumPy array processing functions."""
    return smart_cache(
        ttl=ttl,
        optimization_config=OptimizationConfig(
            enable_caching=True,
            enable_dtype_optimization=True,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            auto_convert_arrays=True
        )
    )

def optimize_heavy_computation():
    """Decorator for heavy computation functions with full optimization."""
    return smart_cache(
        optimization_config=OptimizationConfig(
            enable_caching=True,
            enable_dtype_optimization=True,
            optimization_level=OptimizationLevel.MAXIMUM,
            optimize_inputs=True,
            optimize_outputs=True,
            enable_performance_tracking=True,
            log_performance=True
        )
    )

def memory_aware():
    """Decorator for memory-aware functions with automatic cleanup."""
    return memory_efficient(
        memory_threshold_mb=200.0,
        enable_compression=True,
        auto_cleanup=True
    )

# Global optimization functions
def optimize_all_dataframes(data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize all DataFrames in a dictionary."""
    optimized_data = {}
    
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            optimized_data[key] = optimize_dataframe_default(value)
        elif isinstance(value, dict):
            optimized_data[key] = optimize_all_dataframes(value)
        else:
            optimized_data[key] = value
    
    return optimized_data

def optimize_all_arrays(data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize all NumPy arrays in a dictionary."""
    optimized_data = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            optimized_data[key] = optimize_numpy_array_default(value)
        elif isinstance(value, dict):
            optimized_data[key] = optimize_all_arrays(value)
        else:
            optimized_data[key] = value
    
    return optimized_data

def get_optimization_stats() -> Dict[str, Any]:
    """Get optimization statistics from the global cache."""
    cache = get_global_cache()
    return cache.get_statistics()

def clear_optimization_cache():
    """Clear the optimization cache."""
    cache = get_global_cache()
    cache.clear()
    tprint_info("Optimization cache cleared")

def auto_optimize(optimize_inputs: bool = True, optimize_outputs: bool = True, 
                 optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE):
    """
    Automatically optimize function inputs and outputs.
    
    Args:
        optimize_inputs: Whether to optimize function inputs
        optimize_outputs: Whether to optimize function outputs
        optimization_level: Level of optimization to apply
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Optimize inputs if requested
            if optimize_inputs:
                args, kwargs = _optimize_inputs_enhanced(args, kwargs)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Optimize outputs if requested
            if optimize_outputs:
                result = _optimize_output_enhanced(result)
            
            return result
        return wrapper
    return decorator

def _optimize_inputs_enhanced(args: tuple, kwargs: dict) -> tuple:
    """Enhanced input optimization with advanced features."""
    try:
        from .enhanced_unified_memory_manager import get_enhanced_unified_memory_manager
        memory_manager = get_enhanced_unified_memory_manager()
        
        optimized_args = []
        for arg in args:
            if isinstance(arg, (np.ndarray, pd.DataFrame)):
                optimized_arg = memory_manager.base_manager.optimize_data_for_component(arg, 'cpu')
                optimized_args.append(optimized_arg)
            else:
                optimized_args.append(arg)
        
        optimized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (np.ndarray, pd.DataFrame)):
                optimized_value = memory_manager.base_manager.optimize_data_for_component(value, 'cpu')
                optimized_kwargs[key] = optimized_value
            else:
                optimized_kwargs[key] = value
        
        return tuple(optimized_args), optimized_kwargs
    except ImportError:
        return args, kwargs

def _optimize_output_enhanced(result: Any) -> Any:
    """Enhanced output optimization with advanced features."""
    try:
        from .enhanced_unified_memory_manager import get_enhanced_unified_memory_manager
        memory_manager = get_enhanced_unified_memory_manager()
        
        if isinstance(result, (np.ndarray, pd.DataFrame)):
            return memory_manager.base_manager.optimize_data_for_component(result, 'cpu')
        return result
    except ImportError:
        return result