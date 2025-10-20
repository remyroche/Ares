"""
Optimization Patches for Existing Code.

This module provides patches and updates to make existing code use the new
caching and optimization defaults throughout the codebase.
"""

import logging
import functools
from typing import Any, Dict, List, Optional, Callable, Union
import pandas as pd
import numpy as np

from .optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    cache_dataframe_result, cache_numpy_result, optimize_heavy_computation,
    memory_aware, optimize_all_dataframes, optimize_all_arrays
)
from .integrated_hardware_manager import (
    get_integrated_hardware_manager, optimize_dataframe_default,
    optimize_numpy_array_default, process_market_data, process_ml_training_data
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error
)

logger = logging.getLogger(__name__)

class OptimizationPatcher:
    """Patches existing code to use optimization defaults."""
    
    def __init__(self):
        self.logger = logger.getChild('OptimizationPatcher')
        self.patched_functions = set()
        self.original_functions = {}
    
    def patch_dataframe_operations(self, module):
        """Patch DataFrame operations in a module to use optimization defaults."""
        if hasattr(module, 'optimize_dataframe'):
            self._patch_function(module, 'optimize_dataframe', optimize_dataframe_default)
        
        if hasattr(module, 'optimize_dataframe_memory'):
            self._patch_function(module, 'optimize_dataframe_memory', optimize_dataframe_default)
        
        # Patch common DataFrame operations
        dataframe_methods = [
            'read_csv', 'read_parquet', 'read_json', 'read_excel',
            'concat', 'merge', 'join', 'groupby', 'pivot_table'
        ]
        
        for method_name in dataframe_methods:
            if hasattr(module, method_name):
                self._patch_dataframe_method(module, method_name)
    
    def patch_numpy_operations(self, module):
        """Patch NumPy operations in a module to use optimization defaults."""
        if hasattr(module, 'optimize_numpy_array'):
            self._patch_function(module, 'optimize_numpy_array', optimize_numpy_array_default)
        
        # Patch common NumPy operations
        numpy_functions = [
            'array', 'asarray', 'zeros', 'ones', 'empty', 'full',
            'arange', 'linspace', 'logspace', 'meshgrid'
        ]
        
        for func_name in numpy_functions:
            if hasattr(module, func_name):
                self._patch_numpy_function(module, func_name)
    
    def patch_memory_management(self, module):
        """Patch memory management functions in a module."""
        memory_functions = [
            'get_memory_usage', 'optimize_memory', 'memory_cleanup',
            'clear_memory', 'free_memory'
        ]
        
        for func_name in memory_functions:
            if hasattr(module, func_name):
                self._patch_memory_function(module, func_name)
    
    def patch_caching_functions(self, module):
        """Patch caching functions in a module."""
        if hasattr(module, 'cache_result'):
            self._patch_function(module, 'cache_result', smart_cache())
        
        if hasattr(module, 'memoize'):
            self._patch_function(module, 'memoize', smart_cache())
    
    def _patch_function(self, module, func_name: str, new_func: Callable):
        """Patch a function in a module."""
        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            self.original_functions[f"{module.__name__}.{func_name}"] = original_func
            setattr(module, func_name, new_func)
            self.patched_functions.add(f"{module.__name__}.{func_name}")
            tprint_debug(f"Patched {module.__name__}.{func_name}")
    
    def _patch_dataframe_method(self, module, method_name: str):
        """Patch a DataFrame method to return optimized DataFrames."""
        original_method = getattr(module, method_name)
        
        @functools.wraps(original_method)
        def optimized_method(*args, **kwargs):
            result = original_method(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return optimize_dataframe_default(result)
            return result
        
        self.original_functions[f"{module.__name__}.{method_name}"] = original_method
        setattr(module, method_name, optimized_method)
        self.patched_functions.add(f"{module.__name__}.{method_name}")
        tprint_debug(f"Patched DataFrame method {module.__name__}.{method_name}")
    
    def _patch_numpy_function(self, module, func_name: str):
        """Patch a NumPy function to return optimized arrays."""
        original_func = getattr(module, func_name)
        
        @functools.wraps(original_func)
        def optimized_func(*args, **kwargs):
            result = original_func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                return optimize_numpy_array_default(result)
            return result
        
        self.original_functions[f"{module.__name__}.{func_name}"] = original_func
        setattr(module, func_name, optimized_func)
        self.patched_functions.add(f"{module.__name__}.{func_name}")
        tprint_debug(f"Patched NumPy function {module.__name__}.{func_name}")
    
    def _patch_memory_function(self, module, func_name: str):
        """Patch a memory management function."""
        original_func = getattr(module, func_name)
        
        @functools.wraps(original_func)
        def optimized_func(*args, **kwargs):
            # Use integrated hardware manager for memory operations
            manager = get_integrated_hardware_manager()
            return manager.memory_optimizer.get_memory_stats()
        
        self.original_functions[f"{module.__name__}.{func_name}"] = original_func
        setattr(module, func_name, optimized_func)
        self.patched_functions.add(f"{module.__name__}.{func_name}")
        tprint_debug(f"Patched memory function {module.__name__}.{func_name}")
    
    def restore_original_functions(self):
        """Restore original functions."""
        for func_path, original_func in self.original_functions.items():
            module_name, func_name = func_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[func_name])
            setattr(module, func_name, original_func)
        
        self.patched_functions.clear()
        self.original_functions.clear()
        tprint_info("Restored original functions")

# Global patcher instance
_global_patcher = OptimizationPatcher()

def apply_optimization_patches():
    """Apply optimization patches to common modules."""
    try:
        # Patch common utility modules
        import src.utils.common_operations as common_ops
        _global_patcher.patch_dataframe_operations(common_ops)
        _global_patcher.patch_numpy_operations(common_ops)
        _global_patcher.patch_memory_management(common_ops)
        
        # Patch matrix operations
        try:
            import src.utils.matrix_operations as matrix_ops
            _global_patcher.patch_dataframe_operations(matrix_ops)
            _global_patcher.patch_numpy_operations(matrix_ops)
        except ImportError:
            pass
        
        # Patch hardware utilities
        try:
            import src.utils.hardware.m1_memory_optimizer as m1_mem
            _global_patcher.patch_memory_management(m1_mem)
        except ImportError:
            pass
        
        tprint_success("✅ Optimization patches applied successfully")
        
    except Exception as e:
        tprint_error(f"Failed to apply optimization patches: {e}")
        logger.error(f"Failed to apply optimization patches: {e}")

def remove_optimization_patches():
    """Remove optimization patches and restore original functions."""
    _global_patcher.restore_original_functions()
    tprint_info("Optimization patches removed")

# Convenience functions for common operations
def optimize_dataframe_auto(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically optimize DataFrame with default settings."""
    return optimize_dataframe_default(df)

def optimize_numpy_array_auto(arr: np.ndarray) -> np.ndarray:
    """Automatically optimize NumPy array with default settings."""
    return optimize_numpy_array_default(arr)

def optimize_data_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize all data in a dictionary."""
    return optimize_all_dataframes(optimize_all_arrays(data))

def cache_function_result(func: Callable, ttl: Optional[float] = None):
    """Cache function result with default settings."""
    return smart_cache(ttl=ttl)(func)

def optimize_heavy_function(func: Callable):
    """Apply heavy computation optimization to function."""
    return optimize_heavy_computation()(func)

def make_memory_efficient(func: Callable):
    """Make function memory efficient."""
    return memory_efficient()(func)

# Decorator for automatic optimization
def auto_optimize_function(optimize_inputs: bool = True, optimize_outputs: bool = True):
    """Decorator to automatically optimize function inputs and outputs."""
    def decorator(func: Callable) -> Callable:
        return auto_optimize(
            optimize_inputs=optimize_inputs,
            optimize_outputs=optimize_outputs
        )(func)
    return decorator

# Performance tracking decorator
def track_performance(func: Callable) -> Callable:
    """Decorator to track function performance."""
    return performance_tracked(log_performance=True, track_memory=True)(func)

# Apply patches on import
apply_optimization_patches()