"""
Minimal Hardware Utilities for BaseStep Compatibility.

This module provides minimal hardware optimization functions that are required
by BaseStep but don't depend on external libraries like psutil, numpy, or pandas.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import functools
import gc
import time

# Minimal configuration classes
@dataclass
class IntegratedHardwareConfig:
    enable_automatic_optimization: bool = True
    enable_caching: bool = True
    enable_memory_monitoring: bool = True
    memory_limit_gb: float = 4.0
    cache_memory_limit_mb: float = 256.0

class WorkloadCategory(Enum):
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    INFERENCE = "inference"

class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class MemoryOptimizationLevel(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"

@dataclass
class OptimizationConfig:
    memory_threshold_mb: float = 100.0
    enable_compression: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

# Minimal hardware manager class
class MinimalHardwareManager:
    def __init__(self, config: IntegratedHardwareConfig):
        self.config = config
        self.performance_metrics = {
            'optimizations_applied': 0,
            'memory_savings_mb': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def optimize_dataframe(self, df):
        """Minimal DataFrame optimization."""
        if df is None:
            return df
        # Just return the DataFrame as-is for minimal implementation
        self.performance_metrics['optimizations_applied'] += 1
        return df
    
    def optimize_for_workload(self, workload: str):
        """Minimal workload optimization."""
        pass
    
    def clear_all_caches(self):
        """Clear all caches."""
        gc.collect()
    
    def get_performance_metrics(self):
        """Get performance metrics."""
        return self.performance_metrics.copy()

# Minimal decorators
def memory_optimized(optimization_level: str = "balanced"):
    """Minimal memory optimization decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Just call the function as-is for minimal implementation
            return func(*args, **kwargs)
        return wrapper
    return decorator

def memory_efficient(memory_threshold_mb: float = 100.0, enable_compression: bool = True, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
    """Minimal memory efficient decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def performance_tracked(func):
    """Minimal performance tracking decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # Could add performance tracking here if needed
        return result
    return wrapper

def smart_cache(ttl: int = 1800, max_size: int = 50):
    """Minimal smart cache decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def auto_optimize(config: OptimizationConfig = None):
    """Minimal auto optimization decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Minimal utility functions
def get_integrated_hardware_manager(config: IntegratedHardwareConfig = None):
    """Get integrated hardware manager."""
    if config is None:
        config = IntegratedHardwareConfig()
    return MinimalHardwareManager(config)

def optimize_dataframe(df):
    """Minimal DataFrame optimization."""
    return df

def optimize_array(arr):
    """Minimal array optimization."""
    return arr

def force_cleanup():
    """Force cleanup."""
    gc.collect()

def get_memory_stats():
    """Get minimal memory stats."""
    return {
        'memory_available_mb': 0,
        'memory_used_mb': 0,
        'memory_percent': 0
    }

def m1_optimized(func):
    """Minimal M1 optimization decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def cache_result(ttl: int = 1800, max_size: int = 50):
    """Minimal cache result decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator