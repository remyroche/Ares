"""
Memory Optimization System for Profit Labeling (Phase 1)

This module provides hardware-optimized memory management for data structures
using M1MemoryOptimizer and advanced memory management tools.

Key Features:
- Hardware-optimized memory management
- Data structure optimization
- Memory pool management
- Automatic garbage collection
- Memory pressure monitoring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import gc
import weakref
from collections import defaultdict

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.memory_management import MemoryManager, MemoryManagerConfig, MemoryStrategy
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_performance
)


class DataStructureType(Enum):
    """Types of data structures that can be optimized."""
    DATAFRAME = "dataframe"
    SERIES = "series"
    NDARRAY = "ndarray"
    DICT = "dict"
    LIST = "list"
    TUPLE = "tuple"


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization system."""
    
    # Basic settings
    enable_optimization: bool = True
    memory_limit_gb: float = 2.0
    enable_monitoring: bool = True
    
    # Data structure optimization
    enable_dataframe_optimization: bool = True
    enable_series_optimization: bool = True
    enable_ndarray_optimization: bool = True
    
    # Memory management
    enable_memory_pools: bool = True
    enable_weak_references: bool = True
    enable_automatic_cleanup: bool = True
    cleanup_interval: float = 30.0  # seconds
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    memory_pressure_threshold: float = 0.8
    
    # Performance settings
    enable_compression: bool = False
    enable_serialization: bool = False
    serialization_threshold_mb: float = 50.0


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_utilization: float
    peak_memory_mb: float
    garbage_collections: int
    optimization_count: int


class MemoryOptimizationSystem:
    """
    Hardware-optimized memory management system.
    
    Provides advanced memory optimization for data structures
    with hardware acceleration and automatic management.
    """
    
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        """Initialize memory optimization system."""
        self.config = config or MemoryOptimizationConfig()
        self.logger = logging.getLogger("MemoryOptimizationSystem")
        
        # Initialize hardware optimization tools
        self._initialize_hardware_tools()
        
        # Memory tracking
        self.memory_stats = MemoryStats(
            total_memory_mb=0.0,
            used_memory_mb=0.0,
            available_memory_mb=0.0,
            memory_utilization=0.0,
            peak_memory_mb=0.0,
            garbage_collections=0,
            optimization_count=0
        )
        
        # Memory pools for different data types
        self._memory_pools = defaultdict(list)
        self._weak_refs = {}
        
        # Performance tracking
        self.performance_metrics = {
            'optimizations_performed': 0,
            'memory_saved_mb': 0.0,
            'garbage_collections': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Last cleanup time
        self._last_cleanup = time.time()
        
        tprint_success("✅ MemoryOptimizationSystem initialized with hardware optimizations")
    
    def _initialize_hardware_tools(self):
        """Initialize hardware optimization tools."""
        try:
            if HARDWARE_AVAILABLE and self.config.enable_hardware_optimization:
                # Initialize memory optimizer
                self.memory_optimizer = M1MemoryOptimizer(
                    memory_limit_gb=self.config.memory_limit_gb
                )
                tprint_info("   → M1MemoryOptimizer: Initialized")
                
                # Initialize CPU optimizer
                self.cpu_optimizer = M1CPUOptimizer()
                tprint_info("   → M1CPUOptimizer: Initialized")
                
                # Initialize memory manager
                memory_config = MemoryManagerConfig(
                    strategy=MemoryStrategy.MODERATE,
                    enable_monitoring=True,
                    memory_threshold_mb=self.config.memory_limit_gb * 1024 * 0.8,
                    max_memory_mb=self.config.memory_limit_gb * 1024
                )
                self.memory_manager = MemoryManager(memory_config)
                tprint_info("   → MemoryManager: Initialized")
            else:
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.memory_manager = None
                tprint_warning("   → Hardware optimization: Not available")
                
        except Exception as e:
            tprint_error(f"Failed to initialize hardware tools: {e}")
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.memory_manager = None
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if not self.config.enable_dataframe_optimization:
            return df
        
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            # Optimize dtypes
            optimized_df = df.copy()
            
            for col in optimized_df.columns:
                col_type = optimized_df[col].dtype
                
                if col_type != 'object':
                    c_min = optimized_df[col].min()
                    c_max = optimized_df[col].max()
                    
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            optimized_df[col] = optimized_df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            optimized_df[col] = optimized_df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            optimized_df[col] = optimized_df[col].astype(np.int32)
                    
                    elif str(col_type)[:5] == 'float':
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            optimized_df[col] = optimized_df[col].astype(np.float32)
            
            # Convert object columns to category if beneficial
            for col in optimized_df.select_dtypes(include=['object']).columns:
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')
            
            optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            memory_saved = original_memory - optimized_memory
            
            self.performance_metrics['optimizations_performed'] += 1
            self.performance_metrics['memory_saved_mb'] += memory_saved
            
            if memory_saved > 0:
                tprint_performance(f"DataFrame optimized: {memory_saved:.2f}MB saved")
            
            return optimized_df
            
        except Exception as e:
            tprint_warning(f"DataFrame optimization failed: {e}")
            return df
    
    def optimize_series(self, series: pd.Series) -> pd.Series:
        """Optimize Series memory usage."""
        if not self.config.enable_series_optimization:
            return series
        
        try:
            original_memory = series.memory_usage(deep=True) / 1024 / 1024  # MB
            
            # Optimize dtype
            optimized_series = series.copy()
            
            if optimized_series.dtype != 'object':
                c_min = optimized_series.min()
                c_max = optimized_series.max()
                
                if str(optimized_series.dtype)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_series = optimized_series.astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_series = optimized_series.astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_series = optimized_series.astype(np.int32)
                
                elif str(optimized_series.dtype)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_series = optimized_series.astype(np.float32)
            
            # Convert to category if beneficial
            elif optimized_series.dtype == 'object':
                if optimized_series.nunique() / len(optimized_series) < 0.5:
                    optimized_series = optimized_series.astype('category')
            
            optimized_memory = optimized_series.memory_usage(deep=True) / 1024 / 1024  # MB
            memory_saved = original_memory - optimized_memory
            
            self.performance_metrics['optimizations_performed'] += 1
            self.performance_metrics['memory_saved_mb'] += memory_saved
            
            if memory_saved > 0:
                tprint_performance(f"Series optimized: {memory_saved:.2f}MB saved")
            
            return optimized_series
            
        except Exception as e:
            tprint_warning(f"Series optimization failed: {e}")
            return series
    
    def optimize_ndarray(self, arr: np.ndarray) -> np.ndarray:
        """Optimize NumPy array memory usage."""
        if not self.config.enable_ndarray_optimization:
            return arr
        
        try:
            original_memory = arr.nbytes / 1024 / 1024  # MB
            
            # Optimize dtype
            if arr.dtype != 'object':
                c_min = arr.min()
                c_max = arr.max()
                
                if str(arr.dtype)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_arr = arr.astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_arr = arr.astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_arr = arr.astype(np.int32)
                    else:
                        optimized_arr = arr
                
                elif str(arr.dtype)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized_arr = arr.astype(np.float32)
                    else:
                        optimized_arr = arr
                else:
                    optimized_arr = arr
            else:
                optimized_arr = arr
            
            optimized_memory = optimized_arr.nbytes / 1024 / 1024  # MB
            memory_saved = original_memory - optimized_memory
            
            self.performance_metrics['optimizations_performed'] += 1
            self.performance_metrics['memory_saved_mb'] += memory_saved
            
            if memory_saved > 0:
                tprint_performance(f"NumPy array optimized: {memory_saved:.2f}MB saved")
            
            return optimized_arr
            
        except Exception as e:
            tprint_warning(f"NumPy array optimization failed: {e}")
            return arr
    
    def optimize_data_structure(self, data: Any) -> Any:
        """Optimize any data structure based on its type."""
        if isinstance(data, pd.DataFrame):
            return self.optimize_dataframe(data)
        elif isinstance(data, pd.Series):
            return self.optimize_series(data)
        elif isinstance(data, np.ndarray):
            return self.optimize_ndarray(data)
        else:
            return data
    
    def get_memory_pool(self, data_type: DataStructureType) -> List[Any]:
        """Get memory pool for specific data type."""
        if self.config.enable_memory_pools:
            return self._memory_pools[data_type]
        return []
    
    def add_to_memory_pool(self, data_type: DataStructureType, data: Any):
        """Add data to memory pool."""
        if self.config.enable_memory_pools:
            self._memory_pools[data_type].append(data)
            self.performance_metrics['pool_hits'] += 1
    
    def get_from_memory_pool(self, data_type: DataStructureType) -> Optional[Any]:
        """Get data from memory pool."""
        if self.config.enable_memory_pools and self._memory_pools[data_type]:
            data = self._memory_pools[data_type].pop()
            self.performance_metrics['pool_hits'] += 1
            return data
        else:
            self.performance_metrics['pool_misses'] += 1
            return None
    
    def optimize_memory_usage(self):
        """Perform comprehensive memory optimization."""
        try:
            # Hardware optimization
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory_usage()
            
            # Memory manager optimization
            if self.memory_manager:
                self.memory_manager.optimize_memory_usage()
            
            # Garbage collection
            if self.config.enable_automatic_cleanup:
                gc.collect()
                self.performance_metrics['garbage_collections'] += 1
            
            # Update memory stats
            self._update_memory_stats()
            
            self.performance_metrics['optimizations_performed'] += 1
            
        except Exception as e:
            tprint_error(f"Memory optimization failed: {e}")
    
    def _update_memory_stats(self):
        """Update memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.memory_stats.used_memory_mb = memory_info.rss / 1024 / 1024
            self.memory_stats.total_memory_mb = psutil.virtual_memory().total / 1024 / 1024
            self.memory_stats.available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
            self.memory_stats.memory_utilization = (
                self.memory_stats.used_memory_mb / self.memory_stats.total_memory_mb
            )
            self.memory_stats.peak_memory_mb = max(
                self.memory_stats.peak_memory_mb,
                self.memory_stats.used_memory_mb
            )
            
        except ImportError:
            # Fallback if psutil not available
            self.memory_stats.used_memory_mb = 0.0
            self.memory_stats.total_memory_mb = 0.0
            self.memory_stats.available_memory_mb = 0.0
            self.memory_stats.memory_utilization = 0.0
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high."""
        self._update_memory_stats()
        return self.memory_stats.memory_utilization > self.config.memory_pressure_threshold
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        self._update_memory_stats()
        return self.memory_stats
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **self.performance_metrics,
            'memory_stats': self.memory_stats,
            'hardware_optimization_enabled': self.memory_optimizer is not None,
            'memory_pools_enabled': self.config.enable_memory_pools,
            'automatic_cleanup_enabled': self.config.enable_automatic_cleanup
        }
    
    def cleanup(self):
        """Perform cleanup operations."""
        # Clear memory pools
        if self.config.enable_memory_pools:
            for pool in self._memory_pools.values():
                pool.clear()
        
        # Clear weak references
        self._weak_refs.clear()
        
        # Garbage collection
        gc.collect()
        
        # Update memory stats
        self._update_memory_stats()
        
        tprint_info("🧹 Memory optimization cleanup performed")


# Decorator for automatic memory optimization
def memory_optimized(optimization_system: MemoryOptimizationSystem):
    """Decorator for automatic memory optimization of function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Optimize result
            optimized_result = optimization_system.optimize_data_structure(result)
            
            return optimized_result
        
        return wrapper
    return decorator


# Factory function for easy instantiation
def get_memory_optimization_system(config: Optional[MemoryOptimizationConfig] = None) -> MemoryOptimizationSystem:
    """
    Get a memory optimization system instance.
    
    Args:
        config: Optional configuration for the optimization system
        
    Returns:
        MemoryOptimizationSystem instance
    """
    return MemoryOptimizationSystem(config)