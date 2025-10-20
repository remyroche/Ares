"""
Advanced memory management for clustering components.

This module provides comprehensive memory management with monitoring,
cleanup, and optimization for large-scale clustering operations.
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
import warnings

from src.utils.tprint import (
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_structured
)

# Import M1 hardware optimizations
try:
    _usage(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize memory usage of data structures."""
        try:
            if isinstance(data, np.ndarray):
                return self._optimize_numpy_array(data)
            elif isinstance(data, pd.DataFrame):
                return self._optimize_dataframe(data)
            else:
                return data

        except Exception as e:
            tprint_warning(f"Memory optimization failed: {e}")
            return data

    def _optimize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage."""
        try:
            # Use M1 optimization if available
            if self.m1_optimizer:
                return self.m1_optimizer.optimize_array(array)

            # Standard optimization
            if array.dtype == np.float64:
                # Try to convert to float32 if precision allows
                finite_mask = np.isfinite(array)
                if finite_mask.all() and np.max(np.abs(array)) < 3.4e38:
                    return array.astype(np.float32)

            return array

        except Exception as e:
            tprint_warning(f"Numpy array optimization failed: {e}")
            return array

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame memory usage."""
        try:
            # Use M1 optimization if available
            if self.m1_pool_manager:
                return self.m1_pool_manager.optimize_dataframe(df)

            # Standard optimization
            optimized_df = df.copy()

            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]).columns:
                col_data = optimized_df[col]
                if col_data.dtype == np.float64:
                    # Try to convert to float32
                    if col_data.notna().all() and col_data.abs().max() < 3.4e38:
                        optimized_df[col] = col_data.astype(np.float32)
                elif col_data.dtype == np.int64:
                    # Try to convert to smaller int types
                    if col_data.min() >= 0:
                        if col_data.max() < 255:
                            optimized_df[col] = col_data.astype(np.uint8)
                        elif col_data.max() < 65535:
                            optimized_df[col] = col_data.astype(np.uint16)
                        elif col_data.max() < 4294967295:
                            optimized_df[col] = col_data.astype(np.uint32)
                    else:
                        if col_data.min() >= -128 and col_data.max() <= 127:
                            optimized_df[col] = col_data.astype(np.int8)
                        elif col_data.min() >= -32768 and col_data.max() <= 32767:
                            optimized_df[col] = col_data.astype(np.int16)
                        elif col_data.min() >= -2147483648 and col_data.max() <= 2147483647:
                            optimized_df[col] = col_data.astype(np.int32)

            return optimized_df

        except Exception as e:
            tprint_warning(f"DataFrame optimization failed: {e}")
            return df

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add cleanup callback."""
        self.cleanup_callbacks.append(callback)

    def remove_cleanup_callback(self, callback: Callable) -> None:
        """Remove cleanup callback."""
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        stats = self.get_memory_stats()

        return {
            'current_stats': stats.to_dict(),
            'peak_memory_mb': self.peak_memory_mb,
            'memory_limit_mb': self.memory_limit_mb,
            'within_limit': self.check_memory_limit(),
            'm1_optimization_enabled': self.enable_m1_optimization,
            'cleanup_callbacks_count': len(self.cleanup_callbacks),
            'history_length': len(self.memory_history)
        }

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if self.m1_monitor:
                self.m1_monitor.stop_monitoring()
        except Exception:
            pass

@contextmanager
def memory_checkpoint(operation_name: str, memory_manager: Optional[MemoryManager] = None):
    """Context manager for memory monitoring during operations."""
    if memory_manager is None:
        memory_manager = MemoryManager()

    start_stats = memory_manager.get_memory_stats()
    start_time = time.time()

    try:
        tprint_info(f"Starting memory checkpoint: {operation_name}")
        yield memory_manager

    finally:
        end_stats = memory_manager.get_memory_stats()
        end_time = time.time()

        memory_delta = end_stats.process_memory_mb - start_stats.process_memory_mb
        time_delta = end_time - start_time

        tprint_performance(f"Memory checkpoint {operation_name}: "
                          f"{memory_delta:+.2f}MB in {time_delta:.2f}s")

        # Force cleanup if memory usage is high
        if end_stats.memory_percentage > 80:
            memory_manager.force_cleanup()

class MemoryOptimizedArray:
    """Memory-optimized numpy array wrapper."""

    def __init__(self, array: np.ndarray, memory_manager: Optional[MemoryManager] = None):
        """Initialize with memory manager."""
        self.memory_manager = memory_manager or MemoryManager()
        self._array = self.memory_manager.optimize_memory_usage(array)
        self._original_shape = array.shape
        self._original_dtype = array.dtype

    @property
    def array(self) -> np.ndarray:
        """Get the optimized array."""
        return self._array

    def get_memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self._array.nbytes / (1024 * 1024)

    def cleanup(self) -> None:
        """Cleanup array memory."""
        del self._array
        self.memory_manager.force_cleanup()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass

def create_memory_manager(
    memory_limit_mb: Optional[int] = None,
    enable_m1_optimization: bool = True
) -> MemoryManager:
    """Create and configure memory manager."""
    return MemoryManager(
        memory_limit_mb=memory_limit_mb,
        enable_m1_optimization=enable_m1_optimization
    )

def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage of functions."""
    def wrapper(*args, **kwargs):
        memory_manager = MemoryManager()

        with memory_checkpoint(f"function_{func.__name__}", memory_manager):
            result = func(*args, **kwargs)

            # Log memory report
            report = memory_manager.get_memory_report()
            tprint_structured({
                'function': func.__name__,
                'memory_report': report
            })

            return result

    return wrapper
