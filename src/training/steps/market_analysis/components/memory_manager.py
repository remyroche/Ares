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
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_structured
)

# Import M1 hardware optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_memory_pool_manager import get_m1_memory_pool_manager
    from src.utils.hardware.m1_memory_monitor import get_m1_memory_monitor
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    get_m1_memory_optimizer = lambda: None
    get_m1_memory_pool_manager = lambda: None
    get_m1_memory_monitor = lambda: None

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percentage: float
    process_memory_mb: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_memory_mb': self.total_memory_mb,
            'available_memory_mb': self.available_memory_mb,
            'used_memory_mb': self.used_memory_mb,
            'memory_percentage': self.memory_percentage,
            'process_memory_mb': self.process_memory_mb,
            'timestamp': self.timestamp
        }

class MemoryManager:
    """Advanced memory management with monitoring and optimization."""

    def __init__(
        self,
        memory_limit_mb: Optional[int] = None,
        enable_m1_optimization: bool = True,
        monitoring_interval: float = 1.0
    ):
        """Initialize memory manager."""
        self.memory_limit_mb = memory_limit_mb
        self.enable_m1_optimization = enable_m1_optimization and M1_HARDWARE_AVAILABLE
        self.monitoring_interval = monitoring_interval

        # Initialize hardware optimizers
        if self.enable_m1_optimization:
            self.m1_optimizer = get_m1_memory_optimizer()
            self.m1_pool_manager = get_m1_memory_pool_manager()
            self.m1_monitor = get_m1_memory_monitor()
        else:
            self.m1_optimizer = None
            self.m1_pool_manager = None
            self.m1_monitor = None

        # Memory tracking
        self.memory_history: List[MemoryStats] = []
        self.peak_memory_mb = 0.0
        self.cleanup_callbacks: List[Callable] = []

        # Start monitoring if available
        if self.m1_monitor:
            self.m1_monitor.start_monitoring()

        tprint_info(f"Memory manager initialized (M1: {self.enable_m1_optimization})")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            total_mb = memory.total / (1024 * 1024)
            available_mb = memory.available / (1024 * 1024)
            used_mb = memory.used / (1024 * 1024)
            percentage = memory.percent

            # Process memory
            process = psutil.Process()
            process_mb = process.memory_info().rss / (1024 * 1024)

            stats = MemoryStats(
                total_memory_mb=total_mb,
                available_memory_mb=available_mb,
                used_memory_mb=used_mb,
                memory_percentage=percentage,
                process_memory_mb=process_mb
            )

            # Track peak memory
            if process_mb > self.peak_memory_mb:
                self.peak_memory_mb = process_mb

            # Add to history
            self.memory_history.append(stats)

            # Keep only recent history (last 100 entries)
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-100:]

            return stats

        except Exception as e:
            tprint_error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0)

    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        if self.memory_limit_mb is None:
            return True

        stats = self.get_memory_stats()
        return stats.process_memory_mb <= self.memory_limit_mb

    def force_cleanup(self) -> None:
        """Force memory cleanup."""
        try:
            # 🔍 LOGGING: Gestion mémoire sous pression
            stats_before = self.get_memory_stats()
            tprint_info("🔍 [MEMORY] GESTION MÉMOIRE SOUS PRESSION")
            tprint_info(f"   → Mémoire avant nettoyage: {stats_before.memory_percentage:.1f}% ({stats_before.process_memory_mb:.1f}MB)")
            tprint_info("   → Seuil de déclenchement: 80% d'utilisation mémoire")
            
            # Run garbage collection
            collected = gc.collect()

            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    tprint_warning(f"Cleanup callback failed: {e}")

            # M1-specific cleanup
            if self.m1_optimizer:
                self.m1_optimizer.force_cleanup()

            # 🔍 LOGGING: Impact du nettoyage
            stats_after = self.get_memory_stats()
            memory_freed = stats_before.process_memory_mb - stats_after.process_memory_mb
            tprint_info(f"Memory cleanup completed (collected {collected} objects)")
            tprint_info(f"   → Mémoire après nettoyage: {stats_after.memory_percentage:.1f}% ({stats_after.process_memory_mb:.1f}MB)")
            tprint_info(f"   → Mémoire libérée: {memory_freed:.1f}MB")
            tprint_info(f"   → Objets collectés: {collected}")

        except Exception as e:
            tprint_error(f"Memory cleanup failed: {e}")

    def optimize_memory_usage(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize memory usage of data structures."""
        try:
            # 🔍 LOGGING: Décisions de réduction non documentées
            original_size = 0
            if isinstance(data, np.ndarray):
                original_size = data.nbytes / (1024 * 1024)  # MB
            elif isinstance(data, pd.DataFrame):
                original_size = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            tprint_info("🔍 [MEMORY] OPTIMISATION MÉMOIRE")
            tprint_info(f"   → Taille originale des données: {original_size:.1f}MB")
            tprint_info(f"   → Type de données: {type(data).__name__}")
            
            if isinstance(data, np.ndarray):
                optimized = self._optimize_numpy_array(data)
            elif isinstance(data, pd.DataFrame):
                optimized = self._optimize_dataframe(data)
            else:
                optimized = data
            
            # 🔍 LOGGING: Impact de l'optimisation
            optimized_size = 0
            if isinstance(optimized, np.ndarray):
                optimized_size = optimized.nbytes / (1024 * 1024)  # MB
            elif isinstance(optimized, pd.DataFrame):
                optimized_size = optimized.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            reduction_pct = ((original_size - optimized_size) / original_size * 100) if original_size > 0 else 0
            memory_saved = original_size - optimized_size
            
            tprint_info(f"   → Taille optimisée: {optimized_size:.1f}MB")
            tprint_info(f"   → Réduction: {memory_saved:.1f}MB ({reduction_pct:.1f}%)")
            
            return optimized

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
