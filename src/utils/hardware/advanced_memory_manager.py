"""
Advanced Memory Manager with Efficient Garbage Collection and Chunking.

This module provides advanced memory management capabilities including
intelligent garbage collection, chunking for large data processing,
memory pressure detection, and adaptive cleanup strategies.
"""

import gc
import logging
import threading
import time
import weakref
import psutil
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from contextlib import contextmanager
import tracemalloc
from functools import wraps

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ChunkingStrategy(Enum):
    """Chunking strategies for large data processing."""
    FIXED_SIZE = "fixed_size"          # Fixed chunk size
    MEMORY_AWARE = "memory_aware"      # Based on available memory
    ADAPTIVE = "adaptive"              # Adaptive based on processing speed
    STREAMING = "streaming"            # Streaming processing

@dataclass
class MemoryConfig:
    """Configuration for advanced memory management."""
    # Garbage collection
    enable_aggressive_gc: bool = True
    gc_threshold_mb: float = 100.0      # Trigger GC when memory exceeds this
    gc_interval_seconds: float = 30.0   # Regular GC interval
    gc_generation_threshold: int = 2    # GC generation threshold
    
    # Memory pressure detection
    enable_memory_pressure_detection: bool = True
    pressure_check_interval: float = 5.0
    low_pressure_threshold: float = 0.6    # 60% memory usage
    medium_pressure_threshold: float = 0.75 # 75% memory usage
    high_pressure_threshold: float = 0.85   # 85% memory usage
    critical_pressure_threshold: float = 0.95 # 95% memory usage
    
    # Chunking
    enable_chunking: bool = True
    default_chunk_size_mb: float = 50.0
    max_chunk_size_mb: float = 200.0
    min_chunk_size_mb: float = 1.0
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.MEMORY_AWARE
    
    # Memory pools
    enable_memory_pools: bool = True
    pool_size_mb: float = 100.0
    pool_cleanup_interval: float = 300.0  # 5 minutes
    
    # Weak references
    enable_weak_references: bool = True
    weak_ref_cleanup_interval: float = 60.0  # 1 minute
    
    # Memory monitoring
    enable_detailed_monitoring: bool = True
    monitoring_interval: float = 10.0
    enable_memory_tracing: bool = False

@dataclass
class MemoryStats:
    """Memory statistics."""
    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_percent: float = 0.0
    pressure_level: MemoryPressureLevel = MemoryPressureLevel.LOW
    gc_count: int = 0
    gc_collections: int = 0
    objects_tracked: int = 0
    weak_refs_count: int = 0
    pool_usage_mb: float = 0.0

class MemoryPool:
    """Memory pool for efficient object reuse."""
    
    def __init__(self, pool_size_mb: float = 100.0):
        self.pool_size_bytes = int(pool_size_mb * 1024 * 1024)
        self.pools = {
            'numpy_arrays': deque(),
            'dataframes': deque(),
            'lists': deque(),
            'dicts': deque()
        }
        self.pool_usage = 0
        self.lock = threading.RLock()
        self.logger = logger.getChild('MemoryPool')
    
    def get_numpy_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Get a numpy array from the pool or create a new one."""
        with self.lock:
            # Try to reuse from pool
            for arr in list(self.pools['numpy_arrays']):
                if arr.shape == shape and arr.dtype == dtype:
                    self.pools['numpy_arrays'].remove(arr)
                    self.pool_usage -= arr.nbytes
                    return arr
            
            # Create new array
            return np.zeros(shape, dtype=dtype)
    
    def return_numpy_array(self, arr: np.ndarray):
        """Return a numpy array to the pool."""
        if not isinstance(arr, np.ndarray):
            return
        
        with self.lock:
            if self.pool_usage + arr.nbytes <= self.pool_size_bytes:
                # Clear the array
                arr.fill(0)
                self.pools['numpy_arrays'].append(arr)
                self.pool_usage += arr.nbytes
            else:
                # Pool is full, let it be garbage collected
                del arr
    
    def get_dataframe(self, columns: List[str], index: Optional[pd.Index] = None) -> pd.DataFrame:
        """Get a DataFrame from the pool or create a new one."""
        with self.lock:
            # Try to reuse from pool
            for df in list(self.pools['dataframes']):
                if list(df.columns) == columns and (index is None or df.index.equals(index)):
                    self.pools['dataframes'].remove(df)
                    self.pool_usage -= df.memory_usage(deep=True).sum()
                    return df
            
            # Create new DataFrame
            return pd.DataFrame(columns=columns, index=index)
    
    def return_dataframe(self, df: pd.DataFrame):
        """Return a DataFrame to the pool."""
        if not isinstance(df, pd.DataFrame):
            return
        
        with self.lock:
            df_size = df.memory_usage(deep=True).sum()
            if self.pool_usage + df_size <= self.pool_size_bytes:
                # Clear the DataFrame
                df.drop(df.index, inplace=True)
                self.pools['dataframes'].append(df)
                self.pool_usage += df_size
            else:
                # Pool is full, let it be garbage collected
                del df
    
    def cleanup(self):
        """Clean up the memory pool."""
        with self.lock:
            for pool in self.pools.values():
                pool.clear()
            self.pool_usage = 0
            tprint_debug("Memory pool cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'pool_usage_mb': self.pool_usage / (1024 * 1024),
                'pool_size_mb': self.pool_size_bytes / (1024 * 1024),
                'arrays_in_pool': len(self.pools['numpy_arrays']),
                'dataframes_in_pool': len(self.pools['dataframes']),
                'lists_in_pool': len(self.pools['lists']),
                'dicts_in_pool': len(self.pools['dicts'])
            }

class ChunkingManager:
    """Manages chunking for large data processing."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logger.getChild('ChunkingManager')
        self.memory_manager = None  # Will be set by parent
    
    def calculate_chunk_size(self, data_size_mb: float, available_memory_mb: float) -> int:
        """Calculate optimal chunk size based on available memory."""
        if self.config.chunking_strategy == ChunkingStrategy.FIXED_SIZE:
            return int(self.config.default_chunk_size_mb * 1024 * 1024)
        
        elif self.config.chunking_strategy == ChunkingStrategy.MEMORY_AWARE:
            # Use 20% of available memory, but within limits
            chunk_size_mb = min(
                max(available_memory_mb * 0.2, self.config.min_chunk_size_mb),
                self.config.max_chunk_size_mb
            )
            return int(chunk_size_mb * 1024 * 1024)
        
        elif self.config.chunking_strategy == ChunkingStrategy.ADAPTIVE:
            # Adaptive based on data size and available memory
            if data_size_mb < available_memory_mb * 0.5:
                return int(data_size_mb * 1024 * 1024)  # Process all at once
            else:
                # Process in chunks
                chunk_size_mb = min(
                    available_memory_mb * 0.3,
                    self.config.max_chunk_size_mb
                )
                return int(chunk_size_mb * 1024 * 1024)
        
        else:  # STREAMING
            return int(self.config.min_chunk_size_mb * 1024 * 1024)
    
    def chunk_dataframe(self, df: pd.DataFrame, chunk_size_bytes: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Chunk a DataFrame for processing."""
        if chunk_size_bytes is None:
            available_memory = psutil.virtual_memory().available
            chunk_size_bytes = self.calculate_chunk_size(
                df.memory_usage(deep=True).sum() / (1024 * 1024),
                available_memory / (1024 * 1024)
            )
        
        # Calculate number of rows per chunk
        row_size = df.memory_usage(deep=True).sum() / len(df)
        rows_per_chunk = max(1, int(chunk_size_bytes / row_size))
        
        tprint_debug(f"Chunking DataFrame: {len(df)} rows, {rows_per_chunk} rows per chunk")
        
        for i in range(0, len(df), rows_per_chunk):
            chunk = df.iloc[i:i + rows_per_chunk].copy()
            yield chunk
            
            # Force garbage collection after each chunk
            if self.config.enable_aggressive_gc:
                gc.collect()
    
    def chunk_numpy_array(self, arr: np.ndarray, chunk_size_bytes: Optional[int] = None) -> Iterator[np.ndarray]:
        """Chunk a NumPy array for processing."""
        if chunk_size_bytes is None:
            available_memory = psutil.virtual_memory().available
            chunk_size_bytes = self.calculate_chunk_size(
                arr.nbytes / (1024 * 1024),
                available_memory / (1024 * 1024)
            )
        
        # Calculate number of elements per chunk
        element_size = arr.itemsize
        elements_per_chunk = max(1, int(chunk_size_bytes / element_size))
        
        tprint_debug(f"Chunking NumPy array: {arr.size} elements, {elements_per_chunk} elements per chunk")
        
        for i in range(0, arr.size, elements_per_chunk):
            chunk = arr.flat[i:i + elements_per_chunk].copy()
            yield chunk
            
            # Force garbage collection after each chunk
            if self.config.enable_aggressive_gc:
                gc.collect()
    
    def chunk_dict(self, data: Dict[str, Any], chunk_size_bytes: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Chunk a dictionary for processing."""
        if chunk_size_bytes is None:
            available_memory = psutil.virtual_memory().available
            chunk_size_bytes = self.calculate_chunk_size(
                sys.getsizeof(data) / (1024 * 1024),
                available_memory / (1024 * 1024)
            )
        
        items = list(data.items())
        items_per_chunk = max(1, int(chunk_size_bytes / (sys.getsizeof(items[0]) if items else 1)))
        
        tprint_debug(f"Chunking dictionary: {len(items)} items, {items_per_chunk} items per chunk")
        
        for i in range(0, len(items), items_per_chunk):
            chunk_items = items[i:i + items_per_chunk]
            chunk = dict(chunk_items)
            yield chunk
            
            # Force garbage collection after each chunk
            if self.config.enable_aggressive_gc:
                gc.collect()

class WeakReferenceManager:
    """Manages weak references for large objects."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.weak_refs = weakref.WeakSet()
        self.weak_ref_callbacks = {}
        self.lock = threading.RLock()
        self.logger = logger.getChild('WeakReferenceManager')
    
    def track_object(self, obj: Any, callback: Optional[Callable] = None) -> weakref.ref:
        """Track an object with a weak reference."""
        with self.lock:
            weak_ref = weakref.ref(obj, callback or self._default_callback)
            self.weak_refs.add(weak_ref)
            if callback:
                self.weak_ref_callbacks[weak_ref] = callback
            return weak_ref
    
    def _default_callback(self, weak_ref):
        """Default callback for weak reference cleanup."""
        tprint_debug(f"Weak reference cleaned up: {weak_ref}")
    
    def cleanup_dead_references(self):
        """Clean up dead weak references."""
        with self.lock:
            dead_refs = []
            for ref in self.weak_refs:
                if ref() is None:
                    dead_refs.append(ref)
            
            for ref in dead_refs:
                self.weak_refs.discard(ref)
                self.weak_ref_callbacks.pop(ref, None)
            
            if dead_refs:
                tprint_debug(f"Cleaned up {len(dead_refs)} dead weak references")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get weak reference statistics."""
        with self.lock:
            alive_refs = sum(1 for ref in self.weak_refs if ref() is not None)
            return {
                'total_refs': len(self.weak_refs),
                'alive_refs': alive_refs,
                'dead_refs': len(self.weak_refs) - alive_refs
            }

class AdvancedMemoryManager:
    """Advanced memory manager with efficient garbage collection and chunking."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logger.getChild('AdvancedMemoryManager')
        
        # Initialize components
        self.memory_pool = MemoryPool(self.config.pool_size_mb) if self.config.enable_memory_pools else None
        self.chunking_manager = ChunkingManager(self.config)
        self.weak_ref_manager = WeakReferenceManager(self.config)
        
        # Memory monitoring
        self.stats = MemoryStats()
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Garbage collection
        self.gc_thread = None
        self.gc_active = False
        
        # Memory tracing
        if self.config.enable_memory_tracing:
            tracemalloc.start()
        
        # Start monitoring
        if self.config.enable_memory_pressure_detection:
            self._start_memory_monitoring()
        
        if self.config.enable_aggressive_gc:
            self._start_gc_monitoring()
        
        tprint_success("✅ Advanced Memory Manager initialized")
        self.logger.info("Advanced memory manager with GC and chunking initialized")
    
    def _start_memory_monitoring(self):
        """Start memory monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._memory_monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitoring_thread.start()
        self.logger.debug("Memory monitoring started")
    
    def _start_gc_monitoring(self):
        """Start garbage collection monitoring thread."""
        if self.gc_active:
            return
        
        self.gc_active = True
        self.gc_thread = threading.Thread(
            target=self._gc_monitoring_loop,
            daemon=True,
            name="GCMonitor"
        )
        self.gc_thread.start()
        self.logger.debug("GC monitoring started")
    
    def _memory_monitoring_loop(self):
        """Memory monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_memory_stats()
                self._check_memory_pressure()
                time.sleep(self.config.pressure_check_interval)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(10)
    
    def _gc_monitoring_loop(self):
        """Garbage collection monitoring loop."""
        while self.gc_active:
            try:
                self._perform_gc_if_needed()
                time.sleep(self.config.gc_interval_seconds)
            except Exception as e:
                self.logger.error(f"GC monitoring error: {e}")
                time.sleep(10)
    
    def _update_memory_stats(self):
        """Update memory statistics."""
        try:
            memory = psutil.virtual_memory()
            self.stats.total_memory_mb = memory.total / (1024 * 1024)
            self.stats.used_memory_mb = memory.used / (1024 * 1024)
            self.stats.available_memory_mb = memory.available / (1024 * 1024)
            self.stats.memory_percent = memory.percent / 100.0
            
            # GC stats
            self.stats.gc_count = gc.get_count()[0]
            self.stats.gc_collections = sum(gc.get_count())
            
            # Object tracking
            self.stats.objects_tracked = len(gc.get_objects())
            
            # Weak reference stats
            if self.config.enable_weak_references:
                weak_stats = self.weak_ref_manager.get_stats()
                self.stats.weak_refs_count = weak_stats['total_refs']
            
            # Memory pool stats
            if self.memory_pool:
                pool_stats = self.memory_pool.get_stats()
                self.stats.pool_usage_mb = pool_stats['pool_usage_mb']
            
        except Exception as e:
            self.logger.error(f"Failed to update memory stats: {e}")
    
    def _check_memory_pressure(self):
        """Check memory pressure and trigger cleanup if needed."""
        memory_percent = self.stats.memory_percent
        
        if memory_percent >= self.config.critical_pressure_threshold:
            self.stats.pressure_level = MemoryPressureLevel.CRITICAL
            self._critical_memory_cleanup()
        elif memory_percent >= self.config.high_pressure_threshold:
            self.stats.pressure_level = MemoryPressureLevel.HIGH
            self._high_memory_cleanup()
        elif memory_percent >= self.config.medium_pressure_threshold:
            self.stats.pressure_level = MemoryPressureLevel.MEDIUM
            self._medium_memory_cleanup()
        else:
            self.stats.pressure_level = MemoryPressureLevel.LOW
    
    def _critical_memory_cleanup(self):
        """Critical memory cleanup - aggressive measures."""
        tprint_warning("🚨 CRITICAL memory pressure detected - performing aggressive cleanup")
        
        # Force garbage collection
        self._force_gc_all_generations()
        
        # Clear memory pools
        if self.memory_pool:
            self.memory_pool.cleanup()
        
        # Clean up weak references
        if self.config.enable_weak_references:
            self.weak_ref_manager.cleanup_dead_references()
        
        # Clear caches if available
        try:
            from .enhanced_caching_system import get_global_cache
            cache = get_global_cache()
            cache._aggressive_cleanup()
        except ImportError:
            pass
    
    def _high_memory_cleanup(self):
        """High memory cleanup - moderate measures."""
        tprint_warning("⚠️ HIGH memory pressure detected - performing cleanup")
        
        # Force garbage collection
        self._force_gc_all_generations()
        
        # Clean up weak references
        if self.config.enable_weak_references:
            self.weak_ref_manager.cleanup_dead_references()
    
    def _medium_memory_cleanup(self):
        """Medium memory cleanup - light measures."""
        tprint_debug("Medium memory pressure detected - performing light cleanup")
        
        # Light garbage collection
        gc.collect()
        
        # Clean up dead weak references
        if self.config.enable_weak_references:
            self.weak_ref_manager.cleanup_dead_references()
    
    def _perform_gc_if_needed(self):
        """Perform garbage collection if needed."""
        if self.stats.used_memory_mb > self.config.gc_threshold_mb:
            self._force_gc_all_generations()
    
    def _force_gc_all_generations(self):
        """Force garbage collection on all generations."""
        before_count = sum(gc.get_count())
        collected = gc.collect()
        after_count = sum(gc.get_count())
        
        self.stats.gc_collections += collected
        
        if collected > 0:
            tprint_debug(f"GC collected {collected} objects")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        self._update_memory_stats()
        return self.stats
    
    def get_detailed_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information."""
        self._update_memory_stats()
        
        info = {
            'memory_stats': {
                'total_mb': self.stats.total_memory_mb,
                'used_mb': self.stats.used_memory_mb,
                'available_mb': self.stats.available_memory_mb,
                'percent': self.stats.memory_percent,
                'pressure_level': self.stats.pressure_level.value
            },
            'gc_stats': {
                'gc_count': self.stats.gc_count,
                'gc_collections': self.stats.gc_collections,
                'objects_tracked': self.stats.objects_tracked
            }
        }
        
        if self.config.enable_weak_references:
            info['weak_refs'] = self.weak_ref_manager.get_stats()
        
        if self.memory_pool:
            info['memory_pool'] = self.memory_pool.get_stats()
        
        return info
    
    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for memory-aware operations."""
        start_memory = self.stats.used_memory_mb
        start_objects = len(gc.get_objects())
        
        tprint_debug(f"Starting {operation_name} - Memory: {start_memory:.1f}MB, Objects: {start_objects}")
        
        try:
            yield self
        finally:
            # Force cleanup after operation
            if self.config.enable_aggressive_gc:
                self._force_gc_all_generations()
            
            end_memory = self.stats.used_memory_mb
            end_objects = len(gc.get_objects())
            memory_delta = end_memory - start_memory
            object_delta = end_objects - start_objects
            
            tprint_debug(f"Completed {operation_name} - Memory delta: {memory_delta:+.1f}MB, "
                        f"Objects delta: {object_delta:+d}")
    
    def chunk_data(self, data: Any, chunk_size_bytes: Optional[int] = None) -> Iterator[Any]:
        """Chunk data for processing."""
        if isinstance(data, pd.DataFrame):
            return self.chunking_manager.chunk_dataframe(data, chunk_size_bytes)
        elif isinstance(data, np.ndarray):
            return self.chunking_manager.chunk_numpy_array(data, chunk_size_bytes)
        elif isinstance(data, dict):
            return self.chunking_manager.chunk_dict(data, chunk_size_bytes)
        else:
            # For other types, yield as single chunk
            yield data
    
    def process_in_chunks(self, data: Any, process_func: Callable, 
                         chunk_size_bytes: Optional[int] = None) -> List[Any]:
        """Process data in chunks with automatic memory management."""
        results = []
        
        with self.memory_context("chunked_processing"):
            for i, chunk in enumerate(self.chunk_data(data, chunk_size_bytes)):
                tprint_debug(f"Processing chunk {i+1}")
                
                # Process chunk
                result = process_func(chunk)
                results.append(result)
                
                # Clean up chunk
                del chunk
                
                # Force GC after each chunk
                if self.config.enable_aggressive_gc:
                    gc.collect()
        
        return results
    
    def track_object(self, obj: Any, callback: Optional[Callable] = None) -> weakref.ref:
        """Track an object with weak reference."""
        if not self.config.enable_weak_references:
            return weakref.ref(obj)
        
        return self.weak_ref_manager.track_object(obj, callback)
    
    def get_memory_pool(self) -> Optional[MemoryPool]:
        """Get the memory pool for object reuse."""
        return self.memory_pool
    
    def cleanup_all(self):
        """Perform comprehensive cleanup."""
        tprint_info("Performing comprehensive memory cleanup")
        
        # Force garbage collection
        self._force_gc_all_generations()
        
        # Clean up memory pools
        if self.memory_pool:
            self.memory_pool.cleanup()
        
        # Clean up weak references
        if self.config.enable_weak_references:
            self.weak_ref_manager.cleanup_dead_references()
        
        # Clear caches
        try:
            from .enhanced_caching_system import get_global_cache
            cache = get_global_cache()
            cache.clear()
        except ImportError:
            pass
        
        tprint_success("Memory cleanup completed")
    
    def shutdown(self):
        """Shutdown the memory manager."""
        self.monitoring_active = False
        self.gc_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        if self.gc_thread:
            self.gc_thread.join(timeout=2.0)
        
        self.cleanup_all()
        
        if self.config.enable_memory_tracing:
            tracemalloc.stop()
        
        tprint_info("Advanced memory manager shutdown complete")

# Global instance
_global_memory_manager: Optional[AdvancedMemoryManager] = None

def get_advanced_memory_manager(config: Optional[MemoryConfig] = None) -> AdvancedMemoryManager:
    """Get or create the global advanced memory manager."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = AdvancedMemoryManager(config)
    
    return _global_memory_manager

# Convenience functions
def memory_efficient_processing(func: Callable) -> Callable:
    """Decorator for memory-efficient processing with automatic chunking."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = get_advanced_memory_manager()
        
        with manager.memory_context(func.__name__):
            return func(*args, **kwargs)
    
    return wrapper

def chunked_processing(chunk_size_mb: Optional[float] = None):
    """Decorator for chunked processing of large data."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            manager = get_advanced_memory_manager()
            
            if chunk_size_mb:
                chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
            else:
                chunk_size_bytes = None
            
            return manager.process_in_chunks(data, func, chunk_size_bytes)
        
        return wrapper
    return decorator

def track_memory_usage(func: Callable) -> Callable:
    """Decorator to track memory usage of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        manager = get_advanced_memory_manager()
        
        start_stats = manager.get_memory_stats()
        result = func(*args, **kwargs)
        end_stats = manager.get_memory_stats()
        
        memory_delta = end_stats.used_memory_mb - start_stats.used_memory_mb
        tprint_performance(f"{func.__name__} memory delta: {memory_delta:+.1f}MB")
        
        return result
    
    return wrapper