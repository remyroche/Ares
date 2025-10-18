"""
Memory Pool Optimizer

This module provides memory pool optimization for repeated VectorBT operations,
reducing memory allocation overhead and improving performance for feature generation.
"""

import logging
import time
import gc
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemoryPoolConfig:
    """Configuration for memory pool optimization."""
    max_pool_size: int = 100
    max_object_size_mb: float = 10.0
    cleanup_frequency: int = 100  # Cleanup every N operations
    enable_adaptive_sizing: bool = True
    enable_thread_safety: bool = True
    memory_threshold_mb: float = 512.0
    gc_threshold: float = 0.8  # Trigger GC when memory usage exceeds this ratio

@dataclass
class PooledObject:
    """Represents an object in the memory pool."""
    obj: Any
    obj_type: Type
    size_bytes: int
    last_used: float
    use_count: int = 0
    in_use: bool = False

class MemoryPoolOptimizer:
    """
    Optimizes memory usage for repeated operations by maintaining pools of reusable objects.
    
    This class reduces memory allocation overhead by reusing objects like DataFrames,
    Series, and numpy arrays for similar operations.
    """
    
    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        """
        Initialize the memory pool optimizer.
        
        Args:
            config: Configuration for the memory pool
        """
        self.config = config or MemoryPoolConfig()
        self.logger = logger.getChild('MemoryPoolOptimizer')
        
        # Memory pools organized by object type
        self.pools: Dict[Type, deque] = defaultdict(deque)
        self.pool_stats: Dict[Type, Dict[str, int]] = defaultdict(lambda: {
            'created': 0,
            'reused': 0,
            'discarded': 0,
            'current_size': 0
        })
        
        # Thread safety
        self.lock = threading.Lock() if self.config.enable_thread_safety else None
        self.operation_count = 0
        
        # Memory monitoring
        self.total_memory_usage = 0
        self.peak_memory_usage = 0
        
        self.logger.info("Memory pool optimizer initialized")
    
    def get_object(self, obj_type: Type, size_hint: Optional[tuple] = None, **kwargs) -> Any:
        """
        Get an object from the pool or create a new one.
        
        Args:
            obj_type: Type of object to get
            size_hint: Hint about the size/shape of the object
            **kwargs: Additional arguments for object creation
            
        Returns:
            Object from pool or newly created object
        """
        if self.lock:
            with self.lock:
                return self._get_object_unsafe(obj_type, size_hint, **kwargs)
        else:
            return self._get_object_unsafe(obj_type, size_hint, **kwargs)
    
    def _get_object_unsafe(self, obj_type: Type, size_hint: Optional[tuple] = None, **kwargs) -> Any:
        """
        Get an object from the pool (unsafe version, must be called with lock if needed).
        
        Args:
            obj_type: Type of object to get
            size_hint: Hint about the size/shape of the object
            **kwargs: Additional arguments for object creation
            
        Returns:
            Object from pool or newly created object
        """
        pool = self.pools[obj_type]
        
        # Try to find a suitable object in the pool
        for i, pooled_obj in enumerate(pool):
            if not pooled_obj.in_use and self._is_suitable_size(pooled_obj, size_hint):
                # Found a suitable object
                pooled_obj.in_use = True
                pooled_obj.last_used = time.time()
                pooled_obj.use_count += 1
                
                # Move to end of pool (most recently used)
                pool.remove(pooled_obj)
                pool.append(pooled_obj)
                
                self.pool_stats[obj_type]['reused'] += 1
                self.logger.debug(f"Reused {obj_type.__name__} object from pool")
                
                return pooled_obj.obj
        
        # No suitable object found, create a new one
        new_obj = self._create_object(obj_type, size_hint, **kwargs)
        
        if new_obj is not None:
            obj_size = self._estimate_object_size(new_obj)
            
            # Check if we should add to pool
            if (len(pool) < self.config.max_pool_size and 
                obj_size < self.config.max_object_size_mb * 1024 * 1024):
                
                pooled_obj = PooledObject(
                    obj=new_obj,
                    obj_type=obj_type,
                    size_bytes=obj_size,
                    last_used=time.time(),
                    use_count=1,
                    in_use=True
                )
                
                pool.append(pooled_obj)
                self.pool_stats[obj_type]['created'] += 1
                self.pool_stats[obj_type]['current_size'] += 1
                self.total_memory_usage += obj_size
                
                self.logger.debug(f"Created new {obj_type.__name__} object and added to pool")
            else:
                self.pool_stats[obj_type]['created'] += 1
                self.logger.debug(f"Created new {obj_type.__name__} object (not pooled)")
        
        return new_obj
    
    def return_object(self, obj: Any) -> None:
        """
        Return an object to the pool for reuse.
        
        Args:
            obj: Object to return to pool
        """
        if self.lock:
            with self.lock:
                self._return_object_unsafe(obj)
        else:
            self._return_object_unsafe(obj)
    
    def _return_object_unsafe(self, obj: Any) -> None:
        """
        Return an object to the pool (unsafe version, must be called with lock if needed).
        
        Args:
            obj: Object to return to pool
        """
        obj_type = type(obj)
        pool = self.pools[obj_type]
        
        # Find the object in the pool
        for pooled_obj in pool:
            if pooled_obj.obj is obj:
                pooled_obj.in_use = False
                pooled_obj.last_used = time.time()
                
                # Clean the object if it's a DataFrame or Series
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    obj.drop(obj.index, inplace=True)
                    if isinstance(obj, pd.DataFrame):
                        obj.drop(obj.columns, axis=1, inplace=True)
                
                self.logger.debug(f"Returned {obj_type.__name__} object to pool")
                return
        
        # Object not found in pool, just clean up
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            del obj
    
    def _is_suitable_size(self, pooled_obj: PooledObject, size_hint: Optional[tuple]) -> bool:
        """
        Check if a pooled object is suitable for the given size hint.
        
        Args:
            pooled_obj: Pooled object to check
            size_hint: Size hint for the operation
            
        Returns:
            True if the object is suitable
        """
        if size_hint is None:
            return True
        
        obj = pooled_obj.obj
        
        # Check DataFrame shape compatibility
        if isinstance(obj, pd.DataFrame) and len(size_hint) >= 2:
            if obj.shape[0] >= size_hint[0] and obj.shape[1] >= size_hint[1]:
                return True
        
        # Check Series length compatibility
        elif isinstance(obj, pd.Series) and len(size_hint) >= 1:
            if len(obj) >= size_hint[0]:
                return True
        
        # Check numpy array shape compatibility
        elif isinstance(obj, np.ndarray) and hasattr(obj, 'shape'):
            if len(obj.shape) == len(size_hint):
                if all(obj.shape[i] >= size_hint[i] for i in range(len(size_hint))):
                    return True
        
        return False
    
    def _create_object(self, obj_type: Type, size_hint: Optional[tuple], **kwargs) -> Any:
        """
        Create a new object of the specified type.
        
        Args:
            obj_type: Type of object to create
            size_hint: Hint about the size/shape of the object
            **kwargs: Additional arguments for object creation
            
        Returns:
            Newly created object
        """
        try:
            if obj_type == pd.DataFrame:
                if size_hint and len(size_hint) >= 2:
                    return pd.DataFrame(index=range(size_hint[0]), columns=range(size_hint[1]))
                else:
                    return pd.DataFrame()
            
            elif obj_type == pd.Series:
                if size_hint and len(size_hint) >= 1:
                    return pd.Series(index=range(size_hint[0]))
                else:
                    return pd.Series(dtype=float)
            
            elif obj_type == np.ndarray:
                if size_hint:
                    return np.empty(size_hint, dtype=kwargs.get('dtype', float))
                else:
                    return np.array([])
            
            else:
                # Try to create with size_hint as first argument
                if size_hint:
                    return obj_type(size_hint, **kwargs)
                else:
                    return obj_type(**kwargs)
                    
        except Exception as e:
            self.logger.warning(f"Failed to create object of type {obj_type}: {e}")
            return None
    
    def _estimate_object_size(self, obj: Any) -> int:
        """
        Estimate the memory size of an object in bytes.
        
        Args:
            obj: Object to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            if hasattr(obj, 'memory_usage'):
                return obj.memory_usage(deep=True)
            elif hasattr(obj, 'nbytes'):
                return obj.nbytes
            else:
                return len(str(obj))  # Rough estimate
        except:
            return 1024  # Default estimate
    
    def cleanup(self, force: bool = False) -> None:
        """
        Clean up the memory pool by removing unused objects.
        
        Args:
            force: If True, clean up all unused objects regardless of age
        """
        if self.lock:
            with self.lock:
                self._cleanup_unsafe(force)
        else:
            self._cleanup_unsafe(force)
    
    def _cleanup_unsafe(self, force: bool = False) -> None:
        """
        Clean up the memory pool (unsafe version, must be called with lock if needed).
        
        Args:
            force: If True, clean up all unused objects regardless of age
        """
        current_time = time.time()
        cleanup_threshold = 300.0  # 5 minutes
        
        for obj_type, pool in self.pools.items():
            to_remove = []
            
            for pooled_obj in pool:
                if not pooled_obj.in_use:
                    age = current_time - pooled_obj.last_used
                    if force or age > cleanup_threshold:
                        to_remove.append(pooled_obj)
            
            # Remove old objects
            for pooled_obj in to_remove:
                pool.remove(pooled_obj)
                self.total_memory_usage -= pooled_obj.size_bytes
                self.pool_stats[obj_type]['discarded'] += 1
                self.pool_stats[obj_type]['current_size'] -= 1
                
                # Clean up the object
                del pooled_obj.obj
                del pooled_obj
        
        # Force garbage collection if memory usage is high
        if self.total_memory_usage > self.config.memory_threshold_mb * 1024 * 1024:
            gc.collect()
        
        self.logger.debug(f"Cleanup completed. Memory usage: {self.total_memory_usage / 1024 / 1024:.1f}MB")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory pool.
        
        Returns:
            Dictionary of pool statistics
        """
        stats = {
            'total_memory_mb': self.total_memory_usage / 1024 / 1024,
            'peak_memory_mb': self.peak_memory_usage / 1024 / 1024,
            'pool_stats': dict(self.pool_stats),
            'total_objects': sum(stat['current_size'] for stat in self.pool_stats.values()),
            'operation_count': self.operation_count
        }
        
        return stats
    
    def increment_operation_count(self) -> None:
        """Increment the operation count and trigger cleanup if needed."""
        self.operation_count += 1
        
        if self.operation_count % self.config.cleanup_frequency == 0:
            self.cleanup()
    
    @contextmanager
    def get_dataframe(self, rows: int, cols: int, **kwargs):
        """
        Context manager for getting and returning a DataFrame.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            **kwargs: Additional arguments for DataFrame creation
            
        Yields:
            DataFrame from pool or newly created
        """
        df = self.get_object(pd.DataFrame, (rows, cols), **kwargs)
        try:
            yield df
        finally:
            self.return_object(df)
    
    @contextmanager
    def get_series(self, length: int, **kwargs):
        """
        Context manager for getting and returning a Series.
        
        Args:
            length: Length of the Series
            **kwargs: Additional arguments for Series creation
            
        Yields:
            Series from pool or newly created
        """
        series = self.get_object(pd.Series, (length,), **kwargs)
        try:
            yield series
        finally:
            self.return_object(series)

# Global memory pool instance
_global_memory_pool = None

def get_global_memory_pool() -> MemoryPoolOptimizer:
    """Get the global memory pool optimizer instance."""
    global _global_memory_pool
    if _global_memory_pool is None:
        _global_memory_pool = MemoryPoolOptimizer()
    return _global_memory_pool

def cleanup_global_memory_pool() -> None:
    """Clean up the global memory pool."""
    global _global_memory_pool
    if _global_memory_pool is not None:
        _global_memory_pool.cleanup(force=True)
