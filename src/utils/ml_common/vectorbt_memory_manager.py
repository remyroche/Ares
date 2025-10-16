"""
VectorBT Memory Manager

This module provides centralized memory management for VectorBT operations
to optimize memory usage and prevent out-of-memory errors.
"""

import psutil
import gc
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_gb: float = 8.0
    warning_threshold: float = 0.8  # 80% of max memory
    critical_threshold: float = 0.95  # 95% of max memory
    enable_compression: bool = True
    enable_memory_mapping: bool = True
    chunk_size_multiplier: float = 0.1  # 10% of available memory
    cleanup_frequency: int = 10  # Cleanup every 10 operations

class VectorBTMemoryManager:
    """
    Centralized memory management for VectorBT operations.

    This class provides:
    - Memory allocation tracking
    - Automatic cleanup
    - Memory optimization strategies
    - Out-of-memory prevention
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize VectorBT memory manager."""
        self.config = config or MemoryConfig()
        self.logger = logger.getChild('VectorBTMemoryManager')

        # Get system memory info
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Set max memory based on system capabilities
        if self.config.max_memory_gb > self.total_memory_gb * 0.8:
            self.config.max_memory_gb = self.total_memory_gb * 0.8
            self.logger.warning(f"Adjusted max memory to {self.config.max_memory_gb:.2f}GB based on system")

        # Memory tracking
        self.current_usage_gb = 0.0
        self.allocated_objects: Dict[str, Dict[str, Any]] = {}
        self.operation_count = 0
        self.cleanup_count = 0

        # Performance tracking
        self.memory_stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'peak_usage_gb': 0.0,
            'cleanup_operations': 0,
            'compression_saves_gb': 0.0,
            'memory_mapping_saves_gb': 0.0
        }

        self.logger.info(f"✅ VectorBT Memory Manager initialized")
        self.logger.info(f"📊 Total system memory: {self.total_memory_gb:.2f}GB")
        self.logger.info(f"📊 Available memory: {self.available_memory_gb:.2f}GB")
        self.logger.info(f"📊 Max allocated memory: {self.config.max_memory_gb:.2f}GB")

    def can_allocate(self, size_gb: float, operation_type: str = "general") -> bool:
        """
        Check if we can allocate memory for an operation.

        Args:
            size_gb: Size to allocate in GB
            operation_type: Type of operation for prioritization

        Returns:
            True if allocation is possible
        """
        # Check if we have enough memory
        if self.current_usage_gb + size_gb > self.config.max_memory_gb:
            self.logger.warning(f"⚠️ Cannot allocate {size_gb:.2f}GB - would exceed limit")
            return False

        # Check system memory
        current_system_usage = psutil.virtual_memory().percent / 100
        if current_system_usage > self.config.critical_threshold:
            self.logger.warning(f"⚠️ System memory usage critical: {current_system_usage:.1%}")
            return False

        return True

    def allocate(self, size_gb: float, key: str, operation_type: str = "general",
                priority: int = 1, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Allocate memory with tracking.

        Args:
            size_gb: Size to allocate in GB
            key: Unique key for this allocation
            operation_type: Type of operation
            priority: Priority level (1=highest, 5=lowest)
            metadata: Additional metadata

        Returns:
            True if allocation successful
        """
        if not self.can_allocate(size_gb, operation_type):
            return False

        # Check if key already exists
        if key in self.allocated_objects:
            self.logger.warning(f"⚠️ Key {key} already exists, deallocating first")
            self.deallocate(key)

        # Allocate memory
        self.current_usage_gb += size_gb
        self.allocated_objects[key] = {
            'size_gb': size_gb,
            'operation_type': operation_type,
            'priority': priority,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        # Update peak usage
        self.memory_stats['peak_usage_gb'] = max(
            self.memory_stats['peak_usage_gb'],
            self.current_usage_gb
        )

        self.memory_stats['total_allocations'] += 1
        self.operation_count += 1

        # Check if cleanup is needed
        if self.operation_count % self.config.cleanup_frequency == 0:
            self._cleanup_memory()

        self.logger.debug(f"✅ Allocated {size_gb:.2f}GB for {key} ({operation_type})")
        return True

    def deallocate(self, key: str) -> bool:
        """
        Deallocate memory.

        Args:
            key: Key of allocation to deallocate

        Returns:
            True if deallocation successful
        """
        if key not in self.allocated_objects:
            self.logger.warning(f"⚠️ Key {key} not found for deallocation")
            return False

        # Get allocation info
        allocation = self.allocated_objects[key]
        size_gb = allocation['size_gb']

        # Deallocate
        self.current_usage_gb -= size_gb
        del self.allocated_objects[key]

        self.memory_stats['total_deallocations'] += 1

        self.logger.debug(f"✅ Deallocated {size_gb:.2f}GB for {key}")
        return True

    def get_optimal_chunk_size(self, total_size: int, data_type: str = "float64") -> int:
        """
        Calculate optimal chunk size based on available memory.

        Args:
            total_size: Total size of data
            data_type: Data type for size calculation

        Returns:
            Optimal chunk size
        """
        # Calculate bytes per element
        if data_type == "float64":
            bytes_per_element = 8
        elif data_type == "float32":
            bytes_per_element = 4
        elif data_type == "int64":
            bytes_per_element = 8
        elif data_type == "int32":
            bytes_per_element = 4
        else:
            bytes_per_element = 8  # Default

        # Calculate available memory for chunks
        available_memory_gb = self.config.max_memory_gb - self.current_usage_gb
        available_memory_bytes = available_memory_gb * (1024**3)

        # Calculate chunk size (use 10% of available memory)
        chunk_memory_bytes = available_memory_bytes * self.config.chunk_size_multiplier
        max_elements_per_chunk = int(chunk_memory_bytes / bytes_per_element)

        # Ensure chunk size is reasonable
        chunk_size = min(max_elements_per_chunk, total_size)
        chunk_size = max(chunk_size, 1000)  # Minimum chunk size

        self.logger.debug(f"📊 Optimal chunk size: {chunk_size} elements ({data_type})")
        return chunk_size

    def optimize_data_types(self, data: Any) -> Any:
        """
        Optimize data types for memory efficiency.

        Args:
            data: Data to optimize

        Returns:
            Optimized data
        """
        if not NUMPY_AVAILABLE:
            return data

        if isinstance(data, np.ndarray):
            return self._optimize_numpy_array(data)
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return self._optimize_dataframe(data)
        else:
            return data

    def _optimize_numpy_array(self, arr: np.ndarray) -> np.ndarray:
        """Optimize numpy array data types."""
        if arr.dtype == np.float64:
            # Check if float32 precision is sufficient
            if arr.max() < 3.4e38 and arr.min() > -3.4e38:
                self.logger.debug("📊 Converting float64 to float32 for memory efficiency")
                return arr.astype(np.float32)
        elif arr.dtype == np.int64:
            # Check if int32 range is sufficient
            if arr.max() < 2147483647 and arr.min() > -2147483648:
                self.logger.debug("📊 Converting int64 to int32 for memory efficiency")
                return arr.astype(np.int32)

        return arr

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types."""
        optimized_df = df.copy()

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == np.float64:
                if df[col].max() < 3.4e38 and df[col].min() > -3.4e38:
                    optimized_df[col] = df[col].astype(np.float32)
            elif df[col].dtype == np.int64:
                if df[col].max() < 2147483647 and df[col].min() > -2147483648:
                    optimized_df[col] = df[col].astype(np.int32)

        return optimized_df

    def _cleanup_memory(self):
        """Perform memory cleanup operations."""
        self.logger.debug("🧹 Performing memory cleanup...")

        # Force garbage collection
        gc.collect()

        # Update current usage
        self.current_usage_gb = psutil.virtual_memory().used / (1024**3)

        # Remove old allocations if memory is still high
        if self.current_usage_gb > self.config.max_memory_gb * self.config.warning_threshold:
            self._remove_low_priority_allocations()

        self.cleanup_count += 1
        self.memory_stats['cleanup_operations'] += 1

        self.logger.debug(f"✅ Memory cleanup completed. Current usage: {self.current_usage_gb:.2f}GB")

    def _remove_low_priority_allocations(self):
        """Remove low priority allocations to free memory."""
        # Sort by priority and timestamp
        sorted_allocations = sorted(
            self.allocated_objects.items(),
            key=lambda x: (x[1]['priority'], x[1]['timestamp'])
        )

        # Remove low priority allocations
        removed_count = 0
        for key, allocation in sorted_allocations:
            if allocation['priority'] > 3:  # Remove low priority
                self.deallocate(key)
                removed_count += 1

                # Stop if we've freed enough memory
                if self.current_usage_gb < self.config.max_memory_gb * self.config.warning_threshold:
                    break

        if removed_count > 0:
            self.logger.info(f"🧹 Removed {removed_count} low priority allocations")

    @contextmanager
    def memory_context(self, size_gb: float, key: str, operation_type: str = "general"):
        """
        Context manager for automatic memory management.

        Args:
            size_gb: Size to allocate in GB
            key: Unique key for this allocation
            operation_type: Type of operation
        """
        if not self.allocate(size_gb, key, operation_type):
            raise MemoryError(f"Cannot allocate {size_gb:.2f}GB for {key}")

        try:
            yield
        finally:
            self.deallocate(key)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_system_usage = psutil.virtual_memory().percent / 100

        return {
            'current_usage_gb': self.current_usage_gb,
            'max_usage_gb': self.config.max_memory_gb,
            'usage_percentage': self.current_usage_gb / self.config.max_memory_gb,
            'system_usage_percentage': current_system_usage,
            'allocated_objects': len(self.allocated_objects),
            'total_allocations': self.memory_stats['total_allocations'],
            'total_deallocations': self.memory_stats['total_deallocations'],
            'peak_usage_gb': self.memory_stats['peak_usage_gb'],
            'cleanup_operations': self.memory_stats['cleanup_operations'],
            'available_memory_gb': self.available_memory_gb - self.current_usage_gb
        }

    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        usage_percentage = self.current_usage_gb / self.config.max_memory_gb

        if usage_percentage > 0.8:
            recommendations.append("Consider reducing chunk sizes or enabling compression")

        if self.memory_stats['cleanup_operations'] > 10:
            recommendations.append("Frequent cleanup detected - consider increasing max memory limit")

        if len(self.allocated_objects) > 50:
            recommendations.append("Many active allocations - consider consolidating operations")

        return recommendations

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> VectorBTMemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = VectorBTMemoryManager()
    return _memory_manager

def optimize_memory_usage(data: Any) -> Any:
    """Convenience function to optimize data types."""
    manager = get_memory_manager()
    return manager.optimize_data_types(data)

@contextmanager
def memory_managed_operation(size_gb: float, key: str, operation_type: str = "general"):
    """Convenience context manager for memory management."""
    manager = get_memory_manager()
    with manager.memory_context(size_gb, key, operation_type):
        yield
