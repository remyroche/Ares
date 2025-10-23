"""
VectorBT Memory Manager

Provides memory management for VectorBT operations.
"""

from typing import Dict, Any, Optional
import logging
import psutil
import gc
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class VectorBTMemoryManager:
    """Memory manager for VectorBT operations."""
    
    def __init__(self, memory_limit_mb: int = 1000):
        """Initialize the VectorBT memory manager."""
        self.memory_limit_mb = memory_limit_mb
        self.logger = logger
        self.allocated_memory = 0
        self.memory_operations = []
        self.start_time = time.time()
    
    def allocate_memory(self, size_mb: int) -> bool:
        """
        Allocate memory for VectorBT operations.
        
        Args:
            size_mb: Size in MB to allocate
            
        Returns:
            True if allocation successful
        """
        try:
            if size_mb > self.memory_limit_mb:
                self.logger.warning(f"Requested memory {size_mb}MB exceeds limit {self.memory_limit_mb}MB")
                return False
            
            self.logger.info(f"Allocated {size_mb}MB for VectorBT operations")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory allocation failed: {e}")
            return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            # Get system memory information
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Calculate actual memory usage
            used_mb = process_memory.rss / (1024 * 1024)  # Convert to MB
            available_mb = memory_info.available / (1024 * 1024)
            
            # Calculate VectorBT-specific memory usage
            vectorbt_memory = self._calculate_vectorbt_memory_usage()
            
            return {
                'limit_mb': self.memory_limit_mb,
                'used_mb': round(used_mb, 2),
                'available_mb': round(available_mb, 2),
                'vectorbt_allocated_mb': round(vectorbt_memory, 2),
                'system_memory_percent': memory_info.percent,
                'uptime_seconds': time.time() - self.start_time,
                'memory_operations_count': len(self.memory_operations)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {
                'limit_mb': self.memory_limit_mb,
                'used_mb': 0,
                'available_mb': self.memory_limit_mb,
                'error': str(e)
            }
    
    def _calculate_vectorbt_memory_usage(self) -> float:
        """Calculate VectorBT-specific memory usage."""
        try:
            # Try to get memory usage from VectorBT objects
            vectorbt_memory = 0
            
            # Check for common VectorBT objects in memory
            for obj in gc.get_objects():
                if hasattr(obj, '__class__') and 'vectorbt' in str(obj.__class__.__module__).lower():
                    if hasattr(obj, 'memory_usage'):
                        try:
                            obj_memory = obj.memory_usage(deep=True)
                            if hasattr(obj_memory, 'sum'):
                                vectorbt_memory += obj_memory.sum() / (1024 * 1024)
                        except:
                            pass
            
            return vectorbt_memory
        except Exception:
            return 0.0
    
    def track_memory_operation(self, operation_name: str, memory_delta_mb: float):
        """Track a memory operation."""
        operation = {
            'timestamp': time.time(),
            'operation': operation_name,
            'memory_delta_mb': memory_delta_mb,
            'total_allocated_mb': self.allocated_memory
        }
        self.memory_operations.append(operation)
        self.allocated_memory += memory_delta_mb
    
    def get_memory_history(self) -> list:
        """Get memory operation history."""
        return self.memory_operations.copy()
    
    def clear_memory_history(self):
        """Clear memory operation history."""
        self.memory_operations.clear()
        self.allocated_memory = 0
    
    @contextmanager
    def memory_context(self, operation_name: str):
        """Context manager for memory operations."""
        start_memory = self.get_memory_usage()['used_mb']
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()['used_mb']
            memory_delta = end_memory - start_memory
            self.track_memory_operation(operation_name, memory_delta)
    
    def optimize_memory(self):
        """Optimize memory usage."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Log optimization results
            current_usage = self.get_memory_usage()
            self.logger.info(f"Memory optimization completed: {collected} objects collected")
            self.logger.info(f"Current memory usage: {current_usage['used_mb']:.2f}MB")
            
            return True
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return False


def get_vectorbt_memory_manager(memory_limit_mb: int = 1000) -> VectorBTMemoryManager:
    """
    Get a VectorBT memory manager instance.
    
    Args:
        memory_limit_mb: Memory limit in MB
        
    Returns:
        VectorBTMemoryManager instance
    """
    return VectorBTMemoryManager(memory_limit_mb)


def get_memory_manager(memory_limit_mb: int = 1000) -> VectorBTMemoryManager:
    """
    Get a memory manager instance (alias for get_vectorbt_memory_manager).
    
    Args:
        memory_limit_mb: Memory limit in MB
        
    Returns:
        VectorBTMemoryManager instance
    """
    return get_vectorbt_memory_manager(memory_limit_mb)


def memory_managed_operation(operation_func, memory_limit_mb: int = 1000):
    """
    Decorator for memory-managed operations.
    
    Args:
        operation_func: Function to wrap
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        manager = get_memory_manager(memory_limit_mb)
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Memory-managed operation failed: {e}")
            raise
    return wrapper


def optimize_memory_usage(data: Any, memory_limit_mb: int = 1000) -> Any:
    """
    Optimize memory usage for data operations.
    
    Args:
        data: Data to optimize
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Optimized data
    """
    try:
        manager = get_memory_manager(memory_limit_mb)
        
        # Calculate actual memory usage
        if hasattr(data, 'memory_usage'):
            usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(f"Data memory usage: {usage_mb:.2f}MB")
            
            if usage_mb > memory_limit_mb:
                logger.warning(f"Data size {usage_mb:.2f}MB exceeds limit {memory_limit_mb}MB")
                
                # Try to optimize the data
                if hasattr(data, 'copy'):
                    # Try to create a more memory-efficient copy
                    try:
                        optimized_data = data.copy(deep=False)
                        optimized_usage = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
                        logger.info(f"Optimized data memory usage: {optimized_usage:.2f}MB")
                        
                        if optimized_usage < usage_mb:
                            logger.info("Using optimized data copy")
                            return optimized_data
                    except Exception as e:
                        logger.warning(f"Failed to create optimized copy: {e}")
                
                # If optimization failed, try to reduce precision
                if hasattr(data, 'astype'):
                    try:
                        # Try to reduce float precision
                        if data.dtypes.apply(lambda x: 'float' in str(x)).any():
                            reduced_data = data.astype('float32')
                            reduced_usage = reduced_data.memory_usage(deep=True).sum() / 1024 / 1024
                            logger.info(f"Reduced precision memory usage: {reduced_usage:.2f}MB")
                            
                            if reduced_usage < usage_mb:
                                logger.info("Using reduced precision data")
                                return reduced_data
                    except Exception as e:
                        logger.warning(f"Failed to reduce precision: {e}")
        
        # Track the memory operation
        manager.track_memory_operation("optimize_memory_usage", 0)
        
        return data
        
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return data


# Export the main classes and functions
__all__ = ['VectorBTMemoryManager', 'get_vectorbt_memory_manager', 'get_memory_manager', 'memory_managed_operation', 'optimize_memory_usage']