"""
VectorBT Memory Manager

Provides memory management for VectorBT operations.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VectorBTMemoryManager:
    """Memory manager for VectorBT operations."""
    
    def __init__(self, memory_limit_mb: int = 1000):
        """Initialize the VectorBT memory manager."""
        self.memory_limit_mb = memory_limit_mb
        self.logger = logger
    
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
        import psutil
        import os
        
        try:
            # Get current process memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            used_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Calculate available memory (limit - used)
            available_mb = max(0, self.memory_limit_mb - used_mb)
            
            return {
                'limit_mb': self.memory_limit_mb,
                'used_mb': round(used_mb, 2),
                'available_mb': round(available_mb, 2),
                'usage_percentage': round((used_mb / self.memory_limit_mb) * 100, 2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {
                'limit_mb': self.memory_limit_mb,
                'used_mb': 0.0,
                'available_mb': self.memory_limit_mb,
                'usage_percentage': 0.0,
                'error': str(e)
            }


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
        
        # Real memory optimization
        if hasattr(data, 'memory_usage'):
            usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            if usage_mb > memory_limit_mb:
                logger.warning(f"Data size {usage_mb:.2f}MB exceeds limit {memory_limit_mb}MB")
                
                # Apply memory optimization strategies
                if hasattr(data, 'astype'):
                    # Convert to more memory-efficient dtypes
                    data = data.astype('float32')
                    logger.info("Converted data to float32 for memory efficiency")
                
                if hasattr(data, 'sample') and len(data) > memory_limit_mb * 1000:
                    # Sample data if too large
                    sample_size = int(memory_limit_mb * 1000)
                    data = data.sample(n=sample_size, random_state=42)
                    logger.info(f"Sampled data to {sample_size} rows for memory efficiency")
        
        return data
        
    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return data


# Export the main classes and functions
__all__ = ['VectorBTMemoryManager', 'get_vectorbt_memory_manager', 'get_memory_manager', 'memory_managed_operation', 'optimize_memory_usage']