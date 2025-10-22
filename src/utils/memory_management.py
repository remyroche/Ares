"""
Memory Management Module with Hardware Integration

This module provides comprehensive memory management using hardware utilities
to prevent memory leaks and optimize resource usage.
"""

import logging
import gc
import weakref
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from functools import wraps
import tracemalloc

# Import hardware utilities for memory management
try:
    from .hardware.memory_optimized_decorators import (
        memory_optimized, gc_optimized, comprehensive_memory_optimization,
        MemoryOptimizationLevel, MemoryOptimizationConfig
    )
    from .hardware.advanced_memory_manager import (
        get_advanced_memory_manager, track_memory_usage, force_garbage_collection,
        cleanup_all_memory, MemoryConfig, MemoryPressureLevel
    )
    from .hardware.enhanced_caching_system import (
        get_global_cache, CacheConfig, DataTypeOptimization
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    # Create dummy decorators if hardware not available
    def memory_optimized(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def gc_optimized(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def comprehensive_memory_optimization(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class MemoryStrategy(Enum):
    """Memory management strategies."""
    CONSERVATIVE = "conservative"  # Basic cleanup
    MODERATE = "moderate"         # Regular cleanup with monitoring
    AGGRESSIVE = "aggressive"     # Frequent cleanup with optimization
    MAXIMUM = "maximum"           # All optimizations enabled

@dataclass
class MemoryManagerConfig:
    """Configuration for memory management."""
    strategy: MemoryStrategy = MemoryStrategy.MODERATE
    enable_monitoring: bool = True
    enable_weak_references: bool = True
    enable_gc_optimization: bool = True
    enable_memory_pools: bool = True
    cleanup_interval: float = 30.0  # seconds
    memory_threshold_mb: float = 500.0
    max_memory_mb: float = 1000.0
    enable_tracemalloc: bool = False
    log_memory_usage: bool = False

class ResourceTracker:
    """Tracks resources for cleanup."""
    
    def __init__(self):
        self.resources: List[weakref.ref] = []
        self.cleanup_functions: List[Callable] = []
        self._lock = threading.Lock()
    
    def register_resource(self, resource: Any, cleanup_func: Optional[Callable] = None):
        """Register a resource for tracking."""
        with self._lock:
            if cleanup_func:
                self.cleanup_functions.append(cleanup_func)
            self.resources.append(weakref.ref(resource))
    
    def cleanup_all(self):
        """Cleanup all tracked resources."""
        with self._lock:
            # Call cleanup functions
            for cleanup_func in self.cleanup_functions:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.warning(f"Cleanup function failed: {e}")
            
            # Clear dead references
            self.resources = [ref for ref in self.resources if ref() is not None]
            
            # Clear cleanup functions
            self.cleanup_functions.clear()

class MemoryManager:
    """Comprehensive memory manager with hardware integration."""
    
    def __init__(self, config: Optional[MemoryManagerConfig] = None):
        self.config = config or MemoryManagerConfig()
        self.resource_tracker = ResourceTracker()
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.Lock()
        
        # Initialize hardware components if available
        if HARDWARE_AVAILABLE:
            self.memory_manager = get_advanced_memory_manager()
            self.cache_system = get_global_cache()
        else:
            self.memory_manager = None
            self.cache_system = None
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread."""
        if self.monitoring_thread:
            self._stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_memory_usage()
                self._cleanup_if_needed()
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
            
            self._stop_monitoring.wait(self.config.cleanup_interval)
    
    def _check_memory_usage(self):
        """Check current memory usage."""
        if not HARDWARE_AVAILABLE:
            return
        
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            current_usage = memory_stats.get('current_usage_mb', 0)
            
            if self.config.log_memory_usage:
                logger.debug(f"Memory usage: {current_usage:.2f}MB")
            
            if current_usage > self.config.memory_threshold_mb:
                logger.warning(f"High memory usage: {current_usage:.2f}MB")
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
    
    def _cleanup_if_needed(self):
        """Perform cleanup if needed."""
        if not HARDWARE_AVAILABLE:
            return
        
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            current_usage = memory_stats.get('current_usage_mb', 0)
            
            if current_usage > self.config.memory_threshold_mb:
                self.cleanup_memory()
        except Exception as e:
            logger.error(f"Failed to cleanup memory: {e}")
    
    def cleanup_memory(self):
        """Perform comprehensive memory cleanup."""
        with self._lock:
            logger.info("Performing memory cleanup")
            
            # Cleanup tracked resources
            self.resource_tracker.cleanup_all()
            
            # Force garbage collection
            if self.config.enable_gc_optimization:
                if HARDWARE_AVAILABLE:
                    force_garbage_collection()
                else:
                    gc.collect()
            
            # Clear caches if available
            if self.cache_system:
                try:
                    self.cache_system.clear_expired()
                except Exception as e:
                    logger.warning(f"Failed to clear cache: {e}")
    
    def register_resource(self, resource: Any, cleanup_func: Optional[Callable] = None):
        """Register a resource for tracking."""
        self.resource_tracker.register_resource(resource, cleanup_func)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
            'strategy': self.config.strategy.value,
            'monitoring_enabled': self.config.enable_monitoring,
            'tracked_resources': len(self.resource_tracker.resources)
        }
        
        if HARDWARE_AVAILABLE and self.memory_manager:
            try:
                hardware_stats = self.memory_manager.get_memory_stats()
                stats.update(hardware_stats)
            except Exception as e:
                logger.error(f"Failed to get hardware memory stats: {e}")
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_memory()
        self.stop_monitoring()

# Global memory manager
_global_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(config: Optional[MemoryManagerConfig] = None) -> MemoryManager:
    """Get the global memory manager."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(config)
    return _global_memory_manager

def memory_managed(strategy: MemoryStrategy = MemoryStrategy.MODERATE):
    """Decorator for memory-managed functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get memory manager
            manager = get_memory_manager()
            
            # Register function for tracking
            manager.register_resource(func)
            
            try:
                # Apply hardware optimizations if available
                if HARDWARE_AVAILABLE:
                    if strategy == MemoryStrategy.CONSERVATIVE:
                        optimized_func = gc_optimized()(func)
                    elif strategy == MemoryStrategy.MODERATE:
                        optimized_func = memory_optimized()(func)
                    elif strategy == MemoryStrategy.AGGRESSIVE:
                        optimized_func = memory_optimized(
                            MemoryOptimizationConfig(
                                optimization_level=MemoryOptimizationLevel.AGGRESSIVE
                            )
                        )(func)
                    else:  # MAXIMUM
                        optimized_func = comprehensive_memory_optimization()(func)
                else:
                    optimized_func = func
                
                # Execute function
                result = optimized_func(*args, **kwargs)
                
                # Cleanup if needed
                if strategy in [MemoryStrategy.AGGRESSIVE, MemoryStrategy.MAXIMUM]:
                    manager.cleanup_memory()
                
                return result
                
            except Exception as e:
                # Ensure cleanup on error
                manager.cleanup_memory()
                raise
        
        return wrapper
    return decorator

@contextmanager
def memory_context(strategy: MemoryStrategy = MemoryStrategy.MODERATE) -> ContextManager[MemoryManager]:
    """Context manager for memory management."""
    config = MemoryManagerConfig(strategy=strategy)
    manager = MemoryManager(config)
    
    try:
        yield manager
    finally:
        manager.cleanup_memory()
        manager.stop_monitoring()

def cleanup_all_memory():
    """Cleanup all memory resources."""
    manager = get_memory_manager()
    manager.cleanup_memory()
    
    if HARDWARE_AVAILABLE:
        try:
            cleanup_all_memory()
        except Exception as e:
            logger.error(f"Failed to cleanup hardware memory: {e}")

def track_memory_usage(func: Callable) -> Callable:
    """Decorator to track memory usage of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if HARDWARE_AVAILABLE:
            return track_memory_usage(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper

def force_cleanup():
    """Force immediate memory cleanup."""
    manager = get_memory_manager()
    manager.cleanup_memory()
    
    if HARDWARE_AVAILABLE:
        try:
            force_garbage_collection()
        except Exception as e:
            logger.error(f"Failed to force garbage collection: {e}")

# Convenience functions
def memory_efficient_decorator(level: str = "moderate"):
    """Create a memory-efficient decorator with specified level."""
    strategy = MemoryStrategy(level.lower())
    return memory_managed(strategy)

def gc_optimized_decorator():
    """Create a GC-optimized decorator."""
    return memory_managed(MemoryStrategy.CONSERVATIVE)

def aggressive_memory_decorator():
    """Create an aggressive memory management decorator."""
    return memory_managed(MemoryStrategy.AGGRESSIVE)

def maximum_memory_decorator():
    """Create a maximum memory management decorator."""
    return memory_managed(MemoryStrategy.MAXIMUM)

# Export main classes and functions
__all__ = [
    'MemoryStrategy', 'MemoryManagerConfig', 'ResourceTracker', 'MemoryManager',
    'get_memory_manager', 'memory_managed', 'memory_context', 'cleanup_all_memory',
    'track_memory_usage', 'force_cleanup', 'memory_efficient_decorator',
    'gc_optimized_decorator', 'aggressive_memory_decorator', 'maximum_memory_decorator'
]