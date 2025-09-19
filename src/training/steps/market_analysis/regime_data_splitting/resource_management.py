"""
Resource management utilities for regime data splitting module.

This module provides consistent resource management patterns, including memory management,
hardware optimization cleanup, and proper resource lifecycle management.
"""

import gc
import logging
import weakref
from typing import Dict, List, Optional, Any, Callable, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import threading
import traceback


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_objects: int = 0
    cleanup_operations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ResourceManager:
    """Centralized resource manager for regime data splitting components."""
    
    def __init__(self, component_name: str, logger: Optional[logging.Logger] = None):
        self.component_name = component_name
        self.logger = logger or logging.getLogger(__name__)
        self._cleanup_functions: List[Callable] = []
        self._managed_objects: List[weakref.ref] = []
        self._hardware_managers: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._metrics = ResourceMetrics()
        self._cleanup_performed = False
    
    def register_cleanup_function(self, cleanup_func: Callable, description: str = "") -> None:
        """Register a cleanup function to be called during resource cleanup."""
        with self._lock:
            self._cleanup_functions.append(cleanup_func)
            self.logger.debug(f"Registered cleanup function: {description}")
    
    def register_managed_object(self, obj: Any, description: str = "") -> None:
        """Register an object for lifecycle management."""
        with self._lock:
            # Use weak reference to avoid circular dependencies
            ref = weakref.ref(obj)
            self._managed_objects.append(ref)
            self.logger.debug(f"Registered managed object: {description}")
    
    def register_hardware_manager(self, name: str, manager: Any) -> None:
        """Register a hardware manager for cleanup."""
        with self._lock:
            self._hardware_managers[name] = manager
            self.logger.debug(f"Registered hardware manager: {name}")
    
    @contextmanager
    def memory_context(self, description: str = ""):
        """Context manager for memory-intensive operations."""
        self.logger.debug(f"Entering memory context: {description}")
        initial_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            # Force garbage collection
            collected = gc.collect()
            final_memory = self._get_memory_usage()
            memory_freed = initial_memory - final_memory
            
            self.logger.debug(f"Memory context cleanup: {description}")
            self.logger.debug(f"  Objects collected: {collected}")
            self.logger.debug(f"  Memory freed: {memory_freed:.2f} MB")
            
            self._metrics.cleanup_operations += 1
    
    @contextmanager
    def gpu_context(self, description: str = ""):
        """Context manager for GPU operations."""
        self.logger.debug(f"Entering GPU context: {description}")
        
        try:
            yield
        finally:
            # Clean up GPU resources if available
            self._cleanup_gpu_resources()
            self.logger.debug(f"GPU context cleanup completed: {description}")
    
    def optimize_memory(self, force_gc: bool = True) -> Dict[str, Any]:
        """Optimize memory usage with optional garbage collection."""
        initial_memory = self._get_memory_usage()
        initial_objects = len(gc.get_objects())
        
        # Clean up dead weak references
        with self._lock:
            self._managed_objects = [ref for ref in self._managed_objects if ref() is not None]
        
        # Force garbage collection if requested
        collected_objects = 0
        if force_gc:
            collected_objects = gc.collect()
        
        final_memory = self._get_memory_usage()
        final_objects = len(gc.get_objects())
        
        optimization_result = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': initial_memory - final_memory,
            'initial_objects': initial_objects,
            'final_objects': final_objects,
            'objects_collected': collected_objects,
            'managed_objects_alive': len(self._managed_objects)
        }
        
        self.logger.info(f"Memory optimization completed: {optimization_result}")
        self._metrics.cleanup_operations += 1
        
        return optimization_result
    
    def cleanup(self, force: bool = False) -> Dict[str, Any]:
        """Perform comprehensive resource cleanup."""
        if self._cleanup_performed and not force:
            self.logger.debug("Cleanup already performed, skipping")
            return {'status': 'already_cleaned'}
        
        cleanup_results = {
            'cleanup_functions_executed': 0,
            'hardware_managers_cleaned': 0,
            'errors': [],
            'warnings': []
        }
        
        self.logger.info(f"Starting resource cleanup for {self.component_name}")
        
        # Execute registered cleanup functions
        with self._lock:
            for i, cleanup_func in enumerate(self._cleanup_functions):
                try:
                    cleanup_func()
                    cleanup_results['cleanup_functions_executed'] += 1
                    self.logger.debug(f"Executed cleanup function {i+1}")
                except Exception as e:
                    error_msg = f"Cleanup function {i+1} failed: {str(e)}"
                    cleanup_results['errors'].append(error_msg)
                    self.logger.warning(error_msg)
        
        # Clean up hardware managers
        for name, manager in self._hardware_managers.items():
            try:
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
                elif hasattr(manager, 'close'):
                    manager.close()
                elif hasattr(manager, 'shutdown'):
                    manager.shutdown()
                
                cleanup_results['hardware_managers_cleaned'] += 1
                self.logger.debug(f"Cleaned up hardware manager: {name}")
            except Exception as e:
                error_msg = f"Hardware manager {name} cleanup failed: {str(e)}"
                cleanup_results['errors'].append(error_msg)
                self.logger.warning(error_msg)
        
        # Clean up GPU resources
        try:
            self._cleanup_gpu_resources()
        except Exception as e:
            cleanup_results['warnings'].append(f"GPU cleanup warning: {str(e)}")
        
        # Final memory optimization
        try:
            memory_result = self.optimize_memory(force_gc=True)
            cleanup_results['memory_optimization'] = memory_result
        except Exception as e:
            cleanup_results['errors'].append(f"Memory optimization failed: {str(e)}")
        
        self._cleanup_performed = True
        self.logger.info(f"Resource cleanup completed: {cleanup_results}")
        
        return cleanup_results
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics."""
        self._metrics.memory_usage_mb = self._get_memory_usage()
        self._metrics.active_objects = len([ref for ref in self._managed_objects if ref() is not None])
        self._metrics.timestamp = datetime.now().isoformat()
        
        # Try to get GPU memory if available
        try:
            self._metrics.gpu_memory_mb = self._get_gpu_memory_usage()
        except Exception:
            self._metrics.gpu_memory_mb = 0.0
        
        return self._metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            # Fallback: use gc stats
            return len(gc.get_objects()) * 0.001  # Rough estimate
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        try:
            # Try to get M1 GPU memory usage
            if 'gpu_manager' in self._hardware_managers:
                gpu_manager = self._hardware_managers['gpu_manager']
                if hasattr(gpu_manager, 'get_memory_usage'):
                    return gpu_manager.get_memory_usage()
        except Exception:
            pass
        
        return 0.0
    
    def _cleanup_gpu_resources(self) -> None:
        """Clean up GPU resources."""
        try:
            # Try M1 GPU cleanup
            if 'gpu_manager' in self._hardware_managers:
                gpu_manager = self._hardware_managers['gpu_manager']
                if hasattr(gpu_manager, 'cleanup_resources'):
                    gpu_manager.cleanup_resources()
        except Exception as e:
            self.logger.debug(f"GPU cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            self.logger.warning(f"Context manager cleanup failed: {e}")
    
    def __del__(self):
        """Destructor with safe cleanup."""
        try:
            if not self._cleanup_performed:
                self.cleanup()
        except Exception:
            # Ignore errors in destructor to avoid issues during interpreter shutdown
            pass


# Hardware-specific resource management utilities
class M1ResourceManager:
    """M1-specific resource management utilities."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._monitoring_active = False
        self._gpu_manager = None
        self._memory_optimizer = None
        self._cpu_optimizer = None
    
    def initialize_m1_resources(self) -> Dict[str, Any]:
        """Initialize M1-specific resources."""
        initialization_result = {
            'gpu_available': False,
            'memory_optimizer_available': False,
            'cpu_optimizer_available': False,
            'monitoring_started': False
        }
        
        try:
            # Try to initialize M1 GPU manager
            from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
            if is_m1_available():
                self._gpu_manager = get_m1_gpu_manager()
                initialization_result['gpu_available'] = True
        except Exception as e:
            self.logger.debug(f"M1 GPU initialization failed: {e}")
        
        try:
            # Try to initialize M1 memory optimizer
            from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, start_m1_memory_monitoring
            self._memory_optimizer = get_m1_memory_optimizer()
            start_m1_memory_monitoring()
            self._monitoring_active = True
            initialization_result['memory_optimizer_available'] = True
            initialization_result['monitoring_started'] = True
        except Exception as e:
            self.logger.debug(f"M1 memory optimizer initialization failed: {e}")
        
        try:
            # Try to initialize M1 CPU optimizer
            from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
            self._cpu_optimizer = get_m1_cpu_optimizer()
            initialization_result['cpu_optimizer_available'] = True
        except Exception as e:
            self.logger.debug(f"M1 CPU optimizer initialization failed: {e}")
        
        return initialization_result
    
    def cleanup_m1_resources(self) -> Dict[str, Any]:
        """Clean up M1-specific resources."""
        cleanup_result = {
            'gpu_cleaned': False,
            'memory_monitoring_stopped': False,
            'cpu_cleaned': False,
            'errors': []
        }
        
        # Clean up GPU resources
        if self._gpu_manager:
            try:
                if hasattr(self._gpu_manager, 'cleanup'):
                    self._gpu_manager.cleanup()
                cleanup_result['gpu_cleaned'] = True
            except Exception as e:
                cleanup_result['errors'].append(f"GPU cleanup failed: {e}")
        
        # Stop memory monitoring
        if self._monitoring_active:
            try:
                from src.utils.hardware.m1_memory_optimizer import stop_m1_memory_monitoring
                stop_m1_memory_monitoring()
                self._monitoring_active = False
                cleanup_result['memory_monitoring_stopped'] = True
            except Exception as e:
                cleanup_result['errors'].append(f"Memory monitoring stop failed: {e}")
        
        # Clean up CPU optimizer
        if self._cpu_optimizer:
            try:
                if hasattr(self._cpu_optimizer, 'cleanup'):
                    self._cpu_optimizer.cleanup()
                cleanup_result['cpu_cleaned'] = True
            except Exception as e:
                cleanup_result['errors'].append(f"CPU cleanup failed: {e}")
        
        return cleanup_result


# Global resource manager instance
_global_resource_manager = None

def get_resource_manager(component_name: str, logger: Optional[logging.Logger] = None) -> ResourceManager:
    """Get or create a resource manager instance."""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(component_name, logger)
    
    return _global_resource_manager

def reset_resource_manager():
    """Reset the global resource manager (useful for testing)."""
    global _global_resource_manager
    if _global_resource_manager:
        try:
            _global_resource_manager.cleanup()
        except Exception:
            pass
    _global_resource_manager = None


# Decorator for automatic resource management
def managed_resources(component_name: str):
    """Decorator for automatic resource management."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with get_resource_manager(component_name) as resource_manager:
                return func(*args, **kwargs)
        return wrapper
    return decorator