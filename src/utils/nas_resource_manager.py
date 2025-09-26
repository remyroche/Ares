#!/usr/bin/env python3
"""
Comprehensive Resource Management System for NAS Components

This module provides unified resource management with proper cleanup,
memory monitoring, and resource tracking to prevent memory leaks.
"""

import gc
import psutil
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import logging
from enum import Enum

from .nas_error_handling import (
    NASResourceError, NASMemoryError, ErrorContext, 
    error_context, safe_execute, get_error_handler
)


class ResourceType(Enum):
    """Types of resources that need management."""
    FILE_HANDLE = "file_handle"
    NETWORK_CONNECTION = "network_connection"
    GPU_MEMORY = "gpu_memory"
    CPU_MEMORY = "cpu_memory"
    THREAD = "thread"
    PROCESS = "process"
    CACHE = "cache"
    MODEL = "model"
    DATASET = "dataset"
    OPTIMIZER = "optimizer"


@dataclass
class ResourceInfo:
    """Information about a managed resource."""
    resource_id: str
    resource_type: ResourceType
    created_at: float
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_func: Optional[Callable] = None
    weak_ref: Optional[weakref.ref] = None


class ResourceTracker:
    """Tracks and manages resources to prevent leaks."""
    
    def __init__(self):
        self._resources: Dict[str, ResourceInfo] = {}
        self._lock = threading.RLock()
        self._cleanup_callbacks: List[Callable] = []
        self._memory_threshold_mb = 1024  # 1GB default threshold
        self._error_handler = get_error_handler()
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        resource: Any,
        size_bytes: Optional[int] = None,
        cleanup_func: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a resource for tracking."""
        with self._lock:
            try:
                # Create weak reference to avoid circular references
                weak_ref = weakref.ref(resource, self._cleanup_callback)
                
                resource_info = ResourceInfo(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    created_at=time.time(),
                    size_bytes=size_bytes,
                    metadata=metadata or {},
                    cleanup_func=cleanup_func,
                    weak_ref=weak_ref
                )
                
                self._resources[resource_id] = resource_info
                
                # Check memory threshold
                self._check_memory_threshold()
                
            except Exception as e:
                context = ErrorContext("register_resource", "resource_tracker")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource and clean it up."""
        with self._lock:
            try:
                if resource_id in self._resources:
                    resource_info = self._resources[resource_id]
                    
                    # Call cleanup function if available
                    if resource_info.cleanup_func:
                        try:
                            resource_info.cleanup_func()
                        except Exception as e:
                            context = ErrorContext("cleanup_resource", "resource_tracker")
                            self._error_handler.handle_error(e, context, reraise=False)
                    
                    del self._resources[resource_id]
                    return True
                
                return False
                
            except Exception as e:
                context = ErrorContext("unregister_resource", "resource_tracker")
                self._error_handler.handle_error(e, context, reraise=False)
                return False
    
    def _cleanup_callback(self, weak_ref: weakref.ref) -> None:
        """Callback when a resource is garbage collected."""
        with self._lock:
            # Find and remove the resource
            resource_id = None
            for rid, info in self._resources.items():
                if info.weak_ref is weak_ref:
                    resource_id = rid
                    break
            
            if resource_id:
                self.unregister_resource(resource_id)
    
    def _check_memory_threshold(self) -> None:
        """Check if memory usage exceeds threshold."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._memory_threshold_mb:
                context = ErrorContext("memory_threshold_exceeded", "resource_tracker")
                self._error_handler.handle_error(
                    NASMemoryError(f"Memory usage {memory_mb:.1f}MB exceeds threshold {self._memory_threshold_mb}MB"),
                    context,
                    reraise=False
                )
                
                # Trigger garbage collection
                self.force_cleanup()
                
        except Exception as e:
            context = ErrorContext("check_memory_threshold", "resource_tracker")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def force_cleanup(self) -> None:
        """Force cleanup of all resources."""
        with self._lock:
            try:
                # Clean up all registered resources
                resource_ids = list(self._resources.keys())
                for resource_id in resource_ids:
                    self.unregister_resource(resource_id)
                
                # Force garbage collection
                gc.collect()
                
                # Call cleanup callbacks
                for callback in self._cleanup_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        context = ErrorContext("cleanup_callback", "resource_tracker")
                        self._error_handler.handle_error(e, context, reraise=False)
                
            except Exception as e:
                context = ErrorContext("force_cleanup", "resource_tracker")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked resources."""
        with self._lock:
            stats = {
                'total_resources': len(self._resources),
                'resources_by_type': {},
                'total_size_bytes': 0,
                'oldest_resource_age': 0,
                'memory_usage_mb': 0
            }
            
            current_time = time.time()
            
            for resource_info in self._resources.values():
                # Count by type
                resource_type = resource_info.resource_type.value
                stats['resources_by_type'][resource_type] = stats['resources_by_type'].get(resource_type, 0) + 1
                
                # Sum sizes
                if resource_info.size_bytes:
                    stats['total_size_bytes'] += resource_info.size_bytes
                
                # Track oldest resource
                age = current_time - resource_info.created_at
                stats['oldest_resource_age'] = max(stats['oldest_resource_age'], age)
            
            # Get current memory usage
            try:
                process = psutil.Process()
                stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            except Exception:
                stats['memory_usage_mb'] = 0
            
            return stats
    
    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a callback to be called during cleanup."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def set_memory_threshold(self, threshold_mb: int) -> None:
        """Set memory threshold for automatic cleanup."""
        with self._lock:
            self._memory_threshold_mb = threshold_mb


class MemoryMonitor:
    """Monitors memory usage and provides optimization suggestions."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._memory_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._error_handler = get_error_handler()
    
    def start_monitoring(self) -> None:
        """Start memory monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._check_memory_usage()
                time.sleep(self.check_interval)
            except Exception as e:
                context = ErrorContext("memory_monitor_loop", "memory_monitor")
                self._error_handler.handle_error(e, context, reraise=False)
                time.sleep(self.check_interval)
    
    def _check_memory_usage(self) -> None:
        """Check current memory usage and record history."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_data = {
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
            
            with self._lock:
                self._memory_history.append(memory_data)
                
                # Keep only last 100 entries
                if len(self._memory_history) > 100:
                    self._memory_history = self._memory_history[-100:]
                
                # Check for memory leaks (continuous growth)
                if len(self._memory_history) >= 10:
                    self._check_memory_leak()
                
        except Exception as e:
            context = ErrorContext("check_memory_usage", "memory_monitor")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def _check_memory_leak(self) -> None:
        """Check for potential memory leaks."""
        try:
            recent_data = self._memory_history[-10:]
            rss_values = [data['rss_mb'] for data in recent_data]
            
            # Check for continuous growth
            if len(rss_values) >= 5:
                growth_rate = (rss_values[-1] - rss_values[0]) / len(rss_values)
                
                if growth_rate > 10:  # More than 10MB per check
                    context = ErrorContext("memory_leak_detected", "memory_monitor")
                    self._error_handler.handle_error(
                        NASMemoryError(f"Potential memory leak detected: {growth_rate:.1f}MB growth per check"),
                        context,
                        reraise=False
                    )
        
        except Exception as e:
            context = ErrorContext("check_memory_leak", "memory_monitor")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            virtual_memory = psutil.virtual_memory()
            
            with self._lock:
                return {
                    'current_rss_mb': memory_info.rss / 1024 / 1024,
                    'current_vms_mb': memory_info.vms / 1024 / 1024,
                    'current_percent': process.memory_percent(),
                    'system_available_mb': virtual_memory.available / 1024 / 1024,
                    'system_total_mb': virtual_memory.total / 1024 / 1024,
                    'system_percent': virtual_memory.percent,
                    'history_length': len(self._memory_history),
                    'monitoring_active': self._monitoring
                }
        
        except Exception as e:
            context = ErrorContext("get_memory_stats", "memory_monitor")
            self._error_handler.handle_error(e, context, reraise=False)
            return {}


class ResourceManager:
    """Main resource manager that coordinates all resource management."""
    
    def __init__(self):
        self._tracker = ResourceTracker()
        self._memory_monitor = MemoryMonitor()
        self._cleanup_registered = False
        self._error_handler = get_error_handler()
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        try:
            self._memory_monitor.start_monitoring()
            self._register_cleanup_handlers()
        except Exception as e:
            context = ErrorContext("start_monitoring", "resource_manager")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        try:
            self._memory_monitor.stop_monitoring()
        except Exception as e:
            context = ErrorContext("stop_monitoring", "resource_manager")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for graceful shutdown."""
        if self._cleanup_registered:
            return
        
        import atexit
        atexit.register(self.cleanup_all_resources)
        self._cleanup_registered = True
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        resource: Any,
        size_bytes: Optional[int] = None,
        cleanup_func: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a resource for management."""
        self._tracker.register_resource(
            resource_id, resource_type, resource, size_bytes, cleanup_func, metadata
        )
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource."""
        return self._tracker.unregister_resource(resource_id)
    
    def cleanup_all_resources(self) -> None:
        """Clean up all managed resources."""
        try:
            self._tracker.force_cleanup()
            self._memory_monitor.stop_monitoring()
        except Exception as e:
            context = ErrorContext("cleanup_all_resources", "resource_manager")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        try:
            tracker_stats = self._tracker.get_resource_stats()
            memory_stats = self._memory_monitor.get_memory_stats()
            
            return {
                'tracker_stats': tracker_stats,
                'memory_stats': memory_stats,
                'timestamp': time.time()
            }
        except Exception as e:
            context = ErrorContext("get_resource_stats", "resource_manager")
            self._error_handler.handle_error(e, context, reraise=False)
            return {}
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Clean up resources
            self._tracker.force_cleanup()
            
            # Get memory stats after optimization
            stats = self.get_resource_stats()
            
            return {
                'garbage_collected': collected,
                'optimization_time': time.time(),
                'stats_after_optimization': stats
            }
        except Exception as e:
            context = ErrorContext("optimize_memory", "resource_manager")
            self._error_handler.handle_error(e, context, reraise=False)
            return {}


# Global resource manager instance
_global_resource_manager = ResourceManager()


@contextmanager
def managed_resource(
    resource_id: str,
    resource_type: ResourceType,
    resource: Any,
    size_bytes: Optional[int] = None,
    cleanup_func: Optional[Callable] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Context manager for automatic resource management."""
    manager = get_resource_manager()
    
    try:
        manager.register_resource(
            resource_id, resource_type, resource, size_bytes, cleanup_func, metadata
        )
        yield resource
    finally:
        manager.unregister_resource(resource_id)


@contextmanager
def memory_monitoring(threshold_mb: int = 1024):
    """Context manager for memory monitoring."""
    manager = get_resource_manager()
    original_threshold = manager._tracker._memory_threshold_mb
    
    try:
        manager._tracker.set_memory_threshold(threshold_mb)
        manager.start_monitoring()
        yield manager
    finally:
        manager.stop_monitoring()
        manager._tracker.set_memory_threshold(original_threshold)


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    return _global_resource_manager


def cleanup_resources() -> None:
    """Clean up all resources (convenience function)."""
    get_resource_manager().cleanup_all_resources()


def optimize_memory() -> Dict[str, Any]:
    """Optimize memory usage (convenience function)."""
    return get_resource_manager().optimize_memory()


# Export main classes and functions
__all__ = [
    'ResourceType',
    'ResourceInfo',
    'ResourceTracker',
    'MemoryMonitor',
    'ResourceManager',
    'managed_resource',
    'memory_monitoring',
    'get_resource_manager',
    'cleanup_resources',
    'optimize_memory'
]