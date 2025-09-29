"""
Performance Monitoring and Memory Tracking Suggestions
====================================================

This file contains suggested implementations for:
1. Initialization timing monitoring
2. Memory usage tracking
3. Duplicate initialization prevention
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from functools import wraps
import threading
from dataclasses import dataclass
from enum import Enum

class ComponentType(Enum):
    """Types of system components for monitoring."""
    DATA_CLEANER = "DataCleaner"
    QUALITY_FRAMEWORK = "DataQualityFramework"
    STREAMING_MANAGER = "DataStreamingManager"
    MATRIX_OPERATIONS = "MatrixOperations"
    FEATURE_REGISTRY = "FeatureRegistry"

@dataclass
class InitializationMetrics:
    """Metrics for component initialization."""
    component_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    success: bool
    error_message: Optional[str] = None

class SystemPerformanceMonitor:
    """Centralized performance monitoring for system initialization."""
    
    def __init__(self):
        self.logger = logging.getLogger('SystemPerformanceMonitor')
        self.metrics: Dict[str, InitializationMetrics] = {}
        self._lock = threading.Lock()
        
    @contextmanager
    def monitor_initialization(self, component_name: str, component_type: ComponentType):
        """Context manager for monitoring component initialization."""
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        self.logger.info(f"🚀 Initializing {component_name}...")
        
        try:
            yield
            end_time = time.time()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            with self._lock:
                self.metrics[component_name] = InitializationMetrics(
                    component_name=component_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    memory_delta=memory_after - memory_before,
                    success=True
                )
            
            self.logger.info(f"✅ {component_name} initialized in {end_time - start_time:.3f}s "
                           f"(Memory: +{memory_after - memory_before:.1f}MB)")
            
        except Exception as e:
            end_time = time.time()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            with self._lock:
                self.metrics[component_name] = InitializationMetrics(
                    component_name=component_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    memory_delta=memory_after - memory_before,
                    success=False,
                    error_message=str(e)
                )
            
            self.logger.error(f"❌ {component_name} initialization failed: {e}")
            raise
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate performance summary report."""
        if not self.metrics:
            return {"message": "No initialization metrics available"}
        
        total_duration = sum(m.duration for m in self.metrics.values())
        total_memory_delta = sum(m.memory_delta for m in self.metrics.values())
        failed_components = [m for m in self.metrics.values() if not m.success]
        
        # Find slowest component
        slowest = max(self.metrics.values(), key=lambda m: m.duration)
        
        # Find highest memory consumer
        highest_memory = max(self.metrics.values(), key=lambda m: m.memory_delta)
        
        return {
            "total_components": len(self.metrics),
            "total_duration": total_duration,
            "total_memory_delta": total_memory_delta,
            "failed_components": len(failed_components),
            "slowest_component": {
                "name": slowest.component_name,
                "duration": slowest.duration
            },
            "highest_memory_consumer": {
                "name": highest_memory.component_name,
                "memory_delta": highest_memory.memory_delta
            },
            "components": {
                name: {
                    "duration": metric.duration,
                    "memory_delta": metric.memory_delta,
                    "success": metric.success
                }
                for name, metric in self.metrics.items()
            }
        }

# Global performance monitor instance
performance_monitor = SystemPerformanceMonitor()

def monitor_initialization(component_name: str, component_type: ComponentType):
    """Decorator for monitoring component initialization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor.monitor_initialization(component_name, component_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage for DataCleaner
class MonitoredDataCleaner:
    """DataCleaner with performance monitoring."""
    
    @monitor_initialization("DataCleaner", ComponentType.DATA_CLEANER)
    def __init__(self, max_forward_fill_gap: int = 5, download_threshold: int = 5, 
                 raise_errors: bool = True, log_details: bool = True, 
                 data_type: str = 'klines') -> None:
        """Initialize with monitoring."""
        # Original DataCleaner initialization code would go here
        self.logger = logging.getLogger('DataCleaner')
        self.data_type = data_type
        # ... rest of initialization

# Memory usage tracking utilities
class MemoryTracker:
    """Utility for tracking memory usage during operations."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    def log_memory_usage(logger: logging.Logger, context: str = ""):
        """Log current memory usage."""
        memory = MemoryTracker.get_memory_usage()
        logger.info(f"💾 Memory Usage {context}: "
                   f"RSS={memory['rss_mb']:.1f}MB, "
                   f"VMS={memory['vms_mb']:.1f}MB, "
                   f"Percent={memory['percent']:.1f}%, "
                   f"Available={memory['available_mb']:.1f}MB")

# Singleton pattern for preventing duplicate initialization
class SingletonMeta(type):
    """Metaclass for implementing singleton pattern."""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# Example of how to implement singleton DataCleaner
class SingletonDataCleaner(metaclass=SingletonMeta):
    """Singleton DataCleaner to prevent duplicate initialization."""
    
    def __init__(self, data_type: str = 'klines'):
        if not hasattr(self, '_initialized'):
            self.logger = logging.getLogger('DataCleaner')
            self.data_type = data_type
            self._initialized = True
            self.logger.info(f"🔧 DataCleaner singleton initialized with data_type='{data_type}'")

# Performance monitoring configuration
PERFORMANCE_CONFIG = {
    "enable_timing": True,
    "enable_memory_tracking": True,
    "log_threshold_ms": 100,  # Log components taking longer than 100ms
    "memory_warning_mb": 100,  # Warn if memory usage increases by more than 100MB
    "enable_singleton_pattern": True
}