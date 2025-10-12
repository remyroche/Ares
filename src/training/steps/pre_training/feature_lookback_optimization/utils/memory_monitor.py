"""
Memory Monitoring Utilities for Feature Lookback Optimization.

This module provides comprehensive memory monitoring and management capabilities.
"""

import gc
import psutil
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# Import centralized tprint utilities
from .tprint_utils import (
    tprint, tprint_warning, tprint_debug, tprint_info,
    TPRINT_AVAILABLE
)


@dataclass
class MemoryStats:
    """Memory statistics snapshot."""
    timestamp: datetime
    process_memory_mb: float
    system_memory_mb: float
    system_memory_percent: float
    available_memory_mb: float
    cache_size: int
    gc_objects: int
    memory_pressure: str  # "low", "medium", "high", "critical"


class MemoryMonitor:
    """Comprehensive memory monitoring and management."""
    
    def __init__(self, 
                 memory_limit_mb: float = 1024.0,
                 warning_threshold: float = 0.7,
                 critical_threshold: float = 0.9,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_mb: Memory limit in MB
            warning_threshold: Warning threshold (0.0-1.0)
            critical_threshold: Critical threshold (0.0-1.0)
            logger: Logger instance
        """
        self.memory_limit_mb = memory_limit_mb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 1000
        self.process = psutil.Process()
        
        # Memory pressure tracking
        self.consecutive_high_pressure = 0
        self.last_cleanup = datetime.now()
        
        tprint_info(f"🔧 Memory monitor initialized (limit: {memory_limit_mb}MB)")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_system_memory_info(self) -> Tuple[float, float, float]:
        """Get system memory information.
        
        Returns:
            Tuple of (total_mb, available_mb, percent_used)
        """
        try:
            memory = psutil.virtual_memory()
            total_mb = memory.total / (1024 * 1024)
            available_mb = memory.available / (1024 * 1024)
            percent_used = memory.percent / 100.0
            
            return total_mb, available_mb, percent_used
        except Exception as e:
            self.logger.warning(f"Failed to get system memory info: {e}")
            return 0.0, 0.0, 0.0
    
    def get_memory_pressure(self) -> str:
        """Get current memory pressure level."""
        try:
            _, available_mb, percent_used = self.get_system_memory_info()
            
            if percent_used >= self.critical_threshold:
                return "critical"
            elif percent_used >= self.warning_threshold:
                return "high"
            elif percent_used >= 0.5:
                return "medium"
            else:
                return "low"
        except Exception:
            return "unknown"
    
    def get_cache_size(self) -> int:
        """Get current cache size (placeholder for actual cache tracking)."""
        # This would be implemented by integrating with actual caches
        return 0
    
    def get_gc_objects(self) -> int:
        """Get number of objects tracked by garbage collector."""
        try:
            return len(gc.get_objects())
        except Exception:
            return 0
    
    def take_snapshot(self) -> MemoryStats:
        """Take a memory statistics snapshot."""
        try:
            process_memory = self.get_memory_usage()
            total_mb, available_mb, percent_used = self.get_system_memory_info()
            pressure = self.get_memory_pressure()
            cache_size = self.get_cache_size()
            gc_objects = self.get_gc_objects()
            
            stats = MemoryStats(
                timestamp=datetime.now(),
                process_memory_mb=process_memory,
                system_memory_mb=total_mb,
                system_memory_percent=percent_used,
                available_memory_mb=available_mb,
                cache_size=cache_size,
                gc_objects=gc_objects,
                memory_pressure=pressure
            )
            
            # Store in history
            self.memory_history.append(stats)
            
            # Trim history if too large
            if len(self.memory_history) > self.max_history_size:
                self.memory_history = self.memory_history[-self.max_history_size//2:]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            return MemoryStats(
                timestamp=datetime.now(),
                process_memory_mb=0.0,
                system_memory_mb=0.0,
                system_memory_percent=0.0,
                available_memory_mb=0.0,
                cache_size=0,
                gc_objects=0,
                memory_pressure="unknown"
            )
    
    def check_memory_pressure(self) -> bool:
        """Check if memory pressure is high and needs attention."""
        stats = self.take_snapshot()
        
        if stats.memory_pressure in ["high", "critical"]:
            self.consecutive_high_pressure += 1
            return True
        else:
            self.consecutive_high_pressure = 0
            return False
    
    def should_cleanup(self) -> bool:
        """Determine if cleanup should be performed."""
        # Check memory pressure
        if self.check_memory_pressure():
            return True
        
        # Check if process memory exceeds limit
        if self.get_memory_usage() > self.memory_limit_mb:
            return True
        
        # Check if we've had consecutive high pressure
        if self.consecutive_high_pressure >= 3:
            return True
        
        # Check time since last cleanup
        time_since_cleanup = (datetime.now() - self.last_cleanup).total_seconds()
        if time_since_cleanup > 300:  # 5 minutes
            return True
        
        return False
    
    def perform_cleanup(self, caches: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Perform memory cleanup operations.
        
        Args:
            caches: List of cache objects to clean
            
        Returns:
            Dictionary with cleanup results
        """
        cleanup_results = {
            'timestamp': datetime.now(),
            'before_memory_mb': self.get_memory_usage(),
            'cleanup_operations': [],
            'success': True
        }
        
        try:
            # Clean caches if provided
            if caches:
                for i, cache in enumerate(caches):
                    try:
                        if hasattr(cache, 'clear_cache'):
                            cache.clear_cache(keep_recent=100)  # Keep recent entries
                            cleanup_results['cleanup_operations'].append(f'cache_{i}_cleared')
                        elif hasattr(cache, 'clear'):
                            cache.clear()
                            cleanup_results['cleanup_operations'].append(f'cache_{i}_cleared')
                    except Exception as e:
                        self.logger.warning(f"Failed to clear cache {i}: {e}")
                        cleanup_results['cleanup_operations'].append(f'cache_{i}_failed')
            
            # Force garbage collection
            collected = gc.collect()
            cleanup_results['cleanup_operations'].append(f'gc_collected_{collected}_objects')
            
            # Update last cleanup time
            self.last_cleanup = datetime.now()
            
            # Reset pressure counter
            self.consecutive_high_pressure = 0
            
            # Get memory after cleanup
            cleanup_results['after_memory_mb'] = self.get_memory_usage()
            cleanup_results['memory_freed_mb'] = (
                cleanup_results['before_memory_mb'] - cleanup_results['after_memory_mb']
            )
            
            tprint_info(f"🧹 Memory cleanup completed: {cleanup_results['memory_freed_mb']:.1f}MB freed")
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            cleanup_results['success'] = False
            cleanup_results['error'] = str(e)
        
        return cleanup_results
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        stats = self.take_snapshot()
        
        # Calculate trends
        if len(self.memory_history) >= 2:
            recent_trend = (
                self.memory_history[-1].process_memory_mb - 
                self.memory_history[-2].process_memory_mb
            )
        else:
            recent_trend = 0.0
        
        return {
            'current': {
                'process_memory_mb': stats.process_memory_mb,
                'system_memory_percent': stats.system_memory_percent,
                'memory_pressure': stats.memory_pressure,
                'cache_size': stats.cache_size,
                'gc_objects': stats.gc_objects
            },
            'trends': {
                'recent_change_mb': recent_trend,
                'consecutive_high_pressure': self.consecutive_high_pressure,
                'history_size': len(self.memory_history)
            },
            'limits': {
                'memory_limit_mb': self.memory_limit_mb,
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold
            },
            'recommendations': self._get_memory_recommendations(stats)
        }
    
    def _get_memory_recommendations(self, stats: MemoryStats) -> List[str]:
        """Get memory management recommendations."""
        recommendations = []
        
        if stats.memory_pressure == "critical":
            recommendations.append("CRITICAL: Immediate memory cleanup required")
        elif stats.memory_pressure == "high":
            recommendations.append("WARNING: High memory pressure detected")
        
        if stats.process_memory_mb > self.memory_limit_mb:
            recommendations.append(f"Process memory ({stats.process_memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
        
        if self.consecutive_high_pressure >= 3:
            recommendations.append("Persistent high memory pressure - consider reducing cache sizes")
        
        if stats.gc_objects > 100000:
            recommendations.append("High number of GC objects - consider reducing object creation")
        
        if not recommendations:
            recommendations.append("Memory usage is within normal limits")
        
        return recommendations
    
    def monitor_memory_usage(self, operation_name: str, caches: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Monitor memory usage during an operation.
        
        Args:
            operation_name: Name of the operation being monitored
            caches: List of cache objects to monitor
            
        Returns:
            Dictionary with monitoring results
        """
        start_stats = self.take_snapshot()
        
        try:
            # Check if cleanup is needed before operation
            if self.should_cleanup():
                cleanup_results = self.perform_cleanup(caches)
                tprint_info(f"🧹 Pre-operation cleanup: {cleanup_results['memory_freed_mb']:.1f}MB freed")
            
            # Monitor during operation
            peak_memory = start_stats.process_memory_mb
            
            # This would be called periodically during the operation
            # For now, we'll just return the start stats
            
            end_stats = self.take_snapshot()
            
            return {
                'operation': operation_name,
                'start_memory_mb': start_stats.process_memory_mb,
                'end_memory_mb': end_stats.process_memory_mb,
                'peak_memory_mb': peak_memory,
                'memory_delta_mb': end_stats.process_memory_mb - start_stats.process_memory_mb,
                'memory_pressure': end_stats.memory_pressure,
                'cleanup_performed': self.should_cleanup()
            }
            
        except Exception as e:
            self.logger.error(f"Memory monitoring failed for {operation_name}: {e}")
            return {
                'operation': operation_name,
                'error': str(e),
                'success': False
            }


# Global memory monitor instance
_global_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor


def monitor_memory(operation_name: str, caches: Optional[List[Any]] = None):
    """Decorator for monitoring memory usage during operations."""
    from .tprint_utils import tprint_debug
    tprint_debug(f"🔍 Setting up memory monitoring for operation: {operation_name}")
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            tprint_debug(f"📊 Starting memory monitoring for {operation_name}")
            monitor = get_memory_monitor()
            result = monitor.monitor_memory_usage(operation_name, caches)
            tprint_debug(f"✅ Memory monitoring completed for {operation_name}")
            return result
        return wrapper
    return decorator