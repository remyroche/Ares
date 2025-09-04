#!/usr/bin/env python3
"""
Performance Monitoring System

This module provides comprehensive performance monitoring and metrics collection
throughout the trading pipeline.
"""

import asyncio
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
import psutil
import logging

from src.utils.logger import system_logger
from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    timestamp: str
    unit: str = "seconds"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    timestamp: str


class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self, max_history_size: int = 1000):
        self.logger = system_logger.getChild("PerformanceMonitor")
        self.max_history_size = max_history_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history_size))
        self.system_metrics_history = deque(maxlen=max_history_size)
        self.active_operations = {}
        self.operation_timers = {}
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'execution_time': 60.0,  # seconds
            'memory_usage': 80.0,    # percent
            'cpu_usage': 90.0,       # percent
            'disk_usage': 85.0       # percent
        }
        
        self.logger.info("Performance monitoring system initialized")
    
    def start_operation(self, operation_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start monitoring an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        with self.lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': time.time(),
                'metadata': metadata or {}
            }
        
        self.logger.debug(f"Started monitoring operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True) -> Optional[PerformanceMetric]:
        """End monitoring an operation and record the metric."""
        with self.lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Operation ID not found: {operation_id}")
                return None
            
            operation = self.active_operations.pop(operation_id)
            duration = time.time() - operation['start_time']
            
            # Create performance metric
            metric = PerformanceMetric(
                name=operation['name'],
                value=duration,
                timestamp=format_datetime(get_current_datetime()),
                unit="seconds",
                metadata={
                    'success': success,
                    'operation_id': operation_id,
                    **operation['metadata']
                }
            )
            
            # Store metric
            self.metrics_history[operation['name']].append(metric)
            
            # Check thresholds
            self._check_thresholds(metric)
            
            self.logger.debug(f"Ended monitoring operation: {operation['name']} (Duration: {duration:.3f}s)")
            return metric
    
    def record_metric(self, name: str, value: float, unit: str = "seconds", metadata: Dict[str, Any] = None):
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=format_datetime(get_current_datetime()),
            unit=unit,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics_history[name].append(metric)
        
        self._check_thresholds(metric)
        self.logger.debug(f"Recorded metric: {name} = {value} {unit}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                timestamp=format_datetime(get_current_datetime())
            )
            
            with self.lock:
                self.system_metrics_history.append(system_metrics)
            
            return system_metrics
            
        except Exception as e:
            self.logger.exception(f"Failed to collect system metrics: {e}")
            return None
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds."""
        if metric.name in self.thresholds:
            threshold = self.thresholds[metric.name]
            if metric.value > threshold:
                self.logger.warning(
                    f"Performance threshold exceeded: {metric.name} = {metric.value} "
                    f"(threshold: {threshold})"
                )
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation_name not in self.metrics_history:
            return {"error": f"No metrics found for operation: {operation_name}"}
        
        metrics = list(self.metrics_history[operation_name])
        if not metrics:
            return {"error": f"No metrics found for operation: {operation_name}"}
        
        values = [m.value for m in metrics]
        success_count = sum(1 for m in metrics if m.metadata.get('success', True))
        
        return {
            'operation_name': operation_name,
            'total_executions': len(metrics),
            'successful_executions': success_count,
            'success_rate': success_count / len(metrics),
            'min_duration': min(values),
            'max_duration': max(values),
            'avg_duration': sum(values) / len(values),
            'recent_duration': values[-1] if values else 0,
            'threshold': self.thresholds.get(operation_name, None)
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        if not self.system_metrics_history:
            return {"error": "No system metrics available"}
        
        metrics = list(self.system_metrics_history)
        recent_metric = metrics[-1] if metrics else None
        
        if not recent_metric:
            return {"error": "No system metrics available"}
        
        return {
            'current_cpu_percent': recent_metric.cpu_percent,
            'current_memory_percent': recent_metric.memory_percent,
            'current_memory_used_mb': recent_metric.memory_used_mb,
            'current_memory_available_mb': recent_metric.memory_available_mb,
            'current_disk_usage_percent': recent_metric.disk_usage_percent,
            'cpu_threshold': self.thresholds['cpu_usage'],
            'memory_threshold': self.thresholds['memory_usage'],
            'disk_threshold': self.thresholds['disk_usage'],
            'metrics_count': len(metrics)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': format_datetime(get_current_datetime()),
            'operations': {},
            'system_metrics': self.get_system_statistics(),
            'thresholds': self.thresholds,
            'active_operations': len(self.active_operations)
        }
        
        # Get statistics for all operations
        for operation_name in self.metrics_history.keys():
            summary['operations'][operation_name] = self.get_operation_statistics(operation_name)
        
        return summary
    
    def clear_history(self):
        """Clear all performance history."""
        with self.lock:
            self.metrics_history.clear()
            self.system_metrics_history.clear()
            self.active_operations.clear()
        
        self.logger.info("Performance history cleared")
    
    def export_metrics(self, file_path: str):
        """Export metrics to file."""
        try:
            import json
            
            export_data = {
                'timestamp': format_datetime(get_current_datetime()),
                'metrics': {},
                'system_metrics': list(self.system_metrics_history),
                'summary': self.get_performance_summary()
            }
            
            # Export operation metrics
            for operation_name, metrics in self.metrics_history.items():
                export_data['metrics'][operation_name] = [
                    {
                        'name': m.name,
                        'value': m.value,
                        'timestamp': m.timestamp,
                        'unit': m.unit,
                        'metadata': m.metadata
                    }
                    for m in metrics
                ]
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Performance metrics exported to: {file_path}")
            
        except Exception as e:
            self.logger.exception(f"Failed to export metrics: {e}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str = None, metadata: Dict[str, Any] = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            operation_id = performance_monitor.start_operation(op_name, metadata)
            
            try:
                result = await func(*args, **kwargs)
                performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(operation_id, success=False)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            operation_id = performance_monitor.start_operation(op_name, metadata)
            
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                performance_monitor.end_operation(operation_id, success=False)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceContext:
    """Context manager for performance monitoring."""
    
    def __init__(self, operation_name: str, metadata: Dict[str, Any] = None):
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = performance_monitor.start_operation(self.operation_name, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        performance_monitor.end_operation(self.operation_id, success=success)


# Import functools for decorator
import functools

# Export commonly used functions
__all__ = [
    'PerformanceMetric',
    'SystemMetrics',
    'PerformanceMonitor',
    'performance_monitor',
    'monitor_performance',
    'PerformanceContext'
]