"""
Feature Engineering Performance Monitoring

Provides comprehensive monitoring for feature engineering operations including
performance tracking, resource monitoring, and optimization recommendations.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
import logging
from collections import defaultdict
import threading

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_performance

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    input_size: int
    output_size: int
    cache_hit: bool = False
    optimization_applied: bool = False

@dataclass
class ResourceLimits:
    """Resource limits for monitoring."""
    max_memory_mb: int = 8000
    max_cpu_percent: float = 80.0
    max_execution_time: float = 300.0  # 5 minutes
    warning_memory_mb: int = 6000
    warning_cpu_percent: float = 70.0

class FeaturePerformanceMonitor:
    """
    Monitor performance of feature engineering operations.
    
    Tracks:
    - Execution time
    - Memory usage
    - CPU usage
    - Cache hit rates
    - Optimization opportunities
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize performance monitor."""
        self.limits = limits or ResourceLimits()
        self.metrics_history = []
        self.operation_counts = defaultdict(int)
        self.total_execution_time = 0.0
        self.total_memory_usage = 0.0
        self.optimization_suggestions = []
        
        tprint_info("🔧 Initialized FeaturePerformanceMonitor")
    
    def monitor_operation(self, operation_name: str):
        """Decorator to monitor operation performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_cpu = psutil.cpu_percent()
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    end_cpu = psutil.cpu_percent()
                    
                    # Calculate metrics
                    execution_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    cpu_usage = max(start_cpu, end_cpu)
                    
                    # Estimate input/output sizes
                    input_size = self._estimate_data_size(args, kwargs)
                    output_size = self._estimate_data_size([result], {})
                    
                    # Create metrics
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        execution_time=execution_time,
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_usage,
                        input_size=input_size,
                        output_size=output_size
                    )
                    
                    # Store metrics
                    self.metrics_history.append(metrics)
                    self.operation_counts[operation_name] += 1
                    self.total_execution_time += execution_time
                    self.total_memory_usage += memory_usage
                    
                    # Check for warnings
                    self._check_resource_limits(metrics)
                    
                    # Generate optimization suggestions
                    self._generate_optimization_suggestions(metrics)
                    
                    return result
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Operation {operation_name} failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def _estimate_data_size(self, args: Tuple, kwargs: Dict) -> int:
        """Estimate data size in bytes."""
        total_size = 0
        
        for arg in args:
            if isinstance(arg, (pd.DataFrame, pd.Series)):
                mem_usage = arg.memory_usage(deep=True)
                total_size += int(mem_usage.sum()) if hasattr(mem_usage, 'sum') else int(mem_usage)
            elif isinstance(arg, np.ndarray):
                total_size += arg.nbytes
            elif isinstance(arg, (list, tuple)):
                total_size += sum(self._estimate_data_size([item], {}) for item in arg)
        
        for value in kwargs.values():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                mem_usage = value.memory_usage(deep=True)
                total_size += int(mem_usage.sum()) if hasattr(mem_usage, 'sum') else int(mem_usage)
            elif isinstance(value, np.ndarray):
                total_size += value.nbytes
        
        return total_size
    
    def _check_resource_limits(self, metrics: PerformanceMetrics):
        """Check if operation exceeded resource limits."""
        warnings = []
        
        if metrics.memory_usage_mb > self.limits.warning_memory_mb:
            warnings.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.cpu_usage_percent > self.limits.warning_cpu_percent:
            warnings.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.execution_time > self.limits.max_execution_time:
            warnings.append(f"Long execution time: {metrics.execution_time:.1f}s")
        
        if warnings:
            tprint_warning(f"⚠️ {metrics.operation_name}: {', '.join(warnings)}")
    
    def _generate_optimization_suggestions(self, metrics: PerformanceMetrics):
        """Generate optimization suggestions based on metrics."""
        suggestions = []
        
        if metrics.execution_time > 10.0:  # More than 10 seconds
            suggestions.append(f"Consider caching for {metrics.operation_name}")
        
        if metrics.memory_usage_mb > 1000:  # More than 1GB
            suggestions.append(f"Consider chunked processing for {metrics.operation_name}")
        
        if metrics.input_size > metrics.output_size * 10:  # Large input, small output
            suggestions.append(f"Consider early filtering for {metrics.operation_name}")
        
        if suggestions:
            self.optimization_suggestions.extend(suggestions)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"message": "No metrics recorded"}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        
        return {
            'total_operations': len(self.metrics_history),
            'total_execution_time': sum(execution_times),
            'total_memory_usage_mb': sum(memory_usage),
            'average_execution_time': np.mean(execution_times),
            'average_memory_usage_mb': np.mean(memory_usage),
            'slowest_operation': max(self.metrics_history, key=lambda x: x.execution_time),
            'most_memory_intensive': max(self.metrics_history, key=lambda x: x.memory_usage_mb),
            'operation_counts': dict(self.operation_counts),
            'optimization_suggestions': self.optimization_suggestions[-10:]  # Last 10 suggestions
        }
    
    def get_operation_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get breakdown by operation type."""
        breakdown = defaultdict(lambda: {'count': 0, 'total_time': 0, 'total_memory': 0})
        
        for metrics in self.metrics_history:
            breakdown[metrics.operation_name]['count'] += 1
            breakdown[metrics.operation_name]['total_time'] += metrics.execution_time
            breakdown[metrics.operation_name]['total_memory'] += metrics.memory_usage_mb
        
        # Calculate averages
        for op_name, stats in breakdown.items():
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['avg_memory'] = stats['total_memory'] / stats['count']
        
        return dict(breakdown)

class ResourceTracker:
    """
    Track resource usage across feature engineering operations.
    
    Monitors:
    - Memory usage
    - CPU usage
    - Disk I/O
    - Network usage (if applicable)
    """
    
    def __init__(self):
        """Initialize resource tracker."""
        self.start_time = time.time()
        self.peak_memory = 0
        self.total_cpu_time = 0
        self.operation_counts = defaultdict(int)
        self.resource_history = []
        
        tprint_info("🔧 Initialized ResourceTracker")
    
    def track_operation(self, operation_name: str):
        """Decorator to track resource usage for an operation."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Record start state
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_cpu = psutil.cpu_percent()
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record end state
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    end_cpu = psutil.cpu_percent()
                    end_time = time.time()
                    
                    # Update tracking
                    self.peak_memory = max(self.peak_memory, end_memory)
                    self.total_cpu_time += end_cpu * (end_time - start_time)
                    self.operation_counts[operation_name] += 1
                    
                    # Record resource usage
                    self.resource_history.append({
                        'operation': operation_name,
                        'timestamp': end_time,
                        'memory_mb': end_memory,
                        'cpu_percent': end_cpu,
                        'execution_time': end_time - start_time
                    })
                    
                    return result
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Resource tracking failed for {operation_name}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_history:
            return {"message": "No resource data recorded"}
        
        memory_usage = [r['memory_mb'] for r in self.resource_history]
        cpu_usage = [r['cpu_percent'] for r in self.resource_history]
        execution_times = [r['execution_time'] for r in self.resource_history]
        
        return {
            'total_operations': len(self.resource_history),
            'total_time': time.time() - self.start_time,
            'peak_memory_mb': self.peak_memory,
            'average_memory_mb': np.mean(memory_usage),
            'average_cpu_percent': np.mean(cpu_usage),
            'average_execution_time': np.mean(execution_times),
            'operation_counts': dict(self.operation_counts),
            'memory_trend': memory_usage[-10:] if len(memory_usage) >= 10 else memory_usage
        }
    
    def get_memory_usage_alert(self) -> Optional[str]:
        """Check for memory usage alerts."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        if current_memory > 6000:  # More than 6GB
            return f"⚠️ High memory usage: {current_memory:.1f}MB"
        elif current_memory > 4000:  # More than 4GB
            return f"⚠️ Moderate memory usage: {current_memory:.1f}MB"
        
        return None
    
    def cleanup_resources(self):
        """Clean up resources and force garbage collection."""
        import gc
        gc.collect()
        
        # Log cleanup
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        gc.collect()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        if memory_before - memory_after > 100:  # Freed more than 100MB
            tprint_info(f"🧹 Cleaned up {memory_before - memory_after:.1f}MB of memory")

# Global instances
_performance_monitor = None
_resource_tracker = None

def get_performance_monitor() -> FeaturePerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = FeaturePerformanceMonitor()
    return _performance_monitor

def get_resource_tracker() -> ResourceTracker:
    """Get global resource tracker instance."""
    global _resource_tracker
    if _resource_tracker is None:
        _resource_tracker = ResourceTracker()
    return _resource_tracker
