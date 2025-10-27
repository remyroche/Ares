"""
Performance monitoring and adaptive parameters for SR detection.

This module provides real-time performance monitoring and adaptive parameter
adjustment based on system performance and data characteristics.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np
import pandas as pd

from src.utils.logger import system_logger

class PerformanceLevel(Enum):
    """Performance levels for adaptive behavior."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    """Performance metrics for SR detection methods."""
    method_name: str
    execution_time: float
    memory_usage: float
    memory_delta: float
    data_size: int
    result_count: int
    timestamp: float
    cpu_percent: float = 0.0
    error_occurred: bool = False
    error_message: str = ""

@dataclass
class AdaptiveParameters:
    """Adaptive parameters based on performance."""
    batch_size: int = 1000
    max_memory_mb: int = 1000
    timeout_seconds: float = 30.0
    enable_caching: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    quality_threshold: float = 0.8
    performance_level: PerformanceLevel = PerformanceLevel.GOOD

class SRPerformanceMonitor:
    """Real-time performance monitoring for SR detection."""
    
    def __init__(self, history_size: int = 1000):
        self.logger = system_logger.getChild('SRPerformanceMonitor')
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.adaptive_params = AdaptiveParameters()
        self.performance_levels = self._initialize_performance_levels()
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'execution_time': {
                'excellent': 1.0,
                'good': 3.0,
                'fair': 10.0,
                'poor': 30.0
            },
            'memory_usage': {
                'excellent': 100,
                'good': 500,
                'fair': 1000,
                'poor': 2000
            },
            'memory_delta': {
                'excellent': 10,
                'good': 50,
                'fair': 100,
                'poor': 200
            },
            'cpu_percent': {
                'excellent': 20,
                'good': 50,
                'fair': 80,
                'poor': 95
            }
        }
    
    def _initialize_performance_levels(self) -> Dict[str, PerformanceLevel]:
        """Initialize performance level mappings."""
        return {
            'fractal': PerformanceLevel.GOOD,
            'pivot': PerformanceLevel.GOOD,
            'volume': PerformanceLevel.GOOD,
            'statistical': PerformanceLevel.GOOD,
            'psychological': PerformanceLevel.GOOD,
            'fibonacci': PerformanceLevel.GOOD,
            'trendline': PerformanceLevel.GOOD,
            'channel': PerformanceLevel.GOOD,
            'volume_profile': PerformanceLevel.GOOD,
            'market_structure': PerformanceLevel.GOOD
        }
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_adaptive_parameters()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Update performance level for this method
            performance_level = self._calculate_performance_level(metrics)
            self.performance_levels[metrics.method_name] = performance_level
            
            # Log performance if concerning
            if performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                self.logger.warning(
                    f"Poor performance detected - {metrics.method_name}: "
                    f"Time: {metrics.execution_time:.2f}s, "
                    f"Memory: {metrics.memory_usage:.1f}MB, "
                    f"CPU: {metrics.cpu_percent:.1f}%"
                )
    
    def _calculate_performance_level(self, metrics: PerformanceMetrics) -> PerformanceLevel:
        """Calculate performance level based on metrics."""
        time_level = self._get_level_for_metric('execution_time', metrics.execution_time)
        memory_level = self._get_level_for_metric('memory_usage', metrics.memory_usage)
        cpu_level = self._get_level_for_metric('cpu_percent', metrics.cpu_percent)
        
        # Return the worst level
        levels = [time_level, memory_level, cpu_level]
        if PerformanceLevel.CRITICAL in levels:
            return PerformanceLevel.CRITICAL
        elif PerformanceLevel.POOR in levels:
            return PerformanceLevel.POOR
        elif PerformanceLevel.FAIR in levels:
            return PerformanceLevel.FAIR
        elif PerformanceLevel.GOOD in levels:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.EXCELLENT
    
    def _get_level_for_metric(self, metric_name: str, value: float) -> PerformanceLevel:
        """Get performance level for a specific metric."""
        thresholds = self.thresholds[metric_name]
        
        if value <= thresholds['excellent']:
            return PerformanceLevel.EXCELLENT
        elif value <= thresholds['good']:
            return PerformanceLevel.GOOD
        elif value <= thresholds['fair']:
            return PerformanceLevel.FAIR
        elif value <= thresholds['poor']:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on recent performance."""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 metrics
        
        # Calculate average performance
        avg_time = np.mean([m.execution_time for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        
        # Update parameters based on performance
        if avg_time > 10.0:  # Slow execution
            self.adaptive_params.batch_size = max(100, self.adaptive_params.batch_size // 2)
            self.adaptive_params.timeout_seconds = min(60.0, self.adaptive_params.timeout_seconds * 1.5)
        
        if avg_memory > 1000:  # High memory usage
            self.adaptive_params.max_memory_mb = max(500, self.adaptive_params.max_memory_mb // 2)
            self.adaptive_params.enable_caching = False
        
        if avg_cpu > 80:  # High CPU usage
            self.adaptive_params.max_workers = max(1, self.adaptive_params.max_workers - 1)
            self.adaptive_params.enable_parallel = False
        
        # Reset parameters if performance improves
        if avg_time < 2.0 and avg_memory < 500 and avg_cpu < 50:
            self.adaptive_params.batch_size = min(5000, self.adaptive_params.batch_size * 2)
            self.adaptive_params.max_memory_mb = min(2000, self.adaptive_params.max_memory_mb * 2)
            self.adaptive_params.max_workers = min(8, self.adaptive_params.max_workers + 1)
            self.adaptive_params.enable_parallel = True
            self.adaptive_params.enable_caching = True
    
    def get_adaptive_parameters(self, method_name: str) -> AdaptiveParameters:
        """Get adaptive parameters for a specific method."""
        # Get method-specific performance level
        method_level = self.performance_levels.get(method_name, PerformanceLevel.GOOD)
        
        # Adjust parameters based on method performance
        params = AdaptiveParameters()
        params.batch_size = self.adaptive_params.batch_size
        params.max_memory_mb = self.adaptive_params.max_memory_mb
        params.timeout_seconds = self.adaptive_params.timeout_seconds
        params.enable_caching = self.adaptive_params.enable_caching
        params.enable_parallel = self.adaptive_params.enable_parallel
        params.max_workers = self.adaptive_params.max_workers
        params.quality_threshold = self.adaptive_params.quality_threshold
        params.performance_level = method_level
        
        # Method-specific adjustments
        if method_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
            params.batch_size = max(100, params.batch_size // 2)
            params.timeout_seconds = min(60.0, params.timeout_seconds * 2)
            params.enable_parallel = False
            params.max_workers = 1
        
        return params
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all methods."""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        summary = {}
        method_stats = {}
        
        for metrics in self.metrics_history:
            method = metrics.method_name
            if method not in method_stats:
                method_stats[method] = {
                    'execution_times': [],
                    'memory_usage': [],
                    'cpu_percent': [],
                    'error_count': 0,
                    'total_calls': 0
                }
            
            stats = method_stats[method]
            stats['execution_times'].append(metrics.execution_time)
            stats['memory_usage'].append(metrics.memory_usage)
            stats['cpu_percent'].append(metrics.cpu_percent)
            stats['total_calls'] += 1
            
            if metrics.error_occurred:
                stats['error_count'] += 1
        
        # Calculate statistics for each method
        for method, stats in method_stats.items():
            if stats['total_calls'] > 0:
                summary[method] = {
                    'total_calls': stats['total_calls'],
                    'error_rate': stats['error_count'] / stats['total_calls'],
                    'avg_execution_time': np.mean(stats['execution_times']),
                    'max_execution_time': np.max(stats['execution_times']),
                    'avg_memory_usage': np.mean(stats['memory_usage']),
                    'max_memory_usage': np.max(stats['memory_usage']),
                    'avg_cpu_percent': np.mean(stats['cpu_percent']),
                    'performance_level': self.performance_levels.get(method, PerformanceLevel.GOOD).value
                }
        
        return summary
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'monitoring_active': self.monitoring_active,
                'metrics_count': len(self.metrics_history)
            }
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

def performance_monitor_decorator(monitor: SRPerformanceMonitor):
    """Decorator to automatically record performance metrics."""
    def decorator(func: Callable) -> Callable:
        def wrapper(self, data, *args, **kwargs):
            method_name = func.__name__.replace('_detect_', '').replace('_levels', '')
            start_time = time.time()
            start_memory = 0
            start_cpu = 0
            
            try:
                # Get initial system metrics
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                start_cpu = psutil.cpu_percent()
                
                # Execute method
                result = func(self, data, *args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                end_cpu = psutil.cpu_percent()
                
                # Record metrics
                metrics = PerformanceMetrics(
                    method_name=method_name,
                    execution_time=execution_time,
                    memory_usage=end_memory,
                    memory_delta=memory_delta,
                    data_size=len(data) if data is not None else 0,
                    result_count=len(result) if result is not None else 0,
                    timestamp=time.time(),
                    cpu_percent=end_cpu
                )
                
                monitor.record_metrics(metrics)
                
                return result
                
            except Exception as e:
                # Record error metrics
                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                
                metrics = PerformanceMetrics(
                    method_name=method_name,
                    execution_time=execution_time,
                    memory_usage=end_memory,
                    memory_delta=memory_delta,
                    data_size=len(data) if data is not None else 0,
                    result_count=0,
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(),
                    error_occurred=True,
                    error_message=str(e)
                )
                
                monitor.record_metrics(metrics)
                raise
        
        return wrapper
    return decorator