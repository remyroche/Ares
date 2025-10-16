"""
Performance Monitor for NAS/TAS Pipeline

This module provides comprehensive performance monitoring using tprint_timer
throughout the entire NAS/TAS joint pipeline, ensuring detailed performance
tracking and optimization insights.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import threading
from pathlib import Path
import json

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, tprint_logged, LogLevel
)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Timing metrics
    execution_time: float = 0.0
    cpu_time: float = 0.0
    memory_usage: float = 0.0

    # Pipeline metrics
    data_processing_time: float = 0.0
    architecture_search_time: float = 0.0
    model_training_time: float = 0.0
    evaluation_time: float = 0.0
    result_storage_time: float = 0.0

    # Component metrics
    component_times: Dict[str, float] = field(default_factory=dict)
    operation_times: Dict[str, float] = field(default_factory=dict)

    # Performance indicators
    throughput: float = 0.0
    efficiency: float = 0.0
    resource_utilization: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    pipeline_type: str = "unknown"
    execution_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'cpu_time': self.cpu_time,
            'memory_usage': self.memory_usage,
            'data_processing_time': self.data_processing_time,
            'architecture_search_time': self.architecture_search_time,
            'model_training_time': self.model_training_time,
            'evaluation_time': self.evaluation_time,
            'result_storage_time': self.result_storage_time,
            'component_times': self.component_times,
            'operation_times': self.operation_times,
            'throughput': self.throughput,
            'efficiency': self.efficiency,
            'resource_utilization': self.resource_utilization,
            'timestamp': self.timestamp.isoformat(),
            'pipeline_type': self.pipeline_type,
            'execution_id': self.execution_id
        }

class PerformanceMonitor:
    """Comprehensive performance monitor for NAS/TAS pipeline."""

    def __init__(self, enable_monitoring: bool = True, output_directory: str = "nas_tas_output"):
        """Initialize performance monitor."""
        tprint_info("Initializing Performance Monitor")

        self.enable_monitoring = enable_monitoring
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.operation_timers: Dict[str, float] = {}
        self.component_timers: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.Lock()

        tprint_success("Performance Monitor initialized successfully")

    def start_monitoring(self, pipeline_type: str, execution_id: str = None) -> PerformanceMetrics:
        """Start performance monitoring for a pipeline execution."""
        if not self.enable_monitoring:
            return PerformanceMetrics()

        tprint_info(f"Starting performance monitoring for {pipeline_type} pipeline")

        with self._lock:
            self.current_metrics = PerformanceMetrics()
            self.current_metrics.pipeline_type = pipeline_type
            self.current_metrics.execution_id = execution_id or f"exec_{int(time.time())}"
            self.current_metrics.timestamp = datetime.now()

        tprint_structured({
            "monitoring_started": {
                "pipeline_type": pipeline_type,
                "execution_id": self.current_metrics.execution_id,
                "timestamp": self.current_metrics.timestamp.isoformat()
            }
        }, LogLevel.INFO)

        return self.current_metrics

    def stop_monitoring(self) -> Optional[PerformanceMetrics]:
        """Stop performance monitoring and return final metrics."""
        if not self.enable_monitoring or not self.current_metrics:
            return None

        tprint_info("Stopping performance monitoring")

        with self._lock:
            # Calculate final metrics
            self.current_metrics.execution_time = time.time() - self.current_metrics.timestamp.timestamp()

            # Store metrics
            self.metrics_history.append(self.current_metrics)

            # Log final metrics
            tprint_structured({
                "performance_summary": self.current_metrics.to_dict()
            }, LogLevel.INFO)

            tprint_success(f"Performance monitoring completed: {self.current_metrics.execution_time:.2f}s")

            # Save metrics to file
            self._save_metrics(self.current_metrics)

            final_metrics = self.current_metrics
            self.current_metrics = None

            return final_metrics

    @contextmanager
    def monitor_operation(self, operation_name: str, component: str = "general"):
        """Context manager for monitoring individual operations."""
        if not self.enable_monitoring:
            yield
            return

        tprint_debug(f"Monitoring operation: {operation_name} in {component}")

        start_time = time.time()
        try:
            with tprint_timer(operation_name, LogLevel.DEBUG):
                yield
        finally:
            duration = time.time() - start_time

            # Update metrics
            if self.current_metrics:
                with self._lock:
                    self.current_metrics.operation_times[operation_name] = duration
                    if component not in self.current_metrics.component_times:
                        self.current_metrics.component_times[component] = 0.0
                    self.current_metrics.component_times[component] += duration

            tprint_performance(operation_name, duration)

    @contextmanager
    def monitor_component(self, component_name: str):
        """Context manager for monitoring pipeline components."""
        if not self.enable_monitoring:
            yield
            return

        tprint_debug(f"Monitoring component: {component_name}")

        start_time = time.time()
        try:
            with tprint_timer(f"{component_name}_component", LogLevel.INFO):
                yield
        finally:
            duration = time.time() - start_time

            # Update component-specific metrics
            if self.current_metrics:
                with self._lock:
                    self.current_metrics.component_times[component_name] = duration

                    # Update specific pipeline metrics
                    if component_name == "data_processing":
                        self.current_metrics.data_processing_time = duration
                    elif component_name == "architecture_search":
                        self.current_metrics.architecture_search_time = duration
                    elif component_name == "model_training":
                        self.current_metrics.model_training_time = duration
                    elif component_name == "evaluation":
                        self.current_metrics.evaluation_time = duration
                    elif component_name == "result_storage":
                        self.current_metrics.result_storage_time = duration

            tprint_performance(f"{component_name} component", duration)

    def monitor_function(self, function_name: str = None, component: str = "general"):
        """Decorator for monitoring function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_monitoring:
                    return func(*args, **kwargs)

                name = function_name or func.__name__
                with self.monitor_operation(name, component):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def monitor_async_function(self, function_name: str = None, component: str = "general"):
        """Decorator for monitoring async function performance."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.enable_monitoring:
                    return await func(*args, **kwargs)

                name = function_name or func.__name__
                with self.monitor_operation(name, component):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executions."""
        if not self.metrics_history:
            return {"message": "No performance data available"}

        # Calculate aggregate metrics
        total_executions = len(self.metrics_history)
        total_time = sum(m.execution_time for m in self.metrics_history)
        avg_time = total_time / total_executions if total_executions > 0 else 0

        # Component averages
        component_averages = {}
        for component in ["data_processing", "architecture_search", "model_training", "evaluation", "result_storage"]:
            times = [getattr(m, f"{component}_time", 0) for m in self.metrics_history]
            component_averages[component] = sum(times) / len(times) if times else 0

        summary = {
            "total_executions": total_executions,
            "total_time": total_time,
            "average_execution_time": avg_time,
            "component_averages": component_averages,
            "latest_execution": self.metrics_history[-1].to_dict() if self.metrics_history else None
        }

        tprint_structured({"performance_summary": summary}, LogLevel.INFO)
        return summary

    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """Export performance metrics to file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_directory / f"performance_metrics_{timestamp}.json"
        else:
            filepath = Path(filepath)

        # Prepare export data
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_executions": len(self.metrics_history),
            "metrics": [m.to_dict() for m in self.metrics_history],
            "summary": self.get_performance_summary()
        }

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        tprint_success(f"Performance metrics exported to: {filepath}")
        return str(filepath)

    def _save_metrics(self, metrics: PerformanceMetrics):
        """Save metrics to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{metrics.execution_id}_{timestamp}.json"
            filepath = self.output_directory / filename

            with open(filepath, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)

            tprint_debug(f"Performance metrics saved to: {filepath}")
        except Exception as e:
            tprint_error(f"Failed to save performance metrics: {e}")

# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def setup_performance_monitoring(
    enable_monitoring: bool = True,
    output_directory: str = "nas_tas_output"
) -> PerformanceMonitor:
    """Setup performance monitoring for NAS/TAS pipeline."""
    global _global_monitor
    _global_monitor = PerformanceMonitor(enable_monitoring, output_directory)
    return _global_monitor

# Convenience decorators for common pipeline components
def monitor_data_processing(func):
    """Decorator for monitoring data processing operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_function(func.__name__, "data_processing")(func)

def monitor_architecture_search(func):
    """Decorator for monitoring architecture search operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_function(func.__name__, "architecture_search")(func)

def monitor_model_training(func):
    """Decorator for monitoring model training operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_function(func.__name__, "model_training")(func)

def monitor_evaluation(func):
    """Decorator for monitoring evaluation operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_function(func.__name__, "evaluation")(func)

def monitor_result_storage(func):
    """Decorator for monitoring result storage operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_function(func.__name__, "result_storage")(func)

# Async versions
def monitor_async_data_processing(func):
    """Decorator for monitoring async data processing operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_function(func.__name__, "data_processing")(func)

def monitor_async_architecture_search(func):
    """Decorator for monitoring async architecture search operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_function(func.__name__, "architecture_search")(func)

def monitor_async_model_training(func):
    """Decorator for monitoring async model training operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_function(func.__name__, "model_training")(func)

def monitor_async_evaluation(func):
    """Decorator for monitoring async evaluation operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_function(func.__name__, "evaluation")(func)

def monitor_async_result_storage(func):
    """Decorator for monitoring async result storage operations."""
    monitor = get_performance_monitor()
    return monitor.monitor_async_function(func.__name__, "result_storage")(func)
