"""
Performance monitoring utilities for feature selection.

This module provides comprehensive performance monitoring and profiling
capabilities for feature selection operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import time
import psutil
import threading
from dataclasses import dataclass, field
from contextlib import contextmanager

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success


@dataclass
class PerformanceMetrics:
    """Performance metrics for feature selection operations."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    operations_count: int
    data_shape: tuple
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSummary:
    """Summary of performance monitoring results."""
    total_operations: int
    total_execution_time: float
    average_execution_time: float
    peak_memory_usage_mb: float
    average_memory_usage_mb: float
    success_rate: float
    operations: List[PerformanceMetrics]
    recommendations: List[str]


class PerformanceMonitor:
    """Performance monitoring for feature selection operations."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.logger = get_logger("PerformanceMonitor")
        
        # Performance tracking
        self.operations: List[PerformanceMetrics] = []
        self.start_time = None
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # System monitoring
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        
        if self.enable_monitoring:
            tprint_success("✅ Performance monitoring enabled")
        else:
            tprint_warning("⚠️ Performance monitoring disabled")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 ** 2)  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent()
        except Exception:
            return 0.0
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if not self.enable_monitoring:
            return
        
        tprint_debug("📊 Starting performance monitoring")
        self.start_time = time.time()
        self.stop_monitoring = False
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        if not self.enable_monitoring:
            return
        
        tprint_debug("📊 Stopping performance monitoring")
        self.stop_monitoring = True
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring:
            try:
                current_memory = self._get_memory_usage()
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                time.sleep(0.1)  # Monitor every 100ms
            except Exception as e:
                tprint_warning(f"⚠️ Monitoring loop error: {e}")
                break
    
    @contextmanager
    def monitor_operation(self, operation_name: str, data_shape: tuple = None):
        """Context manager for monitoring individual operations."""
        if not self.enable_monitoring:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        
        success = True
        error_message = None
        
        try:
            tprint_debug(f"📊 Monitoring operation: {operation_name}")
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=end_cpu,
                peak_memory_mb=self.peak_memory,
                operations_count=len(self.operations) + 1,
                data_shape=data_shape or (0, 0),
                success=success,
                error_message=error_message
            )
            
            self.operations.append(metrics)
            
            tprint_debug(f"   📊 Operation completed: {execution_time:.3f}s, {memory_usage:.1f}MB")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a function execution."""
        if not self.enable_monitoring:
            return func(*args, **kwargs)
        
        operation_name = f"{func.__name__}_profile"
        data_shape = None
        
        # Try to extract data shape from arguments
        for arg in args:
            if isinstance(arg, (pd.DataFrame, np.ndarray)):
                data_shape = arg.shape
                break
        
        with self.monitor_operation(operation_name, data_shape):
            return func(*args, **kwargs)
    
    def get_performance_summary(self) -> PerformanceSummary:
        """Get comprehensive performance summary."""
        tprint_debug("📊 Generating performance summary")
        
        if not self.operations:
            return PerformanceSummary(
                total_operations=0,
                total_execution_time=0.0,
                average_execution_time=0.0,
                peak_memory_usage_mb=0.0,
                average_memory_usage_mb=0.0,
                success_rate=0.0,
                operations=[],
                recommendations=[]
            )
        
        # Calculate summary statistics
        total_operations = len(self.operations)
        total_execution_time = sum(op.execution_time for op in self.operations)
        average_execution_time = total_execution_time / total_operations
        
        peak_memory_usage = max(op.peak_memory_mb for op in self.operations)
        average_memory_usage = np.mean([op.memory_usage_mb for op in self.operations])
        
        successful_operations = sum(1 for op in self.operations if op.success)
        success_rate = successful_operations / total_operations
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        summary = PerformanceSummary(
            total_operations=total_operations,
            total_execution_time=total_execution_time,
            average_execution_time=average_execution_time,
            peak_memory_usage_mb=peak_memory_usage,
            average_memory_usage_mb=average_memory_usage,
            success_rate=success_rate,
            operations=self.operations.copy(),
            recommendations=recommendations
        )
        
        tprint_debug(f"   📊 Summary: {total_operations} operations, {total_execution_time:.2f}s total")
        tprint_debug(f"   📊 Memory: {peak_memory_usage:.1f}MB peak, {average_memory_usage:.1f}MB avg")
        tprint_debug(f"   📊 Success rate: {success_rate:.1%}")
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if not self.operations:
            return recommendations
        
        # Check execution time
        avg_time = np.mean([op.execution_time for op in self.operations])
        if avg_time > 10.0:  # More than 10 seconds average
            recommendations.append("Consider optimizing slow operations (>10s average)")
        
        # Check memory usage
        peak_memory = max(op.peak_memory_mb for op in self.operations)
        if peak_memory > 1000:  # More than 1GB
            recommendations.append("High memory usage detected, consider chunked processing")
        
        # Check success rate
        success_rate = sum(1 for op in self.operations if op.success) / len(self.operations)
        if success_rate < 0.9:  # Less than 90% success
            recommendations.append("Low success rate detected, review error handling")
        
        # Check data size
        large_operations = [op for op in self.operations if op.data_shape[0] > 10000]
        if len(large_operations) > len(self.operations) * 0.5:
            recommendations.append("Many large dataset operations, consider VectorBT optimization")
        
        # Check CPU usage
        high_cpu_operations = [op for op in self.operations if op.cpu_usage_percent > 80]
        if len(high_cpu_operations) > len(self.operations) * 0.3:
            recommendations.append("High CPU usage detected, consider parallel processing")
        
        return recommendations
    
    def get_operation_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of operations by type."""
        tprint_debug("📊 Generating operation breakdown")
        
        if not self.operations:
            return {}
        
        # Group operations by name
        operation_groups = {}
        for op in self.operations:
            if op.operation_name not in operation_groups:
                operation_groups[op.operation_name] = []
            operation_groups[op.operation_name].append(op)
        
        # Calculate statistics for each operation type
        breakdown = {}
        for op_name, ops in operation_groups.items():
            times = [op.execution_time for op in ops]
            memories = [op.memory_usage_mb for op in ops]
            successes = [op.success for op in ops]
            
            breakdown[op_name] = {
                'count': len(ops),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_memory': sum(memories),
                'avg_memory': np.mean(memories),
                'success_rate': sum(successes) / len(successes),
                'operations': ops
            }
        
        tprint_debug(f"   📊 Breakdown: {len(breakdown)} operation types")
        return breakdown
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to file."""
        tprint_debug(f"📊 Exporting metrics to {filepath}")
        
        try:
            import json
            
            # Convert operations to serializable format
            serializable_operations = []
            for op in self.operations:
                serializable_operations.append({
                    'operation_name': op.operation_name,
                    'execution_time': op.execution_time,
                    'memory_usage_mb': op.memory_usage_mb,
                    'cpu_usage_percent': op.cpu_usage_percent,
                    'peak_memory_mb': op.peak_memory_mb,
                    'operations_count': op.operations_count,
                    'data_shape': op.data_shape,
                    'success': op.success,
                    'error_message': op.error_message,
                    'metadata': op.metadata
                })
            
            # Create export data
            export_data = {
                'summary': self.get_performance_summary().__dict__,
                'operations': serializable_operations,
                'breakdown': self.get_operation_breakdown()
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            tprint_success(f"   ✅ Metrics exported to {filepath}")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to export metrics: {e}")
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        tprint_debug("📊 Resetting performance metrics")
        
        self.operations.clear()
        self.start_time = None
        self.peak_memory = self.initial_memory
        
        tprint_success("   ✅ Performance metrics reset")
    
    def cleanup(self):
        """Cleanup performance monitoring resources."""
        tprint_info("🧹 Cleaning up performance monitor")
        
        try:
            # Stop monitoring
            self.stop_monitoring()
            
            # Reset metrics
            self.reset_metrics()
            
            tprint_success("   ✅ Performance monitor cleanup completed")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Performance monitor cleanup failed: {e}")