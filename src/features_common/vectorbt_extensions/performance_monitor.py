"""
VectorBT Performance Monitor with Hardware Optimization

This module provides comprehensive performance monitoring for VectorBT operations
including hardware optimization metrics, adaptive learning tracking, and
intelligent performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    data_size: int
    optimization_strategy: str
    hardware_optimized: bool
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSummary:
    """Performance summary data structure."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_duration: float
    min_duration: float
    max_duration: float
    std_duration: float
    avg_memory_usage: float
    avg_cpu_usage: float
    avg_gpu_usage: float
    hardware_optimization_rate: float
    optimization_effectiveness: Dict[str, float]
    performance_trend: str
    recommendations: List[str]

class VectorBTPerformanceMonitor:
    """
    Comprehensive performance monitor for VectorBT operations with hardware optimization.
    
    This monitor provides detailed performance tracking including hardware optimization
    metrics, adaptive learning analysis, and intelligent performance recommendations.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics: List[PerformanceMetric] = []
        self.current_operations: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[PerformanceMetric] = []
        self.monitoring_lock = threading.Lock()
        
        # Hardware optimization tracking
        self.hardware_stats = {
            'total_hardware_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'adaptive_decisions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'chunked_operations': 0,
            'memory_savings_mb': 0.0,
            'performance_improvements': []
        }
        
        # Performance trends
        self.performance_trends = {
            'duration_trend': [],
            'memory_trend': [],
            'cpu_trend': [],
            'gpu_trend': []
        }
        
        # Monitoring configuration
        self.max_history_size = 10000
        self.trend_window_size = 100
        self.performance_threshold = 1.0  # seconds
        
        logger.debug("VectorBT Performance Monitor initialized with hardware optimization support")

    def start_monitoring(self, operation: str, data_size: int = 0, 
                        optimization_strategy: str = 'standard',
                        hardware_optimized: bool = False) -> str:
        """
        Start monitoring an operation with comprehensive metrics.
        
        Args:
            operation: Name of the operation
            data_size: Size of the data being processed
            optimization_strategy: Optimization strategy being used
            hardware_optimized: Whether hardware optimization is enabled
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        
        with self.monitoring_lock:
            self.current_operations[operation_id] = {
                'operation': operation,
                'start_time': time.time(),
                'data_size': data_size,
                'optimization_strategy': optimization_strategy,
                'hardware_optimized': hardware_optimized,
                'memory_usage_start': self._get_memory_usage(),
                'cpu_usage_start': self._get_cpu_usage(),
                'gpu_usage_start': self._get_gpu_usage()
            }
        
        logger.debug(f"Started monitoring operation: {operation} (ID: {operation_id})")
        return operation_id

    def stop_monitoring(self, operation_id: str, success: bool = True, 
                       error_message: Optional[str] = None,
                       additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Stop monitoring an operation and record comprehensive metrics.
        
        Args:
            operation_id: Operation ID returned by start_monitoring
            success: Whether the operation was successful
            error_message: Error message if operation failed
            additional_metrics: Additional metrics to record
        """
        with self.monitoring_lock:
            if operation_id not in self.current_operations:
                logger.warning(f"Operation ID {operation_id} not found in current operations")
                return
            
            operation_data = self.current_operations.pop(operation_id)
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - operation_data['start_time']
            
            memory_usage = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            gpu_usage = self._get_gpu_usage()
            
            # Create performance metric
            metric = PerformanceMetric(
                operation=operation_data['operation'],
                start_time=operation_data['start_time'],
                end_time=end_time,
                duration=duration,
                memory_usage_mb=memory_usage - operation_data['memory_usage_start'],
                cpu_usage_percent=cpu_usage,
                gpu_usage_percent=gpu_usage,
                data_size=operation_data['data_size'],
                optimization_strategy=operation_data['optimization_strategy'],
                hardware_optimized=operation_data['hardware_optimized'],
                success=success,
                error_message=error_message,
                additional_metrics=additional_metrics or {}
            )
            
            # Store metric
            self.metrics.append(metric)
            self.performance_history.append(metric)
            
            # Update hardware stats
            if operation_data['hardware_optimized']:
                self.hardware_stats['total_hardware_operations'] += 1
                
                if 'memory_optimization' in operation_data.get('optimization_strategy', ''):
                    self.hardware_stats['memory_optimizations'] += 1
                
                if 'gpu' in operation_data.get('optimization_strategy', ''):
                    self.hardware_stats['gpu_operations'] += 1
                
                if 'adaptive' in operation_data.get('optimization_strategy', ''):
                    self.hardware_stats['adaptive_decisions'] += 1
            
            # Update performance trends
            self._update_performance_trends(metric)
            
            # Clean up old history
            if len(self.performance_history) > self.max_history_size:
                self.performance_history = self.performance_history[-self.max_history_size:]
            
            logger.debug(f"Stopped monitoring operation: {operation_data['operation']} (Duration: {duration:.4f}s)")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0

    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        try:
            # This would need to be implemented based on the specific GPU monitoring library
            # For now, return 0
            return 0.0
        except ImportError:
            return 0.0

    def _update_performance_trends(self, metric: PerformanceMetric) -> None:
        """Update performance trends with new metric."""
        # Update duration trend
        self.performance_trends['duration_trend'].append(metric.duration)
        if len(self.performance_trends['duration_trend']) > self.trend_window_size:
            self.performance_trends['duration_trend'] = self.performance_trends['duration_trend'][-self.trend_window_size:]
        
        # Update memory trend
        self.performance_trends['memory_trend'].append(metric.memory_usage_mb)
        if len(self.performance_trends['memory_trend']) > self.trend_window_size:
            self.performance_trends['memory_trend'] = self.performance_trends['memory_trend'][-self.trend_window_size:]
        
        # Update CPU trend
        self.performance_trends['cpu_trend'].append(metric.cpu_usage_percent)
        if len(self.performance_trends['cpu_trend']) > self.trend_window_size:
            self.performance_trends['cpu_trend'] = self.performance_trends['cpu_trend'][-self.trend_window_size:]
        
        # Update GPU trend
        self.performance_trends['gpu_trend'].append(metric.gpu_usage_percent)
        if len(self.performance_trends['gpu_trend']) > self.trend_window_size:
            self.performance_trends['gpu_trend'] = self.performance_trends['gpu_trend'][-self.trend_window_size:]

    def get_performance_summary(self) -> PerformanceSummary:
        """Get comprehensive performance summary."""
        if not self.metrics:
            return PerformanceSummary(
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                avg_duration=0.0,
                min_duration=0.0,
                max_duration=0.0,
                std_duration=0.0,
                avg_memory_usage=0.0,
                avg_cpu_usage=0.0,
                avg_gpu_usage=0.0,
                hardware_optimization_rate=0.0,
                optimization_effectiveness={},
                performance_trend='stable',
                recommendations=[]
            )
        
        # Calculate basic statistics
        successful_metrics = [m for m in self.metrics if m.success]
        failed_metrics = [m for m in self.metrics if not m.success]
        
        durations = [m.duration for m in successful_metrics]
        memory_usages = [m.memory_usage_mb for m in successful_metrics]
        cpu_usages = [m.cpu_usage_percent for m in successful_metrics]
        gpu_usages = [m.gpu_usage_percent for m in successful_metrics]
        
        # Calculate hardware optimization rate
        hardware_optimized_count = sum(1 for m in self.metrics if m.hardware_optimized)
        hardware_optimization_rate = hardware_optimized_count / len(self.metrics) if self.metrics else 0.0
        
        # Calculate optimization effectiveness
        optimization_effectiveness = self._calculate_optimization_effectiveness()
        
        # Determine performance trend
        performance_trend = self._determine_performance_trend()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return PerformanceSummary(
            total_operations=len(self.metrics),
            successful_operations=len(successful_metrics),
            failed_operations=len(failed_metrics),
            avg_duration=statistics.mean(durations) if durations else 0.0,
            min_duration=min(durations) if durations else 0.0,
            max_duration=max(durations) if durations else 0.0,
            std_duration=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            avg_memory_usage=statistics.mean(memory_usages) if memory_usages else 0.0,
            avg_cpu_usage=statistics.mean(cpu_usages) if cpu_usages else 0.0,
            avg_gpu_usage=statistics.mean(gpu_usages) if gpu_usages else 0.0,
            hardware_optimization_rate=hardware_optimization_rate,
            optimization_effectiveness=optimization_effectiveness,
            performance_trend=performance_trend,
            recommendations=recommendations
        )

    def _calculate_optimization_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness of different optimization strategies."""
        strategy_groups = {}
        for metric in self.metrics:
            strategy = metric.optimization_strategy
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(metric.duration)
        
        effectiveness = {}
        for strategy, durations in strategy_groups.items():
            if durations:
                effectiveness[strategy] = {
                    'avg_duration': statistics.mean(durations),
                    'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0.0,
                    'success_rate': sum(1 for m in self.metrics if m.optimization_strategy == strategy and m.success) / len(durations)
                }
        
        return effectiveness

    def _determine_performance_trend(self) -> str:
        """Determine the performance trend based on recent metrics."""
        if len(self.performance_trends['duration_trend']) < 10:
            return 'insufficient_data'
        
        recent_durations = self.performance_trends['duration_trend'][-10:]
        older_durations = self.performance_trends['duration_trend'][-20:-10] if len(self.performance_trends['duration_trend']) >= 20 else recent_durations
        
        recent_avg = statistics.mean(recent_durations)
        older_avg = statistics.mean(older_durations)
        
        if recent_avg < older_avg * 0.9:
            return 'improving'
        elif recent_avg > older_avg * 1.1:
            return 'degrading'
        else:
            return 'stable'

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        # Check performance threshold
        slow_operations = [m for m in self.metrics if m.duration > self.performance_threshold]
        if len(slow_operations) > len(self.metrics) * 0.3:
            recommendations.append("Consider enabling hardware optimization for slow operations")
        
        # Check memory usage
        high_memory_operations = [m for m in self.metrics if m.memory_usage_mb > 100]
        if len(high_memory_operations) > len(self.metrics) * 0.2:
            recommendations.append("High memory usage detected - consider enabling memory optimization")
        
        # Check hardware optimization rate
        hardware_optimized_count = sum(1 for m in self.metrics if m.hardware_optimized)
        if hardware_optimized_count < len(self.metrics) * 0.5:
            recommendations.append("Low hardware optimization usage - consider enabling for more operations")
        
        # Check failure rate
        failed_count = sum(1 for m in self.metrics if not m.success)
        if failed_count > len(self.metrics) * 0.1:
            recommendations.append("High failure rate detected - check error handling and fallback mechanisms")
        
        return recommendations

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware optimization statistics."""
        return self.hardware_stats.copy()

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_metrics = [m for m in self.metrics if m.operation == operation]
        
        if not operation_metrics:
            return {'message': f'No metrics found for operation: {operation}'}
        
        durations = [m.duration for m in operation_metrics if m.success]
        memory_usages = [m.memory_usage_mb for m in operation_metrics if m.success]
        
        return {
            'total_calls': len(operation_metrics),
            'successful_calls': sum(1 for m in operation_metrics if m.success),
            'failed_calls': sum(1 for m in operation_metrics if not m.success),
            'avg_duration': statistics.mean(durations) if durations else 0.0,
            'min_duration': min(durations) if durations else 0.0,
            'max_duration': max(durations) if durations else 0.0,
            'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0.0,
            'avg_memory_usage': statistics.mean(memory_usages) if memory_usages else 0.0,
            'hardware_optimized_calls': sum(1 for m in operation_metrics if m.hardware_optimized)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return {
            'metrics': [
                {
                    'operation': m.operation,
                    'start_time': m.start_time,
                    'end_time': m.end_time,
                    'duration': m.duration,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_usage_percent': m.cpu_usage_percent,
                    'gpu_usage_percent': m.gpu_usage_percent,
                    'data_size': m.data_size,
                    'optimization_strategy': m.optimization_strategy,
                    'hardware_optimized': m.hardware_optimized,
                    'success': m.success,
                    'error_message': m.error_message,
                    'additional_metrics': m.additional_metrics
                }
                for m in self.metrics
            ],
            'summary': self.get_performance_summary(),
            'hardware_stats': self.get_hardware_stats()
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self.monitoring_lock:
            self.metrics.clear()
            self.performance_history.clear()
            self.current_operations.clear()
            self.hardware_stats = {
                'total_hardware_operations': 0,
                'memory_optimizations': 0,
                'gpu_operations': 0,
                'adaptive_decisions': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'chunked_operations': 0,
                'memory_savings_mb': 0.0,
                'performance_improvements': []
            }
            self.performance_trends = {
                'duration_trend': [],
                'memory_trend': [],
                'cpu_trend': [],
                'gpu_trend': []
            }
        
        logger.info("Performance metrics reset")

def get_performance_monitor() -> VectorBTPerformanceMonitor:
    """Get the performance monitor instance."""
    return VectorBTPerformanceMonitor()
