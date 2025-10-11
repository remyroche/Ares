"""
Performance Monitoring and Optimization

This module provides comprehensive performance monitoring and optimization
for OHLCV data processing across all exchanges.

Features:
- Real-time performance monitoring
- Memory usage tracking
- Processing time analysis
- Automatic optimization recommendations
- Performance benchmarking
- Resource usage alerts
"""

import time
import psutil
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import asyncio
from contextlib import contextmanager

# Import our unified components
from .unified_ohlcv_standardizer import ExchangeType
from .unified_exchange_interface import UnifiedExchangeManager

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    memory_usage_mb: float
    cpu_percent: float
    data_size: int
    success: bool
    error_message: Optional[str] = None
    exchange: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    active_threads: int
    network_io: Dict[str, int] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Real-time performance monitoring for OHLCV data processing.
    
    Tracks performance metrics, memory usage, and provides optimization
    recommendations for data processing operations.
    """
    
    def __init__(self, max_history: int = 1000):
        """Initialize the performance monitor"""
        self.max_history = max_history
        self.logger = system_logger.getChild("PerformanceMonitor")
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=max_history)
        self.system_metrics_history: deque = deque(maxlen=max_history)
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Performance thresholds
        self.thresholds = {
            'max_operation_time': 30.0,  # 30 seconds
            'max_memory_usage_mb': 1000,  # 1GB
            'max_cpu_percent': 80.0,  # 80%
            'min_success_rate': 0.95  # 95%
        }
        
        self.logger.info("✅ PerformanceMonitor initialized")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.stop_monitoring.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"✅ Performance monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_monitoring.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("✅ Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Check for performance alerts
                self._check_performance_alerts(system_metrics)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                active_threads=threading.active_count(),
                network_io=network_io
            )
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                active_threads=0
            )
    
    def _check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['max_cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 80:  # 80% memory usage
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > 90:  # 90% disk usage
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if alerts:
            self.logger.warning(f"Performance alerts: {'; '.join(alerts)}")
    
    @contextmanager
    def measure_operation(self, operation_name: str, exchange: str = None, **kwargs):
        """Context manager for measuring operation performance"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        start_cpu = psutil.cpu_percent()
        
        success = True
        error_message = None
        data_size = 0
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            end_cpu = psutil.cpu_percent()
            
            duration = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=datetime.fromtimestamp(start_time, tz=timezone.utc),
                end_time=datetime.fromtimestamp(end_time, tz=timezone.utc),
                duration=duration,
                memory_usage_mb=memory_usage,
                cpu_percent=(start_cpu + end_cpu) / 2,
                data_size=data_size,
                success=success,
                error_message=error_message,
                exchange=exchange,
                additional_metrics=kwargs
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.operation_counts[operation_name] += 1
            self.operation_times[operation_name].append(duration)
            
            # Log performance
            if success:
                self.logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s, "
                               f"memory: {memory_usage:.1f}MB")
            else:
                self.logger.error(f"Operation '{operation_name}' failed after {duration:.3f}s: {error_message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        # Calculate statistics for each operation
        operation_stats = {}
        for operation_name in self.operation_counts:
            times = self.operation_times[operation_name]
            if times:
                operation_stats[operation_name] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times),
                    'total_duration': sum(times)
                }
        
        # Calculate overall statistics
        all_durations = [m.duration for m in self.metrics_history]
        all_memory_usage = [m.memory_usage_mb for m in self.metrics_history]
        success_rate = sum(1 for m in self.metrics_history if m.success) / len(self.metrics_history)
        
        # Recent performance (last 10 operations)
        recent_metrics = list(self.metrics_history)[-10:]
        recent_avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            'total_operations': len(self.metrics_history),
            'success_rate': success_rate,
            'overall_stats': {
                'avg_duration': sum(all_durations) / len(all_durations) if all_durations else 0,
                'min_duration': min(all_durations) if all_durations else 0,
                'max_duration': max(all_durations) if all_durations else 0,
                'avg_memory_usage_mb': sum(all_memory_usage) / len(all_memory_usage) if all_memory_usage else 0
            },
            'recent_performance': {
                'avg_duration': recent_avg_duration,
                'operations_count': len(recent_metrics)
            },
            'operation_breakdown': operation_stats,
            'system_metrics': self._get_latest_system_metrics()
        }
    
    def _get_latest_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest system metrics"""
        if not self.system_metrics_history:
            return None
        
        latest = self.system_metrics_history[-1]
        return {
            'timestamp': latest.timestamp.isoformat(),
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'memory_available_mb': latest.memory_available_mb,
            'disk_usage_percent': latest.disk_usage_percent,
            'active_threads': latest.active_threads
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if not self.metrics_history:
            return ["No performance data available for recommendations"]
        
        # Analyze performance patterns
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 operations
        
        # Check for slow operations
        slow_operations = [m for m in recent_metrics if m.duration > self.thresholds['max_operation_time']]
        if slow_operations:
            recommendations.append(f"Consider optimizing slow operations: {len(slow_operations)} operations exceeded {self.thresholds['max_operation_time']}s")
        
        # Check memory usage
        high_memory_ops = [m for m in recent_metrics if m.memory_usage_mb > self.thresholds['max_memory_usage_mb']]
        if high_memory_ops:
            recommendations.append(f"High memory usage detected: {len(high_memory_ops)} operations exceeded {self.thresholds['max_memory_usage_mb']}MB")
        
        # Check success rate
        recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        if recent_success_rate < self.thresholds['min_success_rate']:
            recommendations.append(f"Low success rate: {recent_success_rate:.1%} (threshold: {self.thresholds['min_success_rate']:.1%})")
        
        # Check for memory leaks
        if len(recent_metrics) > 10:
            memory_trend = [m.memory_usage_mb for m in recent_metrics[-10:]]
            if all(memory_trend[i] <= memory_trend[i+1] for i in range(len(memory_trend)-1)):
                recommendations.append("Potential memory leak detected - memory usage consistently increasing")
        
        # Check CPU usage
        latest_system = self._get_latest_system_metrics()
        if latest_system and latest_system['cpu_percent'] > self.thresholds['max_cpu_percent']:
            recommendations.append(f"High CPU usage: {latest_system['cpu_percent']:.1f}% - consider reducing concurrent operations")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds")
        
        return recommendations
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export performance metrics to file"""
        try:
            if format.lower() == 'json':
                # Convert metrics to JSON-serializable format
                export_data = {
                    'metrics': [
                        {
                            'operation_name': m.operation_name,
                            'start_time': m.start_time.isoformat(),
                            'end_time': m.end_time.isoformat(),
                            'duration': m.duration,
                            'memory_usage_mb': m.memory_usage_mb,
                            'cpu_percent': m.cpu_percent,
                            'data_size': m.data_size,
                            'success': m.success,
                            'error_message': m.error_message,
                            'exchange': m.exchange,
                            'additional_metrics': m.additional_metrics
                        }
                        for m in self.metrics_history
                    ],
                    'system_metrics': [
                        {
                            'timestamp': m.timestamp.isoformat(),
                            'cpu_percent': m.cpu_percent,
                            'memory_percent': m.memory_percent,
                            'memory_available_mb': m.memory_available_mb,
                            'disk_usage_percent': m.disk_usage_percent,
                            'active_threads': m.active_threads,
                            'network_io': m.network_io
                        }
                        for m in self.system_metrics_history
                    ],
                    'summary': self.get_performance_summary()
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.logger.info(f"✅ Performance metrics exported to {filepath}")
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            raise
    
    def clear_metrics(self):
        """Clear all performance metrics"""
        self.metrics_history.clear()
        self.system_metrics_history.clear()
        self.operation_counts.clear()
        self.operation_times.clear()
        self.logger.info("✅ Performance metrics cleared")


class PerformanceOptimizer:
    """
    Performance optimizer for OHLCV data processing operations.
    
    Provides automatic optimization recommendations and implements
    performance improvements for data processing workflows.
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize the performance optimizer"""
        self.monitor = monitor
        self.logger = system_logger.getChild("PerformanceOptimizer")
        
        # Optimization strategies
        self.optimization_strategies = {
            'memory_optimization': self._optimize_memory_usage,
            'processing_optimization': self._optimize_processing,
            'concurrency_optimization': self._optimize_concurrency,
            'caching_optimization': self._optimize_caching
        }
        
        self.logger.info("✅ PerformanceOptimizer initialized")
    
    def analyze_and_optimize(self) -> Dict[str, Any]:
        """Analyze current performance and provide optimization recommendations"""
        analysis_result = {
            'current_performance': self.monitor.get_performance_summary(),
            'recommendations': self.monitor.get_optimization_recommendations(),
            'optimization_plan': [],
            'estimated_improvement': 0.0
        }
        
        # Analyze each optimization strategy
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                strategy_result = strategy_func()
                if strategy_result:
                    analysis_result['optimization_plan'].append({
                        'strategy': strategy_name,
                        'recommendations': strategy_result['recommendations'],
                        'estimated_improvement': strategy_result['estimated_improvement']
                    })
            except Exception as e:
                self.logger.error(f"Error analyzing {strategy_name}: {e}")
        
        # Calculate total estimated improvement
        total_improvement = sum(
            plan['estimated_improvement'] 
            for plan in analysis_result['optimization_plan']
        )
        analysis_result['estimated_improvement'] = min(total_improvement, 50.0)  # Cap at 50%
        
        return analysis_result
    
    def _optimize_memory_usage(self) -> Optional[Dict[str, Any]]:
        """Analyze and recommend memory usage optimizations"""
        recent_metrics = list(self.monitor.metrics_history)[-20:]
        if not recent_metrics:
            return None
        
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        recommendations = []
        estimated_improvement = 0.0
        
        if avg_memory > 100:  # More than 100MB average
            recommendations.append("Consider using chunked processing for large datasets")
            recommendations.append("Implement data type optimization (e.g., float32 instead of float64)")
            estimated_improvement += 20.0
        
        if any(m.memory_usage_mb > 500 for m in recent_metrics):
            recommendations.append("Implement streaming data processing")
            recommendations.append("Use memory-mapped files for large datasets")
            estimated_improvement += 30.0
        
        return {
            'recommendations': recommendations,
            'estimated_improvement': estimated_improvement
        }
    
    def _optimize_processing(self) -> Optional[Dict[str, Any]]:
        """Analyze and recommend processing optimizations"""
        recent_metrics = list(self.monitor.metrics_history)[-20:]
        if not recent_metrics:
            return None
        
        avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
        
        recommendations = []
        estimated_improvement = 0.0
        
        if avg_duration > 5.0:  # More than 5 seconds average
            recommendations.append("Consider parallel processing for independent operations")
            recommendations.append("Optimize data validation algorithms")
            estimated_improvement += 25.0
        
        if any(m.duration > 30 for m in recent_metrics):
            recommendations.append("Implement asynchronous processing")
            recommendations.append("Use vectorized operations instead of loops")
            estimated_improvement += 40.0
        
        return {
            'recommendations': recommendations,
            'estimated_improvement': estimated_improvement
        }
    
    def _optimize_concurrency(self) -> Optional[Dict[str, Any]]:
        """Analyze and recommend concurrency optimizations"""
        latest_system = self.monitor._get_latest_system_metrics()
        if not latest_system:
            return None
        
        recommendations = []
        estimated_improvement = 0.0
        
        if latest_system['cpu_percent'] < 50:  # Low CPU usage
            recommendations.append("Increase concurrency for I/O-bound operations")
            recommendations.append("Use thread pools for parallel data processing")
            estimated_improvement += 15.0
        
        if latest_system['active_threads'] < 4:  # Few active threads
            recommendations.append("Implement concurrent exchange data fetching")
            recommendations.append("Use asyncio for non-blocking operations")
            estimated_improvement += 20.0
        
        return {
            'recommendations': recommendations,
            'estimated_improvement': estimated_improvement
        }
    
    def _optimize_caching(self) -> Optional[Dict[str, Any]]:
        """Analyze and recommend caching optimizations"""
        operation_counts = self.monitor.operation_counts
        
        recommendations = []
        estimated_improvement = 0.0
        
        # Check for repeated operations
        repeated_operations = {op: count for op, count in operation_counts.items() if count > 5}
        if repeated_operations:
            recommendations.append("Implement caching for frequently repeated operations")
            recommendations.append("Cache standardized data to avoid reprocessing")
            estimated_improvement += 30.0
        
        return {
            'recommendations': recommendations,
            'estimated_improvement': estimated_improvement
        }


# Global instances
performance_monitor = PerformanceMonitor()
performance_optimizer = PerformanceOptimizer(performance_monitor)


# Convenience functions
def measure_operation(operation_name: str, exchange: str = None, **kwargs):
    """Convenience function for measuring operation performance"""
    return performance_monitor.measure_operation(operation_name, exchange, **kwargs)


def get_performance_summary() -> Dict[str, Any]:
    """Get current performance summary"""
    return performance_monitor.get_performance_summary()


def get_optimization_recommendations() -> List[str]:
    """Get performance optimization recommendations"""
    return performance_monitor.get_optimization_recommendations()


def analyze_performance() -> Dict[str, Any]:
    """Analyze performance and get optimization recommendations"""
    return performance_optimizer.analyze_and_optimize()