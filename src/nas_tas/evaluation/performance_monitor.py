"""
Performance Monitor for NAS/TAS Systems

This module provides system performance monitoring and resource usage tracking
for both NAS and TAS implementations, consolidating monitoring logic.
"""

import psutil
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
from collections import deque
import asyncio

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

@dataclass
class SystemMetrics:
    """System performance metrics container."""

    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_freq_current: float = 0.0
    cpu_freq_max: float = 0.0

    # Memory metrics
    memory_total: float = 0.0
    memory_available: float = 0.0
    memory_percent: float = 0.0
    memory_used: float = 0.0

    # Disk metrics
    disk_total: float = 0.0
    disk_used: float = 0.0
    disk_free: float = 0.0
    disk_percent: float = 0.0

    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0

    # Process metrics
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    process_num_threads: int = 0
    process_num_fds: int = 0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            else:
                result[field_name] = field_value
        return result

@dataclass
class ResourceUsage:
    """Resource usage tracking container."""

    # Peak usage
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0

    # Average usage
    avg_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0

    # Current usage
    current_memory_mb: float = 0.0
    current_cpu_percent: float = 0.0

    # Usage trends
    memory_trend: str = "stable"  # increasing, decreasing, stable
    cpu_trend: str = "stable"

    # Resource warnings
    memory_warnings: List[str] = field(default_factory=list)
    cpu_warnings: List[str] = field(default_factory=list)

    # Efficiency metrics
    memory_efficiency: float = 0.0  # 0-1, higher is better
    cpu_efficiency: float = 0.0     # 0-1, higher is better

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, list):
                result[field_name] = field_value
            else:
                result[field_name] = field_value
        return result

@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    # Report metadata
    report_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    # System metrics summary
    system_metrics_summary: Dict[str, float] = field(default_factory=dict)

    # Resource usage summary
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)

    # Performance statistics
    performance_stats: Dict[str, float] = field(default_factory=dict)

    # Bottlenecks identified
    bottlenecks: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Detailed metrics (if requested)
    detailed_metrics: List[SystemMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'report_id': self.report_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'system_metrics_summary': self.system_metrics_summary,
            'resource_usage': self.resource_usage.to_dict(),
            'performance_stats': self.performance_stats,
            'bottlenecks': self.bottlenecks,
            'recommendations': self.recommendations,
            'detailed_metrics_count': len(self.detailed_metrics)
        }

        return result

class PerformanceMonitor:
    """
    Performance monitor for NAS/TAS systems.

    This class consolidates performance monitoring logic that was previously
    scattered across NAS and TAS implementations, providing unified system
    monitoring and resource tracking.
    """

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        max_history_size: int = 1000,
        enable_detailed_logging: bool = True
    ):
        """
        Initialize performance monitor.

        Args:
            monitoring_interval: Interval between metric collections (seconds)
            max_history_size: Maximum number of metrics to keep in history
            enable_detailed_logging: Whether to enable detailed logging
        """
        self.monitoring_interval = monitoring_interval
        self.max_history_size = max_history_size
        self.enable_detailed_logging = enable_detailed_logging

        self.logger = logging.getLogger(self.__class__.__name__)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.monitoring_start_time = None

        # Metrics storage
        self.metrics_history = deque(maxlen=max_history_size)
        self.current_metrics = SystemMetrics()

        # Resource tracking
        self.peak_memory = 0.0
        self.peak_cpu = 0.0
        self.total_memory_samples = 0
        self.total_cpu_samples = 0
        self.memory_sum = 0.0
        self.cpu_sum = 0.0

        # Process tracking
        self.process = psutil.Process()

        # Callbacks
        self.metrics_callbacks: List[Callable[[SystemMetrics], None]] = []
        self.warning_callbacks: List[Callable[[str], None]] = []

        # Thresholds
        self.memory_warning_threshold = 80.0  # 80% memory usage
        self.cpu_warning_threshold = 90.0     # 90% CPU usage
        self.disk_warning_threshold = 85.0    # 85% disk usage

        tprint_info(f"Performance monitor initialized (interval: {monitoring_interval}s)")

    def start_monitoring(self) -> bool:
        """Start performance monitoring."""
        if self.is_monitoring:
            tprint_warning("Monitoring is already running")
            return False

        try:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

            tprint_success("Performance monitoring started")
            return True

        except Exception as e:
            tprint_error(f"Failed to start monitoring: {e}")
            self.is_monitoring = False
            return False

    def stop_monitoring(self) -> PerformanceReport:
        """Stop performance monitoring and return report."""
        if not self.is_monitoring:
            tprint_warning("Monitoring is not running")
            return PerformanceReport()

        try:
            self.is_monitoring = False

            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)

            # Generate final report
            report = self._generate_performance_report()

            tprint_success("Performance monitoring stopped")
            return report

        except Exception as e:
            tprint_error(f"Error stopping monitoring: {e}")
            return PerformanceReport()

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._collect_metrics()

    def get_metrics_history(self) -> List[SystemMetrics]:
        """Get metrics history."""
        return list(self.metrics_history)

    def add_metrics_callback(self, callback: Callable[[SystemMetrics], None]):
        """Add callback for metrics updates."""
        self.metrics_callbacks.append(callback)

    def add_warning_callback(self, callback: Callable[[str], None]):
        """Add callback for warnings."""
        self.warning_callbacks.append(callback)

    def set_memory_threshold(self, threshold: float):
        """Set memory warning threshold (percentage)."""
        self.memory_warning_threshold = max(0.0, min(100.0, threshold))

    def set_cpu_threshold(self, threshold: float):
        """Set CPU warning threshold (percentage)."""
        self.cpu_warning_threshold = max(0.0, min(100.0, threshold))

    def _monitoring_loop(self):
        """Main monitoring loop."""
        tprint_info("Performance monitoring loop started")

        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.current_metrics = metrics

                # Store in history
                self.metrics_history.append(metrics)

                # Update resource tracking
                self._update_resource_tracking(metrics)

                # Check for warnings
                self._check_warnings(metrics)

                # Call callbacks
                self._notify_callbacks(metrics)

                # Log detailed metrics if enabled
                if self.enable_detailed_logging:
                    self._log_detailed_metrics(metrics)

                # Sleep until next collection
                time.sleep(self.monitoring_interval)

            except Exception as e:
                tprint_error(f"Error in monitoring loop: {e}")
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.monitoring_interval)

        tprint_info("Performance monitoring loop stopped")

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            metrics = SystemMetrics()

            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=None)
            metrics.cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.cpu_freq_current = cpu_freq.current or 0.0
                metrics.cpu_freq_max = cpu_freq.max or 0.0

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.memory_total = memory.total / (1024**3)  # GB
            metrics.memory_available = memory.available / (1024**3)  # GB
            metrics.memory_percent = memory.percent
            metrics.memory_used = memory.used / (1024**3)  # GB

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_total = disk.total / (1024**3)  # GB
            metrics.disk_used = disk.used / (1024**3)  # GB
            metrics.disk_free = disk.free / (1024**3)  # GB
            metrics.disk_percent = (disk.used / disk.total) * 100

            # Network metrics
            network = psutil.net_io_counters()
            if network:
                metrics.network_bytes_sent = network.bytes_sent
                metrics.network_bytes_recv = network.bytes_recv
                metrics.network_packets_sent = network.packets_sent
                metrics.network_packets_recv = network.packets_recv

            # Process metrics
            try:
                process_memory = self.process.memory_info()
                metrics.process_memory_mb = process_memory.rss / (1024**2)  # MB
                metrics.process_cpu_percent = self.process.cpu_percent()
                metrics.process_num_threads = self.process.num_threads()

                # File descriptors (Unix only)
                try:
                    metrics.process_num_fds = self.process.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    metrics.process_num_fds = 0

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process may have ended or access denied
                pass

            metrics.timestamp = datetime.now()
            return metrics

        except Exception as e:
            tprint_error(f"Error collecting metrics: {e}")
            return SystemMetrics()

    def _update_resource_tracking(self, metrics: SystemMetrics):
        """Update resource usage tracking."""
        # Update peaks
        if metrics.memory_percent > self.peak_memory:
            self.peak_memory = metrics.memory_percent

        if metrics.cpu_percent > self.peak_cpu:
            self.peak_cpu = metrics.cpu_percent

        # Update averages
        self.total_memory_samples += 1
        self.total_cpu_samples += 1
        self.memory_sum += metrics.memory_percent
        self.cpu_sum += metrics.cpu_percent

    def _check_warnings(self, metrics: SystemMetrics):
        """Check for resource warnings."""
        warnings = []

        # Memory warning
        if metrics.memory_percent > self.memory_warning_threshold:
            warning = f"High memory usage: {metrics.memory_percent:.1f}%"
            warnings.append(warning)
            tprint_warning(warning)

        # CPU warning
        if metrics.cpu_percent > self.cpu_warning_threshold:
            warning = f"High CPU usage: {metrics.cpu_percent:.1f}%"
            warnings.append(warning)
            tprint_warning(warning)

        # Disk warning
        if metrics.disk_percent > self.disk_warning_threshold:
            warning = f"High disk usage: {metrics.disk_percent:.1f}%"
            warnings.append(warning)
            tprint_warning(warning)

        # Process memory warning
        if metrics.process_memory_mb > 1000:  # 1GB
            warning = f"High process memory usage: {metrics.process_memory_mb:.1f} MB"
            warnings.append(warning)
            tprint_warning(warning)

        # Notify warning callbacks
        for warning in warnings:
            for callback in self.warning_callbacks:
                try:
                    callback(warning)
                except Exception as e:
                    tprint_error(f"Error in warning callback: {e}")

    def _notify_callbacks(self, metrics: SystemMetrics):
        """Notify metrics callbacks."""
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                tprint_error(f"Error in metrics callback: {e}")

    def _log_detailed_metrics(self, metrics: SystemMetrics):
        """Log detailed metrics if enabled."""
        if len(self.metrics_history) % 10 == 0:  # Log every 10th sample
            tprint_performance(
                f"Metrics: CPU={metrics.cpu_percent:.1f}%, "
                f"Memory={metrics.memory_percent:.1f}%, "
                f"Process={metrics.process_memory_mb:.1f}MB"
            )

    def _generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        tprint_info("Generating performance report")

        try:
            report = PerformanceReport()
            report.report_id = f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if self.monitoring_start_time:
                report.start_time = self.monitoring_start_time
                report.end_time = datetime.now()
                report.duration_seconds = (report.end_time - report.start_time).total_seconds()

            # System metrics summary
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                report.system_metrics_summary = {
                    'cpu_percent': latest_metrics.cpu_percent,
                    'memory_percent': latest_metrics.memory_percent,
                    'disk_percent': latest_metrics.disk_percent,
                    'process_memory_mb': latest_metrics.process_memory_mb
                }

            # Resource usage summary
            resource_usage = ResourceUsage()
            resource_usage.peak_memory_mb = self.peak_memory
            resource_usage.peak_cpu_percent = self.peak_cpu

            if self.total_memory_samples > 0:
                resource_usage.avg_memory_mb = self.memory_sum / self.total_memory_samples

            if self.total_cpu_samples > 0:
                resource_usage.avg_cpu_percent = self.cpu_sum / self.total_cpu_samples

            if self.current_metrics:
                resource_usage.current_memory_mb = self.current_metrics.memory_percent
                resource_usage.current_cpu_percent = self.current_metrics.cpu_percent

            # Calculate trends
            resource_usage.memory_trend = self._calculate_trend('memory_percent')
            resource_usage.cpu_trend = self._calculate_trend('cpu_percent')

            # Calculate efficiency
            resource_usage.memory_efficiency = max(0, 1 - (resource_usage.avg_memory_mb / 100))
            resource_usage.cpu_efficiency = max(0, 1 - (resource_usage.avg_cpu_percent / 100))

            report.resource_usage = resource_usage

            # Performance statistics
            if self.metrics_history:
                cpu_values = [m.cpu_percent for m in self.metrics_history]
                memory_values = [m.memory_percent for m in self.metrics_history]

                report.performance_stats = {
                    'cpu_mean': float(np.mean(cpu_values)),
                    'cpu_std': float(np.std(cpu_values)),
                    'cpu_min': float(np.min(cpu_values)),
                    'cpu_max': float(np.max(cpu_values)),
                    'memory_mean': float(np.mean(memory_values)),
                    'memory_std': float(np.std(memory_values)),
                    'memory_min': float(np.min(memory_values)),
                    'memory_max': float(np.max(memory_values))
                }

            # Identify bottlenecks
            report.bottlenecks = self._identify_bottlenecks()

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            # Store detailed metrics if requested
            report.detailed_metrics = list(self.metrics_history)

            tprint_success(f"Performance report generated: {report.report_id}")
            return report

        except Exception as e:
            tprint_error(f"Error generating performance report: {e}")
            return PerformanceReport()

    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric."""
        if len(self.metrics_history) < 10:
            return "stable"

        try:
            values = [getattr(m, metric_name) for m in list(self.metrics_history)[-10:]]
            if len(values) < 2:
                return "stable"

            # Simple linear trend calculation
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"

        except Exception:
            return "stable"

    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if self.peak_memory > 90:
            bottlenecks.append("High memory usage detected")

        if self.peak_cpu > 95:
            bottlenecks.append("High CPU usage detected")

        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            if latest_metrics.disk_percent > 90:
                bottlenecks.append("High disk usage detected")

            if latest_metrics.process_memory_mb > 2000:  # 2GB
                bottlenecks.append("High process memory usage")

        return bottlenecks

    def _generate_recommendations(self, report: PerformanceReport) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        # Memory recommendations
        if report.resource_usage.avg_memory_mb > 80:
            recommendations.append("Consider optimizing memory usage or increasing system RAM")

        if report.resource_usage.memory_trend == "increasing":
            recommendations.append("Memory usage is increasing - investigate for memory leaks")

        # CPU recommendations
        if report.resource_usage.avg_cpu_percent > 80:
            recommendations.append("Consider optimizing CPU usage or using more efficient algorithms")

        if report.resource_usage.cpu_trend == "increasing":
            recommendations.append("CPU usage is increasing - consider parallel processing optimizations")

        # Process recommendations
        if self.current_metrics and self.current_metrics.process_memory_mb > 1000:
            recommendations.append("Consider implementing memory-efficient data structures")

        # General recommendations
        if report.resource_usage.memory_efficiency < 0.5:
            recommendations.append("Memory efficiency is low - consider memory optimization")

        if report.resource_usage.cpu_efficiency < 0.5:
            recommendations.append("CPU efficiency is low - consider algorithm optimization")

        return recommendations

    def save_report(self, report: PerformanceReport, filepath: Union[str, Path]) -> bool:
        """Save performance report to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            tprint_success(f"Performance report saved to {filepath}")
            return True

        except Exception as e:
            tprint_error(f"Failed to save performance report: {e}")
            return False

    def export_metrics_csv(self, filepath: Union[str, Path]) -> bool:
        """Export metrics history to CSV."""
        try:
            if not self.metrics_history:
                tprint_warning("No metrics history to export")
                return False

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert metrics to DataFrame
            metrics_data = []
            for metrics in self.metrics_history:
                metrics_dict = metrics.to_dict()
                metrics_data.append(metrics_dict)

            df = pd.DataFrame(metrics_data)
            df.to_csv(filepath, index=False)

            tprint_success(f"Metrics exported to {filepath}")
            return True

        except Exception as e:
            tprint_error(f"Failed to export metrics: {e}")
            return False

# Convenience functions
def create_performance_monitor(
    interval: float = 1.0,
    max_history: int = 1000,
    detailed_logging: bool = True
) -> PerformanceMonitor:
    """Create and configure a performance monitor."""
    return PerformanceMonitor(
        monitoring_interval=interval,
        max_history_size=max_history,
        enable_detailed_logging=detailed_logging
    )

def monitor_performance_during_execution(
    func: Callable,
    *args,
    monitoring_interval: float = 1.0,
    **kwargs
) -> Tuple[Any, PerformanceReport]:
    """Monitor performance during function execution."""
    monitor = create_performance_monitor(monitoring_interval)

    try:
        monitor.start_monitoring()
        result = func(*args, **kwargs)
        return result, monitor.stop_monitoring()
    except Exception as e:
        monitor.stop_monitoring()
        raise e
