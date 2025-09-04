#!/usr/bin/env python3
"""
Pipeline Monitoring and Logging System

This module provides comprehensive monitoring, logging, and observability
for the Ares trading pipeline execution.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import threading
from collections import defaultdict, deque
import psutil
import sys

from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.logging import logs_execution
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory
)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: LogLevel
    component: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "context": self.context,
            "exception": self.exception,
            "stack_trace": self.stack_trace
        }


@dataclass
class MetricEntry:
    """Structured metric entry."""
    timestamp: str
    metric_name: str
    metric_type: MetricType
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline execution."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    thread_count: int
    file_descriptors: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "network_io_sent_mb": self.network_io_sent_mb,
            "network_io_recv_mb": self.network_io_recv_mb,
            "thread_count": self.thread_count,
            "file_descriptors": self.file_descriptors
        }


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self):
        self.logger = logging.getLogger("system_monitor")
        self.process = psutil.Process()
        self._baseline_metrics = None
        self._start_time = None
    
    @handles_errors(Exception, fallback=None)
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        self._start_time = time.time()
        self._baseline_metrics = self._collect_system_metrics()
        self.logger.info("System monitoring started")
    
    @handles_errors(Exception, fallback=PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0))
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        
        try:
            current_metrics = self._collect_system_metrics()
            
            if self._baseline_metrics is None:
                self._baseline_metrics = current_metrics
            
            # Calculate deltas
            execution_time = time.time() - (self._start_time or time.time())
            memory_usage = current_metrics["memory_mb"] - self._baseline_metrics["memory_mb"]
            cpu_usage = current_metrics["cpu_percent"]
            disk_read = current_metrics["disk_read_mb"] - self._baseline_metrics["disk_read_mb"]
            disk_write = current_metrics["disk_write_mb"] - self._baseline_metrics["disk_write_mb"]
            net_sent = current_metrics["net_sent_mb"] - self._baseline_metrics["net_sent_mb"]
            net_recv = current_metrics["net_recv_mb"] - self._baseline_metrics["net_recv_mb"]
            
            return PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                disk_io_read_mb=disk_read,
                disk_io_write_mb=disk_write,
                network_io_sent_mb=net_sent,
                network_io_recv_mb=net_recv,
                thread_count=current_metrics["thread_count"],
                file_descriptors=current_metrics["file_descriptors"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Disk I/O
            io_counters = self.process.io_counters()
            disk_read_mb = io_counters.read_bytes / 1024 / 1024
            disk_write_mb = io_counters.write_bytes / 1024 / 1024
            
            # Network I/O
            net_counters = psutil.net_io_counters()
            net_sent_mb = net_counters.bytes_sent / 1024 / 1024
            net_recv_mb = net_counters.bytes_recv / 1024 / 1024
            
            # Thread and file descriptor count
            thread_count = self.process.num_threads()
            try:
                file_descriptors = self.process.num_fds()
            except AttributeError:
                # Windows doesn't have num_fds
                file_descriptors = 0
            
            return {
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "disk_read_mb": disk_read_mb,
                "disk_write_mb": disk_write_mb,
                "net_sent_mb": net_sent_mb,
                "net_recv_mb": net_recv_mb,
                "thread_count": thread_count,
                "file_descriptors": file_descriptors
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {
                "memory_mb": 0,
                "cpu_percent": 0,
                "disk_read_mb": 0,
                "disk_write_mb": 0,
                "net_sent_mb": 0,
                "net_recv_mb": 0,
                "thread_count": 0,
                "file_descriptors": 0
            }


class PipelineLogger:
    """Enhanced logger for pipeline operations."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.logger = logging.getLogger("pipeline_logger")
        self.log_entries: deque = deque(maxlen=10000)  # Keep last 10k entries
        self.component_loggers: Dict[str, logging.Logger] = {}
        
        # Ensure log directory exists
        ensure_directory(self.log_dir)
        
        # Setup main logger
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup the main pipeline logger."""
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.log_dir / f"pipeline_{format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configure logger
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    @handles_errors(Exception, fallback=None)
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """Get a logger for a specific component."""
        
        if component_name not in self.component_loggers:
            logger = logging.getLogger(f"pipeline.{component_name}")
            logger.setLevel(logging.DEBUG)
            
            # Add custom handler that also logs to our structured log
            handler = PipelineLogHandler(self)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.propagate = False
            
            self.component_loggers[component_name] = logger
        
        return self.component_loggers[component_name]
    
    @handles_errors(Exception, fallback=None)
    def log_structured(
        self,
        level: LogLevel,
        component: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> None:
        """Log a structured entry."""
        
        try:
            # Create log entry
            entry = LogEntry(
                timestamp=format_datetime(get_current_datetime()),
                level=level,
                component=component,
                message=message,
                context=context or {},
                exception=str(exception) if exception else None,
                stack_trace=self._get_stack_trace(exception) if exception else None
            )
            
            # Add to in-memory log
            self.log_entries.append(entry)
            
            # Log to standard logger
            log_method = getattr(self.logger, level.value.lower())
            log_message = f"[{component}] {message}"
            if context:
                log_message += f" | Context: {context}"
            
            if exception:
                log_method(log_message, exc_info=True)
            else:
                log_method(log_message)
            
        except Exception as e:
            self.logger.error(f"Failed to log structured entry: {e}")
    
    def _get_stack_trace(self, exception: Optional[Exception]) -> Optional[str]:
        """Get stack trace for an exception."""
        if exception:
            import traceback
            return traceback.format_exc()
        return None
    
    @handles_errors(Exception, fallback=[])
    def get_recent_logs(
        self,
        component: Optional[str] = None,
        level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Get recent log entries with optional filtering."""
        
        try:
            logs = list(self.log_entries)
            
            # Filter by component
            if component:
                logs = [log for log in logs if log.component == component]
            
            # Filter by level
            if level:
                logs = [log for log in logs if log.level == level]
            
            # Limit results
            return logs[-limit:] if limit > 0 else logs
            
        except Exception as e:
            self.logger.error(f"Failed to get recent logs: {e}")
            return []
    
    @handles_errors(Exception, fallback=None)
    def save_logs_to_file(self, file_path: str) -> None:
        """Save logs to a JSON file."""
        
        try:
            log_data = {
                "timestamp": format_datetime(get_current_datetime()),
                "total_entries": len(self.log_entries),
                "entries": [entry.to_dict() for entry in self.log_entries]
            }
            
            safe_json_dump(log_data, file_path, indent=2)
            self.logger.info(f"Saved {len(self.log_entries)} log entries to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save logs to file: {e}")


class PipelineLogHandler(logging.Handler):
    """Custom log handler that integrates with PipelineLogger."""
    
    def __init__(self, pipeline_logger: PipelineLogger):
        super().__init__()
        self.pipeline_logger = pipeline_logger
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        try:
            # Extract component name from logger name
            component = record.name.split('.')[-1] if '.' in record.name else record.name
            
            # Convert log level
            level_mapping = {
                logging.DEBUG: LogLevel.DEBUG,
                logging.INFO: LogLevel.INFO,
                logging.WARNING: LogLevel.WARNING,
                logging.ERROR: LogLevel.ERROR,
                logging.CRITICAL: LogLevel.CRITICAL
            }
            level = level_mapping.get(record.levelno, LogLevel.INFO)
            
            # Create context
            context = {
                "filename": record.filename,
                "lineno": record.lineno,
                "funcName": record.funcName
            }
            
            # Log structured entry
            self.pipeline_logger.log_structured(
                level=level,
                component=component,
                message=record.getMessage(),
                context=context,
                exception=record.exc_info[1] if record.exc_info else None
            )
            
        except Exception:
            # Don't let logging errors break the application
            pass


class MetricsCollector:
    """Collects and manages pipeline metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("metrics_collector")
        self.metrics: deque = deque(maxlen=50000)  # Keep last 50k metrics
        self.metric_counters: Dict[str, int] = defaultdict(int)
        self.metric_gauges: Dict[str, float] = {}
        self.metric_histograms: Dict[str, List[float]] = defaultdict(list)
        self.metric_timers: Dict[str, List[float]] = defaultdict(list)
    
    @handles_errors(Exception, fallback=None)
    def record_metric(
        self,
        metric_name: str,
        metric_type: MetricType,
        value: Union[int, float],
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ) -> None:
        """Record a metric."""
        
        try:
            # Create metric entry
            entry = MetricEntry(
                timestamp=format_datetime(get_current_datetime()),
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                tags=tags or {},
                unit=unit
            )
            
            # Add to metrics list
            self.metrics.append(entry)
            
            # Update metric-specific storage
            if metric_type == MetricType.COUNTER:
                self.metric_counters[metric_name] += int(value)
            elif metric_type == MetricType.GAUGE:
                self.metric_gauges[metric_name] = float(value)
            elif metric_type == MetricType.HISTOGRAM:
                self.metric_histograms[metric_name].append(float(value))
            elif metric_type == MetricType.TIMER:
                self.metric_timers[metric_name].append(float(value))
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
    
    @handles_errors(Exception, fallback=0)
    def get_counter_value(self, metric_name: str) -> int:
        """Get current counter value."""
        return self.metric_counters.get(metric_name, 0)
    
    @handles_errors(Exception, fallback=0.0)
    def get_gauge_value(self, metric_name: str) -> float:
        """Get current gauge value."""
        return self.metric_gauges.get(metric_name, 0.0)
    
    @handles_errors(Exception, fallback={})
    def get_histogram_stats(self, metric_name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        
        values = self.metric_histograms.get(metric_name, [])
        if not values:
            return {}
        
        import statistics
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    @handles_errors(Exception, fallback={})
    def get_timer_stats(self, metric_name: str) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(metric_name)
    
    @handles_errors(Exception, fallback=[])
    def get_metrics_by_name(self, metric_name: str, limit: int = 100) -> List[MetricEntry]:
        """Get metrics by name."""
        
        try:
            metrics = [m for m in self.metrics if m.metric_name == metric_name]
            return metrics[-limit:] if limit > 0 else metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics by name: {e}")
            return []
    
    @handles_errors(Exception, fallback=None)
    def save_metrics_to_file(self, file_path: str) -> None:
        """Save metrics to a JSON file."""
        
        try:
            metrics_data = {
                "timestamp": format_datetime(get_current_datetime()),
                "total_metrics": len(self.metrics),
                "counters": dict(self.metric_counters),
                "gauges": dict(self.metric_gauges),
                "histogram_stats": {
                    name: self.get_histogram_stats(name)
                    for name in self.metric_histograms.keys()
                },
                "timer_stats": {
                    name: self.get_timer_stats(name)
                    for name in self.metric_timers.keys()
                },
                "recent_metrics": [metric.to_dict() for metric in list(self.metrics)[-1000:]]
            }
            
            safe_json_dump(metrics_data, file_path, indent=2)
            self.logger.info(f"Saved metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics to file: {e}")


class PipelineMonitor:
    """Main pipeline monitoring system."""
    
    def __init__(self, monitor_dir: str = "monitoring"):
        self.monitor_dir = Path(monitor_dir)
        self.logger = logging.getLogger("pipeline_monitor")
        self.pipeline_logger = PipelineLogger(str(self.monitor_dir / "logs"))
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Ensure monitoring directory exists
        ensure_directory(self.monitor_dir)
    
    @handles_errors(Exception, fallback=False)
    @logs_execution("pipeline_monitoring")
    def start_monitoring(self) -> bool:
        """Start comprehensive pipeline monitoring."""
        
        try:
            if self.monitoring_active:
                self.logger.warning("Monitoring is already active")
                return True
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            # Start background monitoring thread
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            # Log monitoring start
            self.pipeline_logger.log_structured(
                LogLevel.INFO,
                "pipeline_monitor",
                "Pipeline monitoring started",
                {"monitor_dir": str(self.monitor_dir)}
            )
            
            self.logger.info("Pipeline monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False
    
    @handles_errors(Exception, fallback=None)
    def stop_monitoring(self) -> None:
        """Stop pipeline monitoring."""
        
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Log monitoring stop
            self.pipeline_logger.log_structured(
                LogLevel.INFO,
                "pipeline_monitor",
                "Pipeline monitoring stopped"
            )
            
            self.logger.info("Pipeline monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                performance_metrics = self.system_monitor.get_performance_metrics()
                
                # Record performance metrics
                self.metrics_collector.record_metric(
                    "memory_usage_mb",
                    MetricType.GAUGE,
                    performance_metrics.memory_usage_mb,
                    unit="MB"
                )
                
                self.metrics_collector.record_metric(
                    "cpu_usage_percent",
                    MetricType.GAUGE,
                    performance_metrics.cpu_usage_percent,
                    unit="%"
                )
                
                self.metrics_collector.record_metric(
                    "execution_time",
                    MetricType.TIMER,
                    performance_metrics.execution_time,
                    unit="seconds"
                )
                
                # Sleep for monitoring interval
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    @handles_errors(Exception, fallback=None)
    def record_pipeline_event(
        self,
        event_type: str,
        pipeline_id: str,
        step_name: Optional[str] = None,
        message: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a pipeline event."""
        
        try:
            # Log the event
            self.pipeline_logger.log_structured(
                LogLevel.INFO,
                f"pipeline.{pipeline_id}",
                f"{event_type}: {message}",
                context={
                    "event_type": event_type,
                    "step_name": step_name,
                    **(context or {})
                }
            )
            
            # Record metrics
            self.metrics_collector.record_metric(
                f"pipeline_events_{event_type}",
                MetricType.COUNTER,
                1,
                tags={"pipeline_id": pipeline_id, "step_name": step_name or "unknown"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record pipeline event: {e}")
    
    @handles_errors(Exception, fallback=None)
    def record_step_performance(
        self,
        pipeline_id: str,
        step_name: str,
        execution_time: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record step performance metrics."""
        
        try:
            # Record execution time
            self.metrics_collector.record_metric(
                "step_execution_time",
                MetricType.TIMER,
                execution_time,
                tags={"pipeline_id": pipeline_id, "step_name": step_name}
            )
            
            # Record success/failure
            self.metrics_collector.record_metric(
                "step_success" if success else "step_failure",
                MetricType.COUNTER,
                1,
                tags={"pipeline_id": pipeline_id, "step_name": step_name}
            )
            
            # Log performance
            self.pipeline_logger.log_structured(
                LogLevel.INFO,
                f"pipeline.{pipeline_id}",
                f"Step {step_name} completed",
                context={
                    "execution_time": execution_time,
                    "success": success,
                    **(context or {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record step performance: {e}")
    
    @handles_errors(Exception, fallback={})
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        
        try:
            # Get recent logs
            recent_logs = self.pipeline_logger.get_recent_logs(limit=100)
            
            # Get performance metrics
            performance_metrics = self.system_monitor.get_performance_metrics()
            
            # Get metric summaries
            counters = dict(self.metrics_collector.metric_counters)
            gauges = dict(self.metrics_collector.metric_gauges)
            
            return {
                "monitoring_active": self.monitoring_active,
                "total_log_entries": len(self.pipeline_logger.log_entries),
                "total_metrics": len(self.metrics_collector.metrics),
                "performance_metrics": performance_metrics.to_dict(),
                "counters": counters,
                "gauges": gauges,
                "recent_events": [
                    {
                        "timestamp": log.timestamp,
                        "component": log.component,
                        "level": log.level.value,
                        "message": log.message
                    }
                    for log in recent_logs[-20:]  # Last 20 events
                ],
                "timestamp": format_datetime(get_current_datetime())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring summary: {e}")
            return {}
    
    @handles_errors(Exception, fallback=None)
    def save_monitoring_report(self, file_path: str) -> None:
        """Save comprehensive monitoring report."""
        
        try:
            # Get summary
            summary = self.get_monitoring_summary()
            
            # Save logs
            logs_file = Path(file_path).parent / f"{Path(file_path).stem}_logs.json"
            self.pipeline_logger.save_logs_to_file(str(logs_file))
            
            # Save metrics
            metrics_file = Path(file_path).parent / f"{Path(file_path).stem}_metrics.json"
            self.metrics_collector.save_metrics_to_file(str(metrics_file))
            
            # Save summary
            safe_json_dump(summary, file_path, indent=2)
            
            self.logger.info(f"Monitoring report saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save monitoring report: {e}")


# Global monitoring instance
pipeline_monitor = PipelineMonitor()