#!/usr/bin/env python3
"""
Comprehensive Pipeline Monitoring System

This module provides monitoring capabilities for pipeline execution,
performance tracking, and real-time status reporting.
"""

import asyncio
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import psutil
import pandas as pd

from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_json_dump,
    safe_json_load
)


class MonitorStatus(Enum):
    """Monitor status states."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MetricType(Enum):
    """Types of metrics to monitor."""
    EXECUTION_TIME = "EXECUTION_TIME"
    MEMORY_USAGE = "MEMORY_USAGE"
    CPU_USAGE = "CPU_USAGE"
    DISK_USAGE = "DISK_USAGE"
    DATA_SIZE = "DATA_SIZE"
    ERROR_COUNT = "ERROR_COUNT"
    SUCCESS_RATE = "SUCCESS_RATE"
    THROUGHPUT = "THROUGHPUT"


@dataclass
class MetricData:
    """Metric data point."""
    timestamp: str
    metric_type: MetricType
    value: float
    unit: str
    context: Dict[str, Any]


@dataclass
class StepMetrics:
    """Metrics for a pipeline step."""
    step_name: str
    start_time: str
    end_time: Optional[str]
    duration: float
    status: MonitorStatus
    memory_peak: float
    cpu_peak: float
    data_processed: int
    errors: int
    warnings: int
    custom_metrics: Dict[str, Any]


@dataclass
class PipelineMetrics:
    """Overall pipeline metrics."""
    pipeline_id: str
    start_time: str
    end_time: Optional[str]
    total_duration: float
    status: MonitorStatus
    steps_completed: int
    steps_total: int
    total_memory_peak: float
    total_cpu_peak: float
    total_data_processed: int
    total_errors: int
    total_warnings: int
    step_metrics: List[StepMetrics]
    custom_metrics: Dict[str, Any]


class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: List[MetricData] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start monitoring system performance."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system performance."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.exception(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.append(MetricData(
                timestamp=format_datetime(get_current_datetime()),
                metric_type=MetricType.MEMORY_USAGE,
                value=memory.percent,
                unit="percent",
                context={"available": memory.available, "total": memory.total}
            ))
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics.append(MetricData(
                timestamp=format_datetime(get_current_datetime()),
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit="percent",
                context={}
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.append(MetricData(
                timestamp=format_datetime(get_current_datetime()),
                metric_type=MetricType.DISK_USAGE,
                value=(disk.used / disk.total) * 100,
                unit="percent",
                context={"used": disk.used, "total": disk.total, "free": disk.free}
            ))
            
        except Exception as e:
            self.logger.exception(f"Error collecting system metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            return {
                "memory": {
                    "percent": memory.percent,
                    "available": memory.available,
                    "total": memory.total
                },
                "cpu": {
                    "percent": cpu_percent
                },
                "disk": {
                    "percent": (disk.used / disk.total) * 100,
                    "used": disk.used,
                    "total": disk.total,
                    "free": disk.free
                }
            }
        except Exception as e:
            self.logger.exception(f"Error getting current metrics: {e}")
            return {}


class StepMonitor:
    """Monitor individual pipeline steps."""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.logger = logging.getLogger(f"{__name__}.{step_name}")
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = MonitorStatus.IDLE
        self.memory_peak = 0.0
        self.cpu_peak = 0.0
        self.data_processed = 0
        self.errors = 0
        self.warnings = 0
        self.custom_metrics: Dict[str, Any] = {}
        self.performance_monitor = PerformanceMonitor()
    
    def start_step(self) -> None:
        """Start monitoring a step."""
        self.start_time = time.time()
        self.status = MonitorStatus.RUNNING
        self.performance_monitor.start_monitoring(interval=0.5)
        self.logger.info(f"Started monitoring step: {self.step_name}")
    
    def end_step(self, status: MonitorStatus = MonitorStatus.COMPLETED) -> None:
        """End monitoring a step."""
        self.end_time = time.time()
        self.status = status
        self.performance_monitor.stop_monitoring()
        
        # Get peak metrics
        current_metrics = self.performance_monitor.get_current_metrics()
        if current_metrics:
            self.memory_peak = current_metrics.get("memory", {}).get("percent", 0.0)
            self.cpu_peak = current_metrics.get("cpu", {}).get("percent", 0.0)
        
        self.logger.info(f"Ended monitoring step: {self.step_name} - Status: {status.value}")
    
    def record_error(self) -> None:
        """Record an error in the step."""
        self.errors += 1
        self.logger.warning(f"Error recorded in step: {self.step_name} (Total: {self.errors})")
    
    def record_warning(self) -> None:
        """Record a warning in the step."""
        self.warnings += 1
        self.logger.info(f"Warning recorded in step: {self.step_name} (Total: {self.warnings})")
    
    def record_data_processed(self, size: int) -> None:
        """Record data processed in the step."""
        self.data_processed += size
        self.logger.debug(f"Data processed in step: {self.step_name} - {size} bytes (Total: {self.data_processed})")
    
    def set_custom_metric(self, name: str, value: Any) -> None:
        """Set a custom metric for the step."""
        self.custom_metrics[name] = value
        self.logger.debug(f"Custom metric set in step: {self.step_name} - {name}: {value}")
    
    def get_step_metrics(self) -> StepMetrics:
        """Get metrics for the step."""
        duration = 0.0
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        elif self.start_time:
            duration = time.time() - self.start_time
        
        return StepMetrics(
            step_name=self.step_name,
            start_time=format_datetime(get_current_datetime()) if self.start_time else "",
            end_time=format_datetime(get_current_datetime()) if self.end_time else None,
            duration=duration,
            status=self.status,
            memory_peak=self.memory_peak,
            cpu_peak=self.cpu_peak,
            data_processed=self.data_processed,
            errors=self.errors,
            warnings=self.warnings,
            custom_metrics=self.custom_metrics
        )


class PipelineMonitor:
    """Monitor the entire pipeline execution."""
    
    def __init__(self, pipeline_id: str, config: Dict[str, Any]):
        self.pipeline_id = pipeline_id
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = MonitorStatus.IDLE
        self.steps_total = 0
        self.steps_completed = 0
        self.step_monitors: Dict[str, StepMonitor] = {}
        self.total_memory_peak = 0.0
        self.total_cpu_peak = 0.0
        self.total_data_processed = 0
        self.total_errors = 0
        self.total_warnings = 0
        self.custom_metrics: Dict[str, Any] = {}
        self.monitoring_file = Path(config.get('monitoring_file', f'logs/pipeline_monitor_{pipeline_id}.json'))
        
        # Ensure monitoring directory exists
        self.monitoring_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start_pipeline(self, total_steps: int) -> None:
        """Start monitoring the pipeline."""
        self.start_time = time.time()
        self.status = MonitorStatus.RUNNING
        self.steps_total = total_steps
        self.steps_completed = 0
        self.logger.info(f"Started monitoring pipeline: {self.pipeline_id} ({total_steps} steps)")
    
    def end_pipeline(self, status: MonitorStatus = MonitorStatus.COMPLETED) -> None:
        """End monitoring the pipeline."""
        self.end_time = time.time()
        self.status = status
        
        # Calculate totals
        for step_monitor in self.step_monitors.values():
            step_metrics = step_monitor.get_step_metrics()
            self.total_memory_peak = max(self.total_memory_peak, step_metrics.memory_peak)
            self.total_cpu_peak = max(self.total_cpu_peak, step_metrics.cpu_peak)
            self.total_data_processed += step_metrics.data_processed
            self.total_errors += step_metrics.errors
            self.total_warnings += step_metrics.warnings
        
        self.logger.info(f"Ended monitoring pipeline: {self.pipeline_id} - Status: {status.value}")
        
        # Save final metrics
        self._save_metrics()
    
    def start_step(self, step_name: str) -> StepMonitor:
        """Start monitoring a step."""
        if step_name not in self.step_monitors:
            self.step_monitors[step_name] = StepMonitor(step_name)
        
        step_monitor = self.step_monitors[step_name]
        step_monitor.start_step()
        return step_monitor
    
    def end_step(self, step_name: str, status: MonitorStatus = MonitorStatus.COMPLETED) -> None:
        """End monitoring a step."""
        if step_name in self.step_monitors:
            self.step_monitors[step_name].end_step(status)
            if status == MonitorStatus.COMPLETED:
                self.steps_completed += 1
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get overall pipeline metrics."""
        total_duration = 0.0
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
        elif self.start_time:
            total_duration = time.time() - self.start_time
        
        step_metrics = [monitor.get_step_metrics() for monitor in self.step_monitors.values()]
        
        return PipelineMetrics(
            pipeline_id=self.pipeline_id,
            start_time=format_datetime(get_current_datetime()) if self.start_time else "",
            end_time=format_datetime(get_current_datetime()) if self.end_time else None,
            total_duration=total_duration,
            status=self.status,
            steps_completed=self.steps_completed,
            steps_total=self.steps_total,
            total_memory_peak=self.total_memory_peak,
            total_cpu_peak=self.total_cpu_peak,
            total_data_processed=self.total_data_processed,
            total_errors=self.total_errors,
            total_warnings=self.total_warnings,
            step_metrics=step_metrics,
            custom_metrics=self.custom_metrics
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress."""
        progress_percent = 0.0
        if self.steps_total > 0:
            progress_percent = (self.steps_completed / self.steps_total) * 100
        
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "progress_percent": progress_percent,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "current_step": self._get_current_step(),
            "estimated_remaining_time": self._estimate_remaining_time()
        }
    
    def _get_current_step(self) -> Optional[str]:
        """Get the currently running step."""
        for step_name, monitor in self.step_monitors.items():
            if monitor.status == MonitorStatus.RUNNING:
                return step_name
        return None
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on current progress."""
        if not self.start_time or self.steps_completed == 0:
            return None
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_step = elapsed_time / self.steps_completed
        remaining_steps = self.steps_total - self.steps_completed
        
        return remaining_steps * avg_time_per_step
    
    def _save_metrics(self) -> None:
        """Save pipeline metrics to file."""
        try:
            metrics = self.get_pipeline_metrics()
            metrics_dict = asdict(metrics)
            safe_json_dump(metrics_dict, self.monitoring_file)
            self.logger.info(f"Pipeline metrics saved to: {self.monitoring_file}")
        except Exception as e:
            self.logger.exception(f"Error saving pipeline metrics: {e}")
    
    def print_progress_report(self) -> None:
        """Print a formatted progress report."""
        progress = self.get_progress()
        metrics = self.get_pipeline_metrics()
        
        print("\n" + "="*80)
        print("📊 PIPELINE PROGRESS REPORT")
        print("="*80)
        print(f"Pipeline ID: {progress['pipeline_id']}")
        print(f"Status: {progress['status']}")
        print(f"Progress: {progress['progress_percent']:.1f}% ({progress['steps_completed']}/{progress['steps_total']} steps)")
        
        if progress['current_step']:
            print(f"Current Step: {progress['current_step']}")
        
        if progress['estimated_remaining_time']:
            remaining_minutes = progress['estimated_remaining_time'] / 60
            print(f"Estimated Remaining Time: {remaining_minutes:.1f} minutes")
        
        print(f"Total Duration: {metrics.total_duration:.1f} seconds")
        print(f"Memory Peak: {metrics.total_memory_peak:.1f}%")
        print(f"CPU Peak: {metrics.total_cpu_peak:.1f}%")
        print(f"Data Processed: {metrics.total_data_processed:,} bytes")
        print(f"Errors: {metrics.total_errors}")
        print(f"Warnings: {metrics.total_warnings}")
        
        if metrics.step_metrics:
            print("\nStep Details:")
            for step in metrics.step_metrics:
                status_icon = {
                    MonitorStatus.COMPLETED: "✅",
                    MonitorStatus.FAILED: "❌",
                    MonitorStatus.RUNNING: "🔄",
                    MonitorStatus.PAUSED: "⏸️",
                    MonitorStatus.CANCELLED: "⏹️",
                    MonitorStatus.IDLE: "⏳"
                }.get(step.status, "❓")
                
                print(f"  {status_icon} {step.step_name}: {step.duration:.1f}s | "
                      f"Memory: {step.memory_peak:.1f}% | "
                      f"CPU: {step.cpu_peak:.1f}% | "
                      f"Data: {step.data_processed:,} bytes | "
                      f"Errors: {step.errors} | "
                      f"Warnings: {step.warnings}")
        
        print("="*80)


class RealTimeMonitor:
    """Real-time monitoring with live updates."""
    
    def __init__(self, pipeline_monitor: PipelineMonitor):
        self.pipeline_monitor = pipeline_monitor
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.update_interval = 5.0  # seconds
    
    def start_real_time_monitoring(self) -> None:
        """Start real-time monitoring with live updates."""
        self.monitoring = True
        self.logger.info("Real-time monitoring started")
        
        # Print initial report
        self.pipeline_monitor.print_progress_report()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    def stop_real_time_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring = False
        self.logger.info("Real-time monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Real-time monitoring loop."""
        while self.monitoring:
            try:
                # Clear screen and print updated report
                os.system('clear' if os.name == 'posix' else 'cls')
                self.pipeline_monitor.print_progress_report()
                
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                self.logger.exception(f"Error in real-time monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)


# Export main classes
__all__ = [
    'MonitorStatus',
    'MetricType',
    'MetricData',
    'StepMetrics',
    'PipelineMetrics',
    'PerformanceMonitor',
    'StepMonitor',
    'PipelineMonitor',
    'RealTimeMonitor'
]