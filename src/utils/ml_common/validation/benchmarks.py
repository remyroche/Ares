"""
Performance Benchmarking Utilities

Provides utilities for measuring and monitoring performance metrics
in the Analyst→Tactician pipeline.
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
import threading
import queue

from src.utils.tprint import tprint_info, tprint_warning, tprint_success
from src.utils.logger import system_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    rows_processed: int = 0
    columns_processed: int = 0
    na_ratio: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    monitor_memory: bool = True
    monitor_cpu: bool = True
    memory_sample_interval: float = 1.0  # seconds
    cpu_sample_interval: float = 1.0  # seconds
    enable_detailed_logging: bool = True
    max_benchmark_duration: float = 3600.0  # 1 hour max
    memory_warning_threshold_mb: float = 8192.0  # 8GB


class PerformanceMonitor:
    """Monitors performance metrics during execution."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.logger = system_logger.getChild('PerformanceMonitor')
        self._monitoring = False
        self._monitor_thread = None
        self._memory_samples = []
        self._cpu_samples = []
        self._start_time = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._start_time = time.time()
        self._memory_samples = []
        self._cpu_samples = []
        self._stop_event.clear()
        
        if self.config.monitor_memory or self.config.monitor_cpu:
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
        
        tprint_info(f"🔍 Started performance monitoring for {self.config.max_benchmark_duration}s max")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        tprint_info("⏹️ Stopped performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            current_time = time.time()
            
            if self._start_time and (current_time - self._start_time) > self.config.max_benchmark_duration:
                tprint_warning("⚠️ Benchmark duration exceeded maximum, stopping monitoring")
                break
            
            if self.config.monitor_memory:
                try:
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    self._memory_samples.append((current_time, memory_mb))
                    
                    if memory_mb > self.config.memory_warning_threshold_mb:
                        tprint_warning(f"⚠️ High memory usage: {memory_mb:.1f}MB")
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if self.config.monitor_cpu:
                try:
                    cpu_percent = psutil.Process().cpu_percent()
                    self._cpu_samples.append((current_time, cpu_percent))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sleep until next sample
            sleep_time = min(
                self.config.memory_sample_interval if self.config.monitor_memory else float('inf'),
                self.config.cpu_sample_interval if self.config.monitor_cpu else float('inf')
            )
            
            if sleep_time == float('inf'):
                break
            
            self._stop_event.wait(sleep_time)
    
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_current_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.Process().cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage during monitoring."""
        if not self._memory_samples:
            return self.get_current_memory_mb()
        return max(sample[1] for sample in self._memory_samples)
    
    def get_average_cpu_percent(self) -> float:
        """Get average CPU usage during monitoring."""
        if not self._cpu_samples:
            return self.get_current_cpu_percent()
        return np.mean([sample[1] for sample in self._cpu_samples])


@contextmanager
def benchmark_stage(
    stage_name: str,
    config: Optional[BenchmarkConfig] = None,
    data: Optional[pd.DataFrame] = None,
    custom_metrics: Optional[Dict[str, Any]] = None
):
    """
    Context manager for benchmarking a pipeline stage.
    
    Args:
        stage_name: Name of the stage being benchmarked
        config: Benchmarking configuration
        data: Optional DataFrame to analyze
        custom_metrics: Optional custom metrics to track
    
    Yields:
        PerformanceMetrics object that can be updated during execution
    """
    monitor_config = config or BenchmarkConfig()
    monitor = PerformanceMonitor(monitor_config)
    
    # Initialize metrics
    metrics = PerformanceMetrics(
        stage_name=stage_name,
        start_time=datetime.now(),
        memory_start_mb=monitor.get_current_memory_mb(),
        custom_metrics=custom_metrics or {}
    )
    
    # Analyze input data if provided
    if data is not None:
        metrics.rows_processed = len(data)
        metrics.columns_processed = len(data.columns)
        metrics.na_ratio = data.isna().sum().sum() / (len(data) * len(data.columns)) if len(data) > 0 else 0.0
    
    try:
        monitor.start_monitoring()
        yield metrics
    
    except Exception as e:
        metrics.error_message = str(e)
        raise
    
    finally:
        monitor.stop_monitoring()
        
        # Finalize metrics
        metrics.end_time = datetime.now()
        metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.memory_end_mb = monitor.get_current_memory_mb()
        metrics.memory_peak_mb = monitor.get_peak_memory_mb()
        metrics.cpu_percent = monitor.get_average_cpu_percent()
        
        # Log performance summary
        if monitor_config.enable_detailed_logging:
            _log_performance_summary(metrics)


def _log_performance_summary(metrics: PerformanceMetrics):
    """Log a detailed performance summary."""
    logger = system_logger.getChild('BenchmarkSummary')
    
    tprint_success(f"📊 Performance Summary: {metrics.stage_name}")
    tprint_info(f"   ⏱️  Duration: {metrics.duration_seconds:.2f}s")
    tprint_info(f"   🧠 Memory: {metrics.memory_start_mb:.1f}MB → {metrics.memory_end_mb:.1f}MB (peak: {metrics.memory_peak_mb:.1f}MB)")
    tprint_info(f"   💻 CPU: {metrics.cpu_percent:.1f}% avg")
    
    if metrics.rows_processed > 0:
        tprint_info(f"   📈 Data: {metrics.rows_processed:,} rows × {metrics.columns_processed} cols")
        tprint_info(f"   🔢 NA Ratio: {metrics.na_ratio:.1%}")
        
        # Calculate throughput
        rows_per_second = metrics.rows_processed / metrics.duration_seconds if metrics.duration_seconds > 0 else 0
        tprint_info(f"   🚀 Throughput: {rows_per_second:,.0f} rows/s")
    
    if metrics.custom_metrics:
        tprint_info("   📋 Custom Metrics:")
        for key, value in metrics.custom_metrics.items():
            tprint_info(f"      {key}: {value}")
    
    if metrics.error_message:
        tprint_error(f"   ❌ Error: {metrics.error_message}")


def benchmark_function(
    func: Callable,
    *args,
    func_name: Optional[str] = None,
    config: Optional[BenchmarkConfig] = None,
    **kwargs
) -> tuple:
    """
    Benchmark a function call.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        func_name: Optional name for the function
        config: Benchmarking configuration
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (result, metrics)
    """
    name = func_name or func.__name__
    
    with benchmark_stage(f"function:{name}", config) as metrics:
        result = func(*args, **kwargs)
        
        # Update metrics with result information if it's a DataFrame
        if isinstance(result, pd.DataFrame):
            metrics.rows_processed = len(result)
            metrics.columns_processed = len(result.columns)
            metrics.na_ratio = result.isna().sum().sum() / (len(result) * len(result.columns)) if len(result) > 0 else 0.0
        
        return result, metrics


def create_performance_report(
    metrics_list: List[PerformanceMetrics],
    include_summary: bool = True
) -> Dict[str, Any]:
    """
    Create a comprehensive performance report from multiple metrics.
    
    Args:
        metrics_list: List of PerformanceMetrics objects
        include_summary: Whether to include summary statistics
    
    Returns:
        Dictionary with performance report
    """
    if not metrics_list:
        return {'error': 'No metrics provided'}
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_stages': len(metrics_list),
        'stages': []
    }
    
    # Process individual stage metrics
    total_duration = 0.0
    total_memory_peak = 0.0
    total_rows_processed = 0
    total_columns_processed = 0
    
    for metrics in metrics_list:
        stage_report = {
            'stage_name': metrics.stage_name,
            'duration_seconds': metrics.duration_seconds,
            'memory_start_mb': metrics.memory_start_mb,
            'memory_end_mb': metrics.memory_end_mb,
            'memory_peak_mb': metrics.memory_peak_mb,
            'memory_delta_mb': metrics.memory_end_mb - metrics.memory_start_mb,
            'cpu_percent': metrics.cpu_percent,
            'rows_processed': metrics.rows_processed,
            'columns_processed': metrics.columns_processed,
            'na_ratio': metrics.na_ratio,
            'custom_metrics': metrics.custom_metrics,
            'error_message': metrics.error_message,
            'success': metrics.error_message is None
        }
        
        report['stages'].append(stage_report)
        
        # Accumulate totals
        total_duration += metrics.duration_seconds
        total_memory_peak = max(total_memory_peak, metrics.memory_peak_mb)
        total_rows_processed += metrics.rows_processed
        total_columns_processed += metrics.columns_processed
    
    # Add summary if requested
    if include_summary:
        successful_stages = [m for m in metrics_list if m.error_message is None]
        
        report['summary'] = {
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60.0,
            'peak_memory_mb': total_memory_peak,
            'total_rows_processed': total_rows_processed,
            'total_columns_processed': total_columns_processed,
            'successful_stages': len(successful_stages),
            'failed_stages': len(metrics_list) - len(successful_stages),
            'success_rate': len(successful_stages) / len(metrics_list) if metrics_list else 0.0,
            'average_duration_seconds': total_duration / len(metrics_list) if metrics_list else 0.0,
            'fastest_stage': min(metrics_list, key=lambda m: m.duration_seconds).stage_name if metrics_list else None,
            'slowest_stage': max(metrics_list, key=lambda m: m.duration_seconds).stage_name if metrics_list else None,
            'memory_intensive_stage': max(metrics_list, key=lambda m: m.memory_peak_mb).stage_name if metrics_list else None
        }
        
        # Calculate throughput
        if total_duration > 0:
            report['summary']['overall_throughput_rows_per_second'] = total_rows_processed / total_duration
    
    return report


def validate_performance_requirements(
    metrics: PerformanceMetrics,
    requirements: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate performance metrics against requirements.
    
    Args:
        metrics: PerformanceMetrics to validate
        requirements: Performance requirements
    
    Returns:
        Dictionary with validation results
    """
    if requirements is None:
        requirements = {
            'max_duration_seconds': 300.0,  # 5 minutes
            'max_memory_peak_mb': 8192.0,   # 8GB
            'max_cpu_percent': 90.0,        # 90% CPU
            'min_throughput_rows_per_second': 1000.0,
            'max_na_ratio': 0.5             # 50% NA max
        }
    
    validation_results = {
        'stage_name': metrics.stage_name,
        'passed': True,
        'violations': [],
        'requirements': requirements
    }
    
    # Check duration
    if metrics.duration_seconds > requirements.get('max_duration_seconds', float('inf')):
        validation_results['violations'].append({
            'type': 'duration',
            'actual': metrics.duration_seconds,
            'limit': requirements['max_duration_seconds'],
            'message': f"Duration {metrics.duration_seconds:.1f}s exceeds limit {requirements['max_duration_seconds']:.1f}s"
        })
    
    # Check memory
    if metrics.memory_peak_mb > requirements.get('max_memory_peak_mb', float('inf')):
        validation_results['violations'].append({
            'type': 'memory',
            'actual': metrics.memory_peak_mb,
            'limit': requirements['max_memory_peak_mb'],
            'message': f"Peak memory {metrics.memory_peak_mb:.1f}MB exceeds limit {requirements['max_memory_peak_mb']:.1f}MB"
        })
    
    # Check CPU
    if metrics.cpu_percent > requirements.get('max_cpu_percent', float('inf')):
        validation_results['violations'].append({
            'type': 'cpu',
            'actual': metrics.cpu_percent,
            'limit': requirements['max_cpu_percent'],
            'message': f"CPU usage {metrics.cpu_percent:.1f}% exceeds limit {requirements['max_cpu_percent']:.1f}%"
        })
    
    # Check throughput
    if metrics.rows_processed > 0 and metrics.duration_seconds > 0:
        throughput = metrics.rows_processed / metrics.duration_seconds
        min_throughput = requirements.get('min_throughput_rows_per_second', 0.0)
        if throughput < min_throughput:
            validation_results['violations'].append({
                'type': 'throughput',
                'actual': throughput,
                'limit': min_throughput,
                'message': f"Throughput {throughput:.0f} rows/s below minimum {min_throughput:.0f} rows/s"
            })
    
    # Check NA ratio
    if metrics.na_ratio > requirements.get('max_na_ratio', 1.0):
        validation_results['violations'].append({
            'type': 'na_ratio',
            'actual': metrics.na_ratio,
            'limit': requirements['max_na_ratio'],
            'message': f"NA ratio {metrics.na_ratio:.1%} exceeds limit {requirements['max_na_ratio']:.1%}"
        })
    
    validation_results['passed'] = len(validation_results['violations']) == 0
    
    return validation_results


def benchmark_pipeline_stage(
    stage_name: str,
    stage_func: Callable,
    *args,
    config: Optional[BenchmarkConfig] = None,
    requirements: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a pipeline stage with validation.
    
    Args:
        stage_name: Name of the pipeline stage
        stage_func: Function to benchmark
        *args: Function arguments
        config: Benchmarking configuration
        requirements: Performance requirements
        **kwargs: Function keyword arguments
    
    Returns:
        Dictionary with benchmark results and validation
    """
    benchmark_config = config or BenchmarkConfig()
    
    # Extract data for analysis if present in args
    data = None
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            data = arg
            break
    
    try:
        with benchmark_stage(stage_name, benchmark_config, data) as metrics:
            result = stage_func(*args, **kwargs)
            
            # Update metrics with result information
            if isinstance(result, pd.DataFrame):
                metrics.rows_processed = len(result)
                metrics.columns_processed = len(result.columns)
                metrics.na_ratio = result.isna().sum().sum() / (len(result) * len(result.columns)) if len(result) > 0 else 0.0
            
            # Validate performance if requirements provided
            validation = None
            if requirements:
                validation = validate_performance_requirements(metrics, requirements)
            
            return {
                'success': True,
                'result': result,
                'metrics': metrics,
                'validation': validation
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'metrics': None,
            'validation': None
        }
