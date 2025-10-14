"""
Advanced Performance Monitoring Framework for Unified Data-Driven Pipeline.

This module provides comprehensive performance monitoring infrastructure similar to
FeatureLookbackOptimizationComponent but adapted for the unified pipeline.
"""

import logging
import time
import psutil
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import utility modules
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation, validate_dataframe_columns,
    analyze_nan_values_detailed, calculate_data_quality_metrics,
    create_data_quality_report, get_dataframe_info, create_summary_statistics
)
from src.utils.serialization_utils import UniversalSerializer

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_performance
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)

import numpy as np
import pandas as pd


class MetricType(Enum):
    """Types of metrics."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    BUSINESS = "business"
    TECHNICAL = "technical"


class MetricLevel(Enum):
    """Metric severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    level: MetricLevel = MetricLevel.INFO
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary of metrics for a time period."""
    name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_value: float
    last_value: float
    first_timestamp: datetime
    last_timestamp: datetime
    level_distribution: Dict[str, int] = field(default_factory=dict)


class AdvancedPerformanceMonitor:
    """
    Advanced performance monitoring for unified pipeline.
    
    Provides comprehensive metrics collection, analysis, and alerting capabilities
    for performance monitoring and optimization tracking.
    """

    def __init__(self, component_name: str = "UnifiedDataDrivenPipeline"):
        """Initialize the advanced performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.component_name = component_name
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        tprint_success(f"[AdvancedPerformanceMonitor] Initializing monitoring for component: {self.component_name}")

        # Metrics storage
        self.metrics: List[MetricPoint] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Performance tracking
        self.performance_metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0,
            'peak_memory_mb': 0.0,
            'memory_warnings': 0,
            'operations': {},
            'cache_metrics': {
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0
            }
        }

        # Operation tracking
        self.operation_start_times: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = {}
        self.operation_total_times: Dict[str, float] = {}

        # Memory monitoring
        self.memory_warning_threshold_mb = 1000.0  # 1GB
        self.memory_critical_threshold_mb = 2000.0  # 2GB
        self.max_metrics_history = 10000

        # Initialize process monitoring
        try:
            self.process = psutil.Process()
            self.initial_memory_mb = self.process.memory_info().rss / 1024 / 1024
        except Exception as e:
            tprint_warning(f"⚠️ Could not initialize process monitoring: {e}")
            self.process = None
            self.initial_memory_mb = 0.0

        tprint_success(f"✅ AdvancedPerformanceMonitor initialized for {self.component_name}")

    def monitor_data_quality(self, data: pd.DataFrame, operation_name: str = "data_quality_check") -> Dict[str, Any]:
        """
        Monitor data quality using enhanced utilities.
        
        Args:
            data: DataFrame to analyze
            operation_name: Name of the operation for tracking
            
        Returns:
            Dictionary with data quality metrics
        """
        start_time = time.time()
        
        try:
            # Use utility functions for comprehensive analysis
            nan_analysis = analyze_nan_values_detailed(data)
            quality_metrics = calculate_data_quality_metrics(data)
            quality_report = create_data_quality_report(data)
            dataframe_info = get_dataframe_info(data)
            summary_stats = create_summary_statistics(data)
            
            # Record metrics
            self.record_metric(
                MetricType.QUALITY,
                quality_metrics.get('missing_percentage', 0),
                unit="percentage",
                metadata={
                    'operation': operation_name,
                    'data_shape': data.shape,
                    'quality_score': quality_metrics.get('quality_score', 0)
                }
            )
            
            # Create comprehensive quality report
            quality_summary = {
                'operation_name': operation_name,
                'timestamp': datetime.now().isoformat(),
                'data_shape': data.shape,
                'nan_analysis': nan_analysis,
                'quality_metrics': quality_metrics,
                'dataframe_info': dataframe_info,
                'summary_statistics': summary_stats,
                'analysis_time': time.time() - start_time
            }
            
            tprint_performance(f"📊 Data quality analysis completed for {operation_name}: {quality_metrics.get('missing_percentage', 0):.1f}% missing")
            
            return quality_summary
            
        except Exception as e:
            tprint_error(f"❌ Data quality monitoring failed for {operation_name}: {e}")
            return {
                'operation_name': operation_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'analysis_time': time.time() - start_time
            }

    def start_operation(self, operation_name: str) -> float:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
            
        Returns:
            Start time as timestamp
        """
        start_time = time.time()
        self.operation_start_times[operation_name] = start_time
        
        if operation_name not in self.operation_counts:
            self.operation_counts[operation_name] = 0
            self.operation_total_times[operation_name] = 0.0
        
        self.operation_counts[operation_name] += 1
        
        tprint_debug(f"⏱️ Started operation: {operation_name}")
        return start_time

    def end_operation(self, operation_name: str, start_time: float, success: bool = True) -> float:
        """
        End timing an operation.
        
        Args:
            operation_name: Name of the operation
            start_time: Start time from start_operation
            success: Whether the operation was successful
            
        Returns:
            Execution time in seconds
        """
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update operation statistics
        self.operation_total_times[operation_name] += execution_time
        self.performance_metrics['execution_times'][operation_name] = execution_time
        
        # Record metric
        self.record_metric(
            name=f"{operation_name}_execution_time",
            value=execution_time,
            metric_type=MetricType.PERFORMANCE,
            tags={'operation': operation_name, 'success': str(success)}
        )
        
        # Clean up
        if operation_name in self.operation_start_times:
            del self.operation_start_times[operation_name]
        
        tprint_performance(f"⏱️ Completed operation {operation_name} in {execution_time:.3f}s")
        return execution_time

    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.PERFORMANCE,
                     level: MetricLevel = MetricLevel.INFO, tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            level: Severity level
            tags: Additional tags
            metadata: Additional metadata
        """
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metric_type=metric_type,
            level=level,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Trim metrics if too many
        if len(self.metrics) > self.max_metrics_history:
            self.metrics = self.metrics[-self.max_metrics_history:]
        
        tprint_debug(f"📊 Recorded metric: {name} = {value}")

    def record_memory_usage(self):
        """Record current memory usage."""
        if self.process is None:
            return
        
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.performance_metrics['memory_usage'].append(memory_mb)
            
            # Track peak memory
            if memory_mb > self.performance_metrics['peak_memory_mb']:
                self.performance_metrics['peak_memory_mb'] = memory_mb
            
            # Check for memory warnings
            if memory_mb > self.memory_critical_threshold_mb:
                self.performance_metrics['memory_warnings'] += 1
                self.record_metric(
                    name="memory_usage_critical",
                    value=memory_mb,
                    metric_type=MetricType.RESOURCE,
                    level=MetricLevel.CRITICAL,
                    tags={'threshold': 'critical'}
                )
                tprint_error(f"🚨 CRITICAL: Memory usage {memory_mb:.1f}MB exceeds critical threshold")
            elif memory_mb > self.memory_warning_threshold_mb:
                self.performance_metrics['memory_warnings'] += 1
                self.record_metric(
                    name="memory_usage_warning",
                    value=memory_mb,
                    metric_type=MetricType.RESOURCE,
                    level=MetricLevel.WARNING,
                    tags={'threshold': 'warning'}
                )
                tprint_warning(f"⚠️ WARNING: Memory usage {memory_mb:.1f}MB exceeds warning threshold")
            else:
                self.record_metric(
                    name="memory_usage",
                    value=memory_mb,
                    metric_type=MetricType.RESOURCE,
                    level=MetricLevel.INFO
                )
            
        except Exception as e:
            tprint_warning(f"⚠️ Could not record memory usage: {e}")

    def record_cpu_usage(self):
        """Record current CPU usage."""
        if self.process is None:
            return
        
        try:
            cpu_percent = self.process.cpu_percent()
            self.performance_metrics['cpu_usage'].append(cpu_percent)
            
            self.record_metric(
                name="cpu_usage",
                value=cpu_percent,
                metric_type=MetricType.RESOURCE,
                level=MetricLevel.INFO
            )
            
        except Exception as e:
            tprint_warning(f"⚠️ Could not record CPU usage: {e}")

    def record_cache_metrics(self, hits: int, misses: int):
        """Record cache metrics."""
        self.performance_metrics['cache_metrics']['hits'] += hits
        self.performance_metrics['cache_metrics']['misses'] += misses
        
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        self.performance_metrics['cache_metrics']['hit_rate'] = hit_rate
        
        self.record_metric(
            name="cache_hit_rate",
            value=hit_rate,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={'hits': str(hits), 'misses': str(misses)}
        )

    def record_optimization_metrics(self, optimization_results: Dict[str, Any], 
                                  data_quality_score: float = 0.0,
                                  validation_score: float = 0.0):
        """Record optimization-specific metrics."""
        # Record data quality
        self.record_metric(
            name="data_quality_score",
            value=data_quality_score,
            metric_type=MetricType.QUALITY,
            level=MetricLevel.INFO
        )
        
        # Record validation score
        self.record_metric(
            name="validation_score",
            value=validation_score,
            metric_type=MetricType.QUALITY,
            level=MetricLevel.INFO
        )
        
        # Record optimization results
        if 'total_features' in optimization_results:
            self.record_metric(
                name="total_features_optimized",
                value=float(optimization_results['total_features']),
                metric_type=MetricType.BUSINESS,
                level=MetricLevel.INFO
            )
        
        if 'execution_time' in optimization_results:
            self.record_metric(
                name="optimization_execution_time",
                value=optimization_results['execution_time'],
                metric_type=MetricType.PERFORMANCE,
                level=MetricLevel.INFO
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'component_name': self.component_name,
            'monitoring_duration': self._get_monitoring_duration(),
            'total_metrics': len(self.metrics),
            'operations': self._get_operation_summary(),
            'memory_stats': self._get_memory_stats(),
            'cpu_stats': self._get_cpu_stats(),
            'cache_stats': self.performance_metrics['cache_metrics'].copy(),
            'error_count': self.performance_metrics['error_counts'],
            'memory_warnings': self.performance_metrics['memory_warnings']
        }
        
        return summary

    def _get_monitoring_duration(self) -> float:
        """Get total monitoring duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0

    def _get_operation_summary(self) -> Dict[str, Any]:
        """Get operation execution summary."""
        summary = {}
        for operation, count in self.operation_counts.items():
            total_time = self.operation_total_times.get(operation, 0.0)
            avg_time = total_time / count if count > 0 else 0.0
            
            summary[operation] = {
                'count': count,
                'total_time': total_time,
                'average_time': avg_time
            }
        
        return summary

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        memory_usage = self.performance_metrics['memory_usage']
        if not memory_usage:
            return {'current_mb': 0.0, 'peak_mb': 0.0, 'average_mb': 0.0}
        
        return {
            'current_mb': memory_usage[-1] if memory_usage else 0.0,
            'peak_mb': self.performance_metrics['peak_memory_mb'],
            'average_mb': np.mean(memory_usage),
            'min_mb': np.min(memory_usage),
            'max_mb': np.max(memory_usage),
            'samples': len(memory_usage)
        }

    def _get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU usage statistics."""
        cpu_usage = self.performance_metrics['cpu_usage']
        if not cpu_usage:
            return {'current_percent': 0.0, 'average_percent': 0.0}
        
        return {
            'current_percent': cpu_usage[-1] if cpu_usage else 0.0,
            'average_percent': np.mean(cpu_usage),
            'min_percent': np.min(cpu_usage),
            'max_percent': np.max(cpu_usage),
            'samples': len(cpu_usage)
        }

    def get_metric_summary(self, metric_name: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Optional[MetricSummary]:
        """Get summary for a specific metric."""
        # Filter metrics by name and time range
        filtered_metrics = [
            m for m in self.metrics 
            if m.name == metric_name and 
            (start_time is None or m.timestamp >= start_time) and
            (end_time is None or m.timestamp <= end_time)
        ]
        
        if not filtered_metrics:
            return None
        
        values = [m.value for m in filtered_metrics]
        timestamps = [m.timestamp for m in filtered_metrics]
        levels = [m.level.value for m in filtered_metrics]
        
        # Calculate level distribution
        level_distribution = {}
        for level in levels:
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        return MetricSummary(
            name=metric_name,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=np.mean(values),
            median_value=np.median(values),
            std_value=np.std(values),
            last_value=values[-1],
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps),
            level_distribution=level_distribution
        )

    def reset_stats(self):
        """Reset all performance statistics."""
        self.metrics = []
        self.performance_metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0,
            'peak_memory_mb': 0.0,
            'memory_warnings': 0,
            'operations': {},
            'cache_metrics': {
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0
            }
        }
        self.operation_start_times = {}
        self.operation_counts = {}
        self.operation_total_times = {}
        self.start_time = None
        self.end_time = None
        
        tprint_success("✅ Performance statistics reset")

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = datetime.now()
        tprint_success("📊 Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = datetime.now()
        tprint_success("📊 Performance monitoring stopped")

    def update_cache_metrics(self, cache_metrics: Dict[str, Any]):
        """Update cache metrics from external source."""
        self.performance_metrics['cache_metrics'].update(cache_metrics)
        
        hits = cache_metrics.get('hits', 0)
        misses = cache_metrics.get('misses', 0)
        self.record_cache_metrics(hits, misses)