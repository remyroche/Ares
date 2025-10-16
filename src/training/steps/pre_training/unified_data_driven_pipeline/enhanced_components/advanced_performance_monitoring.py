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

# Import features_common performance utilities
try:
    from src.features_common import (
        PerformanceMixin, MonitoringMixin, VectorBTPerformanceMonitor,
        get_performance_monitor, GPUAccelerator, get_gpu_accelerator,
        VectorBTOptimizationEngine, get_optimization_engine,
        validate_input_data, safe_execute, get_logger, log_operation
    )
    FEATURES_COMMON_PERFORMANCE_AVAILABLE = True
except ImportError:
    FEATURES_COMMON_PERFORMANCE_AVAILABLE = False

# Import feature_generation optimization utilities
try:
    from src.feature_generation.utils import (
        FeatureGenerationOptimizer, FeatureOptimizationConfig,
        validate_feature_quality, validate_features_dataframe
    )
    FEATURE_GENERATION_PERFORMANCE_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_PERFORMANCE_AVAILABLE = False

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
        tprint_debug(f"📊 Starting data quality monitoring for: {operation_name}")
        start_time = time.time()

        # Validate input data
        if data is None:
            tprint_error(f"❌ Data is None for quality monitoring: {operation_name}")
            return {
                'operation_name': operation_name,
                'error': 'Data is None',
                'timestamp': datetime.now().isoformat(),
                'analysis_time': 0.0
            }

        if data.empty:
            tprint_warning(f"⚠️ Data is empty for quality monitoring: {operation_name}")
            return {
                'operation_name': operation_name,
                'warning': 'Data is empty',
                'timestamp': datetime.now().isoformat(),
                'analysis_time': time.time() - start_time
            }

        tprint_debug(f"📊 Data shape: {data.shape}, columns: {len(data.columns)}")

        try:
            # Use utility functions for comprehensive analysis
            tprint_debug("🔍 Analyzing NaN values")
            nan_analysis = analyze_nan_values_detailed(data)

            tprint_debug("🔍 Calculating quality metrics")
            quality_metrics = calculate_data_quality_metrics(data)

            tprint_debug("🔍 Creating quality report")
            quality_report = create_data_quality_report(data)

            tprint_debug("🔍 Getting dataframe info")
            dataframe_info = get_dataframe_info(data)

            tprint_debug("🔍 Creating summary statistics")
            summary_stats = create_summary_statistics(data)

            # Record metrics
            tprint_debug("📝 Recording quality metrics")
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

            # Log quality metrics
            missing_pct = quality_metrics.get('missing_percentage', 0)
            quality_score = quality_metrics.get('quality_score', 0)

            if missing_pct > 50:
                tprint_warning(f"⚠️ High missing data percentage: {missing_pct:.1f}% for {operation_name}")
            elif missing_pct > 20:
                tprint_warning(f"⚠️ Moderate missing data percentage: {missing_pct:.1f}% for {operation_name}")
            else:
                tprint_success(f"✅ Low missing data percentage: {missing_pct:.1f}% for {operation_name}")

            if quality_score < 0.5:
                tprint_warning(f"⚠️ Low quality score: {quality_score:.3f} for {operation_name}")
            else:
                tprint_success(f"✅ Good quality score: {quality_score:.3f} for {operation_name}")

            tprint_performance(f"📊 Data quality analysis completed for {operation_name}: {missing_pct:.1f}% missing, score: {quality_score:.3f}")

            return quality_summary

        except Exception as e:
            tprint_error(f"❌ Data quality monitoring failed for {operation_name}: {e}")
            tprint_error(f"❌ Error type: {type(e).__name__}")
            return {
                'operation_name': operation_name,
                'error': str(e),
                'error_type': type(e).__name__,
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
        tprint_debug(f"⏱️ Starting operation: {operation_name}")

        start_time = time.time()
        self.operation_start_times[operation_name] = start_time

        if operation_name not in self.operation_counts:
            self.operation_counts[operation_name] = 0
            self.operation_total_times[operation_name] = 0.0
            tprint_debug(f"📊 First time tracking operation: {operation_name}")

        self.operation_counts[operation_name] += 1

        # Log operation start with context
        tprint_debug(f"⏱️ Started operation: {operation_name} (count: {self.operation_counts[operation_name]})")

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
        tprint_debug(f"⏱️ Ending operation: {operation_name}")

        end_time = time.time()
        execution_time = end_time - start_time

        # Validate execution time
        if execution_time < 0:
            tprint_warning(f"⚠️ Negative execution time for {operation_name}: {execution_time:.3f}s")
            execution_time = 0.0

        # Update operation statistics
        self.operation_total_times[operation_name] += execution_time
        self.performance_metrics['execution_times'][operation_name] = execution_time

        # Record metric
        try:
            self.record_metric(
                name=f"{operation_name}_execution_time",
                value=execution_time,
                metric_type=MetricType.PERFORMANCE,
                tags={'operation': operation_name, 'success': str(success)}
            )
            tprint_debug(f"📝 Recorded metric for {operation_name}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to record metric for {operation_name}: {e}")

        # Clean up
        if operation_name in self.operation_start_times:
            del self.operation_start_times[operation_name]
            tprint_debug(f"🧹 Cleaned up start time for {operation_name}")

        # Log completion with success status
        if success:
            tprint_success(f"✅ Completed operation {operation_name} in {execution_time:.3f}s")
        else:
            tprint_warning(f"⚠️ Completed operation {operation_name} with issues in {execution_time:.3f}s")

        tprint_performance(f"⏱️ Operation {operation_name} finished: {execution_time:.3f}s (success: {success})")

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

    def track_operation_performance(self, operation_name: str,
                                  start_time: Optional[datetime] = None,
                                  end_time: Optional[datetime] = None,
                                  memory_before: Optional[float] = None,
                                  memory_after: Optional[float] = None,
                                  **metadata) -> None:
        """
        Track detailed performance metrics for a specific operation.

        Args:
            operation_name: Name of the operation
            start_time: Operation start time (defaults to now)
            end_time: Operation end time (defaults to now)
            memory_before: Memory usage before operation (MB)
            memory_after: Memory usage after operation (MB)
            **metadata: Additional metadata
        """
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            end_time = datetime.now()

        execution_time = (end_time - start_time).total_seconds()

        # Record execution time
        self.record_metric(
            name=f"{operation_name}_execution_time",
            value=execution_time,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            metadata=metadata
        )

        # Record memory usage if provided
        if memory_before is not None and memory_after is not None:
            memory_delta = memory_after - memory_before
            self.record_metric(
                name=f"{operation_name}_memory_delta",
                value=memory_delta,
                metric_type=MetricType.RESOURCE,
                level=MetricLevel.INFO if abs(memory_delta) < 100 else MetricLevel.WARNING,
                metadata=metadata
            )

        # Update operation counts
        self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1
        self.operation_total_times[operation_name] = self.operation_total_times.get(operation_name, 0.0) + execution_time

        tprint_debug(f"📊 Tracked operation '{operation_name}': {execution_time:.3f}s")

    def analyze_performance_trends(self, metric_name: str,
                                 window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze performance trends for a specific metric.

        Args:
            metric_name: Name of the metric to analyze
            window_size: Window size for trend analysis

        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Get metric values
            metric_values = [m.value for m in self.metrics if m.name == metric_name]

            if len(metric_values) < window_size:
                return {'error': 'Insufficient data for trend analysis'}

            # Calculate rolling statistics
            rolling_mean = []
            rolling_std = []

            for i in range(window_size - 1, len(metric_values)):
                window_values = metric_values[i - window_size + 1:i + 1]
                rolling_mean.append(np.mean(window_values))
                rolling_std.append(np.std(window_values))

            # Calculate trend direction
            if len(rolling_mean) >= 2:
                trend_direction = 'improving' if rolling_mean[-1] < rolling_mean[0] else 'degrading'
                trend_strength = abs(rolling_mean[-1] - rolling_mean[0]) / rolling_mean[0] if rolling_mean[0] != 0 else 0
            else:
                trend_direction = 'stable'
                trend_strength = 0.0

            # Calculate volatility
            volatility = np.std(rolling_std) if rolling_std else 0.0

            return {
                'metric_name': metric_name,
                'total_samples': len(metric_values),
                'window_size': window_size,
                'current_value': metric_values[-1],
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'analysis_timestamp': datetime.now()
            }

        except Exception as e:
            tprint_warning(f"⚠️ Trend analysis failed for {metric_name}: {e}")
            return {'error': str(e)}

    def detect_performance_anomalies(self, metric_name: str,
                                   threshold_std: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect performance anomalies in a metric.

        Args:
            metric_name: Name of the metric to analyze
            threshold_std: Standard deviation threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        try:
            metric_data = [(m.timestamp, m.value) for m in self.metrics if m.name == metric_name]

            if len(metric_data) < 10:
                return []

            timestamps, values = zip(*metric_data)
            values = np.array(values)

            # Calculate z-scores
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:
                return []

            z_scores = np.abs((values - mean_val) / std_val)

            # Find anomalies
            anomalies = []
            for i, (timestamp, value, z_score) in enumerate(zip(timestamps, values, z_scores)):
                if z_score > threshold_std:
                    anomalies.append({
                        'timestamp': timestamp,
                        'value': value,
                        'z_score': z_score,
                        'severity': 'high' if z_score > threshold_std * 2 else 'medium',
                        'index': i
                    })

            if anomalies:
                tprint_warning(f"⚠️ Detected {len(anomalies)} anomalies in {metric_name}")

            return anomalies

        except Exception as e:
            tprint_warning(f"⚠️ Anomaly detection failed for {metric_name}: {e}")
            return []

    def generate_performance_report(self,
                                  include_trends: bool = True,
                                  include_anomalies: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            include_trends: Include trend analysis
            include_anomalies: Include anomaly detection

        Returns:
            Comprehensive performance report
        """
        try:
            report = {
                'report_timestamp': datetime.now(),
                'component_name': self.component_name,
                'monitoring_duration': self._get_monitoring_duration(),
                'summary': self.get_performance_summary()
            }

            # Get unique metric names
            metric_names = list(set(m.name for m in self.metrics))

            if include_trends:
                report['trends'] = {}
                for metric_name in metric_names:
                    trend_analysis = self.analyze_performance_trends(metric_name)
                    if 'error' not in trend_analysis:
                        report['trends'][metric_name] = trend_analysis

            if include_anomalies:
                report['anomalies'] = {}
                for metric_name in metric_names:
                    anomalies = self.detect_performance_anomalies(metric_name)
                    if anomalies:
                        report['anomalies'][metric_name] = anomalies

            # Add recommendations
            report['recommendations'] = self._generate_performance_recommendations()

            tprint_success("✅ Generated comprehensive performance report")
            return report

        except Exception as e:
            tprint_error(f"❌ Performance report generation failed: {e}")
            return {'error': str(e)}

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []

        # Check memory usage
        memory_stats = self._get_memory_stats()
        if memory_stats['peak_mb'] > 1000:
            recommendations.append("Consider optimizing memory usage - peak usage exceeded 1GB")

        # Check execution times
        operation_summary = self._get_operation_summary()
        for operation, stats in operation_summary.items():
            if stats['average_time'] > 10:  # 10 seconds
                recommendations.append(f"Consider optimizing '{operation}' - average execution time is {stats['average_time']:.1f}s")

        # Check error rates
        if self.performance_metrics['error_counts'] > 0:
            recommendations.append(f"Address {self.performance_metrics['error_counts']} errors detected during monitoring")

        # Check cache performance
        cache_stats = self.performance_metrics['cache_metrics']
        if cache_stats['hit_rate'] < 0.5 and (cache_stats['hits'] + cache_stats['misses']) > 0:
            recommendations.append("Consider improving cache hit rate - currently below 50%")

        return recommendations

    def export_metrics_to_csv(self, filepath: str,
                            metric_names: Optional[List[str]] = None) -> bool:
        """
        Export metrics to CSV file.

        Args:
            filepath: Path to output CSV file
            metric_names: Specific metrics to export (None = all)

        Returns:
            True if successful, False otherwise
        """
        try:
            import csv

            # Filter metrics if specific names provided
            if metric_names:
                filtered_metrics = [m for m in self.metrics if m.name in metric_names]
            else:
                filtered_metrics = self.metrics

            if not filtered_metrics:
                tprint_warning("⚠️ No metrics to export")
                return False

            # Write to CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'name', 'value', 'metric_type', 'level', 'tags', 'metadata']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for metric in filtered_metrics:
                    writer.writerow({
                        'timestamp': metric.timestamp.isoformat(),
                        'name': metric.name,
                        'value': metric.value,
                        'metric_type': metric.metric_type.value,
                        'level': metric.level.value,
                        'tags': str(metric.tags),
                        'metadata': str(metric.metadata)
                    })

            tprint_success(f"✅ Exported {len(filtered_metrics)} metrics to {filepath}")
            return True

        except Exception as e:
            tprint_error(f"❌ CSV export failed: {e}")
            return False
