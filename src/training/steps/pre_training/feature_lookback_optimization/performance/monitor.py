"""
Performance Monitoring Framework for Feature Lookback Optimization.

This module provides comprehensive performance monitoring and metrics collection
throughout the optimization process with real-time tracking and analysis.
"""

import logging
import time
import psutil
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# Import utility modules
from src.utils.common_utilities import CommonUtilities
from src.utils.serialization_utils import UniversalSerializer
from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_performance

from ..dependency_manager import get_dependency

# Get dependencies with fallbacks
np, _ = get_dependency('numpy')
pd, _ = get_dependency('pandas')


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


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for feature lookback optimization.

    Provides real-time metrics collection, analysis, and alerting capabilities
    for performance monitoring and optimization tracking.
    """

    def __init__(self, component_name: str = "FeatureLookbackOptimization"):
        """Initialize the performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.component_name = component_name
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        tprint_success(
            f"[PerformanceMonitor] Initializing monitoring for component: {self.component_name}"
        )

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
            'warning_counts': 0
        }

        # Quality metrics
        self.quality_metrics = {
            'data_quality_score': 0.0,
            'validation_score': 0.0,
            'optimization_score': 0.0,
        }

        # Resource metrics
        self.resource_metrics = {
            'peak_memory_mb': 0.0,
            'avg_cpu_percent': 0.0,
            'disk_io_operations': 0,
            'network_io_bytes': 0
        }

        # Cache metrics
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0
        }

        # Initialize monitoring
        self._initialize_monitoring()

        tprint_debug(
            f"[PerformanceMonitor] Monitoring initialized at {self.start_time.isoformat()}"
        )

    def _initialize_monitoring(self):
        """Initialize monitoring state."""
        self.start_time = datetime.now()
        self._record_system_metrics()

    def start_operation(self, operation_name: str) -> datetime:
        """
        Start monitoring an operation.

        Args:
            operation_name: Name of the operation to monitor

        Returns:
            Start timestamp
        """
        start_timestamp = datetime.now()

        # Record operation start
        self._record_metric(
            name=f"{operation_name}_start",
            value=1.0,
            metric_type=MetricType.TECHNICAL,
            level=MetricLevel.INFO,
            tags={'operation': operation_name, 'event': 'start'}
        )

        # Record system metrics at start
        self._record_system_metrics()

        return start_timestamp

    def end_operation(self, operation_name: str, start_timestamp: datetime, success: bool = True):
        """
        End monitoring an operation.

        Args:
            operation_name: Name of the operation being monitored
            start_timestamp: When the operation started
            success: Whether the operation completed successfully
        """
        end_timestamp = datetime.now()
        duration = (end_timestamp - start_timestamp).total_seconds()

        # Record operation duration
        self._record_metric(
            name=f"{operation_name}_duration",
            value=duration,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={'operation': operation_name, 'unit': 'seconds'}
        )

        # Record operation success/failure
        self._record_metric(
            name=f"{operation_name}_success",
            value=1.0 if success else 0.0,
            metric_type=MetricType.BUSINESS,
            level=MetricLevel.INFO if success else MetricLevel.WARNING,
            tags={'operation': operation_name, 'result': 'success' if success else 'failure'}
        )

        # Record system metrics at end
        self._record_system_metrics()

        # Update execution times tracking
        if operation_name not in self.performance_metrics['execution_times']:
            self.performance_metrics['execution_times'][operation_name] = []

        self.performance_metrics['execution_times'][operation_name].append(duration)

    def record_optimization_metrics(
        self,
        optimization_results: Dict[str, Any],
        data_quality_score: float = 0.0,
        validation_score: float = 0.0
    ):
        """
        Record optimization-specific metrics.

        Args:
            optimization_results: Results from optimization process
            data_quality_score: Score indicating data quality
            validation_score: Score indicating validation success
        """
        # Record data quality metrics
        self._record_metric(
            name="data_quality_score",
            value=data_quality_score,
            metric_type=MetricType.QUALITY,
            level=MetricLevel.INFO,
            tags={'component': 'data_quality'}
        )

        self.quality_metrics['data_quality_score'] = data_quality_score

        # Record validation metrics
        self._record_metric(
            name="validation_score",
            value=validation_score,
            metric_type=MetricType.QUALITY,
            level=MetricLevel.INFO,
            tags={'component': 'validation'}
        )

        self.quality_metrics['validation_score'] = validation_score

        # Record optimization-specific metrics
        if 'best_score' in optimization_results:
            self._record_metric(
                name="optimization_best_score",
                value=optimization_results['best_score'],
                metric_type=MetricType.BUSINESS,
                level=MetricLevel.INFO,
                tags={'component': 'optimization', 'metric': 'best_score'}
            )

        if 'total_trials' in optimization_results:
            self._record_metric(
                name="optimization_trials",
                value=optimization_results['total_trials'],
                metric_type=MetricType.TECHNICAL,
                level=MetricLevel.INFO,
                tags={'component': 'optimization', 'metric': 'trials'}
            )

        if 'optimization_time' in optimization_results:
            self._record_metric(
                name="optimization_time",
                value=optimization_results['optimization_time'],
                metric_type=MetricType.PERFORMANCE,
                level=MetricLevel.INFO,
                tags={'component': 'optimization', 'metric': 'time'}
            )

    def record_error(self, operation_name: str, error_type: str, error_message: str):
        """Record an error occurrence."""
        self.performance_metrics['error_counts'] += 1

        self._record_metric(
            name="error_occurrence",
            value=1.0,
            metric_type=MetricType.TECHNICAL,
            level=MetricLevel.ERROR,
            tags={
                'operation': operation_name,
                'error_type': error_type,
                'error_message': error_message[:100]  # Truncate long messages
            }
        )

    def record_warning(self, operation_name: str, warning_message: str):
        """Record a warning occurrence."""
        self.performance_metrics['warning_counts'] += 1

        self._record_metric(
            name="warning_occurrence",
            value=1.0,
            metric_type=MetricType.TECHNICAL,
            level=MetricLevel.WARNING,
            tags={
                'operation': operation_name,
                'warning_message': warning_message[:100]  # Truncate long messages
            }
        )

    def _record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        level: MetricLevel,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a single metric point."""
        try:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metric_type=metric_type,
                level=level,
                tags=tags or {},
                metadata=metadata or {}
            )

            self.metrics.append(metric_point)

            # Keep only recent metrics (limit to prevent memory issues)
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-5000:]

            tprint_debug(
                f"[PerformanceMonitor] Recorded metric '{name}' with value={value}"
                f" type={metric_type.value} level={level.value}"
            )

        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
            tprint_error(f"[PerformanceMonitor] Failed to record metric {name}: {e}")

    def _record_system_metrics(self):
        """Record current system resource usage."""
        try:
            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            self._record_metric(
                name="memory_usage_mb",
                value=memory_mb,
                metric_type=MetricType.RESOURCE,
                level=MetricLevel.INFO,
                tags={'resource': 'memory'}
            )

            # Update peak memory tracking
            if memory_mb > self.resource_metrics['peak_memory_mb']:
                self.resource_metrics['peak_memory_mb'] = memory_mb

            # CPU usage
            cpu_percent = process.cpu_percent()

            self._record_metric(
                name="cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.RESOURCE,
                level=MetricLevel.INFO,
                tags={'resource': 'cpu'}
            )

            # Store in performance metrics
            self.performance_metrics['memory_usage'].append(memory_mb)
            self.performance_metrics['cpu_usage'].append(cpu_percent)

            # Keep only recent system metrics
            if len(self.performance_metrics['memory_usage']) > 100:
                self.performance_metrics['memory_usage'] = self.performance_metrics['memory_usage'][-50:]
                self.performance_metrics['cpu_usage'] = self.performance_metrics['cpu_usage'][-50:]

        except Exception as e:
            self.logger.debug(f"Failed to record system metrics: {e}")

    def record_cache_event(self, event_type: str, cache_key: str, *, artifact_type: str = "features") -> None:
        """Track cache hits, misses, writes, and forced refreshes."""
        if not event_type:
            return

        event_type = event_type.lower()
        increment_map = {
            'hit': 'hits',
            'miss': 'misses',
            'write': 'writes',
            'force_refresh': 'force_refreshes',
        }

        metric_key = increment_map.get(event_type)
        if metric_key:
            self.cache_metrics[metric_key] = self.cache_metrics.get(metric_key, 0) + 1

        self._record_metric(
            name=f"cache_{event_type}",
            value=1.0,
            metric_type=MetricType.TECHNICAL,
            level=MetricLevel.INFO,
            tags={'cache_key': cache_key, 'artifact_type': artifact_type}
        )

    def update_cache_metrics(self, metrics: Dict[str, Any]) -> None:
        for key, value in (metrics or {}).items():
            if key in self.cache_metrics:
                self.cache_metrics[key] = value

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.start_time:
            return {}

        duration = datetime.now() - self.start_time
        total_seconds = duration.total_seconds()

        # Calculate averages
        memory_usage = self.performance_metrics['memory_usage']
        cpu_usage = self.performance_metrics['cpu_usage']

        avg_memory = np.mean(memory_usage) if memory_usage else 0.0
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0.0

        # Get operation statistics
        operation_stats = {}
        for operation_name, times in self.performance_metrics['execution_times'].items():
            if times:
                operation_stats[operation_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }

        return {
            'monitoring_duration_seconds': total_seconds,
            'total_operations': sum(len(times) for times in self.performance_metrics['execution_times'].values()),
            'error_count': self.performance_metrics['error_counts'],
            'warning_count': self.performance_metrics['warning_counts'],
            'avg_memory_usage_mb': avg_memory,
            'avg_cpu_usage_percent': avg_cpu,
            'peak_memory_usage_mb': self.resource_metrics['peak_memory_mb'],
            'operation_statistics': operation_stats,
            'quality_metrics': self.quality_metrics.copy(),
            'resource_metrics': self.resource_metrics.copy(),
            'cache_metrics': self.cache_metrics.copy(),
        }

    def get_metric_summary(self, metric_name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a specific metric."""
        metric_points = [m for m in self.metrics if m.name == metric_name]

        if not metric_points:
            return None

        values = [m.value for m in metric_points]
        timestamps = [m.timestamp for m in metric_points]

        # Calculate statistics
        count = len(values)
        min_value = min(values)
        max_value = max(values)
        mean_value = np.mean(values)
        median_value = np.median(values)
        std_value = np.std(values)
        last_value = values[-1]
        first_timestamp = min(timestamps)
        last_timestamp = max(timestamps)

        # Calculate level distribution
        level_distribution = {}
        for point in metric_points:
            level_key = point.level.value
            level_distribution[level_key] = level_distribution.get(level_key, 0) + 1

        return MetricSummary(
            name=metric_name,
            count=count,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            median_value=median_value,
            std_value=std_value,
            last_value=last_value,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            level_distribution=level_distribution
        )

    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricPoint]:
        """Get all metrics of a specific type."""
        return [m for m in self.metrics if m.metric_type == metric_type]

    def get_metrics_by_level(self, level: MetricLevel) -> List[MetricPoint]:
        """Get all metrics of a specific level."""
        return [m for m in self.metrics if m.level == level]

    def get_recent_metrics(self, minutes: int = 10) -> List[MetricPoint]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.timestamp >= cutoff_time]

    def export_metrics(self, format: str = 'dict') -> Any:
        """
        Export all metrics in specified format.

        Args:
            format: Export format ('dict', 'json', 'csv')

        Returns:
            Metrics in the requested format
        """
        if format == 'dict':
            export_payload = {
                'metrics': [self._metric_to_dict(m) for m in self.metrics],
                'performance_summary': self.get_performance_summary(),
                'metric_summaries': {
                    name: self._metric_summary_to_dict(summary)
                    for name, summary in [(str(m), self.get_metric_summary(str(m))) for m in set(mp.name for mp in self.metrics)]
                    if summary is not None
                }
            }
            tprint_performance(
                f"[PerformanceMonitor] Exported {len(export_payload['metrics'])} metrics as dictionary"
            )
            return export_payload
        elif format == 'json':
            dict_export = self.export_metrics('dict')
            json_export = self.serializer.serialize_json(dict_export)
            tprint_performance(
                f"[PerformanceMonitor] Exported metrics as JSON payload ({len(json_export)} characters)"
            )
            return json_export
        else:
            tprint_error(
                f"[PerformanceMonitor] Unsupported export format requested: {format}"
            )
            raise ValueError(f"Unsupported export format: {format}")

    def _metric_to_dict(self, metric: MetricPoint) -> Dict[str, Any]:
        """Convert metric point to dictionary."""
        return {
            'name': metric.name,
            'value': metric.value,
            'timestamp': metric.timestamp.isoformat(),
            'type': metric.metric_type.value,
            'level': metric.level.value,
            'tags': metric.tags,
            'metadata': metric.metadata
        }

    def _metric_summary_to_dict(self, summary: MetricSummary) -> Dict[str, Any]:
        """Convert metric summary to dictionary."""
        return {
            'name': summary.name,
            'count': summary.count,
            'min_value': summary.min_value,
            'max_value': summary.max_value,
            'mean_value': summary.mean_value,
            'median_value': summary.median_value,
            'std_value': summary.std_value,
            'last_value': summary.last_value,
            'first_timestamp': summary.first_timestamp.isoformat(),
            'last_timestamp': summary.last_timestamp.isoformat(),
            'level_distribution': summary.level_distribution
        }

    def reset_monitoring(self):
        """Reset all monitoring data."""
        self.metrics.clear()
        self.performance_metrics = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0,
            'warning_counts': 0
        }
        self.quality_metrics = {
            'data_quality_score': 0.0,
            'validation_score': 0.0,
            'optimization_score': 0.0,
        }
        self.resource_metrics = {
            'peak_memory_mb': 0.0,
            'avg_cpu_percent': 0.0,
            'disk_io_operations': 0,
            'network_io_bytes': 0
        }
        self.start_time = datetime.now()
        tprint(
            f"[PerformanceMonitor] Monitoring data reset at {self.start_time.isoformat()}"
        )
