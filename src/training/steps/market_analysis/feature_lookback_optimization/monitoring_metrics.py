"""
Comprehensive Monitoring Metrics for Feature Lookback Optimization.

This module provides detailed monitoring and performance metrics collection
throughout the optimization process with real-time tracking and analysis.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .dependency_manager import get_dependency
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Hardware optimization imports
try:
    from src.utils.hardware import get_advanced_memory_optimizer, get_unified_hardware_manager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

logger = system_logger.getChild('MonitoringMetrics')

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
    value: Union[float, int, str, bool]
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

class MonitoringMetrics:
    """
    Comprehensive monitoring metrics collector for feature lookback optimization.
    
    Provides real-time metrics collection, analysis, and alerting capabilities
    for performance monitoring and optimization tracking.
    """
    
    def __init__(self, component_name: str = "FeatureLookbackOptimization"):
        """Initialize the monitoring metrics collector."""
        self.logger = logger.getChild('MonitoringMetrics')
        self.component_name = component_name
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
            'stability_score': 0.0
        }
        
        # Business metrics
        self.business_metrics = {
            'features_optimized': 0,
            'optimization_success_rate': 0.0,
            'time_to_optimization': 0.0,
            'cost_efficiency': 0.0
        }
        
        # Technical metrics
        self.technical_metrics = {
            'dependencies_available': 0,
            'fallbacks_used': 0,
            'validation_rules_passed': 0,
            'validation_rules_failed': 0
        }
        
        # Alerting thresholds
        self.thresholds = {
            'memory_usage_mb': 1000,
            'cpu_usage_percent': 80,
            'execution_time_seconds': 300,
            'error_rate_percent': 10,
            'data_quality_min': 0.7,
            'optimization_score_min': 0.5
        }
        
        # Initialize system monitoring
        self._initialize_system_monitoring()
        
        # Initialize hardware optimization if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_advanced_memory_optimizer()
                self.hardware_manager = get_unified_hardware_manager()
                tprint("✅ Hardware optimization initialized for monitoring")
            except Exception as e:
                tprint(f"⚠️ Hardware optimization initialization failed: {e}")
                self.memory_optimizer = None
                self.hardware_manager = None
        else:
            self.memory_optimizer = None
            self.hardware_manager = None
        
        # Memory cleanup settings
        self.max_metrics_memory = 10000  # Maximum number of metrics to keep in memory
        self.cleanup_interval = 1000  # Cleanup every 1000 metrics
    
    def _initialize_system_monitoring(self) -> None:
        """Initialize system resource monitoring."""
        try:
            psutil, is_fallback = get_dependency('psutil')
            if psutil is not None:
                self.psutil = psutil
                self.psutil_available = True
                self.psutil_fallback = is_fallback
                
                if is_fallback:
                    tprint("Using fallback psutil for system monitoring")
            else:
                self.psutil_available = False
                tprint("⚠️ psutil not available - limited system monitoring")
                
        except Exception as e:
            self.psutil_available = False
            tprint(f"⚠️ Failed to initialize system monitoring: {e}")
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory leaks."""
        if len(self.metrics) > self.max_metrics_memory:
            # Keep only the most recent metrics
            self.metrics = self.metrics[-self.max_metrics_memory:]
            tprint(f"🧹 Cleaned up old metrics, keeping {len(self.metrics)} most recent")
            
            # Also cleanup performance metrics
            if len(self.performance_metrics['memory_usage']) > 1000:
                self.performance_metrics['memory_usage'] = self.performance_metrics['memory_usage'][-1000:]
            if len(self.performance_metrics['cpu_usage']) > 1000:
                self.performance_metrics['cpu_usage'] = self.performance_metrics['cpu_usage'][-1000:]
    
    def start_monitoring(self) -> None:
        """Start monitoring session."""
        self.start_time = datetime.now()
        tprint(f"🔍 Starting monitoring for {self.component_name}")
        
        # Record start metrics
        self.record_metric(
            name="monitoring_started",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={"component": self.component_name}
        )
        
        # Record initial system state
        self._record_system_metrics()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring session."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0.0
        
        tprint(f"🔍 Stopping monitoring for {self.component_name} (duration: {duration:.2f}s)")
        
        # Record end metrics
        self.record_metric(
            name="monitoring_stopped",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={"component": self.component_name, "duration_seconds": duration}
        )
        
        # Record final system state
        self._record_system_metrics()
        
        # Generate summary
        self._generate_monitoring_summary()
        
        # Final cleanup
        self._cleanup_old_metrics()
    
    def record_metric(
        self,
        name: str,
        value: Union[float, int, str, bool],
        metric_type: MetricType,
        level: MetricLevel = MetricLevel.INFO,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric point."""
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
        
        # Update aggregated metrics
        self._update_aggregated_metrics(metric_point)
        
        # Check for alerts
        self._check_alerts(metric_point)
        
        # Log significant metrics
        if level in [MetricLevel.WARNING, MetricLevel.ERROR, MetricLevel.CRITICAL]:
            tprint(f"📊 {name}: {value} ({metric_type.value})")
        
        # Periodic cleanup
        if len(self.metrics) % self.cleanup_interval == 0:
            self._cleanup_old_metrics()
    
    def record_performance_metric(self, operation_name: str, duration: float) -> None:
        """Record performance metric for an operation."""
        self.record_metric(
            name=f"operation_{operation_name}_duration",
            value=duration,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={"operation": operation_name},
            metadata={"duration_seconds": duration}
        )
        
        # Update performance tracking
        if operation_name not in self.performance_metrics['execution_times']:
            self.performance_metrics['execution_times'][operation_name] = []
        self.performance_metrics['execution_times'][operation_name].append(duration)
    
    def record_quality_metric(self, metric_name: str, score: float) -> None:
        """Record quality metric."""
        level = MetricLevel.INFO
        if score < self.thresholds.get('data_quality_min', 0.7):
            level = MetricLevel.WARNING
        if score < 0.5:
            level = MetricLevel.ERROR
        
        self.record_metric(
            name=f"quality_{metric_name}",
            value=score,
            metric_type=MetricType.QUALITY,
            level=level,
            tags={"quality_metric": metric_name},
            metadata={"score": score}
        )
        
        # Update quality metrics
        self.quality_metrics[metric_name] = score
    
    def record_business_metric(self, metric_name: str, value: Union[float, int]) -> None:
        """Record business metric."""
        self.record_metric(
            name=f"business_{metric_name}",
            value=value,
            metric_type=MetricType.BUSINESS,
            level=MetricLevel.INFO,
            tags={"business_metric": metric_name},
            metadata={"value": value}
        )
        
        # Update business metrics
        self.business_metrics[metric_name] = value
    
    def record_technical_metric(self, metric_name: str, value: Union[float, int, bool]) -> None:
        """Record technical metric."""
        self.record_metric(
            name=f"technical_{metric_name}",
            value=value,
            metric_type=MetricType.TECHNICAL,
            level=MetricLevel.INFO,
            tags={"technical_metric": metric_name},
            metadata={"value": value}
        )
        
        # Update technical metrics
        self.technical_metrics[metric_name] = value
    
    def record_error(self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record an error metric."""
        self.record_metric(
            name=f"error_{error_type}",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.ERROR,
            tags={"error_type": error_type},
            metadata={
                "error_message": error_message,
                "context": context or {}
            }
        )
        
        self.performance_metrics['error_counts'] += 1
    
    def record_warning(self, warning_type: str, warning_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a warning metric."""
        self.record_metric(
            name=f"warning_{warning_type}",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.WARNING,
            tags={"warning_type": warning_type},
            metadata={
                "warning_message": warning_message,
                "context": context or {}
            }
        )
        
        self.performance_metrics['warning_counts'] += 1
    
    def _record_system_metrics(self) -> None:
        """Record current system metrics."""
        if not self.psutil_available:
            return
        
        try:
            process = self.psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.record_metric(
                name="system_memory_usage",
                value=memory_mb,
                metric_type=MetricType.RESOURCE,
                level=MetricLevel.INFO,
                tags={"resource": "memory"},
                metadata={"memory_mb": memory_mb}
            )
            
            self.performance_metrics['memory_usage'].append(memory_mb)
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            self.record_metric(
                name="system_cpu_usage",
                value=cpu_percent,
                metric_type=MetricType.RESOURCE,
                level=MetricLevel.INFO,
                tags={"resource": "cpu"},
                metadata={"cpu_percent": cpu_percent}
            )
            
            self.performance_metrics['cpu_usage'].append(cpu_percent)
            
        except Exception as e:
            tprint(f"⚠️ Failed to record system metrics: {e}")
    
    def _update_aggregated_metrics(self, metric_point: MetricPoint) -> None:
        """Update aggregated metrics based on new metric point."""
        # Update performance metrics
        if metric_point.metric_type == MetricType.PERFORMANCE:
            if "duration" in metric_point.name:
                # This is handled in record_performance_metric
                pass
            elif metric_point.level == MetricLevel.ERROR:
                self.performance_metrics['error_counts'] += 1
            elif metric_point.level == MetricLevel.WARNING:
                self.performance_metrics['warning_counts'] += 1
        
        # Update quality metrics
        elif metric_point.metric_type == MetricType.QUALITY:
            if metric_point.name.startswith("quality_"):
                metric_name = metric_point.name.replace("quality_", "")
                self.quality_metrics[metric_name] = metric_point.value
        
        # Update business metrics
        elif metric_point.metric_type == MetricType.BUSINESS:
            if metric_point.name.startswith("business_"):
                metric_name = metric_point.name.replace("business_", "")
                self.business_metrics[metric_name] = metric_point.value
        
        # Update technical metrics
        elif metric_point.metric_type == MetricType.TECHNICAL:
            if metric_point.name.startswith("technical_"):
                metric_name = metric_point.name.replace("technical_", "")
                self.technical_metrics[metric_name] = metric_point.value
    
    def _check_alerts(self, metric_point: MetricPoint) -> None:
        """Check if metric point triggers any alerts."""
        alerts = []
        
        # Memory usage alert
        if (metric_point.name == "system_memory_usage" and 
            metric_point.value > self.thresholds['memory_usage_mb']):
            alerts.append(f"High memory usage: {metric_point.value:.1f}MB")
        
        # CPU usage alert
        if (metric_point.name == "system_cpu_usage" and 
            metric_point.value > self.thresholds['cpu_usage_percent']):
            alerts.append(f"High CPU usage: {metric_point.value:.1f}%")
        
        # Data quality alert
        if (metric_point.metric_type == MetricType.QUALITY and 
            metric_point.value < self.thresholds['data_quality_min']):
            alerts.append(f"Low data quality: {metric_point.value:.3f}")
        
        # Log alerts
        for alert in alerts:
            tprint(f"🚨 ALERT: {alert}")
    
    def _generate_monitoring_summary(self) -> None:
        """Generate comprehensive monitoring summary."""
        if not self.metrics:
            return
        
        summary = {
            'monitoring_session': {
                'component_name': self.component_name,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0.0,
                'total_metrics': len(self.metrics)
            },
            'performance_summary': {
                'total_operations': len(self.performance_metrics['execution_times']),
                'total_errors': self.performance_metrics['error_counts'],
                'total_warnings': self.performance_metrics['warning_counts'],
                'avg_memory_usage': sum(self.performance_metrics['memory_usage']) / len(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
                'max_memory_usage': max(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
                'avg_cpu_usage': sum(self.performance_metrics['cpu_usage']) / len(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0,
                'max_cpu_usage': max(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0
            },
            'quality_summary': self.quality_metrics,
            'business_summary': self.business_metrics,
            'technical_summary': self.technical_metrics,
            'metric_distribution': self._get_metric_distribution(),
            'alerts_triggered': self._get_alerts_summary()
        }
        
        # Record summary as metric
        self.record_metric(
            name="monitoring_summary",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            metadata=summary
        )
        
        tprint("📊 Monitoring summary generated")
    
    def _get_metric_distribution(self) -> Dict[str, int]:
        """Get distribution of metrics by type and level."""
        distribution = {
            'by_type': {},
            'by_level': {}
        }
        
        for metric in self.metrics:
            # By type
            metric_type = metric.metric_type.value
            distribution['by_type'][metric_type] = distribution['by_type'].get(metric_type, 0) + 1
            
            # By level
            level = metric.level.value
            distribution['by_level'][level] = distribution['by_level'].get(level, 0) + 1
        
        return distribution
    
    def _get_alerts_summary(self) -> List[str]:
        """Get summary of alerts triggered."""
        alerts = []
        
        # Check thresholds
        if self.performance_metrics['memory_usage']:
            max_memory = max(self.performance_metrics['memory_usage'])
            if max_memory > self.thresholds['memory_usage_mb']:
                alerts.append(f"Memory usage exceeded {self.thresholds['memory_usage_mb']}MB (max: {max_memory:.1f}MB)")
        
        if self.performance_metrics['cpu_usage']:
            max_cpu = max(self.performance_metrics['cpu_usage'])
            if max_cpu > self.thresholds['cpu_usage_percent']:
                alerts.append(f"CPU usage exceeded {self.thresholds['cpu_usage_percent']}% (max: {max_cpu:.1f}%)")
        
        if self.performance_metrics['error_counts'] > 0:
            alerts.append(f"Total errors: {self.performance_metrics['error_counts']}")
        
        if self.performance_metrics['warning_counts'] > 0:
            alerts.append(f"Total warnings: {self.performance_metrics['warning_counts']}")
        
        return alerts
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.metrics:
            return {'error': 'No metrics recorded'}
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in self.metrics:
            if metric.name not in metrics_by_name:
                metrics_by_name[metric.name] = []
            metrics_by_name[metric.name].append(metric)
        
        # Generate summaries for each metric
        summaries = {}
        for name, metrics in metrics_by_name.items():
            if all(isinstance(m.value, (int, float)) for m in metrics):
                values = [m.value for m in metrics]
                summaries[name] = MetricSummary(
                    name=name,
                    count=len(values),
                    min_value=min(values),
                    max_value=max(values),
                    mean_value=sum(values) / len(values),
                    median_value=sorted(values)[len(values) // 2],
                    std_value=self._calculate_std(values),
                    last_value=values[-1],
                    first_timestamp=min(m.timestamp for m in metrics),
                    last_timestamp=max(m.timestamp for m in metrics),
                    level_distribution={level.value: sum(1 for m in metrics if m.level == level) for level in MetricLevel}
                )
        
        return {
            'total_metrics': len(self.metrics),
            'unique_metric_names': len(metrics_by_name),
            'summaries': summaries,
            'performance_metrics': self.performance_metrics,
            'quality_metrics': self.quality_metrics,
            'business_metrics': self.business_metrics,
            'technical_metrics': self.technical_metrics
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def export_metrics(self, file_path: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"metrics_{self.component_name}_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'component_name': self.component_name,
                'export_timestamp': datetime.now().isoformat(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None
            },
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'timestamp': m.timestamp.isoformat(),
                    'metric_type': m.metric_type.value,
                    'level': m.level.value,
                    'tags': m.tags,
                    'metadata': m.metadata
                }
                for m in self.metrics
            ],
            'summary': self.get_metrics_summary()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            tprint(f"📊 Metrics exported to: {file_path}")
            return file_path
        except Exception as e:
            tprint(f"❌ Failed to export metrics: {e}")
            return ""
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return {
            'execution_times': self.performance_metrics['execution_times'],
            'memory_usage': {
                'values': self.performance_metrics['memory_usage'],
                'average': sum(self.performance_metrics['memory_usage']) / len(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
                'maximum': max(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
                'minimum': min(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0
            },
            'cpu_usage': {
                'values': self.performance_metrics['cpu_usage'],
                'average': sum(self.performance_metrics['cpu_usage']) / len(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0,
                'maximum': max(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0,
                'minimum': min(self.performance_metrics['cpu_usage']) if self.performance_metrics['cpu_usage'] else 0
            },
            'error_count': self.performance_metrics['error_counts'],
            'warning_count': self.performance_metrics['warning_counts']
        }