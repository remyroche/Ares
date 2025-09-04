#!/usr/bin/env python3
"""
Optimisation Pipeline Monitoring System

Comprehensive monitoring and alerting system for the optimisation pipeline:
- Real-time performance monitoring
- Data quality monitoring
- Pipeline health monitoring
- Alerting and notification system
- Metrics collection and analysis
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

from src.utils.logger import system_logger
from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime
)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Union[int, float, str]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineHealthStatus:
    """Pipeline health status."""
    overall_health: str  # "healthy", "degraded", "unhealthy"
    component_status: Dict[str, str]
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class OptimisationMonitoringSystem:
    """Comprehensive monitoring system for the optimisation pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimisationMonitoringSystem")
        
        # Monitoring configuration
        self.monitoring_interval = config.get("monitoring_interval", 30)  # seconds
        self.alert_cooldown = config.get("alert_cooldown", 300)  # 5 minutes
        self.metrics_retention_days = config.get("metrics_retention_days", 30)
        self.health_check_interval = config.get("health_check_interval", 60)  # seconds
        
        # Storage
        self.metrics_storage: List[Metric] = []
        self.alerts_storage: List[Alert] = []
        self.health_history: List[PipelineHealthStatus] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.alert_queue = queue.Queue()
        
        # Thresholds
        self.thresholds = {
            "execution_time": 3600,  # 1 hour
            "memory_usage": 1000,    # 1GB
            "error_rate": 0.1,       # 10%
            "data_quality_score": 0.8,  # 80%
            "pipeline_success_rate": 0.9  # 90%
        }
        
        # Initialize storage directories
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories."""
        try:
            self.metrics_dir = Path(self.config.get("metrics_dir", "data_cache/monitoring/metrics"))
            self.alerts_dir = Path(self.config.get("alerts_dir", "data_cache/monitoring/alerts"))
            self.health_dir = Path(self.config.get("health_dir", "data_cache/monitoring/health"))
            
            ensure_directory(self.metrics_dir)
            ensure_directory(self.alerts_dir)
            ensure_directory(self.health_dir)
            
            self.logger.info("✅ Monitoring storage directories initialized")
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize monitoring storage: {e}")
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        try:
            if self.is_monitoring:
                self.logger.warning("⚠️ Monitoring system is already running")
                return
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("🚀 Monitoring system started")
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to start monitoring system: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        try:
            self.is_monitoring = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Save final state
            self._save_monitoring_state()
            
            self.logger.info("🛑 Monitoring system stopped")
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to stop monitoring system: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.is_monitoring:
                # Perform health checks
                health_status = self._check_pipeline_health()
                self.health_history.append(health_status)
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Save monitoring state
                self._save_monitoring_state()
                
                # Wait for next check
                time.sleep(self.monitoring_interval)
                
        except Exception as e:
            self.logger.exception(f"❌ Monitoring loop error: {e}")
    
    def record_metric(self, 
                     name: str,
                     value: Union[int, float, str],
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=get_current_datetime(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics_storage.append(metric)
            
            # Check if metric triggers alerts
            self._check_metric_alerts(metric)
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to record metric {name}: {e}")
    
    def create_alert(self, 
                    alert_type: str,
                    severity: AlertSeverity,
                    message: str,
                    source: str,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create an alert."""
        try:
            alert_id = f"{alert_type}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                type=alert_type,
                severity=severity,
                message=message,
                timestamp=get_current_datetime(),
                source=source,
                metadata=metadata or {}
            )
            
            self.alerts_storage.append(alert)
            
            # Send alert if severity is high enough
            if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                self._send_alert(alert)
            
            self.logger.info(f"🚨 Alert created: {alert_type} - {message}")
            return alert_id
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to create alert: {e}")
            return ""
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            for alert in self.alerts_storage:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = get_current_datetime()
                    self.logger.info(f"✅ Alert resolved: {alert_id}")
                    return True
            
            self.logger.warning(f"⚠️ Alert not found or already resolved: {alert_id}")
            return False
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to resolve alert {alert_id}: {e}")
            return False
    
    def _check_pipeline_health(self) -> PipelineHealthStatus:
        """Check overall pipeline health."""
        try:
            component_status = {}
            issues = []
            recommendations = []
            
            # Check data quality
            data_quality_score = self._check_data_quality()
            component_status["data_quality"] = "healthy" if data_quality_score > 0.8 else "degraded"
            if data_quality_score < 0.8:
                issues.append(f"Data quality score is low: {data_quality_score:.2f}")
                recommendations.append("Review data preprocessing and validation steps")
            
            # Check performance metrics
            performance_score = self._check_performance_metrics()
            component_status["performance"] = "healthy" if performance_score > 0.8 else "degraded"
            if performance_score < 0.8:
                issues.append(f"Performance score is low: {performance_score:.2f}")
                recommendations.append("Optimize pipeline performance and resource usage")
            
            # Check error rates
            error_rate = self._check_error_rates()
            component_status["error_handling"] = "healthy" if error_rate < 0.1 else "degraded"
            if error_rate > 0.1:
                issues.append(f"High error rate: {error_rate:.2f}")
                recommendations.append("Review error handling and improve robustness")
            
            # Check resource usage
            resource_usage = self._check_resource_usage()
            component_status["resource_usage"] = "healthy" if resource_usage < 0.8 else "degraded"
            if resource_usage > 0.8:
                issues.append(f"High resource usage: {resource_usage:.2f}")
                recommendations.append("Optimize resource usage and consider scaling")
            
            # Determine overall health
            if not issues:
                overall_health = "healthy"
            elif len(issues) <= 2:
                overall_health = "degraded"
            else:
                overall_health = "unhealthy"
            
            health_status = PipelineHealthStatus(
                overall_health=overall_health,
                component_status=component_status,
                last_check=get_current_datetime(),
                issues=issues,
                recommendations=recommendations
            )
            
            return health_status
            
        except Exception as e:
            self.logger.exception(f"❌ Health check failed: {e}")
            return PipelineHealthStatus(
                overall_health="unhealthy",
                component_status={},
                last_check=get_current_datetime(),
                issues=[f"Health check failed: {str(e)}"],
                recommendations=["Investigate monitoring system issues"]
            )
    
    def _check_data_quality(self) -> float:
        """Check data quality score."""
        try:
            # Get recent data quality metrics
            recent_metrics = [
                m for m in self.metrics_storage
                if m.name.startswith("data_quality") and 
                (get_current_datetime() - m.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if not recent_metrics:
                return 1.0  # Assume good if no recent metrics
            
            # Calculate average data quality score
            quality_scores = [float(m.value) for m in recent_metrics if isinstance(m.value, (int, float))]
            return np.mean(quality_scores) if quality_scores else 1.0
            
        except Exception as e:
            self.logger.exception(f"❌ Data quality check failed: {e}")
            return 0.0
    
    def _check_performance_metrics(self) -> float:
        """Check performance score."""
        try:
            # Get recent performance metrics
            recent_metrics = [
                m for m in self.metrics_storage
                if m.name.startswith("performance") and 
                (get_current_datetime() - m.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if not recent_metrics:
                return 1.0  # Assume good if no recent metrics
            
            # Calculate performance score based on execution times
            execution_times = [
                float(m.value) for m in recent_metrics 
                if m.name == "execution_time" and isinstance(m.value, (int, float))
            ]
            
            if not execution_times:
                return 1.0
            
            # Score based on execution time (lower is better)
            avg_execution_time = np.mean(execution_times)
            max_allowed_time = self.thresholds["execution_time"]
            
            return max(0.0, 1.0 - (avg_execution_time / max_allowed_time))
            
        except Exception as e:
            self.logger.exception(f"❌ Performance check failed: {e}")
            return 0.0
    
    def _check_error_rates(self) -> float:
        """Check error rates."""
        try:
            # Get recent error metrics
            recent_metrics = [
                m for m in self.metrics_storage
                if m.name.startswith("error") and 
                (get_current_datetime() - m.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if not recent_metrics:
                return 0.0  # No errors if no recent metrics
            
            # Calculate error rate
            total_operations = len([m for m in recent_metrics if m.name == "operation_count"])
            error_count = len([m for m in recent_metrics if m.name == "error_count"])
            
            if total_operations == 0:
                return 0.0
            
            return error_count / total_operations
            
        except Exception as e:
            self.logger.exception(f"❌ Error rate check failed: {e}")
            return 1.0  # Assume high error rate if check fails
    
    def _check_resource_usage(self) -> float:
        """Check resource usage."""
        try:
            # Get recent resource metrics
            recent_metrics = [
                m for m in self.metrics_storage
                if m.name.startswith("resource") and 
                (get_current_datetime() - m.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if not recent_metrics:
                return 0.0  # Assume low usage if no recent metrics
            
            # Calculate average resource usage
            memory_usage = [
                float(m.value) for m in recent_metrics 
                if m.name == "memory_usage" and isinstance(m.value, (int, float))
            ]
            
            if not memory_usage:
                return 0.0
            
            avg_memory_usage = np.mean(memory_usage)
            max_allowed_memory = self.thresholds["memory_usage"]
            
            return min(1.0, avg_memory_usage / max_allowed_memory)
            
        except Exception as e:
            self.logger.exception(f"❌ Resource usage check failed: {e}")
            return 1.0  # Assume high usage if check fails
    
    def _check_metric_alerts(self, metric: Metric) -> None:
        """Check if a metric triggers alerts."""
        try:
            # Check execution time alerts
            if metric.name == "execution_time" and isinstance(metric.value, (int, float)):
                if metric.value > self.thresholds["execution_time"]:
                    self.create_alert(
                        "high_execution_time",
                        AlertSeverity.WARNING,
                        f"Execution time exceeded threshold: {metric.value:.2f}s > {self.thresholds['execution_time']}s",
                        "performance_monitor",
                        {"execution_time": metric.value, "threshold": self.thresholds["execution_time"]}
                    )
            
            # Check memory usage alerts
            elif metric.name == "memory_usage" and isinstance(metric.value, (int, float)):
                if metric.value > self.thresholds["memory_usage"]:
                    self.create_alert(
                        "high_memory_usage",
                        AlertSeverity.WARNING,
                        f"Memory usage exceeded threshold: {metric.value:.2f}MB > {self.thresholds['memory_usage']}MB",
                        "resource_monitor",
                        {"memory_usage": metric.value, "threshold": self.thresholds["memory_usage"]}
                    )
            
            # Check error rate alerts
            elif metric.name == "error_rate" and isinstance(metric.value, (int, float)):
                if metric.value > self.thresholds["error_rate"]:
                    self.create_alert(
                        "high_error_rate",
                        AlertSeverity.ERROR,
                        f"Error rate exceeded threshold: {metric.value:.2f} > {self.thresholds['error_rate']}",
                        "error_monitor",
                        {"error_rate": metric.value, "threshold": self.thresholds["error_rate"]}
                    )
            
            # Check data quality alerts
            elif metric.name == "data_quality_score" and isinstance(metric.value, (int, float)):
                if metric.value < self.thresholds["data_quality_score"]:
                    self.create_alert(
                        "low_data_quality",
                        AlertSeverity.WARNING,
                        f"Data quality score below threshold: {metric.value:.2f} < {self.thresholds['data_quality_score']}",
                        "data_quality_monitor",
                        {"data_quality_score": metric.value, "threshold": self.thresholds["data_quality_score"]}
                    )
            
        except Exception as e:
            self.logger.exception(f"❌ Metric alert check failed: {e}")
    
    def _check_alert_conditions(self) -> None:
        """Check for alert conditions."""
        try:
            # Check for unresolved critical alerts
            critical_alerts = [
                alert for alert in self.alerts_storage
                if alert.severity == AlertSeverity.CRITICAL and not alert.resolved
            ]
            
            if len(critical_alerts) > 3:
                self.create_alert(
                    "multiple_critical_alerts",
                    AlertSeverity.CRITICAL,
                    f"Multiple critical alerts unresolved: {len(critical_alerts)}",
                    "alert_monitor",
                    {"critical_alert_count": len(critical_alerts)}
                )
            
            # Check for alert frequency
            recent_alerts = [
                alert for alert in self.alerts_storage
                if (get_current_datetime() - alert.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            if len(recent_alerts) > 10:
                self.create_alert(
                    "high_alert_frequency",
                    AlertSeverity.WARNING,
                    f"High alert frequency: {len(recent_alerts)} alerts in last 5 minutes",
                    "alert_monitor",
                    {"recent_alert_count": len(recent_alerts)}
                )
            
        except Exception as e:
            self.logger.exception(f"❌ Alert condition check failed: {e}")
    
    def _send_alert(self, alert: Alert) -> None:
        """Send alert notification."""
        try:
            # For now, just log the alert
            # In a real implementation, this would send emails, Slack messages, etc.
            self.logger.critical(f"🚨 CRITICAL ALERT: {alert.type} - {alert.message}")
            
            # Save alert to file for external processing
            alert_file = self.alerts_dir / f"alert_{alert.id}.json"
            alert_data = {
                "id": alert.id,
                "type": alert.type,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata
            }
            safe_json_dump(alert_data, alert_file)
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to send alert: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        try:
            cutoff_time = get_current_datetime() - timedelta(days=self.metrics_retention_days)
            
            # Clean up old metrics
            self.metrics_storage = [
                m for m in self.metrics_storage
                if m.timestamp > cutoff_time
            ]
            
            # Clean up old health history
            self.health_history = [
                h for h in self.health_history
                if h.last_check > cutoff_time
            ]
            
            # Clean up old alert files
            for alert_file in self.alerts_dir.glob("alert_*.json"):
                if alert_file.stat().st_mtime < cutoff_time.timestamp():
                    alert_file.unlink()
            
        except Exception as e:
            self.logger.exception(f"❌ Cleanup failed: {e}")
    
    def _save_monitoring_state(self) -> None:
        """Save monitoring state to disk."""
        try:
            # Save metrics
            metrics_file = self.metrics_dir / f"metrics_{get_current_datetime().strftime('%Y%m%d')}.json"
            metrics_data = [
                {
                    "name": m.name,
                    "value": m.value,
                    "metric_type": m.metric_type.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "metadata": m.metadata
                }
                for m in self.metrics_storage[-1000:]  # Keep last 1000 metrics
            ]
            safe_json_dump(metrics_data, metrics_file)
            
            # Save alerts
            alerts_file = self.alerts_dir / f"alerts_{get_current_datetime().strftime('%Y%m%d')}.json"
            alerts_data = [
                {
                    "id": a.id,
                    "type": a.type,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "source": a.source,
                    "metadata": a.metadata,
                    "resolved": a.resolved,
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None
                }
                for a in self.alerts_storage[-100:]  # Keep last 100 alerts
            ]
            safe_json_dump(alerts_data, alerts_file)
            
            # Save health status
            if self.health_history:
                latest_health = self.health_history[-1]
                health_file = self.health_dir / f"health_{get_current_datetime().strftime('%Y%m%d_%H%M')}.json"
                health_data = {
                    "overall_health": latest_health.overall_health,
                    "component_status": latest_health.component_status,
                    "last_check": latest_health.last_check.isoformat(),
                    "issues": latest_health.issues,
                    "recommendations": latest_health.recommendations
                }
                safe_json_dump(health_data, health_file)
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to save monitoring state: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        try:
            # Calculate summary statistics
            total_metrics = len(self.metrics_storage)
            total_alerts = len(self.alerts_storage)
            unresolved_alerts = len([a for a in self.alerts_storage if not a.resolved])
            
            # Get latest health status
            latest_health = self.health_history[-1] if self.health_history else None
            
            # Calculate recent activity
            recent_metrics = len([
                m for m in self.metrics_storage
                if (get_current_datetime() - m.timestamp).total_seconds() < 3600
            ])
            
            recent_alerts = len([
                a for a in self.alerts_storage
                if (get_current_datetime() - a.timestamp).total_seconds() < 3600
            ])
            
            summary = {
                "monitoring_status": "active" if self.is_monitoring else "inactive",
                "total_metrics": total_metrics,
                "total_alerts": total_alerts,
                "unresolved_alerts": unresolved_alerts,
                "recent_metrics": recent_metrics,
                "recent_alerts": recent_alerts,
                "latest_health": {
                    "overall_health": latest_health.overall_health if latest_health else "unknown",
                    "component_status": latest_health.component_status if latest_health else {},
                    "issues": latest_health.issues if latest_health else [],
                    "recommendations": latest_health.recommendations if latest_health else []
                },
                "thresholds": self.thresholds,
                "last_updated": get_current_datetime().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to get monitoring summary: {e}")
            return {"error": str(e)}


# Global monitoring system instance
_monitoring_system: Optional[OptimisationMonitoringSystem] = None


def initialize_monitoring_system(config: Dict[str, Any]) -> OptimisationMonitoringSystem:
    """Initialize the monitoring system."""
    global _monitoring_system
    
    _monitoring_system = OptimisationMonitoringSystem(config)
    _monitoring_system.start_monitoring()
    
    system_logger.info("🚀 Optimisation monitoring system initialized")
    return _monitoring_system


def get_monitoring_system() -> OptimisationMonitoringSystem:
    """Get the monitoring system instance."""
    if _monitoring_system is None:
        raise RuntimeError("Monitoring system not initialized. Call initialize_monitoring_system() first.")
    return _monitoring_system


# Convenience functions
def record_metric(name: str, value: Union[int, float, str], **kwargs) -> None:
    """Record a metric."""
    monitoring_system = get_monitoring_system()
    monitoring_system.record_metric(name, value, **kwargs)


def create_alert(alert_type: str, severity: AlertSeverity, message: str, **kwargs) -> str:
    """Create an alert."""
    monitoring_system = get_monitoring_system()
    return monitoring_system.create_alert(alert_type, severity, message, **kwargs)


def get_monitoring_summary() -> Dict[str, Any]:
    """Get monitoring summary."""
    monitoring_system = get_monitoring_system()
    return monitoring_system.get_monitoring_summary()