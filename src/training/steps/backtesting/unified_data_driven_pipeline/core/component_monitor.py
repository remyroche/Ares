"""
Component Monitor for Backtesting Pipeline

This module provides real-time monitoring of backtesting component health and
performance. It includes component status tracking, performance metrics visualization,
health alerts, dependency graphs, historical analysis, and strategy performance tracking.

Key Features:
- Real-time component status monitoring
- Performance metrics visualization
- Health alerts and notifications
- Component dependency graphs
- Historical performance analysis
- Strategy performance tracking
- Backtesting-specific monitoring
"""

import time
import logging
import threading
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from .modular_architecture import ModularComponent, ErrorSeverity, ErrorCategory
from .component_registry import BacktestingComponentRegistry, ComponentStatus, ComponentType, get_registry
from .component_orchestrator import BacktestingWorkflowOrchestrator, WorkflowStatus, get_orchestrator


class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert information."""
    id: str
    component_name: str
    level: AlertLevel
    message: str
    timestamp: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    component_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Component health information."""
    component_name: str
    status: ComponentStatus
    health_score: float
    last_updated: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_real_time_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_health_monitoring: bool = True
    enable_alerting: bool = True
    enable_visualization: bool = True
    monitoring_interval: float = 10.0  # seconds
    health_check_interval: float = 60.0  # seconds
    alert_retention_days: int = 30
    performance_history_size: int = 1000
    health_history_size: int = 100
    enable_strategy_tracking: bool = True
    enable_portfolio_monitoring: bool = True
    enable_risk_monitoring: bool = True


class BacktestingComponentMonitor:
    """Monitor for backtesting components."""
    
    def __init__(
        self,
        registry: Optional[BacktestingComponentRegistry] = None,
        orchestrator: Optional[BacktestingWorkflowOrchestrator] = None,
        config: Optional[MonitoringConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.registry = registry or get_registry()
        self.orchestrator = orchestrator or get_orchestrator()
        self.config = config or MonitoringConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Monitoring data
        self._component_health: Dict[str, ComponentHealth] = {}
        self._performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.performance_history_size))
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.health_history_size))
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Backtesting-specific monitoring
        self._strategy_performance: Dict[str, Dict[str, Any]] = {}
        self._portfolio_monitoring: Dict[str, Dict[str, Any]] = {}
        self._risk_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Alert thresholds
        self._alert_thresholds = {
            'error_rate': 0.1,  # 10% error rate
            'response_time': 10.0,  # 10 seconds
            'memory_usage': 0.8,  # 80% memory usage
            'cpu_usage': 0.8,  # 80% CPU usage
            'health_score': 0.7,  # 70% health score
            'dependency_failures': 3,  # 3 dependency failures
            'backtesting_accuracy': 0.8,  # 80% backtesting accuracy
            'portfolio_drawdown': 0.15,  # 15% portfolio drawdown
            'risk_breach': 1.0  # 100% risk breach
        }
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        with self._lock:
            if self._monitoring_active:
                return
            
            self._monitoring_active = True
            
            # Start monitoring thread
            if self.config.enable_real_time_monitoring:
                self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self._monitoring_thread.start()
            
            # Start health check thread
            if self.config.enable_health_monitoring:
                self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
                self._health_check_thread.start()
            
            self.logger.info("Component monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        with self._lock:
            self._monitoring_active = False
            
            # Wait for threads to finish
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)
            
            if self._health_check_thread:
                self._health_check_thread.join(timeout=5.0)
            
            self.logger.info("Component monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Monitor components
                self._monitor_components()
                
                # Monitor workflows
                self._monitor_workflows()
                
                # Process alerts
                self._process_alerts()
                
                # Sleep for monitoring interval
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while self._monitoring_active:
            try:
                # Run health checks
                self._run_health_checks()
                
                # Sleep for health check interval
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _monitor_components(self) -> None:
        """Monitor all components."""
        with self._lock:
            # Get all components from registry
            components = self.registry.get_all_components()
            
            for component_info in components:
                component_name = component_info['name']
                
                # Update component health
                self._update_component_health(component_name, component_info)
                
                # Collect performance metrics
                if self.config.enable_performance_tracking:
                    self._collect_performance_metrics(component_name, component_info)
                
                # Check for alerts
                if self.config.enable_alerting:
                    self._check_component_alerts(component_name, component_info)
    
    def _monitor_workflows(self) -> None:
        """Monitor all workflows."""
        with self._lock:
            # Get all workflows from orchestrator
            workflows = self.orchestrator.get_all_workflows()
            
            for workflow_info in workflows:
                workflow_id = workflow_info['workflow_id']
                
                # Monitor workflow performance
                self._monitor_workflow_performance(workflow_id, workflow_info)
                
                # Check for workflow alerts
                if self.config.enable_alerting:
                    self._check_workflow_alerts(workflow_id, workflow_info)
    
    def _update_component_health(self, component_name: str, component_info: Dict[str, Any]) -> None:
        """Update component health information."""
        if component_name not in self._component_health:
            self._component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status=ComponentStatus(component_info['status']),
                health_score=0.0,
                last_updated=time.time()
            )
        
        health = self._component_health[component_name]
        health.status = ComponentStatus(component_info['status'])
        health.last_updated = time.time()
        health.dependencies = component_info.get('dependencies', [])
        health.dependents = component_info.get('dependents', [])
        
        # Calculate health score
        health.health_score = self._calculate_health_score(component_info)
        
        # Update health history
        self._health_history[component_name].append({
            'timestamp': time.time(),
            'health_score': health.health_score,
            'status': health.status.value
        })
    
    def _calculate_health_score(self, component_info: Dict[str, Any]) -> float:
        """Calculate health score for a component."""
        score = 1.0
        
        # Status penalty
        status = component_info.get('status', 'unknown')
        if status == 'error':
            score -= 0.5
        elif status == 'stopped':
            score -= 0.2
        
        # Error count penalty
        error_count = component_info.get('error_count', 0)
        if error_count > 0:
            score -= min(error_count * 0.1, 0.5)
        
        # Warning count penalty
        warning_count = component_info.get('warning_count', 0)
        if warning_count > 0:
            score -= min(warning_count * 0.05, 0.2)
        
        # Performance penalty
        performance_stats = component_info.get('performance_stats', {})
        if performance_stats:
            success_rate = performance_stats.get('success_rate', 1.0)
            score *= success_rate
        
        return max(0.0, min(1.0, score))
    
    def _collect_performance_metrics(self, component_name: str, component_info: Dict[str, Any]) -> None:
        """Collect performance metrics for a component."""
        performance_stats = component_info.get('performance_stats', {})
        
        if performance_stats:
            timestamp = time.time()
            
            # Collect various metrics
            metrics = [
                ('total_operations', MetricType.COUNTER),
                ('successful_operations', MetricType.COUNTER),
                ('failed_operations', MetricType.COUNTER),
                ('total_time', MetricType.TIMER),
                ('success_rate', MetricType.GAUGE),
                ('avg_processing_time', MetricType.TIMER)
            ]
            
            for metric_name, metric_type in metrics:
                if metric_name in performance_stats:
                    metric = PerformanceMetric(
                        name=metric_name,
                        value=performance_stats[metric_name],
                        metric_type=metric_type,
                        timestamp=timestamp,
                        component_name=component_name
                    )
                    
                    self._performance_metrics[f"{component_name}_{metric_name}"].append(metric)
    
    def _check_component_alerts(self, component_name: str, component_info: Dict[str, Any]) -> None:
        """Check for component alerts."""
        # Check error rate
        error_count = component_info.get('error_count', 0)
        total_operations = component_info.get('performance_stats', {}).get('total_operations', 1)
        error_rate = error_count / max(total_operations, 1)
        
        if error_rate > self._alert_thresholds['error_rate']:
            self._create_alert(
                component_name=component_name,
                level=AlertLevel.ERROR,
                message=f"High error rate: {error_rate:.2%}",
                category="performance"
            )
        
        # Check response time
        avg_processing_time = component_info.get('performance_stats', {}).get('avg_processing_time', 0)
        if avg_processing_time > self._alert_thresholds['response_time']:
            self._create_alert(
                component_name=component_name,
                level=AlertLevel.WARNING,
                message=f"High response time: {avg_processing_time:.2f}s",
                category="performance"
            )
        
        # Check health score
        health_score = self._component_health.get(component_name, ComponentHealth(component_name, ComponentStatus.REGISTERED, 0.0, 0.0)).health_score
        if health_score < self._alert_thresholds['health_score']:
            self._create_alert(
                component_name=component_name,
                level=AlertLevel.WARNING,
                message=f"Low health score: {health_score:.2f}",
                category="health"
            )
    
    def _check_workflow_alerts(self, workflow_id: str, workflow_info: Dict[str, Any]) -> None:
        """Check for workflow alerts."""
        status = workflow_info.get('status', 'unknown')
        
        if status == 'failed':
            self._create_alert(
                component_name=f"workflow_{workflow_id}",
                level=AlertLevel.ERROR,
                message=f"Workflow {workflow_id} failed",
                category="workflow"
            )
        
        # Check execution time
        start_time = workflow_info.get('start_time', 0)
        end_time = workflow_info.get('end_time', time.time())
        execution_time = end_time - start_time
        
        if execution_time > 3600:  # 1 hour
            self._create_alert(
                component_name=f"workflow_{workflow_id}",
                level=AlertLevel.WARNING,
                message=f"Workflow {workflow_id} running for {execution_time/3600:.1f} hours",
                category="workflow"
            )
    
    def _monitor_workflow_performance(self, workflow_id: str, workflow_info: Dict[str, Any]) -> None:
        """Monitor workflow performance."""
        # Track workflow execution time
        start_time = workflow_info.get('start_time', 0)
        end_time = workflow_info.get('end_time', time.time())
        execution_time = end_time - start_time
        
        # Track completed vs failed steps
        completed_steps = workflow_info.get('completed_steps', [])
        failed_steps = workflow_info.get('failed_steps', [])
        total_steps = len(completed_steps) + len(failed_steps)
        
        if total_steps > 0:
            success_rate = len(completed_steps) / total_steps
            
            # Store workflow performance metrics
            self._performance_metrics[f"workflow_{workflow_id}_success_rate"].append(
                PerformanceMetric(
                    name="success_rate",
                    value=success_rate,
                    metric_type=MetricType.GAUGE,
                    timestamp=time.time(),
                    component_name=f"workflow_{workflow_id}"
                )
            )
    
    def _run_health_checks(self) -> None:
        """Run health checks on all components."""
        with self._lock:
            # Get health check results from registry
            health_results = self.registry.run_health_checks()
            
            # Update component health based on results
            for component_name, health_info in health_results.get('components', {}).items():
                if component_name in self._component_health:
                    health = self._component_health[component_name]
                    health.health_score = self._calculate_health_score(health_info)
                    health.last_updated = time.time()
    
    def _create_alert(
        self,
        component_name: str,
        level: AlertLevel,
        message: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a new alert."""
        alert = Alert(
            id=f"{component_name}_{int(time.time())}",
            component_name=component_name,
            level=level,
            message=message,
            timestamp=time.time(),
            category=category,
            metadata=metadata or {}
        )
        
        self._alerts.append(alert)
        
        # Call alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        self.logger.warning(f"Alert created: {level.value} - {component_name}: {message}")
    
    def _process_alerts(self) -> None:
        """Process and clean up alerts."""
        with self._lock:
            current_time = time.time()
            retention_time = self.config.alert_retention_days * 24 * 3600
            
            # Remove old alerts
            self._alerts = [
                alert for alert in self._alerts
                if current_time - alert.timestamp < retention_time
            ]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add an alert callback function."""
        self._alert_callbacks.append(callback)
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health information for a component."""
        with self._lock:
            return self._component_health.get(component_name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health information for all components."""
        with self._lock:
            return self._component_health.copy()
    
    def get_performance_metrics(
        self,
        component_name: str,
        metric_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[PerformanceMetric]:
        """Get performance metrics for a component."""
        with self._lock:
            if metric_name:
                key = f"{component_name}_{metric_name}"
                metrics = self._performance_metrics.get(key, deque())
            else:
                # Get all metrics for component
                metrics = []
                for key, metric_deque in self._performance_metrics.items():
                    if key.startswith(f"{component_name}_"):
                        metrics.extend(metric_deque)
            
            # Filter by time range
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                return filtered_metrics
            
            return list(metrics)
    
    def get_alerts(
        self,
        component_name: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        category: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = self._alerts.copy()
            
            # Filter by component name
            if component_name:
                alerts = [alert for alert in alerts if alert.component_name == component_name]
            
            # Filter by level
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            # Filter by category
            if category:
                alerts = [alert for alert in alerts if alert.category == category]
            
            # Filter by time range
            if start_time or end_time:
                filtered_alerts = []
                for alert in alerts:
                    if start_time and alert.timestamp < start_time:
                        continue
                    if end_time and alert.timestamp > end_time:
                        continue
                    filtered_alerts.append(alert)
                alerts = filtered_alerts
            
            return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    return True
            return False
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        with self._lock:
            # Component health summary
            total_components = len(self._component_health)
            healthy_components = sum(1 for h in self._component_health.values() if h.health_score > 0.8)
            unhealthy_components = sum(1 for h in self._component_health.values() if h.health_score < 0.5)
            
            # Alert summary
            total_alerts = len(self._alerts)
            unacknowledged_alerts = sum(1 for alert in self._alerts if not alert.acknowledged)
            critical_alerts = sum(1 for alert in self._alerts if alert.level == AlertLevel.CRITICAL)
            
            # Performance summary
            performance_summary = {}
            for component_name in self._component_health.keys():
                metrics = self.get_performance_metrics(component_name)
                if metrics:
                    latest_metrics = {m.name: m.value for m in metrics[-10:]}  # Last 10 metrics
                    performance_summary[component_name] = latest_metrics
            
            return {
                'timestamp': time.time(),
                'components': {
                    'total': total_components,
                    'healthy': healthy_components,
                    'unhealthy': unhealthy_components,
                    'health_percentage': (healthy_components / max(total_components, 1)) * 100
                },
                'alerts': {
                    'total': total_alerts,
                    'unacknowledged': unacknowledged_alerts,
                    'critical': critical_alerts
                },
                'performance': performance_summary,
                'health_scores': {
                    name: health.health_score
                    for name, health in self._component_health.items()
                }
            }
    
    def generate_performance_report(
        self,
        component_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate a performance report for a component."""
        with self._lock:
            # Get performance metrics
            metrics = self.get_performance_metrics(component_name, start_time=start_time, end_time=end_time)
            
            if not metrics:
                return {'error': 'No metrics available'}
            
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric.value)
            
            # Calculate statistics
            report = {
                'component_name': component_name,
                'period': {
                    'start': start_time,
                    'end': end_time
                },
                'metrics': {}
            }
            
            for metric_name, values in metric_groups.items():
                if values:
                    report['metrics'][metric_name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'latest': values[-1]
                    }
            
            return report
    
    def create_performance_visualization(
        self,
        component_name: str,
        metric_name: str,
        output_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> bool:
        """Create a performance visualization."""
        try:
            # Get metrics
            metrics = self.get_performance_metrics(component_name, metric_name, start_time, end_time)
            
            if not metrics:
                return False
            
            # Prepare data
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
            values = [m.value for m in metrics]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, values, linewidth=2)
            ax.set_title(f"{component_name} - {metric_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.xticks(rotation=45)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            return False
    
    def export_monitoring_data(self, output_path: str) -> bool:
        """Export monitoring data to file."""
        try:
            with self._lock:
                export_data = {
                    'timestamp': time.time(),
                    'config': {
                        'monitoring_interval': self.config.monitoring_interval,
                        'health_check_interval': self.config.health_check_interval,
                        'alert_retention_days': self.config.alert_retention_days,
                        'performance_history_size': self.config.performance_history_size,
                        'health_history_size': self.config.health_history_size
                    },
                    'component_health': {
                        name: {
                            'component_name': health.component_name,
                            'status': health.status.value,
                            'health_score': health.health_score,
                            'last_updated': health.last_updated,
                            'dependencies': health.dependencies,
                            'dependents': health.dependents
                        }
                        for name, health in self._component_health.items()
                    },
                    'alerts': [
                        {
                            'id': alert.id,
                            'component_name': alert.component_name,
                            'level': alert.level.value,
                            'message': alert.message,
                            'timestamp': alert.timestamp,
                            'category': alert.category,
                            'acknowledged': alert.acknowledged,
                            'resolved': alert.resolved
                        }
                        for alert in self._alerts
                    ],
                    'performance_metrics': {
                        key: [
                            {
                                'name': metric.name,
                                'value': metric.value,
                                'metric_type': metric.metric_type.value,
                                'timestamp': metric.timestamp,
                                'component_name': metric.component_name
                            }
                            for metric in metrics
                        ]
                        for key, metrics in self._performance_metrics.items()
                    }
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Monitoring data exported to {output_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {e}")
            return False


# Global monitor instance
_monitor_instance: Optional[BacktestingComponentMonitor] = None


def get_monitor() -> BacktestingComponentMonitor:
    """Get the global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = BacktestingComponentMonitor()
    return _monitor_instance


def start_monitoring() -> None:
    """Start monitoring using the global monitor."""
    get_monitor().start_monitoring()


def stop_monitoring() -> None:
    """Stop monitoring using the global monitor."""
    get_monitor().stop_monitoring()


def get_component_health(component_name: str) -> Optional[ComponentHealth]:
    """Get component health using the global monitor."""
    return get_monitor().get_component_health(component_name)


def get_all_component_health() -> Dict[str, ComponentHealth]:
    """Get all component health using the global monitor."""
    return get_monitor().get_all_component_health()


def get_performance_metrics(
    component_name: str,
    metric_name: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> List[PerformanceMetric]:
    """Get performance metrics using the global monitor."""
    return get_monitor().get_performance_metrics(component_name, metric_name, start_time, end_time)


def get_alerts(
    component_name: Optional[str] = None,
    level: Optional[AlertLevel] = None,
    category: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> List[Alert]:
    """Get alerts using the global monitor."""
    return get_monitor().get_alerts(component_name, level, category, start_time, end_time)


def get_monitoring_dashboard_data() -> Dict[str, Any]:
    """Get monitoring dashboard data using the global monitor."""
    return get_monitor().get_monitoring_dashboard_data()


# Export all public classes and functions
__all__ = [
    'AlertLevel',
    'MetricType',
    'Alert',
    'PerformanceMetric',
    'ComponentHealth',
    'MonitoringConfig',
    'BacktestingComponentMonitor',
    'get_monitor',
    'start_monitoring',
    'stop_monitoring',
    'get_component_health',
    'get_all_component_health',
    'get_performance_metrics',
    'get_alerts',
    'get_monitoring_dashboard_data'
]