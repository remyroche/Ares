"""
ModularComponent Monitoring Dashboard

This module provides comprehensive monitoring and performance tracking for
ModularComponent instances across the pipeline.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from pathlib import Path

from .modular_architecture import ModularComponent


@dataclass
class ComponentMetrics:
    """Metrics for a single component."""
    name: str
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    last_execution_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    health_score: float = 1.0
    status: str = "unknown"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.execution_count == 0:
            return 1.0
        return self.success_count / self.execution_count
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.execution_count == 0:
            return 0.0
        return self.error_count / self.execution_count


@dataclass
class PipelineMetrics:
    """Overall pipeline metrics."""
    total_components: int = 0
    healthy_components: int = 0
    degraded_components: int = 0
    error_components: int = 0
    total_executions: int = 0
    total_successes: int = 0
    total_errors: int = 0
    overall_health_score: float = 1.0
    avg_execution_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    last_updated: Optional[datetime] = None


class ModularComponentMonitor:
    """
    Comprehensive monitoring system for ModularComponent instances.
    
    Features:
    - Real-time performance tracking
    - Health monitoring and alerting
    - Historical metrics collection
    - Memory usage tracking
    - Error tracking and analysis
    - Performance recommendations
    """
    
    def __init__(self, 
                 log_file: Optional[str] = None,
                 metrics_retention_days: int = 30,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """Initialize the monitoring system."""
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.metrics_retention_days = metrics_retention_days
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'error_rate': 0.1,  # 10% error rate
            'avg_execution_time': 5.0,  # 5 seconds
            'memory_usage_mb': 1000.0,  # 1GB
            'health_score': 0.7  # 70% health score
        }
        
        # Metrics storage
        self.component_metrics: Dict[str, ComponentMetrics] = {}
        self.pipeline_metrics = PipelineMetrics()
        self.historical_metrics: deque = deque(maxlen=1000)  # Keep last 1000 data points
        self.alerts: List[Dict[str, Any]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start background monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("ModularComponent monitoring system initialized")
    
    def register_component(self, component: ModularComponent) -> bool:
        """Register a component for monitoring."""
        try:
            with self._lock:
                if component.name in self.component_metrics:
                    self.logger.warning(f"Component {component.name} already registered")
                    return False
                
                self.component_metrics[component.name] = ComponentMetrics(
                    name=component.name,
                    status="registered"
                )
                
                self.logger.info(f"Registered component for monitoring: {component.name}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to register component {component.name}: {e}")
            return False
    
    def unregister_component(self, component_name: str) -> bool:
        """Unregister a component from monitoring."""
        try:
            with self._lock:
                if component_name in self.component_metrics:
                    del self.component_metrics[component_name]
                    self.logger.info(f"Unregistered component from monitoring: {component_name}")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Failed to unregister component {component_name}: {e}")
            return False
    
    def record_execution(self, component_name: str, execution_time: float, 
                        success: bool, error_message: Optional[str] = None,
                        memory_usage_mb: Optional[float] = None) -> None:
        """Record a component execution."""
        try:
            with self._lock:
                if component_name not in self.component_metrics:
                    self.logger.warning(f"Component {component_name} not registered for monitoring")
                    return
                
                metrics = self.component_metrics[component_name]
                metrics.execution_count += 1
                metrics.total_execution_time += execution_time
                metrics.avg_execution_time = metrics.total_execution_time / metrics.execution_count
                metrics.last_execution_time = datetime.now()
                
                if success:
                    metrics.success_count += 1
                else:
                    metrics.error_count += 1
                    metrics.last_error_time = datetime.now()
                    metrics.last_error_message = error_message
                
                if memory_usage_mb is not None:
                    metrics.memory_usage_mb = memory_usage_mb
                    metrics.peak_memory_usage_mb = max(metrics.peak_memory_usage_mb, memory_usage_mb)
                
                # Update health score
                self._update_component_health_score(metrics)
                
                # Check for alerts
                self._check_alerts(component_name, metrics)
                
        except Exception as e:
            self.logger.error(f"Failed to record execution for {component_name}: {e}")
    
    def _update_component_health_score(self, metrics: ComponentMetrics) -> None:
        """Update component health score based on metrics."""
        try:
            # Base health score
            health_score = 1.0
            
            # Penalize for errors
            if metrics.execution_count > 0:
                error_rate = metrics.error_count / metrics.execution_count
                health_score -= error_rate * 0.5  # 50% penalty for errors
            
            # Penalize for slow execution
            if metrics.avg_execution_time > self.alert_thresholds['avg_execution_time']:
                time_penalty = min(0.3, (metrics.avg_execution_time - self.alert_thresholds['avg_execution_time']) / 10.0)
                health_score -= time_penalty
            
            # Penalize for high memory usage
            if metrics.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
                memory_penalty = min(0.2, (metrics.memory_usage_mb - self.alert_thresholds['memory_usage_mb']) / 1000.0)
                health_score -= memory_penalty
            
            # Ensure health score is between 0 and 1
            metrics.health_score = max(0.0, min(1.0, health_score))
            
            # Update status based on health score
            if metrics.health_score >= 0.9:
                metrics.status = "excellent"
            elif metrics.health_score >= 0.7:
                metrics.status = "healthy"
            elif metrics.health_score >= 0.5:
                metrics.status = "degraded"
            else:
                metrics.status = "unhealthy"
                
        except Exception as e:
            self.logger.error(f"Failed to update health score: {e}")
    
    def _check_alerts(self, component_name: str, metrics: ComponentMetrics) -> None:
        """Check for alert conditions."""
        try:
            alerts = []
            
            # Check error rate
            if metrics.error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'high_error_rate',
                    'component': component_name,
                    'value': metrics.error_rate,
                    'threshold': self.alert_thresholds['error_rate'],
                    'message': f"High error rate: {metrics.error_rate:.2%} > {self.alert_thresholds['error_rate']:.2%}"
                })
            
            # Check execution time
            if metrics.avg_execution_time > self.alert_thresholds['avg_execution_time']:
                alerts.append({
                    'type': 'slow_execution',
                    'component': component_name,
                    'value': metrics.avg_execution_time,
                    'threshold': self.alert_thresholds['avg_execution_time'],
                    'message': f"Slow execution: {metrics.avg_execution_time:.2f}s > {self.alert_thresholds['avg_execution_time']:.2f}s"
                })
            
            # Check memory usage
            if metrics.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
                alerts.append({
                    'type': 'high_memory_usage',
                    'component': component_name,
                    'value': metrics.memory_usage_mb,
                    'threshold': self.alert_thresholds['memory_usage_mb'],
                    'message': f"High memory usage: {metrics.memory_usage_mb:.2f}MB > {self.alert_thresholds['memory_usage_mb']:.2f}MB"
                })
            
            # Check health score
            if metrics.health_score < self.alert_thresholds['health_score']:
                alerts.append({
                    'type': 'low_health_score',
                    'component': component_name,
                    'value': metrics.health_score,
                    'threshold': self.alert_thresholds['health_score'],
                    'message': f"Low health score: {metrics.health_score:.2f} < {self.alert_thresholds['health_score']:.2f}"
                })
            
            # Add alerts
            for alert in alerts:
                alert['timestamp'] = datetime.now().isoformat()
                self.alerts.append(alert)
                self.logger.warning(f"ALERT: {alert['message']}")
                
        except Exception as e:
            self.logger.error(f"Failed to check alerts: {e}")
    
    def _background_monitoring(self) -> None:
        """Background monitoring thread."""
        while self._monitoring_active:
            try:
                self._update_pipeline_metrics()
                self._save_historical_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_pipeline_metrics(self) -> None:
        """Update overall pipeline metrics."""
        try:
            with self._lock:
                self.pipeline_metrics.total_components = len(self.component_metrics)
                self.pipeline_metrics.healthy_components = sum(
                    1 for m in self.component_metrics.values() 
                    if m.status in ['excellent', 'healthy']
                )
                self.pipeline_metrics.degraded_components = sum(
                    1 for m in self.component_metrics.values() 
                    if m.status == 'degraded'
                )
                self.pipeline_metrics.error_components = sum(
                    1 for m in self.component_metrics.values() 
                    if m.status == 'unhealthy'
                )
                
                # Calculate totals
                total_executions = sum(m.execution_count for m in self.component_metrics.values())
                total_successes = sum(m.success_count for m in self.component_metrics.values())
                total_errors = sum(m.error_count for m in self.component_metrics.values())
                
                self.pipeline_metrics.total_executions = total_executions
                self.pipeline_metrics.total_successes = total_successes
                self.pipeline_metrics.total_errors = total_errors
                
                # Calculate overall health score
                if self.component_metrics:
                    avg_health = sum(m.health_score for m in self.component_metrics.values()) / len(self.component_metrics)
                    self.pipeline_metrics.overall_health_score = avg_health
                else:
                    self.pipeline_metrics.overall_health_score = 1.0
                
                # Calculate average execution time
                if total_executions > 0:
                    total_time = sum(m.total_execution_time for m in self.component_metrics.values())
                    self.pipeline_metrics.avg_execution_time = total_time / total_executions
                
                # Calculate peak memory usage
                if self.component_metrics:
                    self.pipeline_metrics.peak_memory_usage_mb = max(
                        m.peak_memory_usage_mb for m in self.component_metrics.values()
                    )
                
                self.pipeline_metrics.last_updated = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Failed to update pipeline metrics: {e}")
    
    def _save_historical_metrics(self) -> None:
        """Save historical metrics for analysis."""
        try:
            if not self.log_file:
                return
            
            historical_data = {
                'timestamp': datetime.now().isoformat(),
                'pipeline_metrics': asdict(self.pipeline_metrics),
                'component_metrics': {name: asdict(metrics) for name, metrics in self.component_metrics.items()}
            }
            
            self.historical_metrics.append(historical_data)
            
            # Save to file periodically
            if len(self.historical_metrics) % 10 == 0:  # Every 10 data points
                self._write_metrics_to_file()
                
        except Exception as e:
            self.logger.error(f"Failed to save historical metrics: {e}")
    
    def _write_metrics_to_file(self) -> None:
        """Write metrics to log file."""
        try:
            if not self.log_file:
                return
            
            with open(self.log_file, 'a') as f:
                for data in self.historical_metrics:
                    f.write(json.dumps(data, default=str) + '\n')
            
            # Clear written data
            self.historical_metrics.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to write metrics to file: {e}")
    
    def get_component_metrics(self, component_name: str) -> Optional[ComponentMetrics]:
        """Get metrics for a specific component."""
        with self._lock:
            return self.component_metrics.get(component_name)
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get overall pipeline metrics."""
        with self._lock:
            return self.pipeline_metrics
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        with self._lock:
            return self.alerts[-limit:] if self.alerts else []
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self.alerts.clear()
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        try:
            with self._lock:
                for name, metrics in self.component_metrics.items():
                    # High error rate recommendation
                    if metrics.error_rate > 0.05:  # 5% error rate
                        recommendations.append(
                            f"Component '{name}' has high error rate ({metrics.error_rate:.1%}). "
                            "Consider reviewing error handling and input validation."
                        )
                    
                    # Slow execution recommendation
                    if metrics.avg_execution_time > 2.0:  # 2 seconds
                        recommendations.append(
                            f"Component '{name}' has slow execution ({metrics.avg_execution_time:.2f}s). "
                            "Consider optimizing algorithms or adding caching."
                        )
                    
                    # High memory usage recommendation
                    if metrics.peak_memory_usage_mb > 500:  # 500MB
                        recommendations.append(
                            f"Component '{name}' uses high memory ({metrics.peak_memory_usage_mb:.1f}MB). "
                            "Consider implementing memory optimization or data streaming."
                        )
                    
                    # Low health score recommendation
                    if metrics.health_score < 0.8:  # 80% health
                        recommendations.append(
                            f"Component '{name}' has low health score ({metrics.health_score:.1%}). "
                            "Consider reviewing overall component performance and stability."
                        )
        
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        try:
            with self._lock:
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_metrics': asdict(self.pipeline_metrics),
                    'component_metrics': {name: asdict(metrics) for name, metrics in self.component_metrics.items()},
                    'recent_alerts': self.alerts[-20:],  # Last 20 alerts
                    'recommendations': self.get_performance_recommendations(),
                    'monitoring_status': {
                        'active': self._monitoring_active,
                        'log_file': self.log_file,
                        'metrics_retention_days': self.metrics_retention_days
                    }
                }
                
                return report
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return {'error': str(e)}
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self._monitoring_active = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Save final metrics
        self._write_metrics_to_file()
        self.logger.info("ModularComponent monitoring system stopped")


def create_monitor(log_file: Optional[str] = None, 
                  metrics_retention_days: int = 30,
                  alert_thresholds: Optional[Dict[str, float]] = None) -> ModularComponentMonitor:
    """Create a new ModularComponentMonitor instance."""
    return ModularComponentMonitor(log_file, metrics_retention_days, alert_thresholds)