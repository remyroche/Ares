"""
Production Monitoring for TAS

Comprehensive monitoring and alerting system for production TAS deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import time
import psutil
import threading
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for TAS monitoring."""

    # Monitoring intervals
    health_check_interval: int = 30  # seconds
    performance_check_interval: int = 60  # seconds
    alert_check_interval: int = 10  # seconds

    # Resource monitoring
    cpu_threshold: float = 80.0  # %
    memory_threshold: float = 80.0  # %
    disk_threshold: float = 90.0  # %

    # Performance thresholds
    search_time_threshold: float = 300.0  # seconds
    memory_usage_threshold: float = 8.0  # GB
    error_rate_threshold: float = 0.1  # 10%

    # Alerting
    enable_alerts: bool = True
    alert_cooldown: int = 300  # seconds
    alert_channels: List[str] = field(default_factory=lambda: ['log', 'email'])

    # Data retention
    metrics_retention_days: int = 30
    log_retention_days: int = 7

    # Monitoring endpoints
    enable_metrics_endpoint: bool = True
    metrics_port: int = 8080
    enable_health_endpoint: bool = True
    health_port: int = 8081

class HealthCheck:
    """Health check for TAS components."""

    def __init__(self, name: str, check_function: Callable[[], bool], timeout: int = 5):
        """Initialize health check.

        Args:
            name: Name of the health check
            check_function: Function that returns True if healthy
            timeout: Timeout in seconds
        """
        self.name = name
        self.check_function = check_function
        self.timeout = timeout
        self.last_check = None
        self.last_result = None
        self.failure_count = 0

    def run_check(self) -> Dict[str, Any]:
        """Run the health check."""
        start_time = time.time()

        try:
            # Run check with timeout
            result = self._run_with_timeout()
            duration = time.time() - start_time

            self.last_check = datetime.now()
            self.last_result = result

            if result:
                self.failure_count = 0
            else:
                self.failure_count += 1

            return {
                'name': self.name,
                'healthy': result,
                'duration': duration,
                'failure_count': self.failure_count,
                'timestamp': self.last_check.isoformat()
            }

        except Exception as e:
            self.failure_count += 1
            return {
                'name': self.name,
                'healthy': False,
                'duration': time.time() - start_time,
                'failure_count': self.failure_count,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _run_with_timeout(self) -> bool:
        """Run check with timeout."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)

        try:
            result = self.check_function()
            return result
        finally:
            signal.alarm(0)

class TASMonitor:
    """
    Production monitoring system for TAS.

    Provides comprehensive monitoring of system health, performance,
    and resource usage for production TAS deployment.
    """

    def __init__(self, config: MonitoringConfig):
        """Initialize TAS monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Monitoring state
        self.metrics_history = deque(maxlen=1000)
        self.health_checks = {}
        self.alerts = []
        self.alert_cooldowns = {}

        # Performance tracking
        self.performance_metrics = {
            'search_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'error_counts': deque(maxlen=100),
            'success_rates': deque(maxlen=100)
        }

        # Monitoring threads
        self.monitoring_thread = None
        self.running = False

        # Initialize health checks
        self._initialize_health_checks()

        self.logger.info("✅ TAS Monitor initialized")

    def start_monitoring(self):
        """Start monitoring in background thread."""
        if self.running:
            self.logger.warning("⚠️ Monitoring already running")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("🚀 TAS monitoring started")

    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("🛑 TAS monitoring stopped")

    def add_health_check(self, name: str, check_function: Callable[[], bool], timeout: int = 5):
        """Add a health check."""
        self.health_checks[name] = HealthCheck(name, check_function, timeout)
        self.logger.info(f"✅ Added health check: {name}")

    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric."""
        if timestamp is None:
            timestamp = datetime.now()

        metric = {
            'name': metric_name,
            'value': value,
            'timestamp': timestamp
        }

        self.metrics_history.append(metric)

        # Update performance metrics
        if metric_name in self.performance_metrics:
            self.performance_metrics[metric_name].append(value)

    def record_search_performance(self, search_time: float, memory_usage: float, success: bool):
        """Record search performance metrics."""
        self.record_metric('search_time', search_time)
        self.record_metric('memory_usage', memory_usage)
        self.record_metric('search_success', 1.0 if success else 0.0)

        # Check for alerts
        self._check_performance_alerts(search_time, memory_usage, success)

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        health_status = {
            'overall_healthy': True,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'metrics': self._get_current_metrics(),
            'alerts': len(self.alerts)
        }

        # Run all health checks
        for name, health_check in self.health_checks.items():
            check_result = health_check.run_check()
            health_status['checks'][name] = check_result

            if not check_result['healthy']:
                health_status['overall_healthy'] = False

        return health_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {}

        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        return metrics

    def get_alerts(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get alerts."""
        if active_only:
            return [alert for alert in self.alerts if alert.get('active', True)]
        return self.alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
        self.logger.info("🧹 Alerts cleared")

    def _initialize_health_checks(self):
        """Initialize default health checks."""
        # System resource checks
        self.add_health_check(
            'cpu_usage',
            lambda: psutil.cpu_percent() < self.config.cpu_threshold
        )

        self.add_health_check(
            'memory_usage',
            lambda: psutil.virtual_memory().percent < self.config.memory_threshold
        )

        self.add_health_check(
            'disk_usage',
            lambda: psutil.disk_usage('/').percent < self.config.disk_threshold
        )

        # TAS-specific checks
        self.add_health_check(
            'search_performance',
            self._check_search_performance
        )

        self.add_health_check(
            'error_rate',
            self._check_error_rate
        )

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Record system metrics
                self._record_system_metrics()

                # Check for alerts
                self._check_alerts()

                # Sleep until next check
                time.sleep(self.config.alert_check_interval)

            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(5)

    def _record_system_metrics(self):
        """Record system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.record_metric('cpu_usage', cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_gb = memory.used / (1024**3)
        self.record_metric('memory_usage_percent', memory_percent)
        self.record_metric('memory_usage_gb', memory_gb)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric('disk_usage', disk_percent)

    def _check_performance_alerts(self, search_time: float, memory_usage: float, success: bool):
        """Check for performance alerts."""
        # Search time alert
        if search_time > self.config.search_time_threshold:
            self._create_alert(
                'high_search_time',
                f'Search time {search_time:.2f}s exceeds threshold {self.config.search_time_threshold}s',
                'warning'
            )

        # Memory usage alert
        if memory_usage > self.config.memory_usage_threshold:
            self._create_alert(
                'high_memory_usage',
                f'Memory usage {memory_usage:.2f}GB exceeds threshold {self.config.memory_usage_threshold}GB',
                'warning'
            )

        # Error rate alert
        if not success:
            recent_successes = list(self.performance_metrics['success_rates'])[-10:]
            if recent_successes:
                error_rate = 1.0 - np.mean(recent_successes)
                if error_rate > self.config.error_rate_threshold:
                    self._create_alert(
                        'high_error_rate',
                        f'Error rate {error_rate:.2%} exceeds threshold {self.config.error_rate_threshold:.2%}',
                        'critical'
                    )

    def _check_alerts(self):
        """Check for system alerts."""
        # CPU alert
        if self.performance_metrics['cpu_usage']:
            cpu_usage = self.performance_metrics['cpu_usage'][-1]
            if cpu_usage > self.config.cpu_threshold:
                self._create_alert(
                    'high_cpu_usage',
                    f'CPU usage {cpu_usage:.1f}% exceeds threshold {self.config.cpu_threshold}%',
                    'warning'
                )

        # Memory alert
        if self.performance_metrics['memory_usage']:
            memory_usage = self.performance_metrics['memory_usage'][-1]
            if memory_usage > self.config.memory_threshold:
                self._create_alert(
                    'high_memory_usage',
                    f'Memory usage {memory_usage:.1f}% exceeds threshold {self.config.memory_threshold}%',
                    'warning'
                )

    def _check_search_performance(self) -> bool:
        """Check search performance health."""
        if not self.performance_metrics['search_times']:
            return True

        recent_times = list(self.performance_metrics['search_times'])[-5:]
        avg_time = np.mean(recent_times)

        return avg_time < self.config.search_time_threshold

    def _check_error_rate(self) -> bool:
        """Check error rate health."""
        if not self.performance_metrics['success_rates']:
            return True

        recent_successes = list(self.performance_metrics['success_rates'])[-10:]
        if not recent_successes:
            return True

        error_rate = 1.0 - np.mean(recent_successes)
        return error_rate < self.config.error_rate_threshold

    def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create an alert."""
        # Check cooldown
        if alert_type in self.alert_cooldowns:
            last_alert = self.alert_cooldowns[alert_type]
            if datetime.now() - last_alert < timedelta(seconds=self.config.alert_cooldown):
                return

        alert = {
            'id': f"alert_{len(self.alerts) + 1}",
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'active': True
        }

        self.alerts.append(alert)
        self.alert_cooldowns[alert_type] = datetime.now()

        self.logger.warning(f"🚨 ALERT [{severity.upper()}]: {message}")

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {}

        for metric_name, values in self.performance_metrics.items():
            if values:
                metrics[metric_name] = values[-1]

        return metrics

    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        try:
            export_data = {
                'config': self.config.__dict__,
                'metrics': list(self.metrics_history),
                'performance': self.get_performance_metrics(),
                'health': self.get_system_health(),
                'alerts': self.get_alerts(active_only=False)
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info(f"📁 Metrics exported to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export metrics: {e}")

    def cleanup_old_data(self):
        """Cleanup old monitoring data."""
        cutoff_date = datetime.now() - timedelta(days=self.config.metrics_retention_days)

        # Cleanup metrics history
        self.metrics_history = deque([
            metric for metric in self.metrics_history
            if metric['timestamp'] > cutoff_date
        ], maxlen=1000)

        # Cleanup old alerts
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_date
        ]

        self.logger.info("🧹 Old monitoring data cleaned up")
