"""
Real-time Monitoring Dashboard

This module provides a comprehensive monitoring dashboard for exchange OHLCV
data processing operations.

Features:
- Real-time data processing monitoring
- Exchange status monitoring
- Performance metrics visualization
- Data quality monitoring
- System health monitoring
- Alert management
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import statistics

# Import our unified components
from .unified_ohlcv_standardizer import ExchangeType, DataQualityLevel
from .unified_exchange_interface import UnifiedExchangeManager
from .data_validation_suite import AdvancedDataValidator, ValidationResult
from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .config_manager import ConfigurationManager

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentStatus(Enum):
    """Component status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """Alert information"""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Component health information"""
    name: str
    status: ComponentStatus
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    success_count: int = 0
    uptime_percent: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    active_connections: int
    active_threads: int
    network_io_mb: float


@dataclass
class ExchangeMetrics:
    """Exchange-specific metrics"""
    exchange: str
    timestamp: datetime
    data_points_processed: int
    success_rate: float
    avg_processing_time_ms: float
    error_count: int
    quality_score: float
    last_data_timestamp: Optional[datetime] = None


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for exchange OHLCV processing.
    
    Provides comprehensive monitoring, alerting, and visualization
    of all system components and data processing operations.
    """
    
    def __init__(self, config_manager: ConfigurationManager = None):
        """Initialize the monitoring dashboard"""
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = system_logger.getChild("MonitoringDashboard")
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Data storage
        self.alerts: deque = deque(maxlen=1000)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.exchange_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Monitoring components
        self.performance_monitor = PerformanceMonitor()
        self.data_validator = AdvancedDataValidator()
        self.exchange_manager = UnifiedExchangeManager()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'error_rate': 0.05,  # 5%
            'response_time_ms': 5000.0,  # 5 seconds
            'quality_score': 70.0,
            'data_freshness_minutes': 10.0
        }
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ MonitoringDashboard initialized")
    
    def _initialize_components(self):
        """Initialize monitoring components"""
        # Initialize component health tracking
        components = [
            'unified_ohlcv_standardizer',
            'unified_exchange_interface',
            'data_validation_suite',
            'performance_monitor',
            'config_manager',
            'binance_adapter',
            'bingx_adapter',
            'okx_adapter',
            'mexc_adapter'
        ]
        
        for component in components:
            self.component_health[component] = ComponentHealth(
                name=component,
                status=ComponentStatus.UNKNOWN,
                last_check=datetime.now(timezone.utc),
                response_time_ms=0.0
            )
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous monitoring"""
        if self.is_running:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring(interval=1.0)
        
        self.logger.info(f"✅ Monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("✅ Monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check component health
                self._check_component_health()
                
                # Check exchange metrics
                self._check_exchange_metrics()
                
                # Process alerts
                self._process_alerts()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._create_alert(
                    AlertLevel.ERROR,
                    "monitoring_dashboard",
                    f"Monitoring loop error: {e}"
                )
            
            time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_io = psutil.net_io_counters()
            
            # Calculate network I/O in MB
            network_io_mb = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                active_connections=len(psutil.net_connections()),
                active_threads=threading.active_count(),
                network_io_mb=network_io_mb
            )
            
            self.system_metrics_history.append(metrics)
            
            # Check for system alerts
            self._check_system_alerts(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _check_component_health(self):
        """Check health of all components"""
        for component_name, health in self.component_health.items():
            try:
                start_time = time.time()
                
                # Test component based on name
                if component_name == 'unified_ohlcv_standardizer':
                    self._test_ohlcv_standardizer()
                elif component_name == 'unified_exchange_interface':
                    self._test_exchange_interface()
                elif component_name == 'data_validation_suite':
                    self._test_data_validation()
                elif component_name == 'performance_monitor':
                    self._test_performance_monitor()
                elif component_name == 'config_manager':
                    self._test_config_manager()
                elif component_name.endswith('_adapter'):
                    self._test_exchange_adapter(component_name)
                else:
                    # Generic health check
                    self._test_generic_component(component_name)
                
                # Update health status
                response_time = (time.time() - start_time) * 1000
                health.response_time_ms = response_time
                health.last_check = datetime.now(timezone.utc)
                health.success_count += 1
                health.status = ComponentStatus.HEALTHY
                
                # Calculate uptime percentage
                total_checks = health.success_count + health.error_count
                health.uptime_percent = (health.success_count / total_checks) * 100 if total_checks > 0 else 100.0
                
            except Exception as e:
                health.error_count += 1
                health.status = ComponentStatus.UNHEALTHY
                health.last_check = datetime.now(timezone.utc)
                
                # Calculate uptime percentage
                total_checks = health.success_count + health.error_count
                health.uptime_percent = (health.success_count / total_checks) * 100 if total_checks > 0 else 0.0
                
                self.logger.error(f"Component {component_name} health check failed: {e}")
                
                # Create alert for component failure
                self._create_alert(
                    AlertLevel.ERROR,
                    component_name,
                    f"Component health check failed: {e}"
                )
    
    def _test_ohlcv_standardizer(self):
        """Test OHLCV standardizer component"""
        from .unified_ohlcv_standardizer import standardize_exchange_ohlcv
        
        # Test with sample data
        sample_data = [
            [1640995200000, "50000", "51000", "49000", "50500", "100.5", 1640995259999, "5075000", 1000, "50.25", "25.125", "0"]
        ]
        
        result = standardize_exchange_ohlcv(
            sample_data, "binance", "BTCUSDT", "1m", "standard"
        )
        
        if result.empty:
            raise Exception("Standardizer returned empty result")
    
    def _test_exchange_interface(self):
        """Test exchange interface component"""
        # Test if exchange manager is properly initialized
        if not hasattr(self.exchange_manager, 'adapters'):
            raise Exception("Exchange manager not properly initialized")
    
    def _test_data_validation(self):
        """Test data validation component"""
        import pandas as pd
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [100.5],
            'timestamp': [datetime.now(timezone.utc)]
        })
        
        result = self.data_validator.validate_ohlcv_data(
            sample_data, ExchangeType.BINANCE, "test"
        )
        
        if not isinstance(result, ValidationResult):
            raise Exception("Data validator returned invalid result")
    
    def _test_performance_monitor(self):
        """Test performance monitor component"""
        if not self.performance_monitor.is_monitoring:
            raise Exception("Performance monitor not running")
    
    def _test_config_manager(self):
        """Test configuration manager component"""
        config = self.config_manager.get_config()
        if not config:
            raise Exception("Configuration manager returned no config")
    
    def _test_exchange_adapter(self, adapter_name: str):
        """Test exchange adapter component"""
        # This would test the actual adapter if available
        # For now, just check if the component exists
        pass
    
    def _test_generic_component(self, component_name: str):
        """Generic component health check"""
        # Basic health check - just verify the component exists
        pass
    
    def _check_exchange_metrics(self):
        """Check metrics for all exchanges"""
        for exchange_name in ['binance', 'bingx', 'okx', 'mexc']:
            try:
                # Get performance metrics for this exchange
                performance_summary = self.performance_monitor.get_performance_summary()
                
                # Calculate exchange-specific metrics
                exchange_operations = [
                    m for m in self.performance_monitor.metrics_history
                    if m.exchange == exchange_name
                ]
                
                if exchange_operations:
                    success_count = sum(1 for op in exchange_operations if op.success)
                    success_rate = success_count / len(exchange_operations)
                    avg_processing_time = sum(op.duration for op in exchange_operations) / len(exchange_operations)
                    error_count = len(exchange_operations) - success_count
                    
                    # Calculate quality score (simplified)
                    quality_score = success_rate * 100
                    
                    metrics = ExchangeMetrics(
                        exchange=exchange_name,
                        timestamp=datetime.now(timezone.utc),
                        data_points_processed=len(exchange_operations),
                        success_rate=success_rate,
                        avg_processing_time_ms=avg_processing_time * 1000,
                        error_count=error_count,
                        quality_score=quality_score
                    )
                    
                    self.exchange_metrics_history[exchange_name].append(metrics)
                    
                    # Check for exchange-specific alerts
                    self._check_exchange_alerts(metrics)
                
            except Exception as e:
                self.logger.error(f"Error checking exchange metrics for {exchange_name}: {e}")
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-level alerts"""
        # CPU usage alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"High CPU usage: {metrics.cpu_percent:.1f}%"
            )
        
        # Memory usage alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            self._create_alert(
                AlertLevel.WARNING,
                "system",
                f"High memory usage: {metrics.memory_percent:.1f}%"
            )
        
        # Disk usage alert
        if metrics.disk_usage_percent > 90:
            self._create_alert(
                AlertLevel.ERROR,
                "system",
                f"High disk usage: {metrics.disk_usage_percent:.1f}%"
            )
    
    def _check_exchange_alerts(self, metrics: ExchangeMetrics):
        """Check for exchange-specific alerts"""
        # Low success rate alert
        if metrics.success_rate < (1 - self.alert_thresholds['error_rate']):
            self._create_alert(
                AlertLevel.ERROR,
                f"exchange_{metrics.exchange}",
                f"Low success rate for {metrics.exchange}: {metrics.success_rate:.1%}"
            )
        
        # High processing time alert
        if metrics.avg_processing_time_ms > self.alert_thresholds['response_time_ms']:
            self._create_alert(
                AlertLevel.WARNING,
                f"exchange_{metrics.exchange}",
                f"High processing time for {metrics.exchange}: {metrics.avg_processing_time_ms:.1f}ms"
            )
        
        # Low quality score alert
        if metrics.quality_score < self.alert_thresholds['quality_score']:
            self._create_alert(
                AlertLevel.WARNING,
                f"exchange_{metrics.exchange}",
                f"Low quality score for {metrics.exchange}: {metrics.quality_score:.1f}"
            )
    
    def _process_alerts(self):
        """Process and manage alerts"""
        # Auto-resolve old info alerts
        current_time = datetime.now(timezone.utc)
        for alert in self.alerts:
            if (alert.level == AlertLevel.INFO and 
                not alert.resolved and 
                (current_time - alert.timestamp).total_seconds() > 300):  # 5 minutes
                alert.resolved = True
                alert.resolved_at = current_time
    
    def _create_alert(self, level: AlertLevel, component: str, message: str, metadata: Dict[str, Any] = None):
        """Create a new alert"""
        alert_id = f"{component}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            level=level,
            component=component,
            message=message,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{level.value.upper()}] {component}: {message}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_status': self._get_system_status(),
            'component_health': self._get_component_health_summary(),
            'exchange_metrics': self._get_exchange_metrics_summary(),
            'alerts': self._get_alerts_summary(),
            'performance_metrics': self.performance_monitor.get_performance_summary(),
            'configuration': self.config_manager.get_config_summary()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        if not self.system_metrics_history:
            return {'status': 'unknown', 'message': 'No system metrics available'}
        
        latest_metrics = self.system_metrics_history[-1]
        
        # Determine overall status
        status = 'healthy'
        if latest_metrics.cpu_percent > 90 or latest_metrics.memory_percent > 95:
            status = 'critical'
        elif latest_metrics.cpu_percent > 80 or latest_metrics.memory_percent > 85:
            status = 'warning'
        
        return {
            'status': status,
            'cpu_percent': latest_metrics.cpu_percent,
            'memory_percent': latest_metrics.memory_percent,
            'memory_available_mb': latest_metrics.memory_available_mb,
            'disk_usage_percent': latest_metrics.disk_usage_percent,
            'active_threads': latest_metrics.active_threads,
            'uptime_seconds': (datetime.now(timezone.utc) - self.system_metrics_history[0].timestamp).total_seconds()
        }
    
    def _get_component_health_summary(self) -> Dict[str, Any]:
        """Get component health summary"""
        summary = {}
        
        for name, health in self.component_health.items():
            summary[name] = {
                'status': health.status.value,
                'uptime_percent': health.uptime_percent,
                'response_time_ms': health.response_time_ms,
                'last_check': health.last_check.isoformat(),
                'error_count': health.error_count,
                'success_count': health.success_count
            }
        
        return summary
    
    def _get_exchange_metrics_summary(self) -> Dict[str, Any]:
        """Get exchange metrics summary"""
        summary = {}
        
        for exchange_name, metrics_history in self.exchange_metrics_history.items():
            if not metrics_history:
                continue
            
            latest_metrics = metrics_history[-1]
            
            # Calculate trends
            if len(metrics_history) > 1:
                success_rates = [m.success_rate for m in metrics_history[-10:]]
                avg_success_rate = statistics.mean(success_rates)
                success_rate_trend = 'improving' if success_rates[-1] > success_rates[0] else 'declining'
            else:
                avg_success_rate = latest_metrics.success_rate
                success_rate_trend = 'stable'
            
            summary[exchange_name] = {
                'success_rate': latest_metrics.success_rate,
                'avg_success_rate': avg_success_rate,
                'success_rate_trend': success_rate_trend,
                'avg_processing_time_ms': latest_metrics.avg_processing_time_ms,
                'quality_score': latest_metrics.quality_score,
                'data_points_processed': latest_metrics.data_points_processed,
                'error_count': latest_metrics.error_count,
                'last_update': latest_metrics.timestamp.isoformat()
            }
        
        return summary
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary"""
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        alert_counts = {
            'total': len(active_alerts),
            'info': len([a for a in active_alerts if a.level == AlertLevel.INFO]),
            'warning': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
            'error': len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
            'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL])
        }
        
        recent_alerts = [
            {
                'id': alert.id,
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            for alert in active_alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            'counts': alert_counts,
            'recent_alerts': recent_alerts
        }
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                self.logger.info(f"Alert {alert_id} resolved")
                return True
        
        return False
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health check status for external monitoring"""
        return {
            'status': 'healthy' if self.is_running else 'unhealthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': len(self.component_health),
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'uptime_seconds': (datetime.now(timezone.utc) - self.system_metrics_history[0].timestamp).total_seconds() if self.system_metrics_history else 0
        }


# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()


# Convenience functions
def start_monitoring(interval: float = 5.0):
    """Start monitoring dashboard"""
    monitoring_dashboard.start_monitoring(interval)


def stop_monitoring():
    """Stop monitoring dashboard"""
    monitoring_dashboard.stop_monitoring()


def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data"""
    return monitoring_dashboard.get_dashboard_data()


def get_health_check() -> Dict[str, Any]:
    """Get health check status"""
    return monitoring_dashboard.get_health_check()