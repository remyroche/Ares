"""
Real-Time Performance Monitoring and Alerting System

This module provides comprehensive real-time monitoring, alerting, and
performance tracking for A/B/C testing with advanced analytics and
automated response capabilities.

Key Features:
- Real-time performance monitoring
- Advanced alerting system with multiple channels
- Performance analytics and trend analysis
- Automated risk management
- System health monitoring
- Customizable dashboards
- Historical performance tracking
- Predictive analytics
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
import json
import threading
from queue import Queue, PriorityQueue
import uuid
from collections import deque, defaultdict
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.performance_utils import PerformanceMonitor
from src.utils.monitoring_utils import SystemMonitor

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TELEGRAM = "telegram"
    SMS = "sms"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    BUSINESS = "business"
    CUSTOM = "custom"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    metric_type: MetricType
    condition: str  # e.g., ">", "<", "==", "!=", ">=", "<="
    threshold: float
    alert_level: AlertLevel
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 15
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    metric_id: str
    name: str
    value: float
    timestamp: datetime
    model_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    rule_id: str
    rule_name: str
    alert_level: AlertLevel
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    model_id: Optional[str] = None
    channels: List[AlertChannel] = field(default_factory=list)
    delivered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    # Basic settings
    monitoring_interval: int = 30  # seconds
    data_retention_days: int = 30
    max_metrics_per_model: int = 1000
    
    # Alerting settings
    enable_alerting: bool = True
    alert_cooldown_minutes: int = 15
    max_alerts_per_hour: int = 100
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.15,
        'min_sharpe_ratio': 0.5,
        'max_volatility': 0.3,
        'min_win_rate': 0.4,
        'max_correlation': 0.8
    })
    
    # System monitoring
    enable_system_monitoring: bool = True
    system_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'disk_usage': 90.0,
        'gpu_usage': 95.0
    })
    
    # Data storage
    enable_data_persistence: bool = True
    data_storage_path: str = "data/monitoring"
    
    # Email settings
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    
    # Webhook settings
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Advanced metric collection system."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize metric collector."""
        self.config = config
        self.logger = logger.getChild('MetricCollector')
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.max_metrics_per_model))
        self.model_metrics: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=config.max_metrics_per_model)))
        
        # Performance tracking
        self.collection_stats = {
            'total_metrics_collected': 0,
            'metrics_per_second': 0.0,
            'last_collection_time': None,
            'collection_errors': 0
        }
        
        self.logger.info("🚀 MetricCollector initialized")
        self.logger.info(f"📊 Max metrics per model: {config.max_metrics_per_model}")
        self.logger.info(f"📊 Data retention: {config.data_retention_days} days")
    
    def collect_metric(self, metric: PerformanceMetric) -> None:
        """Collect a performance metric."""
        try:
            # Store metric
            self.metrics[metric.name].append(metric)
            
            # Store model-specific metric
            if metric.model_id:
                self.model_metrics[metric.model_id][metric.name].append(metric)
            
            # Update statistics
            self.collection_stats['total_metrics_collected'] += 1
            self.collection_stats['last_collection_time'] = datetime.now()
            
            # Calculate metrics per second
            if self.collection_stats['last_collection_time']:
                time_diff = (datetime.now() - self.collection_stats['last_collection_time']).total_seconds()
                if time_diff > 0:
                    self.collection_stats['metrics_per_second'] = 1.0 / time_diff
            
            self.logger.debug(f"📊 Collected metric: {metric.name} = {metric.value}")
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(f"❌ Error collecting metric {metric.name}: {e}")
    
    def get_metric_history(self, metric_name: str, model_id: Optional[str] = None, 
                          hours: int = 24) -> List[PerformanceMetric]:
        """Get metric history for a specific time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            if model_id and model_id in self.model_metrics:
                metrics = self.model_metrics[model_id][metric_name]
            else:
                metrics = self.metrics[metric_name]
            
            # Filter by time
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            return recent_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error getting metric history for {metric_name}: {e}")
            return []
    
    def get_metric_statistics(self, metric_name: str, model_id: Optional[str] = None, 
                            hours: int = 24) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        try:
            metrics = self.get_metric_history(metric_name, model_id, hours)
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'percentile_25': np.percentile(values, 25),
                'percentile_75': np.percentile(values, 75),
                'percentile_95': np.percentile(values, 95),
                'latest': values[-1] if values else 0.0,
                'trend': self._calculate_trend(values)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating statistics for {metric_name}: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.collection_stats.copy()


class AlertManager:
    """Advanced alerting system."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize alert manager."""
        self.config = config
        self.logger = logger.getChild('AlertManager')
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Alert history
        self.alert_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        
        # Delivery tracking
        self.delivery_stats = {
            'total_alerts_sent': 0,
            'delivery_success_rate': 0.0,
            'delivery_errors': 0,
            'alerts_by_level': defaultdict(int),
            'alerts_by_channel': defaultdict(int)
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        self.logger.info("🚀 AlertManager initialized")
        self.logger.info(f"📊 Alert rules configured: {len(self.alert_rules)}")
        self.logger.info(f"📊 Alerting enabled: {config.enable_alerting}")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_drawdown",
                name="High Drawdown Alert",
                description="Alert when drawdown exceeds threshold",
                metric_name="max_drawdown",
                metric_type=MetricType.RISK,
                condition=">",
                threshold=self.config.performance_thresholds.get('max_drawdown', 0.15),
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="low_sharpe",
                name="Low Sharpe Ratio Alert",
                description="Alert when Sharpe ratio falls below threshold",
                metric_name="sharpe_ratio",
                metric_type=MetricType.PERFORMANCE,
                condition="<",
                threshold=self.config.performance_thresholds.get('min_sharpe_ratio', 0.5),
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="high_volatility",
                name="High Volatility Alert",
                description="Alert when volatility exceeds threshold",
                metric_name="volatility",
                metric_type=MetricType.RISK,
                condition=">",
                threshold=self.config.performance_thresholds.get('max_volatility', 0.3),
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="low_win_rate",
                name="Low Win Rate Alert",
                description="Alert when win rate falls below threshold",
                metric_name="win_rate",
                metric_type=MetricType.PERFORMANCE,
                condition="<",
                threshold=self.config.performance_thresholds.get('min_win_rate', 0.4),
                alert_level=AlertLevel.WARNING,
                channels=[AlertChannel.LOG]
            ),
            AlertRule(
                rule_id="high_correlation",
                name="High Correlation Alert",
                description="Alert when model correlation exceeds threshold",
                metric_name="correlation",
                metric_type=MetricType.RISK,
                condition=">",
                threshold=self.config.performance_thresholds.get('max_correlation', 0.8),
                alert_level=AlertLevel.INFO,
                channels=[AlertChannel.LOG]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        # Add system monitoring rules if enabled
        if self.config.enable_system_monitoring:
            system_rules = [
                AlertRule(
                    rule_id="high_cpu",
                    name="High CPU Usage Alert",
                    description="Alert when CPU usage exceeds threshold",
                    metric_name="cpu_usage",
                    metric_type=MetricType.SYSTEM,
                    condition=">",
                    threshold=self.config.system_thresholds.get('cpu_usage', 80.0),
                    alert_level=AlertLevel.WARNING,
                    channels=[AlertChannel.LOG]
                ),
                AlertRule(
                    rule_id="high_memory",
                    name="High Memory Usage Alert",
                    description="Alert when memory usage exceeds threshold",
                    metric_name="memory_usage",
                    metric_type=MetricType.SYSTEM,
                    condition=">",
                    threshold=self.config.system_thresholds.get('memory_usage', 85.0),
                    alert_level=AlertLevel.WARNING,
                    channels=[AlertChannel.LOG]
                ),
                AlertRule(
                    rule_id="high_disk",
                    name="High Disk Usage Alert",
                    description="Alert when disk usage exceeds threshold",
                    metric_name="disk_usage",
                    metric_type=MetricType.SYSTEM,
                    condition=">",
                    threshold=self.config.system_thresholds.get('disk_usage', 90.0),
                    alert_level=AlertLevel.ERROR,
                    channels=[AlertChannel.LOG]
                )
            ]
            
            for rule in system_rules:
                self.alert_rules[rule.rule_id] = rule
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule."""
        try:
            self.alert_rules[rule.rule_id] = rule
            self.logger.info(f"✅ Added alert rule: {rule.name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error adding alert rule {rule.name}: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"🗑️ Removed alert rule: {rule_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Alert rule not found: {rule_id}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error removing alert rule {rule_id}: {e}")
            return False
    
    def check_alerts(self, metric: PerformanceMetric) -> List[Alert]:
        """Check if metric triggers any alerts."""
        triggered_alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_name != metric.name:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last.total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            # Check condition
            if self._evaluate_condition(metric.value, rule.condition, rule.threshold):
                # Create alert
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule_id,
                    rule_name=rule.name,
                    alert_level=rule.alert_level,
                    message=f"{rule.description}: {metric.name} = {metric.value:.4f} {rule.condition} {rule.threshold}",
                    metric_name=metric.name,
                    metric_value=metric.value,
                    threshold=rule.threshold,
                    timestamp=datetime.now(),
                    model_id=metric.model_id,
                    channels=rule.channels
                )
                
                triggered_alerts.append(alert)
                
                # Update rule
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1
                
                self.logger.warning(f"🚨 Alert triggered: {rule.name} - {alert.message}")
        
        return triggered_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return abs(value - threshold) < 1e-6
            elif condition == "!=":
                return abs(value - threshold) >= 1e-6
            else:
                self.logger.error(f"❌ Unknown condition: {condition}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Error evaluating condition {condition}: {e}")
            return False
    
    async def deliver_alert(self, alert: Alert) -> bool:
        """Deliver alert through configured channels."""
        success = True
        
        for channel in alert.channels:
            try:
                if channel == AlertChannel.LOG:
                    await self._deliver_log_alert(alert)
                elif channel == AlertChannel.EMAIL:
                    await self._deliver_email_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._deliver_webhook_alert(alert)
                else:
                    self.logger.warning(f"⚠️ Unsupported alert channel: {channel}")
                    success = False
                
                # Update delivery stats
                self.delivery_stats['alerts_by_channel'][channel.value] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error delivering alert via {channel}: {e}")
                self.delivery_stats['delivery_errors'] += 1
                success = False
        
        # Update alert status
        alert.delivered = success
        
        # Store in history
        self.alert_history.append(alert)
        
        # Update delivery stats
        self.delivery_stats['total_alerts_sent'] += 1
        self.delivery_stats['alerts_by_level'][alert.alert_level.value] += 1
        
        if success:
            self.delivery_stats['delivery_success_rate'] = (
                (self.delivery_stats['total_alerts_sent'] - self.delivery_stats['delivery_errors']) /
                self.delivery_stats['total_alerts_sent'] * 100
            )
        
        return success
    
    async def _deliver_log_alert(self, alert: Alert) -> None:
        """Deliver alert via logging."""
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.alert_level, logging.WARNING)
        
        self.logger.log(log_level, f"🚨 ALERT: {alert.message}")
    
    async def _deliver_email_alert(self, alert: Alert) -> None:
        """Deliver alert via email."""
        if not self.config.email_enabled or not self.config.email_recipients:
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ", ".join(self.config.email_recipients)
            msg['Subject'] = f"Trading System Alert - {alert.alert_level.value.upper()}"
            
            # Email body
            body = f"""
            Alert Details:
            - Rule: {alert.rule_name}
            - Level: {alert.alert_level.value.upper()}
            - Message: {alert.message}
            - Metric: {alert.metric_name} = {alert.metric_value:.4f}
            - Threshold: {alert.threshold}
            - Model: {alert.model_id or 'N/A'}
            - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Please investigate this alert immediately.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"📧 Email alert sent: {alert.rule_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending email alert: {e}")
            raise
    
    async def _deliver_webhook_alert(self, alert: Alert) -> None:
        """Deliver alert via webhook."""
        if not self.config.webhook_enabled or not self.config.webhook_url:
            return
        
        try:
            import aiohttp
            
            # Prepare webhook payload
            payload = {
                'alert_id': alert.alert_id,
                'rule_name': alert.rule_name,
                'alert_level': alert.alert_level.value,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'model_id': alert.model_id,
                'timestamp': alert.timestamp.isoformat()
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers=self.config.webhook_headers
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"🔗 Webhook alert sent: {alert.rule_name}")
                    else:
                        raise Exception(f"Webhook returned status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"❌ Error sending webhook alert: {e}")
            raise
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for a specific time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get delivery statistics."""
        return self.delivery_stats.copy()


class SystemMonitor:
    """Advanced system monitoring."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize system monitor."""
        self.config = config
        self.logger = logger.getChild('SystemMonitor')
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'gpu_usage': 0.0,
            'network_io': 0.0,
            'process_count': 0,
            'load_average': 0.0
        }
        
        # Process monitoring
        self.process_metrics = {
            'python_processes': 0,
            'memory_per_process': 0.0,
            'cpu_per_process': 0.0
        }
        
        self.logger.info("🚀 SystemMonitor initialized")
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            # CPU usage
            self.system_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics['disk_usage'] = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            self.system_metrics['network_io'] = network.bytes_sent + network.bytes_recv
            
            # Process count
            self.system_metrics['process_count'] = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.system_metrics['load_average'] = load_avg[0]
            except AttributeError:
                self.system_metrics['load_average'] = 0.0
            
            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.system_metrics['gpu_usage'] = gpus[0].load * 100
            except ImportError:
                self.system_metrics['gpu_usage'] = 0.0
            
            # Process-specific metrics
            self._collect_process_metrics()
            
            return self.system_metrics.copy()
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting system metrics: {e}")
            return {}
    
    def _collect_process_metrics(self) -> None:
        """Collect process-specific metrics."""
        try:
            python_processes = []
            total_memory = 0.0
            total_cpu = 0.0
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc)
                        total_memory += proc.info['memory_percent'] or 0.0
                        total_cpu += proc.info['cpu_percent'] or 0.0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.process_metrics['python_processes'] = len(python_processes)
            self.process_metrics['memory_per_process'] = total_memory / len(python_processes) if python_processes else 0.0
            self.process_metrics['cpu_per_process'] = total_cpu / len(python_processes) if python_processes else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting process metrics: {e}")
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            # Weighted health score
            weights = {
                'cpu_usage': 0.3,
                'memory_usage': 0.3,
                'disk_usage': 0.2,
                'gpu_usage': 0.1,
                'load_average': 0.1
            }
            
            health_score = 100.0
            
            for metric, weight in weights.items():
                if metric in self.system_metrics:
                    usage = self.system_metrics[metric]
                    # Convert usage to health (lower is better)
                    health_contribution = max(0, 100 - usage) * weight
                    health_score = min(health_score, health_contribution / weight)
            
            return health_score
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating health score: {e}")
            return 50.0  # Default moderate health


class PerformanceMonitoringSystem:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize performance monitoring system."""
        self.config = config
        self.logger = logger.getChild('PerformanceMonitoringSystem')
        
        # Core components
        self.metric_collector = MetricCollector(config)
        self.alert_manager = AlertManager(config)
        self.system_monitor = SystemMonitor(config)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.system_monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_stats = {
            'total_metrics_collected': 0,
            'total_alerts_triggered': 0,
            'system_health_score': 100.0,
            'uptime': 0.0,
            'start_time': None
        }
        
        # Data persistence
        if config.enable_data_persistence:
            self.data_path = Path(config.data_storage_path)
            ensure_directory(self.data_path)
        
        self.logger.info("🚀 PerformanceMonitoringSystem initialized")
        self.logger.info(f"📊 Monitoring interval: {config.monitoring_interval}s")
        self.logger.info(f"📊 Alerting enabled: {config.enable_alerting}")
        self.logger.info(f"📊 System monitoring: {config.enable_system_monitoring}")
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self.is_running:
            self.logger.warning("⚠️ Monitoring system is already running")
            return
        
        self.logger.info("🚀 Starting PerformanceMonitoringSystem...")
        
        self.is_running = True
        self.performance_stats['start_time'] = datetime.now()
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.config.enable_system_monitoring:
            self.system_monitoring_task = asyncio.create_task(self._system_monitoring_loop())
        
        self.logger.info("✅ PerformanceMonitoringSystem started")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.is_running:
            self.logger.warning("⚠️ Monitoring system is not running")
            return
        
        self.logger.info("🛑 Stopping PerformanceMonitoringSystem...")
        
        self.is_running = False
        
        # Cancel tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.system_monitoring_task:
            self.system_monitoring_task.cancel()
            try:
                await self.system_monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Update uptime
        if self.performance_stats['start_time']:
            uptime = (datetime.now() - self.performance_stats['start_time']).total_seconds()
            self.performance_stats['uptime'] = uptime
        
        self.logger.info("✅ PerformanceMonitoringSystem stopped")
    
    def collect_metric(self, name: str, value: float, model_id: Optional[str] = None, 
                      metric_type: MetricType = MetricType.PERFORMANCE, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Collect a performance metric."""
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            name=name,
            value=value,
            timestamp=datetime.now(),
            model_id=model_id,
            metadata=metadata or {}
        )
        
        self.metric_collector.collect_metric(metric)
        
        # Check for alerts
        if self.config.enable_alerting:
            alerts = self.alert_manager.check_alerts(metric)
            for alert in alerts:
                asyncio.create_task(self.alert_manager.deliver_alert(alert))
                self.performance_stats['total_alerts_triggered'] += 1
    
    def get_metric_history(self, metric_name: str, model_id: Optional[str] = None, 
                          hours: int = 24) -> List[PerformanceMetric]:
        """Get metric history."""
        return self.metric_collector.get_metric_history(metric_name, model_id, hours)
    
    def get_metric_statistics(self, metric_name: str, model_id: Optional[str] = None, 
                            hours: int = 24) -> Dict[str, float]:
        """Get metric statistics."""
        return self.metric_collector.get_metric_statistics(metric_name, model_id, hours)
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add an alert rule."""
        return self.alert_manager.add_alert_rule(rule)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history."""
        return self.alert_manager.get_alert_history(hours)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        system_metrics = self.system_monitor.collect_system_metrics()
        health_score = self.system_monitor.get_system_health_score()
        
        return {
            'health_score': health_score,
            'system_metrics': system_metrics,
            'process_metrics': self.system_monitor.process_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            'is_running': self.is_running,
            'uptime': self.performance_stats['uptime'],
            'performance_stats': self.performance_stats,
            'collection_stats': self.metric_collector.get_collection_stats(),
            'delivery_stats': self.alert_manager.get_delivery_stats(),
            'system_health': self.get_system_health(),
            'alert_rules_count': len(self.alert_manager.alert_rules),
            'active_alerts_count': len(self.alert_manager.active_alerts)
        }
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("🔄 Starting monitoring loop...")
        
        while self.is_running:
            try:
                # Update performance stats
                self.performance_stats['total_metrics_collected'] = self.metric_collector.collection_stats['total_metrics_collected']
                
                # Update uptime
                if self.performance_stats['start_time']:
                    uptime = (datetime.now() - self.performance_stats['start_time']).total_seconds()
                    self.performance_stats['uptime'] = uptime
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
        
        self.logger.info("🛑 Monitoring loop stopped")
    
    async def _system_monitoring_loop(self) -> None:
        """System monitoring loop."""
        self.logger.info("🔄 Starting system monitoring loop...")
        
        while self.is_running:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.collect_system_metrics()
                
                # Create system metrics
                for metric_name, value in system_metrics.items():
                    self.collect_metric(
                        name=metric_name,
                        value=value,
                        metric_type=MetricType.SYSTEM
                    )
                
                # Update health score
                health_score = self.system_monitor.get_system_health_score()
                self.performance_stats['system_health_score'] = health_score
                
                # Wait for next system monitoring cycle
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in system monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
        
        self.logger.info("🛑 System monitoring loop stopped")


# Convenience function for easy integration
async def create_monitoring_system(config: Optional[MonitoringConfig] = None) -> PerformanceMonitoringSystem:
    """Create and initialize a performance monitoring system."""
    if config is None:
        config = MonitoringConfig()
    
    monitoring_system = PerformanceMonitoringSystem(config)
    await monitoring_system.start()
    
    return monitoring_system