"""
Monitoring manager for pipeline components.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
import queue
from dataclasses import dataclass, asdict
from enum import Enum
import os
import hashlib


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert information."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    message: str
    context: Dict[str, Any]
    source: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class Metric:
    """Metric information."""
    name: str
    value: Any
    timestamp: datetime
    tags: Dict[str, str]
    unit: Optional[str] = None


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert."""
        raise NotImplementedError("Subclasses must implement send_notification")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', '')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
        
    def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.enabled or not self.to_emails:
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] Pipeline Alert: {alert.message[:50]}"
            
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email notification: {e}")
            return False
            
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert information for email body."""
        body = f"""
Pipeline Alert

Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Message: {alert.message}
Source: {alert.source}

Context:
{json.dumps(alert.context, indent=2, default=str)}

---
This is an automated alert from the Ares Trading Pipeline.
        """
        return body.strip()


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url', '')
        self.channel = config.get('channel', '#general')
        self.username = config.get('username', 'Pipeline Monitor')
        
    def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.enabled or not self.webhook_url:
            return False
            
        try:
            color_map = {
                AlertSeverity.INFO: '#36a64f',
                AlertSeverity.WARNING: '#ffcc00',
                AlertSeverity.ERROR: '#ff6b6b',
                AlertSeverity.CRITICAL: '#cc0000'
            }
            
            payload = {
                'channel': self.channel,
                'username': self.username,
                'attachments': [{
                    'color': color_map.get(alert.severity, '#cccccc'),
                    'title': f"Pipeline Alert: {alert.message}",
                    'text': f"*Severity:* {alert.severity.value.upper()}\n*Source:* {alert.source}\n*Time:* {alert.timestamp.strftime('%H:%M:%S')}",
                    'fields': [
                        {
                            'title': 'Context',
                            'value': json.dumps(alert.context, indent=2, default=str),
                            'short': False
                        }
                    ],
                    'footer': 'Ares Trading Pipeline',
                    'ts': int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url', '')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 10)
        
    def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.enabled or not self.webhook_url:
            return False
            
        try:
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'message': alert.message,
                'source': alert.source,
                'context': alert.context
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to send webhook notification: {e}")
            return False


class MonitoringManager:
    """
    Manages monitoring and observability of pipeline components.
    
    This class handles metrics collection, logging, alerting, and
    performance monitoring for the trading pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MonitoringManager.
        
        Args:
            config: Configuration dictionary for monitoring
        """
        self.config = config or {}
        self.metrics: Dict[str, List[Metric]] = {}
        self.alerts: List[Alert] = []
        self.logger = self._setup_logger()
        self.notification_channels: List[NotificationChannel] = []
        self._setup_notification_channels()
        self._alert_queue = queue.Queue()
        self._notification_thread = None
        self._start_notification_worker()
        self._metrics_retention_days = self.config.get('metrics_retention_days', 30)
        self._alerts_retention_days = self.config.get('alerts_retention_days', 90)
        self._cleanup_thread = None
        self._start_cleanup_worker()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _setup_notification_channels(self) -> None:
        """Set up notification channels based on configuration."""
        channels_config = self.config.get('notification_channels', {})
        
        # Email channel
        if 'email' in channels_config:
            self.notification_channels.append(
                EmailNotificationChannel(channels_config['email'])
            )
            
        # Slack channel
        if 'slack' in channels_config:
            self.notification_channels.append(
                SlackNotificationChannel(channels_config['slack'])
            )
            
        # Webhook channel
        if 'webhook' in channels_config:
            self.notification_channels.append(
                WebhookNotificationChannel(channels_config['webhook'])
            )
            
        self.logger.info(f"Set up {len(self.notification_channels)} notification channels")
        
    def _start_notification_worker(self) -> None:
        """Start background thread for sending notifications."""
        self._notification_thread = threading.Thread(
            target=self._notification_worker_loop,
            name="notification-worker",
            daemon=True
        )
        self._notification_thread.start()
        
    def _notification_worker_loop(self) -> None:
        """Background loop for processing notification queue."""
        while True:
            try:
                alert = self._alert_queue.get(timeout=1)
                if alert is None:  # Shutdown signal
                    break
                    
                self._send_notifications(alert)
                self._alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in notification worker: {e}")
                
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications through all channels."""
        for channel in self.notification_channels:
            try:
                success = channel.send_notification(alert)
                if success:
                    self.logger.debug(f"Notification sent successfully via {channel.__class__.__name__}")
                else:
                    self.logger.warning(f"Failed to send notification via {channel.__class__.__name__}")
            except Exception as e:
                self.logger.error(f"Error sending notification via {channel.__class__.__name__}: {e}")
                
    def _start_cleanup_worker(self) -> None:
        """Start background thread for cleanup operations."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker_loop,
            name="cleanup-worker",
            daemon=True
        )
        self._cleanup_thread.start()
        
    def _cleanup_worker_loop(self) -> None:
        """Background loop for cleanup operations."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                
    def _cleanup_old_data(self) -> None:
        """Clean up old metrics and alerts."""
        cutoff_time = datetime.now() - timedelta(days=self._metrics_retention_days)
        
        # Clean up old metrics
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if m.timestamp > cutoff_time
            ]
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]
                
        # Clean up old alerts
        cutoff_time_alerts = datetime.now() - timedelta(days=self._alerts_retention_days)
        self.alerts = [
            a for a in self.alerts
            if a.timestamp > cutoff_time_alerts
        ]
        
        self.logger.debug(f"Cleanup completed. Metrics: {len(self.metrics)}, Alerts: {len(self.alerts)}")
        
    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            unit: Optional unit for the metric
        """
        timestamp = datetime.now()
        if name not in self.metrics:
            self.metrics[name] = []
            
        metric = Metric(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
        
        # Check for metric-based alerts
        self._check_metric_alerts(name, value, tags)
        
    def _check_metric_alerts(self, name: str, value: Any, tags: Optional[Dict[str, str]]) -> None:
        """Check if metric value triggers any alerts."""
        metric_alerts = self.config.get('metric_alerts', {})
        
        if name in metric_alerts:
            alert_config = metric_alerts[name]
            threshold = alert_config.get('threshold')
            operator = alert_config.get('operator', '>')
            severity = AlertSeverity(alert_config.get('severity', 'warning'))
            
            should_alert = False
            if operator == '>' and value > threshold:
                should_alert = True
            elif operator == '<' and value < threshold:
                should_alert = True
            elif operator == '>=' and value >= threshold:
                should_alert = True
            elif operator == '<=' and value <= threshold:
                should_alert = True
            elif operator == '==' and value == threshold:
                should_alert = True
            elif operator == '!=' and value != threshold:
                should_alert = True
                
            if should_alert:
                message = f"Metric {name} value {value} {operator} {threshold}"
                self.create_alert(
                    severity=severity.value,
                    message=message,
                    context={'metric_name': name, 'value': value, 'threshold': threshold, 'operator': operator}
                )
        
    def get_metric(self, name: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Metric]:
        """
        Get metric values by name with optional time filtering.
        
        Args:
            name: Metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of metric entries
        """
        if name not in self.metrics:
            return []
            
        metrics = self.metrics[name]
        
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
            
        return sorted(metrics, key=lambda x: x.timestamp)
        
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get summary statistics for a metric over a time window.
        
        Args:
            name: Metric name
            window_minutes: Time window in minutes
            
        Returns:
            Summary statistics
        """
        if name not in self.metrics:
            return {}
            
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics[name] if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
            
        values = [m.value for m in recent_metrics if isinstance(m.value, (int, float))]
        
        if not values:
            return {}
            
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': recent_metrics[-1].value,
            'window_minutes': window_minutes
        }
        
    def log_info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
        
    def log_error(self, message: str, error: Optional[Exception] = None) -> None:
        """
        Log an error message.
        
        Args:
            message: Error message
            error: Optional exception object
        """
        if error:
            self.logger.error(f"{message}: {error}")
        else:
            self.logger.error(message)
            
    def create_alert(self, severity: str, message: str, context: Optional[Dict[str, Any]] = None, source: str = "pipeline") -> str:
        """
        Create an alert.
        
        Args:
            severity: Alert severity level
            message: Alert message
            context: Optional context information
            source: Source of the alert
            
        Returns:
            Alert ID
        """
        try:
            severity_enum = AlertSeverity(severity.lower())
        except ValueError:
            severity_enum = AlertSeverity.WARNING
            self.logger.warning(f"Invalid severity '{severity}', defaulting to 'warning'")
            
        alert_id = self._generate_alert_id(message, source)
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity_enum,
            message=message,
            context=context or {},
            source=source
        )
        
        self.alerts.append(alert)
        
        # Add to notification queue
        self._alert_queue.put(alert)
        
        self.logger.warning(f"ALERT [{severity_enum.value.upper()}] from {source}: {message}")
        
        return alert_id
        
    def _generate_alert_id(self, message: str, source: str) -> str:
        """Generate a unique alert ID."""
        timestamp = datetime.now().isoformat()
        content = f"{source}:{message}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
        
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Name of person acknowledging
            
        Returns:
            True if alert was acknowledged successfully
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
        
    def get_alerts(self, severity: Optional[str] = None, source: Optional[str] = None, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get alerts with optional filtering.
        
        Args:
            severity: Optional severity filter
            source: Optional source filter
            acknowledged: Optional acknowledgment filter
            
        Returns:
            List of alerts
        """
        filtered_alerts = self.alerts
        
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity_enum]
            except ValueError:
                pass
                
        if source:
            filtered_alerts = [a for a in filtered_alerts if a.source == source]
            
        if acknowledged is not None:
            filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]
            
        return [asdict(alert) for alert in filtered_alerts]
        
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of all alerts.
        
        Returns:
            Alert summary statistics
        """
        total_alerts = len(self.alerts)
        unacknowledged = len([a for a in self.alerts if not a.acknowledged])
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([a for a in self.alerts if a.severity == severity])
            
        source_counts = {}
        for alert in self.alerts:
            source_counts[alert.source] = source_counts.get(alert.source, 0) + 1
            
        return {
            'total_alerts': total_alerts,
            'unacknowledged_alerts': unacknowledged,
            'severity_distribution': severity_counts,
            'source_distribution': source_counts,
            'recent_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)])
        }
        
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """
        Add a new notification channel.
        
        Args:
            channel: Notification channel to add
        """
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {channel.__class__.__name__}")
        
    def remove_notification_channel(self, channel_index: int) -> bool:
        """
        Remove a notification channel.
        
        Args:
            channel_index: Index of the channel to remove
            
        Returns:
            True if channel was removed successfully
        """
        if 0 <= channel_index < len(self.notification_channels):
            removed_channel = self.notification_channels.pop(channel_index)
            self.logger.info(f"Removed notification channel: {removed_channel.__class__.__name__}")
            return True
        return False
        
    def test_notification_channels(self) -> Dict[str, bool]:
        """
        Test all notification channels.
        
        Returns:
            Dictionary mapping channel names to test results
        """
        test_alert = Alert(
            id="test",
            timestamp=datetime.now(),
            severity=AlertSeverity.INFO,
            message="Test notification",
            context={'test': True},
            source="test"
        )
        
        results = {}
        for i, channel in enumerate(self.notification_channels):
            try:
                success = channel.send_notification(test_alert)
                results[f"{channel.__class__.__name__}_{i}"] = success
            except Exception as e:
                results[f"{channel.__class__.__name__}_{i}"] = False
                
        return results
        
    def export_metrics(self, format: str = 'json', file_path: Optional[str] = None) -> str:
        """
        Export metrics to file.
        
        Args:
            format: Export format ('json' or 'csv')
            file_path: Optional file path, defaults to timestamped filename
            
        Returns:
            Path to exported file
        """
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"metrics_export_{timestamp}.{format}"
            
        if format.lower() == 'json':
            export_data = {}
            for name, metrics in self.metrics.items():
                export_data[name] = [asdict(m) for m in metrics]
                
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format.lower() == 'csv':
            import pandas as pd
            all_metrics = []
            for name, metrics in self.metrics.items():
                for metric in metrics:
                    all_metrics.append({
                        'metric_name': name,
                        'value': metric.value,
                        'timestamp': metric.timestamp,
                        'tags': json.dumps(metric.tags),
                        'unit': metric.unit
                    })
                    
            df = pd.DataFrame(all_metrics)
            df.to_csv(file_path, index=False)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Metrics exported to: {file_path}")
        return file_path
        
    def shutdown(self) -> None:
        """Shutdown the monitoring manager."""
        # Signal notification worker to stop
        self._alert_queue.put(None)
        
        # Wait for notification worker to finish
        if self._notification_thread and self._notification_thread.is_alive():
            self._notification_thread.join(timeout=5)
            
        self.logger.info("Monitoring manager shutdown complete")

