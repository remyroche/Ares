"""
Alert Manager

Centralized alert management system for trading operations.
Handles alert creation, routing, escalation, and notification delivery.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.printing import tprint
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..utils.error_handling import (
    TradingError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config

logger = system_logger.getChild('AlertManager')

class AlertType(Enum):
    """Alert types."""
    PRICE = "price"
    VOLUME = "volume"
    REGIME = "regime"
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    ORDER = "order"
    POSITION = "position"

class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Notification channels."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    LOG = "log"
    CONSOLE = "console"

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    alert_type: AlertType
    condition: str  # Simple condition expression
    threshold: float
    priority: AlertPriority
    channels: List[NotificationChannel]
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert information."""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    rule_id: Optional[str] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    notifications_sent: Dict[NotificationChannel, datetime] = field(default_factory=dict)

class NotificationResult:
    """Result of notification delivery."""
    def __init__(self, channel: NotificationChannel, success: bool, message: str = ""):
        self.channel = channel
        self.success = success
        self.message = message
        self.timestamp = datetime.now()

class AlertManager:
    """
    Alert Manager

    Centralized system for managing alerts, routing notifications,
    and handling alert escalation and resolution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.

        Args:
            config: Configuration dictionary
        """
        tprint(f"[ALERT_MGR] __init__: Initializing AlertManager with config keys: {list(config.keys())}")
        self.config = config
        self.logger = logger.getChild('AlertManager')

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Alert aggregation
        self.alert_groups: Dict[str, List[str]] = {}  # group_key -> list of alert_ids
        self.aggregation_window_minutes = config.get('aggregation_window_minutes', 60)

        # Thread safety
        self._lock = asyncio.Lock()

        # Notification settings
        self.notification_channels = config.get('notification_channels', {})
        self.default_channels = config.get('default_channels', [NotificationChannel.LOG])

        # Rate limiting
        self.notification_history: Dict[NotificationChannel, List[datetime]] = {}
        self.cooldowns: Dict[str, datetime] = {}  # rule_id -> last_trigger_time
        self.rate_limits: Dict[NotificationChannel, Dict[str, Any]] = config.get(
            'rate_limits', {
                NotificationChannel.EMAIL: {'max_per_minute': 10, 'max_per_hour': 100},
                NotificationChannel.SMS: {'max_per_minute': 5, 'max_per_hour': 50},
                NotificationChannel.WEBHOOK: {'max_per_minute': 60, 'max_per_hour': 1000},
                NotificationChannel.PUSH: {'max_per_minute': 30, 'max_per_hour': 500},
            }
        )

        # Performance tracking
        self.notification_success_rate = 1.0
        self.avg_response_time = 0.0

        # Escalation settings
        self.escalation_rules = config.get('escalation_rules', {})

        tprint_info("🚨 Initializing Alert Manager...")

    async def initialize(self) -> None:
        """Initialize alert manager."""
        tprint(f"[ALERT_MGR] initialize: Starting AlertManager initialization")
        await self._load_default_rules()
        tprint_success("✅ Alert Manager initialized successfully")
        tprint(f"[ALERT_MGR] initialize: Loaded {len(self.alert_rules)} default alert rules")

    async def _load_default_rules(self) -> None:
        """Load default alert rules."""
        tprint(f"[ALERT_MGR] _load_default_rules: Loading default alert rules")
        # Price movement alerts
        self.alert_rules['price_spike'] = AlertRule(
            rule_id='price_spike',
            alert_type=AlertType.PRICE,
            condition='price_change_percent > threshold',
            threshold=5.0,  # 5% price movement
            priority=AlertPriority.HIGH,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE],
            cooldown_seconds=300
        )

        # Volume spike alerts
        self.alert_rules['volume_spike'] = AlertRule(
            rule_id='volume_spike',
            alert_type=AlertType.VOLUME,
            condition='volume_change_percent > threshold',
            threshold=200.0,  # 200% volume increase
            priority=AlertPriority.MEDIUM,
            channels=[NotificationChannel.LOG],
            cooldown_seconds=600
        )

        # Drawdown alerts
        self.alert_rules['drawdown'] = AlertRule(
            rule_id='drawdown',
            alert_type=AlertType.RISK,
            condition='drawdown_percent > threshold',
            threshold=10.0,  # 10% drawdown
            priority=AlertPriority.URGENT,
            channels=[NotificationChannel.CONSOLE],
            cooldown_seconds=1800
        )

        # Position alerts
        self.alert_rules['position_limit'] = AlertRule(
            rule_id='position_limit',
            alert_type=AlertType.POSITION,
            condition='position_count > threshold',
            threshold=10,  # Max 10 positions
            priority=AlertPriority.MEDIUM,
            channels=[NotificationChannel.LOG],
            cooldown_seconds=900
        )

    @handles_errors
    async def create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        data: Dict[str, Any],
        rule_id: Optional[str] = None
    ) -> str:
        """
        Create a new alert.

        Args:
            alert_type: Type of alert
            priority: Alert priority
            title: Alert title
            message: Alert message
            data: Alert data
            rule_id: Associated rule ID

        Returns:
            Alert ID
        """
        tprint(f"[ALERT_MGR] create_alert: Creating alert type={alert_type.value}, priority={priority.value}, title='{title}', rule_id={rule_id}")
        async with self._lock:
            alert_id = f"alert_{datetime.now().timestamp()}_{len(self.active_alerts)}"

            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                timestamp=datetime.now(),
                data=data,
                rule_id=rule_id
            )

            # Check for duplicate/similar alerts
            duplicate_group = await self._find_duplicate_group(alert)
            if duplicate_group:
                self.alert_groups[duplicate_group].append(alert_id)
                self.logger.debug(f"Alert {alert_id} grouped with existing alerts: {duplicate_group}")
            else:
                # Create new group
                group_key = self._generate_group_key(alert)
                self.alert_groups[group_key] = [alert_id]

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

        # Check if alert should trigger notifications (outside lock)
        if await self._should_send_notifications(alert):
            await self._send_notifications(alert)
            tprint(f"[ALERT_MGR] create_alert: Notifications sent for alert {alert_id}")

        tprint_warning(f"🚨 {priority.value.upper()}: {title}")
        tprint(f"[ALERT_MGR] create_alert: Alert created successfully, alert_id={alert_id}")
        return alert_id

    async def _should_send_notifications(self, alert: Alert) -> bool:
        """Check if alert should trigger notifications."""
        tprint(f"[ALERT_MGR] _should_send_notifications: Checking notification eligibility for alert {alert.alert_id}, rule_id={alert.rule_id}")
        # Check cooldown for rule-based alerts
        if alert.rule_id and alert.rule_id in self.cooldowns:
            last_trigger = self.cooldowns[alert.rule_id]
            rule = self.alert_rules.get(alert.rule_id)
            if rule and (datetime.now() - last_trigger).total_seconds() < rule.cooldown_seconds:
                tprint(f"[ALERT_MGR] _should_send_notifications: Alert in cooldown period, returning False")
                return False

        # Update cooldown
        if alert.rule_id:
            self.cooldowns[alert.rule_id] = datetime.now()
            tprint(f"[ALERT_MGR] _should_send_notifications: Updated cooldown for rule {alert.rule_id}")

        tprint(f"[ALERT_MGR] _should_send_notifications: Notifications approved, returning True")
        return True

    def _generate_group_key(self, alert: Alert) -> str:
        """Generate a key for grouping similar alerts."""
        # Group by type, priority, and rule_id (if present)
        return f"{alert.alert_type.value}:{alert.priority.value}:{alert.rule_id or 'none'}"

    async def _find_duplicate_group(self, alert: Alert) -> Optional[str]:
        """Find if alert belongs to an existing group."""
        group_key = self._generate_group_key(alert)
        now = datetime.now()
        
        # Check existing groups
        for key, alert_ids in self.alert_groups.items():
            if key == group_key:
                # Check if alerts in group are recent
                recent_alerts = [
                    aid for aid in alert_ids
                    if aid in self.active_alerts or
                    (aid in [a.alert_id for a in self.alert_history[-100:]] and
                     (now - next((a.timestamp for a in self.alert_history if a.alert_id == aid), now)).total_seconds() < self.aggregation_window_minutes * 60)
                ]
                if recent_alerts:
                    return key
        
        return None

    async def get_aggregated_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated alert statistics."""
        aggregated = {}
        now = datetime.now()
        
        for group_key, alert_ids in self.alert_groups.items():
            # Count recent alerts
            recent_count = sum(
                1 for alert_id in alert_ids
                if alert_id in self.active_alerts or
                any((now - a.timestamp).total_seconds() < self.aggregation_window_minutes * 60
                    for a in self.alert_history[-100:]
                    if a.alert_id == alert_id)
            )
            
            if recent_count > 0:
                # Get latest alert in group
                latest_alert_id = None
                latest_timestamp = None
                for alert_id in alert_ids:
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        if latest_timestamp is None or alert.timestamp > latest_timestamp:
                            latest_timestamp = alert.timestamp
                            latest_alert_id = alert_id
                
                if latest_alert_id:
                    aggregated[group_key] = {
                        'count': recent_count,
                        'latest_alert_id': latest_alert_id,
                        'latest_timestamp': latest_timestamp.isoformat() if latest_timestamp else None,
                        'alert_ids': alert_ids[:10]  # First 10 alert IDs
                    }
        
        return aggregated

    async def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if rate limit allows sending notification."""
        if channel not in self.rate_limits:
            return True  # No rate limit configured

        limits = self.rate_limits[channel]
        now = datetime.now()
        
        # Clean old history
        if channel in self.notification_history:
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)
            self.notification_history[channel] = [
                ts for ts in self.notification_history[channel]
                if ts > cutoff_hour
            ]

        # Check minute limit
        if channel in self.notification_history:
            recent_minute = [ts for ts in self.notification_history[channel]
                           if ts > now - timedelta(minutes=1)]
            if len(recent_minute) >= limits.get('max_per_minute', float('inf')):
                return False

        # Check hour limit
        if channel in self.notification_history:
            recent_hour = [ts for ts in self.notification_history[channel]
                          if ts > now - timedelta(hours=1)]
            if len(recent_hour) >= limits.get('max_per_hour', float('inf')):
                return False

        return True

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for alert."""
        # Determine channels
        channels = self.default_channels.copy()

        if alert.rule_id and alert.rule_id in self.alert_rules:
            rule = self.alert_rules[alert.rule_id]
            channels = rule.channels

        # Send notifications
        results = []

        for channel in channels:
            try:
                # Check rate limit before sending
                if not await self._check_rate_limit(channel):
                    self.logger.warning(f"Rate limit exceeded for {channel.value}, skipping notification")
                    continue

                result = await self._send_notification(alert, channel)
                results.append(result)
                alert.notifications_sent[channel] = datetime.now()

                # Update rate limiting
                if channel not in self.notification_history:
                    self.notification_history[channel] = []
                self.notification_history[channel].append(datetime.now())

                # Keep only recent history (last 1000)
                if len(self.notification_history[channel]) > 1000:
                    self.notification_history[channel] = self.notification_history[channel][-1000:]

            except Exception as e:
                tprint_error(f"❌ Failed to send {channel.value} notification: {str(e)}")
                results.append(NotificationResult(channel, False, str(e)))

        # Update success rate
        success_count = sum(1 for r in results if r.success)
        if results:
            self.notification_success_rate = success_count / len(results)

    async def _send_notification(self, alert: Alert, channel: NotificationChannel) -> NotificationResult:
        """Send notification via specific channel."""
        try:
            if channel == NotificationChannel.LOG:
                return await self._send_log_notification(alert)
            elif channel == NotificationChannel.CONSOLE:
                return await self._send_console_notification(alert)
            elif channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(alert)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(alert)
            else:
                return NotificationResult(channel, False, f"Unsupported channel: {channel}")

        except Exception as e:
            return NotificationResult(channel, False, str(e))

    async def _send_log_notification(self, alert: Alert) -> NotificationResult:
        """Send notification to log."""
        log_message = f"ALERT [{alert.alert_type.value.upper()}] {alert.title}: {alert.message}"
        logger.warning(log_message)

        # Also print to console for immediate visibility
        tprint_warning(f"🚨 {alert.title}: {alert.message}")

        return NotificationResult(NotificationChannel.LOG, True)

    async def _send_console_notification(self, alert: Alert) -> NotificationResult:
        """Send notification to console."""
        console_message = f"""
🚨 ALERT: {alert.title}
Priority: {alert.priority.value.upper()}
Type: {alert.alert_type.value}
Time: {alert.timestamp}
Message: {alert.message}
Data: {alert.data}
"""
        tprint_structured(console_message.strip())

        return NotificationResult(NotificationChannel.CONSOLE, True)

    async def _send_email_notification(self, alert: Alert) -> NotificationResult:
        """Send notification via email."""
        try:
            email_config = self.notification_channels.get('email', {})
            if not email_config.get('enabled', False):
                return NotificationResult(NotificationChannel.EMAIL, False, "Email notifications disabled")

            # Try to import email sending library
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
            except ImportError:
                return NotificationResult(NotificationChannel.EMAIL, False, "Email library not available")

            smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            sender_email = email_config.get('sender_email')
            sender_password = email_config.get('sender_password')
            recipient_emails = email_config.get('recipient_emails', [])

            if not sender_email or not recipient_emails:
                return NotificationResult(NotificationChannel.EMAIL, False, "Email configuration incomplete")

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipient_emails)
            msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"

            body = f"""
{alert.message}

Priority: {alert.priority.value.upper()}
Type: {alert.alert_type.value}
Time: {alert.timestamp}
Alert ID: {alert.alert_id}

Data:
{json.dumps(alert.data, indent=2)}
"""
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            tprint_info(f"📧 Email sent for alert: {alert.title}")
            return NotificationResult(NotificationChannel.EMAIL, True, "Email sent successfully")

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return NotificationResult(NotificationChannel.EMAIL, False, str(e))

    async def _send_webhook_notification(self, alert: Alert) -> NotificationResult:
        """Send notification via webhook."""
        try:
            webhook_config = self.notification_channels.get('webhook', {})
            if not webhook_config.get('enabled', False):
                return NotificationResult(NotificationChannel.WEBHOOK, False, "Webhook notifications disabled")

            webhook_url = webhook_config.get('url')
            if not webhook_url:
                return NotificationResult(NotificationChannel.WEBHOOK, False, "Webhook URL not configured")

            # Try to import HTTP library
            try:
                import aiohttp
            except ImportError:
                try:
                    import requests
                    use_aiohttp = False
                except ImportError:
                    return NotificationResult(NotificationChannel.WEBHOOK, False, "HTTP library not available")
            else:
                use_aiohttp = True

            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'priority': alert.priority.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data,
                'rule_id': alert.rule_id,
            }

            headers = webhook_config.get('headers', {'Content-Type': 'application/json'})

            # Send webhook
            if use_aiohttp:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            tprint_info(f"🔗 Webhook sent for alert: {alert.title}")
                            return NotificationResult(NotificationChannel.WEBHOOK, True, f"Webhook sent (status: {response.status})")
                        else:
                            return NotificationResult(NotificationChannel.WEBHOOK, False, f"Webhook returned status {response.status}")
            else:
                response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    tprint_info(f"🔗 Webhook sent for alert: {alert.title}")
                    return NotificationResult(NotificationChannel.WEBHOOK, True, f"Webhook sent (status: {response.status_code})")
                else:
                    return NotificationResult(NotificationChannel.WEBHOOK, False, f"Webhook returned status {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook: {e}")
            return NotificationResult(NotificationChannel.WEBHOOK, False, str(e))

    @handles_errors
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            user: User acknowledging the alert

        Returns:
            True if acknowledged successfully
        """
        tprint(f"[ALERT_MGR] acknowledge_alert: Acknowledging alert_id={alert_id}, user={user}")
        if alert_id not in self.active_alerts:
            tprint(f"[ALERT_MGR] acknowledge_alert: Alert {alert_id} not found in active alerts, returning False")
            return False

        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.now()

        tprint_info(f"✅ Alert {alert_id} acknowledged by {user}")
        tprint(f"[ALERT_MGR] acknowledge_alert: Successfully acknowledged alert {alert_id}, returning True")
        return True

    @handles_errors
    async def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID
            resolution: Resolution description

        Returns:
            True if resolved successfully
        """
        tprint(f"[ALERT_MGR] resolve_alert: Resolving alert_id={alert_id}, resolution='{resolution}'")
        if alert_id not in self.active_alerts:
            tprint(f"[ALERT_MGR] resolve_alert: Alert {alert_id} not found in active alerts, returning False")
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()

        # Move to history
        del self.active_alerts[alert_id]

        tprint_success(f"✅ Alert {alert_id} resolved: {resolution}")
        tprint(f"[ALERT_MGR] resolve_alert: Successfully resolved and removed alert {alert_id}, returning True")
        return True

    async def escalate_alert(self, alert_id: str) -> bool:
        """
        Escalate alert priority.

        Args:
            alert_id: Alert ID

        Returns:
            True if escalated successfully
        """
        tprint(f"[ALERT_MGR] escalate_alert: Escalating alert_id={alert_id}")
        if alert_id not in self.active_alerts:
            tprint(f"[ALERT_MGR] escalate_alert: Alert {alert_id} not found, returning False")
            return False

        alert = self.active_alerts[alert_id]

        # Priority escalation: LOW -> MEDIUM -> HIGH -> URGENT -> CRITICAL
        priority_order = [AlertPriority.LOW, AlertPriority.MEDIUM, AlertPriority.HIGH,
                         AlertPriority.URGENT, AlertPriority.CRITICAL]

        current_index = priority_order.index(alert.priority)
        if current_index < len(priority_order) - 1:
            old_priority = alert.priority
            new_priority = priority_order[current_index + 1]
            alert.priority = new_priority
            tprint(f"[ALERT_MGR] escalate_alert: Priority escalated from {old_priority.value} to {new_priority.value}")

            # Send new notifications with higher priority
            await self._send_notifications(alert)

            tprint_warning(f"⚡ Alert {alert_id} escalated to {new_priority.value}")
            tprint(f"[ALERT_MGR] escalate_alert: Successfully escalated alert, returning True")
            return True

        tprint(f"[ALERT_MGR] escalate_alert: Alert already at maximum priority, returning False")
        return False

    async def get_active_alerts(self, alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get active alerts."""
        tprint(f"[ALERT_MGR] get_active_alerts: Retrieving active alerts, filter_type={alert_type.value if alert_type else 'None'}")
        alerts = list(self.active_alerts.values())

        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]
            tprint(f"[ALERT_MGR] get_active_alerts: Filtered {len(alerts)} alerts by type {alert_type.value}")

        tprint(f"[ALERT_MGR] get_active_alerts: Returning {len(alerts)} active alerts")
        return alerts

    async def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        alert_type: Optional[AlertType] = None
    ) -> List[Alert]:
        """Get alert history with filters."""
        alerts = self.alert_history.copy()

        if start_time:
            alerts = [alert for alert in alerts if alert.timestamp >= start_time]

        if end_time:
            alerts = [alert for alert in alerts if alert.timestamp <= end_time]

        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]

        return alerts

    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)

        if total_alerts == 0:
            return {
                "total_alerts": 0,
                "active_alerts": 0,
                "acknowledged_alerts": 0,
                "resolved_alerts": 0,
                "avg_response_time": 0.0
            }

        acknowledged_alerts = sum(1 for alert in self.alert_history if alert.acknowledged)
        resolved_alerts = sum(1 for alert in self.alert_history if alert.resolved)

        # Calculate average response time
        response_times = []
        for alert in self.alert_history:
            if alert.acknowledged_at and alert.timestamp:
                response_time = (alert.acknowledged_at - alert.timestamp).total_seconds()
                response_times.append(response_time)

        avg_response_time = np.mean(response_times) if response_times else 0.0

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "resolved_alerts": resolved_alerts,
            "acknowledgment_rate": acknowledged_alerts / total_alerts,
            "resolution_rate": resolved_alerts / total_alerts,
            "avg_response_time": avg_response_time,
            "notification_success_rate": self.notification_success_rate
        }

    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules[rule.rule_id] = rule
        tprint_info(f"📝 Added alert rule: {rule.rule_id}")

    async def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule."""
        if rule_id not in self.alert_rules:
            return False

        rule = self.alert_rules[rule_id]
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        tprint_info(f"📝 Updated alert rule: {rule_id}")
        return True

    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            tprint_info(f"🗑️ Removed alert rule: {rule_id}")
            return True
        return False

    async def cleanup_old_alerts(self, days: int = 30) -> None:
        """Clean up old alerts."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Keep only recent history
        self.alert_history = [alert for alert in self.alert_history if alert.timestamp > cutoff_date]

        # Remove resolved alerts from active
        resolved_alerts = [alert_id for alert_id, alert in self.active_alerts.items()
                          if alert.resolved]
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]

        tprint_info(f"🧹 Cleaned up alerts older than {days} days")

    async def export_alert_data(self, format: str = "json") -> str:
        """Export alert data."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": await self.get_alert_statistics(),
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "type": alert.alert_type.value,
                    "priority": alert.priority.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in self.active_alerts.values()
            ],
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "type": alert.alert_type.value,
                    "priority": alert.priority.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved
                }
                for alert in self.alert_history[-100:]  # Last 100 alerts
            ]
        }

        if format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return pd.DataFrame(data["recent_alerts"]).to_csv(index=False)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of alert manager."""
        try:
            now = datetime.now()
            
            # Check for stale cooldowns
            stale_cooldowns = sum(
                1 for last_trigger in self.cooldowns.values()
                if (now - last_trigger).total_seconds() > 86400  # 24 hours
            )
            
            # Check memory usage
            alerts_count = len(self.active_alerts)
            history_count = len(self.alert_history)
            
            # Check notification success rate
            health_score = 1.0
            if self.notification_success_rate < 0.8:
                health_score = 0.7
            elif self.notification_success_rate < 0.9:
                health_score = 0.85
            
            return {
                'status': 'healthy' if health_score >= 0.8 else 'degraded',
                'health_score': health_score,
                'active_alerts': alerts_count,
                'alert_history_count': history_count,
                'stale_cooldowns': stale_cooldowns,
                'notification_success_rate': self.notification_success_rate,
                'alert_groups_count': len(self.alert_groups),
                'timestamp': now.isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.alert_rules.clear()
        self.notification_history.clear()
        self.cooldowns.clear()

        tprint_info("🧹 Alert Manager cleaned up successfully")

# Global singleton instance
_global_alert_manager: Optional[AlertManager] = None

# Factory functions
async def create_alert_manager(config: Dict[str, Any]) -> AlertManager:
    """Create and initialize an alert manager."""
    global _global_alert_manager
    manager = AlertManager(config)
    await manager.initialize()
    _global_alert_manager = manager
    return manager

def get_alert_manager() -> Optional[AlertManager]:
    """Get the global alert manager instance."""
    return _global_alert_manager
