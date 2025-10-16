"""
Alert Manager

Centralized alert management system for trading operations.
Handles alert creation, routing, escalation, and notification delivery.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
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
        self.config = config
        self.logger = logger.getChild('AlertManager')

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}

        # Notification settings
        self.notification_channels = config.get('notification_channels', {})
        self.default_channels = config.get('default_channels', [NotificationChannel.LOG])

        # Rate limiting
        self.notification_history: Dict[NotificationChannel, List[datetime]] = {}
        self.cooldowns: Dict[str, datetime] = {}  # rule_id -> last_trigger_time

        # Performance tracking
        self.notification_success_rate = 1.0
        self.avg_response_time = 0.0

        # Escalation settings
        self.escalation_rules = config.get('escalation_rules', {})

        tprint_info("🚨 Initializing Alert Manager...")

    async def initialize(self) -> None:
        """Initialize alert manager."""
        await self._load_default_rules()
        tprint_success("✅ Alert Manager initialized successfully")

    async def _load_default_rules(self) -> None:
        """Load default alert rules."""
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

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Check if alert should trigger notifications
        if await self._should_send_notifications(alert):
            await self._send_notifications(alert)

        tprint_warning(f"🚨 {priority.value.upper()}: {title}")

        return alert_id

    async def _should_send_notifications(self, alert: Alert) -> bool:
        """Check if alert should trigger notifications."""
        # Check cooldown for rule-based alerts
        if alert.rule_id and alert.rule_id in self.cooldowns:
            last_trigger = self.cooldowns[alert.rule_id]
            rule = self.alert_rules.get(alert.rule_id)
            if rule and (datetime.now() - last_trigger).total_seconds() < rule.cooldown_seconds:
                return False

        # Update cooldown
        if alert.rule_id:
            self.cooldowns[alert.rule_id] = datetime.now()

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
        # Placeholder for email notification
        # In real implementation, would use SMTP or email service
        tprint_info(f"📧 Would send email for alert: {alert.title}")
        return NotificationResult(NotificationChannel.EMAIL, True, "Email sent (simulated)")

    async def _send_webhook_notification(self, alert: Alert) -> NotificationResult:
        """Send notification via webhook."""
        # Placeholder for webhook notification
        # In real implementation, would POST to webhook URL
        tprint_info(f"🔗 Would send webhook for alert: {alert.title}")
        return NotificationResult(NotificationChannel.WEBHOOK, True, "Webhook sent (simulated)")

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
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.now()

        tprint_info(f"✅ Alert {alert_id} acknowledged by {user}")
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
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()

        # Move to history
        del self.active_alerts[alert_id]

        tprint_success(f"✅ Alert {alert_id} resolved: {resolution}")
        return True

    async def escalate_alert(self, alert_id: str) -> bool:
        """
        Escalate alert priority.

        Args:
            alert_id: Alert ID

        Returns:
            True if escalated successfully
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]

        # Priority escalation: LOW -> MEDIUM -> HIGH -> URGENT -> CRITICAL
        priority_order = [AlertPriority.LOW, AlertPriority.MEDIUM, AlertPriority.HIGH,
                         AlertPriority.URGENT, AlertPriority.CRITICAL]

        current_index = priority_order.index(alert.priority)
        if current_index < len(priority_order) - 1:
            new_priority = priority_order[current_index + 1]
            alert.priority = new_priority

            # Send new notifications with higher priority
            await self._send_notifications(alert)

            tprint_warning(f"⚡ Alert {alert_id} escalated to {new_priority.value}")
            return True

        return False

    async def get_active_alerts(self, alert_type: Optional[AlertType] = None) -> List[Alert]:
        """Get active alerts."""
        alerts = list(self.active_alerts.values())

        if alert_type:
            alerts = [alert for alert in alerts if alert.alert_type == alert_type]

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

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.alert_rules.clear()
        self.notification_history.clear()
        self.cooldowns.clear()

        tprint_info("🧹 Alert Manager cleaned up successfully")

# Factory functions
async def create_alert_manager(config: Dict[str, Any]) -> AlertManager:
    """Create and initialize an alert manager."""
    manager = AlertManager(config)
    await manager.initialize()
    return manager

def get_alert_manager() -> Optional[AlertManager]:
    """Get the global alert manager instance."""
    return None
