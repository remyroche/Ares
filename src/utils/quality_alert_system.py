"""
Quality Alert System for Data Quality Monitoring

This module provides an alert system that can send notifications when data quality
issues are detected, supporting multiple channels like Slack, email, and webhooks.
"""

import sys
import requests
import smtplib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import json
import asyncio
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.advanced_ml_validation import Alert, AlertConfig, MLValidationResult

@dataclass
class QualityAlert:
    """Represents a quality alert with all necessary information."""
    level: str  # CRITICAL, ERROR, WARNING, INFO
    message: str
    timestamp: datetime
    action_required: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    category: str = ""

@dataclass
class AlertChannel:
    """Base class for alert channels."""
    name: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

class SlackChannel(AlertChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str, channel: str = "#general"):
        super().__init__("slack")
        self.config = {
            "webhook_url": webhook_url,
            "channel": channel
        }
    
    async def send_alert(self, alert: QualityAlert) -> bool:
        """Send alert to Slack."""
        try:
            color_map = {
                "CRITICAL": "#FF0000",
                "ERROR": "#FF6B6B", 
                "WARNING": "#FFA500",
                "INFO": "#4CAF50"
            }
            
            payload = {
                "channel": self.config["channel"],
                "attachments": [{
                    "color": color_map.get(alert.level, "#808080"),
                    "title": f"Quality Alert: {alert.level}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "Category", "value": alert.category, "short": True},
                        {"title": "Action Required", "value": "Yes" if alert.action_required else "No", "short": True}
                    ],
                    "footer": "Quality Alert System"
                }]
            }
            
            response = requests.post(self.config["webhook_url"], json=payload)
            return response.status_code == 200
        except Exception as e:
            system_logger.error(f"Failed to send Slack alert: {e}")
            return False

class EmailChannel(AlertChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        super().__init__("email")
        self.config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "recipients": recipients
        }
    
    async def send_alert(self, alert: QualityAlert) -> bool:
        """Send alert via email."""
        try:
            msg = f"""Quality Alert System Notification

Level: {alert.level}
Message: {alert.message}
Timestamp: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
Category: {alert.category}
Action Required: {"Yes" if alert.action_required else "No"}

Details: {json.dumps(alert.details, indent=2)}

---
This is an automated message from the Quality Alert System.
"""
            
            with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                server.starttls()
                server.login(self.config["username"], self.config["password"])
                
                for recipient in self.config["recipients"]:
                    server.sendmail(
                        self.config["username"],
                        recipient,
                        f"Subject: Quality Alert - {alert.level}\n\n{msg}"
                    )
            
            return True
        except Exception as e:
            system_logger.error(f"Failed to send email alert: {e}")
            return False

class WebhookChannel(AlertChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        super().__init__("webhook")
        self.config = {
            "webhook_url": webhook_url,
            "headers": headers or {}
        }
    
    async def send_alert(self, alert: QualityAlert) -> bool:
        """Send alert via webhook."""
        try:
            payload = {
                "level": alert.level,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "action_required": alert.action_required,
                "category": alert.category,
                "details": alert.details
            }
            
            response = requests.post(
                self.config["webhook_url"],
                json=payload,
                headers=self.config["headers"]
            )
            return response.status_code in [200, 201, 202]
        except Exception as e:
            system_logger.error(f"Failed to send webhook alert: {e}")
            return False

class QualityAlertManager:
    """Manages quality alerts and notifications."""
    
    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.config = alert_config or AlertConfig()
        self.alert_history: List[QualityAlert] = []
        self.channels: List[AlertChannel] = []
        self.logger = system_logger.getChild("QualityAlertManager")
        self.is_initialized = False
        self.alert_thresholds = {
            "critical": 0.6,
            "warning": 0.8,
            "info": 0.9
        }
    
    async def initialize(self) -> bool:
        """Initialize QualityAlertManager."""
        try:
            self.logger.info("🚀 Initializing QualityAlertManager...")
            
            # Initialize default channels if none provided
            if not self.channels:
                self._setup_default_channels()
            
            self.is_initialized = True
            self.logger.info("✅ QualityAlertManager initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing QualityAlertManager: {e}")
            return False
    
    def _setup_default_channels(self):
        """Setup default alert channels based on configuration."""
        # This would typically read from config files or environment variables
        pass
    
    def add_channel(self, channel: AlertChannel):
        """Add a new alert channel."""
        self.channels.append(channel)
        self.logger.info(f"Added alert channel: {channel.name}")
    
    def remove_channel(self, channel_name: str):
        """Remove an alert channel by name."""
        self.channels = [c for c in self.channels if c.name != channel_name]
        self.logger.info(f"Removed alert channel: {channel_name}")
    
    def check_alerts(self, validation_result: MLValidationResult) -> List[QualityAlert]:
        """Check validation results and generate appropriate alerts."""
        alerts = []
        
        # Check quality score
        if hasattr(validation_result, 'quality_score') and validation_result.quality_score:
            score = getattr(validation_result.quality_score, 'overall', 0.0)
            grade = getattr(validation_result.quality_score, 'grade', 'F')
            
            if score < self.alert_thresholds["critical"]:  # Grade F
                alerts.append(QualityAlert(
                    level="CRITICAL",
                    message=f"Critical data quality issue: Quality score {score:.3f} (Grade {grade})",
                    timestamp=datetime.now(),
                    action_required=True,
                    details={"quality_score": score, "grade": grade},
                    category="quality_score"
                ))
            elif score < self.alert_thresholds["warning"]:  # Grade C or D
                alerts.append(QualityAlert(
                    level="WARNING",
                    message=f"Data quality warning: Quality score {score:.3f} (Grade {grade})",
                    timestamp=datetime.now(),
                    action_required=False,
                    details={"quality_score": score, "grade": grade},
                    category="quality_score"
                ))
        
        # Check for drift
        if hasattr(validation_result, 'drift_report') and validation_result.drift_report:
            drift_issues = getattr(validation_result.drift_report, 'issues', [])
            if len(drift_issues) > 0:
                alerts.append(QualityAlert(
                    level="ERROR",
                    message=f"Data drift detected: {len(drift_issues)} drift issues found",
                    timestamp=datetime.now(),
                    action_required=True,
                    details={"drift_issues": drift_issues},
                    category="data_drift"
                ))
        
        # Check for correlation issues
        if hasattr(validation_result, 'correlation_issues') and validation_result.correlation_issues:
            alerts.append(QualityAlert(
                level="WARNING",
                message=f"Feature correlation issues: {len(validation_result.correlation_issues)} issues found",
                timestamp=datetime.now(),
                action_required=False,
                details={"correlation_issues": validation_result.correlation_issues},
                category="correlation"
            ))
        
        # Check for missing data
        if hasattr(validation_result, 'missing_data_report') and validation_result.missing_data_report:
            missing_percentage = getattr(validation_result.missing_data_report, 'overall_missing_percentage', 0.0)
            if missing_percentage > 0.1:  # More than 10% missing
                alerts.append(QualityAlert(
                    level="WARNING",
                    message=f"High missing data percentage: {missing_percentage:.2%}",
                    timestamp=datetime.now(),
                    action_required=False,
                    details={"missing_percentage": missing_percentage},
                    category="missing_data"
                ))
        
        return alerts
    
    async def send_alerts(self, alerts: List[QualityAlert]) -> Dict[str, bool]:
        """Send alerts through all configured channels."""
        if not self.is_initialized:
            self.logger.error("QualityAlertManager not initialized")
            return {}
        
        results = {}
        
        for alert in alerts:
            # Store alert in history
            self.alert_history.append(alert)
            
            # Send through all channels
            for channel in self.channels:
                if channel.enabled:
                    try:
                        success = await channel.send_alert(alert)
                        results[f"{channel.name}_{alert.level}"] = success
                        
                        if success:
                            self.logger.info(f"Alert sent successfully via {channel.name}")
                        else:
                            self.logger.warning(f"Failed to send alert via {channel.name}")
                    except Exception as e:
                        self.logger.error(f"Error sending alert via {channel.name}: {e}")
                        results[f"{channel.name}_{alert.level}"] = False
        
        return results
    
    async def process_validation_result(self, validation_result: MLValidationResult) -> Dict[str, Any]:
        """Process validation result and send alerts."""
        alerts = self.check_alerts(validation_result)
        
        if alerts:
            self.logger.info(f"Generated {len(alerts)} alerts from validation result")
            send_results = await self.send_alerts(alerts)
            
            return {
                "alerts_generated": len(alerts),
                "alerts_sent": send_results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            self.logger.info("No alerts generated from validation result")
            return {
                "alerts_generated": 0,
                "alerts_sent": {},
                "timestamp": datetime.now().isoformat()
            }
    
    def get_alert_history(self, 
                         level: Optional[str] = None, 
                         category: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[QualityAlert]:
        """Get filtered alert history."""
        filtered = self.alert_history
        
        if level:
            filtered = [a for a in filtered if a.level == level.upper()]
        
        if category:
            filtered = [a for a in filtered if a.category == category]
        
        if start_date:
            filtered = [a for a in filtered if a.timestamp >= start_date]
        
        if end_date:
            filtered = [a for a in filtered if a.timestamp <= end_date]
        
        return filtered
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about alerts."""
        if not self.alert_history:
            return {"total_alerts": 0}
        
        total = len(self.alert_history)
        by_level = defaultdict(int)
        by_category = defaultdict(int)
        by_date = defaultdict(int)
        
        for alert in self.alert_history:
            by_level[alert.level] += 1
            by_category[alert.category] += 1
            by_date[alert.timestamp.date()] += 1
        
        return {
            "total_alerts": total,
            "by_level": dict(by_level),
            "by_category": dict(by_category),
            "by_date": {str(date): count for date, count in by_date.items()},
            "critical_alerts": by_level.get("CRITICAL", 0),
            "action_required": sum(1 for a in self.alert_history if a.action_required)
        }
    
    def clear_history(self, older_than_days: Optional[int] = None):
        """Clear alert history, optionally keeping recent alerts."""
        if older_than_days is None:
            self.alert_history.clear()
            self.logger.info("Alert history cleared")
        else:
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            original_count = len(self.alert_history)
            self.alert_history = [a for a in self.alert_history if a.timestamp >= cutoff_date]
            removed_count = original_count - len(self.alert_history)
            self.logger.info(f"Removed {removed_count} old alerts from history")

# Convenience function for quick alert sending
async def send_quality_alert(level: str, message: str, category: str = "", 
                           action_required: bool = False, details: Optional[Dict[str, Any]] = None):
    """Quick function to send a quality alert."""
    manager = QualityAlertManager()
    await manager.initialize()
    
    alert = QualityAlert(
        level=level.upper(),
        message=message,
        timestamp=datetime.now(),
        action_required=action_required,
        details=details or {},
        category=category
    )
    
    await manager.send_alerts([alert])
    return alert

# Example usage
if __name__ == "__main__":
    async def main():
        # Example of setting up and using the alert manager
        manager = QualityAlertManager()
        
        # Add channels
        slack_channel = SlackChannel("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
        manager.add_channel(slack_channel)
        
        # Initialize
        await manager.initialize()
        
        # Create a sample alert
        sample_alert = QualityAlert(
            level="INFO",
            message="Quality Alert System initialized successfully",
            timestamp=datetime.now(),
            category="system"
        )
        
        # Send alert
        await manager.send_alerts([sample_alert])
        
        # Print statistics
        print(manager.get_alert_statistics())
    
    asyncio.run(main())