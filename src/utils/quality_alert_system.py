"""
Quality Alert System for Data Quality Monitoring

This module provides an alert system that can send notifications when data quality
issues are detected, supporting multiple channels like Slack, email, and webhooks.
"""

import os
import sys
import json
import requests
import smtplib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.advanced_ml_validation import Alert, AlertConfig, MLValidationResult


class QualityAlertManager:
    """Manages quality alerts and notifications."""
    
    def __init__(self, alert_config: AlertConfig):
        self.config = alert_config
        self.alert_history: List[Alert] = []
        self.logger = system_logger.getChild("QualityAlertManager")
    
    def check_alerts(self, validation_result: MLValidationResult) -> List[Alert]:
        """Generate alerts based on validation results."""
        alerts = []
        
        # Check quality score
        if validation_result.quality_score:
            score = validation_result.quality_score.overall
            grade = validation_result.quality_score.grade
            
            if score < 0.6:  # Grade F
                alerts.append(Alert(
                    level="CRITICAL",
                    message=f"Critical data quality issue: Quality score {score:.3f} (Grade {grade})",
                    timestamp=datetime.now(),
                    action_required=True,
                    details={"quality_score": score, "grade": grade}
                ))
            elif score < 0.8:  # Grade C or D
                alerts.append(Alert(
                    level="WARNING",
                    message=f"Data quality warning: Quality score {score:.3f} (Grade {grade})",
                    timestamp=datetime.now(),
                    action_required=False,
                    details={"quality_score": score, "grade": grade}
                ))
        
        # Check for drift
        if validation_result.drift_report:
            drift_issues = validation_result.drift_report.issues
            if len(drift_issues) > 0:
                alerts.append(Alert(
                    level="ERROR",
                    message=f"Data drift detected: {len(drift_issues)} drift issues found",
                    timestamp=datetime.now(),
                    action_required=True,
                    details={"drift_issues": drift_issues}
                ))
        
        # Check for correlation issues
        if validation_result.correlation_issues:
            alerts.append(Alert(
                level="WARNING",
                message=f"Feature correlation issues: {len(validation_result.correlation_issues)} issues found",
                timestamp=datetime.now(),
                action_required=False,
                details={"correlation_issues": validation_result.correlation_issues[:5]}  # First 5
            ))
        
        # Check for target issues
        if validation_result.target_issues:
            alerts.append(Alert(
                level="ERROR",
                message=f"Target variable issues: {len(validation_result.target_issues)} issues found",
                timestamp=datetime.now(),
                action_required=True,
                details={"target_issues": validation_result.target_issues[:5]}  # First 5
            ))
        
        # Check for distribution issues
        if validation_result.distribution_issues:
            alerts.append(Alert(
                level="WARNING",
                message=f"Distribution issues: {len(validation_result.distribution_issues)} issues found",
                timestamp=datetime.now(),
                action_required=False,
                details={"distribution_issues": validation_result.distribution_issues[:5]}  # First 5
            ))
        
        # Check for outlier issues
        if validation_result.outlier_issues:
            alerts.append(Alert(
                level="WARNING",
                message=f"Outlier issues: {len(validation_result.outlier_issues)} issues found",
                timestamp=datetime.now(),
                action_required=False,
                details={"outlier_issues": validation_result.outlier_issues[:5]}  # First 5
            ))
        
        # Check for time series issues
        if validation_result.time_series_issues:
            alerts.append(Alert(
                level="WARNING",
                message=f"Time series issues: {len(validation_result.time_series_issues)} issues found",
                timestamp=datetime.now(),
                action_required=False,
                details={"time_series_issues": validation_result.time_series_issues[:5]}  # First 5
            ))
        
        # Check for financial data issues
        if validation_result.financial_issues:
            alerts.append(Alert(
                level="ERROR",
                message=f"Financial data issues: {len(validation_result.financial_issues)} issues found",
                timestamp=datetime.now(),
                action_required=True,
                details={"financial_issues": validation_result.financial_issues[:5]}  # First 5
            ))
        
        return alerts
    
    def send_alerts(self, alerts: List[Alert]) -> Dict[str, bool]:
        """Send alerts through configured channels."""
        results = {}
        
        for alert in alerts:
            self.alert_history.append(alert)
            
            # Send to Slack
            if self.config.slack_webhook:
                results["slack"] = self._send_slack_alert(alert)
            
            # Send email
            if self.config.email_config:
                results["email"] = self._send_email_alert(alert)
            
            # Send webhook
            if self.config.webhook_url:
                results["webhook"] = self._send_webhook_alert(alert)
        
        return results
    
    def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            # Create Slack message
            color_map = {
                "CRITICAL": "#FF0000",  # Red
                "ERROR": "#FF6B6B",     # Light red
                "WARNING": "#FFA500",   # Orange
                "INFO": "#4CAF50"       # Green
            }
            
            color = color_map.get(alert.level, "#808080")  # Gray default
            
            slack_message = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"Data Quality Alert: {alert.level}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            },
                            {
                                "title": "Action Required",
                                "value": "Yes" if alert.action_required else "No",
                                "short": True
                            }
                        ],
                        "footer": "Data Quality Monitoring System"
                    }
                ]
            }
            
            # Add details if available
            if alert.details:
                details_text = "\n".join([f"• {k}: {v}" for k, v in alert.details.items()])
                slack_message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": details_text,
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(
                self.config.slack_webhook,
                json=slack_message,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"✅ Slack alert sent successfully: {alert.level}")
                return True
            else:
                self.logger.error(f"❌ Failed to send Slack alert: {response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Error sending Slack alert: {e}")
            return False
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            email_config = self.config.email_config
            if not email_config:
                return False
            
            # Create email message
            subject = f"Data Quality Alert: {alert.level}"
            
            body = f"""
Data Quality Alert

Level: {alert.level}
Message: {alert.message}
Timestamp: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
Action Required: {"Yes" if alert.action_required else "No"}

Details:
"""
            
            if alert.details:
                for key, value in alert.details.items():
                    body += f"• {key}: {value}\n"
            
            body += "\n---\nData Quality Monitoring System"
            
            # Send email
            with smtplib.SMTP(email_config.get("smtp_server", "localhost"), 
                            email_config.get("smtp_port", 587)) as server:
                
                if email_config.get("use_tls", True):
                    server.starttls()
                
                if email_config.get("username") and email_config.get("password"):
                    server.login(email_config["username"], email_config["password"])
                
                # Create message
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                msg = MIMEMultipart()
                msg["From"] = email_config.get("from_email", "noreply@example.com")
                msg["To"] = email_config.get("to_email", "admin@example.com")
                msg["Subject"] = subject
                
                msg.attach(MIMEText(body, "plain"))
                
                server.send_message(msg)
            
            self.logger.info(f"✅ Email alert sent successfully: {alert.level}")
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Error sending email alert: {e}")
            return False
    
    def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            webhook_data = {
                "level": alert.level,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "action_required": alert.action_required,
                "details": alert.details
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=webhook_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 201, 202]:
                self.logger.info(f"✅ Webhook alert sent successfully: {alert.level}")
                return True
            else:
                self.logger.error(f"❌ Failed to send webhook alert: {response.status_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"❌ Error sending webhook alert: {e}")
            return False
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, int]:
        """Get summary of alerts in the last N hours."""
        recent_alerts = self.get_alert_history(hours)
        
        summary = {
            "total": len(recent_alerts),
            "critical": len([a for a in recent_alerts if a.level == "CRITICAL"]),
            "error": len([a for a in recent_alerts if a.level == "ERROR"]),
            "warning": len([a for a in recent_alerts if a.level == "WARNING"]),
            "info": len([a for a in recent_alerts if a.level == "INFO"])
        }
        
        return summary


class StreamingQualityValidator:
    """Validates streaming data in real-time."""
    
    def __init__(self, validation_rules: List[Any], alert_manager: QualityAlertManager):
        self.validation_rules = validation_rules
        self.alert_manager = alert_manager
        self.quality_metrics = defaultdict(list)
        self.logger = system_logger.getChild("StreamingQualityValidator")
    
    def validate_streaming_data(self, data_chunk: pd.DataFrame) -> Dict[str, Any]:
        """Validate streaming data in real-time."""
        issues = []
        metrics = {}
        
        # Apply validation rules
        for rule in self.validation_rules:
            try:
                rule_result = rule.validate(data_chunk)
                issues.extend(rule_result.get("issues", []))
                metrics[rule.name] = rule_result.get("metrics", {})
            except Exception as e:
                self.logger.error(f"Error applying validation rule {rule.name}: {e}")
        
        # Update rolling metrics
        for metric_name, value in metrics.items():
            self.quality_metrics[metric_name].append(value)
            
            # Keep only last 1000 values
            if len(self.quality_metrics[metric_name]) > 1000:
                self.quality_metrics[metric_name].pop(0)
        
        # Calculate rolling statistics
        rolling_metrics = self._calculate_rolling_metrics()
        
        # Generate alerts if needed
        if issues:
            alert = Alert(
                level="WARNING",
                message=f"Streaming data quality issues: {len(issues)} issues detected",
                timestamp=datetime.now(),
                action_required=len(issues) > 5,
                details={"issues": issues[:5], "rolling_metrics": rolling_metrics}
            )
            
            self.alert_manager.send_alerts([alert])
        
        return {
            "issues": issues,
            "metrics": metrics,
            "rolling_metrics": rolling_metrics,
            "timestamp": datetime.now()
        }
    
    def _calculate_rolling_metrics(self) -> Dict[str, float]:
        """Calculate rolling statistics for quality metrics."""
        rolling_metrics = {}
        
        for metric_name, values in self.quality_metrics.items():
            if values:
                rolling_metrics[f"{metric_name}_mean"] = np.mean(values)
                rolling_metrics[f"{metric_name}_std"] = np.std(values)
                rolling_metrics[f"{metric_name}_min"] = np.min(values)
                rolling_metrics[f"{metric_name}_max"] = np.max(values)
        
        return rolling_metrics


class QualityDashboard:
    """Provides dashboard functionality for quality monitoring."""
    
    def __init__(self, alert_manager: QualityAlertManager):
        self.alert_manager = alert_manager
        self.logger = system_logger.getChild("QualityDashboard")
    
    def generate_quality_report(self, validation_result: MLValidationResult) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_quality": {
                "score": validation_result.quality_score.overall,
                "grade": validation_result.quality_score.grade,
                "components": validation_result.quality_score.components
            },
            "issues_summary": {
                "total_issues": validation_result.summary.get("total_issues", 0),
                "correlation_issues": len(validation_result.correlation_issues),
                "target_issues": len(validation_result.target_issues),
                "distribution_issues": len(validation_result.distribution_issues),
                "outlier_issues": len(validation_result.outlier_issues),
                "time_series_issues": len(validation_result.time_series_issues),
                "financial_issues": len(validation_result.financial_issues)
            },
            "drift_detection": {
                "drift_detected": validation_result.drift_report is not None,
                "drift_issues": validation_result.drift_report.issues if validation_result.drift_report else []
            },
            "recommendations": self._generate_recommendations(validation_result)
        }
        
        return report
    
    def _generate_recommendations(self, validation_result: MLValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Quality score recommendations
        if validation_result.quality_score.overall < 0.6:
            recommendations.append("CRITICAL: Data quality is very poor. Immediate action required.")
        elif validation_result.quality_score.overall < 0.8:
            recommendations.append("WARNING: Data quality needs improvement. Review and fix issues.")
        
        # Correlation recommendations
        if validation_result.correlation_issues:
            recommendations.append("Consider removing highly correlated features to reduce multicollinearity.")
        
        # Target recommendations
        if validation_result.target_issues:
            recommendations.append("Review target variable for class imbalance or target leakage.")
        
        # Distribution recommendations
        if validation_result.distribution_issues:
            recommendations.append("Check for data drift or distribution shifts in features.")
        
        # Outlier recommendations
        if validation_result.outlier_issues:
            recommendations.append("Investigate and handle outliers appropriately.")
        
        # Time series recommendations
        if validation_result.time_series_issues:
            recommendations.append("Check time series data for gaps, duplicates, or ordering issues.")
        
        # Financial data recommendations
        if validation_result.financial_issues:
            recommendations.append("Verify financial data integrity and OHLC relationships.")
        
        return recommendations
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for dashboard."""
        alert_summary = self.alert_manager.get_alert_summary(hours)
        
        return {
            "period_hours": hours,
            "alert_counts": alert_summary,
            "total_alerts": alert_summary["total"],
            "critical_alerts": alert_summary["critical"],
            "error_alerts": alert_summary["error"],
            "warning_alerts": alert_summary["warning"]
        }


# Convenience functions
def create_alert_config(
    slack_webhook: Optional[str] = None,
    email_config: Optional[Dict[str, Any]] = None,
    webhook_url: Optional[str] = None
) -> AlertConfig:
    """Create alert configuration."""
    return AlertConfig(
        slack_webhook=slack_webhook,
        email_config=email_config,
        webhook_url=webhook_url
    )


def setup_quality_monitoring(
    alert_config: AlertConfig,
    validation_rules: Optional[List[Any]] = None
) -> Tuple[QualityAlertManager, StreamingQualityValidator, QualityDashboard]:
    """Set up complete quality monitoring system."""
    alert_manager = QualityAlertManager(alert_config)
    streaming_validator = StreamingQualityValidator(validation_rules or [], alert_manager)
    dashboard = QualityDashboard(alert_manager)
    
    return alert_manager, streaming_validator, dashboard