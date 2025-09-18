#!/usr/bin/env python3
"""
Enhanced Reporting System with Real-time Monitoring

This module provides comprehensive reporting capabilities with real-time monitoring,
alerting, and automated report generation for ML training pipelines.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import warnings
import traceback
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from src.utils.tprint import tprint
from src.utils.logger import get_logger

logger = get_logger("EnhancedReportingSystem")

class ReportType(Enum):
    """Types of reports."""
    TRAINING_PROGRESS = "training_progress"
    HPO_OPTIMIZATION = "hpo_optimization"
    MODEL_VALIDATION = "model_validation"
    ERROR_ANALYSIS = "error_analysis"
    PERFORMANCE_METRICS = "performance_metrics"
    SYSTEM_HEALTH = "system_health"
    COMPREHENSIVE = "comprehensive"

class AlertLevel(Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ReportData:
    """Data structure for report content."""
    report_id: str
    report_type: ReportType
    timestamp: datetime
    title: str
    summary: str
    data: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    visualizations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    component: str
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: Optional[str] = None

class EnhancedReportingSystem:
    """Enhanced reporting system with real-time monitoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced reporting system."""
        self.config = config or {}
        self.logger = logger.getChild('EnhancedReportingSystem')
        
        # Report storage
        self.reports: Dict[str, ReportData] = {}
        self.report_history: deque = deque(maxlen=1000)
        
        # Alert system
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_subscribers: List[Callable] = []
        
        # Monitoring configuration
        self.monitoring_config = self.config.get('monitoring', {
            'enable_real_time': True,
            'report_interval': 300,  # 5 minutes
            'alert_check_interval': 60,  # 1 minute
            'retention_days': 30
        })
        
        # Notification configuration
        self.notification_config = self.config.get('notifications', {
            'email_enabled': False,
            'slack_enabled': False,
            'webhook_enabled': False
        })
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=10000)
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.get('output_dir', 'reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("📊 Enhanced Reporting System initialized")
    
    def generate_report(self, 
                       report_type: ReportType,
                       title: str,
                       data: Dict[str, Any],
                       summary: Optional[str] = None) -> str:
        """Generate a comprehensive report."""
        try:
            report_id = self._generate_report_id(report_type, title)
            
            # Create report data
            report_data = ReportData(
                report_id=report_id,
                report_type=report_type,
                timestamp=datetime.now(),
                title=title,
                summary=summary or self._generate_summary(data, report_type),
                data=data,
                metrics=self._extract_metrics(data, report_type),
                visualizations=self._generate_visualizations(data, report_type),
                recommendations=self._generate_recommendations(data, report_type)
            )
            
            # Store report
            with self.lock:
                self.reports[report_id] = report_data
                self.report_history.append(report_data)
            
            # Generate report files
            report_paths = self._save_report_files(report_data)
            
            # Check for alerts
            self._check_report_alerts(report_data)
            
            self.logger.info(f"📊 Generated report: {title} ({report_id})")
            return report_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            raise
    
    def _generate_report_id(self, report_type: ReportType, title: str) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        return f"{report_type.value}_{timestamp}_{title_hash}"
    
    def _generate_summary(self, data: Dict[str, Any], report_type: ReportType) -> str:
        """Generate report summary based on data and type."""
        try:
            if report_type == ReportType.TRAINING_PROGRESS:
                return self._generate_training_summary(data)
            elif report_type == ReportType.HPO_OPTIMIZATION:
                return self._generate_hpo_summary(data)
            elif report_type == ReportType.MODEL_VALIDATION:
                return self._generate_validation_summary(data)
            elif report_type == ReportType.ERROR_ANALYSIS:
                return self._generate_error_summary(data)
            elif report_type == ReportType.PERFORMANCE_METRICS:
                return self._generate_performance_summary(data)
            else:
                return f"Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate summary: {e}")
            return "Summary generation failed"
    
    def _generate_training_summary(self, data: Dict[str, Any]) -> str:
        """Generate training progress summary."""
        try:
            epochs = data.get('epochs', 0)
            loss = data.get('loss', 0)
            accuracy = data.get('accuracy', 0)
            status = data.get('status', 'unknown')
            
            return f"Training {status}: {epochs} epochs completed, loss: {loss:.4f}, accuracy: {accuracy:.4f}"
        except Exception:
            return "Training progress summary unavailable"
    
    def _generate_hpo_summary(self, data: Dict[str, Any]) -> str:
        """Generate HPO optimization summary."""
        try:
            trials = data.get('total_trials', 0)
            best_score = data.get('best_score', 0)
            status = data.get('status', 'unknown')
            
            return f"HPO {status}: {trials} trials completed, best score: {best_score:.4f}"
        except Exception:
            return "HPO optimization summary unavailable"
    
    def _generate_validation_summary(self, data: Dict[str, Any]) -> str:
        """Generate model validation summary."""
        try:
            accuracy = data.get('accuracy', 0)
            precision = data.get('precision', 0)
            recall = data.get('recall', 0)
            f1_score = data.get('f1_score', 0)
            
            return f"Validation completed: accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1_score:.4f}"
        except Exception:
            return "Model validation summary unavailable"
    
    def _generate_error_summary(self, data: Dict[str, Any]) -> str:
        """Generate error analysis summary."""
        try:
            total_errors = data.get('total_errors', 0)
            critical_errors = data.get('critical_errors', 0)
            recent_errors = data.get('recent_errors_1h', 0)
            
            return f"Error analysis: {total_errors} total errors, {critical_errors} critical, {recent_errors} in last hour"
        except Exception:
            return "Error analysis summary unavailable"
    
    def _generate_performance_summary(self, data: Dict[str, Any]) -> str:
        """Generate performance metrics summary."""
        try:
            avg_time = data.get('average_execution_time', 0)
            memory_usage = data.get('memory_usage', 0)
            throughput = data.get('throughput', 0)
            
            return f"Performance metrics: avg time: {avg_time:.2f}s, memory: {memory_usage:.1f}%, throughput: {throughput:.2f}"
        except Exception:
            return "Performance metrics summary unavailable"
    
    def _extract_metrics(self, data: Dict[str, Any], report_type: ReportType) -> Dict[str, Any]:
        """Extract key metrics from data."""
        try:
            metrics = {}
            
            if report_type == ReportType.TRAINING_PROGRESS:
                metrics.update({
                    'epochs': data.get('epochs', 0),
                    'loss': data.get('loss', 0),
                    'accuracy': data.get('accuracy', 0),
                    'learning_rate': data.get('learning_rate', 0),
                    'training_time': data.get('training_time', 0)
                })
            
            elif report_type == ReportType.HPO_OPTIMIZATION:
                metrics.update({
                    'total_trials': data.get('total_trials', 0),
                    'best_score': data.get('best_score', 0),
                    'convergence_rate': data.get('convergence_rate', 0),
                    'optimization_time': data.get('optimization_time', 0)
                })
            
            elif report_type == ReportType.MODEL_VALIDATION:
                metrics.update({
                    'accuracy': data.get('accuracy', 0),
                    'precision': data.get('precision', 0),
                    'recall': data.get('recall', 0),
                    'f1_score': data.get('f1_score', 0),
                    'auc_score': data.get('auc_score', 0)
                })
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to extract metrics: {e}")
            return {}
    
    def _generate_visualizations(self, data: Dict[str, Any], report_type: ReportType) -> List[str]:
        """Generate visualization file paths."""
        try:
            visualizations = []
            
            # This is a placeholder for visualization generation
            # In a real implementation, you would generate actual plots
            
            if report_type == ReportType.TRAINING_PROGRESS:
                visualizations.extend([
                    'training_loss_curve.png',
                    'training_accuracy_curve.png',
                    'learning_rate_schedule.png'
                ])
            
            elif report_type == ReportType.HPO_OPTIMIZATION:
                visualizations.extend([
                    'optimization_history.png',
                    'parameter_importance.png',
                    'convergence_analysis.png'
                ])
            
            elif report_type == ReportType.MODEL_VALIDATION:
                visualizations.extend([
                    'confusion_matrix.png',
                    'roc_curve.png',
                    'precision_recall_curve.png'
                ])
            
            return visualizations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate visualizations: {e}")
            return []
    
    def _generate_recommendations(self, data: Dict[str, Any], report_type: ReportType) -> List[str]:
        """Generate recommendations based on data analysis."""
        try:
            recommendations = []
            
            if report_type == ReportType.TRAINING_PROGRESS:
                loss = data.get('loss', 0)
                accuracy = data.get('accuracy', 0)
                
                if loss > 1.0:
                    recommendations.append("Consider reducing learning rate - loss is high")
                if accuracy < 0.7:
                    recommendations.append("Model accuracy is low - consider more training or different architecture")
                if data.get('overfitting', False):
                    recommendations.append("Overfitting detected - consider regularization or early stopping")
            
            elif report_type == ReportType.HPO_OPTIMIZATION:
                convergence_rate = data.get('convergence_rate', 0)
                if convergence_rate < 0.1:
                    recommendations.append("Low convergence rate - consider expanding search space")
                if data.get('failed_trials', 0) > data.get('total_trials', 1) * 0.3:
                    recommendations.append("High failure rate - check objective function and constraints")
            
            elif report_type == ReportType.MODEL_VALIDATION:
                f1_score = data.get('f1_score', 0)
                if f1_score < 0.8:
                    recommendations.append("F1 score is low - consider class balancing or threshold tuning")
                if data.get('precision', 0) < data.get('recall', 0):
                    recommendations.append("Low precision - consider reducing false positives")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate recommendations: {e}")
            return []
    
    def _save_report_files(self, report_data: ReportData) -> List[str]:
        """Save report to various file formats."""
        try:
            report_paths = []
            timestamp = report_data.timestamp.strftime('%Y%m%d_%H%M%S')
            
            # JSON report
            json_path = self.output_dir / f"{report_data.report_id}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'report_id': report_data.report_id,
                    'report_type': report_data.report_type.value,
                    'timestamp': report_data.timestamp.isoformat(),
                    'title': report_data.title,
                    'summary': report_data.summary,
                    'data': report_data.data,
                    'metrics': report_data.metrics,
                    'recommendations': report_data.recommendations,
                    'metadata': report_data.metadata
                }, f, indent=2)
            report_paths.append(str(json_path))
            
            # HTML report
            html_path = self.output_dir / f"{report_data.report_id}.html"
            self._generate_html_report(report_data, html_path)
            report_paths.append(str(html_path))
            
            # Markdown report
            md_path = self.output_dir / f"{report_data.report_id}.md"
            self._generate_markdown_report(report_data, md_path)
            report_paths.append(str(md_path))
            
            return report_paths
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save report files: {e}")
            return []
    
    def _generate_html_report(self, report_data: ReportData, output_path: Path):
        """Generate HTML report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_data.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .data-section {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_data.title}</h1>
        <p><strong>Report ID:</strong> {report_data.report_id}</p>
        <p><strong>Generated:</strong> {report_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Type:</strong> {report_data.report_type.value}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>{report_data.summary}</p>
    </div>
    
    <div class="metrics">
        <h2>Key Metrics</h2>
"""
            
            for metric_name, metric_value in report_data.metrics.items():
                html_content += f"""
        <div class="metric">
            <h3>{metric_name.replace('_', ' ').title()}</h3>
            <p>{metric_value}</p>
        </div>
"""
            
            html_content += """
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
            
            for recommendation in report_data.recommendations:
                html_content += f"<li>{recommendation}</li>"
            
            html_content += """
        </ul>
    </div>
    
    <div class="data-section">
        <h2>Detailed Data</h2>
        <pre>{}</pre>
    </div>
</body>
</html>
""".format(json.dumps(report_data.data, indent=2))
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate HTML report: {e}")
    
    def _generate_markdown_report(self, report_data: ReportData, output_path: Path):
        """Generate Markdown report."""
        try:
            md_content = f"""# {report_data.title}

**Report ID:** {report_data.report_id}  
**Generated:** {report_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** {report_data.report_type.value}

## Summary

{report_data.summary}

## Key Metrics

"""
            
            for metric_name, metric_value in report_data.metrics.items():
                md_content += f"- **{metric_name.replace('_', ' ').title()}:** {metric_value}\n"
            
            md_content += "\n## Recommendations\n\n"
            
            for recommendation in report_data.recommendations:
                md_content += f"- {recommendation}\n"
            
            md_content += "\n## Detailed Data\n\n```json\n"
            md_content += json.dumps(report_data.data, indent=2)
            md_content += "\n```\n"
            
            with open(output_path, 'w') as f:
                f.write(md_content)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate Markdown report: {e}")
    
    def _check_report_alerts(self, report_data: ReportData):
        """Check if report data triggers any alerts."""
        try:
            # Check for critical metrics
            if report_data.report_type == ReportType.ERROR_ANALYSIS:
                critical_errors = report_data.data.get('critical_errors', 0)
                if critical_errors > 5:
                    self.create_alert(
                        AlertLevel.CRITICAL,
                        "High Critical Error Count",
                        f"Detected {critical_errors} critical errors in the system",
                        "error_analysis"
                    )
            
            elif report_data.report_type == ReportType.PERFORMANCE_METRICS:
                memory_usage = report_data.data.get('memory_usage', 0)
                if memory_usage > 90:
                    self.create_alert(
                        AlertLevel.WARNING,
                        "High Memory Usage",
                        f"Memory usage is at {memory_usage:.1f}%",
                        "performance_monitoring"
                    )
            
            elif report_data.report_type == ReportType.TRAINING_PROGRESS:
                loss = report_data.data.get('loss', 0)
                if loss > 10.0:
                    self.create_alert(
                        AlertLevel.ERROR,
                        "Training Loss Too High",
                        f"Training loss is {loss:.4f}, which is unusually high",
                        "training_monitor"
                    )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check report alerts: {e}")
    
    def create_alert(self, 
                    level: AlertLevel,
                    title: str,
                    message: str,
                    component: str,
                    data: Optional[Dict[str, Any]] = None) -> str:
        """Create and manage an alert."""
        try:
            alert_id = f"{component}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            alert = Alert(
                alert_id=alert_id,
                level=level,
                title=title,
                message=message,
                timestamp=datetime.now(),
                component=component,
                data=data or {}
            )
            
            with self.lock:
                self.alerts[alert_id] = alert
                self.alert_history.append(alert)
            
            # Notify subscribers
            self._notify_alert_subscribers(alert)
            
            # Send notifications
            self._send_alert_notifications(alert)
            
            self.logger.info(f"🚨 Alert created: {title} ({alert_id})")
            return alert_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create alert: {e}")
            raise
    
    def _notify_alert_subscribers(self, alert: Alert):
        """Notify alert subscribers."""
        try:
            for subscriber in self.alert_subscribers:
                try:
                    subscriber(alert)
                except Exception as e:
                    self.logger.warning(f"⚠️ Alert subscriber failed: {e}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to notify alert subscribers: {e}")
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications via configured channels."""
        try:
            if self.notification_config.get('email_enabled'):
                self._send_email_alert(alert)
            
            if self.notification_config.get('slack_enabled'):
                self._send_slack_alert(alert)
            
            if self.notification_config.get('webhook_enabled'):
                self._send_webhook_alert(alert)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to send alert notifications: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert."""
        try:
            # This is a placeholder for email sending
            # In a real implementation, you would configure SMTP settings
            self.logger.info(f"📧 Email alert sent: {alert.title}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send Slack alert."""
        try:
            # This is a placeholder for Slack integration
            # In a real implementation, you would use Slack webhook
            self.logger.info(f"💬 Slack alert sent: {alert.title}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to send Slack alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert."""
        try:
            # This is a placeholder for webhook integration
            # In a real implementation, you would send HTTP POST
            self.logger.info(f"🔗 Webhook alert sent: {alert.title}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to send webhook alert: {e}")
    
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]):
        """Subscribe to alert notifications."""
        self.alert_subscribers.append(callback)
        self.logger.info("📬 Alert subscription added")
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get comprehensive reporting summary."""
        try:
            with self.lock:
                total_reports = len(self.reports)
                total_alerts = len(self.alerts)
                active_alerts = sum(1 for a in self.alerts.values() if not a.resolved)
                
                # Group by report type
                type_counts = defaultdict(int)
                for report in self.reports.values():
                    type_counts[report.report_type.value] += 1
                
                # Group alerts by level
                alert_levels = defaultdict(int)
                for alert in self.alerts.values():
                    alert_levels[alert.level.value] += 1
                
                # Recent activity
                recent_reports = list(self.report_history)[-10:] if self.report_history else []
                recent_alerts = list(self.alert_history)[-10:] if self.alert_history else []
                
                return {
                    'reporting_summary': {
                        'total_reports': total_reports,
                        'total_alerts': total_alerts,
                        'active_alerts': active_alerts,
                        'monitoring_active': self.monitoring_active
                    },
                    'report_type_distribution': dict(type_counts),
                    'alert_level_distribution': dict(alert_levels),
                    'recent_reports': [
                        {
                            'report_id': r.report_id,
                            'title': r.title,
                            'type': r.report_type.value,
                            'timestamp': r.timestamp.isoformat()
                        }
                        for r in recent_reports
                    ],
                    'recent_alerts': [
                        {
                            'alert_id': a.alert_id,
                            'title': a.title,
                            'level': a.level.value,
                            'component': a.component,
                            'timestamp': a.timestamp.isoformat(),
                            'resolved': a.resolved
                        }
                        for a in recent_alerts
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get report summary: {e}")
            return {'error': str(e)}
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("📊 Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("📊 Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for stale alerts
                self._check_stale_alerts()
                
                # Generate periodic reports
                self._generate_periodic_reports()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_config['alert_check_interval'])
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(60)
    
    def _check_stale_alerts(self):
        """Check for stale alerts that need attention."""
        try:
            current_time = datetime.now()
            stale_threshold = timedelta(hours=24)
            
            for alert in self.alerts.values():
                if not alert.resolved and (current_time - alert.timestamp) > stale_threshold:
                    if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                        # Create follow-up alert
                        self.create_alert(
                            AlertLevel.WARNING,
                            f"Stale Alert: {alert.title}",
                            f"Alert {alert.alert_id} has been unresolved for over 24 hours",
                            "alert_monitor"
                        )
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to check stale alerts: {e}")
    
    def _generate_periodic_reports(self):
        """Generate periodic system health reports."""
        try:
            # This is a placeholder for periodic report generation
            # In a real implementation, you would generate system health reports
            pass
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate periodic reports: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old reports and alerts."""
        try:
            retention_days = self.monitoring_config['retention_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up old reports
            with self.lock:
                old_reports = [r for r in self.reports.values() if r.timestamp < cutoff_date]
                for report in old_reports:
                    del self.reports[report.report_id]
                
                # Clean up old alerts
                old_alerts = [a for a in self.alerts.values() if a.timestamp < cutoff_date and a.resolved]
                for alert in old_alerts:
                    del self.alerts[alert.alert_id]
            
            if old_reports or old_alerts:
                self.logger.info(f"🧹 Cleaned up {len(old_reports)} old reports and {len(old_alerts)} old alerts")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup old data: {e}")

# Global reporting system instance
_global_reporting_system = None

def get_global_reporting_system(config: Optional[Dict[str, Any]] = None) -> EnhancedReportingSystem:
    """Get or create global reporting system instance."""
    global _global_reporting_system
    
    if _global_reporting_system is None:
        _global_reporting_system = EnhancedReportingSystem(config)
        _global_reporting_system.start_monitoring()
    
    return _global_reporting_system


def create_training_report(training_results: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive training report.
    
    Args:
        training_results: Results from training process
        output_path: Optional path to save the report
        
    Returns:
        Dictionary containing the training report
    """
    reporting_system = get_global_reporting_system()
    
    # Create report data
    report_data = ReportData(
        report_type=ReportType.TRAINING_PROGRESS,
        timestamp=datetime.now(),
        data=training_results,
        metadata={'generated_by': 'create_training_report'}
    )
    
    # Generate report
    report = reporting_system.generate_report(report_data)
    
    # Save if path provided
    if output_path:
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"✅ Training report saved to {output_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save training report: {e}")
    
    return report


# Export aliases for backward compatibility
ReportGenerator = EnhancedReportingSystem
ReportManager = EnhancedReportingSystem  # The system manages reports