"""
Monitoring manager for pipeline components.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging


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
        self.metrics = {}
        self.alerts = []
        self.logger = self._setup_logger()
        
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
        
    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        timestamp = datetime.now()
        if name not in self.metrics:
            self.metrics[name] = []
            
        metric_entry = {
            'timestamp': timestamp,
            'value': value,
            'tags': tags or {}
        }
        self.metrics[name].append(metric_entry)
        
    def get_metric(self, name: str) -> List[Dict[str, Any]]:
        """
        Get metric values by name.
        
        Args:
            name: Metric name
            
        Returns:
            List of metric entries
        """
        return self.metrics.get(name, [])
        
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
            
    def create_alert(self, severity: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Create an alert.
        
        Args:
            severity: Alert severity level
            message: Alert message
            context: Optional context information
        """
        alert = {
            'timestamp': datetime.now(),
            'severity': severity,
            'message': message,
            'context': context or {}
        }
        self.alerts.append(alert)
        
        # TODO: Implement alert notification logic
        self.logger.warning(f"ALERT [{severity}]: {message}")
        
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts, optionally filtered by severity.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of alerts
        """
        if severity:
            return [alert for alert in self.alerts if alert['severity'] == severity]
        return self.alerts

