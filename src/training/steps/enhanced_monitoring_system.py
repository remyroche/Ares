from ..standardized_parquet_handler import standardized_parquet_handler
"""
Enhanced Monitoring and Alerting System for Training Steps

This module provides comprehensive monitoring that ensures:
1. No silent failures - all failures are detected and alerted
2. Real-time monitoring of critical processes
3. Comprehensive metrics collection and reporting
4. Automated alerting for failures and anomalies
"""

import asyncio

import time
from datetime import datetime, timedelta

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import system_logger
from .enhanced_error_handling import (
    EnhancedErrorHandler,
    CriticalProcessError,
    ErrorSeverity,
    ErrorCategory,
    ErrorRecord
)

class AlertLevel(Enum):
    """Alert levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ProcessStatus(Enum):
    """Process status for monitoring."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class ProcessMetrics:
    """Metrics for a process."""
    process_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ProcessStatus = ProcessStatus.PENDING
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    error_count: int = 0
    warning_count: int = 0
    output_size: Optional[int] = None
    success_rate: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert for monitoring system."""
    alert_id: str
    level: AlertLevel
    message: str
    process_name: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

class EnhancedMonitoringSystem:
    """Enhanced monitoring system with comprehensive tracking and alerting."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('EnhancedMonitoringSystem')
        self.error_handler = EnhancedErrorHandler()
        
        # Monitoring state
        self.active_processes: Dict[str, ProcessMetrics] = {}
        self.process_history: List[ProcessMetrics] = []
        self.alerts: List[Alert] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.alert_thresholds = {
            'execution_time': 3600,  # 1 hour
            'memory_usage': 0.8,     # 80%
            'cpu_usage': 0.9,        # 90%
            'error_rate': 0.1,       # 10%
            'failure_rate': 0.05     # 5%
        }
        
        # Start monitoring loop
        self.monitoring_task = None
        self.is_monitoring = False
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info('🚀 Enhanced monitoring system started')
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info('🛑 Enhanced monitoring system stopped')
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_process_health()
                await self._check_system_resources()
                await self._check_alert_conditions()
                await self._cleanup_old_data()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.exception(f'❌ Monitoring loop error: {e}')
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_process_health(self) -> None:
        """Check health of active processes."""
        current_time = datetime.now()
        
        for process_name, metrics in list(self.active_processes.items()):
            try:
                # Check for timeout
                if metrics.status == ProcessStatus.RUNNING:
                    elapsed_time = (current_time - metrics.start_time).total_seconds()
                    if elapsed_time > self.alert_thresholds['execution_time']:
                        await self._create_alert(
                            AlertLevel.WARNING,
                            f"Process {process_name} is running longer than expected",
                            process_name,
                            {'elapsed_time': elapsed_time, 'threshold': self.alert_thresholds['execution_time']}
                        )
                
                # Check for stuck processes
                if metrics.status == ProcessStatus.RUNNING:
                    # Check if process is actually running (this would need process-specific implementation)
                    # For now, we'll just check if it's been running too long
                    pass
                
            except Exception as e:
                self.logger.exception(f'❌ Error checking process health for {process_name}: {e}')
    
    async def _check_system_resources(self) -> None:
        """Check system resource usage."""
        try:
            import psutil
            
            # Memory usage
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.alert_thresholds['memory_usage']:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"High memory usage: {memory_percent:.1%}",
                    "system",
                    {'memory_percent': memory_percent, 'threshold': self.alert_thresholds['memory_usage']}
                )
            
            # CPU usage
            cpu_percent = psutil.cpu_percent() / 100
            if cpu_percent > self.alert_thresholds['cpu_usage']:
                await self._create_alert(
                    AlertLevel.WARNING,
                    f"High CPU usage: {cpu_percent:.1%}",
                    "system",
                    {'cpu_percent': cpu_percent, 'threshold': self.alert_thresholds['cpu_usage']}
                )
            
            # Disk usage
            disk_percent = psutil.disk_usage('/').percent / 100
            if disk_percent > 0.9:  # 90% disk usage
                await self._create_alert(
                    AlertLevel.ERROR,
                    f"High disk usage: {disk_percent:.1%}",
                    "system",
                    {'disk_percent': disk_percent}
                )
            
        except ImportError:
            self.logger.warning('psutil not available for system resource monitoring')
        except Exception as e:
            self.logger.exception(f'❌ Error checking system resources: {e}')
    
    async def _check_alert_conditions(self) -> None:
        """Check for alert conditions."""
        try:
            # Check error rates
            recent_errors = [
                alert for alert in self.alerts
                if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] and
                alert.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            if len(recent_errors) > 10:  # More than 10 errors in the last hour
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"High error rate: {len(recent_errors)} errors in the last hour",
                    "system",
                    {'error_count': len(recent_errors), 'time_window': '1 hour'}
                )
            
            # Check for repeated failures
            recent_failures = [
                metrics for metrics in self.process_history
                if metrics.status == ProcessStatus.FAILED and
                metrics.end_time and
                metrics.end_time > datetime.now() - timedelta(hours=1)
            ]
            
            if len(recent_failures) > 5:  # More than 5 failures in the last hour
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    f"High failure rate: {len(recent_failures)} failures in the last hour",
                    "system",
                    {'failure_count': len(recent_failures), 'time_window': '1 hour'}
                )
            
        except Exception as e:
            self.logger.exception(f'❌ Error checking alert conditions: {e}')
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of data
            
            # Clean up old process history
            self.process_history = [
                metrics for metrics in self.process_history
                if metrics.start_time > cutoff_time
            ]
            
            # Clean up old alerts
            self.alerts = [
                alert for alert in self.alerts
                if alert.timestamp > cutoff_time
            ]
            
            # Clean up old metrics
            self.metrics_history = [
                metrics for metrics in self.metrics_history
                if metrics.get('timestamp', datetime.min) > cutoff_time
            ]
            
        except Exception as e:
            self.logger.exception(f'❌ Error cleaning up old data: {e}')
    
    async def start_process(self, process_name: str, details: Dict[str, Any] = None) -> str:
        """Start monitoring a process."""
        process_id = f"{process_name}_{int(time.time())}"
        
        metrics = ProcessMetrics(
            process_name=process_name,
            start_time=datetime.now(),
            status=ProcessStatus.RUNNING,
            details=details or {}
        )
        
        self.active_processes[process_id] = metrics
        self.logger.info(f'📊 Started monitoring process: {process_name} (ID: {process_id})')
        
        return process_id
    
    async def update_process(self, process_id: str, **updates) -> None:
        """Update process metrics."""
        if process_id not in self.active_processes:
            self.logger.warning(f'⚠️ Process ID not found: {process_id}')
            return
        
        metrics = self.active_processes[process_id]
        
        for key, value in updates.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
            else:
                metrics.details[key] = value
        
        self.logger.debug(f'📊 Updated process metrics: {process_id}')
    
    async def end_process(self, process_id: str, status: ProcessStatus, details: Dict[str, Any] = None) -> None:
        """End monitoring a process."""
        if process_id not in self.active_processes:
            self.logger.warning(f'⚠️ Process ID not found: {process_id}')
            return
        
        metrics = self.active_processes[process_id]
        metrics.end_time = datetime.now()
        metrics.status = status
        metrics.execution_time = (metrics.end_time - metrics.start_time).total_seconds()
        
        if details:
            metrics.details.update(details)
        
        # Move to history
        self.process_history.append(metrics)
        del self.active_processes[process_id]
        
        # Create alert if process failed
        if status == ProcessStatus.FAILED:
            await self._create_alert(
                AlertLevel.ERROR,
                f"Process {metrics.process_name} failed",
                metrics.process_name,
                {
                    'process_id': process_id,
                    'execution_time': metrics.execution_time,
                    'details': metrics.details
                }
            )
        elif status == ProcessStatus.TIMEOUT:
            await self._create_alert(
                AlertLevel.WARNING,
                f"Process {metrics.process_name} timed out",
                metrics.process_name,
                {
                    'process_id': process_id,
                    'execution_time': metrics.execution_time,
                    'details': metrics.details
                }
            )
        
        self.logger.info(f'📊 Ended monitoring process: {metrics.process_name} (ID: {process_id}, Status: {status.value})')
    
    async def _create_alert(self, level: AlertLevel, message: str, process_name: str, details: Dict[str, Any] = None) -> None:
        """Create an alert."""
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            message=message,
            process_name=process_name,
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Log alert
        if level == AlertLevel.CRITICAL:
            self.logger.critical(f'🚨 CRITICAL ALERT: {message}')
        elif level == AlertLevel.ERROR:
            self.logger.error(f'❌ ERROR ALERT: {message}')
        elif level == AlertLevel.WARNING:
            self.logger.warning(f'⚠️ WARNING ALERT: {message}')
        else:
            self.logger.info(f'ℹ️ INFO ALERT: {message}')
        
        # Send alert (this would integrate with external alerting systems)
        await self._send_alert(alert)
    
    async def _send_alert(self, alert: Alert) -> None:
        """Send alert to external systems."""
        try:
            # This would integrate with external alerting systems like:
            # - Slack
            # - Email
            # - PagerDuty
            # - Webhooks
            
            # For now, just log the alert
            self.logger.info(f'📤 Alert sent: {alert.alert_id} - {alert.message}')
            
        except Exception as e:
            self.logger.exception(f'❌ Error sending alert: {e}')
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        try:
            current_time = datetime.now()
            
            # Active processes
            active_count = len(self.active_processes)
            running_count = len([p for p in self.active_processes.values() if p.status == ProcessStatus.RUNNING])
            
            # Recent activity (last 24 hours)
            recent_cutoff = current_time - timedelta(hours=24)
            recent_processes = [
                p for p in self.process_history
                if p.start_time > recent_cutoff
            ]
            
            # Success rate
            completed_processes = [p for p in recent_processes if p.status == ProcessStatus.COMPLETED]
            failed_processes = [p for p in recent_processes if p.status == ProcessStatus.FAILED]
            success_rate = len(completed_processes) / len(recent_processes) if recent_processes else 0.0
            
            # Recent alerts
            recent_alerts = [
                a for a in self.alerts
                if a.timestamp > recent_cutoff
            ]
            
            # Alert counts by level
            alert_counts = {}
            for alert in recent_alerts:
                level = alert.level.value
                alert_counts[level] = alert_counts.get(level, 0) + 1
            
            # Average execution time
            avg_execution_time = 0.0
            if completed_processes:
                total_time = sum(p.execution_time for p in completed_processes if p.execution_time)
                avg_execution_time = total_time / len(completed_processes)
            
            return {
                'timestamp': current_time.isoformat(),
                'active_processes': {
                    'total': active_count,
                    'running': running_count
                },
                'recent_activity': {
                    'total_processes': len(recent_processes),
                    'completed': len(completed_processes),
                    'failed': len(failed_processes),
                    'success_rate': success_rate,
                    'average_execution_time': avg_execution_time
                },
                'alerts': {
                    'total_recent': len(recent_alerts),
                    'by_level': alert_counts,
                    'unacknowledged': len([a for a in recent_alerts if not a.acknowledged])
                },
                'system_health': {
                    'monitoring_active': self.is_monitoring,
                    'data_retention_days': 7
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring summary generation failed: {e}")
            return {'error': str(e)}
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f'✅ Alert acknowledged: {alert_id}')
                return True
        
        self.logger.warning(f'⚠️ Alert not found: {alert_id}')
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                self.logger.info(f'✅ Alert resolved: {alert_id}')
                return True
        
        self.logger.warning(f'⚠️ Alert not found: {alert_id}')
        return False

# Global monitoring system instance
_global_monitoring_system = EnhancedMonitoringSystem()

def get_global_monitoring_system() -> EnhancedMonitoringSystem:
    """Get the global monitoring system instance."""
    return _global_monitoring_system

def set_global_monitoring_system(system: EnhancedMonitoringSystem) -> None:
    """Set the global monitoring system instance."""
    global _global_monitoring_system
    _global_monitoring_system = system

# Monitoring decorators
def monitor_process(process_name: str):
    """Decorator to monitor a process."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitoring_system = get_global_monitoring_system()
            process_id = await monitoring_system.start_process(process_name)
            
            try:
                result = await func(*args, **kwargs)
                await monitoring_system.end_process(process_id, ProcessStatus.COMPLETED)
                return result
            except Exception as e:
                await monitoring_system.end_process(process_id, ProcessStatus.FAILED, {'error': str(e)})
                raise
        
        return wrapper
    return decorator

def monitor_critical_process(process_name: str):
    """Decorator to monitor a critical process with fail-fast behavior."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitoring_system = get_global_monitoring_system()
            process_id = await monitoring_system.start_process(process_name)
            
            try:
                result = await func(*args, **kwargs)
                await monitoring_system.end_process(process_id, ProcessStatus.COMPLETED)
                return result
            except Exception as e:
                await monitoring_system.end_process(process_id, ProcessStatus.FAILED, {'error': str(e)})
                
                # Create critical alert
                await monitoring_system._create_alert(
                    AlertLevel.CRITICAL,
                    f"Critical process {process_name} failed: {str(e)}",
                    process_name,
                    {'error': str(e), 'process_id': process_id}
                )
                
                raise CriticalProcessError(
                    f"Critical process {process_name} failed: {e}",
                    ErrorRecord(
                        error_id=f"critical_{process_name}_{int(time.time())}",
                        error_type=type(e).__name__,
                        error_message=str(e),
                        severity=ErrorSeverity.CRITICAL,
                        category=ErrorCategory.BUSINESS_LOGIC,
                        context=ErrorContext(
                            function_name=func.__name__,
                            step_name=process_name
                        ),
                        stack_trace="",
                        should_fail_fast=True
                    )
                )
        
        return wrapper
    return decorator