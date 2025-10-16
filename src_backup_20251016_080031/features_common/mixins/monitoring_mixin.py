"""
Monitoring mixin for comprehensive system monitoring and health checks.

This mixin provides real-time monitoring, health checks, and alerting
capabilities for all features_common components.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Lock

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class MonitoringMixin:
    """
    Mixin class providing comprehensive monitoring capabilities.
    
    This mixin can be added to any class to provide real-time monitoring,
    health checks, and alerting capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize monitoring mixin."""
        super().__init__(*args, **kwargs)
        
        # Get unified configuration
        self.config = get_unified_config()
        
        # Monitoring state
        self._monitoring_enabled = self.config.optimization.enable_performance_monitoring
        self._monitoring_data = []
        self._health_checks = {}
        self._alerts = []
        self._monitoring_lock = Lock()
        
        # Monitoring thresholds
        self._thresholds = {
            'memory_usage': 0.8,  # 80% of available memory
            'cpu_usage': 0.9,     # 90% CPU usage
            'disk_usage': 0.9,    # 90% disk usage
            'response_time': 5.0,  # 5 seconds
            'error_rate': 0.1,    # 10% error rate
            'cache_hit_rate': 0.5  # 50% cache hit rate
        }
        
        # Monitoring statistics
        self._monitoring_stats = {
            'total_checks': 0,
            'healthy_checks': 0,
            'warning_checks': 0,
            'critical_checks': 0,
            'alerts_generated': 0,
            'last_check_time': None,
            'uptime_start': time.time()
        }
        
        # Initialize health checks
        self._initialize_health_checks()
    
    def _initialize_health_checks(self) -> None:
        """Initialize default health checks."""
        self._health_checks = {
            'memory_usage': self._check_memory_usage,
            'cpu_usage': self._check_cpu_usage,
            'disk_usage': self._check_disk_usage,
            'system_responsiveness': self._check_system_responsiveness,
            'cache_health': self._check_cache_health,
            'error_rate': self._check_error_rate
        }
    
    def enable_monitoring(self) -> None:
        """Enable monitoring."""
        self._monitoring_enabled = True
        logger.info("Monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable monitoring."""
        self._monitoring_enabled = False
        logger.info("Monitoring disabled")
    
    def is_monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self._monitoring_enabled
    
    def add_health_check(self, name: str, check_func: Callable) -> None:
        """Add a custom health check."""
        self._health_checks[name] = check_func
        logger.debug(f"Added health check: {name}")
    
    def remove_health_check(self, name: str) -> None:
        """Remove a health check."""
        if name in self._health_checks:
            del self._health_checks[name]
            logger.debug(f"Removed health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        if not self._monitoring_enabled:
            return {'status': 'disabled', 'message': 'Monitoring is disabled'}
        
        with self._monitoring_lock:
            results = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'checks': {},
                'alerts': [],
                'summary': {}
            }
            
            self._monitoring_stats['total_checks'] += 1
            self._monitoring_stats['last_check_time'] = time.time()
            
            # Run each health check
            for name, check_func in self._health_checks.items():
                try:
                    check_result = check_func()
                    results['checks'][name] = check_result
                    
                    # Update statistics
                    if check_result['status'] == 'healthy':
                        self._monitoring_stats['healthy_checks'] += 1
                    elif check_result['status'] == 'warning':
                        self._monitoring_stats['warning_checks'] += 1
                        results['overall_status'] = 'warning'
                    elif check_result['status'] == 'critical':
                        self._monitoring_stats['critical_checks'] += 1
                        results['overall_status'] = 'critical'
                    
                    # Generate alerts if needed
                    if check_result['status'] in ['warning', 'critical']:
                        alert = self._generate_alert(name, check_result)
                        if alert:
                            results['alerts'].append(alert)
                            self._alerts.append(alert)
                            self._monitoring_stats['alerts_generated'] += 1
                
                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'message': f"Health check failed: {e}",
                        'timestamp': datetime.now().isoformat()
                    }
                    results['checks'][name] = error_result
                    logger.error(f"Health check {name} failed: {e}")
            
            # Generate summary
            results['summary'] = self._generate_summary(results)
            
            # Store monitoring data
            self._monitoring_data.append(results)
            
            # Keep only recent data
            if len(self._monitoring_data) > 1000:
                self._monitoring_data = self._monitoring_data[-500:]
            
            return results
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0
            
            if usage_percent >= self._thresholds['memory_usage']:
                status = 'critical' if usage_percent >= 0.95 else 'warning'
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = 'healthy'
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'value': memory.percent,
                'threshold': self._thresholds['memory_usage'] * 100,
                'details': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'free': memory.free
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Memory check failed: {e}",
                'value': None,
                'threshold': None
            }
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            usage_percent = cpu_percent / 100.0
            
            if usage_percent >= self._thresholds['cpu_usage']:
                status = 'critical' if usage_percent >= 0.95 else 'warning'
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = 'healthy'
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'value': cpu_percent,
                'threshold': self._thresholds['cpu_usage'] * 100,
                'details': {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"CPU check failed: {e}",
                'value': None,
                'threshold': None
            }
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent / 100.0
            
            if usage_percent >= self._thresholds['disk_usage']:
                status = 'critical' if usage_percent >= 0.95 else 'warning'
                message = f"High disk usage: {disk.percent:.1f}%"
            else:
                status = 'healthy'
                message = f"Disk usage normal: {disk.percent:.1f}%"
            
            return {
                'status': status,
                'message': message,
                'value': disk.percent,
                'threshold': self._thresholds['disk_usage'] * 100,
                'details': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Disk check failed: {e}",
                'value': None,
                'threshold': None
            }
    
    def _check_system_responsiveness(self) -> Dict[str, Any]:
        """Check system responsiveness."""
        try:
            start_time = time.time()
            
            # Perform a simple operation to test responsiveness
            test_data = pd.Series(np.random.randn(1000))
            test_data.mean()
            
            response_time = time.time() - start_time
            
            if response_time >= self._thresholds['response_time']:
                status = 'critical' if response_time >= self._thresholds['response_time'] * 2 else 'warning'
                message = f"Slow system response: {response_time:.3f}s"
            else:
                status = 'healthy'
                message = f"System responsive: {response_time:.3f}s"
            
            return {
                'status': status,
                'message': message,
                'value': response_time,
                'threshold': self._thresholds['response_time'],
                'details': {
                    'test_operation': 'pandas_mean',
                    'test_size': 1000
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Responsiveness check failed: {e}",
                'value': None,
                'threshold': None
            }
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            # This would check cache statistics if available
            if hasattr(self, 'get_cache_stats'):
                cache_stats = self.get_cache_stats()
                hit_rate = cache_stats.get('hit_rate', 0)
                
                if hit_rate < self._thresholds['cache_hit_rate']:
                    status = 'warning'
                    message = f"Low cache hit rate: {hit_rate:.1%}"
                else:
                    status = 'healthy'
                    message = f"Cache healthy: {hit_rate:.1%} hit rate"
                
                return {
                    'status': status,
                    'message': message,
                    'value': hit_rate,
                    'threshold': self._thresholds['cache_hit_rate'],
                    'details': cache_stats
                }
            else:
                return {
                    'status': 'healthy',
                    'message': 'Cache health check not available',
                    'value': None,
                    'threshold': None
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Cache health check failed: {e}",
                'value': None,
                'threshold': None
            }
    
    def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate."""
        try:
            # This would check error statistics if available
            if hasattr(self, 'get_validation_stats'):
                validation_stats = self.get_validation_stats()
                total_validations = validation_stats.get('total_validations', 0)
                failed_validations = validation_stats.get('failed_validations', 0)
                
                if total_validations > 0:
                    error_rate = failed_validations / total_validations
                    
                    if error_rate >= self._thresholds['error_rate']:
                        status = 'critical' if error_rate >= self._thresholds['error_rate'] * 2 else 'warning'
                        message = f"High error rate: {error_rate:.1%}"
                    else:
                        status = 'healthy'
                        message = f"Error rate normal: {error_rate:.1%}"
                    
                    return {
                        'status': status,
                        'message': message,
                        'value': error_rate,
                        'threshold': self._thresholds['error_rate'],
                        'details': validation_stats
                    }
                else:
                    return {
                        'status': 'healthy',
                        'message': 'No validation data available',
                        'value': 0,
                        'threshold': self._thresholds['error_rate']
                    }
            else:
                return {
                    'status': 'healthy',
                    'message': 'Error rate check not available',
                    'value': None,
                    'threshold': None
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Error rate check failed: {e}",
                'value': None,
                'threshold': None
            }
    
    def _generate_alert(self, check_name: str, check_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate an alert based on health check result."""
        if check_result['status'] in ['warning', 'critical']:
            alert = {
                'id': f"{check_name}_{int(time.time())}",
                'check_name': check_name,
                'status': check_result['status'],
                'message': check_result['message'],
                'timestamp': datetime.now().isoformat(),
                'value': check_result.get('value'),
                'threshold': check_result.get('threshold'),
                'acknowledged': False
            }
            return alert
        return None
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitoring summary."""
        checks = results['checks']
        
        summary = {
            'total_checks': len(checks),
            'healthy_checks': sum(1 for c in checks.values() if c['status'] == 'healthy'),
            'warning_checks': sum(1 for c in checks.values() if c['status'] == 'warning'),
            'critical_checks': sum(1 for c in checks.values() if c['status'] == 'critical'),
            'error_checks': sum(1 for c in checks.values() if c['status'] == 'error'),
            'alerts_count': len(results['alerts']),
            'uptime': time.time() - self._monitoring_stats['uptime_start']
        }
        
        return summary
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = self._monitoring_stats.copy()
        
        # Calculate success rates
        if stats['total_checks'] > 0:
            stats['healthy_rate'] = stats['healthy_checks'] / stats['total_checks']
            stats['warning_rate'] = stats['warning_checks'] / stats['total_checks']
            stats['critical_rate'] = stats['critical_checks'] / stats['total_checks']
        else:
            stats['healthy_rate'] = 0.0
            stats['warning_rate'] = 0.0
            stats['critical_rate'] = 0.0
        
        # Add recent alerts
        stats['recent_alerts'] = self._alerts[-10:] if self._alerts else []
        
        return stats
    
    def get_health_status(self) -> str:
        """Get overall health status."""
        if not self._monitoring_enabled:
            return 'disabled'
        
        if self._monitoring_stats['critical_checks'] > 0:
            return 'critical'
        elif self._monitoring_stats['warning_checks'] > 0:
            return 'warning'
        else:
            return 'healthy'
    
    def get_alerts(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by status."""
        alerts = self._alerts
        
        if status:
            alerts = [a for a in alerts if a['status'] == status]
        
        return alerts[-limit:] if limit else alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False
    
    def set_thresholds(self, **thresholds) -> None:
        """Set monitoring thresholds."""
        for key, value in thresholds.items():
            if key in self._thresholds:
                self._thresholds[key] = value
            else:
                logger.warning(f"Unknown threshold: {key}")
    
    def reset_monitoring_stats(self) -> None:
        """Reset monitoring statistics."""
        self._monitoring_stats = {
            'total_checks': 0,
            'healthy_checks': 0,
            'warning_checks': 0,
            'critical_checks': 0,
            'alerts_generated': 0,
            'last_check_time': None,
            'uptime_start': time.time()
        }
        self._monitoring_data = []
        self._alerts = []
    
    def get_monitoring_recommendations(self) -> List[str]:
        """Get monitoring recommendations."""
        recommendations = []
        stats = self.get_monitoring_stats()
        
        # Check critical rate
        if stats['critical_rate'] > 0.1:
            recommendations.append("High critical check rate - investigate system issues")
        
        # Check warning rate
        if stats['warning_rate'] > 0.3:
            recommendations.append("High warning rate - consider adjusting thresholds")
        
        # Check alert count
        if len(self._alerts) > 50:
            recommendations.append("Many unacknowledged alerts - review and acknowledge")
        
        # Check uptime
        if stats['uptime'] > 86400:  # 24 hours
            recommendations.append("System running for extended period - consider restart")
        
        return recommendations