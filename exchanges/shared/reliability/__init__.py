"""
Reliability and Operations Utilities

Provides utilities for rate limiting, retry management, audit logging,
and system status monitoring.
"""

from .rate_limit_manager import RateLimitManager

# Reliability management classes
class RetryManager:
    """
    Manages retry logic with exponential backoff and circuit breaker patterns.
    
    Provides robust retry mechanisms for API calls, network operations,
    and other potentially failing operations.
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_multiplier: float = 2.0):
        """
        Initialize the RetryManager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retry_stats = {}
        self.circuit_breakers = {}
    
    def execute_with_retry(self, func, *args, operation_name: str = None, 
                          retryable_exceptions: tuple = None, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation_name: Name of the operation for tracking
            retryable_exceptions: Tuple of exceptions that should trigger retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        if retryable_exceptions is None:
            retryable_exceptions = (Exception,)
        
        operation_name = operation_name or func.__name__
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker
                if self._is_circuit_open(operation_name):
                    raise Exception(f"Circuit breaker open for operation: {operation_name}")
                
                result = func(*args, **kwargs)
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(operation_name)
                
                # Record success
                self._record_success(operation_name)
                
                return result
                
            except retryable_exceptions as e:
                last_exception = e
                
                # Record failure
                self._record_failure(operation_name)
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self._log_retry(operation_name, attempt + 1, delay, str(e))
                    
                    import time
                    time.sleep(delay)
                else:
                    self._log_final_failure(operation_name, str(e))
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)
    
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for operation."""
        if operation_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation_name]
        return breaker['state'] == 'open' and breaker['next_attempt'] > self._get_current_time()
    
    def _reset_circuit_breaker(self, operation_name: str):
        """Reset circuit breaker for operation."""
        if operation_name in self.circuit_breakers:
            self.circuit_breakers[operation_name]['state'] = 'closed'
            self.circuit_breakers[operation_name]['failure_count'] = 0
    
    def _record_success(self, operation_name: str):
        """Record successful operation."""
        if operation_name not in self.retry_stats:
            self.retry_stats[operation_name] = {'successes': 0, 'failures': 0, 'retries': 0}
        
        self.retry_stats[operation_name]['successes'] += 1
    
    def _record_failure(self, operation_name: str):
        """Record failed operation."""
        if operation_name not in self.retry_stats:
            self.retry_stats[operation_name] = {'successes': 0, 'failures': 0, 'retries': 0}
        
        self.retry_stats[operation_name]['failures'] += 1
        
        # Update circuit breaker
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = {
                'state': 'closed',
                'failure_count': 0,
                'next_attempt': 0
            }
        
        breaker = self.circuit_breakers[operation_name]
        breaker['failure_count'] += 1
        
        # Open circuit if too many failures
        if breaker['failure_count'] >= 5:  # Threshold for circuit breaker
            breaker['state'] = 'open'
            breaker['next_attempt'] = self._get_current_time() + 300  # 5 minutes
    
    def _log_retry(self, operation_name: str, attempt: int, delay: float, error: str):
        """Log retry attempt."""
        print(f"Retry {attempt} for {operation_name} in {delay:.2f}s. Error: {error}")
    
    def _log_final_failure(self, operation_name: str, error: str):
        """Log final failure after all retries."""
        print(f"Operation {operation_name} failed after all retries. Final error: {error}")
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def get_retry_stats(self) -> dict:
        """Get retry statistics."""
        return self.retry_stats.copy()
    
    def reset_stats(self):
        """Reset retry statistics."""
        self.retry_stats.clear()
        self.circuit_breakers.clear()


class AuditLogger:
    """
    Comprehensive audit logging for trading operations.
    
    Provides structured logging for compliance, debugging, and monitoring.
    """
    
    def __init__(self, log_file: str = None, log_level: str = 'INFO'):
        """
        Initialize the AuditLogger.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        import logging
        import sys
        
        self.logger = logging.getLogger('audit_logger')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.audit_events = []
        self.user_actions = []
    
    def log_trade_execution(self, trade_id: str, symbol: str, side: str, 
                          quantity: float, price: float, user_id: str = None):
        """
        Log trade execution event.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            quantity: Trade quantity
            price: Trade price
            user_id: User identifier (optional)
        """
        event = {
            'type': 'trade_execution',
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'user_id': user_id,
            'timestamp': self._get_timestamp()
        }
        
        self.audit_events.append(event)
        self.logger.info(f"Trade executed: {trade_id} - {side} {quantity} {symbol} @ {price}")
    
    def log_order_placement(self, order_id: str, symbol: str, order_type: str,
                          quantity: float, price: float = None, user_id: str = None):
        """
        Log order placement event.
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            order_type: Type of order ('market', 'limit', 'stop', etc.)
            quantity: Order quantity
            price: Order price (for limit orders)
            user_id: User identifier (optional)
        """
        event = {
            'type': 'order_placement',
            'order_id': order_id,
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'user_id': user_id,
            'timestamp': self._get_timestamp()
        }
        
        self.audit_events.append(event)
        self.logger.info(f"Order placed: {order_id} - {order_type} {quantity} {symbol}")
    
    def log_user_action(self, action: str, user_id: str, details: dict = None):
        """
        Log user action event.
        
        Args:
            action: Action performed
            user_id: User identifier
            details: Additional details (optional)
        """
        event = {
            'type': 'user_action',
            'action': action,
            'user_id': user_id,
            'details': details or {},
            'timestamp': self._get_timestamp()
        }
        
        self.user_actions.append(event)
        self.logger.info(f"User action: {user_id} - {action}")
    
    def log_system_event(self, event_type: str, message: str, severity: str = 'INFO'):
        """
        Log system event.
        
        Args:
            event_type: Type of system event
            message: Event message
            severity: Event severity
        """
        event = {
            'type': 'system_event',
            'event_type': event_type,
            'message': message,
            'severity': severity,
            'timestamp': self._get_timestamp()
        }
        
        self.audit_events.append(event)
        
        # Log based on severity
        if severity.upper() == 'ERROR':
            self.logger.error(f"System event: {event_type} - {message}")
        elif severity.upper() == 'WARNING':
            self.logger.warning(f"System event: {event_type} - {message}")
        else:
            self.logger.info(f"System event: {event_type} - {message}")
    
    def log_security_event(self, event_type: str, user_id: str = None, 
                          ip_address: str = None, details: dict = None):
        """
        Log security-related event.
        
        Args:
            event_type: Type of security event
            user_id: User identifier (optional)
            ip_address: IP address (optional)
            details: Additional details (optional)
        """
        event = {
            'type': 'security_event',
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {},
            'timestamp': self._get_timestamp()
        }
        
        self.audit_events.append(event)
        self.logger.warning(f"Security event: {event_type} - User: {user_id}, IP: {ip_address}")
    
    def get_audit_trail(self, event_type: str = None, user_id: str = None, 
                       start_time: str = None, end_time: str = None) -> list:
        """
        Get filtered audit trail.
        
        Args:
            event_type: Filter by event type (optional)
            user_id: Filter by user ID (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            List of filtered audit events
        """
        events = self.audit_events.copy()
        
        if event_type:
            events = [e for e in events if e.get('type') == event_type]
        
        if user_id:
            events = [e for e in events if e.get('user_id') == user_id]
        
        if start_time:
            events = [e for e in events if e['timestamp'] >= start_time]
        
        if end_time:
            events = [e for e in events if e['timestamp'] <= end_time]
        
        return events
    
    def export_audit_log(self, file_path: str, format: str = 'json'):
        """
        Export audit log to file.
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        import json
        import csv
        
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(self.audit_events, f, indent=2)
        elif format.lower() == 'csv':
            if self.audit_events:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.audit_events[0].keys())
                    writer.writeheader()
                    writer.writerows(self.audit_events)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class SystemStatusManager:
    """
    Monitors and manages system health and status.
    
    Provides comprehensive system monitoring, health checks, and status reporting.
    """
    
    def __init__(self):
        """Initialize the SystemStatusManager."""
        self.system_metrics = {}
        self.health_checks = {}
        self.alerts = []
        self.status_history = []
        self.uptime_start = self._get_current_time()
    
    def register_health_check(self, name: str, check_function, critical: bool = False):
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_function: Function that returns (is_healthy: bool, message: str)
            critical: Whether this check is critical for system health
        """
        self.health_checks[name] = {
            'function': check_function,
            'critical': critical,
            'last_check': None,
            'last_result': None
        }
    
    def run_health_checks(self) -> dict:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary with health check results
        """
        results = {
            'overall_status': 'healthy',
            'checks': {},
            'critical_failures': 0,
            'timestamp': self._get_timestamp()
        }
        
        for name, check_info in self.health_checks.items():
            try:
                is_healthy, message = check_info['function']()
                
                check_result = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'message': message,
                    'timestamp': self._get_timestamp()
                }
                
                results['checks'][name] = check_result
                check_info['last_check'] = self._get_current_time()
                check_info['last_result'] = check_result
                
                if not is_healthy and check_info['critical']:
                    results['critical_failures'] += 1
                    results['overall_status'] = 'unhealthy'
                    self._create_alert(f"Critical health check failed: {name}", 'critical')
                
            except Exception as e:
                check_result = {
                    'status': 'error',
                    'message': f"Health check error: {str(e)}",
                    'timestamp': self._get_timestamp()
                }
                
                results['checks'][name] = check_result
                check_info['last_check'] = self._get_current_time()
                check_info['last_result'] = check_result
                
                if check_info['critical']:
                    results['critical_failures'] += 1
                    results['overall_status'] = 'unhealthy'
                    self._create_alert(f"Critical health check error: {name}", 'critical')
        
        # Record status history
        self.status_history.append(results)
        
        # Keep only last 100 status records
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]
        
        return results
    
    def update_metric(self, name: str, value: float, unit: str = None):
        """
        Update a system metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit (optional)
        """
        self.system_metrics[name] = {
            'value': value,
            'unit': unit,
            'timestamp': self._get_timestamp()
        }
    
    def get_metric(self, name: str) -> dict:
        """Get a system metric."""
        return self.system_metrics.get(name, {})
    
    def get_all_metrics(self) -> dict:
        """Get all system metrics."""
        return self.system_metrics.copy()
    
    def _create_alert(self, message: str, severity: str = 'warning'):
        """Create a system alert."""
        alert = {
            'id': self._generate_alert_id(),
            'message': message,
            'severity': severity,
            'timestamp': self._get_timestamp(),
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False
    
    def get_active_alerts(self) -> list:
        """Get all unacknowledged alerts."""
        return [alert for alert in self.alerts if not alert['acknowledged']]
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        uptime = self._get_current_time() - self.uptime_start
        
        return {
            'overall_status': self._get_overall_status(),
            'uptime_seconds': uptime,
            'uptime_human': self._format_uptime(uptime),
            'total_metrics': len(self.system_metrics),
            'total_health_checks': len(self.health_checks),
            'active_alerts': len(self.get_active_alerts()),
            'last_health_check': self._get_last_health_check_time()
        }
    
    def _get_overall_status(self) -> str:
        """Get overall system status."""
        if not self.status_history:
            return 'unknown'
        
        latest_status = self.status_history[-1]
        return latest_status['overall_status']
    
    def _get_last_health_check_time(self) -> str:
        """Get timestamp of last health check."""
        if not self.status_history:
            return None
        
        return self.status_history[-1]['timestamp']
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()

__all__ = [
    "RateLimitManager",
    "RetryManager",
    "AuditLogger",
    "SystemStatusManager"
]