"""
Advanced Error Recovery System

This module provides comprehensive error recovery capabilities including:
- Circuit breaker patterns
- Exponential backoff with jitter
- Automatic retry with different strategies
- Graceful degradation modes
- Error classification and routing
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta

from src.utils.logger import system_logger

logger = system_logger.getChild('AdvancedErrorRecovery')

class ErrorType(Enum):
    """Types of errors for classification."""
    NETWORK = "network"
    API_RATE_LIMIT = "api_rate_limit"
    API_SERVER_ERROR = "api_server_error"
    DATA_VALIDATION = "data_validation"
    MEMORY_ERROR = "memory_error"
    FILE_IO = "file_io"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 1.0

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: ErrorType
    error_message: str
    retry_count: int = 0
    last_attempt_time: Optional[datetime] = None
    context_data: Dict[str, Any] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger = logger.getChild('CircuitBreaker')
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("✅ Circuit breaker reset to CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"🚨 Circuit breaker OPENED after {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout

class AdvancedErrorRecovery:
    """Advanced error recovery system with circuit breakers and retry strategies."""
    
    def __init__(self):
        self.logger = logger.getChild('AdvancedErrorRecovery')
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.retry_strategies: Dict[ErrorType, RetryConfig] = {
            ErrorType.NETWORK: RetryConfig(max_attempts=5, base_delay=1.0, max_delay=30.0),
            ErrorType.API_RATE_LIMIT: RetryConfig(max_attempts=3, base_delay=5.0, max_delay=60.0),
            ErrorType.API_SERVER_ERROR: RetryConfig(max_attempts=3, base_delay=2.0, max_delay=30.0),
            ErrorType.DATA_VALIDATION: RetryConfig(max_attempts=2, base_delay=0.5, max_delay=5.0),
            ErrorType.MEMORY_ERROR: RetryConfig(max_attempts=2, base_delay=1.0, max_delay=10.0),
            ErrorType.FILE_IO: RetryConfig(max_attempts=3, base_delay=1.0, max_delay=15.0),
            ErrorType.TIMEOUT: RetryConfig(max_attempts=4, base_delay=2.0, max_delay=20.0),
            ErrorType.UNKNOWN: RetryConfig(max_attempts=2, base_delay=1.0, max_delay=10.0)
        }
        
        self.logger.info("🛡️ Advanced Error Recovery system initialized")
    
    def get_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(config)
        return self.circuit_breakers[service_name]
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['connection', 'network', 'socket', 'timeout']):
            return ErrorType.NETWORK
        
        # API rate limiting
        if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429']):
            return ErrorType.API_RATE_LIMIT
        
        # API server errors
        if any(keyword in error_str for keyword in ['500', '502', '503', '504', 'server error']):
            return ErrorType.API_SERVER_ERROR
        
        # Data validation errors
        if any(keyword in error_str for keyword in ['validation', 'invalid', 'format', 'schema']):
            return ErrorType.DATA_VALIDATION
        
        # Memory errors
        if any(keyword in error_str for keyword in ['memory', 'out of memory', 'memoryerror']):
            return ErrorType.MEMORY_ERROR
        
        # File I/O errors
        if any(keyword in error_str for keyword in ['file', 'io', 'permission', 'not found']):
            return ErrorType.FILE_IO
        
        # Timeout errors
        if 'timeout' in error_str or 'TimeoutError' in error_type:
            return ErrorType.TIMEOUT
        
        return ErrorType.UNKNOWN
    
    def calculate_backoff_delay(self, retry_count: int, config: RetryConfig) -> float:
        """Calculate exponential backoff delay with jitter."""
        # Exponential backoff
        delay = config.base_delay * (config.exponential_base ** retry_count)
        
        # Apply backoff multiplier
        delay *= config.backoff_multiplier
        
        # Cap at max delay
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    async def execute_with_retry(self, 
                                func: Callable,
                                service_name: str = "default",
                                retry_config: Optional[RetryConfig] = None,
                                circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                                *args, **kwargs) -> Any:
        """Execute function with advanced error recovery."""
        circuit_breaker = self.get_circuit_breaker(service_name, circuit_breaker_config)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker OPEN for service {service_name}")
        
        last_error = None
        retry_count = 0
        
        while retry_count < (retry_config.max_attempts if retry_config else 3):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                circuit_breaker.record_success()
                self.logger.info(f"✅ Successfully executed {service_name} (attempt {retry_count + 1})")
                return result
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # Classify error
                error_type = self.classify_error(e)
                
                # Record error context
                error_context = ErrorContext(
                    error_type=error_type,
                    error_message=str(e),
                    retry_count=retry_count,
                    last_attempt_time=datetime.now(),
                    context_data={'service': service_name, 'function': func.__name__}
                )
                self.error_history.append(error_context)
                
                # Record failure in circuit breaker
                circuit_breaker.record_failure()
                
                # Get retry configuration
                config = retry_config or self.retry_strategies.get(error_type, RetryConfig())
                
                # Check if we should retry
                if retry_count >= config.max_attempts:
                    self.logger.error(f"❌ Max retry attempts reached for {service_name}: {e}")
                    break
                
                # Calculate delay
                delay = self.calculate_backoff_delay(retry_count - 1, config)
                
                self.logger.warning(f"⚠️ Retry {retry_count}/{config.max_attempts} for {service_name} in {delay:.2f}s: {e}")
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries failed
        raise last_error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}, 'circuit_breakers': {}}
        
        # Count error types
        error_types = {}
        for error in self.error_history:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Circuit breaker states
        circuit_states = {}
        for service, breaker in self.circuit_breakers.items():
            circuit_states[service] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count
            }
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'circuit_breakers': circuit_states,
            'recent_errors': [
                {
                    'type': e.error_type.value,
                    'message': e.error_message,
                    'retry_count': e.retry_count,
                    'timestamp': e.last_attempt_time.isoformat() if e.last_attempt_time else None
                }
                for e in self.error_history[-10:]  # Last 10 errors
            ]
        }

# Global instance
_error_recovery: Optional[AdvancedErrorRecovery] = None

def get_error_recovery() -> AdvancedErrorRecovery:
    """Get the global error recovery instance."""
    global _error_recovery
    if _error_recovery is None:
        _error_recovery = AdvancedErrorRecovery()
    return _error_recovery

def with_error_recovery(service_name: str = "default", 
                       retry_config: Optional[RetryConfig] = None,
                       circuit_breaker_config: Optional[CircuitBreakerConfig] = None):
    """Decorator for automatic error recovery."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            error_recovery = get_error_recovery()
            return await error_recovery.execute_with_retry(
                func, service_name, retry_config, circuit_breaker_config, *args, **kwargs
            )
        return wrapper
    return decorator