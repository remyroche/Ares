"""
Circuit Breaker Implementation

Provides circuit breaker pattern to prevent cascading failures in model training.
Thread-safe implementation with atomic operations.
"""

from typing import Callable, Any
import time
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Thread-safe circuit breaker to prevent cascading failures in model training."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        """
        Initialize circuit breaker with thread safety.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()  # Re-entrant lock for thread safety
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with thread-safe circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit breaker is open or function fails
        """
        with self._lock:
            current_time = time.time()
            
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == "OPEN":
                if self.last_failure_time and current_time - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
                else:
                    remaining_time = self.timeout - (current_time - (self.last_failure_time or current_time))
                    error_msg = f"Circuit breaker is OPEN - too many failures detected. Retry in {remaining_time:.1f}s"
                    logger.error(f"🚨 {error_msg}")
                    raise Exception(error_msg)
        
        # Execute function outside of lock to avoid deadlocks
        try:
            result = func(*args, **kwargs)
            # If we were in HALF_OPEN and succeeded, reset to CLOSED
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("✅ Circuit breaker reset to CLOSED after successful operation")
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"🚨 Circuit breaker opened after {self.failure_count} failures. Will retry after {self.timeout}s")
                else:
                    logger.warning(f"⚠️ Circuit breaker failure count: {self.failure_count}/{self.failure_threshold}")
            
            raise e
    
    def reset(self) -> None:
        """Thread-safe reset circuit breaker to CLOSED state."""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("✅ Circuit breaker manually reset to CLOSED")
    
    def get_state(self) -> str:
        """Thread-safe get current circuit breaker state."""
        with self._lock:
            return self.state
    
    def get_failure_count(self) -> int:
        """Thread-safe get current failure count."""
        with self._lock:
            return self.failure_count
    
    def is_open(self) -> bool:
        """Thread-safe check if circuit breaker is open."""
        with self._lock:
            return self.state == "OPEN"
    
    def is_closed(self) -> bool:
        """Thread-safe check if circuit breaker is closed."""
        with self._lock:
            return self.state == "CLOSED"
    
    def is_half_open(self) -> bool:
        """Thread-safe check if circuit breaker is half-open."""
        with self._lock:
            return self.state == "HALF_OPEN"
    
    @contextmanager
    def protected_execution(self, operation_name: str = "operation"):
        """
        Context manager for protected execution with automatic error handling.
        
        Args:
            operation_name: Name of the operation for logging
        """
        try:
            logger.debug(f"Starting protected execution: {operation_name}")
            yield self
            logger.debug(f"Completed protected execution: {operation_name}")
        except Exception as e:
            logger.error(f"Protected execution failed for {operation_name}: {e}")
            raise