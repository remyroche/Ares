"""
Circuit Breaker Implementation

Provides circuit breaker pattern to prevent cascading failures in model training.
"""

from typing import Callable, Any
import time
import logging

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures in model training."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit breaker is open or function fails
        """
        current_time = time.time()
        
        # Check if we should transition from OPEN to HALF_OPEN
        if self.state == "OPEN":
            if current_time - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
            else:
                remaining_time = self.timeout - (current_time - self.last_failure_time)
                error_msg = f"Circuit breaker is OPEN - too many failures detected. Retry in {remaining_time:.1f}s"
                logger.error(f"🚨 {error_msg}")
                raise Exception(error_msg)
        
        try:
            result = func(*args, **kwargs)
            # If we were in HALF_OPEN and succeeded, reset to CLOSED
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("✅ Circuit breaker reset to CLOSED after successful operation")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"🚨 Circuit breaker opened after {self.failure_count} failures. Will retry after {self.timeout}s")
            else:
                logger.warning(f"⚠️ Circuit breaker failure count: {self.failure_count}/{self.failure_threshold}")
            
            raise e
    
    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("✅ Circuit breaker manually reset to CLOSED")
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state
    
    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self.failure_count
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"
    
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == "CLOSED"
    
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "HALF_OPEN"