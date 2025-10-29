"""
Order Execution Utilities

Enhanced utilities for order management including retry logic, expiry handling,
timeout management, and circuit breaker pattern.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Dict
from enum import Enum
from dataclasses import dataclass
import time

from src.utils.logger import system_logger
from src.utils.tprint import tprint_warning, tprint_error, tprint_info

logger = system_logger.getChild('OrderExecutionUtils')

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing
    timeout: float = 60.0  # Timeout before trying again (seconds)
    failure_timeout: float = 300.0  # Timeout when circuit is open (seconds)

class CircuitBreaker:
    """
    Circuit breaker pattern for exchange operations.
    
    Prevents cascading failures by stopping requests when service is failing.
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.failure_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    tprint_info("🔄 Circuit breaker entering HALF_OPEN state")
                    return True
            return False
        
        # HALF_OPEN state
        return True
    
    def record_success(self):
        """Record successful operation."""
        self.success_count += 1
        self.last_success_time = datetime.now()
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                tprint_info("✅ Circuit breaker CLOSED - service recovered")
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                tprint_warning(f"⚠️ Circuit breaker OPENED after {self.failure_count} failures")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state

async def with_timeout(
    coro: Callable,
    timeout: float,
    error_message: str = "Operation timed out"
) -> Any:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        error_message: Error message if timeout occurs
        
    Returns:
        Result of coroutine
        
    Raises:
        asyncio.TimeoutError: If operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        tprint_error(f"❌ {error_message} after {timeout}s")
        raise

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Result of function
        
    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                tprint_warning(
                    f"⚠️ Retry attempt {attempt + 1}/{max_retries} after {delay:.1f}s: {str(e)}"
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                tprint_error(f"❌ All {max_retries + 1} retry attempts failed")
                raise last_exception
    
    raise last_exception

def check_order_expiry(order_expires_at: Optional[datetime]) -> bool:
    """
    Check if order has expired.
    
    Args:
        order_expires_at: Expiration datetime
        
    Returns:
        True if expired, False otherwise
    """
    if order_expires_at is None:
        return False
    
    return datetime.now() >= order_expires_at
