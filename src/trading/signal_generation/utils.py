"""
Signal Generation Utilities

Common utilities for signal generation including validation, rate limiting,
circuit breaking, and signal deduplication.
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from threading import Lock
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger

logger = system_logger.getChild('SignalGenerationUtils')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for signal generation failure handling."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            success_threshold: Number of successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise RuntimeError("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("✅ Circuit breaker CLOSED - service recovered")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("⚠️ Circuit breaker OPEN - recovery failed")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(f"❌ Circuit breaker OPEN - {self.failure_count} failures")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None


class RateLimiter:
    """Rate limiter for signal generation."""
    
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times: deque = deque(maxlen=max_calls)
        self._lock = Lock()
    
    def acquire(self) -> bool:
        """
        Try to acquire permission for a call.
        
        Returns:
            True if call is allowed, False if rate limit exceeded
        """
        with self._lock:
            now = time.time()
            
            # Remove old calls outside time window
            while self.call_times and self.call_times[0] < now - self.time_window:
                self.call_times.popleft()
            
            if len(self.call_times) >= self.max_calls:
                return False
            
            self.call_times.append(now)
            return True
    
    def wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        with self._lock:
            if len(self.call_times) < self.max_calls:
                return 0.0
            
            oldest_call = self.call_times[0]
            wait_time = self.time_window - (time.time() - oldest_call)
            return max(0.0, wait_time)


class SignalDeduplicator:
    """Signal deduplication to prevent redundant signals."""
    
    def __init__(self, deduplication_window: float = 300.0):
        """
        Initialize signal deduplicator.
        
        Args:
            deduplication_window: Time window in seconds for deduplication
        """
        self.deduplication_window = deduplication_window
        self.recent_signals: deque = deque(maxlen=100)
        self._lock = Lock()
    
    def is_duplicate(
        self,
        symbol: str,
        signal_type: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Check if signal is a duplicate.
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type (buy/sell/hold/close)
            timestamp: Signal timestamp (defaults to now)
        
        Returns:
            True if duplicate, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            # Remove old signals outside window
            cutoff_time = timestamp - timedelta(seconds=self.deduplication_window)
            while self.recent_signals and self.recent_signals[0]['timestamp'] < cutoff_time:
                self.recent_signals.popleft()
            
            # Check for duplicates
            for signal in self.recent_signals:
                if (signal['symbol'] == symbol and
                    signal['signal_type'] == signal_type and
                    (timestamp - signal['timestamp']).total_seconds() < self.deduplication_window):
                    return True
            
            return False
    
    def record_signal(self, symbol: str, signal_type: str, timestamp: Optional[datetime] = None):
        """Record a signal to prevent duplicates."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.recent_signals.append({
                'symbol': symbol,
                'signal_type': signal_type,
                'timestamp': timestamp
            })


def validate_market_data(market_data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate market data DataFrame.
    
    Args:
        market_data: Market data DataFrame to validate
        required_columns: Required columns (default: ['close', 'volume'])
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required_columns is None:
        required_columns = ['close', 'volume']
    
    if market_data is None:
        return False, "Market data is None"
    
    if not isinstance(market_data, pd.DataFrame):
        return False, f"Market data must be DataFrame, got {type(market_data)}"
    
    if market_data.empty:
        return False, "Market data is empty"
    
    missing_columns = [col for col in required_columns if col not in market_data.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    # Check for NaN values in critical columns
    for col in required_columns:
        if market_data[col].isna().all():
            return False, f"Column {col} contains only NaN values"
    
    # Check minimum data length
    if len(market_data) < 20:
        return False, f"Insufficient data points: {len(market_data)} < 20"
    
    return True, None


def validate_regime_probabilities(regime_probabilities: Dict[Any, float]) -> Tuple[bool, Optional[str]]:
    """
    Validate regime probabilities.
    
    Args:
        regime_probabilities: Dictionary of regime probabilities
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not regime_probabilities:
        return False, "Regime probabilities is empty"
    
    # Check that all values are non-negative
    for regime, prob in regime_probabilities.items():
        if prob < 0 or prob > 1:
            return False, f"Invalid probability for {regime}: {prob} (must be in [0, 1])"
    
    # Check that probabilities sum to approximately 1.0 (allow 10% tolerance)
    total_prob = sum(regime_probabilities.values())
    if abs(total_prob - 1.0) > 0.1:
        return False, f"Probabilities sum to {total_prob:.3f}, expected ~1.0"
    
    return True, None


def validate_signal_parameters(
    symbol: Optional[str] = None,
    account_balance: Optional[float] = None,
    confidence_score: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate signal generation parameters.
    
    Args:
        symbol: Trading symbol
        account_balance: Account balance
        confidence_score: Confidence score
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if symbol is not None:
        if not isinstance(symbol, str) or not symbol.strip():
            return False, f"Invalid symbol: {symbol}"
    
    if account_balance is not None:
        if not isinstance(account_balance, (int, float)) or account_balance <= 0:
            return False, f"Invalid account balance: {account_balance} (must be positive)"
    
    if confidence_score is not None:
        if not isinstance(confidence_score, (int, float)):
            return False, f"Invalid confidence score type: {type(confidence_score)}"
        if confidence_score < 0 or confidence_score > 1:
            return False, f"Invalid confidence score: {confidence_score} (must be in [0, 1])"
    
    return True, None


def calculate_weighted_regime_multiplier(
    regime_probabilities: Dict[Any, float],
    regime_multipliers: Dict[Any, float]
) -> float:
    """
    Calculate weighted regime multiplier using weighted average.
    
    This fixes the additive accumulation bug - uses proper weighted average instead.
    
    Args:
        regime_probabilities: Dictionary of regime probabilities
        regime_multipliers: Dictionary of regime multipliers
    
    Returns:
        Weighted regime multiplier
    """
    if not regime_probabilities:
        return 1.0
    
    weighted_sum = 0.0
    total_prob = 0.0
    
    for regime, probability in regime_probabilities.items():
        multiplier = regime_multipliers.get(regime, 1.0)
        weighted_sum += multiplier * probability
        total_prob += probability
    
    if total_prob == 0:
        return 1.0
    
    # Normalize by total probability (should be ~1.0)
    return weighted_sum / total_prob if total_prob > 0 else 1.0
