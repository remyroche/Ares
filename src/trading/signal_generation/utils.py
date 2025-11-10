"""
Signal Generation Utilities

Common utilities for signal generation including validation, rate limiting,
circuit breaking, and signal deduplication.
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar
from threading import Lock
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.printing import tprint

logger = system_logger.getChild('SignalGenerationUtils')

T = TypeVar('T')


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
        tprint(f"[CIRCUIT_BREAKER] __init__: failure_threshold={failure_threshold}, recovery_timeout={recovery_timeout}, success_threshold={success_threshold}")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = Lock()
        tprint(f"[CIRCUIT_BREAKER] __init__ -> initialized (state={self.state.value})")
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        tprint(f"[CIRCUIT_BREAKER] call: func={func.__name__}, state={self.state.value}")
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self.last_failure_time is not None and time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    tprint(f"[CIRCUIT_BREAKER] call: transitioning to HALF_OPEN")
                    logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
                else:
                    tprint(f"[CIRCUIT_BREAKER] call -> ERROR: circuit is OPEN", color="red")
                    raise RuntimeError("Circuit breaker is OPEN - too many failures")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            tprint(f"[CIRCUIT_BREAKER] call -> success (state={self.state.value})")
            return result
        except Exception as e:
            self._on_failure()
            tprint(f"[CIRCUIT_BREAKER] call -> ERROR: {e}", color="red")
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        tprint(f"[CIRCUIT_BREAKER] _on_success: state={self.state.value}, success_count={self.success_count}")
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    tprint(f"[CIRCUIT_BREAKER] _on_success: circuit CLOSED - service recovered")
                    logger.info("✅ Circuit breaker CLOSED - service recovered")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
        tprint(f"[CIRCUIT_BREAKER] _on_success -> state={self.state.value}, failure_count={self.failure_count}")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        tprint(f"[CIRCUIT_BREAKER] _on_failure: state={self.state.value}, failure_count={self.failure_count}")
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                tprint(f"[CIRCUIT_BREAKER] _on_failure: circuit OPEN - recovery failed")
                logger.warning("⚠️ Circuit breaker OPEN - recovery failed")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    tprint(f"[CIRCUIT_BREAKER] _on_failure: circuit OPEN - {self.failure_count} failures")
                    logger.error(f"❌ Circuit breaker OPEN - {self.failure_count} failures")
        tprint(f"[CIRCUIT_BREAKER] _on_failure -> state={self.state.value}, failure_count={self.failure_count}")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        tprint(f"[CIRCUIT_BREAKER] get_state: state={self.state.value}")
        return self.state
    
    def reset(self) -> None:
        """Manually reset circuit breaker."""
        tprint(f"[CIRCUIT_BREAKER] reset: resetting circuit breaker")
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            tprint_info("🔄 Circuit breaker manually reset")
        tprint(f"[CIRCUIT_BREAKER] reset -> state={self.state.value}")


class RateLimiter:
    """Rate limiter for signal generation."""
    
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        tprint(f"[RATE_LIMITER] __init__: max_calls={max_calls}, time_window={time_window}")
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times: deque[float] = deque(maxlen=max_calls)
        self._lock = Lock()
        tprint(f"[RATE_LIMITER] __init__ -> initialized")
    
    def acquire(self) -> bool:
        """
        Try to acquire permission for a call.

        Returns:
            True if call is allowed, False if rate limit exceeded
        """
        tprint(f"[RATE_LIMITER] acquire: current_calls={len(self.call_times)}, max_calls={self.max_calls}")
        with self._lock:
            now = time.time()

            # Remove old calls outside time window
            while self.call_times and self.call_times[0] < now - self.time_window:
                self.call_times.popleft()

            if len(self.call_times) >= self.max_calls:
                tprint_warning(f"⚠️ Rate limit exceeded: {len(self.call_times)}/{self.max_calls} calls in {self.time_window}s window")
                tprint(f"[RATE_LIMITER] acquire -> False (rate limit exceeded)")
                return False

            self.call_times.append(now)
            tprint(f"[RATE_LIMITER] acquire -> True (calls={len(self.call_times)}/{self.max_calls})")
            return True
    
    def wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        tprint(f"[RATE_LIMITER] wait_time: checking wait time")
        with self._lock:
            if len(self.call_times) < self.max_calls:
                tprint(f"[RATE_LIMITER] wait_time -> 0.0 (below limit)")
                return 0.0

            oldest_call = self.call_times[0]
            wait_time = self.time_window - (time.time() - oldest_call)
            result = max(0.0, wait_time)
            tprint(f"[RATE_LIMITER] wait_time -> {result:.2f}s")
            return result


class SignalDeduplicator:
    """Signal deduplication to prevent redundant signals."""
    
    def __init__(self, deduplication_window: float = 300.0):
        """
        Initialize signal deduplicator.

        Args:
            deduplication_window: Time window in seconds for deduplication
        """
        tprint(f"[SIGNAL_DEDUP] __init__: deduplication_window={deduplication_window}")
        self.deduplication_window = deduplication_window
        self.recent_signals: deque[Dict[str, Any]] = deque(maxlen=100)
        self._lock = Lock()
        tprint(f"[SIGNAL_DEDUP] __init__ -> initialized")
    
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
        tprint(f"[SIGNAL_DEDUP] is_duplicate: symbol={symbol}, signal_type={signal_type}")
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
                    tprint(f"[SIGNAL_DEDUP] is_duplicate -> True (found duplicate within {self.deduplication_window}s)")
                    return True

            tprint(f"[SIGNAL_DEDUP] is_duplicate -> False (no duplicate found)")
            return False
    
    def record_signal(self, symbol: str, signal_type: str, timestamp: Optional[datetime] = None):
        """Record a signal to prevent duplicates."""
        tprint(f"[SIGNAL_DEDUP] record_signal: symbol={symbol}, signal_type={signal_type}")
        if timestamp is None:
            timestamp = datetime.now()

        with self._lock:
            self.recent_signals.append({
                'symbol': symbol,
                'signal_type': signal_type,
                'timestamp': timestamp
            })
        tprint(f"[SIGNAL_DEDUP] record_signal -> recorded (total_signals={len(self.recent_signals)})")


def validate_market_data(market_data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate market data DataFrame.

    Args:
        market_data: Market data DataFrame to validate
        required_columns: Required columns (default: ['close', 'volume'])

    Returns:
        Tuple of (is_valid, error_message)
    """
    tprint(f"[UTILS] validate_market_data: validating market data")
    if required_columns is None:
        required_columns = ['close', 'volume']

    if market_data is None:
        tprint(f"[UTILS] validate_market_data -> False (data is None)", color="red")
        return False, "Market data is None"

    if not isinstance(market_data, pd.DataFrame):
        tprint(f"[UTILS] validate_market_data -> False (invalid type: {type(market_data)})", color="red")
        return False, f"Market data must be DataFrame, got {type(market_data)}"

    if market_data.empty:
        tprint(f"[UTILS] validate_market_data -> False (empty dataframe)", color="red")
        return False, "Market data is empty"

    missing_columns = [col for col in required_columns if col not in market_data.columns]
    if missing_columns:
        tprint(f"[UTILS] validate_market_data -> False (missing columns: {missing_columns})", color="red")
        return False, f"Missing required columns: {missing_columns}"

    # Check for NaN values in critical columns
    for col in required_columns:
        if market_data[col].isna().all():
            tprint(f"[UTILS] validate_market_data -> False (column {col} all NaN)", color="red")
            return False, f"Column {col} contains only NaN values"

    # Check minimum data length
    if len(market_data) < 20:
        tprint(f"[UTILS] validate_market_data -> False (insufficient data: {len(market_data)} < 20)", color="red")
        return False, f"Insufficient data points: {len(market_data)} < 20"

    tprint(f"[UTILS] validate_market_data -> True (rows={len(market_data)}, cols={list(market_data.columns)})")
    return True, None


def validate_regime_probabilities(regime_probabilities: Dict[Any, float]) -> Tuple[bool, Optional[str]]:
    """
    Validate regime probabilities.

    Args:
        regime_probabilities: Dictionary of regime probabilities

    Returns:
        Tuple of (is_valid, error_message)
    """
    tprint(f"[UTILS] validate_regime_probabilities: validating {len(regime_probabilities) if regime_probabilities else 0} regime probabilities")
    if not regime_probabilities:
        tprint(f"[UTILS] validate_regime_probabilities -> False (empty dict)", color="red")
        return False, "Regime probabilities is empty"

    # Check that all values are non-negative
    for regime, prob in regime_probabilities.items():
        if prob < 0 or prob > 1:
            tprint(f"[UTILS] validate_regime_probabilities -> False (invalid prob for {regime}: {prob})", color="red")
            return False, f"Invalid probability for {regime}: {prob} (must be in [0, 1])"

    # Check that probabilities sum to approximately 1.0 (allow 10% tolerance)
    total_prob = sum(regime_probabilities.values())
    if abs(total_prob - 1.0) > 0.1:
        tprint(f"[UTILS] validate_regime_probabilities -> False (sum={total_prob:.3f}, expected ~1.0)", color="red")
        return False, f"Probabilities sum to {total_prob:.3f}, expected ~1.0"

    tprint(f"[UTILS] validate_regime_probabilities -> True (regimes={list(regime_probabilities.keys())}, sum={total_prob:.3f})")
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
    tprint(f"[UTILS] validate_signal_parameters: symbol={symbol}, account_balance={account_balance}, confidence_score={confidence_score}")
    if symbol is not None:
        if not isinstance(symbol, str) or not symbol.strip():
            tprint(f"[UTILS] validate_signal_parameters -> False (invalid symbol)", color="red")
            return False, f"Invalid symbol: {symbol}"

    if account_balance is not None:
        if not isinstance(account_balance, (int, float)) or account_balance <= 0:
            tprint(f"[UTILS] validate_signal_parameters -> False (invalid account_balance)", color="red")
            return False, f"Invalid account balance: {account_balance} (must be positive)"

    if confidence_score is not None:
        if not isinstance(confidence_score, (int, float)):
            tprint(f"[UTILS] validate_signal_parameters -> False (invalid confidence_score type)", color="red")
            return False, f"Invalid confidence score type: {type(confidence_score)}"
        if confidence_score < 0 or confidence_score > 1:
            tprint(f"[UTILS] validate_signal_parameters -> False (confidence_score out of range)", color="red")
            return False, f"Invalid confidence score: {confidence_score} (must be in [0, 1])"

    tprint(f"[UTILS] validate_signal_parameters -> True")
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
    tprint(f"[UTILS] calculate_weighted_regime_multiplier: regimes={len(regime_probabilities)}, multipliers={len(regime_multipliers)}")
    if not regime_probabilities:
        tprint(f"[UTILS] calculate_weighted_regime_multiplier -> 1.0 (no probabilities)")
        return 1.0

    weighted_sum = 0.0
    total_prob = 0.0

    for regime, probability in regime_probabilities.items():
        multiplier = regime_multipliers.get(regime, 1.0)
        weighted_sum += multiplier * probability
        total_prob += probability
        tprint(f"[UTILS] calculate_weighted_regime_multiplier: regime={regime}, prob={probability:.3f}, mult={multiplier:.3f}")

    if total_prob == 0:
        tprint(f"[UTILS] calculate_weighted_regime_multiplier -> 1.0 (zero total probability)")
        return 1.0

    # Normalize by total probability (should be ~1.0)
    result = weighted_sum / total_prob if total_prob > 0 else 1.0
    tprint(f"[UTILS] calculate_weighted_regime_multiplier -> {result:.3f} (weighted_sum={weighted_sum:.3f}, total_prob={total_prob:.3f})")
    return result
