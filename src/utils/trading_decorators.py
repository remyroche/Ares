"""
Comprehensive Trading and Backtesting Decorators

This module provides a suite of decorators for enhancing trading and backtesting
pipelines with error handling, trade tracking, monitoring, performance analysis,
and operational management capabilities.
"""

import asyncio
import functools
import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import traceback

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error, warning, critical, problem, failed, invalid, missing,
    timeout, connection_error, validation_error, initialization_error, execution_error
)

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class TradeSide(Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ExecutionMode(Enum):
    """Execution mode enumeration."""
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"
    SIMULATION = "simulation"


@dataclass
class TradeContext:
    """Context information for trade execution."""
    symbol: str
    side: TradeSide
    quantity: float
    price: float
    timestamp: datetime
    execution_mode: ExecutionMode
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    regime_analysis: Dict[str, Any] = field(default_factory=dict)
    support_resistance_levels: Dict[str, float] = field(default_factory=dict)
    hmm_regime: str = ""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    trade_id: Optional[str] = None
    pnl: Optional[float] = None
    drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


class TradeTracker:
    """Centralized trade tracking system."""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.logger = system_logger.getChild("TradeTracker")
        
    def log_trade(self, trade_context: TradeContext, result: Any, metrics: PerformanceMetrics):
        """Log a trade with all context and metrics."""
        trade_record = {
            "trade_id": metrics.trade_id,
            "timestamp": trade_context.timestamp.isoformat(),
            "symbol": trade_context.symbol,
            "side": trade_context.side.value,
            "quantity": trade_context.quantity,
            "price": trade_context.price,
            "execution_mode": trade_context.execution_mode.value,
            "model_weights": trade_context.model_weights,
            "model_confidences": trade_context.model_confidences,
            "regime_analysis": trade_context.regime_analysis,
            "support_resistance_levels": trade_context.support_resistance_levels,
            "hmm_regime": trade_context.hmm_regime,
            "market_conditions": trade_context.market_conditions,
            "risk_metrics": trade_context.risk_metrics,
            "execution_metadata": trade_context.execution_metadata,
            "result": result,
            "performance_metrics": asdict(metrics),
            "success": metrics.success
        }
        
        self.trades.append(trade_record)
        self.performance_history.append(metrics)
        
        # Log to comprehensive logger if available
        try:
            from src.utils.comprehensive_logger import get_comprehensive_logger
            cl = get_comprehensive_logger()
            if cl:
                cl.log_trade(trade_record)
        except Exception:
            pass
        
        self.logger.info(f"Trade logged: {trade_context.symbol} {trade_context.side.value} "
                        f"@ {trade_context.price:.4f} - Success: {metrics.success}")


# Global trade tracker instance
_trade_tracker = TradeTracker()


def get_trade_tracker() -> TradeTracker:
    """Get the global trade tracker instance."""
    return _trade_tracker


# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def trading_error_handler(
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    circuit_breaker_threshold: int = 5,
    fallback_strategy: Optional[Callable] = None
):
    """
    Enhanced error handling decorator for trading operations.
    
    Args:
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
        circuit_breaker_threshold: Number of failures before circuit breaker opens
        fallback_strategy: Fallback function to call if all retries fail
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            failures = 0
            last_exception = None
            
            for attempt in range(retry_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    failures += 1
                    last_exception = e
                    
                    if failures >= circuit_breaker_threshold:
                        system_logger.error(f"Circuit breaker opened for {func.__name__}: {e}")
                        break
                    
                    if attempt < retry_attempts:
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        system_logger.warning(f"Retry {attempt + 1}/{retry_attempts} for {func.__name__}: {e}")
            
            # Try fallback strategy
            if fallback_strategy:
                try:
                    return await fallback_strategy(*args, **kwargs)
                except Exception as fallback_e:
                    system_logger.error(f"Fallback strategy also failed: {fallback_e}")
            
            raise last_exception or Exception(f"All attempts failed for {func.__name__}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            failures = 0
            last_exception = None
            
            for attempt in range(retry_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    failures += 1
                    last_exception = e
                    
                    if failures >= circuit_breaker_threshold:
                        system_logger.error(f"Circuit breaker opened for {func.__name__}: {e}")
                        break
                    
                    if attempt < retry_attempts:
                        time.sleep(retry_delay * (2 ** attempt))
                        system_logger.warning(f"Retry {attempt + 1}/{retry_attempts} for {func.__name__}: {e}")
            
            if fallback_strategy:
                try:
                    return fallback_strategy(*args, **kwargs)
                except Exception as fallback_e:
                    system_logger.error(f"Fallback strategy also failed: {fallback_e}")
            
            raise last_exception or Exception(f"All attempts failed for {func.__name__}")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def market_data_error_handler(
    data_validation: bool = True,
    fallback_to_cached: bool = True,
    max_age_seconds: int = 300
):
    """
    Error handling decorator specifically for market data operations.
    
    Args:
        data_validation: Whether to validate data quality
        fallback_to_cached: Whether to fallback to cached data
        max_age_seconds: Maximum age of cached data to use
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                if data_validation and result is not None:
                    # Validate data quality
                    if isinstance(result, pd.DataFrame) and result.empty:
                        raise ValueError("Empty market data received")
                    
                    # Check for stale data
                    if hasattr(result, 'index') and len(result.index) > 0:
                        latest_timestamp = result.index[-1]
                        if isinstance(latest_timestamp, pd.Timestamp):
                            age_seconds = (datetime.now() - latest_timestamp).total_seconds()
                            if age_seconds > max_age_seconds:
                                system_logger.warning(f"Market data is {age_seconds:.0f}s old")
                
                return result
                
            except Exception as e:
                system_logger.error(f"Market data error in {func.__name__}: {e}")
                
                if fallback_to_cached:
                    # Try to get cached data
                    try:
                        from src.database.efficient_features_database import EfficientFeaturesDatabase
                        db = EfficientFeaturesDatabase()
                        cached_data = await db.get_latest_features(kwargs.get('symbol', ''))
                        if cached_data is not None:
                            system_logger.info(f"Using cached data for {func.__name__}")
                            return cached_data
                    except Exception as cache_e:
                        system_logger.error(f"Cache fallback failed: {cache_e}")
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                if data_validation and result is not None:
                    if isinstance(result, pd.DataFrame) and result.empty:
                        raise ValueError("Empty market data received")
                
                return result
                
            except Exception as e:
                system_logger.error(f"Market data error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# ============================================================================
# TRADE TRACKING DECORATORS
# ============================================================================

def track_trade(
    capture_model_data: bool = True,
    capture_regime_data: bool = True,
    capture_market_conditions: bool = True,
    capture_risk_metrics: bool = True
):
    """
    Decorator to track comprehensive trade data including model weights,
    confidences, regime analysis, and market conditions.
    
    Args:
        capture_model_data: Whether to capture model weights and confidences
        capture_regime_data: Whether to capture regime analysis
        capture_market_conditions: Whether to capture market conditions
        capture_risk_metrics: Whether to capture risk metrics
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_message = None
            result = None
            
            # Extract trade context from function arguments
            trade_context = _extract_trade_context(args, kwargs)
            
            try:
                result = await func(*args, **kwargs)
                success = True
                
                # Capture additional data if available
                if capture_model_data:
                    trade_context.model_weights = _extract_model_weights(result, args, kwargs)
                    trade_context.model_confidences = _extract_model_confidences(result, args, kwargs)
                
                if capture_regime_data:
                    trade_context.regime_analysis = _extract_regime_analysis(result, args, kwargs)
                    trade_context.hmm_regime = _extract_hmm_regime(result, args, kwargs)
                
                if capture_market_conditions:
                    trade_context.market_conditions = _extract_market_conditions(result, args, kwargs)
                    trade_context.support_resistance_levels = _extract_support_resistance(result, args, kwargs)
                
                if capture_risk_metrics:
                    trade_context.risk_metrics = _extract_risk_metrics(result, args, kwargs)
                
            except Exception as e:
                error_message = str(e)
                system_logger.error(f"Trade execution failed in {func.__name__}: {e}")
                raise
            finally:
                # Create performance metrics
                execution_time = time.time() - start_time
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    success=success,
                    error_message=error_message,
                    trade_id=getattr(result, 'trade_id', None) if result else None
                )
                
                # Log the trade
                _trade_tracker.log_trade(trade_context, result, metrics)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            error_message = None
            result = None
            
            trade_context = _extract_trade_context(args, kwargs)
            
            try:
                result = func(*args, **kwargs)
                success = True
                
                if capture_model_data:
                    trade_context.model_weights = _extract_model_weights(result, args, kwargs)
                    trade_context.model_confidences = _extract_model_confidences(result, args, kwargs)
                
                if capture_regime_data:
                    trade_context.regime_analysis = _extract_regime_analysis(result, args, kwargs)
                    trade_context.hmm_regime = _extract_hmm_regime(result, args, kwargs)
                
                if capture_market_conditions:
                    trade_context.market_conditions = _extract_market_conditions(result, args, kwargs)
                    trade_context.support_resistance_levels = _extract_support_resistance(result, args, kwargs)
                
                if capture_risk_metrics:
                    trade_context.risk_metrics = _extract_risk_metrics(result, args, kwargs)
                
            except Exception as e:
                error_message = str(e)
                system_logger.error(f"Trade execution failed in {func.__name__}: {e}")
                raise
            finally:
                execution_time = time.time() - start_time
                metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    success=success,
                    error_message=error_message,
                    trade_id=getattr(result, 'trade_id', None) if result else None
                )
                
                _trade_tracker.log_trade(trade_context, result, metrics)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def track_model_performance(
    model_name: Optional[str] = None,
    capture_predictions: bool = True,
    capture_feature_importance: bool = True,
    capture_confidence: bool = True
):
    """
    Decorator to track model performance metrics.
    
    Args:
        model_name: Name of the model to track
        capture_predictions: Whether to capture predictions
        capture_feature_importance: Whether to capture feature importance
        capture_confidence: Whether to capture confidence scores
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            
            try:
                result = await func(*args, **kwargs)
                
                # Track model performance
                model_name_actual = model_name or func.__name__
                performance_data = {
                    "model_name": model_name_actual,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": time.time() - start_time,
                    "success": True
                }
                
                if capture_predictions and result is not None:
                    performance_data["prediction"] = _extract_prediction(result)
                
                if capture_feature_importance and result is not None:
                    performance_data["feature_importance"] = _extract_feature_importance(result)
                
                if capture_confidence and result is not None:
                    performance_data["confidence"] = _extract_confidence(result)
                
                # Log to monitoring system
                _log_model_performance(performance_data)
                
            except Exception as e:
                # Log failure
                performance_data = {
                    "model_name": model_name or func.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
                _log_model_performance(performance_data)
                raise
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = None
            
            try:
                result = func(*args, **kwargs)
                
                model_name_actual = model_name or func.__name__
                performance_data = {
                    "model_name": model_name_actual,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": time.time() - start_time,
                    "success": True
                }
                
                if capture_predictions and result is not None:
                    performance_data["prediction"] = _extract_prediction(result)
                
                if capture_feature_importance and result is not None:
                    performance_data["feature_importance"] = _extract_feature_importance(result)
                
                if capture_confidence and result is not None:
                    performance_data["confidence"] = _extract_confidence(result)
                
                _log_model_performance(performance_data)
                
            except Exception as e:
                performance_data = {
                    "model_name": model_name or func.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
                _log_model_performance(performance_data)
                raise
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# ============================================================================
# PERFORMANCE MONITORING DECORATORS
# ============================================================================

def monitor_performance(
    alert_threshold_ms: float = 1000.0,
    log_slow_operations: bool = True,
    capture_memory_usage: bool = False
):
    """
    Decorator to monitor function performance and alert on slow operations.
    
    Args:
        alert_threshold_ms: Threshold in milliseconds to trigger alerts
        log_slow_operations: Whether to log slow operations
        capture_memory_usage: Whether to capture memory usage
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = _get_memory_usage() if capture_memory_usage else None
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                if execution_time > alert_threshold_ms:
                    if log_slow_operations:
                        system_logger.warning(
                            f"Slow operation detected: {func.__name__} took {execution_time:.2f}ms"
                        )
                    
                    # Alert monitoring system
                    _alert_slow_operation(func.__name__, execution_time, alert_threshold_ms)
                
                if capture_memory_usage:
                    end_memory = _get_memory_usage()
                    memory_delta = end_memory - start_memory if start_memory else 0
                    _log_memory_usage(func.__name__, memory_delta)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = _get_memory_usage() if capture_memory_usage else None
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.time() - start_time) * 1000
                
                if execution_time > alert_threshold_ms:
                    if log_slow_operations:
                        system_logger.warning(
                            f"Slow operation detected: {func.__name__} took {execution_time:.2f}ms"
                        )
                    
                    _alert_slow_operation(func.__name__, execution_time, alert_threshold_ms)
                
                if capture_memory_usage:
                    end_memory = _get_memory_usage()
                    memory_delta = end_memory - start_memory if start_memory else 0
                    _log_memory_usage(func.__name__, memory_delta)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def validate_trade_parameters(
    validate_price: bool = True,
    validate_quantity: bool = True,
    validate_symbol: bool = True,
    min_price: float = 0.0,
    min_quantity: float = 0.0
):
    """
    Decorator to validate trade parameters before execution.
    
    Args:
        validate_price: Whether to validate price
        validate_quantity: Whether to validate quantity
        validate_symbol: Whether to validate symbol
        min_price: Minimum valid price
        min_quantity: Minimum valid quantity
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Validate parameters
            if validate_price and 'price' in kwargs:
                if kwargs['price'] <= min_price:
                    raise ValueError(f"Invalid price: {kwargs['price']} <= {min_price}")
            
            if validate_quantity and 'quantity' in kwargs:
                if kwargs['quantity'] <= min_quantity:
                    raise ValueError(f"Invalid quantity: {kwargs['quantity']} <= {min_quantity}")
            
            if validate_symbol and 'symbol' in kwargs:
                if not kwargs['symbol'] or not isinstance(kwargs['symbol'], str):
                    raise ValueError(f"Invalid symbol: {kwargs['symbol']}")
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if validate_price and 'price' in kwargs:
                if kwargs['price'] <= min_price:
                    raise ValueError(f"Invalid price: {kwargs['price']} <= {min_price}")
            
            if validate_quantity and 'quantity' in kwargs:
                if kwargs['quantity'] <= min_quantity:
                    raise ValueError(f"Invalid quantity: {kwargs['quantity']} <= {min_quantity}")
            
            if validate_symbol and 'symbol' in kwargs:
                if not kwargs['symbol'] or not isinstance(kwargs['symbol'], str):
                    raise ValueError(f"Invalid symbol: {kwargs['symbol']}")
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# ============================================================================
# OPERATIONAL DECORATORS
# ============================================================================

def rate_limit(
    max_calls: int = 100,
    time_window: float = 60.0
):
    """
    Rate limiting decorator for API calls and trading operations.
    
    Args:
        max_calls: Maximum number of calls allowed
        time_window: Time window in seconds
    """
    call_history = []
    
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Clean old calls
            call_history[:] = [call_time for call_time in call_history 
                             if current_time - call_time < time_window]
            
            if len(call_history) >= max_calls:
                wait_time = time_window - (current_time - call_history[0])
                if wait_time > 0:
                    system_logger.warning(f"Rate limit reached for {func.__name__}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            call_history.append(current_time)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_time = time.time()
            
            call_history[:] = [call_time for call_time in call_history 
                             if current_time - call_time < time_window]
            
            if len(call_history) >= max_calls:
                wait_time = time_window - (current_time - call_history[0])
                if wait_time > 0:
                    system_logger.warning(f"Rate limit reached for {func.__name__}, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
            
            call_history.append(current_time)
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    monitor_interval: float = 10.0
):
    """
    Circuit breaker decorator for trading operations.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        monitor_interval: Interval to check circuit state
    """
    class CircuitState:
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
    
    circuit_state = CircuitState.CLOSED
    failure_count = 0
    last_failure_time = 0
    
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal circuit_state, failure_count, last_failure_time
            
            current_time = time.time()
            
            # Check if circuit should transition
            if circuit_state == CircuitState.OPEN:
                if current_time - last_failure_time > recovery_timeout:
                    circuit_state = CircuitState.HALF_OPEN
                    system_logger.info(f"Circuit breaker for {func.__name__} transitioning to half-open")
            
            if circuit_state == CircuitState.OPEN:
                raise Exception(f"Circuit breaker is open for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Success - close circuit if it was half-open
                if circuit_state == CircuitState.HALF_OPEN:
                    circuit_state = CircuitState.CLOSED
                    failure_count = 0
                    system_logger.info(f"Circuit breaker for {func.__name__} closed")
                
                return result
                
            except Exception as e:
                failure_count += 1
                last_failure_time = current_time
                
                if failure_count >= failure_threshold:
                    circuit_state = CircuitState.OPEN
                    system_logger.error(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal circuit_state, failure_count, last_failure_time
            
            current_time = time.time()
            
            if circuit_state == CircuitState.OPEN:
                if current_time - last_failure_time > recovery_timeout:
                    circuit_state = CircuitState.HALF_OPEN
                    system_logger.info(f"Circuit breaker for {func.__name__} transitioning to half-open")
            
            if circuit_state == CircuitState.OPEN:
                raise Exception(f"Circuit breaker is open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                if circuit_state == CircuitState.HALF_OPEN:
                    circuit_state = CircuitState.CLOSED
                    failure_count = 0
                    system_logger.info(f"Circuit breaker for {func.__name__} closed")
                
                return result
                
            except Exception as e:
                failure_count += 1
                last_failure_time = current_time
                
                if failure_count >= failure_threshold:
                    circuit_state = CircuitState.OPEN
                    system_logger.error(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """
    Retry decorator with exponential backoff and jitter.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        backoff_factor: Factor to multiply delay by each retry
        jitter: Whether to add random jitter to delays
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    if jitter:
                        delay *= 0.5 + np.random.random() * 0.5
                    
                    system_logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception or Exception(f"All retries failed for {func.__name__}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    if jitter:
                        delay *= 0.5 + np.random.random() * 0.5
                    
                    system_logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            raise last_exception or Exception(f"All retries failed for {func.__name__}")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _extract_trade_context(args: tuple, kwargs: dict) -> TradeContext:
    """Extract trade context from function arguments."""
    # Try to extract from kwargs first
    symbol = kwargs.get('symbol', 'UNKNOWN')
    side = TradeSide(kwargs.get('side', 'hold').lower())
    quantity = kwargs.get('quantity', 0.0)
    price = kwargs.get('price', 0.0)
    execution_mode = ExecutionMode(kwargs.get('execution_mode', 'backtest').lower())
    
    # Try to extract from args if not in kwargs
    if len(args) > 0 and isinstance(args[0], dict):
        arg_dict = args[0]
        symbol = arg_dict.get('symbol', symbol)
        side = TradeSide(arg_dict.get('side', side.value).lower())
        quantity = arg_dict.get('quantity', quantity)
        price = arg_dict.get('price', price)
        execution_mode = ExecutionMode(arg_dict.get('execution_mode', execution_mode.value).lower())
    
    return TradeContext(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=datetime.now(),
        execution_mode=execution_mode
    )


def _extract_model_weights(result: Any, args: tuple, kwargs: dict) -> Dict[str, float]:
    """Extract model weights from result or arguments."""
    if hasattr(result, 'model_weights'):
        return result.model_weights
    if hasattr(result, 'weights'):
        return result.weights
    return kwargs.get('model_weights', {})


def _extract_model_confidences(result: Any, args: tuple, kwargs: dict) -> Dict[str, float]:
    """Extract model confidences from result or arguments."""
    if hasattr(result, 'model_confidences'):
        return result.model_confidences
    if hasattr(result, 'confidences'):
        return result.confidences
    return kwargs.get('model_confidences', {})


def _extract_regime_analysis(result: Any, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract regime analysis from result or arguments."""
    if hasattr(result, 'regime_analysis'):
        return result.regime_analysis
    return kwargs.get('regime_analysis', {})


def _extract_hmm_regime(result: Any, args: tuple, kwargs: dict) -> str:
    """Extract HMM regime from result or arguments."""
    if hasattr(result, 'hmm_regime'):
        return result.hmm_regime
    return kwargs.get('hmm_regime', '')


def _extract_market_conditions(result: Any, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract market conditions from result or arguments."""
    if hasattr(result, 'market_conditions'):
        return result.market_conditions
    return kwargs.get('market_conditions', {})


def _extract_support_resistance(result: Any, args: tuple, kwargs: dict) -> Dict[str, float]:
    """Extract support/resistance levels from result or arguments."""
    if hasattr(result, 'support_resistance_levels'):
        return result.support_resistance_levels
    return kwargs.get('support_resistance_levels', {})


def _extract_risk_metrics(result: Any, args: tuple, kwargs: dict) -> Dict[str, float]:
    """Extract risk metrics from result or arguments."""
    if hasattr(result, 'risk_metrics'):
        return result.risk_metrics
    return kwargs.get('risk_metrics', {})


def _extract_prediction(result: Any) -> Any:
    """Extract prediction from result."""
    if hasattr(result, 'prediction'):
        return result.prediction
    if hasattr(result, 'pred'):
        return result.pred
    return result


def _extract_feature_importance(result: Any) -> Any:
    """Extract feature importance from result."""
    if hasattr(result, 'feature_importance'):
        return result.feature_importance
    if hasattr(result, 'feature_importances'):
        return result.feature_importances
    return None


def _extract_confidence(result: Any) -> Any:
    """Extract confidence from result."""
    if hasattr(result, 'confidence'):
        return result.confidence
    if hasattr(result, 'conf'):
        return result.conf
    return None


def _log_model_performance(performance_data: Dict[str, Any]) -> None:
    """Log model performance data."""
    try:
        from src.monitoring.enhanced_ml_tracker import EnhancedMLTracker
        tracker = EnhancedMLTracker()
        tracker.log_performance(performance_data)
    except Exception:
        system_logger.info(f"Model performance: {performance_data}")


def _alert_slow_operation(func_name: str, execution_time: float, threshold: float) -> None:
    """Alert monitoring system about slow operations."""
    try:
        from src.monitoring.performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.alert_slow_operation(func_name, execution_time, threshold)
    except Exception:
        system_logger.warning(f"Slow operation alert: {func_name} took {execution_time:.2f}ms")


def _log_memory_usage(func_name: str, memory_delta: float) -> None:
    """Log memory usage for function."""
    system_logger.debug(f"Memory usage for {func_name}: {memory_delta:.2f} MB")


def _get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return None


# ============================================================================
# COMPOSITE DECORATORS
# ============================================================================

def comprehensive_trade_decorator(
    enable_error_handling: bool = True,
    enable_tracking: bool = True,
    enable_performance_monitoring: bool = True,
    enable_validation: bool = True,
    enable_rate_limiting: bool = True,
    enable_circuit_breaker: bool = True,
    **kwargs
):
    """
    Comprehensive decorator that combines multiple trading decorators.
    
    This is a convenience decorator that applies multiple decorators
    commonly used together for trading operations.
    """
    def decorator(func: F) -> F:
        # Apply decorators in order
        if enable_validation:
            func = validate_trade_parameters(**kwargs)(func)
        
        if enable_rate_limiting:
            func = rate_limit(**kwargs)(func)
        
        if enable_circuit_breaker:
            func = circuit_breaker(**kwargs)(func)
        
        if enable_error_handling:
            func = trading_error_handler(**kwargs)(func)
        
        if enable_tracking:
            func = track_trade(**kwargs)(func)
        
        if enable_performance_monitoring:
            func = monitor_performance(**kwargs)(func)
        
        return func
    
    return decorator


def comprehensive_model_decorator(
    enable_error_handling: bool = True,
    enable_tracking: bool = True,
    enable_performance_monitoring: bool = True,
    enable_retry: bool = True,
    **kwargs
):
    """
    Comprehensive decorator for model operations.
    """
    def decorator(func: F) -> F:
        if enable_performance_monitoring:
            func = monitor_performance(**kwargs)(func)
        
        if enable_tracking:
            func = track_model_performance(**kwargs)(func)
        
        if enable_retry:
            func = retry_with_backoff(**kwargs)(func)
        
        if enable_error_handling:
            func = trading_error_handler(**kwargs)(func)
        
        return func
    
    return decorator