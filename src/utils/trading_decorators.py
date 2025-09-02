"""
Comprehensive Trading and Backtesting Decorators

This module provides a suite of decorators for enhancing trading and backtesting
pipelines with error handling, trade tracking, monitoring, performance analysis,
and operational management capabilities.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, Optional, Dict, List

# Try to import psutil, fallback to basic monitoring if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from src.utils.logger import system_logger

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class TradeSide(Enum):
    """Trading side enumeration."""
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
        """Initialize trade tracker."""
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
            "result": result,
            "metrics": metrics,
        }
        self.trades.append(trade_record)
        self.performance_history.append(metrics)
        self.logger.info(f"Trade logged: {trade_record}")


# Global trade tracker instance
trade_tracker = TradeTracker()


def error_handler(exceptions: tuple = (Exception,), default_return: Any = None):
    """Decorator for comprehensive error handling in trading operations.

    Args:
        exceptions: Tuple of exceptions to catch
        default_return: Default return value on error
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                system_logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


def performance_monitor(func: F) -> F:
    """Decorator for performance monitoring."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss if PSUTIL_AVAILABLE else 0
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss if PSUTIL_AVAILABLE else 0
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        system_logger.info(
            f"Performance: {func.__name__} - "
            f"Time: {execution_time:.3f}s, "
            f"Memory: {memory_delta / 1024 / 1024:.2f}MB, "
            f"Success: {success}"
        )
        
        return result
    return wrapper


def trade_logger(func: F) -> F:
    """Decorator for trade execution logging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        system_logger.info(f"Starting trade execution: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            system_logger.info(f"Trade execution completed: {func.__name__}")
            return result
        except Exception as e:
            system_logger.error(f"Trade execution failed: {func.__name__} - {e}")
            raise
    
    return wrapper


def risk_manager(max_drawdown: float = 0.05, max_position_size: float = 0.1):
    """Decorator for risk management in trading operations.

    Args:
        max_drawdown: Maximum allowed drawdown
        max_position_size: Maximum position size as fraction of portfolio

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check current drawdown
            current_drawdown = get_current_drawdown()
            if current_drawdown > max_drawdown:
                system_logger.warning(
                    f"Risk limit exceeded: drawdown {current_drawdown:.2%} > {max_drawdown:.2%}"
                )
                return None
            
            # Check position size
            position_size = get_position_size(*args, **kwargs)
            if position_size > max_position_size:
                system_logger.warning(
                    f"Position size limit exceeded: {position_size:.2%} > {max_position_size:.2%}"
                )
                return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def regime_aware(func: F) -> F:
    """Decorator for regime-aware trading."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        current_regime = get_current_regime()
        system_logger.info(f"Current regime: {current_regime}")
        
        # Add regime context to kwargs
        kwargs["regime"] = current_regime
        
        return func(*args, **kwargs)
    
    return wrapper


def backtest_mode(func: F) -> F:
    """Decorator for backtest mode execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["execution_mode"] = ExecutionMode.BACKTEST
        system_logger.info(f"Running in backtest mode: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


def live_trading_mode(func: F) -> F:
    """Decorator for live trading mode execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["execution_mode"] = ExecutionMode.LIVE
        system_logger.info(f"Running in live trading mode: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


def paper_trading_mode(func: F) -> F:
    """Decorator for paper trading mode execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["execution_mode"] = ExecutionMode.PAPER
        system_logger.info(f"Running in paper trading mode: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


def async_trade_executor(func: F) -> F:
    """Decorator for async trade execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        system_logger.info(f"Starting async trade execution: {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            system_logger.info(f"Async trade execution completed: {func.__name__}")
            return result
        except Exception as e:
            system_logger.error(f"Async trade execution failed: {func.__name__} - {e}")
            raise
        
        return wrapper
    
    return wrapper


def trade_validation(func: F) -> F:
    """Decorator for trade parameter validation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate trade parameters
        if not validate_trade_params(*args, **kwargs):
            system_logger.error(f"Trade validation failed: {func.__name__}")
            return None
        
        system_logger.info(f"Trade validation passed: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


# Helper functions (implementations would depend on your specific trading system)

def get_current_drawdown() -> float:
    """Get current portfolio drawdown."""
    # Implementation would depend on your portfolio tracking system
    return 0.05  # Placeholder


def get_position_size(*args, **kwargs) -> float:
    """Get current position size as fraction of portfolio."""
    # Implementation would depend on your position sizing logic
    return 0.1  # Placeholder


def get_current_regime() -> str:
    """Get current market regime."""
    # Implementation would depend on your regime detection system
    return "trending"  # Placeholder


def validate_trade_params(*args, **kwargs) -> bool:
    """Validate trade parameters."""
    # Implementation would depend on your validation logic
    return True  # Placeholder


def get_trade_tracker():
    """Get a trade tracker instance for monitoring trade execution."""
    # Simple implementation - in a real system this would be more sophisticated
    class SimpleTradeTracker:
        def __init__(self):
            self.trades = []
            self.current_trade = None
        
        def start_trade(self, trade_id: str, symbol: str, side: str, quantity: float, price: float):
            """Start tracking a new trade."""
            self.current_trade = {
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "start_time": time.time(),
                "status": "executing"
            }
        
        def complete_trade(self, trade_id: str, final_price: float, commission: float = 0.0):
            """Complete tracking a trade."""
            if self.current_trade and self.current_trade["trade_id"] == trade_id:
                self.current_trade["final_price"] = final_price
                self.current_trade["commission"] = commission
                self.current_trade["end_time"] = time.time()
                self.current_trade["status"] = "completed"
                self.trades.append(self.current_trade)
                self.current_trade = None
        
        def get_trade_history(self):
            """Get all tracked trades."""
            return self.trades.copy()
        
        def get_current_trade(self):
            """Get the currently executing trade."""
            return self.current_trade
    
    return SimpleTradeTracker()


def comprehensive_trading_decorator(
    enable_error_handling: bool = True,
    enable_performance_monitoring: bool = True,
    enable_trade_logging: bool = True,
    enable_risk_management: bool = True,
    enable_regime_awareness: bool = True,
    max_drawdown: float = 0.05,
    max_position_size: float = 0.1
):
    """Comprehensive decorator that combines multiple trading enhancements.

    Args:
        enable_error_handling: Enable error handling
        enable_performance_monitoring: Enable performance monitoring
        enable_trade_logging: Enable trade logging
        enable_risk_management: Enable risk management
        enable_regime_awareness: Enable regime awareness
        max_drawdown: Maximum allowed drawdown
        max_position_size: Maximum position size

    Returns:
        Comprehensive decorator
    """
    def decorator(func: F) -> F:
        # Apply decorators based on configuration
        if enable_error_handling:
            func = error_handler()(func)
        
        if enable_performance_monitoring:
            func = performance_monitor(func)
        
        if enable_trade_logging:
            func = trade_logger(func)
        
        if enable_risk_management:
            func = risk_manager(max_drawdown, max_position_size)(func)
        
        if enable_regime_awareness:
            func = regime_aware(func)
        
        return func
    
    return decorator


# Example usage decorators for common trading patterns

def momentum_trading(func: F) -> F:
    """Decorator for momentum-based trading strategies."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add momentum context
        kwargs["strategy_type"] = "momentum"
        kwargs["lookback_period"] = kwargs.get("lookback_period", 20)
        
        system_logger.info(f"Executing momentum strategy: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


def mean_reversion_trading(func: F) -> F:
    """Decorator for mean reversion trading strategies."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add mean reversion context
        kwargs["strategy_type"] = "mean_reversion"
        kwargs["reversion_threshold"] = kwargs.get("reversion_threshold", 2.0)
        
        system_logger.info(f"Executing mean reversion strategy: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


def volatility_breakout_trading(func: F) -> F:
    """Decorator for volatility breakout trading strategies."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add volatility breakout context
        kwargs["strategy_type"] = "volatility_breakout"
        kwargs["volatility_window"] = kwargs.get("volatility_window", 30)
        
        system_logger.info(f"Executing volatility breakout strategy: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper


def sector_rotation_trading(func: F) -> F:
    """Decorator for sector rotation trading strategies."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Add sector rotation context
        kwargs["strategy_type"] = "sector_rotation"
        kwargs["rotation_period"] = kwargs.get("rotation_period", 90)
        
        system_logger.info(f"Executing sector rotation strategy: {func.__name__}")
        return func(*args, **kwargs)
    
    return wrapper
