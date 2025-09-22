"""
Performance Tracker

Comprehensive performance tracking and metrics calculation for trading operations.
Monitors trade performance, calculates key metrics, and provides detailed analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ..utils.error_handling import (
    TradingError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config

logger = system_logger.getChild('PerformanceTracker')

class MetricType(Enum):
    """Performance metric types."""
    CUMULATIVE_RETURN = "cumulative_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVG_TRADE = "avg_trade"
    AVG_WIN = "avg_win"
    AVG_LOSS = "avg_loss"
    LARGEST_WIN = "largest_win"
    LARGEST_LOSS = "largest_loss"
    TOTAL_TRADES = "total_trades"
    TOTAL_FEES = "total_fees"
    EXPECTANCY = "expectancy"

@dataclass
class TradeRecord:
    """Individual trade record."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: float = 0.0
    fees: float = 0.0
    status: str = "open"  # 'open', 'closed', 'cancelled'
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DailyPerformance:
    """Daily performance summary."""
    date: datetime.date
    starting_balance: float
    ending_balance: float
    daily_pnl: float
    daily_return: float
    trades: int
    winning_trades: int
    losing_trades: int
    volume: float
    fees: float
    max_drawdown: float

@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot."""
    timestamp: datetime
    balance: float
    total_pnl: float
    total_return: float
    metrics: Dict[MetricType, float]
    positions: Dict[str, Any]
    open_trades: List[TradeRecord]

class PerformanceTracker:
    """
    Performance Tracker

    Tracks trading performance in real-time, calculates key metrics,
    and provides comprehensive performance analytics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('PerformanceTracker')

        # Performance state
        self.initial_balance = config.get('initial_balance', 10000.0)
        self.current_balance = self.initial_balance
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Trade records
        self.trades: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        self.open_trades: List[TradeRecord] = []

        # Performance history
        self.balance_history: List[float] = [self.initial_balance]
        self.pnl_history: List[float] = [0.0]
        self.performance_snapshots: List[PerformanceSnapshot] = []

        # Risk metrics
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0

        # Daily performance
        self.daily_performance: Dict[str, DailyPerformance] = {}

        # Benchmark comparison
        self.benchmark_symbol = config.get('benchmark_symbol', 'BTCUSDT')
        self.benchmark_returns = []

        tprint_info("📊 Initializing Performance Tracker...")

    async def initialize(self) -> None:
        """Initialize performance tracker."""
        tprint_success("✅ Performance Tracker initialized successfully")

    @handles_errors
    async def record_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_time: datetime,
        entry_price: float,
        quantity: float,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a new trade.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            side: Trade side ('buy' or 'sell')
            entry_time: Entry timestamp
            entry_price: Entry price
            quantity: Trade quantity
            strategy: Trading strategy used
            metadata: Additional trade metadata
        """
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time=entry_time,
            entry_price=entry_price,
            quantity=quantity,
            strategy=strategy,
            metadata=metadata or {}
        )

        self.trades[trade_id] = trade
        self.open_trades.append(trade)
        self.total_trades += 1

        tprint_info(f"📝 Recorded trade: {side} {quantity} {symbol} @ {entry_price}")

    @handles_errors
    async def update_trade(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        fees: float = 0.0
    ) -> float:
        """
        Update trade with exit information.

        Args:
            trade_id: Trade identifier
            exit_time: Exit timestamp
            exit_price: Exit price
            fees: Trading fees

        Returns:
            Trade PnL
        """
        if trade_id not in self.trades:
            raise TradingError(f"Trade {trade_id} not found")

        trade = self.trades[trade_id]
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.fees = fees

        # Calculate PnL
        if trade.side == 'buy':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity - fees
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity - fees

        # Update trade status
        trade.status = "closed"

        # Update statistics
        self.total_pnl += trade.pnl
        self.total_fees += fees
        self.current_balance += trade.pnl

        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Move from open to closed trades
        self.open_trades = [t for t in self.open_trades if t.trade_id != trade_id]
        self.closed_trades.append(trade)

        # Update performance history
        self.balance_history.append(self.current_balance)
        self.pnl_history.append(self.total_pnl)

        # Update risk metrics
        await self._update_risk_metrics()

        tprint_success(f"✅ Updated trade {trade_id}: PnL = {trade.pnl:.2f}")

        return trade.pnl

    async def _update_risk_metrics(self) -> None:
        """Update risk metrics."""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance

        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

    @handles_errors
    async def get_performance_metrics(self) -> Dict[MetricType, float]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        # Basic metrics
        total_return = self.total_pnl / self.initial_balance
        metrics[MetricType.CUMULATIVE_RETURN] = total_return
        metrics[MetricType.TOTAL_TRADES] = self.total_trades
        metrics[MetricType.TOTAL_FEES] = self.total_fees

        if self.total_trades > 0:
            metrics[MetricType.WIN_RATE] = self.winning_trades / self.total_trades

            if self.winning_trades > 0:
                avg_win = sum(t.pnl for t in self.closed_trades if t.pnl > 0) / self.winning_trades
                metrics[MetricType.AVG_WIN] = avg_win

            if self.losing_trades > 0:
                avg_loss = sum(t.pnl for t in self.closed_trades if t.pnl < 0) / self.losing_trades
                metrics[MetricType.AVG_LOSS] = avg_loss

            metrics[MetricType.AVG_TRADE] = self.total_pnl / self.total_trades

            if self.losing_trades > 0:
                profit_factor = abs(sum(t.pnl for t in self.closed_trades if t.pnl > 0) /
                                  sum(t.pnl for t in self.closed_trades if t.pnl < 0))
                metrics[MetricType.PROFIT_FACTOR] = profit_factor
            else:
                metrics[MetricType.PROFIT_FACTOR] = float('inf') if self.winning_trades > 0 else 0.0

            # Find largest win/loss
            if self.closed_trades:
                winning_trades = [t for t in self.closed_trades if t.pnl > 0]
                losing_trades = [t for t in self.closed_trades if t.pnl < 0]

                if winning_trades:
                    metrics[MetricType.LARGEST_WIN] = max(t.pnl for t in winning_trades)

                if losing_trades:
                    metrics[MetricType.LARGEST_LOSS] = min(t.pnl for t in losing_trades)

        # Calculate Sharpe ratio (simplified)
        if len(self.pnl_history) > 1:
            returns = np.diff(self.pnl_history)
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                metrics[MetricType.SHARPE_RATIO] = sharpe_ratio

        # Annualized return
        if len(self.balance_history) > 1:
            days_elapsed = (datetime.now() - self.closed_trades[0].entry_time).days if self.closed_trades else 1
            if days_elapsed > 0:
                annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
                metrics[MetricType.ANNUALIZED_RETURN] = annualized_return

        metrics[MetricType.MAX_DRAWDOWN] = self.max_drawdown

        # Calculate expectancy
        if self.total_trades > 0:
            win_rate = metrics.get(MetricType.WIN_RATE, 0)
            avg_win = metrics.get(MetricType.AVG_WIN, 0)
            avg_loss = metrics.get(MetricType.AVG_LOSS, 0)
            metrics[MetricType.EXPECTANCY] = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return metrics

    @handles_errors
    async def get_daily_performance(self, days: int = 30) -> List[DailyPerformance]:
        """
        Get daily performance for the last N days.

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily performance records
        """
        daily_performance = []
        end_date = datetime.now().date()

        for i in range(days):
            date = end_date - timedelta(days=i)

            # Find trades for this date
            day_trades = [t for t in self.closed_trades
                         if t.entry_time.date() <= date <= (t.exit_time.date() if t.exit_time else date)]

            if day_trades:
                daily_pnl = sum(t.pnl for t in day_trades)
                winning_trades = len([t for t in day_trades if t.pnl > 0])
                losing_trades = len([t for t in day_trades if t.pnl < 0])

                # Calculate daily return (simplified)
                prev_balance = self.initial_balance + sum(t.pnl for t in self.closed_trades
                                                        if t.entry_time.date() < date)
                daily_return = daily_pnl / prev_balance if prev_balance > 0 else 0

                daily_perf = DailyPerformance(
                    date=date,
                    starting_balance=prev_balance,
                    ending_balance=prev_balance + daily_pnl,
                    daily_pnl=daily_pnl,
                    daily_return=daily_return,
                    trades=len(day_trades),
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    volume=sum(t.quantity * t.entry_price for t in day_trades),
                    fees=sum(t.fees for t in day_trades),
                    max_drawdown=self.max_drawdown
                )

                daily_performance.append(daily_perf)

        return daily_performance

    @handles_errors
    async def get_drawdown_analysis(self) -> Dict[str, Any]:
        """Analyze drawdown periods."""
        if len(self.balance_history) < 2:
            return {"drawdown_periods": [], "max_drawdown_duration": 0}

        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        current_peak = self.balance_history[0]

        for i, balance in enumerate(self.balance_history):
            if balance > current_peak:
                if in_drawdown:
                    # End of drawdown period
                    drawdown_periods.append({
                        "start_index": drawdown_start,
                        "end_index": i - 1,
                        "start_date": self._get_date_for_index(drawdown_start),
                        "end_date": self._get_date_for_index(i - 1),
                        "duration": i - drawdown_start,
                        "depth": (current_peak - min(self.balance_history[drawdown_start:i])) / current_peak
                    })
                    in_drawdown = False

                current_peak = balance
            elif balance < current_peak and not in_drawdown:
                # Start of drawdown period
                in_drawdown = True
                drawdown_start = i

        # Handle ongoing drawdown
        if in_drawdown:
            drawdown_periods.append({
                "start_index": drawdown_start,
                "end_index": len(self.balance_history) - 1,
                "start_date": self._get_date_for_index(drawdown_start),
                "end_date": self._get_date_for_index(len(self.balance_history) - 1),
                "duration": len(self.balance_history) - drawdown_start,
                "depth": (current_peak - min(self.balance_history[drawdown_start:])) / current_peak
            })

        return {
            "drawdown_periods": drawdown_periods,
            "max_drawdown_duration": max([p["duration"] for p in drawdown_periods]) if drawdown_periods else 0
        }

    def _get_date_for_index(self, index: int) -> Optional[datetime]:
        """Get date for balance history index."""
        if 0 <= index < len(self.closed_trades):
            return self.closed_trades[index].entry_time
        return None

    @handles_errors
    async def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by strategy."""
        strategy_metrics = {}

        for trade in self.closed_trades:
            strategy = trade.strategy or "unknown"

            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "trades": 0,
                    "total_pnl": 0.0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_fees": 0.0
                }

            metrics = strategy_metrics[strategy]
            metrics["trades"] += 1
            metrics["total_pnl"] += trade.pnl
            metrics["total_fees"] += trade.fees

            if trade.pnl > 0:
                metrics["winning_trades"] += 1
            else:
                metrics["losing_trades"] += 1

        # Calculate derived metrics
        for strategy, metrics in strategy_metrics.items():
            if metrics["trades"] > 0:
                metrics["win_rate"] = metrics["winning_trades"] / metrics["trades"]
                metrics["avg_trade"] = metrics["total_pnl"] / metrics["trades"]
                metrics["profit_factor"] = (
                    abs(sum(t.pnl for t in self.closed_trades
                           if (t.strategy or "unknown") == strategy and t.pnl > 0)) /
                    abs(sum(t.pnl for t in self.closed_trades
                           if (t.strategy or "unknown") == strategy and t.pnl < 0))
                    if metrics["losing_trades"] > 0 else float('inf')
                )

        return strategy_metrics

    @handles_errors
    async def get_symbol_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by symbol."""
        symbol_metrics = {}

        for trade in self.closed_trades:
            symbol = trade.symbol

            if symbol not in symbol_metrics:
                symbol_metrics[symbol] = {
                    "trades": 0,
                    "total_pnl": 0.0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_fees": 0.0,
                    "volume": 0.0
                }

            metrics = symbol_metrics[symbol]
            metrics["trades"] += 1
            metrics["total_pnl"] += trade.pnl
            metrics["total_fees"] += trade.fees
            metrics["volume"] += trade.quantity * trade.entry_price

            if trade.pnl > 0:
                metrics["winning_trades"] += 1
            else:
                metrics["losing_trades"] += 1

        # Calculate derived metrics
        for symbol, metrics in symbol_metrics.items():
            if metrics["trades"] > 0:
                metrics["win_rate"] = metrics["winning_trades"] / metrics["trades"]
                metrics["avg_trade"] = metrics["total_pnl"] / metrics["trades"]

        return symbol_metrics

    async def export_performance_report(self, format: str = "json") -> str:
        """
        Export comprehensive performance report.

        Args:
            format: Export format ('json', 'csv')

        Returns:
            Report data as string
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "total_pnl": self.total_pnl,
                "total_return": self.total_pnl / self.initial_balance,
                "total_trades": self.total_trades,
                "win_rate": self.winning_trades / max(self.total_trades, 1),
                "total_fees": self.total_fees
            },
            "metrics": await self.get_performance_metrics(),
            "strategy_performance": await self.get_strategy_performance(),
            "symbol_performance": await self.get_symbol_performance(),
            "recent_trades": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "pnl": t.pnl,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None
                }
                for t in self.closed_trades[-10:]  # Last 10 trades
            ]
        }

        if format == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            # CSV format
            return pd.DataFrame(report["recent_trades"]).to_csv(index=False)

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.trades.clear()
        self.closed_trades.clear()
        self.open_trades.clear()
        self.balance_history.clear()
        self.pnl_history.clear()
        self.performance_snapshots.clear()
        self.daily_performance.clear()

        tprint_info("🧹 Performance Tracker cleaned up successfully")

# Factory functions
async def create_performance_tracker(config: Dict[str, Any]) -> PerformanceTracker:
    """Create and initialize a performance tracker."""
    tracker = PerformanceTracker(config)
    await tracker.initialize()
    return tracker

def get_performance_tracker() -> Optional[PerformanceTracker]:
    """Get the global performance tracker instance."""
    return None