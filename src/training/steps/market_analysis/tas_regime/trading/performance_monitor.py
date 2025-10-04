"""Performance monitoring utilities for the TAS trading engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import math

logger = logging.getLogger(__name__)

try:
    # Avoid optional dependency failures when pandas is unavailable in tests.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback when pandas missing
    pd = None  # type: ignore


@dataclass
class PerformanceConfig:
    """Configuration for :class:`TradingPerformanceMonitor`."""

    initial_equity: float = 100000.0
    track_sharpe: bool = True
    min_trades_for_sharpe: int = 2
    rolling_window: int = 100

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")


class TradingPerformanceMonitor:
    """Track realised trading performance for TAS."""

    def __init__(self, config: PerformanceConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    def update_performance(self, trade_result: object) -> None:
        """Update monitoring state with a new trade result."""

        pnl = float(getattr(trade_result, "pnl", 0.0))
        self.total_pnl += pnl
        self.trade_history.append(pnl)
        self.total_trades += 1

        if pnl >= 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if self.current_equity is None:
            initial_capital = self.config.initial_equity
            metadata = getattr(trade_result, "metadata", {}) or {}
            trade_config = metadata.get("config") or {}
            initial_capital = float(trade_config.get("initial_capital", initial_capital))
            self.current_equity = initial_capital
            self.peak_equity = initial_capital

        self.current_equity += pnl
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        self.equity_curve.append(self.current_equity)
        self.logger.debug(
            "Performance updated – trades: %d, pnl: %.2f, equity: %.2f",
            self.total_trades,
            self.total_pnl,
            self.current_equity,
        )

    def get_metrics(self) -> Dict[str, float]:
        """Return a snapshot of current performance metrics."""

        win_rate = (
            self.winning_trades / self.total_trades if self.total_trades else 0.0
        )
        sharpe_ratio = 0.0
        if self.config.track_sharpe and len(self.trade_history) >= self.config.min_trades_for_sharpe:
            returns = self.trade_history[-self.config.rolling_window :]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / max(len(returns) - 1, 1)
            std_dev = math.sqrt(variance)
            if std_dev > 0:
                sharpe_ratio = mean_return / std_dev

        return {
            "total_trades": float(self.total_trades),
            "total_pnl": float(self.total_pnl),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(self.max_drawdown),
            "equity_curve_length": float(len(self.equity_curve)),
        }

    def get_equity_curve(self) -> Optional[pd.Series]:  # type: ignore[name-defined]
        """Return the equity curve as a pandas Series when pandas is available."""

        if pd is None:
            return None
        if not self.equity_curve:
            return pd.Series(dtype=float)
        index = range(len(self.equity_curve))
        return pd.Series(self.equity_curve, index=index, dtype=float)

    def reset(self) -> None:
        """Reset all tracked statistics."""

        self.trade_history: List[float] = []
        self.total_trades: int = 0
        self.total_pnl: float = 0.0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.max_drawdown: float = 0.0
        self.equity_curve: List[float] = []
        self.current_equity: Optional[float] = None
        self.peak_equity: float = self.config.initial_equity
        self.logger.debug("Performance monitor reset")
