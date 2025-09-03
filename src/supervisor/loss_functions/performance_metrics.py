"""
Performance Metrics Calculator Module.

This module handles various performance metric calculations for trading strategies.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.decorators import handles_errors

from .base import PnLLossFunctionsBase


class PerformanceMetricsCalculator(PnLLossFunctionsBase):
    """
    Performance Metrics Calculator for computing strategy performance measures.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize performance metrics calculator."""
        super().__init__(config)
        self.enable_performance_metrics: bool = self.pnl_config.get(
            "enable_performance_metrics", True
        )

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_win_rate(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate win rate and related metrics.

        Args:
            trades: List of trade dictionaries

        Returns:
            Dictionary containing win rate metrics
        """
        try:
            if not trades:
                return {
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                }

            winning_trades = 0
            losing_trades = 0
            total_pnl_wins = 0.0
            total_pnl_losses = 0.0

            for trade in trades:
                pnl = trade.get("pnl", 0.0)
                if pnl > 0:
                    winning_trades += 1
                    total_pnl_wins += pnl
                elif pnl < 0:
                    losing_trades += 1
                    total_pnl_losses += abs(pnl)

            total_trades = winning_trades + losing_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Calculate profit factor
            profit_factor = total_pnl_wins / total_pnl_losses if total_pnl_losses > 0 else 0.0

            # Calculate average win/loss
            avg_win = total_pnl_wins / winning_trades if winning_trades > 0 else 0.0
            avg_loss = total_pnl_losses / losing_trades if losing_trades > 0 else 0.0

            return {
                "win_rate": float(win_rate),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "profit_factor": float(profit_factor),
                "average_win": float(avg_win),
                "average_loss": float(avg_loss),
                "expectancy": float((win_rate * avg_win) - ((1 - win_rate) * avg_loss)),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating win rate: {e}")
            return {"win_rate": 0.0, "total_trades": 0}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_sortino_ratio(self, returns: np.ndarray, 
                               target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio.

        Args:
            returns: Array of returns
            target_return: Target/minimum acceptable return

        Returns:
            Sortino ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = returns - target_return
            mean_excess_return = np.mean(excess_returns)

            # Calculate downside deviation
            downside_returns = np.minimum(excess_returns, 0)
            downside_std = np.std(downside_returns)

            if downside_std == 0:
                return 0.0

            # Annualize assuming daily returns
            sortino_ratio = (mean_excess_return / downside_std) * np.sqrt(252)
            return float(sortino_ratio)

        except Exception as e:
            self.logger.exception(f"Error calculating Sortino ratio: {e}")
            return 0.0

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_calmar_ratio(self, returns: np.ndarray, 
                              equity_curve: np.ndarray) -> float:
        """
        Calculate Calmar ratio.

        Args:
            returns: Array of returns
            equity_curve: Array of equity values

        Returns:
            Calmar ratio
        """
        try:
            if len(returns) < 252 or len(equity_curve) < 2:  # Need at least 1 year
                return 0.0

            # Calculate annualized return
            total_return = (equity_curve[-1] / equity_curve[0]) - 1
            years = len(returns) / 252
            annualized_return = (1 + total_return) ** (1 / years) - 1

            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))

            if max_drawdown == 0:
                return 0.0

            calmar_ratio = annualized_return / max_drawdown
            return float(calmar_ratio)

        except Exception as e:
            self.logger.exception(f"Error calculating Calmar ratio: {e}")
            return 0.0

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_recovery_factor(self, total_pnl: float, 
                                 max_drawdown: float) -> float:
        """
        Calculate recovery factor.

        Args:
            total_pnl: Total profit/loss
            max_drawdown: Maximum drawdown (absolute value)

        Returns:
            Recovery factor
        """
        try:
            if max_drawdown == 0 or total_pnl <= 0:
                return 0.0

            recovery_factor = total_pnl / max_drawdown
            return float(recovery_factor)

        except Exception as e:
            self.logger.exception(f"Error calculating recovery factor: {e}")
            return 0.0