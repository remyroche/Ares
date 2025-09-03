"""
PnL Calculator Module.

This module handles profit and loss calculations for trading positions
and strategies.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.core.decorators import handles_errors

from .base import PnLLossFunctionsBase


class PnLCalculator(PnLLossFunctionsBase):
    """
    PnL Calculator for computing various profit and loss metrics.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize PnL calculator."""
        super().__init__(config)
        self.enable_pnl_calculation: bool = self.pnl_config.get(
            "enable_pnl_calculation", True
        )

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_total_pnl(self, calculation_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total PnL from trades.

        Args:
            calculation_input: Dictionary containing trade data

        Returns:
            Dictionary containing PnL metrics
        """
        try:
            trades = calculation_input.get("trades", [])
            if not trades:
                return {"total_pnl": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0}

            total_pnl = 0.0
            realized_pnl = 0.0
            unrealized_pnl = 0.0

            for trade in trades:
                if trade.get("status") == "closed":
                    pnl = trade.get("pnl", 0.0)
                    realized_pnl += pnl
                    total_pnl += pnl
                elif trade.get("status") == "open":
                    # Calculate unrealized PnL
                    entry_price = trade.get("entry_price", 0.0)
                    current_price = trade.get("current_price", entry_price)
                    quantity = trade.get("quantity", 0.0)
                    side = trade.get("side", "long")

                    if side == "long":
                        pnl = (current_price - entry_price) * quantity
                    else:  # short
                        pnl = (entry_price - current_price) * quantity

                    unrealized_pnl += pnl
                    total_pnl += pnl

            return {
                "total_pnl": total_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "trade_count": len(trades),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating total PnL: {e}")
            return {}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = returns - risk_free_rate
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)

            if std_excess_return == 0:
                return 0.0

            # Annualize assuming daily returns
            sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)
            return float(sharpe_ratio)

        except Exception as e:
            self.logger.exception(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Array of equity values

        Returns:
            Dictionary containing drawdown metrics
        """
        try:
            if len(equity_curve) < 2:
                return {"max_drawdown": 0.0, "max_drawdown_duration": 0}

            # Calculate running maximum
            running_max = np.maximum.accumulate(equity_curve)
            
            # Calculate drawdown
            drawdown = (equity_curve - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = np.min(drawdown)
            
            # Calculate drawdown duration
            in_drawdown = drawdown < 0
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

            return {
                "max_drawdown": abs(float(max_drawdown)),
                "max_drawdown_duration": int(max_drawdown_duration),
                "current_drawdown": abs(float(drawdown[-1])) if len(drawdown) > 0 else 0.0,
            }

        except Exception as e:
            self.logger.exception(f"Error calculating max drawdown: {e}")
            return {"max_drawdown": 0.0, "max_drawdown_duration": 0}