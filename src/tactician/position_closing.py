

import src.utils.warning_symbols
import numpy as np
from ...utils.logger import system_logger
from .core.decorators import handles_errors

# src/tactician/position_closing.py


"""
Position Closing Module for Tactician.
Handles position closure based on dual model confidence scores and ATR-based exit rules.
"""
from datetime import datetime
from typing import Any

from ...utils.logger import system_logger
from src.core.exceptions import (
import logging
import time

    failed,
    invalid,
)


class PositionCloser:
    """
    Position Closer that handles position closure based on dual model confidence scores
    and ATR-based exit rules.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize Position Closer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PositionCloser")

        # Configuration from step17 optimization results
        self.position_config = config.get("position_closing", {})

        # Load step17 optimized parameters
        step17_config = config.get("step17_optimization", {})
        tpsl_optimization = step17_config.get("tpsl", {})

        # Load optimized position closing parameters
        # Note: ATR-based stop loss, confidence threshold, and min hold time removed
        # Keep only maximum hold time at 3 hours (10800 seconds)
        self.max_hold_time = tpsl_optimization.get("max_hold_time", 10800)  # 3 hours

        # State tracking
        self.closed_positions = []
        self.position_history = []

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closer initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the position closer.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Position Closer...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid position closer configuration"))
                return False

            self.logger.info("✅ Position Closer initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                failed(f"❌ Position Closer initialization failed: {e}")
            )
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate position closer configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.max_hold_time <= 0:
                self.logger.error(invalid("Maximum hold time must be positive"))
                return False

            return True

        except Exception as e:
            self.logger.exception(failed(f"❌ Configuration validation failed: {e}"))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            if "tpsl" in step17_results:
                tpsl_optimization = step17_results["tpsl"]

                # Update only maximum hold time (other parameters removed)
                self.max_hold_time = tpsl_optimization.get(
                    "max_hold_time", self.max_hold_time
                )

                self.logger.info(
                    "✅ Position closer configuration refreshed from step17 results"
                )

        except Exception as e:
            self.logger.exception(f"Error refreshing step17 configuration: {e}")

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closure evaluation",
    )
    async def should_close_position(
        self,
        position_data: dict[str, Any],
    ) -> bool:
        """
        Determine if a position should be closed based on maximum hold time.

        Args:
            position_data: Position information

        Returns:
            bool: True if position should be closed
        """
        try:
            # Check maximum hold time only
            if self._should_close_by_max_time(position_data):
                self.logger.info("Closing position due to maximum hold time")
                return True

            return False

        except Exception as e:
            self.logger.exception(failed(f"❌ Position closure evaluation failed: {e}"))
            return False

    def _should_close_by_max_time(self, position_data: dict[str, Any]) -> bool:
        """
        Check if position should be closed based on maximum hold time.

        Args:
            position_data: Position information

        Returns:
            bool: True if should close by maximum time
        """
        try:
            entry_time = position_data.get("entry_time")
            if not entry_time:
                return False

            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)

            hold_time = (datetime.now() - entry_time).total_seconds()
            return hold_time >= self.max_hold_time

        except Exception as e:
            self.logger.exception(failed(f"❌ Maximum time-based closure check failed: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position closure execution",
    )
    async def close_position(
        self,
        position_data: dict[str, Any],
        close_reason: str,
    ) -> dict[str, Any] | None:
        """
        Execute position closure.

        Args:
            position_data: Position information
            close_reason: Reason for closure

        Returns:
            Dict: Closure result or None if failed
        """
        try:
            self.logger.info(f"Closing position: {close_reason}")

            # Record closure
            closure_record = {
                "position_id": position_data.get("position_id"),
                "symbol": position_data.get("symbol"),
                "side": position_data.get("side"),
                "entry_price": position_data.get("entry_price"),
                "exit_price": position_data.get("current_price"),
                "quantity": position_data.get("quantity"),
                "close_reason": close_reason,
                "close_time": datetime.now().isoformat(),
                "pnl": self._calculate_pnl(position_data),
            }

            self.closed_positions.append(closure_record)
            self.position_history.append(closure_record)

            self.logger.info(
                f"✅ Position closed successfully: {closure_record['pnl']:.4f} PnL"
            )
            return closure_record

        except Exception as e:
            self.logger.exception(failed(f"❌ Position closure failed: {e}"))
            return None

    def _calculate_pnl(self, position_data: dict[str, Any]) -> float:
        """
        Calculate position PnL.

        Args:
            position_data: Position information

        Returns:
            float: Calculated PnL
        """
        try:
            entry_price = position_data.get("entry_price", 0)
            current_price = position_data.get("current_price", 0)
            quantity = position_data.get("quantity", 0)
            side = position_data.get("side", "").upper()

            if entry_price <= 0 or current_price <= 0 or quantity <= 0:
                return 0.0

            if side == "LONG":
                return (current_price - entry_price) * quantity
            if side == "SHORT":
                return (entry_price - current_price) * quantity
            return 0.0

        except Exception as e:
            self.logger.exception(failed(f"❌ PnL calculation failed: {e}"))
            return 0.0

    def get_closed_positions(self) -> list[dict[str, Any]]:
        """
        Get list of closed positions.

        Returns:
            List[Dict[str, Any]]: Closed positions
        """
        return self.closed_positions.copy()

    def get_position_history(self) -> list[dict[str, Any]]:
        """
        Get complete position history.

        Returns:
            List[Dict[str, Any]]: Position history
        """
        return self.position_history.copy()

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for closed positions.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            if not self.closed_positions:
                return {
                    "total_positions": 0,
                    "winning_positions": 0,
                    "losing_positions": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "average_pnl": 0.0,
                }

            total_positions = len(self.closed_positions)
            winning_positions = len(
                [p for p in self.closed_positions if p.get("pnl", 0) > 0]
            )
            losing_positions = len(
                [p for p in self.closed_positions if p.get("pnl", 0) < 0]
            )
            total_pnl = sum(p.get("pnl", 0) for p in self.closed_positions)

            return {
                "total_positions": total_positions,
                "winning_positions": winning_positions,
                "losing_positions": losing_positions,
                "win_rate": (
                    winning_positions / total_positions if total_positions > 0 else 0.0
                ),
                "total_pnl": total_pnl,
                "average_pnl": (
                    total_pnl / total_positions if total_positions > 0 else 0.0
                ),
            }

        except Exception as e:
            self.logger.exception(
                failed(f"❌ Performance metrics calculation failed: {e}")
            )
            return {}

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info("Cleaning up Position Closer...")

            # Save position history if needed
            if self.position_history:
                self.logger.info(
                    f"Saving {len(self.position_history)} position records"
                )

            self.logger.info("✅ Position Closer cleanup completed")

        except Exception as e:
            self.logger.exception(failed(f"❌ Position Closer cleanup failed: {e}"))
