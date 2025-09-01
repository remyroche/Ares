# src/tactician/position_closing.py

"""
Position Closing Module for Tactician.
Handles position closure based on dual model confidence scores and ATR-based exit rules.
"""

from datetime import datetime
from typing import Any, Dict, Optional, List

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
)

class PositionCloser:
    """
    Position Closer that handles position closure based on dual model confidence scores
    and ATR-based exit rules.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
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
        self.atr_multiplier = tpsl_optimization.get("atr_multiplier", 2.0)
        self.confidence_threshold = tpsl_optimization.get("confidence_threshold", 0.7)
        self.min_hold_time = tpsl_optimization.get("min_hold_time", 300)  # 5 minutes

        # Load additional optimized parameters
        self.stop_loss_multiplier = tpsl_optimization.get("stop_loss_multiplier", 1.5)
        self.take_profit_multiplier = tpsl_optimization.get("take_profit_multiplier", 2.0)
        self.trailing_stop_enabled = tpsl_optimization.get("trailing_stop_enabled", True)
        self.trailing_stop_distance = tpsl_optimization.get("trailing_stop_distance", 0.02)
        self.max_hold_time = tpsl_optimization.get("max_hold_time", 3600)  # 1 hour

        # State tracking
        self.closed_positions = []
        self.position_history = []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closer initialization"
    )
    def _validate_configuration(self) -> bool:
        """
        Validate position closer configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.atr_multiplier <= 0:
                self.logger.error(invalid("ATR multiplier must be positive"))
                return False

            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid("Confidence threshold must be between 0 and 1"))
                return False

            if self.min_hold_time < 0:
                self.logger.error(invalid("Minimum hold time must be non-negative"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position closure evaluation"
    )
    def _should_close_by_atr(
        self,
        position_data: Dict[str, Any],
        atr_value: float,
        current_price: float
    ) -> bool:
        """
        Check if position should be closed based on ATR.

        Args:
            position_data: Position information
            atr_value: ATR value
            current_price: Current market price

        Returns:
            bool: True if should close by ATR
        """
        try:
            entry_price = position_data.get("entry_price", 0)
            if entry_price <= 0:
                return False

            # Calculate ATR-based exit levels
            atr_exit_distance = atr_value * self.atr_multiplier

            # For long positions
            if position_data.get("side", "").upper() == "LONG":
                stop_loss = entry_price - atr_exit_distance
                return current_price <= stop_loss

            # For short positions
            elif position_data.get("side", "").upper() == "SHORT":
                stop_loss = entry_price + atr_exit_distance
                return current_price >= stop_loss

            return False

        except Exception as e:
            self.logger.error(failed(f"❌ ATR-based closure check failed: {e}"))
            return False

    def _should_close_by_time(self, position_data: Dict[str, Any]) -> bool:
        """
        Check if position should be closed based on minimum hold time.

        Args:
            position_data: Position information

        Returns:
            bool: True if should close by time
        """
        try:
            entry_time = position_data.get("entry_time")
            if not entry_time:
                return False

            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

            hold_time = (datetime.now() - entry_time).total_seconds()
            return hold_time >= self.min_hold_time

        except Exception as e:
            self.logger.error(failed(f"❌ Time-based closure check failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position closure execution"
    )
    def _calculate_pnl(self, position_data: Dict[str, Any]) -> float:
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
            elif side == "SHORT":
                return (entry_price - current_price) * quantity
            else:
                return 0.0

        except Exception as e:
            self.logger.error(failed(f"❌ PnL calculation failed: {e}"))
            return 0.0
