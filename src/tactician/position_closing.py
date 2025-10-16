

"""
Position Closing Module for Tactician.
Handles position closure based on dual model confidence scores and ATR-based exit rules.
"""
import logging
import time
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import src.utils.warning_symbols
from ..utils.logger import system_logger
from ..core.decorators import handles_errors

# Optional imports with fallbacks
try:
    from ..core.exceptions import (  # type: ignore
        failed,
        invalid,
    )
except ImportError:
    # Fallback functions if exceptions module is not available
    def failed(message: str) -> str:
        return f"FAILED: {message}"
    def invalid(message: str) -> str:
        return f"INVALID: {message}"

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

        decay_defaults = {
            "enabled": True,
            "max_bars_without_progress": 8,
            "min_unrealized_atr": 0.3,
            "min_unrealized_pct": 0.0,
            "grace_bars": 0,
        }
        self.decay_config = {
            **decay_defaults,
            **self.position_config.get("decay", {}),
        }

        # Track last exit evaluation metadata for downstream analytics
        self.last_exit_metadata: dict[str, Any] | None = None

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

            if self.decay_config.get("max_bars_without_progress", 0) < 0:
                self.logger.error(invalid("Decay bars threshold must be non-negative"))
                return False

            if self.decay_config.get("min_unrealized_atr", 0) < 0:
                self.logger.error(invalid("Decay ATR threshold must be non-negative"))
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
        default_return=(False, {"trigger": "error"}),
        context="position closure evaluation",
    )
    async def should_close_position(
        self,
        position_data: dict[str, Any],
    ) -> Tuple[bool, dict[str, Any]]:
        """
        Determine if a position should be closed based on exit triggers.

        Args:
            position_data: Position information

        Returns:
            Tuple[bool, Dict[str, Any]]: Decision flag and exit metadata
        """
        try:
            metadata: dict[str, Any] = {"trigger": None, "evaluations": {}}

            decay_result = self._evaluate_decay_triggers(position_data)
            metadata["evaluations"].update(decay_result.get("details", {}))

            if decay_result.get("should_close"):
                metadata["trigger"] = decay_result.get("trigger", "time_profit_decay")
                metadata["details"] = decay_result.get("details", {})
                self.last_exit_metadata = metadata
                self.logger.info(
                    "Closing position via %s decay trigger", metadata["trigger"]
                )
                return True, metadata

            if self._should_close_by_max_time(position_data):
                hold_seconds = self._calculate_hold_time_seconds(position_data)
                metadata["trigger"] = "max_hold_time"
                metadata["details"] = {
                    "hold_seconds": hold_seconds,
                    "max_hold_time": self.max_hold_time,
                }
                self.last_exit_metadata = metadata
                self.logger.info("Closing position due to maximum hold time")
                return True, metadata

            self.last_exit_metadata = metadata
            return False, metadata

        except Exception as e:
            self.logger.exception(failed(f"❌ Position closure evaluation failed: {e}"))
            error_metadata = {"trigger": "error", "details": {"message": str(e)}}
            self.last_exit_metadata = error_metadata
            return False, error_metadata

    def _should_close_by_max_time(self, position_data: dict[str, Any]) -> bool:
        """
        Check if position should be closed based on maximum hold time.

        Args:
            position_data: Position information

        Returns:
            bool: True if should close by maximum time
        """
        try:
            hold_time = self._calculate_hold_time_seconds(position_data)
            return hold_time >= self.max_hold_time

        except Exception as e:
            self.logger.exception(failed(f"❌ Maximum time-based closure check failed: {e}"))
            return False

    def _calculate_hold_time_seconds(self, position_data: dict[str, Any]) -> float:
        try:
            entry_time = position_data.get("entry_time")
            if not entry_time:
                return 0.0

            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)

            return (datetime.now() - entry_time).total_seconds()
        except Exception:
            return 0.0

    def _evaluate_decay_triggers(self, position_data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate time-decay and profit-decay conditions before forced closures."""

        result = {
            "should_close": False,
            "trigger": None,
            "details": {},
        }

        try:
            if not self.decay_config.get("enabled", True):
                return result

            bars_held = position_data.get("bars_held") or position_data.get("age_bars")
            if bars_held is None:
                bar_seconds = position_data.get("bar_duration_seconds") or 0
                hold_seconds = self._calculate_hold_time_seconds(position_data)
                bars_held = hold_seconds / bar_seconds if bar_seconds else None

            atr_value = (
                position_data.get("atr")
                or position_data.get("atr_value")
                or position_data.get("atr_current")
            )
            unrealized_pnl = position_data.get("unrealized_pnl")
            quantity = position_data.get("quantity") or 0.0
            reference_price = position_data.get("current_price") or position_data.get("entry_price")

            unrealized_atr = position_data.get("unrealized_atr")
            if unrealized_atr is None and atr_value:
                if unrealized_pnl is not None and quantity:
                    price_change = unrealized_pnl / max(quantity, 1e-9)
                    unrealized_atr = abs(price_change) / max(atr_value, 1e-9)

            result["details"].update(
                {
                    "bars_held": bars_held,
                    "atr_value": atr_value,
                    "unrealized_atr": unrealized_atr,
                }
            )

            if bars_held is None:
                return result

            threshold_bars = max(0, self.decay_config.get("max_bars_without_progress", 0))
            grace_bars = max(0, self.decay_config.get("grace_bars", 0))
            min_unrealized_atr = self.decay_config.get("min_unrealized_atr", 0.0)
            min_unrealized_pct = self.decay_config.get("min_unrealized_pct", 0.0)

            if bars_held >= threshold_bars + grace_bars:
                below_atr = (
                    unrealized_atr is not None and unrealized_atr < min_unrealized_atr
                )
                below_pct = False
                if unrealized_pnl is not None and reference_price:
                    notional = reference_price * max(quantity, 1e-9)
                    unrealized_pct = unrealized_pnl / max(notional, 1e-9)
                    result["details"]["unrealized_pct"] = unrealized_pct
                    below_pct = abs(unrealized_pct) < min_unrealized_pct if min_unrealized_pct else False

                if below_atr or below_pct:
                    result.update(
                        {
                            "should_close": True,
                            "trigger": "time_profit_decay",
                        }
                    )
                    result["details"].update(
                        {
                            "bars_threshold": threshold_bars,
                            "grace_bars": grace_bars,
                            "min_unrealized_atr": min_unrealized_atr,
                            "min_unrealized_pct": min_unrealized_pct,
                            "below_atr": below_atr,
                            "below_pct": below_pct,
                        }
                    )

            return result

        except Exception as exc:
            self.logger.exception(failed(f"❌ Decay trigger evaluation failed: {exc}"))
            result["details"]["error"] = str(exc)
            return result

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position closure execution",
    )
    async def close_position(
        self,
        position_data: dict[str, Any],
        close_reason: str,
        exit_metadata: dict[str, Any] | None = None,
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
            metadata_source = exit_metadata or self.last_exit_metadata or {}
            metadata = dict(metadata_source)
            trailing_state = position_data.get("trailing_state")
            if trailing_state and "trailing_state" not in metadata:
                metadata = {**metadata, "trailing_state": trailing_state}

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
                "exit_metadata": metadata,
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

            self.last_exit_metadata = None
            self.logger.info("✅ Position Closer cleanup completed")

        except Exception as e:
            self.logger.exception(failed(f"❌ Position Closer cleanup failed: {e}"))
