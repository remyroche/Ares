# src/tactician/position_closing.py

"""
Position Closing Module for Tactician.
Handles position closure based on dual model confidence scores and ATR-based exit rules.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.config_optuna import get_parameter_value
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


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
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PositionCloser")

        # Configuration
        self.closing_config: dict[str, Any] = self.config.get("position_closing", {})

        # Confidence thresholds
        self.neutral_signal_threshold: float = get_parameter_value(
            "confidence_thresholds.neutral_signal_threshold",
            0.5,
        )
        self.close_signal_threshold: float = get_parameter_value(
            "confidence_thresholds.position_close_threshold",
            0.4,
        )
        self.tactician_close_threshold: float = get_parameter_value(
            "confidence_thresholds.tactician_close_threshold",
            0.6,
        )

        # ATR-based exit rules
        self.atr_multiplier: float = self.closing_config.get("atr_multiplier", 1.5)
        self.atr_timeframe: str = self.closing_config.get("atr_timeframe", "1m")

        # Hard stop-loss settings
        self.hard_stop_loss_enabled: bool = self.closing_config.get(
            "hard_stop_loss_enabled",
            True,
        )
        self.max_position_hold_hours: float = self.closing_config.get(
            "max_position_hold_hours",
            2.0,
        )

        # Position tracking
        self.closed_positions: list[dict[str, Any]] = []
        self.is_initialized: bool = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position closer configuration"),
            AttributeError: (False, "Missing required closer parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position closer initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize Position Closer with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Position Closer...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for position closer")
                return False

            self.is_initialized = True
            self.logger.info("✅ Position Closer initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Position Closer initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate position closer configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate confidence thresholds
            if not (0.0 <= self.neutral_signal_threshold <= 1.0):
                self.logger.error("Neutral signal threshold must be between 0 and 1")
                return False

            if not (0.0 <= self.close_signal_threshold <= 1.0):
                self.logger.error("Close signal threshold must be between 0 and 1")
                return False

            if not (0.0 <= self.tactician_close_threshold <= 1.0):
                self.logger.error("Tactician close threshold must be between 0 and 1")
                return False

            # Validate ATR settings
            if self.atr_multiplier <= 0:
                self.logger.error("ATR multiplier must be positive")
                return False

            if self.max_position_hold_hours <= 0:
                self.logger.error("Max position hold hours must be positive")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid position data for closing analysis"),
            AttributeError: (None, "Closer not properly initialized"),
        },
        default_return=None,
        context="position closing analysis",
    )
    async def analyze_position_closing(
        self,
        current_position: dict[str, Any],
        analyst_confidence: float,
        tactician_confidence: float,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Analyze whether to close a position based on dual model confidence scores.

        Args:
            current_position: Current position information
            analyst_confidence: Analyst model confidence score
            tactician_confidence: Tactician model confidence score
            market_data: Market data for ATR calculation
            current_price: Current asset price

        Returns:
            Dictionary with closing analysis
        """
        try:
            if not self.is_initialized:
                raise ValueError("Position Closer not initialized")

            self.logger.info("🔍 Analyzing position closing...")

            # Calculate ATR for exit rules
            atr_value = self._calculate_atr(market_data, self.atr_timeframe)

            # Check time-based exit rule
            time_exit_triggered = self._check_time_based_exit(current_position)

            # Check hard stop-loss
            hard_stop_triggered = self._check_hard_stop_loss(
                current_position,
                current_price,
            )

            # Determine closing decision based on signals
            closing_decision = self._determine_closing_decision(
                analyst_confidence,
                tactician_confidence,
                current_position,
                current_price,
                atr_value,
                time_exit_triggered,
                hard_stop_triggered,
            )

            # Generate closing analysis
            closing_analysis = {
                "timestamp": datetime.now().isoformat(),
                "position_id": current_position.get("position_id", "unknown"),
                "analyst_confidence": analyst_confidence,
                "tactician_confidence": tactician_confidence,
                "atr_value": atr_value,
                "time_exit_triggered": time_exit_triggered,
                "hard_stop_triggered": hard_stop_triggered,
                "closing_decision": closing_decision,
                "current_price": current_price,
                "position_entry_price": current_position.get("entry_price", 0.0),
                "position_size": current_position.get("size", 0.0),
                "position_direction": current_position.get("direction", "unknown"),
            }

            self.logger.info(
                f"✅ Position closing analysis completed: {closing_decision['action']}",
            )
            return closing_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing position closing: {e}")
            return self._get_fallback_closing_decision()

    def _determine_closing_decision(
        self,
        analyst_confidence: float,
        tactician_confidence: float,
        current_position: dict[str, Any],
        current_price: float,
        atr_value: float,
        time_exit_triggered: bool,
        hard_stop_triggered: bool,
    ) -> dict[str, Any]:
        """Determine closing decision based on confidence scores and exit rules."""
        try:
            position_direction = current_position.get("direction", "LONG")
            entry_price = current_position.get("entry_price", current_price)

            # Check for hard stop-loss first
            if hard_stop_triggered:
                return {
                    "action": "CLOSE_ALL",
                    "reason": "Hard stop-loss triggered",
                    "close_percentage": 1.0,
                    "priority": "high",
                }

            # Check for time-based exit
            if time_exit_triggered:
                return {
                    "action": "CLOSE_ALL",
                    "reason": "Maximum position hold time exceeded",
                    "close_percentage": 1.0,
                    "priority": "high",
                }

            # Check for NEUTRAL signal (close 50% when tactician confidence drops)
            if analyst_confidence < self.neutral_signal_threshold:
                if tactician_confidence < self.tactician_close_threshold:
                    return {
                        "action": "CLOSE_PARTIAL",
                        "reason": f"NEUTRAL signal: Analyst confidence {analyst_confidence:.2f} < {self.neutral_signal_threshold}, Tactician confidence {tactician_confidence:.2f} < {self.tactician_close_threshold}",
                        "close_percentage": 0.5,
                        "priority": "medium",
                    }

            # Check for CLOSE signal
            if analyst_confidence < self.close_signal_threshold:
                if tactician_confidence < self.tactician_close_threshold:
                    return {
                        "action": "CLOSE_ALL",
                        "reason": f"CLOSE signal: Analyst confidence {analyst_confidence:.2f} < {self.close_signal_threshold}, Tactician confidence {tactician_confidence:.2f} < {self.tactician_close_threshold}",
                        "close_percentage": 1.0,
                        "priority": "high",
                    }

            # Check for ATR-based exit (price reversal by 1.5x ATR)
            atr_exit_triggered = self._check_atr_based_exit(
                current_position,
                current_price,
                atr_value,
                position_direction,
            )

            if atr_exit_triggered:
                return {
                    "action": "CLOSE_ALL",
                    "reason": f"ATR-based exit: Price reversed by {self.atr_multiplier}x ATR ({atr_value:.4f})",
                    "close_percentage": 1.0,
                    "priority": "high",
                }

            # No closing action needed
            return {
                "action": "HOLD",
                "reason": "No closing conditions met",
                "close_percentage": 0.0,
                "priority": "low",
            }

        except Exception as e:
            self.logger.error(f"Error determining closing decision: {e}")
            return {
                "action": "HOLD",
                "reason": f"Error determining closing decision: {e}",
                "close_percentage": 0.0,
                "priority": "low",
            }

    def _check_time_based_exit(self, current_position: dict[str, Any]) -> bool:
        """Check if position should be closed based on time limit."""
        try:
            entry_time = current_position.get("entry_time")
            if entry_time is None:
                return False

            # Convert to datetime if it's a string
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))

            current_time = datetime.now()
            time_diff = (
                current_time - entry_time
            ).total_seconds() / 3600  # Convert to hours

            return time_diff >= self.max_position_hold_hours

        except Exception as e:
            self.logger.error(f"Error checking time-based exit: {e}")
            return False

    def _check_hard_stop_loss(
        self,
        current_position: dict[str, Any],
        current_price: float,
    ) -> bool:
        """Check if hard stop-loss should be triggered."""
        try:
            if not self.hard_stop_loss_enabled:
                return False

            entry_price = current_position.get("entry_price", 0.0)
            direction = current_position.get("direction", "LONG")

            if entry_price <= 0:
                return False

            # Calculate price change percentage
            if direction == "LONG":
                price_change_pct = (current_price - entry_price) / entry_price
            elif direction == "SHORT":
                price_change_pct = (entry_price - current_price) / entry_price
            else:
                return False

            # Hard stop-loss at 90% of liquidation point (conservative)
            return price_change_pct <= -0.9

        except Exception as e:
            self.logger.error(f"Error checking hard stop-loss: {e}")
            return False

    def _check_atr_based_exit(
        self,
        current_position: dict[str, Any],
        current_price: float,
        atr_value: float,
        position_direction: str,
    ) -> bool:
        """Check if ATR-based exit should be triggered."""
        try:
            entry_price = current_position.get("entry_price", 0.0)

            if entry_price <= 0:
                return False

            # Calculate price change
            price_change = current_price - entry_price

            # Calculate ATR threshold
            atr_threshold = atr_value * self.atr_multiplier

            # Check if price has reversed by ATR threshold
            if position_direction == "LONG":
                # For long positions, check if price dropped by ATR threshold
                return price_change <= -atr_threshold
            if position_direction == "SHORT":
                # For short positions, check if price rose by ATR threshold
                return price_change >= atr_threshold
            return False

        except Exception as e:
            self.logger.error(f"Error checking ATR-based exit: {e}")
            return False

    def _calculate_atr(self, market_data: pd.DataFrame, timeframe: str) -> float:
        """Calculate Average True Range (ATR) for the specified timeframe."""
        try:
            if market_data.empty:
                return 0.0

            # Ensure we have the required columns
            required_columns = ["high", "low", "close"]
            if not all(col in market_data.columns for col in required_columns):
                self.logger.warning("Missing required columns for ATR calculation")
                return 0.0

            # Calculate True Range
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]

            # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR (14-period average)
            atr_period = 14
            atr = true_range.rolling(window=atr_period).mean()

            # Return the latest ATR value
            latest_atr = atr.iloc[-1] if not atr.empty else 0.0

            return float(latest_atr)

        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0

    def _get_fallback_closing_decision(self) -> dict[str, Any]:
        """Get fallback closing decision when analysis fails."""
        return {
            "timestamp": datetime.now().isoformat(),
            "position_id": "unknown",
            "analyst_confidence": 0.0,
            "tactician_confidence": 0.0,
            "atr_value": 0.0,
            "time_exit_triggered": False,
            "hard_stop_triggered": False,
            "closing_decision": {
                "action": "HOLD",
                "reason": "Fallback decision - analysis failed",
                "close_percentage": 0.0,
                "priority": "low",
            },
            "current_price": 0.0,
            "position_entry_price": 0.0,
            "position_size": 0.0,
            "position_direction": "unknown",
        }

    def record_closed_position(self, position_data: dict[str, Any]) -> None:
        """Record a closed position for historical tracking."""
        try:
            self.closed_positions.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    **position_data,
                },
            )

            # Keep only last 1000 closed positions
            if len(self.closed_positions) > 1000:
                self.closed_positions = self.closed_positions[-1000:]

        except Exception as e:
            self.logger.error(f"Error recording closed position: {e}")

    def get_closed_positions(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get closed positions history."""
        try:
            if limit is None:
                return self.closed_positions.copy()
            return self.closed_positions[-limit:]

        except Exception as e:
            self.logger.error(f"Error getting closed positions: {e}")
            return []

    def get_closing_statistics(self) -> dict[str, Any]:
        """Get closing statistics."""
        try:
            if not self.closed_positions:
                return {
                    "total_closed": 0,
                    "partial_closes": 0,
                    "full_closes": 0,
                    "avg_hold_time": 0.0,
                    "success_rate": 0.0,
                }

            total_closed = len(self.closed_positions)
            partial_closes = sum(
                1 for p in self.closed_positions if p.get("close_percentage", 1.0) < 1.0
            )
            full_closes = total_closed - partial_closes

            # Calculate average hold time
            hold_times = []
            for position in self.closed_positions:
                entry_time = position.get("entry_time")
                exit_time = position.get("exit_time")
                if entry_time and exit_time:
                    try:
                        if isinstance(entry_time, str):
                            entry_time = datetime.fromisoformat(
                                entry_time.replace("Z", "+00:00"),
                            )
                        if isinstance(exit_time, str):
                            exit_time = datetime.fromisoformat(
                                exit_time.replace("Z", "+00:00"),
                            )
                        hold_time = (
                            exit_time - entry_time
                        ).total_seconds() / 3600  # hours
                        hold_times.append(hold_time)
                    except Exception:
                        continue

            avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0.0

            # Calculate success rate (profitable closes)
            profitable_closes = sum(
                1 for p in self.closed_positions if p.get("profit", 0.0) > 0
            )
            success_rate = profitable_closes / total_closed if total_closed > 0 else 0.0

            return {
                "total_closed": total_closed,
                "partial_closes": partial_closes,
                "full_closes": full_closes,
                "avg_hold_time": avg_hold_time,
                "success_rate": success_rate,
            }

        except Exception as e:
            self.logger.error(f"Error getting closing statistics: {e}")
            return {
                "total_closed": 0,
                "partial_closes": 0,
                "full_closes": 0,
                "avg_hold_time": 0.0,
                "success_rate": 0.0,
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position closer cleanup",
    )
    async def stop(self) -> None:
        """Stop the position closer."""
        self.logger.info("🛑 Stopping Position Closer...")

        try:
            self.is_initialized = False
            self.logger.info("✅ Position Closer stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping position closer: {e}")


# Global position closer instance
position_closer: PositionCloser | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position closer setup",
)
async def setup_position_closer(
    config: dict[str, Any] | None = None,
) -> PositionCloser | None:
    """
    Setup global position closer.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[PositionCloser]: Global position closer instance
    """
    try:
        global position_closer

        if config is None:
            config = {
                "position_closing": {
                    "neutral_signal_threshold": 0.5,
                    "close_signal_threshold": 0.4,
                    "tactician_close_threshold": 0.6,
                    "atr_multiplier": 1.5,
                    "atr_timeframe": "1m",
                    "hard_stop_loss_enabled": True,
                    "max_position_hold_hours": 2.0,
                },
            }

        # Create position closer
        position_closer = PositionCloser(config)

        # Initialize position closer
        success = await position_closer.initialize()
        if success:
            return position_closer
        return None

    except Exception as e:
        print(f"Error setting up position closer: {e}")
        return None
