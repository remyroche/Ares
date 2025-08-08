import asyncio
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.analyst.ml_dynamic_target_predictor import MLDynamicTargetPredictor
from src.utils.error_handler import (
    handle_errors,
    handle_network_operations,
)
from src.utils.logger import system_logger


class MLTargetUpdater:
    """
    Continuously monitors active positions and updates their targets based on:
    - Real-time ML predictions
    - Changing market conditions
    - Position performance
    - Risk management rules

    This ensures targets are constantly optimized rather than being set once at entry.
    """

    def __init__(
        self,
        ml_target_predictor: MLDynamicTargetPredictor,
        exchange_client,
        state_manager,
        config: dict[str, Any],
    ):
        self.ml_target_predictor = ml_target_predictor
        self.exchange = exchange_client  # expected to be a client from exchange/factory
        self.state_manager = state_manager
        self.config = config.get("ml_target_updater", {})
        self.logger = system_logger.getChild("MLTargetUpdater")

        # Update configuration
        self.update_interval_seconds = self.config.get(
            "update_interval_seconds",
            300,
        )  # 5 minutes
        self.min_time_between_updates = self.config.get(
            "min_time_between_updates_seconds",
            60,
        )  # 1 minute
        from src.config_optuna import get_parameter_value

        self.confidence_threshold_for_update = get_parameter_value(
            "confidence_thresholds.ml_target_update_threshold",
            0.6,
        )
        self.max_target_change_percent = self.config.get(
            "max_target_change_percent",
            0.25,
        )  # 25% max change
        self.enable_stop_loss_updates = self.config.get(
            "enable_stop_loss_updates",
            True,
        )
        self.enable_take_profit_updates = self.config.get(
            "enable_take_profit_updates",
            True,
        )
        self.trailing_stop_enabled = self.config.get("trailing_stop_enabled", True)

        # Risk management settings
        self.max_sl_distance_from_entry = self.config.get(
            "max_sl_distance_from_entry",
            0.03,
        )  # 3% for high leverage
        self.min_tp_distance_from_current = self.config.get(
            "min_tp_distance_from_current",
            0.005,
        )  # 0.5% for high leverage

        # High leverage optimizations
        self.high_leverage_mode = self.config.get("high_leverage_mode", True)
        self.emergency_update_threshold = self.config.get(
            "emergency_update_threshold",
            0.02,
        )
        self.volatility_scaling = self.config.get("volatility_scaling", True)

        # State tracking
        self.last_update_time = {}
        self.update_history = {}
        self.is_running = False

    @handle_errors(
        exceptions=(asyncio.CancelledError,),
        default_return=None,
        context="target updater start",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="target updater monitoring",
    )
    async def start_monitoring(self):
        """Start the continuous target monitoring and updating process."""
        if self.is_running:
            self.logger.warning("Target updater is already running")
            return

        self.is_running = True
        self.logger.info("Started ML target updater monitoring")

        while self.is_running:
            await self._update_cycle()
            await asyncio.sleep(self.update_interval_seconds)

        self.is_running = False

    def stop_monitoring(self):
        """Stop the target monitoring process."""
        self.is_running = False
        self.logger.info("Stopped ML target updater monitoring")

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="target update cycle",
    )
    async def _update_cycle(self):
        """Single update cycle - check and update targets for active positions."""
        try:
            # Get current position from state manager
            current_position = self.state_manager.get_state("current_position", {})

            if not current_position or current_position.get("direction") is None:
                return  # No active position

            # Check if enough time has passed since last update
            position_id = current_position.get("trade_id", "unknown")
            last_update = self.last_update_time.get(position_id, 0)

            # Check for emergency update conditions
            should_emergency_update = await self._check_emergency_update_conditions(
                current_position,
                last_update,
            )

            if (
                not should_emergency_update
                and time.time() - last_update < self.min_time_between_updates
            ):
                return  # Too soon to update

            # Get current market data
            current_market_data = await self._get_current_market_data()
            if not current_market_data:
                return

            # Determine signal type from position
            signal_type = self._determine_signal_type(current_position)
            if not signal_type:
                return

            # Get ML prediction for current conditions
            ml_targets = await self._get_ml_targets(
                signal_type,
                current_market_data,
                current_position,
            )

            if not ml_targets:
                return

            # Evaluate if targets should be updated
            update_decision = self._evaluate_target_update(
                current_position,
                ml_targets,
                current_market_data,
            )

            if update_decision.get("should_update", False):
                await self._execute_target_update(
                    current_position,
                    update_decision,
                    ml_targets,
                    current_market_data,
                )

                # Record update time
                self.last_update_time[position_id] = time.time()

        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}", exc_info=True)

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="emergency update check",
    )
    async def _check_emergency_update_conditions(
        self,
        position: dict[str, Any],
        last_update: float,
    ) -> bool:
        """Check if emergency update is needed due to significant price movement."""
        if not self.high_leverage_mode:
            return False

        try:
            # Get current market data
            market_data = await self._get_current_market_data()
            if not market_data:
                return False

            current_price = market_data.get("current_price", 0)
            entry_price = position.get("entry_price", 0)

            if entry_price <= 0:
                return False

            # Calculate price movement since entry
            price_movement = abs(current_price - entry_price) / entry_price

            # Emergency update if price moved more than threshold
            if price_movement > self.emergency_update_threshold:
                self.logger.warning(
                    f"Emergency update triggered: Price moved {price_movement:.2%} "
                    f"(threshold: {self.emergency_update_threshold:.2%})",
                )
                return True

            # Emergency update if too much time has passed (volatility scaling)
            if self.volatility_scaling:
                time_since_update = time.time() - last_update
                if (
                    time_since_update > self.update_interval_seconds * 2
                ):  # Double the normal interval
                    self.logger.info(
                        "Emergency update triggered: Too much time since last update",
                    )
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking emergency update conditions: {e}")
            return False

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def _get_current_market_data(self) -> dict[str, Any] | None:
        """Get current market data needed for ML predictions."""
        try:
            # Get current price data
            ticker = await self.exchange.get_ticker(
                self.state_manager.get_state("trade_symbol", "BTCUSDT"),
            )

            # Get recent klines for technical analysis
            klines = await self.exchange.get_klines(
                symbol=self.state_manager.get_state("trade_symbol", "BTCUSDT"),
                interval="1m",
                limit=100,
            )

            if not klines:
                return None

            # Convert to DataFrame for technical analysis
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "count",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
            )

            # Convert to numeric
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            current_price = float(ticker.get("price", df["close"].iloc[-1]))

            return {
                "current_price": current_price,
                "close": current_price,
                "high": df["high"].iloc[-1],
                "low": df["low"].iloc[-1],
                "volume": df["volume"].iloc[-1],
                "klines_df": df,
                "ticker": ticker,
            }

        except Exception as e:
            self.logger.error(f"Error getting current market data: {e}")
            return None

    def _determine_signal_type(self, position: dict[str, Any]) -> str | None:
        """Determine the signal type from position data."""
        direction = position.get("direction")
        entry_context = position.get("entry_context", {})

        # Try to get signal type from entry context
        original_signal = entry_context.get("signal_type", "")

        if original_signal in [
            "SR_FADE_LONG",
            "SR_FADE_SHORT",
            "SR_BREAKOUT_LONG",
            "SR_BREAKOUT_SHORT",
        ]:
            return original_signal

        # Fallback: infer from direction and other context
        if direction == "long":
            # Default to breakout long if we can't determine
            return "SR_BREAKOUT_LONG"
        if direction == "short":
            return "SR_BREAKOUT_SHORT"

        return None

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="ML target predictions",
    )
    async def _get_ml_targets(
        self,
        signal_type: str,
        market_data: dict[str, Any],
        position: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Get ML target predictions for current conditions."""
        try:
            # Calculate current ATR from recent data
            klines_df = market_data.get("klines_df")
            if klines_df is None or len(klines_df) < 14:
                return None

            # Simple ATR calculation
            high_low = klines_df["high"] - klines_df["low"]
            high_close = np.abs(klines_df["high"] - klines_df["close"].shift(1))
            low_close = np.abs(klines_df["low"] - klines_df["close"].shift(1))

            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            current_atr = tr.rolling(window=14).mean().iloc[-1]

            # Prepare features DataFrame (simplified for real-time)
            current_features = pd.DataFrame(
                {
                    "close": [market_data["current_price"]],
                    "high": [market_data["high"]],
                    "low": [market_data["low"]],
                    "volume": [market_data["volume"]],
                    "ATR": [current_atr],
                },
            )

            # Get ML predictions
            ml_targets = self.ml_target_predictor.predict_dynamic_targets(
                signal_type=signal_type,
                technical_analysis_data=market_data,
                current_features=current_features,
                current_atr=current_atr,
            )

            return ml_targets

        except Exception as e:
            self.logger.error(f"Error getting ML targets: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="target update evaluation",
    )
    def _evaluate_target_update(
        self,
        position: dict[str, Any],
        ml_targets: dict[str, Any],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate whether targets should be updated and how."""
        update_decision = self._initialize_update_decision(position, ml_targets)

        # Early exit if confidence is too low
        if not self._check_confidence_threshold(update_decision):
            return update_decision

        # Process target updates through validation pipeline
        update_decision = self._process_target_updates(
            position,
            ml_targets,
            market_data,
            update_decision,
        )

        # Finalize update decision
        update_decision = self._finalize_update_decision(update_decision)

        return update_decision

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="target update application",
    )
    def _process_target_updates(
        self,
        position: dict[str, Any],
        ml_targets: dict[str, Any],
        market_data: dict[str, Any],
        update_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Process target updates through validation pipeline."""
        # Apply change limits
        new_tp, new_sl = self._apply_change_limits(
            position,
            ml_targets,
            update_decision,
        )

        # Apply risk management rules
        new_tp, new_sl = self._apply_risk_management_rules(
            position,
            new_tp,
            new_sl,
            market_data,
            update_decision,
        )

        # Evaluate update opportunities
        self._evaluate_update_opportunities(position, new_tp, new_sl, update_decision)

        # Store processed targets
        update_decision["new_tp"] = new_tp
        update_decision["new_sl"] = new_sl

        return update_decision

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="target update finalization",
    )
    def _finalize_update_decision(
        self,
        update_decision: dict[str, Any],
    ) -> dict[str, Any]:
        """Finalize the update decision."""
        if update_decision["update_tp"] or update_decision["update_sl"]:
            update_decision["should_update"] = True

        return update_decision

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="target update initialization",
    )
    def _initialize_update_decision(
        self,
        position: dict[str, Any],
        ml_targets: dict[str, Any],
    ) -> dict[str, Any]:
        """Initialize the update decision structure."""
        return {
            "should_update": False,
            "update_tp": False,
            "update_sl": False,
            "new_tp": ml_targets.get("take_profit", position.get("take_profit", 0)),
            "new_sl": ml_targets.get("stop_loss", position.get("stop_loss", 0)),
            "confidence": ml_targets.get("prediction_confidence", 0),
            "reasons": [],
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=False,
        context="confidence threshold check",
    )
    def _check_confidence_threshold(self, update_decision: dict[str, Any]) -> bool:
        """Check if confidence meets the threshold for updates."""
        confidence = update_decision["confidence"]
        if confidence < self.confidence_threshold_for_update:
            update_decision["reasons"].append(f"Low confidence: {confidence:.2f}")
            return False
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="change limits application",
    )
    def _apply_change_limits(
        self,
        position: dict[str, Any],
        ml_targets: dict[str, Any],
        update_decision: dict[str, Any],
    ) -> tuple[float, float]:
        """Apply change limits to prevent excessive updates."""
        current_tp = position.get("take_profit", 0)
        current_sl = position.get("stop_loss", 0)
        new_tp = ml_targets.get("take_profit", current_tp)
        new_sl = ml_targets.get("stop_loss", current_sl)

        # Calculate percentage changes
        tp_change_percent = (
            abs(new_tp - current_tp) / current_tp if current_tp > 0 else 0
        )
        sl_change_percent = (
            abs(new_sl - current_sl) / current_sl if current_sl > 0 else 0
        )

        # Apply limits with high leverage considerations
        if self.high_leverage_mode:
            # For high leverage, allow larger changes but with safety checks
            max_change = self.max_target_change_percent

            # Allow larger changes for stop loss (risk management)
            if sl_change_percent > max_change * 1.5:  # 75% for SL
                update_decision["reasons"].append(
                    f"SL change too large: {sl_change_percent:.2%}",
                )
                new_sl = current_sl
            elif tp_change_percent > max_change:
                update_decision["reasons"].append(
                    f"TP change too large: {tp_change_percent:.2%}",
                )
                new_tp = current_tp
        else:
            # Standard limits for non-high leverage
            if tp_change_percent > self.max_target_change_percent:
                update_decision["reasons"].append(
                    f"TP change too large: {tp_change_percent:.2%}",
                )
                new_tp = current_tp

            if sl_change_percent > self.max_target_change_percent:
                update_decision["reasons"].append(
                    f"SL change too large: {sl_change_percent:.2%}",
                )
                new_sl = current_sl

        return new_tp, new_sl

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="risk management application",
    )
    def _apply_risk_management_rules(
        self,
        position: dict[str, Any],
        new_tp: float,
        new_sl: float,
        market_data: dict[str, Any],
        update_decision: dict[str, Any],
    ) -> tuple[float, float]:
        """Apply risk management rules to target updates."""
        direction = position.get("direction")
        entry_price = position.get("entry_price", 0)
        current_price = market_data.get("current_price", 0)

        if direction == "long":
            new_tp, new_sl = self._apply_long_position_rules(
                entry_price,
                current_price,
                new_tp,
                new_sl,
                update_decision,
            )
        else:  # short position
            new_tp, new_sl = self._apply_short_position_rules(
                entry_price,
                current_price,
                new_tp,
                new_sl,
                update_decision,
            )

        return new_tp, new_sl

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="long position risk management",
    )
    def _apply_long_position_rules(
        self,
        entry_price: float,
        current_price: float,
        new_tp: float,
        new_sl: float,
        update_decision: dict[str, Any],
    ) -> tuple[float, float]:
        """Apply risk management rules for long positions."""
        # Don't move SL too far from entry
        sl_distance_from_entry = abs(entry_price - new_sl) / entry_price
        if sl_distance_from_entry > self.max_sl_distance_from_entry:
            new_sl = entry_price * (1 - self.max_sl_distance_from_entry)
            update_decision["reasons"].append("SL capped due to risk limits")

        # Don't set TP too close to current price
        tp_distance_from_current = abs(current_price - new_tp) / current_price
        if tp_distance_from_current < self.min_tp_distance_from_current:
            new_tp = current_price * (1 + self.min_tp_distance_from_current)
            update_decision["reasons"].append("TP adjusted for minimum distance")

        return new_tp, new_sl

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="short position risk management",
    )
    def _apply_short_position_rules(
        self,
        entry_price: float,
        current_price: float,
        new_tp: float,
        new_sl: float,
        update_decision: dict[str, Any],
    ) -> tuple[float, float]:
        """Apply risk management rules for short positions."""
        # Don't move SL too far from entry
        sl_distance_from_entry = abs(new_sl - entry_price) / entry_price
        if sl_distance_from_entry > self.max_sl_distance_from_entry:
            new_sl = entry_price * (1 + self.max_sl_distance_from_entry)
            update_decision["reasons"].append("SL capped due to risk limits")

        # Don't set TP too close to current price
        tp_distance_from_current = abs(new_tp - current_price) / current_price
        if tp_distance_from_current < self.min_tp_distance_from_current:
            new_tp = current_price * (1 - self.min_tp_distance_from_current)
            update_decision["reasons"].append("TP adjusted for minimum distance")

        return new_tp, new_sl

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="update opportunity evaluation",
    )
    def _evaluate_update_opportunities(
        self,
        position: dict[str, Any],
        new_tp: float,
        new_sl: float,
        update_decision: dict[str, Any],
    ) -> None:
        """Evaluate specific update opportunities."""
        direction = position.get("direction")
        current_tp = position.get("take_profit", 0)
        current_sl = position.get("stop_loss", 0)

        if direction == "long":
            self._evaluate_long_position_updates(
                new_tp,
                new_sl,
                current_tp,
                current_sl,
                update_decision,
            )
        else:  # short position
            self._evaluate_short_position_updates(
                new_tp,
                new_sl,
                current_tp,
                current_sl,
                update_decision,
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="long position update evaluation",
    )
    def _evaluate_long_position_updates(
        self,
        new_tp: float,
        new_sl: float,
        current_tp: float,
        current_sl: float,
        update_decision: dict[str, Any],
    ) -> None:
        """Evaluate update opportunities for long positions."""
        # Trailing stop logic
        if self.trailing_stop_enabled and new_sl > current_sl:
            update_decision["update_sl"] = True
            update_decision["reasons"].append("Trailing stop adjustment")

        # Take profit improvement
        if self.enable_take_profit_updates and new_tp > current_tp:
            update_decision["update_tp"] = True
            update_decision["reasons"].append("TP improvement opportunity")

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="short position update evaluation",
    )
    def _evaluate_short_position_updates(
        self,
        new_tp: float,
        new_sl: float,
        current_tp: float,
        current_sl: float,
        update_decision: dict[str, Any],
    ) -> None:
        """Evaluate update opportunities for short positions."""
        # Trailing stop logic
        if self.trailing_stop_enabled and new_sl < current_sl:
            update_decision["update_sl"] = True
            update_decision["reasons"].append("Trailing stop adjustment")

        # Take profit improvement
        if self.enable_take_profit_updates and new_tp < current_tp:
            update_decision["update_tp"] = True
            update_decision["reasons"].append("TP improvement opportunity")

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="target update execution",
    )
    async def _execute_target_update(
        self,
        position: dict[str, Any],
        update_decision: dict[str, Any],
        ml_targets: dict[str, Any],
        market_data: dict[str, Any],
    ):
        """Execute the target update by modifying position orders."""
        try:
            trade_id = position.get("trade_id")
            # symbol = self.state_manager.get_state("trade_symbol", "BTCUSDT")

            updates_made = []

            # Update take profit if needed
            if update_decision.get("update_tp") and self.enable_take_profit_updates:
                new_tp = update_decision["new_tp"]

                # Cancel existing TP order and place new one
                # Note: This would need to be adapted based on your exchange's order management
                try:
                    # This is a simplified example - actual implementation would depend on order tracking
                    old_tp = position.get("take_profit")
                    position["take_profit"] = new_tp
                    updates_made.append(f"TP: {old_tp:.6f} -> {new_tp:.6f}")

                except Exception as e:
                    self.logger.error(f"Failed to update take profit: {e}")

            # Update stop loss if needed
            if update_decision.get("update_sl") and self.enable_stop_loss_updates:
                new_sl = update_decision["new_sl"]

                try:
                    old_sl = position.get("stop_loss")
                    position["stop_loss"] = new_sl
                    updates_made.append(f"SL: {old_sl:.6f} -> {new_sl:.6f}")

                except Exception as e:
                    self.logger.error(f"Failed to update stop loss: {e}")

            if updates_made:
                # Update position in state manager
                self.state_manager.set_state("current_position", position)

                # Log the update
                self.logger.info(
                    f"ML Target Update for {trade_id}: {', '.join(updates_made)}. "
                    f"Confidence: {update_decision['confidence']:.2f}. "
                    f"Reasons: {', '.join(update_decision['reasons'])}",
                )

                # Record in update history
                self._record_update_history(
                    trade_id,
                    update_decision,
                    ml_targets,
                    market_data,
                    updates_made,
                )

        except Exception as e:
            self.logger.error(f"Error executing target update: {e}", exc_info=True)

    @handle_errors(
        exceptions=(ValueError, AttributeError, TypeError),
        default_return=None,
        context="update history recording",
    )
    def _record_update_history(
        self,
        trade_id: str,
        update_decision: dict[str, Any],
        ml_targets: dict[str, Any],
        market_data: dict[str, Any],
        updates_made: list[str],
    ):
        """Record the update in history for analysis."""
        if trade_id not in self.update_history:
            self.update_history[trade_id] = []

        update_record = {
            "timestamp": datetime.now(),
            "market_price": market_data.get("current_price"),
            "ml_confidence": update_decision.get("confidence"),
            "updates_made": updates_made,
            "reasons": update_decision.get("reasons"),
            "tp_multiplier": ml_targets.get("tp_multiplier"),
            "sl_multiplier": ml_targets.get("sl_multiplier"),
        }

        self.update_history[trade_id].append(update_record)

        # Keep only last 50 updates per trade
        if len(self.update_history[trade_id]) > 50:
            self.update_history[trade_id] = self.update_history[trade_id][-50:]

    def get_update_statistics(self) -> dict[str, Any]:
        """Get statistics about target updates."""
        total_updates = sum(len(history) for history in self.update_history.values())

        if total_updates == 0:
            return {"total_updates": 0}

        all_updates = []
        for history in self.update_history.values():
            all_updates.extend(history)

        confidences = [update.get("ml_confidence", 0) for update in all_updates]
        avg_confidence = np.mean(confidences) if confidences else 0

        return {
            "total_updates": total_updates,
            "active_trades_monitored": len(self.update_history),
            "average_ml_confidence": avg_confidence,
            "last_update_time": max(self.last_update_time.values())
            if self.last_update_time
            else None,
            "is_monitoring": self.is_running,
        }
