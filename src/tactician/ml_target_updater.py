import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

from src.analyst.ml_dynamic_target_predictor import MLDynamicTargetPredictor
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
        config: Dict[str, Any],
    ):
        self.ml_target_predictor = ml_target_predictor
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.config = config.get("ml_target_updater", {})
        self.logger = system_logger.getChild("MLTargetUpdater")

        # Update configuration
        self.update_interval_seconds = self.config.get(
            "update_interval_seconds", 300
        )  # 5 minutes
        self.min_time_between_updates = self.config.get(
            "min_time_between_updates_seconds", 60
        )  # 1 minute
        self.confidence_threshold_for_update = self.config.get(
            "confidence_threshold_for_update", 0.6
        )
        self.max_target_change_percent = self.config.get(
            "max_target_change_percent", 0.25
        )  # 25% max change
        self.enable_stop_loss_updates = self.config.get(
            "enable_stop_loss_updates", True
        )
        self.enable_take_profit_updates = self.config.get(
            "enable_take_profit_updates", True
        )
        self.trailing_stop_enabled = self.config.get("trailing_stop_enabled", True)

        # Risk management settings
        self.max_sl_distance_from_entry = self.config.get(
            "max_sl_distance_from_entry", 0.05
        )  # 5%
        self.min_tp_distance_from_current = self.config.get(
            "min_tp_distance_from_current", 0.01
        )  # 1%

        # State tracking
        self.last_update_time = {}
        self.update_history = {}
        self.is_running = False

    async def start_monitoring(self):
        """Start the continuous target monitoring and updating process."""
        if self.is_running:
            self.logger.warning("Target updater is already running")
            return

        self.is_running = True
        self.logger.info("Started ML target updater monitoring")

        try:
            while self.is_running:
                await self._update_cycle()
                await asyncio.sleep(self.update_interval_seconds)
        except Exception as e:
            self.logger.error(
                f"Error in target updater monitoring loop: {e}", exc_info=True
            )
        finally:
            self.is_running = False

    def stop_monitoring(self):
        """Stop the target monitoring process."""
        self.is_running = False
        self.logger.info("Stopped ML target updater monitoring")

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

            if time.time() - last_update < self.min_time_between_updates:
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
                signal_type, current_market_data, current_position
            )

            if not ml_targets:
                return

            # Evaluate if targets should be updated
            update_decision = self._evaluate_target_update(
                current_position, ml_targets, current_market_data
            )

            if update_decision.get("should_update", False):
                await self._execute_target_update(
                    current_position, update_decision, ml_targets, current_market_data
                )

                # Record update time
                self.last_update_time[position_id] = time.time()

        except Exception as e:
            self.logger.error(f"Error in update cycle: {e}", exc_info=True)

    async def _get_current_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data needed for ML predictions."""
        try:
            # Get current price data
            ticker = await self.exchange.get_ticker(
                self.state_manager.get_state("trade_symbol", "BTCUSDT")
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

    def _determine_signal_type(self, position: Dict[str, Any]) -> Optional[str]:
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
        elif direction == "short":
            return "SR_BREAKOUT_SHORT"

        return None

    async def _get_ml_targets(
        self, signal_type: str, market_data: Dict[str, Any], position: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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
                }
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

    def _evaluate_target_update(
        self,
        position: Dict[str, Any],
        ml_targets: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate whether targets should be updated and how."""

        current_tp = position.get("take_profit", 0)
        current_sl = position.get("stop_loss", 0)
        entry_price = position.get("entry_price", 0)
        current_price = market_data.get("current_price", 0)
        direction = position.get("direction")

        new_tp = ml_targets.get("take_profit", current_tp)
        new_sl = ml_targets.get("stop_loss", current_sl)
        confidence = ml_targets.get("prediction_confidence", 0)

        update_decision = {
            "should_update": False,
            "update_tp": False,
            "update_sl": False,
            "new_tp": new_tp,
            "new_sl": new_sl,
            "confidence": confidence,
            "reasons": [],
        }

        # Check confidence threshold
        if confidence < self.confidence_threshold_for_update:
            update_decision["reasons"].append(f"Low confidence: {confidence:.2f}")
            return update_decision

        # Calculate percentage changes
        tp_change_percent = (
            abs(new_tp - current_tp) / current_tp if current_tp > 0 else 0
        )
        sl_change_percent = (
            abs(new_sl - current_sl) / current_sl if current_sl > 0 else 0
        )

        # Check if changes are within acceptable limits
        if tp_change_percent > self.max_target_change_percent:
            update_decision["reasons"].append(
                f"TP change too large: {tp_change_percent:.2%}"
            )
            new_tp = current_tp  # Don't update

        if sl_change_percent > self.max_target_change_percent:
            update_decision["reasons"].append(
                f"SL change too large: {sl_change_percent:.2%}"
            )
            new_sl = current_sl  # Don't update

        # Risk management checks
        if direction == "long":
            # For long positions
            sl_distance_from_entry = abs(entry_price - new_sl) / entry_price
            tp_distance_from_current = abs(current_price - new_tp) / current_price

            # Don't move SL too far from entry
            if sl_distance_from_entry > self.max_sl_distance_from_entry:
                new_sl = entry_price * (1 - self.max_sl_distance_from_entry)
                update_decision["reasons"].append("SL capped due to risk limits")

            # Don't set TP too close to current price
            if (
                new_tp > current_tp
                and tp_distance_from_current < self.min_tp_distance_from_current
            ):
                new_tp = current_price * (1 + self.min_tp_distance_from_current)
                update_decision["reasons"].append("TP adjusted for minimum distance")

            # Trailing stop logic
            if self.trailing_stop_enabled and new_sl > current_sl:
                update_decision["update_sl"] = True
                update_decision["reasons"].append("Trailing stop adjustment")

            # Take profit improvement
            if self.enable_take_profit_updates and new_tp > current_tp:
                update_decision["update_tp"] = True
                update_decision["reasons"].append("TP improvement opportunity")

        else:  # short position
            # Similar logic for short positions
            sl_distance_from_entry = abs(new_sl - entry_price) / entry_price
            tp_distance_from_current = abs(new_tp - current_price) / current_price

            if sl_distance_from_entry > self.max_sl_distance_from_entry:
                new_sl = entry_price * (1 + self.max_sl_distance_from_entry)
                update_decision["reasons"].append("SL capped due to risk limits")

            if (
                new_tp < current_tp
                and tp_distance_from_current < self.min_tp_distance_from_current
            ):
                new_tp = current_price * (1 - self.min_tp_distance_from_current)
                update_decision["reasons"].append("TP adjusted for minimum distance")

            if self.trailing_stop_enabled and new_sl < current_sl:
                update_decision["update_sl"] = True
                update_decision["reasons"].append("Trailing stop adjustment")

            if self.enable_take_profit_updates and new_tp < current_tp:
                update_decision["update_tp"] = True
                update_decision["reasons"].append("TP improvement opportunity")

        # Final decision
        if update_decision["update_tp"] or update_decision["update_sl"]:
            update_decision["should_update"] = True
            update_decision["new_tp"] = new_tp
            update_decision["new_sl"] = new_sl

        return update_decision

    async def _execute_target_update(
        self,
        position: Dict[str, Any],
        update_decision: Dict[str, Any],
        ml_targets: Dict[str, Any],
        market_data: Dict[str, Any],
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
                    f"Reasons: {', '.join(update_decision['reasons'])}"
                )

                # Record in update history
                self._record_update_history(
                    trade_id, update_decision, ml_targets, market_data, updates_made
                )

        except Exception as e:
            self.logger.error(f"Error executing target update: {e}", exc_info=True)

    def _record_update_history(
        self,
        trade_id: str,
        update_decision: Dict[str, Any],
        ml_targets: Dict[str, Any],
        market_data: Dict[str, Any],
        updates_made: List[str],
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

    def get_update_statistics(self) -> Dict[str, Any]:
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
