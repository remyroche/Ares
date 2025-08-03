# src/training/two_tier_decision_system.py

"""
Two-Tier Decision System for High Leverage Trading

Tier 1: All timeframes decide trade direction
Tier 2: Only 1m+5m decide exact entry timing
"""

from datetime import datetime
from typing import Any

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.training.multi_timeframe_training_manager import MultiTimeframeTrainingManager
from src.utils.logger import system_logger


class TwoTierDecisionSystem:
    """
    Two-tier decision system for high leverage trading.

    Tier 1: Ensemble decision using all timeframes (direction)
    Tier 2: Precision timing using only 1m+5m (exact entry)
    """

    def __init__(self, db_manager: SQLiteManager):
        self.db_manager = db_manager
        self.logger = system_logger.getChild("TwoTierDecisionSystem")

        # Configuration
        self.config = CONFIG.get(
            "TWO_TIER_DECISION",
            {
                "tier1_timeframes": [
                    "1m",
                    "5m",
                    "15m",
                    "1h",
                ],  # All timeframes for direction
                "tier2_timeframes": ["1m", "5m"],  # Only shortest for timing
                "direction_threshold": 0.7,  # Threshold for trade direction
                "timing_threshold": 0.8,  # Threshold for precise timing
                "high_leverage_mode": True,
            },
        )

        # Initialize managers
        self.mtf_manager = MultiTimeframeTrainingManager(db_manager)
        self.timing_manager = None  # Will be initialized for 1m+5m only

    async def train_two_tier_system(self, symbol: str) -> dict[str, Any]:
        """Train both tiers of the decision system."""
        self.logger.info("🚀 Training Two-Tier Decision System")

        # Tier 1: Train ensemble with all timeframes
        tier1_results = await self._train_tier1_direction_models(symbol)

        # Tier 2: Train timing models with 1m+5m only
        tier2_results = await self._train_tier2_timing_models(symbol)

        # Combine results
        results = {
            "tier1_direction": tier1_results,
            "tier2_timing": tier2_results,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info("✅ Two-tier system training completed")
        return results

    async def _train_tier1_direction_models(self, symbol: str) -> dict[str, Any]:
        """Train Tier 1: Direction models using all timeframes."""
        self.logger.info("🎯 Training Tier 1: Direction Models")

        # Use all timeframes for direction decision
        direction_timeframes = self.config["tier1_timeframes"]

        # Train ensemble with all timeframes
        direction_results = await self.mtf_manager.run_multi_timeframe_training(
            symbol=symbol,
            exchange_name=self.config.get("exchange_name", "BINANCE"),
            timeframes=direction_timeframes,
            enable_ensemble=True,
        )

        return {
            "timeframes": direction_timeframes,
            "ensemble_model": direction_results.get("ensemble_results", {}),
            "purpose": "Trade direction decision using all timeframes",
        }

    async def _train_tier2_timing_models(self, symbol: str) -> dict[str, Any]:
        """Train Tier 2: Timing models using only 1m+5m."""
        self.logger.info("⚡ Training Tier 2: Timing Models")

        # Use only shortest timeframes for precise timing
        timing_timeframes = self.config["tier2_timeframes"]

        # Create specialized timing manager
        timing_manager = MultiTimeframeTrainingManager(self.db_manager)

        # Configure for timing-specific training
        timing_config = {
            "enable_ensemble": True,
            "enable_cross_validation": True,
            "ensemble_method": "timing_optimized",
            "high_leverage_settings": {
                "prioritize_short_timeframes": True,
                "timing_precision": True,
                "entry_optimization": True,
            },
        }

        # Train timing models
        timing_results = await timing_manager.run_multi_timeframe_training(
            symbol=symbol,
            exchange_name=self.config.get("exchange_name", "BINANCE"),
            timeframes=timing_timeframes,
            enable_ensemble=True,
        )

        return {
            "timeframes": timing_timeframes,
            "ensemble_model": timing_results.get("ensemble_results", {}),
            "purpose": "Precise entry timing using 1m+5m only",
        }

    async def make_trading_decision(
        self,
        symbol: str,
        current_data: dict[str, Any],
        current_position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make trading decision using two-tier system (entry and exit)."""
        self.logger.info("🎯 Making Two-Tier Trading Decision")

        # Check if we have an open position for exit logic
        if current_position:
            return await self._make_exit_decision(
                symbol,
                current_data,
                current_position,
            )
        return await self._make_entry_decision(symbol, current_data)

    async def _make_entry_decision(
        self,
        symbol: str,
        current_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Make entry decision using two-tier system."""
        # Tier 1: Get direction decision from all timeframes
        direction_decision = await self._get_tier1_direction(current_data)

        # If direction is clear, get precise timing
        if direction_decision["should_trade"]:
            strategy = direction_decision["strategy"]
            timing_decision = await self._get_tier2_timing(current_data, strategy)

            final_decision = {
                "action": "ENTRY",
                "direction": direction_decision["direction"],
                "strategy": strategy,
                "confidence": direction_decision["confidence"],
                "timing_signal": timing_decision["timing_signal"],
                "should_execute": timing_decision["should_execute"],
                "position_size": self._calculate_position_size(
                    direction_decision,
                    timing_decision,
                ),
                "tier1_timeframes": self.config["tier1_timeframes"],
                "tier2_timeframes": self.config["tier2_timeframes"],
                "timestamp": datetime.now().isoformat(),
            }
        else:
            final_decision = {
                "action": "HOLD",
                "direction": "HOLD",
                "confidence": direction_decision["confidence"],
                "reason": "No clear direction from ensemble",
                "timestamp": datetime.now().isoformat(),
            }

        return final_decision

    async def _make_exit_decision(
        self,
        symbol: str,
        current_data: dict[str, Any],
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Make exit decision using two-tier system."""
        self.logger.info("🚪 Making Two-Tier Exit Decision")

        # Tier 1: Get exit direction from all timeframes
        exit_direction_decision = await self._get_tier1_exit_direction(
            current_data,
            current_position,
        )

        # If exit signal is clear, get precise exit timing
        if exit_direction_decision["should_exit"]:
            strategy = exit_direction_decision["strategy"]
            exit_timing_decision = await self._get_tier2_exit_timing(
                current_data,
                strategy,
            )

            final_decision = {
                "action": "EXIT",
                "exit_type": exit_direction_decision["exit_type"],
                "strategy": strategy,
                "confidence": exit_direction_decision["confidence"],
                "timing_signal": exit_timing_decision["timing_signal"],
                "should_execute": exit_timing_decision["should_execute"],
                "exit_reason": exit_direction_decision["exit_reason"],
                "tier1_timeframes": self.config["tier1_timeframes"],
                "tier2_timeframes": self.config["tier2_timeframes"],
                "timestamp": datetime.now().isoformat(),
            }
        else:
            final_decision = {
                "action": "HOLD_POSITION",
                "exit_type": "HOLD",
                "confidence": exit_direction_decision["confidence"],
                "reason": "No clear exit signal from ensemble",
                "timestamp": datetime.now().isoformat(),
            }

        return final_decision

    async def _get_tier1_direction(
        self,
        current_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Get trade direction and strategy classification from all timeframes."""
        # Simulate ensemble prediction from all timeframes
        # In practice, this would use the trained ensemble model

        ensemble_prediction = 0.65  # Example: 65% bullish
        threshold = self.config["direction_threshold"]

        # Strategy classification based on all timeframes
        strategy = self._classify_strategy(current_data, ensemble_prediction)

        if ensemble_prediction > threshold:
            direction = "LONG"
            should_trade = True
        elif ensemble_prediction < (1 - threshold):
            direction = "SHORT"
            should_trade = True
        else:
            direction = "HOLD"
            should_trade = False

        return {
            "direction": direction,
            "strategy": strategy,
            "confidence": ensemble_prediction,
            "should_trade": should_trade,
            "ensemble_prediction": ensemble_prediction,
        }

    def _classify_strategy(
        self,
        current_data: dict[str, Any],
        ensemble_prediction: float,
    ) -> str:
        """Classify market strategy based on all timeframes."""
        # This would analyze price action, indicators, and timeframe patterns

        # Simulate strategy classification
        price = current_data.get("price", 2000)
        sr_levels = current_data.get("sr_levels", [])

        # Enhanced SR Strategy - both breakouts and bounces
        if self._is_near_sr_level(price, sr_levels):
            sr_type = self._classify_sr_scenario(price, sr_levels, ensemble_prediction)
            return sr_type

        # Trend Strategies
        if ensemble_prediction > 0.7:
            return "BULLISH_TREND"
        if ensemble_prediction < 0.3:
            return "BEARISH_TREND"
        if 0.4 <= ensemble_prediction <= 0.6:
            return "SIDEWAYS_RANGE"

        # Momentum Strategy
        if ensemble_prediction > 0.8:
            return "MOMENTUM_BREAKOUT"

        return "WAIT"

    def _classify_sr_scenario(
        self,
        price: float,
        sr_levels: list,
        ensemble_prediction: float,
    ) -> str:
        """Classify SR scenario as breakout or bounce."""
        nearest_level = self._find_nearest_sr_level(price, sr_levels)

        if price > nearest_level:
            # Price above SR level - potential breakout
            if ensemble_prediction > 0.6:
                return "SR_BREAKOUT_UP"
            return "SR_BOUNCE_DOWN"
        # Price below SR level - potential bounce
        if ensemble_prediction > 0.6:
            return "SR_BOUNCE_UP"
        return "SR_BREAKOUT_DOWN"

    def _find_nearest_sr_level(self, price: float, sr_levels: list) -> float:
        """Find the nearest SR level to current price."""
        if not sr_levels:
            return price

        nearest = min(sr_levels, key=lambda x: abs(x - price))
        return nearest

    def _is_near_sr_level(self, price: float, sr_levels: list) -> bool:
        """Check if price is near support/resistance levels."""
        for level in sr_levels:
            if abs(price - level) / level < 0.01:  # Within 1%
                return True
        return False

    async def _get_tier2_timing(
        self,
        current_data: dict[str, Any],
        strategy: str,
    ) -> dict[str, Any]:
        """Get strategy-specific precise timing from 1m+5m only."""
        # Simulate timing prediction from 1m+5m
        # In practice, this would use the specialized timing model

        timing_signal = 0.85  # Example: 85% timing signal
        threshold = self.config["timing_threshold"]

        # Strategy-specific timing logic
        strategy_timing = self._get_strategy_specific_timing(strategy, timing_signal)

        should_execute = strategy_timing["should_execute"]

        return {
            "timing_signal": timing_signal,
            "should_execute": should_execute,
            "timing_confidence": timing_signal,
            "strategy_timing": strategy_timing,
        }

    def _get_strategy_specific_timing(
        self,
        strategy: str,
        timing_signal: float,
    ) -> dict[str, Any]:
        """Get timing decision based on specific strategy."""

        if strategy == "SR_BREAKOUT_UP":
            # Wait for strong upward breakout confirmation
            return {
                "should_execute": timing_signal > 0.8,
                "entry_type": "SR_BREAKOUT_UP",
                "timing_reason": "Waiting for upward SR breakout confirmation",
            }
        if strategy == "SR_BREAKOUT_DOWN":
            # Wait for strong downward breakout confirmation
            return {
                "should_execute": timing_signal < 0.2,
                "entry_type": "SR_BREAKOUT_DOWN",
                "timing_reason": "Waiting for downward SR breakout confirmation",
            }
        if strategy == "SR_BOUNCE_UP":
            # Wait for upward bounce confirmation
            return {
                "should_execute": timing_signal > 0.7,
                "entry_type": "SR_BOUNCE_UP",
                "timing_reason": "Waiting for upward SR bounce confirmation",
            }
        if strategy == "SR_BOUNCE_DOWN":
            # Wait for downward bounce confirmation
            return {
                "should_execute": timing_signal < 0.3,
                "entry_type": "SR_BOUNCE_DOWN",
                "timing_reason": "Waiting for downward SR bounce confirmation",
            }

        if strategy == "BULLISH_TREND":
            # Look for pullback entries in bullish trend
            return {
                "should_execute": timing_signal > 0.7,
                "entry_type": "BULLISH_PULLBACK",
                "timing_reason": "Looking for bullish pullback entry",
            }

        if strategy == "BEARISH_TREND":
            # Look for bounce entries in bearish trend
            return {
                "should_execute": timing_signal < 0.3,
                "entry_type": "BEARISH_BOUNCE",
                "timing_reason": "Looking for bearish bounce entry",
            }

        if strategy == "SIDEWAYS_RANGE":
            # Look for range breakout
            return {
                "should_execute": timing_signal > 0.8 or timing_signal < 0.2,
                "entry_type": "RANGE_BREAKOUT",
                "timing_reason": "Waiting for range breakout",
            }

        if strategy == "MOMENTUM_BREAKOUT":
            # Execute immediately on momentum
            return {
                "should_execute": timing_signal > 0.9,
                "entry_type": "MOMENTUM_BREAKOUT",
                "timing_reason": "Executing momentum breakout",
            }

        return {
            "should_execute": False,
            "entry_type": "WAIT",
            "timing_reason": "No clear strategy timing",
        }

    def _calculate_position_size(
        self,
        direction_decision: dict[str, Any],
        timing_decision: dict[str, Any],
    ) -> float:
        """Calculate position size based on both tiers."""
        base_size = 1.0

        # Adjust based on direction confidence
        direction_confidence = direction_decision["confidence"]
        if direction_confidence > 0.8:
            direction_multiplier = 1.2
        elif direction_confidence > 0.6:
            direction_multiplier = 1.0
        else:
            direction_multiplier = 0.8

        # Adjust based on timing precision
        timing_confidence = timing_decision["timing_signal"]
        if timing_confidence > 0.9:
            timing_multiplier = 1.3
        elif timing_confidence > 0.8:
            timing_multiplier = 1.1
        else:
            timing_multiplier = 0.9

        # Calculate final position size
        final_size = base_size * direction_multiplier * timing_multiplier

        return min(final_size, 1.5)  # Cap at 150% position size

    async def _get_tier1_exit_direction(
        self,
        current_data: dict[str, Any],
        current_position: dict[str, Any],
    ) -> dict[str, Any]:
        """Get exit direction from all timeframes."""
        # Simulate exit ensemble prediction
        # In practice, this would use the trained ensemble model

        exit_prediction = 0.75  # Example: 75% exit signal
        threshold = self.config["direction_threshold"]

        # Exit strategy classification
        exit_strategy = self._classify_exit_strategy(
            current_data,
            current_position,
            exit_prediction,
        )

        if exit_prediction > threshold:
            exit_type = "TAKE_PROFIT"
            should_exit = True
        elif exit_prediction < (1 - threshold):
            exit_type = "STOP_LOSS"
            should_exit = True
        else:
            exit_type = "HOLD"
            should_exit = False

        return {
            "exit_type": exit_type,
            "strategy": exit_strategy,
            "confidence": exit_prediction,
            "should_exit": should_exit,
            "exit_prediction": exit_prediction,
            "exit_reason": self._get_exit_reason(exit_strategy, exit_prediction),
        }

    def _classify_exit_strategy(
        self,
        current_data: dict[str, Any],
        current_position: dict[str, Any],
        exit_prediction: float,
    ) -> str:
        """Classify exit strategy based on position and market conditions."""
        position_type = current_position.get("type", "LONG")
        entry_strategy = current_position.get("entry_strategy", "UNKNOWN")

        # Exit based on entry strategy
        if entry_strategy.startswith("SR_"):
            if exit_prediction > 0.8:
                return "SR_TAKE_PROFIT"
            if exit_prediction < 0.2:
                return "SR_STOP_LOSS"
            return "SR_HOLD"

        if entry_strategy in ["BULLISH_TREND", "BEARISH_TREND"]:
            if exit_prediction > 0.7:
                return "TREND_TAKE_PROFIT"
            if exit_prediction < 0.3:
                return "TREND_STOP_LOSS"
            return "TREND_HOLD"

        if entry_strategy == "MOMENTUM_BREAKOUT":
            if exit_prediction > 0.6:
                return "MOMENTUM_TAKE_PROFIT"
            if exit_prediction < 0.4:
                return "MOMENTUM_STOP_LOSS"
            return "MOMENTUM_HOLD"

        return "GENERAL_EXIT"

    def _get_exit_reason(self, exit_strategy: str, exit_prediction: float) -> str:
        """Get human-readable exit reason."""
        if "TAKE_PROFIT" in exit_strategy:
            return f"Take profit signal ({exit_prediction:.1%} confidence)"
        if "STOP_LOSS" in exit_strategy:
            return f"Stop loss signal ({exit_prediction:.1%} confidence)"
        return f"Hold position ({exit_prediction:.1%} confidence)"

    async def _get_tier2_exit_timing(
        self,
        current_data: dict[str, Any],
        exit_strategy: str,
    ) -> dict[str, Any]:
        """Get strategy-specific exit timing from 1m+5m only."""
        # Simulate exit timing prediction from 1m+5m
        timing_signal = 0.85  # Example: 85% timing signal
        threshold = self.config["timing_threshold"]

        # Strategy-specific exit timing logic
        strategy_timing = self._get_exit_strategy_specific_timing(
            exit_strategy,
            timing_signal,
        )

        should_execute = strategy_timing["should_execute"]

        return {
            "timing_signal": timing_signal,
            "should_execute": should_execute,
            "timing_confidence": timing_signal,
            "strategy_timing": strategy_timing,
        }

    def _get_exit_strategy_specific_timing(
        self,
        exit_strategy: str,
        timing_signal: float,
    ) -> dict[str, Any]:
        """Get exit timing decision based on specific strategy."""

        if "TAKE_PROFIT" in exit_strategy:
            # Execute take profit immediately on strong signal
            return {
                "should_execute": timing_signal > 0.7,
                "exit_type": "TAKE_PROFIT",
                "timing_reason": "Executing take profit on strong signal",
            }

        if "STOP_LOSS" in exit_strategy:
            # Execute stop loss immediately on weak signal
            return {
                "should_execute": timing_signal < 0.3,
                "exit_type": "STOP_LOSS",
                "timing_reason": "Executing stop loss on weak signal",
            }

        if "SR_" in exit_strategy:
            # SR exits need strong confirmation
            return {
                "should_execute": timing_signal > 0.8 or timing_signal < 0.2,
                "exit_type": "SR_EXIT",
                "timing_reason": "Waiting for SR exit confirmation",
            }

        if "MOMENTUM_" in exit_strategy:
            # Momentum exits are quick
            return {
                "should_execute": timing_signal > 0.6 or timing_signal < 0.4,
                "exit_type": "MOMENTUM_EXIT",
                "timing_reason": "Executing momentum exit",
            }

        return {
            "should_execute": False,
            "exit_type": "HOLD",
            "timing_reason": "No clear exit timing",
        }

    def get_system_info(self) -> dict[str, Any]:
        """Get information about the two-tier system."""
        return {
            "tier1_timeframes": self.config["tier1_timeframes"],
            "tier2_timeframes": self.config["tier2_timeframes"],
            "direction_threshold": self.config["direction_threshold"],
            "timing_threshold": self.config["timing_threshold"],
            "high_leverage_mode": self.config["high_leverage_mode"],
            "description": "Two-tier decision system for high leverage trading",
        }
