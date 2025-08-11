# src/analyst/predictive_ensembles/two_tier_integration.py

"""
Two-Tier Integration Layer

This integrates two-tier decision logic into the existing ensemble system
without replacing the current confidence levels and liquidation risk calculations.
"""

import time
from typing import Any

from src.config import CONFIG
from src.utils.logger import system_logger


class TwoTierIntegration:
    """
    Integrates two-tier decision logic into existing ensemble system.

    Tier 1: Uses existing ensemble models for direction/strategy
    Tier 2: Uses 1m+5m for precise timing
    """

    def __init__(self):
        self.logger = system_logger.getChild("TwoTierIntegration")

        # Configuration
        self.config = CONFIG.get(
            "TWO_TIER_INTEGRATION",
            {
                "enable_two_tier": True,
                "tier1_timeframes": [
                    "1m",
                    "5m",
                    "15m",
                    "1h",
                ],  # All timeframes for direction
                "tier2_timeframes": ["1m", "5m"],  # Only shortest for timing
                "direction_threshold": 0.7,
                "timing_threshold": 0.8,
                "high_leverage_mode": True,
            },
        )

        # Log initialization
        self.logger.info("🚀 Initializing TwoTierIntegration")
        self.logger.info(f"📊 Tier 1 timeframes: {self.config['tier1_timeframes']}")
        self.logger.info(f"⏰ Tier 2 timeframes: {self.config['tier2_timeframes']}")
        self.logger.info(
            f"🎯 Direction threshold: {self.config['direction_threshold']}",
        )
        self.logger.info(f"⚡ Timing threshold: {self.config['timing_threshold']}")
        self.logger.info(f"🔥 High leverage mode: {self.config['high_leverage_mode']}")

    def enhance_ensemble_prediction(
        self,
        ensemble_prediction: dict[str, Any],
        current_data: dict[str, Any],
        current_position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Enhance existing ensemble prediction with two-tier logic.

        Args:
            ensemble_prediction: Existing ensemble prediction with confidence
            current_data: Current market data
            current_position: Current position (if any)

        Returns:
            Enhanced prediction with two-tier timing
        """
        start_time = time.time()

        if not self.config["enable_two_tier"]:
            self.logger.info(
                "⚠️ Two-tier integration disabled, returning original prediction",
            )
            return ensemble_prediction

        self.logger.info("🎯 Enhancing ensemble prediction with two-tier logic...")

        # Extract existing ensemble info
        base_prediction = ensemble_prediction.get("prediction", "HOLD")
        base_confidence = ensemble_prediction.get("confidence", 0.5)
        regime = ensemble_prediction.get("regime", "UNKNOWN")

        self.logger.info(
            f"📊 Base prediction: {base_prediction} (confidence: {base_confidence:.3f})",
        )
        self.logger.info(f"📈 Regime: {regime}")

        # Tier 1: Use existing ensemble for direction/strategy
        self.logger.info("🔧 Processing Tier 1 (direction/strategy)...")
        tier1_result = self._get_tier1_from_ensemble(
            base_prediction,
            base_confidence,
            regime,
            current_data,
        )

        self.logger.info(
            f"📊 Tier 1 result: {tier1_result['direction']} ({tier1_result['strategy']})",
        )
        self.logger.info(f"🎯 Should trade: {tier1_result['should_trade']}")

        # Tier 2: Add precise timing from 1m+5m
        if tier1_result["should_trade"]:
            self.logger.info("⏰ Processing Tier 2 (precise timing)...")
            tier2_result = self._get_tier2_timing(
                current_data,
                tier1_result["strategy"],
            )

            self.logger.info(
                f"📊 Tier 2 result: timing signal {tier2_result['timing_signal']:.3f}",
            )
            self.logger.info(f"⚡ Should execute: {tier2_result['should_execute']}")

            # Combine with existing ensemble prediction
            enhanced_prediction = {
                **ensemble_prediction,  # Keep all existing ensemble data
                "two_tier_enhanced": True,
                "tier1_direction": tier1_result["direction"],
                "tier1_strategy": tier1_result["strategy"],
                "tier1_confidence": tier1_result["confidence"],
                "tier2_timing_signal": tier2_result["timing_signal"],
                "tier2_should_execute": tier2_result["should_execute"],
                "final_decision": self._combine_tier_decisions(
                    tier1_result,
                    tier2_result,
                ),
                "position_size_multiplier": self._calculate_position_multiplier(
                    tier1_result,
                    tier2_result,
                ),
                "liquidation_risk_adjustment": self._calculate_risk_adjustment(
                    tier1_result,
                    tier2_result,
                ),
            }

            self.logger.info(
                f"🎯 Final decision: {enhanced_prediction['final_decision']}",
            )
            self.logger.info(
                f"📈 Position multiplier: {enhanced_prediction['position_size_multiplier']:.2f}",
            )

            risk_adj = enhanced_prediction["liquidation_risk_adjustment"]
            self.logger.info(f"🛡️ Risk multiplier: {risk_adj['risk_multiplier']:.2f}")
            self.logger.info(f"📊 Confidence level: {risk_adj['confidence_level']:.3f}")

        else:
            self.logger.info("⏸️ No clear direction from Tier 1, holding position")
            enhanced_prediction = {
                **ensemble_prediction,
                "two_tier_enhanced": True,
                "final_decision": "HOLD",
                "reason": "No clear direction from ensemble",
            }

        processing_time = time.time() - start_time
        self.logger.info(f"✅ Two-tier enhancement completed in {processing_time:.3f}s")

        return enhanced_prediction

    def _get_tier1_from_ensemble(
        self,
        base_prediction: str,
        base_confidence: float,
        regime: str,
        current_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract Tier 1 direction from existing ensemble prediction."""

        self.logger.debug("🔧 Extracting Tier 1 direction from ensemble...")

        # Use existing ensemble confidence as Tier 1 confidence
        threshold = self.config["direction_threshold"]

        if base_confidence > threshold:
            direction = "LONG" if base_prediction == "BUY" else "SHORT"
            should_trade = True
            self.logger.debug(
                f"📊 High confidence ({base_confidence:.3f} > {threshold}), direction: {direction}",
            )
        elif base_confidence < (1 - threshold):
            direction = "SHORT" if base_prediction == "BUY" else "LONG"
            should_trade = True
            self.logger.debug(
                f"📊 Low confidence ({base_confidence:.3f} < {1-threshold}), direction: {direction}",
            )
        else:
            direction = "HOLD"
            should_trade = False
            self.logger.debug(f"📊 Medium confidence ({base_confidence:.3f}), holding")

        # Classify strategy based on regime and prediction
        strategy = self._classify_strategy_from_regime(
            regime,
            base_prediction,
            current_data,
        )

        self.logger.debug(f"📊 Strategy classified as: {strategy}")

        return {
            "direction": direction,
            "strategy": strategy,
            "confidence": base_confidence,
            "should_trade": should_trade,
            "regime": regime,
            "base_prediction": base_prediction,
        }

    def _classify_strategy_from_regime(
        self,
        regime: str,
        prediction: str,
        current_data: dict[str, Any],
    ) -> str:
        """Classify strategy based on existing regime classification."""

        self.logger.debug(
            f"🔧 Classifying strategy from regime: {regime}, prediction: {prediction}",
        )

        # Map regimes to strategies
        regime_strategy_map = {
            "BULL_TREND": "BULLISH_TREND",
            "BEAR_TREND": "BEARISH_TREND",
            "SIDEWAYS_RANGE": "SIDEWAYS_RANGE",
            "SR_ZONE_ACTION": "SR_BREAKOUT_UP",  # Will be refined below
            "HIGH_IMPACT_CANDLE": "MOMENTUM_BREAKOUT",
        }

        base_strategy = regime_strategy_map.get(regime, "GENERAL")

        # Refine SR strategy based on price action
        if regime == "SR_ZONE_ACTION":
            price = current_data.get("price", 0)
            sr_levels = current_data.get("sr_levels", [])
            if sr_levels:
                nearest_level = min(sr_levels, key=lambda x: abs(x - price))
                if price > nearest_level:
                    base_strategy = (
                        "SR_BREAKOUT_UP" if prediction == "BUY" else "SR_BOUNCE_DOWN"
                    )
                else:
                    base_strategy = (
                        "SR_BOUNCE_UP" if prediction == "BUY" else "SR_BREAKOUT_DOWN"
                    )

                self.logger.debug(
                    f"📊 SR strategy refined: price {price}, nearest level {nearest_level}, strategy: {base_strategy}",
                )

        return base_strategy

    def _get_tier2_timing(
        self,
        current_data: dict[str, Any],
        strategy: str,
    ) -> dict[str, Any]:
        """Get Tier 2 timing from 1m+5m data."""

        self.logger.debug(f"⏰ Getting Tier 2 timing for strategy: {strategy}")

        # This would use actual 1m+5m model predictions
        # For now, simulate based on strategy

        timing_signal = self._simulate_timing_signal(strategy, current_data)
        self.config["timing_threshold"]

        self.logger.debug(f"📊 Timing signal: {timing_signal:.3f}")

        strategy_timing = self._get_strategy_specific_timing(strategy, timing_signal)

        self.logger.debug(
            f"📊 Strategy timing: {strategy_timing['entry_type']} - {strategy_timing['timing_reason']}",
        )

        return {
            "timing_signal": timing_signal,
            "should_execute": strategy_timing["should_execute"],
            "timing_confidence": timing_signal,
            "strategy_timing": strategy_timing,
        }

    def _simulate_timing_signal(
        self,
        strategy: str,
        current_data: dict[str, Any],
    ) -> float:
        """Simulate timing signal based on strategy."""

        self.logger.debug(f"🔧 Simulating timing signal for strategy: {strategy}")

        # In practice, this would use actual 1m+5m model predictions

        base_signal = 0.75  # Base timing signal

        # Adjust based on strategy
        if "SR_" in strategy:
            base_signal += 0.1  # SR strategies need stronger confirmation
            self.logger.debug("📊 SR strategy adjustment: +0.1")
        elif "MOMENTUM" in strategy:
            base_signal += 0.15  # Momentum strategies are more sensitive
            self.logger.debug("📊 Momentum strategy adjustment: +0.15")
        elif "TREND" in strategy:
            base_signal += 0.05  # Trend strategies are more patient
            self.logger.debug("📊 Trend strategy adjustment: +0.05")

        final_signal = min(1.0, base_signal)
        self.logger.debug(f"📊 Final timing signal: {final_signal:.3f}")

        return final_signal

    def _get_strategy_specific_timing(
        self,
        strategy: str,
        timing_signal: float,
    ) -> dict[str, Any]:
        """Get strategy-specific timing decision."""

        self.logger.debug(f"🔧 Getting strategy-specific timing for: {strategy}")

        if strategy == "SR_BREAKOUT_UP":
            should_execute = timing_signal > 0.8
            result = {
                "should_execute": should_execute,
                "entry_type": "SR_BREAKOUT_UP",
                "timing_reason": "Waiting for upward SR breakout confirmation",
            }
        elif strategy == "SR_BREAKOUT_DOWN":
            should_execute = timing_signal < 0.2
            result = {
                "should_execute": should_execute,
                "entry_type": "SR_BREAKOUT_DOWN",
                "timing_reason": "Waiting for downward SR breakout confirmation",
            }
        elif strategy == "SR_BOUNCE_UP":
            should_execute = timing_signal > 0.7
            result = {
                "should_execute": should_execute,
                "entry_type": "SR_BOUNCE_UP",
                "timing_reason": "Waiting for upward SR bounce confirmation",
            }
        elif strategy == "SR_BOUNCE_DOWN":
            should_execute = timing_signal < 0.3
            result = {
                "should_execute": should_execute,
                "entry_type": "SR_BOUNCE_DOWN",
                "timing_reason": "Waiting for downward SR bounce confirmation",
            }
        elif strategy == "BULLISH_TREND":
            should_execute = timing_signal > 0.7
            result = {
                "should_execute": should_execute,
                "entry_type": "BULLISH_PULLBACK",
                "timing_reason": "Looking for bullish pullback entry",
            }
        elif strategy == "BEARISH_TREND":
            should_execute = timing_signal < 0.3
            result = {
                "should_execute": should_execute,
                "entry_type": "BEARISH_BOUNCE",
                "timing_reason": "Looking for bearish bounce entry",
            }
        elif strategy == "SIDEWAYS_RANGE":
            should_execute = timing_signal > 0.8 or timing_signal < 0.2
            result = {
                "should_execute": should_execute,
                "entry_type": "RANGE_BREAKOUT",
                "timing_reason": "Waiting for range breakout",
            }
        elif strategy == "MOMENTUM_BREAKOUT":
            should_execute = timing_signal > 0.9
            result = {
                "should_execute": should_execute,
                "entry_type": "MOMENTUM_BREAKOUT",
                "timing_reason": "Executing momentum breakout",
            }
        else:
            should_execute = False
            result = {
                "should_execute": should_execute,
                "entry_type": "WAIT",
                "timing_reason": "No clear strategy timing",
            }

        self.logger.debug(
            f"📊 Strategy timing result: {result['entry_type']} - {result['timing_reason']}",
        )
        self.logger.debug(f"⚡ Should execute: {should_execute}")

        return result

    def _combine_tier_decisions(
        self,
        tier1_result: dict[str, Any],
        tier2_result: dict[str, Any],
    ) -> str:
        """Combine Tier 1 and Tier 2 decisions."""

        self.logger.debug("🔧 Combining Tier 1 and Tier 2 decisions...")

        if not tier1_result["should_trade"]:
            self.logger.debug("📊 Tier 1 says no trade, final decision: HOLD")
            return "HOLD"

        if not tier2_result["should_execute"]:
            self.logger.debug(
                "📊 Tier 2 says wait for timing, final decision: WAIT_FOR_TIMING",
            )
            return "WAIT_FOR_TIMING"

        final_decision = tier1_result["direction"]
        self.logger.debug(f"📊 Both tiers agree, final decision: {final_decision}")

        return final_decision

    def _calculate_position_multiplier(
        self,
        tier1_result: dict[str, Any],
        tier2_result: dict[str, Any],
    ) -> float:
        """Calculate position size multiplier based on two-tier confidence."""

        self.logger.debug("🔧 Calculating position size multiplier...")

        tier1_confidence = tier1_result["confidence"]
        tier2_confidence = tier2_result["timing_confidence"]

        self.logger.debug(f"📊 Tier 1 confidence: {tier1_confidence:.3f}")
        self.logger.debug(f"📊 Tier 2 confidence: {tier2_confidence:.3f}")

        # Base multiplier
        base_multiplier = 1.0

        # Adjust based on Tier 1 confidence
        if tier1_confidence > 0.8:
            tier1_multiplier = 1.2
        elif tier1_confidence > 0.6:
            tier1_multiplier = 1.0
        else:
            tier1_multiplier = 0.8

        # Adjust based on Tier 2 timing
        if tier2_confidence > 0.9:
            tier2_multiplier = 1.3
        elif tier2_confidence > 0.8:
            tier2_multiplier = 1.1
        else:
            tier2_multiplier = 0.9

        final_multiplier = base_multiplier * tier1_multiplier * tier2_multiplier
        final_multiplier = min(final_multiplier, 1.5)  # Cap at 150%

        self.logger.debug(f"📊 Tier 1 multiplier: {tier1_multiplier:.2f}")
        self.logger.debug(f"📊 Tier 2 multiplier: {tier2_multiplier:.2f}")
        self.logger.debug(f"📊 Final multiplier: {final_multiplier:.2f}")

        return final_multiplier

    def _calculate_risk_adjustment(
        self,
        tier1_result: dict[str, Any],
        tier2_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate liquidation risk adjustment based on two-tier analysis."""

        self.logger.debug("🔧 Calculating liquidation risk adjustment...")

        # Higher confidence = lower risk
        tier1_confidence = tier1_result["confidence"]
        tier2_confidence = tier2_result["timing_confidence"]

        # Risk reduction based on confidence
        risk_reduction = (tier1_confidence + tier2_confidence) / 2

        risk_multiplier = 1.0 - (risk_reduction * 0.3)  # Reduce risk by up to 30%

        self.logger.debug(f"📊 Tier 1 confidence: {tier1_confidence:.3f}")
        self.logger.debug(f"📊 Tier 2 confidence: {tier2_confidence:.3f}")
        self.logger.debug(f"📊 Average confidence: {risk_reduction:.3f}")
        self.logger.debug(f"📊 Risk reduction: {risk_reduction * 0.3:.3f}")
        self.logger.debug(f"📊 Risk multiplier: {risk_multiplier:.3f}")

        return {
            "risk_multiplier": risk_multiplier,
            "confidence_level": risk_reduction,
            "strategy": tier1_result["strategy"],
            "timing_precision": tier2_confidence,
        }
