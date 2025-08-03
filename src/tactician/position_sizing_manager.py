# src/tactician/position_sizing_manager.py

from datetime import datetime
from typing import Any

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class PositionSizingManager:
    """
    Manages position sizing based on confidence scores, volatility, and risk management rules.
    Supports multiple positions for the same signal when confidence is very high.
    """

    def __init__(self, config: dict[str, Any], state_manager=None):
        self.config = config
        self.state_manager = state_manager
        self.risk_config = config.get("risk_management", {})
        self.position_config = self.risk_config.get("position_sizing", {})
        self.dynamic_config = self.risk_config.get("dynamic_risk_management", {})
        self.logger = system_logger.getChild("PositionSizingManager")

        # Track position history for successive position management
        self.position_history = []
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=0.0,
        context="position size calculation",
    )
    def calculate_position_size(
        self,
        current_price: float,
        stop_loss_price: float,
        leverage: int,
        confidence: float = 0.0,
        market_conditions: dict[str, Any] = None,
        portfolio_value: float = None,
        existing_positions: list[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Calculate optimal position size based on confidence and risk rules.

        Args:
            current_price: Current asset price
            stop_loss_price: Stop loss price
            leverage: Leverage to use
            confidence: ML confidence score (0.0 to 1.0)
            market_conditions: Market data including volatility indicators
            portfolio_value: Total portfolio value
            existing_positions: List of existing positions for this asset

        Returns:
            Dict containing position size and metadata
        """
        try:
            # Get portfolio value if not provided
            if portfolio_value is None:
                portfolio_value = (
                    self.state_manager.get_state("account_equity", 10000)
                    if self.state_manager
                    else 10000
                )

            # Get base position size
            base_size = self._get_base_position_size(portfolio_value)

            # Apply confidence-based scaling
            confidence_size = self._apply_confidence_scaling(base_size, confidence)

            # Apply volatility adjustment
            volatility_size = self._apply_volatility_adjustment(
                confidence_size,
                market_conditions,
            )

            # Apply regime-based adjustment
            regime_size = self._apply_regime_adjustment(
                volatility_size,
                market_conditions,
            )

            # Apply dynamic risk management adjustments
            risk_adjusted_size = self._apply_dynamic_risk_adjustment(regime_size)

            # Check if we can add successive positions
            successive_positions = self._check_successive_positions(
                confidence,
                market_conditions,
                existing_positions,
            )

            # Calculate final position size
            final_size = self._calculate_final_position_size(
                risk_adjusted_size,
                successive_positions,
            )

            # Validate against risk limits
            validated_size = self._validate_risk_limits(
                final_size,
                portfolio_value,
                existing_positions,
            )

            return {
                "position_size": validated_size,
                "confidence_score": confidence,
                "base_size": base_size,
                "confidence_multiplier": self._get_confidence_multiplier(confidence),
                "volatility_multiplier": self._get_volatility_multiplier(
                    market_conditions,
                ),
                "regime_multiplier": self._get_regime_multiplier(market_conditions),
                "risk_multiplier": self._get_risk_multiplier(),
                "successive_positions_allowed": successive_positions > 0,
                "total_exposure_after": self.current_exposure + validated_size,
                "calculation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return {"position_size": 0.0, "error": str(e)}

    def _get_base_position_size(self, portfolio_value: float) -> float:
        """Get base position size as fraction of portfolio."""
        base_size_fraction = self.position_config.get("base_position_size", 0.05)
        return portfolio_value * base_size_fraction

    def _apply_confidence_scaling(self, base_size: float, confidence: float) -> float:
        """Apply confidence-based position size scaling."""
        if not self.position_config.get("confidence_based_scaling", True):
            return base_size

        thresholds = self.position_config.get("confidence_thresholds", {})
        multipliers = self.position_config.get("position_size_multipliers", {})

        # Determine confidence level
        if confidence >= thresholds.get("very_high_confidence", 0.95):
            multiplier = multipliers.get("very_high_confidence", 2.0)
        elif confidence >= thresholds.get("high_confidence", 0.85):
            multiplier = multipliers.get("high_confidence", 1.5)
        elif confidence >= thresholds.get("medium_confidence", 0.75):
            multiplier = multipliers.get("medium_confidence", 1.0)
        else:
            multiplier = multipliers.get("low_confidence", 0.5)

        return base_size * multiplier

    def _apply_volatility_adjustment(
        self,
        position_size: float,
        market_conditions: dict[str, Any],
    ) -> float:
        """Apply volatility-based position size adjustment."""
        if not self.position_config.get("volatility_adjustment", {}).get(
            "enable_volatility_scaling",
            True,
        ):
            return position_size

        # Get ATR or volatility measure
        atr = market_conditions.get("atr", 0.0) if market_conditions else 0.0
        current_price = (
            market_conditions.get("current_price", 1.0) if market_conditions else 1.0
        )
        atr_ratio = atr / current_price if current_price > 0 else 0.0

        thresholds = self.position_config.get("volatility_adjustment", {}).get(
            "volatility_thresholds",
            {},
        )
        multipliers = self.position_config.get("volatility_adjustment", {}).get(
            "volatility_multipliers",
            {},
        )

        # Determine volatility level
        if atr_ratio <= thresholds.get("low_volatility", 0.02):
            multiplier = multipliers.get("low_volatility", 1.2)
        elif atr_ratio <= thresholds.get("medium_volatility", 0.05):
            multiplier = multipliers.get("medium_volatility", 1.0)
        else:
            multiplier = multipliers.get("high_volatility", 0.7)

        return position_size * multiplier

    def _apply_regime_adjustment(
        self,
        position_size: float,
        market_conditions: dict[str, Any],
    ) -> float:
        """Apply regime-based position size adjustment."""
        if not self.position_config.get("regime_based_adjustment", {}).get(
            "enable_regime_adjustment",
            True,
        ):
            return position_size

        regime = (
            market_conditions.get("market_regime", "SIDEWAYS_RANGE")
            if market_conditions
            else "SIDEWAYS_RANGE"
        )
        multipliers = self.position_config.get("regime_based_adjustment", {}).get(
            "regime_multipliers",
            {},
        )

        multiplier = multipliers.get(regime, 1.0)
        return position_size * multiplier

    def _apply_dynamic_risk_adjustment(self, position_size: float) -> float:
        """Apply dynamic risk management adjustments based on current performance."""
        if not self.dynamic_config.get("enable_dynamic_risk", True):
            return position_size

        # Apply drawdown-based adjustment
        if self.dynamic_config.get("drawdown_adjustment", {}).get(
            "enable_drawdown_scaling",
            True,
        ):
            position_size = self._apply_drawdown_adjustment(position_size)

        # Apply daily loss-based adjustment
        if self.dynamic_config.get("daily_loss_adjustment", {}).get(
            "enable_daily_loss_scaling",
            True,
        ):
            position_size = self._apply_daily_loss_adjustment(position_size)

        return position_size

    def _apply_drawdown_adjustment(self, position_size: float) -> float:
        """Apply drawdown-based position size reduction."""
        thresholds = self.dynamic_config.get("drawdown_adjustment", {}).get(
            "drawdown_thresholds",
            {},
        )
        reduction_factors = self.dynamic_config.get("drawdown_adjustment", {}).get(
            "size_reduction_factors",
            {},
        )

        if self.max_drawdown >= thresholds.get("emergency", 0.4):
            return position_size * reduction_factors.get("emergency", 0.2)
        if self.max_drawdown >= thresholds.get("aggressive", 0.3):
            return position_size * reduction_factors.get("aggressive", 0.5)
        if self.max_drawdown >= thresholds.get("reduction", 0.2):
            return position_size * reduction_factors.get("reduction", 0.7)
        if self.max_drawdown >= thresholds.get("warning", 0.1):
            return position_size * reduction_factors.get("warning", 0.9)

        return position_size

    def _apply_daily_loss_adjustment(self, position_size: float) -> float:
        """Apply daily loss-based position size reduction."""
        thresholds = self.dynamic_config.get("daily_loss_adjustment", {}).get(
            "daily_loss_thresholds",
            {},
        )
        reduction_factors = self.dynamic_config.get("daily_loss_adjustment", {}).get(
            "size_reduction_factors",
            {},
        )

        if self.daily_pnl <= -thresholds.get("emergency", 0.10):
            return position_size * reduction_factors.get("emergency", 0.2)
        if self.daily_pnl <= -thresholds.get("reduction", 0.08):
            return position_size * reduction_factors.get("reduction", 0.5)
        if self.daily_pnl <= -thresholds.get("warning", 0.05):
            return position_size * reduction_factors.get("warning", 0.8)

        return position_size

    def _check_successive_positions(
        self,
        confidence: float,
        market_conditions: dict[str, Any],
        existing_positions: list[dict[str, Any]] = None,
    ) -> int:
        """Check if we can add successive positions for high confidence signals."""
        if not self.position_config.get("successive_position_rules", {}).get(
            "enable_successive_positions",
            True,
        ):
            return 0

        min_confidence = self.position_config.get("successive_position_rules", {}).get(
            "min_confidence_for_successive",
            0.85,
        )
        max_positions = self.position_config.get("successive_position_rules", {}).get(
            "max_successive_positions",
            3,
        )

        if confidence < min_confidence:
            return 0

        # Count existing positions for this signal type
        existing_count = 0
        if existing_positions:
            # This would need to be implemented based on your position tracking system
            existing_count = len(existing_positions)

        return max(0, max_positions - existing_count)

    def _calculate_final_position_size(
        self,
        base_size: float,
        successive_positions: int,
    ) -> float:
        """Calculate final position size including successive positions."""
        if successive_positions == 0:
            return base_size

        # Calculate size for successive positions
        reduction_factor = self.position_config.get(
            "successive_position_rules",
            {},
        ).get("size_reduction_factor", 0.8)
        successive_size = base_size * (reduction_factor**successive_positions)

        return base_size + successive_size

    def _validate_risk_limits(
        self,
        position_size: float,
        portfolio_value: float,
        existing_positions: list[dict[str, Any]] = None,
    ) -> float:
        """Validate position size against risk limits."""
        risk_limits = self.position_config.get("risk_limits", {})

        # Check single position limit
        max_single = portfolio_value * risk_limits.get("max_single_position", 0.15)
        position_size = min(position_size, max_single)

        # Check total exposure limit
        total_exposure = self.current_exposure + position_size
        max_total = portfolio_value * risk_limits.get("max_total_exposure", 0.3)
        if total_exposure > max_total:
            position_size = max(0, max_total - self.current_exposure)

        # Check minimum position size
        min_size = portfolio_value * risk_limits.get("min_position_size", 0.01)
        if position_size < min_size:
            position_size = 0.0  # Don't take position if too small

        return position_size

    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get confidence-based multiplier."""
        thresholds = self.position_config.get("confidence_thresholds", {})
        multipliers = self.position_config.get("position_size_multipliers", {})

        if confidence >= thresholds.get("very_high_confidence", 0.95):
            return multipliers.get("very_high_confidence", 2.0)
        if confidence >= thresholds.get("high_confidence", 0.85):
            return multipliers.get("high_confidence", 1.5)
        if confidence >= thresholds.get("medium_confidence", 0.75):
            return multipliers.get("medium_confidence", 1.0)
        return multipliers.get("low_confidence", 0.5)

    def _get_volatility_multiplier(self, market_conditions: dict[str, Any]) -> float:
        """Get volatility-based multiplier."""
        atr = market_conditions.get("atr", 0.0) if market_conditions else 0.0
        current_price = (
            market_conditions.get("current_price", 1.0) if market_conditions else 1.0
        )
        atr_ratio = atr / current_price if current_price > 0 else 0.0

        thresholds = self.position_config.get("volatility_adjustment", {}).get(
            "volatility_thresholds",
            {},
        )
        multipliers = self.position_config.get("volatility_adjustment", {}).get(
            "volatility_multipliers",
            {},
        )

        if atr_ratio <= thresholds.get("low_volatility", 0.02):
            return multipliers.get("low_volatility", 1.2)
        if atr_ratio <= thresholds.get("medium_volatility", 0.05):
            return multipliers.get("medium_volatility", 1.0)
        return multipliers.get("high_volatility", 0.7)

    def _get_regime_multiplier(self, market_conditions: dict[str, Any]) -> float:
        """Get regime-based multiplier."""
        regime = (
            market_conditions.get("market_regime", "SIDEWAYS_RANGE")
            if market_conditions
            else "SIDEWAYS_RANGE"
        )
        multipliers = self.position_config.get("regime_based_adjustment", {}).get(
            "regime_multipliers",
            {},
        )
        return multipliers.get(regime, 1.0)

    def _get_risk_multiplier(self) -> float:
        """Get current risk-based multiplier."""
        # This would be calculated based on current drawdown and daily loss
        drawdown_multiplier = 1.0
        daily_loss_multiplier = 1.0

        # Apply drawdown adjustment
        if self.max_drawdown >= 0.4:
            drawdown_multiplier = 0.2
        elif self.max_drawdown >= 0.3:
            drawdown_multiplier = 0.5
        elif self.max_drawdown >= 0.2:
            drawdown_multiplier = 0.7
        elif self.max_drawdown >= 0.1:
            drawdown_multiplier = 0.9

        # Apply daily loss adjustment
        if self.daily_pnl <= -0.10:
            daily_loss_multiplier = 0.2
        elif self.daily_pnl <= -0.08:
            daily_loss_multiplier = 0.5
        elif self.daily_pnl <= -0.05:
            daily_loss_multiplier = 0.8

        return min(drawdown_multiplier, daily_loss_multiplier)

    def calculate_leverage(
        self,
        base_leverage: float,
        max_leverage_cap: int,
        confidence: float = 0.0,
        market_conditions: dict[str, Any] = None,
    ) -> int:
        """Calculate optimal leverage based on confidence and risk rules."""
        # Apply confidence-based leverage adjustments
        if confidence >= self.position_config.get("confidence_thresholds", {}).get(
            "very_high_confidence",
            0.95,
        ):
            leverage_boost = self.position_config.get("leverage_multipliers", {}).get(
                "very_high_confidence",
                2.0,
            )
            base_leverage = min(base_leverage * leverage_boost, max_leverage_cap)
            self.logger.info(f"Very high confidence leverage boost: {leverage_boost}x")
        elif confidence >= self.position_config.get("confidence_thresholds", {}).get(
            "high_confidence",
            0.85,
        ):
            leverage_boost = self.position_config.get("leverage_multipliers", {}).get(
                "high_confidence",
                1.8,
            )
            base_leverage = min(base_leverage * leverage_boost, max_leverage_cap)
            self.logger.info(f"High confidence leverage boost: {leverage_boost}x")

        # Apply market condition adjustments
        if market_conditions:
            opportunity_type = market_conditions.get("opportunity_type", "STANDARD")

            # Enhanced leverage for S/R opportunities
            if opportunity_type in ["SR_FADE", "SR_BREAKOUT"]:
                sr_leverage_boost = self.position_config.get(
                    "leverage_multipliers",
                    {},
                ).get("sr_opportunity", 2.0)
                base_leverage = min(base_leverage * sr_leverage_boost, max_leverage_cap)
                self.logger.info(
                    f"S/R opportunity leverage boost: {sr_leverage_boost}x",
                )

            # S/R zone leverage boost
            if market_conditions.get("near_sr_zone", False):
                sr_boost = self.position_config.get("leverage_multipliers", {}).get(
                    "sr_zone",
                    1.5,
                )
                base_leverage = min(base_leverage * sr_boost, max_leverage_cap)
                self.logger.info(f"S/R zone leverage boost: {sr_boost}x")

            # Huge candle leverage boost
            if market_conditions.get("huge_candle", False):
                huge_candle_boost = self.position_config.get(
                    "leverage_multipliers",
                    {},
                ).get("huge_candle", 2.0)
                base_leverage = min(base_leverage * huge_candle_boost, max_leverage_cap)
                self.logger.info(f"Huge candle leverage boost: {huge_candle_boost}x")

        # Apply dynamic risk management to leverage
        if self.dynamic_config.get("enable_dynamic_risk", True):
            base_leverage = self._apply_dynamic_risk_to_leverage(base_leverage)

        # Ensure leverage doesn't exceed maximum cap
        final_leverage = min(int(base_leverage), max_leverage_cap)

        # Log leverage calculation details
        self.logger.info("Enhanced Leverage Calculation:")
        self.logger.info(f"   Base Leverage: {base_leverage}")
        self.logger.info(f"   Max Cap: {max_leverage_cap}")
        self.logger.info(f"   Confidence: {confidence:.2f}")
        if market_conditions:
            self.logger.info(
                f"   Opportunity Type: {market_conditions.get('opportunity_type', 'STANDARD')}",
            )
        self.logger.info(f"   Final Leverage: {final_leverage}x")

        return final_leverage

    def _apply_dynamic_risk_to_leverage(self, base_leverage: float) -> float:
        """Apply dynamic risk management to leverage calculation."""
        # Apply drawdown-based leverage reduction
        if self.max_drawdown >= 0.4:
            base_leverage *= 0.2
        elif self.max_drawdown >= 0.3:
            base_leverage *= 0.5
        elif self.max_drawdown >= 0.2:
            base_leverage *= 0.7
        elif self.max_drawdown >= 0.1:
            base_leverage *= 0.9

        # Apply daily loss-based leverage reduction
        if self.daily_pnl <= -0.10:
            base_leverage *= 0.2
        elif self.daily_pnl <= -0.08:
            base_leverage *= 0.5
        elif self.daily_pnl <= -0.05:
            base_leverage *= 0.8

        return base_leverage

    def update_performance_metrics(self, daily_pnl: float, max_drawdown: float):
        """Update performance metrics for dynamic risk management."""
        self.daily_pnl = daily_pnl
        self.max_drawdown = max_drawdown

    def update_exposure(self, new_exposure: float):
        """Update current total exposure."""
        self.current_exposure = new_exposure

    def get_position_summary(self) -> dict[str, Any]:
        """Get summary of position sizing rules and current state."""
        return {
            "current_exposure": self.current_exposure,
            "daily_pnl": self.daily_pnl,
            "max_drawdown": self.max_drawdown,
            "risk_multiplier": self._get_risk_multiplier(),
            "position_config": self.position_config,
            "dynamic_config": self.dynamic_config,
        }
