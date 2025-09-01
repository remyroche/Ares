# src/tactician/leverage_sizer.py

"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

from datetime import datetime
from src.utils.logger import system_logger
import contextlib
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors


class LeverageSizer:
    """
    Simplified leverage sizer that uses ML confidence scores and liquidation risk model
    to set leverage between 10x and 100x.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LeverageSizer")
        # Backward-compatibility shim for legacy self.print calls
        if not hasattr(self, "print"):
            self.print = _shim_print  # type: ignore[attr-defined]

        # Load configuration from step17 optimization results
        self.leverage_config: dict[str, Any] = self.config.get("leverage_sizing", {})

        # Load step17 optimized parameters
        step17_config = self.config.get("step17_optimization", {})
        leverage_optimization = step17_config.get("leverage", {})

        # Load optimized leverage parameters
        self.min_leverage: float = leverage_optimization.get("min_leverage", 10.0)
        self.max_leverage: float = leverage_optimization.get("max_leverage", 100.0)
        self.confidence_threshold: float = leverage_optimization.get("confidence_threshold", 0.6)
        self.liquidation_buffer: float = leverage_optimization.get("liquidation_buffer", 0.05)

        # NEW: Combined confidence threshold for leverage sizing (optimizable in step17)
        self.leverage_combined_threshold: float = leverage_optimization.get("leverage_combined_threshold", 0.75)

        # Load optimized component weights
        self.ml_weight: float = leverage_optimization.get("ml_weight", 0.6)
        self.liquidation_weight: float = leverage_optimization.get("liquidation_weight", 0.4)

        # Load additional optimized parameters
        self.leverage_multiplier: float = leverage_optimization.get("leverage_multiplier", 1.0)
        self.risk_adjustment_factor: float = leverage_optimization.get("risk_adjustment_factor", 1.0)
        self.confidence_boost_threshold: float = leverage_optimization.get("confidence_boost_threshold", 0.8)
        self.max_risk_leverage: float = leverage_optimization.get("max_risk_leverage", 50.0)

        self.is_initialized: bool = False
        self.leverage_sizing_history: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid leverage sizer configuration"),
            AttributeError: (False, "Missing required leverage parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="leverage sizer initialization",
    )
    def _validate_configuration(self) -> bool:
        """Validate leverage sizer configuration."""
        try:
            required_keys = [
                "min_leverage",
                "max_leverage",
                "confidence_threshold",
                "liquidation_buffer",
            ]
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            # Validate values
            if self.min_leverage <= 0 or self.min_leverage >= self.max_leverage:
                self.logger.error("Invalid leverage range configuration")
                return False

            if self.liquidation_buffer <= 0 or self.liquidation_buffer >= 1:
                self.logger.error("Invalid liquidation_buffer configuration")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for leverage sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return={},
        context="leverage sizing calculation",
    )
    def _calculate_ml_leverage(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
    ) -> float:
        """Calculate leverage based on ML confidence scores."""
        try:
            # Get average confidence for target levels (0.5% to 2.0%)
            target_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []

            for level in target_levels:
                closest_level = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidence = price_target_confidences.get(closest_level, 0.5)
                confidences.append(confidence)

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)

            # Get average adverse risk
            adverse_risks = []
            for level in target_levels:
                closest_level = min(
                    adversarial_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                risk = adversarial_confidences.get(closest_level, 0.3)
                adverse_risks.append(risk)

            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)

            # Calculate confidence-based leverage
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk

            # Base leverage calculation
            base_leverage = (
                self.min_leverage
                + (self.max_leverage - self.min_leverage)
                * confidence_factor
                * risk_factor
            )

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, base_leverage),
            )

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f"Error calculating ML leverage: {e}")
            return self.min_leverage

    def _calculate_liquidation_safe_leverage(
        self,
        current_price: float,
        account_balance: float,
        market_health_analysis: dict[str, Any] | None,
    ) -> float:
        """Calculate safe leverage to avoid liquidation."""
        try:
            # Base liquidation calculation
            # Assume worst-case scenario: 10% price move against position
            worst_case_move = 0.10

            # Adjust for market volatility
            if market_health_analysis:
                vol_analysis = market_health_analysis.get("volatility_analysis", {})
                current_vol = float(vol_analysis.get("current_volatility", 0.02))

                # If volatility is high, increase the worst-case scenario
                if current_vol > 0.03:
                    worst_case_move = min(0.20, current_vol * 2)
                elif current_vol > 0.02:
                    worst_case_move = 0.15

            # Calculate safe leverage with buffer
            safe_leverage = (1.0 - self.liquidation_buffer) / worst_case_move

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, safe_leverage),
            )

        except (ValueError, TypeError) as e:
            self.logger.exception(f"Error calculating liquidation safe leverage: {e}")
            return self.min_leverage

    def _calculate_weighted_leverage(
        self,
        ml_leverage: float,
        liquidation_leverage: float,
    ) -> float:
        """Calculate weighted leverage using ML and liquidation risk models."""
        try:
            # Calculate weighted leverage
            weighted_leverage = (
                ml_leverage * self.ml_weight
                + liquidation_leverage * self.liquidation_weight
            ) / (self.ml_weight + self.liquidation_weight)

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, weighted_leverage),
            )

        except Exception as e:
            self.logger.exception(f"Error calculating weighted leverage: {e}")
            return self.min_leverage

    def _apply_leverage_modifiers(
        self,
        base_leverage: float,
        *,
        market_health_analysis: dict[str, Any] | None,
        strategist_risk_parameters: dict[str, Any] | None,
        analyst_confidence: float,
        tactician_confidence: float,
    ) -> float:
        """Adjust leverage based on market health and risk parameters."""
        try:
            adjusted = base_leverage

            # Apply market health modifiers
            if market_health_analysis:
                volatility_modifier = market_health_analysis.get("volatility_modifier", 1.0)
                liquidity_modifier = market_health_analysis.get("liquidity_modifier", 1.0)
                stress_modifier = market_health_analysis.get("stress_modifier", 1.0)

                adjusted *= volatility_modifier * liquidity_modifier * stress_modifier

            # Apply strategist risk parameters
            if strategist_risk_parameters:
                risk_modifier = strategist_risk_parameters.get("leverage_modifier", 1.0)
                adjusted *= risk_modifier

            # Apply confidence modifiers
            confidence_modifier = (analyst_confidence + tactician_confidence) / 2
            adjusted *= confidence_modifier

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, adjusted),
            )

        except Exception as e:
            self.logger.exception(f"Error applying leverage modifiers: {e}")
            return base_leverage

    def _generate_leverage_reason(
        self,
        final_leverage: float,
        ml_leverage: float,
        liquidation_leverage: float,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        combined_confidence: float = 0.5,
    ) -> str:
        """Generate reason for leverage sizing decision."""
        try:
            # Get average confidence and risk
            key_levels = [0.5, 1.0, 1.5, 2.0]
            avg_confidence = 0.0
            avg_risk = 0.0

            for level in key_levels:
                closest_level = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidence = price_target_confidences.get(closest_level, 0.5)
                risk = adversarial_confidences.get(closest_level, 0.3)
                avg_confidence += confidence
                avg_risk += risk

            avg_confidence /= len(key_levels)
            avg_risk /= len(key_levels)

            # NEW: Include combined confidence in leverage reason
            if combined_confidence < self.leverage_combined_threshold:
                return (
                    f"Leverage: {final_leverage:.1f}x (minimum due to low combined confidence "
                    f"{combined_confidence:.2f} below threshold {self.leverage_combined_threshold:.2f})"
                )

            return (
                f"Leverage: {final_leverage:.1f}x "
                f"(ML: {ml_leverage:.1f}x, Liquidation: {liquidation_leverage:.1f}x, "
                f"Combined Confidence: {combined_confidence:.3f}, Risk: {avg_risk:.3f})"
            )

        except Exception as e:
            self.logger.exception(f"Error generating leverage reason: {e}")
            return f"Leverage: {final_leverage:.1f}x (Error generating reason)"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="leverage sizer cleanup",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="leverage sizer cleanup",
    )

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="leverage sizer setup",
)