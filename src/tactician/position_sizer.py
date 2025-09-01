# src/tactician/position_sizer.py

"""
Simplified Position Sizer for high leverage trading.
Uses ML confidence scores and Kelly criterion for position sizing.
"""

from datetime import datetime
from src.utils.logger import system_logger
from typing import Any
import contextlib

from src.utils.confidence import normalize_dual_confidence
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, initialization_error, missing
from src.utils.centralized_decorators import validate_data_quality
from kelly_criterion_formula import calculate_kelly_multiplier


class PositionSizer:
    """
    Position Sizer component responsible for:
    - Position sizing decisions based on ML confidence scores and Kelly criterion
    - Integration with Strategist for strategy input
    - Position size optimization for high leverage trading

    This is the primary component responsible for position sizing across the system.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PositionSizer")
        # Backward-compatibility shim for legacy self.print calls
        if not hasattr(self, "print"):
            self.print = _shim_print  # type: ignore[attr-defined]

        # Load configuration from step17 optimization results
        self.sizing_config: dict[str, Any] = self.config.get("position_sizing", {})

        # Load step17 optimized parameters
        step17_config = self.config.get("step17_optimization", {})
        position_sizing_optimization = step17_config.get("position_sizing", {})

        # Load optimized position sizing parameters
        self.kelly_multiplier: float = position_sizing_optimization.get("kelly_multiplier", 0.25)
        self.max_position_size: float = position_sizing_optimization.get("max_position_size", 0.5)
        self.min_position_size: float = position_sizing_optimization.get("min_position_size", 0.01)
        self.confidence_threshold: float = position_sizing_optimization.get("confidence_threshold", 0.6)

        # NEW: Combined confidence threshold for position sizing (optimizable in step17)
        self.positionsize_combined_threshold: float = position_sizing_optimization.get("positionsize_combined_threshold", 0.7)

        # Load optimized component weights
        # Removed config-driven weights; internal weighting is handled without exposed params

        # Load additional optimized parameters
        # Removed deprecated parameters: risk_adjustment_factor, confidence_boost_threshold,
        # volatility_adjustment, market_regime_multiplier

        self.is_initialized: bool = False
        self.position_sizing_history: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position sizer configuration"),
            AttributeError: (False, "Missing required sizing parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position sizer initialization",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="position sizer initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )

    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            required_keys = [
                "kelly_multiplier",
                "max_position_size",
                "min_position_size",
            ]
            for key in required_keys:
                if key not in self.sizing_config:
                    self.print(missing(f"Missing required configuration key: {key}"))
                    return False

            if self.max_position_size <= self.min_position_size:
                self.logger.error(
                    "max_position_size must be greater than min_position_size",
                )
                return False

            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                self.print(error("kelly_multiplier must be between 0 and 1"))
                return False

            return True

        except Exception as e:
            self.print(error(f"Error validating configuration: {e}"))
            return False

    @validate_data_quality(
        required_columns=None,  # This method validates dict input, not DataFrame
        min_rows=1,
        max_null_ratio=0.0,
        check_duplicates=False,
        check_timestamps=False,
        context="position sizing calculation input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for position sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return=None,
        context="position sizing calculation",
    )
    def _calculate_kelly_position_size(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
    ) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
        try:
            # Use the new Kelly criterion formula module
            kelly_multiplier = calculate_kelly_multiplier(
                price_target_confidences=price_target_confidences,
                adversarial_confidences=adversarial_confidences,
                kelly_multiplier=self.kelly_multiplier,
            )
            
            # The Kelly multiplier is already scaled by the conservative multiplier
            # and normalized to 0-1 range, so we can use it directly
            # Scale it to our position size range
            kelly_position_size = (
                self.min_position_size
                + (self.max_position_size - self.min_position_size) * kelly_multiplier
            )

            # Ensure within bounds
            return max(
                self.min_position_size, min(self.max_position_size, kelly_position_size),
            )

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f"Error calculating Kelly position size: {e}")
            return self.min_position_size
        except ZeroDivisionError as e:
            self.logger.exception(f"Division by zero in Kelly calculation: {e}")
            return self.min_position_size

    def _calculate_ml_position_size(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
    ) -> float:
        """Calculate position size based on ML confidence scores."""
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

            # Calculate ML-based position size
            # Higher confidence and lower risk = larger position
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk

            # Base position size calculation
            base_position_size = (
                self.min_position_size
                + (self.max_position_size - self.min_position_size)
                * confidence_factor
                * risk_factor
            )

            # Ensure within bounds
            return max(
                self.min_position_size, min(self.max_position_size, base_position_size),
            )

        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f"Error calculating ML position size: {e}")
            return self.min_position_size
        except ZeroDivisionError as e:
            self.logger.exception(f"Division by zero in ML position calculation: {e}")
            return self.min_position_size

    def _calculate_weighted_position_size(
        self,
        kelly_position_size: float,
        ml_position_size: float,
    ) -> float:
        """Calculate weighted position size using Kelly criterion and ML confidence."""
        try:
            # Calculate weighted position size
            # Combine Kelly and ML sizes multiplicatively as requested
            weighted_size = (kelly_position_size * ml_position_size)

            return max(
                self.min_position_size, min(self.max_position_size, weighted_size),
            )

        except Exception as e:
            self.print(error(f"Error calculating weighted position size: {e}"))
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

    def _apply_position_size_modifiers(
        self,
        base_size: float,
        *,
        market_health_analysis: dict[str, Any] | None,
        strategist_risk_parameters: dict[str, Any] | None,
        analyst_confidence: float,
        tactician_confidence: float,
    ) -> float:
        """Adjust position size based on market health (vol/liquidity/stress), strategist risk, and dynamic confidence."""
        try:
            adjusted = base_size

            # Market health: downscale size under high volatility or stress; upscale when healthy
            if market_health_analysis:
                vol = market_health_analysis.get("volatility_analysis", {})
                stress = market_health_analysis.get("stress_analysis", {})
                liq = market_health_analysis.get("liquidity_analysis", {})

                current_vol = float(vol.get("current_volatility", 0.02))
                vol_regime = vol.get("volatility_regime", "normal")
                stress_level = float(stress.get("stress_level", 0.5))  # 0..1
                liquidity_score = float(liq.get("liquidity_score", 0.5))  # 0..1

                # Volatility adjustment
                if vol_regime in ("high", "extreme") or current_vol > 0.03:
                    adjusted *= 0.6
                elif vol_regime == "low" and current_vol < 0.015:
                    adjusted *= 1.1

                # Stress adjustment
                if stress_level >= 0.8:
                    adjusted *= 0.4
                elif stress_level >= 0.6:
                    adjusted *= 0.6
                elif stress_level >= 0.4:
                    adjusted *= 0.8

                # Liquidity adjustment
                if liquidity_score < 0.3:
                    adjusted *= 0.6
                elif liquidity_score > 0.7:
                    adjusted *= 1.05

            # Strategist risk parameters: respect max risk caps without using fixed TP/SL distances
            if strategist_risk_parameters:
                # Example: cap size based on max daily loss or risk per trade signals
                max_position_risk = float(
                    strategist_risk_parameters.get("max_position_risk", 0.01),
                )
                # Ensure final size does not exceed configured max_position_size
                configured_max = float(self.max_position_size)
                adjusted = min(adjusted, configured_max)
                # If max_position_risk is very small, reduce size further
                if max_position_risk <= 0.005:
                    adjusted *= 0.8

            # Dynamic confidence-based modulation (analyst and tactician)
            # Use dual confidence similar to monitor normalization
            _, normalized = normalize_dual_confidence(
                analyst_confidence, tactician_confidence,
            )
            # Scale position by a gentle factor around 1.0 (0.8..1.2)
            conf_scale = 0.8 + 0.4 * normalized
            adjusted *= conf_scale

            return max(self.min_position_size, min(self.max_position_size, adjusted))
        except Exception as e:
            self.print(error(f"Error applying size modifiers: {e}"))
            return max(self.min_position_size, min(self.max_position_size, base_size))

    def _generate_sizing_reason(
        self,
        final_position_size: float,
        kelly_position_size: float,
        ml_position_size: float,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        combined_confidence: float = 0.5,
    ) -> str:
        """Generate reason for position sizing decision."""
        try:
            # Get average confidence and risk
            key_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            risks = []

            for level in key_levels:
                closest_confidence = min(
                    price_target_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                closest_risk = min(
                    adversarial_confidences.keys(),
                    key=lambda x: abs(float(x.replace("%", "")) - level),
                )
                confidences.append(
                    price_target_confidences.get(closest_confidence, 0.5),
                )
                risks.append(adversarial_confidences.get(closest_risk, 0.3))

            avg_confidence = sum(confidences) / len(confidences)
            avg_risk = sum(risks) / len(risks)

            # NEW: Include combined confidence in sizing reason
            if final_position_size >= self.max_position_size * 0.8:
                return f"Maximum position size due to high combined confidence ({combined_confidence:.2f}) and low risk ({avg_risk:.2f})"
            if final_position_size >= self.max_position_size * 0.5:
                return f"Large position size based on combined confidence ({combined_confidence:.2f}) and Kelly criterion ({kelly_position_size:.3f})"
            if final_position_size >= self.min_position_size * 2:
                return f"Moderate position size with combined confidence ({combined_confidence:.2f}) and balanced risk-reward profile"
            if combined_confidence < self.positionsize_combined_threshold:
                return f"Minimum position size due to low combined confidence ({combined_confidence:.2f}) below threshold ({self.positionsize_combined_threshold:.2f})"
            return f"Conservative position size due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})"

        except Exception as e:
            self.print(error(f"Error generating sizing reason: {e}"))
            return "Position size calculated using ML intelligence and Kelly criterion"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position sizer cleanup",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position sizer cleanup",
    )

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position sizer setup",
)