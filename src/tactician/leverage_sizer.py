# src/tactician/leverage_sizer.py

"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

from datetime import datetime
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    initialization_error,
    missing,
)


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

            def _shim_print(message: str) -> None:
                try:
                    self.logger.error(str(message))
                except Exception:
                    pass

            self.print = _shim_print  # type: ignore[attr-defined]

        # Load configuration
        from src.config_optuna import get_parameter_value

        self.leverage_config: dict[str, Any] = self.config.get("leverage_sizing", {})
        self.symbol_risk_limits: dict[str, Any] = self.leverage_config.get(
            "symbol_risk_limits",
            {},
        )
        # Example: {"BTCUSDT": {"max_leverage": 50.0, "margin_mode": "isolated"}}
        self.max_leverage: float = get_parameter_value(
            "position_sizing_parameters.max_leverage",
            100.0,
        )
        self.min_leverage: float = get_parameter_value(
            "position_sizing_parameters.min_leverage",
            10.0,
        )
        self.confidence_threshold: float = get_parameter_value(
            "position_sizing_parameters.leverage_confidence_threshold",
            0.6,  # More aggressive: lowered from 0.7 to 0.6
        )
        self.risk_tolerance: float = get_parameter_value(
            "position_sizing_parameters.risk_tolerance",
            0.3,
        )

        # Component weights
        self.ml_weight: float = self.leverage_config.get("ml_weight", 0.5)
        self.liquidation_risk_weight: float = self.leverage_config.get(
            "liquidation_risk_weight",
            0.5,  # Increased from 0.3 since market_health_weight is removed
        )

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
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="leverage sizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the leverage sizer."""
        self.logger.info("Initializing leverage sizer...")

        # Validate configuration
        if not self._validate_configuration():
            return False

        self.is_initialized = True
        self.logger.info("✅ Leverage sizer initialized successfully")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate leverage sizer configuration."""
        try:
            required_keys = ["max_leverage", "min_leverage", "confidence_threshold"]
            for key in required_keys:
                if key not in self.leverage_config:
                    self.print(missing("Missing required configuration key: {key}"))
                    return False

            if self.max_leverage <= self.min_leverage:
                self.print(error("max_leverage must be greater than min_leverage"))
                return False

            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.print(error("confidence_threshold must be between 0 and 1"))
                return False

            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for leverage sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return=None,
        context="leverage sizing calculation",
    )
    async def calculate_leverage(
        self,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
        liquidation_risk_analysis: dict[str, Any] | None = None,
        current_price: float = 0.0,
        target_direction: str = "long",
        symbol: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Calculate leverage using ML confidence scores and liquidation risk analysis.

        Args:
            price_target_confidences: ML confidence predictions for price targets
            adversarial_confidences: ML confidence predictions for adverse movements
            liquidation_risk_analysis: Liquidation risk analysis results
            current_price: Current market price
            target_direction: Target trading direction ("long" or "short")
            symbol: Trading symbol for symbol-specific limits

        Returns:
            dict: Leverage analysis results
        """
        if not self.is_initialized:
            self.print(initialization_error("Leverage sizer not initialized"))
            return None

        self.logger.info(f"Calculating leverage for {target_direction} position...")

        try:
            # Extract ML confidence scores
            # price_target_confidences = ml_predictions.get(
            #     "price_target_confidences",
            #     {},
            # )
            # adversarial_confidences = ml_predictions.get("adversarial_confidences", {})
            # directional_confidence = ml_predictions.get("directional_confidence", {})

            # Calculate base leverage from ML confidence
            ml_leverage = self._calculate_ml_leverage(
                price_target_confidences,
                adversarial_confidences,
            )

            # Get liquidation risk leverage recommendations
            liquidation_leverage = self._extract_liquidation_leverage(
                liquidation_risk_analysis,
            )

            # Calculate weighted leverage (ML + Liquidation only)
            final_leverage = self._calculate_weighted_leverage(
                ml_leverage,
                liquidation_leverage,
            )

            # Apply hard risk guardrails based on liquidation proximity
            final_leverage = self._apply_leverage_guards(
                final_leverage,
                current_price=current_price,
                liquidation_risk_analysis=liquidation_risk_analysis,
            )

            # Enforce symbol-specific leverage caps and include margin mode
            symbol_limits = (
                self.symbol_risk_limits.get(symbol or "", {}) if symbol else {}
            )
            per_symbol_max = float(symbol_limits.get("max_leverage", self.max_leverage))
            final_leverage = min(final_leverage, per_symbol_max)
            margin_mode = symbol_limits.get(
                "margin_mode",
                self.leverage_config.get("default_margin_mode", "cross"),
            )

            # Create leverage sizing analysis
            leverage_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "target_direction": target_direction,
                "symbol": symbol,
                "ml_leverage": ml_leverage,
                "liquidation_leverage": liquidation_leverage,
                "final_leverage": final_leverage,
                "per_symbol_max_leverage": per_symbol_max,
                "margin_mode": margin_mode,
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "directional_confidence": {},  # This line was removed from the new_code, so it's removed here.
                "leverage_reason": self._generate_leverage_reason(
                    final_leverage,
                    ml_leverage,
                    liquidation_leverage,
                    price_target_confidences,
                    adversarial_confidences,
                ),
            }

            # Store in history
            self.leverage_sizing_history.append(leverage_analysis)
            if len(self.leverage_sizing_history) > 100:  # Keep last 100 entries
                self.leverage_sizing_history = self.leverage_sizing_history[-100:]

            self.logger.info(f"✅ Leverage calculated: {final_leverage:.2f}x")
            return leverage_analysis

        except Exception:
            self.print(error("Error calculating leverage: {e}"))
            return None

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

            # Calculate leverage based on confidence and risk
            # Higher confidence and lower risk = higher leverage
            # More aggressive: 100x leverage at 0.9 confidence instead of 1.0
            confidence_factor = (avg_confidence / 0.9) if avg_confidence <= 0.9 else 1.0
            risk_factor = 1.0 - avg_adverse_risk

            # Base leverage calculation (10x to 100x range)
            # More aggressive scaling: 100x leverage achievable at 0.9 confidence
            ml_leverage = (
                self.min_leverage
                + (self.max_leverage - self.min_leverage)
                * confidence_factor
                * risk_factor
            )

            # Apply risk tolerance adjustment
            # More aggressive: reduce risk tolerance impact for higher leverage
            risk_adjusted_leverage = ml_leverage * (
                1.0 - self.risk_tolerance * 0.9
            )  # 10% less conservative
            return max(
                self.min_leverage,
                min(self.max_leverage, risk_adjusted_leverage),
            )

        except Exception:
            self.print(error("Error calculating ML leverage: {e}"))
            return self.min_leverage

    def _extract_liquidation_leverage(
        self,
        liquidation_risk_analysis: dict[str, Any] | None,
    ) -> float:
        """Extract leverage recommendations from liquidation risk analysis."""
        try:
            if not liquidation_risk_analysis:
                return self.min_leverage

            # Get safe leverage levels
            safe_leverage_levels = liquidation_risk_analysis.get(
                "safe_leverage_levels",
                {},
            )

            if not safe_leverage_levels:
                return self.min_leverage

            # Get average safe leverage
            safe_leverages = []
            for leverage_data in safe_leverage_levels.values():
                safe_leverage = leverage_data.get("safe_leverage", self.min_leverage)
                safe_leverages.append(safe_leverage)

            if safe_leverages:
                avg_safe_leverage = sum(safe_leverages) / len(safe_leverages)
                return max(self.min_leverage, min(self.max_leverage, avg_safe_leverage))
            return self.min_leverage

        except Exception:
            self.print(error("Error extracting liquidation leverage: {e}"))
            return self.min_leverage

    def _apply_leverage_guards(
        self,
        proposed_leverage: float,
        *,
        current_price: float,
        liquidation_risk_analysis: dict[str, Any] | None,
    ) -> float:
        """Apply hard guardrails to leverage based on liquidation proximity."""
        try:
            adjusted = proposed_leverage

            # Guard 1: Liquidation proximity buffer
            # Expect liquidation_risk_analysis to contain an estimated liquidation price per-symbol
            # and/or a liquidation buffer ratio.
            if liquidation_risk_analysis:
                liq_price = liquidation_risk_analysis.get("estimated_liquidation_price")
                min_buffer_ratio = liquidation_risk_analysis.get(
                    "min_liquidation_buffer_ratio",
                    0.015,
                )  # require at least 1.5% distance
                if liq_price and current_price:
                    distance = abs(current_price - liq_price) / current_price
                    if distance < min_buffer_ratio:
                        # Soft scale down (no more than 50% cut) to increase buffer
                        risk_scale = max(0.5, distance / max(min_buffer_ratio, 1e-6))
                        adjusted = max(
                            self.min_leverage,
                            proposed_leverage * risk_scale,
                        )

            # Clamp to global bounds and return
            return max(self.min_leverage, min(self.max_leverage, adjusted))
        except Exception:
            self.print(error("Error applying leverage guards: {e}"))
            return max(self.min_leverage, min(self.max_leverage, proposed_leverage))

    def _calculate_weighted_leverage(
        self,
        ml_leverage: float,
        liquidation_leverage: float,
    ) -> float:
        """Calculate weighted leverage using component indicators."""
        try:
            # Calculate weighted leverage
            weighted_leverage = (
                ml_leverage * self.ml_weight
                + liquidation_leverage * self.liquidation_risk_weight
            ) / (self.ml_weight + self.liquidation_risk_weight)

            return max(self.min_leverage, min(self.max_leverage, weighted_leverage))

        except Exception:
            self.print(error("Error calculating weighted leverage: {e}"))
            return ml_leverage

    def _generate_leverage_reason(
        self,
        final_leverage: float,
        ml_leverage: float,
        liquidation_leverage: float,
        price_target_confidences: dict[str, float],
        adversarial_confidences: dict[str, float],
    ) -> str:
        """Generate reason for leverage decision."""
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

            if final_leverage >= self.max_leverage * 0.8:
                return f"Maximum leverage due to high confidence ({avg_confidence:.2f}) and low risk ({avg_risk:.2f})"
            if final_leverage >= self.max_leverage * 0.5:
                return f"High leverage based on ML confidence ({ml_leverage:.2f}x) and liquidation safety ({liquidation_leverage:.2f}x)"
            if final_leverage >= self.min_leverage * 2:
                return "Moderate leverage with balanced risk-reward profile"
            return f"Conservative leverage due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})"

        except Exception:
            self.print(error("Error generating leverage reason: {e}"))
            return "Leverage calculated using ML intelligence and liquidation risk analysis"

    def get_leverage_sizing_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get leverage sizing history."""
        if limit:
            return self.leverage_sizing_history[-limit:]
        return self.leverage_sizing_history.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="leverage sizer cleanup",
    )
    async def stop(self) -> None:
        """Stop the leverage sizer."""
        try:
            self.logger.info("Stopping leverage sizer...")
            self.is_initialized = False
            self.logger.info("✅ Leverage sizer stopped successfully")
        except Exception:
            self.print(error("Error stopping leverage sizer: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="leverage sizer setup",
)
async def setup_leverage_sizer(
    config: dict[str, Any] | None = None,
) -> LeverageSizer | None:
    """
    Setup leverage sizer.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[LeverageSizer]: Initialized leverage sizer or None
    """
    try:
        if config is None:
            config = {}

        leverage_sizer = LeverageSizer(config)

        if await leverage_sizer.initialize():
            return leverage_sizer
        return None

    except Exception:
        system_logger.exception(error("Failed to setup Leverage Sizer: {e}"))
        return None
