# src/tactician/leverage_sizer.py

"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

from datetime import datetime
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class LeverageSizer:
    """
    Simplified leverage sizer that uses ML confidence scores, liquidation risk model,
    and market health analysis to set leverage between 10x and 100x.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LeverageSizer")

        # Load configuration
        from src.config_optuna import get_parameter_value

        self.leverage_config: dict[str, Any] = self.config.get("leverage_sizing", {})
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
            0.7,
        )
        self.risk_tolerance: float = get_parameter_value(
            "position_sizing_parameters.risk_tolerance",
            0.3,
        )

        # Component weights
        self.ml_weight: float = self.leverage_config.get("ml_weight", 0.5)
        self.liquidation_risk_weight: float = self.leverage_config.get(
            "liquidation_risk_weight",
            0.3,
        )
        self.market_health_weight: float = self.leverage_config.get(
            "market_health_weight",
            0.2,
        )

        self.is_initialized: bool = False
        self.leverage_sizing_history: list[dict[str, Any]] = []
        # Smoothing and governance
        self._last_leverage: float | None = None
        self._last_update_ts: float | None = None
        self.smoothing_alpha: float = float(self.leverage_config.get("smoothing_alpha", 0.3))
        self.min_step: float = float(self.leverage_config.get("min_step", 2.0))
        self.cooldown_seconds: float = float(self.leverage_config.get("cooldown_seconds", 15.0))

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
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.max_leverage <= self.min_leverage:
                self.logger.error("max_leverage must be greater than min_leverage")
                return False

            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.logger.error("confidence_threshold must be between 0 and 1")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
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
        ml_predictions: dict[str, Any],
        liquidation_risk_analysis: dict[str, Any] | None = None,
        market_health_analysis: dict[str, Any] | None = None,
        current_price: float = 0.0,
        target_direction: str = "long",
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """
        Calculate leverage using ML confidence scores, liquidation risk analysis, and market health.

        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            liquidation_risk_analysis: Liquidation risk analysis from liquidation_risk_model
            market_health_analysis: Market health analysis from market_health_analyzer
            current_price: Current market price
            target_direction: Target direction ("long" or "short")

        Returns:
            dict[str, Any]: Leverage sizing analysis
        """
        if not self.is_initialized:
            self.logger.error("Leverage sizer not initialized")
            return None

        self.logger.info(f"Calculating leverage for {target_direction} position...")

        try:
            # Extract ML confidence scores
            price_target_confidences = ml_predictions.get(
                "price_target_confidences",
                {},
            )
            adversarial_confidences = ml_predictions.get("adversarial_confidences", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})

            # Calculate base leverage from ML confidence
            ml_leverage = self._calculate_ml_leverage(
                price_target_confidences,
                adversarial_confidences,
            )

            # Get liquidation risk leverage recommendations
            liquidation_leverage = self._extract_liquidation_leverage(
                liquidation_risk_analysis,
            )

            # Get market health leverage adjustment
            market_health_leverage = self._extract_market_health_leverage(
                market_health_analysis,
            )

            # Calculate weighted leverage
            final_leverage = self._calculate_weighted_leverage(
                ml_leverage,
                liquidation_leverage,
                market_health_leverage,
            )

            # Apply hard risk guardrails based on liquidation and market stress
            final_leverage = self._apply_leverage_guards(
                final_leverage,
                current_price=current_price,
                liquidation_risk_analysis=liquidation_risk_analysis,
                market_health_analysis=market_health_analysis,
            )

            # Create leverage sizing analysis
            leverage_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "target_direction": target_direction,
                "ml_leverage": ml_leverage,
                "liquidation_leverage": liquidation_leverage,
                "market_health_leverage": market_health_leverage,
                "final_leverage": final_leverage,
                "price_target_confidences": price_target_confidences,
                "adversarial_confidences": adversarial_confidences,
                "directional_confidence": directional_confidence,
                "leverage_reason": self._generate_leverage_reason(
                    final_leverage,
                    ml_leverage,
                    liquidation_leverage,
                    market_health_leverage,
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

        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
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
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk

            # Base leverage calculation (10x to 100x range)
            ml_leverage = (
                self.min_leverage
                + (self.max_leverage - self.min_leverage)
                * confidence_factor
                * risk_factor
            )

            # Apply risk tolerance adjustment
            risk_adjusted_leverage = ml_leverage * (1.0 - self.risk_tolerance)
            return max(self.min_leverage, min(self.max_leverage, risk_adjusted_leverage))

        except Exception as e:
            self.logger.error(f"Error calculating ML leverage: {e}")
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

        except Exception as e:
            self.logger.error(f"Error extracting liquidation leverage: {e}")
            return self.min_leverage

    def _apply_leverage_guards(
        self,
        proposed_leverage: float,
        *,
        current_price: float,
        liquidation_risk_analysis: dict[str, Any] | None,
        market_health_analysis: dict[str, Any] | None,
    ) -> float:
        """Apply hard guardrails to leverage based on liquidation proximity and market stress."""
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
                        adjusted = max(self.min_leverage, proposed_leverage * risk_scale)

            # Guard 2: Market stress clamp
            if market_health_analysis:
                stress = market_health_analysis.get("stress_analysis", {})
                stress_level = float(stress.get("stress_level", 0.5))  # 0..1
                # In high stress, reduce leverage using gentle caps
                if stress_level >= 0.8:
                    adjusted = min(adjusted, max(self.min_leverage, self.max_leverage * 0.2))
                elif stress_level >= 0.6:
                    adjusted = min(adjusted, self.max_leverage * 0.35)
                elif stress_level >= 0.4:
                    adjusted = min(adjusted, self.max_leverage * 0.6)

            # Clamp to global bounds
            adjusted = max(self.min_leverage, min(self.max_leverage, adjusted))

            # Smoothing (EMA) and minimum step to avoid oscillations
            now_ts = datetime.utcnow().timestamp()
            if self._last_leverage is not None:
                # Enforce cooldown between changes
                if self._last_update_ts and (now_ts - self._last_update_ts) < self.cooldown_seconds:
                    adjusted = self._last_leverage
                else:
                    ema = self.smoothing_alpha * adjusted + (1 - self.smoothing_alpha) * self._last_leverage
                    # Enforce minimum step size
                    if abs(ema - self._last_leverage) < self.min_step:
                        adjusted = self._last_leverage
                    else:
                        adjusted = ema
            # Record
            self._last_leverage = float(adjusted)
            self._last_update_ts = now_ts

            return float(adjusted)
        except Exception as e:
            self.logger.error(f"Error applying leverage guards: {e}")
            return max(self.min_leverage, min(self.max_leverage, proposed_leverage))

    def _extract_market_health_leverage(
        self,
        market_health_analysis: dict[str, Any] | None,
    ) -> float:
        """Extract leverage adjustment from market health analysis."""
        try:
            if not market_health_analysis:
                return self.min_leverage

            # Get volatility analysis
            volatility_analysis = market_health_analysis.get("volatility_analysis", {})
            current_volatility = volatility_analysis.get("current_volatility", 0.02)
            historical_volatility = volatility_analysis.get(
                "historical_volatility",
                0.02,
            )
            volatility_regime = volatility_analysis.get("volatility_regime", "normal")

            # Get liquidity analysis
            liquidity_analysis = market_health_analysis.get("liquidity_analysis", {})
            liquidity_score = liquidity_analysis.get("liquidity_score", 0.5)
            bid_ask_spread = liquidity_analysis.get("bid_ask_spread", 0.001)
            market_depth = liquidity_analysis.get("market_depth", 0.5)

            # Get market stress analysis
            stress_analysis = market_health_analysis.get("stress_analysis", {})
            stress_level = stress_analysis.get("stress_level", 0.5)
            stress_regime = stress_analysis.get("stress_regime", "normal")

            # Calculate volatility factor with regime consideration
            volatility_factor = self._calculate_volatility_factor(
                current_volatility,
                historical_volatility,
                volatility_regime,
            )

            # Calculate liquidity factor with multiple indicators
            liquidity_factor = self._calculate_liquidity_factor(
                liquidity_score,
                bid_ask_spread,
                market_depth,
            )

            # Calculate stress factor with regime consideration
            stress_factor = self._calculate_stress_factor(stress_level, stress_regime)

            # Combine factors with weighted average
            market_health_factor = (
                volatility_factor * 0.4  # Volatility has highest weight
                + liquidity_factor * 0.35  # Liquidity is second most important
                + stress_factor * 0.25  # Stress is least important
            )

            # Calculate market health leverage
            market_health_leverage = (
                self.min_leverage
                + (self.max_leverage - self.min_leverage) * market_health_factor
            )

            return max(
                self.min_leverage,
                min(self.max_leverage, market_health_leverage),
            )

        except Exception as e:
            self.logger.error(f"Error extracting market health leverage: {e}")
            return self.min_leverage

    def _calculate_volatility_factor(
        self,
        current_vol: float,
        historical_vol: float,
        regime: str,
    ) -> float:
        """Calculate volatility factor with regime consideration."""
        try:
            # Define volatility thresholds
            low_vol_threshold = 0.01  # 1%
            normal_vol_threshold = 0.03  # 3%
            high_vol_threshold = 0.05  # 5%
            extreme_vol_threshold = 0.08  # 8%

            # Calculate volatility ratio (current vs historical)
            vol_ratio = current_vol / max(historical_vol, 0.001)

            # Base factor based on current volatility
            if current_vol <= low_vol_threshold:
                base_factor = 1.0  # Full leverage in low volatility
            elif current_vol <= normal_vol_threshold:
                base_factor = 0.9  # Slight reduction
            elif current_vol <= high_vol_threshold:
                base_factor = 0.7  # Moderate reduction
            elif current_vol <= extreme_vol_threshold:
                base_factor = 0.4  # Significant reduction
            else:
                base_factor = 0.2  # Extreme reduction

            # Adjust based on volatility regime
            if regime == "low_volatility":
                base_factor *= 1.1  # Increase leverage in low vol regime
            elif regime == "high_volatility":
                base_factor *= 0.8  # Decrease leverage in high vol regime
            elif regime == "extreme_volatility":
                base_factor *= 0.5  # Significant decrease in extreme vol

            # Adjust based on volatility ratio (current vs historical)
            if vol_ratio > 1.5:  # Current vol is 50% higher than historical
                base_factor *= 0.8
            elif vol_ratio < 0.7:  # Current vol is 30% lower than historical
                base_factor *= 1.1

            return max(0.1, min(1.0, base_factor))

        except Exception as e:
            self.logger.error(f"Error calculating volatility factor: {e}")
            return 0.5

    def _calculate_liquidity_factor(
        self,
        liquidity_score: float,
        bid_ask_spread: float,
        market_depth: float,
    ) -> float:
        """Calculate liquidity factor with multiple indicators."""
        try:
            # Define liquidity thresholds
            tight_spread = 0.0005  # 0.05%
            normal_spread = 0.001  # 0.1%
            wide_spread = 0.002  # 0.2%

            # Calculate spread factor
            if bid_ask_spread <= tight_spread:
                spread_factor = 1.0
            elif bid_ask_spread <= normal_spread:
                spread_factor = 0.9
            elif bid_ask_spread <= wide_spread:
                spread_factor = 0.7
            else:
                spread_factor = 0.5

            # Calculate depth factor
            depth_factor = market_depth  # Direct use of market depth score

            # Calculate overall liquidity factor
            liquidity_factor = (
                liquidity_score * 0.4 + spread_factor * 0.4 + depth_factor * 0.2
            )

            return max(0.1, min(1.0, liquidity_factor))

        except Exception as e:
            self.logger.error(f"Error calculating liquidity factor: {e}")
            return 0.5

    def _calculate_stress_factor(self, stress_level: float, regime: str) -> float:
        """Calculate stress factor with regime consideration."""
        try:
            # Define stress thresholds
            low_stress = 0.2
            normal_stress = 0.5
            high_stress = 0.7
            extreme_stress = 0.9

            # Base factor based on stress level
            if stress_level <= low_stress:
                base_factor = 1.0  # Full leverage in low stress
            elif stress_level <= normal_stress:
                base_factor = 0.9  # Slight reduction
            elif stress_level <= high_stress:
                base_factor = 0.7  # Moderate reduction
            elif stress_level <= extreme_stress:
                base_factor = 0.4  # Significant reduction
            else:
                base_factor = 0.2  # Extreme reduction

            # Adjust based on stress regime
            if regime == "low_stress":
                base_factor *= 1.1  # Increase leverage in low stress
            elif regime == "high_stress":
                base_factor *= 0.8  # Decrease leverage in high stress
            elif regime == "extreme_stress":
                base_factor *= 0.5  # Significant decrease in extreme stress
            elif regime == "crisis":
                base_factor *= 0.3  # Minimal leverage in crisis

            return max(0.1, min(1.0, base_factor))

        except Exception as e:
            self.logger.error(f"Error calculating stress factor: {e}")
            return 0.5

    def _calculate_weighted_leverage(
        self,
        ml_leverage: float,
        liquidation_leverage: float,
        market_health_leverage: float,
    ) -> float:
        """Calculate weighted leverage using component indicators."""
        try:
            # Calculate weighted leverage
            weighted_leverage = (
                ml_leverage * self.ml_weight
                + liquidation_leverage * self.liquidation_risk_weight
                + market_health_leverage * self.market_health_weight
            ) / (
                self.ml_weight
                + self.liquidation_risk_weight
                + self.market_health_weight
            )

            return max(self.min_leverage, min(self.max_leverage, weighted_leverage))

        except Exception as e:
            self.logger.error(f"Error calculating weighted leverage: {e}")
            return ml_leverage

    def _generate_leverage_reason(
        self,
        final_leverage: float,
        ml_leverage: float,
        liquidation_leverage: float,
        market_health_leverage: float,
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

        except Exception as e:
            self.logger.error(f"Error generating leverage reason: {e}")
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
        except Exception as e:
            self.logger.error(f"Error stopping leverage sizer: {e}")


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

    except Exception as e:
        system_logger.error(f"Error setting up leverage sizer: {e}")
        return None
