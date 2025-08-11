# src/analyst/liquidation_risk_model.py
from typing import Any

import pandas as pd

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
)


class LiquidationRiskModel:
    """
    Simplified Liquidation Risk Model that takes ML confidence predictions
    and determines safe leverage levels based on adverse price change risk.
    Optimized for 10x-100x leverage trading.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize Liquidation Risk Model.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LiquidationRiskModel")

        # Model state
        self.is_initialized: bool = False
        self.risk_assessments: dict[str, Any] = {}

        # Configuration
        self.risk_config: dict[str, Any] = self.config.get("liquidation_risk_model", {})

        # Risk thresholds - adjusted for high leverage trading
        self.max_adverse_risk: float = self.risk_config.get("max_adverse_risk", 0.3)
        self.safe_leverage_multiplier: float = self.risk_config.get(
            "safe_leverage_multiplier",
            0.8,
        )
        self.max_leverage: int = self.risk_config.get(
            "max_leverage",
            100,
        )  # Increased for high leverage
        self.min_leverage: int = self.risk_config.get(
            "min_leverage",
            10,
        )  # Increased minimum

        # Adverse movement thresholds for different leverage levels (10x-100x)
        self.leverage_risk_levels: dict[int, float] = {
            10: 0.1,  # 10x leverage: can handle 10% adverse movement
            15: 0.08,  # 15x leverage: can handle 8% adverse movement
            20: 0.07,  # 20x leverage: can handle 7% adverse movement
            25: 0.06,  # 25x leverage: can handle 6% adverse movement
            30: 0.05,  # 30x leverage: can handle 5% adverse movement
            40: 0.04,  # 40x leverage: can handle 4% adverse movement
            50: 0.035,  # 50x leverage: can handle 3.5% adverse movement
            60: 0.03,  # 60x leverage: can handle 3% adverse movement
            75: 0.025,  # 75x leverage: can handle 2.5% adverse movement
            100: 0.02,  # 100x leverage: can handle 2% adverse movement
        }

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid liquidation risk model configuration"),
            AttributeError: (False, "Missing required risk model parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="liquidation risk model initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize Liquidation Risk Model with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Liquidation Risk Model...")

            # Load risk model configuration
            await self._load_risk_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for liquidation risk model"))
                return False

            self.is_initialized = True
            self.logger.info(
                "✅ Liquidation Risk Model initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Liquidation Risk Model initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk configuration loading",
    )
    async def _load_risk_configuration(self) -> None:
        """Load risk model configuration."""
        self.logger.info("Loading liquidation risk configuration...")

        # Additional configuration can be loaded here
        self.logger.info("Risk configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate risk model configuration."""
        try:
            if self.max_adverse_risk <= 0 or self.max_adverse_risk > 1:
                self.print(error("max_adverse_risk must be between 0 and 1"))
                return False

            if self.safe_leverage_multiplier <= 0 or self.safe_leverage_multiplier > 1:
                self.print(error("safe_leverage_multiplier must be between 0 and 1"))
                return False

            if self.max_leverage < 10:
                self.logger.error(
                    "max_leverage must be at least 10 for high leverage trading",
                )
                return False

            if self.min_leverage < 10:
                self.logger.error(
                    "min_leverage must be at least 10 for high leverage trading",
                )
                return False

            self.logger.info("Risk model configuration validation passed")
            return True

        except Exception:
            self.print(failed("Configuration validation failed: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for liquidation risk calculation"),
            AttributeError: (None, "Model not properly initialized"),
        },
        default_return=None,
        context="liquidation risk calculation",
    )
    async def calculate_liquidation_risk(
        self,
        ml_predictions: dict[str, Any],
        current_price: float,
        target_direction: str = "long",
    ) -> dict[str, Any]:
        """
        Calculate liquidation risk based on ML confidence predictions.

        Args:
            ml_predictions: Predictions from ml_confidence_predictor.py
            current_price: Current asset price
            target_direction: "long" or "short"

        Returns:
            dict: Risk assessment with safe leverage levels and liquidation prices
        """
        if not self.is_initialized:
            self.print(initialization_error("Liquidation risk model not initialized"))
            return None

        try:
            self.logger.info(
                f"Calculating liquidation risk for {target_direction} position",
            )

            # Extract adverse movement probabilities from ML predictions
            adverse_risk = self._extract_adverse_risk(ml_predictions, target_direction)

            # Calculate safe leverage levels
            safe_leverage = self._calculate_safe_leverage(
                adverse_risk,
                target_direction,
            )

            # Calculate liquidation prices for different leverage levels
            liquidation_prices = self._calculate_liquidation_prices(
                current_price,
                target_direction,
            )

            # Generate risk assessment
            risk_assessment = {
                "adverse_risk": adverse_risk,
                "safe_leverage": safe_leverage,
                "max_safe_leverage": self._get_max_safe_leverage(adverse_risk),
                "risk_level": self._classify_risk_level(adverse_risk),
                "recommendation": self._generate_risk_recommendation(
                    adverse_risk,
                    safe_leverage,
                ),
                "liquidation_prices": liquidation_prices,
                "target_direction": target_direction,
                "current_price": current_price,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            self.risk_assessments = risk_assessment
            self.logger.info(
                f"Risk assessment completed: safe leverage = {safe_leverage}x, adverse risk = {adverse_risk:.3f}",
            )

            return risk_assessment

        except Exception:
            self.print(error("Error calculating liquidation risk: {e}"))
            return None

    def _extract_adverse_risk(
        self,
        ml_predictions: dict[str, Any],
        target_direction: str,
    ) -> float:
        """
        Extract adverse movement risk from ML predictions.

        Args:
            ml_predictions: Predictions from ml_confidence_predictor
            target_direction: "long" or "short"

        Returns:
            float: Adverse risk probability (0-1)
        """
        try:
            # Look for adverse movement probabilities in ML predictions
            ml_predictions.get("adverse_probabilities", {})

            if target_direction == "long":
                # For long positions, adverse movement is downward
                adverse_key = "decrease_probabilities"
            else:
                # For short positions, adverse movement is upward
                adverse_key = "increase_probabilities"

            # Get the highest adverse probability as the risk measure
            if adverse_key in ml_predictions:
                adverse_probs = ml_predictions[adverse_key]
                if isinstance(adverse_probs, dict):
                    max_adverse_prob = (
                        max(adverse_probs.values()) if adverse_probs else 0.0
                    )
                else:
                    max_adverse_prob = float(adverse_probs) if adverse_probs else 0.0
            else:
                # Fallback: use a default risk based on confidence
                confidence = ml_predictions.get("confidence", 0.5)
                max_adverse_prob = 1.0 - confidence

            self.logger.info(
                f"Extracted adverse risk: {max_adverse_prob:.3f} for {target_direction}",
            )
            return max_adverse_prob

        except Exception:
            self.print(error("Error extracting adverse risk: {e}"))
            return 0.5  # Default moderate risk

    def _calculate_safe_leverage(
        self,
        adverse_risk: float,
        target_direction: str,
    ) -> int:
        """
        Calculate safe leverage level based on adverse risk.

        Args:
            adverse_risk: Probability of adverse movement (0-1)
            target_direction: "long" or "short"

        Returns:
            int: Safe leverage level
        """
        try:
            # Find the highest leverage level where adverse risk is acceptable
            safe_leverage = self.min_leverage

            for leverage, max_risk in sorted(self.leverage_risk_levels.items()):
                if adverse_risk <= max_risk:
                    safe_leverage = leverage
                else:
                    break

            # Apply safety multiplier
            safe_leverage = int(safe_leverage * self.safe_leverage_multiplier)

            # Ensure within bounds
            safe_leverage = max(
                self.min_leverage,
                min(safe_leverage, self.max_leverage),
            )

            self.logger.info(
                f"Calculated safe leverage: {safe_leverage}x (adverse risk: {adverse_risk:.3f})",
            )
            return safe_leverage

        except Exception:
            self.print(error("Error calculating safe leverage: {e}"))
            return self.min_leverage

    def _get_max_safe_leverage(self, adverse_risk: float) -> int:
        """Get maximum safe leverage without safety multiplier."""
        try:
            max_leverage = self.min_leverage

            for leverage, max_risk in sorted(self.leverage_risk_levels.items()):
                if adverse_risk <= max_risk:
                    max_leverage = leverage
                else:
                    break

            return max_leverage

        except Exception:
            self.print(error("Error calculating max safe leverage: {e}"))
            return self.min_leverage

    def _calculate_liquidation_prices(
        self,
        current_price: float,
        target_direction: str,
    ) -> dict[str, Any]:
        """
        Calculate liquidation prices for different leverage levels.

        Args:
            current_price: Current asset price
            target_direction: "long" or "short"

        Returns:
            dict: Liquidation prices for different leverage levels
        """
        try:
            liquidation_prices = {}

            for leverage in sorted(self.leverage_risk_levels.keys()):
                if target_direction == "long":
                    # For long positions, liquidation price is below current price
                    liquidation_price = current_price * (1 - 1 / leverage)
                    distance_to_liquidation = (
                        current_price - liquidation_price
                    ) / current_price
                else:
                    # For short positions, liquidation price is above current price
                    liquidation_price = current_price * (1 + 1 / leverage)
                    distance_to_liquidation = (
                        liquidation_price - current_price
                    ) / current_price

                liquidation_prices[f"{leverage}x"] = {
                    "liquidation_price": liquidation_price,
                    "distance_to_liquidation": distance_to_liquidation,
                    "distance_percentage": distance_to_liquidation * 100,
                }

            self.logger.info(
                f"Calculated liquidation prices for {len(liquidation_prices)} leverage levels",
            )
            return liquidation_prices

        except Exception:
            self.print(error("Error calculating liquidation prices: {e}"))
            return {}

    def _classify_risk_level(self, adverse_risk: float) -> str:
        """Classify risk level based on adverse probability."""
        if adverse_risk <= 0.05:  # 5% adverse risk
            return "VERY_LOW"
        if adverse_risk <= 0.1:  # 10% adverse risk
            return "LOW"
        if adverse_risk <= 0.2:  # 20% adverse risk
            return "MODERATE"
        if adverse_risk <= 0.3:  # 30% adverse risk
            return "HIGH"
        return "VERY_HIGH"

    def _generate_risk_recommendation(
        self,
        adverse_risk: float,
        safe_leverage: int,
    ) -> str:
        """Generate risk recommendation based on assessment."""
        risk_level = self._classify_risk_level(adverse_risk)

        if risk_level == "VERY_LOW":
            return f"Very low risk environment. Safe to use up to {safe_leverage}x leverage."
        if risk_level == "LOW":
            return f"Low risk environment. Safe to use up to {safe_leverage}x leverage."
        if risk_level == "MODERATE":
            return f"Moderate risk. Use {safe_leverage}x leverage with caution."
        if risk_level == "HIGH":
            return f"High risk environment. Limit leverage to {safe_leverage}x maximum."
        return f"Very high risk. Avoid leverage or use minimum {safe_leverage}x only."

    def get_risk_assessment(self) -> dict[str, Any]:
        """Get the latest risk assessment."""
        return self.risk_assessments

    def get_risk_summary(self) -> dict[str, Any]:
        """Get a summary of risk metrics."""
        if not self.risk_assessments:
            return {}

        return {
            "safe_leverage": self.risk_assessments.get("safe_leverage", 10),
            "risk_level": self.risk_assessments.get("risk_level", "UNKNOWN"),
            "adverse_risk": self.risk_assessments.get("adverse_risk", 0.0),
            "recommendation": self.risk_assessments.get("recommendation", ""),
            "current_price": self.risk_assessments.get("current_price", 0.0),
            "timestamp": self.risk_assessments.get("timestamp", ""),
        }

    def get_liquidation_price_for_leverage(
        self,
        leverage: int,
        current_price: float,
        direction: str,
    ) -> float:
        """
        Get liquidation price for a specific leverage level.

        Args:
            leverage: Leverage level
            current_price: Current asset price
            direction: "long" or "short"

        Returns:
            float: Liquidation price
        """
        try:
            if direction == "long":
                return current_price * (1 - 1 / leverage)
            return current_price * (1 + 1 / leverage)
        except Exception:
            self.print(error("Error calculating liquidation price: {e}"))
            return current_price

    def get_distance_to_liquidation(
        self,
        leverage: int,
        current_price: float,
        direction: str,
    ) -> float:
        """
        Get distance to liquidation as a percentage.

        Args:
            leverage: Leverage level
            current_price: Current asset price
            direction: "long" or "short"

        Returns:
            float: Distance to liquidation as percentage
        """
        try:
            liquidation_price = self.get_liquidation_price_for_leverage(
                leverage,
                current_price,
                direction,
            )

            if direction == "long":
                return ((current_price - liquidation_price) / current_price) * 100
            return ((liquidation_price - current_price) / current_price) * 100
        except Exception:
            self.print(error("Error calculating distance to liquidation: {e}"))
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="liquidation risk model cleanup",
    )
    async def stop(self) -> None:
        """Clean up liquidation risk model resources."""
        try:
            self.logger.info("Stopping Liquidation Risk Model...")
            self.is_initialized = False
            self.risk_assessments = {}
            self.logger.info("✅ Liquidation Risk Model stopped successfully")
        except Exception:
            self.print(error("❌ Error stopping Liquidation Risk Model: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="liquidation risk model setup",
)
async def setup_liquidation_risk_model(
    config: dict[str, Any] | None = None,
) -> LiquidationRiskModel | None:
    """
    Setup and initialize Liquidation Risk Model.

    Args:
        config: Configuration dictionary

    Returns:
        LiquidationRiskModel: Initialized liquidation risk model or None if failed
    """
    try:
        if config is None:
            config = {}

        risk_model = LiquidationRiskModel(config)

        if await risk_model.initialize():
            system_logger.info("✅ Liquidation Risk Model setup completed successfully")
            return risk_model
        system_print(failed("❌ Liquidation Risk Model setup failed"))
        return None

    except Exception:
        system_print(error("❌ Error setting up Liquidation Risk Model: {e}"))
        return None
