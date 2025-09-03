# src/analyst/liquidation_risk_model.py
from src.utils.logger import system_logger
from typing import Any
from src.core.decorators import handles_errors
import pandas as pd
import logging
import datetime as datetime
import asyncio
from src.utils.centralized_decorators_simple import (
    comprehensive_data_validation,
    validate_data_quality,
    with_tracing_span,
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
        self.logger = logging.getLogger(self.__class__.__name__)
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

    @handles_errors(
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
                self.logger.error("Invalid configuration for liquidation risk model")
                return False

            self.is_initialized = True
            self.logger.info("Liquidation Risk Model initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Liquidation Risk Model: {e}")
            return False

    @handles_errors(fallback=None)
    async def _load_risk_configuration(self) -> None:
        """Load risk model configuration."""
        self.logger.info("Loading liquidation risk model configuration...")

        # Additional configuration can be loaded here
        self.logger.info("Risk model configuration loaded successfully")

    @handles_errors(fallback=False)
    def _validate_configuration(self) -> bool:
        """Validate risk model configuration."""
        try:
            if self.max_adverse_risk <= 0 or self.max_adverse_risk > 1:
                self.logger.error("max_adverse_risk must be between 0 and 1")
                return False

            if self.safe_leverage_multiplier <= 0 or self.safe_leverage_multiplier > 1:
                self.logger.error("safe_leverage_multiplier must be between 0 and 1")
                return False

            if self.max_leverage <= 0:
                self.logger.error("max_leverage must be positive")
                return False

            if self.min_leverage <= 0 or self.min_leverage > self.max_leverage:
                self.logger.error("min_leverage must be positive and <= max_leverage")
                return False

            self.logger.info("Liquidation risk model configuration validation passed")
            return True

        except Exception:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handles_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for liquidation risk calculation"),
            AttributeError: (None, "Model not properly initialized"),
        },
        default_return=None,
        context="liquidation risk calculation",
    )
    async def calculate_liquidation_risk(
        self, ml_predictions: dict[str, Any], current_price: float, target_direction: str = "long"
    ) -> dict[str, Any]:
        """
        Calculate liquidation risk and safe leverage levels.

        Args:
            ml_predictions: ML confidence predictions
            current_price: Current market price
            target_direction: Target trading direction ("long" or "short")

        Returns:
            dict: Risk assessment results
        """
        try:
            if not self.is_initialized:
                self.logger.error("Liquidation Risk Model not initialized")
                return None

            if not ml_predictions or current_price <= 0:
                self.logger.error("Invalid input data for risk calculation")
                return None

            # Extract adverse risk from ML predictions
            adverse_risk = self._extract_adverse_risk(ml_predictions, target_direction)

            # Calculate safe leverage levels
            safe_leverage = self._calculate_safe_leverage(adverse_risk, target_direction)

            # Calculate liquidation prices for different leverage levels
            liquidation_prices = self._calculate_liquidation_prices(current_price, target_direction)

            # Generate risk assessment
            risk_assessment = {
                "adverse_risk": adverse_risk, "safe_leverage": safe_leverage,
                "max_safe_leverage": self._get_max_safe_leverage(adverse_risk),
                "risk_level": self._classify_risk_level(adverse_risk),
                "recommendation": self._generate_risk_recommendation(adverse_risk, safe_leverage),
                "liquidation_prices": liquidation_prices, "target_direction": target_direction,
                "current_price": current_price,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            self.risk_assessments = risk_assessment
            self.logger.info(f"Risk assessment completed: safe leverage = {safe_leverage}x, adverse risk = {adverse_risk:.3f}")

            return risk_assessment

        except Exception:
            self.logger.error(f"Error calculating liquidation risk: {e}")
            return None

    @validate_data_quality(validation_level="WARNING")
    @with_tracing_span("adverse_risk_extraction")
    def _extract_adverse_risk(
        self, ml_predictions: dict[str, Any], target_direction: str = "long"
    ) -> float:
        """
        Extract adverse risk from ML predictions.

        Args:
            ml_predictions: ML confidence predictions
            target_direction: Target trading direction

        Returns:
            float: Adverse risk score (0-1)
        """
        try:
            # Get confidence from ML predictions
            confidence = ml_predictions.get("confidence", 0.5)

            # Get probability distributions
            increase_probs = ml_predictions.get("increase_probabilities", {})
            decrease_probs = ml_predictions.get("decrease_probabilities", {})

            if target_direction == "long":
                # For long positions, adverse risk is probability of significant decrease
                adverse_risk = (
                    sum(decrease_probs.values()) / len(decrease_probs)
                    if decrease_probs
                    else 0.5
                )
            else:
                # For short positions, adverse risk is probability of significant increase
                adverse_risk = (
                    sum(increase_probs.values()) / len(increase_probs)
                    if increase_probs
                    else 0.5
                )

            # Adjust based on confidence
            if confidence < 0.3:
                adverse_risk *= 1.5  # Increase risk for low confidence
            elif confidence > 0.7:
                adverse_risk *= 0.8  # Decrease risk for high confidence

            return max(0.0, min(1.0, adverse_risk))

        except Exception:
            self.logger.error(f"Error extracting adverse risk: {e}")
            return 0.5

    def _calculate_safe_leverage(
        self, adverse_risk: float, target_direction: str = "long"
    ) -> int:
        """
        Calculate safe leverage level based on adverse risk.

        Args:
            adverse_risk: Adverse risk score (0-1)
            target_direction: Target trading direction

        Returns:
            int: Safe leverage level
        """
        try:
            # Find the highest leverage level that can handle the adverse risk
            safe_leverage = self.min_leverage

            for leverage, max_risk in sorted(self.leverage_risk_levels.items()):
                if adverse_risk <= max_risk:
                    safe_leverage = leverage
                else:
                    break

            # Apply safety multiplier
            safe_leverage = int(safe_leverage * self.safe_leverage_multiplier)

            # Ensure within bounds
            return max(
                self.min_leverage, min(self.max_leverage, safe_leverage)
            )

        except Exception:
            self.logger.error(f"Error calculating safe leverage: {e}")
            return self.min_leverage

    def _get_max_safe_leverage(self, adverse_risk: float) -> int:
        """
        Get maximum safe leverage for given adverse risk.

        Args:
            adverse_risk: Adverse risk score (0-1)

        Returns:
            int: Maximum safe leverage
        """
        try:
            max_leverage = self.min_leverage

            for leverage, max_risk in sorted(self.leverage_risk_levels.items()):
                if adverse_risk <= max_risk:
                    max_leverage = leverage
                else:
                    break

            return max_leverage

        except Exception:
            self.logger.error(f"Error getting max safe leverage: {e}")
            return self.min_leverage

    def _classify_risk_level(self, adverse_risk: float) -> str:
        """
        Classify risk level based on adverse risk.

        Args:
            adverse_risk: Adverse risk score (0-1)

        Returns:
            str: Risk level classification
        """
        try:
            if adverse_risk <= 0.2:
                return "LOW"
            if adverse_risk <= 0.4:
                return "MEDIUM"
            if adverse_risk <= 0.6:
                return "HIGH"
            return "EXTREME"

        except Exception:
            self.logger.error(f"Error classifying risk level: {e}")
            return "UNKNOWN"

    def _generate_risk_recommendation(
        self, adverse_risk: float, safe_leverage: int
    ) -> str:
        """
        Generate risk recommendation.

        Args:
            adverse_risk: Adverse risk score (0-1)
            safe_leverage: Safe leverage level

        Returns:
            str: Risk recommendation
        """
        try:
            if adverse_risk > 0.7:
                return "AVOID_TRADING"
            if adverse_risk > 0.5:
                return "REDUCE_POSITION_SIZE"
            if safe_leverage < 20:
                return "USE_LOW_LEVERAGE"
            return "NORMAL_TRADING"

        except Exception:
            self.logger.error(f"Error generating risk recommendation: {e}")
            return "UNKNOWN"

    def _calculate_liquidation_prices(
        self, current_price: float, target_direction: str = "long"
    ) -> dict[str, float]:
        """
        Calculate liquidation prices for different leverage levels.

        Args:
            current_price: Current market price
            target_direction: Target trading direction

        Returns:
            dict: Liquidation prices for different leverage levels
        """
        try:
            liquidation_prices = {}

            for leverage in [10, 20, 30, 50, 75, 100]:
                if leverage in self.leverage_risk_levels:
                    max_adverse_move = self.leverage_risk_levels[leverage]

                    if target_direction == "long":
                        # For long positions, liquidation price is below current price
                        liquidation_price = current_price * (1 - max_adverse_move)
                    else:
                        # For short positions, liquidation price is above current price
                        liquidation_price = current_price * (1 + max_adverse_move)

                    liquidation_prices[f"{leverage}x"] = liquidation_price

            return liquidation_prices

        except Exception:
            self.logger.error(f"Error calculating liquidation prices: {e}")
            return {}

    def get_risk_assessments(self) -> dict[str, Any]:
        """Get current risk assessments."""
        return self.risk_assessments

    def get_model_status(self) -> dict[str, Any]:
        """Get model status."""
        return {
            "is_initialized": self.is_initialized, "max_leverage": self.max_leverage,
            "min_leverage": self.min_leverage, "safe_leverage_multiplier": self.safe_leverage_multiplier,
            "max_adverse_risk": self.max_adverse_risk,
        }

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Clean up liquidation risk model resources."""
        try:
            self.logger.info("Stopping Liquidation Risk Model...")
            self.is_initialized = False
            self.risk_assessments = {}
            self.logger.info("Liquidation Risk Model stopped successfully")
        except Exception:
            self.logger.error(f"Error stopping Liquidation Risk Model: {e}")
