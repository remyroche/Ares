# src/analyst/liquidation_risk_model.py
from typing import Any

from src.utils.error_handler import (
    handle_data_processing_errors,
    handle_errors,
)
from src.utils.logger import system_logger


class ProbabilisticLiquidationRiskModel:
    """
    Calculates a probabilistic Liquidation Safety Score (LSS) based on market conditions,
    position size, leverage, and historical volatility.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = system_logger.getChild("LiquidationRiskModel")

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return=50.0,
        context="calculate_lss",
    )
    def calculate_lss(
        self,
        current_price: float,
        position_size: float,
        leverage: int,
        side: str,
        atr: float,
        market_volatility: float = None,
        account_balance: float = None,
    ) -> float:
        """
        Calculate Liquidation Safety Score (LSS) - higher is safer.

        Args:
            current_price: Current asset price
            position_size: Position size in base currency
            leverage: Current leverage
            side: 'long' or 'short'
            atr: Average True Range
            market_volatility: Optional market volatility metric
            account_balance: Optional account balance for additional risk assessment

        Returns:
            float: LSS score between 0-100 (higher = safer)
        """
        try:
            if current_price <= 0 or position_size <= 0 or leverage <= 0:
                self.logger.warning("Invalid parameters for LSS calculation")
                return 50.0

            # Calculate base liquidation distance
            liquidation_distance = self._calculate_liquidation_distance(
                current_price,
                position_size,
                leverage,
                side,
            )

            # Calculate volatility risk
            volatility_risk = self._calculate_volatility_risk(atr, market_volatility)

            # Calculate position size risk
            position_risk = self._calculate_position_risk(
                position_size,
                account_balance,
            )

            # Calculate leverage risk
            leverage_risk = self._calculate_leverage_risk(leverage)

            # Combine risks into final LSS
            lss = self._combine_risk_factors(
                liquidation_distance,
                volatility_risk,
                position_risk,
                leverage_risk,
            )

            self.logger.info(
                f"LSS calculated: {lss:.2f} "
                f"(Distance: {liquidation_distance:.4f}, "
                f"Vol Risk: {volatility_risk:.2f}, "
                f"Pos Risk: {position_risk:.2f}, "
                f"Lev Risk: {leverage_risk:.2f})",
            )

            return max(0.0, min(100.0, lss))

        except Exception as e:
            self.logger.error(f"Error calculating LSS: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_liquidation_distance",
    )
    def _calculate_liquidation_distance(
        self,
        current_price: float,
        position_size: float,
        leverage: int,
        side: str,
    ) -> float:
        """Calculate the distance to liquidation price."""
        try:
            # Calculate liquidation price
            if side.lower() == "long":
                liquidation_price = current_price * (1 - 1 / leverage)
                distance = (current_price - liquidation_price) / current_price
            elif side.lower() == "short":
                liquidation_price = current_price * (1 + 1 / leverage)
                distance = (liquidation_price - current_price) / current_price
            else:
                self.logger.warning(f"Invalid side: {side}")
                return 0.0

            return distance

        except Exception as e:
            self.logger.error(f"Error calculating liquidation distance: {e}")
            return 0.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_volatility_risk",
    )
    def _calculate_volatility_risk(
        self,
        atr: float,
        market_volatility: float = None,
    ) -> float:
        """Calculate risk based on volatility."""
        try:
            # Use ATR as base volatility measure
            volatility_measure = atr if atr > 0 else 0.01

            # Normalize volatility (higher volatility = higher risk)
            # Assuming 5% daily volatility is "normal"
            normalized_volatility = volatility_measure / 0.05

            # Convert to risk score (0-100, higher = more risky)
            volatility_risk = min(100.0, normalized_volatility * 50)

            return volatility_risk

        except Exception as e:
            self.logger.error(f"Error calculating volatility risk: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_position_risk",
    )
    def _calculate_position_risk(
        self,
        position_size: float,
        account_balance: float = None,
    ) -> float:
        """Calculate risk based on position size relative to account."""
        try:
            if account_balance is None or account_balance <= 0:
                # Default risk if no account balance provided
                return 25.0

            # Calculate position size as percentage of account
            position_pct = (position_size / account_balance) * 100

            # Higher position percentage = higher risk
            if position_pct <= 1.0:
                risk = 10.0  # Very small position
            elif position_pct <= 5.0:
                risk = 25.0  # Small position
            elif position_pct <= 10.0:
                risk = 50.0  # Medium position
            elif position_pct <= 20.0:
                risk = 75.0  # Large position
            else:
                risk = 100.0  # Very large position

            return risk

        except Exception as e:
            self.logger.error(f"Error calculating position risk: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_leverage_risk",
    )
    def _calculate_leverage_risk(self, leverage: int) -> float:
        """Calculate risk based on leverage."""
        try:
            # Higher leverage = higher risk
            if leverage <= 5:
                risk = 10.0  # Very low leverage
            elif leverage <= 10:
                risk = 25.0  # Low leverage
            elif leverage <= 25:
                risk = 50.0  # Medium leverage
            elif leverage <= 50:
                risk = 75.0  # High leverage
            else:
                risk = 100.0  # Very high leverage

            return risk

        except Exception as e:
            self.logger.error(f"Error calculating leverage risk: {e}")
            return 50.0

    @handle_data_processing_errors(default_return=50.0, context="combine_risk_factors")
    def _combine_risk_factors(
        self,
        liquidation_distance: float,
        volatility_risk: float,
        position_risk: float,
        leverage_risk: float,
    ) -> float:
        """Combine all risk factors into final LSS score."""
        try:
            # Weights for different risk factors
            weights = self.config.get(
                "lss_weights",
                {
                    "liquidation_distance": 0.4,
                    "volatility_risk": 0.25,
                    "position_risk": 0.2,
                    "leverage_risk": 0.15,
                },
            )

            # Convert liquidation distance to safety score (higher distance = safer)
            distance_safety = min(100.0, liquidation_distance * 1000)  # Scale factor

            # Calculate weighted average of safety scores
            lss = (
                distance_safety * weights["liquidation_distance"]
                + (100.0 - volatility_risk) * weights["volatility_risk"]
                + (100.0 - position_risk) * weights["position_risk"]
                + (100.0 - leverage_risk) * weights["leverage_risk"]
            )

            return lss

        except Exception as e:
            self.logger.error(f"Error combining risk factors: {e}")
            return 50.0

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="get_risk_metrics",
    )
    def get_risk_metrics(
        self,
        current_price: float,
        position_size: float,
        leverage: int,
        side: str,
        atr: float,
    ) -> dict[str, Any]:
        """
        Get comprehensive risk metrics for a position.

        Returns:
            Dict containing various risk metrics
        """
        try:
            lss = self.calculate_lss(current_price, position_size, leverage, side, atr)

            # Calculate liquidation price
            if side.lower() == "long":
                liquidation_price = current_price * (1 - 1 / leverage)
            else:
                liquidation_price = current_price * (1 + 1 / leverage)

            # Calculate distance to liquidation
            distance_to_liquidation = (
                abs(current_price - liquidation_price) / current_price
            )

            return {
                "lss": lss,
                "liquidation_price": liquidation_price,
                "distance_to_liquidation": distance_to_liquidation,
                "leverage": leverage,
                "position_size": position_size,
                "side": side,
                "current_price": current_price,
                "atr": atr,
            }

        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}







# src/analyst/liquidation_risk_model.py

"""
Liquidation Risk Model for high leverage trading.
Simplified model that uses ML confidence scores and volatility indicators.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class LiquidationRiskModel:
    """
    Simplified liquidation risk model for high leverage trading.
    Uses ML confidence scores and volatility indicators to determine liquidation thresholds.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("LiquidationRiskModel")

        # Load configuration
        self.liquidation_config: dict[str, Any] = self.config.get("liquidation_risk", {})
        self.max_leverage: float = self.liquidation_config.get("max_leverage", 100.0)
        self.min_leverage: float = self.liquidation_config.get("min_leverage", 10.0)
        self.leverage_steps: List[float] = self.liquidation_config.get("leverage_steps", [1.0, 2.0, 3.0, 5.0, 10.0])
        self.volatility_multiplier: float = self.liquidation_config.get("volatility_multiplier", 1.5)
        self.confidence_threshold: float = self.liquidation_config.get("confidence_threshold", 0.7)
        
        # Risk parameters
        self.risk_free_rate: float = self.liquidation_config.get("risk_free_rate", 0.02)
        self.position_sizing_multiplier: float = self.liquidation_config.get("position_sizing_multiplier", 0.25)
        
        self.is_initialized: bool = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid liquidation risk model configuration"),
            AttributeError: (False, "Missing required liquidation parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="liquidation risk model initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the liquidation risk model."""
        try:
            self.logger.info("Initializing liquidation risk model...")

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Initialize liquidation thresholds
            await self._initialize_liquidation_thresholds()

            self.is_initialized = True
            self.logger.info("✅ Liquidation risk model initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing liquidation risk model: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate liquidation risk model configuration."""
        try:
            required_keys = ["max_leverage", "min_leverage"]
            for key in required_keys:
                if key not in self.liquidation_config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.max_leverage <= self.min_leverage:
                self.logger.error("max_leverage must be greater than min_leverage")
                return False
                
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_liquidation_distance",
    )
    def _calculate_liquidation_distance(
        self,
        current_price: float,
        position_size: float,
        leverage: int,
        side: str,
    ) -> float:
        """Calculate the distance to liquidation price."""
        try:
            # Calculate liquidation price
            if side.lower() == "long":
                liquidation_price = current_price * (1 - 1 / leverage)
                distance = (current_price - liquidation_price) / current_price
            elif side.lower() == "short":
                liquidation_price = current_price * (1 + 1 / leverage)
                distance = (liquidation_price - current_price) / current_price
            else:
                self.logger.warning(f"Invalid side: {side}")
                return 0.0

            return distance_to_liquidation

        except Exception as e:
            self.logger.error(f"Error calculating liquidation distance: {e}")
            return 0.0

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
        current_price: float,
        ml_predictions: dict[str, Any],
        volatility_indicators: dict[str, float],
        target_direction: str = "long",  # "long" or "short"
    ) -> dict[str, Any]:
        """
        Calculate liquidation risk based on ML predictions and volatility indicators.
        
        Args:
            current_price: Current market price
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            volatility_indicators: Pre-computed volatility indicators
            target_direction: Target direction ("long" or "short")
            
        Returns:
            dict[str, Any]: Liquidation risk analysis
        """
        try:
            if not self.is_initialized:
                self.logger.error("Liquidation risk model not initialized")
                return None

            self.logger.info(f"Calculating liquidation risk for {target_direction} position...")

            # Extract ML predictions
            movement_confidence = ml_predictions.get("movement_confidence_scores", {})
            adverse_movement_risks = ml_predictions.get("adverse_movement_risks", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})

            # Get volatility indicators
            current_volatility = volatility_indicators.get("current_volatility", 0.02)
            volatility_ratio = volatility_indicators.get("volatility_ratio", 1.0)

            # Calculate liquidation risk analysis
            liquidation_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "target_direction": target_direction,
                "volatility_indicators": volatility_indicators,
                "risk_adjusted_leverage": {},
            }

            # Calculate liquidation thresholds for each leverage level
            for leverage in self.leverage_steps:
                leverage_key = f"leverage_{leverage}"
                base_threshold = self.liquidation_thresholds.get(leverage_key, 1.0 / leverage)
                
                # Adjust threshold based on volatility
                volatility_adjusted_threshold = base_threshold * (current_volatility / 0.02) * self.volatility_multiplier
                
                # Calculate risk-adjusted threshold based on ML predictions
                ml_confidence = self._get_ml_confidence_for_leverage(leverage, movement_confidence)
                adverse_risk = self._get_adverse_risk_for_leverage(leverage, adverse_movement_risks)
                
                # Risk adjustment factor
                risk_factor = 1.0 + (1.0 - ml_confidence) + adverse_risk
                risk_adjusted_threshold = volatility_adjusted_threshold * risk_factor
                
                # Calculate safe leverage for this threshold
                safe_leverage = max(self.min_leverage, min(self.max_leverage, 1.0 / risk_adjusted_threshold))
                
                liquidation_analysis["liquidation_thresholds"][leverage_key] = {
                    "leverage": leverage,
                    "base_threshold": base_threshold,
                    "volatility_adjusted_threshold": volatility_adjusted_threshold,
                    "risk_adjusted_threshold": risk_adjusted_threshold,
                    "ml_confidence": ml_confidence,
                    "adverse_risk": adverse_risk,
                    "risk_factor": risk_factor,
                }
                                
                # Risk-adjusted leverage recommendations
                liquidation_analysis["risk_adjusted_leverage"][leverage_key] = {
                    "leverage": leverage,
                    "recommended_leverage": safe_leverage,
                    "confidence_level": "high" if ml_confidence >= 0.7 else "medium" if ml_confidence >= 0.5 else "low",
                    "risk_level": "low" if adverse_risk <= 0.3 else "medium" if adverse_risk <= 0.6 else "high",
                }
                
                # Position recommendations
                should_enter = ml_confidence >= self.confidence_threshold and adverse_risk <= 0.5
                
                liquidation_analysis["position_recommendations"][leverage_key] = {
                    "should_enter": should_enter,
                    "confidence": ml_confidence,
                    "adverse_risk": adverse_risk,
                    "reason": self._generate_position_reason(should_enter, ml_confidence, adverse_risk),
                }

            self.logger.info("✅ Liquidation risk calculation completed successfully")
            return liquidation_analysis

        except Exception as e:
            self.logger.error(f"Error calculating liquidation risk: {e}")
            return None


    def _generate_position_reason(self, should_enter: bool, confidence: float, adverse_risk: float) -> str:
        """Generate reason for position recommendation."""
        try:
            if should_enter:
                if confidence >= 0.8 and adverse_risk <= 0.2:
                    return f"Strong entry signal (confidence: {confidence:.2f}, risk: {adverse_risk:.2f})"
                elif confidence >= 0.6 and adverse_risk <= 0.4:
                    return f"Moderate entry signal (confidence: {confidence:.2f}, risk: {adverse_risk:.2f})"
                else:
                    return f"Weak entry signal (confidence: {confidence:.2f}, risk: {adverse_risk:.2f})"
            else:
                if confidence < 0.5:
                    return f"Low confidence ({confidence:.2f})"
                elif adverse_risk > 0.5:
                    return f"High adverse risk ({adverse_risk:.2f})"
                else:
                    return f"Unfavorable conditions (confidence: {confidence:.2f}, risk: {adverse_risk:.2f})"
                    
        except Exception as e:
            self.logger.error(f"Error generating position reason: {e}")
            return "Unable to determine reason"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="liquidation risk model cleanup",
    )
    async def stop(self) -> None:
        """Stop the liquidation risk model."""
        try:
            self.logger.info("Stopping liquidation risk model...")
            self.is_initialized = False
            self.logger.info("✅ Liquidation risk model stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping liquidation risk model: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="liquidation risk model setup",
)
async def setup_liquidation_risk_model(
    config: dict[str, Any] | None = None,
) -> LiquidationRiskModel | None:
    """
    Setup liquidation risk model.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[LiquidationRiskModel]: Initialized liquidation risk model or None
    """
    try:
        if config is None:
            config = {}

        liquidation_model = LiquidationRiskModel(config)
        
        if await liquidation_model.initialize():
            return liquidation_model
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up liquidation risk model: {e}")
        return None
