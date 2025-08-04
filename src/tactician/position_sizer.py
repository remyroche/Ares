# src/tactician/position_sizer.py

"""
Simplified Position Sizer for high leverage trading.
Uses ML confidence scores and Kelly criterion for position sizing.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class PositionSizer:
    """
    Simplified position sizer that uses ML confidence scores and Kelly criterion
    for position sizing decisions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PositionSizer")

        # Load configuration
        self.sizing_config: dict[str, Any] = self.config.get("position_sizing", {})
        self.kelly_multiplier: float = self.sizing_config.get("kelly_multiplier", 0.25)
        self.max_position_size: float = self.sizing_config.get("max_position_size", 0.5)
        self.min_position_size: float = self.sizing_config.get("min_position_size", 0.01)
        self.confidence_threshold: float = self.sizing_config.get("confidence_threshold", 0.6)
        
        # Component weights
        self.ml_weight: float = self.sizing_config.get("ml_weight", 0.7)
        self.kelly_weight: float = self.sizing_config.get("kelly_weight", 0.3)
        
        self.is_initialized: bool = False
        self.position_sizing_history: List[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position sizer configuration"),
            AttributeError: (False, "Missing required sizing parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position sizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the position sizer."""
        try:
            self.logger.info("Initializing position sizer...")

            # Validate configuration
            if not self._validate_configuration():
                return False

            self.is_initialized = True
            self.logger.info("✅ Position sizer initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing position sizer: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            required_keys = ["kelly_multiplier", "max_position_size", "min_position_size"]
            for key in required_keys:
                if key not in self.sizing_config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.max_position_size <= self.min_position_size:
                self.logger.error("max_position_size must be greater than min_position_size")
                return False

            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                self.logger.error("kelly_multiplier must be between 0 and 1")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for position sizing"),
            AttributeError: (None, "Sizer not properly initialized"),
        },
        default_return=None,
        context="position sizing calculation",
    )
    async def calculate_position_size(
        self,
        ml_predictions: dict[str, Any],
        current_price: float = 0.0,
        account_balance: float = 1000.0,
    ) -> dict[str, Any]:
        """
        Calculate position size using ML confidence scores and Kelly criterion.
        
        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            current_price: Current market price
            account_balance: Account balance for position sizing
            
        Returns:
            dict[str, Any]: Position sizing analysis
        """
        try:
            if not self.is_initialized:
                self.logger.error("Position sizer not initialized")
                return None

            self.logger.info("Calculating position size using ML intelligence...")

            # Extract ML confidence scores
            movement_confidence = ml_predictions.get("movement_confidence_scores", {})
            adverse_movement_risks = ml_predictions.get("adverse_movement_risks", {})
            directional_confidence = ml_predictions.get("directional_confidence", {})

            # Calculate base Kelly criterion position size
            kelly_position_size = self._calculate_kelly_position_size(movement_confidence, adverse_movement_risks)

            # Calculate ML-based position size
            ml_position_size = self._calculate_ml_position_size(movement_confidence, adverse_movement_risks)

            # Calculate weighted position size
            final_position_size = self._calculate_weighted_position_size(
                kelly_position_size,
                ml_position_size,
            )

            # Create position sizing analysis
            sizing_analysis = {
                "timestamp": datetime.now(),
                "current_price": current_price,
                "account_balance": account_balance,
                "kelly_position_size": kelly_position_size,
                "ml_position_size": ml_position_size,
                "final_position_size": final_position_size,
                "ml_confidence_scores": movement_confidence,
                "adverse_movement_risks": adverse_movement_risks,
                "directional_confidence": directional_confidence,
                "sizing_reason": self._generate_sizing_reason(
                    final_position_size, kelly_position_size, ml_position_size, movement_confidence, adverse_movement_risks
                ),
            }

            # Store in history
            self.position_sizing_history.append(sizing_analysis)
            if len(self.position_sizing_history) > 100:  # Keep last 100 entries
                self.position_sizing_history = self.position_sizing_history[-100:]

            self.logger.info(f"✅ Position size calculated: {final_position_size:.4f}")
            return sizing_analysis

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return None

    def _calculate_kelly_position_size(
        self, movement_confidence: dict[str, float], adverse_movement_risks: dict[str, float]
    ) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
        try:
            # Get average confidence for target levels (0.5% to 2.0%)
            target_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            
            for level in target_levels:
                closest_level = min(movement_confidence.keys(), key=lambda x: abs(float(x) - level))
                confidence = movement_confidence.get(closest_level, 0.5)
                confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            # Get average adverse risk
            adverse_risks = []
            for level in target_levels:
                closest_level = min(adverse_movement_risks.keys(), key=lambda x: abs(float(x) - level))
                risk = adverse_movement_risks.get(closest_level, 0.3)
                adverse_risks.append(risk)
            
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)
            
            # Kelly criterion: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            # For our case: b = 1 (1:1 odds), p = avg_confidence, q = avg_adverse_risk
            kelly_fraction = avg_confidence - avg_adverse_risk
            
            # Apply Kelly multiplier for conservative sizing
            kelly_position_size = kelly_fraction * self.kelly_multiplier
            
            # Ensure within bounds
            kelly_position_size = max(self.min_position_size, min(self.max_position_size, kelly_position_size))
            
            return kelly_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            return self.min_position_size

    def _calculate_ml_position_size(
        self, movement_confidence: dict[str, float], adverse_movement_risks: dict[str, float]
    ) -> float:
        """Calculate position size based on ML confidence scores."""
        try:
            # Get average confidence for target levels (0.5% to 2.0%)
            target_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            
            for level in target_levels:
                closest_level = min(movement_confidence.keys(), key=lambda x: abs(float(x) - level))
                confidence = movement_confidence.get(closest_level, 0.5)
                confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            # Get average adverse risk
            adverse_risks = []
            for level in target_levels:
                closest_level = min(adverse_movement_risks.keys(), key=lambda x: abs(float(x) - level))
                risk = adverse_movement_risks.get(closest_level, 0.3)
                adverse_risks.append(risk)
            
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)
            
            # Calculate ML-based position size
            # Higher confidence and lower risk = larger position
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk
            
            # Base position size calculation
            base_position_size = self.min_position_size + (self.max_position_size - self.min_position_size) * confidence_factor * risk_factor
            
            # Ensure within bounds
            ml_position_size = max(self.min_position_size, min(self.max_position_size, base_position_size))
            
            return ml_position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating ML position size: {e}")
            return self.min_position_size

    def _calculate_weighted_position_size(
        self,
        kelly_position_size: float,
        ml_position_size: float,
    ) -> float:
        """Calculate weighted position size using Kelly criterion and ML confidence."""
        try:
            # Calculate weighted position size
            weighted_size = (
                kelly_position_size * self.kelly_weight +
                ml_position_size * self.ml_weight
            ) / (self.kelly_weight + self.ml_weight)
            
            return max(self.min_position_size, min(self.max_position_size, weighted_size))
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted position size: {e}")
            return kelly_position_size

    def _generate_sizing_reason(
        self,
        final_position_size: float,
        kelly_position_size: float,
        ml_position_size: float,
        movement_confidence: dict[str, float],
        adverse_movement_risks: dict[str, float],
    ) -> str:
        """Generate reason for position sizing decision."""
        try:
            # Get average confidence and risk
            key_levels = [0.5, 1.0, 1.5, 2.0]
            confidences = []
            risks = []
            
            for level in key_levels:
                closest_confidence = min(movement_confidence.keys(), key=lambda x: abs(float(x) - level))
                closest_risk = min(adverse_movement_risks.keys(), key=lambda x: abs(float(x) - level))
                confidences.append(movement_confidence.get(closest_confidence, 0.5))
                risks.append(adverse_movement_risks.get(closest_risk, 0.3))
            
            avg_confidence = sum(confidences) / len(confidences)
            avg_risk = sum(risks) / len(risks)
            
            if final_position_size >= self.max_position_size * 0.8:
                return f"Maximum position size due to high confidence ({avg_confidence:.2f}) and low risk ({avg_risk:.2f})"
            elif final_position_size >= self.max_position_size * 0.5:
                return f"Large position size based on Kelly criterion ({kelly_position_size:.3f}) and ML confidence ({ml_position_size:.3f})"
            elif final_position_size >= self.min_position_size * 2:
                return f"Moderate position size with balanced risk-reward profile"
            else:
                return f"Conservative position size due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})"
                
        except Exception as e:
            self.logger.error(f"Error generating sizing reason: {e}")
            return "Position size calculated using ML intelligence and Kelly criterion"

    def get_position_sizing_history(self, limit: Optional[int] = None) -> List[dict[str, Any]]:
        """Get position sizing history."""
        if limit:
            return self.position_sizing_history[-limit:]
        return self.position_sizing_history.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position sizer cleanup",
    )
    async def stop(self) -> None:
        """Stop the position sizer."""
        try:
            self.logger.info("Stopping position sizer...")
            self.is_initialized = False
            self.logger.info("✅ Position sizer stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping position sizer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position sizer setup",
)
async def setup_position_sizer(
    config: dict[str, Any] | None = None,
) -> PositionSizer | None:
    """
    Setup position sizer.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[PositionSizer]: Initialized position sizer or None
    """
    try:
        if config is None:
            config = {}

        position_sizer = PositionSizer(config)
        
        if await position_sizer.initialize():
            return position_sizer
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up position sizer: {e}")
        return None
