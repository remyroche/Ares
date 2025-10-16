"""
Position Sizer

Simplified position sizing using ML confidence scores and Kelly criterion.
Based on existing tactician approach with ML confidence and Kelly calculations.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import math

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel
from ..config.trading_config import TradingConfig

logger = system_logger.getChild('PositionSizer')

@dataclass
class PositionSizeResult:
    """Position sizing result."""
    symbol: str
    recommended_size: float
    max_size: float
    min_size: float
    leverage: float
    confidence: float
    kelly_size: float
    ml_size: float
    sizing_method: str
    metadata: Dict[str, Any]

class PositionSizer:
    """
    Simplified position sizing engine using ML confidence scores and Kelly criterion.

    Based on existing tactician approach:
    - Uses ML confidence scores for position sizing
    - Implements Kelly criterion for optimal position sizing
    - Simple leverage management with configurable limits
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('PositionSizer')

        # Position sizing configuration
        self.kelly_multiplier: float = 0.25  # Kelly fraction multiplier
        self.max_position_size: float = 0.5  # Maximum position size (50% of portfolio)
        self.min_position_size: float = 0.01  # Minimum position size (1% of portfolio)
        self.confidence_threshold: float = 0.6  # Minimum confidence threshold
        self.ml_weight: float = 0.7  # Weight for ML-based sizing vs Kelly

        # State management
        self.is_initialized: bool = False
        self.position_sizing_history: list[dict[str, Any]] = []

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize position sizer."""
        try:
            tprint_info("🔄 Initializing Position Sizer...")
            self.logger.info("Initializing Position Sizer...")

            # Validate configuration
            if not self._validate_configuration():
                tprint_error("❌ Position Sizer configuration validation failed")
                return False

            tprint_success("✅ Position Sizer configuration validated")

            self.is_initialized = True
            tprint_success("✅ Position Sizer initialized successfully")
            self.logger.info("✅ Position Sizer initialized successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Position Sizer: {e}")
            self.logger.error(f"❌ Failed to initialize Position Sizer: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            if self.max_position_size <= self.min_position_size:
                self.logger.error("max_position_size must be greater than min_position_size")
                return False
            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                self.logger.error("kelly_multiplier must be between 0 and 1")
                return False
            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                self.logger.error("confidence_threshold must be between 0 and 1")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handles_errors
    @log_execution_time()
    @traced(span_name="calculate_position_size")
    async def calculate_position_size(
        self,
        symbol: str,
        ml_predictions: Dict[str, Any],
        current_price: float,
        account_balance: float,
        analyst_confidence: float = 0.5,
        tactician_confidence: float = 0.5
    ) -> PositionSizeResult:
        """
        Calculate position size using ML confidence scores and Kelly criterion.

        Args:
            symbol: Trading symbol
            ml_predictions: ML confidence predictions
            current_price: Current market price
            account_balance: Account balance for position sizing
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score

        Returns:
            PositionSizeResult: Position sizing recommendation
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Position Sizer not initialized")

            # Extract ML predictions
            combined_confidence = ml_predictions.get('combined_confidence', 0.5)
            price_target_confidences = ml_predictions.get('price_target_confidences', {})
            adversarial_confidences = ml_predictions.get('adversarial_confidences', {})
            intensity = ml_predictions.get('intensity', 1.0)
            reliability = ml_predictions.get('reliability', 1.0)
            risk_score = ml_predictions.get('risk_score', 0.0)

            # Calculate Kelly position size
            kelly_size = self._calculate_kelly_position_size(price_target_confidences, adversarial_confidences)

            # Calculate ML-based position size
            ml_size = self._calculate_ml_position_size(price_target_confidences, adversarial_confidences)

            # Calculate weighted position size
            base_size = self._calculate_weighted_position_size(kelly_size, ml_size)

            # Apply confidence multiplier
            confidence_multiplier = self._calculate_confidence_multiplier(
                combined_confidence, intensity, reliability, risk_score
            )
            confidence_adjusted_size = base_size * confidence_multiplier

            # Apply final modifiers
            final_size = self._apply_position_size_modifiers(
                confidence_adjusted_size, analyst_confidence, tactician_confidence
            )

            # Calculate leverage
            leverage = self._calculate_leverage(final_size, account_balance)

            # Create result
            result = PositionSizeResult(
                symbol=symbol,
                recommended_size=final_size,
                max_size=self.max_position_size,
                min_size=self.min_position_size,
                leverage=leverage,
                confidence=combined_confidence,
                kelly_size=kelly_size,
                ml_size=ml_size,
                sizing_method="ml_kelly_hybrid",
                metadata={
                    'current_price': current_price,
                    'account_balance': account_balance,
                    'analyst_confidence': analyst_confidence,
                    'tactician_confidence': tactician_confidence,
                    'confidence_multiplier': confidence_multiplier,
                    'intensity': intensity,
                    'reliability': reliability,
                    'risk_score': risk_score
                }
            )

            # Store in history
            self.position_sizing_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'final_size': final_size,
                'kelly_size': kelly_size,
                'ml_size': ml_size,
                'combined_confidence': combined_confidence,
                'current_price': current_price,
                'account_balance': account_balance
            })

            # Maintain history size
            if len(self.position_sizing_history) > 100:
                self.position_sizing_history = self.position_sizing_history[-100:]

            self.logger.debug(f"Position size calculated for {symbol}: {final_size:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Position sizing failed for {symbol}: {e}")
            raise

    def _calculate_kelly_position_size(self, price_target_confidences: Dict[str, float], adversarial_confidences: Dict[str, float]) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
        try:
            # Simplified Kelly calculation based on ML confidence scores
            # This is a simplified version of the Kelly criterion adapted for ML predictions

            # Get average confidence from price targets
            target_levels = [0.25, 0.5, 0.75, 1.0]
            confidences = []
            for level in target_levels:
                # Find closest confidence level
                closest_level = min(price_target_confidences.keys(),
                                  key=lambda x: abs(float(x.replace('%', '')) - level))
                confidence = price_target_confidences.get(closest_level, 0.5)
                confidences.append(confidence)

            avg_confidence = sum(confidences) / len(confidences)

            # Get average risk from adversarial confidences
            adverse_risks = []
            for level in target_levels:
                closest_level = min(adversarial_confidences.keys(),
                                  key=lambda x: abs(float(x.replace('%', '')) - level))
                risk = adversarial_confidences.get(closest_level, 0.3)
                adverse_risks.append(risk)

            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)

            # Kelly formula: f = (bp - q) / b
            # where b = odds (1/avg_adverse_risk), p = win probability (avg_confidence), q = loss probability (avg_adverse_risk)
            if avg_adverse_risk > 0:
                odds = 1.0 / avg_adverse_risk
                kelly_fraction = (odds * avg_confidence - avg_adverse_risk) / odds
            else:
                kelly_fraction = avg_confidence - 0.5  # Simplified fallback

            # Apply Kelly multiplier and clamp to limits
            kelly_position_size = kelly_fraction * self.kelly_multiplier
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

        except Exception as e:
            self.logger.error(f"❌ Kelly position size calculation failed: {e}")
            return self.min_position_size

    def _calculate_ml_position_size(self, price_target_confidences: Dict[str, float], adversarial_confidences: Dict[str, float]) -> float:
        """Calculate position size based on ML confidence scores."""
        try:
            target_levels = [0.25, 0.5, 0.75, 1.0]
            confidences = []
            for level in target_levels:
                closest_level = min(price_target_confidences.keys(),
                                  key=lambda x: abs(float(x.replace('%', '')) - level))
                confidence = price_target_confidences.get(closest_level, 0.5)
                confidences.append(confidence)

            avg_confidence = sum(confidences) / len(confidences)

            adverse_risks = []
            for level in target_levels:
                closest_level = min(adversarial_confidences.keys(),
                                  key=lambda x: abs(float(x.replace('%', '')) - level))
                risk = adversarial_confidences.get(closest_level, 0.3)
                adverse_risks.append(risk)

            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)

            # Calculate confidence factor
            confidence_factor = avg_confidence / self.confidence_threshold if self.confidence_threshold > 0 else 1.0
            risk_factor = 1.0 - avg_adverse_risk

            # Ensure risk factor is positive
            risk_factor = max(0.0, min(1.0, risk_factor))

            # Calculate position size
            base_position_size = self.min_position_size + (self.max_position_size - self.min_position_size) * confidence_factor * risk_factor
            return max(self.min_position_size, min(self.max_position_size, base_position_size))

        except Exception as e:
            self.logger.error(f"❌ ML position size calculation failed: {e}")
            return self.min_position_size

    def _calculate_weighted_position_size(self, kelly_position_size: float, ml_position_size: float) -> float:
        """Calculate weighted position size using logarithmic computations."""
        try:
            # Use logarithmic computations to prevent multiplicative compounding
            log_kelly = math.log(kelly_position_size) if kelly_position_size > 0 else math.log(self.min_position_size)
            log_ml = math.log(ml_position_size) if ml_position_size > 0 else math.log(self.min_position_size)

            # Weighted average of log values
            weighted_log = (1 - self.ml_weight) * log_kelly + self.ml_weight * log_ml
            weighted_size = math.exp(weighted_log)

            # Ensure result is finite
            if not math.isfinite(weighted_size):
                self.logger.warning("Non-finite result in weighted position size calculation")
                return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

            return max(self.min_position_size, min(self.max_position_size, weighted_size))

        except Exception as e:
            self.logger.error(f"❌ Weighted position size calculation failed: {e}")
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

    def _calculate_confidence_multiplier(self, combined_confidence: float, intensity: float, reliability: float, risk_score: float) -> float:
        """Calculate confidence multiplier for position size adjustment."""
        try:
            # Base multiplier from combined confidence
            base_multiplier = 0.5 + (combined_confidence * 0.5)  # 0.5 to 1.0

            # Adjust for intensity and reliability
            intensity_factor = 0.8 + (intensity * 0.4)  # 0.8 to 1.2
            reliability_factor = 0.8 + (reliability * 0.4)  # 0.8 to 1.2

            # Adjust for risk score (higher risk = lower multiplier)
            risk_factor = 1.0 - (risk_score * 0.3)  # 0.7 to 1.0

            # Combine all factors
            final_multiplier = base_multiplier * intensity_factor * reliability_factor * risk_factor

            return max(0.1, min(2.0, final_multiplier))  # Clamp between 0.1 and 2.0

        except Exception as e:
            self.logger.error(f"❌ Confidence multiplier calculation failed: {e}")
            return 1.0

    def _apply_position_size_modifiers(self, base_size: float, analyst_confidence: float, tactician_confidence: float) -> float:
        """Apply final position size modifiers."""
        try:
            # Use raw confidence inputs to determine scaling
            confidence_values = [value for value in (analyst_confidence, tactician_confidence)
                                 if value is not None and math.isfinite(value)]

            if confidence_values:
                average_confidence = sum(confidence_values) / len(confidence_values)
            else:
                average_confidence = 0.5

            # Clamp the average confidence to a sensible range (0 to 1)
            average_confidence = max(0.0, min(1.0, average_confidence))

            # Calculate confidence scale directly from the raw scores (0.8 to 1.2 window)
            confidence_scale = 0.8 + 0.4 * average_confidence

            # Apply logarithmic adjustment with safeguarded base size
            epsilon = 1e-8
            safe_base = max(self.min_position_size, min(self.max_position_size, base_size))
            log_adjusted = math.log(safe_base + epsilon) + math.log(confidence_scale)
            adjusted = math.exp(log_adjusted)

            return max(self.min_position_size, min(self.max_position_size, adjusted))

        except Exception as e:
            self.logger.error(f"❌ Position size modifiers application failed: {e}")
            return max(self.min_position_size, min(self.max_position_size, base_size))

    def _calculate_leverage(self, position_size: float, account_balance: float) -> float:
        """Calculate leverage for position."""
        try:
            if account_balance <= 0:
                return 1.0

            leverage = position_size / account_balance
            return min(leverage, 10.0)  # Cap at 10x leverage

        except Exception as e:
            self.logger.error(f"❌ Leverage calculation failed: {e}")
            return 1.0

    def get_position_sizing_history(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get position sizing history."""
        if limit:
            return self.position_sizing_history[-limit:]
        return self.position_sizing_history.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for position sizing."""
        try:
            if not self.position_sizing_history:
                return {
                    'total_sizings': 0,
                    'avg_size': 0.0,
                    'avg_confidence': 0.0,
                    'kelly_usage': 0.0,
                    'ml_usage': 0.0
                }

            recent_history = self.position_sizing_history[-50:]  # Last 50 sizings

            avg_size = sum(h['final_size'] for h in recent_history) / len(recent_history)
            avg_confidence = sum(h['combined_confidence'] for h in recent_history) / len(recent_history)
            avg_kelly = sum(h['kelly_size'] for h in recent_history) / len(recent_history)
            avg_ml = sum(h['ml_size'] for h in recent_history) / len(recent_history)

            return {
                'total_sizings': len(self.position_sizing_history),
                'avg_size': avg_size,
                'avg_confidence': avg_confidence,
                'kelly_usage': avg_kelly,
                'ml_usage': avg_ml,
                'kelly_multiplier': self.kelly_multiplier,
                'ml_weight': self.ml_weight,
                'confidence_threshold': self.confidence_threshold
            }

        except Exception as e:
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {}

    async def stop(self):
        """Stop position sizer."""
        try:
            self.logger.info("🛑 Stopping Position Sizer...")
            self.is_initialized = False
            self.logger.info("✅ Position Sizer stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping Position Sizer: {e}")

# Convenience function
async def setup_position_sizer(config: TradingConfig) -> Optional[PositionSizer]:
    """Setup and initialize position sizer."""
    try:
        position_sizer = PositionSizer(config)
        success = await position_sizer.initialize()
        if success:
            return position_sizer
        return None
    except Exception as e:
        logger.error(f"❌ Failed to setup position sizer: {e}")
        return None
