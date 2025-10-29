"""
Position Sizer

Simplified position sizing using ML confidence scores and Kelly criterion.
Based on existing tactician approach with ML confidence and Kelly calculations.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import math

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, tprint_debug, LogLevel
from ..config.trading_config import TradingConfig
from .leverage_manager import LeverageManager
from .risk_calculator import RiskCalculator

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
    - Integrates with LeverageManager and RiskCalculator
    """

    def __init__(self, config: TradingConfig, leverage_manager: Optional[LeverageManager] = None, risk_calculator: Optional[RiskCalculator] = None):
        self.config = config
        self.logger = logger.getChild('PositionSizer')
        
        # External dependencies (optional, can be set later)
        self.leverage_manager = leverage_manager
        self.risk_calculator = risk_calculator

        # Position sizing configuration - read from config with defaults
        self.kelly_multiplier: float = 0.25  # Kelly fraction multiplier
        self.max_position_size: float = getattr(config, 'max_position_size', 0.5)  # Maximum position size (50% of portfolio)
        self.min_position_size: float = getattr(config, 'min_position_size', 0.01)  # Minimum position size (1% of portfolio)
        self.confidence_threshold: float = 0.6  # Minimum confidence threshold
        self.ml_weight: float = 0.7  # Weight for ML-based sizing vs Kelly
        
        # Confidence multiplier from final_parameters_optimization (loaded from optimized params)
        # This is set by optimized_parameters_integration or can be set directly
        self.confidence_multiplier: float = 1.0  # Default to 1.0 if not set from optimized params
        
        # Position size rounding configuration
        self.min_order_size: float = 0.0  # Minimum order size (exchange-specific, set via config)
        self.tick_size: float = 0.0  # Tick size for rounding (exchange-specific, set via config)

        # State management
        self.is_initialized: bool = False
        self.position_sizing_history: List[Dict[str, Any]] = []

    def set_leverage_manager(self, leverage_manager: LeverageManager) -> None:
        """Set the leverage manager for integration."""
        self.leverage_manager = leverage_manager
        tprint_debug("Leverage manager set for Position Sizer")

    def set_risk_calculator(self, risk_calculator: RiskCalculator) -> None:
        """Set the risk calculator for integration."""
        self.risk_calculator = risk_calculator
        tprint_debug("Risk calculator set for Position Sizer")

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
                tprint_error("max_position_size must be greater than min_position_size")
                self.logger.error("max_position_size must be greater than min_position_size")
                return False
            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                tprint_error("kelly_multiplier must be between 0 and 1")
                self.logger.error("kelly_multiplier must be between 0 and 1")
                return False
            if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
                tprint_error("confidence_threshold must be between 0 and 1")
                self.logger.error("confidence_threshold must be between 0 and 1")
                return False
            if self.ml_weight < 0 or self.ml_weight > 1:
                tprint_error("ml_weight must be between 0 and 1")
                self.logger.error("ml_weight must be between 0 and 1")
                return False
            tprint_debug("✅ Position Sizer configuration validated")
            return True
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_inputs(
        self,
        symbol: str,
        current_price: float,
        account_balance: float,
        analyst_confidence: float,
        tactician_confidence: float
    ) -> None:
        """Validate inputs for position sizing."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"symbol must be a non-empty string, got {symbol}")
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError(f"current_price must be a positive finite number, got {current_price}")
        if not math.isfinite(account_balance) or account_balance <= 0:
            raise ValueError(f"account_balance must be a positive finite number, got {account_balance}")
        if not math.isfinite(analyst_confidence) or not (0 <= analyst_confidence <= 1):
            raise ValueError(f"analyst_confidence must be between 0 and 1, got {analyst_confidence}")
        if not math.isfinite(tactician_confidence) or not (0 <= tactician_confidence <= 1):
            raise ValueError(f"tactician_confidence must be between 0 and 1, got {tactician_confidence}")

    def _extract_confidence_levels(
        self,
        confidence_dict: Dict[str, float],
        target_levels: List[float],
        default_value: float = 0.5
    ) -> List[float]:
        """
        Extract confidence values for target levels from dictionary.
        
        Safely handles empty dictionaries and missing keys.
        """
        if not confidence_dict:
            return [default_value] * len(target_levels)
        
        confidences = []
        for level in target_levels:
            try:
                # Try to find closest key
                closest_key = None
                min_diff = float('inf')
                
                for key in confidence_dict.keys():
                    try:
                        # Try to parse key as percentage or number
                        key_value = float(str(key).replace('%', '').replace(' ', ''))
                        diff = abs(key_value - level)
                        if diff < min_diff:
                            min_diff = diff
                            closest_key = key
                    except (ValueError, AttributeError):
                        continue
                
                if closest_key is not None:
                    confidence = confidence_dict.get(closest_key, default_value)
                else:
                    confidence = default_value
                    
                confidences.append(confidence)
            except Exception as e:
                self.logger.debug(f"Error extracting confidence for level {level}: {e}")
                confidences.append(default_value)
        
        return confidences

    def _calculate_kelly_position_size(
        self,
        price_target_confidences: Dict[str, float],
        adversarial_confidences: Dict[str, float]
    ) -> float:
        """
        Calculate position size using Kelly criterion based on ML confidence scores.
        
        Kelly formula: f = (bp - q) / b
        where:
            f = fraction of capital to bet
            b = net odds (win_amount / loss_amount)
            p = win probability
            q = loss probability (1 - p)
        
        Simplified adaptation for ML:
            p = average confidence from price targets
            q = average adversarial risk (probability of loss)
            b = risk/reward ratio (estimated from confidences)
        """
        try:
            target_levels = [0.25, 0.5, 0.75, 1.0]
            
            # Extract confidences safely
            confidences = self._extract_confidence_levels(price_target_confidences, target_levels, 0.5)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Extract adversarial risks safely
            adverse_risks = self._extract_confidence_levels(adversarial_confidences, target_levels, 0.3)
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks) if adverse_risks else 0.3
            
            # Calculate win and loss probabilities
            win_probability = avg_confidence
            loss_probability = max(0.0, min(1.0, avg_adverse_risk))
            
            # Estimate risk/reward ratio from confidences
            # Higher confidence difference suggests better risk/reward
            if loss_probability > 0:
                # Estimate odds: if win prob is much higher than loss prob, better odds
                if win_probability > loss_probability:
                    # Estimated odds ratio
                    odds_ratio = win_probability / loss_probability
                else:
                    odds_ratio = 1.0
                
                # Kelly formula: f = (bp - q) / b
                # Simplified: f = p - q/b, where b = odds_ratio
                if odds_ratio > 0:
                    kelly_fraction = win_probability - (loss_probability / odds_ratio)
                else:
                    kelly_fraction = win_probability - loss_probability
            else:
                # No loss probability data, use simplified calculation
                kelly_fraction = win_probability - 0.5
            
            # Ensure Kelly fraction is non-negative
            kelly_fraction = max(0.0, kelly_fraction)
            
            # Apply Kelly multiplier and clamp to limits
            kelly_position_size = kelly_fraction * self.kelly_multiplier
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

        except Exception as e:
            tprint_warning(f"❌ Kelly position size calculation failed: {e}, using min_position_size")
            self.logger.error(f"❌ Kelly position size calculation failed: {e}")
            return self.min_position_size

    def _calculate_ml_position_size(
        self,
        price_target_confidences: Dict[str, float],
        adversarial_confidences: Dict[str, float]
    ) -> float:
        """Calculate position size based on ML confidence scores."""
        try:
            target_levels = [0.25, 0.5, 0.75, 1.0]
            
            # Extract confidences safely (reuse helper method)
            confidences = self._extract_confidence_levels(price_target_confidences, target_levels, 0.5)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Extract adversarial risks safely
            adverse_risks = self._extract_confidence_levels(adversarial_confidences, target_levels, 0.3)
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks) if adverse_risks else 0.3

            # Calculate confidence factor
            confidence_factor = avg_confidence / self.confidence_threshold if self.confidence_threshold > 0 else 1.0
            risk_factor = max(0.0, min(1.0, 1.0 - avg_adverse_risk))

            # Calculate position size
            base_position_size = self.min_position_size + (self.max_position_size - self.min_position_size) * confidence_factor * risk_factor
            return max(self.min_position_size, min(self.max_position_size, base_position_size))

        except Exception as e:
            tprint_warning(f"❌ ML position size calculation failed: {e}, using min_position_size")
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
                tprint_warning("Non-finite result in weighted position size calculation, using Kelly size")
                self.logger.warning("Non-finite result in weighted position size calculation")
                return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

            return max(self.min_position_size, min(self.max_position_size, weighted_size))

        except Exception as e:
            tprint_warning(f"❌ Weighted position size calculation failed: {e}, using Kelly size")
            self.logger.error(f"❌ Weighted position size calculation failed: {e}")
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

    def _calculate_confidence_multiplier(
        self,
        combined_confidence: float,
        intensity: float,
        reliability: float,
        risk_score: float
    ) -> float:
        """
        Calculate confidence multiplier for position size adjustment.
        
        Uses the confidence_multiplier from final_parameters_optimization (backtested).
        Applies risk adjustment based on risk_score.
        
        Args:
            combined_confidence: Combined ML confidence score
            intensity: Intensity score
            reliability: Reliability score
            risk_score: Risk score (higher = more risk)
            
        Returns:
            Confidence multiplier for position size adjustment
        """
        try:
            # Use the backtested confidence multiplier from final_parameters_optimization
            # This is set via set_confidence_multiplier() or from optimized parameters
            base_multiplier = self.confidence_multiplier
            
            # Apply risk adjustment (higher risk = lower multiplier)
            # Risk factor: 0.7 to 1.0 range (reduces multiplier by up to 30% for high risk)
            risk_factor = 1.0 - (risk_score * 0.3)
            risk_factor = max(0.7, min(1.0, risk_factor))
            
            # Final multiplier: base multiplier adjusted for risk
            final_multiplier = base_multiplier * risk_factor
            
            # Clamp to reasonable bounds
            return max(0.1, min(2.0, final_multiplier))

        except Exception as e:
            tprint_warning(f"❌ Confidence multiplier calculation failed: {e}, using default 1.0")
            self.logger.error(f"❌ Confidence multiplier calculation failed: {e}")
            return 1.0

    def _apply_position_size_modifiers(
        self,
        base_size: float,
        analyst_confidence: float,
        tactician_confidence: float
    ) -> float:
        """Apply final position size modifiers."""
        try:
            # Use raw confidence inputs to determine scaling
            confidence_values = [
                value for value in (analyst_confidence, tactician_confidence)
                if value is not None and math.isfinite(value)
            ]

            if confidence_values:
                average_confidence = sum(confidence_values) / len(confidence_values)
            else:
                average_confidence = 0.5

            # Clamp the average confidence to a sensible range (0 to 1)
            average_confidence = max(0.0, min(1.0, average_confidence))

            # Calculate confidence scale directly from the raw scores (0.8 to 1.2 window)
            # Simple linear scaling based on confidence
            confidence_scale = 0.8 + (average_confidence * 0.4)

            # Apply logarithmic adjustment with safeguarded base size
            epsilon = 1e-8
            safe_base = max(self.min_position_size, min(self.max_position_size, base_size))
            log_adjusted = math.log(safe_base + epsilon) + math.log(confidence_scale)
            adjusted = math.exp(log_adjusted)

            return max(self.min_position_size, min(self.max_position_size, adjusted))

        except Exception as e:
            tprint_warning(f"❌ Position size modifiers application failed: {e}, using base size")
            self.logger.error(f"❌ Position size modifiers application failed: {e}")
            return max(self.min_position_size, min(self.max_position_size, base_size))

    def _round_position_size(
        self,
        position_size: float,
        current_price: float,
        account_balance: float
    ) -> float:
        """
        Round position size to exchange-acceptable values.
        
        Args:
            position_size: Position size as fraction of account balance (0-1)
            current_price: Current market price
            account_balance: Account balance
            
        Returns:
            Rounded position size as fraction of account balance
        """
        try:
            # Convert fractional position size to units
            position_value = position_size * account_balance
            position_units = position_value / current_price if current_price > 0 else 0.0
            
            # Apply minimum order size constraint
            if self.min_order_size > 0 and position_units < self.min_order_size:
                position_units = self.min_order_size
            
            # Round to tick size if specified
            if self.tick_size > 0:
                position_units = round(position_units / self.tick_size) * self.tick_size
            
            # Convert back to fractional position size
            rounded_value = position_units * current_price
            rounded_size = rounded_value / account_balance if account_balance > 0 else 0.0
            
            # Ensure within limits
            return max(self.min_position_size, min(self.max_position_size, rounded_size))
            
        except Exception as e:
            tprint_warning(f"Position size rounding failed: {e}, using original size")
            self.logger.warning(f"Position size rounding failed: {e}, using original size")
            return position_size

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
        tactician_confidence: float = 0.5,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None
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
            stop_loss_price: Stop loss price (optional, for risk validation)
            volatility: Market volatility (optional, for risk validation)

        Returns:
            PositionSizeResult: Position sizing recommendation
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Position Sizer not initialized")

            # Validate inputs
            self._validate_inputs(symbol, current_price, account_balance, analyst_confidence, tactician_confidence)

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

            # Round position size to exchange requirements
            final_size = self._round_position_size(final_size, current_price, account_balance)

            # Calculate leverage using LeverageManager if available
            leverage = 1.0
            if self.leverage_manager:
                try:
                    leverage_result = await self.leverage_manager.calculate_leverage(
                        symbol=symbol,
                        ml_predictions=ml_predictions,
                        current_price=current_price,
                        account_balance=account_balance,
                        analyst_confidence=analyst_confidence,
                        tactician_confidence=tactician_confidence
                    )
                    leverage = leverage_result.recommended_leverage
                except Exception as e:
                    tprint_warning(f"Failed to calculate leverage: {e}, using default 1.0")
                    self.logger.warning(f"Failed to calculate leverage: {e}, using default 1.0")

            # Validate risk if RiskCalculator is available
            risk_warnings = []
            if self.risk_calculator and stop_loss_price:
                try:
                    # Convert fractional position size to units for risk calculation
                    position_value = final_size * account_balance
                    position_units = position_value / current_price if current_price > 0 else 0.0
                    
                    risk_validation = await self.risk_calculator.validate_position_risk(
                        position_size=position_units,
                        current_price=current_price,
                        account_balance=account_balance,
                        volatility=volatility,
                        stop_loss_price=stop_loss_price,
                        leverage=leverage
                    )
                    
                    if not risk_validation.get('is_valid', False):
                        risk_warnings = risk_validation.get('warnings', [])
                        tprint_warning(f"Position size exceeds risk limits: {risk_warnings}")
                        self.logger.warning(f"Position size exceeds risk limits: {risk_warnings}")
                        # Optionally reduce position size if risk is too high
                        if risk_validation.get('position_risk', 0) > self.risk_calculator.max_position_risk:
                            reduction_factor = self.risk_calculator.max_position_risk / risk_validation.get('position_risk', 1.0)
                            final_size = final_size * reduction_factor
                            final_size = max(self.min_position_size, min(self.max_position_size, final_size))
                            tprint_info(f"Position size reduced by risk validation to {final_size:.4f}")
                except Exception as e:
                    tprint_warning(f"Risk validation failed: {e}")
                    self.logger.warning(f"Risk validation failed: {e}")

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
                    'risk_score': risk_score,
                    'stop_loss_price': stop_loss_price,
                    'volatility': volatility,
                    'risk_warnings': risk_warnings
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
                'account_balance': account_balance,
                'leverage': leverage
            })

            # Maintain history size
            if len(self.position_sizing_history) > 100:
                self.position_sizing_history = self.position_sizing_history[-100:]

            tprint_debug(f"Position size calculated for {symbol}: {final_size:.4f} (leverage: {leverage:.1f}x)")
            self.logger.debug(f"Position size calculated for {symbol}: {final_size:.4f} (leverage: {leverage:.1f}x)")

            return result

        except Exception as e:
            tprint_error(f"❌ Position sizing failed for {symbol}: {e}")
            self.logger.error(f"❌ Position sizing failed for {symbol}: {e}")
            raise

    def get_position_sizing_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
            avg_leverage = sum(h.get('leverage', 1.0) for h in recent_history) / len(recent_history)

            return {
                'total_sizings': len(self.position_sizing_history),
                'avg_size': avg_size,
                'avg_confidence': avg_confidence,
                'kelly_usage': avg_kelly,
                'ml_usage': avg_ml,
                'avg_leverage': avg_leverage,
                'kelly_multiplier': self.kelly_multiplier,
                'ml_weight': self.ml_weight,
                'confidence_threshold': self.confidence_threshold
            }

        except Exception as e:
            tprint_error(f"❌ Performance metrics calculation failed: {e}")
            self.logger.error(f"❌ Performance metrics calculation failed: {e}")
            return {}

    def set_confidence_multiplier(self, multiplier: float) -> None:
        """
        Set the confidence multiplier from final_parameters_optimization.
        
        This multiplier is backtested and optimized, and should be used directly
        instead of calculating it from intensity/reliability factors.
        
        Args:
            multiplier: Confidence multiplier from optimized parameters
        """
        if not isinstance(multiplier, (int, float)) or multiplier <= 0:
            tprint_warning(f"Invalid confidence multiplier: {multiplier}, using default 1.0")
            self.logger.warning(f"Invalid confidence multiplier: {multiplier}, using default 1.0")
            self.confidence_multiplier = 1.0
        else:
            self.confidence_multiplier = float(multiplier)
            tprint_info(f"✅ Confidence multiplier set to {self.confidence_multiplier}")
            self.logger.info(f"✅ Confidence multiplier set to {self.confidence_multiplier}")

    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Update position sizing configuration."""
        try:
            if 'kelly_multiplier' in new_config:
                self.kelly_multiplier = new_config['kelly_multiplier']
            if 'max_position_size' in new_config:
                self.max_position_size = new_config['max_position_size']
            if 'min_position_size' in new_config:
                self.min_position_size = new_config['min_position_size']
            if 'confidence_threshold' in new_config:
                self.confidence_threshold = new_config['confidence_threshold']
            if 'ml_weight' in new_config:
                self.ml_weight = new_config['ml_weight']
            if 'confidence_multiplier' in new_config:
                self.set_confidence_multiplier(new_config['confidence_multiplier'])
            if 'min_order_size' in new_config:
                self.min_order_size = new_config['min_order_size']
            if 'tick_size' in new_config:
                self.tick_size = new_config['tick_size']
            
            tprint_success("✅ Position sizing configuration updated")
            self.logger.info("✅ Position sizing configuration updated")

        except Exception as e:
            tprint_error(f"❌ Failed to update position sizing configuration: {e}")
            self.logger.error(f"❌ Failed to update position sizing configuration: {e}")

    async def stop(self) -> None:
        """Stop position sizer."""
        try:
            tprint_info("🛑 Stopping Position Sizer...")
            self.logger.info("🛑 Stopping Position Sizer...")
            self.is_initialized = False
            tprint_success("✅ Position Sizer stopped successfully")
            self.logger.info("✅ Position Sizer stopped successfully")

        except Exception as e:
            tprint_error(f"❌ Error stopping Position Sizer: {e}")
            self.logger.error(f"❌ Error stopping Position Sizer: {e}")

# Convenience function
async def setup_position_sizer(
    config: TradingConfig,
    leverage_manager: Optional[LeverageManager] = None,
    risk_calculator: Optional[RiskCalculator] = None
) -> Optional[PositionSizer]:
    """Setup and initialize position sizer."""
    try:
        tprint_info("🔄 Setting up Position Sizer...")
        position_sizer = PositionSizer(config, leverage_manager, risk_calculator)
        success = await position_sizer.initialize()
        if success:
            tprint_success("✅ Position Sizer setup completed successfully")
            return position_sizer
        tprint_warning("⚠️ Position Sizer setup completed but initialization failed")
        return None
    except Exception as e:
        tprint_error(f"❌ Failed to setup position sizer: {e}")
        logger.error(f"❌ Failed to setup position sizer: {e}")
        return None
