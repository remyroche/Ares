"""
Signal Validator

Validates trading signals for quality, consistency, and risk compliance.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from .signal_pipeline import SignalGenerationResult

logger = system_logger.getChild('SignalValidator')

@dataclass
class ValidationResult:
    """Signal validation result."""
    is_valid: bool
    confidence_score: float
    risk_score: float
    quality_score: float
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]

class SignalValidator:
    """
    Signal validator for trading signals.
    
    Validates signals for:
    - Quality and consistency
    - Risk compliance
    - Confidence thresholds
    - Historical performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.getChild('SignalValidator')
        
        # Validation parameters
        self.min_confidence_threshold = 0.6
        self.max_risk_threshold = 0.8
        self.quality_threshold = 0.7
        
        # State management
        self.is_initialized = False
        self.validation_history: List[ValidationResult] = []
    
    @handles_errors
    async def initialize(self) -> bool:
        """Initialize signal validator."""
        try:
            self.logger.info("Initializing Signal Validator...")
            
            # Load validation configuration
            self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
            self.max_risk_threshold = self.config.get('max_risk_threshold', 0.8)
            self.quality_threshold = self.config.get('quality_threshold', 0.7)
            
            self.is_initialized = True
            self.logger.info("✅ Signal Validator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Signal Validator: {e}")
            return False
    
    @handles_errors
    async def validate_signal(self, signal_result: SignalGenerationResult) -> ValidationResult:
        """
        Validate a trading signal.
        
        Args:
            signal_result: Signal generation result to validate
            
        Returns:
            ValidationResult: Validation result with scores and issues
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Signal Validator not initialized")
            
            warnings = []
            errors = []
            
            # Validate confidence
            confidence_score = self._validate_confidence(signal_result, warnings, errors)
            
            # Validate risk
            risk_score = self._validate_risk(signal_result, warnings, errors)
            
            # Validate quality
            quality_score = self._validate_quality(signal_result, warnings, errors)
            
            # Overall validation
            is_valid = len(errors) == 0 and confidence_score >= self.min_confidence_threshold
            
            # Create result
            result = ValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                risk_score=risk_score,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors,
                metadata={
                    'timestamp': datetime.now(),
                    'symbol': signal_result.symbol,
                    'signal': signal_result.final_signal,
                    'validation_thresholds': {
                        'min_confidence': self.min_confidence_threshold,
                        'max_risk': self.max_risk_threshold,
                        'quality': self.quality_threshold
                    }
                }
            )
            
            # Store in history
            self.validation_history.append(result)
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]
            
            self.logger.debug(f"Signal validated for {signal_result.symbol}: valid={is_valid}, confidence={confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Signal validation failed: {e}")
            raise
    
    def _validate_confidence(self, signal_result: SignalGenerationResult, warnings: List[str], errors: List[str]) -> float:
        """Validate signal confidence."""
        try:
            confidence = signal_result.final_confidence
            
            # Check minimum confidence threshold
            if confidence < self.min_confidence_threshold:
                errors.append(f"Confidence below threshold: {confidence:.3f} < {self.min_confidence_threshold:.3f}")
            
            # Check regime confidence
            regime_confidence = signal_result.hmm_output.confidence
            if regime_confidence < 0.5:
                warnings.append(f"Low regime confidence: {regime_confidence:.3f}")
            
            # Check analyst confidence
            analyst_confidence = signal_result.analyst_output.analyst_confidence
            if analyst_confidence < 0.5:
                warnings.append(f"Low analyst confidence: {analyst_confidence:.3f}")
            
            # Check tactician confidence
            tactician_confidence = signal_result.tactician_output.tactician_confidence
            if tactician_confidence < 0.5:
                warnings.append(f"Low tactician confidence: {tactician_confidence:.3f}")
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Confidence validation failed: {e}")
            return 0.0
    
    def _validate_risk(self, signal_result: SignalGenerationResult, warnings: List[str], errors: List[str]) -> float:
        """Validate signal risk."""
        try:
            # Calculate risk score based on various factors
            risk_factors = []
            
            # Regime transition risk
            transition_prob = signal_result.hmm_output.transition_probability
            if transition_prob > 0.3:
                risk_factors.append(transition_prob)
                warnings.append(f"High regime transition probability: {transition_prob:.3f}")
            
            # Signal strength risk
            signal_strength = signal_result.signal_strength
            if signal_strength < 0.3:
                risk_factors.append(0.8)  # High risk for weak signals
                warnings.append(f"Weak signal strength: {signal_strength:.3f}")
            
            # Market health risk
            market_health = signal_result.analyst_output.market_health_score
            if market_health < 0.3:
                risk_factors.append(0.7)
                warnings.append(f"Poor market health: {market_health:.3f}")
            
            # Calculate overall risk score
            if risk_factors:
                risk_score = sum(risk_factors) / len(risk_factors)
            else:
                risk_score = 0.2  # Low risk if no risk factors
            
            # Check risk threshold
            if risk_score > self.max_risk_threshold:
                errors.append(f"Risk above threshold: {risk_score:.3f} > {self.max_risk_threshold:.3f}")
            
            return risk_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Risk validation failed: {e}")
            return 0.5
    
    def _validate_quality(self, signal_result: SignalGenerationResult, warnings: List[str], errors: List[str]) -> float:
        """Validate signal quality."""
        try:
            quality_factors = []
            
            # Data quality
            data_points = signal_result.metadata.get('data_points', 0)
            if data_points < 100:
                quality_factors.append(0.3)
                warnings.append(f"Low data points: {data_points}")
            else:
                quality_factors.append(0.9)
            
            # Model consistency
            analyst_conf = signal_result.analyst_output.analyst_confidence
            tactician_conf = signal_result.tactician_output.tactician_confidence
            confidence_diff = abs(analyst_conf - tactician_conf)
            
            if confidence_diff > 0.3:
                quality_factors.append(0.4)
                warnings.append(f"High confidence difference: {confidence_diff:.3f}")
            else:
                quality_factors.append(0.8)
            
            # Signal consistency
            if signal_result.final_signal == 'hold':
                quality_factors.append(0.6)  # Hold signals are lower quality
            else:
                quality_factors.append(0.8)
            
            # Calculate overall quality score
            quality_score = sum(quality_factors) / len(quality_factors)
            
            # Check quality threshold
            if quality_score < self.quality_threshold:
                warnings.append(f"Low quality score: {quality_score:.3f} < {self.quality_threshold:.3f}")
            
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality validation failed: {e}")
            return 0.5
    
    def get_validation_history(self, limit: int = 100) -> List[ValidationResult]:
        """Get recent validation history."""
        return self.validation_history[-limit:] if self.validation_history else []
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation performance metrics."""
        try:
            if not self.validation_history:
                return {
                    'total_validations': 0,
                    'validation_rate': 0.0,
                    'avg_confidence': 0.0,
                    'avg_risk': 0.0,
                    'avg_quality': 0.0
                }
            
            recent_validations = self.validation_history[-100:]  # Last 100 validations
            
            valid_count = sum(1 for v in recent_validations if v.is_valid)
            validation_rate = valid_count / len(recent_validations)
            
            avg_confidence = sum(v.confidence_score for v in recent_validations) / len(recent_validations)
            avg_risk = sum(v.risk_score for v in recent_validations) / len(recent_validations)
            avg_quality = sum(v.quality_score for v in recent_validations) / len(recent_validations)
            
            return {
                'total_validations': len(self.validation_history),
                'validation_rate': validation_rate,
                'avg_confidence': avg_confidence,
                'avg_risk': avg_risk,
                'avg_quality': avg_quality,
                'thresholds': {
                    'min_confidence': self.min_confidence_threshold,
                    'max_risk': self.max_risk_threshold,
                    'quality': self.quality_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Validation metrics calculation failed: {e}")
            return {}
    
    async def stop(self):
        """Stop signal validator."""
        try:
            self.logger.info("🛑 Stopping Signal Validator...")
            self.is_initialized = False
            self.logger.info("✅ Signal Validator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Signal Validator: {e}")

# Convenience function
async def setup_signal_validator(config: Dict[str, Any]) -> Optional[SignalValidator]:
    """Setup and initialize signal validator."""
    try:
        validator = SignalValidator(config)
        success = await validator.initialize()
        if success:
            return validator
        return None
    except Exception as e:
        logger.error(f"❌ Failed to setup signal validator: {e}")
        return None