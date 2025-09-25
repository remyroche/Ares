"""
Unified Confidence Calculation Utilities

This module provides shared confidence calculation methods for both TAS and NAS components,
ensuring consistent confidence scoring and risk assessment across the trading system.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('ConfidenceCalculator')

@dataclass
class ConfidenceMetrics:
    """Container for confidence calculation results."""
    base_confidence: float
    enhanced_confidence: float
    combined_confidence: float
    risk_adjusted_confidence: float
    final_confidence: float
    confidence_components: Dict[str, float]
    risk_factors: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class ConfidenceWeights:
    """Weights for confidence calculation."""
    base_weight: float = 0.4
    enhancement_weight: float = 0.3
    risk_weight: float = 0.2
    regime_weight: float = 0.1

class UnifiedConfidenceCalculator:
    """
    Unified confidence calculation engine for both TAS and NAS components.
    
    Provides consistent confidence scoring, risk adjustment, and enhancement
    across all trading signal generation components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified confidence calculator.
        
        Args:
            config: Configuration dictionary for confidence calculation
        """
        self.config = config or {}
        self.logger = logger.getChild('UnifiedConfidenceCalculator')
        
        # Confidence calculation parameters
        self.weights = ConfidenceWeights(
            base_weight=self.config.get('base_weight', 0.4),
            enhancement_weight=self.config.get('enhancement_weight', 0.3),
            risk_weight=self.config.get('risk_weight', 0.2),
            regime_weight=self.config.get('regime_weight', 0.1)
        )
        
        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.max_confidence = self.config.get('max_confidence', 0.95)
        self.enhancement_threshold = self.config.get('enhancement_threshold', 0.7)
        
        # Risk adjustment parameters
        self.volatility_penalty = self.config.get('volatility_penalty', 0.1)
        self.liquidation_penalty = self.config.get('liquidation_penalty', 0.2)
        self.regime_instability_penalty = self.config.get('regime_instability_penalty', 0.15)
        
        # Performance tracking
        self.confidence_calculation_count = 0
        self.calculation_times = []
        
    @handles_errors
    @traced(span_name="calculate_confidence")
    @log_execution_time()
    async def calculate_confidence(
        self,
        base_confidence: float,
        enhancement_confidence: Optional[float] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
        regime_metrics: Optional[Dict[str, float]] = None,
        signal_type: str = "both",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceMetrics:
        """
        Calculate unified confidence score for trading signals.
        
        Args:
            base_confidence: Base confidence from signal analysis
            enhancement_confidence: Confidence from NAS/TAS enhancement
            risk_metrics: Risk-related metrics for adjustment
            regime_metrics: Regime-related metrics for adjustment
            signal_type: Type of signal ("nas", "tas", or "both")
            additional_context: Additional context for calculation
            
        Returns:
            ConfidenceMetrics: Comprehensive confidence calculation result
        """
        try:
            tprint_info(f"🔄 Calculating {signal_type} confidence (base: {base_confidence:.3f})")
            
            # Validate and clamp base confidence
            base_confidence = np.clip(base_confidence, 0.0, 1.0)
            
            # Calculate enhancement confidence
            if enhancement_confidence is not None:
                enhancement_confidence = np.clip(enhancement_confidence, 0.0, 1.0)
            else:
                enhancement_confidence = base_confidence * 0.8  # Default fallback
            
            # Calculate combined confidence
            combined_confidence = self._calculate_combined_confidence(
                base_confidence, enhancement_confidence
            )
            
            # Apply risk adjustments
            risk_adjusted_confidence = self._apply_risk_adjustments(
                combined_confidence, risk_metrics, regime_metrics
            )
            
            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(
                risk_adjusted_confidence, signal_type, additional_context
            )
            
            # Create confidence components breakdown
            confidence_components = {
                'base_confidence': base_confidence,
                'enhancement_confidence': enhancement_confidence,
                'combined_confidence': combined_confidence,
                'risk_adjusted_confidence': risk_adjusted_confidence
            }
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(risk_metrics, regime_metrics)
            
            # Create result
            result = ConfidenceMetrics(
                base_confidence=base_confidence,
                enhanced_confidence=enhancement_confidence,
                combined_confidence=combined_confidence,
                risk_adjusted_confidence=risk_adjusted_confidence,
                final_confidence=final_confidence,
                confidence_components=confidence_components,
                risk_factors=risk_factors,
                metadata={
                    'calculation_timestamp': datetime.now().isoformat(),
                    'signal_type': signal_type,
                    'weights': {
                        'base_weight': self.weights.base_weight,
                        'enhancement_weight': self.weights.enhancement_weight,
                        'risk_weight': self.weights.risk_weight,
                        'regime_weight': self.weights.regime_weight
                    },
                    'additional_context': additional_context or {}
                }
            )
            
            self.confidence_calculation_count += 1
            tprint_success(f"✅ Confidence calculated: {final_confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Confidence calculation failed: {e}")
            return self._create_fallback_confidence(base_confidence)
    
    def _calculate_combined_confidence(
        self,
        base_confidence: float,
        enhancement_confidence: float
    ) -> float:
        """Calculate combined confidence from base and enhancement."""
        try:
            # Weighted combination
            combined = (
                base_confidence * self.weights.base_weight +
                enhancement_confidence * self.weights.enhancement_weight
            )
            
            # Normalize weights
            total_weight = self.weights.base_weight + self.weights.enhancement_weight
            if total_weight > 0:
                combined /= total_weight
            
            return np.clip(combined, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Combined confidence calculation failed: {e}")
            return base_confidence
    
    def _apply_risk_adjustments(
        self,
        combined_confidence: float,
        risk_metrics: Optional[Dict[str, float]],
        regime_metrics: Optional[Dict[str, float]]
    ) -> float:
        """Apply risk-based adjustments to confidence."""
        try:
            if not risk_metrics and not regime_metrics:
                return combined_confidence
            
            risk_adjustment = 1.0
            
            # Volatility penalty
            if risk_metrics and 'volatility' in risk_metrics:
                volatility = risk_metrics['volatility']
                if volatility > 0.05:  # High volatility threshold
                    penalty = min(volatility * self.volatility_penalty, 0.3)
                    risk_adjustment -= penalty
            
            # Liquidation risk penalty
            if risk_metrics and 'liquidation_risk' in risk_metrics:
                liquidation_risk = risk_metrics['liquidation_risk']
                if liquidation_risk > 0.1:  # High liquidation risk threshold
                    penalty = min(liquidation_risk * self.liquidation_penalty, 0.4)
                    risk_adjustment -= penalty
            
            # Regime instability penalty
            if regime_metrics and 'regime_stability' in regime_metrics:
                regime_stability = regime_metrics['regime_stability']
                if regime_stability < 0.5:  # Low regime stability
                    penalty = (0.5 - regime_stability) * self.regime_instability_penalty
                    risk_adjustment -= penalty
            
            # Apply risk adjustment
            risk_adjusted = combined_confidence * max(risk_adjustment, 0.1)  # Minimum 10% confidence
            
            return np.clip(risk_adjusted, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Risk adjustment failed: {e}")
            return combined_confidence
    
    def _calculate_final_confidence(
        self,
        risk_adjusted_confidence: float,
        signal_type: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate final confidence with additional adjustments."""
        try:
            final_confidence = risk_adjusted_confidence
            
            # Apply signal type specific adjustments
            if signal_type == "nas":
                # NAS signals might be more conservative
                final_confidence *= 0.95
            elif signal_type == "tas":
                # TAS signals might be more aggressive
                final_confidence *= 1.05
            
            # Apply additional context adjustments
            if additional_context:
                intensity = additional_context.get('intensity', 1.0)
                reliability = additional_context.get('reliability', 1.0)
                
                # Adjust based on intensity and reliability
                adjustment_factor = (intensity + reliability) / 2
                final_confidence *= adjustment_factor
            
            # Apply final clamping
            final_confidence = np.clip(final_confidence, self.min_confidence, self.max_confidence)
            
            return final_confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Final confidence calculation failed: {e}")
            return risk_adjusted_confidence
    
    def _calculate_risk_factors(
        self,
        risk_metrics: Optional[Dict[str, float]],
        regime_metrics: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate risk factors for confidence adjustment."""
        try:
            risk_factors = {}
            
            if risk_metrics:
                risk_factors.update({
                    'volatility_factor': risk_metrics.get('volatility', 0.0),
                    'liquidation_factor': risk_metrics.get('liquidation_risk', 0.0),
                    'market_health_factor': risk_metrics.get('market_health', 0.5)
                })
            
            if regime_metrics:
                risk_factors.update({
                    'regime_stability_factor': regime_metrics.get('regime_stability', 0.5),
                    'regime_confidence_factor': regime_metrics.get('regime_confidence', 0.5),
                    'transition_probability_factor': regime_metrics.get('transition_probability', 0.0)
                })
            
            return risk_factors
            
        except Exception as e:
            self.logger.warning(f"⚠️ Risk factors calculation failed: {e}")
            return {}
    
    def _create_fallback_confidence(self, base_confidence: float) -> ConfidenceMetrics:
        """Create fallback confidence metrics."""
        return ConfidenceMetrics(
            base_confidence=base_confidence,
            enhanced_confidence=base_confidence * 0.8,
            combined_confidence=base_confidence * 0.9,
            risk_adjusted_confidence=base_confidence * 0.8,
            final_confidence=base_confidence * 0.8,
            confidence_components={
                'base_confidence': base_confidence,
                'enhancement_confidence': base_confidence * 0.8,
                'combined_confidence': base_confidence * 0.9,
                'risk_adjusted_confidence': base_confidence * 0.8
            },
            risk_factors={},
            metadata={
                'calculation_timestamp': datetime.now().isoformat(),
                'fallback': True,
                'error': 'Confidence calculation failed'
            }
        )
    
    def calculate_weighted_confidence(
        self,
        confidences: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """Calculate weighted average of multiple confidence scores."""
        try:
            if not confidences:
                return 0.0
            
            if weights is None:
                weights = [1.0 / len(confidences)] * len(confidences)
            
            if len(confidences) != len(weights):
                weights = [1.0 / len(confidences)] * len(confidences)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Calculate weighted average
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights))
            
            return np.clip(weighted_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Weighted confidence calculation failed: {e}")
            return np.mean(confidences) if confidences else 0.0
    
    def calculate_enhancement_boost(
        self,
        base_confidence: float,
        enhancement_confidence: float,
        boost_threshold: float = 0.7
    ) -> float:
        """Calculate confidence boost from enhancement."""
        try:
            if enhancement_confidence < boost_threshold:
                return base_confidence  # No boost if enhancement is low
            
            # Calculate boost factor
            boost_factor = (enhancement_confidence - boost_threshold) / (1.0 - boost_threshold)
            boost_factor = np.clip(boost_factor, 0.0, 1.0)
            
            # Apply boost
            boosted_confidence = base_confidence + (enhancement_confidence - base_confidence) * boost_factor
            
            return np.clip(boosted_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhancement boost calculation failed: {e}")
            return base_confidence
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get confidence calculation performance metrics."""
        return {
            'total_calculations': self.confidence_calculation_count,
            'avg_calculation_time': np.mean(self.calculation_times) if self.calculation_times else 0.0,
            'weights': {
                'base_weight': self.weights.base_weight,
                'enhancement_weight': self.weights.enhancement_weight,
                'risk_weight': self.weights.risk_weight,
                'regime_weight': self.weights.regime_weight
            },
            'thresholds': {
                'min_confidence': self.min_confidence,
                'max_confidence': self.max_confidence,
                'enhancement_threshold': self.enhancement_threshold
            }
        }

# Convenience functions
def create_confidence_calculator(config: Optional[Dict[str, Any]] = None) -> UnifiedConfidenceCalculator:
    """Create a configured confidence calculator."""
    return UnifiedConfidenceCalculator(config)

async def calculate_signal_confidence(
    base_confidence: float,
    enhancement_confidence: Optional[float] = None,
    risk_metrics: Optional[Dict[str, float]] = None,
    regime_metrics: Optional[Dict[str, float]] = None,
    signal_type: str = "both",
    config: Optional[Dict[str, Any]] = None
) -> ConfidenceMetrics:
    """Calculate signal confidence with convenience function."""
    calculator = create_confidence_calculator(config)
    return await calculator.calculate_confidence(
        base_confidence, enhancement_confidence, risk_metrics, 
        regime_metrics, signal_type
    )