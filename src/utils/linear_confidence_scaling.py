"""
from .logger import system_logger
Linear Confidence Scaling Utilities

This module provides linear confidence scaling functions to replace
threshold-based approaches with smooth, continuous scaling.
"""
from typing import Dict, Any, Tuple
from .logger import system_logger
import numpy as np
import logging

class LinearConfidenceScaler:
    """
    Linear confidence scaling utility that replaces threshold-based approaches
    with smooth, continuous scaling based on confidence levels.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize linear confidence scaler with configuration.
        
        Args:
            config: Configuration dictionary containing scaling parameters
        """
        self.config = config
        self.logger = system_logger.getChild('LinearConfidenceScaler')
        confidence_config = config.get('confidence', {})
        self.confidence_min_threshold = confidence_config.get('confidence_min_threshold', 0.6)
        self.confidence_max_threshold = confidence_config.get('confidence_max_threshold', 0.95)
        self.confidence_min_multiplier = confidence_config.get('confidence_min_multiplier', 0.5)
        self.confidence_max_multiplier = confidence_config.get('confidence_max_multiplier', 2.0)
        self.entry_risk_threshold = confidence_config.get('entry_risk_threshold', 0.15)
        self.profit_confidence_threshold = confidence_config.get('profit_confidence_threshold', 0.6)
        self.confidence_scaling_factor = confidence_config.get('confidence_scaling_factor', 1.0)
        self.risk_scaling_factor = confidence_config.get('risk_scaling_factor', 1.0)
        self.profit_scaling_factor = confidence_config.get('profit_scaling_factor', 1.0)
        intensity_config = config.get('intensity', {})
        self.intensity_position_multiplier = intensity_config.get('intensity_position_multiplier', 1.0)
        self.high_intensity_boost = intensity_config.get('high_intensity_boost', 1.3)
        self.low_intensity_reduction = intensity_config.get('low_intensity_reduction', 0.7)

    def calculate_linear_confidence_multiplier(self, confidence: float, intensity: float = 1.0, reliability: float = 1.0) -> float:
        """
        Calculate non-linear confidence multiplier based on confidence, intensity, and reliability.
        Uses logarithmic and fractional power transformations for more nuanced scaling.
        
        Args:
            confidence: Base confidence score (0.0 to 1.0)
            intensity: Signal intensity (0.0 to 1.0)
            reliability: Signal reliability (0.0 to 1.0)
            
        Returns:
            Non-linear confidence multiplier
        """
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Apply logarithmic transformation to confidence for more nuanced scaling
        if confidence < self.confidence_min_threshold:
            # Use exponential decay for low confidence
            base_multiplier = self.confidence_min_multiplier * np.exp(-2 * (self.confidence_min_threshold - confidence))
        else:
            # Use logarithmic scaling for higher confidence values
            normalized_confidence = (confidence - self.confidence_min_threshold) / (self.confidence_max_threshold - self.confidence_min_threshold)
            # Apply log transformation with base 2 for more gradual increase
            log_scaled = np.log2(1 + normalized_confidence)
            base_multiplier = self.confidence_min_multiplier + (self.confidence_max_multiplier - self.confidence_min_multiplier) * log_scaled
        
        intensity_multiplier = self._calculate_intensity_multiplier(intensity)
        reliability_multiplier = self._calculate_reliability_multiplier(reliability)
        final_multiplier = base_multiplier * intensity_multiplier * reliability_multiplier * self.confidence_scaling_factor
        final_multiplier = np.clip(final_multiplier, 0.1, 5.0)
        return float(final_multiplier)

    def _calculate_intensity_multiplier(self, intensity: float) -> float:
        """Calculate intensity-based multiplier using non-linear transformations."""
        intensity = np.clip(intensity, 0.0, 1.0)
        
        if intensity < 0.5:
            # Use square root transformation for low intensity (more gradual increase)
            normalized_low = intensity / 0.5
            sqrt_scaled = np.sqrt(normalized_low)
            intensity_multiplier = self.low_intensity_reduction + (1.0 - self.low_intensity_reduction) * sqrt_scaled
        else:
            # Use fractional power (0.7) for high intensity (more gradual increase than linear)
            normalized_high = (intensity - 0.5) / 0.5
            power_scaled = np.power(normalized_high, 0.7)
            intensity_multiplier = 1.0 + (self.high_intensity_boost - 1.0) * power_scaled
        
        return float(intensity_multiplier)

    def _calculate_reliability_multiplier(self, reliability: float) -> float:
        """Calculate reliability-based multiplier using non-linear transformations."""
        reliability = np.clip(reliability, 0.0, 1.0)
        
        if reliability >= 0.8:
            # Use logarithmic scaling for high reliability (diminishing returns)
            high_reliability = (reliability - 0.8) / 0.2
            log_scaled = np.log2(1 + high_reliability)
            reliability_multiplier = 1.0 + log_scaled * 0.3
        elif reliability <= 0.5:
            # Use exponential decay for low reliability
            low_reliability = reliability / 0.5
            exp_scaled = np.exp(-2 * (1 - low_reliability))
            reliability_multiplier = 0.7 + 0.3 * exp_scaled
        else:
            # Use fractional power for medium reliability
            medium_reliability = (reliability - 0.5) / 0.3
            power_scaled = np.power(medium_reliability, 0.8)
            reliability_multiplier = 1.0 + power_scaled * 0.1
        
        return float(reliability_multiplier)

    def calculate_position_size_multiplier(self, confidence: float, intensity: float = 1.0, reliability: float = 1.0, risk_score: float = 0.0) -> float:
        """
        Calculate position size multiplier using non-linear confidence scaling.
        
        Args:
            confidence: Base confidence score
            intensity: Signal intensity
            reliability: Signal reliability
            risk_score: Current risk score (0.0 to 1.0)
            
        Returns:
            Position size multiplier
        """
        confidence_multiplier = self.calculate_linear_confidence_multiplier(confidence, intensity, reliability)
        
        # Apply non-linear risk adjustment using exponential decay
        risk_adjustment = np.exp(-risk_score * self.risk_scaling_factor * 2)
        risk_adjustment = np.clip(risk_adjustment, 0.3, 1.0)
        
        # Apply non-linear intensity adjustment using square root
        intensity_adjustment = np.sqrt(self.intensity_position_multiplier)
        
        final_multiplier = confidence_multiplier * risk_adjustment * intensity_adjustment
        return float(np.clip(final_multiplier, 0.1, 3.0))

    def calculate_leverage_multiplier(self, confidence: float, intensity: float = 1.0, reliability: float = 1.0, risk_score: float = 0.0) -> float:
        """
        Calculate leverage multiplier using non-linear confidence scaling.
        
        Args:
            confidence: Base confidence score
            intensity: Signal intensity
            reliability: Signal reliability
            risk_score: Current risk score
            
        Returns:
            Leverage multiplier
        """
        confidence_multiplier = self.calculate_linear_confidence_multiplier(confidence, intensity, reliability)
        
        # Apply non-linear risk adjustment using exponential decay (more aggressive for leverage)
        risk_adjustment = np.exp(-risk_score * self.risk_scaling_factor * 3)
        risk_adjustment = np.clip(risk_adjustment, 0.2, 1.0)
        
        leverage_multiplier = confidence_multiplier * risk_adjustment
        return float(np.clip(leverage_multiplier, 0.3, 2.0))

    def should_enter_trade(self, confidence: float, profit_confidence: float, risk_score: float, intensity: float = 1.0) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if trade should be entered based on linear thresholds.
        
        Args:
            confidence: Base confidence score
            profit_confidence: Profit prediction confidence
            risk_score: Current risk score
            intensity: Signal intensity
            
        Returns:
            Tuple of (should_enter, reasoning_dict)
        """
        min_confidence_met = confidence >= self.confidence_min_threshold
        profit_confidence_met = profit_confidence >= self.profit_confidence_threshold
        risk_acceptable = risk_score <= self.entry_risk_threshold
        intensity_acceptable = intensity >= 0.3
        should_enter = all([min_confidence_met, profit_confidence_met, risk_acceptable, intensity_acceptable])
        reasoning = {'should_enter': should_enter, 'confidence_met': min_confidence_met, 'profit_confidence_met': profit_confidence_met, 'risk_acceptable': risk_acceptable, 'intensity_acceptable': intensity_acceptable, 'confidence_score': confidence, 'profit_confidence_score': profit_confidence, 'risk_score': risk_score, 'intensity_score': intensity, 'confidence_multiplier': self.calculate_linear_confidence_multiplier(confidence, intensity), 'position_multiplier': self.calculate_position_size_multiplier(confidence, intensity, 1.0, risk_score), 'leverage_multiplier': self.calculate_leverage_multiplier(confidence, intensity, 1.0, risk_score)}
        return (should_enter, reasoning)

def create_linear_confidence_scaler(config: Dict[str, Any]) -> LinearConfidenceScaler:
    """
    Factory function to create a linear confidence scaler.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LinearConfidenceScaler instance
    """
    return LinearConfidenceScaler(config)