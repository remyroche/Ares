"""Confidence-Based Position Entry Logic for Step 17 Optimization.

This module implements advanced position entry logic that considers:
1. Confidence levels for both 25% and 50% barrier predictions
2. Dynamic timeframe weighting based on confidence
3. Optimized entry conditions using both barrier levels
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger


@dataclass
class BarrierPrediction:
    """Container for barrier prediction results."""
    barrier_type: str  # "barrier_25_25" or "barrier_50_50"
    timeframe: str  # "1m" or "5m"
    confidence: float
    predicted_outcome: int  # 1 for success, -1 for failure, 0 for neutral
    expected_profit: float
    risk_score: float


@dataclass
class EntryDecision:
    """Container for position entry decision."""
    should_enter: bool
    position_size_multiplier: float
    selected_timeframe: str
    selected_barriers: Tuple[float, float]
    confidence_score: float
    risk_adjusted_confidence: float
    reasoning: str


class ConfidenceBasedEntryLogic:
    """Implements confidence-based position entry logic with optimized barrier selection."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize confidence-based entry logic."""
        self.config = config
        self.logger = system_logger.getChild("ConfidenceBasedEntryLogic")
        
        # Entry configuration
        self.entry_config = config.get("confidence_based_entry", {})
        
        # Base thresholds (to be optimized by step17)
        self.min_combined_confidence = self.entry_config.get("min_combined_confidence", 0.65)
        self.min_individual_confidence = self.entry_config.get("min_individual_confidence", 0.55)
        self.confidence_difference_threshold = self.entry_config.get("confidence_difference_threshold", 0.2)
        
        # Confidence weights (to be optimized)
        self.barrier_25_weight = self.entry_config.get("barrier_25_weight", 0.4)
        self.barrier_50_weight = self.entry_config.get("barrier_50_weight", 0.6)
        self.timeframe_1m_weight = self.entry_config.get("timeframe_1m_weight", 0.5)
        self.timeframe_5m_weight = self.entry_config.get("timeframe_5m_weight", 0.5)
        
        # Risk adjustment factors
        self.volatility_penalty = self.entry_config.get("volatility_penalty", 0.1)
        self.regime_confidence_boost = self.entry_config.get("regime_confidence_boost", 0.05)
        
        # Position sizing factors
        self.high_confidence_size_boost = self.entry_config.get("high_confidence_size_boost", 1.5)
        self.low_confidence_size_reduction = self.entry_config.get("low_confidence_size_reduction", 0.5)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="evaluate position entry"
    )
    @traced(span_name="ConfidenceEntry.evaluate")
    def evaluate_position_entry(
        self,
        barrier_predictions: List[BarrierPrediction],
        market_context: Dict[str, Any],
        regime_info: Dict[str, Any]
    ) -> Optional[EntryDecision]:
        """
        Evaluate whether to enter a position based on confidence levels.
        
        Args:
            barrier_predictions: List of predictions for different barrier/timeframe combinations
            market_context: Current market conditions
            regime_info: Current regime information
            
        Returns:
            EntryDecision object or None if no entry
        """
        try:
            # Organize predictions by barrier type and timeframe
            predictions_dict = self._organize_predictions(barrier_predictions)
            
            # Calculate confidence scores for each combination
            confidence_scores = self._calculate_confidence_scores(
                predictions_dict, market_context, regime_info
            )
            
            # Apply dynamic optimization based on confidence patterns
            optimized_weights = self._optimize_weights_by_confidence(confidence_scores)
            
            # Calculate combined confidence using optimized weights
            combined_confidence = self._calculate_combined_confidence(
                confidence_scores, optimized_weights
            )
            
            # Make entry decision
            entry_decision = self._make_entry_decision(
                confidence_scores, 
                combined_confidence,
                optimized_weights,
                market_context
            )
            
            return entry_decision
            
        except Exception as e:
            self.logger.error(f"Error evaluating position entry: {e}")
            return None
    
    def _organize_predictions(
        self, 
        barrier_predictions: List[BarrierPrediction]
    ) -> Dict[str, Dict[str, BarrierPrediction]]:
        """Organize predictions by barrier type and timeframe."""
        organized = {
            "barrier_25_25": {},
            "barrier_50_50": {}
        }
        
        for pred in barrier_predictions:
            organized[pred.barrier_type][pred.timeframe] = pred
        
        return organized
    
    def _calculate_confidence_scores(
        self,
        predictions_dict: Dict[str, Dict[str, BarrierPrediction]],
        market_context: Dict[str, Any],
        regime_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate adjusted confidence scores for each prediction."""
        scores = {}
        
        for barrier_type, timeframe_preds in predictions_dict.items():
            for timeframe, pred in timeframe_preds.items():
                key = f"{barrier_type}_{timeframe}"
                
                # Base confidence
                base_confidence = pred.confidence
                
                # Adjust for market volatility
                volatility = market_context.get("volatility", 0.01)
                volatility_adjustment = -self.volatility_penalty * (volatility / 0.01 - 1)
                
                # Adjust for regime confidence
                regime_confidence = regime_info.get("confidence", 0.5)
                regime_adjustment = self.regime_confidence_boost * (regime_confidence - 0.5)
                
                # Adjust for risk score
                risk_adjustment = -0.1 * pred.risk_score
                
                # Calculate final adjusted confidence
                adjusted_confidence = np.clip(
                    base_confidence + volatility_adjustment + regime_adjustment + risk_adjustment,
                    0, 1
                )
                
                scores[key] = adjusted_confidence
        
        return scores
    
    def _optimize_weights_by_confidence(
        self, 
        confidence_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Dynamically optimize weights based on confidence patterns."""
        
        # Extract scores
        barrier_25_1m = confidence_scores.get("barrier_25_25_1m", 0)
        barrier_25_5m = confidence_scores.get("barrier_25_25_5m", 0)
        barrier_50_1m = confidence_scores.get("barrier_50_50_1m", 0)
        barrier_50_5m = confidence_scores.get("barrier_50_50_5m", 0)
        
        # Calculate average confidences
        barrier_25_avg = (barrier_25_1m + barrier_25_5m) / 2
        barrier_50_avg = (barrier_50_1m + barrier_50_5m) / 2
        timeframe_1m_avg = (barrier_25_1m + barrier_50_1m) / 2
        timeframe_5m_avg = (barrier_25_5m + barrier_50_5m) / 2
        
        # Optimize barrier weights based on relative confidence
        if barrier_50_avg > barrier_25_avg * 1.2:
            # 50% barriers much more confident
            barrier_50_weight = 0.7
            barrier_25_weight = 0.3
        elif barrier_25_avg > barrier_50_avg * 1.2:
            # 25% barriers much more confident (rare but possible)
            barrier_50_weight = 0.3
            barrier_25_weight = 0.7
        else:
            # Similar confidence - use configured weights
            barrier_50_weight = self.barrier_50_weight
            barrier_25_weight = self.barrier_25_weight
        
        # Optimize timeframe weights based on relative confidence
        if timeframe_1m_avg > timeframe_5m_avg * 1.1:
            # 1m timeframe more confident
            timeframe_1m_weight = 0.6
            timeframe_5m_weight = 0.4
        elif timeframe_5m_avg > timeframe_1m_avg * 1.1:
            # 5m timeframe more confident
            timeframe_1m_weight = 0.4
            timeframe_5m_weight = 0.6
        else:
            # Similar confidence - equal weights
            timeframe_1m_weight = 0.5
            timeframe_5m_weight = 0.5
        
        return {
            "barrier_25_weight": barrier_25_weight,
            "barrier_50_weight": barrier_50_weight,
            "timeframe_1m_weight": timeframe_1m_weight,
            "timeframe_5m_weight": timeframe_5m_weight
        }
    
    def _calculate_combined_confidence(
        self,
        confidence_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate combined confidence using optimized weights."""
        
        # Extract individual scores
        scores = {
            "barrier_25_25_1m": confidence_scores.get("barrier_25_25_1m", 0),
            "barrier_25_25_5m": confidence_scores.get("barrier_25_25_5m", 0),
            "barrier_50_50_1m": confidence_scores.get("barrier_50_50_1m", 0),
            "barrier_50_50_5m": confidence_scores.get("barrier_50_50_5m", 0)
        }
        
        # Calculate weighted combinations
        barrier_25_combined = (
            scores["barrier_25_25_1m"] * weights["timeframe_1m_weight"] +
            scores["barrier_25_25_5m"] * weights["timeframe_5m_weight"]
        )
        
        barrier_50_combined = (
            scores["barrier_50_50_1m"] * weights["timeframe_1m_weight"] +
            scores["barrier_50_50_5m"] * weights["timeframe_5m_weight"]
        )
        
        # Final combined confidence
        combined = (
            barrier_25_combined * weights["barrier_25_weight"] +
            barrier_50_combined * weights["barrier_50_weight"]
        )
        
        return combined
    
    def _make_entry_decision(
        self,
        confidence_scores: Dict[str, float],
        combined_confidence: float,
        optimized_weights: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> EntryDecision:
        """Make final entry decision based on confidence analysis."""
        
        # Check if all individual confidences meet minimum threshold
        all_above_minimum = all(
            score >= self.min_individual_confidence 
            for score in confidence_scores.values()
        )
        
        # Check if combined confidence meets threshold
        combined_meets_threshold = combined_confidence >= self.min_combined_confidence
        
        # Check confidence spread (we want consistent confidence across barriers)
        confidence_values = list(confidence_scores.values())
        confidence_spread = max(confidence_values) - min(confidence_values)
        consistent_confidence = confidence_spread <= self.confidence_difference_threshold
        
        # Determine if we should enter
        should_enter = (
            all_above_minimum and 
            combined_meets_threshold and 
            consistent_confidence
        )
        
        # Select optimal timeframe based on confidence
        timeframe_1m_score = (
            confidence_scores.get("barrier_25_25_1m", 0) * optimized_weights["barrier_25_weight"] +
            confidence_scores.get("barrier_50_50_1m", 0) * optimized_weights["barrier_50_weight"]
        )
        timeframe_5m_score = (
            confidence_scores.get("barrier_25_25_5m", 0) * optimized_weights["barrier_25_weight"] +
            confidence_scores.get("barrier_50_50_5m", 0) * optimized_weights["barrier_50_weight"]
        )
        
        selected_timeframe = "1m" if timeframe_1m_score >= timeframe_5m_score else "5m"
        
        # Use 50% barriers for actual stop/take profit (wider barriers for safety)
        # But entry decision considers both 25% and 50% confidence
        selected_barriers = (0.001, 0.0005)  # 50% of analyst barriers
        
        # Calculate position size multiplier based on confidence
        if combined_confidence >= 0.8:
            size_multiplier = self.high_confidence_size_boost
        elif combined_confidence >= 0.7:
            size_multiplier = 1.0
        else:
            size_multiplier = self.low_confidence_size_reduction
        
        # Adjust for market conditions
        if market_context.get("high_volatility", False):
            size_multiplier *= 0.8
        
        # Generate reasoning
        reasoning = self._generate_entry_reasoning(
            should_enter,
            all_above_minimum,
            combined_meets_threshold,
            consistent_confidence,
            confidence_spread,
            combined_confidence
        )
        
        return EntryDecision(
            should_enter=should_enter,
            position_size_multiplier=size_multiplier,
            selected_timeframe=selected_timeframe,
            selected_barriers=selected_barriers,
            confidence_score=combined_confidence,
            risk_adjusted_confidence=combined_confidence * (1 - market_context.get("risk_score", 0)),
            reasoning=reasoning
        )
    
    def _generate_entry_reasoning(
        self,
        should_enter: bool,
        all_above_minimum: bool,
        combined_meets_threshold: bool,
        consistent_confidence: bool,
        confidence_spread: float,
        combined_confidence: float
    ) -> str:
        """Generate human-readable reasoning for entry decision."""
        
        if should_enter:
            return (
                f"Entry approved: Combined confidence {combined_confidence:.1%} exceeds threshold. "
                f"All barrier predictions consistent (spread: {confidence_spread:.1%})."
            )
        else:
            reasons = []
            if not all_above_minimum:
                reasons.append("some predictions below minimum confidence")
            if not combined_meets_threshold:
                reasons.append(f"combined confidence {combined_confidence:.1%} below threshold")
            if not consistent_confidence:
                reasons.append(f"inconsistent predictions (spread: {confidence_spread:.1%})")
            
            return f"Entry rejected: {', '.join(reasons)}."
    
    def get_optimizable_parameters(self) -> Dict[str, Tuple[float, float]]:
        """Get parameters that can be optimized by step17."""
        return {
            "min_combined_confidence": (0.55, 0.75),
            "min_individual_confidence": (0.45, 0.65),
            "confidence_difference_threshold": (0.1, 0.3),
            "barrier_25_weight": (0.2, 0.5),
            "barrier_50_weight": (0.5, 0.8),
            "volatility_penalty": (0.05, 0.2),
            "regime_confidence_boost": (0.0, 0.1),
            "high_confidence_size_boost": (1.2, 2.0),
            "low_confidence_size_reduction": (0.3, 0.7)
        }