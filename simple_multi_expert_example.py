#!/usr/bin/env python3
"""
Simple Multi-Expert Trading System for Transition States

This example demonstrates how to use multiple regime experts simultaneously
when the market is in transition states or between clear regimes.
"""

import math
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class RegimeType(Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    TRANSITION = "TRANSITION"
    MIXED = "MIXED"

@dataclass
class ExpertPrediction:
    expert_name: str
    prediction: float  # -1 to 1 (bearish to bullish)
    confidence: float  # 0 to 1
    weight: float  # 0 to 1
    regime_type: RegimeType
    reasoning: str

@dataclass
class TransitionState:
    primary_regime: RegimeType
    secondary_regimes: List[RegimeType]
    transition_probability: float
    regime_intensities: Dict[RegimeType, float]
    uncertainty_level: float  # 0 to 1
    is_transitioning: bool

class SimpleMultiExpertTradingSystem:
    """
    Simple system for trading with multiple regime experts during transition states.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Expert activation thresholds
        self.primary_confidence_threshold = config.get("primary_confidence_threshold", 0.7)
        self.secondary_confidence_threshold = config.get("secondary_confidence_threshold", 0.5)
        self.transition_threshold = config.get("transition_threshold", 0.6)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.4)
        
        # Expert weights for different scenarios
        self.expert_weights = {
            "transition": {
                "BULL_TREND_EXPERT": 0.3,
                "BEAR_TREND_EXPERT": 0.3,
                "MOMENTUM_EXPERT": 0.2,
                "VOLATILITY_EXPERT": 0.2
            },
            "mixed": {
                "BULL_TREND_EXPERT": 0.25,
                "BEAR_TREND_EXPERT": 0.25,
                "SIDEWAYS_EXPERT": 0.25,
                "VOLATILITY_EXPERT": 0.25
            },
            "uncertain": {
                "GENERAL_EXPERT": 0.4,
                "MOMENTUM_EXPERT": 0.3,
                "VOLATILITY_EXPERT": 0.3
            }
        }
    
    def analyze_market_state(self, hmm_probs: List[List[float]]) -> TransitionState:
        """
        Analyze current market state to determine if we're in a transition.
        
        Args:
            hmm_probs: HMM state probabilities for each regime
            
        Returns:
            TransitionState object with regime information
        """
        try:
            # Calculate regime probabilities (max probability for each timepoint)
            regime_probs = [max(probs) for probs in hmm_probs]
            
            # Calculate regime entropy (uncertainty measure)
            regime_entropy = []
            for probs in hmm_probs:
                entropy = 0
                for p in probs:
                    if p > 0:
                        entropy -= p * math.log(p)
                regime_entropy.append(entropy)
            
            # Determine primary regime
            primary_regime_idx = regime_probs.index(max(regime_probs))
            primary_regime = self._map_regime_index_to_type(primary_regime_idx)
            
            # Check for multiple high-probability regimes (transition state)
            high_prob_threshold = 0.3
            high_prob_regimes = []
            regime_intensities = {}
            
            for i, prob in enumerate(regime_probs):
                regime_type = self._map_regime_index_to_type(i)
                regime_intensities[regime_type] = prob
                
                if prob > high_prob_threshold:
                    high_prob_regimes.append(regime_type)
            
            # Determine if we're in transition
            avg_entropy = sum(regime_entropy) / len(regime_entropy)
            is_transitioning = len(high_prob_regimes) > 1 or avg_entropy > self.uncertainty_threshold
            
            # Calculate transition probability
            transition_prob = avg_entropy if is_transitioning else 0.0
            
            # Get secondary regimes (regimes with significant probability)
            secondary_regimes = [r for r in high_prob_regimes if r != primary_regime]
            
            return TransitionState(
                primary_regime=primary_regime,
                secondary_regimes=secondary_regimes,
                transition_probability=transition_prob,
                regime_intensities=regime_intensities,
                uncertainty_level=avg_entropy,
                is_transitioning=is_transitioning
            )
            
        except Exception as e:
            print(f"Error analyzing market state: {e}")
            return TransitionState(
                primary_regime=RegimeType.MIXED,
                secondary_regimes=[],
                transition_probability=0.0,
                regime_intensities={},
                uncertainty_level=1.0,
                is_transitioning=True
            )
    
    def _map_regime_index_to_type(self, regime_idx: int) -> RegimeType:
        """Map HMM regime index to regime type."""
        regime_mapping = {
            0: RegimeType.BULL_TREND,
            1: RegimeType.BEAR_TREND,
            2: RegimeType.SIDEWAYS,
            3: RegimeType.VOLATILE
        }
        return regime_mapping.get(regime_idx, RegimeType.MIXED)
    
    def get_expert_predictions(self, market_data: Dict[str, float], transition_state: TransitionState) -> List[ExpertPrediction]:
        """
        Get predictions from multiple experts based on transition state.
        
        Args:
            market_data: Current market data (simplified)
            transition_state: Current transition state analysis
            
        Returns:
            List of expert predictions
        """
        predictions = []
        
        try:
            # Determine which experts to activate
            active_experts = self._determine_active_experts(transition_state)
            
            # Get predictions from each active expert
            for expert_name in active_experts:
                prediction = self._get_expert_prediction(expert_name, market_data, transition_state)
                if prediction:
                    predictions.append(prediction)
            
            # Sort by confidence
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            return predictions
            
        except Exception as e:
            print(f"Error getting expert predictions: {e}")
            return []
    
    def _determine_active_experts(self, transition_state: TransitionState) -> List[str]:
        """Determine which experts should be activated based on transition state."""
        active_experts = []
        
        if transition_state.is_transitioning:
            # During transitions, activate multiple experts
            if transition_state.uncertainty_level > 0.7:
                # High uncertainty - use general and volatility experts
                active_experts = ["GENERAL_EXPERT", "VOLATILITY_EXPERT", "MOMENTUM_EXPERT"]
            else:
                # Moderate uncertainty - use regime-specific experts
                active_experts = ["BULL_TREND_EXPERT", "BEAR_TREND_EXPERT", "MOMENTUM_EXPERT"]
                
                # Add experts based on secondary regimes
                for regime in transition_state.secondary_regimes:
                    if regime == RegimeType.SIDEWAYS:
                        active_experts.append("SIDEWAYS_EXPERT")
                    elif regime == RegimeType.VOLATILE:
                        active_experts.append("VOLATILITY_EXPERT")
        else:
            # Clear regime - use primary expert
            if transition_state.primary_regime == RegimeType.BULL_TREND:
                active_experts = ["BULL_TREND_EXPERT"]
            elif transition_state.primary_regime == RegimeType.BEAR_TREND:
                active_experts = ["BEAR_TREND_EXPERT"]
            elif transition_state.primary_regime == RegimeType.SIDEWAYS:
                active_experts = ["SIDEWAYS_EXPERT"]
            elif transition_state.primary_regime == RegimeType.VOLATILE:
                active_experts = ["VOLATILITY_EXPERT"]
        
        return active_experts
    
    def _get_expert_prediction(self, expert_name: str, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Get prediction from a specific expert."""
        try:
            # Simulate expert predictions based on market data
            if expert_name == "BULL_TREND_EXPERT":
                prediction = self._simulate_bull_expert(market_data, transition_state)
            elif expert_name == "BEAR_TREND_EXPERT":
                prediction = self._simulate_bear_expert(market_data, transition_state)
            elif expert_name == "SIDEWAYS_EXPERT":
                prediction = self._simulate_sideways_expert(market_data, transition_state)
            elif expert_name == "VOLATILITY_EXPERT":
                prediction = self._simulate_volatility_expert(market_data, transition_state)
            elif expert_name == "MOMENTUM_EXPERT":
                prediction = self._simulate_momentum_expert(market_data, transition_state)
            elif expert_name == "GENERAL_EXPERT":
                prediction = self._simulate_general_expert(market_data, transition_state)
            else:
                return None
            
            return prediction
            
        except Exception as e:
            print(f"Error getting prediction from {expert_name}: {e}")
            return None
    
    def _simulate_bull_expert(self, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Simulate bull trend expert prediction."""
        # Calculate bullish indicators
        price_change = market_data.get('price_change', 0.0)
        momentum = market_data.get('momentum', 0.0)
        
        # Base prediction on price momentum
        prediction = max(-1.0, min(1.0, price_change * 2 + momentum * 5))
        confidence = min(abs(prediction) * 0.8 + 0.2, 1.0)
        
        # Weight based on regime intensity
        weight = transition_state.regime_intensities.get(RegimeType.BULL_TREND, 0.3)
        
        return ExpertPrediction(
            expert_name="BULL_TREND_EXPERT",
            prediction=prediction,
            confidence=confidence,
            weight=weight,
            regime_type=RegimeType.BULL_TREND,
            reasoning=f"Price momentum: {momentum:.3f}, price change: {price_change:.3f}"
        )
    
    def _simulate_bear_expert(self, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Simulate bear trend expert prediction."""
        # Calculate bearish indicators
        price_change = market_data.get('price_change', 0.0)
        momentum = market_data.get('momentum', 0.0)
        
        # Base prediction on negative price momentum
        prediction = max(-1.0, min(1.0, -price_change * 2 - momentum * 5))
        confidence = min(abs(prediction) * 0.8 + 0.2, 1.0)
        
        # Weight based on regime intensity
        weight = transition_state.regime_intensities.get(RegimeType.BEAR_TREND, 0.3)
        
        return ExpertPrediction(
            expert_name="BEAR_TREND_EXPERT",
            prediction=prediction,
            confidence=confidence,
            weight=weight,
            regime_type=RegimeType.BEAR_TREND,
            reasoning=f"Negative momentum: {momentum:.3f}, price change: {price_change:.3f}"
        )
    
    def _simulate_sideways_expert(self, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Simulate sideways expert prediction."""
        # Calculate range indicators
        price_range = market_data.get('price_range', 0.1)
        
        # Sideways expert prefers small ranges and neutral predictions
        prediction = max(-0.5, min(0.5, (0.5 - price_range) * 2))
        confidence = min(0.6, 1.0 - price_range)
        
        weight = transition_state.regime_intensities.get(RegimeType.SIDEWAYS, 0.3)
        
        return ExpertPrediction(
            expert_name="SIDEWAYS_EXPERT",
            prediction=prediction,
            confidence=confidence,
            weight=weight,
            regime_type=RegimeType.SIDEWAYS,
            reasoning=f"Price range: {price_range:.3f}, suggesting sideways movement"
        )
    
    def _simulate_volatility_expert(self, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Simulate volatility expert prediction."""
        # Calculate volatility indicators
        volatility = market_data.get('volatility', 0.02)
        vol_change = market_data.get('volatility_change', 0.0)
        
        # Volatility expert predicts based on volatility expansion/contraction
        prediction = max(-1.0, min(1.0, vol_change * 10))
        confidence = min(volatility * 50, 1.0)
        
        weight = transition_state.regime_intensities.get(RegimeType.VOLATILE, 0.3)
        
        return ExpertPrediction(
            expert_name="VOLATILITY_EXPERT",
            prediction=prediction,
            confidence=confidence,
            weight=weight,
            regime_type=RegimeType.VOLATILE,
            reasoning=f"Volatility: {volatility:.3f}, vol change: {vol_change:.3f}"
        )
    
    def _simulate_momentum_expert(self, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Simulate momentum expert prediction."""
        # Calculate momentum indicators
        short_momentum = market_data.get('short_momentum', 0.0)
        long_momentum = market_data.get('long_momentum', 0.0)
        momentum_divergence = short_momentum - long_momentum
        
        # Momentum expert focuses on momentum changes during transitions
        prediction = max(-1.0, min(1.0, momentum_divergence * 5))
        confidence = min(abs(momentum_divergence) * 2 + 0.3, 1.0)
        
        # Higher weight during transitions
        weight = 0.4 if transition_state.is_transitioning else 0.2
        
        return ExpertPrediction(
            expert_name="MOMENTUM_EXPERT",
            prediction=prediction,
            confidence=confidence,
            weight=weight,
            regime_type=RegimeType.TRANSITION,
            reasoning=f"Short momentum: {short_momentum:.3f}, long momentum: {long_momentum:.3f}, divergence: {momentum_divergence:.3f}"
        )
    
    def _simulate_general_expert(self, market_data: Dict[str, float], transition_state: TransitionState) -> ExpertPrediction:
        """Simulate general expert prediction for uncertain conditions."""
        # General expert uses multiple indicators
        price_change = market_data.get('price_change', 0.0)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # Conservative prediction
        prediction = max(-0.5, min(0.5, price_change * 0.5 + (volume_ratio - 1) * 0.2))
        confidence = 0.5  # Moderate confidence for uncertain conditions
        
        weight = 0.4  # Higher weight during high uncertainty
        
        return ExpertPrediction(
            expert_name="GENERAL_EXPERT",
            prediction=prediction,
            confidence=confidence,
            weight=weight,
            regime_type=RegimeType.MIXED,
            reasoning=f"Conservative approach: price change {price_change:.3f}, volume ratio {volume_ratio:.3f}"
        )
    
    def combine_expert_predictions(self, predictions: List[ExpertPrediction], transition_state: TransitionState) -> Dict[str, Any]:
        """
        Combine predictions from multiple experts into a final trading decision.
        
        Args:
            predictions: List of expert predictions
            transition_state: Current transition state
            
        Returns:
            Combined trading decision
        """
        try:
            if not predictions:
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "No expert predictions available",
                    "expert_contributions": {}
                }
            
            # Calculate weighted prediction
            total_weight = 0.0
            weighted_prediction = 0.0
            expert_contributions = {}
            
            for pred in predictions:
                # Adjust weight based on confidence and transition state
                adjusted_weight = pred.weight * pred.confidence
                
                if transition_state.is_transitioning:
                    # During transitions, give more weight to momentum and volatility experts
                    if pred.expert_name in ["MOMENTUM_EXPERT", "VOLATILITY_EXPERT"]:
                        adjusted_weight *= 1.2
                
                weighted_prediction += pred.prediction * adjusted_weight
                total_weight += adjusted_weight
                
                expert_contributions[pred.expert_name] = {
                    "prediction": pred.prediction,
                    "confidence": pred.confidence,
                    "weight": adjusted_weight,
                    "reasoning": pred.reasoning
                }
            
            # Normalize prediction
            if total_weight > 0:
                final_prediction = weighted_prediction / total_weight
            else:
                final_prediction = 0.0
            
            # Determine action based on prediction threshold
            action_threshold = 0.2
            if final_prediction > action_threshold:
                action = "BUY"
            elif final_prediction < -action_threshold:
                action = "SELL"
            else:
                action = "HOLD"
            
            # Calculate overall confidence
            avg_confidence = sum(pred.confidence for pred in predictions) / len(predictions)
            
            # Adjust confidence based on transition state
            if transition_state.is_transitioning:
                # Reduce confidence during transitions
                final_confidence = avg_confidence * 0.8
            else:
                final_confidence = avg_confidence
            
            return {
                "action": action,
                "prediction_value": final_prediction,
                "confidence": final_confidence,
                "transition_state": {
                    "is_transitioning": transition_state.is_transitioning,
                    "uncertainty_level": transition_state.uncertainty_level,
                    "primary_regime": transition_state.primary_regime.value,
                    "secondary_regimes": [r.value for r in transition_state.secondary_regimes]
                },
                "expert_contributions": expert_contributions,
                "active_experts": [pred.expert_name for pred in predictions],
                "reasoning": f"Combined prediction from {len(predictions)} experts during {'transition' if transition_state.is_transitioning else 'stable'} state"
            }
            
        except Exception as e:
            print(f"Error combining expert predictions: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error combining predictions: {e}",
                "expert_contributions": {}
            }
    
    def execute_trading_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the trading decision with risk management.
        
        Args:
            decision: Combined trading decision
            
        Returns:
            Execution result
        """
        try:
            action = decision["action"]
            confidence = decision["confidence"]
            
            # Risk management based on confidence and transition state
            transition_state = decision.get("transition_state", {})
            is_transitioning = transition_state.get("is_transitioning", False)
            
            # Adjust position size based on confidence and transition state
            if is_transitioning:
                # Reduce position size during transitions
                position_size_multiplier = 0.5
                risk_reason = "Reduced position size due to transition state"
            else:
                position_size_multiplier = 1.0
                risk_reason = "Normal position size for stable regime"
            
            # Further reduce if confidence is low
            if confidence < 0.6:
                position_size_multiplier *= 0.7
                risk_reason += " and low confidence"
            
            execution_result = {
                "action": action,
                "confidence": confidence,
                "position_size_multiplier": position_size_multiplier,
                "risk_reason": risk_reason,
                "expert_contributions": decision.get("expert_contributions", {}),
                "transition_state": transition_state
            }
            
            print(f"Trading decision: {action} (confidence: {confidence:.3f}, size multiplier: {position_size_multiplier:.2f})")
            print(f"Risk reason: {risk_reason}")
            
            return execution_result
            
        except Exception as e:
            print(f"Error executing trading decision: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "error": str(e)
            }

def main():
    """Example of using the simple multi-expert trading system."""
    
    # Configuration
    config = {
        "primary_confidence_threshold": 0.7,
        "secondary_confidence_threshold": 0.5,
        "transition_threshold": 0.6,
        "uncertainty_threshold": 0.4
    }
    
    # Initialize system
    trading_system = SimpleMultiExpertTradingSystem(config)
    
    # Simulate HMM probabilities (transition state)
    hmm_probs = [
        [0.3, 0.3, 0.2, 0.2],  # Mixed probabilities indicating transition
        [0.4, 0.3, 0.2, 0.1],  # Slightly more bullish
        [0.2, 0.4, 0.3, 0.1],  # Slightly more bearish
    ]
    
    # Simulate market data
    market_data = {
        'price_change': 0.02,      # 2% price increase
        'momentum': 0.01,          # Positive momentum
        'price_range': 0.15,       # Moderate price range
        'volatility': 0.025,       # Moderate volatility
        'volatility_change': 0.005, # Slight volatility increase
        'short_momentum': 0.015,   # Short-term momentum
        'long_momentum': 0.008,    # Long-term momentum
        'volume_ratio': 1.2        # 20% volume increase
    }
    
    print("🔍 Analyzing market state...")
    transition_state = trading_system.analyze_market_state(hmm_probs)
    
    print(f"📊 Market State Analysis:")
    print(f"   Primary Regime: {transition_state.primary_regime.value}")
    print(f"   Secondary Regimes: {[r.value for r in transition_state.secondary_regimes]}")
    print(f"   Is Transitioning: {transition_state.is_transitioning}")
    print(f"   Uncertainty Level: {transition_state.uncertainty_level:.3f}")
    print(f"   Transition Probability: {transition_state.transition_probability:.3f}")
    
    print("\n🧠 Getting expert predictions...")
    expert_predictions = trading_system.get_expert_predictions(market_data, transition_state)
    
    print(f"📈 Expert Predictions ({len(expert_predictions)} experts):")
    for pred in expert_predictions:
        print(f"   {pred.expert_name}: {pred.prediction:.3f} (confidence: {pred.confidence:.3f}, weight: {pred.weight:.3f})")
        print(f"     Reasoning: {pred.reasoning}")
    
    print("\n⚖️ Combining expert predictions...")
    combined_decision = trading_system.combine_expert_predictions(expert_predictions, transition_state)
    
    print(f"🎯 Combined Decision:")
    print(f"   Action: {combined_decision['action']}")
    print(f"   Prediction Value: {combined_decision['prediction_value']:.3f}")
    print(f"   Confidence: {combined_decision['confidence']:.3f}")
    print(f"   Reasoning: {combined_decision['reasoning']}")
    
    print("\n💼 Executing trading decision...")
    execution_result = trading_system.execute_trading_decision(combined_decision)
    
    print(f"✅ Execution Result:")
    print(f"   Action: {execution_result['action']}")
    print(f"   Position Size Multiplier: {execution_result['position_size_multiplier']:.2f}")
    print(f"   Risk Reason: {execution_result['risk_reason']}")
    
    print(f"\n🔧 Expert Contributions:")
    for expert, contrib in execution_result['expert_contributions'].items():
        print(f"   {expert}: {contrib['prediction']:.3f} (weight: {contrib['weight']:.3f})")

if __name__ == "__main__":
    main()