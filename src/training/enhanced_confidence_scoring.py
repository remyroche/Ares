#!/usr/bin/env python3
"""Enhanced Confidence Scoring System.

This module provides intelligent confidence scoring based on multi-output predictions
(direction, profit, and price) to achieve more accurate trading decisions with
threshold-based filtering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from src.utils.logger import system_logger


@dataclass
class ConfidenceConfig:
    """Configuration for enhanced confidence scoring."""
    
    # Direction confidence settings
    direction_threshold: float = 0.6  # Minimum direction confidence
    direction_weight: float = 0.4     # Weight for direction in overall confidence
    
    # Profit confidence settings
    profit_threshold: float = 0.001   # Minimum expected profit (0.1%)
    profit_weight: float = 0.3        # Weight for profit in overall confidence
    profit_volatility_penalty: float = 0.1  # Penalty for high profit volatility
    
    # Price prediction settings
    price_threshold: float = 0.005    # Minimum price movement (0.5%)
    price_weight: float = 0.3         # Weight for price prediction in overall confidence
    price_confidence_decay: float = 0.95  # Decay factor for price confidence over time
    
    # Risk-adjusted settings
    risk_free_rate: float = 0.02      # Risk-free rate (2% annual)
    sharpe_threshold: float = 0.5     # Minimum Sharpe ratio
    max_drawdown_threshold: float = 0.1  # Maximum acceptable drawdown (10%)
    
    # Market regime settings
    regime_confidence_boost: float = 0.1  # Boost confidence in favorable regimes
    volatility_adjustment: bool = True    # Adjust confidence based on market volatility
    
    # Ensemble settings
    ensemble_method: str = "weighted_average"  # "weighted_average", "geometric_mean", "harmonic_mean"
    min_ensemble_confidence: float = 0.7  # Minimum ensemble confidence
    ensemble_diversity_bonus: float = 0.05  # Bonus for diverse model predictions


class EnhancedConfidenceScorer:
    """Enhanced confidence scoring system using multi-output predictions."""
    
    def __init__(self, config: ConfidenceConfig):
        self.config = config
        self.logger = system_logger.getChild("EnhancedConfidenceScorer")
        
        # Historical confidence tracking
        self.confidence_history = []
        self.prediction_history = []
        
        self.logger.info("🔧 Enhanced confidence scorer initialized")
    
    def calculate_direction_confidence(
        self,
        direction_probability: np.ndarray,
        direction_prediction: np.ndarray
    ) -> np.ndarray:
        """Calculate confidence score for direction predictions.
        
        Args:
            direction_probability: Model probability for direction (0-1)
            direction_prediction: Binary direction prediction (0/1)
            
        Returns:
            Direction confidence scores (0-1)
        """
        # Base confidence from probability
        base_confidence = np.abs(direction_probability - 0.5) * 2  # Convert to 0-1 scale
        
        # Apply threshold filtering
        threshold_mask = base_confidence >= self.config.direction_threshold
        
        # Calculate final confidence
        direction_confidence = base_confidence * threshold_mask
        
        # Add uncertainty penalty for predictions near 0.5
        uncertainty_penalty = np.exp(-10 * np.abs(direction_probability - 0.5))
        direction_confidence *= uncertainty_penalty
        
        return direction_confidence
    
    def calculate_profit_confidence(
        self,
        profit_prediction: np.ndarray,
        profit_volatility: Optional[np.ndarray] = None,
        historical_profit: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate confidence score for profit predictions.
        
        Args:
            profit_prediction: Predicted profit percentages
            profit_volatility: Historical profit volatility (optional)
            historical_profit: Historical profit data (optional)
            
        Returns:
            Profit confidence scores (0-1)
        """
        # Base confidence from absolute profit magnitude
        profit_abs = np.abs(profit_prediction)
        base_confidence = np.tanh(profit_abs * 100)  # Sigmoid-like function
        
        # Apply minimum profit threshold
        threshold_mask = profit_abs >= self.config.profit_threshold
        profit_confidence = base_confidence * threshold_mask
        
        # Apply volatility penalty if available
        if profit_volatility is not None:
            volatility_penalty = np.exp(-self.config.profit_volatility_penalty * profit_volatility)
            profit_confidence *= volatility_penalty
        
        # Apply historical consistency bonus
        if historical_profit is not None:
            # Calculate consistency with historical predictions
            if len(historical_profit) > 0:
                historical_mean = np.mean(historical_profit)
                historical_std = np.std(historical_profit)
                
                if historical_std > 0:
                    z_score = np.abs(profit_prediction - historical_mean) / historical_std
                    consistency_bonus = np.exp(-z_score / 2)  # Higher confidence for consistent predictions
                    profit_confidence *= consistency_bonus
        
        return profit_confidence
    
    def calculate_price_confidence(
        self,
        current_price: np.ndarray,
        predicted_price: np.ndarray,
        price_volatility: Optional[np.ndarray] = None,
        time_horizon: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate confidence score for price predictions.
        
        Args:
            current_price: Current price levels
            predicted_price: Predicted price levels
            price_volatility: Price volatility (optional)
            time_horizon: Prediction time horizon in periods (optional)
            
        Returns:
            Price confidence scores (0-1)
        """
        # Calculate price movement percentage
        price_movement = np.abs(predicted_price - current_price) / current_price
        
        # Base confidence from price movement magnitude
        base_confidence = np.tanh(price_movement * 50)  # Sigmoid-like function
        
        # Apply minimum price movement threshold
        threshold_mask = price_movement >= self.config.price_threshold
        price_confidence = base_confidence * threshold_mask
        
        # Note: Volatility adjustment removed as requested
        # if price_volatility is not None and self.config.volatility_adjustment:
        #     # Higher confidence for predictions in stable volatility periods
        #     volatility_factor = np.exp(-price_volatility / np.mean(price_volatility))
        #     price_confidence *= volatility_factor
        
        # Apply time decay
        if time_horizon is not None:
            decay_factor = self.config.price_confidence_decay ** time_horizon
            price_confidence *= decay_factor
        
        return price_confidence
    
    def calculate_risk_adjusted_confidence(
        self,
        direction_confidence: np.ndarray,
        profit_confidence: np.ndarray,
        price_confidence: np.ndarray,
        profit_prediction: np.ndarray,
        risk_metrics: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Calculate simplified confidence score (no weighting since all from same model).
        
        Args:
            direction_confidence: Direction confidence scores
            profit_confidence: Profit confidence scores
            price_confidence: Price confidence scores
            profit_prediction: Predicted profit percentages
            risk_metrics: Dictionary of risk metrics (optional) - IGNORED
            
        Returns:
            Simplified confidence scores (0-1)
        """
        # Since all predictions come from the same model, use simple average
        # This avoids arbitrary weighting of related predictions
        simple_confidence = (direction_confidence + profit_confidence + price_confidence) / 3.0
        
        # Note: All risk adjustments removed as requested
        # No weighting since all predictions come from the same model
        
        return np.clip(simple_confidence, 0, 1)
    
    def calculate_ensemble_confidence(
        self,
        model_predictions: List[Dict[str, np.ndarray]],
        model_weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ensemble confidence from multiple model predictions.
        
        Args:
            model_predictions: List of model prediction dictionaries
            model_weights: Optional weights for each model
            
        Returns:
            Tuple of (ensemble_direction, ensemble_profit, ensemble_confidence)
        """
        if not model_predictions:
            return np.array([]), np.array([]), np.array([])
        
        # Extract predictions
        direction_probs = []
        profit_preds = []
        price_preds = []
        confidences = []
        
        for pred in model_predictions:
            if 'direction_probability' in pred:
                direction_probs.append(pred['direction_probability'])
            if 'profit_prediction' in pred:
                profit_preds.append(pred['profit_prediction'])
            if 'price_prediction' in pred:
                price_preds.append(pred['price_prediction'])
            if 'confidence' in pred:
                confidences.append(pred['confidence'])
        
        # Calculate ensemble predictions
        if direction_probs:
            if self.config.ensemble_method == "weighted_average":
                weights = model_weights or [1.0] * len(direction_probs)
                ensemble_direction_prob = np.average(direction_probs, weights=weights, axis=0)
                ensemble_direction = (ensemble_direction_prob > 0.5).astype(int)
            else:
                ensemble_direction_prob = np.mean(direction_probs, axis=0)
                ensemble_direction = (ensemble_direction_prob > 0.5).astype(int)
        else:
            ensemble_direction_prob = np.array([])
            ensemble_direction = np.array([])
        
        if profit_preds:
            if self.config.ensemble_method == "weighted_average":
                weights = model_weights or [1.0] * len(profit_preds)
                ensemble_profit = np.average(profit_preds, weights=weights, axis=0)
            else:
                ensemble_profit = np.mean(profit_preds, axis=0)
        else:
            ensemble_profit = np.array([])
        
        # Calculate ensemble confidence
        if confidences:
            if self.config.ensemble_method == "weighted_average":
                weights = model_weights or [1.0] * len(confidences)
                ensemble_confidence = np.average(confidences, weights=weights, axis=0)
            else:
                ensemble_confidence = np.mean(confidences, axis=0)
            
            # Add diversity bonus
            if len(confidences) > 1:
                diversity = np.std(confidences, axis=0)
                diversity_bonus = self.config.ensemble_diversity_bonus * diversity
                ensemble_confidence += diversity_bonus
                ensemble_confidence = np.clip(ensemble_confidence, 0, 1)
        else:
            ensemble_confidence = np.array([])
        
        return ensemble_direction, ensemble_profit, ensemble_confidence
    
    def calculate_comprehensive_confidence(
        self,
        direction_probability: np.ndarray,
        direction_prediction: np.ndarray,
        profit_prediction: np.ndarray,
        current_price: np.ndarray,
        predicted_price: np.ndarray,
        profit_volatility: Optional[np.ndarray] = None,
        price_volatility: Optional[np.ndarray] = None,
        risk_metrics: Optional[Dict[str, np.ndarray]] = None,
        market_regime: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Calculate comprehensive confidence score using all available information.
        
        Args:
            direction_probability: Model probability for direction
            direction_prediction: Binary direction prediction
            profit_prediction: Predicted profit percentages
            current_price: Current price levels
            predicted_price: Predicted price levels
            profit_volatility: Profit volatility (optional)
            price_volatility: Price volatility (optional)
            risk_metrics: Risk metrics dictionary (optional)
            market_regime: Market regime indicators (optional)
            
        Returns:
            Dictionary containing all confidence scores and final confidence
        """
        # Calculate individual confidence scores
        direction_confidence = self.calculate_direction_confidence(
            direction_probability, direction_prediction
        )
        
        profit_confidence = self.calculate_profit_confidence(
            profit_prediction, profit_volatility
        )
        
        price_confidence = self.calculate_price_confidence(
            current_price, predicted_price, price_volatility
        )
        
        # Calculate simplified confidence (no risk adjustments, no weighting)
        simple_confidence = self.calculate_risk_adjusted_confidence(
            direction_confidence, profit_confidence, price_confidence,
            profit_prediction, risk_metrics
        )
        
        # Note: Market regime adjustment removed as requested
        # if market_regime is not None:
        #     # Boost confidence in favorable regimes
        #     regime_boost = np.where(
        #         market_regime > 0,
        #         self.config.regime_confidence_boost,
        #         0.0
        #     )
        #     simple_confidence += regime_boost
        #     simple_confidence = np.clip(simple_confidence, 0, 1)
        
        # Apply minimum ensemble confidence threshold
        final_confidence = np.where(
            simple_confidence >= self.config.min_ensemble_confidence,
            simple_confidence,
            0.0
        )
        
        # Store in history
        self.confidence_history.append({
            'direction_confidence': direction_confidence,
            'profit_confidence': profit_confidence,
            'price_confidence': price_confidence,
            'simple_confidence': simple_confidence,
            'final_confidence': final_confidence
        })
        
        return {
            'direction_confidence': direction_confidence,
            'profit_confidence': profit_confidence,
            'price_confidence': price_confidence,
            'simple_confidence': simple_confidence,
            'final_confidence': final_confidence,
            'direction_prediction': direction_prediction,
            'profit_prediction': profit_prediction,
            'predicted_price': predicted_price
        }
    
    def get_confidence_threshold_signals(
        self,
        confidence_scores: Dict[str, np.ndarray],
        threshold: float = 0.7
    ) -> np.ndarray:
        """Get trading signals based on confidence threshold.
        
        Args:
            confidence_scores: Dictionary of confidence scores
            threshold: Minimum confidence threshold
            
        Returns:
            Binary trading signals (1 for trade, 0 for no trade)
        """
        final_confidence = confidence_scores['final_confidence']
        direction_prediction = confidence_scores['direction_prediction']
        
        # Generate signals based on confidence threshold
        signals = np.where(
            (final_confidence >= threshold) & (direction_prediction == 1),
            1,  # Long signal
            np.where(
                (final_confidence >= threshold) & (direction_prediction == 0),
                -1,  # Short signal
                0  # No signal
            )
        )
        
        return signals
    
    def get_confidence_statistics(self) -> Dict[str, float]:
        """Get statistics about confidence scores.
        
        Returns:
            Dictionary of confidence statistics
        """
        if not self.confidence_history:
            return {}
        
        # Calculate statistics from history
        all_final_confidences = np.concatenate([
            hist['final_confidence'] for hist in self.confidence_history
        ])
        
        stats = {
            'mean_confidence': np.mean(all_final_confidences),
            'std_confidence': np.std(all_final_confidences),
            'min_confidence': np.min(all_final_confidences),
            'max_confidence': np.max(all_final_confidences),
            'median_confidence': np.median(all_final_confidences),
            'high_confidence_rate': np.mean(all_final_confidences >= 0.8),
            'low_confidence_rate': np.mean(all_final_confidences <= 0.3)
        }
        
        return stats


def create_enhanced_confidence_scorer(
    direction_threshold: float = 0.6,
    profit_threshold: float = 0.001,
    price_threshold: float = 0.005,
    min_ensemble_confidence: float = 0.7
) -> EnhancedConfidenceScorer:
    """Factory function to create enhanced confidence scorer.
    
    Args:
        direction_threshold: Minimum direction confidence
        profit_threshold: Minimum expected profit
        price_threshold: Minimum price movement
        min_ensemble_confidence: Minimum ensemble confidence
        
    Returns:
        Configured EnhancedConfidenceScorer instance
    """
    config = ConfidenceConfig(
        direction_threshold=direction_threshold,
        profit_threshold=profit_threshold,
        price_threshold=price_threshold,
        min_ensemble_confidence=min_ensemble_confidence
    )
    
    return EnhancedConfidenceScorer(config)