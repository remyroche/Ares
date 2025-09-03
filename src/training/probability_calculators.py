"""
Probability Calculation Framework

This module provides base probability calculation functions for different model types
and market scenarios. It serves as the foundation for generating the 4 required
probability outputs for the Enhanced Prediction Service.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)

class BaseProbabilityCalculator:
    """Base class for probability calculations across different model types."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_probability(self, prob: float, name: str) -> float:
        """Validate that probability is between 0.0 and 1.0."""
        if not 0.0 <= prob <= 1.0:
            self.logger.warning(f"{name} probability {prob} out of range [0,1], clamping")
            return np.clip(prob, 0.0, 1.0)
        return prob
    
    def calculate_confidence_from_proba(self, y_pred_proba: np.ndarray) -> float:
        """Calculate confidence from prediction probabilities."""
        if y_pred_proba.ndim == 1:
            # Binary classification
            return np.mean(np.maximum(y_pred_proba, 1 - y_pred_proba))
        else:
            # Multi-class classification
            return np.mean(np.max(y_pred_proba, axis=1))

class ClassificationProbabilityCalculator(BaseProbabilityCalculator):
    """Probability calculator for classification models."""
    
    def calculate_triple_barrier_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame,
        profit_target: float = 0.02,
        stop_loss: float = 0.01,
        volatility_window: int = 20
    ) -> float:
        """
        Calculate probability of reaching profit target without hitting stop-loss.
        
        Args:
            model: Trained classification model
            X_test: Test features
            market_data: Market data with OHLCV information
            profit_target: Profit target percentage (default: 2%)
            stop_loss: Stop loss percentage (default: 1%)
            volatility_window: Window for volatility calculation
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get model predictions and probabilities
            y_pred_proba = model.predict_proba(X_test)
            confidence = self.calculate_confidence_from_proba(y_pred_proba)
            
            # Calculate market volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(window=volatility_window).std().mean()
            else:
                volatility = 0.02  # Default volatility
            
            # Adjust probability based on volatility and target ratios
            volatility_factor = max(0.1, 1 - volatility * 10)
            target_ratio = profit_target / stop_loss
            
            # Base probability from model confidence
            base_prob = confidence * volatility_factor
            
            # Adjust for target ratio (higher ratio = lower probability)
            ratio_factor = min(1.0, 2.0 / target_ratio)
            
            final_prob = base_prob * ratio_factor
            return self.validate_probability(final_prob, "triple_barrier")
            
        except Exception as e:
            self.logger.error(f"Error calculating triple barrier probability: {e}")
            return 0.5  # Default fallback
    
    def calculate_direction_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> float:
        """
        Calculate probability of price moving in predicted direction.
        
        Args:
            model: Trained classification model
            X_test: Test features
            y_test: True labels
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate confidence
            confidence = self.calculate_confidence_from_proba(y_pred_proba)
            
            # Combine accuracy and confidence
            direction_prob = (accuracy + confidence) / 2
            return self.validate_probability(direction_prob, "direction")
            
        except Exception as e:
            self.logger.error(f"Error calculating direction probability: {e}")
            return 0.5  # Default fallback
    
    def calculate_magnitude_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame,
        threshold_factor: float = 0.8
    ) -> float:
        """
        Calculate probability of price moving by expected magnitude.
        
        Args:
            model: Trained classification model
            X_test: Test features
            market_data: Market data
            threshold_factor: Factor for magnitude threshold
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)
            confidence = self.calculate_confidence_from_proba(y_pred_proba)
            
            # Calculate market volatility for magnitude context
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                avg_magnitude = returns.abs().mean()
                volatility = returns.std()
            else:
                avg_magnitude = 0.01  # Default 1%
                volatility = 0.02
            
            # Adjust probability based on confidence and market conditions
            magnitude_prob = confidence * (1 - volatility * 5) * threshold_factor
            return self.validate_probability(magnitude_prob, "magnitude")
            
        except Exception as e:
            self.logger.error(f"Error calculating magnitude probability: {e}")
            return 0.5  # Default fallback
    
    def calculate_barrier_avoidance_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame,
        adverse_threshold: float = 0.01
    ) -> float:
        """
        Calculate probability of avoiding adverse price movements.
        
        Args:
            model: Trained classification model
            X_test: Test features
            market_data: Market data
            adverse_threshold: Threshold for adverse movements
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)
            confidence = self.calculate_confidence_from_proba(y_pred_proba)
            
            # Calculate market risk metrics
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                adverse_prob = (returns.abs() > adverse_threshold).mean()
                volatility = returns.std()
            else:
                adverse_prob = 0.3  # Default 30% chance of adverse movement
                volatility = 0.02
            
            # Calculate avoidance probability
            base_avoidance = 1 - adverse_prob
            volatility_adjustment = max(0.1, 1 - volatility * 10)
            
            avoidance_prob = base_avoidance * volatility_adjustment * confidence
            return self.validate_probability(avoidance_prob, "barrier_avoidance")
            
        except Exception as e:
            self.logger.error(f"Error calculating barrier avoidance probability: {e}")
            return 0.5  # Default fallback

class RegressionProbabilityCalculator(BaseProbabilityCalculator):
    """Probability calculator for regression models."""
    
    def calculate_triple_barrier_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame,
        profit_target: float = 0.02,
        stop_loss: float = 0.01
    ) -> float:
        """
        Calculate probability of reaching profit target without hitting stop-loss.
        
        Args:
            model: Trained regression model
            X_test: Test features
            market_data: Market data
            profit_target: Profit target percentage
            stop_loss: Stop loss percentage
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate prediction confidence based on variance
            if hasattr(model, 'predict_proba'):
                # Some regression models support predict_proba
                y_pred_proba = model.predict_proba(X_test)
                confidence = self.calculate_confidence_from_proba(y_pred_proba)
            else:
                # Use prediction magnitude as confidence proxy
                pred_magnitude = np.abs(y_pred)
                confidence = np.mean(np.minimum(pred_magnitude / 0.02, 1.0))
            
            # Calculate market volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std()
            else:
                volatility = 0.02
            
            # Adjust for volatility and target ratio
            volatility_factor = max(0.1, 1 - volatility * 10)
            target_ratio = profit_target / stop_loss
            ratio_factor = min(1.0, 2.0 / target_ratio)
            
            final_prob = confidence * volatility_factor * ratio_factor
            return self.validate_probability(final_prob, "triple_barrier")
            
        except Exception as e:
            self.logger.error(f"Error calculating triple barrier probability: {e}")
            return 0.5
    
    def calculate_direction_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> float:
        """
        Calculate probability of price moving in predicted direction.
        
        Args:
            model: Trained regression model
            X_test: Test features
            y_test: True values
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate direction accuracy
            direction_correct = np.sign(y_pred) == np.sign(y_test)
            direction_accuracy = np.mean(direction_correct)
            
            # Calculate prediction confidence
            pred_magnitude = np.abs(y_pred)
            confidence = np.mean(np.minimum(pred_magnitude / 0.02, 1.0))
            
            # Combine accuracy and confidence
            direction_prob = (direction_accuracy + confidence) / 2
            return self.validate_probability(direction_prob, "direction")
            
        except Exception as e:
            self.logger.error(f"Error calculating direction probability: {e}")
            return 0.5
    
    def calculate_magnitude_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame,
        threshold_factor: float = 0.8
    ) -> float:
        """
        Calculate probability of price moving by expected magnitude.
        
        Args:
            model: Trained regression model
            X_test: Test features
            market_data: Market data
            threshold_factor: Factor for magnitude threshold
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            predicted_magnitude = np.abs(y_pred)
            
            # Calculate actual magnitude from market data
            if 'close' in market_data.columns:
                actual_magnitude = np.abs(market_data['close'].pct_change().dropna())
                avg_actual_magnitude = actual_magnitude.mean()
            else:
                avg_actual_magnitude = 0.01  # Default 1%
            
            # Calculate magnitude accuracy
            magnitude_accuracy = np.mean(
                predicted_magnitude >= avg_actual_magnitude * threshold_factor
            )
            
            # Adjust for prediction confidence
            confidence = np.mean(np.minimum(predicted_magnitude / 0.02, 1.0))
            
            magnitude_prob = (magnitude_accuracy + confidence) / 2
            return self.validate_probability(magnitude_prob, "magnitude")
            
        except Exception as e:
            self.logger.error(f"Error calculating magnitude probability: {e}")
            return 0.5
    
    def calculate_barrier_avoidance_probability(
        self, 
        model, 
        X_test: np.ndarray, 
        market_data: pd.DataFrame,
        adverse_threshold: float = 0.01
    ) -> float:
        """
        Calculate probability of avoiding adverse price movements.
        
        Args:
            model: Trained regression model
            X_test: Test features
            market_data: Market data
            adverse_threshold: Threshold for adverse movements
            
        Returns:
            float: Probability between 0.0 and 1.0
        """
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate prediction confidence
            pred_magnitude = np.abs(y_pred)
            confidence = np.mean(np.minimum(pred_magnitude / 0.02, 1.0))
            
            # Calculate market risk
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                adverse_prob = (returns.abs() > adverse_threshold).mean()
                volatility = returns.std()
            else:
                adverse_prob = 0.3
                volatility = 0.02
            
            # Calculate avoidance probability
            base_avoidance = 1 - adverse_prob
            volatility_adjustment = max(0.1, 1 - volatility * 10)
            
            avoidance_prob = base_avoidance * volatility_adjustment * confidence
            return self.validate_probability(avoidance_prob, "barrier_avoidance")
            
        except Exception as e:
            self.logger.error(f"Error calculating barrier avoidance probability: {e}")
            return 0.5

def get_probability_calculator(model_type: str) -> Union[ClassificationProbabilityCalculator, RegressionProbabilityCalculator]:
    """
    Factory function to get appropriate probability calculator.
    
    Args:
        model_type: Type of model ('classification' or 'regression')
        
    Returns:
        Appropriate probability calculator instance
    """
    if model_type.lower() in ['classification', 'classifier', 'clf']:
        return ClassificationProbabilityCalculator()
    elif model_type.lower() in ['regression', 'regressor', 'reg']:
        return RegressionProbabilityCalculator()
    else:
        # Default to classification
        logger.warning(f"Unknown model type '{model_type}', defaulting to classification")
        return ClassificationProbabilityCalculator()