"""
Weighted Loss Framework for Negative Learning Approximation

This module provides a comprehensive framework for implementing weighted losses
that approximate negative learning by emphasizing difficult samples and failure contexts.

Key Features:
1. Failure context detection and weighting
2. Sample difficulty assessment
3. Dynamic loss weighting strategies
4. Integration with existing models
5. Performance monitoring and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.metrics import log_loss, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

from .error_handling import (
    handle_errors, validate_data, safe_import,
    MLModelTrainerError, DataValidationError, ModelTrainingError, PredictionError
)

logger = logging.getLogger(__name__)

class FailureContextType(Enum):
    """Types of failure contexts for negative learning."""
    HIGH_VOLATILITY = "high_volatility"
    CHOP = "chop"
    WIDE_SPREAD = "wide_spread"
    LOW_LIQUIDITY = "low_liquidity"
    REGIME_CHANGE = "regime_change"
    OUTLIER = "outlier"
    UNCERTAINTY = "uncertainty"

class WeightingStrategy(Enum):
    """Weighting strategies for loss calculation."""
    DIFFICULTY_BASED = "difficulty_based"
    FAILURE_CONTEXT = "failure_context"
    ADAPTIVE = "adaptive"
    FOCAL_LOSS = "focal_loss"
    GRADIENT_BASED = "gradient_based"

@dataclass
class WeightedLossConfig:
    """Configuration for weighted loss framework."""
    # Core settings
    enable_weighted_loss: bool = True
    weighting_strategy: WeightingStrategy = WeightingStrategy.ADAPTIVE
    
    # Failure context detection
    volatility_threshold: float = 0.02
    chop_threshold: float = 0.5
    spread_threshold: float = 0.01
    liquidity_threshold: float = 1000.0
    
    # Weighting parameters
    base_weight: float = 1.0
    max_weight: float = 5.0
    min_weight: float = 0.1
    weight_smoothing: float = 0.1
    
    # Adaptive parameters
    adaptation_rate: float = 0.01
    memory_decay: float = 0.95
    stability_threshold: float = 0.1
    
    # Focal loss parameters
    alpha: float = 0.25
    gamma: float = 2.0
    
    # Performance monitoring
    enable_monitoring: bool = True
    log_frequency: int = 100

class FailureContextDetector:
    """Detects failure contexts in trading data."""
    
    def __init__(self, config: WeightedLossConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, market_data: Optional[Dict[str, np.ndarray]] = None):
        """Fit the failure context detector."""
        tprint_info("Fitting failure context detector")
        
        # Fit scaler for normalization
        self.scaler.fit(X)
        self.is_fitted = True
        
        tprint_success("Failure context detector fitted")
        
    def detect_failure_contexts(self, X: np.ndarray, y: np.ndarray, 
                              market_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Detect failure contexts in the data."""
        if not self.is_fitted:
            raise ModelTrainingError("Detector must be fitted before detection")
            
        tprint_debug(f"Detecting failure contexts for {X.shape[0]} samples")
        
        # Normalize features
        X_norm = self.scaler.transform(X)
        
        failure_contexts = {}
        
        # High volatility detection
        if market_data and 'returns' in market_data:
            returns = market_data['returns']
            volatility = self._calculate_rolling_volatility(returns)
            failure_contexts[FailureContextType.HIGH_VOLATILITY.value] = (
                volatility > self.config.volatility_threshold
            ).astype(float)
        else:
            # Use feature-based volatility estimation
            price_features = self._extract_price_features(X_norm)
            volatility = self._calculate_feature_volatility(price_features)
            failure_contexts[FailureContextType.HIGH_VOLATILITY.value] = (
                volatility > self.config.volatility_threshold
            ).astype(float)
        
        # Chop detection (sideways movement)
        if market_data and 'high' in market_data and 'low' in market_data:
            high = market_data['high']
            low = market_data['low']
            chop_score = self._calculate_chop_score(high, low)
            failure_contexts[FailureContextType.CHOP.value] = (
                chop_score > self.config.chop_threshold
            ).astype(float)
        else:
            # Use feature-based chop estimation
            chop_score = self._calculate_feature_chop(X_norm)
            failure_contexts[FailureContextType.CHOP.value] = (
                chop_score > self.config.chop_threshold
            ).astype(float)
        
        # Wide spread detection
        if market_data and 'bid' in market_data and 'ask' in market_data:
            bid = market_data['bid']
            ask = market_data['ask']
            spread = (ask - bid) / ((ask + bid) / 2)
            failure_contexts[FailureContextType.WIDE_SPREAD.value] = (
                spread > self.config.spread_threshold
            ).astype(float)
        else:
            # Use feature-based spread estimation
            spread_score = self._calculate_feature_spread(X_norm)
            failure_contexts[FailureContextType.WIDE_SPREAD.value] = (
                spread_score > self.config.spread_threshold
            ).astype(float)
        
        # Outlier detection
        outlier_score = self._calculate_outlier_score(X_norm)
        failure_contexts[FailureContextType.OUTLIER.value] = outlier_score
        
        # Uncertainty detection (prediction confidence)
        uncertainty_score = self._calculate_uncertainty_score(X_norm, y)
        failure_contexts[FailureContextType.UNCERTAINTY.value] = uncertainty_score
        
        tprint_debug(f"Detected failure contexts: {list(failure_contexts.keys())}")
        
        return failure_contexts
    
    def _calculate_rolling_volatility(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility."""
        if len(returns) < window:
            return np.zeros_like(returns)
        
        volatility = np.zeros_like(returns)
        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i])
        
        return volatility
    
    def _extract_price_features(self, X: np.ndarray) -> np.ndarray:
        """Extract price-related features from the feature matrix."""
        # Assume price features are in the first few columns
        # This is a heuristic - in practice, you'd know your feature structure
        price_cols = min(10, X.shape[1])
        return X[:, :price_cols]
    
    def _calculate_feature_volatility(self, price_features: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate volatility from price features."""
        if price_features.shape[0] < window:
            return np.zeros(price_features.shape[0])
        
        volatility = np.zeros(price_features.shape[0])
        for i in range(window, price_features.shape[0]):
            feature_returns = np.diff(price_features[i-window:i], axis=0)
            volatility[i] = np.mean(np.std(feature_returns, axis=0))
        
        return volatility
    
    def _calculate_chop_score(self, high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate chop score (sideways movement indicator)."""
        if len(high) < window:
            return np.zeros_like(high)
        
        chop_score = np.zeros_like(high)
        for i in range(window, len(high)):
            window_high = high[i-window:i]
            window_low = low[i-window:i]
            range_high = np.max(window_high) - np.min(window_low)
            sum_range = np.sum(window_high - window_low)
            chop_score[i] = sum_range / range_high if range_high > 0 else 0
        
        return chop_score
    
    def _calculate_feature_chop(self, X: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate chop score from features."""
        if X.shape[0] < window:
            return np.zeros(X.shape[0])
        
        chop_score = np.zeros(X.shape[0])
        for i in range(window, X.shape[0]):
            window_data = X[i-window:i]
            range_data = np.max(window_data, axis=0) - np.min(window_data, axis=0)
            sum_range = np.sum(np.abs(np.diff(window_data, axis=0)), axis=0)
            chop_score[i] = np.mean(sum_range / (range_data + 1e-8))
        
        return chop_score
    
    def _calculate_feature_spread(self, X: np.ndarray) -> np.ndarray:
        """Calculate spread score from features."""
        # Use feature variance as a proxy for spread
        feature_std = np.std(X, axis=1)
        return feature_std / (np.mean(np.abs(X), axis=1) + 1e-8)
    
    def _calculate_outlier_score(self, X: np.ndarray) -> np.ndarray:
        """Calculate outlier score using isolation forest approach."""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_scores = iso_forest.fit_predict(X)
        return (outlier_scores == -1).astype(float)
    
    def _calculate_uncertainty_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate uncertainty score based on feature complexity."""
        # Use feature variance and target variance as uncertainty indicators
        feature_uncertainty = np.var(X, axis=1)
        target_uncertainty = np.var(y) if hasattr(y, 'var') else np.var(y)
        
        # Normalize and combine
        feature_uncertainty = feature_uncertainty / (np.mean(feature_uncertainty) + 1e-8)
        uncertainty_score = feature_uncertainty * target_uncertainty
        
        return uncertainty_score

class SampleDifficultyAssessor:
    """Assesses sample difficulty for weighting."""
    
    def __init__(self, config: WeightedLossConfig):
        self.config = config
        self.difficulty_history = []
        
    def assess_difficulty(self, X: np.ndarray, y: np.ndarray, 
                         predictions: Optional[np.ndarray] = None) -> np.ndarray:
        """Assess difficulty of each sample."""
        tprint_debug(f"Assessing difficulty for {X.shape[0]} samples")
        
        # Feature-based difficulty
        feature_difficulty = self._calculate_feature_difficulty(X)
        
        # Prediction-based difficulty
        if predictions is not None:
            prediction_difficulty = self._calculate_prediction_difficulty(y, predictions)
        else:
            prediction_difficulty = np.zeros(len(y))
        
        # Target-based difficulty
        target_difficulty = self._calculate_target_difficulty(y)
        
        # Combine difficulties
        total_difficulty = (
            feature_difficulty * 0.4 + 
            prediction_difficulty * 0.4 + 
            target_difficulty * 0.2
        )
        
        # Normalize to [0, 1]
        if np.max(total_difficulty) > 0:
            total_difficulty = total_difficulty / np.max(total_difficulty)
        
        # Update history for adaptive weighting
        self.difficulty_history.append(np.mean(total_difficulty))
        if len(self.difficulty_history) > 100:
            self.difficulty_history = self.difficulty_history[-100:]
        
        return total_difficulty
    
    def _calculate_feature_difficulty(self, X: np.ndarray) -> np.ndarray:
        """Calculate difficulty based on feature complexity."""
        # Use feature variance and correlation as difficulty indicators
        feature_variance = np.var(X, axis=1)
        feature_correlation = self._calculate_feature_correlation(X)
        
        # Combine variance and correlation
        difficulty = feature_variance * (1 + feature_correlation)
        
        return difficulty
    
    def _calculate_feature_correlation(self, X: np.ndarray) -> np.ndarray:
        """Calculate average correlation between features for each sample."""
        if X.shape[1] < 2:
            return np.zeros(X.shape[0])
        
        correlations = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sample = X[i:i+1, :]
            if np.std(sample) > 0:
                # Calculate correlation with other samples
                other_samples = np.vstack([X[:i], X[i+1:]])
                if other_samples.shape[0] > 0:
                    corr_matrix = np.corrcoef(sample, other_samples)
                    if corr_matrix.shape[0] > 1:
                        correlations[i] = np.mean(np.abs(corr_matrix[0, 1:]))
        
        return correlations
    
    def _calculate_prediction_difficulty(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate difficulty based on prediction accuracy."""
        if len(y_true) != len(y_pred):
            return np.zeros(len(y_true))
        
        # Use absolute error as difficulty indicator
        error = np.abs(y_true - y_pred)
        
        # Normalize by target variance
        target_std = np.std(y_true) if np.std(y_true) > 0 else 1.0
        difficulty = error / target_std
        
        return difficulty
    
    def _calculate_target_difficulty(self, y: np.ndarray) -> np.ndarray:
        """Calculate difficulty based on target complexity."""
        if len(y) < 3:
            return np.zeros(len(y))
        
        # Use local variance as difficulty indicator
        difficulty = np.zeros(len(y))
        window = min(5, len(y) // 2)
        
        for i in range(len(y)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(y), i + window // 2 + 1)
            local_y = y[start_idx:end_idx]
            difficulty[i] = np.var(local_y)
        
        return difficulty

class WeightedLossCalculator:
    """Calculates weighted losses for negative learning approximation."""
    
    def __init__(self, config: WeightedLossConfig):
        self.config = config
        self.failure_detector = FailureContextDetector(config)
        self.difficulty_assessor = SampleDifficultyAssessor(config)
        self.weight_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, market_data: Optional[Dict[str, np.ndarray]] = None):
        """Fit the weighted loss calculator."""
        tprint_info("Fitting weighted loss calculator")
        
        self.failure_detector.fit(X, y, market_data)
        
        tprint_success("Weighted loss calculator fitted")
        
    def calculate_weights(self, X: np.ndarray, y: np.ndarray, 
                         predictions: Optional[np.ndarray] = None,
                         market_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate sample weights for loss calculation."""
        tprint_debug(f"Calculating weights for {X.shape[0]} samples")
        
        # Detect failure contexts
        failure_contexts = self.failure_detector.detect_failure_contexts(X, y, market_data)
        
        # Assess sample difficulty
        difficulty = self.difficulty_assessor.assess_difficulty(X, y, predictions)
        
        # Calculate weights based on strategy
        if self.config.weighting_strategy == WeightingStrategy.DIFFICULTY_BASED:
            weights = self._calculate_difficulty_weights(difficulty)
        elif self.config.weighting_strategy == WeightingStrategy.FAILURE_CONTEXT:
            weights = self._calculate_failure_context_weights(failure_contexts)
        elif self.config.weighting_strategy == WeightingStrategy.ADAPTIVE:
            weights = self._calculate_adaptive_weights(difficulty, failure_contexts)
        elif self.config.weighting_strategy == WeightingStrategy.FOCAL_LOSS:
            weights = self._calculate_focal_weights(y, predictions)
        elif self.config.weighting_strategy == WeightingStrategy.GRADIENT_BASED:
            weights = self._calculate_gradient_weights(X, y, predictions)
        else:
            weights = np.ones(len(y))
        
        # Apply weight constraints
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        # Smooth weights
        if self.config.weight_smoothing > 0:
            weights = self._smooth_weights(weights)
        
        # Update weight history
        self.weight_history.append(np.mean(weights))
        if len(self.weight_history) > 100:
            self.weight_history = self.weight_history[-100:]
        
        tprint_debug(f"Weight statistics - Mean: {np.mean(weights):.3f}, Std: {np.std(weights):.3f}")
        
        return weights
    
    def _calculate_difficulty_weights(self, difficulty: np.ndarray) -> np.ndarray:
        """Calculate weights based on sample difficulty."""
        # Higher difficulty = higher weight
        weights = self.config.base_weight + difficulty * (self.config.max_weight - self.config.base_weight)
        return weights
    
    def _calculate_failure_context_weights(self, failure_contexts: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate weights based on failure contexts."""
        weights = np.ones(len(list(failure_contexts.values())[0]))
        
        for context_type, context_scores in failure_contexts.items():
            # Higher failure context score = higher weight
            context_weights = self.config.base_weight + context_scores * (self.config.max_weight - self.config.base_weight)
            weights = np.maximum(weights, context_weights)
        
        return weights
    
    def _calculate_adaptive_weights(self, difficulty: np.ndarray, 
                                  failure_contexts: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate adaptive weights combining difficulty and failure contexts."""
        # Start with difficulty-based weights
        weights = self._calculate_difficulty_weights(difficulty)
        
        # Adjust based on failure contexts
        for context_type, context_scores in failure_contexts.items():
            context_adjustment = context_scores * (self.config.max_weight - self.config.base_weight)
            weights = np.maximum(weights, weights + context_adjustment)
        
        # Apply adaptive learning
        if len(self.weight_history) > 1:
            recent_avg = np.mean(self.weight_history[-10:])
            if recent_avg > self.config.stability_threshold:
                # Increase weights for more difficult samples
                weights = weights * (1 + self.config.adaptation_rate)
            else:
                # Decrease weights for stability
                weights = weights * (1 - self.config.adaptation_rate)
        
        return weights
    
    def _calculate_focal_weights(self, y_true: np.ndarray, y_pred: Optional[np.ndarray]) -> np.ndarray:
        """Calculate focal loss weights."""
        if y_pred is None:
            return np.ones(len(y_true))
        
        # Convert to probabilities if needed
        if y_pred.ndim == 1:
            # Binary classification
            p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        else:
            # Multiclass classification
            p_t = y_pred[np.arange(len(y_true)), y_true.astype(int)]
        
        # Calculate focal weights
        alpha_t = self.config.alpha
        focal_weights = alpha_t * (1 - p_t) ** self.config.gamma
        
        return focal_weights
    
    def _calculate_gradient_weights(self, X: np.ndarray, y: np.ndarray, 
                                  predictions: Optional[np.ndarray]) -> np.ndarray:
        """Calculate weights based on gradient magnitudes."""
        if predictions is None:
            return np.ones(len(y))
        
        # Calculate gradient magnitude as difficulty indicator
        error = y - predictions
        gradient_magnitude = np.abs(error)
        
        # Normalize and convert to weights
        if np.max(gradient_magnitude) > 0:
            gradient_weights = gradient_magnitude / np.max(gradient_magnitude)
        else:
            gradient_weights = np.ones(len(y))
        
        # Scale to weight range
        weights = self.config.base_weight + gradient_weights * (self.config.max_weight - self.config.base_weight)
        
        return weights
    
    def _smooth_weights(self, weights: np.ndarray, window: int = 5) -> np.ndarray:
        """Smooth weights using moving average."""
        if len(weights) < window:
            return weights
        
        smoothed_weights = np.zeros_like(weights)
        for i in range(len(weights)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(weights), i + window // 2 + 1)
            smoothed_weights[i] = np.mean(weights[start_idx:end_idx])
        
        return smoothed_weights
    
    def calculate_weighted_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              weights: np.ndarray, loss_type: str = "mse") -> float:
        """Calculate weighted loss."""
        if loss_type == "mse":
            loss = mean_squared_error(y_true, y_pred, sample_weight=weights)
        elif loss_type == "mae":
            loss = mean_absolute_error(y_true, y_pred, sample_weight=weights)
        elif loss_type == "log_loss":
            # For log loss, we need to handle the weights differently
            loss = self._calculate_weighted_log_loss(y_true, y_pred, weights)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss
    
    def _calculate_weighted_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   weights: np.ndarray) -> float:
        """Calculate weighted log loss."""
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Calculate log loss for each sample
        sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Apply weights
        weighted_losses = sample_losses * weights
        
        # Return mean weighted loss
        return np.mean(weighted_losses)

class WeightedLossManager:
    """Main manager for weighted loss implementation."""
    
    def __init__(self, config: WeightedLossConfig):
        self.config = config
        self.calculator = WeightedLossCalculator(config)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, market_data: Optional[Dict[str, np.ndarray]] = None):
        """Fit the weighted loss manager."""
        tprint_info("Fitting weighted loss manager")
        
        self.calculator.fit(X, y, market_data)
        self.is_fitted = True
        
        tprint_success("Weighted loss manager fitted")
        
    def get_sample_weights(self, X: np.ndarray, y: np.ndarray, 
                          predictions: Optional[np.ndarray] = None,
                          market_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Get sample weights for training."""
        if not self.is_fitted:
            raise ModelTrainingError("Manager must be fitted before getting weights")
        
        return self.calculator.calculate_weights(X, y, predictions, market_data)
    
    def calculate_weighted_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              X: np.ndarray, loss_type: str = "mse",
                              market_data: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Calculate weighted loss."""
        if not self.is_fitted:
            raise ModelTrainingError("Manager must be fitted before calculating loss")
        
        weights = self.calculator.calculate_weights(X, y_true, y_pred, market_data)
        return self.calculator.calculate_weighted_loss(y_true, y_pred, weights, loss_type)
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get weight statistics for monitoring."""
        if not self.weight_history:
            return {}
        
        return {
            "mean_weight": np.mean(self.weight_history),
            "std_weight": np.std(self.weight_history),
            "min_weight": np.min(self.weight_history),
            "max_weight": np.max(self.weight_history),
            "recent_trend": np.mean(self.weight_history[-10:]) - np.mean(self.weight_history[-20:-10]) if len(self.weight_history) >= 20 else 0.0
        }

# Factory function
def create_weighted_loss_manager(config: Optional[WeightedLossConfig] = None) -> WeightedLossManager:
    """Create a weighted loss manager with configuration."""
    if config is None:
        config = WeightedLossConfig()
    
    return WeightedLossManager(config)