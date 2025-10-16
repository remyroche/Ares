"""
Adaptive Regime Learner for NAS Regime Detection.

This module provides adaptive learning capabilities for regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """Learning metrics."""
    adaptation_rate: float
    learning_accuracy: float
    model_stability: float
    learning_progress: float

class AdaptiveRegimeLearner:
    """Adaptive learner for regime detection."""
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize the adaptive regime learner."""
        self.learning_rate = learning_rate
        self.learning_history = []
        self.model_weights = {}
        self.logger = logging.getLogger(__name__)
    
    def adapt_to_new_data(
        self, 
        new_data: pd.DataFrame,
        previous_predictions: np.ndarray,
        new_predictions: np.ndarray
    ) -> LearningMetrics:
        """Adapt the model to new data."""
        try:
            # Calculate adaptation rate
            adaptation_rate = self._calculate_adaptation_rate(
                previous_predictions, new_predictions
            )
            
            # Calculate learning accuracy
            learning_accuracy = self._calculate_learning_accuracy(
                new_predictions, new_data
            )
            
            # Calculate model stability
            model_stability = self._calculate_model_stability()
            
            # Calculate learning progress
            learning_progress = self._calculate_learning_progress()
            
            # Update learning history
            self.learning_history.append({
                'adaptation_rate': adaptation_rate,
                'learning_accuracy': learning_accuracy,
                'model_stability': model_stability,
                'learning_progress': learning_progress
            })
            
            return LearningMetrics(
                adaptation_rate=adaptation_rate,
                learning_accuracy=learning_accuracy,
                model_stability=model_stability,
                learning_progress=learning_progress
            )
            
        except Exception as e:
            self.logger.error(f"Error in adaptive learning: {e}")
            return LearningMetrics(0, 0, 0, 0)
    
    def _calculate_adaptation_rate(
        self, 
        previous: np.ndarray, 
        current: np.ndarray
    ) -> float:
        """Calculate adaptation rate."""
        if len(previous) == 0 or len(current) == 0:
            return 0.0
        
        # Calculate change in predictions
        change = np.mean(np.abs(current - previous))
        return min(change, 1.0)
    
    def _calculate_learning_accuracy(
        self, 
        predictions: np.ndarray, 
        data: pd.DataFrame
    ) -> float:
        """Calculate learning accuracy."""
        if len(predictions) == 0:
            return 0.0
        
        # Simple accuracy based on prediction consistency
        consistency = 1.0 / (1.0 + np.var(predictions))
        return consistency
    
    def _calculate_model_stability(self) -> float:
        """Calculate model stability."""
        if len(self.learning_history) < 2:
            return 1.0
        
        # Calculate stability based on recent learning history
        recent_rates = [h['adaptation_rate'] for h in self.learning_history[-5:]]
        stability = 1.0 - np.mean(recent_rates)
        return max(0.0, stability)
    
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress."""
        if len(self.learning_history) < 2:
            return 0.0
        
        # Calculate progress based on improvement in accuracy
        recent_accuracy = [h['learning_accuracy'] for h in self.learning_history[-5:]]
        if len(recent_accuracy) >= 2:
            progress = recent_accuracy[-1] - recent_accuracy[0]
            return max(0.0, progress)
        
        return 0.0