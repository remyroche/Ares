"""
Transition-Aware HPO Scoring for Regime Models

Provides custom scoring functions that balance accuracy and stability
for hyperparameter optimization.
"""

import numpy as np
from typing import Callable, Any
from sklearn.metrics import accuracy_score, make_scorer
from src.utils.ml_common.evaluation.regime_temporal_metrics import (
    RegimeTemporalMetricsCalculator,
    calculate_temporal_smoothness_penalty
)


def create_transition_aware_scorer(
    alpha: float = 0.1,
    accuracy_weight: float = 0.7,
    stability_weight: float = 0.3,
    min_episode_length: int = 3
) -> Callable[[Any, np.ndarray, np.ndarray], float]:
    """
    Create a transition-aware scorer that balances accuracy and stability.
    
    Args:
        alpha: Temporal smoothness penalty weight
        accuracy_weight: Weight for accuracy component (default 0.7)
        stability_weight: Weight for stability component (default 0.3)
        min_episode_length: Minimum desired episode length
        
    Returns:
        Scorer function compatible with sklearn
    """
    temporal_calc = RegimeTemporalMetricsCalculator(min_episode_length=min_episode_length)
    
    def transition_aware_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate transition-aware score combining accuracy and stability.
        
        Score = accuracy_weight * accuracy - stability_weight * (transition_rate + smoothness_penalty)
        """
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate temporal metrics
        temporal_metrics = temporal_calc._calculate_temporal_metrics(y_pred, None)
        transition_rate = temporal_metrics.get('transition_rate', 0.0)
        
        # Calculate smoothness penalty
        smoothness_penalty = calculate_temporal_smoothness_penalty(y_pred, alpha=alpha)
        
        # Normalize transition rate (typical range: 0-1)
        normalized_transition_rate = min(transition_rate, 1.0)
        
        # Normalize smoothness penalty (divide by number of samples for scale)
        normalized_penalty = smoothness_penalty / len(y_pred) if len(y_pred) > 0 else 0.0
        
        # Combine metrics
        stability_score = normalized_transition_rate + normalized_penalty
        composite_score = accuracy_weight * accuracy - stability_weight * stability_score
        
        return composite_score
    
    return make_scorer(transition_aware_score, greater_is_better=True)


def create_multi_objective_scorer(
    min_episode_length: int = 3
) -> Callable[[Any, np.ndarray, np.ndarray], dict]:
    """
    Create a multi-objective scorer that returns both accuracy and stability metrics.
    
    Returns a dictionary with:
    - accuracy
    - mean_episode_length
    - transition_rate
    - smoothness_penalty
    
    Args:
        min_episode_length: Minimum desired episode length
        
    Returns:
        Scorer function that returns dict of metrics
    """
    temporal_calc = RegimeTemporalMetricsCalculator(min_episode_length=min_episode_length)
    
    def multi_objective_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate multiple objectives for regime models."""
        accuracy = accuracy_score(y_true, y_pred)
        
        temporal_metrics = temporal_calc._calculate_temporal_metrics(y_pred, None)
        smoothness_penalty = calculate_temporal_smoothness_penalty(y_pred, alpha=0.1)
        
        return {
            'accuracy': accuracy,
            'mean_episode_length': temporal_metrics.get('mean_episode_length', 0.0),
            'transition_rate': temporal_metrics.get('transition_rate', 0.0),
            'smoothness_penalty': smoothness_penalty,
            'short_episode_count': temporal_metrics.get('short_episode_count', 0),
            'switch_false_positive_rate': temporal_metrics.get('switch_false_positive_rate', 0.0)
        }
    
    return multi_objective_score
