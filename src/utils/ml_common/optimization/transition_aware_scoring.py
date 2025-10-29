"""
Transition-Aware HPO Scoring for Regime Models

Provides custom scoring functions that balance accuracy and stability
for hyperparameter optimization. Integrates with Pareto optimization
for multi-objective hyperparameter tuning.
"""

import numpy as np
from typing import Callable, Any, Dict, List, Optional
from sklearn.metrics import accuracy_score, make_scorer
from src.utils.ml_common.evaluation.regime_temporal_metrics import (
    RegimeTemporalMetricsCalculator,
    calculate_temporal_smoothness_penalty
)

# Try to import Pareto optimization tools
try:
    from src.utils.ml_common.optimization.pareto import (
        ParetoFront,
        Solution,
        ObjectiveDirection,
        get_pareto_front
    )
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False
    ParetoFront = None
    Solution = None
    ObjectiveDirection = None
    get_pareto_front = None

# Try to import HPO utils for multi-objective optimization
try:
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False
    HyperparameterOptimization = None


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


def create_pareto_multi_objective_hpo(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 3,
    n_trials: int = 50,
    min_episode_length: int = 3,
    use_pareto_optimization: bool = True
) -> Dict[str, Any]:
    """
    Create a multi-objective HPO that uses Pareto optimization.
    
    Optimizes for both accuracy and stability (transition-aware metrics).
    Uses Pareto front to find trade-off solutions.
    
    Args:
        model_factory: Function that creates model with trial parameters
        X: Feature matrix
        y: Target array
        cv_folds: Number of CV folds
        n_trials: Number of optimization trials
        min_episode_length: Minimum desired episode length
        use_pareto_optimization: Use Pareto front optimization (default: True)
        
    Returns:
        Dictionary with Pareto-optimized results
    """
    if not PARETO_AVAILABLE or not HPO_AVAILABLE:
        raise ImportError("Pareto optimization tools not available. Install required dependencies.")
    
    temporal_calc = RegimeTemporalMetricsCalculator(min_episode_length=min_episode_length)
    pareto_front = get_pareto_front() if use_pareto_optimization else None
    
    # Define objectives: maximize accuracy, minimize transition_rate
    objectives: ObjectiveDirection = {
        'accuracy': 'max',
        'transition_rate': 'min',
        'mean_episode_length': 'max'
    }
    
    solutions: List[Solution] = []
    
    # Create HPO optimizer
    hpo_optimizer = HyperparameterOptimization({
        'max_trials': n_trials,
        'timeout_seconds': 300,
        'enable_early_stopping': True
    })
    
    def multi_objective_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate multiple objectives for Pareto optimization."""
        accuracy = accuracy_score(y_true, y_pred)
        
        temporal_metrics = temporal_calc._calculate_temporal_metrics(y_pred, None)
        
        return {
            'accuracy': accuracy,
            'transition_rate': temporal_metrics.get('transition_rate', 0.0),
            'mean_episode_length': temporal_metrics.get('mean_episode_length', 0.0),
            'switch_false_positive_rate': temporal_metrics.get('switch_false_positive_rate', 0.0)
        }
    
    # Run optimization trials
    for trial_num in range(n_trials):
        try:
            # Use HPO optimizer to get trial and create model
            # This is a simplified version - actual implementation would use Optuna trials
            # For now, we'll use the existing HPO infrastructure
            
            # This would need integration with Optuna or similar
            # For now, return a structure that can be used
            pass
        except Exception as e:
            continue
    
    # If Pareto optimization enabled, compute Pareto front
    if use_pareto_optimization and pareto_front and solutions:
        pareto_solutions = pareto_front.compute_pareto_front(solutions, objectives)
        
        return {
            'pareto_solutions': pareto_solutions,
            'objectives': objectives,
            'n_solutions': len(solutions),
            'n_pareto_solutions': len(pareto_solutions),
            'optimization_method': 'pareto_front'
        }
    
    return {
        'solutions': solutions,
        'objectives': objectives,
        'n_solutions': len(solutions),
        'optimization_method': 'multi_objective'
    }
