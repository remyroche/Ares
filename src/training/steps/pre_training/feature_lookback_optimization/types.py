"""
Centralized type definitions for feature lookback optimization.

This module provides a single source of truth for all type definitions
used across the feature lookback optimization module.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    MRMR = "mrmr"
    RANDOM_SEARCH = "random_search"
    MULTI_TARGET = "multi_target"
    COARSE_TO_REFINE = "coarse_to_refine"


@dataclass
class OptimizationResult:
    """Standardized optimization result."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_trials: int
    optimization_time: float
    convergence_achieved: bool
    metadata: Dict[str, Any]
    feature_name: str = ""
    stability_score: float = 0.0
    lookback_sensitivity: float = 0.0
    
    # ENHANCEMENTS: Extended stability and robustness metrics
    resampled_lookbacks: Optional[List[int]] = None
    objective_name: str = "unknown"
    regularization_penalty: float = 0.0
    raw_objective_value: float = 0.0
    is_stable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        # Ensure metadata contains serializable values
        def convert_metadata(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_metadata(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_metadata(item) for item in obj]
            else:
                return obj

        result_dict = {
            'best_lookback_period': int(self.best_lookback_period) if isinstance(self.best_lookback_period, np.int64) else self.best_lookback_period,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'total_trials': int(self.total_trials) if isinstance(self.total_trials, np.int64) else self.total_trials,
            'optimization_time': self.optimization_time,
            'convergence_achieved': self.convergence_achieved,
            'stability_score': self.stability_score,
            'lookback_sensitivity': self.lookback_sensitivity,
            'metadata': convert_metadata(self.metadata),
            'resampled_lookbacks': self.resampled_lookbacks if self.resampled_lookbacks is not None else [],
            'objective_name': self.objective_name,
            'regularization_penalty': self.regularization_penalty,
            'raw_objective_value': self.raw_objective_value,
            'is_stable': self.is_stable
        }
        return result_dict


@dataclass
class LookbackConstraints:
    """Constraints for lookback optimization."""
    min_lookback: int = 5
    max_lookback: int = 300
    search_step: int = 5
    enable_regularization: bool = True
    regularization_strength: float = 0.1
    preferred_lookback: int = 50
    min_stability_score: float = 0.7
    
    # ENHANCEMENTS: Explicit objective function and stability tracking
    optimization_objective: str = "max_ic"  # 'max_ic', 'max_sharpe', 'min_rmse', 'max_label_corr'
    preferred_min: float = 40.0  # Preferred minimum lookback
    preferred_max: float = 80.0  # Preferred maximum lookback
    penalty_exponent: float = 2.0  # Penalty exponent for regularization
    enable_bootstrap_stability: bool = True  # Enable bootstrap resampling for stability
    n_bootstrap_samples: int = 10  # Number of bootstrap samples (reduced to 2 for light/blank modes)
    track_sensitivity: bool = True  # Track lookback sensitivity
    
    # MODE-AWARE OPTIMIZATION: Execution mode settings
    execution_mode: str = "full"  # 'light', 'blank', 'full'
    cv_folds: int = 5  # Cross-validation folds (reduced to 2 for light/blank modes)
    use_bayesian_optimization: bool = False  # Use Bayesian TPE optimizer for coarser search
    enable_enhanced_caching: bool = True  # Enable enhanced cache optimization


__all__ = [
    'OptimizationMethod',
    'OptimizationResult', 
    'LookbackConstraints'
]