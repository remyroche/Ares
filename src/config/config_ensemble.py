# src/config/config_ensemble.py

"""
Configuration file for optimizable ensemble parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass
from enum import Enum


class EnsembleMethod(Enum):
    """Enum for ensemble gathering methods."""
ALL_THRESHOLD , "all_threshold"
MAJORITY_VOTE = "majority_vote"
WEIGHTED_AVERAGE = "weighted_average"
META_LEARNER = "meta_learner"
CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class EnsembleConfig:
    pass  # TODO: Add implementation
class EnsembleConfig:
    pass  # TODO: Add implementation
class EnsembleConfig:
    """Optimizable ensemble parameters."""

# Ensemble method
ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE

# Threshold-based ensemble
all_threshold_confidence: float = 0.8
majority_vote_threshold: float = 0.6

# Weighted ensemble
analyst_weight: float = 0.4
tactician_weight: float = 0.6

# Meta-learner parameters
meta_learner_type: str = "lightgbm"
meta_learner_learning_rate: float = 0.1
meta_learner_n_estimators: int = 100
meta_learner_max_depth: int = 6
meta_learner_min_child_samples: int = 20

# Ensemble validation
min_ensemble_agreement: float = 0.7
max_ensemble_disagreement: float = 0.3

# Confidence-weighted ensemble
confidence_weight_power: float = 2.0
min_confidence_weight: float = 0.1

# Ensemble diversity
enable_diversity_penalty: bool = True
diversity_penalty_weight: float = 0.1

# Ensemble stability
enable_stability_check: bool = True
stability_threshold: float = 0.8
stability_lookback_periods: int = 10


def get_ensemble_config() -> EnsembleConfig:
    """Get ensemble configuration."""
return EnsembleConfig()


def get_ensemble_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for ensemble optimization."""
return {
"all_threshold_confidence": {"min": 0.7, "max": 0.95, "type": "float"},
"majority_vote_threshold": {"min": 0.5, "max": 0.8, "type": "float"},
"analyst_weight": {"min": 0.2, "max": 0.6, "type": "float"},
"tactician_weight": {"min": 0.4, "max": 0.8, "type": "float"},
"meta_learner_learning_rate": {"min": 0.05, "max": 0.3, "type": "float"},
"meta_learner_n_estimators": {"min": 50, "max": 200, "type": "int"},
"meta_learner_max_depth": {"min": 3, "max": 10, "type": "int"},
"meta_learner_min_child_samples": {"min": 10, "max": 50, "type": "int"},
"min_ensemble_agreement": {"min": 0.6, "max": 0.9, "type": "float"},
"max_ensemble_disagreement": {"min": 0.2, "max": 0.5, "type": "float"},
"confidence_weight_power": {"min": 1.0, "max": 4.0, "type": "float"},
"min_confidence_weight": {"min": 0.05, "max": 0.2, "type": "float"},
"diversity_penalty_weight": {"min": 0.05, "max": 0.3, "type": "float"},
"stability_threshold": {"min": 0.7, "max": 0.95, "type": "float"},
"stability_lookback_periods": {"min": 5, "max": 20, "type": "int"},
}