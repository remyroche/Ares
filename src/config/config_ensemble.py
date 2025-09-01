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


