# src/config/multi_timeframe_hmm_ensemble_config.py

"""
Multi-Timeframe HMM Ensemble Configuration

Configuration settings for the multi-timeframe HMM cluster ensemble system
that combines predictions from HMM clusters across multiple timeframes.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class TimeframeConfig:
    """Configuration for each timeframe in the ensemble."""

    timeframe: str
    weight: float
    min_samples: int = 50
    enable_hazard_model: bool = True
    enable_price_prediction: bool = (
        False  # Hazard models are for regime transitions only
    )


@dataclass
class EnsembleConfig:
    """Configuration for the multi-timeframe ensemble."""

    timeframes: list[TimeframeConfig]
    meta_learner_type: str = "lgbm"  # "lgbm", "random_forest", "logistic"
    enable_dynamic_weighting: bool = True
    weight_update_frequency: int = 100  # Update weights every N predictions
    min_confidence_threshold: float = 0.6
    ensemble_method: str = (
        "meta_learner"  # "weighted_average", "meta_learner", "stacking"
    )





def validate_ensemble_config(config: dict[str, Any]) -> bool:
    """
    Validate ensemble configuration.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if valid = False otherwise
    """
    try:
        ensemble_config = config.get("MULTI_TIMEFRAME_HMM_ENSEMBLE", {})

        # Check if enabled
        if not ensemble_config.get("enabled", False):
            return False

        # Check timeframes
        timeframes = ensemble_config.get("timeframes", {})
        if not timeframes:
            return False

        # Validate timeframe weights sum to 1.0
        total_weight = sum(tf.get("weight", 0) for tf in timeframes.values())
        if abs(total_weight - 1.0) > 0.01:
            return False

        # Check ensemble method
        ensemble_method = ensemble_config.get("ensemble_method", "")
        valid_methods = ["weighted_average", "meta_learner", "stacking"]
        if ensemble_method not in valid_methods:
            return False

        # Check meta-learner type
        meta_learner_type = ensemble_config.get("meta_learner", {}).get("type", "")
        valid_learner_types = ["lgbm", "random_forest", "logistic"]
        return meta_learner_type in valid_learner_types

    except Exception:
        return False


