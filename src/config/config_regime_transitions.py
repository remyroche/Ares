# src/config/config_regime_transitions.py

"""
Configuration file for optimizable regime transition parameters.
These parameters control how we handle transitions between HMM states.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class RegimeTransitionConfig:
    """Optimizable regime transition parameters."""

    # Transition detection thresholds
    transition_intensity_threshold: float = 0.3
    min_combined_intensity: float = 0.6
    max_regimes_to_consider: int = 3

    # Transition confidence thresholds
    transition_confidence_threshold: float = 0.7
    regime_transition_prob_threshold: float = 0.6
    transition_imminent_threshold: float = 0.8

    # Model blending during transitions
    step09_5_weight: float = 0.4
    step10_weight: float = 0.3
    regime_expert_weight: float = 0.3

    # Transition type classification
    range_breakout_threshold: float = 0.65
    volatility_spike_threshold: float = 0.55

    # Weight adjustment factors
    step09_5_boost_factor: float = 0.5
    step10_boost_factor: float = 0.3
    regime_expert_boost_factor: float = 0.2

    # Transition timing
    transition_lookback_periods: int = 5
    transition_confirmation_periods: int = 3
    transition_stability_threshold: float = 0.7

    # Risk management during transitions
    transition_risk_multiplier: float = 1.2
    transition_position_multiplier: float = 0.8
    transition_confidence_penalty: float = 0.1

    # Model switching behavior
    enable_smooth_transitions: bool = True
    transition_blending_periods: int = 3
    min_transition_duration: int = 2

    # Performance monitoring
    transition_performance_threshold: float = 0.6
    transition_accuracy_threshold: float = 0.7
    transition_stability_weight: float = 0.3


def get_regime_transition_config() -> RegimeTransitionConfig:
    """Get regime transition configuration."""
    return RegimeTransitionConfig()


def get_regime_transition_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for regime transition optimization."""
    return {
        # Transition detection thresholds
        "transition_intensity_threshold": {"min": 0.2, "max": 0.5, "type": "float"},
        "min_combined_intensity": {"min": 0.5, "max": 0.8, "type": "float"},
        "max_regimes_to_consider": {"min": 2, "max": 5, "type": "int"},

        # Transition confidence thresholds
        "transition_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "regime_transition_prob_threshold": {"min": 0.5, "max": 0.8, "type": "float"},
        "transition_imminent_threshold": {"min": 0.7, "max": 0.95, "type": "float"},

        # Model blending during transitions
        "step09_5_weight": {"min": 0.2, "max": 0.6, "type": "float"},
        "step10_weight": {"min": 0.2, "max": 0.5, "type": "float"},
        "regime_expert_weight": {"min": 0.2, "max": 0.5, "type": "float"},

        # Transition type classification
        "range_breakout_threshold": {"min": 0.55, "max": 0.8, "type": "float"},
        "volatility_spike_threshold": {"min": 0.45, "max": 0.7, "type": "float"},

        # Weight adjustment factors
        "step09_5_boost_factor": {"min": 0.3, "max": 0.7, "type": "float"},
        "step10_boost_factor": {"min": 0.2, "max": 0.5, "type": "float"},
        "regime_expert_boost_factor": {"min": 0.1, "max": 0.4, "type": "float"},

        # Transition timing
        "transition_lookback_periods": {"min": 3, "max": 10, "type": "int"},
        "transition_confirmation_periods": {"min": 2, "max": 5, "type": "int"},
        "transition_stability_threshold": {"min": 0.6, "max": 0.85, "type": "float"},

        # Risk management during transitions
        "transition_risk_multiplier": {"min": 1.0, "max": 1.5, "type": "float"},
        "transition_position_multiplier": {"min": 0.6, "max": 1.0, "type": "float"},
        "transition_confidence_penalty": {"min": 0.05, "max": 0.2, "type": "float"},

        # Model switching behavior
        "transition_blending_periods": {"min": 2, "max": 5, "type": "int"},
        "min_transition_duration": {"min": 1, "max": 4, "type": "int"},

        # Performance monitoring
        "transition_performance_threshold": {"min": 0.5, "max": 0.8, "type": "float"},
        "transition_accuracy_threshold": {"min": 0.6, "max": 0.85, "type": "float"},
        "transition_stability_weight": {"min": 0.2, "max": 0.5, "type": "float"},
    }