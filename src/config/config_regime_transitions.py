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


