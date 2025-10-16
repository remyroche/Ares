
# src/config/config_intensity.py

"""
Configuration file for optimizable intensity parameters.
These parameters control signal intensity, event triggering, and regime transitions.
"""

from dataclasses import dataclass
from typing import Any

@dataclass
class IntensityConfig:
    """Optimizable intensity parameters for signal processing and event triggering."""

    # Event trigger intensity thresholds
    transition_intensity_threshold: float = 0.3
    min_combined_intensity: float = 0.6
    signal_intensity_threshold: float = 0.5

    # Intensity weighting and reliability
    intensity_reliability_weight: float = 0.8
    intensity_decay_rate: float = 0.2
    intensity_boost_factor: float = 1.2

    # Regime transition intensity
    regime_transition_intensity: float = 0.4
    regime_stability_threshold: float = 0.7
    regime_change_boost: float = 1.5

    # Signal strength intensity
    breakout_intensity_threshold: float = 0.6
    volume_intensity_threshold: float = 0.5
    momentum_intensity_threshold: float = 0.4

    # Intensity-based position sizing
    intensity_position_multiplier: float = 1.0
    high_intensity_boost: float = 1.3
    low_intensity_reduction: float = 0.7

    # Non-maximum suppression
    intensity_nms_threshold: float = 0.5
    intensity_overlap_threshold: float = 0.3

    # Time-based intensity decay
    intensity_time_decay: float = 0.1
    intensity_persistence: float = 0.8

def get_intensity_config() -> IntensityConfig:
    """Get intensity configuration."""
    return IntensityConfig()

def get_intensity_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for intensity parameter optimization."""
    return {
        # Event trigger intensity thresholds
        "transition_intensity_threshold": {"min": 0.2, "max": 0.5, "type": "float"},
        "min_combined_intensity": {"min": 0.5, "max": 0.8, "type": "float"},
        "signal_intensity_threshold": {"min": 0.3, "max": 0.7, "type": "float"},

        # Intensity weighting and reliability
        "intensity_reliability_weight": {"min": 0.5, "max": 1.0, "type": "float"},
        "intensity_decay_rate": {"min": 0.1, "max": 0.5, "type": "float"},
        "intensity_boost_factor": {"min": 1.0, "max": 2.0, "type": "float"},

        # Regime transition intensity
        "regime_transition_intensity": {"min": 0.3, "max": 0.6, "type": "float"},
        "regime_stability_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "regime_change_boost": {"min": 1.2, "max": 2.0, "type": "float"},

        # Signal strength intensity
        "breakout_intensity_threshold": {"min": 0.5, "max": 0.8, "type": "float"},
        "volume_intensity_threshold": {"min": 0.4, "max": 0.7, "type": "float"},
        "momentum_intensity_threshold": {"min": 0.3, "max": 0.6, "type": "float"},

        # Intensity-based position sizing
        "intensity_position_multiplier": {"min": 0.8, "max": 1.5, "type": "float"},
        "high_intensity_boost": {"min": 1.1, "max": 1.8, "type": "float"},
        "low_intensity_reduction": {"min": 0.5, "max": 0.9, "type": "float"},

        # Non-maximum suppression
        "intensity_nms_threshold": {"min": 0.3, "max": 0.7, "type": "float"},
        "intensity_overlap_threshold": {"min": 0.2, "max": 0.5, "type": "float"},

        # Time-based intensity decay
        "intensity_time_decay": {"min": 0.05, "max": 0.2, "type": "float"},
        "intensity_persistence": {"min": 0.6, "max": 0.9, "type": "float"},
    }
