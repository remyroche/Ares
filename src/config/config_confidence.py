# src/config/config_confidence.py

"""
Configuration file for optimizable confidence threshold parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class ConfidenceThresholdsConfig:
    """Optimizable confidence thresholds for different trading decisions."""

    # Entry thresholds
    base_entry_threshold: float = 0.7
    volatility_modulated_entry: bool = True
    volatility_multiplier: float = 0.5
    volatility_zscore_threshold: float = 1.0

    # Two-tier system thresholds
    analyst_confidence_threshold: float = 0.7
    tactician_confidence_threshold: float = 0.8

    # Position management thresholds
    position_scale_up_threshold: float = 0.85
    position_scale_down_threshold: float = 0.6
    position_close_threshold: float = 0.3

    # ML target update thresholds
    ml_target_update_threshold: float = 0.5
    emergency_update_threshold: float = 0.02

    # Ensemble thresholds
    ensemble_agreement_threshold: float = 0.8
    ensemble_minimum_models: int = 3

    # Position closing thresholds
    neutral_signal_threshold: float = 0.5
    tactician_close_threshold: float = 0.6

    # Model performance thresholds
    model_performance_threshold: float = 0.6
    model_degradation_threshold: float = 0.4
    model_retrain_threshold: float = 0.3

    # S/R specific confidence thresholds
    min_sr_confidence: float = 0.6
    high_confidence_threshold: float = 0.8
    confidence_decay_rate: float = 0.2
    ensemble_confidence_threshold: float = 0.7

    # Breakout confidence thresholds
    breakout_confidence_threshold: float = 0.7
    false_breakout_filter: float = 0.2


