# src/config/config_two_tier.py

"""
Configuration file for optimizable two-tier system parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class TwoTierConfig:
    """Optimizable two-tier system parameters."""

    # Two-tier system thresholds
    tier1_confidence_threshold: float = 0.7
    tier2_confidence_threshold: float = 0.8
    tier1_weight: float = 0.6
    tier2_weight: float = 0.4

    # Direction thresholds
    direction_threshold: float = 0.6
    neutral_threshold: float = 0.4

    # Timeframe configuration
    tier1_timeframes: list[str] = None
    tier2_timeframes: list[str] = None

    # Integration thresholds
    step09_5_weight: float = 0.4
    step10_weight: float = 0.3
    regime_expert_weight: float = 0.3

    # Decision thresholds
    min_combined_confidence: float = 0.6
    max_confidence_discrepancy: float = 0.3

    # Performance thresholds
    min_tier1_performance: float = 0.55
    min_tier2_performance: float = 0.6
    performance_lookback_periods: int = 20

    # Risk management
    tier1_risk_multiplier: float = 1.0
    tier2_risk_multiplier: float = 1.1
    combined_risk_limit: float = 0.15

    # Stability thresholds
    tier1_stability_threshold: float = 0.7
    tier2_stability_threshold: float = 0.75
    stability_lookback_periods: int = 10

    # Integration timing
    integration_lookback_periods: int = 5
    integration_confirmation_periods: int = 3

    # Confidence calibration
    confidence_calibration_factor: float = 1.0
    confidence_smoothing_periods: int = 5

    # Decision validation
    min_decision_confidence: float = 0.5
    decision_validation_periods: int = 3

    # Performance weighting
    enable_performance_weighting: bool = True
    performance_weight_decay: float = 0.95
    min_performance_weight: float = 0.1

    # Risk adjustment
    enable_risk_adjustment: bool = True
    risk_adjustment_factor: float = 1.0
    max_risk_adjustment: float = 0.2


