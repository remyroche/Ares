# src/config/config_two_tier.py

"""
Configuration file for optimizable two-tier system parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


import @dataclass
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

    def __post_init__(self):
    pass
    pass
    pass
    pass
        if self.tier1_timeframes is None:
    pass
    pass
    pass
    pass
            self.tier1_timeframes , ["1m", "5m", "15m", "1h"]

        if self.tier2_timeframes is None:
    pass
    pass
    pass
    pass
            self.tier2_timeframes = ["4h", "1d"]


def get_two_tier_config() -> TwoTierConfig:
    pass
    pass
    pass
    pass
    """Get two-tier configuration."""
    return TwoTierConfig()


def get_two_tier_search_space() -> dict[str, dict[str, Any]]:
    pass
    pass
    pass
    pass
    """Get search space for two-tier optimization."""
    return {
        "tier1_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "tier2_confidence_threshold": {"min": 0.7, "max": 0.95, "type": "float"},
        "tier1_weight": {"min": 0.4, "max": 0.8, "type": "float"},
        "tier2_weight": {"min": 0.2, "max": 0.6, "type": "float"},
        "direction_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "neutral_threshold": {"min": 0.3, "max": 0.6, "type": "float"},
        "min_combined_confidence": {"min": 0.6, "max": 0.9, "type": "float"},
        "max_confidence_discrepancy": {"min": 0.2, "max": 0.5, "type": "float"},
        "min_tier1_performance": {"min": 0.5, "max": 0.7, "type": "float"},
        "min_tier2_performance": {"min": 0.55, "max": 0.75, "type": "float"},
        "performance_lookback_periods": {"min": 10, "max": 30, "type": "int"},
        "tier1_risk_multiplier": {"min": 0.8, "max": 1.2, "type": "float"},
        "tier2_risk_multiplier": {"min": 0.9, "max": 1.2, "type": "float"},
        "combined_risk_limit": {"min": 0.1, "max": 0.2, "type": "float"},
        "tier1_stability_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "tier2_stability_threshold": {"min": 0.65, "max": 0.95, "type": "float"},
        "stability_lookback_periods": {"min": 5, "max": 15, "type": "int"},
        "integration_lookback_periods": {"min": 3, "max": 10, "type": "int"},
        "integration_confirmation_periods": {"min": 2, "max": 5, "type": "int"},
        "confidence_calibration_factor": {"min": 0.8, "max": 1.2, "type": "float"},
        "confidence_smoothing_periods": {"min": 3, "max": 10, "type": "int"},
        "min_decision_confidence": {"min": 0.4, "max": 0.7, "type": "float"},
        "decision_validation_periods": {"min": 2, "max": 5, "type": "int"},
        "enable_performance_weighting": {"type": "bool"},
        "performance_weight_decay": {"min": 0.9, "max": 0.99, "type": "float"},
        "min_performance_weight": {"min": 0.05, "max": 0.2, "type": "float"},
        "enable_risk_adjustment": {"type": "bool"},
        "risk_adjustment_factor": {"min": 0.8, "max": 1.2, "type": "float"},
        "max_risk_adjustment": {"min": 0.1, "max": 0.3, "type": "float"},
    }