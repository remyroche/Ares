# src/config/config_sr.py

"""
Configuration file for optimizable S/R (Support/Resistance) parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class SRConfig:
    """Optimizable S/R (Support/Resistance) parameters."""

    # Strength score weights
    touch_count_weight: float = 0.3
    total_volume_weight: float = 0.25
    level_age_weight: float = 0.2
    bounce_rate_weight: float = 0.15
    isolation_score_weight: float = 0.1

    # Level detection parameters
    min_touch_count: int = 3
    min_level_age_hours: int = 24
    price_tolerance_pct: float = 0.5
    volume_threshold: float = 1.0
    strength_threshold: float = 0.5

    # Breakout thresholds
    breakout_threshold: float = 0.75
    confirmation_periods: int = 2
    volume_confirmation: float = 1.5
    momentum_threshold: float = 0.2
    false_breakout_filter: float = 0.2

    # Zone multipliers
    support_zone_multiplier: float = 1.0
    resistance_zone_multiplier: float = 1.0
    sr_zone_threshold: float = 0.7
    zone_expansion_factor: float = 1.2
    zone_contraction_factor: float = 0.8

    # Confidence thresholds
    min_sr_confidence: float = 0.6
    high_confidence_threshold: float = 0.8
    confidence_decay_rate: float = 0.2
    ensemble_confidence_threshold: float = 0.7

    # Optimization configuration
    multi_objective: bool = True
    objectives: list[str] = None
    objective_weights: dict[str, float] = None

    # Optimization constraints
    n_trials: int = 100
    cv_folds: int = 5
    early_stopping_patience: int = 20
    subsample_fraction: float = 0.7

    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.15
    min_win_rate: float = 0.55
    min_profit_factor: float = 1.3
    min_signal_clarity: float = 0.1


