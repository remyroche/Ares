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

def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["sharpe_ratio", "win_rate", "signal_clarity"]

if self.objective_weights is None:
            self.objective_weights = {
"sharpe_ratio": 0.4,
"win_rate": 0.3,
"signal_clarity": 0.3,
}

def get_strength_score_weights(self) -> dict[str, float]:
        """Get strength score weights as a dictionary."""
return {
"touch_count": self.touch_count_weight,
"total_volume": self.total_volume_weight,
"level_age": self.level_age_weight,
"bounce_rate": self.bounce_rate_weight,
"isolation_score": self.isolation_score_weight,
}

def get_level_detection_params(self) -> dict[str, Any]:
        """Get level detection parameters as a dictionary."""
return {
"min_touch_count": self.min_touch_count,
"min_level_age_hours": self.min_level_age_hours,
"price_tolerance_pct": self.price_tolerance_pct,
"volume_threshold": self.volume_threshold,
"strength_threshold": self.strength_threshold,
}

def get_breakout_thresholds(self) -> dict[str, float]:
        """Get breakout thresholds as a dictionary."""
return {
"breakout_threshold": self.breakout_threshold,
"confirmation_periods": self.confirmation_periods,
"volume_confirmation": self.volume_confirmation,
"momentum_threshold": self.momentum_threshold,
"false_breakout_filter": self.false_breakout_filter,
}

def get_zone_multipliers(self) -> dict[str, float]:
        """Get zone multipliers as a dictionary."""
return {
"support_zone_multiplier": self.support_zone_multiplier,
"resistance_zone_multiplier": self.resistance_zone_multiplier,
"sr_zone_threshold": self.sr_zone_threshold,
"zone_expansion_factor": self.zone_expansion_factor,
"zone_contraction_factor": self.zone_contraction_factor,
}

def get_confidence_thresholds(self) -> dict[str, float]:
        """Get confidence thresholds as a dictionary."""
return {
"min_sr_confidence": self.min_sr_confidence,
"high_confidence_threshold": self.high_confidence_threshold,
"confidence_decay_rate": self.confidence_decay_rate,
"ensemble_confidence_threshold": self.ensemble_confidence_threshold,
}


def get_sr_config() -> SRConfig:
    """Get S/R configuration."""
return SRConfig()


def get_sr_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for S/R optimization."""
return {
# Strength score weights
"touch_count_weight": {"min": 0.1, "max": 0.5, "type": "float"},
"total_volume_weight": {"min": 0.1, "max": 0.4, "type": "float"},
"level_age_weight": {"min": 0.1, "max": 0.4, "type": "float"},
"bounce_rate_weight": {"min": 0.1, "max": 0.4, "type": "float"},
"isolation_score_weight": {"min": 0.05, "max": 0.3, "type": "float"},
# Level detection parameters
"min_touch_count": {"min": 2, "max": 10, "type": "int"},
"min_level_age_hours": {"min": 1, "max": 48, "type": "int"},
"price_tolerance_pct": {"min": 0.1, "max": 2.0, "type": "float"},
"volume_threshold": {"min": 0.5, "max": 2.0, "type": "float"},
"strength_threshold": {"min": 0.3, "max": 0.8, "type": "float"},
# Breakout thresholds
"breakout_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
"confirmation_periods": {"min": 1, "max": 5, "type": "int"},
"volume_confirmation": {"min": 1.2, "max": 3.0, "type": "float"},
"momentum_threshold": {"min": 0.1, "max": 0.5, "type": "float"},
"false_breakout_filter": {"min": 0.1, "max": 0.3, "type": "float"},
# Zone multipliers
"support_zone_multiplier": {"min": 0.8, "max": 1.5, "type": "float"},
"resistance_zone_multiplier": {"min": 0.8, "max": 1.5, "type": "float"},
"sr_zone_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
"zone_expansion_factor": {"min": 1.0, "max": 2.0, "type": "float"},
"zone_contraction_factor": {"min": 0.5, "max": 1.0, "type": "float"},
# Confidence thresholds
"min_sr_confidence": {"min": 0.5, "max": 0.8, "type": "float"},
"high_confidence_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
"confidence_decay_rate": {"min": 0.1, "max": 0.5, "type": "float"},
"ensemble_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
# Performance thresholds
"min_sharpe_ratio": {"min": 0.3, "max": 0.8, "type": "float"},
"max_drawdown_threshold": {"min": -0.25, "max": -0.05, "type": "float"},
"min_win_rate": {"min": 0.45, "max": 0.7, "type": "float"},
"min_profit_factor": {"min": 1.1, "max": 2.0, "type": "float"},
"min_signal_clarity": {"min": 0.05, "max": 0.2, "type": "float"},
}