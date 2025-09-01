# src/config/config_leverage.py

"""
Configuration file for optimizable leverage parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class LeverageConfig:
    pass  # TODO: Add implementation
class LeverageConfig:
class LeverageConfig:
    """Optimizable leverage parameters."""

# Base leverage settings
safe_leverage_multiplier: float = 0.8
max_leverage: int = 100
min_leverage: int = 1

# Leverage risk levels (leverage: max_adverse_movement)
leverage_risk_levels: dict[int, float] = None

# Dynamic leverage adjustment
enable_dynamic_leverage: bool = True
volatility_based_leverage: bool = True

# Volatility-based leverage
volatility_leverage_multiplier: float = 1.0
low_volatility_leverage_boost: float = 1.2
high_volatility_leverage_reduction: float = 0.7

# Confidence-based leverage
enable_confidence_leverage: bool = True
confidence_leverage_thresholds: dict[str, float] = None
confidence_leverage_multipliers: dict[str, float] = None

# Liquidation risk management
enable_liquidation_protection: bool = True
liquidation_buffer: float = 0.1  # 10% buffer from liquidation
max_liquidation_risk: float = 0.05  # 5% max liquidation risk

# Leverage decay
enable_leverage_decay: bool = True
leverage_decay_rate: float = 0.1
leverage_decay_threshold: float = 0.8

def __post_init__(self):
    def __post_init__(self):
    def __post_init__(self):
    def __post_init__(self):
        if self.leverage_risk_levels is None:
            self.leverage_risk_levels = {
10: 0.1,   # 10x leverage: can handle 10% adverse movement
15: 0.08,  # 15x leverage: can handle 8% adverse movement
20: 0.07,  # 20x leverage: can handle 7% adverse movement
25: 0.06,  # 25x leverage: can handle 6% adverse movement
30: 0.05,  # 30x leverage: can handle 5% adverse movement
40: 0.04,  # 40x leverage: can handle 4% adverse movement
50: 0.035, # 50x leverage: can handle 3.5% adverse movement
60: 0.03,  # 60x leverage: can handle 3% adverse movement
75: 0.025, # 75x leverage: can handle 2.5% adverse movement
100: 0.02, # 100x leverage: can handle 2% adverse movement
}

if self.confidence_leverage_thresholds is None:
            self.confidence_leverage_thresholds = {
"low_confidence": 0.6,
"medium_confidence": 0.75,
"high_confidence": 0.85,
"very_high_confidence": 0.95,
}

if self.confidence_leverage_multipliers is None:
            self.confidence_leverage_multipliers = {
"low_confidence": 0.5,
"medium_confidence": 0.8,
"high_confidence": 1.0,
"very_high_confidence": 1.2,
}


def get_leverage_config() -> LeverageConfig:
    """Get leverage configuration."""
return LeverageConfig()


def get_leverage_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for leverage optimization."""
return {
"safe_leverage_multiplier": {"min": 0.5, "max": 1.0, "type": "float"},
"max_leverage": {"min": 50, "max": 100, "type": "int"},
"min_leverage": {"min": 1, "max": 10, "type": "int"},
"volatility_leverage_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},
"low_volatility_leverage_boost": {"min": 1.0, "max": 1.5, "type": "float"},
"high_volatility_leverage_reduction": {"min": 0.5, "max": 0.9, "type": "float"},
"liquidation_buffer": {"min": 0.05, "max": 0.2, "type": "float"},
"max_liquidation_risk": {"min": 0.02, "max": 0.1, "type": "float"},
"leverage_decay_rate": {"min": 0.05, "max": 0.2, "type": "float"},
"leverage_decay_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
# Confidence leverage multipliers
"confidence_leverage_multipliers.low_confidence": {"min": 0.3, "max": 0.7, "type": "float"},
"confidence_leverage_multipliers.medium_confidence": {"min": 0.6, "max": 1.0, "type": "float"},
"confidence_leverage_multipliers.high_confidence": {"min": 0.8, "max": 1.2, "type": "float"},
"confidence_leverage_multipliers.very_high_confidence": {"min": 1.0, "max": 1.5, "type": "float"},
}