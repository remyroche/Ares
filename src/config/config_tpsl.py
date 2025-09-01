# src/config/config_tpsl.py

"""
Configuration file for optimizable take profit and stop loss parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class PlaceholderDataClass:
    pass  # TODO: Add implementation
class TPSLConfig:
    pass  # TODO: Add implementation
class TPSLConfig:
    pass  # TODO: Add implementation
class TPSLConfig:
    """Optimizable take profit and stop loss parameters."""

# Base TP/SL settings
enable_tp_sl: bool = True
enable_ml_early_exit: bool = True
early_exit_confidence: float = 0.8

# Take Profit parameters
tp_long: float = 0.03  # 3% take profit for long positions
tp_short: float = 0.03  # 3% take profit for short positions
tp_volatility_multiplier: float = 1.5
max_tp: float = 0.15  # Maximum 15% take profit

# Stop Loss parameters
sl_long: float = 0.02  # 2% stop loss for long positions
sl_short: float = 0.02  # 2% stop loss for short positions
sl_volatility_multiplier: float = 2.0
max_sl: float = 0.10  # Maximum 10% stop loss

# Dynamic TP/SL based on volatility
enable_volatility_based_tpsl: bool = True
volatility_lookback_period: int = 20
volatility_thresholds: dict[str, float] = None
volatility_tp_multipliers: dict[str, float] = None
volatility_sl_multipliers: dict[str, float] = None

# Confidence-based TP/SL
enable_confidence_based_tpsl: bool = True
confidence_tp_multipliers: dict[str, float] = None
confidence_sl_multipliers: dict[str, float] = None

# Trailing stop loss
enable_trailing_stop: bool = True
trailing_stop_activation_threshold: float = 0.01
trailing_stop_distance: float = 0.005
lock_profit_threshold: float = 0.03

# Time-based exits
enable_time_based_exit: bool = True
max_holding_time_hours: int = 24
profit_lock_time_hours: int = 4
loss_cut_time_hours: int = 2

# Risk-reward ratios
min_risk_reward_ratio: float = 1.5
target_risk_reward_ratio: float = 2.0
max_risk_reward_ratio: float = 5.0

def __post_init__(self):
    def __post_init__(self):
    def __post_init__(self):
    def __post_init__(self):
        if self.volatility_thresholds is None:
            self.volatility_thresholds , {
"low_volatility": 0.02,
"medium_volatility": 0.05,
"high_volatility": 0.10,
}

if self.volatility_tp_multipliers is None:
            self.volatility_tp_multipliers = {
"low_volatility": 0.8,
"medium_volatility": 1.0,
"high_volatility": 1.5,
}

if self.volatility_sl_multipliers is None:
            self.volatility_sl_multipliers = {
"low_volatility": 1.2,
"medium_volatility": 1.0,
"high_volatility": 0.8,
}

if self.confidence_tp_multipliers is None:
            self.confidence_tp_multipliers = {
"low_confidence": 0.8,
"medium_confidence": 1.0,
"high_confidence": 1.2,
"very_high_confidence": 1.5,
}

if self.confidence_sl_multipliers is None:
            self.confidence_sl_multipliers = {
"low_confidence": 1.2,
"medium_confidence": 1.0,
"high_confidence": 0.8,
"very_high_confidence": 0.6,
}


def get_tpsl_config() -> TPSLConfig:
    """Get TP/SL configuration."""
return TPSLConfig()


def get_tpsl_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for TP/SL optimization."""
return {
"early_exit_confidence": {"min": 0.7, "max": 0.95, "type": "float"},
"tp_long": {"min": 0.01, "max": 0.08, "type": "float"},
"tp_short": {"min": 0.01, "max": 0.08, "type": "float"},
"tp_volatility_multiplier": {"min": 1.0, "max": 3.0, "type": "float"},
"max_tp": {"min": 0.08, "max": 0.25, "type": "float"},
"sl_long": {"min": 0.01, "max": 0.05, "type": "float"},
"sl_short": {"min": 0.01, "max": 0.05, "type": "float"},
"sl_volatility_multiplier": {"min": 1.0, "max": 3.0, "type": "float"},
"max_sl": {"min": 0.05, "max": 0.15, "type": "float"},
"volatility_lookback_period": {"min": 10, "max": 50, "type": "int"},
"trailing_stop_activation_threshold": {"min": 0.005, "max": 0.02, "type": "float"},
"trailing_stop_distance": {"min": 0.002, "max": 0.01, "type": "float"},
"lock_profit_threshold": {"min": 0.02, "max": 0.05, "type": "float"},
"max_holding_time_hours": {"min": 12, "max": 48, "type": "int"},
"profit_lock_time_hours": {"min": 2, "max": 8, "type": "int"},
"loss_cut_time_hours": {"min": 1, "max": 4, "type": "int"},
"min_risk_reward_ratio": {"min": 1.2, "max": 2.0, "type": "float"},
"target_risk_reward_ratio": {"min": 1.5, "max": 3.0, "type": "float"},
"max_risk_reward_ratio": {"min": 3.0, "max": 8.0, "type": "float"},
# Volatility thresholds
"volatility_thresholds.low_volatility": {"min": 0.01, "max": 0.03, "type": "float"},
"volatility_thresholds.medium_volatility": {"min": 0.03, "max": 0.07, "type": "float"},
"volatility_thresholds.high_volatility": {"min": 0.07, "max": 0.15, "type": "float"},
# Volatility TP multipliers
"volatility_tp_multipliers.low_volatility": {"min": 0.6, "max": 1.0, "type": "float"},
"volatility_tp_multipliers.medium_volatility": {"min": 0.8, "max": 1.2, "type": "float"},
"volatility_tp_multipliers.high_volatility": {"min": 1.2, "max": 2.0, "type": "float"},
# Volatility SL multipliers
"volatility_sl_multipliers.low_volatility": {"min": 1.0, "max": 1.5, "type": "float"},
"volatility_sl_multipliers.medium_volatility": {"min": 0.8, "max": 1.2, "type": "float"},
"volatility_sl_multipliers.high_volatility": {"min": 0.6, "max": 1.0, "type": "float"},
# Confidence TP multipliers
"confidence_tp_multipliers.low_confidence": {"min": 0.6, "max": 1.0, "type": "float"},
"confidence_tp_multipliers.medium_confidence": {"min": 0.8, "max": 1.2, "type": "float"},
"confidence_tp_multipliers.high_confidence": {"min": 1.0, "max": 1.4, "type": "float"},
"confidence_tp_multipliers.very_high_confidence": {"min": 1.2, "max": 2.0, "type": "float"},
# Confidence SL multipliers
"confidence_sl_multipliers.low_confidence": {"min": 1.0, "max": 1.5, "type": "float"},
"confidence_sl_multipliers.medium_confidence": {"min": 0.8, "max": 1.2, "type": "float"},
"confidence_sl_multipliers.high_confidence": {"min": 0.6, "max": 1.0, "type": "float"},
"confidence_sl_multipliers.very_high_confidence": {"min": 0.4, "max": 0.8, "type": "float"},
}