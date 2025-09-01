# src/config/config_tpsl.py

"""
Configuration file for optimizable take profit and stop loss parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
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


