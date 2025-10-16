# TPSL CONFIG - TEMPORARILY DISABLED
# This file is temporarily disabled as TPSL parameters are commented out in config.yaml
# Uncomment when TPSL optimization is re-enabled

'\nConfiguration file for optimizable take profit and stop loss parameters.\nThese parameters can be optimized in step12.\n'
from dataclasses import dataclass
from typing import Any

@dataclass
class TPSLConfig:
    """Optimizable take profit and stop loss parameters."""
    enable_tp_sl: bool = True
    enable_ml_early_exit: bool = True
    early_exit_confidence: float = 0.8
    tp_long: float = 0.03
    tp_short: float = 0.03
    tp_volatility_multiplier: float = 1.5
    max_tp: float = 0.15
    sl_long: float = 0.02
    sl_short: float = 0.02
    sl_volatility_multiplier: float = 2.0
    max_sl: float = 0.1
    enable_volatility_based_tpsl: bool = True
    volatility_lookback_period: int = 20
    volatility_thresholds: dict[str, float] = None
    volatility_tp_multipliers: dict[str, float] = None
    volatility_sl_multipliers: dict[str, float] = None
    enable_confidence_based_tpsl: bool = True
    confidence_tp_multipliers: dict[str, float] = None
    confidence_sl_multipliers: dict[str, float] = None
    enable_trailing_stop: bool = True
    trailing_stop_activation_threshold: float = 0.01
    trailing_stop_distance: float = 0.005
    lock_profit_threshold: float = 0.03
    enable_time_based_exit: bool = True
    max_holding_time_hours: int = 24
    profit_lock_time_hours: int = 4
    loss_cut_time_hours: int = 2
    min_risk_reward_ratio: float = 1.5
    max_risk_reward_ratio: float = 5.0

    def __post_init__(self) -> None:
        if self.volatility_thresholds is None:
            self.volatility_thresholds = {'low_volatility': 0.02, 'medium_volatility': 0.05, 'high_volatility': 0.1}
        if self.volatility_tp_multipliers is None:
            self.volatility_tp_multipliers = {'low_volatility': 0.8, 'medium_volatility': 1.0, 'high_volatility': 1.5}
        if self.volatility_sl_multipliers is None:
            self.volatility_sl_multipliers = {'low_volatility': 1.2, 'medium_volatility': 1.0, 'high_volatility': 0.8}
        if self.confidence_tp_multipliers is None:
            self.confidence_tp_multipliers = {'low_confidence': 0.8, 'medium_confidence': 1.0, 'high_confidence': 1.2, 'very_high_confidence': 1.5}
        if self.confidence_sl_multipliers is None:
            self.confidence_sl_multipliers = {'low_confidence': 1.2, 'medium_confidence': 1.0, 'high_confidence': 0.8, 'very_high_confidence': 0.6}

def get_tpsl_config() -> TPSLConfig:
    """Get TP/SL configuration."""
    return TPSLConfig()

def get_tpsl_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for TP/SL optimization."""
