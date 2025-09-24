'\nConfiguration file for optimizable leverage parameters.\nThese parameters can be optimized in step12.\n'
from dataclasses import dataclass
from typing import Any
from ..utils.leverage_constants import MIN_LEVERAGE, MAX_LEVERAGE, LEVERAGE_RISK_THRESHOLDS

@dataclass
class LeverageConfig:
    """Optimizable leverage parameters."""
    max_leverage: float = MAX_LEVERAGE
    min_leverage: float = MIN_LEVERAGE
    leverage_risk_levels: dict[int, float] = None
    enable_dynamic_leverage: bool = True
    volatility_based_leverage: bool = True
    volatility_leverage_multiplier: float = 1.0
    low_volatility_leverage_boost: float = 1.2
    high_volatility_leverage_reduction: float = 0.7
    enable_confidence_leverage: bool = True
    confidence_leverage_thresholds: dict[str, float] = None
    confidence_leverage_multipliers: dict[str, float] = None
    enable_liquidation_protection: bool = True
    liquidation_buffer: float = 0.1
    max_liquidation_risk: float = 0.05
    enable_leverage_decay: bool = True
    leverage_decay_rate: float = 0.1
    leverage_decay_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.leverage_risk_levels is None:
            self.leverage_risk_levels = LEVERAGE_RISK_THRESHOLDS
        if self.confidence_leverage_thresholds is None:
            self.confidence_leverage_thresholds = {'low_confidence': 0.6, 'medium_confidence': 0.75, 'high_confidence': 0.85, 'very_high_confidence': 0.95}
        if self.confidence_leverage_multipliers is None:
            self.confidence_leverage_multipliers = {'low_confidence': 0.5, 'medium_confidence': 0.8, 'high_confidence': 1.0, 'very_high_confidence': 1.2}

def get_leverage_config() -> LeverageConfig:
    """Get leverage configuration."""
    return LeverageConfig()

def get_leverage_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for leverage optimization."""