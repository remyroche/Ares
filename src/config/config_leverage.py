# src/config/config_leverage.py

"""
Configuration file for optimizable leverage parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
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


