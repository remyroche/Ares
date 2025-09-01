# src/config/config_position_sizing.py

"""
Configuration file for optimizable position sizing parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class PositionSizingConfig:
    """Optimizable position sizing parameters."""

    # Base position sizing
    base_position_size: float = 0.05  # 5% of portfolio
    max_position_size: float = 0.3  # 30% maximum position size
    min_position_size: float = 0.01  # 1% minimum position size

    # Confidence-based scaling
    confidence_based_scaling: bool = True
    confidence_thresholds: dict[str, float] = None
    position_size_multipliers: dict[str, float] = None

    # Volatility adjustment
    enable_volatility_scaling: bool = True
    atr_multiplier: float = 1.0
    volatility_thresholds: dict[str, float] = None
    volatility_multipliers: dict[str, float] = None

    # Liquidation risk adjustment
    enable_liquidation_scaling: bool = True
    lss_thresholds: dict[str, float] = None
    lss_multipliers: dict[str, float] = None

    # Successive position rules
    enable_successive_positions: bool = True
    min_confidence_for_successive: float = 0.85
    max_successive_positions: int = 3
    position_spacing_minutes: int = 15
    size_reduction_factor: float = 0.8
    max_total_exposure: float = 0.3

    # Risk limits
    max_single_position: float = 0.15
    max_total_exposure: float = 0.3
    max_correlation_exposure: float = 0.2
    max_leverage: float = 10.0

    # Kelly criterion parameters
    kelly_multiplier: float = 0.5
    kelly_max_fraction: float = 0.25

    # Dynamic risk management
    enable_dynamic_risk: bool = True
    drawdown_thresholds: dict[str, float] = None
    position_size_reductions: dict[str, float] = None


