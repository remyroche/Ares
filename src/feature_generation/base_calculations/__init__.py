"""
Base Calculations for Feature Generation

This module provides base calculation methods that can be used by different
feature generators, including price returns and returns-based VWAP calculations.
"""

from .base_calculator import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    PriceReturnsCalculator,
    ReturnsVWAPCalculator,
    PriceLevelsCalculator,
    VolumeWeightedCalculator,
    VolumeReturnsCalculator,
    get_base_calculator,
    create_base_calculator,
    calculate_price_returns,
    calculate_returns_vwap,
    calculate_price_levels,
    calculate_volume_weighted,
    calculate_volume_returns
)

__all__ = [
    "BaseCalculator",
    "BaseCalculationType",
    "BaseCalculationConfig",
    "PriceReturnsCalculator",
    "ReturnsVWAPCalculator",
    "PriceLevelsCalculator",
    "VolumeWeightedCalculator",
    "VolumeReturnsCalculator",
    "get_base_calculator",
    "create_base_calculator",
    "calculate_price_returns",
    "calculate_returns_vwap",
    "calculate_price_levels",
    "calculate_volume_weighted",
    "calculate_volume_returns"
]
