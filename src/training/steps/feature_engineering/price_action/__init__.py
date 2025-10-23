"""
Price Action Feature Engineering

This module contains features related to price action analysis including:
- Bar Efficiency Ratio: Measures directional price action vs. choppy conditions
- Close-Location Value (CLV): Tracks buying/selling pressure and control
"""

from .bar_efficiency_ratio import (
    BarEfficiencyRatioFeature, BarEfficiencyConfig, BarEfficiencyRatioGenerator,
    calculate_bar_efficiency_features
)
from .close_location_value import (
    CloseLocationValueFeature, CLVConfig, CloseLocationValueGenerator,
    calculate_clv_features
)

__all__ = [
    'BarEfficiencyRatioFeature',
    'BarEfficiencyConfig',
    'BarEfficiencyRatioGenerator',
    'calculate_bar_efficiency_features',
    'CloseLocationValueFeature',
    'CLVConfig',
    'CloseLocationValueGenerator',
    'calculate_clv_features'
]
