"""
Volatility Feature Engineering

This module contains features related to volatility analysis including:
- ATR Volatility Ratio: Normalizes volatility for adaptive filtering
"""

from .atr_volatility_ratio import (
    ATRVolatilityRatioFeature, ATRVolatilityRatioConfig, ATRVolatilityRatioGenerator,
    calculate_atr_volatility_ratio_features
)

__all__ = [
    'ATRVolatilityRatioFeature',
    'ATRVolatilityRatioConfig',
    'ATRVolatilityRatioGenerator',
    'calculate_atr_volatility_ratio_features'
]