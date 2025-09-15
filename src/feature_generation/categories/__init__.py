"""
Category-Specific Feature Generators

This module provides feature generators organized by category, including:
- Returns features (price returns, log returns, etc.)
- Momentum features (RSI, MACD, etc.)
- Volume features (volume ratios, OBV, etc.)
- Volatility features (Bollinger Bands, ATR, etc.)
- Trend features (moving averages, trend indicators, etc.)
- Oscillator features (stochastic, Williams %R, etc.)
- Support/Resistance features (pivot points, levels, etc.)
- Candlestick pattern features (doji, hammer, etc.)
- HMM regime features (regime detection, regime-specific features, etc.)
"""

from .returns import ReturnsFeatureGenerator
from .momentum import MomentumFeatureGenerator
from .volume import VolumeFeatureGenerator
from .volatility import VolatilityFeatureGenerator
from .trend import TrendFeatureGenerator
from .oscillator import OscillatorFeatureGenerator
from .support_resistance import SupportResistanceFeatureGenerator
from .candlestick_pattern import CandlestickPatternFeatureGenerator
from .hmm_regime import HMMRegimeFeatureGenerator
from .interaction import (
    InteractionFeatureGenerator,
    CrossTimeframeInteractionGenerator,
    FeatureRatioGenerator,
    PolynomialFeatureGenerator,
    CorrelationInteractionGenerator,
    create_interaction_generators,
    create_default_interaction_generators
)

__all__ = [
    "ReturnsFeatureGenerator",
    "MomentumFeatureGenerator",
    "VolumeFeatureGenerator", 
    "VolatilityFeatureGenerator",
    "TrendFeatureGenerator",
    "OscillatorFeatureGenerator",
    "SupportResistanceFeatureGenerator",
    "CandlestickPatternFeatureGenerator",
    "HMMRegimeFeatureGenerator",
    "InteractionFeatureGenerator",
    "CrossTimeframeInteractionGenerator",
    "FeatureRatioGenerator",
    "PolynomialFeatureGenerator",
    "CorrelationInteractionGenerator",
    "create_interaction_generators",
    "create_default_interaction_generators"
]