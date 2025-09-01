# src/config/config_technical_indicators.py

"""
Configuration file for optimizable technical indicator parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class TechnicalIndicatorsConfig:
    """Optimizable technical indicator parameters."""

    # RSI parameters
    rsi_period: int = 14
    rsi_overbought_threshold: float = 70.0
    rsi_oversold_threshold: float = 30.0
    rsi_signal_threshold: float = 50.0

    # MACD parameters
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    macd_threshold: float = 0.0

    # ADX parameters
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_sideways_threshold: float = 20.0

    # Moving averages
    sma_short_period: int = 9
    sma_medium_period: int = 21
    sma_long_period: int = 50
    ema_short_period: int = 12
    ema_long_period: int = 26

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_squeeze_threshold: float = 0.5

    # Volatility indicators
    atr_period: int = 14
    volatility_period: int = 20
    volatility_threshold: float = 0.025
    volatility_percentile_threshold: float = 0.75
    atr_normalized_threshold: float = 0.02
    bb_width_volatility_threshold: float = 0.03

    # Volume indicators
    volume_sma_period: int = 20
    volume_threshold: float = 1.5
    volume_price_divergence_threshold: float = 0.3

    # Divergence detection
    divergence_lookback_period: int = 10
    divergence_threshold: float = 0.2
    enable_rsi_macd_divergence: bool = True
    enable_volume_price_divergence: bool = True

    # Feature engineering
    enable_interaction_features: bool = True
    enable_cross_timeframe_features: bool = True
    enable_regime_features: bool = True

    # Regime detection
    regime_lookback_period: int = 50
    regime_volatility_threshold: float = 0.02
    regime_trend_threshold: float = 0.01
    regime_stability_threshold: float = 0.7
    regime_transition_threshold: float = 0.6
    regime_confirmation_periods: int = 3

    # Regime classification parameters (from simple_regime_rules.py)
    ema_fast_period: int = 21
    ema_slow_period: int = 55
    ema_sep_min_ratio: float = 0.0

    # Unified regime classifier parameters
    volatility_period: int = 10
    atr_normalized_threshold: float = 0.035
    volatility_percentile_threshold: float = 0.80
    bb_width_volatility_threshold: float = 0.045

    # Transition regime handler parameters
    transition_intensity_threshold: float = 0.3
    min_combined_intensity: float = 0.6
    max_regimes_to_consider: int = 3

    # Data quality thresholds
    correlation_threshold: float = 0.95
    nan_threshold: float = 0.1
    infinite_threshold: float = 0.05
    zero_variance_threshold: float = 1e-6
    constant_threshold: float = 0.1
    extreme_value_threshold: float = 5.0
    missing_warning: float = 0.05
    missing_error: float = 0.2
    variance_threshold: float = 1e-6


