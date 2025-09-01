# src/config/config_technical_indicators.py

"""
Configuration file for optimizable technical indicator parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class TechnicalIndicatorsConfig:
    pass  # TODO: Add implementation
class TechnicalIndicatorsConfig:
    pass  # TODO: Add implementation
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


def get_technical_indicators_config() -> TechnicalIndicatorsConfig:
    """Get technical indicators configuration."""
return TechnicalIndicatorsConfig()


def get_technical_indicators_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for technical indicators optimization."""
return {
# RSI parameters
"rsi_period": {"min": 10, "max": 20, "type": "int"},
"rsi_overbought_threshold": {"min": 65.0, "max": 80.0, "type": "float"},
"rsi_oversold_threshold": {"min": 20.0, "max": 35.0, "type": "float"},
"rsi_signal_threshold": {"min": 45.0, "max": 55.0, "type": "float"},

# MACD parameters
"macd_fast_period": {"min": 8, "max": 16, "type": "int"},
"macd_slow_period": {"min": 20, "max": 30, "type": "int"},
"macd_signal_period": {"min": 7, "max": 12, "type": "int"},
"macd_threshold": {"min": -0.1, "max": 0.1, "type": "float"},

# ADX parameters
"adx_period": {"min": 10, "max": 20, "type": "int"},
"adx_trend_threshold": {"min": 20.0, "max": 30.0, "type": "float"},
"adx_sideways_threshold": {"min": 15.0, "max": 25.0, "type": "float"},

# Moving averages
"sma_short_period": {"min": 5, "max": 15, "type": "int"},
"sma_medium_period": {"min": 15, "max": 30, "type": "int"},
"sma_long_period": {"min": 40, "max": 60, "type": "int"},
"ema_short_period": {"min": 8, "max": 16, "type": "int"},
"ema_long_period": {"min": 20, "max": 30, "type": "int"},

# Bollinger Bands
"bb_period": {"min": 15, "max": 25, "type": "int"},
"bb_std_dev": {"min": 1.5, "max": 2.5, "type": "float"},
"bb_squeeze_threshold": {"min": 0.3, "max": 0.7, "type": "float"},

# Volatility indicators
"atr_period": {"min": 10, "max": 20, "type": "int"},
"volatility_period": {"min": 15, "max": 25, "type": "int"},
"volatility_threshold": {"min": 0.015, "max": 0.035, "type": "float"},
"volatility_percentile_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
"atr_normalized_threshold": {"min": 0.015, "max": 0.025, "type": "float"},
"bb_width_volatility_threshold": {"min": 0.02, "max": 0.04, "type": "float"},

# Volume indicators
"volume_sma_period": {"min": 15, "max": 25, "type": "int"},
"volume_threshold": {"min": 1.2, "max": 2.0, "type": "float"},
"volume_price_divergence_threshold": {"min": 0.2, "max": 0.4, "type": "float"},

# Divergence detection
"divergence_lookback_period": {"min": 5, "max": 15, "type": "int"},
"divergence_threshold": {"min": 0.1, "max": 0.3, "type": "float"},

# Regime detection
"regime_lookback_period": {"min": 30, "max": 100, "type": "int"},
"regime_volatility_threshold": {"min": 0.01, "max": 0.05, "type": "float"},
"regime_trend_threshold": {"min": 0.005, "max": 0.02, "type": "float"},
"regime_stability_threshold": {"min": 0.6, "max": 0.8, "type": "float"},
"regime_transition_threshold": {"min": 0.5, "max": 0.7, "type": "float"},
"regime_confirmation_periods": {"min": 2, "max": 5, "type": "int"},

# Regime classification parameters
"ema_fast_period": {"min": 15, "max": 30, "type": "int"},
"ema_slow_period": {"min": 40, "max": 70, "type": "int"},
"ema_sep_min_ratio": {"min": 0.0, "max": 0.1, "type": "float"},

# Unified regime classifier parameters
"volatility_period": {"min": 5, "max": 20, "type": "int"},
"atr_normalized_threshold": {"min": 0.02, "max": 0.05, "type": "float"},
"volatility_percentile_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
"bb_width_volatility_threshold": {"min": 0.03, "max": 0.06, "type": "float"},

# Transition regime handler parameters
"transition_intensity_threshold": {"min": 0.2, "max": 0.5, "type": "float"},
"min_combined_intensity": {"min": 0.5, "max": 0.8, "type": "float"},
"max_regimes_to_consider": {"min": 2, "max": 5, "type": "int"},

# Data quality thresholds
"correlation_threshold": {"min": 0.9, "max": 0.98, "type": "float"},
"nan_threshold": {"min": 0.05, "max": 0.2, "type": "float"},
"infinite_threshold": {"min": 0.02, "max": 0.1, "type": "float"},
"constant_threshold": {"min": 0.05, "max": 0.2, "type": "float"},
"extreme_value_threshold": {"min": 3.0, "max": 7.0, "type": "float"},
"missing_warning": {"min": 0.02, "max": 0.1, "type": "float"},
"missing_error": {"min": 0.1, "max": 0.3, "type": "float"},
}