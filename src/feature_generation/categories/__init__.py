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
- Acceleration features (momentum, acceleration, jerk, trend strength, etc.)
- Interaction features (momentum divergence, volatility-volume, etc.)
- Cross-timeframe features (multi-timeframe momentum, volatility, etc.)
- Entropy features (15 comprehensive entropy indicators)
"""

from .returns import ReturnsFeatureGenerator
from .momentum import MomentumFeatureGenerator
from .volume import VolumeFeatureGenerator
from .volatility import VolatilityFeatureGenerator
from .trend import TrendFeatureGenerator
from .oscillator import OscillatorFeatureGenerator
from .support_resistance import SupportResistanceFeatureGenerator
from .candlestick_pattern import CandlestickPatternFeatureGenerator
from .hmm_regime import (
    HMMRegimeFeatureGenerator,
    HMMRegimeLabelGenerator,
    HMMRegimeProbabilityGenerator,
    HMMRegimeTransitionGenerator,
    HMMRegimeDurationGenerator,
    HMMRegimeStabilityGenerator,
    create_hmm_regime_generators,
    create_default_hmm_regime_generators,
    create_advanced_hmm_regime_generators,
    create_minimal_hmm_regime_generators
)
from .hmm_performance_metrics import (
    HMMPerformanceMetricsFeatureGenerator,
    create_hmm_performance_features_from_result,
    integrate_hmm_metrics_with_features
)

# New consolidated categories
from .acceleration import (
    AccelerationFeatureGenerator,
    MomentumGenerator,
    PriceAccelerationGenerator,
    PriceJerkGenerator,
    TrendStrengthGenerator,
    TrendConsistencyGenerator,
    VolumeAccelerationGenerator,
    VolatilityAccelerationGenerator,
    create_acceleration_generators
)
from .interaction import (
    InteractionFeatureGenerator,
    MomentumDivergenceGenerator,
    MomentumVolumeGenerator,
    MomentumVolatilityGenerator,
    MomentumTrendGenerator,
    VolatilityVolumeGenerator,
    VolatilityPriceGenerator,
    VolatilityHighLowGenerator,
    VolatilityMomentumGenerator,
    VolatilityTrendGenerator,
    create_interaction_generators,
    # Legacy interaction generators
    CrossTimeframeInteractionGenerator,
    FeatureRatioGenerator,
    PolynomialFeatureGenerator,
    CorrelationInteractionGenerator,
    create_default_interaction_generators
)
from .cross_timeframe import (
    CrossTimeframeFeatureGenerator,
    CrossTimeframeMomentumGenerator,
    CrossTimeframeVolatilityGenerator,
    CrossTimeframeVolumeGenerator,
    CrossTimeframeTrendGenerator,
    CrossTimeframeHighLowGenerator,
    CrossTimeframeRatioGenerator,
    CrossTimeframeCorrelationGenerator,
    CrossTimeframeDivergenceGenerator,
    create_cross_timeframe_generators
)
from .entropy import (
    EntropyFeatureGenerator,
    PriceEntropyGenerator,
    VolumeEntropyGenerator,
    ReturnEntropyGenerator,
    PriceEntropyMAGenerator,
    VolumeEntropyMAGenerator,
    ReturnEntropyMAGenerator,
    HighLowEntropyGenerator,
    VolatilityEntropyGenerator,
    MomentumEntropyGenerator,
    RSIEntropyGenerator,
    MACDEntropyGenerator,
    BollingerBandsEntropyGenerator,
    CrossAssetEntropyGenerator,
    RegimeEntropyGenerator,
    create_entropy_generators,
    create_default_entropy_generators
)

from .microstructure import create_default_microstructure_generators
from .autoencoder import create_default_autoencoder_generators
from .order_flow import create_default_order_flow_generators
from .cross_timeframe import create_default_cross_timeframe_generators
# regime.py deleted - replaced by advanced HMM regime system
from .legacy import create_default_legacy_generators
from .time import create_default_time_generators

__all__ = [
    # Core categories
    "ReturnsFeatureGenerator",
    "MomentumFeatureGenerator",
    "VolumeFeatureGenerator", 
    "VolatilityFeatureGenerator",
    "TrendFeatureGenerator",
    "OscillatorFeatureGenerator",
    "SupportResistanceFeatureGenerator",
    "CandlestickPatternFeatureGenerator",
    
    # HMM Regime
    "HMMRegimeFeatureGenerator",
    "HMMRegimeLabelGenerator",
    "HMMRegimeProbabilityGenerator",
    "HMMRegimeTransitionGenerator",
    "HMMRegimeDurationGenerator",
    "HMMRegimeStabilityGenerator",
    "create_hmm_regime_generators",
    "create_default_hmm_regime_generators",
    "create_advanced_hmm_regime_generators",
    "create_minimal_hmm_regime_generators",
    "HMMPerformanceMetricsFeatureGenerator",
    "create_hmm_performance_features_from_result",
    "integrate_hmm_metrics_with_features",
    
    # New consolidated categories
    "AccelerationFeatureGenerator",
    "MomentumGenerator",
    "PriceAccelerationGenerator",
    "PriceJerkGenerator",
    "TrendStrengthGenerator",
    "TrendConsistencyGenerator",
    "VolumeAccelerationGenerator",
    "VolatilityAccelerationGenerator",
    "create_acceleration_generators",
    
    "InteractionFeatureGenerator",
    "MomentumDivergenceGenerator",
    "MomentumVolumeGenerator",
    "MomentumVolatilityGenerator",
    "MomentumTrendGenerator",
    "VolatilityVolumeGenerator",
    "VolatilityPriceGenerator",
    "VolatilityHighLowGenerator",
    "VolatilityMomentumGenerator",
    "VolatilityTrendGenerator",
    "create_interaction_generators",
    
    "CrossTimeframeFeatureGenerator",
    "CrossTimeframeMomentumGenerator",
    "CrossTimeframeVolatilityGenerator",
    "CrossTimeframeVolumeGenerator",
    "CrossTimeframeTrendGenerator",
    "CrossTimeframeHighLowGenerator",
    "CrossTimeframeRatioGenerator",
    "CrossTimeframeCorrelationGenerator",
    "CrossTimeframeDivergenceGenerator",
    "create_cross_timeframe_generators",
    
    "EntropyFeatureGenerator",
    "PriceEntropyGenerator",
    "VolumeEntropyGenerator",
    "ReturnEntropyGenerator",
    "PriceEntropyMAGenerator",
    "VolumeEntropyMAGenerator",
    "ReturnEntropyMAGenerator",
    "HighLowEntropyGenerator",
    "VolatilityEntropyGenerator",
    "MomentumEntropyGenerator",
    "RSIEntropyGenerator",
    "MACDEntropyGenerator",
    "BollingerBandsEntropyGenerator",
    "CrossAssetEntropyGenerator",
    "RegimeEntropyGenerator",
    "create_entropy_generators",
    
    # Legacy interaction generators
    "CrossTimeframeInteractionGenerator",
    "FeatureRatioGenerator",
    "PolynomialFeatureGenerator",
    "CorrelationInteractionGenerator",
    "create_default_interaction_generators",
    
    # Other categories
    "create_default_microstructure_generators",
    "create_default_entropy_generators",
    "create_default_autoencoder_generators",
    "create_default_order_flow_generators",
    "create_default_cross_timeframe_generators",
    "create_default_legacy_generators",
    "create_default_time_generators"
]