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
- Regime transition features (transition probabilities, switch patterns, dynamics)
- Regime persistence features (duration, survival probability, exhaustion indicators)
- Market structure features (support/resistance within regimes, swing structure, fractals)
- Regime probability features (HMM probabilities, confidence, dynamics, patterns)
- Regime uncertainty features (classification entropy, confusion scores, ambiguity)
"""

# Returns and momentum features are handled by other categories
# Returns: handled by base calculations and other generators
# Momentum: handled by acceleration.py and interaction.py
# Volume: now handled by volume.py with comprehensive basic volume features
from .returns import (
    ReturnsFeatureGenerator,
    LogReturnsGenerator,
    SimpleReturnsGenerator,
    CumulativeReturnsGenerator,
    RollingReturnsGenerator,
    ReturnsVolatilityGenerator,
    ReturnsSkewnessGenerator,
    ReturnsKurtosisGenerator,
    ReturnGenerator,
    SharpeRatioGenerator,
    AmihudIlliquidityRatioGenerator,
    # NEW FEATURES - Advanced Returns Analysis
    AdvancedCumulativeReturnsGenerator,
    RollingZScoreReturnsGenerator,
    ARCoefficientsGenerator,
    LjungBoxTestGenerator,
    create_returns_generators,
    create_default_returns_generators
)
from .momentum import (
    MomentumFeatureGenerator,
    RSIGenerator,
    MACDGenerator,
    StochasticGenerator,
    WilliamsRGenerator,
    MomentumOscillatorGenerator,
    RateOfChangeGenerator,
    AdvancedMomentumGenerator,
    PriceAccelerationGenerator,
    VolumeMomentumGenerator,
    # NEW FEATURES - Advanced Momentum Analysis
    MomentumEndpointsGenerator,
    MACDDeltaGenerator,
    RSIZScoreGenerator,
    StochasticKDGenerator,
    DonchianChannelGenerator,
    create_momentum_generators,
    create_default_momentum_generators
)
from .volume import (
    VolumeFeatureGenerator,
    # NEW FEATURES - Enhanced Volume Analysis
    VolumeZScoreGenerator,
    VolumeMARatiosGenerator,
    CMFGenerator,
    VWAPDeviationsGenerator,
    OrderFlowImbalanceGenerator,
    VolumeVolatilityElasticityGenerator,
    create_default_volume_generators
)
from .volatility import (
    VolatilityFeatureGenerator,
    create_default_volatility_generators
)
# Note: Removed direct import of NormalizationFeatureGenerator to avoid circular import
# These will be imported lazily when needed
# from src.features_common.normalization import (
#     NormalizationFeatureGenerator,
#     RollingZScoreGenerator,
#     VolatilityScalingGenerator,
#     CrossSectionalNormalizer
# )
from .trend import TrendFeatureGenerator
from .oscillator import OscillatorFeatureGenerator
from .support_resistance import SupportResistanceFeatureGenerator
from .custom_support_resistance import (
    SRStrengthGenerator,
    SRDistanceGenerator,
    SRTouchCountGenerator,
    SRQualityGenerator,
    VolumeWeightedSRGenerator,
    DynamicSRGenerator,
    create_default_custom_sr_generators
)
from .candlestick_pattern import CandlestickPatternFeatureGenerator

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
    # Note: RegimeDependentFeatureGenerator and other enhanced generators
    # are not currently implemented - removed from imports to prevent errors
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
    CrossTimeframeFractionalChangeGenerator,
    CrossTimeframeAlignmentGenerator,
    CrossTimeframeLearnedProjectionGenerator,
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
    # NEW FEATURES - Advanced Entropy Analysis
    ShannonEntropyGenerator,
    PermutationEntropyGenerator,
    SampleEntropyGenerator,
    LempelZivComplexityGenerator,
    EntropyRateGenerator,
    SpectralEntropyGenerator,
    create_entropy_generators,
    create_default_entropy_generators
)

from .advanced_regime_features import (
    RegimeEntropyGenerator,
    RegimeComplexityGenerator,
    RegimeFractalDimensionGenerator,
    RegimeHurstExponentGenerator,
    RegimeMemoryStrengthGenerator,
    create_advanced_regime_generators
)

# NEW REGIME FEATURE CATEGORIES - Enhanced Regime Analysis
from .regime_transitions import (
    RegimeTransitionGenerator,
    create_regime_transition_generators
)
from .regime_persistence import (
    RegimePersistenceGenerator,
    create_regime_persistence_generators
)
from .market_structure import (
    MarketStructureGenerator,
    create_market_structure_generators
)
from .regime_probability import (
    RegimeProbabilityGenerator,
    create_regime_probability_generators
)
from .regime_uncertainty import (
    RegimeUncertaintyGenerator,
    create_regime_uncertainty_generators
)

# Ensemble disagreement features for ensemble models
from .ensemble_disagreement import (
    EnsembleDisagreementFeatures,
    calculate_ensemble_disagreement_features,
    get_core_feature_names
)

# Labeling features for meta-labeling (signal quality assessment)
from .labeling import (
    LabelingFeatureGenerator,
    MarketFeaturesGenerator,
    SignalFeaturesGenerator,
    LabelingFeatureConfig
)

from .microstructure_features import create_default_microstructure_generators
from .autoencoder import create_default_autoencoder_generators
from .representation_learning import (
    PatchTSTRepresentationGenerator,
    TFTEncoderRepresentationGenerator,
    AutoencoderRepresentationGenerator,
    ContrastiveLearningGenerator
)
from .order_flow import create_default_order_flow_generators
from .cross_timeframe import create_default_cross_timeframe_generators
# regime.py deleted - replaced by advanced HMM regime system
# from .legacy import create_default_legacy_generators  # File doesn't exist
from .time import create_default_time_generators

# NEW CATEGORIES - Advanced Statistical and Spectral/Wavelet
from .advanced_statistical import (
    HurstExponentGenerator,
    JumpIndicatorsGenerator,
    CVaRGenerator,
    MaxDrawdownGenerator,
    RollingSkewnessKurtosisGenerator,
    TrendPersistenceGenerator,
    create_advanced_statistical_generators
)

from .spectral_wavelet import (
    WaveletEnergyGenerator,
    BandLimitedVolatilityGenerator,
    CycleLengthGenerator,
    FractalDimensionGenerator,
    DFASlopesGenerator,
    create_default_spectral_wavelet_generators
)

# Regime feature integration
from .regime_feature_integration import (
    RegimeFeatureIntegration,
    RegimeFeatureConfig,
    generate_regime_features
)

# Multi-timeframe EWMA features (inspired by rolling_hmm_clustering)
from .multi_timeframe_ewma import (
    MultiTimeframeEWMAReturnsGenerator,
    MultiTimeframeEWMAVolatilityGenerator,
    MultiTimeframeEWMATrendGenerator,
    MultiTimeframeEWMAVolumeGenerator,
)

__all__ = [
    # Core categories
    "ReturnsFeatureGenerator",
    "LogReturnsGenerator",
    "SimpleReturnsGenerator",
    "CumulativeReturnsGenerator",
    "RollingReturnsGenerator",
    "ReturnsVolatilityGenerator",
    "ReturnsSkewnessGenerator",
    "ReturnsKurtosisGenerator",
    "ReturnGenerator",
    "SharpeRatioGenerator",
    # NEW FEATURES - Advanced Returns Analysis
    "AdvancedCumulativeReturnsGenerator",
    "RollingZScoreReturnsGenerator",
    "ARCoefficientsGenerator",
    "LjungBoxTestGenerator",
    "MomentumFeatureGenerator",
    "RSIGenerator",
    "MACDGenerator",
    "StochasticGenerator",
    "WilliamsRGenerator",
    "MomentumOscillatorGenerator",
    "RateOfChangeGenerator",
    # NEW FEATURES - Advanced Momentum Analysis
    "MomentumEndpointsGenerator",
    "MACDDeltaGenerator",
    "RSIZScoreGenerator",
    "StochasticKDGenerator",
    "DonchianChannelGenerator",
    "VolumeFeatureGenerator",
    # NEW FEATURES - Enhanced Volume Analysis
    "VolumeZScoreGenerator",
    "VolumeMARatiosGenerator",
    "CMFGenerator",
    "VWAPDeviationsGenerator",
    "OrderFlowImbalanceGenerator",
    "VolumeVolatilityElasticityGenerator",
    "VolatilityFeatureGenerator",
    "GARCHFeatureGenerator",
    # Note: NormalizationFeatureGenerator and related classes removed from __all__
    # to avoid circular imports - they will be imported lazily when needed
    "TrendFeatureGenerator",
    "OscillatorFeatureGenerator",
    "SupportResistanceFeatureGenerator",
    # Custom SR features (disabled by default)
    "SRStrengthGenerator",
    "SRDistanceGenerator",
    "SRTouchCountGenerator",
    "SRQualityGenerator",
    "VolumeWeightedSRGenerator",
    "DynamicSRGenerator",
    "create_default_custom_sr_generators",
    "CandlestickPatternFeatureGenerator",

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
    "CrossTimeframeFractionalChangeGenerator",
    "CrossTimeframeAlignmentGenerator",
    "CrossTimeframeLearnedProjectionGenerator",
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
    # NEW FEATURES - Advanced Entropy Analysis
    "ShannonEntropyGenerator",
    "PermutationEntropyGenerator",
    "SampleEntropyGenerator",
    "LempelZivComplexityGenerator",
    "EntropyRateGenerator",
    "SpectralEntropyGenerator",
    "create_entropy_generators",

    # Legacy interaction generators
    "CrossTimeframeInteractionGenerator",
    "FeatureRatioGenerator",
    "PolynomialFeatureGenerator",
    "CorrelationInteractionGenerator",
    "create_default_interaction_generators",

    # Other categories
    "create_returns_generators",
    "create_default_returns_generators",
    "create_momentum_generators",
    "create_default_momentum_generators",
    "create_default_volume_generators",
    "create_default_microstructure_generators",
    "create_default_entropy_generators",
    "create_default_autoencoder_generators",
    "PatchTSTRepresentationGenerator",
    "TFTEncoderRepresentationGenerator",
    "AutoencoderRepresentationGenerator",
    "ContrastiveLearningGenerator",
    "create_default_order_flow_generators",
    "create_default_cross_timeframe_generators",
    "create_default_time_generators",
    # NEW CATEGORIES - Advanced Statistical and Spectral/Wavelet
    "HurstExponentGenerator",
    "JumpIndicatorsGenerator",
    "CVaRGenerator",
    "MaxDrawdownGenerator",
    "RollingSkewnessKurtosisGenerator",
    "TrendPersistenceGenerator",
    "create_advanced_statistical_generators",
    "WaveletEnergyGenerator",
    "BandLimitedVolatilityGenerator",
    "CycleLengthGenerator",
    "FractalDimensionGenerator",
    "DFASlopesGenerator",
    "create_default_spectral_wavelet_generators",
    # Regime feature integration
    "RegimeFeatureIntegration",
    "RegimeFeatureConfig",
    "generate_regime_features",
    # NEW REGIME FEATURE CATEGORIES - Enhanced Regime Analysis
    "RegimeTransitionGenerator",
    "create_regime_transition_generators",
    "RegimePersistenceGenerator",
    "create_regime_persistence_generators",
    "MarketStructureGenerator",
    "create_market_structure_generators",
    "RegimeProbabilityGenerator",
    "create_regime_probability_generators",
    "RegimeUncertaintyGenerator",
    "create_regime_uncertainty_generators",
    # Multi-timeframe EWMA features
    "MultiTimeframeEWMAReturnsGenerator",
    "MultiTimeframeEWMAVolatilityGenerator",
    "MultiTimeframeEWMATrendGenerator",
    "MultiTimeframeEWMAVolumeGenerator",
    # Labeling features for meta-labeling
    "LabelingFeatureGenerator",
    "MarketFeaturesGenerator",
    "SignalFeaturesGenerator",
    "LabelingFeatureConfig",
    # Meta features
    "BarsSinceLastEventGenerator",
    "EventMeanReturnGenerator",
    "KaufmanEfficiencyRatioGenerator",
    "AveragedACFGenerator",
]

from .efficiency_noise import (
    KaufmanEfficiencyRatioGenerator,
    AveragedACFGenerator
)
