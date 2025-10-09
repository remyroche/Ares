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
    create_volatility_generators,
    create_default_volatility_generators
)
from .normalization import (
    NormalizationFeatureGenerator,
    RollingZScoreGenerator,
    VolatilityScalingGenerator,
    CrossSectionalNormalizer
)
from .trend import TrendFeatureGenerator
from .oscillator import OscillatorFeatureGenerator
from .support_resistance import SupportResistanceFeatureGenerator
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
    RegimeEntropyGenerator,
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

from .microstructure import create_default_microstructure_generators
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
from .legacy import create_default_legacy_generators
from .time import create_default_time_generators

# NEW CATEGORIES - Advanced Statistical and Spectral/Wavelet
from .advanced_statistical import (
    HurstExponentGenerator,
    JumpIndicatorsGenerator,
    CVaRGenerator,
    MaxDrawdownGenerator,
    RollingSkewnessKurtosisGenerator,
    TrendPersistenceGenerator,
    create_default_advanced_statistical_generators
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
    "NormalizationFeatureGenerator",
    "RollingZScoreGenerator",
    "VolatilityScalingGenerator",
    "CrossSectionalNormalizer",
    "TrendFeatureGenerator",
    "OscillatorFeatureGenerator",
    "SupportResistanceFeatureGenerator",
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
    "RegimeEntropyGenerator",
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
    # Note: RegimeDependentFeatureGenerator, CointegrationResidualGenerator,
    # StructuralRatioGenerator, PairwiseInteractionGenerator are not implemented yet
    
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
    "create_default_legacy_generators",
    "create_default_time_generators",
    # NEW CATEGORIES - Advanced Statistical and Spectral/Wavelet
    "HurstExponentGenerator",
    "JumpIndicatorsGenerator",
    "CVaRGenerator",
    "MaxDrawdownGenerator",
    "RollingSkewnessKurtosisGenerator",
    "TrendPersistenceGenerator",
    "create_default_advanced_statistical_generators",
    "WaveletEnergyGenerator",
    "BandLimitedVolatilityGenerator",
    "CycleLengthGenerator",
    "FractalDimensionGenerator",
    "DFASlopesGenerator",
    "create_default_spectral_wavelet_generators",
    # Regime feature integration
    "RegimeFeatureIntegration",
    "RegimeFeatureConfig",
    "generate_regime_features"
]