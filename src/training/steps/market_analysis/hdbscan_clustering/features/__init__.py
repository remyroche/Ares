"""
Feature extraction module for regime discovery system.

This module now imports features from the centralized feature_generation/ system
instead of maintaining duplicate implementations.
"""

# Import from the centralized feature_generation system
from src.feature_generation.categories.entropy import (
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
    # Advanced entropy features
    ShannonEntropyGenerator,
    PermutationEntropyGenerator,
    SampleEntropyGenerator,
    LempelZivComplexityGenerator,
    EntropyRateGenerator,
    SpectralEntropyGenerator,
    create_entropy_generators,
    create_default_entropy_generators
)

from src.feature_generation.categories.spectral_features import (
    SpectralFeatureGenerator,
    WaveletEnergyGenerator,
    BandLimitedVolatilityGenerator,
    CycleLengthGenerator,
    FractalDimensionGenerator,
    DFASlopesGenerator,
    create_default_spectral_wavelet_generators
)

from src.feature_generation.categories.regime_features import (
    RegimeFeatureGenerator,
    StatisticalRegimeFeatureGenerator,
    StructuralTrendRegimeFeatureGenerator,
    VolatilityRegimeFeatureGenerator,
    VolumeRegimeFeatureGenerator,
    AdvancedRegimeFeatureGenerator,
    create_regime_generators,
    create_default_regime_generators
)

# Legacy aliases for backward compatibility
EntropyComplexityFeatureGenerator = EntropyFeatureGenerator
EntropyFeatureResult = dict  # Simple alias for backward compatibility

SpectralFeatureGenerator = SpectralFeatureGenerator
SpectralFeatureResult = dict  # Simple alias for backward compatibility

RegimeFeatureExtractor = RegimeFeatureGenerator
RegimeFeatureExtractionResult = dict  # Simple alias for backward compatibility

__all__ = [
    # Entropy features
    'EntropyFeatureGenerator',
    'PriceEntropyGenerator',
    'VolumeEntropyGenerator',
    'ReturnEntropyGenerator',
    'PriceEntropyMAGenerator',
    'VolumeEntropyMAGenerator',
    'ReturnEntropyMAGenerator',
    'HighLowEntropyGenerator',
    'VolatilityEntropyGenerator',
    'MomentumEntropyGenerator',
    'RSIEntropyGenerator',
    'MACDEntropyGenerator',
    'BollingerBandsEntropyGenerator',
    'CrossAssetEntropyGenerator',
    'RegimeEntropyGenerator',
    'ShannonEntropyGenerator',
    'PermutationEntropyGenerator',
    'SampleEntropyGenerator',
    'LempelZivComplexityGenerator',
    'EntropyRateGenerator',
    'SpectralEntropyGenerator',
    'create_entropy_generators',
    'create_default_entropy_generators',
    
    # Spectral features
    'SpectralFeatureGenerator',
    'WaveletEnergyGenerator',
    'BandLimitedVolatilityGenerator',
    'CycleLengthGenerator',
    'FractalDimensionGenerator',
    'DFASlopesGenerator',
    'create_default_spectral_wavelet_generators',
    
    # Regime features
    'RegimeFeatureGenerator',
    'StatisticalRegimeFeatureGenerator',
    'StructuralTrendRegimeFeatureGenerator',
    'VolatilityRegimeFeatureGenerator',
    'VolumeRegimeFeatureGenerator',
    'AdvancedRegimeFeatureGenerator',
    'create_regime_generators',
    'create_default_regime_generators',
    
    # Legacy aliases for backward compatibility
    'EntropyComplexityFeatureGenerator',
    'EntropyFeatureResult',
    'SpectralFeatureGenerator',
    'SpectralFeatureResult',
    'RegimeFeatureExtractor',
    'RegimeFeatureExtractionResult'
]