"""
Spectral and Wavelet Feature Generators

This module provides spectral and wavelet-based feature generators by importing from
the spectral features module.
"""

# Import spectral and wavelet generators from spectral features
from .spectral_features import (
    SpectralFeatureGenerator,
    WaveletFeatureGenerator,
    FractalDimensionGenerator,
    DetrendedFluctuationAnalysisGenerator,
    VectorBTSpectralFeatureGenerator,
    # VectorBTWaveletFeatureGenerator,  # This class doesn't exist, using VectorBTSpectralFeatureGenerator instead
    # VectorBTFractalDimensionGenerator,  # This class doesn't exist
    # VectorBTDetrendedFluctuationAnalysisGenerator,  # This class doesn't exist
    WaveletEnergyGenerator,
    BandLimitedVolatilityGenerator,
    CycleLengthGenerator,
    DFASlopesGenerator,
    VectorBTSpectralWaveletBatchGenerator,
    create_default_spectral_generators,
    create_default_wavelet_generators,
    create_default_fractal_generators,
    create_default_dfa_generators
)

def create_default_spectral_wavelet_generators():
    """Create default spectral and wavelet generators."""
    generators = []
    
    # Add spectral generators
    generators.extend(create_default_spectral_generators())
    
    # Add wavelet generators
    generators.extend(create_default_wavelet_generators())
    
    # Add fractal generators
    generators.extend(create_default_fractal_generators())
    
    # Add DFA generators
    generators.extend(create_default_dfa_generators())
    
    return generators

# Export all the classes and functions
__all__ = [
    'SpectralFeatureGenerator',
    'WaveletFeatureGenerator',
    'FractalDimensionGenerator',
    'DetrendedFluctuationAnalysisGenerator',
    'VectorBTSpectralFeatureGenerator',
    'WaveletEnergyGenerator',
    'BandLimitedVolatilityGenerator',
    'CycleLengthGenerator',
    'DFASlopesGenerator',
    'VectorBTSpectralWaveletBatchGenerator',
    'create_default_spectral_generators',
    'create_default_wavelet_generators',
    'create_default_fractal_generators',
    'create_default_dfa_generators',
    'create_default_spectral_wavelet_generators'
]
