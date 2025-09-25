"""
Neural Architecture Search (NAS) module.

This module contains NAS-specific implementations for automated neural network architecture discovery.
"""

from .neural_architecture_search import (
    NeuralArchitectureSearch,
    ArchitectureConfig,
    ArchitectureCandidate,
    ArchitectureSearchSpace,
    search_neural_architecture
)

from .adaptive_regime_nas import (
    AdaptiveRegimeNAS,
    AdaptiveRegimeNASConfig,
    RegimeDetector
)

__all__ = [
    'NeuralArchitectureSearch',
    'ArchitectureConfig', 
    'ArchitectureCandidate',
    'ArchitectureSearchSpace',
    'search_neural_architecture',
    'AdaptiveRegimeNAS',
    'AdaptiveRegimeNASConfig',
    'RegimeDetector'
]