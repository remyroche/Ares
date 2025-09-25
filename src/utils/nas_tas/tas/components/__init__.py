"""
TAS Components Module

This module provides neural architecture components and micro-regime detection for TAS.
"""

from .neural_architecture import (
    LSTMTradingModel,
    AttentionTradingModel,
    NeuralODETradingModel,
    NeuralStateSpaceModel,
    HybridTreeNeuralModel,
    TASNeuralModel,
    NeuralArchitectureConfig,
    TreeInspiredLayer,
    ODENet
)

from .micro_regime_detector import (
    MicroRegimeDetector,
    MicroRegimeDetectionResult
)

__all__ = [
    'LSTMTradingModel',
    'AttentionTradingModel',
    'NeuralODETradingModel',
    'NeuralStateSpaceModel',
    'HybridTreeNeuralModel',
    'TASNeuralModel',
    'NeuralArchitectureConfig',
    'TreeInspiredLayer',
    'ODENet',
    'MicroRegimeDetector',
    'MicroRegimeDetectionResult'
]