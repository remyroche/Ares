"""
Perfect NAS Regime Qualification System - Standalone Implementation

This module implements the perfect NAS regime detection system that combines:
- Advanced neural architectures (Neural ODEs, Vision Transformers, State Space Models)
- True NAS search with evolutionary algorithms
- Economic significance evaluation
- Trading viability assessment
- Meta-learning for regime adaptation
- Production optimization

The system is fully standalone with no external dependencies.
Provides the ultimate regime qualification for ML model training.
"""

from .core.perfect_nas_config import PerfectNASConfig
from .core.perfect_nas_regime_detector import PerfectNASRegimeDetector
from .core.hybrid_architecture import HybridRegimeArchitecture
from .core.neural_architectures import (
    NeuralODE, VisionTransformer, NeuralStateSpaceModel,
    ContinuousTimeRegimeDetector, TransformerRegimeDetector
)
from .core.nas_search import EssentialNASClusterer, NSGAIIOptimizer
from .evaluation.economic_evaluator import EconomicSignificanceEvaluator
from .evaluation.trading_viability_evaluator import TradingViabilityEvaluator
from .optimization.multi_objective_optimizer import PerfectMultiObjectiveOptimizer
from .meta_learning.adaptive_regime_learner import AdaptiveRegimeLearner

__version__ = "1.0.0-standalone"
__author__ = "Perfect NAS Regime System"
__description__ = "Fully standalone NAS regime detection system with no external dependencies"

__all__ = [
    'PerfectNASConfig',
    'PerfectNASRegimeDetector',
    'HybridRegimeArchitecture',
    'NeuralODE',
    'VisionTransformer',
    'NeuralStateSpaceModel',
    'ContinuousTimeRegimeDetector',
    'TransformerRegimeDetector',
    'EssentialNASClusterer',
    'NSGAIIOptimizer',
    'EconomicSignificanceEvaluator',
    'TradingViabilityEvaluator',
    'PerfectMultiObjectiveOptimizer',
    'AdaptiveRegimeLearner'
]
