"""
Perfect NAS Regime Qualification System

This module implements the perfect NAS regime detection system that combines:
- Advanced neural architectures from nas_modeling
- True NAS search from nas_clustering  
- Economic significance evaluation
- Trading viability assessment
- Meta-learning for regime adaptation
- Production optimization

The system provides the ultimate regime qualification for ML model training.
"""

from .core.perfect_nas_config import PerfectNASConfig
from .core.perfect_nas_regime_detector import PerfectNASRegimeDetector
from .core.hybrid_architecture import HybridRegimeArchitecture
from .evaluation.economic_evaluator import EconomicSignificanceEvaluator
from .evaluation.trading_viability_evaluator import TradingViabilityEvaluator
from .optimization.multi_objective_optimizer import PerfectMultiObjectiveOptimizer
from .meta_learning.adaptive_regime_learner import AdaptiveRegimeLearner

__version__ = "1.0.0"
__author__ = "Perfect NAS Regime System"

__all__ = [
    'PerfectNASConfig',
    'PerfectNASRegimeDetector', 
    'HybridRegimeArchitecture',
    'EconomicSignificanceEvaluator',
    'TradingViabilityEvaluator',
    'PerfectMultiObjectiveOptimizer',
    'AdaptiveRegimeLearner'
]