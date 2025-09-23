"""
Trading Architecture Search (TAS) Package

This package provides comprehensive Trading Architecture Search capabilities
for financial machine learning applications, including:

- Advanced TAS with micro-regime detection
- Neural architecture integration
- Economic significance validation
- Hardware acceleration
- Meta-learning and ensemble optimization
"""

from .core.tas_config import TASConfig, TASArchitectureType, TradingObjective, MarketRegime, MicroRegimeType
from .core.advanced_tas_search import AdvancedTradingArchitectureSearch, AdvancedTASResult
from .components.micro_regime_detector import MicroRegimeDetector, MicroRegimeDetectionResult
from .components.neural_architecture import TASNeuralModel, NeuralArchitectureConfig
from .evaluation.tas_evaluator import TASEvaluator, EvaluationResult

# Convenience functions
from .core.advanced_tas_search import optimize_advanced_trading_architecture

__version__ = "2.0.0"
__author__ = "Advanced TAS Team"

# Package configuration
DEFAULT_CONFIG = TASConfig.create_advanced_trading_config()

__all__ = [
    # Core classes
    'TASConfig',
    'TASArchitectureType',
    'TradingObjective',
    'MarketRegime',
    'MicroRegimeType',
    'AdvancedTradingArchitectureSearch',
    'AdvancedTASResult',

    # Components
    'MicroRegimeDetector',
    'MicroRegimeDetectionResult',
    'TASNeuralModel',
    'NeuralArchitectureConfig',

    # Evaluation
    'TASEvaluator',
    'EvaluationResult',

    # Convenience functions
    'optimize_advanced_trading_architecture',

    # Configuration
    'DEFAULT_CONFIG'
]