"""
Neural Architecture Search (NAS) for Market Analysis

This module provides a comprehensive Neural Architecture Search implementation
specifically designed for financial market analysis, regime detection, and
HMM state modeling.

Key Features:
- True Neural Architecture Search implementation
- Optimized for financial time series
- HMM state optimization capabilities
- Regime detection architecture search
- Hardware acceleration support
"""

from .core.nas_search import NASArchitectureSearch
from .core.nas_model import NASModel
from .core.nas_trainer import NASTrainer
from .core.nas_evaluator import NASEvaluator

from .search.search_space import SearchSpace, ArchitectureConfig
from .search.random_search import RandomSearch
from .search.bayesian_search import BayesianSearch
from .search.evolutionary_search import EvolutionarySearch

from .evaluation.nas_metrics import NASMetrics
from .evaluation.regime_metrics import RegimeMetrics
from .evaluation.hmm_metrics import HMMMetrics

from .applications.hmm_nas import HMM_NAS_Optimizer
from .applications.regime_nas import Regime_NAS_Detector

from .utils.nas_utils import NASUtils, ArchitectureUtils
from .utils.logging_utils import NASLogger

__version__ = "1.0.0"
__author__ = "Ares Trading System"

__all__ = [
    # Core NAS
    'NASArchitectureSearch',
    'NASModel',
    'NASTrainer',
    'NASEvaluator',

    # Search Strategies
    'SearchSpace',
    'ArchitectureConfig',
    'RandomSearch',
    'BayesianSearch',
    'EvolutionarySearch',

    # Evaluation
    'NASMetrics',
    'RegimeMetrics',
    'HMMMetrics',

    # Applications
    'HMM_NAS_Optimizer',
    'Regime_NAS_Detector',

    # Utils
    'NASUtils',
    'ArchitectureUtils',
    'NASLogger'
]