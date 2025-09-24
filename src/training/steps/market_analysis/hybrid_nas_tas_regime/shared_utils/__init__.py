"""
Shared utilities for NAS and TAS regime detection systems.

This module contains common utilities used by both Neural Architecture Search (NAS)
and Tree Architecture Search (TAS) regime detection systems, including:
- Feature collection and preprocessing
- Economic significance and trading viability assessment
- Data pipeline management
- Advanced search strategies
- Bayesian and evolutionary optimization
- Hardware optimization
- Advanced analysis components

These utilities are designed to be used by both individual regime detectors
and the hybrid regime system.
"""

from .feature_collection import *
from .economic_analysis import *
from .data_pipeline import *
from .search_strategies import *
from .optimization import *
from .hardware import *
from .analysis import *

__all__ = [
    # Feature collection
    'SharedFeatureCollector',
    'StandardizedFeatureCalculator',

    # Economic analysis
    'EconomicSignificanceAnalyzer',
    'TradingViabilityAssessor',

    # Data pipeline
    'SharedDataPipeline',
    'DataPreprocessor',

    # Search strategies
    'AdvancedSearchStrategy',
    'HybridSearchStrategy',

    # Optimization
    'BayesianOptimizer',
    'EvolutionaryOptimizer',
    'GridOptimizer',

    # Hardware
    'HardwareOptimizer',

    # Analysis
    'RegimeAnalyzer',
    'PerformanceAnalyzer',
    'ClusteringAnalyzer'
]