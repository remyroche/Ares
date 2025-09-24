"""
Shared utilities for hybrid NAS-TAS regime detection.

This module provides common utilities used by both NAS and TAS regime detection systems,
including:
- Feature collection using pre-existing feature_generator/
- Economic significance evaluation
- Trading viability assessment
- Data pipeline utilities
- Advanced Search Strategies
- Bayesian optimization with grid optimization
- Advanced evolutionary algorithms (NSGA-II, SPEA2)
- Hardware optimization based on hardware/
- Advanced Analysis Components
"""

from .data_pipeline import DataPipelineManager, MarketDataProcessor
from .feature_collection import FeatureCollectionManager, StandardizedFeatureCalculator
from .economic_significance import EconomicSignificanceEvaluator, EconomicSignificanceResult
from .trading_viability import TradingViabilityEvaluator, TradingViabilityResult
from .search_strategies import AdvancedSearchStrategy, BayesianOptimizer, GridOptimizer
from .evolutionary_algorithms import NSGA2Optimizer, SPEA2Optimizer, EvolutionaryAlgorithm
from .hardware_optimization import HardwareOptimizer, PerformanceMonitor
from .analysis_components import AdvancedAnalysisComponent, RegimeAnalyzer, ClusterAnalyzer
from .metrics_reporting import MetricsReporter, ConsolidatedMetricsReport

__all__ = [
    # Data Pipeline
    'DataPipelineManager', 'MarketDataProcessor',
    
    # Feature Collection
    'FeatureCollectionManager', 'StandardizedFeatureCalculator',
    
    # Economic Significance
    'EconomicSignificanceEvaluator', 'EconomicSignificanceResult',
    
    # Trading Viability
    'TradingViabilityEvaluator', 'TradingViabilityResult',
    
    # Search Strategies
    'AdvancedSearchStrategy', 'BayesianOptimizer', 'GridOptimizer',
    
    # Evolutionary Algorithms
    'NSGA2Optimizer', 'SPEA2Optimizer', 'EvolutionaryAlgorithm',
    
    # Hardware Optimization
    'HardwareOptimizer', 'PerformanceMonitor',
    
    # Analysis Components
    'AdvancedAnalysisComponent', 'RegimeAnalyzer', 'ClusterAnalyzer',
    
    # Metrics Reporting
    'MetricsReporter', 'ConsolidatedMetricsReport'
]