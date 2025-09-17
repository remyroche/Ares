"""
Research Module for Advanced Market Analysis.

This module contains research frameworks and tools for advanced market analysis,
including regime clustering, dimension discovery, and economic validation.

Submodules:
- clusters: Market regime clustering research framework
"""

from .clusters import *

__all__ = [
    # Re-export everything from clusters
    'MarketDimensionAnalyzer',
    'RegimeClusterer',
    'RegimeFeatureImportance', 
    'RegimeValidationMetrics',
    'HMMIntegrationLayer',
    'RegimeVisualization',
    'EconomicValidator',
    'TradingMetricCalibrator',
    'LookaheadBiasPrevention',
    'MetricOrthogonalizer',
    'ComprehensiveFeatureGenerator',
    'StatisticalDimensionAnalyzer',
    'DimensionEconomicRelevanceAnalyzer'
]