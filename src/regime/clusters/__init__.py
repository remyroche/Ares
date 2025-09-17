"""
Market Regime Clustering Research Framework.

This module provides comprehensive research tools for discovering and analyzing
market regime dimensions and clustering strategies for trading applications.

Key Components:
- Dimension Analysis: Research framework for market dimension importance
- Regime Clustering: Advanced clustering algorithms for market regimes
- Feature Importance: Analysis of regime-relevant features
- ML Model Training: Regime-specific model training framework
- Validation: Quality metrics and validation tools
- Integration: Bridge with existing HMM systems
"""

from .dimension_analyzer import MarketDimensionAnalyzer, DimensionAnalysisConfig, MarketDimension
from .regime_clusterer import RegimeClusterer, ClusteringConfig, ClusteringMethod
from .feature_importance import RegimeFeatureImportance, ImportanceConfig, ImportanceMethod
from .validation_metrics import RegimeValidationMetrics, ValidationConfig
from .integration_layer import HMMIntegrationLayer, IntegrationConfig, IntegrationMethod
from .visualization import RegimeVisualization, VisualizationConfig

# Enhanced components
from .economic_metrics import EconomicValidator, EconomicValidationConfig, EconomicMetric
from .trading_calibration import TradingMetricCalibrator, TradingCalibration, generate_complete_trading_calibration_report
from .lookahead_bias_prevention import LookaheadBiasPrevention, create_bias_free_analysis_wrapper
from .metric_orthogonalization import MetricOrthogonalizer, OrthogonalMetric, OrthogonalMetricResult
from .comprehensive_feature_integration import ComprehensiveFeatureGenerator
from .statistical_dimension_analysis import StatisticalDimensionAnalyzer, DimensionalityMethod
from .dimension_economic_relevance import DimensionEconomicRelevanceAnalyzer, analyze_all_dimensions_economic_relevance

__all__ = [
    # Main classes
    'MarketDimensionAnalyzer',
    'RegimeClusterer', 
    'RegimeFeatureImportance',
    'RegimeValidationMetrics',
    'HMMIntegrationLayer',
    'RegimeVisualization',
    
    # Enhanced components
    'EconomicValidator',
    'TradingMetricCalibrator',
    'LookaheadBiasPrevention',
    'MetricOrthogonalizer',
    'ComprehensiveFeatureGenerator',
    'StatisticalDimensionAnalyzer',
    'DimensionEconomicRelevanceAnalyzer',
    
    # Configuration classes
    'DimensionAnalysisConfig',
    'ClusteringConfig',
    'ImportanceConfig',
    'ValidationConfig',
    'IntegrationConfig',
    'VisualizationConfig',
    'EconomicValidationConfig',
    'TradingCalibration',
    
    # Enums
    'MarketDimension',
    'ClusteringMethod',
    'ImportanceMethod',
    'IntegrationMethod',
    'EconomicMetric',
    'OrthogonalMetric',
    'DimensionalityMethod',
    
    # Utility functions
    'generate_complete_trading_calibration_report',
    'create_bias_free_analysis_wrapper',
    'analyze_all_dimensions_economic_relevance'
]

__version__ = "1.0.0"