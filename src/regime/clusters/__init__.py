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

__all__ = [
    # Main classes
    'MarketDimensionAnalyzer',
    'RegimeClusterer', 
    'RegimeFeatureImportance',
    'RegimeValidationMetrics',
    'HMMIntegrationLayer',
    'RegimeVisualization',
    
    # Configuration classes
    'DimensionAnalysisConfig',
    'ClusteringConfig',
    'ImportanceConfig',
    'ValidationConfig',
    'IntegrationConfig',
    'VisualizationConfig',
    
    # Enums
    'MarketDimension',
    'ClusteringMethod',
    'ImportanceMethod',
    'IntegrationMethod'
]

__version__ = "1.0.0"