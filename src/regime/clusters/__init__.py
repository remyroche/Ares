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

from .dimension_analyzer import MarketDimensionAnalyzer
from .regime_clusterer import RegimeClusterer
from .feature_importance import RegimeFeatureImportance
from .ml_training import RegimeMLTrainer
from .validation_metrics import RegimeValidationMetrics
from .integration_layer import HMMIntegrationLayer
from .visualization import RegimeVisualization

__all__ = [
    'MarketDimensionAnalyzer',
    'RegimeClusterer', 
    'RegimeFeatureImportance',
    'RegimeMLTrainer',
    'RegimeValidationMetrics',
    'HMMIntegrationLayer',
    'RegimeVisualization'
]

__version__ = "1.0.0"