"""
Enhanced Feature Engineering for Statsmodels Clustering

This module provides comprehensive feature engineering capabilities with anti-leakage safeguards,
multiple feature types, and covariance stabilization for robust regime detection.

Key Features:
- Multiple feature types (returns, volatility, momentum, etc.)
- Rolling features with proper shift() to avoid look-ahead bias
- Factor exposures (market, size, value, momentum)
- Cross-sectional rank normalization
- Covariance stabilization using Ledoit-Wolf shrinkage
"""

from .enhanced_features import EnhancedFeatureEngineer
from .temporal_features import TemporalFeatureExtractor
from .factor_exposures import FactorExposureCalculator
from .rank_normalization import RankNormalizer
from .covariance_stabilization import CovarianceStabilizer

__all__ = [
    'EnhancedFeatureEngineer',
    'TemporalFeatureExtractor', 
    'FactorExposureCalculator',
    'RankNormalizer',
    'CovarianceStabilizer'
]