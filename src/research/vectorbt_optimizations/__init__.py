"""
VectorBT Optimizations for Research Framework

This module provides VectorBT-based optimizations for the research framework,
enhancing performance, accuracy, and capabilities across all research components.

Key Optimizations:
1. Technical Analysis Enhancement
2. Backtesting Integration
3. Performance Metrics Improvement
4. Signal Generation Optimization
5. Portfolio Analysis Enhancement
"""

from .crypto_analysis_optimizer import VectorBTCryptoOptimizer
from .feature_comparison_optimizer import VectorBTFeatureOptimizer
from .profit_labeling_optimizer import VectorBTProfitLabelingOptimizer
from .price_patterns_optimizer import VectorBTPricePatternsOptimizer
from .clustering_optimizer import VectorBTClusteringOptimizer

__all__ = [
    'VectorBTCryptoOptimizer',
    'VectorBTFeatureOptimizer', 
    'VectorBTProfitLabelingOptimizer',
    'VectorBTPricePatternsOptimizer',
    'VectorBTClusteringOptimizer'
]