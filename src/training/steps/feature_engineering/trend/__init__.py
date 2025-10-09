"""
Trend Feature Engineering

This module contains features related to trend analysis including:
- Trend Coherence: Ensures trend continuity and direction consistency
"""

from .trend_coherence import TrendCoherenceFeature, TrendCoherenceConfig, calculate_trend_coherence_features

__all__ = [
    'TrendCoherenceFeature',
    'TrendCoherenceConfig',
    'calculate_trend_coherence_features'
]