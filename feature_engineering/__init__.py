"""
Feature Engineering Module

This module contains the feature engineering components for the trading system,
including the analyst features bank and other feature generation utilities.
"""

from .analyst_features import AnalystFeatureBank, get_analyst_feature_bank

__all__ = [
    'AnalystFeatureBank',
    'get_analyst_feature_bank'
]