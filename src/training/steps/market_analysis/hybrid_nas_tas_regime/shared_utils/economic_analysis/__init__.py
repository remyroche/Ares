"""
Shared economic analysis utilities for regime detection systems.

This module provides economic significance and trading viability assessment
utilities that can be used by both NAS and TAS regime detection systems.
"""

from .economic_significance import EconomicSignificanceAnalyzer
from .trading_viability import TradingViabilityAssessor

__all__ = [
    'EconomicSignificanceAnalyzer',
    'TradingViabilityAssessor'
]