"""
VectorBT Integration for Backtesting

This module provides VectorBT integration for the backtesting infrastructure,
offering significant performance improvements and enhanced functionality.
"""

from .vectorbt_base import VectorBTBase
from .vectorbt_portfolio import VectorBTPortfolio
from .vectorbt_indicators import VectorBTIndicators
from .vectorbt_metrics import VectorBTMetrics
from .vectorbt_simulation import VectorBTSimulation
from .vectorbt_comparison import VectorBTComparison
from .vectorbt_config import VectorBTConfig

__all__ = [
    'VectorBTBase',
    'VectorBTPortfolio', 
    'VectorBTIndicators',
    'VectorBTMetrics',
    'VectorBTSimulation',
    'VectorBTComparison',
    'VectorBTConfig'
]