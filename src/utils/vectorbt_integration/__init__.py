"""
VectorBT Integration Module

This module provides VectorBT integration utilities for enhanced backtesting,
portfolio optimization, and research capabilities.
"""

from .vectorbt_backtesting_engine import VectorBTBacktestingEngine
from .vectorbt_research_utils import VectorBTResearchUtils
from .vectorbt_portfolio_optimizer import VectorBTPortfolioOptimizer
from .vectorbt_performance_analyzer import VectorBTPerformanceAnalyzer

__all__ = [
    'VectorBTBacktestingEngine',
    'VectorBTResearchUtils', 
    'VectorBTPortfolioOptimizer',
    'VectorBTPerformanceAnalyzer'
]