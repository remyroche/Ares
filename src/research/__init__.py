"""
Research Module for Advanced Trading Strategy Development

This module contains research components for various aspects of algorithmic trading,
providing a structured approach to investigating and optimizing trading strategies.

Core Research Areas:
- Market microstructure analysis
- Risk management optimization
- Portfolio construction research
- Execution quality analysis
- Alternative data integration
- Behavioral finance modeling
- Performance attribution analysis
"""

from .market_microstructure import MarketMicrostructureResearcher
from .risk_management import RiskManagementResearcher
from .portfolio_optimization import PortfolioOptimizationResearcher
from .execution_analysis import ExecutionAnalysisResearcher
from .alternative_data import AlternativeDataResearcher
from .behavioral_modeling import BehavioralModelingResearcher
from .performance_attribution import PerformanceAttributionResearcher
from .research_framework import ResearchFramework

__all__ = [
    'MarketMicrostructureResearcher',
    'RiskManagementResearcher', 
    'PortfolioOptimizationResearcher',
    'ExecutionAnalysisResearcher',
    'AlternativeDataResearcher',
    'BehavioralModelingResearcher',
    'PerformanceAttributionResearcher',
    'ResearchFramework'
]