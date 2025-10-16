"""
Backtesting Framework for TAS

Comprehensive backtesting framework for tree architecture search including:
- Historical data backtesting
- Walk-forward analysis
- Out-of-sample testing
- Performance attribution
- Risk analysis
- Scenario testing
- Monte Carlo simulation
"""

from .backtesting_engine import BacktestingEngine, BacktestingConfig, BacktestingResult
from .walk_forward_analysis import WalkForwardAnalyzer, WalkForwardConfig
from .performance_attribution import PerformanceAttributor, AttributionConfig
from .risk_analysis import RiskAnalyzer, RiskConfig
from .scenario_testing import ScenarioTester, ScenarioConfig
from .monte_carlo import MonteCarloSimulator, MonteCarloConfig
from .data_manager import BacktestingDataManager, DataConfig

__all__ = [
    'BacktestingEngine', 'BacktestingConfig', 'BacktestingResult',
    'WalkForwardAnalyzer', 'WalkForwardConfig',
    'PerformanceAttributor', 'AttributionConfig',
    'RiskAnalyzer', 'RiskConfig',
    'ScenarioTester', 'ScenarioConfig',
    'MonteCarloSimulator', 'MonteCarloConfig',
    'BacktestingDataManager', 'DataConfig'
]
