"""
NAS-TAS Backtesting and Validation Framework

Comprehensive backtesting framework that integrates with regime detection
and model training systems to provide complete validation capabilities.
"""

from .backtesting_engine import BacktestingEngine, BacktestingConfig
from .walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig
from .performance_attribution import PerformanceAttributor, AttributionConfig
from .scenario_tester import ScenarioTester, ScenarioConfig
from .validation_orchestrator import ValidationOrchestrator, ValidationConfig

__all__ = [
    'BacktestingEngine',
    'BacktestingConfig',
    'WalkForwardAnalyzer',
    'WalkForwardConfig',
    'PerformanceAttributor',
    'AttributionConfig',
    'ScenarioTester',
    'ScenarioConfig',
    'ValidationOrchestrator',
    'ValidationConfig'
]
