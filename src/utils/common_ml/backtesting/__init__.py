"""
Common ML Backtesting Utilities

This package provides comprehensive backtesting utilities for the ML pipeline,
including walk-forward validation, Monte Carlo simulations, A/B testing, and model saving.

Available modules:
- backtesting_engine: Core backtesting functionality with M1 optimizations
- monte_carlo_engine: Monte Carlo simulation engine with GPU acceleration
- ab_testing_engine: A/B testing framework with statistical validation
- model_saver: Model saving and persistence utilities
- analytics_reporter: Comprehensive analytics and reporting
"""

from .backtesting_engine import (
    BacktestingEngine,
    WalkForwardValidator,
    BacktestingConfig,
    BacktestingResults
)

from .monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloConfig,
    MonteCarloResults,
    SimulationParameters
)

from .ab_testing_engine import (
    ABTestingEngine,
    ABTestConfig,
    ABTestResults,
    StatisticalTest
)

from .model_saver import (
    ModelSaver,
    ModelSaveConfig,
    ModelMetadata,
    ModelVersion
)

from .analytics_reporter import (
    AnalyticsReporter,
    AnalyticsConfig,
    PerformanceMetrics,
    RiskMetrics
)

from .turnover import (
    calculate_turnover_metrics,
    apply_market_impact_model,
    reject_high_turnover_configs,
)

__all__ = [
    # Backtesting
    'BacktestingEngine',
    'WalkForwardValidator',
    'BacktestingConfig',
    'BacktestingResults',

    # Monte Carlo
    'MonteCarloEngine',
    'MonteCarloConfig',
    'MonteCarloResults',
    'SimulationParameters',

    # A/B Testing
    'ABTestingEngine',
    'ABTestConfig',
    'ABTestResults',
    'StatisticalTest',

    # Model Saving
    'ModelSaver',
    'ModelSaveConfig',
    'ModelMetadata',
    'ModelVersion',

    # Analytics
    'AnalyticsReporter',
    'AnalyticsConfig',
    'PerformanceMetrics',
    'RiskMetrics',

    # Turnover utilities
    'calculate_turnover_metrics',
    'apply_market_impact_model',
    'reject_high_turnover_configs',
]

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Common ML Backtesting Utilities - Comprehensive backtesting, Monte Carlo, A/B testing, and model saving"
