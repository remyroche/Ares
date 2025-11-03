"""
Common ML Backtesting Utilities

This package provides comprehensive backtesting utilities for the ML pipeline,
including walk-forward validation, Monte Carlo simulations, and model saving.

Available modules:
- monte_carlo_engine: Monte Carlo simulation engine with GPU acceleration
- model_saver: Model saving and persistence utilities
- analytics_reporter: Comprehensive analytics and reporting
- turnover: Turnover calculation utilities
"""

# Import only modules that exist
try:
    from .monte_carlo_engine import (
        MonteCarloEngine,
        MonteCarloConfig,
        MonteCarloResults,
        SimulationParameters,
        SimulationType
    )
    _has_monte_carlo = True
except ImportError:
    _has_monte_carlo = False

try:
    from .model_saver import (
        ModelSaver,
        get_model_saver
    )
    _has_model_saver = True
except ImportError:
    _has_model_saver = False

try:
    from .analytics_reporter import (
        AnalyticsReporter,
        get_analytics_reporter
    )
    _has_analytics = True
except ImportError:
    _has_analytics = False

try:
    from .turnover import (
        calculate_turnover_metrics,
        apply_market_impact_model,
        reject_high_turnover_configs,
    )
    _has_turnover = True
except ImportError:
    _has_turnover = False

# Build __all__ based on what's actually available
__all__ = []

if _has_monte_carlo:
    __all__.extend([
        'MonteCarloEngine',
        'MonteCarloConfig',
        'MonteCarloResults',
        'SimulationParameters',
        'SimulationType',
    ])

if _has_model_saver:
    __all__.extend([
        'ModelSaver',
        'get_model_saver',
    ])

if _has_analytics:
    __all__.extend([
        'AnalyticsReporter',
        'get_analytics_reporter',
    ])

if _has_turnover:
    __all__.extend([
        'calculate_turnover_metrics',
        'apply_market_impact_model',
        'reject_high_turnover_configs',
    ])

__version__ = "1.0.0"
__author__ = "Ares Trading Bot Team"
__description__ = "Common ML Backtesting Utilities - Monte Carlo simulation and model saving"
