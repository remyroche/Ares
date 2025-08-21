"""
Pipeline framework for Ares trading bot.

This module provides the base framework and common components for all
pipeline implementations (live trading, backtesting, training).
"""

from src.utils.warning_symbols import (
    connection_error,
    critical,
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    missing,
    problem,
    timeout,
    validation_error,
    warning,
)

# Base and live trading pipelines are required
from .base_pipeline import BasePipeline, PipelineConfig
from .live_trading_pipeline import LiveTradingPipeline

# Optional pipelines that may not be present in some builds
BacktestingPipeline = None
TrainingPipeline = None
try:
    from .backtesting_pipeline import BacktestingPipeline  # type: ignore
except Exception:
    BacktestingPipeline = None

try:
    from .training_pipeline import TrainingPipeline  # type: ignore
except Exception:
    TrainingPipeline = None

__all__ = [
    "BasePipeline",
    "PipelineConfig",
    "LiveTradingPipeline",
]

if BacktestingPipeline is not None:
    __all__.append("BacktestingPipeline")
if TrainingPipeline is not None:
    __all__.append("TrainingPipeline")
