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

from .backtesting_pipeline import BacktestingPipeline
from .base_pipeline import BasePipeline, PipelineConfig
from .live_trading_pipeline import LiveTradingPipeline
from .training_pipeline import TrainingPipeline

__all__ = [
    "BasePipeline",
    "PipelineConfig",
    "LiveTradingPipeline",
    "BacktestingPipeline",
    "TrainingPipeline",
]
