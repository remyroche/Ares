"""
Pipeline framework for Ares trading bot.

This module provides the base framework and common components for all
pipeline implementations (live trading, backtesting, training).
"""

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
