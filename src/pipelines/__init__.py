"""
Pipeline framework for Ares trading bot.

This module provides the base framework and common components for all
pipeline implementations (live trading, backtesting, training).
"""

from .base_pipeline import BasePipeline, PipelineConfig
# Optional imports if modules exist; keep namespace clean
try:
    passpasspassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from .live_trading_pipeline import LiveTradingPipeline
except Exception:  # Module may be optional in minimal envs
LiveTradingPipeline = None  # type: ignore

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from .backtesting_pipeline import BacktestingPipeline
except Exception:
    passpassBacktestingPipeline = None  # type: ignore

try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
from .training_pipeline import TrainingPipeline
except Exception:
    passpassTrainingPipeline = None  # type: ignore

__all__ = [
"BasePipeline",
"PipelineConfig",
"LiveTradingPipeline",
"BacktestingPipeline",
"TrainingPipeline",
]
