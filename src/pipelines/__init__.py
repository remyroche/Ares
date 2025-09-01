""""""""
Pipeline framework for Ares trading bot.
"
This module provides the base framework and common components for all"""
pipeline implementations (live trading, backtesting, training)."""
""""""""

from .base_pipeline import BasePipeline, PipelineConfig
# Optional imports if modules exist; keep namespace clean
try:
    except Exception as e:
        pass
    from .live_trading_pipeline import LiveTradingPipeline
except Exception:  # Module may be optional in minimal envs
    LiveTradingPipeline = None  # type: ignore

try:
    except Exception as e:
        pass
    from .backtesting_pipeline import BacktestingPipeline
except Exception:
    BacktestingPipeline = None  # type: ignore

try:
    except Exception as e:
        pass
    from .training_pipeline import TrainingPipeline
except Exception:
    TrainingPipeline = None  # type: ignore"
"""
__all__ = []"""
    "BasePipeline","""
    "PipelineConfig","""
    "LiveTradingPipeline","""
    "BacktestingPipeline","""
    "TrainingPipeline"","
""
"""""""