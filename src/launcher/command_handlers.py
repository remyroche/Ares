#!/usr/bin/env python3
"""
Command Handlers for Ares Launcher

This module contains specialized command handlers that extract command-specific
logic from the main AresLauncher class, reducing complexity and improving maintainability.
"""

import asyncio

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.utils.common_operations import format_datetime, get_current_datetime
import logging

class BaseCommandHandler(ABC):
    """Base class for all command handlers."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.logger = launcher.logger

    @abstractmethod
    def execute(self, **kwargs) -> bool:
        """Execute the command with the given parameters."""
        pass

class TrainingCommandHandler(BaseCommandHandler):
    """Handles training-related commands (light, blank, full)."""

    def execute(self, training_mode: str, symbol: str, exchange: str,
                lookback_days: Optional[int] = None, with_gui: bool = False) -> bool:
        """Execute unified training with the specified mode."""
        return self.launcher._run_unified_training(
            symbol = symbol,
            exchange = exchange,
            training_mode = training_mode,
            lookback_days = lookback_days,
            with_gui = with_gui,
        )

class TradingCommandHandler(BaseCommandHandler):
    """Handles trading-related commands (paper, live, challenger)."""

    def execute(self, trading_mode: str, symbol: str, exchange: str,
                with_gui: bool = False) -> bool:
        """Execute unified trading with the specified mode."""
        return self.launcher._run_unified_trading(
            symbol = symbol,
            exchange = exchange,
            trading_mode = trading_mode,
            with_gui = with_gui,
        )

class StepBasedCommandHandler(BaseCommandHandler):
    """Handles step-based training commands."""

    def execute(self, start_step: str, symbol: str, exchange: str,
                training_mode: str = "blank", force_rerun: bool = False,
                with_gui: bool = False) -> bool:
        """Execute step-based training with validation."""
        return asyncio.run(
            self.launcher.run_step_based_training_with_validation(
                symbol = symbol,
                exchange = exchange,
                start_step = start_step,
                training_mode = training_mode,
                force_rerun = force_rerun,
                with_gui = with_gui,
            )
        )

class DataLoadingCommandHandler(BaseCommandHandler):
    """Handles data loading commands."""

    def execute(self, symbol: str, exchange: str,
                lookback_days: int = 1460, blank_mode: bool = False) -> bool:
        """Execute data loading and consolidation."""
        actual_lookback = 30 if blank_mode else lookback_days
        return self.launcher.run_data_loading(
            symbol = symbol,
            exchange = exchange,
            lookback_days = actual_lookback,
        )

class PipelineCommandHandler(BaseCommandHandler):
    """Handles pipeline execution commands."""

    def execute(self, pipeline_type: str, symbol: str, exchange: str,
                with_gui: bool = False) -> bool:
        """Execute the specified pipeline."""
        pipeline_methods = {
            "data-collection": self.launcher.run_data_collection_pipeline,
            "market-analysis": self.launcher.run_market_analysis_pipeline,
            "model-training": self.launcher.run_model_training_pipeline,
            "optimisation": self.launcher.run_optimisation_pipeline,
            "backtesting": self.launcher.run_backtesting_pipeline,
            "all-pipelines": self.launcher.run_all_pipelines,
        }

        if pipeline_type not in pipeline_methods:
            self.logger.error(f"Unknown pipeline type: {pipeline_type}")
            return False

        return pipeline_methods[pipeline_type](
            symbol = symbol,
            exchange = exchange,
            with_gui = with_gui,
        )

class RegimeCommandHandler(BaseCommandHandler):
    """Handles regime-related commands."""

    def execute(self, subcommand: str, symbol: str, exchange: str,
                with_gui: bool = False) -> bool:
        """Execute regime operations."""
        return asyncio.run(
            self.launcher.run_regime_operations(
                symbol = symbol,
                exchange = exchange,
                subcommand = subcommand,
                with_gui = with_gui,
            )
        )

class UtilityCommandHandler(BaseCommandHandler):
    """Handles utility commands (modes, precompute, resume)."""

    def execute(self, command: str, **kwargs) -> bool:
        """Execute utility commands."""
        if command == "modes":
            return self.launcher.show_training_modes()
        elif command == "precompute":
            return self.launcher.precompute_wavelet_features(
                kwargs.get("symbol"), kwargs.get("exchange")
            )
        elif command == "resume":
            return self.launcher.resume_training(
                kwargs.get("symbol"), kwargs.get("exchange"), kwargs.get("with_gui", False)
            )
        else:
            self.logger.error(f"Unknown utility command: {command}")
            return False

class CommandHandlerFactory:
    """Factory for creating command handlers."""

    @staticmethod
    def create_handler(command: str, launcher) -> BaseCommandHandler:
        """Create the appropriate command handler for the given command."""
        training_commands = ["light", "blank", "full"]
        trading_commands = ["paper", "live", "challenger"]
        step_commands = [
            "step01", "step01_5", "step1_5", "step02",
            "step03", "step3_5", "step04", "step05", "step06", "step07", "step08",
            "step8_5", "step09", "step9_5", "step10", "step11", "step12", "step13",
            "step14", "step15", "step16", "step17", "step18", "step19", "step20", "step21"
        ]
        pipeline_commands = [
            "data-collection", "market-analysis", "model-training",
            "optimisation", "backtesting", "all-pipelines"
        ]
        utility_commands = ["modes", "precompute", "resume"]

        if command in training_commands:
            return TrainingCommandHandler(launcher)
        elif command in trading_commands:
            return TradingCommandHandler(launcher)
        elif command in step_commands:
            return StepBasedCommandHandler(launcher)
        elif command == "load":
            return DataLoadingCommandHandler(launcher)
        elif command in pipeline_commands:
            return PipelineCommandHandler(launcher)
        elif command == "regime":
            return RegimeCommandHandler(launcher)
        elif command in utility_commands:
            return UtilityCommandHandler(launcher)
        else:
            raise ValueError(f"No handler available for command: {command}")
