#!/usr/bin/env python3
"""
Step Orchestrator for Training Pipeline

This module orchestrates the execution of training steps with progress saving
and resuming capabilities. Now uses EnhancedTrainingManager for 16-step pipeline.
"""

import importlib
import os
import inspect
from typing import Any

from src.training.progress_manager import ProgressManager
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
)


class StepOrchestrator:
    """Orchestrates training step execution with progress management using EnhancedTrainingManager."""

    def __init__(self, symbol: str, exchange: str, data_dir: str = "data/training"):
        self.symbol = symbol
        self.exchange = exchange
        self.data_dir = data_dir
        self.logger = system_logger.getChild("StepOrchestrator")

        # Initialize progress manager
        self.progress_manager = ProgressManager(symbol, exchange, data_dir)

        # Define available steps in order (for reference)
        self.available_steps = [
            "step1_data_collection",
            "step1_5_data_converter",  # NEW STEP
            "step2_processing_labeling_feature_engineering",
            "step3_regime_data_splitting",
            "step4_model_training_ready",
            "step5_analyst_specialist_training",
            "step6_analyst_enhancement",
            "step7_analyst_ensemble_creation",
            "step8_tactician_labeling",
            "step9_tactician_specialist_training",
            "step10_tactician_ensemble_creation",
            "step11_confidence_calibration",
            "step12_final_parameters_optimization",
            "step13_walk_forward_validation",
            "step14_monte_carlo_validation",
            "step15_ab_testing",
            "step16_saving",
        ]

        # Enhanced training manager
        self.enhanced_training_manager = None

        self.logger.info(f"Initialized StepOrchestrator for {symbol} on {exchange}")

    def print(self, message: str) -> None:
        """Print a message using the logger."""
        self.logger.info(message)

    async def _setup_enhanced_training_manager(self, config: dict[str, Any]) -> bool:
        """
        Set up the enhanced training manager.

        Args:
            config: Configuration dictionary

        Returns:
            True if setup successful, False otherwise
        """
        try:
            from src.training.enhanced_training_manager import (
                setup_enhanced_training_manager,
            )

            self.enhanced_training_manager = await setup_enhanced_training_manager(
                config,
            )
            if not self.enhanced_training_manager:
                self.print(failed("❌ Failed to setup enhanced training manager"))
                return False

            # The enhanced training manager is already initialized when returned from setup_enhanced_training_manager
            # No need to call initialize() again

            self.logger.info("✅ Enhanced training manager setup successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to setup enhanced training manager: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    def get_step_module(self, step_name: str) -> Any | None:
        """
        Import and return a step module.

        Args:
            step_name: Name of the step (e.g., 'step1_data_collection')

        Returns:
            Step module if found, None otherwise
        """
        try:
            module_path = f"src.training.steps.{step_name}"
            module = importlib.import_module(module_path)
            self.logger.info(f"✅ Loaded step module: {step_name}")
            return module
        except ImportError:
            self.print(failed("❌ Failed to import step module {step_name}: {e}"))
            return None

    def get_step_class(self, step_name: str) -> Any | None:
        """
        Get the main step class from a step module.

        Args:
            step_name: Name of the step

        Returns:
            Step class if found, None otherwise
        """
        module = self.get_step_module(step_name)
        if not module:
            return None

        # Look for the main step class (usually ends with 'Step')
        step_classes = [
            attr
            for attr in dir(module)
            if inspect.isclass(getattr(module, attr)) and attr.endswith("Step")
        ]

        if step_classes:
            step_class = getattr(module, step_classes[0])
            self.logger.info(f"✅ Found step class: {step_classes[0]}")
            return step_class

        self.print(error("❌ No step class found in {step_name}"))
        return None

    async def execute_step(
        self,
        step_name: str,
        config: dict[str, Any],
        force_rerun: bool = False,
    ) -> bool:
        """
        Execute a single training step using enhanced training manager.

        Args:
            step_name: Name of the step to execute
            config: Configuration dictionary
            force_rerun: If True, rerun even if progress exists

        Returns:
            True if step executed successfully, False otherwise
        """
        self.logger.info(f"🚀 Executing step: {step_name}")

        # Check if step already completed (unless force_rerun)
        if not force_rerun and self.progress_manager.step_exists(step_name):
            self.logger.info(f"⏭️  Step {step_name} already completed, skipping")
            return True

        try:
            # Set up enhanced training manager if not already done
            if not self.enhanced_training_manager:
                setup_success = await self._setup_enhanced_training_manager(config)
                if not setup_success:
                    return False

            # Prepare training input for enhanced training manager
            training_input = {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": "1m",
                "data_dir": self.data_dir,
                "start_step": step_name,
                "force_rerun": force_rerun,
                # Respect blank training defaults: 180 days if BLANK_TRAINING_MODE=1
                "lookback_days": 180
                if os.getenv("BLANK_TRAINING_MODE", "0") == "1"
                else 730,
                "exclude_recent_days": 2,  # Always exclude the last 2 days for both blank and full mode
            }

            # Execute the enhanced training pipeline
            success = await self.enhanced_training_manager.execute_enhanced_training(
                training_input,
            )

            if success:
                # Save progress
                step_data = {
                    "result": {"status": "SUCCESS"},
                    "pipeline_state": {},
                    "training_input": training_input,
                }

                metadata = {
                    "step_name": step_name,
                    "symbol": self.symbol,
                    "exchange": self.exchange,
                    "force_rerun": force_rerun,
                }

                if self.progress_manager.save_step_progress(
                    step_name,
                    step_data,
                    metadata,
                ):
                    self.logger.info(f"✅ Step {step_name} completed successfully")
                    return True
                self.print(failed("❌ Failed to save progress for {step_name}"))
                return False
            self.print(failed("❌ Step {step_name} failed"))
            return False

        except Exception as e:
            error_msg = f"Step {step_name} failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    def _build_pipeline_state(self, current_step: str) -> dict[str, Any]:
        """
        Build pipeline state from previous step progress.

        Args:
            current_step: Current step being executed

        Returns:
            Pipeline state dictionary
        """
        pipeline_state = {}

        # Load progress from all previous steps
        for step_name in self.available_steps:
            if step_name == current_step:
                break  # Stop at current step

            progress = self.progress_manager.load_step_progress(step_name)
            if progress and "data" in progress:
                step_data = progress["data"]
                if "result" in step_data:
                    pipeline_state[step_name] = step_data["result"]
                if "pipeline_state" in step_data:
                    pipeline_state.update(step_data["pipeline_state"])

        self.logger.info(f"📋 Built pipeline state with {len(pipeline_state)} items")
        return pipeline_state

    async def execute_from_step(
        self,
        start_step: str,
        config: dict[str, Any],
        force_rerun: bool = False,
    ) -> bool:
        """
        Execute training pipeline starting from a specific step using enhanced training manager.

        Args:
            start_step: Step to start from
            config: Configuration dictionary
            force_rerun: If True, rerun completed steps

        Returns:
            True if all steps completed successfully, False otherwise
        """
        self.logger.info(f"🚀 Starting execution from step: {start_step}")

        # Find the starting step index
        try:
            self.available_steps.index(start_step)
        except ValueError:
            self.print(error("❌ Unknown step: {start_step}"))
            return False

        # Set up enhanced training manager
        setup_success = await self._setup_enhanced_training_manager(config)
        if not setup_success:
            return False

        # Prepare training input for enhanced training manager
        training_input = {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "timeframe": "1m",
            "data_dir": self.data_dir,
            "start_step": start_step,
            "force_rerun": force_rerun,
            "lookback_days": 30,  # Add missing lookback_days field
            "exclude_recent_days": 2,  # Always exclude the last 2 days for both blank and full mode
        }

        # Execute the enhanced training pipeline
        success = await self.enhanced_training_manager.execute_enhanced_training(
            training_input,
        )

        if success:
            self.logger.info(
                "✅ Enhanced 16-step training pipeline completed successfully",
            )
            return True
        self.print(failed("❌ Enhanced 16-step training pipeline failed"))
        return False

    async def execute_all_steps(
        self,
        config: dict[str, Any],
        force_rerun: bool = False,
    ) -> bool:
        """
        Execute all training steps from the beginning using enhanced training manager.

        Args:
            config: Configuration dictionary
            force_rerun: If True, rerun completed steps

        Returns:
            True if all steps completed successfully, False otherwise
        """
        return await self.execute_from_step(
            self.available_steps[0],
            config,
            force_rerun,
        )

    def get_execution_status(self) -> dict[str, Any]:
        """
        Get the current execution status.

        Returns:
            Dictionary with execution status information
        """
        status = {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "total_steps": len(self.available_steps),
            "completed_steps": [],
            "pending_steps": [],
            "latest_step": None,
        }

        latest_step = self.progress_manager.get_latest_step()
        if latest_step:
            status["latest_step"] = latest_step

        for step_name in self.available_steps:
            if self.progress_manager.step_exists(step_name):
                status["completed_steps"].append(step_name)
            else:
                status["pending_steps"].append(step_name)

        return status

    def clear_progress(self, step_name: str | None = None) -> bool:
        """
        Clear progress for specific step or all steps.

        Args:
            step_name: Step name to clear, or None to clear all

        Returns:
            True if cleared successfully, False otherwise
        """
        return self.progress_manager.clear_progress(step_name)

    def list_available_steps(self) -> list[str]:
        """
        Get list of available steps.

        Returns:
            List of available step names
        """
        return self.available_steps.copy()
