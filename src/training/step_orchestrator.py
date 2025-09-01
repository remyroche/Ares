#!/usr/bin/env python3
"""Step Orchestrator for Training Pipeline.

This module orchestrates the execution of training steps with progress saving
and resuming capabilities. Now uses EnhancedTrainingManager for 16-step pipeline.
"""

import importlib
import inspect
import os
from typing import Any

from src.training.progress_manager import ProgressManager
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error, failed)


class StepOrchestrator:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="steporchestrator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize StepOrchestrator."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
                """Orchestrates training step execution with progress management using EnhancedTrainingManager."""

    def __init__(self, symbol: str, exchange: str, data_dir: str = "data/training") -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.data_dir = data_dir
        self.logger = system_logger.getChild("StepOrchestrator")

        # Initialize progress manager
        self.progress_manager = ProgressManager(symbol, exchange, data_dir)

        # Define available steps in order (for reference)
        self.available_steps = [
            "step01_data_collection", "step01_5_data_converter",
            "step02_feature_engineering",
            "step02_5_sr_optimization",
            "step03_hmm_regime_discovery",
            "step04_processing_labeling",
            "step05_regime_data_splitting",
            "step06_hmm_based_training",
            "step06_5_unified_regime_intelligence",
            "step07_analyst_enhancement",
            "step08_tactician_labeling",
            "step09_tactician_specialist_training",
            "step10_confidence_calibration",
            "step11_final_parameters_optimization",
            "step12_walk_forward_validation",
            "step13_monte_carlo_validation",
            "step14_ab_testing",
            "step15_saving",
            "step16_confidence_calibration",
            "step17_final_parameters_optimization",
            "step18_walk_forward_validation",
            "step19_monte_carlo_validation",
            "step20_ab_testing",
            "step21_saving",
        ]

        # Enhanced training manager
        self.enhanced_training_manager = None

        self.logger.info(f"Initialized StepOrchestrator for {symbol} on {exchange}")

            from src.training.enhanced_training_manager import (
                setup_enhanced_training_manager)

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

        if not module:
                return None

        # Look for the main step class (usually ends with 'Step')
        step_classes = [
            attr
            for attr in dir(module)
            if inspect.isclass(getattr(module, attr)) and attr.endswith("Step")
        ]

        if step_classes:
            self.logger.info(f"✅ Found step class: {step_classes[0]}")
            return step_class

        self.print(error(f"❌ No step class found in {step_name}"))
        return None


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

            # Determine training mode and apply mode-specific parameters
            training_mode = "blank" if os.getenv("BLANK_TRAINING_MODE", "0") == "1" else "full"

            # Apply mode-specific parameters to the configuration
            config = apply_mode_parameters_to_config(config, training_mode, step_name)

            # Get step-specific parameters
            step_params = get_step_specific_parameters(training_mode, step_name)

            # Prepare training input for enhanced training manager
            training_input = {
                "symbol": self.symbol, "exchange": self.exchange, "timeframe": "1m",
                "data_dir": self.data_dir, "start_step": step_name, "force_rerun": force_rerun,
                **step_params  # Include all step-specific parameters
            }

            # Execute the enhanced training pipeline
            success = await self.enhanced_training_manager.execute_enhanced_training(
                training_input)

            if success:
# Save progress
                step_data = {
                    "result": {"status": "SUCCESS"},
                    "pipeline_state": {},
                    "training_input": training_input,
                }

                metadata = {
                    "step_name": step_name, "symbol": self.symbol,
                    "exchange": self.exchange, "force_rerun": force_rerun
                }

                if self.progress_manager.save_step_progress(
                    step_name,
                    return True
                self.print(failed(f"❌ Failed to save progress for {step_name}"))
                return False
            self.print(failed(f"❌ Step {step_name} failed"))
            return False

        except Exception as e:
            error_msg = f"Step {step_name} failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

    def _build_pipeline_state(...) -> ...:
    """..."""
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

            return False

        # Set up enhanced training manager
        setup_success = await self._setup_enhanced_training_manager(config)
        if not setup_success:
                return False

        # Prepare training input for enhanced training manager
        # Use proper lookback_days based on training mode
        from src.config.training_modes import (
            BLANK_TRAINING_LOOKBACK_DAYS,
            FULL_TRAINING_LOOKBACK_DAYS,
            get_step_specific_parameters, apply_mode_parameters_to_config,
        )

        # Determine training mode and apply mode-specific parameters
        training_mode = "blank" if os.getenv("BLANK_TRAINING_MODE", "0") == "1" else "full"

        # Apply mode-specific parameters to the configuration
        config = apply_mode_parameters_to_config(config, training_mode, start_step)

        # Get step-specific parameters
        step_params = get_step_specific_parameters(training_mode, start_step)

        training_input = {
            "symbol": self.symbol, "exchange": self.exchange, "timeframe": "1m",
            "data_dir": self.data_dir, "start_step": start_step, "force_rerun": force_rerun,
            **step_params  # Include all step-specific parameters
        }

        # Execute the enhanced training pipeline
        success = await self.enhanced_training_manager.execute_enhanced_training(
            training_input)

        if success:
                "✅ Enhanced 16-step training pipeline completed successfully",
            )
            return True
        self.print(failed("❌ Enhanced 16-step training pipeline failed"))
        return False

            self.available_steps[0],
            config, force_rerun)

            "completed_steps": [],
            "pending_steps": [],
            "latest_step": None,
        }

        latest_step = self.progress_manager.get_latest_step()
        if latest_step:

        for step_name in self.available_steps:
                if self.progress_manager.step_exists(step_name):
status["completed_steps"].append(step_name)
            else:
status["pending_steps"].append(step_name)

        return status


    def list_available_steps(...) -> ...:
    """..."""
                return self.available_steps.copy()
    def _calculate_confidence(self, prediction):
        """Calculate prediction confidence."""
        try:
            if hasattr(prediction, 'predict_proba'):
                return np.max(prediction.predict_proba())
            elif isinstance(prediction, (list, np.ndarray)):
                return np.max(prediction)
            else:
                return 0.5
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0

