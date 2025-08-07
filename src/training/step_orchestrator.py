#!/usr/bin/env python3
"""
Step Orchestrator for Training Pipeline

This module orchestrates the execution of training steps with progress saving
and resuming capabilities.
"""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.progress_manager import ProgressManager
from src.utils.logger import system_logger


class StepOrchestrator:
    """Orchestrates training step execution with progress management."""

    def __init__(self, symbol: str, exchange: str, data_dir: str = "data/training"):
        self.symbol = symbol
        self.exchange = exchange
        self.data_dir = data_dir
        self.logger = system_logger.getChild("StepOrchestrator")
        
        # Initialize progress manager
        self.progress_manager = ProgressManager(symbol, exchange, data_dir)
        
        # Define available steps in order
        self.available_steps = [
            "step1_data_collection",
            "step2_market_regime_classification",
            "step3_regime_data_splitting",
            "step4_analyst_labeling_feature_engineering",
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
            "step16_saving"
        ]
        
        self.logger.info(f"Initialized StepOrchestrator for {symbol} on {exchange}")

    def get_step_module(self, step_name: str) -> Optional[Any]:
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
        except ImportError as e:
            self.logger.error(f"❌ Failed to import step module {step_name}: {e}")
            return None

    def get_step_class(self, step_name: str) -> Optional[Any]:
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
            attr for attr in dir(module)
            if inspect.isclass(getattr(module, attr)) and attr.endswith('Step')
        ]
        
        if step_classes:
            step_class = getattr(module, step_classes[0])
            self.logger.info(f"✅ Found step class: {step_classes[0]}")
            return step_class
        
        self.logger.error(f"❌ No step class found in {step_name}")
        return None

    async def execute_step(
        self, 
        step_name: str, 
        config: Dict[str, Any], 
        force_rerun: bool = False
    ) -> bool:
        """
        Execute a single training step.
        
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
            # Get step class
            step_class = self.get_step_class(step_name)
            if not step_class:
                self.logger.error(f"❌ Could not load step class for {step_name}")
                return False
            
            # Initialize step
            step_instance = step_class(config)
            
            # Initialize step if it has an initialize method
            if hasattr(step_instance, 'initialize'):
                await step_instance.initialize()
            
            # Prepare training input
            training_input = {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "data_dir": self.data_dir
            }
            
            # Load previous progress to build pipeline state
            pipeline_state = self._build_pipeline_state(step_name)
            
            # Execute step
            if hasattr(step_instance, 'execute'):
                result = await step_instance.execute(training_input, pipeline_state)
            else:
                # Fallback to run_step function if execute method doesn't exist
                run_step_func = getattr(step_instance, 'run_step', None)
                if run_step_func:
                    result = await run_step_func(self.symbol, self.exchange, self.data_dir)
                else:
                    self.logger.error(f"❌ No execute or run_step method found in {step_name}")
                    return False
            
            # Save progress
            step_data = {
                "result": result,
                "pipeline_state": pipeline_state,
                "training_input": training_input
            }
            
            metadata = {
                "step_name": step_name,
                "symbol": self.symbol,
                "exchange": self.exchange,
                "force_rerun": force_rerun
            }
            
            if self.progress_manager.save_step_progress(step_name, step_data, metadata):
                self.logger.info(f"✅ Step {step_name} completed successfully")
                return True
            else:
                self.logger.error(f"❌ Failed to save progress for {step_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Step {step_name} failed: {e}")
            return False

    def _build_pipeline_state(self, current_step: str) -> Dict[str, Any]:
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
        config: Dict[str, Any], 
        force_rerun: bool = False
    ) -> bool:
        """
        Execute training pipeline starting from a specific step.
        
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
            start_index = self.available_steps.index(start_step)
        except ValueError:
            self.logger.error(f"❌ Unknown step: {start_step}")
            return False
        
        # Execute steps from start_step onwards
        steps_to_execute = self.available_steps[start_index:]
        
        for step_name in steps_to_execute:
            self.logger.info(f"🔄 Executing step {step_name} ({steps_to_execute.index(step_name) + 1}/{len(steps_to_execute)})")
            
            success = await self.execute_step(step_name, config, force_rerun)
            if not success:
                self.logger.error(f"❌ Pipeline failed at step: {step_name}")
                return False
        
        self.logger.info("✅ All steps completed successfully")
        return True

    async def execute_all_steps(
        self, 
        config: Dict[str, Any], 
        force_rerun: bool = False
    ) -> bool:
        """
        Execute all training steps from the beginning.
        
        Args:
            config: Configuration dictionary
            force_rerun: If True, rerun completed steps
            
        Returns:
            True if all steps completed successfully, False otherwise
        """
        return await self.execute_from_step(self.available_steps[0], config, force_rerun)

    def get_execution_status(self) -> Dict[str, Any]:
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
            "latest_step": None
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

    def clear_progress(self, step_name: Optional[str] = None) -> bool:
        """
        Clear progress for specific step or all steps.
        
        Args:
            step_name: Step name to clear, or None to clear all
            
        Returns:
            True if cleared successfully, False otherwise
        """
        return self.progress_manager.clear_progress(step_name)

    def list_available_steps(self) -> List[str]:
        """
        Get list of available steps.
        
        Returns:
            List of available step names
        """
        return self.available_steps.copy()
