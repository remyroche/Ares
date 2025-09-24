#!/usr/bin/env python3
"""
Step Orchestrator Wrapper for Ares Launcher

This module provides a simplified wrapper around the step orchestrator,
reducing complexity in the main launcher class.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import CONFIG
from src.training.step_orchestrator import StepOrchestrator
from src.launcher.validation_utilities import ValidationFactory
import logging

class StepOrchestratorWrapper:
    """Simplified wrapper for step-based training operations."""
    
    def __init__(self, launcher):
        self.launcher = launcher
        self.logger = launcher.logger
        self.validation_factory = ValidationFactory()
    
    async def run_step_based_training(
        self,
        symbol: str,
        exchange: str,
        start_step: str,
        training_mode: str = "blank",
        force_rerun: bool = False,
        with_gui: bool = False,
    ) -> bool:
        """Run step-based training with comprehensive validation."""
        self.logger.info(f"🚀 Running step-based training for {symbol} on {exchange}")
        self.logger.info(f"📊 Starting from: {start_step}")
        self.logger.info(f"🎯 Training mode: {training_mode}")
        
        # Normalize step name
        start_step = self._normalize_step_name(start_step)
        self.logger.info(f"Starting from step: {start_step}")
        
        # Set training mode environment
        self._set_training_mode_environment(training_mode)
        
        # Prevent blank mode with step01 data collection
        if training_mode == "blank" and start_step == "step1_data_collection":
            self.logger.error("❌ Cannot use blank mode with step1_data_collection")
            self.logger.error("Blank mode is designed for quick testing with limited data")
            self.logger.error("step1_data_collection processes all available data files")
            self.logger.error("Use one of the following instead:")
            self.logger.error("  - python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE (for full data)")
            self.logger.error("  - python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step2_processing_labeling_feature_engineering (for blank mode)")
            return False
        
        if with_gui and not self.launcher.launch_gui("training", symbol, exchange):
            return False
        
        try:
            # Initialize step orchestrator
            orchestrator = StepOrchestrator(symbol, exchange)
            
            # When forcing, set env flags and clear progress/checkpoints from the start step
            if force_rerun:
                os.environ["FORCE"] = "1"
                self._force_fresh_start_from_step(orchestrator, start_step)
                self._clear_checkpoint_files(symbol, exchange, timeframe="1m")
            
            # Validation is now handled by EnhancedTrainingManager
            self.logger.info("🔍 Step validation will be performed by EnhancedTrainingManager")
            
            # Run the step-based training using the orchestrator
            success = await orchestrator.execute_from_step(
                start_step = start_step, 
                config = CONFIG, 
                force_rerun = force_rerun,
            )
            
            if success:
                self.logger.info("✅ Step-based training pipeline completed successfully")
                return True
            else:
                self.logger.error("❌ Step-based training pipeline failed")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Failed to run step-based training pipeline: {e}")
            return False
    
    async def run_step_based_training_with_validation(
        self,
        symbol: str,
        exchange: str,
        start_step: str,
        training_mode: str = "blank",
        force_rerun: bool = False,
        with_gui: bool = False,
    ) -> bool:
        """Run step-based training with comprehensive validation of previous steps."""
        self.logger.info(f"🚀 Running step-based training with validation for {symbol} on {exchange}")
        self.logger.info(f"📊 Starting from: {start_step}")
        self.logger.info(f"🎯 Training mode: {training_mode}")
        
        # Validate previous steps before proceeding
        validator = self.validation_factory.create_validator("step_validation", self.logger)
        validation_success = validator.validate(
            start_step = start_step,
            symbol = symbol,
            exchange = exchange,
            config = CONFIG
        )
        
        if not validation_success:
            self.logger.error(f"❌ Cannot start from {start_step} - previous step validation failed")
            return False
        
        # Run the step pipeline with the specified training mode
        return await self.run_step_based_training(
            symbol = symbol,
            exchange = exchange,
            start_step = start_step,
            training_mode = training_mode,
            force_rerun = force_rerun,
            with_gui = with_gui,
        )
    
    async def run_step2_with_existing_data(
        self,
        symbol: str,
        exchange: str,
        start_step: str = "step2_feature_engineering",
        force_rerun: bool = False,
        with_gui: bool = False,
    ) -> bool:
        """Run step02 with existing data from step01 and step1_5 without triggering new downloads."""
        self.logger.info(f"🚀 Running step02 with existing data for {symbol} on {exchange}")
        self.logger.info("📊 Using existing data from step01 and step1_5 - no new downloads")
        
        # Validate data for step02 readiness
        validator = self.validation_factory.create_validator("data_validation", self.logger)
        validation_success, validation_data = validator.validate(
            symbol = symbol,
            exchange = exchange,
            config = CONFIG
        )
        
        if not validation_success:
            self.logger.error("❌ Cannot start from step02 - data validation failed")
            self.logger.error("Please run step01 and step1_5 first to collect and process data")
            return False
        
        # Log warnings if any issues found
        warnings = validation_data.get("warnings", [])
        if warnings:
            self.logger.warning(f"⚠️ Data validation found {len(warnings)} warnings - proceeding with existing data")
            for warning in warnings:
                self.logger.warning(f"   • {warning}")
        
        self.logger.info("✅ Data validation passed - proceeding with existing data")
        
        return await self.run_step_based_training(
            symbol = symbol,
            exchange = exchange,
            start_step = start_step,
            training_mode="blank",  # Use blank mode for step02 with existing data
            force_rerun = force_rerun,
            with_gui = with_gui,
        )
    
    def _normalize_step_name(self, step_name: Optional[str]) -> Optional[str]:
        """Normalize legacy step names to the current ones used by the orchestrator."""
        if not step_name:
            return None
        
        mapping = {
            # Legacy -> Current
            "step2_market_regime_classification": "step2_processing_labeling_feature_engineering",
            "step3_regime_data_splitting": "step4_regime_data_splitting",
            "step4_analyst_labeling_feature_engineering": "step2_processing_labeling_feature_engineering",
        }
        
        normalized = mapping.get(step_name, step_name)
        if normalized != step_name:
            self.logger.info(f"🔁 Normalized requested step '{step_name}' -> '{normalized}'")
        
        return normalized
    
    def _set_training_mode_environment(self, training_mode: str):
        """Set environment variables for training mode."""
        if training_mode == "light":
            os.environ["LIGHT_TRAINING_MODE"] = "1"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            os.environ["FULL_TRAINING_MODE"] = "0"
            self.logger.info("💡 LIGHT TRAINING MODE: Set LIGHT_TRAINING_MODE = 1 for step-based training (10 days)")
        elif training_mode == "blank":
            os.environ["BLANK_TRAINING_MODE"] = "1"
            os.environ["LIGHT_TRAINING_MODE"] = "0"
            os.environ["FULL_TRAINING_MODE"] = "0"
            self.logger.info("🧪 BLANK TRAINING MODE: Set BLANK_TRAINING_MODE = 1 for step-based training (180 days)")
        elif training_mode == "full":
            os.environ["FULL_TRAINING_MODE"] = "1"
            os.environ["LIGHT_TRAINING_MODE"] = "0"
            os.environ["BLANK_TRAINING_MODE"] = "0"
            self.logger.info("📊 FULL TRAINING MODE: Set FULL_TRAINING_MODE = 1 for step-based training (1460 days)")
    
    def _clear_checkpoint_files(self, symbol: str, exchange: str, timeframe: str = "1m") -> None:
        """Remove enhanced training checkpoints to guarantee a fresh start."""
        try:
            ns_dir = Path("checkpoints") / exchange / symbol / timeframe
            target_file = ns_dir / "training_progress.json"
            if target_file.exists():
                target_file.unlink()
                self.logger.info(f"🗑️  Cleared checkpoint: {target_file}")
            
            # Also clear any individual step checkpoint files
            for checkpoint_file in ns_dir.glob("*.json"):
                if checkpoint_file.name != "training_progress.json":
                    checkpoint_file.unlink()
                    self.logger.info(f"🗑️  Cleared step checkpoint: {checkpoint_file}")
                    
        except OSError as e:
            self.logger.warning(f"Failed to clear checkpoint: {e}")
    
    def _force_fresh_start_from_step(self, orchestrator, start_step: str) -> None:
        """Clear progress from the specified start step onward to enforce a fresh run."""
        try:
            steps = orchestrator.list_available_steps()
            if start_step not in steps:
                self.logger.warning(f"Cannot clear progress: step '{start_step}' is not in available steps")
                return
            
            # Find the index of the starting step
            try:
                start_index = steps.index(start_step)
            except ValueError:
                self.logger.warning(f"⚠️ Unknown step {start_step}, clearing all progress")
                start_index = 0
            
            # Clear progress for the starting step and all subsequent steps
            steps_to_clear = steps[start_index:]
            
            for step in steps_to_clear:
                if orchestrator.clear_progress(step):
                    self.logger.info(f"🧹 Cleared progress for '{step}' (force)")
                else:
                    self.logger.warning(f"⚠️ Failed to clear progress for '{step}'")
            
            self.logger.info(f"✅ Cleared progress for {len(steps_to_clear)} steps: {steps_to_clear}")
            
        except OSError as e:
            self.logger.warning(f"Failed clearing progress from step '{start_step}': {e}")