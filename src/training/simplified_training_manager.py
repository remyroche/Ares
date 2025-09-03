"""Simplified training manager with clear separation of concerns.

This module provides a clean, maintainable training manager that orchestrates
the training pipeline using the standardized step system.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.decorators import handles_errors
from src.training.progress_manager import ProgressManager
from src.training.step_config import (
import asyncio

    get_all_steps,
    get_step_config,
    get_step_execution_order,
    validate_step_sequence,
)
from src.utils.logger import system_logger
from src.utils.step_dependency_validator import StepDependencyValidator


class SimplifiedTrainingManager:
    """Simplified training manager for orchestrating the training pipeline.
    
    This manager provides:
    - Clear step execution flow
    - Dependency management
    - Progress tracking
    - Error handling and recovery
    - Modular architecture
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the training manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("SimplifiedTrainingManager")
        
        # Extract basic configuration
        self.symbol = config.get("symbol", "BTCUSDT")
        self.exchange = config.get("exchange", "binance")
        self.data_dir = config.get("data_dir", "data/training")
        
        # Initialize components
        self.progress_manager = ProgressManager(self.symbol, self.exchange, self.data_dir)
        self.dependency_validator = StepDependencyValidator()
        
        # Pipeline state
        self.pipeline_state: Dict[str, Any] = {}
        self.step_instances: Dict[str, Any] = {}
        self.execution_report: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "steps_executed": [],
            "steps_skipped": [],
            "steps_failed": [],
            "total_duration": 0
        }
        
        self.logger.info(f"Initialized SimplifiedTrainingManager for {self.symbol} on {self.exchange}")
    
    @handles_errors(
        exceptions=(Exception,),
        default_return=False,
        context="training manager initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the training manager and validate configuration.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("🔧 Initializing training manager...")
            
            # Validate step sequence
            validation_result = validate_step_sequence()
            if not validation_result["valid"]:
                self.logger.error(f"❌ Step sequence validation failed: {validation_result['issues']}")
                return False
            
            self.logger.info(
                f"✅ Step sequence validated: {validation_result['total_steps']} steps, "
                f"{validation_result['enabled_steps']} enabled"
            )
            
            # Initialize progress tracking
            self.progress_manager.initialize()
            
            # Load previous pipeline state if resuming
            latest_step = self.progress_manager.get_latest_completed_step()
            if latest_step:
                self.logger.info(f"📂 Found previous execution, latest step: {latest_step}")
                self._load_pipeline_state()
            
            self.logger.info("✅ Training manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize training manager: {e}")
            return False
    
    async def execute_pipeline(
        self,
        start_step: Optional[str] = None,
        end_step: Optional[str] = None,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """Execute the training pipeline.
        
        Args:
            start_step: Step number to start from (e.g., "01", "02")
            end_step: Step number to end at (inclusive)
            force_rerun: Force re-execution of completed steps
            
        Returns:
            Execution results
        """
        self.logger.info("🚀 Starting pipeline execution...")
        self.execution_report["start_time"] = datetime.now().isoformat()
        pipeline_start = time.time()
        
        try:
            # Get execution order
            execution_order = get_step_execution_order()
            
            # Filter steps based on start/end
            if start_step:
                try:
                    start_idx = execution_order.index(start_step)
                    execution_order = execution_order[start_idx:]
                except ValueError:
                    self.logger.error(f"❌ Invalid start step: {start_step}")
                    return {"success": False, "error": f"Invalid start step: {start_step}"}
            
            if end_step:
                try:
                    end_idx = execution_order.index(end_step)
                    execution_order = execution_order[:end_idx + 1]
                except ValueError:
                    self.logger.error(f"❌ Invalid end step: {end_step}")
                    return {"success": False, "error": f"Invalid end step: {end_step}"}
            
            # Execute steps in order
            for step_num in execution_order:
                step_config = get_step_config(step_num)
                
                # Check if step is enabled
                if not step_config.enabled:
                    self.logger.info(f"⏭️ Skipping disabled step: {step_config.full_name}")
                    self.execution_report["steps_skipped"].append(step_config.full_name)
                    continue
                
                # Check if already completed (unless force rerun)
                if not force_rerun and self.progress_manager.step_exists(step_config.full_name):
                    self.logger.info(f"✓ Step already completed: {step_config.full_name}")
                    self.execution_report["steps_skipped"].append(step_config.full_name)
                    # Load the step's output into pipeline state
                    self._load_step_output(step_config.full_name)
                    continue
                
                # Validate dependencies
                if not await self._validate_step_dependencies(step_config):
                    self.logger.error(f"❌ Dependencies not met for: {step_config.full_name}")
                    self.execution_report["steps_failed"].append(step_config.full_name)
                    if not step_config.optional:
                        return {
                            "success": False,
                            "error": f"Dependencies not met for required step: {step_config.full_name}",
                            "execution_report": self.execution_report
                        }
                    continue
                
                # Execute the step
                success = await self._execute_step(step_config)
                
                if success:
                    self.execution_report["steps_executed"].append(step_config.full_name)
                else:
                    self.execution_report["steps_failed"].append(step_config.full_name)
                    if not step_config.optional:
                        return {
                            "success": False,
                            "error": f"Required step failed: {step_config.full_name}",
                            "execution_report": self.execution_report
                        }
            
            # Pipeline completed
            self.execution_report["end_time"] = datetime.now().isoformat()
            self.execution_report["total_duration"] = time.time() - pipeline_start
            
            self.logger.info(
                f"✅ Pipeline execution completed in {self.execution_report['total_duration']:.2f}s"
            )
            
            return {
                "success": True,
                "execution_report": self.execution_report,
                "pipeline_state": self.pipeline_state
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline execution failed: {e}")
            self.execution_report["end_time"] = datetime.now().isoformat()
            self.execution_report["total_duration"] = time.time() - pipeline_start
            return {
                "success": False,
                "error": str(e),
                "execution_report": self.execution_report
            }
    
    async def _validate_step_dependencies(self, step_config) -> bool:
        """Validate that all dependencies for a step are satisfied.
        
        Args:
            step_config: StepConfig object
            
        Returns:
            True if all dependencies are satisfied
        """
        for dep_step_num in step_config.dependencies:
            dep_config = get_step_config(dep_step_num)
            
            # Check if dependency was executed or exists in progress
            if (dep_config.full_name not in self.execution_report["steps_executed"] and
                not self.progress_manager.step_exists(dep_config.full_name)):
                self.logger.error(
                    f"❌ Missing dependency {dep_config.full_name} for {step_config.full_name}"
                )
                return False
        
        return True
    
    async def _execute_step(self, step_config) -> bool:
        """Execute a single step.
        
        Args:
            step_config: StepConfig object
            
        Returns:
            True if step executed successfully
        """
        self.logger.info(f"🔄 Executing step: {step_config.full_name}")
        step_start = time.time()
        
        try:
            # Dynamically import and instantiate the step
            step_instance = await self._load_step_instance(step_config)
            if not step_instance:
                return False
            
            # Initialize the step
            await step_instance.initialize()
            
            # Prepare training input
            training_input = {
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": self.config.get("timeframe", "1m"),
                "data_dir": self.data_dir,
                **self.config.get("step_params", {}).get(step_config.step_number, {})
            }
            
            # Execute the step
            result = await step_instance.execute(training_input, self.pipeline_state)
            
            # Check if step succeeded
            if result.get(f"{step_config.full_name}_completed", False):
                # Update pipeline state
                self.pipeline_state.update(result)
                
                # Save progress
                self.progress_manager.save_step_progress(
                    step_config.full_name,
                    {
                        "completed": True,
                        "duration": time.time() - step_start,
                        "timestamp": datetime.now().isoformat(),
                        "outputs": step_config.produced_outputs
                    }
                )
                
                self.logger.info(
                    f"✅ Step completed successfully: {step_config.full_name} "
                    f"({time.time() - step_start:.2f}s)"
                )
                return True
            else:
                self.logger.error(
                    f"❌ Step failed: {step_config.full_name} - "
                    f"{result.get(f'{step_config.full_name}_failure_reason', 'Unknown error')}"
                )
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Error executing step {step_config.full_name}: {e}")
            return False
    
    async def _load_step_instance(self, step_config):
        """Dynamically load and instantiate a step class.
        
        Args:
            step_config: StepConfig object
            
        Returns:
            Step instance or None if loading failed
        """
        try:
            # Import the module
            import importlib
            module = importlib.import_module(step_config.module_path)
            
            # Get the class
            step_class = getattr(module, step_config.class_name)
            
            # Instantiate with config
            step_instance = step_class(self.config)
            
            return step_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load step {step_config.full_name}: {e}")
            return None
    
    def _load_pipeline_state(self) -> None:
        """Load pipeline state from previous executions."""
        # This would load saved pipeline state from disk
        # For now, we'll just initialize an empty state
        self.pipeline_state = {}
    
    def _load_step_output(self, step_name: str) -> None:
        """Load output from a previously completed step.
        
        Args:
            step_name: Full step name
        """
        # This would load the step's output data
        # For now, we'll just mark it as loaded
        self.pipeline_state[f"{step_name}_loaded"] = True
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status.
        
        Returns:
            Status dictionary
        """
        all_steps = get_all_steps()
        completed_steps = []
        pending_steps = []
        
        for step in all_steps:
            if self.progress_manager.step_exists(step.full_name):
                completed_steps.append(step.full_name)
            else:
                pending_steps.append(step.full_name)
        
        return {
            "total_steps": len(all_steps),
            "completed_steps": completed_steps,
            "pending_steps": pending_steps,
            "execution_report": self.execution_report,
            "pipeline_state_keys": list(self.pipeline_state.keys())
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("🧹 Cleaning up training manager resources...")
        # Clean up any resources
        self.step_instances.clear()
        self.pipeline_state.clear()


# Factory function to create and initialize the training manager
async def create_training_manager(config: Dict[str, Any]) -> SimplifiedTrainingManager:
    """Create and initialize a training manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized SimplifiedTrainingManager
    """
    manager = SimplifiedTrainingManager(config)
    if await manager.initialize():
        return manager
    else:
        raise RuntimeError("Failed to initialize training manager")