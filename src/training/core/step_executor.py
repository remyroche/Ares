"""Step executor for running individual pipeline steps.

This module handles the execution of individual steps with proper
error handling, validation, and progress tracking.
"""

from typing import Any, Dict, Optional, Type
import importlib
import time
from datetime import datetime
from pathlib import Path

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.base_step import BaseStep
from src.training.step_config import StepConfig


class StepExecutor:
    """Executes individual pipeline steps."""
    
    def __init__(self, progress_manager=None):
        """Initialize step executor.
        
        Args:
            progress_manager: Optional progress manager for tracking
        """
        self.logger = system_logger.getChild("StepExecutor")
        self.progress_manager = progress_manager
        self.step_cache: Dict[str, BaseStep] = {}
        
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="step loading"
    )
    async def load_step(self, step_config: StepConfig, config: Dict[str, Any]) -> Optional[BaseStep]:
        """Load and instantiate a step.
        
        Args:
            step_config: Step configuration
            config: Pipeline configuration
            
        Returns:
            Step instance or None if loading failed
        """
        # Check cache first
        if step_config.full_name in self.step_cache:
            return self.step_cache[step_config.full_name]
        
        try:
            # Import module
            module = importlib.import_module(step_config.module_path)
            
            # Get class
            step_class = getattr(module, step_config.class_name)
            
            # Validate it's a BaseStep subclass
            if not issubclass(step_class, BaseStep):
                self.logger.error(
                    f"Step class {step_config.class_name} does not inherit from BaseStep"
                )
                return None
            
            # Instantiate
            step_instance = step_class(config)
            
            # Cache for reuse
            self.step_cache[step_config.full_name] = step_instance
            
            self.logger.info(f"✅ Loaded step: {step_config.full_name}")
            return step_instance
            
        except ImportError as e:
            self.logger.error(f"Failed to import step module {step_config.module_path}: {e}")
            return None
        except AttributeError as e:
            self.logger.error(f"Step class {step_config.class_name} not found: {e}")
            return None
        except Exception as e:
            self.logger.exception(f"Error loading step {step_config.full_name}: {e}")
            return None
    
    async def execute_step(
        self,
        step_config: StepConfig,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single step.
        
        Args:
            step_config: Step configuration
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Pipeline configuration
            
        Returns:
            Execution result dictionary
        """
        self.logger.info(f"🔄 Executing step: {step_config.full_name}")
        start_time = time.time()
        
        # Create result structure
        result = {
            "step_name": step_config.full_name,
            "success": False,
            "duration": 0,
            "error": None,
            "outputs": {}
        }
        
        try:
            # Load step instance
            step_instance = await self.load_step(step_config, config)
            if not step_instance:
                result["error"] = "Failed to load step"
                return result
            
            # Initialize step
            await step_instance.initialize()
            
            # Execute step
            updated_state = await step_instance.execute(training_input, pipeline_state)
            
            # Check success
            if updated_state.get(f"{step_config.full_name}_completed", False):
                result["success"] = True
                result["outputs"] = {
                    k: v for k, v in updated_state.items()
                    if k not in pipeline_state  # Only new keys
                }
                
                # Update pipeline state
                pipeline_state.update(updated_state)
                
                # Save progress if manager available
                if self.progress_manager:
                    self.progress_manager.save_step_progress(
                        step_config.full_name,
                        {
                            "completed": True,
                            "duration": time.time() - start_time,
                            "timestamp": datetime.now().isoformat(),
                            "outputs": list(result["outputs"].keys())
                        }
                    )
            else:
                result["error"] = updated_state.get(
                    f"{step_config.full_name}_failure_reason",
                    "Unknown error"
                )
                
        except Exception as e:
            self.logger.exception(f"Step execution error: {e}")
            result["error"] = str(e)
        
        # Record duration
        result["duration"] = time.time() - start_time
        
        # Log result
        if result["success"]:
            self.logger.info(
                f"✅ Step completed: {step_config.full_name} ({result['duration']:.2f}s)"
            )
        else:
            self.logger.error(
                f"❌ Step failed: {step_config.full_name} - {result['error']}"
            )
        
        return result
    
    def validate_dependencies(
        self,
        step_config: StepConfig,
        completed_steps: set,
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step dependencies.
        
        Args:
            step_config: Step configuration
            completed_steps: Set of completed step names
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (dependencies_met, missing_dependencies)
        """
        missing = []
        
        # Check step dependencies
        for dep in step_config.dependencies:
            dep_full_name = f"step{dep}_{dep}"  # This needs proper mapping
            if dep_full_name not in completed_steps:
                missing.append(dep_full_name)
        
        # Check required inputs in pipeline state
        for required_input in step_config.required_inputs:
            if required_input not in pipeline_state:
                missing.append(f"input:{required_input}")
        
        # Check required files
        for file_pattern in step_config.required_files:
            # This is a simplified check - in practice would use glob
            if "*" not in file_pattern:
                if not Path(file_pattern).exists():
                    missing.append(f"file:{file_pattern}")
        
        return len(missing) == 0, missing
    
    def clear_cache(self) -> None:
        """Clear the step instance cache."""
        self.step_cache.clear()
        self.logger.info("🧹 Step cache cleared")