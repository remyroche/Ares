"""Base class for all training pipeline steps.

This module provides a standardized interface and common functionality
for all steps in the training pipeline.
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards


class BaseStep(ABC):
    """Abstract base class for all pipeline steps.
    
    Provides common functionality including:
    - Standardized initialization
    - Progress tracking
    - Error handling
    - Validation
    - Reporting
    """
    
    def __init__(self, config: Dict[str, Any], step_number: str, step_name: str):
        """Initialize base step.
        
        Args:
            config: Configuration dictionary
            step_number: Step number (e.g., "01", "02", "03")
            step_name: Descriptive step name
        """
        self.config = config
        self.step_number = step_number
        self.step_name = step_name
        self.full_step_name = f"step{step_number}_{step_name}"
        
        self.logger = system_logger.getChild(self.full_step_name)
        self.standards = PipelineStandards()
        
        # Execution tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.execution_duration: Optional[float] = None
        self.execution_status: str = "pending"
        self.execution_errors: list = []
        
        # Data tracking
        self.input_data_info: Dict[str, Any] = {}
        self.output_data_info: Dict[str, Any] = {}
        
        # Initialize step-specific components
        self._initialize_step()
    
    @abstractmethod
    def _initialize_step(self) -> None:
        """Initialize step-specific components.
        
        This method should be implemented by each step to perform
        any step-specific initialization.
        """
    
    @abstractmethod
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
    
    @abstractmethod
    def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main step logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
    
    @abstractmethod
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="step initialization"
    )
    async def initialize(self) -> None:
        """Initialize the step."""
        self.logger.info(f"🔧 Initializing {self.full_step_name}...")
        self.execution_status = "initialized"
        self.logger.info(f"✅ {self.full_step_name} initialized successfully")
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="step execution"
    )
    async def execute(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the step with full validation and error handling.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with execution results
        """
        self.logger.info(f"🚀 Starting {self.full_step_name}...")
        self.start_time = time.time()
        self.execution_status = "running"
        
        try:
            # Validate inputs
            self.logger.info("🔍 Validating inputs...")
            is_valid, errors = self.validate_inputs(training_input, pipeline_state)
            if not is_valid:
                self.execution_errors.extend(errors)
                self.execution_status = "failed"
                self.logger.error(f"❌ Input validation failed: {errors}")
                return self._create_failure_result(pipeline_state, "Input validation failed", errors)
            
            # Record input data info
            self._record_input_info(training_input, pipeline_state)
            
            # Execute main logic
            self.logger.info("⚙️ Executing step logic...")
            result = await self.execute_logic(training_input, pipeline_state)
            
            # Validate outputs
            self.logger.info("🔍 Validating outputs...")
            is_valid, errors = self.validate_outputs(result)
            if not is_valid:
                self.execution_errors.extend(errors)
                self.execution_status = "failed"
                self.logger.error(f"❌ Output validation failed: {errors}")
                return self._create_failure_result(result, "Output validation failed", errors)
            
            # Record output data info
            self._record_output_info(result)
            
            # Mark success
            self.end_time = time.time()
            self.execution_duration = self.end_time - self.start_time
            self.execution_status = "completed"
            
            # Add step metadata to result
            result[f"{self.full_step_name}_completed"] = True
            result[f"{self.full_step_name}_duration"] = self.execution_duration
            result[f"{self.full_step_name}_timestamp"] = datetime.now().isoformat()
            
            self.logger.info(
                f"✅ {self.full_step_name} completed successfully in {self.execution_duration:.2f}s"
            )
            
            # Generate and save step report
            await self._generate_step_report(result)
            
            return result
            
        except Exception as e:
            self.execution_status = "failed"
            self.execution_errors.append(str(e))
            self.logger.exception(f"❌ {self.full_step_name} failed with error: {e}")
            return self._create_failure_result(pipeline_state, "Execution failed", [str(e)])
    
    def _record_input_info(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Record information about input data."""
        self.input_data_info = {
            "training_input_keys": list(training_input.keys()),
            "pipeline_state_keys": list(pipeline_state.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    def _record_output_info(self, result: Dict[str, Any]) -> None:
        """Record information about output data."""
        self.output_data_info = {
            "result_keys": list(result.keys()),
            "new_keys": [k for k in result.keys() if k not in self.input_data_info.get("pipeline_state_keys", [])],
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_failure_result(
        self,
        pipeline_state: Dict[str, Any],
        reason: str,
        errors: list
    ) -> Dict[str, Any]:
        """Create a failure result."""
        result = pipeline_state.copy()
        result[f"{self.full_step_name}_completed"] = False
        result[f"{self.full_step_name}_failure_reason"] = reason
        result[f"{self.full_step_name}_errors"] = errors
        result[f"{self.full_step_name}_timestamp"] = datetime.now().isoformat()
        return result
    
    async def _generate_step_report(self, result: Dict[str, Any]) -> None:
        """Generate and save a step execution report."""
        report = {
            "step_info": {
                "step_number": self.step_number,
                "step_name": self.step_name,
                "full_step_name": self.full_step_name
            },
            "execution_info": {
                "status": self.execution_status,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": self.execution_duration,
                "errors": self.execution_errors
            },
            "data_info": {
                "input": self.input_data_info,
                "output": self.output_data_info
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        report_dir = Path("reports") / "step_execution" / self.full_step_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Step report saved to {report_file}")
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step.
        
        Returns:
            List of required input keys
        """
        return []
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step.
        
        Returns:
            List of output keys
        """
        return []
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies.
        
        Returns:
            List of step names this step depends on
        """
        return []