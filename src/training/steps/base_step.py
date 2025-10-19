"""
Base Step Class for Autonomous Pipeline Steps

This module provides the abstract base class that all pipeline steps must inherit from.
Each step becomes autonomous with standardized artifact management and outcome file generation.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback

from src.utils.artifact_manager import ArtifactManager


class BaseStep(ABC):
    """
    Abstract base class for all autonomous pipeline steps.
    
    Each step must:
    - Inherit from this class
    - Implement the execute() method
    - Use artifact_manager for all data I/O
    - Generate Markdown outcome files
    - Be callable only via launcher (no standalone CLI)
    """
    
    def __init__(self, step_name: str):
        """
        Initialize the base step.
        
        Args:
            step_name: Unique name for this step (used for artifact paths and outcomes)
        """
        self.step_name = step_name
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        self.artifact_manager = ArtifactManager(config={})
        
        # Set up artifact manager context with step-category organization
        self.artifact_manager.set_context(
            step_name=step_name,
            datetime=datetime.now()
        )
        
    @abstractmethod
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step logic.
        
        Args:
            config: Configuration dictionary containing all necessary parameters
                   (symbol, exchange, timeframes, execution_mode, etc.)
        
        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': list of artifact paths/metadata created
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
            - 'execution_time': float seconds taken to execute
        """
        pass
    
    def _save_artifact(self, data: Any, artifact_name: str, 
                      artifact_type: str = "data", 
                      compression: str = "auto",
                      metadata: Optional[Dict] = None) -> str:
        """
        Save an artifact using the enhanced artifact manager with step-category organization.
        
        Args:
            data: Data to save (DataFrame, dict, model, etc.)
            artifact_name: Name for the artifact
            artifact_type: Type of artifact ("data", "model", "metadata", etc.)
            compression: Compression method ("auto", "gzip", "lz4", "none")
            metadata: Additional metadata to store with artifact
            
        Returns:
            Path where artifact was saved
        """
        try:
            artifact_path = self.artifact_manager.save(
                data=data,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                compression=compression,
                metadata=metadata
            )
            self.logger.info(f"Saved artifact: {artifact_name} -> {artifact_path}")
            return artifact_path
        except Exception as e:
            self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
            raise
    
    def _get_artifact(self, artifact_name: str, 
                     artifact_type: str = "data") -> Any:
        """
        Retrieve an artifact using the enhanced artifact manager with step-category fallback.
        
        Args:
            artifact_name: Name of the artifact to retrieve
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Retrieved data
        """
        try:
            data = self.artifact_manager.get_artifact(
                artifact_name=artifact_name,
                artifact_type=artifact_type
            )
            self.logger.info(f"Retrieved artifact: {artifact_name}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact {artifact_name}: {e}")
            raise
    
    
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the step with error handling and outcome generation.
        
        This is the main entry point called by the launcher.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution result with outcome report path
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting execution of {self.step_name}")
            
            # Execute the step (async)
            execution_result = await self.execute(config)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_result['execution_time'] = execution_time
            
            # Log completion
            if execution_result.get('success', False):
                self.logger.info(f"Successfully completed {self.step_name} in {execution_time:.2f}s")
            else:
                self.logger.error(f"Failed to complete {self.step_name} after {execution_time:.2f}s")
            
            return execution_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Step {self.step_name} failed: {str(e)}\n{traceback.format_exc()}"
            
            self.logger.error(error_msg)
            
            # Create failure result
            failure_result = {
                'success': False,
                'error': error_msg,
                'execution_time': execution_time,
                'artifacts': [],
                'metrics': {}
            }
            
            
            return failure_result


class StepRegistry:
    """
    Registry for all autonomous steps.
    
    Used by the launcher to discover and execute steps.
    """
    
    def __init__(self):
        self._steps: Dict[str, type] = {}
    
    def register(self, step_name: str, step_class: type):
        """
        Register a step class.
        
        Args:
            step_name: Unique name for the step
            step_class: Step class that inherits from BaseStep
        """
        if not issubclass(step_class, BaseStep):
            raise ValueError(f"Step class {step_class} must inherit from BaseStep")
        
        self._steps[step_name] = step_class
        logging.getLogger("ares.registry").info(f"Registered step: {step_name}")
    
    def get_step(self, step_name: str) -> type:
        """
        Get a registered step class.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step class
            
        Raises:
            KeyError: If step is not registered
        """
        if step_name not in self._steps:
            raise KeyError(f"Step '{step_name}' not found in registry. Available steps: {list(self._steps.keys())}")
        
        return self._steps[step_name]
    
    def list_steps(self) -> list:
        """
        List all registered step names.
        
        Returns:
            List of step names
        """
        return list(self._steps.keys())
    
    def is_registered(self, step_name: str) -> bool:
        """
        Check if a step is registered.
        
        Args:
            step_name: Name of the step
            
        Returns:
            True if step is registered
        """
        return step_name in self._steps


# Global step registry instance
step_registry = StepRegistry()
