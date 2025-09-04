"""
Enhanced Pipeline Orchestrator

This module provides a comprehensive pipeline orchestrator that replaces the
monolithic enhanced_training_manager.py with a clean, configurable, and
maintainable architecture.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from enum import Enum
import json

from .dependency_injection import EnhancedDIContainer, ServiceLifetime
from .enhanced_interfaces import (
    IPipelineStep, StepResult, StepStatus, StepConfig, StepFactory,
    BasePipelineStep
)
from .enhanced_config_system import (
    ConfigurationManager, PipelineConfiguration, StepConfiguration,
    ConfigurationError, Environment
)

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    status: PipelineStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    execution_id: Optional[str] = None
    pipeline_name: Optional[str] = None
    pipeline_version: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate total pipeline execution duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.status == PipelineStatus.COMPLETED

    @property
    def failed_steps(self) -> List[str]:
        """Get list of failed step names."""
        return [
            name for name, result in self.step_results.items()
            if result.status == StepStatus.FAILED
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'execution_id': self.execution_id,
            'pipeline_name': self.pipeline_name,
            'pipeline_version': self.pipeline_version,
            'failed_steps': self.failed_steps,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'artifacts': {k: str(v) for k, v in self.artifacts.items()},
            'step_results': {k: v.to_dict() for k, v in self.step_results.items()}
        }

class EnhancedPipelineOrchestrator:
    """
    Enhanced Pipeline Orchestrator for ML Training Pipelines.
    
    This orchestrator replaces the monolithic enhanced_training_manager.py with:
    - Configuration-driven pipeline execution
    - Dependency injection for clean component management
    - Standard interfaces for all pipeline steps
    - Comprehensive error handling and recovery
    - Real-time monitoring and metrics collection
    - Parallel execution where possible
    - Checkpointing and resume capabilities
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[PipelineConfiguration] = None,
        logger: Optional[logging.Logger] = None,
        di_container: Optional[EnhancedDIContainer] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.di_container = di_container or EnhancedDIContainer(self.logger)
        self.config_manager = ConfigurationManager(logger=self.logger)
        
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")

        # Pipeline state
        self._current_execution: Optional[PipelineResult] = None
        self._step_instances: Dict[str, IPipelineStep] = {}
        self._execution_history: List[PipelineResult] = []
        self._is_running = False
        self._cancellation_requested = False

        # Register core services
        self._register_core_services()

    async def run(
        self,
        start_from_step: Optional[str] = None,
        stop_at_step: Optional[str] = None,
        parallel_execution: bool = True,
        checkpoint_interval: int = 5
    ) -> PipelineResult:
        """
        Execute the complete pipeline.
        
        Args:
            start_from_step: Step to start execution from (for resuming)
            stop_at_step: Step to stop execution at
            parallel_execution: Whether to execute independent steps in parallel
            checkpoint_interval: Save checkpoint every N steps
            
        Returns:
            PipelineResult with execution details
        """
        if self._is_running:
            raise RuntimeError("Pipeline is already running")

        self._is_running = True
        self._cancellation_requested = False

        execution_id = f"pipeline_{int(time.time())}_{self.config.name}"
        
        result = PipelineResult(
            status=PipelineStatus.PENDING,
            start_time=datetime.now(),
            execution_id=execution_id,
            pipeline_name=self.config.name,
            pipeline_version=self.config.version
        )

        self._current_execution = result

        try:
            self.logger.info(f"Starting pipeline execution: {self.config.name} v{self.config.version}")
            self.logger.info(f"Execution ID: {execution_id}")
            
            result.status = PipelineStatus.RUNNING

            # Initialize step instances
            await self._initialize_steps()

            # Determine execution order
            execution_order = self._determine_execution_order(start_from_step, stop_at_step)
            
            self.logger.info(f"Execution order: {[step.name for step in execution_order]}")

            # Execute steps
            if parallel_execution:
                await self._execute_steps_parallel(execution_order, checkpoint_interval)
            else:
                await self._execute_steps_sequential(execution_order, checkpoint_interval)

            # Check for failures
            failed_steps = result.failed_steps
            if failed_steps:
                result.status = PipelineStatus.FAILED
                result.errors.append(f"Failed steps: {', '.join(failed_steps)}")
                self.logger.error(f"Pipeline failed with failed steps: {failed_steps}")
            else:
                result.status = PipelineStatus.COMPLETED
                self.logger.info("Pipeline completed successfully")

        except asyncio.CancelledError:
            result.status = PipelineStatus.CANCELLED
            self.logger.warning("Pipeline execution was cancelled")
        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        finally:
            result.end_time = datetime.now()
            self._is_running = False
            self._execution_history.append(result)
            
            # Cleanup
            await self._cleanup_steps()

        return result

    async def cancel(self) -> None:
        """Cancel the currently running pipeline."""
        if not self._is_running:
            self.logger.warning("No pipeline is currently running")
            return
        
        self._cancellation_requested = True
        self.logger.info("Pipeline cancellation requested")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        if not self._current_execution:
            return {"status": "not_started"}

        return {
            "status": self._current_execution.status.value,
            "execution_id": self._current_execution.execution_id,
            "start_time": self._current_execution.start_time.isoformat() if self._current_execution.start_time else None,
            "duration": self._current_execution.duration,
            "completed_steps": len([r for r in self._current_execution.step_results.values() if r.is_success]),
            "total_steps": len(self.config.steps),
            "failed_steps": self._current_execution.failed_steps,
            "is_running": self._is_running
        }

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return [result.to_dict() for result in self._execution_history]

    def get_step_health_status(self) -> Dict[str, Any]:
        """Get health status of all pipeline steps."""
        health_status = {}
        for name, step in self._step_instances.items():
            health_status[name] = step.get_health_status()
        return health_status

    def _register_core_services(self) -> None:
        """Register core services in the DI container."""
        # Register configuration
        self.di_container.register_instance("pipeline_config", self.config)
        
        # Register logger
        self.di_container.register_instance("logger", self.logger)
        
        # Register DI container itself
        self.di_container.register_instance("di_container", self.di_container)

    async def _initialize_steps(self) -> None:
        """Initialize all pipeline step instances."""
        self.logger.info("Initializing pipeline steps...")
        
        for step_config in self.config.steps:
            if not step_config.enabled:
                self.logger.info(f"Skipping disabled step: {step_config.name}")
                continue

            try:
                # Create step configuration
                config = StepConfig(
                    name=step_config.name,
                    enabled=step_config.enabled,
                    timeout_seconds=step_config.timeout_seconds,
                    retry_count=step_config.retry_count,
                    retry_delay_seconds=step_config.retry_delay_seconds,
                    fail_fast=step_config.fail_fast,
                    parameters=step_config.parameters,
                    dependencies=step_config.dependencies,
                    output_schema=step_config.output_schema,
                    validation_rules=step_config.validation_rules,
                    resource_limits=step_config.resource_limits,
                    metadata=step_config.metadata
                )

                # Create step instance
                step_instance = StepFactory.create_step(
                    config, 
                    logger=self.logger.getChild(step_config.name),
                    di_container=self.di_container
                )

                # Register step in DI container
                self.di_container.register_instance(
                    f"step_{step_config.name}",
                    step_instance,
                    metadata={"step_config": step_config}
                )

                self._step_instances[step_config.name] = step_instance
                self.logger.info(f"Initialized step: {step_config.name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize step {step_config.name}: {e}")
                raise

    def _determine_execution_order(
        self, 
        start_from_step: Optional[str] = None,
        stop_at_step: Optional[str] = None
    ) -> List[StepConfiguration]:
        """Determine the execution order based on dependencies."""
        # Build dependency graph
        step_map = {step.name: step for step in self.config.steps if step.enabled}
        
        # Topological sort
        visited = set()
        temp_visited = set()
        execution_order = []

        def visit(step_name: str):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {step_name}")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            
            if step_name in step_map:
                step = step_map[step_name]
                for dep in step.dependencies:
                    if dep in step_map:
                        visit(dep)
                
                visited.add(step_name)
                temp_visited.remove(step_name)
                execution_order.append(step)

        # Visit all steps
        for step in self.config.steps:
            if step.enabled and step.name not in visited:
                visit(step.name)

        # Apply start/stop filters
        if start_from_step:
            start_index = next(
                (i for i, step in enumerate(execution_order) if step.name == start_from_step),
                None
            )
            if start_index is not None:
                execution_order = execution_order[start_index:]

        if stop_at_step:
            stop_index = next(
                (i for i, step in enumerate(execution_order) if step.name == stop_at_step),
                None
            )
            if stop_index is not None:
                execution_order = execution_order[:stop_index + 1]

        return execution_order

    async def _execute_steps_sequential(
        self, 
        execution_order: List[StepConfiguration],
        checkpoint_interval: int
    ) -> None:
        """Execute steps sequentially."""
        for i, step_config in enumerate(execution_order):
            if self._cancellation_requested:
                break

            await self._execute_single_step(step_config.name)
            
            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                await self._save_checkpoint()

    async def _execute_steps_parallel(
        self, 
        execution_order: List[StepConfiguration],
        checkpoint_interval: int
    ) -> None:
        """Execute steps in parallel where possible."""
        completed_steps = set()
        step_index = 0

        while step_index < len(execution_order):
            if self._cancellation_requested:
                break

            # Find steps that can be executed in parallel
            ready_steps = []
            for i in range(step_index, len(execution_order)):
                step = execution_order[i]
                if all(dep in completed_steps for dep in step.dependencies):
                    ready_steps.append(step)
                else:
                    break

            if not ready_steps:
                # No ready steps, execute next step sequentially
                step = execution_order[step_index]
                await self._execute_single_step(step.name)
                completed_steps.add(step.name)
                step_index += 1
            else:
                # Execute ready steps in parallel
                tasks = [
                    self._execute_single_step(step.name)
                    for step in ready_steps
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                for step in ready_steps:
                    completed_steps.add(step.name)
                
                step_index += len(ready_steps)

            # Save checkpoint
            if len(completed_steps) % checkpoint_interval == 0:
                await self._save_checkpoint()

    async def _execute_single_step(self, step_name: str) -> None:
        """Execute a single pipeline step."""
        if step_name not in self._step_instances:
            self.logger.error(f"Step not found: {step_name}")
            return

        step = self._step_instances[step_name]
        self.logger.info(f"Executing step: {step_name}")

        try:
            # Get step dependencies from previous results
            dependencies = {}
            for dep_name in step.config.dependencies:
                if dep_name in self._current_execution.step_results:
                    dep_result = self._current_execution.step_results[dep_name]
                    if dep_result.is_success:
                        dependencies[dep_name] = dep_result.data
                    else:
                        raise ValueError(f"Dependency {dep_name} failed")

            # Execute step
            result = await step.execute(**dependencies)
            self._current_execution.step_results[step_name] = result

            if result.is_success:
                self.logger.info(f"Step {step_name} completed successfully")
            else:
                self.logger.error(f"Step {step_name} failed: {result.error}")
                if step.config.fail_fast:
                    raise RuntimeError(f"Step {step_name} failed and fail_fast is enabled")

        except Exception as e:
            self.logger.error(f"Step {step_name} execution failed: {e}")
            error_result = StepResult(
                status=StepStatus.FAILED,
                error=e,
                start_time=datetime.now(),
                end_time=datetime.now()
            )
            self._current_execution.step_results[step_name] = error_result
            
            if step.config.fail_fast:
                raise

    async def _save_checkpoint(self) -> None:
        """Save pipeline execution checkpoint."""
        if not self._current_execution:
            return

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{self._current_execution.execution_id}_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(self._current_execution.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    async def _cleanup_steps(self) -> None:
        """Clean up all step instances."""
        self.logger.info("Cleaning up pipeline steps...")
        
        cleanup_tasks = []
        for step in self._step_instances.values():
            cleanup_tasks.append(step.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._step_instances.clear()
        self.logger.info("Pipeline cleanup completed")

# Factory function for easy pipeline creation
def create_pipeline(
    config_path: Union[str, Path],
    environment: Optional[Environment] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedPipelineOrchestrator:
    """
    Create a pipeline orchestrator from configuration file.
    
    Args:
        config_path: Path to pipeline configuration file
        environment: Environment override
        logger: Logger instance
        
    Returns:
        Configured pipeline orchestrator
    """
    config_manager = ConfigurationManager(logger=logger)
    config = config_manager.load_config(config_path, environment)
    
    return EnhancedPipelineOrchestrator(
        config=config,
        logger=logger
    )

# Example usage
async def example_usage():
    """Example of using the enhanced pipeline orchestrator."""
    # Create pipeline from configuration
    pipeline = create_pipeline("config/basic_ml_pipeline.yaml")
    
    # Run pipeline
    result = await pipeline.run()
    
    if result.is_success:
        print("Pipeline completed successfully!")
        print(f"Duration: {result.duration:.2f} seconds")
        print(f"Steps completed: {len(result.step_results)}")
    else:
        print(f"Pipeline failed: {result.errors}")
        print(f"Failed steps: {result.failed_steps}")

if __name__ == "__main__":
    asyncio.run(example_usage())