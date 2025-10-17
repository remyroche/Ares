"""
Component Orchestrator for Backtesting Pipeline

This module provides orchestration capabilities for complex backtesting workflows
using ModularComponent instances. It includes workflow definition, execution,
dependency management, parallel execution coordination, error handling, and
strategy checkpointing.

Key Features:
- Backtesting workflow definition and execution
- Component dependency management
- Parallel execution coordination
- Error handling and recovery
- Progress monitoring
- Strategy checkpointing
- Backtesting-specific orchestration patterns
"""

import time
import logging
import threading
import asyncio
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .modular_architecture import ModularComponent, ValidationResult, ErrorInfo, ErrorSeverity, ErrorCategory
from .component_registry import BacktestingComponentRegistry, ComponentStatus, ComponentType, get_registry


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionMode(Enum):
    """Execution mode for workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONDITIONAL = "conditional"


@dataclass
class WorkflowStep:
    """A step in a backtesting workflow."""
    name: str
    component_name: str
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: Optional[float] = None


@dataclass
class WorkflowDefinition:
    """Definition of a backtesting workflow."""
    name: str
    description: str
    steps: List[WorkflowStep]
    execution_mode: ExecutionMode
    max_parallel_workers: int = 4
    timeout: Optional[float] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    checkpoint_interval: float = 300.0  # seconds
    enable_checkpointing: bool = True
    enable_monitoring: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Execution context for a backtesting workflow."""
    workflow_id: str
    definition: WorkflowDefinition
    status: WorkflowStatus
    start_time: float
    end_time: Optional[float] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[ErrorInfo] = field(default_factory=list)
    warnings: List[ErrorInfo] = field(default_factory=list)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    execution_stats: Dict[str, Any] = field(default_factory=dict)


class BacktestingWorkflowOrchestrator:
    """Orchestrator for backtesting workflows."""
    
    def __init__(self, registry: Optional[BacktestingComponentRegistry] = None, logger: Optional[logging.Logger] = None):
        self.registry = registry or get_registry()
        self.logger = logger or logging.getLogger(__name__)
        self._executions: Dict[str, WorkflowExecution] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # Orchestrator configuration
        self._max_concurrent_workflows = 10
        self._default_timeout = 3600.0  # 1 hour
        self._checkpoint_interval = 300.0  # 5 minutes
        self._enable_monitoring = True
        self._enable_checkpointing = True
        
        # Backtesting-specific settings
        self._enable_strategy_checkpointing = True
        self._enable_portfolio_state_tracking = True
        self._enable_performance_monitoring = True
        self._enable_risk_monitoring = True
    
    def define_workflow(
        self,
        name: str,
        description: str,
        steps: List[WorkflowStep],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        max_parallel_workers: int = 4,
        timeout: Optional[float] = None,
        enable_checkpointing: bool = True,
        enable_monitoring: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowDefinition:
        """
        Define a new backtesting workflow.
        
        Args:
            name: Name of the workflow
            description: Description of the workflow
            steps: List of workflow steps
            execution_mode: Execution mode for the workflow
            max_parallel_workers: Maximum number of parallel workers
            timeout: Timeout for the workflow
            enable_checkpointing: Enable checkpointing
            enable_monitoring: Enable monitoring
            metadata: Additional metadata
            
        Returns:
            WorkflowDefinition object
        """
        # Validate workflow definition
        self._validate_workflow_definition(name, steps)
        
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            steps=steps,
            execution_mode=execution_mode,
            max_parallel_workers=max_parallel_workers,
            timeout=timeout or self._default_timeout,
            enable_checkpointing=enable_checkpointing,
            enable_monitoring=enable_monitoring,
            metadata=metadata or {}
        )
        
        self.logger.info(f"Workflow {name} defined with {len(steps)} steps")
        return workflow
    
    def execute_workflow(
        self,
        definition: WorkflowDefinition,
        workflow_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a backtesting workflow.
        
        Args:
            definition: Workflow definition
            workflow_id: Optional workflow ID
            input_data: Optional input data for the workflow
            
        Returns:
            Workflow execution ID
        """
        with self._lock:
            # Generate workflow ID if not provided
            if workflow_id is None:
                workflow_id = f"{definition.name}_{int(time.time())}"
            
            # Check if workflow ID already exists
            if workflow_id in self._executions:
                raise ValueError(f"Workflow {workflow_id} already exists")
            
            # Check concurrent workflow limit
            running_workflows = sum(1 for exec in self._executions.values() if exec.status == WorkflowStatus.RUNNING)
            if running_workflows >= self._max_concurrent_workflows:
                raise RuntimeError(f"Maximum concurrent workflows ({self._max_concurrent_workflows}) exceeded")
            
            # Create workflow execution
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                definition=definition,
                status=WorkflowStatus.PENDING,
                start_time=time.time(),
                checkpoint_data=input_data or {}
            )
            
            self._executions[workflow_id] = execution
            
            # Start execution in background
            if definition.execution_mode == ExecutionMode.SEQUENTIAL:
                self._execute_sequential_workflow(execution)
            elif definition.execution_mode == ExecutionMode.PARALLEL:
                self._execute_parallel_workflow(execution)
            elif definition.execution_mode == ExecutionMode.PIPELINE:
                self._execute_pipeline_workflow(execution)
            elif definition.execution_mode == ExecutionMode.CONDITIONAL:
                self._execute_conditional_workflow(execution)
            else:
                raise ValueError(f"Unknown execution mode: {definition.execution_mode}")
            
            # Start monitoring if enabled
            if definition.enable_monitoring and not self._monitoring_active:
                self._start_monitoring()
            
            self.logger.info(f"Workflow {workflow_id} execution started")
            return workflow_id
    
    def _validate_workflow_definition(self, name: str, steps: List[WorkflowStep]) -> None:
        """Validate a workflow definition."""
        if not name:
            raise ValueError("Workflow name cannot be empty")
        
        if not steps:
            raise ValueError("Workflow must have at least one step")
        
        # Check for duplicate step names
        step_names = [step.name for step in steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Duplicate step names found")
        
        # Validate dependencies
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(f"Step {step.name} has invalid dependency: {dep}")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(steps):
            raise ValueError("Circular dependencies detected in workflow")
    
    def _has_circular_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Check for circular dependencies in workflow steps."""
        # Build dependency graph
        graph = {step.name: step.dependencies for step in steps}
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for step in steps:
            if step.name not in visited:
                if has_cycle(step.name):
                    return True
        
        return False
    
    def _execute_sequential_workflow(self, execution: WorkflowExecution) -> None:
        """Execute workflow steps sequentially."""
        def run_workflow():
            try:
                execution.status = WorkflowStatus.RUNNING
                
                # Sort steps by dependencies
                sorted_steps = self._topological_sort_steps(execution.definition.steps)
                
                for step in sorted_steps:
                    if execution.status == WorkflowStatus.CANCELLED:
                        break
                    
                    execution.current_step = step.name
                    
                    try:
                        # Execute step
                        result = self._execute_step(step, execution)
                        
                        # Update step status
                        step.status = WorkflowStatus.COMPLETED
                        step.end_time = time.time()
                        step.execution_time = step.end_time - (step.start_time or step.end_time)
                        step.output_data = result
                        
                        execution.completed_steps.append(step.name)
                        execution.results[step.name] = result
                        
                        # Checkpoint if enabled
                        if execution.definition.enable_checkpointing:
                            self._create_checkpoint(execution)
                        
                        self.logger.info(f"Step {step.name} completed successfully")
                        
                    except Exception as e:
                        # Handle step failure
                        step.status = WorkflowStatus.FAILED
                        step.error = str(e)
                        step.end_time = time.time()
                        step.execution_time = step.end_time - (step.start_time or step.end_time)
                        
                        execution.failed_steps.append(step.name)
                        execution.errors.append(ErrorInfo(
                            message=f"Step {step.name} failed: {e}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.PROCESSING
                        ))
                        
                        self.logger.error(f"Step {step.name} failed: {e}")
                        
                        # Check retry policy
                        if step.retry_count < step.max_retries:
                            step.retry_count += 1
                            step.status = WorkflowStatus.PENDING
                            step.error = None
                            step.start_time = None
                            step.end_time = None
                            step.execution_time = None
                            
                            self.logger.info(f"Retrying step {step.name} (attempt {step.retry_count + 1})")
                            continue
                        else:
                            execution.status = WorkflowStatus.FAILED
                            break
                
                # Finalize execution
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.end_time = time.time()
                execution.execution_stats = {
                    'total_time': execution.end_time - execution.start_time,
                    'completed_steps': len(execution.completed_steps),
                    'failed_steps': len(execution.failed_steps),
                    'total_errors': len(execution.errors),
                    'total_warnings': len(execution.warnings)
                }
                
                self.logger.info(f"Workflow {execution.workflow_id} completed with status {execution.status.value}")
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = time.time()
                execution.errors.append(ErrorInfo(
                    message=f"Workflow execution failed: {e}",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.PROCESSING
                ))
                
                self.logger.error(f"Workflow {execution.workflow_id} failed: {e}")
        
        # Start execution in background thread
        self._executor.submit(run_workflow)
    
    def _execute_parallel_workflow(self, execution: WorkflowExecution) -> None:
        """Execute workflow steps in parallel where possible."""
        def run_workflow():
            try:
                execution.status = WorkflowStatus.RUNNING
                
                # Group steps by dependency level
                dependency_levels = self._group_steps_by_dependency_level(execution.definition.steps)
                
                for level, steps in dependency_levels.items():
                    if execution.status == WorkflowStatus.CANCELLED:
                        break
                    
                    # Execute steps in parallel within the same level
                    with ThreadPoolExecutor(max_workers=execution.definition.max_parallel_workers) as executor:
                        future_to_step = {
                            executor.submit(self._execute_step, step, execution): step
                            for step in steps
                        }
                        
                        for future in as_completed(future_to_step):
                            step = future_to_step[future]
                            
                            try:
                                result = future.result()
                                
                                # Update step status
                                step.status = WorkflowStatus.COMPLETED
                                step.end_time = time.time()
                                step.execution_time = step.end_time - (step.start_time or step.end_time)
                                step.output_data = result
                                
                                execution.completed_steps.append(step.name)
                                execution.results[step.name] = result
                                
                                self.logger.info(f"Step {step.name} completed successfully")
                                
                            except Exception as e:
                                # Handle step failure
                                step.status = WorkflowStatus.FAILED
                                step.error = str(e)
                                step.end_time = time.time()
                                step.execution_time = step.end_time - (step.start_time or step.end_time)
                                
                                execution.failed_steps.append(step.name)
                                execution.errors.append(ErrorInfo(
                                    message=f"Step {step.name} failed: {e}",
                                    severity=ErrorSeverity.HIGH,
                                    category=ErrorCategory.PROCESSING
                                ))
                                
                                self.logger.error(f"Step {step.name} failed: {e}")
                                
                                # Check if we should fail the entire workflow
                                if step.max_retries == 0:
                                    execution.status = WorkflowStatus.FAILED
                                    break
                    
                    # Checkpoint after each level
                    if execution.definition.enable_checkpointing:
                        self._create_checkpoint(execution)
                
                # Finalize execution
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.end_time = time.time()
                execution.execution_stats = {
                    'total_time': execution.end_time - execution.start_time,
                    'completed_steps': len(execution.completed_steps),
                    'failed_steps': len(execution.failed_steps),
                    'total_errors': len(execution.errors),
                    'total_warnings': len(execution.warnings)
                }
                
                self.logger.info(f"Workflow {execution.workflow_id} completed with status {execution.status.value}")
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = time.time()
                execution.errors.append(ErrorInfo(
                    message=f"Workflow execution failed: {e}",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.PROCESSING
                ))
                
                self.logger.error(f"Workflow {execution.workflow_id} failed: {e}")
        
        # Start execution in background thread
        self._executor.submit(run_workflow)
    
    def _execute_pipeline_workflow(self, execution: WorkflowExecution) -> None:
        """Execute workflow as a pipeline with data flow between steps."""
        def run_workflow():
            try:
                execution.status = WorkflowStatus.RUNNING
                
                # Sort steps by dependencies
                sorted_steps = self._topological_sort_steps(execution.definition.steps)
                
                # Initialize data flow
                data_flow = execution.checkpoint_data.copy()
                
                for step in sorted_steps:
                    if execution.status == WorkflowStatus.CANCELLED:
                        break
                    
                    execution.current_step = step.name
                    
                    try:
                        # Prepare input data for step
                        step_input = self._prepare_step_input(step, data_flow)
                        
                        # Execute step
                        result = self._execute_step(step, execution, step_input)
                        
                        # Update step status
                        step.status = WorkflowStatus.COMPLETED
                        step.end_time = time.time()
                        step.execution_time = step.end_time - (step.start_time or step.end_time)
                        step.output_data = result
                        
                        execution.completed_steps.append(step.name)
                        execution.results[step.name] = result
                        
                        # Update data flow
                        data_flow[step.name] = result
                        
                        # Checkpoint if enabled
                        if execution.definition.enable_checkpointing:
                            self._create_checkpoint(execution)
                        
                        self.logger.info(f"Step {step.name} completed successfully")
                        
                    except Exception as e:
                        # Handle step failure
                        step.status = WorkflowStatus.FAILED
                        step.error = str(e)
                        step.end_time = time.time()
                        step.execution_time = step.end_time - (step.start_time or step.end_time)
                        
                        execution.failed_steps.append(step.name)
                        execution.errors.append(ErrorInfo(
                            message=f"Step {step.name} failed: {e}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.PROCESSING
                        ))
                        
                        self.logger.error(f"Step {step.name} failed: {e}")
                        
                        # Pipeline fails if any step fails
                        execution.status = WorkflowStatus.FAILED
                        break
                
                # Finalize execution
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.end_time = time.time()
                execution.execution_stats = {
                    'total_time': execution.end_time - execution.start_time,
                    'completed_steps': len(execution.completed_steps),
                    'failed_steps': len(execution.failed_steps),
                    'total_errors': len(execution.errors),
                    'total_warnings': len(execution.warnings)
                }
                
                self.logger.info(f"Workflow {execution.workflow_id} completed with status {execution.status.value}")
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = time.time()
                execution.errors.append(ErrorInfo(
                    message=f"Workflow execution failed: {e}",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.PROCESSING
                ))
                
                self.logger.error(f"Workflow {execution.workflow_id} failed: {e}")
        
        # Start execution in background thread
        self._executor.submit(run_workflow)
    
    def _execute_conditional_workflow(self, execution: WorkflowExecution) -> None:
        """Execute workflow with conditional step execution."""
        def run_workflow():
            try:
                execution.status = WorkflowStatus.RUNNING
                
                # Sort steps by dependencies
                sorted_steps = self._topological_sort_steps(execution.definition.steps)
                
                for step in sorted_steps:
                    if execution.status == WorkflowStatus.CANCELLED:
                        break
                    
                    execution.current_step = step.name
                    
                    # Check step condition
                    if step.condition and not step.condition(execution.results):
                        self.logger.info(f"Step {step.name} skipped due to condition")
                        continue
                    
                    try:
                        # Execute step
                        result = self._execute_step(step, execution)
                        
                        # Update step status
                        step.status = WorkflowStatus.COMPLETED
                        step.end_time = time.time()
                        step.execution_time = step.end_time - (step.start_time or step.end_time)
                        step.output_data = result
                        
                        execution.completed_steps.append(step.name)
                        execution.results[step.name] = result
                        
                        # Checkpoint if enabled
                        if execution.definition.enable_checkpointing:
                            self._create_checkpoint(execution)
                        
                        self.logger.info(f"Step {step.name} completed successfully")
                        
                    except Exception as e:
                        # Handle step failure
                        step.status = WorkflowStatus.FAILED
                        step.error = str(e)
                        step.end_time = time.time()
                        step.execution_time = step.end_time - (step.start_time or step.end_time)
                        
                        execution.failed_steps.append(step.name)
                        execution.errors.append(ErrorInfo(
                            message=f"Step {step.name} failed: {e}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.PROCESSING
                        ))
                        
                        self.logger.error(f"Step {step.name} failed: {e}")
                        
                        # Check retry policy
                        if step.retry_count < step.max_retries:
                            step.retry_count += 1
                            step.status = WorkflowStatus.PENDING
                            step.error = None
                            step.start_time = None
                            step.end_time = None
                            step.execution_time = None
                            
                            self.logger.info(f"Retrying step {step.name} (attempt {step.retry_count + 1})")
                            continue
                        else:
                            execution.status = WorkflowStatus.FAILED
                            break
                
                # Finalize execution
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.end_time = time.time()
                execution.execution_stats = {
                    'total_time': execution.end_time - execution.start_time,
                    'completed_steps': len(execution.completed_steps),
                    'failed_steps': len(execution.failed_steps),
                    'total_errors': len(execution.errors),
                    'total_warnings': len(execution.warnings)
                }
                
                self.logger.info(f"Workflow {execution.workflow_id} completed with status {execution.status.value}")
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = time.time()
                execution.errors.append(ErrorInfo(
                    message=f"Workflow execution failed: {e}",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.PROCESSING
                ))
                
                self.logger.error(f"Workflow {execution.workflow_id} failed: {e}")
        
        # Start execution in background thread
        self._executor.submit(run_workflow)
    
    def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution, input_data: Optional[Any] = None) -> Any:
        """Execute a single workflow step."""
        step.start_time = time.time()
        
        try:
            # Get component from registry
            component = self.registry.get_component(step.component_name)
            if component is None:
                raise RuntimeError(f"Component {step.component_name} not found in registry")
            
            # Prepare input data
            if input_data is not None:
                step.input_data = input_data
            elif step.input_data is not None:
                input_data = step.input_data
            else:
                input_data = execution.checkpoint_data.get(step.name)
            
            # Execute component
            result = component.process(input_data, **step.parameters)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step {step.name} execution failed: {e}")
            raise
    
    def _prepare_step_input(self, step: WorkflowStep, data_flow: Dict[str, Any]) -> Any:
        """Prepare input data for a step based on dependencies."""
        if step.input_data is not None:
            return step.input_data
        
        # Collect data from dependencies
        input_data = {}
        for dep in step.dependencies:
            if dep in data_flow:
                input_data[dep] = data_flow[dep]
        
        return input_data if input_data else None
    
    def _topological_sort_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Sort steps by dependencies using topological sort."""
        # Build dependency graph
        graph = {step.name: step.dependencies for step in steps}
        step_map = {step.name: step for step in steps}
        
        # Topological sort
        visited = set()
        result = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor)
            
            result.append(node)
        
        for step in steps:
            if step.name not in visited:
                dfs(step.name)
        
        return [step_map[name] for name in result]
    
    def _group_steps_by_dependency_level(self, steps: List[WorkflowStep]) -> Dict[int, List[WorkflowStep]]:
        """Group steps by dependency level for parallel execution."""
        # Build dependency graph
        graph = {step.name: step.dependencies for step in steps}
        step_map = {step.name: step for step in steps}
        
        # Calculate dependency levels
        levels = {}
        visited = set()
        
        def calculate_level(node):
            if node in visited:
                return levels.get(node, 0)
            
            visited.add(node)
            
            if not graph.get(node, []):
                levels[node] = 0
            else:
                max_dep_level = max(calculate_level(dep) for dep in graph[node])
                levels[node] = max_dep_level + 1
            
            return levels[node]
        
        for step in steps:
            calculate_level(step.name)
        
        # Group by level
        level_groups = {}
        for step in steps:
            level = levels[step.name]
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(step)
        
        return level_groups
    
    def _create_checkpoint(self, execution: WorkflowExecution) -> None:
        """Create a checkpoint for the workflow execution."""
        if not execution.definition.enable_checkpointing:
            return
        
        try:
            checkpoint_data = {
                'workflow_id': execution.workflow_id,
                'definition': execution.definition,
                'status': execution.status.value,
                'current_step': execution.current_step,
                'completed_steps': execution.completed_steps,
                'failed_steps': execution.failed_steps,
                'results': execution.results,
                'errors': [error.__dict__ for error in execution.errors],
                'warnings': [warning.__dict__ for warning in execution.warnings],
                'checkpoint_data': execution.checkpoint_data,
                'timestamp': time.time()
            }
            
            # Save checkpoint to file
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_file = checkpoint_dir / f"{execution.workflow_id}_checkpoint.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            self.logger.debug(f"Checkpoint created for workflow {execution.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint for workflow {execution.workflow_id}: {e}")
    
    def _start_monitoring(self) -> None:
        """Start monitoring thread for workflows."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Monitoring loop for workflows."""
        while self._monitoring_active:
            try:
                with self._lock:
                    # Check for timed out workflows
                    current_time = time.time()
                    for execution in self._executions.values():
                        if execution.status == WorkflowStatus.RUNNING:
                            if execution.definition.timeout and (current_time - execution.start_time) > execution.definition.timeout:
                                execution.status = WorkflowStatus.FAILED
                                execution.errors.append(ErrorInfo(
                                    message="Workflow timed out",
                                    severity=ErrorSeverity.HIGH,
                                    category=ErrorCategory.PROCESSING
                                ))
                                self.logger.warning(f"Workflow {execution.workflow_id} timed out")
                    
                    # Create checkpoints for running workflows
                    if self._enable_checkpointing:
                        for execution in self._executions.values():
                            if execution.status == WorkflowStatus.RUNNING:
                                if (current_time - execution.start_time) % self._checkpoint_interval < 1.0:
                                    self._create_checkpoint(execution)
                
                # Sleep for monitoring interval
                time.sleep(10.0)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(10.0)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution."""
        with self._lock:
            if workflow_id not in self._executions:
                return None
            
            execution = self._executions[workflow_id]
            
            return {
                'workflow_id': workflow_id,
                'status': execution.status.value,
                'start_time': execution.start_time,
                'end_time': execution.end_time,
                'current_step': execution.current_step,
                'completed_steps': execution.completed_steps,
                'failed_steps': execution.failed_steps,
                'total_errors': len(execution.errors),
                'total_warnings': len(execution.warnings),
                'execution_stats': execution.execution_stats
            }
    
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get status of all workflow executions."""
        with self._lock:
            return [self.get_workflow_status(workflow_id) for workflow_id in self._executions.keys()]
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        with self._lock:
            if workflow_id not in self._executions:
                return False
            
            execution = self._executions[workflow_id]
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.CANCELLED
                execution.end_time = time.time()
                self.logger.info(f"Workflow {workflow_id} cancelled")
                return True
            
            return False
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        with self._lock:
            if workflow_id not in self._executions:
                return False
            
            execution = self._executions[workflow_id]
            
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.PAUSED
                self.logger.info(f"Workflow {workflow_id} paused")
                return True
            
            return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        with self._lock:
            if workflow_id not in self._executions:
                return False
            
            execution = self._executions[workflow_id]
            
            if execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                self.logger.info(f"Workflow {workflow_id} resumed")
                return True
            
            return False
    
    def cleanup_workflow(self, workflow_id: str) -> bool:
        """Cleanup a workflow execution."""
        with self._lock:
            if workflow_id not in self._executions:
                return False
            
            execution = self._executions[workflow_id]
            
            # Cleanup components used in the workflow
            for step in execution.definition.steps:
                component = self.registry.get_component(step.component_name)
                if component:
                    try:
                        component.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup component {step.component_name}: {e}")
            
            # Remove execution
            del self._executions[workflow_id]
            
            self.logger.info(f"Workflow {workflow_id} cleaned up")
            return True
    
    def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        with self._lock:
            # Stop monitoring
            self._monitoring_active = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)
            
            # Cancel all running workflows
            for workflow_id in list(self._executions.keys()):
                if self._executions[workflow_id].status == WorkflowStatus.RUNNING:
                    self.cancel_workflow(workflow_id)
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            self.logger.info("Orchestrator shutdown completed")


# Global orchestrator instance
_orchestrator_instance: Optional[BacktestingWorkflowOrchestrator] = None


def get_orchestrator() -> BacktestingWorkflowOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = BacktestingWorkflowOrchestrator()
    return _orchestrator_instance


def define_workflow(
    name: str,
    description: str,
    steps: List[WorkflowStep],
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    max_parallel_workers: int = 4,
    timeout: Optional[float] = None,
    enable_checkpointing: bool = True,
    enable_monitoring: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> WorkflowDefinition:
    """Define a workflow using the global orchestrator."""
    return get_orchestrator().define_workflow(
        name, description, steps, execution_mode, max_parallel_workers,
        timeout, enable_checkpointing, enable_monitoring, metadata
    )


def execute_workflow(
    definition: WorkflowDefinition,
    workflow_id: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None
) -> str:
    """Execute a workflow using the global orchestrator."""
    return get_orchestrator().execute_workflow(definition, workflow_id, input_data)


def get_workflow_status(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow status using the global orchestrator."""
    return get_orchestrator().get_workflow_status(workflow_id)


def cancel_workflow(workflow_id: str) -> bool:
    """Cancel a workflow using the global orchestrator."""
    return get_orchestrator().cancel_workflow(workflow_id)


# Export all public classes and functions
__all__ = [
    'WorkflowStatus',
    'ExecutionMode',
    'WorkflowStep',
    'WorkflowDefinition',
    'WorkflowExecution',
    'BacktestingWorkflowOrchestrator',
    'get_orchestrator',
    'define_workflow',
    'execute_workflow',
    'get_workflow_status',
    'cancel_workflow'
]