"""
Base Classes and Interfaces for Data Qualification Pipeline

This module provides standardized base classes, interfaces, and protocols for all data qualification steps,
ensuring consistent implementation patterns, type safety, and comprehensive documentation.

Key Features:
- Abstract base classes for all data qualification steps
- Standardized interfaces with comprehensive type hints
- Protocol definitions for step implementations
- Result classes with detailed type information
- Performance monitoring and metrics collection
- Comprehensive error handling integration
- Async/await support for modern Python patterns
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Type, Protocol, Generic, TypeVar, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

# Import our custom utilities
from .data_qualification_config import DataQualificationConfig, ValidationResult
from .data_qualification_error_handler import DataQualificationErrorHandler, ErrorContext
from .data_qualification_imports import DataQualificationImportManager, UtilitySuite

# Initialize logger
logger = logging.getLogger(__name__)

# Type variables for generic classes
T = TypeVar('T')
StepResult = TypeVar('StepResult', bound='DataQualificationResult')

@dataclass
class StepMetrics:
    """
    Metrics for data qualification step execution.
    
    This class provides comprehensive metrics tracking for step execution,
    including performance, memory usage, and success rates.
    
    Attributes:
        step_name: Name of the step
        execution_time: Total execution time in seconds
        memory_usage_mb: Peak memory usage in MB
        cpu_usage_percent: Average CPU usage percentage
        success_rate: Success rate (0.0 to 1.0)
        error_count: Number of errors encountered
        retry_count: Number of retries performed
        fallback_used: Whether fallback mechanisms were used
        input_size: Size of input data
        output_size: Size of output data
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
    """
    step_name: str
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    retry_count: int = 0
    fallback_used: bool = False
    input_size: int = 0
    output_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'step_name': self.step_name,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'retry_count': self.retry_count,
            'fallback_used': self.fallback_used,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class DataQualificationResult:
    """
    Base result class for data qualification steps.
    
    This class provides a standardized result format for all data qualification steps,
    including success status, data, metadata, and performance metrics.
    
    Attributes:
        success: Whether the step executed successfully
        data: The main result data (DataFrame, dict, etc.)
        metadata: Additional metadata about the execution
        metrics: Performance and execution metrics
        errors: List of errors encountered (if any)
        warnings: List of warnings generated
        execution_time: Total execution time
        step_name: Name of the step that produced this result
        timestamp: When the result was created
    """
    success: bool
    data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[StepMetrics] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    step_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'metadata': self.metadata,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'errors': self.errors,
            'warnings': self.warnings,
            'execution_time': self.execution_time,
            'step_name': self.step_name,
            'timestamp': self.timestamp.isoformat()
        }
    
    def save_to_file(self, file_path: str, format: str = "json"):
        """
        Save result to file.
        
        Args:
            file_path: Path to save the result
            format: File format ("json" or "yaml")
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif format == "yaml":
            import yaml
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

class DataQualificationStep(ABC):
    """
    Abstract base class for all data qualification steps.
    
    This class provides a standardized interface and common functionality
    for all data qualification steps, including configuration management,
    error handling, performance monitoring, and utility access.
    
    Example:
        >>> class MyStep(DataQualificationStep):
        ...     async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        ...         # Implementation here
        ...         return DataQualificationResult(success=True, data=result)
    """
    
    def __init__(self, config: DataQualificationConfig):
        """
        Initialize the data qualification step.
        
        Args:
            config: Configuration for the step
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self.error_handler = DataQualificationErrorHandler()
        self.import_manager = DataQualificationImportManager()
        self.utilities: Optional[UtilitySuite] = None
        self.metrics = StepMetrics(step_name=self.__class__.__name__)
        
        # Validate configuration
        validation_result = config.validate()
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        # Initialize utilities
        self.utilities = self.import_manager.get_utility_suite()
        
        self.logger.info(f"🚀 {self.__class__.__name__} initialized")
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        """
        Execute the data qualification step.
        
        This method must be implemented by all subclasses and should contain
        the main logic for the step.
        
        Args:
            input_data: Input data for the step
            
        Returns:
            DataQualificationResult with execution results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute method")
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate input data for the step.
        
        This method should validate that the input data contains all required
        fields and is in the correct format.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        raise NotImplementedError("Subclasses must implement validate_input method")
    
    def get_step_config(self) -> Dict[str, Any]:
        """
        Get step-specific configuration.
        
        Returns:
            Dictionary containing step-specific configuration
        """
        step_name = self.__class__.__name__.lower().replace('step', '')
        return self.config.get_step_config(step_name)
    
    def get_utility(self, utility_name: str) -> Optional[Any]:
        """
        Get a utility from the utility suite.
        
        Args:
            utility_name: Name of the utility to get
            
        Returns:
            Utility instance or None if not available
        """
        if self.utilities is None:
            return None
        
        # Navigate through utility suite
        utility_path = utility_name.split('.')
        utility = self.utilities
        
        for path_part in utility_path:
            if hasattr(utility, path_part):
                utility = getattr(utility, path_part)
            elif isinstance(utility, dict) and path_part in utility:
                utility = utility[path_part]
            else:
                return None
        
        return utility
    
    def handle_error(
        self, 
        error: Exception, 
        operation: str,
        fallback_func: Optional[Callable] = None
    ) -> Any:
        """
        Handle errors with automatic recovery.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            fallback_func: Optional fallback function
            
        Returns:
            Result from fallback function or raises exception
        """
        context = ErrorContext(
            step_name=self.__class__.__name__,
            operation=operation
        )
        
        return self.error_handler.handle_utility_failure(
            step_name=self.__class__.__name__,
            utility_name=operation,
            error=error,
            fallback_func=fallback_func,
            context=context
        )
    
    def start_metrics_collection(self):
        """Start collecting performance metrics."""
        self.metrics = StepMetrics(step_name=self.__class__.__name__)
        self.metrics.timestamp = datetime.now()
    
    def stop_metrics_collection(self):
        """Stop collecting performance metrics."""
        if self.metrics:
            self.metrics.execution_time = time.time() - self.metrics.timestamp.timestamp()
    
    def get_metrics(self) -> StepMetrics:
        """
        Get current step metrics.
        
        Returns:
            StepMetrics with current performance data
        """
        return self.metrics
    
    def log_step_start(self, input_data: Dict[str, Any]):
        """Log step execution start."""
        self.logger.info(f"🎯 Starting {self.__class__.__name__}")
        self.logger.debug(f"Input data keys: {list(input_data.keys())}")
        self.start_metrics_collection()
    
    def log_step_completion(self, result: DataQualificationResult):
        """Log step execution completion."""
        self.stop_metrics_collection()
        result.metrics = self.metrics
        
        if result.success:
            self.logger.info(f"✅ {self.__class__.__name__} completed successfully in {result.execution_time:.2f}s")
        else:
            self.logger.error(f"❌ {self.__class__.__name__} failed: {result.errors}")
        
        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(f"⚠️ {warning}")
    
    @asynccontextmanager
    async def execution_context(self, input_data: Dict[str, Any]):
        """
        Context manager for step execution with automatic logging and metrics.
        
        Args:
            input_data: Input data for the step
            
        Yields:
            Dict[str, Any]: Input data
        """
        self.log_step_start(input_data)
        
        try:
            yield input_data
        except Exception as e:
            self.logger.exception(f"Error in {self.__class__.__name__}: {e}")
            raise
        finally:
            self.stop_metrics_collection()

class DataQualificationStepProtocol(Protocol):
    """
    Protocol for data qualification step implementations.
    
    This protocol defines the interface that all data qualification steps
    must implement, providing type safety and consistency.
    """
    
    def __init__(self, config: DataQualificationConfig) -> None:
        """Initialize the step with configuration."""
        ...
    
    async def execute(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        """Execute the step with input data."""
        ...
    
    def validate_input(self, input_data: Dict[str, Any]) -> ValidationResult:
        """Validate input data."""
        ...
    
    def get_step_config(self) -> Dict[str, Any]:
        """Get step-specific configuration."""
        ...

class DataQualificationPipeline:
    """
    Pipeline for executing multiple data qualification steps.
    
    This class provides orchestration for executing multiple steps in sequence
    or parallel, with comprehensive error handling, metrics collection, and
    result aggregation.
    
    Example:
        >>> pipeline = DataQualificationPipeline(config)
        >>> pipeline.add_step(SROptimizationStep(config))
        >>> pipeline.add_step(HMMRegimeDiscoveryStep(config))
        >>> result = await pipeline.execute(input_data)
    """
    
    def __init__(self, config: DataQualificationConfig):
        """
        Initialize the data qualification pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config
        self.steps: List[DataQualificationStep] = []
        self.logger = logger.getChild('DataQualificationPipeline')
        self.error_handler = DataQualificationErrorHandler()
        self.pipeline_metrics: List[StepMetrics] = []
    
    def add_step(self, step: DataQualificationStep):
        """
        Add a step to the pipeline.
        
        Args:
            step: Data qualification step to add
        """
        self.steps.append(step)
        self.logger.info(f"Added step: {step.__class__.__name__}")
    
    def remove_step(self, step_name: str):
        """
        Remove a step from the pipeline.
        
        Args:
            step_name: Name of the step to remove
        """
        self.steps = [step for step in self.steps if step.__class__.__name__ != step_name]
        self.logger.info(f"Removed step: {step_name}")
    
    async def execute_sequential(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        """
        Execute all steps in sequence.
        
        Args:
            input_data: Input data for the first step
            
        Returns:
            DataQualificationResult with aggregated results
        """
        self.logger.info(f"🚀 Starting sequential pipeline execution with {len(self.steps)} steps")
        
        pipeline_start_time = time.time()
        current_data = input_data
        all_results: List[DataQualificationResult] = []
        pipeline_errors: List[str] = []
        pipeline_warnings: List[str] = []
        
        for i, step in enumerate(self.steps):
            try:
                self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.__class__.__name__}")
                
                # Validate input
                validation_result = step.validate_input(current_data)
                if not validation_result.is_valid:
                    error_msg = f"Input validation failed for {step.__class__.__name__}: {validation_result.errors}"
                    pipeline_errors.append(error_msg)
                    self.logger.error(error_msg)
                    continue
                
                # Execute step
                result = await step.execute(current_data)
                all_results.append(result)
                
                # Update current data for next step
                if result.success and result.data is not None:
                    current_data = result.data if isinstance(result.data, dict) else {"data": result.data}
                
                # Collect errors and warnings
                pipeline_errors.extend(result.errors)
                pipeline_warnings.extend(result.warnings)
                
                # Collect metrics
                if result.metrics:
                    self.pipeline_metrics.append(result.metrics)
                
                # Stop on critical failure
                if not result.success and len(result.errors) > 0:
                    self.logger.error(f"Step {step.__class__.__name__} failed, stopping pipeline")
                    break
                
            except Exception as e:
                error_msg = f"Unexpected error in step {step.__class__.__name__}: {e}"
                pipeline_errors.append(error_msg)
                self.logger.exception(error_msg)
                break
        
        pipeline_execution_time = time.time() - pipeline_start_time
        
        # Create aggregated result
        pipeline_result = DataQualificationResult(
            success=len(pipeline_errors) == 0,
            data=current_data,
            metadata={
                "steps_executed": len(all_results),
                "total_steps": len(self.steps),
                "pipeline_execution_time": pipeline_execution_time,
                "step_results": [result.to_dict() for result in all_results]
            },
            errors=pipeline_errors,
            warnings=pipeline_warnings,
            execution_time=pipeline_execution_time,
            step_name="DataQualificationPipeline"
        )
        
        self.logger.info(f"✅ Pipeline execution completed in {pipeline_execution_time:.2f}s")
        return pipeline_result
    
    async def execute_parallel(self, input_data: Dict[str, Any]) -> DataQualificationResult:
        """
        Execute all steps in parallel.
        
        Args:
            input_data: Input data for all steps
            
        Returns:
            DataQualificationResult with aggregated results
        """
        self.logger.info(f"🚀 Starting parallel pipeline execution with {len(self.steps)} steps")
        
        pipeline_start_time = time.time()
        
        # Execute all steps in parallel
        tasks = []
        for step in self.steps:
            task = asyncio.create_task(step.execute(input_data))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_results: List[DataQualificationResult] = []
        pipeline_errors: List[str] = []
        pipeline_warnings: List[str] = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Step {self.steps[i].__class__.__name__} failed with exception: {result}"
                pipeline_errors.append(error_msg)
                self.logger.error(error_msg)
            else:
                all_results.append(result)
                pipeline_errors.extend(result.errors)
                pipeline_warnings.extend(result.warnings)
                
                if result.metrics:
                    self.pipeline_metrics.append(result.metrics)
        
        pipeline_execution_time = time.time() - pipeline_start_time
        
        # Create aggregated result
        pipeline_result = DataQualificationResult(
            success=len(pipeline_errors) == 0,
            data={"results": [result.to_dict() for result in all_results]},
            metadata={
                "steps_executed": len(all_results),
                "total_steps": len(self.steps),
                "pipeline_execution_time": pipeline_execution_time,
                "execution_mode": "parallel"
            },
            errors=pipeline_errors,
            warnings=pipeline_warnings,
            execution_time=pipeline_execution_time,
            step_name="DataQualificationPipeline"
        )
        
        self.logger.info(f"✅ Parallel pipeline execution completed in {pipeline_execution_time:.2f}s")
        return pipeline_result
    
    async def execute(self, input_data: Dict[str, Any], mode: str = "sequential") -> DataQualificationResult:
        """
        Execute the pipeline.
        
        Args:
            input_data: Input data for the pipeline
            mode: Execution mode ("sequential" or "parallel")
            
        Returns:
            DataQualificationResult with pipeline results
        """
        if mode == "sequential":
            return await self.execute_sequential(input_data)
        elif mode == "parallel":
            return await self.execute_parallel(input_data)
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")
    
    def get_pipeline_metrics(self) -> List[StepMetrics]:
        """
        Get metrics for all steps in the pipeline.
        
        Returns:
            List of StepMetrics for all executed steps
        """
        return self.pipeline_metrics
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        if not self.pipeline_metrics:
            return {"total_steps": len(self.steps), "executed_steps": 0}
        
        total_execution_time = sum(metric.execution_time for metric in self.pipeline_metrics)
        total_memory_usage = sum(metric.memory_usage_mb for metric in self.pipeline_metrics)
        average_success_rate = sum(metric.success_rate for metric in self.pipeline_metrics) / len(self.pipeline_metrics)
        
        return {
            "total_steps": len(self.steps),
            "executed_steps": len(self.pipeline_metrics),
            "total_execution_time": total_execution_time,
            "total_memory_usage_mb": total_memory_usage,
            "average_success_rate": average_success_rate,
            "step_metrics": [metric.to_dict() for metric in self.pipeline_metrics]
        }

# Utility functions
def create_step_result(
    success: bool,
    data: Optional[Any] = None,
    step_name: str = "",
    execution_time: float = 0.0,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> DataQualificationResult:
    """
    Create a standardized step result.
    
    Args:
        success: Whether the step was successful
        data: Result data
        step_name: Name of the step
        execution_time: Execution time in seconds
        errors: List of errors
        warnings: List of warnings
        metadata: Additional metadata
        
    Returns:
        DataQualificationResult instance
    """
    return DataQualificationResult(
        success=success,
        data=data,
        step_name=step_name,
        execution_time=execution_time,
        errors=errors or [],
        warnings=warnings or [],
        metadata=metadata or {}
    )

def validate_dataframe_input(
    data: Any,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> ValidationResult:
    """
    Validate DataFrame input for steps.
    
    Args:
        data: Data to validate
        required_columns: List of required columns
        min_rows: Minimum number of rows required
        
    Returns:
        ValidationResult with validation status
    """
    errors = []
    warnings = []
    
    if not isinstance(data, pd.DataFrame):
        errors.append("Input data must be a pandas DataFrame")
        return ValidationResult(is_valid=False, errors=errors)
    
    if len(data) < min_rows:
        errors.append(f"DataFrame must have at least {min_rows} rows, got {len(data)}")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
    
    if data.isnull().any().any():
        warnings.append("DataFrame contains null values")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

# Export main classes and functions
__all__ = [
    'DataQualificationStep',
    'DataQualificationStepProtocol',
    'DataQualificationPipeline',
    'DataQualificationResult',
    'StepMetrics',
    'create_step_result',
    'validate_dataframe_input'
]