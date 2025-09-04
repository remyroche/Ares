"""
Enhanced Standard Interfaces for ML Pipeline Components

This module defines comprehensive interfaces that replace the monolithic architecture
with clean, testable, and extensible components.
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union, Callable
import numpy as np
import pandas as pd
from copy import copy

T = TypeVar('T')
DataType = Union[pd.DataFrame, np.ndarray, Dict[str, Any]]

class StepStatus(Enum):
    """Status of a pipeline step execution."""
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    CANCELLED = 'cancelled'

class StepPriority(Enum):
    """Priority levels for pipeline steps."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class StepResult:
    """Enhanced result from a pipeline step with comprehensive metadata."""
    status: StepStatus
    data: Optional[Any] = None
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_id: Optional[str] = None
    step_version: str = "1.0.0"
    dependencies_satisfied: bool = True
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_success(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        """Check if step failed."""
        return self.status == StepStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'status': self.status.value,
            'data_type': type(self.data).__name__ if self.data is not None else None,
            'error': str(self.error) if self.error else None,
            'metrics': self.metrics,
            'artifacts': {k: str(v) for k, v in self.artifacts.items()},
            'warnings': self.warnings,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'execution_id': self.execution_id,
            'step_version': self.step_version,
            'dependencies_satisfied': self.dependencies_satisfied,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent
        }

@dataclass
class StepConfig:
    """Enhanced configuration for a pipeline step."""
    name: str
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: int = 1
    fail_fast: bool = True
    priority: StepPriority = StepPriority.NORMAL
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.name or not isinstance(self.name, str):
            errors.append("Step name must be a non-empty string")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        if self.retry_count < 0:
            errors.append("Retry count cannot be negative")
        if self.retry_delay_seconds < 0:
            errors.append("Retry delay cannot be negative")
        return errors

class IPipelineStep(ABC):
    """
    Enhanced base interface for all pipeline steps.
    
    This interface ensures consistency and provides comprehensive functionality
    for all pipeline components.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this step."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of this step implementation."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this step does."""

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Schema defining expected input data structure."""

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Schema defining output data structure."""

    @abstractmethod
    async def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.
        
        Returns:
            True if inputs are valid, False otherwise
        """

    @abstractmethod
    async def execute(self, **kwargs) -> StepResult:
        """
        Execute the step logic.
        
        Returns:
            StepResult containing output data and metadata
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources used by this step."""

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status and diagnostics for this step."""

    def can_handle(self, data_type: str) -> bool:
        """Check if this step can handle the given data type."""
        return True  # Override in subclasses for specific type handling

class IDataStep(IPipelineStep):
    """Interface for data loading and preprocessing steps."""

    @abstractmethod
    async def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from specified source."""

    @abstractmethod
    async def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data meets requirements."""

    @abstractmethod
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for downstream steps."""

    @abstractmethod
    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get data quality metrics."""

class ILabelingStep(IPipelineStep):
    """Interface for labeling steps."""

    @abstractmethod
    async def create_labels(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create labels for the input data."""

    @abstractmethod
    def get_label_distribution(self, labels: pd.Series) -> Dict[str, int]:
        """Get distribution of labels."""

    @abstractmethod
    def validate_labels(self, labels: pd.Series) -> bool:
        """Validate created labels."""

    @abstractmethod
    def get_labeling_metadata(self) -> Dict[str, Any]:
        """Get metadata about the labeling process."""

class IFeatureStep(IPipelineStep):
    """Interface for feature engineering steps."""

    @abstractmethod
    async def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Engineer features from input data."""

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""

    @abstractmethod
    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about engineered features."""

    @abstractmethod
    async def validate_features(self, features: pd.DataFrame) -> bool:
        """Validate engineered features."""

class ITrainingStep(IPipelineStep):
    """Interface for model training steps."""

    @abstractmethod
    async def train_model(self, features: pd.DataFrame, labels: pd.Series, **kwargs) -> Any:
        """Train a model on the provided features and labels."""

    @abstractmethod
    async def save_model(self, model: Any, path: Path) -> None:
        """Save trained model to disk."""

    @abstractmethod
    async def load_model(self, path: Path) -> Any:
        """Load model from disk."""

    @abstractmethod
    def get_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Get metadata about the trained model."""

    @abstractmethod
    async def validate_model(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model performance on test data."""

class IValidationStep(IPipelineStep):
    """Interface for validation steps."""

    @abstractmethod
    async def validate_model(self, model: Any, test_data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Validate model performance on test data."""

    @abstractmethod
    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report."""

    @abstractmethod
    async def cross_validate(self, model: Any, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform cross-validation."""

    @abstractmethod
    def get_validation_metrics(self) -> Dict[str, float]:
        """Get validation metrics."""

class IOptimizationStep(IPipelineStep):
    """Interface for optimization steps."""

    @abstractmethod
    async def optimize_parameters(self, model: Any, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Optimize model parameters."""

    @abstractmethod
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get optimization results."""

    @abstractmethod
    async def validate_optimization(self, results: Dict[str, Any]) -> bool:
        """Validate optimization results."""

class BasePipelineStep(IPipelineStep):
    """
    Enhanced base implementation of IPipelineStep with comprehensive functionality.
    
    Concrete steps should inherit from this class and implement the abstract methods.
    """

    def __init__(self, config: StepConfig, logger: Optional[logging.Logger] = None, di_container: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.di_container = di_container
        self._metrics: Dict[str, Any] = {}
        self._artifacts: Dict[str, Path] = {}
        self._warnings: List[str] = []
        self._execution_count = 0
        self._last_execution_time: Optional[datetime] = None
        self._health_status = {"status": "healthy", "last_check": datetime.now()}

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return f"Base implementation of {self.name}"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def validate_inputs(self, **kwargs) -> bool:
        """Enhanced input validation with schema checking."""
        try:
            # Basic validation
            if not self.config.enabled:
                self.logger.warning(f"Step {self.name} is disabled")
                return False

            # Schema validation if provided
            if self.config.output_schema:
                # Implement schema validation logic here
                pass

            # Custom validation rules
            for rule in self.config.validation_rules:
                if not await self._validate_rule(rule, **kwargs):
                    self.logger.error(f"Validation rule failed: {rule}")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Input validation failed for {self.name}: {e}")
            return False

    async def execute(self, **kwargs) -> StepResult:
        """
        Enhanced execution wrapper with comprehensive error handling and metrics.
        
        Subclasses should implement _execute_impl instead of this method.
        """
        execution_id = f"{self.name}_{int(time.time())}_{self._execution_count}"
        result = StepResult(
            status=StepStatus.PENDING,
            start_time=datetime.now(),
            execution_id=execution_id,
            step_version=self.version
        )

        self._execution_count += 1
        self._last_execution_time = result.start_time

        try:
            # Pre-execution validation
            if not await self.validate_inputs(**kwargs):
                result.status = StepStatus.FAILED
                result.error = ValueError('Input validation failed')
                return result

            result.status = StepStatus.RUNNING
            self.logger.info(f"Starting execution of {self.name} (ID: {execution_id})")

            # Execute with timeout if specified
            if self.config.timeout_seconds:
                output = await asyncio.wait_for(
                    self._execute_impl(**kwargs),
                    timeout=self.config.timeout_seconds
                )
            else:
                output = await self._execute_impl(**kwargs)

            result.data = output
            result.status = StepStatus.COMPLETED
            result.metrics = self._metrics.copy()
            result.artifacts = self._artifacts.copy()
            result.warnings = self._warnings.copy()

            self.logger.info(f"Completed execution of {self.name} in {result.duration:.2f}s")

        except asyncio.TimeoutError:
            result.status = StepStatus.FAILED
            result.error = TimeoutError(f'Step timed out after {self.config.timeout_seconds}s')
            self.logger.error(f"Step {self.name} timed out")

        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            self.logger.error(f"Step {self.name} failed: {e}", exc_info=True)

        finally:
            result.end_time = datetime.now()
            # Add resource usage metrics if available
            try:
                import psutil
                process = psutil.Process()
                result.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                result.cpu_usage_percent = process.cpu_percent()
            except ImportError:
                pass

        return result

    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Any:
        """
        Actual implementation of step logic.
        
        Subclasses must implement this method.
        """

    async def cleanup(self) -> None:
        """Enhanced cleanup with resource monitoring."""
        try:
            self._metrics.clear()
            self._artifacts.clear()
            self._warnings.clear()
            self.logger.debug(f"Cleaned up resources for {self.name}")
        except Exception as e:
            self.logger.error(f"Cleanup failed for {self.name}: {e}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "status": self._health_status["status"],
            "last_check": self._health_status["last_check"].isoformat(),
            "execution_count": self._execution_count,
            "last_execution": self._last_execution_time.isoformat() if self._last_execution_time else None,
            "config_valid": len(self.config.validate()) == 0,
            "dependencies_available": self._check_dependencies(),
            "resource_usage": self._get_resource_usage()
        }

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to be included in the result."""
        self._metrics[name] = value

    def add_artifact(self, name: str, path: Path) -> None:
        """Add an artifact path to be included in the result."""
        self._artifacts[name] = path

    def add_warning(self, message: str) -> None:
        """Add a warning to be included in the result."""
        self._warnings.append(message)

    async def _validate_rule(self, rule: Dict[str, Any], **kwargs) -> bool:
        """Validate a custom validation rule."""
        # Implement custom validation logic here
        return True

    def _check_dependencies(self) -> bool:
        """Check if all dependencies are available."""
        if not self.di_container:
            return True
        
        for dep in self.config.dependencies:
            try:
                self.di_container.get(dep)
            except Exception:
                return False
        return True

    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads()
            }
        except ImportError:
            return {"error": "psutil not available"}

class StepFactory:
    """Enhanced factory for creating pipeline steps based on configuration."""
    _step_registry: Dict[str, type[IPipelineStep]] = {}

    @classmethod
    def register_step(cls, name: str, step_class: type[IPipelineStep]) -> None:
        """Register a new step type."""
        cls._step_registry[name] = step_class

    @classmethod
    def create_step(cls, config: StepConfig, logger: Optional[logging.Logger] = None, di_container: Optional[Any] = None) -> IPipelineStep:
        """Create a step instance from configuration."""
        step_type = config.parameters.get('type', config.name)
        if step_type not in cls._step_registry:
            raise ValueError(f'Unknown step type: {step_type}')
        step_class = cls._step_registry[step_type]
        return step_class(config, logger, di_container)

    @classmethod
    def list_available_steps(cls) -> List[str]:
        """List all available step types."""
        return list(cls._step_registry.keys())

    @classmethod
    def get_step_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered step type."""
        if name not in cls._step_registry:
            return None
        
        step_class = cls._step_registry[name]
        return {
            'name': name,
            'class': step_class.__name__,
            'module': step_class.__module__,
            'description': getattr(step_class, 'description', 'No description available'),
            'version': getattr(step_class, 'version', '1.0.0')
        }

# Example implementations for common step types
class SimpleDataStep(BasePipelineStep, IDataStep):
    """Example implementation of a data loading step."""

    @property
    def version(self) -> str:
        return '1.0.1'

    @property
    def description(self) -> str:
        return "Loads and validates data from various sources"

    async def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from parquet file."""
        self.logger.info(f'Loading data from {source}')
        data = pd.read_parquet(source)
        self.add_metric('rows_loaded', len(data))
        self.add_metric('columns_loaded', len(data.columns))
        return data

    async def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data has required columns and no nulls."""
        required_columns = self.config.parameters.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.add_warning(f'Missing columns: {missing_columns}')
            return False
        
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            self.add_warning(f'Null values found: {null_counts[null_counts > 0].to_dict()}')
            return False
        
        return True

    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic data preprocessing."""
        # Remove duplicates
        initial_rows = len(data)
        data = data.drop_duplicates()
        if len(data) < initial_rows:
            self.add_metric('duplicates_removed', initial_rows - len(data))
        
        return data

    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data quality metrics."""
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'null_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_rows': data.duplicated().sum(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }

    async def _execute_impl(self, source: str, **kwargs) -> pd.DataFrame:
        """Implementation of data loading step."""
        data = await self.load_data(source, **kwargs)
        if not await self.validate_data(data):
            raise ValueError('Data validation failed')
        
        data = await self.preprocess_data(data)
        
        if self.config.parameters.get('save_snapshot', False):
            snapshot_path = Path(f'data/snapshots/{self.name}_{int(time.time())}.parquet')
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(snapshot_path)
            self.add_artifact('data_snapshot', snapshot_path)
        
        # Add quality metrics
        quality_metrics = self.get_data_quality_metrics(data)
        for key, value in quality_metrics.items():
            self.add_metric(f'quality_{key}', value)
        
        return data

# Register example steps
StepFactory.register_step('data_loader', SimpleDataStep)