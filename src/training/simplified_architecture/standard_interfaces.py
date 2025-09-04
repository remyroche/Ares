"""
Standard Interfaces for ML Pipeline Components

This module defines the standard interfaces that all pipeline steps must implement,
ensuring consistency and predictability across the system.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union
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

@dataclass
class StepResult:
    """Standardized result from a pipeline step."""
    status: StepStatus
    data: Optional[Any] = None
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

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

@dataclass
class StepConfig:
    """Configuration for a pipeline step."""
    name: str
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: int = 1
    fail_fast: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)

class IPipelineStep(ABC):
    """
    Base interface for all pipeline steps.
    
    Every step must implement this interface to ensure consistency.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this step."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of this step implementation."""

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

class IDataStep(IPipelineStep):
    """Interface for data loading/preprocessing steps."""

    @abstractmethod
    async def load_data(self, source: str) -> pd.DataFrame:
        """Load data from specified source."""

    @abstractmethod
    async def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data meets requirements."""

class ILabelingStep(IPipelineStep):
    """Interface for labeling steps."""

    @abstractmethod
    async def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create labels for the input data."""

    @abstractmethod
    def get_label_distribution(self, labels: pd.Series) -> Dict[str, int]:
        """Get distribution of labels."""

class IFeatureStep(IPipelineStep):
    """Interface for feature engineering steps."""

    @abstractmethod
    async def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from input data."""

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""

class ITrainingStep(IPipelineStep):
    """Interface for model training steps."""

    @abstractmethod
    async def train_model(self, features: pd.DataFrame, labels: pd.Series) -> Any:
        """Train a model on the provided features and labels."""

    @abstractmethod
    async def save_model(self, model: Any, path: Path) -> None:
        """Save trained model to disk."""

    @abstractmethod
    async def load_model(self, path: Path) -> Any:
        """Load model from disk."""

class IValidationStep(IPipelineStep):
    """Interface for validation steps."""

    @abstractmethod
    async def validate_model(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model performance on test data."""

    @abstractmethod
    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report."""

class BasePipelineStep(IPipelineStep):
    """
    Base implementation of IPipelineStep with common functionality.
    
    Concrete steps should inherit from this class.
    """

    def __init__(self, config: StepConfig, logger: logging.Logger=None) -> None:
        self.config = config
        self.logger = logger
        self._metrics: Dict[str, Any] = {}
        self._artifacts: Dict[str, Path] = {}
        self._warnings: list[str] = []

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def version(self) -> str:
        return '1.0.0'

    async def validate_inputs(self, **kwargs) -> bool:
        """Default input validation - override in subclasses."""
        return True

    async def execute(self, **kwargs) -> StepResult:
        """
        Standard execution wrapper with error handling and metrics.
        
        Subclasses should implement _execute_impl instead of this method.
        """
        result = StepResult(status=StepStatus.PENDING, start_time=datetime.now())
        try:
            if not await self.validate_inputs(**kwargs):
                raise ValueError('Input validation failed')
            result.status = StepStatus.RUNNING
            if self.config.timeout_seconds:
                output = await asyncio.wait_for(self._execute_impl(**kwargs), timeout=self.config.timeout_seconds)
            else:
                output = await self._execute_impl(**kwargs)
            result.data = output
            result.status = StepStatus.COMPLETED
            result.metrics = self._metrics.copy()
            result.artifacts = self._artifacts.copy()
            result.warnings = self._warnings.copy()
        except asyncio.TimeoutError:
            result.status = StepStatus.FAILED
            result.error = TimeoutError(f'Step timed out after {self.config.timeout_seconds}s')
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = e
            if self.logger:
                self.logger.error(f'Step {self.name} failed: {e}')
        finally:
            result.end_time = datetime.now()
        return result

    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Any:
        """
        Actual implementation of step logic.
        
        Subclasses must implement this method.
        """

    async def cleanup(self) -> None:
        """Default cleanup - override if needed."""
        self._metrics.clear()
        self._artifacts.clear()
        self._warnings.clear()

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to be included in the result."""
        self._metrics[name] = value

    def add_artifact(self, name: str, path: Path) -> None:
        """Add an artifact path to be included in the result."""
        self._artifacts[name] = path

    def add_warning(self, message: str) -> None:
        """Add a warning to be included in the result."""
        self._warnings.append(message)

class SimpleDataStep(BasePipelineStep, IDataStep):
    """Example implementation of a data loading step."""

    @property
    def version(self) -> str:
        return '1.0.1'

    async def load_data(self, source: str) -> pd.DataFrame:
        """Load data from parquet file."""
        if self.logger:
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

    async def _execute_impl(self, source: str) -> pd.DataFrame:
        """Implementation of data loading step."""
        data = await self.load_data(source)
        if not await self.validate_data(data):
            raise ValueError('Data validation failed')
        if self.config.parameters.get('save_snapshot', False):
            snapshot_path = Path(f'data/snapshots/{self.name}_{int(time.time())}.parquet')
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(snapshot_path)
            self.add_artifact('data_snapshot', snapshot_path)
        return data

class SimpleLabelingStep(BasePipelineStep, ILabelingStep):
    """Example implementation of a labeling step."""

    async def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple price direction labels."""
        data['returns'] = data['close'].pct_change()
        data['label'] = (data['returns'] > 0).astype(int)
        distribution = self.get_label_distribution(data['label'])
        for label, count in distribution.items():
            self.add_metric(f'label_{label}_count', count)
        return data

    def get_label_distribution(self, labels: pd.Series) -> Dict[str, int]:
        """Get distribution of labels."""
        return labels.value_counts().to_dict()

    async def _execute_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implementation of labeling step."""
        return await self.create_labels(data)

class StepFactory:
    """Factory for creating pipeline steps based on configuration."""
    _step_registry: Dict[str, type[IPipelineStep]] = {'data_loader': SimpleDataStep, 'labeling': SimpleLabelingStep}

    @classmethod
    def register_step(cls, name: str, step_class: type[IPipelineStep]) -> None:
        """Register a new step type."""
        cls._step_registry[name] = step_class

    @classmethod
    def create_step(cls, config: StepConfig, logger: logging.Logger=None) -> IPipelineStep:
        """Create a step instance from configuration."""
        step_type = config.parameters.get('type', config.name)
        if step_type not in cls._step_registry:
            raise ValueError(f'Unknown step type: {step_type}')
        step_class = cls._step_registry[step_type]
        return step_class(config, logger)

async def example_usage() -> None:
    """Example of using standard interfaces."""
    data_config = StepConfig(name='data_loader', parameters={'required_columns': ['open', 'high', 'low', 'close', 'volume'], 'save_snapshot': True})
    labeling_config = StepConfig(name='labeling', timeout_seconds=30)
    data_step = StepFactory.create_step(data_config)
    labeling_step = StepFactory.create_step(labeling_config)
    data_result = await data_step.execute(source='data/raw/prices.parquet')
    if not data_result.is_success:
        print(f'Data loading failed: {data_result.error}')
        return
    label_result = await labeling_step.execute(data=data_result.data)
    if not label_result.is_success:
        print(f'Labeling failed: {label_result.error}')
        return
    print(f'Pipeline completed successfully!')
    print(f'Data metrics: {data_result.metrics}')
    print(f'Label metrics: {label_result.metrics}')
    await data_step.cleanup()
    await labeling_step.cleanup()
if __name__ == '__main__':
    asyncio.run(example_usage())