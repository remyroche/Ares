"""
Base Pipeline Class - Eliminates Code Duplication

This module provides a unified base class for all pipeline stages:
1. Common pipeline execution patterns
2. Unified error handling and logging
3. Standardized configuration management
4. Consistent artifact management
5. Reusable sub-pipeline orchestration

Usage:
    from src.training.core.base_pipeline import BasePipeline, PipelineStage

    class DataCollectionPipeline(BasePipeline):
        def __init__(self, config):
            super().__init__(config, PipelineStage.DATA_COLLECTION)

        async def execute_sub_pipeline(self, name: str, config):
            # Stage-specific implementation
            pass
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from src.training.core.errors import (
    TrainingError, PipelineError, ErrorContext, ErrorHandler,
    get_error_handler, with_error_context
)
from src.training.core.config_schema import ConfigSchema, ConfigValidator

T = TypeVar('T')

class PipelineStage(Enum):
    """Pipeline execution stages."""
    DATA_COLLECTION = "data_collection"
    MARKET_ANALYSIS = "market_analysis"
    MODEL_TRAINING = "model_training"
    BACKTESTING = "backtesting"
    VALIDATION = "validation"

class ExecutionMode(Enum):
    """Execution modes for pipelines."""
    FULL = "full"
    LIGHT = "light"
    BLANK = "blank"

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PipelineConfig:
    """Base pipeline configuration."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    data_dir: Union[str, Path] = "./historical_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    single_stage_only: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResult:
    """Base pipeline execution result."""
    pipeline_name: str
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.status == PipelineStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Check if pipeline failed."""
        return self.status == PipelineStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'pipeline_name': self.pipeline_name,
            'stage': self.stage.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'output_files': self.output_files,
            'metadata': self.metadata,
            'error_message': self.error_message,
            'artifacts': self.artifacts
        }

class SubPipelineRegistry:
    """Registry for sub-pipeline functions."""

    def __init__(self):
        self._pipelines: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Register a sub-pipeline function."""
        self._pipelines[name] = func

    def get(self, name: str) -> Optional[Callable]:
        """Get a sub-pipeline function."""
        return self._pipelines.get(name)

    def list_available(self) -> List[str]:
        """List all available sub-pipelines."""
        return list(self._pipelines.keys())

    def has_pipeline(self, name: str) -> bool:
        """Check if a sub-pipeline exists."""
        return name in self._pipelines

class BasePipeline(ABC, Generic[T]):
    """
    Base class for all pipeline stages.

    Provides common functionality:
    - Unified error handling
    - Standardized logging
    - Configuration management
    - Artifact management
    - Sub-pipeline orchestration
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        stage: PipelineStage = PipelineStage.DATA_COLLECTION
    ):
        self.config = config or PipelineConfig()
        self.stage = stage
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.error_handler = get_error_handler()
        self.results: List[PipelineResult] = []
        self.sub_pipeline_registry = SubPipelineRegistry()

        # Initialize artifact manager
        try:
            from src.utils.enhanced_artifact_manager import get_artifact_manager
            self.artifact_manager = get_artifact_manager()
        except ImportError:
            self.artifact_manager = None

        # Register common sub-pipelines
        self._register_common_pipelines()

    @abstractmethod
    def _register_common_pipelines(self):
        """Register stage-specific sub-pipelines."""
        pass

    @abstractmethod
    async def execute_sub_pipeline(
        self,
        sub_pipeline_name: str,
        config: Optional[PipelineConfig] = None
    ) -> PipelineResult:
        """Execute a specific sub-pipeline."""
        pass

    async def execute_multiple_sub_pipelines(
        self,
        sub_pipeline_names: List[str],
        config: Optional[PipelineConfig] = None,
        sequential: bool = False
    ) -> List[PipelineResult]:
        """Execute multiple sub-pipelines."""
        config = config or self.config

        with self._execution_context("execute_multiple_sub_pipelines"):
            if sequential:
                results = []
                for name in sub_pipeline_names:
                    result = await self.execute_sub_pipeline(name, config)
                    results.append(result)
                    if result.failed:
                        self.logger.warning(f"Stopping sequential execution due to failure in {name}")
                        break
                return results
            else:
                # Execute in parallel
                tasks = [
                    self.execute_sub_pipeline(name, config)
                    for name in sub_pipeline_names
                ]
                return await asyncio.gather(*tasks, return_exceptions=True)

    @with_error_context("pipeline_execution")
    async def execute_pipeline_with_next(
        self,
        starting_sub_pipeline: str,
        config: Optional[PipelineConfig] = None
    ) -> PipelineResult:
        """Execute a sub-pipeline and automatically trigger next ones."""
        config = config or self.config

        with self._execution_context(f"execute_pipeline_with_next_{starting_sub_pipeline}"):
            # Execute the starting sub-pipeline
            first_result = await self.execute_sub_pipeline(starting_sub_pipeline, config)

            if not first_result.success:
                self.logger.warning(f"Starting sub-pipeline {starting_sub_pipeline} failed")
                return first_result

            # For now, just return the first result
            # In a more sophisticated implementation, this would chain to next pipelines
            return first_result

    @asynccontextmanager
    async def _execution_context(self, operation: str):
        """Context manager for pipeline execution."""
        start_time = datetime.now()
        context = ErrorContext(
            operation=operation,
            stage=self.stage.value,
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe
        )

        self.logger.info(f"🚀 Starting {operation} for {self.stage.value}")

        try:
            yield
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Create error with context
            if isinstance(e, TrainingError):
                e.context = context
                e.context.execution_time = duration
                raise

            # Convert to PipelineError
            pipeline_error = PipelineError(
                f"Pipeline execution failed: {str(e)}",
                context=context,
                cause=e
            )
            raise pipeline_error
        finally:
            # Log execution time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"⏱️  {operation} completed in {duration:.2f}s")

    def create_pipeline_result(
        self,
        sub_pipeline_name: str,
        status: PipelineStatus,
        start_time: datetime,
        **kwargs
    ) -> PipelineResult:
        """Create a standardized pipeline result."""
        end_time = kwargs.get('end_time')
        if not end_time:
            end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        return PipelineResult(
            pipeline_name=sub_pipeline_name,
            stage=self.stage,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            output_files=kwargs.get('output_files', []),
            metadata=kwargs.get('metadata', {}),
            error_message=kwargs.get('error_message'),
            artifacts=kwargs.get('artifacts', {})
        )

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline executions."""
        total_executions = len(self.results)
        completed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if r.failed)
        total_duration = sum(r.duration_seconds or 0 for r in self.results)

        return {
            'total_executions': total_executions,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_executions if total_executions > 0 else 0,
            'total_duration_seconds': total_duration,
            'stage': self.stage.value,
            'results': [r.to_dict() for r in self.results]
        }

    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return self.sub_pipeline_registry.list_available()

    def validate_config(self, config: PipelineConfig) -> bool:
        """Validate pipeline configuration."""
        try:
            # Basic validation
            if not config.symbol or not isinstance(config.symbol, str):
                raise ConfigurationError("Invalid symbol")

            if config.max_workers < 1:
                raise ConfigurationError("max_workers must be >= 1")

            if config.mode not in ExecutionMode:
                raise ConfigurationError(f"Invalid execution mode: {config.mode}")

            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def log_execution_summary(self):
        """Log a summary of pipeline executions."""
        summary = self.get_execution_summary()

        self.logger.info("=" * 60)
        self.logger.info(f"PIPELINE EXECUTION SUMMARY - {self.stage.value.upper()}")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Executions: {summary['total_executions']}")
        self.logger.info(f"Completed: {summary['completed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        self.logger.info(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        self.logger.info("=" * 60)

    def cleanup(self):
        """Clean up pipeline resources."""
        try:
            if self.artifact_manager:
                self.artifact_manager.cleanup()
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

# Convenience functions
def create_base_config(**kwargs) -> PipelineConfig:
    """Create a base pipeline configuration."""
    return PipelineConfig(**kwargs)

def validate_base_config(config: PipelineConfig) -> bool:
    """Validate base pipeline configuration."""
    return BasePipeline().validate_config(config)

# Export all classes and functions
__all__ = [
    'PipelineStage', 'ExecutionMode', 'PipelineStatus',
    'PipelineConfig', 'PipelineResult', 'SubPipelineRegistry',
    'BasePipeline', 'create_base_config', 'validate_base_config'
]