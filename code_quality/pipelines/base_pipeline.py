#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Base Pipeline Architecture for Code Quality Analysis

This provides a proper pipeline framework with real value:
- Multi-stage processing with checkpoints
- Parallel execution capabilities
- Comprehensive error handling and recovery
- Progress tracking and reporting
- Configuration management
- Result aggregation and visualization
"""

import asyncio
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZATION = "initialization"
    PREPARATION = "preparation"
    ANALYSIS = "analysis"
    PROCESSING = "processing"
    AGGREGATION = "aggregation"
    REPORTING = "reporting"
    CLEANUP = "cleanup"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    project_root: Union[str, Path]
    output_dir: Optional[Union[str, Path]] = None
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_per_stage: int = 300
    retry_attempts: int = 3
    log_level: str = "INFO"
    dry_run: bool = False
    verbose: bool = False
    cache_enabled: bool = True
    cache_dir: Optional[Union[str, Path]] = None
    checkpoint_enabled: bool = True
    checkpoint_dir: Optional[Union[str, Path]] = None
    continue_on_failure: bool = False
    
    def __post_init__(self):
        """Post-initialization setup."""
        self.project_root = Path(self.project_root)
        if self.output_dir is None:
            self.output_dir = self.project_root / "code_quality" / "reports"
        else:
            self.output_dir = Path(self.output_dir)
        
        if self.cache_dir is None:
            self.cache_dir = self.project_root / "code_quality" / "cache"
        else:
            self.cache_dir = Path(self.cache_dir)
            
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "code_quality" / "checkpoints"
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, data: Dict[str, Any] = None, errors: List[str] = None, warnings: List[str] = None):
        """Mark stage as completed."""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        else:
            self.duration_seconds = 0.0
        self.status = PipelineStatus.COMPLETED
        if data:
            self.data.update(data)
        if errors:
            self.errors.extend(errors)
        if warnings:
            self.warnings.extend(warnings)
    
    def fail(self, errors: List[str]):
        """Mark stage as failed."""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        else:
            self.duration_seconds = 0.0
        self.status = PipelineStatus.FAILED
        self.errors.extend(errors)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    pipeline_name: str
    config: PipelineConfig
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    stages: List[StageResult] = field(default_factory=list)
    aggregated_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark pipeline as completed."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = PipelineStatus.COMPLETED
        
        # Aggregate all stage data
        for stage in self.stages:
            self.aggregated_data.update(stage.data)
            self.errors.extend(stage.errors)
            self.warnings.extend(stage.warnings)
            self.metrics.update(stage.metrics)
    
    def fail(self, errors: List[str]):
        """Mark pipeline as failed."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = PipelineStatus.FAILED
        self.errors.extend(errors)


class BasePipeline(ABC):
    """Base class for all code quality pipelines."""
    
    def __init__(self, config: PipelineConfig, pipeline_name: str = None):
        """Initialize the pipeline."""
        self.config = config
        self.pipeline_name = pipeline_name or self.__class__.__name__
        self.logger = self._setup_logger()
        self.result = PipelineResult(
            pipeline_name=self.pipeline_name,
            config=config,
            status=PipelineStatus.PENDING,
            start_time=datetime.now()
        )
        self._executor = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup pipeline logger."""
        logger = logging.getLogger(f"code_quality.{self.pipeline_name}")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.config.log_level.upper()))
        return logger
    
    @abstractmethod
    def get_stages(self) -> List[PipelineStage]:
        """Get the list of stages for this pipeline."""
        pass
    
    @abstractmethod
    async def execute_stage(self, stage: PipelineStage, context: Dict[str, Any]) -> StageResult:
        """Execute a specific pipeline stage."""
        pass
    
    async def run(self, context: Dict[str, Any] = None) -> PipelineResult:
        """Run the complete pipeline."""
        if context is None:
            context = {}
        
        self.logger.info(f"Starting pipeline: {self.pipeline_name}")
        self.result.status = PipelineStatus.RUNNING
        
        try:
            # Initialize executor if parallel execution is enabled
            if self.config.parallel_execution:
                self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            # Execute stages sequentially (but stages can use parallel execution internally)
            for stage in self.get_stages():
                self.logger.info(f"Executing stage: {stage.value}")
                
                stage_result = StageResult(
                    stage=stage,
                    status=PipelineStatus.RUNNING,
                    start_time=datetime.now()
                )
                self.result.stages.append(stage_result)
                
                try:
                    # Execute stage with timeout
                    stage_result = await asyncio.wait_for(
                        self.execute_stage(stage, context),
                        timeout=self.config.timeout_per_stage
                    )
                    
                    # Update stage result
                    self.result.stages[-1] = stage_result
                    
                    if stage_result.status == PipelineStatus.FAILED:
                        self.logger.error(f"Stage {stage.value} failed: {stage_result.errors}")
                        if not self.config.continue_on_failure:
                            self.result.fail([f"Stage {stage.value} failed"])
                            return self.result
                    else:
                        self.logger.info(f"Stage {stage.value} completed in {stage_result.duration_seconds:.2f}s")
                        
                except asyncio.TimeoutError:
                    error_msg = f"Stage {stage.value} timed out after {self.config.timeout_per_stage}s"
                    self.logger.error(error_msg)
                    stage_result.fail([error_msg])
                    self.result.stages[-1] = stage_result
                    if not self.config.continue_on_failure:
                        self.result.fail([error_msg])
                        return self.result
                        
                except Exception as e:
                    error_msg = f"Stage {stage.value} failed with exception: {e}"
                    self.logger.error(error_msg, exc_info=True)
                    stage_result.fail([error_msg])
                    self.result.stages[-1] = stage_result
                    if not self.config.continue_on_failure:
                        self.result.fail([error_msg])
                        return self.result
            
            # Pipeline completed successfully
            self.result.complete()
            self.logger.info(f"Pipeline {self.pipeline_name} completed in {self.result.duration_seconds:.2f}s")
            
        except Exception as e:
            error_msg = f"Pipeline {self.pipeline_name} failed with exception: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.result.fail([error_msg])
            
        finally:
            # Cleanup
            if self._executor:
                self._executor.shutdown(wait=True)
        
        return self.result
    
    def save_result(self, filename: str = None) -> Path:
        """Save pipeline result to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.pipeline_name}_{timestamp}.json"
        
        result_path = self.config.output_dir / filename
        
        # Convert result to serializable format
        result_data = {
            "pipeline_name": self.result.pipeline_name,
            "status": self.result.status.value,
            "start_time": self.result.start_time.isoformat(),
            "end_time": self.result.end_time.isoformat() if self.result.end_time else None,
            "duration_seconds": self.result.duration_seconds,
            "stages": [
                {
                    "stage": stage.stage.value,
                    "status": stage.status.value,
                    "start_time": stage.start_time.isoformat(),
                    "end_time": stage.end_time.isoformat() if stage.end_time else None,
                    "duration_seconds": stage.duration_seconds,
                    "data": stage.data,
                    "errors": stage.errors,
                    "warnings": stage.warnings,
                    "metrics": stage.metrics
                }
                for stage in self.result.stages
            ],
            "aggregated_data": self.result.aggregated_data,
            "errors": self.result.errors,
            "warnings": self.result.warnings,
            "metrics": self.result.metrics
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline result saved to: {result_path}")
        return result_path
    
    def print_summary(self):
        """Print pipeline execution summary."""
        tprint(f"\n{'='*80}")
        tprint(f"PIPELINE EXECUTION SUMMARY: {self.pipeline_name}")
        tprint(f"{'='*80}")
        tprint(f"Status: {self.result.status.value}")
        tprint(f"Duration: {self.result.duration_seconds:.2f}s" if self.result.duration_seconds else "N/A")
        tprint(f"Stages: {len(self.result.stages)}")
        
        if self.result.errors:
            tprint(f"Errors: {len(self.result.errors)}")
            for error in self.result.errors:
                tprint(f"  - {error}")
        
        if self.result.warnings:
            tprint(f"Warnings: {len(self.result.warnings)}")
            for warning in self.result.warnings:
                tprint(f"  - {warning}")
        
        tprint(f"\nStage Details:")
        for stage in self.result.stages:
            tprint(f"  {stage.stage.value}: {stage.status.value} ({stage.duration_seconds:.2f}s)")
            if stage.errors:
                for error in stage.errors:
                    tprint(f"    - {error}")
        
        tprint(f"{'='*80}")