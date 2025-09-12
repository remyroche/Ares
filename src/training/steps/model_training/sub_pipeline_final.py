"""
Final Model Training Sub-Pipeline

This module provides the final model training sub-pipeline with only the 4 required steps:

1. analyst_models_training - Per-regime individual model training with HPO, saving, and metrics
2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics
3. tactician_models_training - All-regime individual model training with HPO, saving, and metrics
4. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics
"""

import asyncio
import json
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.enhanced_artifact_manager import get_artifact_manager
from src.utils.version_manager import get_version_manager

logger = system_logger.getChild('ModelTrainingSubPipelineFinal')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    data_dir: str = "data/training"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

class ModelTrainingSubPipelineFinal:
    """
    Final Model Training Sub-Pipeline Manager.
    
    Provides the 4 required model training steps:
    1. analyst_models_training - Per-regime individual model training
    2. analyst_ensemble_training - Per-regime ensemble training
    3. tactician_models_training - All-regime individual model training
    4. tactician_ensemble_training - All-regime ensemble training
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the final model training sub-pipeline."""
        self.config = config or SubPipelineConfig()
        self.logger = logger.getChild('ModelTrainingSubPipelineFinal')
        self.results: List[SubPipelineResult] = []
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize sub-pipeline registry with only the 4 required steps
        self.sub_pipelines = {
            'analyst_models_training': self._analyst_models_training_pipeline,
            'analyst_ensemble_training': self._analyst_ensemble_training_pipeline,
            'tactician_models_training': self._tactician_models_training_pipeline,
            'tactician_ensemble_training': self._tactician_ensemble_training_pipeline,
        }
        
        # Define pipeline order
        self.pipeline_order = [
            'analyst_models_training',
            'analyst_ensemble_training',
            'tactician_models_training',
            'tactician_ensemble_training'
        ]
    
    async def execute_sub_pipeline(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Optional configuration override
            
        Returns:
            SubPipelineResult with execution details
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting model training sub-pipeline: {sub_pipeline_name} (mode: {config.mode.value})")
        
        start_time = datetime.now()
        result = SubPipelineResult(
            sub_pipeline_name=sub_pipeline_name,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            if sub_pipeline_name not in self.sub_pipelines:
                raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
            
            # Execute the sub-pipeline
            pipeline_func = self.sub_pipelines[sub_pipeline_name]
            artifacts = await pipeline_func(config)
            
            # Update result
            end_time = datetime.now()
            result.status = SubPipelineStatus.COMPLETED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.artifacts = artifacts
            result.metadata = {
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe
            }
            
            self.logger.info(f"✅ Model training sub-pipeline {sub_pipeline_name} completed in {result.duration_seconds:.2f}s")
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"❌ Model training sub-pipeline {sub_pipeline_name} failed: {e}")
        
        self.results.append(result)
        return result
    
    async def execute_sub_pipeline_with_next(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a sub-pipeline and automatically trigger the next ones in sequence.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to start from
            config: Optional configuration override
            
        Returns:
            SubPipelineResult of the starting sub-pipeline
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting model training pipeline chain from: {sub_pipeline_name}")
        
        # Find the starting index
        try:
            start_index = self.pipeline_order.index(sub_pipeline_name)
        except ValueError:
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
        
        # Execute sub-pipelines in sequence starting from the specified one
        for i in range(start_index, len(self.pipeline_order)):
            pipeline_name = self.pipeline_order[i]
            self.logger.info(f"🔄 Executing pipeline step {i+1}/{len(self.pipeline_order)}: {pipeline_name}")
            
            result = await self.execute_sub_pipeline(pipeline_name, config)
            
            # Check if this step failed
            if result.status == SubPipelineStatus.FAILED:
                self.logger.error(f"❌ Pipeline chain stopped due to failure in {pipeline_name}")
                break
        
        # Return the result of the starting sub-pipeline
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result
        
        # If not found, return the last result
        return self.results[-1] if self.results else None
    
    # Sub-pipeline implementations
    async def _analyst_models_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst Models Training sub-pipeline - Per-regime individual model training."""
        self.logger.info("👨‍💼 Executing analyst models training pipeline (per-regime individual models)")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual analyst models training")
            return {'analyst_models': [], 'analyst_training_metadata': {}}
        
        # Import and execute analyst models training
        try:
            from .analyst_models_training import AnalystModelsTrainingStep
            trainer = AnalystModelsTrainingStep()
            result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Analyst models training not available, using mock training")
            return {'analyst_models': [], 'analyst_training_metadata': {}}
    
    async def _analyst_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst Ensemble Training sub-pipeline - Per-regime ensemble training."""
        self.logger.info("🎭 Executing analyst ensemble training pipeline (per-regime ensemble models)")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual analyst ensemble training")
            return {'analyst_ensembles': [], 'analyst_ensemble_metadata': {}}
        
        # Import and execute analyst ensemble training
        try:
            from .analyst_ensemble_training import AnalystEnsembleTrainingStep
            trainer = AnalystEnsembleTrainingStep()
            result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Analyst ensemble training not available, using mock training")
            return {'analyst_ensembles': [], 'analyst_ensemble_metadata': {}}
    
    async def _tactician_models_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician Models Training sub-pipeline - All-regime individual model training."""
        self.logger.info("🎯 Executing tactician models training pipeline (all-regime individual models)")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual tactician models training")
            return {'tactician_models': [], 'tactician_training_metadata': {}}
        
        # Import and execute tactician models training
        try:
            from .tactician_models_training import TacticianModelsTrainingStep
            trainer = TacticianModelsTrainingStep()
            result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Tactician models training not available, using mock training")
            return {'tactician_models': [], 'tactician_training_metadata': {}}
    
    async def _tactician_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician Ensemble Training sub-pipeline - All-regime ensemble training."""
        self.logger.info("🎪 Executing tactician ensemble training pipeline (all-regime ensemble models)")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual tactician ensemble training")
            return {'tactician_ensembles': [], 'tactician_ensemble_metadata': {}}
        
        # Import and execute tactician ensemble training
        try:
            from .tactician_ensemble_training import TacticianEnsembleTrainingStep
            trainer = TacticianEnsembleTrainingStep()
            result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={}
            )
            return result
        except ImportError:
            self.logger.warning("⚠️ Tactician ensemble training not available, using mock training")
            return {'tactician_ensembles': [], 'tactician_ensemble_metadata': {}}
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.sub_pipelines.keys())
    
    def get_pipeline_order(self) -> List[str]:
        """Get the pipeline execution order."""
        return self.pipeline_order.copy()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all sub-pipeline executions."""
        total_executions = len(self.results)
        completed = sum(1 for r in self.results if r.status == SubPipelineStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == SubPipelineStatus.FAILED)
        total_duration = sum(r.duration_seconds or 0 for r in self.results)
        
        return {
            'total_executions': total_executions,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_executions if total_executions > 0 else 0,
            'total_duration_seconds': total_duration,
            'results': self.results
        }

# Convenience functions
def get_model_training_sub_pipeline_final(config: Optional[SubPipelineConfig] = None) -> ModelTrainingSubPipelineFinal:
    """Get a configured final model training sub-pipeline."""
    return ModelTrainingSubPipelineFinal(config)

async def execute_model_training_sub_pipeline_final(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute a final model training sub-pipeline."""
    pipeline = get_model_training_sub_pipeline_final(config)
    return await pipeline.execute_sub_pipeline(sub_pipeline_name, config)