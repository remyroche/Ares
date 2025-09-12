"""
Updated Model Training Sub-Pipeline (HMM Training Moved to MARKET_ANALYSIS)

This module provides granular sub-pipeline functionality for model training,
with HMM training moved to MARKET_ANALYSIS stage.

Sub-pipelines:
1. General Model Training - Train general ML models
2. Analyst Model Training - Train analyst-specific models
3. Tactician Model Training - Train tactician-specific models
4. Ensemble Training - Ensemble model training
5. Multi-timeframe Training - Multi-timeframe model training
6. Model Validation - Model validation and testing
7. Model Persistence - Save and load models
8. Model Evaluation - Comprehensive model evaluation

Note: HMM Training has been moved to MARKET_ANALYSIS stage.
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

logger = system_logger.getChild('ModelTrainingSubPipelineUpdated')

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

class ModelTrainingSubPipelineUpdated:
    """
    Updated Model Training Sub-Pipeline Manager (HMM Training Moved to MARKET_ANALYSIS).
    
    Provides granular control over model training processes with different
    execution modes and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the updated model training sub-pipeline."""
        self.config = config or SubPipelineConfig()
        self.logger = logger.getChild('ModelTrainingSubPipelineUpdated')
        self.results: List[SubPipelineResult] = []
        
        # Initialize artifact and version managers
        self.artifact_manager = get_artifact_manager()
        self.version_manager = get_version_manager()
        
        # Initialize sub-pipeline registry (HMM training removed)
        self.sub_pipelines = {
            'general_model_training': self._general_model_training_pipeline,
            'analyst_model_training': self._analyst_model_training_pipeline,
            'tactician_model_training': self._tactician_model_training_pipeline,
            'ensemble_training': self._ensemble_training_pipeline,
            'multi_timeframe_training': self._multi_timeframe_training_pipeline,
            'model_validation': self._model_validation_pipeline,
            'model_persistence': self._model_persistence_pipeline,
            'model_evaluation': self._model_evaluation_pipeline,
        }
        
        # Define pipeline order (HMM training removed)
        self.pipeline_order = [
            'general_model_training',
            'analyst_model_training',
            'tactician_model_training',
            'ensemble_training',
            'multi_timeframe_training',
            'model_validation',
            'model_persistence',
            'model_evaluation'
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
    async def _general_model_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """General Model Training sub-pipeline."""
        self.logger.info("🤖 Executing general model training pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual general model training")
            return {'general_models': [], 'general_training_metadata': {}}
        
        # Import and execute general model training
        try:
            from .general_model_training import GeneralModelTrainingStep
            trainer = GeneralModelTrainingStep()
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
            self.logger.warning("⚠️ General model training not available, using mock training")
            return {'general_models': [], 'general_training_metadata': {}}
    
    async def _analyst_model_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst Model Training sub-pipeline."""
        self.logger.info("👨‍💼 Executing analyst model training pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual analyst model training")
            return {'analyst_models': [], 'analyst_training_metadata': {}}
        
        # Import and execute analyst model training
        try:
            from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored as AnalystModelTrainingStep
            trainer = AnalystModelTrainingStep()
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
            self.logger.warning("⚠️ Analyst model training not available, using mock training")
            return {'analyst_models': [], 'analyst_training_metadata': {}}
    
    async def _tactician_model_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician Model Training sub-pipeline."""
        self.logger.info("🎯 Executing tactician model training pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual tactician model training")
            return {'tactician_models': [], 'tactician_training_metadata': {}}
        
        # Import and execute tactician model training
        try:
            from .tactician_models_training import TacticianModelTrainingStep
            trainer = TacticianModelTrainingStep()
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
            self.logger.warning("⚠️ Tactician model training not available, using mock training")
            return {'tactician_models': [], 'tactician_training_metadata': {}}
    
    async def _ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Ensemble Training sub-pipeline."""
        self.logger.info("🎭 Executing ensemble training pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual ensemble training")
            return {'ensemble_models': [], 'ensemble_training_metadata': {}}
        
        # Import and execute ensemble training
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
            self.logger.warning("⚠️ Ensemble training not available, using mock training")
            return {'ensemble_models': [], 'ensemble_training_metadata': {}}
    
    async def _multi_timeframe_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Multi-timeframe Training sub-pipeline."""
        self.logger.info("⏰ Executing multi-timeframe training pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual multi-timeframe training")
            return {'multi_tf_models': [], 'multi_tf_training_metadata': {}}
        
        # Import and execute multi-timeframe training
        try:
            from .multi_timeframe_hmm_ensemble import MultiTimeframeHMMEnsembleStep
            trainer = MultiTimeframeHMMEnsembleStep()
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
            self.logger.warning("⚠️ Multi-timeframe training not available, using mock training")
            return {'multi_tf_models': [], 'multi_tf_training_metadata': {}}
    
    async def _model_validation_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Model Validation sub-pipeline."""
        self.logger.info("✅ Executing model validation pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual model validation")
            return {'validation_results': [], 'validation_metadata': {}}
        
        # Import and execute model validation
        try:
            from .model_validation import ModelValidationStep
            validator = ModelValidationStep()
            result = await validator.execute(
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
            self.logger.warning("⚠️ Model validation not available, using mock validation")
            return {'validation_results': [], 'validation_metadata': {}}
    
    async def _model_persistence_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Model Persistence sub-pipeline."""
        self.logger.info("💾 Executing model persistence pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual model persistence")
            return {'persisted_models': [], 'persistence_metadata': {}}
        
        # Import and execute model persistence
        try:
            from .model_persistence import ModelPersistenceStep
            persister = ModelPersistenceStep()
            result = await persister.execute(
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
            self.logger.warning("⚠️ Model persistence not available, using mock persistence")
            return {'persisted_models': [], 'persistence_metadata': {}}
    
    async def _model_evaluation_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Model Evaluation sub-pipeline."""
        self.logger.info("📊 Executing model evaluation pipeline")
        
        if config.mode == ExecutionMode.BLANK:
            self.logger.info("🔄 Blank mode: Skipping actual model evaluation")
            return {'evaluation_results': [], 'evaluation_metadata': {}}
        
        # Import and execute model evaluation
        try:
            from .model_evaluation import ModelEvaluationStep
            evaluator = ModelEvaluationStep()
            result = await evaluator.execute(
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
            self.logger.warning("⚠️ Model evaluation not available, using mock evaluation")
            return {'evaluation_results': [], 'evaluation_metadata': {}}
    
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
def get_model_training_sub_pipeline_updated(config: Optional[SubPipelineConfig] = None) -> ModelTrainingSubPipelineUpdated:
    """Get a configured updated model training sub-pipeline."""
    return ModelTrainingSubPipelineUpdated(config)

async def execute_model_training_sub_pipeline_updated(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute an updated model training sub-pipeline."""
    pipeline = get_model_training_sub_pipeline_updated(config)
    return await pipeline.execute_sub_pipeline(sub_pipeline_name, config)