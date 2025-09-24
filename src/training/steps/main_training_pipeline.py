"""
Refactored Main Training Pipeline

This module provides a simplified training pipeline orchestrator that coordinates
stage-specific pipelines using the new base pipeline architecture.

Key Features:
- Clean separation of concerns
- Unified error handling
- Configuration validation
- Performance monitoring
- Simplified execution model

Architecture:
    MainTrainingPipeline (Orchestrator)
    ├── DataCollectionPipeline (Stage-specific)
    ├── MarketAnalysisPipeline (Stage-specific)
    ├── ModelTrainingPipeline (Stage-specific)
    └── BacktestingPipeline (Stage-specific)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

# New core imports
from src.training.core.errors import (
    TrainingError, PipelineError, ErrorContext, ErrorHandler,
    get_error_handler, with_error_context, pipeline_execution_error
)
from src.training.core.config_schema import (
    validate_pipeline_config, ConfigSchema, ConfigValidator,
    PIPELINE_CONFIG_SCHEMA
)
from src.training.core.base_pipeline import (
    BasePipeline, PipelineStage, ExecutionMode, PipelineStatus,
    PipelineConfig as BasePipelineConfig, PipelineResult
)
from src.training.utils.dataframes import get_dataframe_manager, log_memory_usage

# Lazy imports for sub-pipelines
def _get_system_logger():
    try:
        from src.utils.logger import system_logger
        return system_logger
    except ImportError:
        return logging.getLogger('MainTrainingPipeline')

logger = _get_system_logger().getChild('MainTrainingPipeline')

@dataclass
class MainPipelineConfig:
    """Simplified main pipeline configuration."""
    # Basic configuration
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    data_dir: str = "historical_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False

    # Execution control
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True

    # Stage control - simplified to just enable/disable stages
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: [
        PipelineStage.DATA_COLLECTION,
        PipelineStage.MARKET_ANALYSIS,
        PipelineStage.MODEL_TRAINING,
        PipelineStage.BACKTESTING
    ])

    # Custom parameters for each stage
    stage_params: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class MainPipelineResult:
    """Simplified main pipeline result."""
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Stage results
    stage_results: Dict[PipelineStage, PipelineResult] = field(default_factory=dict)

    # Overall metrics
    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    success_rate: float = 0.0

    # Error information
    error_message: Optional[str] = None
    failed_stages_list: List[PipelineStage] = field(default_factory=list)

    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class MainTrainingPipeline:
    """
    Simplified Main Training Pipeline Manager.

    Orchestrates the execution of stage-specific pipelines using the new
    base pipeline architecture with unified error handling and monitoring.
    """

    def __init__(self, config: Optional[MainPipelineConfig] = None):
        """Initialize the main training pipeline."""
        self.config = config or MainPipelineConfig()
        self.logger = logger.getChild('MainTrainingPipeline')
        self.error_handler = get_error_handler()

        # Initialize stage-specific pipelines
        self.stage_pipelines: Dict[PipelineStage, BasePipeline] = {}
        self._initialize_stage_pipelines()

        # Pipeline state
        self.current_stage: Optional[PipelineStage] = None
        self.pipeline_results: List[MainPipelineResult] = []

    def _initialize_stage_pipelines(self):
        """Initialize stage-specific pipelines."""
        # Data Collection Pipeline
        if self._is_stage_enabled(PipelineStage.DATA_COLLECTION):
            try:
                from .data_collection.sub_pipeline import DataCollectionSubPipeline
                self.stage_pipelines[PipelineStage.DATA_COLLECTION] = DataCollectionSubPipeline()
            except ImportError as e:
                self.logger.warning(f"Data collection pipeline not available: {e}")

        # Market Analysis Pipeline
        if self._is_stage_enabled(PipelineStage.MARKET_ANALYSIS):
            try:
                from .market_analysis.sub_pipeline import MarketAnalysisSubPipeline
                self.stage_pipelines[PipelineStage.MARKET_ANALYSIS] = MarketAnalysisSubPipeline()
            except ImportError as e:
                self.logger.warning(f"Market analysis pipeline not available: {e}")

        # Model Training Pipeline
        if self._is_stage_enabled(PipelineStage.MODEL_TRAINING):
            try:
                from .model_training.sub_pipeline import ModelTrainingSubPipeline
                self.stage_pipelines[PipelineStage.MODEL_TRAINING] = ModelTrainingSubPipeline()
            except ImportError as e:
                self.logger.warning(f"Model training pipeline not available: {e}")

        # Backtesting Pipeline
        if self._is_stage_enabled(PipelineStage.BACKTESTING):
            try:
                from .backtesting.sub_pipeline import BacktestingSubPipeline
                self.stage_pipelines[PipelineStage.BACKTESTING] = BacktestingSubPipeline()
            except ImportError as e:
                self.logger.warning(f"Backtesting pipeline not available: {e}")

    def _is_stage_enabled(self, stage: PipelineStage) -> bool:
        """Check if a stage is enabled."""
        return stage in self.config.enabled_stages
    
    @with_error_context("execute_pipeline")
    async def execute_pipeline(
        self,
        config: Optional[MainPipelineConfig] = None
    ) -> MainPipelineResult:
        """
        Execute the complete training pipeline using stage-specific pipelines.

        Args:
            config: Optional configuration override

        Returns:
            MainPipelineResult with execution details
        """
        config = config or self.config

        # Validate configuration
        try:
            validated_config = validate_pipeline_config(config.__dict__)
            config = MainPipelineConfig(**validated_config)
        except Exception as e:
            raise PipelineError(
                f"Configuration validation failed: {e}",
                stage="configuration",
                context=ErrorContext(operation="configuration_validation")
            )

        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"🚀 Starting main training pipeline: {pipeline_id} (mode: {config.mode.value})")
        log_memory_usage("pipeline_start")

        start_time = datetime.now()
        result = MainPipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Execute each enabled stage using stage-specific pipelines
            for stage in config.enabled_stages:
                self.logger.info(f"📋 Executing stage: {stage.value}")
                self.current_stage = stage

                # Get stage-specific pipeline
                stage_pipeline = self.stage_pipelines.get(stage)
                if not stage_pipeline:
                    self.logger.warning(f"⚠️ Stage pipeline not available: {stage.value}")
                    result.failed_stages_list.append(stage)
                    continue

                # Execute stage pipeline
                stage_config = self._create_stage_config(stage, config)
                stage_result = await stage_pipeline.execute_pipeline(stage_config)

                result.stage_results[stage] = stage_result
                result.total_stages += 1

                if stage_result.success:
                    result.completed_stages += 1
                else:
                    result.failed_stages += 1
                    result.failed_stages_list.append(stage)

            # Calculate overall metrics
            self._calculate_pipeline_metrics(result)

            # Update result status
            end_time = datetime.now()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()

            if result.failed_stages == 0:
                result.status = PipelineStatus.COMPLETED
                self.logger.info(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
            else:
                result.status = PipelineStatus.FAILED
                result.error_message = f"Pipeline failed with {result.failed_stages} failed stages"
                self.logger.error(f"❌ Main training pipeline {pipeline_id} failed: {result.error_message}")

        except Exception as e:
            end_time = datetime.now()
            result.status = PipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()

            # Use enhanced error handling
            if isinstance(e, TrainingError):
                result.error_message = str(e)
            else:
                result.error_message = f"Unexpected error: {str(e)}"

            self.logger.error(f"❌ Main training pipeline {pipeline_id} failed with exception: {e}")

        log_memory_usage("pipeline_end")
        self.pipeline_results.append(result)
        return result
    
    async def execute_sub_pipeline_with_chain(
        self,
        stage: PipelineStage,
        starting_sub_pipeline: str,
        config: Optional[MainPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline and let it automatically trigger the next ones in sequence.
        
        Args:
            stage: Pipeline stage containing the sub-pipeline
            starting_sub_pipeline: Sub-pipeline to start from
            config: Optional configuration override
            
        Returns:
            SubPipelineResult of the starting sub-pipeline (which will have triggered the chain)
        """
        config = config or self.config
        self.logger.info(f"🚀 Starting sub-pipeline chain: {stage.value} -> {starting_sub_pipeline}")
        
        # Create stage-specific configuration
        stage_config = self._create_stage_config(stage, config)
        
        # Execute the sub-pipeline with automatic next triggering
        if stage == PipelineStage.MARKET_ANALYSIS:
            if not MARKET_ANALYSIS_AVAILABLE:
                self.logger.warning("⚠️ Market analysis sub-pipeline not available")
                return None
            return await self.market_analysis_pipeline.execute_sub_pipeline_with_next(starting_sub_pipeline, stage_config)
        elif stage == PipelineStage.MODEL_TRAINING:
            if not MODEL_TRAINING_AVAILABLE:
                self.logger.warning("⚠️ Model training sub-pipeline not available")
                return None
            return await self.model_training_pipeline.execute_sub_pipeline_with_next(starting_sub_pipeline, stage_config)
        elif stage == PipelineStage.BACKTESTING:
            if not BACKTESTING_AVAILABLE:
                self.logger.warning("⚠️ Backtesting sub-pipeline not available")
                return None
            return await self.backtesting_pipeline.execute_sub_pipeline_with_next(starting_sub_pipeline, stage_config)
        else:
            self.logger.warning(f"⚠️ Auto-chaining not implemented for stage: {stage.value}")
            return None
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        config: MainPipelineConfig
    ) -> PipelineResult:
        """
        Execute a specific pipeline stage using stage-specific pipeline.

        Args:
            stage: Pipeline stage to execute
            config: Pipeline configuration

        Returns:
            PipelineResult for the stage
        """
        self.logger.info(f"🎯 Executing stage: {stage.value}")

        # Get stage-specific pipeline
        stage_pipeline = self.stage_pipelines.get(stage)
        if not stage_pipeline:
            error_msg = f"Stage pipeline not available: {stage.value}"
            self.logger.error(error_msg)
            raise PipelineError(
                error_msg,
                stage=stage.value,
                context=ErrorContext(operation="stage_execution")
            )

        # Create stage-specific configuration
        stage_config = self._create_stage_config(stage, config)

        # Execute stage pipeline
        try:
            # Most stages expect a different config type, so we need to handle this
            if hasattr(stage_pipeline, 'config'):
                # Set the config on the stage pipeline
                stage_pipeline.config = stage_config

            # Execute the stage
            result = await stage_pipeline.execute_pipeline(stage_config)
            return result

        except Exception as e:
            error_msg = f"Stage execution failed for {stage.value}: {e}"
            self.logger.error(error_msg)
            raise PipelineError(
                error_msg,
                stage=stage.value,
                context=ErrorContext(operation="stage_execution")
            )
    
    def _create_stage_config(self, stage: PipelineStage, config: MainPipelineConfig) -> BasePipelineConfig:
        """Create stage-specific configuration."""
        # Create base configuration with stage-specific parameters
        stage_config = BasePipelineConfig(
            mode=config.mode,
            symbol=config.symbol,
            exchange=config.exchange,
            timeframe=config.timeframe,
            data_dir=config.data_dir,
            start_date=config.start_date,
            end_date=config.end_date,
            force_rerun=config.force_rerun,
            parallel_processing=config.parallel_processing,
            max_workers=config.max_workers,
            validation_enabled=config.validation_enabled,
            monitoring_enabled=config.monitoring_enabled,
            single_stage_only=config.single_stage_only,
            custom_params=config.stage_params.get(stage, {})
        )

        return stage_config
    
    
    
    
    
    def _calculate_pipeline_metrics(self, result: MainPipelineResult) -> None:
        """Calculate overall pipeline metrics."""
        total_stages = len(result.stage_results)
        completed_stages = sum(1 for stage_result in result.stage_results.values() if stage_result.success)
        failed_stages = sum(1 for stage_result in result.stage_results.values() if not stage_result.success)

        result.total_stages = total_stages
        result.completed_stages = completed_stages
        result.failed_stages = failed_stages
        result.success_rate = completed_stages / total_stages if total_stages > 0 else 0

        self.logger.info(f"📈 Final metrics: Total={total_stages}, Completed={completed_stages}, Failed={failed_stages}, Rate={result.success_rate:.1%}")

        # Calculate performance metrics
        result.performance_metrics = {
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'failed_stages': failed_stages,
            'success_rate': result.success_rate,
            'failed_stages_list': result.failed_stages_list
        }
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """Get status of a specific pipeline execution."""
        for result in self.pipeline_results:
            if result.pipeline_id == pipeline_id:
                return result.status
        return None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline executions."""
        total_executions = len(self.pipeline_results)
        completed = sum(1 for r in self.pipeline_results if r.status == PipelineStatus.COMPLETED)
        failed = sum(1 for r in self.pipeline_results if r.status == PipelineStatus.FAILED)
        total_duration = sum(r.duration_seconds or 0 for r in self.pipeline_results)

        return {
            'total_executions': total_executions,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_executions if total_executions > 0 else 0,
            'total_duration_seconds': total_duration,
            'results': self.pipeline_results
        }
    
    def get_available_sub_pipelines(self, stage: PipelineStage) -> List[str]:
        """Get available sub-pipelines for a specific stage."""
        stage_pipeline = self.stage_pipelines.get(stage)
        if stage_pipeline:
            return stage_pipeline.get_available_sub_pipelines()
        return []

    def get_stage_execution_summary(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get execution summary for a specific stage."""
        stage_pipeline = self.stage_pipelines.get(stage)
        if stage_pipeline:
            return stage_pipeline.get_execution_summary()
        return {}

# Convenience functions
def get_main_training_pipeline(config: Optional[MainPipelineConfig] = None) -> MainTrainingPipeline:
    """Get a configured main training pipeline."""
    return MainTrainingPipeline(config)

async def execute_main_training_pipeline(
    config: Optional[MainPipelineConfig] = None
) -> MainPipelineResult:
    """Convenience function to execute the main training pipeline."""
    pipeline = get_main_training_pipeline(config)
    return await pipeline.execute_pipeline(config)

# Predefined pipeline configurations - simplified
def get_full_pipeline_config(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a full pipeline configuration."""
    return MainPipelineConfig(
        mode=ExecutionMode.FULL,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=False,
        parallel_processing=True,
        max_workers=4,
        validation_enabled=True,
        monitoring_enabled=True,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ]
    )

def get_light_pipeline_config(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a light pipeline configuration."""
    return MainPipelineConfig(
        mode=ExecutionMode.LIGHT,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=False,
        parallel_processing=True,
        max_workers=2,  # Fewer workers for light mode
        validation_enabled=True,
        monitoring_enabled=False,  # Less monitoring for light mode
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.BACKTESTING  # Skip model training for light mode
        ]
    )

def get_blank_pipeline_config(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a blank pipeline configuration for testing."""
    return MainPipelineConfig(
        mode=ExecutionMode.BLANK,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=False,
        parallel_processing=False,  # No parallel processing for blank mode
        max_workers=1,
        validation_enabled=False,   # Minimal validation for blank mode
        monitoring_enabled=False,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION  # Only data collection for blank mode
        ]
    )