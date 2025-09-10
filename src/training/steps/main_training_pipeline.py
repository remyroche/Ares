"""
Main Training Pipeline

This module provides a comprehensive training pipeline that orchestrates all
sub-pipelines across different modules (data_collection, market_analysis,
model_training, backtesting) with granular control and monitoring.

Key Features:
- Sequential execution of all sub-pipelines
- Granular control at sub-pipeline level
- Multiple execution modes (full, light, blank)
- Comprehensive monitoring and reporting
- Error handling and recovery
- Performance tracking
- Artifact management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time

# Import sub-pipelines
from .data_collection.sub_pipeline import (
    DataCollectionSubPipeline, SubPipelineConfig as DataCollectionConfig,
    SubPipelineResult as DataCollectionResult, ExecutionMode, SubPipelineStatus
)
from .market_analysis.sub_pipeline import (
    MarketAnalysisSubPipeline, SubPipelineConfig as MarketAnalysisConfig,
    SubPipelineResult as MarketAnalysisResult
)
from .model_training.sub_pipeline import (
    ModelTrainingSubPipeline, SubPipelineConfig as ModelTrainingConfig,
    SubPipelineResult as ModelTrainingResult
)
from .backtesting.sub_pipeline import (
    BacktestingSubPipeline, SubPipelineConfig as BacktestingConfig,
    SubPipelineResult as BacktestingResult
)

logger = system_logger.getChild('MainTrainingPipeline')

class PipelineStage(Enum):
    """Pipeline execution stages."""
    DATA_COLLECTION = "data_collection"
    MARKET_ANALYSIS = "market_analysis"
    MODEL_TRAINING = "model_training"
    BACKTESTING = "backtesting"

@dataclass
class MainPipelineConfig:
    """Configuration for the main training pipeline."""
    # General configuration
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    data_dir: str = "data/training"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    
    # Execution control
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    
    # Stage control
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: [
        PipelineStage.DATA_COLLECTION,
        PipelineStage.MARKET_ANALYSIS,
        PipelineStage.MODEL_TRAINING,
        PipelineStage.BACKTESTING
    ])
    
    # Sub-pipeline control
    enabled_sub_pipelines: Dict[PipelineStage, List[str]] = field(default_factory=lambda: {
        PipelineStage.DATA_COLLECTION: [
            'data_download', 'data_conversion', 'data_validation', 'data_preparation',
            'feature_engineering', 'data_quality_check', 'data_storage'
        ],
        PipelineStage.MARKET_ANALYSIS: [
            'sr_detection', 'sr_clustering', 'hmm_regime_discovery', 'regime_data_splitting',
            'triple_barrier_labeling', 'feature_lookback_optimization'
        ],
        PipelineStage.MODEL_TRAINING: [
            'general_model_training', 'analyst_model_training', 'tactician_model_training',
            'model_validation', 'model_persistence'
        ],
        PipelineStage.BACKTESTING: [
            'walk_forward_validation', 'monte_carlo_simulation', 'final_parameters_optimization',
            'performance_analytics', 'reporting'
        ]
    })
    
    # Custom parameters for each stage
    stage_params: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class MainPipelineResult:
    """Result of main pipeline execution."""
    pipeline_id: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Stage results
    stage_results: Dict[PipelineStage, List[Any]] = field(default_factory=dict)
    
    # Overall metrics
    total_sub_pipelines: int = 0
    completed_sub_pipelines: int = 0
    failed_sub_pipelines: int = 0
    success_rate: float = 0.0
    
    # Artifacts and outputs
    artifacts: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    
    # Error information
    error_message: Optional[str] = None
    failed_stages: List[PipelineStage] = field(default_factory=list)
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class MainTrainingPipeline:
    """
    Main Training Pipeline Manager.
    
    Orchestrates the execution of all sub-pipelines across different stages
    with comprehensive monitoring and error handling.
    """
    
    def __init__(self, config: Optional[MainPipelineConfig] = None):
        """Initialize the main training pipeline."""
        self.config = config or MainPipelineConfig()
        self.logger = logger.getChild('MainTrainingPipeline')
        
        # Initialize sub-pipeline managers
        self.data_collection_pipeline = DataCollectionSubPipeline()
        self.market_analysis_pipeline = MarketAnalysisSubPipeline()
        self.model_training_pipeline = ModelTrainingSubPipeline()
        self.backtesting_pipeline = BacktestingSubPipeline()
        
        # Pipeline state
        self.current_stage: Optional[PipelineStage] = None
        self.pipeline_results: List[MainPipelineResult] = []
    
    async def execute_pipeline(
        self,
        config: Optional[MainPipelineConfig] = None
    ) -> MainPipelineResult:
        """
        Execute the complete training pipeline.
        
        Args:
            config: Optional configuration override
            
        Returns:
            MainPipelineResult with execution details
        """
        config = config or self.config
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"🚀 Starting main training pipeline: {pipeline_id} (mode: {config.mode.value})")
        
        start_time = datetime.now()
        result = MainPipelineResult(
            pipeline_id=pipeline_id,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Execute each enabled stage
            for stage in config.enabled_stages:
                self.logger.info(f"📋 Executing stage: {stage.value}")
                self.current_stage = stage
                
                stage_result = await self._execute_stage(stage, config)
                result.stage_results[stage] = stage_result
                
                # Check if stage failed and handle accordingly
                failed_sub_pipelines = [r for r in stage_result if r.status == SubPipelineStatus.FAILED]
                if failed_sub_pipelines and config.mode != ExecutionMode.BLANK:
                    self.logger.warning(f"⚠️ Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                    result.failed_stages.append(stage)
            
            # Calculate overall metrics
            self._calculate_pipeline_metrics(result)
            
            # Update result status
            end_time = datetime.now()
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            if result.failed_sub_pipelines == 0:
                result.status = SubPipelineStatus.COMPLETED
                self.logger.info(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Pipeline failed with {result.failed_sub_pipelines} failed sub-pipelines"
                self.logger.error(f"❌ Main training pipeline {pipeline_id} failed: {result.error_message}")
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            self.logger.error(f"❌ Main training pipeline {pipeline_id} failed with exception: {e}")
        
        self.pipeline_results.append(result)
        return result
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        config: MainPipelineConfig
    ) -> List[Any]:
        """
        Execute a specific pipeline stage.
        
        Args:
            stage: Pipeline stage to execute
            config: Pipeline configuration
            
        Returns:
            List of sub-pipeline results for the stage
        """
        self.logger.info(f"🎯 Executing stage: {stage.value}")
        
        # Get enabled sub-pipelines for this stage
        enabled_sub_pipelines = config.enabled_sub_pipelines.get(stage, [])
        if not enabled_sub_pipelines:
            self.logger.warning(f"⚠️ No sub-pipelines enabled for stage: {stage.value}")
            return []
        
        # Create stage-specific configuration
        stage_config = self._create_stage_config(stage, config)
        
        # Execute sub-pipelines based on stage
        if stage == PipelineStage.DATA_COLLECTION:
            return await self._execute_data_collection_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.MARKET_ANALYSIS:
            return await self._execute_market_analysis_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.MODEL_TRAINING:
            return await self._execute_model_training_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.BACKTESTING:
            return await self._execute_backtesting_stage(enabled_sub_pipelines, stage_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")
    
    def _create_stage_config(self, stage: PipelineStage, config: MainPipelineConfig) -> Any:
        """Create stage-specific configuration."""
        base_config = {
            'mode': config.mode,
            'symbol': config.symbol,
            'exchange': config.exchange,
            'timeframe': config.timeframe,
            'data_dir': config.data_dir,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'force_rerun': config.force_rerun,
            'parallel_processing': config.parallel_processing,
            'max_workers': config.max_workers,
            'validation_enabled': config.validation_enabled,
            'monitoring_enabled': config.monitoring_enabled,
            'custom_params': config.stage_params.get(stage, {})
        }
        
        if stage == PipelineStage.DATA_COLLECTION:
            return DataCollectionConfig(**base_config)
        elif stage == PipelineStage.MARKET_ANALYSIS:
            return MarketAnalysisConfig(**base_config)
        elif stage == PipelineStage.MODEL_TRAINING:
            return ModelTrainingConfig(**base_config)
        elif stage == PipelineStage.BACKTESTING:
            return BacktestingConfig(**base_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")
    
    async def _execute_data_collection_stage(
        self,
        sub_pipeline_names: List[str],
        config: DataCollectionConfig
    ) -> List[DataCollectionResult]:
        """Execute data collection stage."""
        self.logger.info(f"📥 Executing data collection stage with {len(sub_pipeline_names)} sub-pipelines")
        
        # Update pipeline configuration
        self.data_collection_pipeline.config = config
        
        # Execute sub-pipelines
        if config.parallel_processing:
            results = await self.data_collection_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=False
            )
        else:
            results = await self.data_collection_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=True
            )
        
        return results
    
    async def _execute_market_analysis_stage(
        self,
        sub_pipeline_names: List[str],
        config: MarketAnalysisConfig
    ) -> List[MarketAnalysisResult]:
        """Execute market analysis stage."""
        self.logger.info(f"📊 Executing market analysis stage with {len(sub_pipeline_names)} sub-pipelines")
        
        # Update pipeline configuration
        self.market_analysis_pipeline.config = config
        
        # Execute sub-pipelines
        if config.parallel_processing:
            results = await self.market_analysis_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=False
            )
        else:
            results = await self.market_analysis_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=True
            )
        
        return results
    
    async def _execute_model_training_stage(
        self,
        sub_pipeline_names: List[str],
        config: ModelTrainingConfig
    ) -> List[ModelTrainingResult]:
        """Execute model training stage."""
        self.logger.info(f"🤖 Executing model training stage with {len(sub_pipeline_names)} sub-pipelines")
        
        # Update pipeline configuration
        self.model_training_pipeline.config = config
        
        # Execute sub-pipelines
        if config.parallel_processing:
            results = await self.model_training_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=False
            )
        else:
            results = await self.model_training_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=True
            )
        
        return results
    
    async def _execute_backtesting_stage(
        self,
        sub_pipeline_names: List[str],
        config: BacktestingConfig
    ) -> List[BacktestingResult]:
        """Execute backtesting stage."""
        self.logger.info(f"📈 Executing backtesting stage with {len(sub_pipeline_names)} sub-pipelines")
        
        # Update pipeline configuration
        self.backtesting_pipeline.config = config
        
        # Execute sub-pipelines
        if config.parallel_processing:
            results = await self.backtesting_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=False
            )
        else:
            results = await self.backtesting_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=True
            )
        
        return results
    
    def _calculate_pipeline_metrics(self, result: MainPipelineResult) -> None:
        """Calculate overall pipeline metrics."""
        total_sub_pipelines = 0
        completed_sub_pipelines = 0
        failed_sub_pipelines = 0
        
        # Aggregate metrics from all stages
        for stage, stage_results in result.stage_results.items():
            for sub_result in stage_results:
                total_sub_pipelines += 1
                if sub_result.status == SubPipelineStatus.COMPLETED:
                    completed_sub_pipelines += 1
                elif sub_result.status == SubPipelineStatus.FAILED:
                    failed_sub_pipelines += 1
        
        result.total_sub_pipelines = total_sub_pipelines
        result.completed_sub_pipelines = completed_sub_pipelines
        result.failed_sub_pipelines = failed_sub_pipelines
        result.success_rate = completed_sub_pipelines / total_sub_pipelines if total_sub_pipelines > 0 else 0
        
        # Calculate performance metrics
        result.performance_metrics = {
            'total_sub_pipelines': total_sub_pipelines,
            'completed_sub_pipelines': completed_sub_pipelines,
            'failed_sub_pipelines': failed_sub_pipelines,
            'success_rate': result.success_rate,
            'stages_completed': len([s for s in result.stage_results.keys() if s not in result.failed_stages]),
            'stages_failed': len(result.failed_stages)
        }
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific pipeline execution."""
        for result in self.pipeline_results:
            if result.pipeline_id == pipeline_id:
                return result.status
        return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline executions."""
        total_executions = len(self.pipeline_results)
        completed = sum(1 for r in self.pipeline_results if r.status == SubPipelineStatus.COMPLETED)
        failed = sum(1 for r in self.pipeline_results if r.status == SubPipelineStatus.FAILED)
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
        if stage == PipelineStage.DATA_COLLECTION:
            return self.data_collection_pipeline.get_available_sub_pipelines()
        elif stage == PipelineStage.MARKET_ANALYSIS:
            return self.market_analysis_pipeline.get_available_sub_pipelines()
        elif stage == PipelineStage.MODEL_TRAINING:
            return self.model_training_pipeline.get_available_sub_pipelines()
        elif stage == PipelineStage.BACKTESTING:
            return self.backtesting_pipeline.get_available_sub_pipelines()
        else:
            return []
    
    def get_stage_execution_summary(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get execution summary for a specific stage."""
        if stage == PipelineStage.DATA_COLLECTION:
            return self.data_collection_pipeline.get_execution_summary()
        elif stage == PipelineStage.MARKET_ANALYSIS:
            return self.market_analysis_pipeline.get_execution_summary()
        elif stage == PipelineStage.MODEL_TRAINING:
            return self.model_training_pipeline.get_execution_summary()
        elif stage == PipelineStage.BACKTESTING:
            return self.backtesting_pipeline.get_execution_summary()
        else:
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

# Predefined pipeline configurations
def get_full_pipeline_config(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "data/training"
) -> MainPipelineConfig:
    """Get a full pipeline configuration with all stages and sub-pipelines enabled."""
    return MainPipelineConfig(
        mode=ExecutionMode.FULL,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: [
                'data_download', 'data_conversion', 'data_validation', 'data_preparation',
                'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring'
            ],
            PipelineStage.MARKET_ANALYSIS: [
                'sr_detection', 'sr_clustering', 'sr_ml_learning', 'hmm_clustering',
                'hmm_regime_discovery', 'regime_data_splitting', 'triple_barrier_labeling',
                'feature_lookback_optimization', 'fractional_differentiation', 'cross_timeframe_analysis'
            ],
            PipelineStage.MODEL_TRAINING: [
                'general_model_training', 'analyst_model_training', 'tactician_model_training',
                'hmm_training', 'ensemble_training', 'multi_timeframe_training',
                'regime_specific_training', 'model_validation', 'model_persistence', 'model_evaluation'
            ],
            PipelineStage.BACKTESTING: [
                'walk_forward_validation', 'monte_carlo_simulation', 'ab_testing',
                'model_persistence', 'final_parameters_optimization', 'performance_analytics',
                'risk_analysis', 'trade_analysis', 'portfolio_analysis', 'reporting'
            ]
        }
    )

def get_light_pipeline_config(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "data/training"
) -> MainPipelineConfig:
    """Get a light pipeline configuration with essential sub-pipelines only."""
    return MainPipelineConfig(
        mode=ExecutionMode.LIGHT,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: [
                'data_download', 'data_conversion', 'data_validation'
            ],
            PipelineStage.MARKET_ANALYSIS: [
                'sr_detection', 'hmm_regime_discovery', 'triple_barrier_labeling'
            ],
            PipelineStage.MODEL_TRAINING: [
                'general_model_training', 'model_validation'
            ],
            PipelineStage.BACKTESTING: [
                'walk_forward_validation', 'performance_analytics'
            ]
        }
    )

def get_blank_pipeline_config(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "data/training"
) -> MainPipelineConfig:
    """Get a blank pipeline configuration for testing/validation."""
    return MainPipelineConfig(
        mode=ExecutionMode.BLANK,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: ['data_download', 'data_conversion'],
            PipelineStage.MARKET_ANALYSIS: ['sr_detection', 'hmm_regime_discovery'],
            PipelineStage.MODEL_TRAINING: ['general_model_training'],
            PipelineStage.BACKTESTING: ['walk_forward_validation']
        }
    )