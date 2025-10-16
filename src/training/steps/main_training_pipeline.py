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
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

class ExecutionModeType(Enum):
    """Execution mode types for stage/sub-pipeline specific execution."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class DirectionType(Enum):
    """Direction types for trading."""
    LONGS = "longs"        # Long positions only
    SHORTS = "shorts"      # Short positions only
    BOTH = "both"          # Both long and short positions

# Import enhanced utilities for comprehensive logging and operations
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug
from src.utils.common_operations import (
    safe_divide, safe_mean, safe_std, validate_finite, optimize_dataframe_dtypes,
    calculate_data_quality_metrics, get_dataframe_info
)
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Define ExecutionMode locally as fallback
class ExecutionMode(Enum):
    """Execution modes for the pipeline."""
    FULL = "full"
    LIGHT = "light"
    BLANK = "blank"

# Define SubPipelineStatus locally as fallback
class SubPipelineStatus(Enum):
    """Status values for sub-pipelines."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Define PipelineStage enum
class PipelineStage(Enum):
    """Pipeline stages."""
    DATA_COLLECTION = "data_collection"
    MARKET_ANALYSIS = "market_analysis"
    PRE_TRAINING = "pre_training"
    MODEL_TRAINING = "model_training"
    BACKTESTING = "backtesting"

# Lazy imports to avoid circular dependency
def get_system_logger():
    from src.utils.logger import system_logger
    return system_logger

# Import standardized logging
try:
    from src.training.steps.market_analysis.logging_standards import (
        get_logger, log_info, log_warning, log_error, log_success, log_debug,
        LoggingContext, log_step_progress, log_data_info, log_validation_result
    )
    STANDARDIZED_LOGGING_AVAILABLE = True
except ImportError:
    STANDARDIZED_LOGGING_AVAILABLE = False

from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_performance
from src.utils.common_operations import (
    safe_json_load, safe_json_dump, ensure_directory, safe_file_exists,
    timed_operation, format_bytes, get_memory_usage, optimize_memory
)

# Import sub-pipelines with optional imports
try:
    from .data_collection.sub_pipeline import (
        DataCollectionSubPipeline, SubPipelineConfig as DataCollectionConfig,
        SubPipelineResult as DataCollectionResult
    )
    # Import base classes for general use
    from .data_collection.sub_pipeline import SubPipelineResult, SubPipelineConfig
    DATA_COLLECTION_AVAILABLE = True
except ImportError:
    DATA_COLLECTION_AVAILABLE = False
    DataCollectionSubPipeline = None
    DataCollectionConfig = None
    DataCollectionResult = None
    SubPipelineResult = None
    SubPipelineConfig = None

try:
    from .market_analysis.sub_pipeline import (
        MarketAnalysisSubPipeline, SubPipelineConfig as MarketAnalysisConfig,
        SubPipelineResult as MarketAnalysisResult
    )
    MARKET_ANALYSIS_AVAILABLE = True
except ImportError:
    MARKET_ANALYSIS_AVAILABLE = False
    MarketAnalysisSubPipeline = None
    MarketAnalysisConfig = None
    MarketAnalysisResult = None

try:
    from .pre_training.sub_pipeline import (
        PreTrainingSubPipeline, SubPipelineConfig as PreTrainingConfig,
        SubPipelineResult as PreTrainingResult
    )
    PRE_TRAINING_AVAILABLE = True
except ImportError:
    PRE_TRAINING_AVAILABLE = False
    PreTrainingSubPipeline = None
    PreTrainingConfig = None
    PreTrainingResult = None

try:
    from .model_training.sub_pipeline import (
        ModelTrainingSubPipeline, SubPipelineConfig as ModelTrainingConfig,
        SubPipelineResult as ModelTrainingResult
    )
    MODEL_TRAINING_AVAILABLE = True
except ImportError:
    MODEL_TRAINING_AVAILABLE = False
    ModelTrainingSubPipeline = None
    ModelTrainingConfig = None
    ModelTrainingResult = None

try:
    from .backtesting.sub_pipeline import (
        BacktestingSubPipeline, SubPipelineConfig as BacktestingConfig,
        SubPipelineResult as BacktestingResult
    )
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    BacktestingSubPipeline = None
    BacktestingConfig = None
    BacktestingResult = None

logger = get_system_logger().getChild('MainTrainingPipeline')

class PipelineStage(Enum):
    """Pipeline execution stages."""
    DATA_COLLECTION = "data_collection"
    MARKET_ANALYSIS = "market_analysis"
    PRE_TRAINING = "pre_training"
    MODEL_TRAINING = "model_training"
    BACKTESTING = "backtesting"

@dataclass
class MainPipelineConfig:
    """Configuration for the main training pipeline."""
    # General configuration
    mode: ExecutionMode = ExecutionMode.FULL
    execution_mode: ExecutionModeType = ExecutionModeType.LIGHT
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "15m"
    data_dir: str = "historical_data"
    direction: DirectionType = DirectionType.LONGS
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
        PipelineStage.PRE_TRAINING,
        PipelineStage.MODEL_TRAINING,
        PipelineStage.BACKTESTING
    ])

    # Sub-pipeline control
    enabled_sub_pipelines: Dict[PipelineStage, List[str]] = field(default_factory=lambda: {
        PipelineStage.DATA_COLLECTION: [
            'data_download', 'data_conversion', 'data_validation', 'data_preparation',
            'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring',
            'data_integration', 'data_export'
        ],
        PipelineStage.MARKET_ANALYSIS: [
            'sr_parameter_optimization', 'sr_detection', 'sr_clustering',
            'nas_tas_regime_discovery', 'nas_tas_clustering', 'regime_models_training', 'regime_ensemble_training',
            'regime_data_splitting', 'sr_feature_integration'
        ],
        PipelineStage.PRE_TRAINING: [
            'multi_horizon_profit_labeler',
            'interactive_feature_generation', 'final_feature_selection'
        ],
        PipelineStage.MODEL_TRAINING: [
            'analyst_pre_ml_orchestration', 'analyst_models_training', 'analyst_ensemble_training',
            'tactician_pre_ml_orchestration', 'tactician_models_training', 'tactician_ensemble_training'
        ],
        PipelineStage.BACKTESTING: [
            'basic_backtesting_pre', 'final_parameters_optimization', 'basic_backtesting_post', 'walk_forward_validation', 'monte_carlo_simulation', 'ab_testing',
            'model_persistence', 'performance_analytics',
            'risk_analysis', 'trade_analysis', 'portfolio_analysis', 'reporting'
        ]
    })

    # Custom parameters for each stage
    stage_params: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)

    # Intensity parameters for ML training
    intensity_percentage: float = 1.0  # Default to 100% intensity
    training_mode_config: Optional[Dict[str, Any]] = None

    # Direction control for trading (longs, shorts, or both)
    direction_type: DirectionType = DirectionType.LONGS  # Default to long positions

    # Single stage execution control
    single_stage_only: bool = False  # Control whether to execute only the requested stage

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
        # Use standardized logging if available
        if STANDARDIZED_LOGGING_AVAILABLE:
            self.logger = get_logger('MainTrainingPipeline')
        else:
            self.logger = logger.getChild('MainTrainingPipeline')

        # Initialize sub-pipeline managers (only if available)
        self.data_collection_pipeline = DataCollectionSubPipeline() if DATA_COLLECTION_AVAILABLE else None
        self.market_analysis_pipeline = MarketAnalysisSubPipeline() if MARKET_ANALYSIS_AVAILABLE else None
        self.pre_training_pipeline = PreTrainingSubPipeline() if PRE_TRAINING_AVAILABLE else None
        self.model_training_pipeline = ModelTrainingSubPipeline() if MODEL_TRAINING_AVAILABLE else None
        self.backtesting_pipeline = BacktestingSubPipeline() if BACKTESTING_AVAILABLE else None

        # Pipeline state
        self.current_stage: Optional[PipelineStage] = None
        self.pipeline_results: List[MainPipelineResult] = []

    @staticmethod
    def _extract_subpipeline_failure(result: Any) -> Tuple[Optional[str], Optional[str]]:
        error_code = getattr(result, 'error_code', None)
        message = getattr(result, 'error_message', None)
        failure = getattr(result, 'failure', None)
        if not message and failure is not None:
            message = getattr(failure, 'message', None) or message
        if not error_code and failure is not None:
            error_code = getattr(failure, 'error_code', None) or error_code
        return error_code, message

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

        # Enhanced logging with tprint
        tprint(f"🚀 Starting main training pipeline: {pipeline_id} (mode: {config.mode.value})")
        tprint_debug(f"📋 Pipeline configuration: {len(config.enabled_stages)} stages enabled")

        if STANDARDIZED_LOGGING_AVAILABLE:
            log_info(f"🚀 Starting main training pipeline: {pipeline_id} (mode: {config.mode.value})")
        else:
            self.logger.info(f"🚀 Starting main training pipeline: {pipeline_id} (mode: {config.mode.value})")

        start_time = datetime.now()
        result = MainPipelineResult(
            pipeline_id=pipeline_id,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Execute each enabled stage with comprehensive logging
            total_stages = len(config.enabled_stages)
            for i, stage in enumerate(config.enabled_stages, 1):
                tprint(f"📋 [{i}/{total_stages}] Executing stage: {stage.value}")
                tprint_info(f"🔄 Stage progress: {i}/{total_stages} ({100*i//total_stages}%)")

                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_info(f"📋 Executing stage: {stage.value}")
                else:
                    self.logger.info(f"📋 Executing stage: {stage.value}")
                self.current_stage = stage

                stage_result = await self._execute_stage(stage, config)
                result.stage_results[stage] = stage_result

                # Log stage completion with timing
                stage_duration = (datetime.now() - start_time).total_seconds()
                tprint_success(f"✅ Stage {stage.value} completed in {stage_duration:.1f}s")

                # Check if stage failed and handle accordingly
                failed_sub_pipelines = [r for r in stage_result if r.status == SubPipelineStatus.FAILED]
                if failed_sub_pipelines and config.mode != ExecutionMode.BLANK:
                    if STANDARDIZED_LOGGING_AVAILABLE:
                        log_warning(f"⚠️ Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                    else:
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
                # Enhanced success logging with tprint
                tprint_success(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
                tprint(f"📊 Execution summary: {result.completed_sub_pipelines}/{result.total_sub_pipelines} sub-pipelines completed")
                tprint(f"🏆 Success rate: {result.success_rate:.1%}")

                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_success(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
                else:
                    self.logger.info(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
            else:
                result.status = SubPipelineStatus.FAILED
                result.error_message = f"Pipeline failed with {result.failed_sub_pipelines} failed sub-pipelines"
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_error(f"❌ Main training pipeline {pipeline_id} failed: {result.error_message}")
                else:
                    self.logger.error(f"❌ Main training pipeline {pipeline_id} failed: {result.error_message}")

        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)

            if STANDARDIZED_LOGGING_AVAILABLE:
                log_error(f"❌ Main training pipeline {pipeline_id} failed with exception: {e}")
            else:
                self.logger.error(f"❌ Main training pipeline {pipeline_id} failed with exception: {e}")

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
        if STANDARDIZED_LOGGING_AVAILABLE:
            log_info(f"🚀 Starting sub-pipeline chain: {stage.value} -> {starting_sub_pipeline}")
        else:
            self.logger.info(f"🚀 Starting sub-pipeline chain: {stage.value} -> {starting_sub_pipeline}")

        # Create stage-specific configuration
        stage_config = self._create_stage_config(stage, config)

        # Execute the sub-pipeline with automatic next triggering
        if stage == PipelineStage.MARKET_ANALYSIS:
            if not MARKET_ANALYSIS_AVAILABLE:
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_warning("⚠️ Market analysis sub-pipeline not available")
                else:
                    self.logger.warning("⚠️ Market analysis sub-pipeline not available")
                return None
            return await self.market_analysis_pipeline.execute_sub_pipeline_with_next(starting_sub_pipeline, stage_config)
        elif stage == PipelineStage.PRE_TRAINING:
            if not PRE_TRAINING_AVAILABLE:
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_warning("⚠️ Pre-training sub-pipeline not available")
                else:
                    self.logger.warning("⚠️ Pre-training sub-pipeline not available")
                return None
            return await self.pre_training_pipeline.execute_sub_pipeline_with_next(starting_sub_pipeline, stage_config)
        elif stage == PipelineStage.MODEL_TRAINING:
            if not MODEL_TRAINING_AVAILABLE:
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_warning("⚠️ Model training sub-pipeline not available")
                else:
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
            if not DATA_COLLECTION_AVAILABLE:
                self.logger.warning("⚠️ Data collection sub-pipeline not available")
                return []
            return await self._execute_data_collection_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.MARKET_ANALYSIS:
            if not MARKET_ANALYSIS_AVAILABLE:
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_warning("⚠️ Market analysis sub-pipeline not available")
                else:
                    self.logger.warning("⚠️ Market analysis sub-pipeline not available")
                return []
            return await self._execute_market_analysis_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.PRE_TRAINING:
            if not PRE_TRAINING_AVAILABLE:
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_warning("⚠️ Pre-training sub-pipeline not available")
                else:
                    self.logger.warning("⚠️ Pre-training sub-pipeline not available")
                return []
            return await self._execute_pre_training_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.MODEL_TRAINING:
            if not MODEL_TRAINING_AVAILABLE:
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_warning("⚠️ Model training sub-pipeline not available")
                else:
                    self.logger.warning("⚠️ Model training sub-pipeline not available")
                return []
            return await self._execute_model_training_stage(enabled_sub_pipelines, stage_config)
        elif stage == PipelineStage.BACKTESTING:
            if not BACKTESTING_AVAILABLE:
                self.logger.warning("⚠️ Backtesting sub-pipeline not available")
                return []
            return await self._execute_backtesting_stage(enabled_sub_pipelines, stage_config)
        else:
            raise ValueError(f"Unknown pipeline stage: {stage}")

    def _create_stage_config(self, stage: PipelineStage, config: MainPipelineConfig) -> Any:
        """Create stage-specific configuration."""
        # Base configuration that all stage configs support
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
            'custom_params': config.stage_params.get(stage, {}),
            # Direction settings from main pipeline config
            'enable_long_positions': config.direction_type in [DirectionType.LONGS, DirectionType.BOTH],
            'enable_short_positions': config.direction_type in [DirectionType.SHORTS, DirectionType.BOTH]
        }

        # Add flag to use existing data instead of downloading new data
        # This is crucial for the current setup where we want to use pre-existing data
        base_config['use_existing_data'] = True

        # Remove parameters that are specific to MainPipelineConfig and not supported by stage configs
        # single_stage_only is a MainPipelineConfig parameter used for controlling pipeline execution
        # but it's not needed by individual stage configurations

        if stage == PipelineStage.DATA_COLLECTION:
            if DATA_COLLECTION_AVAILABLE:
                return DataCollectionConfig(**base_config)
            else:
                return base_config
        elif stage == PipelineStage.MARKET_ANALYSIS:
            if MARKET_ANALYSIS_AVAILABLE:
                return MarketAnalysisConfig(**base_config)
            else:
                return base_config
        elif stage == PipelineStage.PRE_TRAINING:
            if PRE_TRAINING_AVAILABLE:
                return PreTrainingConfig(**base_config)
            else:
                return base_config
        elif stage == PipelineStage.MODEL_TRAINING:
            if MODEL_TRAINING_AVAILABLE:
                return ModelTrainingConfig(**base_config)
            else:
                return base_config
        elif stage == PipelineStage.BACKTESTING:
            if BACKTESTING_AVAILABLE:
                return BacktestingConfig(**base_config)
            else:
                return base_config
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

    async def _execute_stage_from_sub_pipeline(
        self,
        stage: PipelineStage,
        sub_pipeline_names: List[str],
        config: MainPipelineConfig
    ) -> List[Any]:
        """
        Execute a stage starting from a specific sub-pipeline, running subsequent
        sub-pipelines sequentially.

        Args:
            stage: Pipeline stage to execute
            sub_pipeline_names: List of sub-pipelines to execute (starting from the specified one)
            config: Pipeline configuration

        Returns:
            List of sub-pipeline results for the stage
        """
        self.logger.info(f"🎯 Executing stage from sub-pipeline: {stage.value} with {len(sub_pipeline_names)} sub-pipelines")

        # Create stage-specific configuration
        stage_config = self._create_stage_config(stage, config)

        # Execute sub-pipelines sequentially (not in parallel) to ensure proper order
        results = []
        for i, sub_pipeline_name in enumerate(sub_pipeline_names):
            self.logger.info(f"🔄 Executing sub-pipeline {i+1}/{len(sub_pipeline_names)}: {sub_pipeline_name}")

            try:
                if stage == PipelineStage.DATA_COLLECTION:
                    if not DATA_COLLECTION_AVAILABLE:
                        self.logger.warning("⚠️ Data collection sub-pipeline not available")
                        continue
                    result = await self.data_collection_pipeline.execute_sub_pipeline(sub_pipeline_name, stage_config)
                elif stage == PipelineStage.MARKET_ANALYSIS:
                    if not MARKET_ANALYSIS_AVAILABLE:
                        if STANDARDIZED_LOGGING_AVAILABLE:
                            log_warning("⚠️ Market analysis sub-pipeline not available")
                        else:
                            self.logger.warning("⚠️ Market analysis sub-pipeline not available")
                    # Use execute_sub_pipeline_with_next for automatic sequential execution
                    result = await self.market_analysis_pipeline.execute_sub_pipeline_with_next(sub_pipeline_name, stage_config)
                elif stage == PipelineStage.PRE_TRAINING:
                    if not PRE_TRAINING_AVAILABLE:
                        if STANDARDIZED_LOGGING_AVAILABLE:
                            log_warning("⚠️ Pre-training sub-pipeline not available")
                        else:
                            self.logger.warning("⚠️ Pre-training sub-pipeline not available")
                        continue
                    # Use execute_sub_pipeline_with_next for automatic sequential execution
                    result = await self.pre_training_pipeline.execute_sub_pipeline_with_next(sub_pipeline_name, stage_config)
                elif stage == PipelineStage.MODEL_TRAINING:
                    if not MODEL_TRAINING_AVAILABLE:
                        if STANDARDIZED_LOGGING_AVAILABLE:
                            log_warning("⚠️ Model training sub-pipeline not available")
                        else:
                            self.logger.warning("⚠️ Model training sub-pipeline not available")
                        continue
                    # Use execute_sub_pipeline_with_next for automatic sequential execution
                    result = await self.model_training_pipeline.execute_sub_pipeline_with_next(sub_pipeline_name, stage_config)
                elif stage == PipelineStage.BACKTESTING:
                    if not BACKTESTING_AVAILABLE:
                        self.logger.warning("⚠️ Backtesting sub-pipeline not available")
                        continue
                    # Use execute_sub_pipeline_with_next for automatic sequential execution
                    result = await self.backtesting_pipeline.execute_sub_pipeline_with_next(sub_pipeline_name, stage_config)
                else:
                    raise ValueError(f"Unknown pipeline stage: {stage}")

                results.append(result)

                # Check if this sub-pipeline failed
                if result.status == SubPipelineStatus.FAILED:
                    error_code, message = self._extract_subpipeline_failure(result)
                    code_text = f"[{error_code}] " if error_code else ''
                    detail = f"{code_text}{message}" if message else code_text.strip()
                    failure_msg = (
                        f"❌ Sub-pipeline {sub_pipeline_name} failed"
                        f"{': ' + detail if detail else ''}, stopping sequential execution"
                    )
                    self.logger.error(failure_msg)
                    break
                else:
                    self.logger.info(f"✅ Sub-pipeline {sub_pipeline_name} completed successfully")

            except Exception as e:
                self.logger.error(f"❌ Error executing sub-pipeline {sub_pipeline_name}: {e}")
                # Create a failed result
                failed_result = type(results[0] if results else SubPipelineResult)(
                    sub_pipeline_name=sub_pipeline_name,
                    status=SubPipelineStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(e)
                )
                if hasattr(failed_result, 'error_code'):
                    setattr(failed_result, 'error_code', 'PRETRAIN_STAGE_EXEC_ERROR')
                results.append(failed_result)
                break

        return results

    async def _execute_market_analysis_stage(
        self,
        sub_pipeline_names: List[str],
        config: MarketAnalysisConfig
    ) -> List[MarketAnalysisResult]:
        """Execute market analysis stage with automatic sequential progression."""
        self.logger.info(f"📊 Executing market analysis stage with {len(sub_pipeline_names)} sub-pipelines")
        self.logger.info("🔄 MARKET_ANALYSIS stage configured for automatic sequential execution")

        # Load market data for analysis using existing klines parquet utility
        from src.utils.data.klines_parquet import get_klines_manager

        self.logger.info("📂 Loading market data for analysis...")
        klines_manager = get_klines_manager(data_dir=config.data_dir)

        # Parse start_date and end_date from config if provided
        start_date = None
        end_date = None
        if hasattr(config, 'start_date') and config.start_date:
            start_date = datetime.strptime(config.start_date, '%Y-%m-%d')
            self.logger.info(f"📅 Using start_date filter: {start_date} (mode: {config.mode.value})")
        if hasattr(config, 'end_date') and config.end_date:
            end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
            self.logger.info(f"📅 Using end_date filter: {end_date} (mode: {config.mode.value})")

        # Load data with date filtering if specified
        market_data = klines_manager.read_data(
            symbol=config.symbol,
            interval=config.timeframe,
            data_type="processed",  # Use processed data
            start_date=start_date,
            end_date=end_date
        )

        if market_data is None or len(market_data) == 0:
            self.logger.error(f"❌ Failed to load market data for {config.symbol} {config.timeframe}")
            return []

        self.logger.info(f"✅ Loaded market data: {market_data.shape[0]} rows, {market_data.shape[1]} columns")
        self.logger.info(f"📊 Data columns: {list(market_data.columns)}")
        self.logger.info(f"📅 Date range: {market_data.index.min()} to {market_data.index.max()}")

        # Prepare pipeline state with data
        # Ensure timestamp column exists for data quality framework
        if 'timestamp' not in market_data.columns and isinstance(market_data.index, pd.DatetimeIndex):
            market_data = market_data.copy()
            market_data['timestamp'] = market_data.index
            self.logger.info("✅ Added timestamp column from DatetimeIndex for data quality framework")

        pipeline_state = {
            'dataframe': market_data,
            'symbol': config.symbol,
            'exchange': config.exchange,
            'timeframe': config.timeframe,
            'data_points': len(market_data)
        }

        # Update pipeline configuration
        self.market_analysis_pipeline.config = config

        # Pass the loaded data to the sub-pipeline
        self.market_analysis_pipeline._current_data = pipeline_state['dataframe']
        self.market_analysis_pipeline._current_pipeline_state = pipeline_state

        # For MARKET_ANALYSIS, use sequential execution with automatic progression
        # Start with the first sub-pipeline and let it trigger the next ones
        self.logger.info("🚀 Starting automatic sequential execution: sr_parameter_optimization -> sr_detection -> sr_clustering -> nas_tas_regime_discovery -> nas_tas_clustering -> regime_models_training -> regime_ensemble_training -> hybrid_nas_tas_regime_discovery -> nas_tas_clustering -> regime_models_training -> regime_ensemble_training -> regime_data_splitting")

        results = []
        if sub_pipeline_names:
            # Execute the first sub-pipeline with automatic next triggering
            first_result = await self.market_analysis_pipeline.execute_sub_pipeline_with_next(
                sub_pipeline_names[0], config
            )
            results.append(first_result)

            # The first sub-pipeline will have automatically triggered all subsequent ones
            # Add their results to our results list
            for result in self.market_analysis_pipeline.results:
                if result not in results:  # Avoid duplicates
                    results.append(result)

        return results

    async def _execute_pre_training_stage(
        self,
        sub_pipeline_names: List[str],
        config: PreTrainingConfig
    ) -> List[PreTrainingResult]:
        """Execute pre-training stage with automatic sequential progression."""
        self.logger.info(f"📊 Executing pre-training stage with {len(sub_pipeline_names)} sub-pipelines")
        self.logger.info("🔄 PRE_TRAINING stage configured for automatic sequential execution")

        results = []

        try:
            # Execute each sub-pipeline in sequence
            for sub_pipeline_name in sub_pipeline_names:
                self.logger.info(f"🚀 Executing pre-training sub-pipeline: {sub_pipeline_name}")

                # Execute the sub-pipeline
                result = await self.pre_training_pipeline.execute_sub_pipeline(sub_pipeline_name, config)
                results.append(result)

                # Check sub-pipeline result status
                if result.status == SubPipelineStatus.FAILED:
                    self.logger.error(f"❌ Sub-pipeline {sub_pipeline_name} FAILED")
                    if result.error_message:
                        self.logger.error(f"   Error: {result.error_message}")
                    self.logger.error(f"   Stopping sequential execution")
                    break
                elif result.status == SubPipelineStatus.COMPLETED:
                    self.logger.info(f"✅ Sub-pipeline {sub_pipeline_name} completed successfully")
                else:
                    self.logger.warning(f"⚠️ Sub-pipeline {sub_pipeline_name} finished with status: {result.status.value}")
                    # Continue execution for non-FAILED statuses

        except Exception as e:
            self.logger.error(f"❌ Error executing pre-training stage: {e}")
            # Create a failed result for the problematic sub-pipeline
            if sub_pipeline_names:
                failed_result = PreTrainingResult(
                    sub_pipeline_name=sub_pipeline_names[0],
                    status=SubPipelineStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(e)
                )
                results.append(failed_result)

        return results

    async def _execute_model_training_stage(
        self,
        sub_pipeline_names: List[str],
        config: ModelTrainingConfig
    ) -> List[ModelTrainingResult]:
        """Execute model training stage."""
        self.logger.info(f"🤖 Executing model training stage with {len(sub_pipeline_names)} sub-pipelines")
        tprint(f"🤖 [MAIN_TRAINING] Starting model training stage with {len(sub_pipeline_names)} sub-pipelines", color="blue")
        tprint(f"📊 [MAIN_TRAINING] Sub-pipelines: {', '.join(sub_pipeline_names)}", color="cyan")

        # Update pipeline configuration
        tprint("⚙️ [MAIN_TRAINING] Updating model training pipeline configuration", color="yellow")
        self.model_training_pipeline.config = config

        # Execute sub-pipelines
        if config.parallel_processing:
            tprint("⚡ [MAIN_TRAINING] Executing sub-pipelines in parallel mode", color="magenta")
            results = await self.model_training_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=False
            )
            tprint("✅ [MAIN_TRAINING] Parallel execution completed", color="green")
        else:
            tprint("🔄 [MAIN_TRAINING] Executing sub-pipelines sequentially", color="yellow")
            results = await self.model_training_pipeline.execute_multiple_sub_pipelines(
                sub_pipeline_names, config, sequential=True
            )
            tprint("✅ [MAIN_TRAINING] Sequential execution completed", color="green")

        tprint(f"📊 [MAIN_TRAINING] Model training stage completed with {len(results)} results", color="cyan")
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

        self.logger.info(f"🔍 Calculating metrics from {len(result.stage_results)} stages")

        # Aggregate metrics from all stages
        for stage, stage_results in result.stage_results.items():
            self.logger.info(f"📊 Stage {stage.value}: {len(stage_results)} results")
            for sub_result in stage_results:
                total_sub_pipelines += 1
                self.logger.info(f"   Sub-pipeline: {sub_result.sub_pipeline_name}, Status: {sub_result.status.value}")
                # Use success attribute for accurate success rate calculation
                if sub_result.success:
                    completed_sub_pipelines += 1
                elif sub_result.status == SubPipelineStatus.FAILED or not sub_result.success:
                    failed_sub_pipelines += 1

        result.total_sub_pipelines = total_sub_pipelines
        result.completed_sub_pipelines = completed_sub_pipelines
        result.failed_sub_pipelines = failed_sub_pipelines
        result.success_rate = completed_sub_pipelines / total_sub_pipelines if total_sub_pipelines > 0 else 0

        if STANDARDIZED_LOGGING_AVAILABLE:
            log_info(f"📈 Final metrics: Total={total_sub_pipelines}, Completed={completed_sub_pipelines}, Failed={failed_sub_pipelines}, Rate={result.success_rate:.1%}")
        else:
            self.logger.info(f"📈 Final metrics: Total={total_sub_pipelines}, Completed={completed_sub_pipelines}, Failed={failed_sub_pipelines}, Rate={result.success_rate:.1%}")

        # Calculate performance metrics with int64 conversion
        result.performance_metrics = {
            'total_sub_pipelines': int(total_sub_pipelines),
            'completed_sub_pipelines': int(completed_sub_pipelines),
            'failed_sub_pipelines': int(failed_sub_pipelines),
            'success_rate': float(result.success_rate),
            'stages_completed': int(len([s for s in result.stage_results.keys() if s not in result.failed_stages])),
            'stages_failed': int(len(result.failed_stages))
        }

        # Convert any numpy types in artifacts to Python types to avoid JSON serialization errors
        result.artifacts = self._convert_numpy_types(result.artifacts)

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to Python types to avoid JSON serialization errors."""
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def get_pipeline_status(self, pipeline_id: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific pipeline execution."""
        for result in self.pipeline_results:
            if result.pipeline_id == pipeline_id:
                return result.status
        return None

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline executions."""
        total_executions = len(self.pipeline_results)
        completed = sum(1 for r in self.pipeline_results if r.status.value == "completed")
        failed = sum(1 for r in self.pipeline_results if r.status.value == "failed")
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
            if self.data_collection_pipeline:
                return self.data_collection_pipeline.get_available_sub_pipelines()
            else:
                return []
        elif stage == PipelineStage.MARKET_ANALYSIS:
            if self.market_analysis_pipeline:
                return self.market_analysis_pipeline.get_available_sub_pipelines(stage)
            else:
                return []
        elif stage == PipelineStage.PRE_TRAINING:
            if self.pre_training_pipeline:
                return self.pre_training_pipeline.get_available_sub_pipelines()
            else:
                return []
        elif stage == PipelineStage.MODEL_TRAINING:
            if self.model_training_pipeline:
                return self.model_training_pipeline.get_available_sub_pipelines()
            else:
                return []
        elif stage == PipelineStage.BACKTESTING:
            if self.backtesting_pipeline:
                return self.backtesting_pipeline.get_available_sub_pipelines()
            else:
                return []
        else:
            return []

    def get_stage_execution_summary(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get execution summary for a specific stage."""
        if stage == PipelineStage.DATA_COLLECTION:
            return self.data_collection_pipeline.get_execution_summary()
        elif stage == PipelineStage.MARKET_ANALYSIS:
            return self.market_analysis_pipeline.get_execution_summary()
        elif stage == PipelineStage.PRE_TRAINING:
            return self.pre_training_pipeline.get_execution_summary()
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
    """Convenience function to execute the main training pipeline with comprehensive logging."""
    tprint("🚀 Executing main training pipeline via convenience function")
    tprint_info(f"📋 Configuration: symbol={config.symbol if config else 'default'}, exchange={config.exchange if config else 'default'}")

    pipeline = get_main_training_pipeline(config)
    result = await pipeline.execute_pipeline(config)

    tprint_success(f"✅ Main training pipeline execution completed: {result.status.value}")
    return result

# Predefined pipeline configurations
def get_full_pipeline_config(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a full pipeline configuration with all stages and sub-pipelines enabled."""
    from src.config.pipeline_modes import get_full_mode_config

    # Get centralized full mode configuration
    mode_config = get_full_mode_config()

    # Full mode: Use last available data instead of current date
    # This ensures we use actual historical data rather than future dates
    try:
        # Try to determine the last available date from the data
        from src.utils.data.klines_parquet import KlinesParquetManager
        manager = KlinesParquetManager(data_dir=data_dir)

        from datetime import datetime, timedelta

        # First, get data info to find the actual available date range without filtering
        data_info = manager.get_data_info(symbol=symbol, interval=timeframe, data_type="processed")

        if data_info and data_info.get("available") and data_info.get("date_range"):
            # Use the last date from the available data
            _, max_date = data_info["date_range"]
            end_date = pd.to_datetime(max_date)
            start_date = end_date - timedelta(days=mode_config.lookback_days)
            print(f"✅ Using last available data date: {end_date.strftime('%Y-%m-%d')}")
        else:
            # Fallback: Load without date filtering to find the actual date range
            print("⚠️ Could not get data info, loading sample to determine date range...")
            sample_data = manager.read_data(
                symbol=symbol,
                interval=timeframe,
                start_date=None,
                end_date=None,
                data_type="processed"
            )

            if sample_data is not None and not len(sample_data) == 0:
                # Get the last available date from the data
                if 'timestamp' in sample_data.columns:
                    timestamps = pd.to_datetime(sample_data['timestamp'], unit='s')
                    end_date = timestamps.max()
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
                else:
                    # Fallback to using the index if available
                    if hasattr(sample_data.index, 'max'):
                        end_date = sample_data.index.max()
                        start_date = end_date - timedelta(days=mode_config.lookback_days)
                    else:
                        # Final fallback to current date
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=mode_config.lookback_days)
                print(f"✅ Determined date range from data: {end_date.strftime('%Y-%m-%d')}")
            else:
                # No data available, use current date as fallback
                print("⚠️ No data available, using current date as fallback")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=mode_config.lookback_days)

    except Exception as e:
        # If there's any error, fall back to current date
        print(f"⚠️ Could not determine available data range: {e}")
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=mode_config.lookback_days)

    # Set intensity percentage for full mode
    intensity_pct = 1.0  # 100% intensity for full mode

    data_dir_path = Path(data_dir).expanduser()
    cache_base = data_dir_path if 'cache' in data_dir_path.name.lower() else data_dir_path.parent / 'data_cache'
    cache_base = cache_base.expanduser()
    pre_training_stage_params = {
        'regime_cache_path': str((cache_base / 'nas_tas_clustering')),
        'data_cache_dir': str(cache_base)
    }

    return MainPipelineConfig(
        mode=ExecutionMode.FULL,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        intensity_percentage=intensity_pct,
        training_mode_config=mode_config.__dict__,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.PRE_TRAINING,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: [
                'data_download', 'data_conversion', 'data_validation', 'data_preparation',
                'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring'
            ],
            PipelineStage.MARKET_ANALYSIS: [
                'sr_parameter_optimization', 'sr_detection', 'sr_clustering',
                'hmm_regime_discovery', 'hmm_clustering', 'hmm_models_training', 'hmm_ensemble_training',
                'regime_data_splitting', 'sr_feature_integration'
            ],
            PipelineStage.PRE_TRAINING: [
                'multi_horizon_profit_labeler',
                'interactive_feature_generation', 'final_feature_selection'
            ],
            PipelineStage.MODEL_TRAINING: [
                'analyst_model_training', 'analyst_ensemble_training',
                'tactician_lookback_optimization', 'tactician_models_training', 'tactician_ensemble_training'
            ],
            PipelineStage.BACKTESTING: [
                'basic_backtesting_pre', 'final_parameters_optimization', 'basic_backtesting_post', 'walk_forward_validation', 'monte_carlo_simulation', 'ab_testing',
                'model_persistence', 'performance_analytics',
                'risk_analysis', 'trade_analysis', 'portfolio_analysis', 'reporting'
            ]
        },
        stage_params={
            PipelineStage.PRE_TRAINING: pre_training_stage_params
        }
    )

def get_light_pipeline_config(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a light pipeline configuration with essential sub-pipelines only."""
    from src.config.pipeline_modes import get_light_mode_config

    # Get centralized light mode configuration
    mode_config = get_light_mode_config()

    # Light mode: Use last 10 days of available data instead of current date
    # This ensures we use actual historical data rather than future dates
    try:
        # Try to determine the last available date from the data
        from src.utils.data.klines_parquet import KlinesParquetManager
        manager = KlinesParquetManager(data_dir=data_dir)

        from datetime import datetime, timedelta

        # First, get data info to find the actual available date range without filtering
        data_info = manager.get_data_info(symbol=symbol, interval=timeframe, data_type="processed")

        if data_info and data_info.get("available") and data_info.get("date_range"):
            # Use the last date from the available data
            _, max_date = data_info["date_range"]
            end_date = pd.to_datetime(max_date)
            start_date = end_date - timedelta(days=mode_config.lookback_days)
            print(f"✅ Using last available data date: {end_date.strftime('%Y-%m-%d')}")
        else:
            # Fallback: Load without date filtering to find the actual date range
            print("⚠️ Could not get data info, loading sample to determine date range...")
            sample_data = manager.read_data(
                symbol=symbol,
                interval=timeframe,
                start_date=None,
                end_date=None,
                data_type="processed"
            )

            if sample_data is not None and not len(sample_data) == 0:
                # Get the last available date from the data
                if 'timestamp' in sample_data.columns:
                    timestamps = pd.to_datetime(sample_data['timestamp'], unit='s')
                    end_date = timestamps.max()
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
                else:
                    # Fallback to using the index if available
                    if hasattr(sample_data.index, 'max'):
                        end_date = sample_data.index.max()
                        start_date = end_date - timedelta(days=mode_config.lookback_days)
                    else:
                        # Final fallback to current date
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=mode_config.lookback_days)
                print(f"✅ Determined date range from data: {end_date.strftime('%Y-%m-%d')}")
            else:
                # No data available, use current date as fallback
                print("⚠️ No data available, using current date as fallback")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=mode_config.lookback_days)

    except Exception as e:
        # If there's any error, fall back to current date
        print(f"⚠️ Could not determine available data range: {e}")
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=mode_config.lookback_days)

    # Set intensity percentage for light mode
    intensity_pct = 0.5  # 50% intensity for light mode

    data_dir_path = Path(data_dir).expanduser()
    cache_base = data_dir_path if 'cache' in data_dir_path.name.lower() else data_dir_path.parent / 'data_cache'
    cache_base = cache_base.expanduser()
    pre_training_stage_params = {
        'regime_cache_path': str((cache_base / 'nas_tas_clustering')),
        'data_cache_dir': str(cache_base)
    }

    return MainPipelineConfig(
        mode=ExecutionMode.LIGHT,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        intensity_percentage=intensity_pct,
        training_mode_config=mode_config.__dict__,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.PRE_TRAINING,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: [
                'data_download', 'data_conversion', 'data_validation'
            ],
            PipelineStage.MARKET_ANALYSIS: [
                'sr_parameter_optimization', 'sr_detection', 'sr_clustering',
                'hmm_regime_discovery', 'hmm_clustering', 'hmm_models_training', 'hmm_ensemble_training',
                'regime_data_splitting',
                'interactive_feature_generation', 'multi_horizon_profit_labeler'
            ],
            PipelineStage.MODEL_TRAINING: [
                'analyst_model_training', 'analyst_ensemble_training', 'tactician_lookback_optimization', 'tactician_models_training', 'tactician_ensemble_training'
            ],
            PipelineStage.BACKTESTING: [
                'walk_forward_validation', 'performance_analytics'
            ]
        },
        stage_params={
            PipelineStage.PRE_TRAINING: pre_training_stage_params
        }
    )

def get_blank_pipeline_config(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "15m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a blank pipeline configuration for testing/validation."""
    from src.config.pipeline_modes import get_blank_mode_config

    # Get centralized blank mode configuration
    mode_config = get_blank_mode_config()

    # Blank mode: Use last available data instead of current date
    # This ensures we use actual historical data rather than future dates
    try:
        # Try to determine the last available date from the data
        from src.utils.data.klines_parquet import KlinesParquetManager
        manager = KlinesParquetManager(data_dir=data_dir)

        # Load a small sample of recent data to determine the date range
        # Use last 30 days to get a representative sample without loading everything
        from datetime import datetime, timedelta
        recent_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        recent_end = datetime.now().strftime('%Y-%m-%d')

        sample_data = manager.read_data(
            symbol=symbol,
            interval=timeframe,
            start_date=recent_start,
            end_date=recent_end,
            data_type="processed"
        )

        if sample_data is not None and not len(sample_data) == 0:
            # Get the last available date from the data
            if 'timestamp' in sample_data.columns:
                timestamps = pd.to_datetime(sample_data['timestamp'], unit='s')
                end_date = timestamps.max()
                start_date = end_date - timedelta(days=mode_config.lookback_days)
            else:
                # Fallback to using the index if available
                if hasattr(sample_data.index, 'max'):
                    end_date = sample_data.index.max()
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
                else:
                    # Final fallback to current date
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=mode_config.lookback_days)
        else:
            # No data available, use current date as fallback
            end_date = datetime.now()
            start_date = end_date - timedelta(days=mode_config.lookback_days)

    except Exception as e:
        # If there's any error, fall back to current date
        print(f"⚠️ Could not determine available data range: {e}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=mode_config.lookback_days)

    # Set intensity percentage for blank mode
    intensity_pct = 0.1  # 10% intensity for blank mode

    data_dir_path = Path(data_dir).expanduser()
    cache_base = data_dir_path if 'cache' in data_dir_path.name.lower() else data_dir_path.parent / 'data_cache'
    cache_base = cache_base.expanduser()
    pre_training_stage_params = {
        'regime_cache_path': str((cache_base / 'nas_tas_clustering')),
        'data_cache_dir': str(cache_base)
    }

    return MainPipelineConfig(
        mode=ExecutionMode.BLANK,
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        intensity_percentage=intensity_pct,
        training_mode_config=mode_config.__dict__,
        enabled_stages=[
            PipelineStage.DATA_COLLECTION,
            PipelineStage.MARKET_ANALYSIS,
            PipelineStage.PRE_TRAINING,
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: ['data_download', 'data_conversion'],
            PipelineStage.MARKET_ANALYSIS: ['sr_parameter_optimization', 'sr_detection', 'hmm_regime_discovery', 'hmm_models_training', 'hmm_ensemble_training'],
            PipelineStage.MODEL_TRAINING: ['analyst_model_training'],
            PipelineStage.BACKTESTING: ['walk_forward_validation']
        },
        stage_params={
            PipelineStage.PRE_TRAINING: pre_training_stage_params
        }
    )
