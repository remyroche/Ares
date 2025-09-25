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
- Enhanced error handling and recovery
- Performance tracking with M1 optimization
- Artifact management with serialization utilities
- ML utilities integration (Bayesian TPE, grid search)
- Hardware optimization integration
"""

import asyncio
import json
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

# Import utilities with error handling
try:
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        safe_dataframe_operation, validate_dataframe_columns, optimize_dataframe_dtypes
    )
    from src.utils.math_validation import (
        validate_finite, validate_positive, safe_divide, safe_log, safe_sqrt,
        safe_correlation, safe_covariance, safe_mean, safe_std
    )
    from src.utils.serialization_utils import UniversalSerializer
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Some utilities not available: {e}")
    UTILS_AVAILABLE = False
    from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_info

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
    data_dir: str = "historical_data"
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
            'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring',
            'data_integration', 'data_export'
        ],
        PipelineStage.MARKET_ANALYSIS: [
            'sr_parameter_optimization', 'sr_detection', 'sr_clustering',
            'hmm_regime_discovery', 'hmm_clustering', 'hmm_models_training', 'hmm_ensemble_training',
            'regime_data_splitting', 'feature_lookback_optimization',
            'pid_based_feature_generation', 'multi_horizon_profit_labeler', 'final_feature_selection'
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
    })
    
    # Custom parameters for each stage
    stage_params: Dict[PipelineStage, Dict[str, Any]] = field(default_factory=dict)
    
    # Intensity parameters for ML training
    intensity_percentage: float = 1.0  # Default to 100% intensity
    training_mode_config: Optional[Dict[str, Any]] = None
    
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
        """Initialize the main training pipeline with enhanced error handling and utility integration."""
        try:
            tprint("🚀 Starting MainTrainingPipeline initialization...")
            self.config = config or MainPipelineConfig()
            
            # Use standardized logging if available
            if STANDARDIZED_LOGGING_AVAILABLE:
                self.logger = get_logger('MainTrainingPipeline')
            else:
                self.logger = logger.getChild('MainTrainingPipeline')
            
            # Initialize utility systems
            self.utils_available = UTILS_AVAILABLE
            self.serializer = None
            self.m1_optimizers = None
            
            if self.utils_available:
                try:
                    self._initialize_utilities()
                    tprint_success("✅ Utilities initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Utilities initialization failed: {e}")
                    self.utils_available = False
            
            # Initialize sub-pipeline managers
            self.data_collection_pipeline = DataCollectionSubPipeline() if DATA_COLLECTION_AVAILABLE else None
            self.market_analysis_pipeline = MarketAnalysisSubPipeline() if MARKET_ANALYSIS_AVAILABLE else None
            self.model_training_pipeline = ModelTrainingSubPipeline() if MODEL_TRAINING_AVAILABLE else None
            self.backtesting_pipeline = BacktestingSubPipeline() if BACKTESTING_AVAILABLE else None
            
            # Pipeline state
            self.current_stage: Optional[PipelineStage] = None
            self.pipeline_results: List[MainPipelineResult] = []
            
            tprint_success("✅ MainTrainingPipeline initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ MainTrainingPipeline initialization failed: {e}")
            raise
    
    def _initialize_utilities(self):
        """Initialize utility systems including serialization, M1 optimizers, and ML utilities."""
        try:
            # Initialize serialization utilities
            if self.utils_available:
                self.serializer = UniversalSerializer()
                tprint_info("✅ Serialization utilities initialized")
            
            # Initialize M1 optimizers
            try:
                self.m1_optimizers = integrate_with_m1_optimizers()
                if self.m1_optimizers.get('success', False):
                    tprint_success("✅ M1 optimizers integrated successfully")
                else:
                    tprint_warning("⚠️ M1 optimizers integration failed")
            except Exception as e:
                tprint_warning(f"⚠️ M1 optimizers not available: {e}")
                self.m1_optimizers = None
            
                
        except Exception as e:
            tprint_error(f"❌ Utility initialization failed: {e}")
            raise
    
    async def _create_outcome_file(self, stage: str, sub_pipeline: str, result: Any, config: MainPipelineConfig) -> str:
        """Create outcome file for stage/sub-pipeline completion with enhanced error handling."""
        try:
            # Ensure directory exists with proper error handling
            if self.utils_available:
                try:
                    ensure_directory(Path("outcomes"))
                    tprint_info("📁 Created outcomes directory")
                except Exception as e:
                    tprint_error(f"❌ Failed to create outcomes directory: {e}")
                    raise
            else:
                outcome_dir = Path("outcomes")
                if not outcome_dir.exists():
                    outcome_dir.mkdir(exist_ok=True)
                    tprint_info(f"📁 Created outcomes directory: {outcome_dir}")
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{stage}_{sub_pipeline}_outcome_{timestamp}.json"
            outcome_file = Path("outcomes") / filename
            
            # Create outcome data with validation
            outcome_data = {
                'stage': stage,
                'sub_pipeline': sub_pipeline,
                'timestamp': datetime.now().isoformat(),
                'status': result.status.value if hasattr(result, 'status') else 'completed',
                'output_files': result.output_files if hasattr(result, 'output_files') else [],
                'metadata': result.metadata if hasattr(result, 'metadata') else {},
                'artifacts': result.artifacts if hasattr(result, 'artifacts') else {},
                'config': {
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'mode': config.mode.value,
                    'intensity_percentage': config.intensity_percentage,
                    'training_mode_config': config.training_mode_config
                },
                'next_stage_requirements': self._get_next_stage_requirements(stage, sub_pipeline),
                'utility_integration': {
                    'utils_available': self.utils_available,
                    'm1_optimizers_active': self.m1_optimizers.get('success', False) if self.m1_optimizers else False,
                    'serialization_available': self.serializer is not None
                }
            }
            
            # Save outcome file with proper error handling
            if self.utils_available and self.serializer:
                try:
                    success = self.serializer.save(outcome_data, str(outcome_file), format='json')
                    if not success:
                        raise Exception("Serialization failed")
                    tprint_success(f"💾 Outcome file created with serializer: {outcome_file}")
                except Exception as e:
                    tprint_warning(f"⚠️ Serializer failed, falling back to standard JSON: {e}")
                    with open(outcome_file, 'w') as f:
                        json.dump(outcome_data, f, indent=2, default=str)
                    tprint_success(f"💾 Outcome file created with fallback: {outcome_file}")
            else:
                with open(outcome_file, 'w') as f:
                    json.dump(outcome_data, f, indent=2, default=str)
                tprint_success(f"💾 Outcome file created: {outcome_file}")
            
            self.logger.info(f"💾 Outcome file created: {outcome_file}")
            return str(outcome_file)
            
        except Exception as e:
            tprint_error(f"❌ Failed to create outcome file: {e}")
            self.logger.error(f"❌ Outcome file creation failed: {e}")
            return f"failed_outcome_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def _get_next_stage_requirements(self, current_stage: str, current_sub_pipeline: str) -> Dict[str, Any]:
        """Get requirements for the next stage/sub-pipeline."""
        requirements = {
            'required_files': [],
            'required_artifacts': [],
            'data_dependencies': []
        }
        
        # Define stage dependencies and requirements
        stage_requirements = {
            'data_collection': {
                'next_stage': 'market_analysis',
                'required_files': ['processed_data.parquet', 'data_quality_report.json', 'exported_data.parquet'],
                'required_artifacts': ['data_metadata', 'quality_metrics', 'integration_results'],
                'sub_pipelines': ['data_download', 'data_conversion', 'data_validation', 'data_preparation', 
                                'feature_engineering', 'data_quality_check', 'data_storage', 'data_monitoring',
                                'data_integration', 'data_export']
            },
            'market_analysis': {
                'next_stage': 'model_training',
                'required_files': ['sr_levels.json', 'regime_assignments.parquet', 'labels.parquet', 'features.parquet'],
                'required_artifacts': ['sr_clusters', 'regime_model', 'feature_metadata'],
                'sub_pipelines': ['sr_detection', 'sr_clustering', 'hybrid_nas_tas_regime_discovery', 'nas_tas_regime_discovery', 'nas_tas_clustering', 'nas_regime_discovery', 'nas_clustering',
                                'hmm_models_training', 'hmm_ensemble_training',
                                'feature_lookback_optimization', 'pid_based_feature_generation',
                                'multi_horizon_profit_labeler', 'triple_barrier_labeling',
                                'sr_feature_integration']
            },
            'model_training': {
                'next_stage': 'backtesting',
                'required_files': ['trained_models.pkl', 'validation_results.json', 'evaluation_results.json'],
                'required_artifacts': ['model_metadata', 'performance_metrics', 'ensemble_models'],
                'sub_pipelines': ['hmm_training', 'analyst_model_training', 'analyst_ensemble_training',
                                'tactician_pre_ml_orchestration', 'tactician_dual_training',
                                'regime_specific_training', 'model_validation', 'model_persistence', 'model_evaluation']
            },
            'backtesting': {
                'next_stage': 'reporting',
                'required_files': ['backtest_results.json', 'performance_report.json', 'final_report.pdf'],
                'required_artifacts': ['trade_analysis', 'risk_metrics', 'portfolio_analysis'],
                'sub_pipelines': ['basic_backtesting_pre', 'final_parameters_optimization', 'basic_backtesting_post', 'walk_forward_validation', 'monte_carlo_simulation', 'ab_testing',
                                'model_persistence', 'performance_analytics',
                                'risk_analysis', 'trade_analysis', 'portfolio_analysis', 'reporting']
            }
        }
        
        if current_stage in stage_requirements:
            requirements.update(stage_requirements[current_stage])
        
        return requirements
    
    async def _check_outcome_files(self, stage: str, sub_pipeline: str) -> Optional[Dict[str, Any]]:
        """Check for existing outcome files from previous stages."""
        outcome_dir = Path("outcomes")
        if not outcome_dir.exists():
            return None
        
        # Look for the most recent outcome file for this stage/sub-pipeline
        pattern = f"{stage}_{sub_pipeline}_outcome_*.json"
        outcome_files = list(outcome_dir.glob(pattern))
        
        if not outcome_files:
            return None
        
        # Get the most recent file
        latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            self.logger.info(f"📂 Found existing outcome file: {latest_file}")
            return outcome_data
        except Exception as e:
            self.logger.warning(f"⚠️ Could not read outcome file {latest_file}: {e}")
            return None
    
    async def execute_pipeline(
        self,
        config: Optional[MainPipelineConfig] = None
    ) -> MainPipelineResult:
        """
        Execute the complete training pipeline with enhanced error handling and utility integration.
        
        Args:
            config: Optional configuration override
            
        Returns:
            MainPipelineResult with execution details
        """
        try:
            tprint("🚀 [EXECUTE_PIPELINE] Starting main training pipeline execution...")
            config = config or self.config
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Input validation
            if self.utils_available:
                try:
                    validate_positive(len(config.symbol), "symbol length")
                    validate_positive(len(config.exchange), "exchange length")
                    validate_positive(len(config.timeframe), "timeframe length")
                    validate_positive(len(config.data_dir), "data_dir length")
                except Exception as e:
                    tprint_warning(f"⚠️ Input validation warning: {e}")
            
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
                # Execute each enabled stage
                for stage in config.enabled_stages:
                    try:
                        tprint(f"📋 [EXECUTE_PIPELINE] Executing stage: {stage.value}")
                        if STANDARDIZED_LOGGING_AVAILABLE:
                            log_info(f"📋 Executing stage: {stage.value}")
                        else:
                            self.logger.info(f"📋 Executing stage: {stage.value}")
                        self.current_stage = stage
                        
                        stage_result = await self._execute_stage(stage, config)
                        result.stage_results[stage] = stage_result
                        
                        # Check if stage failed and handle accordingly
                        failed_sub_pipelines = [r for r in stage_result if r.status == SubPipelineStatus.FAILED]
                        if failed_sub_pipelines and config.mode != ExecutionMode.BLANK:
                            tprint_warning(f"⚠️ [EXECUTE_PIPELINE] Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                            if STANDARDIZED_LOGGING_AVAILABLE:
                                log_warning(f"⚠️ Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                            else:
                                self.logger.warning(f"⚠️ Stage {stage.value} had {len(failed_sub_pipelines)} failed sub-pipelines")
                            result.failed_stages.append(stage)
                        else:
                            tprint_success(f"✅ [EXECUTE_PIPELINE] Stage {stage.value} completed successfully")
                            
                    except Exception as e:
                        tprint_error(f"❌ [EXECUTE_PIPELINE] Stage {stage.value} execution failed: {e}")
                        if STANDARDIZED_LOGGING_AVAILABLE:
                            log_error(f"❌ Stage {stage.value} execution failed: {e}")
                        else:
                            self.logger.error(f"❌ Stage {stage.value} execution failed: {e}")
                        result.failed_stages.append(stage)
                        continue
            
                # Calculate overall metrics
                try:
                    self._calculate_pipeline_metrics(result)
                    tprint_success("✅ [EXECUTE_PIPELINE] Pipeline metrics calculated successfully")
                except Exception as e:
                    tprint_warning(f"⚠️ [EXECUTE_PIPELINE] Pipeline metrics calculation failed: {e}")
                
                # Update result status
                end_time = datetime.now()
                result.end_time = end_time
                result.duration_seconds = (end_time - start_time).total_seconds()
                
                if result.failed_sub_pipelines == 0:
                    result.status = SubPipelineStatus.COMPLETED
                    tprint_success(f"✅ [EXECUTE_PIPELINE] Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
                    if STANDARDIZED_LOGGING_AVAILABLE:
                        log_success(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
                    else:
                        self.logger.info(f"✅ Main training pipeline {pipeline_id} completed successfully in {result.duration_seconds:.2f}s")
                else:
                    result.status = SubPipelineStatus.FAILED
                    result.error_message = f"Pipeline failed with {result.failed_sub_pipelines} failed sub-pipelines"
                    tprint_error(f"❌ [EXECUTE_PIPELINE] Main training pipeline {pipeline_id} failed: {result.error_message}")
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
                
                tprint_error(f"❌ [EXECUTE_PIPELINE] Main training pipeline {pipeline_id} failed with exception: {e}")
                if STANDARDIZED_LOGGING_AVAILABLE:
                    log_error(f"❌ Main training pipeline {pipeline_id} failed with exception: {e}")
                else:
                    self.logger.error(f"❌ Main training pipeline {pipeline_id} failed with exception: {e}")
            
            self.pipeline_results.append(result)
            return result
            
        except Exception as e:
            tprint_error(f"❌ [EXECUTE_PIPELINE] Critical error in pipeline execution: {e}")
            # Create a failed result
            end_time = datetime.now()
            failed_result = MainPipelineResult(
                pipeline_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status=SubPipelineStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                error_message=str(e)
            )
            self.pipeline_results.append(failed_result)
            return failed_result
    
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
            'single_stage_only': config.single_stage_only,
            'custom_params': config.stage_params.get(stage, {})
        }
        
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
                        continue
                    # Use execute_sub_pipeline_with_next for automatic sequential execution
                    result = await self.market_analysis_pipeline.execute_sub_pipeline_with_next(sub_pipeline_name, stage_config)
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
                    self.logger.error(f"❌ Sub-pipeline {sub_pipeline_name} failed, stopping sequential execution")
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
        
        if market_data is None or market_data.empty:
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
        self.logger.info("🚀 Starting automatic sequential execution: sr_parameter_optimization -> sr_detection -> sr_clustering -> hmm_regime_discovery -> hmm_clustering -> regime_data_splitting -> feature_lookback_optimization -> pid_based_feature_generation -> multi_horizon_profit_labeler -> final_feature_selection -> cross_timeframe_analysis")

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

        self.logger.info(f"🔍 Calculating metrics from {len(result.stage_results)} stages")

        # Aggregate metrics from all stages
        for stage, stage_results in result.stage_results.items():
            self.logger.info(f"📊 Stage {stage.value}: {len(stage_results)} results")
            for sub_result in stage_results:
                total_sub_pipelines += 1
                self.logger.info(f"   Sub-pipeline: {sub_result.sub_pipeline_name}, Status: {sub_result.status.value}")
                if sub_result.status.value == "completed":
                    completed_sub_pipelines += 1
                elif sub_result.status.value == "failed":
                    failed_sub_pipelines += 1

        result.total_sub_pipelines = total_sub_pipelines
        result.completed_sub_pipelines = completed_sub_pipelines
        result.failed_sub_pipelines = failed_sub_pipelines
        result.success_rate = completed_sub_pipelines / total_sub_pipelines if total_sub_pipelines > 0 else 0

        if STANDARDIZED_LOGGING_AVAILABLE:
            log_info(f"📈 Final metrics: Total={total_sub_pipelines}, Completed={completed_sub_pipelines}, Failed={failed_sub_pipelines}, Rate={result.success_rate:.1%}")
        else:
            self.logger.info(f"📈 Final metrics: Total={total_sub_pipelines}, Completed={completed_sub_pipelines}, Failed={failed_sub_pipelines}, Rate={result.success_rate:.1%}")
        
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
                return self.market_analysis_pipeline.get_available_sub_pipelines()
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
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a full pipeline configuration with all stages and sub-pipelines enabled."""
    from src.config.pipeline_modes import get_full_mode_config
    
    # Get centralized full mode configuration
    mode_config = get_full_mode_config()
    
    # Full mode: 1460 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=mode_config.lookback_days)
    
    # Set intensity percentage for full mode
    intensity_pct = 1.0  # 100% intensity for full mode
    
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
                'hmm_regime_discovery', 'hmm_clustering', 'regime_base_training', 'regime_metamodel_training',
                'regime_data_splitting', 'feature_lookback_optimization',
                'pid_based_feature_generation', 'multi_horizon_profit_labeler', 'final_feature_selection'
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
        }
    )

def get_light_pipeline_config(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a light pipeline configuration with essential sub-pipelines only."""
    from src.config.pipeline_modes import get_light_mode_config
    
    # Get centralized light mode configuration
    mode_config = get_light_mode_config()
    
    # Light mode: 10 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=mode_config.lookback_days)
    
    # Set intensity percentage for light mode
    intensity_pct = 0.5  # 50% intensity for light mode
    
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
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: [
                'data_download', 'data_conversion', 'data_validation'
            ],
            PipelineStage.MARKET_ANALYSIS: [
                'sr_parameter_optimization', 'sr_detection', 'sr_clustering',
                'hmm_regime_discovery', 'hmm_clustering', 'regime_base_training', 'regime_metamodel_training',
                'regime_data_splitting', 'feature_lookback_optimization',
                'pid_based_feature_generation', 'multi_horizon_profit_labeler'
            ],
            PipelineStage.MODEL_TRAINING: [
                'analyst_model_training', 'analyst_ensemble_training', 'tactician_lookback_optimization', 'tactician_models_training', 'tactician_ensemble_training'
            ],
            PipelineStage.BACKTESTING: [
                'walk_forward_validation', 'performance_analytics'
            ]
        }
    )

def get_blank_pipeline_config(
    symbol: str = "ETHUSDT",
    exchange: str = "binance",
    timeframe: str = "1m",
    data_dir: str = "historical_data"
) -> MainPipelineConfig:
    """Get a blank pipeline configuration for testing/validation."""
    from src.config.pipeline_modes import get_blank_mode_config
    
    # Get centralized blank mode configuration
    mode_config = get_blank_mode_config()
    
    # Blank mode: 180 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=mode_config.lookback_days)
    
    # Set intensity percentage for blank mode
    intensity_pct = 0.1  # 10% intensity for blank mode
    
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
            PipelineStage.MODEL_TRAINING,
            PipelineStage.BACKTESTING
        ],
        enabled_sub_pipelines={
            PipelineStage.DATA_COLLECTION: ['data_download', 'data_conversion'],
            PipelineStage.MARKET_ANALYSIS: ['sr_parameter_optimization', 'sr_detection', 'hmm_regime_discovery', 'regime_base_training', 'regime_metamodel_training'],
            PipelineStage.MODEL_TRAINING: ['analyst_model_training'],
            PipelineStage.BACKTESTING: ['walk_forward_validation']
        }
    )