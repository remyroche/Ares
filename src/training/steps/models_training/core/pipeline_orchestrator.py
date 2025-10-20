"""
Training Pipeline Orchestrator - Unified Training Pipeline Management

This module provides a comprehensive orchestrator that manages the entire
training pipeline, coordinating between different roles, models, and ensemble
strategies with advanced monitoring and error handling.

Key Features:
- Unified pipeline orchestration for all training components
- Role-specific training coordination (Analyst, Tactician, Ensemble)
- Advanced pipeline monitoring and health checks
- Comprehensive error handling and recovery
- Performance optimization and resource management
- Cross-component validation and integration
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, WorkloadType, process_ml_training_data
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time

from .base_trainer import TrainingConfig, TrainingRole, ModelType
from .model_trainer import ModelTrainer
from .ensemble_trainer import EnsembleTrainer


class PipelinePhase(Enum):
    """Training pipeline phases."""
    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    ANALYST_TRAINING = "analyst_training"
    TACTICIAN_TRAINING = "tactician_training"
    ENSEMBLE_TRAINING = "ensemble_training"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    COMPLETION = "completion"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Unified pipeline configuration."""
    # Core configuration
    symbol: str = "ETHUSDT"
    timeframe: str = "15m"
    execution_mode: str = "full"  # full, light, blank
    
    # Role configuration
    enable_analyst: bool = True
    enable_tactician: bool = True
    enable_ensemble: bool = True
    
    # Training configuration
    analyst_config: Optional[Dict[str, Any]] = None
    tactician_config: Optional[Dict[str, Any]] = None
    ensemble_config: Optional[Dict[str, Any]] = None
    
    # Performance configuration
    max_parallel_tasks: int = 3
    memory_limit_mb: Optional[int] = None
    timeout_seconds: Optional[int] = None
    
    # Monitoring configuration
    enable_monitoring: bool = True
    monitoring_interval: float = 30.0
    enable_health_checks: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    status: PipelineStatus
    execution_time: float
    phases_completed: List[PipelinePhase]
    phases_failed: List[PipelinePhase]
    
    # Results by role
    analyst_result: Optional[Dict[str, Any]] = None
    tactician_result: Optional[Dict[str, Any]] = None
    ensemble_result: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingPipelineOrchestrator:
    """
    Training pipeline orchestrator.
    
    This class orchestrates the entire training pipeline, coordinating between
    different roles, models, and ensemble strategies with comprehensive
    monitoring and error handling.
    """
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or system_logger.getChild("TrainingPipelineOrchestrator")
        
        # Pipeline state
        self._pipeline_state = {
            'status': PipelineStatus.PENDING,
            'current_phase': None,
            'start_time': None,
            'end_time': None,
            'phases_completed': [],
            'phases_failed': [],
            'errors': [],
            'warnings': []
        }
        
        # Component instances
        self._analyst_trainer = None
        self._tactician_trainer = None
        self._ensemble_trainer = None
        
        # Performance tracking
        self._performance_metrics = {
            'total_execution_time': 0.0,
            'phase_times': {},
            'memory_usage': {},
            'cpu_usage': {},
            'error_counts': {}
        }
        
        # Monitoring
        self._monitoring_task = None
        self._health_check_interval = config.monitoring_interval
        
        self.logger.info(f"Initialized TrainingPipelineOrchestrator for {config.symbol}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=PipelineResult(
            success=False, 
            status=PipelineStatus.FAILED,
            execution_time=0.0,
            phases_completed=[],
            phases_failed=[PipelinePhase.INITIALIZATION],
            errors=["Pipeline initialization failed"]
        ),
        context="pipeline execution"
    )
    async def execute_pipeline(
        self, 
        data: pd.DataFrame, 
        analyst_targets: Optional[pd.Series] = None,
        tactician_targets: Optional[pd.Series] = None
    ) -> PipelineResult:
        """
        Execute the complete training pipeline.
        
        Args:
            data: Training data
            analyst_targets: Analyst target variables
            tactician_targets: Tactician target variables
            
        Returns:
            Pipeline execution result
        """
        try:
            self.logger.info("🚀 Starting training pipeline execution...")
            start_time = time.time()
            
            # Initialize pipeline
            if not await self._initialize_pipeline():
                return self._create_failure_result("Pipeline initialization failed", start_time)
            
            # Execute phases
            result = await self._execute_phases(data, analyst_targets, tactician_targets)
            
            # Finalize pipeline
            await self._finalize_pipeline()
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            if result.success:
                self.logger.info(f"✅ Pipeline completed successfully in {execution_time:.2f}s")
                tprint_success(f"Training pipeline completed for {self.config.symbol}")
            else:
                self.logger.error(f"❌ Pipeline failed after {execution_time:.2f}s")
                tprint_error(f"Training pipeline failed for {self.config.symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return self._create_failure_result(f"Pipeline execution failed: {e}", start_time)
    
    async def _initialize_pipeline(self) -> bool:
        """Initialize the training pipeline."""
        try:
            self.logger.info("🔧 Initializing training pipeline...")
            
            # Update state
            self._pipeline_state['status'] = PipelineStatus.RUNNING
            self._pipeline_state['start_time'] = time.time()
            self._pipeline_state['current_phase'] = PipelinePhase.INITIALIZATION
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            # Initialize components based on configuration
            if self.config.enable_analyst:
                self._analyst_trainer = await self._create_analyst_trainer()
                if self._analyst_trainer is None:
                    self.logger.error("Failed to create analyst trainer")
                    return False
            
            if self.config.enable_tactician:
                self._tactician_trainer = await self._create_tactician_trainer()
                if self._tactician_trainer is None:
                    self.logger.error("Failed to create tactician trainer")
                    return False
            
            if self.config.enable_ensemble:
                self._ensemble_trainer = await self._create_ensemble_trainer()
                if self._ensemble_trainer is None:
                    self.logger.error("Failed to create ensemble trainer")
                    return False
            
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            # Mark initialization complete
            self._pipeline_state['phases_completed'].append(PipelinePhase.INITIALIZATION)
            
            self.logger.info("✅ Pipeline initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate pipeline configuration."""
        try:
            # Validate required fields
            if not self.config.symbol:
                self.logger.error("Symbol is required")
                return False
            
            if not self.config.timeframe:
                self.logger.error("Timeframe is required")
                return False
            
            # Validate role configuration
            if not any([self.config.enable_analyst, self.config.enable_tactician, self.config.enable_ensemble]):
                self.logger.error("At least one role must be enabled")
                return False
            
            # Validate performance configuration
            if self.config.max_parallel_tasks < 1:
                self.logger.error("Max parallel tasks must be at least 1")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _create_analyst_trainer(self) -> Optional[ModelTrainer]:
        """Create analyst trainer."""
        try:
            # Default analyst configuration
            analyst_config = TrainingConfig(
                role=TrainingRole.ANALYST,
                model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST],
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                enable_ensemble=False,  # Individual models only
                custom_params=self.config.analyst_config or {}
            )
            
            # Merge with custom configuration
            if self.config.analyst_config:
                analyst_config.custom_params.update(self.config.analyst_config)
            
            trainer = ModelTrainer(analyst_config, self.logger)
            
            # Initialize trainer
            if await trainer.initialize():
                self.logger.info("✅ Analyst trainer created successfully")
                return trainer
            else:
                self.logger.error("Failed to initialize analyst trainer")
                return None
                
        except Exception as e:
            self.logger.error(f"Analyst trainer creation failed: {e}")
            return None
    
    async def _create_tactician_trainer(self) -> Optional[ModelTrainer]:
        """Create tactician trainer."""
        try:
            # Default tactician configuration
            tactician_config = TrainingConfig(
                role=TrainingRole.TACTICIAN,
                model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST, ModelType.NEURAL_NETWORK],
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                enable_ensemble=False,  # Individual models only
                custom_params=self.config.tactician_config or {}
            )
            
            # Merge with custom configuration
            if self.config.tactician_config:
                tactician_config.custom_params.update(self.config.tactician_config)
            
            trainer = ModelTrainer(tactician_config, self.logger)
            
            # Initialize trainer
            if await trainer.initialize():
                self.logger.info("✅ Tactician trainer created successfully")
                return trainer
            else:
                self.logger.error("Failed to initialize tactician trainer")
                return None
                
        except Exception as e:
            self.logger.error(f"Tactician trainer creation failed: {e}")
            return None
    
    async def _create_ensemble_trainer(self) -> Optional[EnsembleTrainer]:
        """Create ensemble trainer."""
        try:
            # Default ensemble configuration
            ensemble_config = TrainingConfig(
                role=TrainingRole.ENSEMBLE,
                model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST],
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                enable_ensemble=True,
                custom_params=self.config.ensemble_config or {}
            )
            
            # Merge with custom configuration
            if self.config.ensemble_config:
                ensemble_config.custom_params.update(self.config.ensemble_config)
            
            trainer = EnsembleTrainer(ensemble_config, self.logger)
            
            # Initialize trainer
            if await trainer.initialize():
                self.logger.info("✅ Ensemble trainer created successfully")
                return trainer
            else:
                self.logger.error("Failed to initialize ensemble trainer")
                return None
                
        except Exception as e:
            self.logger.error(f"Ensemble trainer creation failed: {e}")
            return None
    
    async def _execute_phases(
        self, 
        data: pd.DataFrame, 
        analyst_targets: Optional[pd.Series],
        tactician_targets: Optional[pd.Series]
    ) -> PipelineResult:
        """Execute all pipeline phases with proper artifact chaining."""
        try:
            tprint_info("🔄 Executing pipeline phases with artifact chaining...")
            
            # Initialize artifact storage for chaining
            artifacts = {
                'analyst_base_models': None,
                'analyst_ensemble_model': None,
                'tactician_base_models': None,
                'tactician_ensemble_model': None,
                'analyst_predictions': None,
                'tactician_predictions': None
            }
            
            # Create result object
            result = PipelineResult(
                success=True,
                status=PipelineStatus.RUNNING,
                execution_time=0.0,
                phases_completed=[],
                phases_failed=[],
                analyst_result=None,
                tactician_result=None,
                ensemble_result=None,
                performance_metrics={},
                errors=[],
                warnings=[]
            )
            
            # Phase 1: Analyst Base Models Training
            if self.config.enable_analyst:
                tprint_info("🎯 Phase 1: Training Analyst base models...")
                analyst_base_result = await self._execute_analyst_base_training(data, analyst_targets)
                result.analyst_result = analyst_base_result
                
                if analyst_base_result.get('success', False):
                    artifacts['analyst_base_models'] = analyst_base_result.get('models', {})
                    artifacts['analyst_predictions'] = analyst_base_result.get('predictions', None)
                    result.phases_completed.append(PipelinePhase.ANALYST_TRAINING)
                    tprint_success("✅ Analyst base models trained successfully")
                else:
                    result.phases_failed.append(PipelinePhase.ANALYST_TRAINING)
                    result.errors.append("Analyst base models training failed")
                    tprint_error("❌ Analyst base models training failed")
            
            # Phase 2: Analyst Ensemble Training (uses Analyst base models)
            if self.config.enable_analyst and artifacts['analyst_base_models']:
                tprint_info("🎯 Phase 2: Training Analyst ensemble model...")
                analyst_ensemble_result = await self._execute_analyst_ensemble_training(
                    data, analyst_targets, artifacts['analyst_base_models'], artifacts['analyst_predictions']
                )
                
                if analyst_ensemble_result.get('success', False):
                    artifacts['analyst_ensemble_model'] = analyst_ensemble_result.get('model', None)
                    artifacts['analyst_predictions'] = analyst_ensemble_result.get('predictions', None)
                    result.phases_completed.append(PipelinePhase.ENSEMBLE_TRAINING)
                    tprint_success("✅ Analyst ensemble model trained successfully")
                else:
                    result.phases_failed.append(PipelinePhase.ENSEMBLE_TRAINING)
                    result.errors.append("Analyst ensemble training failed")
                    tprint_error("❌ Analyst ensemble training failed")
            
            # Phase 3: Tactician Base Models Training (uses Analyst ensemble outputs)
            if self.config.enable_tactician and artifacts['analyst_ensemble_model']:
                tprint_info("🎯 Phase 3: Training Tactician base models...")
                tactician_base_result = await self._execute_tactician_base_training(
                    data, tactician_targets, artifacts['analyst_predictions']
                )
                result.tactician_result = tactician_base_result
                
                if tactician_base_result.get('success', False):
                    artifacts['tactician_base_models'] = tactician_base_result.get('models', {})
                    artifacts['tactician_predictions'] = tactician_base_result.get('predictions', None)
                    result.phases_completed.append(PipelinePhase.TACTICIAN_TRAINING)
                    tprint_success("✅ Tactician base models trained successfully")
                else:
                    result.phases_failed.append(PipelinePhase.TACTICIAN_TRAINING)
                    result.errors.append("Tactician base models training failed")
                    tprint_error("❌ Tactician base models training failed")
            
            # Phase 4: Tactician Ensemble Training (uses Tactician base models)
            if self.config.enable_ensemble and artifacts['tactician_base_models']:
                tprint_info("🎯 Phase 4: Training Tactician ensemble model...")
                tactician_ensemble_result = await self._execute_tactician_ensemble_training(
                    data, tactician_targets, artifacts['tactician_base_models'], artifacts['tactician_predictions']
                )
                
                if tactician_ensemble_result.get('success', False):
                    artifacts['tactician_ensemble_model'] = tactician_ensemble_result.get('model', None)
                    result.phases_completed.append(PipelinePhase.ENSEMBLE_TRAINING)
                    tprint_success("✅ Tactician ensemble model trained successfully")
                else:
                    result.phases_failed.append(PipelinePhase.ENSEMBLE_TRAINING)
                    result.errors.append("Tactician ensemble training failed")
                    tprint_error("❌ Tactician ensemble training failed")
            
            # Store artifacts in result for downstream use
            result.artifacts = artifacts
            
            # Determine overall success
            result.success = len(result.phases_failed) == 0
            result.status = PipelineStatus.COMPLETED if result.success else PipelineStatus.FAILED
            
            tprint_info(f"📊 Pipeline execution completed: {len(result.phases_completed)} phases successful, {len(result.phases_failed)} phases failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Phase execution failed: {e}")
            tprint_error(f"❌ Phase execution failed: {e}")
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                execution_time=0.0,
                phases_completed=[],
                phases_failed=[PipelinePhase.ANALYST_TRAINING, PipelinePhase.TACTICIAN_TRAINING, PipelinePhase.ENSEMBLE_TRAINING],
                errors=[f"Phase execution failed: {e}"],
                warnings=[]
            )
    
    async def _execute_analyst_base_training(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Dict[str, Any]:
        """Execute analyst base models training phase."""
        try:
            if self._analyst_trainer is None:
                raise ValueError("Analyst trainer not initialized")
            
            tprint_info("📊 Starting analyst base models training...")
            
            # Train analyst base models
            result = await self._analyst_trainer.train(data, targets)
            
            if result.success:
                tprint_success("✅ Analyst base models trained successfully")
                return {
                    'success': True,
                    'models': result.models,
                    'predictions': result.predictions,
                    'metrics': result.metrics,
                    'training_time': result.training_time,
                    'metadata': result.metadata
                }
            else:
                tprint_error(f"❌ Analyst base models training failed: {result.error_message}")
                return {'success': False, 'error_message': result.error_message}
            
        except Exception as e:
            tprint_error(f"❌ Analyst base models training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_analyst_ensemble_training(self, data: pd.DataFrame, targets: Optional[pd.Series], base_models: Dict[str, Any], base_predictions: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Execute analyst ensemble training phase using base models."""
        try:
            if self._ensemble_trainer is None:
                raise ValueError("Ensemble trainer not initialized")
            
            tprint_info("📊 Starting analyst ensemble training...")
            
            # Prepare data with base model predictions
            enhanced_data = self._enhance_data_with_predictions(data, base_predictions)
            
            # Train analyst ensemble
            result = await self._ensemble_trainer.train(enhanced_data, targets)
            
            if result.success:
                tprint_success("✅ Analyst ensemble trained successfully")
                return {
                    'success': True,
                    'model': result.model,
                    'predictions': result.predictions,
                    'metrics': result.metrics,
                    'training_time': result.training_time,
                    'metadata': result.metadata
                }
            else:
                tprint_error(f"❌ Analyst ensemble training failed: {result.error_message}")
                return {'success': False, 'error_message': result.error_message}
            
        except Exception as e:
            tprint_error(f"❌ Analyst ensemble training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_tactician_base_training(self, data: pd.DataFrame, targets: Optional[pd.Series], analyst_predictions: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Execute tactician base models training phase using analyst predictions."""
        try:
            if self._tactician_trainer is None:
                raise ValueError("Tactician trainer not initialized")
            
            tprint_info("⚔️ Starting tactician base models training...")
            
            # Prepare data with analyst predictions
            enhanced_data = self._enhance_data_with_predictions(data, analyst_predictions)
            
            # Train tactician base models
            result = await self._tactician_trainer.train(enhanced_data, targets)
            
            if result.success:
                tprint_success("✅ Tactician base models trained successfully")
                return {
                    'success': True,
                    'models': result.models,
                    'predictions': result.predictions,
                    'metrics': result.metrics,
                    'training_time': result.training_time,
                    'metadata': result.metadata
                }
            else:
                tprint_error(f"❌ Tactician base models training failed: {result.error_message}")
                return {'success': False, 'error_message': result.error_message}
            
        except Exception as e:
            tprint_error(f"❌ Tactician base models training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_tactician_ensemble_training(self, data: pd.DataFrame, targets: Optional[pd.Series], base_models: Dict[str, Any], base_predictions: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Execute tactician ensemble training phase using base models."""
        try:
            if self._ensemble_trainer is None:
                raise ValueError("Ensemble trainer not initialized")
            
            tprint_info("⚔️ Starting tactician ensemble training...")
            
            # Prepare data with base model predictions
            enhanced_data = self._enhance_data_with_predictions(data, base_predictions)
            
            # Train tactician ensemble
            result = await self._ensemble_trainer.train(enhanced_data, targets)
            
            if result.success:
                tprint_success("✅ Tactician ensemble trained successfully")
                return {
                    'success': True,
                    'model': result.model,
                    'predictions': result.predictions,
                    'metrics': result.metrics,
                    'training_time': result.training_time,
                    'metadata': result.metadata
                }
            else:
                tprint_error(f"❌ Tactician ensemble training failed: {result.error_message}")
                return {'success': False, 'error_message': result.error_message}
            
        except Exception as e:
            tprint_error(f"❌ Tactician ensemble training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    def _enhance_data_with_predictions(self, data: pd.DataFrame, predictions: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Enhance data with predictions from previous models."""
        try:
            if predictions is None or predictions.empty:
                tprint_warning("⚠️ No predictions provided, using original data")
                return data
            
            # Ensure predictions align with data index
            if not predictions.index.equals(data.index):
                tprint_warning("⚠️ Prediction index doesn't match data index, aligning...")
                predictions = predictions.reindex(data.index)
            
            # Add prediction columns to data
            enhanced_data = data.copy()
            for col in predictions.columns:
                enhanced_data[f'pred_{col}'] = predictions[col]
            
            tprint_info(f"📊 Enhanced data with {len(predictions.columns)} prediction columns")
            return enhanced_data
            
        except Exception as e:
            tprint_error(f"❌ Data enhancement failed: {e}")
            return data
    
    async def _execute_analyst_training(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Dict[str, Any]:
        """Execute analyst training phase."""
        try:
            if self._analyst_trainer is None:
                raise ValueError("Analyst trainer not initialized")
            
            self.logger.info("📊 Starting analyst training...")
            
            # Train analyst models
            result = await self._analyst_trainer.train(data, targets)
            
            if result.success:
                self.logger.info("✅ Analyst training completed successfully")
                tprint_success("Analyst models trained successfully")
            else:
                self.logger.error(f"❌ Analyst training failed: {result.error_message}")
                tprint_error(f"Analyst training failed: {result.error_message}")
            
            return {
                'success': result.success,
                'metrics': result.metrics,
                'training_time': result.training_time,
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Analyst training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_tactician_training(self, data: pd.DataFrame, targets: Optional[pd.Series]) -> Dict[str, Any]:
        """Execute tactician training phase."""
        try:
            if self._tactician_trainer is None:
                raise ValueError("Tactician trainer not initialized")
            
            self.logger.info("⚔️ Starting tactician training...")
            
            # Train tactician models
            result = await self._tactician_trainer.train(data, targets)
            
            if result.success:
                self.logger.info("✅ Tactician training completed successfully")
                tprint_success("Tactician models trained successfully")
            else:
                self.logger.error(f"❌ Tactician training failed: {result.error_message}")
                tprint_error(f"Tactician training failed: {result.error_message}")
            
            return {
                'success': result.success,
                'metrics': result.metrics,
                'training_time': result.training_time,
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Tactician training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_ensemble_training(
        self, 
        data: pd.DataFrame, 
        analyst_targets: Optional[pd.Series],
        tactician_targets: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Execute ensemble training phase."""
        try:
            if self._ensemble_trainer is None:
                raise ValueError("Ensemble trainer not initialized")
            
            self.logger.info("🎯 Starting ensemble training...")
            
            # Use analyst targets for ensemble training (primary signal)
            ensemble_targets = analyst_targets if analyst_targets is not None else tactician_targets
            
            # Train ensemble models
            result = await self._ensemble_trainer.train(data, ensemble_targets)
            
            if result.success:
                self.logger.info("✅ Ensemble training completed successfully")
                tprint_success("Ensemble models trained successfully")
            else:
                self.logger.error(f"❌ Ensemble training failed: {result.error_message}")
                tprint_error(f"Ensemble training failed: {result.error_message}")
            
            return {
                'success': result.success,
                'metrics': result.metrics,
                'training_time': result.training_time,
                'error_message': result.error_message,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble training execution failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_validation(self, data: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation phase."""
        try:
            self.logger.info("🔍 Starting validation phase...")
            
            validation_results = {}
            
            # Validate analyst models
            if self._analyst_trainer and results.get('analyst', {}).get('success', False):
                analyst_validation = await self._analyst_trainer.validate(data)
                validation_results['analyst'] = {
                    'success': analyst_validation.success,
                    'metrics': analyst_validation.metrics,
                    'error_message': analyst_validation.error_message
                }
            
            # Validate tactician models
            if self._tactician_trainer and results.get('tactician', {}).get('success', False):
                tactician_validation = await self._tactician_trainer.validate(data)
                validation_results['tactician'] = {
                    'success': tactician_validation.success,
                    'metrics': tactician_validation.metrics,
                    'error_message': tactician_validation.error_message
                }
            
            # Validate ensemble models
            if self._ensemble_trainer and results.get('ensemble', {}).get('success', False):
                ensemble_validation = await self._ensemble_trainer.validate(data)
                validation_results['ensemble'] = {
                    'success': ensemble_validation.success,
                    'metrics': ensemble_validation.metrics,
                    'error_message': ensemble_validation.error_message
                }
            
            self.logger.info("✅ Validation phase completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation phase failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_integration(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integration phase."""
        try:
            self.logger.info("🔗 Starting integration phase...")
            
            # Integration logic would go here
            # This could include model combination, cross-validation, etc.
            
            integration_results = {
                'analyst_integrated': results.get('analyst', {}).get('success', False),
                'tactician_integrated': results.get('tactician', {}).get('success', False),
                'ensemble_integrated': results.get('ensemble', {}).get('success', False)
            }
            
            self.logger.info("✅ Integration phase completed")
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Integration phase failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _execute_completion(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute completion phase."""
        try:
            self.logger.info("🏁 Starting completion phase...")
            
            # Completion logic would go here
            # This could include final model saving, report generation, etc.
            
            completion_results = {
                'pipeline_completed': True,
                'total_models_trained': sum(1 for r in results.values() if r.get('success', False)),
                'completion_time': time.time()
            }
            
            self.logger.info("✅ Completion phase finished")
            return completion_results
            
        except Exception as e:
            self.logger.error(f"Completion phase failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def _start_monitoring(self):
        """Start pipeline monitoring."""
        try:
            if self.config.enable_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
                self.logger.info("📊 Pipeline monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Pipeline monitoring loop."""
        try:
            while self._pipeline_state['status'] == PipelineStatus.RUNNING:
                await self._perform_health_check()
                await asyncio.sleep(self._health_check_interval)
        except Exception as e:
            self.logger.error(f"Monitoring loop failed: {e}")
    
    async def _perform_health_check(self):
        """Perform pipeline health check."""
        try:
            # Check memory usage
            import psutil
            memory_usage = psutil.virtual_memory().percent
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent()
            
            # Update performance metrics
            self._performance_metrics['memory_usage'][time.time()] = memory_usage
            self._performance_metrics['cpu_usage'][time.time()] = cpu_usage
            
            # Check for issues
            if memory_usage > 90:
                self.logger.warning(f"High memory usage: {memory_usage}%")
                self._pipeline_state['warnings'].append(f"High memory usage: {memory_usage}%")
            
            if cpu_usage > 95:
                self.logger.warning(f"High CPU usage: {cpu_usage}%")
                self._pipeline_state['warnings'].append(f"High CPU usage: {cpu_usage}%")
                
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    async def _finalize_pipeline(self):
        """Finalize the training pipeline."""
        try:
            # Stop monitoring
            if self._monitoring_task:
                self._monitoring_task.cancel()
            
            # Update final state
            self._pipeline_state['end_time'] = time.time()
            self._pipeline_state['status'] = PipelineStatus.COMPLETED
            
            # Calculate final metrics
            if self._pipeline_state['start_time']:
                total_time = self._pipeline_state['end_time'] - self._pipeline_state['start_time']
                self._performance_metrics['total_execution_time'] = total_time
            
            self.logger.info("✅ Pipeline finalized")
            
        except Exception as e:
            self.logger.error(f"Pipeline finalization failed: {e}")
    
    def _create_failure_result(self, error_message: str, start_time: float) -> PipelineResult:
        """Create failure result."""
        return PipelineResult(
            success=False,
            status=PipelineStatus.FAILED,
            execution_time=time.time() - start_time,
            phases_completed=self._pipeline_state['phases_completed'],
            phases_failed=self._pipeline_state['phases_failed'],
            errors=[error_message] + self._pipeline_state['errors'],
            warnings=self._pipeline_state['warnings']
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'status': self._pipeline_state['status'].value,
            'current_phase': self._pipeline_state['current_phase'].value if self._pipeline_state['current_phase'] else None,
            'phases_completed': [phase.value for phase in self._pipeline_state['phases_completed']],
            'phases_failed': [phase.value for phase in self._pipeline_state['phases_failed']],
            'execution_time': time.time() - self._pipeline_state['start_time'] if self._pipeline_state['start_time'] else 0,
            'errors': self._pipeline_state['errors'],
            'warnings': self._pipeline_state['warnings'],
            'performance_metrics': self._performance_metrics
        }
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return [
            'pandas', 'numpy', 'scikit-learn', 'lightgbm', 'catboost',
            'torch', 'psutil', 'asyncio'
        ]
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get pipeline processing capabilities."""
        return {
            'supports_parallel_processing': True,
            'max_parallel_tasks': self.config.max_parallel_tasks,
            'supports_monitoring': self.config.enable_monitoring,
            'supports_health_checks': self.config.enable_health_checks,
            'memory_efficient': True,
            'supports_ensemble': self.config.enable_ensemble
        }
