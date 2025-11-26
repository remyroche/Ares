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
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int,
    get_memory_usage, optimize_dataframe_memory, memory_checkpoint
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.hardware.m1_memory_optimizer import optimize_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
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
                model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST, ModelType.NGBOOST, ModelType.DEPTHWISE_CNN],  # Removed DEPTHWISE_CNN (R²≈0, not suitable for tabular data)
                timeframe=self.config.timeframe,
                symbol=self.config.symbol,
                enable_ensemble=False,  # Individual models only
                enable_hyperparameter_optimization=False,  # Enable HPO (we'll control when it's run)
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
                enable_hyperparameter_optimization=False,
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
            skip_base = self.config.custom_params.get('skip_base_training', False)
            if self.config.enable_analyst and not skip_base:
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
            elif skip_base and self.config.enable_analyst:
                tprint_info("⏭️ Skipping base training, loading existing artifacts for ensemble...")
                # Load base models from artifacts for ensemble training
                try:
                    from src.utils.ml_common.artifact_manager import get_artifact_manager
                    artifact_mgr = get_artifact_manager()
                    # Try to load analyst base outputs (predictions from base models)
                    base_outputs = artifact_mgr.load_artifact('analyst_base_outputs', artifact_type='data')
                    if base_outputs is not None:
                        artifacts['analyst_base_models'] = {'loaded_from_artifacts': True}  # Dummy models dict
                        artifacts['analyst_predictions'] = base_outputs
                        tprint_success(f"✅ Loaded base model outputs: {base_outputs.shape}")
                    else:
                        tprint_warning("⚠️ Could not load analyst_base_outputs, ensemble may not have base features")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to load base artifacts: {e}")
            
            # Phase 2: Analyst Ensemble Training (uses Analyst base models)
            if self.config.enable_analyst and self.config.enable_ensemble and artifacts['analyst_base_models'] is not None:
                tprint_info("🎯 Phase 2: Training Analyst ensemble model...")
                analyst_ensemble_result = await self._execute_analyst_ensemble_training(
                    data, analyst_targets, artifacts['analyst_base_models'], artifacts['analyst_predictions']
                )
                result.ensemble_result = analyst_ensemble_result  # Store ensemble result in PipelineResult
                
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
            if self.config.enable_tactician and artifacts['analyst_ensemble_model'] is not None:
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
            if self.config.enable_ensemble and artifacts['tactician_base_models'] is not None:
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

            # Generate OOF predictions using time-series splits for leakage-safe stacking
            try:
                n_splits = int(self.config.custom_params.get('oof_splits', 5)) if hasattr(self, 'config') else 5
            except Exception:
                n_splits = 5
            tprint_info(f"🧪 Generating OOF predictions with TimeSeriesSplit(n_splits={n_splits})...")
            # Disable HPO during OOF folds to avoid repeated optimization cycles
            _orig_hpo_flag = getattr(self._analyst_trainer.config, 'enable_hyperparameter_optimization', False)
            self._analyst_trainer.config.enable_hyperparameter_optimization = False
            # In BLANK mode, pre-configure smaller HPO trials for the later final pass
            if getattr(self.config, 'execution_mode', '').lower() == 'blank':
                self._analyst_trainer.config.custom_params['hpo_n_trials'] = min(
                    int(self._analyst_trainer.config.custom_params.get('hpo_n_trials', 10)), 5
                )
            oof_df = pd.DataFrame(index=data.index)
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Compute embargo gap in bars if available
            def _bars_per_day(tf: str) -> int:
                mapping = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '2h': 12, '4h': 6, '1d': 1}
                return mapping.get(str(self.config.timeframe).lower(), 96)
            embargo_days = int(self.config.custom_params.get('wf_embargo_days', 1))
            gap_bars = max(0, embargo_days * _bars_per_day(str(self.config.timeframe)))
            tprint_info(f"🔒 OOF CV setup: n_splits={n_splits}, timeframe={self.config.timeframe}, bars_per_day={_bars_per_day(str(self.config.timeframe))}, wf_embargo_days={embargo_days}, gap_bars={gap_bars}")
            if gap_bars == 0:
                tprint_warning("⚠️ OOF CV running with gap_bars=0 (no embargo). Increase embargo to reduce leakage risk.")

            fold_num = 0
            for train_idx, val_idx in tscv.split(data):
                # Apply purged gap between train and val
                if gap_bars > 0:
                    # Trim the last gap_bars from train and first gap_bars from val if possible
                    if len(train_idx) > gap_bars:
                        train_idx = train_idx[:len(train_idx)-gap_bars]
                    if len(val_idx) > gap_bars:
                        val_idx = val_idx[gap_bars:]
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue
                fold_num += 1
                tprint_info(f"   ↪ OOF fold {fold_num}/{n_splits}: train={len(train_idx)}, val={len(val_idx)} (gap={gap_bars})")
                train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
                train_targets = targets.iloc[train_idx] if targets is not None else None
                # Sanity: drop potential leak columns
                leak_cols = [c for c in train_data.columns if any(term in c.lower() for term in ['label', 'target', 'future_', 'lead_'])]
                if leak_cols:
                    tprint_info(f"   🔍 Dropping potential leakage columns from fold data: {len(leak_cols)} (e.g., {leak_cols[:5]})")
                    train_data = train_data.drop(columns=leak_cols, errors='ignore')
                    val_data = val_data.drop(columns=leak_cols, errors='ignore')
                # Train on fold-train only
                fold_result = await self._analyst_trainer.train(train_data, train_targets)
                if not getattr(fold_result, 'success', False):
                    tprint_warning(f"⚠️ OOF fold {fold_num} training failed, skipping fold")
                    continue
                # Determine models to predict
                models_to_predict = {}
                if hasattr(fold_result, 'metadata') and 'trained_models' in fold_result.metadata:
                    models_to_predict = fold_result.metadata['trained_models']
                elif hasattr(fold_result, 'metadata') and 'model_instances' in fold_result.metadata:
                    models_to_predict = fold_result.metadata['model_instances']
                elif hasattr(fold_result, 'models') and fold_result.models is not None:
                    models_to_predict = fold_result.models
                else:
                    models_to_predict = {'best_model': getattr(fold_result, 'model', None)}
                # Select trained features if available
                data_for_prediction = val_data
                if hasattr(fold_result, 'metadata') and 'trained_feature_columns' in fold_result.metadata:
                    trained_features = fold_result.metadata['trained_feature_columns']
                    missing = set(trained_features) - set(data_for_prediction.columns)
                    if missing:
                        tprint_warning(f"⚠️ OOF fold {fold_num}: missing {len(missing)} trained features; skipping fold predictions")
                        continue
                    data_for_prediction = data_for_prediction[trained_features]
                # Predict on fold-val
                fold_preds = await self._generate_predictions(models_to_predict, data_for_prediction)
                if fold_preds is None or fold_preds.empty:
                    tprint_warning(f"⚠️ OOF fold {fold_num}: no predictions generated")
                    continue
                # Allocate columns in oof_df as needed and fill fold indices
                for col in fold_preds.columns:
                    if col not in oof_df.columns:
                        oof_df[col] = np.nan
                    oof_df.iloc[val_idx, oof_df.columns.get_loc(col)] = fold_preds[col].values
            # End OOF loop
            if not oof_df.empty and oof_df.isna().any().any():
                filled = oof_df.notna().sum().min()
                tprint_info(f"   OOF coverage: min non-NaN count per column = {filled}")
            # If walk-forward config is available, compute honest fold-aggregated metrics
            wf_cfg = self.config.custom_params.get('walkforward_config') if hasattr(self.config, 'custom_params') else None
            walkforward_metrics = []
            if wf_cfg is not None and hasattr(wf_cfg, 'folds') and len(wf_cfg.folds) > 0:
                import pandas as _pd
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                embargo_days = int(self.config.custom_params.get('wf_embargo_days', 1))
                def _bars_per_day(tf: str) -> int:
                    mapping = {'1m': 1440, '5m': 288, '15m': 96, '30m': 48, '1h': 24, '2h': 12, '4h': 6, '1d': 1}
                    return mapping.get(str(self.config.timeframe).lower(), 96)
                gap_bars = max(0, embargo_days * _bars_per_day(str(self.config.timeframe)))

                for fold in wf_cfg.folds:
                    tr_start, tr_end = fold.training.start, fold.training.effective_end
                    va_start, va_end = fold.validation.start, fold.validation.effective_end
                    train_mask = (data.index >= _pd.Timestamp(tr_start)) & (data.index <= _pd.Timestamp(tr_end))
                    val_mask = (data.index >= _pd.Timestamp(va_start)) & (data.index <= _pd.Timestamp(va_end))
                    train_idx = data.index[train_mask]
                    val_idx = data.index[val_mask]
                    if len(train_idx) == 0 or len(val_idx) == 0:
                        continue
                    # Apply gap by trimming trailing/leading bars
                    if gap_bars > 0:
                        if len(train_idx) > gap_bars:
                            train_idx = train_idx[:-gap_bars]
                        if len(val_idx) > gap_bars:
                            val_idx = val_idx[gap_bars:]
                    if len(train_idx) == 0 or len(val_idx) == 0:
                        continue
                    X_tr = data.loc[train_idx]
                    X_va = data.loc[val_idx]
                    y_tr = targets.loc[train_idx] if targets is not None else None
                    # Sanity: drop potential leak columns
                    leak_cols = [c for c in X_tr.columns if any(term in c.lower() for term in ['label', 'target', 'future_', 'lead_'])]
                    if leak_cols:
                        X_tr = X_tr.drop(columns=leak_cols, errors='ignore')
                        X_va = X_va.drop(columns=leak_cols, errors='ignore')
                    fold_result = await self._analyst_trainer.train(X_tr, y_tr)
                    if not getattr(fold_result, 'success', False):
                        continue
                    models_to_predict = {}
                    if hasattr(fold_result, 'metadata') and 'trained_models' in fold_result.metadata:
                        models_to_predict = fold_result.metadata['trained_models']
                    elif hasattr(fold_result, 'metadata') and 'model_instances' in fold_result.metadata:
                        models_to_predict = fold_result.metadata['model_instances']
                    elif hasattr(fold_result, 'models') and fold_result.models is not None:
                        models_to_predict = fold_result.models
                    else:
                        models_to_predict = {'best_model': getattr(fold_result, 'model', None)}
                    X_pred = X_va
                    if hasattr(fold_result, 'metadata') and 'trained_feature_columns' in fold_result.metadata:
                        trained_features = fold_result.metadata['trained_feature_columns']
                        if set(trained_features).issubset(set(X_pred.columns)):
                            X_pred = X_pred[trained_features]
                    y_va = targets.loc[val_idx] if targets is not None else None
                    preds = await self._generate_predictions(models_to_predict, X_pred)
                    if preds is None or preds.empty or y_va is None:
                        continue
                    # Use average across model columns for fold metric
                    y_hat = preds.mean(axis=1).loc[y_va.index]
                    fold_metrics = {
                        'val_mse': mean_squared_error(y_va, y_hat),
                        'val_mae': mean_absolute_error(y_va, y_hat),
                        'val_r2': r2_score(y_va, y_hat),
                        'n_train': len(X_tr),
                        'n_val': len(X_va),
                    }
                    walkforward_metrics.append(fold_metrics)
                # Aggregate
                if walkforward_metrics:
                    import numpy as _np
                    agg = {
                        'wf_val_mse_mean': float(_np.mean([m['val_mse'] for m in walkforward_metrics])),
                        'wf_val_mae_mean': float(_np.mean([m['val_mae'] for m in walkforward_metrics])),
                        'wf_val_r2_mean': float(_np.mean([m['val_r2'] for m in walkforward_metrics])),
                        'wf_folds': len(walkforward_metrics),
                    }
                else:
                    agg = None
            else:
                agg = None

            # Train analyst base models on full data after OOF generation
            # Re-enable HPO for the single final training pass
            self._analyst_trainer.config.enable_hyperparameter_optimization = _orig_hpo_flag or True
            result = await self._analyst_trainer.train(data, targets)

            if result.success:
                # Attach walk-forward metrics if available
                if agg is not None:
                    if not hasattr(result, 'metrics') or result.metrics is None:
                        result.metrics = {}
                    result.metrics.update(agg)
                # CRITICAL: Models were trained on 104 features (60 base + analyst engineered features)
                # We need to use the SAME feature set for predictions
                # The training data flow is: base features (60) -> analyst feature engineering -> 104 features
                # So for prediction, we need to apply the same analyst feature engineering
                
                # Use the original 60 selected features as input
                data_for_prediction = data
                
                # Get all trained models for per-model predictions
                models_to_predict = {}
                if hasattr(result, 'metadata') and 'trained_models' in result.metadata:
                    models_to_predict = result.metadata['trained_models']
                elif hasattr(result, 'metadata') and 'model_instances' in result.metadata:
                    models_to_predict = result.metadata['model_instances']
                else:
                    # Fallback to single best model
                    models_to_predict = {'best_model': result.model}
                
                # CRITICAL: Use the EXACT same features that models were trained on
                # Models expect: base features (~60) + regime probabilities (3-7) = ~63-67 features
                # We should NOT apply analyst feature engineering here - that happens inside model_trainer
                # We just need to ensure we have the same features that were used for training
                
                tprint_info("🔧 Preparing prediction features (base + regime probabilities)...")
                tprint_info(f"   Input data shape: {data_for_prediction.shape}")
                tprint_info(f"   Input columns (first 10): {list(data_for_prediction.columns[:10])}")
                tprint_info(f"   Input columns (last 10): {list(data_for_prediction.columns[-10:])}")
                
                # The data_for_prediction should already have regime probabilities if they were loaded
                # Just verify we have reasonable feature count
                expected_min = 63  # ~60 base + 3 regime
                expected_max = 67  # ~60 base + 7 regime
                if data_for_prediction.shape[1] < expected_min:
                    tprint_warning(f"⚠️ Expected {expected_min}-{expected_max} features, got {data_for_prediction.shape[1]}")
                    tprint_warning("   This may cause prediction failures for some models")
                elif data_for_prediction.shape[1] > expected_max:
                    tprint_warning(f"⚠️ More features than expected: {data_for_prediction.shape[1]} > {expected_max}")
                else:
                    tprint_success(f"✅ Prediction data has {data_for_prediction.shape[1]} features (within expected range)")
                
                # Step 2: Select ONLY the features that models were trained on (from metadata)
                if hasattr(result, 'metadata') and 'trained_feature_columns' in result.metadata:
                    trained_features = result.metadata['trained_feature_columns']
                    tprint_info(f"📊 Using stored feature columns from training: {len(trained_features)} features")
                    
                    # Ensure all required features are available
                    missing_features = set(trained_features) - set(data_for_prediction.columns)
                    if missing_features:
                        tprint_error(f"❌ Missing required features ({len(missing_features)}): {list(missing_features)[:10]}...")
                        tprint_error("   Cannot generate predictions without all training features!")
                        predictions = None
                    else:
                        # Select ONLY the features used during training
                        data_for_prediction = data_for_prediction[trained_features]
                        tprint_success(f"✅ Selected {len(trained_features)} features for prediction (matches training)")
                        predictions = await self._generate_predictions(models_to_predict, data_for_prediction)
                else:
                    # Fallback: try to infer from model
                    tprint_warning("⚠️ No stored feature columns found, trying to infer from models...")
                    for model_name, model_obj in models_to_predict.items():
                        if hasattr(model_obj, 'feature_names_in_'):
                            expected_features = list(model_obj.feature_names_in_)
                            tprint_info(f"📊 {model_name} expects {len(expected_features)} features")
                            
                            missing_features = set(expected_features) - set(data_for_prediction.columns)
                            if missing_features:
                                tprint_error(f"❌ Missing features: {missing_features}")
                                predictions = None
                            else:
                                data_for_prediction = data_for_prediction[expected_features]
                                tprint_success(f"✅ Selected {len(expected_features)} features for prediction")
                                predictions = await self._generate_predictions(models_to_predict, data_for_prediction)
                            break
                        else:
                            tprint_error(f"❌ Model {model_name} has no feature_names_in_ attribute")
                            predictions = None

                # Store predictions for ensemble training (will be saved by calling step)
                if predictions is not None:
                    tprint_info("=" * 80)
                    tprint_info("📊 PREDICTION GENERATION SUMMARY")
                    tprint_info("=" * 80)
                    tprint_info(f"✅ Generated predictions from {len(models_to_predict)} models")
                    tprint_info(f"   Prediction shape: {predictions.shape}")
                    tprint_info(f"   Prediction columns: {list(predictions.columns)}")
                    tprint_info(f"   Models used: {list(models_to_predict.keys())}")
                    
                    # Verify we have predictions from all models
                    if predictions.shape[1] != len(models_to_predict):
                        tprint_error(f"❌ MISMATCH: Expected {len(models_to_predict)} prediction columns, got {predictions.shape[1]}")
                        tprint_error(f"   Missing models: {set(models_to_predict.keys()) - set(predictions.columns)}")
                    else:
                        tprint_success(f"✅ All {len(models_to_predict)} models have predictions")
                    
                    # Also compute confidence scores
                    confidence = predictions.abs()
                    tprint_info(f"   Confidence shape: {confidence.shape}")
                    tprint_info("=" * 80)
                else:
                    tprint_error("❌ No predictions generated!")

                tprint_success("✅ Analyst base models trained successfully")
                # Extract models from metadata if not available as attribute
                trained_models = {}
                if hasattr(result, 'models'):
                    trained_models = result.models
                elif hasattr(result, 'metadata') and 'trained_models' in result.metadata:
                    trained_models = result.metadata['trained_models']
                elif hasattr(result, 'metadata') and 'model_instances' in result.metadata:
                    trained_models = result.metadata['model_instances']
                
                return {
                    'success': True,
                    'model': result.model,
                    'models': trained_models,
                    'predictions': predictions,
                    'confidence': confidence if predictions is not None else None,
                    'oof_predictions': oof_df if 'oof_df' in locals() and not oof_df.empty else None,
                    'metrics': result.metrics,
                    'training_time': result.training_time,
                    'validation_metrics': result.validation_metrics,
                    'feature_importance': result.feature_importance,
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

            # CRITICAL: Pass base_predictions directly to ensemble trainer
            # The ensemble trainer will use these predictions for meta-learning
            # Do NOT add them as features - they are the input to the meta-learner
            
            # Train analyst ensemble with base predictions
            result = await self._ensemble_trainer.train(data, targets, base_predictions=base_predictions)

            if result.success:
                # Generate ensemble predictions
                predictions = result.predictions if hasattr(result, 'predictions') and result.predictions is not None else await self._generate_predictions(result.model, data)

                # Note: analyst_ensemble_outputs will be saved by unified_models_training_step
                tprint_success("✅ Analyst ensemble trained successfully")
                return {
                    'success': True,
                    'model': result.model,
                    'predictions': predictions,
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

            # Prepare data with analyst ensemble predictions (NOT base model predictions)
            enhanced_data = self._enhance_data_with_predictions(data, analyst_predictions)

            # Train tactician base models
            result = await self._tactician_trainer.train(enhanced_data, targets)

            if result.success:
                # Generate predictions on training data
                predictions = result.predictions if hasattr(result, 'predictions') and result.predictions is not None else await self._generate_predictions(result.model if hasattr(result, 'model') else result.models, enhanced_data)

                # Save predictions to HDF5 as tactician_base_outputs
                if predictions is not None:
                    try:
                        from src.utils.ml_common.artifact_manager import get_artifact_manager
                        artifact_mgr = get_artifact_manager()
                        artifact_mgr.save_artifact(
                            predictions,
                            'tactician_base_outputs',
                            artifact_type='data',
                            metadata={
                                'phase': 'tactician_base',
                                'shape': predictions.shape,
                                'columns': list(predictions.columns) if hasattr(predictions, 'columns') else []
                            }
                        )
                        tprint_success(f"✅ Saved tactician_base_outputs: {predictions.shape}")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to save tactician_base_outputs: {e}")

                tprint_success("✅ Tactician base models trained successfully")
                return {
                    'success': True,
                    'models': result.models if hasattr(result, 'models') else {},
                    'predictions': predictions,
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

            # Prepare data with tactician base model predictions + disagreement features
            enhanced_data = self._enhance_data_with_predictions(data, base_predictions)

            # Add disagreement features for ensemble
            if base_predictions is not None and not base_predictions.empty:
                disagreement_features = self._calculate_disagreement_features(base_predictions)
                if disagreement_features is not None:
                    enhanced_data = pd.concat([enhanced_data, disagreement_features], axis=1)
                    tprint_info(f"📊 Added {len(disagreement_features.columns)} disagreement features")

            # Train tactician ensemble
            result = await self._ensemble_trainer.train(enhanced_data, targets)

            if result.success:
                # Generate ensemble predictions
                predictions = result.predictions if hasattr(result, 'predictions') and result.predictions is not None else await self._generate_predictions(result.model, enhanced_data)

                # Save predictions to HDF5 as tactician_ensemble_outputs
                if predictions is not None:
                    try:
                        from src.utils.ml_common.artifact_manager import get_artifact_manager
                        artifact_mgr = get_artifact_manager()
                        artifact_mgr.save_artifact(
                            predictions,
                            'tactician_ensemble_outputs',
                            artifact_type='data',
                            metadata={
                                'phase': 'tactician_ensemble',
                                'shape': predictions.shape,
                                'columns': list(predictions.columns) if hasattr(predictions, 'columns') else []
                            }
                        )
                        tprint_success(f"✅ Saved tactician_ensemble_outputs: {predictions.shape}")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to save tactician_ensemble_outputs: {e}")

                tprint_success("✅ Tactician ensemble trained successfully")
                return {
                    'success': True,
                    'model': result.model,
                    'predictions': predictions,
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

    async def _generate_predictions(self, model: Any, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate predictions from a trained model.

        Args:
            model: Trained model (can be single model or dict of models)
            data: Input data for predictions

        Returns:
            DataFrame with predictions or None if failed
        """
        try:
            import pandas as pd

            # Handle dict of models (base models)
            if isinstance(model, dict):
                tprint_info(f"🔮 Generating predictions from {len(model)} models...")
                predictions_dict = {}
                for model_name, model_obj in model.items():
                    try:
                        tprint_info(f"   → Predicting with {model_name}...")
                        if hasattr(model_obj, 'predict'):
                            pred = model_obj.predict(data)
                            predictions_dict[model_name] = pred
                            tprint_success(f"   ✅ {model_name}: {len(pred)} predictions")
                        elif hasattr(model_obj, 'predict_proba'):
                            pred = model_obj.predict_proba(data)
                            # Use probability of positive class if binary
                            if pred.ndim == 2 and pred.shape[1] == 2:
                                predictions_dict[model_name] = pred[:, 1]
                            else:
                                predictions_dict[model_name] = pred
                            tprint_success(f"   ✅ {model_name}: {len(pred)} predictions (proba)")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                        continue

                if predictions_dict:
                    result_df = pd.DataFrame(predictions_dict, index=data.index)
                    tprint_success(f"✅ Combined predictions: {result_df.shape} with columns {list(result_df.columns)}")
                    return result_df
                else:
                    tprint_error("❌ No predictions generated from any model!")
                    return None

            # Handle single model
            elif hasattr(model, 'predict'):
                pred = model.predict(data)
                return pd.DataFrame({'prediction': pred}, index=data.index)
            elif hasattr(model, 'predict_proba'):
                pred = model.predict_proba(data)
                # Use probability of positive class if binary
                if pred.ndim == 2 and pred.shape[1] == 2:
                    return pd.DataFrame({'prediction_proba': pred[:, 1]}, index=data.index)
                else:
                    return pd.DataFrame(pred, index=data.index)
            else:
                tprint_warning("⚠️ Model does not have predict or predict_proba method")
                return None

        except Exception as e:
            tprint_error(f"❌ Failed to generate predictions: {e}")
            return None

    def _calculate_disagreement_features(self, predictions: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate disagreement features from base model predictions.

        Only uses Coefficient of Variation (CV) as the disagreement metric,
        which provides normalized diversity measure across base models.

        Args:
            predictions: DataFrame with predictions from multiple base models

        Returns:
            DataFrame with CV disagreement feature
        """
        try:
            import pandas as pd
            import numpy as np

            if predictions is None or predictions.empty or len(predictions.columns) < 2:
                tprint_warning("⚠️ Need at least 2 base model predictions for disagreement features")
                return None

            disagreement_features = pd.DataFrame(index=predictions.index)

            # Coefficient of variation (normalized diversity) - ONLY disagreement metric
            std_pred = predictions.std(axis=1)
            mean_pred = predictions.mean(axis=1)
            disagreement_features['pred_cv'] = std_pred / (mean_pred.abs() + 1e-8)

            tprint_info(f"✅ Calculated {len(disagreement_features.columns)} disagreement feature (CV only)")
            return disagreement_features

        except Exception as e:
            tprint_error(f"❌ Failed to calculate disagreement features: {e}")
            return None

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
