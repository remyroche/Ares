"""
Analyst Model Training + Ensemble Management

This module provides specialized model training for the Analyst system with
ensemble management capabilities, utilizing M1 optimizations and regime-specific features.
Enhanced with ML commons utilities for better integration and performance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path

# Enhanced Model Trainer - Automatic post-training integration
from src.utils.ml_common.enhanced_model_trainer import EnhancedModelTrainer, EnhancedTrainingConfig
from src.utils.ml_common.multi_timeframe_training import MultiTimeframeTrainer, MultiTimeframeTrainingConfig, TimeframeConfig

# Import ensemble manager and general training
from src.utils.ml_common.ensemble_manager import EnsembleManager, EnsembleConfig, EnsembleType
from .general_model_training import GeneralModelTrainer, ModelTrainingConfig, ModelType, TaskType

# ML Commons utilities - Enhanced integration
from src.utils.ml_common import (
    ModelEvaluator, HPOptimizer, FeatureSelectionFramework,
    DataLabelingUtilities, MemoryEfficientTraining, 
    ParallelProcessingCoordinator, ModelRegistry,
    DataQualityUtilities, CrossValidationUtilities,
    LookaheadProtection, MLTrainingSafeguards,
    HMMRegimeDetector, RegimeDataProcessor
)

# M1 Optimization imports
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials, 
    get_scaled_hpo_timeout, log_intensity_info, apply_intensity_scaling
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class AnalystModelType(Enum):
    """Types of Analyst models."""
    REGIME_CLASSIFIER = "regime_classifier"
    SIGNAL_PREDICTOR = "signal_predictor"
    CONFIDENCE_ESTIMATOR = "confidence_estimator"
    RISK_ASSESSOR = "risk_assessor"
    META_LABELER = "meta_labeler"


@dataclass
class AnalystTrainingConfig:
    """Configuration for Analyst model training."""
    # Basic configuration
    analyst_name: str
    output_dir: str
    
    # Model configuration
    model_types: List[AnalystModelType] = field(default_factory=lambda: [
        AnalystModelType.REGIME_CLASSIFIER,
        AnalystModelType.SIGNAL_PREDICTOR,
        AnalystModelType.CONFIDENCE_ESTIMATOR
    ])
    
    # Ensemble configuration
    enable_ensemble: bool = True
    ensemble_type: EnsembleType = EnsembleType.STACKING
    max_ensemble_models: int = 5
    
    # Regime-specific configuration (analyst models are inherently regime-specific)
    regime_columns: List[str] = field(default_factory=lambda: ['hmm_cluster', 'regime_id'])
    
    # Feature configuration
    feature_columns: List[str] = field(default_factory=list)
    target_columns: Dict[str, str] = field(default_factory=lambda: {
        'regime_classifier': 'regime_label',
        'signal_predictor': 'signal',
        'confidence_estimator': 'confidence_score',
        'risk_assessor': 'risk_score',
        'meta_labeler': 'meta_label'
    })
    
    # Training configuration
    enable_hyperparameter_optimization: bool = True
    hpo_trials: int = 50  # Reduced for multiple models
    hpo_timeout: int = 1800  # 30 minutes per model
    hpo_sampler: str = "TPE"  # TPE, Random, CMA-ES
    hpo_pruner: str = "MedianPruner"  # MedianPruner, PercentilePruner, SuccessiveHalvingPruner
    hpo_strategy: str = "coarse_first_tpe"  # Always use coarse-first then Full TPE
    launch_mode: str = "full"  # full, blank, light - determines HPO intensity
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Apply intensity scaling and launch mode scaling after initialization."""
        # Apply launch mode scaling first
        self._apply_launch_mode_scaling()
        
        # Then apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.hpo_trials = get_scaled_hpo_trials(self.hpo_trials, intensity_pct)
            self.hpo_timeout = get_scaled_hpo_timeout(self.hpo_timeout, intensity_pct)
            self.early_stopping_patience = max(1, int(self.early_stopping_patience * intensity_pct))
            logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%): HPO trials={self.hpo_trials}")
    
    def _apply_launch_mode_scaling(self):
        """Apply launch mode scaling to HPO parameters."""
        
        if self.launch_mode == "light":
            # Light mode: Minimal HPO for quick testing
            self.hpo_trials = 10
            self.hpo_timeout = 300  # 5 minutes
            self.early_stopping_patience = 3
            logger.info("🚀 Light mode: Minimal HPO (10 trials, 5 min)")
            
        elif self.launch_mode == "blank":
            # Blank mode: Moderate HPO for development
            self.hpo_trials = 25
            self.hpo_timeout = 900  # 15 minutes
            self.early_stopping_patience = 5
            logger.info("🔧 Blank mode: Moderate HPO (25 trials, 15 min)")
            
        elif self.launch_mode == "full":
            # Full mode: Comprehensive HPO for production
            self.hpo_trials = 50
            self.hpo_timeout = 1800  # 30 minutes
            self.early_stopping_patience = 8
            logger.info("🎯 Full mode: Comprehensive HPO (50 trials, 30 min)")
            
        else:
            # Default to full mode if unknown
            self.launch_mode = "full"
            self.hpo_trials = 50
            self.hpo_timeout = 1800
            self.early_stopping_patience = 8
            logger.warning(f"⚠️ Unknown launch mode '{self.launch_mode}', defaulting to full mode")
        
        logger.info(f"📊 Launch mode '{self.launch_mode}': HPO trials={self.hpo_trials}, timeout={self.hpo_timeout}s")
    
    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 3  # Reduced for multiple models
    
    # Output settings
    save_models: bool = True
    save_ensemble: bool = True
    save_predictions: bool = True
    generate_reports: bool = True


@dataclass
class AnalystTrainingResults:
    """Results from Analyst model training."""
    # Basic info
    analyst_name: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Individual model results
    individual_models: Dict[str, Any] = field(default_factory=dict)
    model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Ensemble results
    ensemble_manager: Optional[EnsembleManager] = None
    ensemble_performance: Dict[str, float] = field(default_factory=dict)
    
    # Regime-specific results
    regime_specific_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall performance
    overall_performance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    config: AnalystTrainingConfig = field(default_factory=AnalystTrainingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class AnalystModelTrainer:
    """Analyst model trainer with ensemble management and ML commons integration."""
    
    def __init__(self, config: AnalystTrainingConfig):
        """Initialize Analyst model trainer."""
        self.config = config
        self.logger = logger.getChild('AnalystModelTrainer')
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize ML Commons utilities
        self.model_evaluator = ModelEvaluator()
        self.hpo_optimizer = HPOptimizer()
        self.feature_selector = FeatureSelectionFramework()
        self.data_labeler = DataLabelingUtilities()
        self.memory_efficient_training = MemoryEfficientTraining()
        self.parallel_coordinator = ParallelProcessingCoordinator()
        self.model_registry = ModelRegistry()
        self.data_quality = DataQualityUtilities()
        self.cv_utils = CrossValidationUtilities()
        self.lookahead_protection = LookaheadProtection()
        self.ml_safeguards = MLTrainingSafeguards()
        self.hmm_regime_detector = HMMRegimeDetector()
        self.regime_processor = RegimeDataProcessor()
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        # Initialize ensemble manager if enabled
        self.ensemble_manager = None
        if config.enable_ensemble:
            ensemble_config = EnsembleConfig(
                ensemble_name=f"{config.analyst_name}_ensemble",
                output_dir=f"{config.output_dir}/ensemble",
                ensemble_type=config.ensemble_type,
                max_models=config.max_ensemble_models,
                enable_gpu_acceleration=config.enable_gpu_acceleration,
                enable_memory_optimization=config.enable_memory_optimization,
                enable_parallel_processing=config.enable_parallel_processing,
                memory_limit_gb=config.memory_limit_gb,
                max_workers=config.max_workers
            )
            self.ensemble_manager = EnsembleManager(ensemble_config)
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        self.logger.info(f"🚀 AnalystModelTrainer initialized for {config.analyst_name}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎯 Model types: {[mt.value for mt in config.model_types]}")
        self.logger.info(f"🤖 Ensemble enabled: {config.enable_ensemble}")
        self.logger.info(f"🔧 ML Commons integration: Enhanced")
    
    @traced(span_name='train_analyst_models')
    @log_execution_time
    async def train_analyst_models(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> AnalystTrainingResults:
        """Train all Analyst models with ensemble management and ML commons integration."""
        
        self.logger.info("🚀 Starting enhanced Analyst model training with ML commons...")
        start_time = time.time()
        
        # Validate inputs with ML safeguards
        self._validate_data_with_safeguards(data)
        
        # Apply lookahead bias protection
        data = self.lookahead_protection.apply_protection(data)
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._train_analyst_models_internal(data, **kwargs)
        else:
            results = await self._train_analyst_models_internal(data, **kwargs)
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ Enhanced Analyst model training completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Overall performance: {results.overall_performance}")
        
        return results
    
    def _validate_data_with_safeguards(self, data: pd.DataFrame) -> None:
        """Validate input data for Analyst training using ML safeguards."""
        
        if data.empty:
            raise ValidationError("Input data is empty")
        
        # Check required columns
        missing_features = [col for col in self.config.feature_columns if col not in data.columns]
        if missing_features:
            raise ValidationError(f"Missing feature columns: {missing_features}")
        
        # Check target columns
        for model_type, target_col in self.config.target_columns.items():
            if target_col not in data.columns:
                self.logger.warning(f"⚠️ Missing target column for {model_type}: {target_col}")
        
        # Check regime columns (analyst models are inherently regime-specific)
        missing_regime_cols = [col for col in self.config.regime_columns if col not in data.columns]
        if missing_regime_cols:
            self.logger.warning(f"⚠️ Missing regime columns: {missing_regime_cols}")
        
        # Check for sufficient data
        if len(data) < 100:
            raise ValidationError(f"Insufficient data: {len(data)} < 100")
        
        # Use ML safeguards for advanced validation
        try:
            self.ml_safeguards.validate_training_data(data, list(self.config.target_columns.values())[0])
            self.logger.info("✅ ML safeguards validation passed")
        except Exception as e:
            self.logger.warning(f"⚠️ ML safeguards validation warning: {e}")
        
        # Data quality assessment
        quality_score = self.data_quality.calculate_data_quality_score(data)
        self.logger.info(f"📊 Data quality score: {quality_score:.2f}")
        
        if quality_score < 0.7:
            self.logger.warning("⚠️ Low data quality score detected")
    
    async def _train_analyst_models_internal(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> AnalystTrainingResults:
        """Internal Analyst model training logic using EnhancedModelTrainer for automatic post-training integration."""
        
        results = AnalystTrainingResults(
            analyst_name=self.config.analyst_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        # Prepare multi-timeframe data if available
        multi_timeframe_data = self._prepare_multi_timeframe_data(data)
        
        # Train individual models with EnhancedModelTrainer
        individual_results = {}
        model_performance = {}
        
        for model_type in self.config.model_types:
            self.logger.info(f"🔄 Training {model_type.value} with EnhancedModelTrainer...")
            
            try:
                # Prepare data for this model type
                model_data = self._prepare_model_data(data, model_type)
                
                # Create EnhancedTrainingConfig for analyst models
                enhanced_config = EnhancedTrainingConfig(
                    model_types=[model_type.value],
                    enable_hyperparameter_optimization=self.config.enable_hyperparameter_optimization,
                    hpo_trials=self.config.hpo_trials,
                    hpo_timeout=self.config.hpo_timeout,
                    enable_multi_timeframe_training=self.config.enable_multi_timeframe_training,
                    timeframes=self.config.timeframes,
                    timeframe_weights=self.config.timeframe_weights,
                    enable_pre_hpo_evaluation=True,
                    enable_post_hpo_evaluation=True,
                    evaluation_metrics=['accuracy', 'f1_score', 'r2_score', 'precision', 'recall', 'sharpe_ratio'],
                    enable_cross_validation=True,
                    cv_folds=self.config.cross_validation_folds,
                    enable_holdout_validation=True,
                    holdout_ratio=self.config.validation_split,
                    enable_model_persistence=True,
                    enable_versioning=True,
                    max_versions=10,
                    min_accuracy_threshold=0.5,
                    min_f1_threshold=0.5,
                    min_r2_threshold=0.0,
                    min_sharpe_threshold=0.0,
                    save_training_results=True,
                    generate_training_report=True,
                    training_report_path=f"{self.config.output_dir}/analyst_training_report_{model_type.value}.json"
                )
                
                # Initialize EnhancedModelTrainer
                enhanced_trainer = EnhancedModelTrainer(enhanced_config)
                
                # Prepare data for training
                X, y = self._prepare_features_target(model_data, model_type)
                X_train, X_test, y_train, y_test = self._split_data(X, y)
                
                # Convert to numpy arrays
                X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
                X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
                y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
                
                # Train model with automatic post-training integration
                training_result = await enhanced_trainer.train_model(
                    X_train=X_train_array,
                    y_train=y_train_array,
                    X_test=X_test_array,
                    y_test=y_test_array,
                    model_name=f"{self.config.analyst_name}_{model_type.value}",
                    model_type=model_type.value,
                    multi_timeframe_data=multi_timeframe_data
                )
                
                # Convert result to analyst format
                individual_results[model_type.value] = self._convert_enhanced_result_to_analyst_result(training_result, model_type)
                
                # Track performance
                if training_result.post_hpo_metrics and training_result.post_hpo_metrics.post_hpo_metrics:
                    model_performance[model_type.value] = training_result.post_hpo_metrics.post_hpo_metrics.get('accuracy', 0.0)
                else:
                    model_performance[model_type.value] = 0.0
                
                self.logger.info(f"✅ {model_type.value} training completed with EnhancedModelTrainer")
                
            except Exception as e:
                self.logger.error(f"❌ Error training {model_type.value}: {e}")
                individual_results[model_type.value] = None
                model_performance[model_type.value] = 0.0
        
        # Store individual results
        results.individual_results = individual_results
        results.model_performance = model_performance
        results.overall_performance = safe_mean(list(model_performance.values())) if model_performance else 0.0
        
        # Train ensemble if enabled and we have successful individual models
        if self.config.enable_ensemble_training and any(result is not None for result in individual_results.values()):
            self.logger.info("🧠 Training ensemble with EnhancedModelTrainer...")
            ensemble_result = await self._train_ensemble_with_enhanced_trainer(individual_results, data)
            results.ensemble_result = ensemble_result
            results.ensemble_manager = ensemble_result.ensemble_manager if ensemble_result else None
        
        return results
    
    def _prepare_multi_timeframe_data(self, data: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
        """Prepare multi-timeframe data for EnhancedModelTrainer."""
        try:
            if not self.config.enable_multi_timeframe_training:
                return None
            
            multi_timeframe_data = {}
            for timeframe in self.config.timeframes:
                # For now, use the same data for all timeframes
                # In a real implementation, you'd have different data for each timeframe
                multi_timeframe_data[timeframe] = data.copy()
            
            return multi_timeframe_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to prepare multi-timeframe data: {e}")
            return None
    
    def _prepare_model_data(self, data: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
        """Prepare data for a specific model type."""
        try:
            # Apply analyst-specific preprocessing
            model_data = data.copy()
            
            # Add regime-specific features if available
            if self.config.regime_columns:
                for regime_col in self.config.regime_columns:
                    if regime_col in model_data.columns:
                        # Create regime-specific features
                        model_data[f'{regime_col}_encoded'] = pd.Categorical(model_data[regime_col]).codes
            
            return model_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to prepare model data for {model_type.value}: {e}")
            return data
    
    def _prepare_features_target(self, data: pd.DataFrame, model_type: ModelType) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for a specific model type."""
        try:
            # Select features
            X = data[self.config.feature_columns].copy()
            
            # Select target for this model type
            target_col = self.config.target_columns.get(model_type.value)
            if target_col and target_col in data.columns:
                y = data[target_col].copy()
            else:
                # Fallback to first available target
                available_targets = [col for col in self.config.target_columns.values() if col in data.columns]
                if available_targets:
                    y = data[available_targets[0]].copy()
                else:
                    raise ValueError(f"No valid target column found for {model_type.value}")
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare features/target for {model_type.value}: {e}")
            raise
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train/test sets."""
        try:
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_split,
                random_state=42,
                stratify=y if len(y.unique()) < 10 else None
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"❌ Failed to split data: {e}")
            raise
    
    def _convert_enhanced_result_to_analyst_result(self, training_result, model_type: ModelType) -> Optional[Any]:
        """Convert EnhancedModelTrainer result to analyst-specific format."""
        try:
            if not training_result.success:
                return None
            
            # Create a simplified result object for analyst models
            result = type('AnalystModelResult', (), {
                'model_type': model_type.value,
                'trained_model': training_result.best_model,
                'best_params': training_result.best_params,
                'training_metrics': training_result.post_hpo_metrics.post_hpo_metrics if training_result.post_hpo_metrics else {},
                'validation_metrics': training_result.validation_result.validation_metrics if training_result.validation_result else {},
                'pre_hpo_metrics': training_result.pre_hpo_metrics.post_hpo_metrics if training_result.pre_hpo_metrics else {},
                'post_hpo_metrics': training_result.post_hpo_metrics.post_hpo_metrics if training_result.post_hpo_metrics else {},
                'improvement_achieved': training_result.improvement_achieved,
                'hpo_trials_completed': training_result.hpo_trials_completed,
                'persistence_result': training_result.persistence_result,
                'training_time': training_result.training_time
            })()
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert enhanced result for {model_type.value}: {e}")
            return None
    
    async def _train_ensemble_with_enhanced_trainer(self, individual_results: Dict[str, Any], data: pd.DataFrame) -> Optional[Any]:
        """Train ensemble using EnhancedModelTrainer."""
        try:
            # Create ensemble config
            ensemble_config = EnsembleConfig(
                ensemble_type=EnsembleType.VOTING,
                model_types=list(individual_results.keys()),
                enable_hyperparameter_optimization=True,
                voting_strategy='soft'
            )
            
            # Initialize ensemble manager
            ensemble_manager = EnsembleManager(ensemble_config)
            
            # Add trained models to ensemble
            for model_type, result in individual_results.items():
                if result and result.trained_model:
                    ensemble_manager.add_model(result.trained_model, model_type)
            
            # Train ensemble
            ensemble_result = await ensemble_manager.train_ensemble(data)
            
            return type('EnsembleResult', (), {
                'ensemble_manager': ensemble_manager,
                'ensemble_result': ensemble_result,
                'training_successful': ensemble_result is not None
            })()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train ensemble: {e}")
            return None
