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
    
    # Regime-specific configuration
    enable_regime_specific_training: bool = True
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
        
        # Check regime columns if regime-specific training is enabled
        if self.config.enable_regime_specific_training:
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
        """Internal Analyst model training logic with ML commons integration."""
        
        results = AnalystTrainingResults(
            analyst_name=self.config.analyst_name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        # Train individual models with enhanced ML commons integration
        individual_results = {}
        model_performance = {}
        
        for model_type in self.config.model_types:
            self.logger.info(f"🔄 Training {model_type.value} with ML commons...")
            
            try:
                # Perform analyst-specific HPO if enabled
                analyst_hpo_params = {}
                if self.config.enable_hyperparameter_optimization:
                    analyst_hpo_params = await self._optimize_analyst_hyperparameters(data, model_type)
                
        # Create training config for this model type
        training_config = self._create_training_config(model_type)
        
        # Apply analyst-specific HPO parameters if available
        if analyst_hpo_params:
            training_config.model_params.update(analyst_hpo_params)
            self.logger.info(f"🔧 Applied analyst-specific HPO parameters for {model_type.value}")
        
        # Pass launch mode to GeneralModelTrainer
        training_config.launch_mode = self.config.launch_mode
        training_config.hpo_strategy = self.config.hpo_strategy
                
                # Train model with enhanced ML commons integration
                trainer = GeneralModelTrainer(training_config)
                model_result = await trainer.train_model(data, **kwargs)
                
                # Store results
                individual_results[model_type.value] = model_result
                model_performance[model_type.value] = model_result.validation_metrics
                
                # Add to ensemble if enabled
                if self.ensemble_manager and model_result.trained_model is not None:
                    await self.ensemble_manager.add_model(
                        model_name=model_type.value,
                        model=model_result.trained_model,
                        performance_metrics=model_result.validation_metrics
                    )
                
                self.logger.info(f"✅ {model_type.value} training completed")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train {model_type.value}: {e}")
                continue
        
        # Train ensemble if enabled with ML commons
        ensemble_performance = {}
        if self.ensemble_manager and len(self.ensemble_manager.models) > 0:
            self.logger.info("🔄 Training ensemble with ML commons...")
            
            try:
                # Prepare data for ensemble training with enhanced preprocessing
                X, y = self._prepare_ensemble_data_enhanced(data)
                
                # Create ensemble using ML commons
                ensemble_result = await self.ensemble_manager.create_ensemble(X, y)
                ensemble_performance = ensemble_result.ensemble_performance
                
                results.ensemble_manager = self.ensemble_manager
                self.logger.info("✅ Enhanced ensemble training completed")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train ensemble: {e}")
        
        # Train regime-specific models if enabled with ML commons
        regime_specific_results = {}
        if self.config.enable_regime_specific_training:
            regime_specific_results = await self._train_regime_specific_models_enhanced(data, **kwargs)
        
        # Calculate overall performance
        overall_performance = self._calculate_overall_performance(model_performance, ensemble_performance)
        
        # Update results
        results.individual_models = individual_results
        results.model_performance = model_performance
        results.ensemble_performance = ensemble_performance
        results.regime_specific_results = regime_specific_results
        results.overall_performance = overall_performance
        
        return results
    
    def _create_training_config(self, model_type: AnalystModelType) -> ModelTrainingConfig:
        """Create training configuration for specific model type."""
        
        # Determine task type based on model type
        if model_type in [AnalystModelType.REGIME_CLASSIFIER, AnalystModelType.SIGNAL_PREDICTOR, AnalystModelType.META_LABELER]:
            task_type = TaskType.CLASSIFICATION
        else:
            task_type = TaskType.REGRESSION
        
        # Determine model type
        if model_type == AnalystModelType.REGIME_CLASSIFIER:
            ml_model_type = ModelType.RANDOM_FOREST
        elif model_type == AnalystModelType.SIGNAL_PREDICTOR:
            ml_model_type = ModelType.XGBOOST
        elif model_type == AnalystModelType.CONFIDENCE_ESTIMATOR:
            ml_model_type = ModelType.LIGHTGBM
        else:
            ml_model_type = ModelType.RANDOM_FOREST
        
        # Get target column
        target_column = self.config.target_columns.get(model_type.value, 'target')
        
        return ModelTrainingConfig(
            model_name=f"{self.config.analyst_name}_{model_type.value}",
            task_type=task_type,
            model_type=ml_model_type,
            output_dir=f"{self.config.output_dir}/{model_type.value}",
            feature_columns=self.config.feature_columns,
            target_column=target_column,
            validation_split=self.config.validation_split,
            test_split=self.config.test_split,
            enable_hyperparameter_optimization=self.config.enable_hyperparameter_optimization,
            hpo_trials=self.config.hpo_trials,
            hpo_timeout=self.config.hpo_timeout,
            hpo_sampler=self.config.hpo_sampler,
            hpo_pruner=self.config.hpo_pruner,
            enable_early_stopping=self.config.enable_early_stopping,
            early_stopping_patience=self.config.early_stopping_patience,
            enable_gpu_acceleration=self.config.enable_gpu_acceleration,
            enable_memory_optimization=self.config.enable_memory_optimization,
            enable_parallel_processing=self.config.enable_parallel_processing,
            memory_limit_gb=self.config.memory_limit_gb,
            max_workers=self.config.max_workers,
            cross_validation_folds=self.config.cross_validation_folds
        )
    
    def _prepare_ensemble_data_enhanced(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ensemble training with enhanced preprocessing."""
        
        # Use regime classifier as primary target for ensemble
        target_column = self.config.target_columns.get('regime_classifier', 'regime_label')
        
        if target_column not in data.columns:
            # Fallback to first available target
            available_targets = [col for col in self.config.target_columns.values() if col in data.columns]
            if available_targets:
                target_column = available_targets[0]
            else:
                raise ValidationError("No valid target column found for ensemble training")
        
        X = data[self.config.feature_columns].copy()
        y = data[target_column].copy()
        
        # Enhanced missing value handling using ML commons
        X = self.data_quality.enhanced_automated_data_cleaning(X)
        y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        
        return X, y
    
    async def _train_regime_specific_models_enhanced(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Train regime-specific models with ML commons integration."""
        
        regime_results = {}
        
        # Get unique regimes
        regime_col = self.config.regime_columns[0] if self.config.regime_columns else 'hmm_cluster'
        
        if regime_col not in data.columns:
            self.logger.warning(f"⚠️ Regime column {regime_col} not found, skipping regime-specific training")
            return regime_results
        
        # Use ML commons regime processor for enhanced regime analysis
        try:
            regime_analysis = await self.regime_processor.analyze_regime_transitions(data, regime_col)
            self.logger.info(f"📊 Regime analysis completed: {regime_analysis}")
        except Exception as e:
            self.logger.warning(f"⚠️ Regime analysis failed: {e}")
        
        unique_regimes = data[regime_col].unique()
        self.logger.info(f"🔄 Training regime-specific models for {len(unique_regimes)} regimes with ML commons")
        
        for regime in unique_regimes:
            if pd.isna(regime):
                continue
            
            self.logger.info(f"🔄 Training models for regime {regime} with ML commons...")
            
            try:
                # Filter data for this regime
                regime_data = data[data[regime_col] == regime].copy()
                
                if len(regime_data) < 50:  # Minimum samples for training
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data)} samples")
                    continue
                
                # Train models for this regime with enhanced ML commons
                regime_model_results = {}
                
                for model_type in self.config.model_types:
                    try:
                        # Create training config for this regime
                        training_config = self._create_training_config(model_type)
                        training_config.model_name = f"{self.config.analyst_name}_{model_type.value}_regime_{regime}"
                        training_config.output_dir = f"{self.config.output_dir}/regime_{regime}/{model_type.value}"
                        
                        # Train model with enhanced ML commons integration
                        trainer = GeneralModelTrainer(training_config)
                        model_result = await trainer.train_model(regime_data, **kwargs)
                        
                        regime_model_results[model_type.value] = model_result
                        
                    except Exception as e:
                        self.logger.error(f"❌ Failed to train {model_type.value} for regime {regime}: {e}")
                        continue
                
                regime_results[f"regime_{regime}"] = regime_model_results
                self.logger.info(f"✅ Regime {regime} training completed with ML commons")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to train regime-specific models for regime {regime}: {e}")
                continue
        
        return regime_results
    
    def _calculate_overall_performance(
        self, 
        model_performance: Dict[str, Dict[str, float]], 
        ensemble_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        
        overall_metrics = {}
        
        # Aggregate individual model performance
        if model_performance:
            # Calculate average performance across models
            all_scores = []
            for model_name, metrics in model_performance.items():
                if 'accuracy' in metrics:
                    all_scores.append(metrics['accuracy'])
                elif 'r2_score' in metrics:
                    all_scores.append(metrics['r2_score'])
                elif 'f1_score' in metrics:
                    all_scores.append(metrics['f1_score'])
            
            if all_scores:
                overall_metrics['average_model_performance'] = np.mean(all_scores)
                overall_metrics['best_model_performance'] = np.max(all_scores)
                overall_metrics['model_count'] = len(model_performance)
        
        # Add ensemble performance
        if ensemble_performance:
            for metric, value in ensemble_performance.items():
                overall_metrics[f'ensemble_{metric}'] = value
        
        return overall_metrics
    
    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
        
        optimizations.append("ml_commons_integration")
        
        return optimizations
    
    async def _optimize_analyst_hyperparameters(
        self, 
        data: pd.DataFrame, 
        model_type: AnalystModelType
    ) -> Dict[str, Any]:
        """Perform analyst-specific hyperparameter optimization."""
        
        self.logger.info(f"🔬 Starting analyst-specific HPO for {model_type.value}...")
        
        try:
            # Get analyst-specific search space
            search_space = self._get_analyst_search_space(model_type)
            
            # Prepare data for this model type
            target_column = self.config.target_columns.get(model_type.value, 'target')
            X = data[self.config.feature_columns].copy()
            y = data[target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Use ML commons HPO optimizer with analyst-specific configuration
            best_params = await self.hpo_optimizer.optimize(
                model_type=self._get_analyst_model_type(model_type).value,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=self.config.hpo_trials,
                timeout=self.config.hpo_timeout,
                task_type=self._get_analyst_task_type(model_type).value,
                sampler=self.config.hpo_sampler,
                pruner=self.config.hpo_pruner,
                search_space=search_space
            )
            
            self.logger.info(f"✅ Analyst HPO completed for {model_type.value}")
            self.logger.info(f"📊 Best parameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            self.logger.warning(f"⚠️ Analyst HPO failed for {model_type.value}: {e}")
            return {}
    
    def _get_analyst_search_space(self, model_type: AnalystModelType) -> Dict[str, Any]:
        """Get analyst-specific hyperparameter search space."""
        
        if model_type == AnalystModelType.REGIME_CLASSIFIER:
            # Regime classifier needs robust parameters
            return {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 800, 'step': 20},
                'max_depth': {'type': 'int', 'low': 5, 'high': 25},
                'min_samples_split': {'type': 'int', 'low': 5, 'high': 50},
                'min_samples_leaf': {'type': 'int', 'low': 2, 'high': 20},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.7, 0.8, 0.9]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]},
                'class_weight': {'type': 'categorical', 'choices': [None, 'balanced', 'balanced_subsample']}
            }
        
        elif model_type == AnalystModelType.SIGNAL_PREDICTOR:
            # Signal predictor needs fast and accurate parameters
            return {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500, 'step': 10},
                'max_depth': {'type': 'int', 'low': 3, 'high': 15},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 5.0, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 5.0, 'log': True}
            }
        
        elif model_type == AnalystModelType.CONFIDENCE_ESTIMATOR:
            # Confidence estimator needs calibrated parameters
            return {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 400, 'step': 10},
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.15, 'log': True},
                'subsample': {'type': 'float', 'low': 0.8, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.8, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 2.0, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 2.0, 'log': True},
                'min_child_samples': {'type': 'int', 'low': 10, 'high': 100}
            }
        
        elif model_type == AnalystModelType.RISK_ASSESSOR:
            # Risk assessor needs conservative parameters
            return {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 600, 'step': 20},
                'max_depth': {'type': 'int', 'low': 4, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 10, 'high': 100},
                'min_samples_leaf': {'type': 'int', 'low': 5, 'high': 50},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.6, 0.7, 0.8]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]},
                'max_samples': {'type': 'float', 'low': 0.7, 'high': 1.0}
            }
        
        elif model_type == AnalystModelType.META_LABELER:
            # Meta labeler needs balanced parameters
            return {
                'n_estimators': {'type': 'int', 'low': 75, 'high': 500, 'step': 15},
                'max_depth': {'type': 'int', 'low': 4, 'high': 18},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 3.0, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 3.0, 'log': True}
            }
        
        else:
            # Default search space
            return {}
    
    def _get_analyst_model_type(self, analyst_model_type: AnalystModelType) -> ModelType:
        """Get the underlying ML model type for analyst model type."""
        
        if analyst_model_type == AnalystModelType.REGIME_CLASSIFIER:
            return ModelType.RANDOM_FOREST
        elif analyst_model_type == AnalystModelType.SIGNAL_PREDICTOR:
            return ModelType.XGBOOST
        elif analyst_model_type == AnalystModelType.CONFIDENCE_ESTIMATOR:
            return ModelType.LIGHTGBM
        elif analyst_model_type == AnalystModelType.RISK_ASSESSOR:
            return ModelType.RANDOM_FOREST
        elif analyst_model_type == AnalystModelType.META_LABELER:
            return ModelType.XGBOOST
        else:
            return ModelType.RANDOM_FOREST
    
    def _get_analyst_task_type(self, analyst_model_type: AnalystModelType) -> TaskType:
        """Get the task type for analyst model type."""
        
        if analyst_model_type in [AnalystModelType.REGIME_CLASSIFIER, AnalystModelType.SIGNAL_PREDICTOR, AnalystModelType.META_LABELER]:
            return TaskType.CLASSIFICATION
        else:
            return TaskType.REGRESSION
    
    async def predict(
        self, 
        data: pd.DataFrame, 
        use_ensemble: bool = True
    ) -> Dict[str, Any]:
        """Make predictions using trained models."""
        
        predictions = {}
        
        # Individual model predictions
        for model_name, model_result in self.individual_models.items():
            if model_result.trained_model is not None:
                try:
                    X = data[self.config.feature_columns]
                    if hasattr(model_result.trained_model, 'predict_proba'):
                        pred_proba = model_result.trained_model.predict_proba(X)
                        predictions[f"{model_name}_probabilities"] = pred_proba
                        predictions[f"{model_name}_predictions"] = np.argmax(pred_proba, axis=1)
                    else:
                        predictions[f"{model_name}_predictions"] = model_result.trained_model.predict(X)
                except Exception as e:
                    self.logger.error(f"Error making predictions with {model_name}: {e}")
        
        # Ensemble predictions
        if use_ensemble and self.ensemble_manager and self.ensemble_manager.ensemble_model is not None:
            try:
                X = data[self.config.feature_columns]
                ensemble_pred, ensemble_proba = await self.ensemble_manager.predict(X)
                predictions['ensemble_predictions'] = ensemble_pred
                if ensemble_proba is not None:
                    predictions['ensemble_probabilities'] = ensemble_proba
            except Exception as e:
                self.logger.error(f"Error making ensemble predictions: {e}")
        
        return predictions
    
    async def save_analyst_models(self, results: AnalystTrainingResults) -> None:
        """Save all trained Analyst models using ML commons registry."""
        
        try:
            # Save individual models using ML commons registry
            for model_name, model_result in results.individual_models.items():
                if model_result.trained_model is not None:
                    model_path = f"{self.config.output_dir}/{model_name}_model.pkl"
                    await self.model_registry.save_model(
                        model=model_result.trained_model,
                        model_name=model_name,
                        file_path=model_path,
                        metadata={
                            'analyst_name': self.config.analyst_name,
                            'model_type': model_name,
                            'training_time': datetime.now().isoformat()
                        }
                    )
            
            # Save ensemble if available
            if results.ensemble_manager:
                ensemble_path = f"{self.config.output_dir}/ensemble.pkl"
                await results.ensemble_manager.save_ensemble(ensemble_path)
            
            # Save results metadata
            results_path = f"{self.config.output_dir}/training_results.json"
            await safe_json_dump(results_path, results.__dict__)
            
            self.logger.info(f"💾 All Analyst models saved to {self.config.output_dir} using ML commons registry")
            
        except Exception as e:
            self.logger.error(f"Error saving Analyst models: {e}")
            raise