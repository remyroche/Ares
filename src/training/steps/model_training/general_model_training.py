"""
General Model Training + HPO

This module provides comprehensive model training with hyperparameter optimization
for general ML models, utilizing M1 optimizations and modern ML practices.
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
import optuna
from pathlib import Path

# M1 Optimization imports
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer

# ML Commons utilities - Enhanced integration
from src.utils.ml_common import (
    ModelEvaluator, HPOptimizer, FeatureSelectionFramework,
    DataLabelingUtilities, MemoryEfficientTraining, 
    ParallelProcessingCoordinator, ModelRegistry,
    DataQualityUtilities, CrossValidationUtilities,
    LookaheadProtection, MLTrainingSafeguards
)

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
    get_scaled_hpo_timeout, log_intensity_info
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models supported."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class TaskType(Enum):
    """Types of ML tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_CLASS = "multi_class"
    MULTI_LABEL = "multi_label"


@dataclass
class ModelTrainingConfig:
    """Configuration for model training."""
    # Basic configuration
    model_name: str
    task_type: TaskType
    model_type: ModelType
    output_dir: str
    
    # Data configuration
    feature_columns: List[str]
    target_column: str
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Training configuration
    enable_hyperparameter_optimization: bool = True
    hpo_trials: int = 100
    hpo_timeout: int = 3600  # 1 hour
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Apply intensity scaling after initialization."""
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.hpo_trials = get_scaled_hpo_trials(self.hpo_trials, intensity_pct)
            self.hpo_timeout = get_scaled_hpo_timeout(self.hpo_timeout, intensity_pct)
            self.early_stopping_patience = max(1, int(self.early_stopping_patience * intensity_pct))
            logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%): HPO trials={self.hpo_trials}, timeout={self.hpo_timeout}s")
    
    # Model-specific configuration
    model_params: Dict[str, Any] = field(default_factory=dict)
    hpo_search_space: Dict[str, Any] = field(default_factory=dict)
    
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
    cross_validation_folds: int = 5
    scoring_metric: str = "accuracy"  # accuracy, f1, precision, recall, roc_auc, etc.
    
    # Output settings
    save_model: bool = True
    save_predictions: bool = True
    generate_reports: bool = True


@dataclass
class ModelTrainingResults:
    """Results from model training."""
    # Basic info
    model_name: str
    task_type: TaskType
    model_type: ModelType
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # Model information
    trained_model: Any
    best_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[np.ndarray] = None
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: List[float] = field(default_factory=list)
    
    # HPO results
    hpo_trials: int = 0
    best_hpo_score: float = 0.0
    hpo_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data information
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    
    # Metadata
    config: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class GeneralModelTrainer:
    """General model trainer with HPO and M1 optimizations, enhanced with ML commons."""
    
    def __init__(self, config: ModelTrainingConfig):
        """Initialize general model trainer."""
        self.config = config
        self.logger = logger.getChild('GeneralModelTrainer')
        
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
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        # Log intensity information
        log_intensity_info()
        
        # Initialize model factory
        self.model_factory = ModelFactory()
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        self.logger.info(f"🚀 GeneralModelTrainer initialized for {config.model_name}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎯 Task type: {config.task_type.value}")
        self.logger.info(f"🤖 Model type: {config.model_type.value}")
        self.logger.info(f"🔧 ML Commons integration: Enhanced")
    
    @traced(span_name='train_model')
    @log_execution_time
    async def train_model(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> ModelTrainingResults:
        """Train model with HPO and M1 optimizations, enhanced with ML commons."""
        
        self.logger.info("🚀 Starting enhanced model training with ML commons...")
        start_time = time.time()
        
        # Validate inputs with ML safeguards
        self._validate_data_with_safeguards(data)
        
        # Apply lookahead bias protection
        data = self.lookahead_protection.apply_protection(data)
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._train_model_internal(data, **kwargs)
        else:
            results = await self._train_model_internal(data, **kwargs)
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ Enhanced model training completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Best validation score: {results.validation_metrics.get('score', 0.0):.4f}")
        
        return results
    
    def _validate_data_with_safeguards(self, data: pd.DataFrame) -> None:
        """Validate input data using ML safeguards."""
        
        # Basic validation
        if data.empty:
            raise ValidationError("Input data is empty")
        
        # Check required columns
        missing_features = [col for col in self.config.feature_columns if col not in data.columns]
        if missing_features:
            raise ValidationError(f"Missing feature columns: {missing_features}")
        
        if self.config.target_column not in data.columns:
            raise ValidationError(f"Missing target column: {self.config.target_column}")
        
        # Check for sufficient data
        if len(data) < 100:
            raise ValidationError(f"Insufficient data: {len(data)} < 100")
        
        # Use ML safeguards for advanced validation
        try:
            self.ml_safeguards.validate_training_data(data, self.config.target_column)
            self.logger.info("✅ ML safeguards validation passed")
        except Exception as e:
            self.logger.warning(f"⚠️ ML safeguards validation warning: {e}")
        
        # Data quality assessment
        quality_score = self.data_quality.calculate_data_quality_score(data)
        self.logger.info(f"📊 Data quality score: {quality_score:.2f}")
        
        if quality_score < 0.7:
            self.logger.warning("⚠️ Low data quality score detected")
    
    async def _train_model_internal(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> ModelTrainingResults:
        """Internal model training logic with ML commons integration."""
        
        # Prepare data with enhanced preprocessing
        X, y = self._prepare_data_enhanced(data)
        
        # Feature selection using ML commons
        if len(self.config.feature_columns) > 50:  # Only if many features
            X = await self._apply_feature_selection(X, y)
        
        # Split data with temporal integrity
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data_temporal(X, y)
        
        # Perform hyperparameter optimization using ML commons
        if self.config.enable_hyperparameter_optimization:
            best_params = await self._optimize_hyperparameters_enhanced(X_train, y_train, X_val, y_val)
        else:
            best_params = self.config.model_params
        
        # Train final model with best parameters
        trained_model = await self._train_final_model_enhanced(X_train, y_train, best_params)
        
        # Evaluate model using ML commons
        training_metrics = await self._evaluate_model_enhanced(trained_model, X_train, y_train)
        validation_metrics = await self._evaluate_model_enhanced(trained_model, X_val, y_val)
        test_metrics = await self._evaluate_model_enhanced(trained_model, X_test, y_test)
        
        # Perform cross-validation using ML commons
        cv_scores = await self._cross_validate_model_enhanced(trained_model, X_train, y_train)
        
        # Extract feature importance if available
        feature_importance = self._extract_feature_importance(trained_model)
        
        # Create results
        results = ModelTrainingResults(
            model_name=self.config.model_name,
            task_type=self.config.task_type,
            model_type=self.config.model_type,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            trained_model=trained_model,
            best_params=best_params,
            feature_importance=feature_importance,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            cross_validation_scores=cv_scores,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            test_samples=len(X_test),
            feature_count=len(self.config.feature_columns),
            config=self.config,
            optimization_used=self._get_optimization_used()
        )
        
        return results
    
    def _prepare_data_enhanced(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training with enhanced preprocessing."""
        
        # Select features and target
        X = data[self.config.feature_columns].copy()
        y = data[self.config.target_column].copy()
        
        # Enhanced missing value handling
        X = self.data_quality.enhanced_automated_data_cleaning(X)
        y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Convert categorical variables if needed
        X = self._handle_categorical_variables(X)
        
        return X, y
    
    async def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature selection using ML commons."""
        
        self.logger.info("🔍 Applying feature selection...")
        
        # Use feature selection framework
        selected_features = await self.feature_selector.select_features(
            X, y, 
            method='mutual_info',
            k_best=min(50, len(X.columns))
        )
        
        self.logger.info(f"📊 Selected {len(selected_features)} features from {len(X.columns)}")
        
        return X[selected_features]
    
    def _split_data_temporal(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data with temporal integrity using ML commons."""
        
        # Use cross-validation utilities for temporal splitting
        splits = self.cv_utils.temporal_train_test_split(
            X, y,
            test_size=self.config.test_split,
            validation_size=self.config.validation_split
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        self.logger.info(f"📊 Temporal data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    async def _optimize_hyperparameters_enhanced(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using ML commons HPO."""
        
        self.logger.info("🔄 Starting enhanced hyperparameter optimization...")
        
        # Use ML commons HPO optimizer
        best_params = await self.hpo_optimizer.optimize(
            model_type=self.config.model_type.value,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=self.config.hpo_trials,
            timeout=self.config.hpo_timeout,
            task_type=self.config.task_type.value
        )
        
        self.logger.info(f"✅ Enhanced HPO completed")
        self.logger.info(f"📊 Best parameters: {best_params}")
        
        return best_params
    
    async def _train_final_model_enhanced(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        params: Dict[str, Any]
    ) -> Any:
        """Train final model with enhanced training using ML commons."""
        
        self.logger.info("🔄 Training final model with enhanced training...")
        
        # Create model with best parameters
        model = self.model_factory.create_model(self.config.model_type, params)
        
        # Use memory efficient training if enabled
        if self.config.enable_memory_optimization:
            trained_model = await self.memory_efficient_training.train_model(
                model, X_train, y_train
            )
        else:
            model.fit(X_train, y_train)
            trained_model = model
        
        self.logger.info("✅ Enhanced final model trained")
        
        return trained_model
    
    async def _evaluate_model_enhanced(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance using ML commons."""
        
        # Use ML commons model evaluator
        metrics = await self.model_evaluator.evaluate_model(
            model, X, y,
            task_type=self.config.task_type.value,
            return_detailed=True
        )
        
        return metrics
    
    async def _cross_validate_model_enhanced(self, model: Any, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Perform cross-validation using ML commons."""
        
        # Use ML commons cross-validation utilities
        cv_scores = await self.cv_utils.cross_validate_model(
            model, X, y,
            cv_folds=self.config.cross_validation_folds,
            scoring=self.config.scoring_metric
        )
        
        self.logger.info(f"📊 Enhanced cross-validation scores: {cv_scores}")
        self.logger.info(f"📊 Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        
        return cv_scores
    
    def _handle_categorical_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical variables."""
        
        # Simple one-hot encoding for categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_columns) > 0:
            X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            self.logger.info(f"📊 Encoded {len(categorical_columns)} categorical columns")
            return X_encoded
        
        return X
    
    def _extract_feature_importance(self, model: Any) -> Optional[np.ndarray]:
        """Extract feature importance if available."""
        
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {e}")
            return None
    
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
    
    async def save_model(self, model: Any, file_path: str) -> None:
        """Save trained model using ML commons registry."""
        
        try:
            # Use ML commons model registry for saving
            await self.model_registry.save_model(
                model=model,
                model_name=self.config.model_name,
                file_path=file_path,
                metadata={
                    'task_type': self.config.task_type.value,
                    'model_type': self.config.model_type.value,
                    'training_time': datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"💾 Model saved to {file_path} using ML commons registry")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    async def load_model(self, file_path: str) -> Any:
        """Load trained model using ML commons registry."""
        
        try:
            # Use ML commons model registry for loading
            model = await self.model_registry.load_model(file_path)
            
            self.logger.info(f"📂 Model loaded from {file_path} using ML commons registry")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise


class ModelFactory:
    """Factory for creating ML models."""
    
    def create_model(self, model_type: ModelType, params: Dict[str, Any]) -> Any:
        """Create model instance based on type and parameters."""
        
        if model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            # Determine if classification or regression based on params
            if 'task_type' in params and params['task_type'] == TaskType.REGRESSION:
                return RandomForestRegressor(**params)
            else:
                return RandomForestClassifier(**params)
        
        elif model_type == ModelType.XGBOOST:
            import xgboost as xgb
            if 'task_type' in params and params['task_type'] == TaskType.REGRESSION:
                return xgb.XGBRegressor(**params)
            else:
                return xgb.XGBClassifier(**params)
        
        elif model_type == ModelType.LIGHTGBM:
            import lightgbm as lgb
            if 'task_type' in params and params['task_type'] == TaskType.REGRESSION:
                return lgb.LGBMRegressor(**params)
            else:
                return lgb.LGBMClassifier(**params)
        
        elif model_type == ModelType.CATBOOST:
            import catboost as cb
            if 'task_type' in params and params['task_type'] == TaskType.REGRESSION:
                return cb.CatBoostRegressor(**params)
            else:
                return cb.CatBoostClassifier(**params)
        
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        
        elif model_type == ModelType.SVM:
            from sklearn.svm import SVC, SVR
            if 'task_type' in params and params['task_type'] == TaskType.REGRESSION:
                return SVR(**params)
            else:
                return SVC(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")