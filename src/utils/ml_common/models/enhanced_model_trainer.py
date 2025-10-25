"""
Enhanced Model Trainer

This module provides a comprehensive model trainer that integrates evaluation,
validation, and persistence components with pre/post HPO metrics comparison.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path

# Suppress LightGBM warnings about no further splits
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

# ML library imports
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)

# Import evaluation result types
# Note: Removed dependency on hybrid_nas_tas_regime module
# from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_evaluation_framework import EvaluationResult
# from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_validation_system import ValidationResult
from src.utils.ml_common.post_training.model_persistence import PersistenceResult
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
    ConfigurationError, ModelTrainingError
)
from src.utils.logger import system_logger

# Import model evaluation components
from .model_registry import (
    ModelRegistry
)

# Import post-training components
from ..post_training.model_evaluation import ModelEvaluator, EvaluationConfig, EvaluationResult
from ..post_training.model_validation import ModelValidator, ValidationConfig, ValidationResult
from ..post_training.model_persistence import ModelPersistence, PersistenceConfig, PersistenceResult

# Import multi-timeframe training - commented out as module doesn't exist
# from .multi_timeframe_training import MultiTimeframeTrainer, MultiTimeframeTrainingConfig, TimeframeConfig

@dataclass
class EnhancedTrainingConfig:
    """Configuration for enhanced model training."""

    # Model training settings
    model_types: List[str] = field(default_factory=lambda: ['lightgbm', 'random_forest', 'neural_network'])
    enable_hyperparameter_optimization: bool = True
    hpo_trials: int = 100
    hpo_timeout: int = 3600  # 1 hour

    # Multi-timeframe settings
    enable_multi_timeframe_training: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '30m', '1h'])
    timeframe_weights: List[float] = field(default_factory=lambda: [0.3, 0.2, 0.2, 0.2, 0.1])

    # Evaluation settings
    enable_pre_hpo_evaluation: bool = True
    enable_post_hpo_evaluation: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1_score', 'r2_score', 'sharpe_ratio'])

    # Validation settings
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_holdout_validation: bool = True
    holdout_ratio: float = 0.2

    # Persistence settings
    enable_model_persistence: bool = True
    enable_versioning: bool = True
    max_versions: int = 10

    # Performance thresholds
    min_accuracy_threshold: float = 0.5
    min_f1_threshold: float = 0.5
    min_r2_threshold: float = 0.0
    min_sharpe_threshold: float = 0.0

    # Output settings
    save_training_results: bool = True
    generate_training_report: bool = True
    training_report_path: Optional[str] = None

@dataclass
class TrainingResult:
    """Result of enhanced model training."""

    # Training status
    success: bool = False
    training_time: float = 0.0
    training_timestamp: str = ""

    # Model information
    model_name: str = ""
    model_type: str = ""
    best_model: Optional[Any] = None
    best_params: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    pre_hpo_metrics: Optional[EvaluationResult] = None
    post_hpo_metrics: Optional[EvaluationResult] = None
    validation_result: Optional[ValidationResult] = None

    # Persistence information
    persistence_result: Optional[PersistenceResult] = None

    # Multi-timeframe results
    multi_timeframe_results: Optional[Dict[str, Any]] = None

    # Training summary
    models_trained: int = 0
    hpo_trials_completed: int = 0
    best_score: float = 0.0
    improvement_achieved: bool = False

    # Error information
    error_message: Optional[str] = None

class EnhancedModelTrainer:
    """Enhanced model trainer with integrated evaluation, validation, and persistence."""

    def __init__(self, config: EnhancedTrainingConfig):
        """Initialize the enhanced model trainer.

        Args:
            config: Enhanced training configuration
        """
        self.config = config
        self.logger = system_logger.getChild('EnhancedModelTrainer')

        # Initialize post-training components
        self.evaluator = ModelEvaluator(EvaluationConfig(
            enable_pre_hpo_evaluation=config.enable_pre_hpo_evaluation,
            enable_post_hpo_evaluation=config.enable_post_hpo_evaluation,
            enable_cross_validation=config.enable_cross_validation,
            cv_folds=config.cv_folds,
            calculate_classification_metrics=True,
            calculate_regression_metrics=True,
            calculate_trading_metrics=True,
            min_accuracy_threshold=config.min_accuracy_threshold,
            min_f1_threshold=config.min_f1_threshold,
            min_r2_threshold=config.min_r2_threshold,
            min_sharpe_threshold=config.min_sharpe_threshold,
            save_evaluation_results=True,
            generate_evaluation_report=True
        ))

        self.validator = ModelValidator(ValidationConfig(
            enable_cross_validation=config.enable_cross_validation,
            enable_holdout_validation=config.enable_holdout_validation,
            cv_folds=config.cv_folds,
            holdout_ratio=config.holdout_ratio,
            min_cv_score=config.min_accuracy_threshold,
            min_holdout_score=config.min_accuracy_threshold,
            save_validation_results=True,
            generate_validation_report=True
        ))

        self.persistence = ModelPersistence(PersistenceConfig(
            base_model_dir="models",
            enable_versioning=config.enable_versioning,
            max_versions=config.max_versions,
            save_metadata=True,
            enable_backup=True,
            validate_on_save=True,
            validate_on_load=True,
            save_persistence_log=True
        ))

        # Initialize multi-timeframe trainer if enabled - commented out as module doesn't exist
        self.multi_timeframe_trainer = None
        # if config.enable_multi_timeframe_training:
        #     timeframe_configs = [
        #         TimeframeConfig(timeframe=tf, weight=weight)
        #         for tf, weight in zip(config.timeframes, config.timeframe_weights)
        #     ]
        #
        #     mtf_config = MultiTimeframeTrainingConfig(
        #         timeframes=timeframe_configs,
        #         enable_cross_timeframe_features=True,
        #         enable_timeframe_ensemble=True,
        #         ensemble_method="weighted_average"
        #     )
        #
        #     self.multi_timeframe_trainer = MultiTimeframeTrainer(mtf_config, "default", "default")

        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = self._apply_intensity_scaling(intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to enhanced training config")

    def _apply_intensity_scaling(self, intensity_pct: float) -> EnhancedTrainingConfig:
        """Apply intensity scaling to the configuration."""
        return EnhancedTrainingConfig(
            model_types=self.config.model_types,
            enable_hyperparameter_optimization=self.config.enable_hyperparameter_optimization,
            hpo_trials=max(10, int(self.config.hpo_trials * intensity_pct)),
            hpo_timeout=max(300, int(self.config.hpo_timeout * intensity_pct)),
            enable_multi_timeframe_training=self.config.enable_multi_timeframe_training,
            timeframes=self.config.timeframes,
            timeframe_weights=self.config.timeframe_weights,
            enable_pre_hpo_evaluation=self.config.enable_pre_hpo_evaluation,
            enable_post_hpo_evaluation=self.config.enable_post_hpo_evaluation,
            evaluation_metrics=self.config.evaluation_metrics,
            enable_cross_validation=self.config.enable_cross_validation,
            cv_folds=max(3, int(self.config.cv_folds * intensity_pct)),
            enable_holdout_validation=self.config.enable_holdout_validation,
            holdout_ratio=self.config.holdout_ratio,
            enable_model_persistence=self.config.enable_model_persistence,
            enable_versioning=self.config.enable_versioning,
            max_versions=max(3, int(self.config.max_versions * intensity_pct)),
            min_accuracy_threshold=self.config.min_accuracy_threshold,
            min_f1_threshold=self.config.min_f1_threshold,
            min_r2_threshold=self.config.min_r2_threshold,
            min_sharpe_threshold=self.config.min_sharpe_threshold,
            save_training_results=self.config.save_training_results,
            generate_training_report=self.config.generate_training_report,
            training_report_path=self.config.training_report_path
        )

    @handles_errors(default_return=TrainingResult(success=False), context='Enhanced model training')
    # @log_execution_time  # Temporarily disabled due to import conflicts
    async def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         model_name: str = "enhanced_model",
                         model_type: str = "lightgbm",
                         multi_timeframe_data: Optional[Dict[str, pd.DataFrame]] = None) -> TrainingResult:
        """Train a model with enhanced evaluation, validation, and persistence.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            model_type: Type of model to train
            multi_timeframe_data: Multi-timeframe training data (optional)

        Returns:
            TrainingResult with comprehensive training information
        """
        try:
            self.logger.info(f"🚀 Starting enhanced model training: {model_name}")
            start_time = time.time()

            # Initialize training result
            result = TrainingResult(
                model_name=model_name,
                model_type=model_type,
                training_timestamp=get_current_datetime()
            )

            # 1. Train initial model (pre-HPO)
            initial_model = await self._train_initial_model(X_train, y_train, model_type)
            if initial_model is None:
                result.error_message = "Failed to train initial model"
                return result

            # 2. Evaluate initial model (pre-HPO)
            pre_hpo_metrics = None
            if self.config.enable_pre_hpo_evaluation:
                self.logger.info("🔍 Evaluating initial model (pre-HPO)...")
                pre_hpo_metrics = await self.evaluator.evaluate_model(
                    initial_model, X_test, y_test, f"{model_name}_pre_hpo"
                )
                result.pre_hpo_metrics = pre_hpo_metrics

            # 3. Hyperparameter optimization
            best_model = initial_model
            best_params = {}
            hpo_trials_completed = 0

            if self.config.enable_hyperparameter_optimization:
                self.logger.info("🔧 Starting hyperparameter optimization...")
                best_model, best_params, hpo_trials_completed = await self._perform_hyperparameter_optimization(
                    X_train, y_train, X_test, y_test, model_type, model_name
                )
                result.hpo_trials_completed = hpo_trials_completed
                result.best_params = best_params

            # 4. Evaluate optimized model (post-HPO)
            post_hpo_metrics = None
            if self.config.enable_post_hpo_evaluation:
                self.logger.info("🔍 Evaluating optimized model (post-HPO)...")
                post_hpo_metrics = await self.evaluator.evaluate_model(
                    best_model, X_test, y_test, f"{model_name}_post_hpo", pre_hpo_metrics
                )
                result.post_hpo_metrics = post_hpo_metrics

                # Check if improvement was achieved
                if pre_hpo_metrics and post_hpo_metrics:
                    result.improvement_achieved = self._check_improvement(pre_hpo_metrics, post_hpo_metrics)

            # 5. Validate model
            self.logger.info("✅ Validating model...")
            validation_result = await self.validator.validate_model(
                best_model, X_train, y_train, f"{model_name}_validation"
            )
            result.validation_result = validation_result

            # 6. Multi-timeframe training (if enabled and data provided)
            if self.config.enable_multi_timeframe_training and multi_timeframe_data and self.multi_timeframe_trainer:
                self.logger.info("🔄 Performing multi-timeframe training...")
                mtf_result = await self.multi_timeframe_trainer.train_models(
                    multi_timeframe_data, self, {'model_type': model_type}
                )
                result.multi_timeframe_results = mtf_result

            # 7. Persist model
            if self.config.enable_model_persistence:
                self.logger.info("💾 Persisting model...")
                persistence_result = await self.persistence.save_model(
                    best_model, model_name, model_type
                )
                result.persistence_result = persistence_result

            # 8. Finalize result
            result.success = True
            result.best_model = best_model
            result.training_time = time.time() - start_time
            result.models_trained = 1

            # Calculate best score
            if post_hpo_metrics and post_hpo_metrics.post_hpo_metrics:
                if post_hpo_metrics.post_hpo_metrics.accuracy is not None:
                    result.best_score = post_hpo_metrics.post_hpo_metrics.accuracy
                elif post_hpo_metrics.post_hpo_metrics.r2_score is not None:
                    result.best_score = post_hpo_metrics.post_hpo_metrics.r2_score

            # Save training results
            if self.config.save_training_results:
                await self._save_training_results(result)

            # Generate training report
            if self.config.generate_training_report:
                await self._generate_training_report(result)

            self.logger.info(f"✅ Enhanced model training completed: {model_name}")
            return result

        except Exception as e:
            self.logger.exception(f"💥 Error in enhanced model training: {e}")
            return TrainingResult(
                success=False,
                model_name=model_name,
                model_type=model_type,
                training_timestamp=get_current_datetime(),
                error_message=str(e)
            )

    @handles_errors(default_return=None, context='Initial model training')
    async def _train_initial_model(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str) -> Optional[Any]:
        """Train an initial model with default parameters."""
        try:
            if model_type == "lightgbm":
                model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            elif model_type == "random_forest":
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
            elif model_type == "neural_network":
                model = MLPClassifier(random_state=42, max_iter=1000)
            else:
                # Default to LightGBM
                model = lgb.LGBMClassifier(random_state=42, verbose=-1)

            model.fit(X_train, y_train)
            return model

        except Exception as e:
            self.logger.exception(f"💥 Error training initial model: {e}")
            return None

    @handles_errors(default_return=(None, {}, 0), context='Hyperparameter optimization')
    async def _perform_hyperparameter_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                                 X_test: np.ndarray, y_test: np.ndarray,
                                                 model_type: str, model_name: str) -> Tuple[Optional[Any], Dict[str, Any], int]:
        """Perform hyperparameter optimization."""
        try:
            import optuna

            def objective(trial):
                # Define hyperparameter search space based on model type
                if model_type == "lightgbm":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 1.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 1.0, log=True)
                    }
                    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                elif model_type == "random_forest":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                    }
                    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                else:
                    # Default to LightGBM
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
                    }
                    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)

                # Train and evaluate
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                return score

            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.hpo_trials, timeout=self.config.hpo_timeout)

            # Get best model
            best_params = study.best_params
            best_score = study.best_value

            # Train final model with best parameters
            if model_type == "lightgbm":
                best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
            elif model_type == "random_forest":
                best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            else:
                best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)

            best_model.fit(X_train, y_train)

            self.logger.info(f"✅ HPO completed: {len(study.trials)} trials, best score: {best_score:.4f}")
            return best_model, best_params, len(study.trials)

        except Exception as e:
            self.logger.exception(f"💥 Error in hyperparameter optimization: {e}")
            return None, {}, 0

    def _check_improvement(self, pre_hpo: EvaluationResult, post_hpo: EvaluationResult) -> bool:
        """Check if improvement was achieved through HPO."""
        try:
            if not pre_hpo.post_hpo_metrics or not post_hpo.post_hpo_metrics:
                return False

            pre_metrics = pre_hpo.post_hpo_metrics
            post_metrics = post_hpo.post_hpo_metrics

            # Check accuracy improvement
            if pre_metrics.accuracy and post_metrics.accuracy:
                if post_metrics.accuracy > pre_metrics.accuracy:
                    return True

            # Check F1 improvement
            if pre_metrics.f1_score and post_metrics.f1_score:
                if post_metrics.f1_score > pre_metrics.f1_score:
                    return True

            # Check R2 improvement
            if pre_metrics.r2_score and post_metrics.r2_score:
                if post_metrics.r2_score > pre_metrics.r2_score:
                    return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️ Error checking improvement: {e}")
            return False

    @handles_errors(default_return=None, context='Training results saving')
    async def _save_training_results(self, result: TrainingResult):
        """Save training results to file."""
        try:
            results_data = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'success': result.success,
                'training_time': result.training_time,
                'training_timestamp': result.training_timestamp,
                'models_trained': result.models_trained,
                'hpo_trials_completed': result.hpo_trials_completed,
                'best_score': result.best_score,
                'improvement_achieved': result.improvement_achieved,
                'best_params': result.best_params,
                'pre_hpo_metrics': result.pre_hpo_metrics.__dict__ if result.pre_hpo_metrics else None,
                'post_hpo_metrics': result.post_hpo_metrics.__dict__ if result.post_hpo_metrics else None,
                'validation_result': result.validation_result.__dict__ if result.validation_result else None,
                'persistence_result': result.persistence_result.__dict__ if result.persistence_result else None,
                'multi_timeframe_results': result.multi_timeframe_results,
                'error_message': result.error_message
            }

            # Save to file
            results_path = f"data_cache/enhanced_training_results_{result.model_name}_{get_current_datetime()}.json"
            ensure_directory(Path(results_path).parent)
            safe_json_dump(results_data, results_path)

            self.logger.info(f"💾 Training results saved to {results_path}")

        except Exception as e:
            self.logger.exception(f"💥 Error saving training results: {e}")

    @handles_errors(default_return=None, context='Training report generation')
    async def _generate_training_report(self, result: TrainingResult):
        """Generate comprehensive training report."""
        try:
            report_path = self.config.training_report_path or f"data_cache/enhanced_training_report_{result.model_name}_{get_current_datetime()}.txt"
            ensure_directory(Path(report_path).parent)

            with open(report_path, 'w') as f:
                f.write(f"Enhanced Model Training Report\n")
                f.write(f"============================\n\n")
                f.write(f"Model Name: {result.model_name}\n")
                f.write(f"Model Type: {result.model_type}\n")
                f.write(f"Training Timestamp: {result.training_timestamp}\n")
                f.write(f"Training Time: {result.training_time:.2f}s\n")
                f.write(f"Success: {result.success}\n")
                f.write(f"Models Trained: {result.models_trained}\n")
                f.write(f"HPO Trials Completed: {result.hpo_trials_completed}\n")
                f.write(f"Best Score: {result.best_score:.4f}\n")
                f.write(f"Improvement Achieved: {result.improvement_achieved}\n\n")

                if result.best_params:
                    f.write(f"Best Hyperparameters:\n")
                    f.write(f"---------------------\n")
                    for param, value in result.best_params.items():
                        f.write(f"{param}: {value}\n")
                    f.write(f"\n")

                if result.pre_hpo_metrics:
                    f.write(f"Pre-HPO Evaluation:\n")
                    f.write(f"-------------------\n")
                    f.write(f"Grade: {result.pre_hpo_metrics.performance_grade}\n")
                    f.write(f"Passed: {result.pre_hpo_metrics.evaluation_passed}\n")
                    if result.pre_hpo_metrics.post_hpo_metrics:
                        metrics = result.pre_hpo_metrics.post_hpo_metrics
                        if metrics.accuracy:
                            f.write(f"Accuracy: {metrics.accuracy:.4f}\n")
                        if metrics.f1_score:
                            f.write(f"F1 Score: {metrics.f1_score:.4f}\n")
                        if metrics.r2_score:
                            f.write(f"R2 Score: {metrics.r2_score:.4f}\n")
                    f.write(f"\n")

                if result.post_hpo_metrics:
                    f.write(f"Post-HPO Evaluation:\n")
                    f.write(f"--------------------\n")
                    f.write(f"Grade: {result.post_hpo_metrics.performance_grade}\n")
                    f.write(f"Passed: {result.post_hpo_metrics.evaluation_passed}\n")
                    if result.post_hpo_metrics.post_hpo_metrics:
                        metrics = result.post_hpo_metrics.post_hpo_metrics
                        if metrics.accuracy:
                            f.write(f"Accuracy: {metrics.accuracy:.4f}\n")
                        if metrics.f1_score:
                            f.write(f"F1 Score: {metrics.f1_score:.4f}\n")
                        if metrics.r2_score:
                            f.write(f"R2 Score: {metrics.r2_score:.4f}\n")
                    f.write(f"\n")

                if result.validation_result:
                    f.write(f"Validation Results:\n")
                    f.write(f"-------------------\n")
                    f.write(f"Grade: {result.validation_result.validation_grade}\n")
                    f.write(f"Passed: {result.validation_result.validation_passed}\n")
                    f.write(f"Stable: {result.validation_result.is_stable}\n")
                    f.write(f"Stability Grade: {result.validation_result.stability_grade}\n\n")

                if result.persistence_result:
                    f.write(f"Persistence Results:\n")
                    f.write(f"--------------------\n")
                    f.write(f"Success: {result.persistence_result.success}\n")
                    f.write(f"File Path: {result.persistence_result.file_path}\n")
                    f.write(f"File Size: {result.persistence_result.file_size} bytes\n")
                    f.write(f"Version: {result.persistence_result.version}\n\n")

                if result.error_message:
                    f.write(f"Error Message:\n")
                    f.write(f"--------------\n")
                    f.write(f"{result.error_message}\n")

            self.logger.info(f"📊 Training report generated: {report_path}")

        except Exception as e:
            self.logger.exception(f"💥 Error generating training report: {e}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get training status and configuration."""
        return {
            'config': self.config.__dict__,
            'evaluator_available': self.evaluator is not None,
            'validator_available': self.validator is not None,
            'persistence_available': self.persistence is not None,
            'multi_timeframe_available': self.multi_timeframe_trainer is not None,
            'intensity_scaling_applied': get_intensity_from_environment() < 1.0
        }
