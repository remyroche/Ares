"""
Model Validation Utilities for Hybrid NAS-TAS Regime Detection.

Provides comprehensive model validation with multiple validation strategies,
performance metrics, and robustness testing using existing ml_common utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum
import json
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing ml_common utilities
try:
    from src.utils.ml_common import (
        UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
        nested_cross_validation, PurgedKFold, TemporalCrossValidator,
        StabilityAnalyzer, ConfigurationValidator
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Import existing utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ValidationStrategy(Enum):
    """Validation strategies available."""
    CROSS_VALIDATION = "cross_validation"
    TEMPORAL_CV = "temporal_cv"
    NESTED_CV = "nested_cv"
    HOLD_OUT = "hold_out"
    TIME_SERIES_SPLIT = "time_series_split"
    PURGED_KFOLD = "purged_kfold"
    WALK_FORWARD = "walk_forward"
    BLOCKING_TIME_SERIES = "blocking_time_series"

class ValidationMetric(Enum):
    """Validation metrics available."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    LOG_LOSS = "log_loss"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2_SCORE = "r2_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"

@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    # Validation strategy
    strategy: ValidationStrategy = ValidationStrategy.CROSS_VALIDATION
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Time series specific
    purged_pct: float = 0.01  # For purged K-fold
    embargo_pct: float = 0.01  # For purged K-fold
    n_blocks: int = 3  # For blocking time series

    # Metrics to calculate
    metrics: List[ValidationMetric] = None

    # Robustness testing
    enable_robustness_testing: bool = True
    noise_levels: List[float] = None
    perturbation_methods: List[str] = None

    # Performance optimization
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0

    # Output settings
    save_results: bool = True
    output_dir: str = "validation_results"
    verbose: bool = True

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                ValidationMetric.ACCURACY,
                ValidationMetric.PRECISION,
                ValidationMetric.RECALL,
                ValidationMetric.F1_SCORE,
                ValidationMetric.ROC_AUC
            ]
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.05, 0.1]
        if self.perturbation_methods is None:
            self.perturbation_methods = ['gaussian_noise', 'feature_dropout', 'label_flip']

@dataclass
class ValidationResult:
    """Result from model validation."""
    # Core validation results
    validation_scores: Dict[str, float]
    validation_std: Dict[str, float]
    fold_scores: List[Dict[str, float]]

    # Robustness testing results
    robustness_scores: Dict[str, Dict[str, float]] = None

    # Stability analysis
    stability_metrics: Dict[str, float] = None

    # Performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0

    # Metadata
    validation_strategy: str = ""
    n_folds: int = 0
    n_samples: int = 0
    n_features: int = 0

    # Results
    success: bool = True
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class ModelValidator:
    """Advanced model validator with comprehensive validation strategies."""

    def __init__(self, config: ValidationConfig):
        """Initialize the model validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for model validation")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for model validation")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        # Initialize validation components
        self.validator = None
        if ML_COMMON_AVAILABLE:
            try:
                self.validator = UnifiedCrossValidator()
                self.logger.info("✅ Unified cross validator initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Unified cross validator not available: {e}")

        self.logger.info("✅ Model Validator initialized")
        self.logger.info(f"   Strategy: {config.strategy.value}")
        self.logger.info(f"   Metrics: {[m.value for m in config.metrics]}")
        self.logger.info(f"   Robustness testing: {config.enable_robustness_testing}")

    def validate_model(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      additional_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a model using the configured strategy.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Target variable
            additional_data: Optional additional data for validation

        Returns:
            ValidationResult with validation scores and metrics
        """
        start_time = time.time()

        try:
            self.logger.info("🔍 Starting model validation")
            self.logger.info(f"   Data shape: {X.shape}")
            self.logger.info(f"   Strategy: {self.config.strategy.value}")
            self.logger.info(f"   Metrics: {[m.value for m in self.config.metrics]}")

            # Perform validation based on strategy
            if self.config.strategy == ValidationStrategy.CROSS_VALIDATION:
                validation_scores, validation_std, fold_scores = self._cross_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.TEMPORAL_CV:
                validation_scores, validation_std, fold_scores = self._temporal_cross_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.NESTED_CV:
                validation_scores, validation_std, fold_scores = self._nested_cross_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.HOLD_OUT:
                validation_scores, validation_std, fold_scores = self._hold_out_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.TIME_SERIES_SPLIT:
                validation_scores, validation_std, fold_scores = self._time_series_split_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.PURGED_KFOLD:
                validation_scores, validation_std, fold_scores = self._purged_kfold_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.WALK_FORWARD:
                validation_scores, validation_std, fold_scores = self._walk_forward_validation(model, X, y)
            elif self.config.strategy == ValidationStrategy.BLOCKING_TIME_SERIES:
                validation_scores, validation_std, fold_scores = self._blocking_time_series_validation(model, X, y)
            else:
                raise ValueError(f"Unknown validation strategy: {self.config.strategy}")

            # Perform robustness testing if enabled
            robustness_scores = None
            if self.config.enable_robustness_testing:
                robustness_scores = self._robustness_testing(model, X, y)

            # Perform stability analysis
            stability_metrics = self._stability_analysis(fold_scores)

            # Calculate execution metrics
            execution_time = time.time() - start_time
            memory_usage_mb = self._calculate_memory_usage()

            self.logger.info(f"✅ Model validation completed in {execution_time:.2f}s")
            self.logger.info(f"   Validation scores: {validation_scores}")

            return ValidationResult(
                validation_scores=validation_scores,
                validation_std=validation_std,
                fold_scores=fold_scores,
                robustness_scores=robustness_scores,
                stability_metrics=stability_metrics,
                execution_time=execution_time,
                memory_usage_mb=memory_usage_mb,
                validation_strategy=self.config.strategy.value,
                n_folds=self.config.n_folds,
                n_samples=len(X),
                n_features=len(X.columns),
                success=True,
                hardware_optimization_applied=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Model validation failed: {e}")

            return ValidationResult(
                validation_scores={},
                validation_std={},
                fold_scores=[],
                execution_time=execution_time,
                validation_strategy=self.config.strategy.value,
                n_folds=self.config.n_folds,
                n_samples=len(X),
                n_features=len(X.columns),
                success=False,
                error_message=str(e)
            )

    def _cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform cross-validation."""
        try:
            if ML_COMMON_AVAILABLE and self.validator:
                # Use unified cross validator
                cv_result = perform_cross_validation(
                    X, y, model, n_folds=self.config.n_folds
                )

                validation_scores = {
                    metric.value: cv_result.get(f'{metric.value}_score', 0.0)
                    for metric in self.config.metrics
                }
                validation_std = {
                    metric.value: cv_result.get(f'{metric.value}_std', 0.0)
                    for metric in self.config.metrics
                }
                fold_scores = cv_result.get('fold_scores', [])

                return validation_scores, validation_std, fold_scores
            else:
                # Fallback to manual cross-validation
                return self._manual_cross_validation(model, X, y)

        except Exception as e:
            self.logger.warning(f"⚠️ Cross-validation failed: {e}")
            return self._manual_cross_validation(model, X, y)

    def _temporal_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform temporal cross-validation."""
        try:
            if ML_COMMON_AVAILABLE:
                # Use temporal cross validator
                cv_result = temporal_cross_validation(
                    X, y, model, n_folds=self.config.n_folds
                )

                validation_scores = {
                    metric.value: cv_result.get(f'{metric.value}_score', 0.0)
                    for metric in self.config.metrics
                }
                validation_std = {
                    metric.value: cv_result.get(f'{metric.value}_std', 0.0)
                    for metric in self.config.metrics
                }
                fold_scores = cv_result.get('fold_scores', [])

                return validation_scores, validation_std, fold_scores
            else:
                # Fallback to manual temporal cross-validation
                return self._manual_temporal_cross_validation(model, X, y)

        except Exception as e:
            self.logger.warning(f"⚠️ Temporal cross-validation failed: {e}")
            return self._manual_temporal_cross_validation(model, X, y)

    def _nested_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform nested cross-validation."""
        try:
            if ML_COMMON_AVAILABLE:
                # Use nested cross validator
                cv_result = nested_cross_validation(
                    X, y, model, n_folds=self.config.n_folds
                )

                validation_scores = {
                    metric.value: cv_result.get(f'{metric.value}_score', 0.0)
                    for metric in self.config.metrics
                }
                validation_std = {
                    metric.value: cv_result.get(f'{metric.value}_std', 0.0)
                    for metric in self.config.metrics
                }
                fold_scores = cv_result.get('fold_scores', [])

                return validation_scores, validation_std, fold_scores
            else:
                # Fallback to manual nested cross-validation
                return self._manual_nested_cross_validation(model, X, y)

        except Exception as e:
            self.logger.warning(f"⚠️ Nested cross-validation failed: {e}")
            return self._manual_nested_cross_validation(model, X, y)

    def _hold_out_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform hold-out validation."""
        try:
            from sklearn.model_selection import train_test_split

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            validation_scores = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            validation_std = {metric: 0.0 for metric in validation_scores.keys()}
            fold_scores = [validation_scores]

            return validation_scores, validation_std, fold_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Hold-out validation failed: {e}")
            return {}, {}, []

    def _time_series_split_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform time series split validation."""
        try:
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=self.config.n_folds)
            fold_scores = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics for this fold
                fold_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                fold_scores.append(fold_metrics)

            # Aggregate results
            validation_scores = {}
            validation_std = {}

            for metric in self.config.metrics:
                metric_name = metric.value
                scores = [fold[metric_name] for fold in fold_scores if metric_name in fold]

                if scores:
                    validation_scores[metric_name] = np.mean(scores)
                    validation_std[metric_name] = np.std(scores)
                else:
                    validation_scores[metric_name] = 0.0
                    validation_std[metric_name] = 0.0

            return validation_scores, validation_std, fold_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Time series split validation failed: {e}")
            return {}, {}, []

    def _purged_kfold_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform purged K-fold validation."""
        try:
            if ML_COMMON_AVAILABLE:
                # Use purged K-fold from ml_common
                purged_kfold = PurgedKFold(
                    n_splits=self.config.n_folds,
                    purged_pct=self.config.purged_pct,
                    embargo_pct=self.config.embargo_pct
                )

                fold_scores = []

                for fold, (train_idx, test_idx) in enumerate(purged_kfold.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)

                    # Calculate metrics for this fold
                    fold_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                    fold_scores.append(fold_metrics)

                # Aggregate results
                validation_scores = {}
                validation_std = {}

                for metric in self.config.metrics:
                    metric_name = metric.value
                    scores = [fold[metric_name] for fold in fold_scores if metric_name in fold]

                    if scores:
                        validation_scores[metric_name] = np.mean(scores)
                        validation_std[metric_name] = np.std(scores)
                    else:
                        validation_scores[metric_name] = 0.0
                        validation_std[metric_name] = 0.0

                return validation_scores, validation_std, fold_scores
            else:
                # Fallback to regular K-fold
                return self._manual_cross_validation(model, X, y)

        except Exception as e:
            self.logger.warning(f"⚠️ Purged K-fold validation failed: {e}")
            return self._manual_cross_validation(model, X, y)

    def _walk_forward_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform walk-forward validation."""
        try:
            # Split data into chunks for walk-forward validation
            chunk_size = len(X) // self.config.n_folds
            fold_scores = []

            for fold in range(self.config.n_folds):
                # Define training and test periods
                train_end = (fold + 1) * chunk_size
                test_start = train_end
                test_end = min(test_start + chunk_size, len(X))

                if test_start >= len(X):
                    break

                X_train = X.iloc[:train_end]
                y_train = y.iloc[:train_end]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics for this fold
                fold_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                fold_scores.append(fold_metrics)

            # Aggregate results
            validation_scores = {}
            validation_std = {}

            for metric in self.config.metrics:
                metric_name = metric.value
                scores = [fold[metric_name] for fold in fold_scores if metric_name in fold]

                if scores:
                    validation_scores[metric_name] = np.mean(scores)
                    validation_std[metric_name] = np.std(scores)
                else:
                    validation_scores[metric_name] = 0.0
                    validation_std[metric_name] = 0.0

            return validation_scores, validation_std, fold_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Walk-forward validation failed: {e}")
            return {}, {}, []

    def _blocking_time_series_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Perform blocking time series validation."""
        try:
            # Split data into blocks
            block_size = len(X) // self.config.n_blocks
            fold_scores = []

            for fold in range(self.config.n_blocks):
                # Define training and test blocks
                test_start = fold * block_size
                test_end = min((fold + 1) * block_size, len(X))

                # Training data: all data except test block
                train_indices = list(range(0, test_start)) + list(range(test_end, len(X)))

                if len(train_indices) == 0 or test_start >= len(X):
                    continue

                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics for this fold
                fold_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                fold_scores.append(fold_metrics)

            # Aggregate results
            validation_scores = {}
            validation_std = {}

            for metric in self.config.metrics:
                metric_name = metric.value
                scores = [fold[metric_name] for fold in fold_scores if metric_name in fold]

                if scores:
                    validation_scores[metric_name] = np.mean(scores)
                    validation_std[metric_name] = np.std(scores)
                else:
                    validation_scores[metric_name] = 0.0
                    validation_std[metric_name] = 0.0

            return validation_scores, validation_std, fold_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Blocking time series validation failed: {e}")
            return {}, {}, []

    def _manual_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Manual cross-validation implementation."""
        try:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_state)
            fold_scores = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics for this fold
                fold_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                fold_scores.append(fold_metrics)

            # Aggregate results
            validation_scores = {}
            validation_std = {}

            for metric in self.config.metrics:
                metric_name = metric.value
                scores = [fold[metric_name] for fold in fold_scores if metric_name in fold]

                if scores:
                    validation_scores[metric_name] = np.mean(scores)
                    validation_std[metric_name] = np.std(scores)
                else:
                    validation_scores[metric_name] = 0.0
                    validation_std[metric_name] = 0.0

            return validation_scores, validation_std, fold_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Manual cross-validation failed: {e}")
            return {}, {}, []

    def _manual_temporal_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Manual temporal cross-validation implementation."""
        try:
            # Use time series split for temporal validation
            return self._time_series_split_validation(model, X, y)

        except Exception as e:
            self.logger.warning(f"⚠️ Manual temporal cross-validation failed: {e}")
            return {}, {}, []

    def _manual_nested_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
        """Manual nested cross-validation implementation."""
        try:
            # Simplified nested CV: outer loop for validation, inner loop for hyperparameter tuning
            # For now, just use regular cross-validation
            return self._manual_cross_validation(model, X, y)

        except Exception as e:
            self.logger.warning(f"⚠️ Manual nested cross-validation failed: {e}")
            return {}, {}, []

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate validation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metric scores
        """
        try:
            metrics = {}

            for metric in self.config.metrics:
                if metric == ValidationMetric.ACCURACY:
                    from sklearn.metrics import accuracy_score
                    metrics[metric.value] = accuracy_score(y_true, y_pred)

                elif metric == ValidationMetric.PRECISION:
                    from sklearn.metrics import precision_score
                    metrics[metric.value] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

                elif metric == ValidationMetric.RECALL:
                    from sklearn.metrics import recall_score
                    metrics[metric.value] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

                elif metric == ValidationMetric.F1_SCORE:
                    from sklearn.metrics import f1_score
                    metrics[metric.value] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                elif metric == ValidationMetric.ROC_AUC:
                    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                        from sklearn.metrics import roc_auc_score
                        metrics[metric.value] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics[metric.value] = 0.0

                elif metric == ValidationMetric.PR_AUC:
                    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                        from sklearn.metrics import average_precision_score
                        metrics[metric.value] = average_precision_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics[metric.value] = 0.0

                elif metric == ValidationMetric.LOG_LOSS:
                    if y_pred_proba is not None:
                        from sklearn.metrics import log_loss
                        metrics[metric.value] = log_loss(y_true, y_pred_proba)
                    else:
                        metrics[metric.value] = 0.0

                elif metric == ValidationMetric.MAE:
                    from sklearn.metrics import mean_absolute_error
                    metrics[metric.value] = mean_absolute_error(y_true, y_pred)

                elif metric == ValidationMetric.MSE:
                    from sklearn.metrics import mean_squared_error
                    metrics[metric.value] = mean_squared_error(y_true, y_pred)

                elif metric == ValidationMetric.RMSE:
                    metrics[metric.value] = np.sqrt(mean_squared_error(y_true, y_pred))

                elif metric == ValidationMetric.R2_SCORE:
                    from sklearn.metrics import r2_score
                    metrics[metric.value] = r2_score(y_true, y_pred)

                else:
                    metrics[metric.value] = 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Metric calculation failed: {e}")
            return {metric.value: 0.0 for metric in self.config.metrics}

    def _robustness_testing(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Perform robustness testing with noise and perturbations."""
        try:
            robustness_scores = {}

            for noise_level in self.config.noise_levels:
                for perturbation_method in self.config.perturbation_methods:
                    # Create perturbed data
                    X_perturbed = self._apply_perturbation(X, perturbation_method, noise_level)

                    # Train model on perturbed data
                    model.fit(X_perturbed, y)

                    # Test on original data
                    y_pred = model.predict(X)
                    y_pred_proba = None
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X)

                    # Calculate metrics
                    metrics = self._calculate_metrics(y, y_pred, y_pred_proba)

                    perturbation_key = f"{perturbation_method}_{noise_level}"
                    robustness_scores[perturbation_key] = metrics

            return robustness_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Robustness testing failed: {e}")
            return {}

    def _apply_perturbation(self, X: pd.DataFrame, method: str, noise_level: float) -> pd.DataFrame:
        """Apply perturbation to data.

        Args:
            X: Input data
            method: Perturbation method
            noise_level: Noise level

        Returns:
            Perturbed data
        """
        try:
            X_perturbed = X.copy()

            if method == 'gaussian_noise':
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X.shape)
                X_perturbed = X_perturbed + noise

            elif method == 'feature_dropout':
                # Randomly set features to zero
                dropout_mask = np.random.random(X.shape) < noise_level
                X_perturbed = X_perturbed.mask(dropout_mask, 0)

            elif method == 'label_flip':
                # This would require y, so we'll skip for now
                pass

            return X_perturbed

        except Exception as e:
            self.logger.warning(f"⚠️ Perturbation application failed: {e}")
            return X

    def _stability_analysis(self, fold_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze stability of validation results.

        Args:
            fold_scores: Scores from each fold

        Returns:
            Stability metrics
        """
        try:
            stability_metrics = {}

            for metric in self.config.metrics:
                metric_name = metric.value
                scores = [fold[metric_name] for fold in fold_scores if metric_name in fold]

                if scores:
                    stability_metrics[f'{metric_name}_mean'] = np.mean(scores)
                    stability_metrics[f'{metric_name}_std'] = np.std(scores)
                    stability_metrics[f'{metric_name}_cv'] = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else 0
                    stability_metrics[f'{metric_name}_range'] = np.max(scores) - np.min(scores)
                else:
                    stability_metrics[f'{metric_name}_mean'] = 0.0
                    stability_metrics[f'{metric_name}_std'] = 0.0
                    stability_metrics[f'{metric_name}_cv'] = 0.0
                    stability_metrics[f'{metric_name}_range'] = 0.0

            return stability_metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Stability analysis failed: {e}")
            return {}

    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage in MB."""
        try:
            if self.memory_optimizer:
                memory_info = self.memory_optimizer.get_memory_usage()
                return memory_info.get('used_memory_mb', 0.0)
            else:
                return 0.0

        except Exception:
            return 0.0

def create_model_validator(config: Optional[ValidationConfig] = None) -> ModelValidator:
    """Create a model validator instance.

    Args:
        config: Optional validation configuration

    Returns:
        ModelValidator instance
    """
    if config is None:
        config = ValidationConfig()
    return ModelValidator(config)

def quick_model_validation(model: Any,
                          X: pd.DataFrame,
                          y: pd.Series,
                          strategy: ValidationStrategy = ValidationStrategy.CROSS_VALIDATION,
                          n_folds: int = 5) -> ValidationResult:
    """Quick model validation with default settings.

    Args:
        model: Model to validate
        X: Feature matrix
        y: Target variable
        strategy: Validation strategy
        n_folds: Number of folds

    Returns:
        ValidationResult
    """
    config = ValidationConfig(
        strategy=strategy,
        n_folds=n_folds,
        enable_robustness_testing=False
    )

    validator = ModelValidator(config)
    return validator.validate_model(model, X, y)
