"""
Shared Validation Utilities for Hybrid NAS-TAS Regime Detection.

Provides common validation utilities that can be used by both NAS and TAS systems
for model validation, performance assessment, and robustness testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum
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

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of validation available."""
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
class SharedValidationConfig:
    """Configuration for shared validation utilities."""
    # Validation type
    validation_type: ValidationType = ValidationType.CROSS_VALIDATION
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # Time series specific
    purged_pct: float = 0.01
    embargo_pct: float = 0.01
    n_blocks: int = 3

    # Metrics to calculate
    metrics: List[ValidationMetric] = None

    # Robustness testing
    enable_robustness_testing: bool = True
    noise_levels: List[float] = None
    perturbation_methods: List[str] = None

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
            self.perturbation_methods = ['gaussian_noise', 'feature_dropout']

@dataclass
class SharedValidationResult:
    """Result from shared validation."""
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

    # Metadata
    validation_type: str = ""
    n_folds: int = 0
    n_samples: int = 0
    n_features: int = 0

    # Results
    success: bool = True
    error_message: Optional[str] = None

class SharedValidator:
    """Shared validator for both NAS and TAS systems."""

    def __init__(self, config: SharedValidationConfig):
        """Initialize the shared validator.

        Args:
            config: Shared validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize validation components
        self.validator = None
        if ML_COMMON_AVAILABLE:
            try:
                self.validator = UnifiedCrossValidator()
                self.logger.info("✅ Unified cross validator initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Unified cross validator not available: {e}")

        self.logger.info("✅ Shared Validator initialized")
        self.logger.info(f"   Type: {config.validation_type.value}")
        self.logger.info(f"   Metrics: {[m.value for m in config.metrics]}")
        self.logger.info(f"   Robustness testing: {config.enable_robustness_testing}")

    def validate(self,
                 model: Any,
                 X: pd.DataFrame,
                 y: pd.Series,
                 additional_data: Optional[Dict[str, Any]] = None) -> SharedValidationResult:
        """Validate a model using the configured strategy.

        Args:
            model: Model to validate
            X: Feature matrix
            y: Target variable
            additional_data: Optional additional data for validation

        Returns:
            SharedValidationResult with validation scores and metrics
        """
        start_time = time.time()

        try:
            self.logger.info("🔍 Starting shared validation")
            self.logger.info(f"   Data shape: {X.shape}")
            self.logger.info(f"   Type: {self.config.validation_type.value}")
            self.logger.info(f"   Metrics: {[m.value for m in self.config.metrics]}")

            # Perform validation based on type
            if self.config.validation_type == ValidationType.CROSS_VALIDATION:
                validation_scores, validation_std, fold_scores = self._cross_validation(model, X, y)
            elif self.config.validation_type == ValidationType.TEMPORAL_CV:
                validation_scores, validation_std, fold_scores = self._temporal_cross_validation(model, X, y)
            elif self.config.validation_type == ValidationType.NESTED_CV:
                validation_scores, validation_std, fold_scores = self._nested_cross_validation(model, X, y)
            elif self.config.validation_type == ValidationType.HOLD_OUT:
                validation_scores, validation_std, fold_scores = self._hold_out_validation(model, X, y)
            elif self.config.validation_type == ValidationType.TIME_SERIES_SPLIT:
                validation_scores, validation_std, fold_scores = self._time_series_split_validation(model, X, y)
            elif self.config.validation_type == ValidationType.PURGED_KFOLD:
                validation_scores, validation_std, fold_scores = self._purged_kfold_validation(model, X, y)
            elif self.config.validation_type == ValidationType.WALK_FORWARD:
                validation_scores, validation_std, fold_scores = self._walk_forward_validation(model, X, y)
            elif self.config.validation_type == ValidationType.BLOCKING_TIME_SERIES:
                validation_scores, validation_std, fold_scores = self._blocking_time_series_validation(model, X, y)
            else:
                raise ValueError(f"Unknown validation type: {self.config.validation_type}")

            # Perform robustness testing if enabled
            robustness_scores = None
            if self.config.enable_robustness_testing:
                robustness_scores = self._robustness_testing(model, X, y)

            # Perform stability analysis
            stability_metrics = self._stability_analysis(fold_scores)

            # Calculate execution time
            execution_time = time.time() - start_time

            self.logger.info(f"✅ Shared validation completed in {execution_time:.2f}s")
            self.logger.info(f"   Validation scores: {validation_scores}")

            return SharedValidationResult(
                validation_scores=validation_scores,
                validation_std=validation_std,
                fold_scores=fold_scores,
                robustness_scores=robustness_scores,
                stability_metrics=stability_metrics,
                execution_time=execution_time,
                validation_type=self.config.validation_type.value,
                n_folds=self.config.n_folds,
                n_samples=len(X),
                n_features=len(X.columns),
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Shared validation failed: {e}")

            return SharedValidationResult(
                validation_scores={},
                validation_std={},
                fold_scores=[],
                execution_time=execution_time,
                validation_type=self.config.validation_type.value,
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
        """Calculate validation metrics."""
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
        """Apply perturbation to data."""
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

            return X_perturbed

        except Exception as e:
            self.logger.warning(f"⚠️ Perturbation application failed: {e}")
            return X

    def _stability_analysis(self, fold_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze stability of validation results."""
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

def create_shared_validator(config: Optional[SharedValidationConfig] = None) -> SharedValidator:
    """Create a shared validator instance.

    Args:
        config: Optional shared validation configuration

    Returns:
        SharedValidator instance
    """
    if config is None:
        config = SharedValidationConfig()
    return SharedValidator(config)

def quick_shared_validation(model: Any,
                           X: pd.DataFrame,
                           y: pd.Series,
                           validation_type: ValidationType = ValidationType.CROSS_VALIDATION,
                           n_folds: int = 5) -> SharedValidationResult:
    """Quick shared validation with default settings.

    Args:
        model: Model to validate
        X: Feature matrix
        y: Target variable
        validation_type: Validation type
        n_folds: Number of folds

    Returns:
        SharedValidationResult
    """
    config = SharedValidationConfig(
        validation_type=validation_type,
        n_folds=n_folds,
        enable_robustness_testing=False
    )

    validator = SharedValidator(config)
    return validator.validate(model, X, y)
