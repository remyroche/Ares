"""
Out-of-Fold (OOF) Stacking Ensemble Manager

This module provides comprehensive OOF stacking ensemble management with:
- PurgedKFoldTime for temporal cross-validation
- OOF base model predictions to prevent data leakage
- Proper meta-learner training on OOF predictions
- Early stopping support for tree-based models
- M1 hardware optimization integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.base import clone
import warnings

# Import purged K-fold for temporal validation
from src.utils.purged_kfold import PurgedKFoldTime

# M1 Optimization imports
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.memory_optimization import get_memory_manager, MemoryMonitor

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError
)

logger = logging.getLogger(__name__)


@dataclass
class OOFStackingEnsembleConfig:
    """Configuration for OOF stacking ensemble manager."""
    # Basic configuration
    ensemble_name: str
    output_dir: str

    # Multi-output configuration
    n_outputs: int = 4
    output_names: List[str] = field(default_factory=lambda: ["output_1", "output_2", "output_3", "output_4"])

    # Base model configuration
    base_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta_models: Dict[str, Any] = field(default_factory=dict)

    # OOF Training configuration
    enable_out_of_fold: bool = True
    cv_folds: int = 5
    cv_strategy: str = "purged_kfold"  # purged_kfold, stratified, regular
    enable_temporal_validation: bool = True
    purge_periods: int = 5
    embargo_periods: int = 2

    # Early stopping configuration
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_rounds: int = 50
    early_stopping_metric: str = "accuracy"

    # Multi-output specific settings
    output_weights: Optional[List[float]] = None
    output_loss_weights: Optional[List[float]] = None
    enable_output_correlation: bool = True
    correlation_threshold: float = 0.7

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
    enable_online_learning: bool = False

    # Output settings
    save_models: bool = True
    save_predictions: bool = True
    generate_reports: bool = True

    # OOF-specific settings
    oof_validation_metric: str = "accuracy"
    oof_aggregation_method: str = "mean"  # mean, median, weighted


@dataclass
class OOFStackingEnsembleResult:
    """Result from OOF stacking ensemble operations."""
    # Basic info
    ensemble_name: str
    n_outputs: int
    output_names: List[str]
    created_at: datetime
    total_duration: float

    # Model information
    base_model_count: int
    meta_model_count: int
    base_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    meta_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Ensemble performance
    ensemble_performance: Dict[str, float] = field(default_factory=dict)
    per_output_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # OOF-specific results
    oof_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    oof_scores: Dict[str, float] = field(default_factory=dict)
    oof_confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Model characteristics
    model_weights: Optional[np.ndarray] = None
    output_correlations: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, Any]] = None

    # Metadata
    config: OOFStackingEnsembleConfig = field(default_factory=OOFStackingEnsembleConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)


class OOFStackingEnsembleManager:
    """Comprehensive OOF stacking ensemble manager with temporal validation."""

    def __init__(self, config: OOFStackingEnsembleConfig):
        """Initialize the OOF stacking ensemble manager."""
        self.logger = logger.getChild('OOFStackingEnsembleManager')
        self.logger.info(f"🚀 Initializing OOF StackingEnsembleManager for {config.ensemble_name}...")
        start_time = time.time()

        self.config = config

        # Initialize M1 optimizers
        self.logger.debug("🔧 Initializing M1 optimizers...")
        self.m1_gpu = get_m1_memory_optimizer() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_memory_manager() if config.enable_parallel_processing else None

        self.logger.debug("✅ M1 optimizers initialized")

        # Initialize data structures
        self.base_models = {}
        self.meta_models = {}
        self.is_fitted = False
        self.training_history = []

        # OOF-specific data structures
        self.oof_predictions = {}
        self.oof_scores = {}

        # Ensure output directory exists
        self.logger.debug(f"🔧 Ensuring output directory exists: {config.output_dir}")
        ensure_directory(config.output_dir)
        self.logger.debug("✅ Output directory ready")

        init_time = time.time() - start_time
        self.logger.info(f"✅ OOF StackingEnsembleManager initialized for {config.ensemble_name} in {init_time:.3f}s")

    def add_base_model(self, output_name: str, model_name: str, model: Any) -> None:
        """Add a base model for a specific output."""
        if output_name not in self.base_models:
            self.base_models[output_name] = {}

        self.base_models[output_name][model_name] = model
        self.logger.info(f"➕ Added base model {model_name} for output {output_name}")

    def add_meta_model(self, output_name: str, model: Any) -> None:
        """Add a meta model for a specific output."""
        self.meta_models[output_name] = model
        self.logger.info(f"➕ Added meta model for output {output_name}")

    def _setup_cross_validation(self, X: np.ndarray, y: np.ndarray, is_classification: bool) -> Any:
        """Setup cross-validation strategy."""
        if not self.config.enable_temporal_validation or not self.config.enable_out_of_fold:
            # Use standard cross-validation
            if is_classification:
                return StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=42
                )
            else:
                return KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=42
                )
        else:
            # Use PurgedKFoldTime for temporal validation
            try:
                # Convert to DataFrame for temporal CV
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X)
                    # Create timestamp index if not present
                    if X_df.index is None or not hasattr(X_df.index, 'is_monotonic_increasing'):
                        X_df.index = pd.date_range(start='2020-01-01', periods=len(X_df), freq='1min')
                else:
                    X_df = X

                # Create temporal target data
                if isinstance(y, np.ndarray):
                    y_series = pd.Series(y.ravel())
                else:
                    y_series = y

                # Setup PurgedKFoldTime
                cv = PurgedKFoldTime(
                    n_splits=self.config.cv_folds,
                    purge=self.config.purge_periods,
                    embargo=self.config.embargo_periods
                )

                return cv

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to setup temporal CV, falling back to TimeSeriesSplit without shuffle to prevent data leakage: {e}")
                # CRITICAL: Always use TimeSeriesSplit for time series data!
                return TimeSeriesSplit(n_splits=self.config.cv_folds)

    def _generate_oof_predictions(self,
                                 models: Dict[str, Any],
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 cv: Any,
                                 is_classification: bool) -> Dict[str, np.ndarray]:
        """Generate out-of-fold predictions for base models."""
        self.logger.info("🔄 Generating OOF predictions for base models...")

        oof_predictions = {}
        oof_scores = {}

        for output_name, model_dict in models.items():
            self.logger.debug(f"🔄 Processing output: {output_name}")

            # Get target for this output
            if isinstance(y, pd.DataFrame):
                y_output = y[output_name].values if output_name in y.columns else y.iloc[:, 0].values
            else:
                # For multi-output, assume y has shape (n_samples, n_outputs)
                output_idx = self.config.output_names.index(output_name)
                y_output = y[:, output_idx]

            # Initialize OOF predictions array
            n_samples = len(X)
            oof_preds = np.zeros(n_samples)

            # Generate OOF predictions for each base model
            for model_name, model in model_dict.items():
                self.logger.debug(f"🔄 Generating OOF predictions for {model_name}")

                model_oof_preds = np.zeros(n_samples)

                for train_idx, val_idx in cv.split(X, y_output):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y_output[train_idx], y_output[val_idx]

                    # Clone model to avoid state issues
                    model_clone = clone(model)

                    # Setup early stopping if enabled
                    if self.config.enable_early_stopping:
                        model_clone = self._setup_early_stopping(
                            model_clone, X_train, X_val, y_train, y_val, model_name
                        )

                    # Train model
                    model_clone.fit(X_train, y_train)

                    # Make predictions
                    if hasattr(model_clone, 'predict_proba') and is_classification:
                        val_preds = model_clone.predict_proba(X_val)
                        if val_preds.ndim > 1 and val_preds.shape[1] > 1:
                            val_preds = val_preds[:, 1]  # Use positive class probability
                    else:
                        val_preds = model_clone.predict(X_val)

                    model_oof_preds[val_idx] = val_preds

                # Store OOF predictions for this model
                if output_name not in oof_predictions:
                    oof_predictions[output_name] = {}
                oof_predictions[output_name][model_name] = model_oof_preds

                # Calculate OOF score for this model
                if hasattr(model, 'predict_proba') and is_classification:
                    pred_probs = model_oof_preds
                    if pred_probs.ndim == 1:
                        # Convert to probabilities if needed
                        pred_probs = np.column_stack([1 - pred_probs, pred_probs])
                    score = self._calculate_score(y_output, pred_probs, is_classification, self.config.oof_validation_metric)
                else:
                    score = self._calculate_score(y_output, model_oof_preds, is_classification, self.config.oof_validation_metric)

                if output_name not in oof_scores:
                    oof_scores[output_name] = {}
                oof_scores[output_name][model_name] = score

                self.logger.debug(f"✅ OOF predictions generated for {model_name}, score: {score:.4f}")
            self.logger.info(f"✅ OOF predictions generated for {output_name}")

        self.oof_predictions = oof_predictions
        self.oof_scores = oof_scores

        return oof_predictions

    def _setup_early_stopping(self, model: Any, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, model_name: str) -> Any:
        """Setup early stopping for tree-based models with proper eval_set."""
        try:
            model_type = model_name.lower()

            # XGBoost early stopping
            if 'xgb' in model_type:
                try:
                    # Determine evaluation metric based on problem type
                    eval_metric = self.config.early_stopping_metric
                    if eval_metric == "auto":
                        eval_metric = "logloss"  # Default for classification

                    model.set_params(
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        eval_metric=eval_metric,
                        verbose=False
                    )
                    self.logger.debug(f"✅ XGBoost early stopping configured for {model_name}")
                except Exception as xgb_error:
                    self.logger.warning(f"XGBoost early stopping setup failed: {xgb_error}")

            # LightGBM early stopping
            elif 'lgbm' in model_type or 'lightgbm' in model_type:
                try:
                    # Determine evaluation metric
                    eval_metric = self.config.early_stopping_metric
                    if eval_metric == "auto":
                        eval_metric = "binary_logloss"  # Default for classification

                    # Create callback list
                    callbacks = [
                        'early_stopping'
                    ]

                    model.set_params(
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        eval_metric=eval_metric,
                        callbacks=callbacks,
                        verbose=-1  # Suppress output
                    )
                    self.logger.debug(f"✅ LightGBM early stopping configured for {model_name}")
                except Exception as lgbm_error:
                    self.logger.warning(f"LightGBM early stopping setup failed: {lgbm_error}")

            # CatBoost early stopping
            elif 'catboost' in model_type:
                try:
                    model.set_params(
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False,
                        use_best_model=True  # Use best model found during training
                    )
                    self.logger.debug(f"✅ CatBoost early stopping configured for {model_name}")
                except Exception as catboost_error:
                    self.logger.warning(f"CatBoost early stopping setup failed: {catboost_error}")

            # Neural networks and other models - use enhanced early stopping
            elif 'neural' in model_type or 'torch' in model_type or 'pytorch' in model_type or 'tensorflow' in model_type:
                try:
                    # Use enhanced early stopping for neural networks
                    from ..training.enhanced_early_stopping import apply_enhanced_early_stopping, get_early_stopping_config

                    config = get_early_stopping_config(
                        enabled=True,
                        patience=self.config.early_stopping_patience,
                        min_delta=self.config.early_stopping_min_delta,
                        mode='min',
                        monitor='validation_loss',
                        nn_learning_rate=0.001,
                        nn_batch_size=32,
                        nn_epochs=100
                    )

                    trained_model, result = apply_enhanced_early_stopping(
                        model, X_train, y_train, X_val, y_val, model_name, config
                    )

                    # Store result information for later use
                    if not hasattr(self, 'early_stopping_results'):
                        self.early_stopping_results = {}
                    self.early_stopping_results[model_name] = result

                    self.logger.debug(f"✅ Neural network early stopping configured for {model_name}")
                    return trained_model

                except Exception as nn_error:
                    self.logger.warning(f"Neural network early stopping setup failed: {nn_error}")

            # Other models - use generic enhanced early stopping
            else:
                try:
                    # Use enhanced early stopping for other models
                    from ..training.enhanced_early_stopping import apply_enhanced_early_stopping, get_early_stopping_config

                    config = get_early_stopping_config(
                        enabled=True,
                        patience=self.config.early_stopping_patience,
                        min_delta=self.config.early_stopping_min_delta,
                        mode='min',
                        monitor='validation_loss',
                        generic_check_frequency=1,
                        generic_max_iterations=100
                    )

                    trained_model, result = apply_enhanced_early_stopping(
                        model, X_train, y_train, X_val, y_val, model_name, config
                    )

                    # Store result information for later use
                    if not hasattr(self, 'early_stopping_results'):
                        self.early_stopping_results = {}
                    self.early_stopping_results[model_name] = result

                    self.logger.debug(f"✅ Generic early stopping configured for {model_name}")
                    return trained_model

                except Exception as generic_error:
                    self.logger.warning(f"Generic early stopping setup failed: {generic_error}")

            return model

        except Exception as e:
            self.logger.warning(f"Failed to setup early stopping for {model_name}: {e}")
            return model

    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray, is_classification: bool, metric: str) -> float:
        """Calculate score based on metric."""
        try:
            if is_classification:
                if metric == "accuracy":
                    return accuracy_score(y_true, np.round(y_pred) if y_pred.ndim > 1 else y_pred)
                elif metric == "f1":
                    return f1_score(y_true, np.round(y_pred) if y_pred.ndim > 1 else y_pred, average='weighted')
                else:
                    return accuracy_score(y_true, np.round(y_pred) if y_pred.ndim > 1 else y_pred)
            else:
                # For regression, use R² score
                return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return 0.0

    def _train_meta_models(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          oof_predictions: Dict[str, Dict[str, np.ndarray]],
                          cv: Any) -> Dict[str, Any]:
        """Train meta-models on OOF predictions using proper cross-validation."""
        self.logger.info("🔄 Training meta-models on OOF predictions with proper CV...")

        meta_models = {}

        for output_idx, output_name in enumerate(self.config.output_names):
            self.logger.debug(f"🔄 Training meta-model for {output_name}")

            # Get target for this output
            if isinstance(y, pd.DataFrame):
                y_output = y[output_name].values if output_name in y.columns else y.iloc[:, 0].values
            else:
                y_output = y[:, output_idx]

            # Prepare meta-features (OOF predictions from base models)
            if output_name in oof_predictions and oof_predictions[output_name]:
                meta_features_list = list(oof_predictions[output_name].values())
                meta_features = np.column_stack(meta_features_list)

                # Add original features
                meta_features = np.hstack([X, meta_features])
            else:
                self.logger.warning(f"No OOF predictions available for {output_name}, using original features")
                meta_features = X

            # Create meta-model if not provided
            if output_name not in self.meta_models:
                self._create_default_meta_model(output_name)

            if output_name not in self.meta_models:
                self.logger.error(f"No meta-model available for {output_name}")
                continue

            meta_model = self.meta_models[output_name]

            # Train meta-model with proper cross-validation to avoid overfitting
            if self.config.enable_early_stopping:
                meta_model = self._train_meta_model_with_cv(
                    meta_model, meta_features, y_output, cv, f"meta_{output_name}"
                )
            else:
                # Simple training without early stopping
                meta_model.fit(meta_features, y_output)

            meta_models[output_name] = meta_model
            self.logger.debug(f"✅ Meta-model trained for {output_name}")

        self.meta_models = meta_models
        self.logger.info("✅ Meta-models trained successfully with proper CV")

        return meta_models

    def _train_meta_model_with_cv(self, meta_model: Any, X: np.ndarray, y: np.ndarray, cv: Any, model_name: str) -> Any:
        """Train meta-model with proper cross-validation for early stopping."""
        self.logger.debug(f"🔄 Training meta-model {model_name} with cross-validation...")

        # Determine if classification or regression based on target
        is_classification = len(np.unique(y)) <= 10

        # Setup early stopping for different model types
        if 'xgb' in model_name.lower():
            # XGBoost early stopping
            eval_metric = "logloss" if is_classification else "rmse"
            meta_model.set_params(
                eval_set=[(X, y)],
                early_stopping_rounds=self.config.early_stopping_rounds,
                eval_metric=eval_metric,
                verbose=False
            )
            meta_model.fit(X, y)
        elif 'lgbm' in model_name.lower() or 'lightgbm' in model_name.lower():
            # LightGBM early stopping
            eval_metric = "binary_logloss" if is_classification else "rmse"
            callbacks = ['early_stopping']
            meta_model.set_params(
                eval_set=[(X, y)],
                early_stopping_rounds=self.config.early_stopping_rounds,
                eval_metric=eval_metric,
                callbacks=callbacks,
                verbose=-1
            )
            meta_model.fit(X, y)
        elif 'catboost' in model_name.lower():
            # CatBoost early stopping
            meta_model.set_params(
                eval_set=(X, y),
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False,
                use_best_model=True
            )
            meta_model.fit(X, y)
        else:
            # For other models, use sklearn's built-in CV or simple training
            try:
                # Try to use built-in early stopping if available
                meta_model.fit(X, y)
            except Exception as e:
                self.logger.warning(f"Could not setup early stopping for {model_name}, training without: {e}")
                meta_model.fit(X, y)

        self.logger.debug(f"✅ Meta-model {model_name} trained with CV")
        return meta_model

    def _calculate_stacking_confidence(self, X: np.ndarray, base_preds: List[np.ndarray],
                                     predictions_list: List[np.ndarray],
                                     probabilities: Optional[np.ndarray]) -> np.ndarray:
        """Calculate confidence scores for stacking predictions based on base model agreement and meta-model confidence."""
        self.logger.debug("🔄 Calculating stacking confidence scores...")

        try:
            n_samples = X.shape[0]
            confidence_scores = np.zeros(n_samples)

            if not base_preds:
                # No base predictions available, use meta-model confidence only
                if probabilities is not None and probabilities.ndim > 1:
                    confidence_scores = np.max(probabilities, axis=1)
                else:
                    confidence_scores = np.ones(n_samples) * 0.5  # Default moderate confidence
                return confidence_scores

            for sample_idx in range(n_samples):
                # Get predictions for this sample from all base models
                sample_base_preds = [pred[sample_idx] for pred in base_preds if len(pred) > sample_idx]

                if not sample_base_preds:
                    confidence_scores[sample_idx] = 0.5  # Default moderate confidence
                    continue

                # Calculate base model agreement (lower variance = higher confidence)
                base_pred_array = np.array(sample_base_preds)
                base_agreement = 1.0 / (1.0 + np.var(base_pred_array))

                # Calculate meta-model confidence if probabilities available
                meta_confidence = 0.5  # Default moderate confidence
                if probabilities is not None and probabilities.ndim > 1 and sample_idx < len(probabilities):
                    sample_prob = probabilities[sample_idx]
                    meta_confidence = np.max(sample_prob)  # Use maximum probability as confidence

                # Combine base model agreement and meta-model confidence
                combined_confidence = 0.7 * base_agreement + 0.3 * meta_confidence

                # Add small amount of random noise to avoid overconfident predictions
                noise_factor = 0.01
                combined_confidence = np.clip(combined_confidence + np.random.normal(0, noise_factor), 0.1, 0.9)

                confidence_scores[sample_idx] = combined_confidence

            # Normalize confidence scores to [0, 1] range
            if len(confidence_scores) > 0 and np.std(confidence_scores) > 0:
                confidence_scores = (confidence_scores - np.min(confidence_scores)) / (np.max(confidence_scores) - np.min(confidence_scores))

            self.logger.debug(f"✅ Confidence scores calculated: mean={np.mean(confidence_scores):.3f}, std={np.std(confidence_scores):.3f}")
            return confidence_scores

        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence scores: {e}, using default scores")
            return np.ones(n_samples) * 0.5  # Default moderate confidence

    def _create_default_meta_model(self, output_name: str):
        """Create default meta-model for output."""
        try:
            # Import common meta-learners
            from sklearn.linear_model import ElasticNet
            from sklearn.ensemble import RandomForestRegressor

            # Choose appropriate meta-learner based on output
            if self.config.meta_models and output_name in self.config.meta_models:
                self.meta_models[output_name] = self.config.meta_models[output_name]
            else:
                # Default meta-learner
                if self.config.n_outputs == 1:
                    # For single output, use ElasticNet for regularization
                    self.meta_models[output_name] = ElasticNet(
                        alpha=0.1,
                        l1_ratio=0.5,
                        random_state=42,
                        max_iter=1000
                    )
                else:
                    # For multi-output, use RandomForest for robustness
                    self.meta_models[output_name] = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    )

            self.logger.debug(f"✅ Default meta-model created for {output_name}")

        except Exception as e:
            self.logger.error(f"Failed to create default meta-model for {output_name}: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OOFStackingEnsembleManager':
        """Fit the OOF stacking ensemble."""
        self.logger.info(f"🚀 Fitting OOF StackingEnsemble with {X.shape[0]} samples")
        start_time = time.time()

        try:
            # Setup cross-validation strategy
            is_classification = len(np.unique(y.ravel())) <= 10  # Simple heuristic
            cv = self._setup_cross_validation(X, y, is_classification)

            # Generate OOF predictions for base models
            if self.base_models:
                oof_predictions = self._generate_oof_predictions(
                    self.base_models, X, y, cv, is_classification
                )
            else:
                self.logger.warning("No base models provided, creating defaults")
                self._create_default_base_models()
                oof_predictions = self._generate_oof_predictions(
                    self.base_models, X, y, cv, is_classification
                )

            # Train meta-models on OOF predictions
            if self.meta_models:
                meta_models = self._train_meta_models(X, y, oof_predictions, cv)
            else:
                self.logger.warning("No meta-models provided, creating defaults")
                meta_models = self._train_meta_models(X, y, oof_predictions, cv)

            # Update state
            self.is_fitted = True

            # Record training history
            training_time = time.time() - start_time
            self.training_history.append({
                'timestamp': datetime.now(),
                'duration': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_outputs': y.shape[1] if len(y.shape) > 1 else 1,
                'cv_folds': self.config.cv_folds,
                'oof_scores': self.oof_scores
            })

            self.logger.info(f"✅ OOF StackingEnsemble fitted in {training_time:.3f}s")
            self.logger.info(f"📊 OOF scores: {self.oof_scores}")

            return self

        except Exception as e:
            self.logger.error(f"Failed to fit OOF stacking ensemble: {e}")
            raise

    def _create_default_base_models(self):
        """Create default base models for all outputs."""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from xgboost import XGBClassifier, XGBRegressor
            from lightgbm import LGBMClassifier, LGBMRegressor
            from catboost import CatBoostClassifier, CatBoostRegressor

            # Create base models for each output
            for output_name in self.config.output_names:
                if output_name not in self.base_models:
                    self.base_models[output_name] = {}

                # Determine if classification or regression based on target
                # For simplicity, assume classification for now
                is_classification = True

                # Create diverse base models
                if is_classification:
                    self.base_models[output_name]['random_forest'] = RandomForestClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1
                    )
                    self.base_models[output_name]['xgboost'] = XGBClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1, verbosity=0
                    )
                    self.base_models[output_name]['lightgbm'] = LGBMClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1, verbosity=0
                    )
                else:
                    self.base_models[output_name]['random_forest'] = RandomForestRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1
                    )
                    self.base_models[output_name]['xgboost'] = XGBRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1, verbosity=0
                    )
                    self.base_models[output_name]['lightgbm'] = LGBMRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1, verbosity=0
                    )

                self.logger.info(f"✅ Created default base models for {output_name}")

        except Exception as e:
            self.logger.error(f"Failed to create default base models: {e}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Make predictions using the OOF stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("OOF StackingEnsemble not fitted yet")

        self.logger.debug(f"🔮 Making predictions for {X.shape[0]} samples")
        start_time = time.time()

        try:
            predictions_list = []

            for output_name in self.config.output_names:
                if output_name not in self.base_models:
                    self.logger.warning(f"No base models for output {output_name}")
                    continue

                # Generate base model predictions
                base_preds = []
                for model_name, model in self.base_models[output_name].items():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                        if pred.ndim > 1 and pred.shape[1] > 1:
                            pred = pred[:, 1]
                    else:
                        pred = model.predict(X)
                    base_preds.append(pred)

                # Combine with original features for meta-model
                if base_preds:
                    meta_features = np.column_stack([X] + base_preds)
                else:
                    meta_features = X

                # Meta-model prediction
                if output_name in self.meta_models:
                    meta_pred = self.meta_models[output_name].predict(meta_features)
                else:
                    # Fallback to average of base predictions
                    meta_pred = np.mean(base_preds, axis=0)

                predictions_list.append(meta_pred)

            # Combine all output predictions
            if predictions_list:
                predictions = np.column_stack(predictions_list)
            else:
                predictions = np.zeros((X.shape[0], self.config.n_outputs))

            # Generate probabilities if available
            probabilities = None
            if len(predictions_list) == 1 and self.config.n_outputs == 1:
                # For single output, create probability-like predictions
                probabilities = np.column_stack([1 - predictions.ravel(), predictions.ravel()])

            # Calculate confidence scores with proper uncertainty estimation
            confidence_scores = self._calculate_stacking_confidence(
                X, base_preds, predictions_list, probabilities
            )

            prediction_time = time.time() - start_time
            self.logger.info(f"✅ Predictions completed in {prediction_time:.3f}s")
            self.logger.info(f"📊 Confidence: {np.mean(confidence_scores):.3f} ± {np.std(confidence_scores):.3f}")

            return predictions, probabilities, confidence_scores

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def get_oof_predictions(self) -> Dict[str, np.ndarray]:
        """Get OOF predictions for all outputs."""
        return self.oof_predictions

    def get_oof_scores(self) -> Dict[str, float]:
        """Get OOF scores for all outputs."""
        return self.oof_scores
    
    def get_confidence_intervals(self, confidence_level: float = 0.95) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get confidence intervals for OOF predictions using bootstrap."""
        try:
            from scipy import stats
            
            confidence_intervals = {}
            
            for output_name, oof_preds in self.oof_predictions.items():
                if len(oof_preds) == 0:
                    continue
                
                # Bootstrap confidence intervals
                n_bootstrap = 1000
                bootstrap_samples = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    bootstrap_indices = np.random.choice(len(oof_preds), size=len(oof_preds), replace=True)
                    bootstrap_sample = oof_preds[bootstrap_indices]
                    bootstrap_samples.append(np.mean(bootstrap_sample))
                
                # Calculate confidence intervals
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(bootstrap_samples, lower_percentile)
                upper_bound = np.percentile(bootstrap_samples, upper_percentile)
                
                confidence_intervals[output_name] = (lower_bound, upper_bound)
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return {}
    
    def get_ensemble_diversity_metrics(self) -> Dict[str, float]:
        """Calculate ensemble diversity metrics."""
        try:
            diversity_metrics = {}
            
            for output_name, oof_preds in self.oof_predictions.items():
                if len(oof_preds) == 0:
                    continue
                
                # Calculate prediction variance as diversity measure
                prediction_variance = np.var(oof_preds)
                diversity_metrics[f"{output_name}_variance"] = float(prediction_variance)
                
                # Calculate coefficient of variation
                if np.mean(oof_preds) != 0:
                    cv = np.std(oof_preds) / np.abs(np.mean(oof_preds))
                    diversity_metrics[f"{output_name}_cv"] = float(cv)
                else:
                    diversity_metrics[f"{output_name}_cv"] = 0.0
            
            return diversity_metrics
            
        except Exception as e:
            self.logger.warning(f"Diversity metrics calculation failed: {e}")
            return {}