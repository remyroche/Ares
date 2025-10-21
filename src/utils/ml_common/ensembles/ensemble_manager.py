"""
Ensemble Manager for ML Models

This module provides comprehensive ensemble management capabilities for creating,
training, and managing ensembles of ML models, particularly for the Analyst system.
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
import pickle
import joblib
from pathlib import Path

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
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError
)
from src.utils.tprint import tprint_data_format, LogLevel

logger = logging.getLogger(__name__)

class EnsembleType(Enum):
    """Types of ensemble methods."""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    MULTI_OUTPUT_STACKING = "multi_output_stacking"

class VotingStrategy(Enum):
    """Voting strategies for ensemble."""
    HARD = "hard"
    SOFT = "soft"
    WEIGHTED = "weighted"

@dataclass
class ModelMetadata:
    """Metadata for individual models in ensemble."""
    model_name: str
    model_type: str
    model_object: Any
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[np.ndarray] = None
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage_mb: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    weight: float = 1.0

@dataclass
class EnsembleConfig:
    """Configuration for ensemble manager."""
    # Basic configuration
    ensemble_name: str
    output_dir: str

    # Ensemble type and strategy
    ensemble_type: EnsembleType = EnsembleType.VOTING
    voting_strategy: VotingStrategy = VotingStrategy.SOFT

    # Model selection
    max_models: int = 10
    min_models: int = 2
    model_selection_criteria: str = "performance"  # performance, diversity, stability

    # Training configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10

    # Weight optimization
    enable_weight_optimization: bool = True
    weight_optimization_method: str = "performance_based"  # performance_based, diversity_based, stability_based
    weight_update_frequency: int = 100  # Update weights every N predictions

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

@dataclass
class EnsembleResults:
    """Results from ensemble operations."""
    # Basic info
    ensemble_name: str
    ensemble_type: EnsembleType
    created_at: datetime
    total_duration: float

    # Model information
    model_count: int
    active_models: int
    model_metadata: List[ModelMetadata] = field(default_factory=list)

    # Performance metrics
    ensemble_performance: Dict[str, float] = field(default_factory=dict)
    individual_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Ensemble characteristics
    diversity_score: float = 0.0
    stability_score: float = 0.0
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))

    # Metadata
    config: EnsembleConfig = field(default_factory=EnsembleConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)

class EnsembleManager:
    """Comprehensive ensemble manager with M1 optimizations."""

    def __init__(self, config: EnsembleConfig):
        """Initialize ensemble manager."""
        self.logger = logger.getChild('EnsembleManager')
        self.logger.info(f"🚀 Initializing EnsembleManager for {config.ensemble_name}...")
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

        # Initialize utilities
        self.logger.debug("🔧 Initializing utilities...")
        self.parquet_utils = get_parquet_utils()
        self.logger.debug("✅ Utilities initialized")

        # Ensemble state
        self.models: Dict[str, ModelMetadata] = {}
        self.ensemble_model: Optional[Any] = None
        self.ensemble_weights: Optional[np.ndarray] = None
        self.prediction_count = 0

        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.weight_history: List[np.ndarray] = []

        # Ensure output directory exists
        self.logger.debug(f"🔧 Ensuring output directory exists: {config.output_dir}")
        ensure_directory(config.output_dir)
        self.logger.debug("✅ Output directory ready")

        init_time = time.time() - start_time
        self.logger.info(f"✅ EnsembleManager initialized for {config.ensemble_name} in {init_time:.3f}s")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎯 Ensemble type: {config.ensemble_type.value}")
        self.logger.info(f"📊 Max models: {config.max_models}, Min models: {config.min_models}")
        self.logger.info(f"💾 Output directory: {config.output_dir}")

    async def add_model(
        self,
        model_name: str,
        model: Any,
        performance_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """Add a model to the ensemble."""

        self.logger.info(f"🔄 Adding model {model_name} to ensemble...")
        start_time = time.time()

        try:
            # Validate model
            self.logger.debug(f"🔍 Validating model {model_name}...")
            if not self._validate_model(model):
                self.logger.error(f"❌ Model {model_name} validation failed")
                return False
            self.logger.debug(f"✅ Model {model_name} validation passed")

            # Check if we have space for more models
            if len(self.models) >= self.config.max_models:
                self.logger.warning(f"⚠️ Ensemble at capacity ({len(self.models)}/{self.config.max_models}), removing worst model...")
                await self._remove_worst_model()

            # Create model metadata
            self.logger.debug(f"📊 Creating metadata for model {model_name}...")
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=type(model).__name__,
                model_object=model,
                performance_metrics=performance_metrics or {},
                created_at=datetime.now(),
                last_updated=datetime.now()
            )

            # Calculate initial weight based on performance
            if performance_metrics:
                metadata.weight = self._calculate_initial_weight(performance_metrics)
                self.logger.debug(f"⚖️ Initial weight calculated: {metadata.weight:.4f}")
            else:
                self.logger.debug("⚖️ No performance metrics provided, using default weight")

            # Add to ensemble
            self.models[model_name] = metadata

            add_time = time.time() - start_time
            self.logger.info(f"✅ Model {model_name} added to ensemble in {add_time:.3f}s")
            self.logger.info(f"📊 Current ensemble size: {len(self.models)}")
            self.logger.info(f"🎯 Model type: {metadata.model_type}")
            self.logger.info(f"⚖️ Model weight: {metadata.weight:.4f}")

            return True

        except Exception as e:
            add_time = time.time() - start_time
            self.logger.error(f"❌ Failed to add model {model_name} after {add_time:.3f}s: {e}")
            self.logger.error(f"📋 Model type: {type(model).__name__}")
            self.logger.error(f"📊 Performance metrics: {performance_metrics}")
            self.logger.warning("⚠️ Model addition failed - ensemble may be incomplete")
            return False

    async def create_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> EnsembleResults:
        """Create ensemble from available models."""

        self.logger.info("🚀 Creating ensemble...")
        start_time = time.time()

        self.logger.info(f"📊 Training data shape: {X_train.shape}")
        self.logger.info(f"📊 Target data shape: {y_train.shape}")
        if X_val is not None:
            self.logger.info(f"📊 Validation data shape: {X_val.shape}")
        if y_val is not None:
            self.logger.info(f"📊 Validation target shape: {y_val.shape}")

        # Debug data format
        tprint_data_format(X_train, "ensemble_training_features", level=LogLevel.DEBUG)
        tprint_data_format(y_train, "ensemble_training_targets", level=LogLevel.DEBUG)
        if X_val is not None:
            tprint_data_format(X_val, "ensemble_validation_features", level=LogLevel.DEBUG)
        if y_val is not None:
            tprint_data_format(y_val, "ensemble_validation_targets", level=LogLevel.DEBUG)

        # Validate inputs
        self.logger.debug(f"🔍 Validating ensemble creation requirements...")
        if len(self.models) < self.config.min_models:
            self.logger.error(f"❌ Insufficient models: {len(self.models)} < {self.config.min_models}")
            raise ValidationError(f"Insufficient models: {len(self.models)} < {self.config.min_models}")

        self.logger.info(f"✅ Model count validation passed: {len(self.models)} models available")

        # Memory optimization context
        if self.m1_memory:
            self.logger.debug("🧠 Using memory optimization context...")
            with self.m1_memory.optimization_context():
                results = await self._create_ensemble_internal(X_train, y_train, X_val, y_val, **kwargs)
        else:
            self.logger.debug("🧠 No memory optimization available, proceeding normally...")
            results = await self._create_ensemble_internal(X_train, y_train, X_val, y_val, **kwargs)

        execution_time = time.time() - start_time
        results.execution_time = execution_time

        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
            self.logger.info(f"🧠 Memory usage: {results.memory_usage_mb:.1f} MB")

        self.logger.info(f"✅ Ensemble created in {execution_time:.2f}s")
        self.logger.info(f"📊 Ensemble performance: {results.ensemble_performance}")
        self.logger.info(f"🎯 Model count: {results.model_count}")
        self.logger.info(f"📈 Diversity score: {results.diversity_score:.4f}")
        self.logger.info(f"🔒 Stability score: {results.stability_score:.4f}")

        return results

    async def _create_ensemble_internal(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        **kwargs
    ) -> EnsembleResults:
        """Internal ensemble creation logic."""

        self.logger.debug("🔄 Starting internal ensemble creation...")
        internal_start_time = time.time()

        # Select best models
        self.logger.debug("🔍 Selecting best models for ensemble...")
        selection_start_time = time.time()
        selected_models = await self._select_best_models()
        selection_time = time.time() - selection_start_time
        self.logger.info(f"✅ Model selection completed in {selection_time:.3f}s: {len(selected_models)} models selected")

        # Create ensemble based on type
        self.logger.info(f"🔄 Creating {self.config.ensemble_type.value} ensemble...")
        creation_start_time = time.time()

        if self.config.ensemble_type == EnsembleType.VOTING:
            ensemble_model = await self._create_voting_ensemble(selected_models)
        elif self.config.ensemble_type == EnsembleType.STACKING:
            ensemble_model = await self._create_stacking_ensemble(selected_models, X_train, y_train)
        elif self.config.ensemble_type == EnsembleType.BLENDING:
            ensemble_model = await self._create_blending_ensemble(selected_models, X_train, y_train, X_val, y_val)
        elif self.config.ensemble_type == EnsembleType.WEIGHTED_AVERAGE:
            ensemble_model = await self._create_weighted_average_ensemble(selected_models)
        elif self.config.ensemble_type == EnsembleType.MULTI_OUTPUT_STACKING:
            ensemble_model = await self._create_multi_output_stacking_ensemble(selected_models, X_train, y_train)
        else:
            self.logger.error(f"❌ Unsupported ensemble type: {self.config.ensemble_type}")
            raise ValueError(f"Unsupported ensemble type: {self.config.ensemble_type}")

        creation_time = time.time() - creation_start_time
        self.logger.info(f"✅ Ensemble model created in {creation_time:.3f}s")

        # Train ensemble
        if hasattr(ensemble_model, 'fit'):
            self.logger.info("🔄 Training ensemble model...")
            training_start_time = time.time()
            ensemble_model.fit(X_train, y_train)
            training_time = time.time() - training_start_time
            self.logger.info(f"✅ Ensemble training completed in {training_time:.3f}s")
        else:
            self.logger.debug("ℹ️ Ensemble model does not require training")

        # Evaluate ensemble
        self.logger.debug("🔍 Evaluating ensemble performance...")
        eval_start_time = time.time()
        ensemble_performance = await self._evaluate_ensemble(ensemble_model, X_val, y_val)
        eval_time = time.time() - eval_start_time
        self.logger.info(f"✅ Ensemble evaluation completed in {eval_time:.3f}s")

        # Calculate ensemble characteristics
        self.logger.debug("📊 Calculating ensemble characteristics...")
        char_start_time = time.time()
        diversity_score = await self._calculate_diversity_score(selected_models, X_train)
        stability_score = await self._calculate_stability_score(selected_models, X_train, y_train)
        char_time = time.time() - char_start_time
        self.logger.info(f"✅ Characteristic calculation completed in {char_time:.3f}s")

        # Create results
        self.logger.debug("📊 Creating ensemble results...")
        results = EnsembleResults(
            ensemble_name=self.config.ensemble_name,
            ensemble_type=self.config.ensemble_type,
            created_at=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            model_count=len(selected_models),
            active_models=len([m for m in selected_models.values() if m.is_active]),
            model_metadata=list(selected_models.values()),
            ensemble_performance=ensemble_performance,
            individual_model_performance={name: meta.performance_metrics for name, meta in selected_models.items()},
            diversity_score=diversity_score,
            stability_score=stability_score,
            config=self.config,
            optimization_used=self._get_optimization_used()
        )

        # Store ensemble model
        self.ensemble_model = ensemble_model

        internal_time = time.time() - internal_start_time
        self.logger.info(f"✅ Internal ensemble creation completed in {internal_time:.3f}s")

        return results

    async def _select_best_models(self) -> Dict[str, ModelMetadata]:
        """Select best models for ensemble."""

        self.logger.debug(f"🔍 Selecting best models from {len(self.models)} available models...")
        self.logger.debug(f"📊 Selection criteria: {self.config.model_selection_criteria}")
        self.logger.debug(f"📊 Max models allowed: {self.config.max_models}")

        if len(self.models) <= self.config.max_models:
            self.logger.info(f"✅ All {len(self.models)} models selected (within limit)")
            return self.models.copy()

        # Sort models by performance
        self.logger.debug("🔄 Sorting models by selection criteria...")
        if self.config.model_selection_criteria == "performance":
            self.logger.debug("📊 Sorting by performance metrics...")
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].performance_metrics.get('accuracy', 0.0),
                reverse=True
            )
        elif self.config.model_selection_criteria == "diversity":
            self.logger.debug("📊 Sorting by diversity (simplified implementation)...")
            # Select diverse models (simplified implementation)
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].performance_metrics.get('accuracy', 0.0),
                reverse=True
            )
        else:
            self.logger.debug("📊 Using default performance sorting...")
            # Default to performance
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].performance_metrics.get('accuracy', 0.0),
                reverse=True
            )

        # Select top models
        selected = dict(sorted_models[:self.config.max_models])

        # Log selection details
        self.logger.info(f"📊 Selected {len(selected)} models for ensemble")
        for i, (name, metadata) in enumerate(sorted_models[:self.config.max_models]):
            accuracy = metadata.performance_metrics.get('accuracy', 0.0)
            self.logger.debug(f"   {i+1}. {name}: accuracy={accuracy:.4f}, weight={metadata.weight:.4f}")

        # Log excluded models
        if len(sorted_models) > self.config.max_models:
            excluded_count = len(sorted_models) - self.config.max_models
            self.logger.info(f"📊 Excluded {excluded_count} models due to capacity limit")
            for i, (name, metadata) in enumerate(sorted_models[self.config.max_models:]):
                accuracy = metadata.performance_metrics.get('accuracy', 0.0)
                self.logger.debug(f"   Excluded {i+1}. {name}: accuracy={accuracy:.4f}")

        return selected

    async def _create_voting_ensemble(self, models: Dict[str, ModelMetadata]) -> Any:
        """Create voting ensemble."""

        self.logger.debug("🔄 Creating voting ensemble...")
        start_time = time.time()

        from sklearn.ensemble import VotingClassifier, VotingRegressor

        # Determine if classification or regression
        self.logger.debug("🔍 Determining task type...")
        is_classification = self._is_classification_task(models)
        self.logger.info(f"📊 Task type: {'Classification' if is_classification else 'Regression'}")

        # Prepare estimators
        self.logger.debug("🔧 Preparing estimators...")
        estimators = [(name, meta.model_object) for name, meta in models.items()]
        self.logger.debug(f"📊 Prepared {len(estimators)} estimators")

        # Create voting ensemble
        self.logger.debug("🔄 Creating voting ensemble object...")
        if is_classification:
            voting = 'soft' if self.config.voting_strategy == VotingStrategy.SOFT else 'hard'
            self.logger.debug(f"📊 Voting strategy: {voting}")
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting,
                n_jobs=-1
            )
        else:
            self.logger.debug("📊 Creating regression voting ensemble")
            ensemble = VotingRegressor(
                estimators=estimators,
                n_jobs=-1
            )

        creation_time = time.time() - start_time
        self.logger.info(f"✅ Created voting ensemble with {len(estimators)} models in {creation_time:.3f}s")
        self.logger.info(f"🎯 Ensemble type: {type(ensemble).__name__}")

        return ensemble

    async def _create_stacking_ensemble(self, models: Dict[str, ModelMetadata], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Create stacking ensemble."""

        self.logger.debug("🔄 Creating stacking ensemble...")
        start_time = time.time()

        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression

        # Determine if classification or regression
        self.logger.debug("🔍 Determining task type...")
        is_classification = self._is_classification_task(models)
        self.logger.info(f"📊 Task type: {'Classification' if is_classification else 'Regression'}")

        # Prepare base estimators
        self.logger.debug("🔧 Preparing base estimators...")
        base_estimators = [(name, meta.model_object) for name, meta in models.items()]
        self.logger.debug(f"📊 Prepared {len(base_estimators)} base estimators")

        # Create meta-learner
        self.logger.debug("🔧 Creating meta-learner...")
        if is_classification:
            meta_learner = LogisticRegression(random_state=42)
            self.logger.debug("📊 Using LogisticRegression as meta-learner")
            ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                stack_method='predict_proba',
                n_jobs=-1
            )
        else:
            meta_learner = LinearRegression()
            self.logger.debug("📊 Using LinearRegression as meta-learner")
            ensemble = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                n_jobs=-1
            )

        creation_time = time.time() - start_time
        self.logger.info(f"✅ Created stacking ensemble with {len(base_estimators)} base models in {creation_time:.3f}s")
        self.logger.info(f"🎯 Ensemble type: {type(ensemble).__name__}")
        self.logger.info(f"📊 CV folds: {self.config.cv_folds}")
        self.logger.info(f"🎯 Meta-learner: {type(meta_learner).__name__}")

        return ensemble

    async def _create_blending_ensemble(self, models: Dict[str, ModelMetadata], X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> Any:
        """Create blending ensemble."""

        self.logger.debug("🔄 Creating blending ensemble...")
        start_time = time.time()

        # Determine if classification or regression
        self.logger.debug("🔍 Determining task type...")
        is_classification = self._is_classification_task(models)
        self.logger.info(f"📊 Task type: {'Classification' if is_classification else 'Regression'}")

        # Create blending ensemble class
        self.logger.debug("🔧 Creating BlendingEnsemble class...")
        class BlendingEnsemble:
            def __init__(self, base_models, meta_learner, is_classification):
                self.base_models = base_models
                self.meta_learner = meta_learner
                self.is_classification = is_classification
                self.is_fitted = False

            def fit(self, X, y):
                # Train base models
                for name, model in self.base_models.items():
                    model.fit(X, y)

                # Generate meta-features
                meta_features = self._generate_meta_features(X)

                # Train meta-learner
                self.meta_learner.fit(meta_features, y)
                self.is_fitted = True

            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble not fitted")

                meta_features = self._generate_meta_features(X)
                return self.meta_learner.predict(meta_features)

            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble not fitted")

                meta_features = self._generate_meta_features(X)
                if hasattr(self.meta_learner, 'predict_proba'):
                    return self.meta_learner.predict_proba(meta_features)
                else:
                    return None

            def _generate_meta_features(self, X):
                meta_features = []
                for name, model in self.base_models.items():
                    if hasattr(model, 'predict_proba') and self.is_classification:
                        pred = model.predict_proba(X)
                        meta_features.append(pred)
                    else:
                        pred = model.predict(X).reshape(-1, 1)
                        meta_features.append(pred)

                return np.hstack(meta_features)

        # Create meta-learner
        self.logger.debug("🔧 Creating meta-learner...")
        if is_classification:
            meta_learner = LogisticRegression(random_state=42)
            self.logger.debug("📊 Using LogisticRegression as meta-learner")
        else:
            meta_learner = LinearRegression()
            self.logger.debug("📊 Using LinearRegression as meta-learner")

        # Create blending ensemble
        self.logger.debug("🔧 Creating blending ensemble object...")
        ensemble = BlendingEnsemble(
            base_models={name: meta.model_object for name, meta in models.items()},
            meta_learner=meta_learner,
            is_classification=is_classification
        )

        creation_time = time.time() - start_time
        self.logger.info(f"✅ Created blending ensemble with {len(models)} base models in {creation_time:.3f}s")
        self.logger.info(f"🎯 Ensemble type: BlendingEnsemble")
        self.logger.info(f"🎯 Meta-learner: {type(meta_learner).__name__}")

        return ensemble

    async def _create_weighted_average_ensemble(self, models: Dict[str, ModelMetadata]) -> Any:
        """Create weighted average ensemble."""

        self.logger.debug("🔄 Creating weighted average ensemble...")
        start_time = time.time()

        self.logger.debug("🔧 Creating WeightedAverageEnsemble class...")
        class WeightedAverageEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                self.is_fitted = False

            def fit(self, X, y):
                # Train all models
                for model in self.models.values():
                    model.fit(X, y)
                self.is_fitted = True

            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble not fitted")

                predictions = []
                for model, weight in zip(self.models.values(), self.weights):
                    pred = model.predict(X) * weight
                    predictions.append(pred)

                return np.sum(predictions, axis=0)

            def predict_proba(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble not fitted")

                predictions = []
                for model, weight in zip(self.models.values(), self.weights):
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X) * weight
                        predictions.append(pred)

                if predictions:
                    return np.sum(predictions, axis=0)
                else:
                    return None

        # Calculate weights based on performance
        self.logger.debug("⚖️ Calculating model weights...")
        weights = np.array([meta.weight for meta in models.values()])
        weights = weights / weights.sum()  # Normalize weights

        self.logger.debug(f"📊 Model weights: {weights}")
        self.logger.debug(f"📊 Weight sum: {weights.sum():.6f}")

        ensemble = WeightedAverageEnsemble(
            models={name: meta.model_object for name, meta in models.items()},
            weights=weights
        )

        creation_time = time.time() - start_time
        self.logger.info(f"✅ Created weighted average ensemble with {len(models)} models in {creation_time:.3f}s")
        self.logger.info(f"🎯 Ensemble type: WeightedAverageEnsemble")
        self.logger.info(f"⚖️ Weight range: {weights.min():.4f} - {weights.max():.4f}")

        return ensemble

    async def _create_multi_output_stacking_ensemble(self, models: Dict[str, ModelMetadata], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Create multi-output stacking ensemble."""

        self.logger.debug("🔄 Creating multi-output stacking ensemble...")
        start_time = time.time()

        try:
            # Import multi-output stacking components
            from ..models.multi_output_models import MultiOutputStackingModel, MultiOutputConfig
            from .stacking_ensemble_manager import StackingEnsembleManager, StackingEnsembleConfig

            # Determine output configuration based on target shape
            if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                n_outputs = y_train.shape[1]
                output_names = [f"output_{i+1}" for i in range(n_outputs)]
            else:
                n_outputs = 1
                output_names = ["output_1"]

            # Create multi-output configuration
            multi_output_config = MultiOutputConfig(
                model_name=f"{self.config.ensemble_name}_multi_output",
                n_outputs=n_outputs,
                output_names=output_names,
                base_models={name: meta.model_object for name, meta in models.items()},
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_memory_optimization=self.config.enable_memory_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                memory_limit_gb=self.config.memory_limit_gb
            )

            # Create multi-output stacking model
            ensemble = MultiOutputStackingModel(multi_output_config)

            creation_time = time.time() - start_time
            self.logger.info(f"✅ Created multi-output stacking ensemble with {n_outputs} outputs in {creation_time:.3f}s")
            self.logger.info(f"🎯 Ensemble type: MultiOutputStackingModel")
            self.logger.info(f"📊 Outputs: {output_names}")

            return ensemble

        except Exception as e:
            creation_time = time.time() - start_time
            self.logger.error(f"❌ Failed to create multi-output stacking ensemble after {creation_time:.3f}s: {e}")
            self.logger.warning("⚠️ Multi-output stacking ensemble creation failed - ensemble may not be available")
            raise

    async def _evaluate_ensemble(self, ensemble_model: Any, X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> Dict[str, float]:
        """Evaluate ensemble performance."""

        self.logger.debug("🔍 Evaluating ensemble performance...")
        start_time = time.time()

        if X_val is None or y_val is None:
            self.logger.warning("⚠️ No validation data provided, skipping evaluation")
            return {}

        self.logger.debug(f"📊 Validation data shape: {X_val.shape}")
        self.logger.debug(f"📊 Validation target shape: {y_val.shape}")

        try:
            # Make predictions
            self.logger.debug("🔄 Making predictions...")
            pred_start_time = time.time()

            if hasattr(ensemble_model, 'predict_proba'):
                y_pred_proba = ensemble_model.predict_proba(X_val)
                y_pred = np.argmax(y_pred_proba, axis=1)
                self.logger.debug("📊 Using predict_proba for predictions")
            else:
                y_pred = ensemble_model.predict(X_val)
                y_pred_proba = None
                self.logger.debug("📊 Using predict for predictions")

            pred_time = time.time() - pred_start_time
            self.logger.debug(f"✅ Predictions completed in {pred_time:.3f}s")

            # Calculate metrics
            self.logger.debug("📊 Calculating performance metrics...")
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
            }

            self.logger.debug(f"📊 Basic metrics calculated: {metrics}")

            # Add ROC AUC if available
            if y_pred_proba is not None and len(np.unique(y_val)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba[:, 1])
                    self.logger.debug(f"📊 ROC AUC calculated: {metrics['roc_auc']:.4f}")
                except Exception as e:
                    self.logger.debug(f"⚠️ ROC AUC calculation failed: {e}")

            eval_time = time.time() - start_time
            self.logger.info(f"✅ Ensemble evaluation completed in {eval_time:.3f}s")
            self.logger.info(f"📊 Performance metrics: {metrics}")

            return metrics

        except Exception as e:
            eval_time = time.time() - start_time
            self.logger.error(f"❌ Error evaluating ensemble after {eval_time:.3f}s: {e}")
            self.logger.warning("⚠️ Ensemble evaluation failed - returning empty metrics")
            return {'error': str(e)}

    async def _calculate_diversity_score(self, models: Dict[str, ModelMetadata], X: pd.DataFrame) -> float:
        """Calculate diversity score of models."""

        self.logger.debug("🔍 Calculating diversity score...")
        start_time = time.time()

        if len(models) < 2:
            self.logger.debug("⚠️ Insufficient models for diversity calculation (need at least 2)")
            return 0.0

        try:
            # Get predictions from all models
            self.logger.debug("🔄 Getting predictions from all models...")
            predictions = []
            for name, meta in models.items():
                if hasattr(meta.model_object, 'predict'):
                    pred = meta.model_object.predict(X)
                    predictions.append(pred)
                    self.logger.debug(f"📊 Got predictions from {name}: shape {pred.shape}")
                else:
                    self.logger.warning(f"⚠️ Model {name} does not have predict method")

            if len(predictions) < 2:
                self.logger.warning("⚠️ Insufficient predictions for diversity calculation")
                return 0.0

            # Calculate pairwise disagreement
            self.logger.debug("🔄 Calculating pairwise disagreements...")
            disagreements = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    disagreement = np.mean(predictions[i] != predictions[j])
                    disagreements.append(disagreement)
                    self.logger.debug(f"📊 Disagreement between models {i} and {j}: {disagreement:.4f}")

            # Diversity score is average disagreement
            diversity_score = np.mean(disagreements) if disagreements else 0.0

            calc_time = time.time() - start_time
            self.logger.info(f"✅ Diversity score calculated in {calc_time:.3f}s: {diversity_score:.4f}")
            self.logger.info(f"📊 Pairwise disagreements: {len(disagreements)} comparisons")

            return diversity_score

        except Exception as e:
            calc_time = time.time() - start_time
            self.logger.error(f"❌ Error calculating diversity score after {calc_time:.3f}s: {e}")
            self.logger.warning("⚠️ Diversity score calculation failed - returning zero diversity")
            return 0.0

    async def _calculate_stability_score(self, models: Dict[str, ModelMetadata], X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate stability score of models."""

        self.logger.debug("🔍 Calculating stability score...")
        start_time = time.time()

        try:
            # Use cross-validation to measure stability
            from sklearn.model_selection import KFold

            stability_scores = []
            for name, meta in models.items():
                self.logger.debug(f"🔄 Calculating stability for model {name}...")

                if hasattr(meta.model_object, 'predict'):
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    fold_predictions = []

                    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                        self.logger.debug(f"📊 Processing fold {fold_idx + 1}/3 for {name}...")

                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]

                        # Train model on fold
                        model_copy = type(meta.model_object)(**meta.model_object.get_params())
                        model_copy.fit(X_train_fold, y_train_fold)

                        # Make predictions
                        pred = model_copy.predict(X_val_fold)
                        fold_predictions.append(pred)
                        self.logger.debug(f"📊 Fold {fold_idx + 1} predictions: shape {pred.shape}")

                    # Calculate stability as inverse of variance across folds
                    if len(fold_predictions) > 1:
                        stability = 1.0 / (1.0 + np.var(fold_predictions))
                        stability_scores.append(stability)
                        self.logger.debug(f"📊 Stability for {name}: {stability:.4f}")
                    else:
                        self.logger.warning(f"⚠️ Insufficient folds for stability calculation for {name}")
                else:
                    self.logger.warning(f"⚠️ Model {name} does not have predict method")

            final_stability = np.mean(stability_scores) if stability_scores else 0.0

            calc_time = time.time() - start_time
            self.logger.info(f"✅ Stability score calculated in {calc_time:.3f}s: {final_stability:.4f}")
            self.logger.info(f"📊 Individual stability scores: {stability_scores}")

            return final_stability

        except Exception as e:
            calc_time = time.time() - start_time
            self.logger.error(f"❌ Error calculating stability score after {calc_time:.3f}s: {e}")
            self.logger.warning("⚠️ Stability score calculation failed - returning zero stability")
            return 0.0

    def _validate_model(self, model: Any) -> bool:
        """Validate model has required methods."""

        self.logger.debug(f"🔍 Validating model: {type(model).__name__}")

        required_methods = ['fit', 'predict']
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                self.logger.error(f"❌ Model missing required method: {method}")
                self.logger.warning("⚠️ Model validation failed - model cannot be used in ensemble")
                return False
            else:
                self.logger.debug(f"✅ Model has required method: {method}")

        self.logger.debug("✅ Model validation passed")
        return True

    def _is_classification_task(self, models: Dict[str, ModelMetadata]) -> bool:
        """Determine if this is a classification task."""

        self.logger.debug("🔍 Determining task type from model types...")

        # Check model types
        classification_models = ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'GradientBoostingClassifier']

        for name, meta in models.items():
            self.logger.debug(f"📊 Checking model {name}: {meta.model_type}")
            if any(cls in meta.model_type for cls in classification_models):
                self.logger.debug(f"✅ Classification task detected from model {name}")
                return True

        self.logger.debug("📊 No classification models found, assuming regression task")
        return False

    def _calculate_initial_weight(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate initial weight based on performance metrics."""

        self.logger.debug(f"⚖️ Calculating initial weight from metrics: {performance_metrics}")

        # Use accuracy as primary metric, fallback to other metrics
        if 'accuracy' in performance_metrics:
            weight = performance_metrics['accuracy']
            self.logger.debug(f"⚖️ Using accuracy for weight: {weight:.4f}")
        elif 'f1_score' in performance_metrics:
            weight = performance_metrics['f1_score']
            self.logger.debug(f"⚖️ Using f1_score for weight: {weight:.4f}")
        elif 'precision' in performance_metrics:
            weight = performance_metrics['precision']
            self.logger.debug(f"⚖️ Using precision for weight: {weight:.4f}")
        else:
            weight = 0.5  # Default weight
            self.logger.debug(f"⚖️ Using default weight: {weight:.4f}")

        self.logger.debug(f"✅ Initial weight calculated: {weight:.4f}")
        return weight

    async def _remove_worst_model(self) -> None:
        """Remove the worst performing model."""

        if not self.models:
            return

        # Find worst model
        worst_model = min(
            self.models.items(),
            key=lambda x: x[1].performance_metrics.get('accuracy', 0.0)
        )

        model_name = worst_model[0]
        del self.models[model_name]

        self.logger.info(f"🗑️ Removed worst model: {model_name}")

    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        self.logger.debug("🔍 Getting list of optimizations used...")

        optimizations = []

        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
            self.logger.debug("✅ M1 GPU acceleration enabled")

        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
            self.logger.debug("✅ M1 memory optimization enabled")

        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
            self.logger.debug("✅ M1 parallel processing enabled")

        self.logger.debug(f"📊 Optimizations used: {optimizations}")
        return optimizations

    @staticmethod
    def _create_voting_ensemble_static(models: Dict[str, Any]) -> Any:
        """Create voting ensemble (static method for utilities)."""
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        # Determine if classification or regression
        is_classification = any('Classifier' in str(type(model)) for model in models.values())

        # Prepare estimators
        estimators = [(name, model) for name, model in models.items()]

        # Create voting ensemble
        if is_classification:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
        else:
            ensemble = VotingRegressor(
                estimators=estimators,
                n_jobs=-1
            )

        return ensemble

    @staticmethod
    def _create_stacking_ensemble_static(models: Dict[str, Any]) -> Any:
        """Create stacking ensemble (static method for utilities)."""
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression

        # Determine if classification or regression
        is_classification = any('Classifier' in str(type(model)) for model in models.values())

        # Prepare base estimators
        base_estimators = [(name, model) for name, model in models.items()]

        # Create meta-learner
        if is_classification:
            meta_learner = LogisticRegression(random_state=42)
            ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )
        else:
            meta_learner = LinearRegression()
            ensemble = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )

        return ensemble

    @staticmethod
    def _create_weighted_average_ensemble_static(models: Dict[str, Any]) -> Any:
        """Create weighted average ensemble (static method for utilities)."""
        class WeightedAverageEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                self.is_fitted = False

            def fit(self, X, y):
                # Train all models
                for model in self.models.values():
                    model.fit(X, y)
                self.is_fitted = True

            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Ensemble not fitted")

                predictions = []
                for model, weight in zip(self.models.values(), self.weights):
                    pred = model.predict(X) * weight
                    predictions.append(pred)

                return np.sum(predictions, axis=0)

        # Calculate weights based on model count
        weights = np.ones(len(models)) / len(models)

        ensemble = WeightedAverageEnsemble(
            models=models,
            weights=weights
        )

        return ensemble

    @traced(span_name='predict')
    async def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions using ensemble."""

        if self.ensemble_model is None:
            raise ValueError("Ensemble not created yet")

        try:
            # Make predictions
            if hasattr(self.ensemble_model, 'predict_proba'):
                y_pred_proba = self.ensemble_model.predict_proba(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = self.ensemble_model.predict(X)
                y_pred_proba = None

            # Update prediction count
            self.prediction_count += 1

            # Update weights if needed
            if (self.config.enable_weight_optimization and
                self.prediction_count % self.config.weight_update_frequency == 0):
                await self._update_weights()

            return y_pred, y_pred_proba

        except Exception as e:
            self.logger.error(f"❌ Error making predictions: {e}")
            self.logger.warning("⚠️ Prediction failed - ensemble predictions may be incomplete")
            raise

    async def _update_weights(self) -> None:
        """Update model weights based on recent performance."""

        # This is a simplified implementation
        # In practice, you would track recent performance and update weights accordingly

        for meta in self.models.values():
            # Update weight based on recent performance (simplified)
            if meta.performance_metrics:
                meta.weight = self._calculate_initial_weight(meta.performance_metrics)

        self.logger.info("🔄 Updated model weights")

    async def save_ensemble(self, file_path: str) -> None:
        """Save ensemble to disk."""

        try:
            ensemble_data = {
                'ensemble_model': self.ensemble_model,
                'models': self.models,
                'config': self.config,
                'performance_history': self.performance_history,
                'weight_history': self.weight_history
            }

            with open(file_path, 'wb') as f:
                pickle.dump(ensemble_data, f)

            self.logger.info(f"💾 Ensemble saved to {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Error saving ensemble: {e}")
            self.logger.warning("⚠️ Ensemble save failed - ensemble data may not be persisted")
            raise

    async def load_ensemble(self, file_path: str) -> None:
        """Load ensemble from disk."""

        try:
            with open(file_path, 'rb') as f:
                ensemble_data = pickle.load(f)

            self.ensemble_model = ensemble_data['ensemble_model']
            self.models = ensemble_data['models']
            self.performance_history = ensemble_data.get('performance_history', [])
            self.weight_history = ensemble_data.get('weight_history', [])

            self.logger.info(f"📂 Ensemble loaded from {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Error loading ensemble: {e}")
            self.logger.warning("⚠️ Ensemble load failed - ensemble may not be restored from disk")
            raise
