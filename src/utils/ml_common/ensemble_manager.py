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
from src.utils.m1_gpu_utils import get_m1_gpu_manager
from src.utils.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer

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
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

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
        self.config = config
        self.logger = logger.getChild('EnsembleManager')
        
        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None
        
        # Initialize utilities
        self.parquet_utils = get_parquet_utils()
        
        # Ensemble state
        self.models: Dict[str, ModelMetadata] = {}
        self.ensemble_model: Optional[Any] = None
        self.ensemble_weights: Optional[np.ndarray] = None
        self.prediction_count = 0
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.weight_history: List[np.ndarray] = []
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        self.logger.info(f"🚀 EnsembleManager initialized for {config.ensemble_name}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎯 Ensemble type: {config.ensemble_type.value}")
    
    @traced(span_name='add_model')
    @log_execution_time
    async def add_model(
        self, 
        model_name: str, 
        model: Any, 
        performance_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """Add a model to the ensemble."""
        
        self.logger.info(f"🔄 Adding model {model_name} to ensemble...")
        
        try:
            # Validate model
            if not self._validate_model(model):
                self.logger.error(f"❌ Model {model_name} validation failed")
                return False
            
            # Check if we have space for more models
            if len(self.models) >= self.config.max_models:
                await self._remove_worst_model()
            
            # Create model metadata
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
            
            # Add to ensemble
            self.models[model_name] = metadata
            
            self.logger.info(f"✅ Model {model_name} added to ensemble")
            self.logger.info(f"📊 Current ensemble size: {len(self.models)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add model {model_name}: {e}")
            return False
    
    @traced(span_name='create_ensemble')
    @log_execution_time
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
        
        # Validate inputs
        if len(self.models) < self.config.min_models:
            raise ValidationError(f"Insufficient models: {len(self.models)} < {self.config.min_models}")
        
        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._create_ensemble_internal(X_train, y_train, X_val, y_val, **kwargs)
        else:
            results = await self._create_ensemble_internal(X_train, y_train, X_val, y_val, **kwargs)
        
        execution_time = time.time() - start_time
        results.execution_time = execution_time
        
        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
        
        self.logger.info(f"✅ Ensemble created in {execution_time:.2f}s")
        self.logger.info(f"📊 Ensemble performance: {results.ensemble_performance}")
        
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
        
        # Select best models
        selected_models = await self._select_best_models()
        
        # Create ensemble based on type
        if self.config.ensemble_type == EnsembleType.VOTING:
            ensemble_model = await self._create_voting_ensemble(selected_models)
        elif self.config.ensemble_type == EnsembleType.STACKING:
            ensemble_model = await self._create_stacking_ensemble(selected_models, X_train, y_train)
        elif self.config.ensemble_type == EnsembleType.BLENDING:
            ensemble_model = await self._create_blending_ensemble(selected_models, X_train, y_train, X_val, y_val)
        elif self.config.ensemble_type == EnsembleType.WEIGHTED_AVERAGE:
            ensemble_model = await self._create_weighted_average_ensemble(selected_models)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.config.ensemble_type}")
        
        # Train ensemble
        if hasattr(ensemble_model, 'fit'):
            ensemble_model.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_performance = await self._evaluate_ensemble(ensemble_model, X_val, y_val)
        
        # Calculate ensemble characteristics
        diversity_score = await self._calculate_diversity_score(selected_models, X_train)
        stability_score = await self._calculate_stability_score(selected_models, X_train, y_train)
        
        # Create results
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
        
        return results
    
    async def _select_best_models(self) -> Dict[str, ModelMetadata]:
        """Select best models for ensemble."""
        
        if len(self.models) <= self.config.max_models:
            return self.models.copy()
        
        # Sort models by performance
        if self.config.model_selection_criteria == "performance":
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].performance_metrics.get('accuracy', 0.0),
                reverse=True
            )
        elif self.config.model_selection_criteria == "diversity":
            # Select diverse models (simplified implementation)
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].performance_metrics.get('accuracy', 0.0),
                reverse=True
            )
        else:
            # Default to performance
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].performance_metrics.get('accuracy', 0.0),
                reverse=True
            )
        
        # Select top models
        selected = dict(sorted_models[:self.config.max_models])
        
        self.logger.info(f"📊 Selected {len(selected)} models for ensemble")
        
        return selected
    
    async def _create_voting_ensemble(self, models: Dict[str, ModelMetadata]) -> Any:
        """Create voting ensemble."""
        
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        # Determine if classification or regression
        is_classification = self._is_classification_task(models)
        
        # Prepare estimators
        estimators = [(name, meta.model_object) for name, meta in models.items()]
        
        # Create voting ensemble
        if is_classification:
            voting = 'soft' if self.config.voting_strategy == VotingStrategy.SOFT else 'hard'
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=voting,
                n_jobs=-1
            )
        else:
            ensemble = VotingRegressor(
                estimators=estimators,
                n_jobs=-1
            )
        
        self.logger.info(f"✅ Created voting ensemble with {len(estimators)} models")
        
        return ensemble
    
    async def _create_stacking_ensemble(self, models: Dict[str, ModelMetadata], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Create stacking ensemble."""
        
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # Determine if classification or regression
        is_classification = self._is_classification_task(models)
        
        # Prepare base estimators
        base_estimators = [(name, meta.model_object) for name, meta in models.items()]
        
        # Create meta-learner
        if is_classification:
            meta_learner = LogisticRegression(random_state=42)
            ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                stack_method='predict_proba',
                n_jobs=-1
            )
        else:
            meta_learner = LinearRegression()
            ensemble = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                n_jobs=-1
            )
        
        self.logger.info(f"✅ Created stacking ensemble with {len(base_estimators)} base models")
        
        return ensemble
    
    async def _create_blending_ensemble(self, models: Dict[str, ModelMetadata], X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> Any:
        """Create blending ensemble."""
        
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # Determine if classification or regression
        is_classification = self._is_classification_task(models)
        
        # Create blending ensemble class
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
        if is_classification:
            meta_learner = LogisticRegression(random_state=42)
        else:
            meta_learner = LinearRegression()
        
        # Create blending ensemble
        ensemble = BlendingEnsemble(
            base_models={name: meta.model_object for name, meta in models.items()},
            meta_learner=meta_learner,
            is_classification=is_classification
        )
        
        self.logger.info(f"✅ Created blending ensemble with {len(models)} base models")
        
        return ensemble
    
    async def _create_weighted_average_ensemble(self, models: Dict[str, ModelMetadata]) -> Any:
        """Create weighted average ensemble."""
        
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
        weights = np.array([meta.weight for meta in models.values()])
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble = WeightedAverageEnsemble(
            models={name: meta.model_object for name, meta in models.items()},
            weights=weights
        )
        
        self.logger.info(f"✅ Created weighted average ensemble with {len(models)} models")
        
        return ensemble
    
    async def _evaluate_ensemble(self, ensemble_model: Any, X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series]) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        
        if X_val is None or y_val is None:
            return {}
        
        try:
            # Make predictions
            if hasattr(ensemble_model, 'predict_proba'):
                y_pred_proba = ensemble_model.predict_proba(X_val)
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                y_pred = ensemble_model.predict(X_val)
                y_pred_proba = None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC if available
            if y_pred_proba is not None and len(np.unique(y_val)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba[:, 1])
                except:
                    pass
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble: {e}")
            return {}
    
    async def _calculate_diversity_score(self, models: Dict[str, ModelMetadata], X: pd.DataFrame) -> float:
        """Calculate diversity score of models."""
        
        if len(models) < 2:
            return 0.0
        
        try:
            # Get predictions from all models
            predictions = []
            for meta in models.values():
                if hasattr(meta.model_object, 'predict'):
                    pred = meta.model_object.predict(X)
                    predictions.append(pred)
            
            if len(predictions) < 2:
                return 0.0
            
            # Calculate pairwise disagreement
            disagreements = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    disagreement = np.mean(predictions[i] != predictions[j])
                    disagreements.append(disagreement)
            
            # Diversity score is average disagreement
            diversity_score = np.mean(disagreements) if disagreements else 0.0
            
            return diversity_score
            
        except Exception as e:
            self.logger.error(f"Error calculating diversity score: {e}")
            return 0.0
    
    async def _calculate_stability_score(self, models: Dict[str, ModelMetadata], X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate stability score of models."""
        
        try:
            # Use cross-validation to measure stability
            from sklearn.model_selection import KFold
            
            stability_scores = []
            for meta in models.values():
                if hasattr(meta.model_object, 'predict'):
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    fold_predictions = []
                    
                    for train_idx, val_idx in kf.split(X):
                        X_train_fold = X.iloc[train_idx]
                        y_train_fold = y.iloc[train_idx]
                        X_val_fold = X.iloc[val_idx]
                        
                        # Train model on fold
                        model_copy = type(meta.model_object)(**meta.model_object.get_params())
                        model_copy.fit(X_train_fold, y_train_fold)
                        
                        # Make predictions
                        pred = model_copy.predict(X_val_fold)
                        fold_predictions.append(pred)
                    
                    # Calculate stability as inverse of variance across folds
                    if len(fold_predictions) > 1:
                        stability = 1.0 / (1.0 + np.var(fold_predictions))
                        stability_scores.append(stability)
            
            return np.mean(stability_scores) if stability_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating stability score: {e}")
            return 0.0
    
    def _validate_model(self, model: Any) -> bool:
        """Validate model has required methods."""
        
        required_methods = ['fit', 'predict']
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                return False
        
        return True
    
    def _is_classification_task(self, models: Dict[str, ModelMetadata]) -> bool:
        """Determine if this is a classification task."""
        
        # Check model types
        classification_models = ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'GradientBoostingClassifier']
        
        for meta in models.values():
            if any(cls in meta.model_type for cls in classification_models):
                return True
        
        return False
    
    def _calculate_initial_weight(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate initial weight based on performance metrics."""
        
        # Use accuracy as primary metric, fallback to other metrics
        if 'accuracy' in performance_metrics:
            return performance_metrics['accuracy']
        elif 'f1_score' in performance_metrics:
            return performance_metrics['f1_score']
        elif 'precision' in performance_metrics:
            return performance_metrics['precision']
        else:
            return 0.5  # Default weight
    
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
        optimizations = []
        
        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
        
        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
        
        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
        
        return optimizations
    
    @traced(span_name='predict')
    @log_execution_time
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
            self.logger.error(f"Error making predictions: {e}")
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
            self.logger.error(f"Error saving ensemble: {e}")
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
            self.logger.error(f"Error loading ensemble: {e}")
            raise