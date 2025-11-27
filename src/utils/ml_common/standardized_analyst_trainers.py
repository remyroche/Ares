"""
Standardized Analyst Model Trainers with OOF Predictions and Scheduled Retraining

This module provides unified interfaces for training Analyst base models (KNN, NGBoost, LGBM)
with proper out-of-fold (OOF) predictions, scheduled retraining, and HPO.

Features:
- OOF predictions only (no data leakage)
- Respects burn-in periods
- Retrain every 10 days without HPO
- Full HPO every 30 days using BOHB (TPE + Hyperband)
- Warm start from previous HPO runs
- Stratified sampling (10-50% of data)
- early_stopping_rounds=35

Model-Specific Optimizations:

KNN:
- Approximate nearest neighbors (Faiss/HNSW)
- Batch predictions
- Euclidean distance
- Stratified sampling

NGBoost:
- Gaussian distribution
- Stochastic methods (colsample_bytree + subsample)
- Subsample rows (10-50%)

LGBM:
- Binary Dataset format
- Full-sized validation
- Optimized parameters (min_data_in_leaf, min_gain_to_split)
- Bagging/feature fractions
- Warm start from previous best params

Usage:
    ```python
    from src.utils.ml_common.standardized_analyst_trainers import (
        StandardizedKNNTrainer, StandardizedNGBoostTrainer, StandardizedLGBMTrainer
    )

    # KNN Trainer
    knn_trainer = StandardizedKNNTrainer(model_id="ETHUSDT_binance_15m_analyst_knn")
    knn_results = knn_trainer.train_and_predict(X, y, data_start, data_end)

    # NGBoost Trainer
    ngb_trainer = StandardizedNGBoostTrainer(model_id="ETHUSDT_binance_15m_analyst_ngboost")
    ngb_results = ngb_trainer.train_and_predict(X, y, data_start, data_end)

    # LGBM Trainer
    lgbm_trainer = StandardizedLGBMTrainer(model_id="ETHUSDT_binance_15m_analyst_lgbm")
    lgbm_results = lgbm_trainer.train_and_predict(X, y, data_start, data_end)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

# Import retraining scheduler
from .retraining_scheduler import (
    OOFPredictionGenerator,
    RetrainingSchedule,
    RetrainingManager,
)

# Import optimization tools
from .optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Optional Dependencies
# ============================================================================

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import ngboost
    from ngboost import NGBRegressor, NGBClassifier
    from ngboost.distns import Normal  # Gaussian distribution
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    ngboost = None
    NGBRegressor = None
    NGBClassifier = None
    Normal = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KNeighborsClassifier = None
    KNeighborsRegressor = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


# ============================================================================
# Base Configuration and Results
# ============================================================================

@dataclass
class AnalystTrainingConfig:
    """Base configuration for standardized Analyst model training."""
    
    # Model identification
    model_id: str
    
    # Historical data retraining schedule (for OOF windows)
    retrain_interval_days: int = 10
    hpo_interval_days: int = 30
    burnin_pct: float = 1/12  # 3 months burn-in
    min_samples_for_training: int = 500
    
    # Task type
    task_type: str = "classification"  # "classification" or "regression"
    
    # HPO configuration
    hpo_n_trials: int = 50
    hpo_timeout: int = 1800  # 30 minutes max
    enable_warm_start: bool = True
    early_stopping_rounds: int = 35
    
    # Stratified sampling
    stratified_sampling_range: Tuple[float, float] = (0.1, 0.5)  # 10-50%
    
    # Paths
    cache_dir: Path = field(default_factory=lambda: Path("cache/analyst_models"))
    hpo_cache_dir: Path = field(default_factory=lambda: Path("cache/analyst_hpo"))
    
    def __post_init__(self):
        """Create cache directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hpo_cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AnalystTrainingResults:
    """Results from standardized Analyst model training."""
    
    oof_predictions: pd.DataFrame  # OOF predictions
    models: List[Any]  # List of trained models (one per window)
    metadata: List[Dict[str, Any]]  # Metadata for each training window
    hpo_history: Optional[Dict[str, Any]] = None
    training_windows: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# Base Analyst Trainer
# ============================================================================

class BaseAnalystTrainer(ABC):
    """
    Abstract base class for standardized Analyst model trainers.
    
    Implements common functionality:
    - OOF prediction generation
    - Retraining scheduling
    - HPO management
    - Stratified sampling
    - Warm start support
    """
    
    def __init__(self, model_id: str, config: Optional[AnalystTrainingConfig] = None):
        self.model_id = model_id
        self.config = config or AnalystTrainingConfig(model_id=model_id)
        self.retrain_manager = RetrainingManager(cache_dir=self.config.cache_dir)
        self._best_params_cache: Dict[str, Any] = {}
        self._hpo_window_counter = 0
        
        logger.info(f"Initialized {self.__class__.__name__} for model: {model_id}")
    
    def _get_stratified_sample_size(self, n_samples: int) -> int:
        """Calculate stratified sample size based on dataset size."""
        min_pct, max_pct = self.config.stratified_sampling_range
        
        # More data → smaller sampling percentage
        if n_samples < 5000:
            sample_pct = max_pct  # 50%
        elif n_samples < 20000:
            sample_pct = (max_pct + min_pct) / 2  # 30%
        elif n_samples < 50000:
            sample_pct = 0.2  # 20%
        else:
            sample_pct = min_pct  # 10%
        
        return max(int(n_samples * sample_pct), self.config.min_samples_for_training)
    
    def _should_run_hpo(self, window_id: int) -> bool:
        """Determine if HPO should run for this window."""
        windows_per_hpo = self.config.hpo_interval_days // self.config.retrain_interval_days
        return (window_id % windows_per_hpo) == 0
    
    def _load_warm_start_params(self) -> Optional[Dict[str, Any]]:
        """Load best parameters from previous HPO runs."""
        if not self.config.enable_warm_start:
            return None
        
        cache_file = self.config.hpo_cache_dir / f"{self.model_id}_best_params.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    params = json.load(f)
                logger.info(f"Loaded warm start params from {cache_file}")
                return params
            except Exception as e:
                logger.warning(f"Failed to load warm start params: {e}")
        
        return None
    
    def _save_best_params(self, params: Dict[str, Any]) -> None:
        """Save best parameters for warm start."""
        cache_file = self.config.hpo_cache_dir / f"{self.model_id}_best_params.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"Saved best params to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save best params: {e}")
    
    def train_and_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data_start: datetime,
        data_end: datetime,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> AnalystTrainingResults:
        """
        Train model with OOF predictions and scheduled retraining.
        
        Args:
            X: Feature dataframe with DatetimeIndex
            y: Target series with DatetimeIndex
            data_start: Start of available data
            data_end: End of available data
            sample_weight: Optional sample weights
            verbose: Whether to print progress
            
        Returns:
            AnalystTrainingResults with OOF predictions, models, metadata
        """
        if verbose:
            logger.info("=" * 80)
            logger.info(f"🚀 Starting {self.__class__.__name__} Training with OOF Predictions")
            logger.info("=" * 80)
            logger.info(f"Model ID: {self.model_id}")
            logger.info(f"Data range: {data_start} → {data_end}")
            logger.info(f"Samples: {len(X)}, Features: {len(X.columns)}")
            logger.info(f"Task: {self.config.task_type}")
        
        # Create retraining schedule
        schedule = RetrainingSchedule(
            model_type=self.__class__.__name__.lower(),
            retrain_interval_days=self.config.retrain_interval_days,
            burnin_pct=self.config.burnin_pct,
            min_samples_for_training=self.config.min_samples_for_training,
            enable_warm_start=False
        )
        
        # Create OOF prediction generator
        oof_generator = OOFPredictionGenerator(
            schedule=schedule,
            data_start=data_start,
            data_end=data_end
        )
        
        if verbose:
            logger.info(f"📊 Created {len(oof_generator.windows)} training windows")
        
        # Prepare aligned data
        aligned_data = pd.DataFrame(index=X.index)
        for col in X.columns:
            aligned_data[col] = X[col]
        aligned_data['__target__'] = y
        if sample_weight is not None:
            aligned_data['__weight__'] = sample_weight
        
        # Process windows
        all_predictions = []
        all_models = []
        all_metadata = []
        hpo_history = {}
        
        start_time = time.time()
        
        for window in oof_generator.windows:
            if verbose:
                logger.info(f"\n🔄 Processing Window {window.window_id + 1}/{len(oof_generator.windows)}")
            
            # Get training data
            train_mask = (aligned_data.index >= window.training_start) & (aligned_data.index < window.training_end)
            train_data = aligned_data.loc[train_mask].copy()
            
            if len(train_data) < schedule.min_samples_for_training:
                logger.warning(f"⚠️ Window {window.window_id}: Insufficient samples. Skipping.")
                continue
            
            # Train model
            window_start_time = time.time()
            use_hpo = self._should_run_hpo(window.window_id)
            
            model = self._train_single_window(
                train_data=train_data,
                window_id=window.window_id,
                use_hpo=use_hpo,
                verbose=verbose
            )
            all_models.append(model)
            training_time = time.time() - window_start_time
            
            # Get prediction data
            pred_mask = (aligned_data.index >= window.prediction_start) & (aligned_data.index <= window.prediction_end)
            pred_data = aligned_data.loc[pred_mask].copy()
            
            if len(pred_data) == 0:
                continue
            
            # Make predictions
            predictions = self._predict_single_window(model, pred_data)
            all_predictions.append(predictions)
            
            # Store metadata
            metadata = {
                'window_id': window.window_id,
                'training_samples': len(train_data),
                'prediction_samples': len(pred_data),
                'training_start': window.training_start.isoformat(),
                'training_end': window.training_end.isoformat(),
                'prediction_start': window.prediction_start.isoformat(),
                'prediction_end': window.prediction_end.isoformat(),
                'training_time': training_time,
                'used_hpo': use_hpo,
            }
            all_metadata.append(metadata)
            
            if verbose:
                logger.info(f"   ✅ Window complete in {training_time:.2f}s (HPO: {use_hpo})")
        
        # Combine predictions
        if all_predictions:
            oof_predictions = pd.concat(all_predictions, axis=0).sort_index()
        else:
            oof_predictions = pd.DataFrame()
        
        total_time = time.time() - start_time
        
        if verbose:
            logger.info("=" * 80)
            logger.info(f"✅ OOF Training Complete in {total_time:.2f}s")
            logger.info(f"   Total windows: {len(all_models)}")
            logger.info(f"   Total predictions: {len(oof_predictions)}")
            logger.info("=" * 80)
        
        return AnalystTrainingResults(
            oof_predictions=oof_predictions,
            models=all_models,
            metadata=all_metadata,
            hpo_history=hpo_history,
            training_windows=[{
                'window_id': w.window_id,
                'training_start': w.training_start.isoformat(),
                'training_end': w.training_end.isoformat(),
                'prediction_start': w.prediction_start.isoformat(),
                'prediction_end': w.prediction_end.isoformat()
            } for w in oof_generator.windows]
        )
    
    @abstractmethod
    def _train_single_window(
        self,
        train_data: pd.DataFrame,
        window_id: int,
        use_hpo: bool,
        verbose: bool
    ) -> Any:
        """Train model for a single window. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict_single_window(
        self,
        model: Any,
        pred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions for a single window. Implemented by subclasses."""
        pass


# ============================================================================
# KNN Configuration and Trainer
# ============================================================================

@dataclass
class KNNTrainingConfig(AnalystTrainingConfig):
    """Configuration for KNN training."""
    
    # KNN-specific parameters
    n_neighbors: int = 15
    metric: str = "euclidean"
    weights: str = "distance"
    
    # Faiss/HNSW settings
    use_faiss: bool = True
    faiss_nlist: int = 100  # Number of cells for IVF index
    faiss_nprobe: int = 10  # Number of cells to visit during search
    
    # HPO parameter ranges
    n_neighbors_range: Tuple[int, int] = (5, 50)
    
    def __post_init__(self):
        super().__post_init__()
        self.model_id = self.model_id or "knn_analyst"


class StandardizedKNNTrainer(BaseAnalystTrainer):
    """
    Standardized KNN trainer with OOF predictions and scheduled retraining.
    
    Features:
    - Approximate nearest neighbors (Faiss/HNSW) when available
    - Batch predictions
    - Euclidean distance
    - Stratified sampling
    """
    
    def __init__(self, model_id: str, config: Optional[KNNTrainingConfig] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for StandardizedKNNTrainer")
        
        self.knn_config = config or KNNTrainingConfig(model_id=model_id)
        super().__init__(model_id, self.knn_config)
        
        # Check Faiss availability
        self.use_faiss = self.knn_config.use_faiss and FAISS_AVAILABLE
        if self.knn_config.use_faiss and not FAISS_AVAILABLE:
            logger.warning("Faiss not available, falling back to sklearn KNN")
    
    def _create_faiss_index(self, X: np.ndarray) -> Any:
        """Create Faiss index for approximate nearest neighbors."""
        if not FAISS_AVAILABLE:
            return None
        
        d = X.shape[1]  # Dimensionality
        n = X.shape[0]  # Number of samples
        
        # Use IVF (Inverted File) index for larger datasets
        if n > 10000:
            nlist = min(self.knn_config.faiss_nlist, int(np.sqrt(n)))
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(X.astype(np.float32))
            index.nprobe = self.knn_config.faiss_nprobe
        else:
            # Use flat index for smaller datasets
            index = faiss.IndexFlatL2(d)
        
        index.add(X.astype(np.float32))
        return index
    
    def _train_single_window(
        self,
        train_data: pd.DataFrame,
        window_id: int,
        use_hpo: bool,
        verbose: bool
    ) -> Dict[str, Any]:
        """Train KNN model for a single window."""
        
        # Extract features and target
        feature_cols = [c for c in train_data.columns if not c.startswith('__')]
        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data['__target__'].values
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Stratified sampling for HPO
        if use_hpo:
            sample_size = self._get_stratified_sample_size(len(X_train))
            if sample_size < len(X_train):
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_sample = X_train[indices]
                y_sample = y_train[indices]
            else:
                X_sample = X_train
                y_sample = y_train
            
            # Run HPO
            best_params = self._run_knn_hpo(X_sample, y_sample, verbose)
            n_neighbors = best_params.get('n_neighbors', self.knn_config.n_neighbors)
        else:
            # Use warm start params or defaults
            warm_params = self._load_warm_start_params()
            n_neighbors = warm_params.get('n_neighbors', self.knn_config.n_neighbors) if warm_params else self.knn_config.n_neighbors
        
        # Create model package
        model_package = {
            'X_train': X_train,
            'y_train': y_train,
            'n_neighbors': n_neighbors,
            'feature_cols': feature_cols,
            'use_faiss': self.use_faiss,
            'is_classification': self.config.task_type == 'classification'
        }
        
        # Build Faiss index if available
        if self.use_faiss:
            model_package['faiss_index'] = self._create_faiss_index(X_train)
        else:
            # Create sklearn KNN
            if self.config.task_type == 'classification':
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    metric=self.knn_config.metric,
                    weights=self.knn_config.weights,
                    n_jobs=-1
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    metric=self.knn_config.metric,
                    weights=self.knn_config.weights,
                    n_jobs=-1
                )
            model.fit(X_train, y_train)
            model_package['sklearn_model'] = model
        
        return model_package
    
    def _run_knn_hpo(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> Dict[str, Any]:
        """Run HPO for KNN parameters."""
        if not OPTUNA_AVAILABLE:
            return {'n_neighbors': self.knn_config.n_neighbors}
        
        warm_params = self._load_warm_start_params()
        
        def objective(trial):
            # Suggest n_neighbors with warm start
            if warm_params and 'n_neighbors' in warm_params:
                n_neighbors = trial.suggest_int(
                    'n_neighbors',
                    self.knn_config.n_neighbors_range[0],
                    self.knn_config.n_neighbors_range[1],
                )
            else:
                n_neighbors = trial.suggest_int(
                    'n_neighbors',
                    self.knn_config.n_neighbors_range[0],
                    self.knn_config.n_neighbors_range[1]
                )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            if self.config.task_type == 'classification':
                model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            else:
                model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            
            return scores.mean()
        
        # Create study with warm start
        study = optuna.create_study(direction='maximize')
        
        # Add warm start trial
        if warm_params:
            try:
                study.enqueue_trial(warm_params)
            except Exception:
                pass
        
        study.optimize(objective, n_trials=min(self.config.hpo_n_trials, 20), show_progress_bar=verbose)
        
        best_params = study.best_params
        self._save_best_params(best_params)
        
        return best_params
    
    def _predict_single_window(
        self,
        model: Dict[str, Any],
        pred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions using batch prediction."""
        
        feature_cols = model['feature_cols']
        X_pred = pred_data[feature_cols].values.astype(np.float32)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        if model['use_faiss'] and 'faiss_index' in model:
            # Use Faiss for prediction
            k = model['n_neighbors']
            distances, indices = model['faiss_index'].search(X_pred, k)
            
            # Get neighbor labels
            y_neighbors = model['y_train'][indices]
            
            if model['is_classification']:
                # Majority vote with distance weighting
                from scipy.stats import mode
                predictions = mode(y_neighbors, axis=1).mode.flatten()
                
                # Calculate class probabilities
                unique_classes = np.unique(model['y_train'])
                probs = np.zeros((len(X_pred), len(unique_classes)))
                for i, pred in enumerate(predictions):
                    weights = 1 / (distances[i] + 1e-8)
                    for j, cls in enumerate(unique_classes):
                        cls_mask = y_neighbors[i] == cls
                        probs[i, j] = np.sum(weights[cls_mask]) / np.sum(weights)
                
                results = pd.DataFrame({
                    'prediction': predictions,
                    'probability': probs.max(axis=1)
                }, index=pred_data.index)
            else:
                # Weighted average for regression
                weights = 1 / (distances + 1e-8)
                predictions = np.sum(y_neighbors * weights, axis=1) / np.sum(weights, axis=1)
                
                results = pd.DataFrame({
                    'prediction': predictions
                }, index=pred_data.index)
        else:
            # Use sklearn model
            sklearn_model = model['sklearn_model']
            
            # Batch prediction
            predictions = sklearn_model.predict(X_pred)
            
            if model['is_classification'] and hasattr(sklearn_model, 'predict_proba'):
                probs = sklearn_model.predict_proba(X_pred)
                results = pd.DataFrame({
                    'prediction': predictions,
                    'probability': probs.max(axis=1)
                }, index=pred_data.index)
            else:
                results = pd.DataFrame({
                    'prediction': predictions
                }, index=pred_data.index)
        
        return results


# ============================================================================
# NGBoost Configuration and Trainer
# ============================================================================

@dataclass
class NGBoostTrainingConfig(AnalystTrainingConfig):
    """Configuration for NGBoost training."""
    
    # NGBoost-specific parameters (using Gaussian distribution)
    n_estimators: int = 500
    learning_rate: float = 0.01
    minibatch_frac: float = 0.5  # Subsample rows
    col_sample: float = 0.7  # Subsample columns (like colsample_bytree)
    
    # HPO parameter ranges
    learning_rate_range: Tuple[float, float] = (0.005, 0.1)
    n_estimators_range: Tuple[int, int] = (100, 1000)
    minibatch_frac_range: Tuple[float, float] = (0.1, 0.5)
    col_sample_range: Tuple[float, float] = (0.5, 0.9)
    
    def __post_init__(self):
        super().__post_init__()
        self.model_id = self.model_id or "ngboost_analyst"


class StandardizedNGBoostTrainer(BaseAnalystTrainer):
    """
    Standardized NGBoost trainer with OOF predictions and scheduled retraining.
    
    Features:
    - Gaussian distribution (not NormalWithNegBin or multi-parameter)
    - Stochastic methods (colsample + subsample)
    - Subsample rows (10-50%)
    - Early stopping
    """
    
    def __init__(self, model_id: str, config: Optional[NGBoostTrainingConfig] = None):
        if not NGBOOST_AVAILABLE:
            raise ImportError("ngboost is required for StandardizedNGBoostTrainer")
        
        self.ngb_config = config or NGBoostTrainingConfig(model_id=model_id)
        super().__init__(model_id, self.ngb_config)
    
    def _train_single_window(
        self,
        train_data: pd.DataFrame,
        window_id: int,
        use_hpo: bool,
        verbose: bool
    ) -> Any:
        """Train NGBoost model for a single window."""
        
        # Extract features and target
        feature_cols = [c for c in train_data.columns if not c.startswith('__')]
        X_train = train_data[feature_cols].values.astype(np.float64)
        y_train = train_data['__target__'].values
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split for validation (for early stopping)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Stratified sampling for HPO
        if use_hpo:
            sample_size = self._get_stratified_sample_size(len(X_tr))
            if sample_size < len(X_tr):
                indices = np.random.choice(len(X_tr), sample_size, replace=False)
                X_hpo = X_tr[indices]
                y_hpo = y_tr[indices]
            else:
                X_hpo = X_tr
                y_hpo = y_tr
            
            # Run HPO
            best_params = self._run_ngboost_hpo(X_hpo, y_hpo, X_val, y_val, verbose)
        else:
            warm_params = self._load_warm_start_params()
            best_params = warm_params if warm_params else {}
        
        # Create model with best/warm start params
        params = {
            'n_estimators': best_params.get('n_estimators', self.ngb_config.n_estimators),
            'learning_rate': best_params.get('learning_rate', self.ngb_config.learning_rate),
            'minibatch_frac': best_params.get('minibatch_frac', self.ngb_config.minibatch_frac),
            'col_sample': best_params.get('col_sample', self.ngb_config.col_sample),
            'verbose': verbose,
        }
        
        # Use Gaussian distribution (Normal)
        if self.config.task_type == 'classification':
            model = NGBClassifier(
                Dist=Normal,
                **params
            )
        else:
            model = NGBRegressor(
                Dist=Normal,
                **params
            )
        
        # Train with early stopping
        model.fit(
            X_tr, y_tr,
            X_val=X_val, Y_val=y_val,
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        # Store feature columns for prediction
        model.__feature_cols__ = feature_cols
        
        return model
    
    def _run_ngboost_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run HPO for NGBoost parameters."""
        if not OPTUNA_AVAILABLE:
            return {}
        
        warm_params = self._load_warm_start_params()
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    self.ngb_config.n_estimators_range[0],
                    self.ngb_config.n_estimators_range[1]
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.ngb_config.learning_rate_range[0],
                    self.ngb_config.learning_rate_range[1],
                    log=True
                ),
                'minibatch_frac': trial.suggest_float(
                    'minibatch_frac',
                    self.ngb_config.minibatch_frac_range[0],
                    self.ngb_config.minibatch_frac_range[1]
                ),
                'col_sample': trial.suggest_float(
                    'col_sample',
                    self.ngb_config.col_sample_range[0],
                    self.ngb_config.col_sample_range[1]
                ),
                'verbose': False
            }
            
            try:
                if self.config.task_type == 'classification':
                    model = NGBClassifier(Dist=Normal, **params)
                    model.fit(
                        X_train, y_train,
                        X_val=X_val, Y_val=y_val,
                        early_stopping_rounds=self.config.early_stopping_rounds
                    )
                    preds = model.predict_proba(X_val)[:, 1]
                    from sklearn.metrics import log_loss
                    return -log_loss(y_val, preds)
                else:
                    model = NGBRegressor(Dist=Normal, **params)
                    model.fit(
                        X_train, y_train,
                        X_val=X_val, Y_val=y_val,
                        early_stopping_rounds=self.config.early_stopping_rounds
                    )
                    preds = model.predict(X_val)
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y_val, preds)
            except Exception as e:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        
        # Add warm start trial
        if warm_params:
            try:
                study.enqueue_trial(warm_params)
            except Exception:
                pass
        
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials,
            timeout=self.config.hpo_timeout,
            show_progress_bar=verbose
        )
        
        best_params = study.best_params
        self._save_best_params(best_params)
        
        return best_params
    
    def _predict_single_window(
        self,
        model: Any,
        pred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions for a single window."""
        
        feature_cols = model.__feature_cols__
        X_pred = pred_data[feature_cols].values.astype(np.float64)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.config.task_type == 'classification':
            predictions = model.predict(X_pred)
            probs = model.predict_proba(X_pred)
            
            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            }, index=pred_data.index)
        else:
            predictions = model.predict(X_pred)
            # NGBoost also provides distribution parameters
            dist = model.pred_dist(X_pred)
            
            results = pd.DataFrame({
                'prediction': predictions,
                'std': dist.params['scale'] if hasattr(dist, 'params') else np.zeros_like(predictions)
            }, index=pred_data.index)
        
        return results


# ============================================================================
# LGBM Configuration and Trainer
# ============================================================================

@dataclass
class LGBMTrainingConfig(AnalystTrainingConfig):
    """Configuration for LightGBM training."""
    
    # LGBM-specific parameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    min_gain_to_split: float = 0.01
    
    # Bagging/sampling parameters
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    feature_fraction: float = 0.8
    feature_fraction_bynode: float = 0.8
    
    # Binary dataset
    use_binary_dataset: bool = True
    
    # For ensemble model: fixed max_depth=4
    is_ensemble_model: bool = False
    ensemble_max_depth: int = 4
    
    # HPO parameter ranges
    learning_rate_range: Tuple[float, float] = (0.01, 0.3)  # Higher range for faster training
    n_estimators_range: Tuple[int, int] = (100, 500)  # Lower iterations with higher LR
    max_depth_range: Tuple[int, int] = (4, 8)
    num_leaves_range: Tuple[int, int] = (16, 64)
    min_data_in_leaf_range: Tuple[int, int] = (10, 50)
    bagging_fraction_range: Tuple[float, float] = (0.6, 0.9)
    feature_fraction_range: Tuple[float, float] = (0.6, 0.9)
    
    def __post_init__(self):
        super().__post_init__()
        self.model_id = self.model_id or "lgbm_analyst"


class StandardizedLGBMTrainer(BaseAnalystTrainer):
    """
    Standardized LightGBM trainer with OOF predictions and scheduled retraining.
    
    Features:
    - Binary Dataset format for speed
    - Full-sized validation
    - Optimized min_data_in_leaf, min_gain_to_split
    - Bagging/feature fractions
    - Higher learning rate with fewer iterations option
    - Warm start from previous best params
    """
    
    def __init__(self, model_id: str, config: Optional[LGBMTrainingConfig] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required for StandardizedLGBMTrainer")
        
        self.lgbm_config = config or LGBMTrainingConfig(model_id=model_id)
        super().__init__(model_id, self.lgbm_config)
        
        # Temp directory for binary datasets
        self._temp_dir = tempfile.mkdtemp(prefix="lgbm_binary_")
    
    def _create_binary_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reference: Optional[lgb.Dataset] = None,
        save_path: Optional[str] = None
    ) -> lgb.Dataset:
        """Create LightGBM Dataset, optionally in binary format."""
        
        dataset = lgb.Dataset(X, label=y, reference=reference, free_raw_data=False)
        
        if save_path and self.lgbm_config.use_binary_dataset:
            dataset.save_binary(save_path)
            # Reload from binary for speed
            return lgb.Dataset(save_path)
        
        return dataset
    
    def _train_single_window(
        self,
        train_data: pd.DataFrame,
        window_id: int,
        use_hpo: bool,
        verbose: bool
    ) -> Any:
        """Train LightGBM model for a single window."""
        
        # Extract features and target
        feature_cols = [c for c in train_data.columns if not c.startswith('__')]
        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data['__target__'].values
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split for validation (keep validation full-sized)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Create binary datasets
        train_path = os.path.join(self._temp_dir, f"train_{window_id}.bin")
        val_path = os.path.join(self._temp_dir, f"val_{window_id}.bin")
        
        train_dataset = self._create_binary_dataset(X_tr, y_tr, save_path=train_path)
        val_dataset = self._create_binary_dataset(X_val, y_val, reference=train_dataset, save_path=val_path)
        
        # Stratified sampling for HPO
        if use_hpo:
            sample_size = self._get_stratified_sample_size(len(X_tr))
            if sample_size < len(X_tr):
                indices = np.random.choice(len(X_tr), sample_size, replace=False)
                X_hpo = X_tr[indices]
                y_hpo = y_tr[indices]
            else:
                X_hpo = X_tr
                y_hpo = y_tr
            
            # Run HPO
            best_params = self._run_lgbm_hpo(X_hpo, y_hpo, X_val, y_val, verbose)
        else:
            warm_params = self._load_warm_start_params()
            best_params = warm_params if warm_params else {}
        
        # Build parameters
        params = {
            'objective': 'binary' if self.config.task_type == 'classification' else 'regression',
            'metric': 'binary_logloss' if self.config.task_type == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1 if not verbose else 0,
            'n_jobs': -1,
            
            # Training params
            'learning_rate': best_params.get('learning_rate', self.lgbm_config.learning_rate),
            'num_leaves': best_params.get('num_leaves', self.lgbm_config.num_leaves),
            'min_data_in_leaf': best_params.get('min_data_in_leaf', self.lgbm_config.min_data_in_leaf),
            'min_gain_to_split': self.lgbm_config.min_gain_to_split,
            
            # Bagging/sampling
            'bagging_fraction': best_params.get('bagging_fraction', self.lgbm_config.bagging_fraction),
            'bagging_freq': self.lgbm_config.bagging_freq,
            'feature_fraction': best_params.get('feature_fraction', self.lgbm_config.feature_fraction),
            'feature_fraction_bynode': self.lgbm_config.feature_fraction_bynode,
        }
        
        # Handle max_depth
        if self.lgbm_config.is_ensemble_model:
            # Hardcoded max_depth=4 for ensemble
            params['max_depth'] = self.lgbm_config.ensemble_max_depth
        else:
            params['max_depth'] = best_params.get('max_depth', self.lgbm_config.max_depth)
        
        n_estimators = best_params.get('n_estimators', self.lgbm_config.n_estimators)
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[train_dataset, val_dataset],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
                lgb.log_evaluation(period=100) if verbose else lgb.log_evaluation(period=0)
            ]
        )
        
        # Store feature columns for prediction
        model.__feature_cols__ = feature_cols
        
        # Cleanup temporary binary files
        for path in [train_path, val_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        
        return model
    
    def _run_lgbm_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool
    ) -> Dict[str, Any]:
        """Run HPO for LightGBM parameters."""
        if not OPTUNA_AVAILABLE:
            return {}
        
        warm_params = self._load_warm_start_params()
        
        def objective(trial):
            params = {
                'objective': 'binary' if self.config.task_type == 'classification' else 'regression',
                'metric': 'binary_logloss' if self.config.task_type == 'classification' else 'rmse',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'n_jobs': -1,
                
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.lgbm_config.learning_rate_range[0],
                    self.lgbm_config.learning_rate_range[1],
                    log=True
                ),
                'num_leaves': trial.suggest_int(
                    'num_leaves',
                    self.lgbm_config.num_leaves_range[0],
                    self.lgbm_config.num_leaves_range[1]
                ),
                'max_depth': trial.suggest_int(
                    'max_depth',
                    self.lgbm_config.max_depth_range[0],
                    self.lgbm_config.max_depth_range[1]
                ) if not self.lgbm_config.is_ensemble_model else self.lgbm_config.ensemble_max_depth,
                'min_data_in_leaf': trial.suggest_int(
                    'min_data_in_leaf',
                    self.lgbm_config.min_data_in_leaf_range[0],
                    self.lgbm_config.min_data_in_leaf_range[1]
                ),
                'bagging_fraction': trial.suggest_float(
                    'bagging_fraction',
                    self.lgbm_config.bagging_fraction_range[0],
                    self.lgbm_config.bagging_fraction_range[1]
                ),
                'bagging_freq': self.lgbm_config.bagging_freq,
                'feature_fraction': trial.suggest_float(
                    'feature_fraction',
                    self.lgbm_config.feature_fraction_range[0],
                    self.lgbm_config.feature_fraction_range[1]
                ),
            }
            
            n_estimators = trial.suggest_int(
                'n_estimators',
                self.lgbm_config.n_estimators_range[0],
                self.lgbm_config.n_estimators_range[1]
            )
            
            train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
            
            try:
                model = lgb.train(
                    params,
                    train_dataset,
                    num_boost_round=n_estimators,
                    valid_sets=[val_dataset],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
                    ]
                )
                
                preds = model.predict(X_val)
                
                if self.config.task_type == 'classification':
                    from sklearn.metrics import log_loss
                    return -log_loss(y_val, preds)
                else:
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y_val, preds)
            except Exception as e:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        
        # Add warm start trial
        if warm_params:
            try:
                study.enqueue_trial(warm_params)
            except Exception:
                pass
        
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials,
            timeout=self.config.hpo_timeout,
            show_progress_bar=verbose
        )
        
        best_params = study.best_params
        self._save_best_params(best_params)
        
        return best_params
    
    def _predict_single_window(
        self,
        model: Any,
        pred_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions for a single window."""
        
        feature_cols = model.__feature_cols__
        X_pred = pred_data[feature_cols].values.astype(np.float32)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = model.predict(X_pred)
        
        if self.config.task_type == 'classification':
            # Predictions are probabilities for classification
            results = pd.DataFrame({
                'prediction': (predictions > 0.5).astype(int),
                'probability': predictions
            }, index=pred_data.index)
        else:
            results = pd.DataFrame({
                'prediction': predictions
            }, index=pred_data.index)
        
        return results
    
    def __del__(self):
        """Cleanup temporary directory on deletion."""
        import shutil
        try:
            if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception:
            pass


# ============================================================================
# Factory Function
# ============================================================================

def create_analyst_trainer(
    model_type: str,
    model_id: str,
    task_type: str = "classification",
    is_ensemble: bool = False,
    **kwargs
) -> BaseAnalystTrainer:
    """
    Factory function to create the appropriate Analyst trainer.
    
    Args:
        model_type: One of "knn", "ngboost", "lgbm"
        model_id: Unique model identifier
        task_type: "classification" or "regression"
        is_ensemble: If True and model_type=="lgbm", uses ensemble config
        **kwargs: Additional config parameters
        
    Returns:
        Appropriate standardized trainer instance
    """
    model_type = model_type.lower()
    
    if model_type == "knn":
        config = KNNTrainingConfig(model_id=model_id, task_type=task_type, **kwargs)
        return StandardizedKNNTrainer(model_id=model_id, config=config)
    
    elif model_type == "ngboost":
        config = NGBoostTrainingConfig(model_id=model_id, task_type=task_type, **kwargs)
        return StandardizedNGBoostTrainer(model_id=model_id, config=config)
    
    elif model_type == "lgbm":
        config = LGBMTrainingConfig(
            model_id=model_id,
            task_type=task_type,
            is_ensemble_model=is_ensemble,
            **kwargs
        )
        return StandardizedLGBMTrainer(model_id=model_id, config=config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: knn, ngboost, lgbm")


# ============================================================================
# Ensemble Helper
# ============================================================================

def create_lgbm_ensemble_trainer(model_id: str, task_type: str = "classification", **kwargs) -> StandardizedLGBMTrainer:
    """
    Create LGBM trainer specifically configured for ensemble model.
    
    This uses:
    - Hardcoded max_depth=4
    - Binary Dataset format
    - Same training/HPO schedule as other Analyst models
    
    Args:
        model_id: Unique model identifier
        task_type: "classification" or "regression"
        **kwargs: Additional config parameters
        
    Returns:
        StandardizedLGBMTrainer configured for ensemble use
    """
    config = LGBMTrainingConfig(
        model_id=model_id,
        task_type=task_type,
        is_ensemble_model=True,  # Forces max_depth=4
        **kwargs
    )
    return StandardizedLGBMTrainer(model_id=model_id, config=config)
