"""
Incremental Analyst Model Trainers with Rolling OOF Predictions

This module provides incremental training for Analyst base models:
- LGBM: incremental training via init_model
- NGBoost: incremental training via warm_start/init_model
- KNN: incremental addition of points
- BayesianRidge: incremental training via partial_fit

Key Features:
- Train on burn-in period -> Generate OOF for next 14 days -> Resume training
- Incremental HPO at each round (neighborhood search around best params)
- Rolling incremental training ensures OOF predictions come from models that haven't seen that batch
- Burn-in period = full_period - 4 months

Usage:
    ```python
    from src.utils.ml_common.incremental_analyst_trainers import IncrementalAnalystTrainer
    
    trainer = IncrementalAnalystTrainer(
        model_id="ETHUSDT_binance_15m",
        execution_mode="blank",  # 1 year for blank, 3+ years for full
    )
    results = trainer.train_all_models(X, y, data_start, data_end)
    ```
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

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
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    ngboost = None
    NGBRegressor = None
    NGBClassifier = None
    Normal = None

try:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KNeighborsClassifier = None
    KNeighborsRegressor = None
    BayesianRidge = None
    StandardScaler = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


# ============================================================================
# Configuration
# ============================================================================

# Burn-in is full_period - 4 months
BURN_IN_BUFFER_DAYS = 120  # 4 months

# OOF batch size (14 days)
OOF_BATCH_DAYS = 14

# Mode lookback days (from ares_launcher)
MODE_LOOKBACK_DAYS = {
    "light": 30,
    "blank": 360,  # 1 year
    "full": 365 * 3  # 3 years
}


@dataclass
class IncrementalTrainingConfig:
    """Configuration for incremental training."""
    
    # Model identification
    model_id: str
    
    # Execution mode
    execution_mode: str = "blank"
    
    # OOF batch configuration
    oof_batch_days: int = OOF_BATCH_DAYS
    
    # Task type
    task_type: str = "regression"  # "classification" or "regression"
    
    # HPO configuration (incremental)
    enable_incremental_hpo: bool = True
    hpo_n_trials_per_round: int = 10  # Small number for incremental search
    hpo_timeout_per_round: int = 300  # 5 minutes max per round
    early_stopping_rounds: int = 35
    
    # Sensitive hyperparameters to tune incrementally
    sensitive_params_lgbm: List[str] = field(default_factory=lambda: [
        'learning_rate', 'num_leaves', 'max_depth', 'reg_alpha', 'reg_lambda'
    ])
    sensitive_params_ngboost: List[str] = field(default_factory=lambda: [
        'learning_rate', 'n_estimators', 'minibatch_frac'
    ])
    sensitive_params_knn: List[str] = field(default_factory=lambda: [
        'n_neighbors', 'leaf_size'
    ])
    sensitive_params_bayesian: List[str] = field(default_factory=lambda: [
        'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2'
    ])
    
    # Neighborhood search radius (percentage of current value)
    hpo_neighborhood_radius: float = 0.3  # 30% around current best
    
    # Paths
    cache_dir: Path = field(default_factory=lambda: Path("cache/incremental_models"))
    hpo_cache_dir: Path = field(default_factory=lambda: Path("cache/incremental_hpo"))
    
    def __post_init__(self):
        """Create cache directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hpo_cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def full_period_days(self) -> int:
        """Get full period in days based on execution mode."""
        return MODE_LOOKBACK_DAYS.get(self.execution_mode, 360)
    
    @property
    def burn_in_days(self) -> int:
        """Calculate burn-in period as full_period - 4 months."""
        return max(30, self.full_period_days - BURN_IN_BUFFER_DAYS)


@dataclass
class IncrementalTrainingWindow:
    """A single training window for incremental OOF generation."""
    
    window_id: int
    training_start: datetime
    training_end: datetime  # End of training data (exclusive)
    prediction_start: datetime  # Start of OOF prediction batch
    prediction_end: datetime  # End of OOF prediction batch
    is_burn_in: bool = False  # True if this is the initial burn-in window
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_id': self.window_id,
            'training_start': self.training_start.isoformat(),
            'training_end': self.training_end.isoformat(),
            'prediction_start': self.prediction_start.isoformat(),
            'prediction_end': self.prediction_end.isoformat(),
            'is_burn_in': self.is_burn_in
        }


@dataclass 
class IncrementalTrainingResults:
    """Results from incremental training."""
    
    oof_predictions: pd.DataFrame  # OOF predictions for all windows
    final_model: Any  # Final trained model
    window_metadata: List[Dict[str, Any]]  # Metadata for each window
    hpo_history: Dict[str, Any]  # HPO history
    best_params: Dict[str, Any]  # Final best parameters
    training_time: float  # Total training time


# ============================================================================
# Incremental Training Windows Generator
# ============================================================================

class IncrementalWindowGenerator:
    """
    Generates training windows for incremental OOF prediction generation.
    
    Burn-in period = full_period - 4 months
    After burn-in, generate OOF predictions for 14-day batches
    """
    
    def __init__(
        self,
        config: IncrementalTrainingConfig,
        data_start: datetime,
        data_end: datetime
    ):
        self.config = config
        self.data_start = data_start
        self.data_end = data_end
        self.windows = self._create_windows()
    
    def _create_windows(self) -> List[IncrementalTrainingWindow]:
        """Create training windows for incremental OOF generation."""
        windows = []
        
        # Calculate burn-in end
        burn_in_end = self.data_start + timedelta(days=self.config.burn_in_days)
        
        # Ensure burn-in end doesn't exceed data end
        if burn_in_end >= self.data_end:
            logger.warning(
                f"Burn-in period ({self.config.burn_in_days} days) exceeds data range. "
                f"Adjusting to 50% of data."
            )
            total_days = (self.data_end - self.data_start).days
            burn_in_end = self.data_start + timedelta(days=int(total_days * 0.5))
        
        # First window: burn-in training (no OOF for this period)
        current_training_end = burn_in_end
        window_id = 0
        
        # Generate windows for OOF prediction
        while current_training_end < self.data_end:
            # Prediction period: next oof_batch_days
            prediction_start = current_training_end
            prediction_end = min(
                prediction_start + timedelta(days=self.config.oof_batch_days),
                self.data_end
            )
            
            window = IncrementalTrainingWindow(
                window_id=window_id,
                training_start=self.data_start,
                training_end=current_training_end,
                prediction_start=prediction_start,
                prediction_end=prediction_end,
                is_burn_in=(window_id == 0)
            )
            windows.append(window)
            
            # Move to next window - extend training to include the prediction batch
            current_training_end = prediction_end
            window_id += 1
        
        logger.info(
            f"Created {len(windows)} incremental training windows. "
            f"Burn-in: {self.config.burn_in_days} days, "
            f"OOF batch: {self.config.oof_batch_days} days"
        )
        
        return windows


# ============================================================================
# Base Incremental Trainer
# ============================================================================

class BaseIncrementalTrainer(ABC):
    """
    Abstract base class for incremental model trainers.
    
    Implements:
    - Rolling OOF prediction generation
    - Incremental training (model resumption)
    - Incremental HPO (neighborhood search)
    """
    
    def __init__(
        self,
        model_id: str,
        config: Optional[IncrementalTrainingConfig] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.config = config or IncrementalTrainingConfig(model_id=model_id)
        self.model_config = model_config or {}
        self._current_model = None
        self._best_params: Dict[str, Any] = {}
        self._training_data_accumulated: Optional[pd.DataFrame] = None
        self._specialist_feature_names: Optional[List[str]] = None
        
        logger.info(f"Initialized {self.__class__.__name__} for {model_id}")
    
    def _load_best_params(self) -> Dict[str, Any]:
        """Load best parameters from cache."""
        cache_file = self.config.hpo_cache_dir / f"{self.model_id}_{self.__class__.__name__}_params.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached params: {e}")
        return {}
    
    def _save_best_params(self, params: Dict[str, Any]) -> None:
        """Save best parameters to cache."""
        cache_file = self.config.hpo_cache_dir / f"{self.model_id}_{self.__class__.__name__}_params.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save params: {e}")
    
    def train_and_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data_start: datetime,
        data_end: datetime,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True,
        specialist_feature_names: Optional[List[str]] = None
    ) -> IncrementalTrainingResults:
        """
        Train model incrementally with rolling OOF predictions.
        
        1. Train on burn-in period
        2. Generate OOF predictions for next 14 days
        3. Resume training with additional data
        4. Repeat until end of data
        """
        start_time = time.time()
        self._specialist_feature_names = specialist_feature_names
        
        if verbose:
            logger.info("=" * 80)
            logger.info(f"🚀 Starting Incremental {self.__class__.__name__} Training")
            logger.info("=" * 80)
            logger.info(f"Model ID: {self.model_id}")
            logger.info(f"Execution mode: {self.config.execution_mode}")
            logger.info(f"Burn-in period: {self.config.burn_in_days} days")
            logger.info(f"OOF batch size: {self.config.oof_batch_days} days")
            logger.info(f"Data range: {data_start} → {data_end}")
            logger.info(f"Samples: {len(X)}, Features: {len(X.columns)}")

            if self.model_config.get('use_specialist_outputs_only', False):
                spec_count = len(specialist_feature_names) if specialist_feature_names else 0
                logger.info(f"⚠️ Using ONLY specialist outputs ({spec_count} features)")
        
        # Load initial best params
        self._best_params = self._load_best_params()
        
        # Create training windows
        window_generator = IncrementalWindowGenerator(
            config=self.config,
            data_start=data_start,
            data_end=data_end
        )
        
        if verbose:
            logger.info(f"📊 Created {len(window_generator.windows)} training windows")
        
        # Prepare aligned data
        aligned_data = pd.DataFrame(index=X.index)
        for col in X.columns:
            aligned_data[col] = X[col]
        aligned_data['__target__'] = y
        if sample_weight is not None:
            aligned_data['__weight__'] = sample_weight
        
        # Process windows
        all_predictions = []
        all_metadata = []
        hpo_history = {}
        previous_training_end = None  # Track previous window's training boundary
        
        for window in window_generator.windows:
            if verbose:
                logger.info(f"\n🔄 Window {window.window_id + 1}/{len(window_generator.windows)}")
                logger.info(f"   Training: {window.training_start} → {window.training_end}")
                logger.info(f"   Predict:  {window.prediction_start} → {window.prediction_end}")
            
            window_start_time = time.time()
            
            # Get training data (all data from start up to training_end)
            # CRITICAL: training_end MUST be < prediction_start to ensure no leakage
            train_mask = (aligned_data.index >= window.training_start) & (aligned_data.index < window.training_end)
            train_data = aligned_data.loc[train_mask].copy()
            
            # Get prediction data (data that model has NOT seen yet)
            pred_mask = (aligned_data.index >= window.prediction_start) & (aligned_data.index < window.prediction_end)
            pred_data = aligned_data.loc[pred_mask].copy()
            
            # VALIDATION: Ensure no overlap between training and prediction data
            # Note: training uses `< training_end` (exclusive), prediction uses `>= prediction_start` (inclusive)
            # So if training_end == prediction_start, there's no overlap
            if len(train_data) > 0 and len(pred_data) > 0:
                train_max_ts = train_data.index.max()  # Actual last training sample
                pred_min_ts = pred_data.index.min()    # Actual first prediction sample
                
                # Check for actual data overlap (training sample timestamp >= prediction sample timestamp)
                if train_max_ts >= pred_min_ts:
                    logger.error(
                        f"⚠️ DATA LEAKAGE DETECTED! Last training sample ({train_max_ts}) >= "
                        f"First prediction sample ({pred_min_ts}). Skipping window."
                    )
                    previous_training_end = window.training_end
                    continue
                else:
                    if verbose:
                        gap_seconds = (pred_min_ts - train_max_ts).total_seconds()
                        logger.info(f"   ✓ OOF integrity verified: {gap_seconds/60:.1f} min gap between train/predict")
            
            if len(train_data) < 100:
                logger.warning(f"Window {window.window_id}: Insufficient training samples ({len(train_data)}). Skipping.")
                previous_training_end = window.training_end
                continue
            
            # Train or resume training
            if window.is_burn_in or self._current_model is None:
                # Full initial training on burn-in data
                self._current_model = self._train_initial(
                    train_data=train_data,
                    window=window,
                    verbose=verbose
                )
            else:
                # Incremental training (resume from previous model)
                # Get only the NEW data that wasn't in the previous training set
                # This is the data from PREVIOUS training_end to CURRENT training_end
                # (i.e., the previous OOF batch that we can now add to training)
                if previous_training_end is not None:
                    new_data_mask = (
                        (aligned_data.index >= previous_training_end) & 
                        (aligned_data.index < window.training_end)
                    )
                    new_data = aligned_data.loc[new_data_mask].copy()
                    
                    if verbose:
                        logger.info(f"   New data for incremental training: {len(new_data)} samples")
                    
                    if len(new_data) > 0:
                        self._current_model = self._train_incremental(
                            new_data=new_data,
                            full_train_data=train_data,
                            window=window,
                            verbose=verbose
                        )
                    else:
                        # No new data, just continue with current model
                        logger.info(f"   No new data for incremental training, using current model")
                else:
                    # Fallback: full training if we don't have previous boundary
                    self._current_model = self._train_initial(
                        train_data=train_data,
                        window=window,
                        verbose=verbose
                    )
            
            # Update previous training boundary for next iteration
            previous_training_end = window.training_end
            
            if len(pred_data) == 0:
                logger.warning(f"Window {window.window_id}: No prediction data. Skipping predictions.")
                continue
            
            # Generate OOF predictions
            predictions = self._predict(pred_data)
            all_predictions.append(predictions)
            
            # Run incremental HPO if enabled
            if self.config.enable_incremental_hpo and not window.is_burn_in:
                hpo_result = self._run_incremental_hpo(
                    train_data=train_data,
                    window=window,
                    verbose=verbose
                )
                if hpo_result:
                    hpo_history[f"window_{window.window_id}"] = hpo_result
            
            window_time = time.time() - window_start_time
            
            # Store metadata
            metadata = {
                'window_id': window.window_id,
                'training_samples': len(train_data),
                'prediction_samples': len(pred_data),
                'training_time': window_time,
                'is_burn_in': window.is_burn_in,
                **window.to_dict()
            }
            all_metadata.append(metadata)
            
            if verbose:
                logger.info(f"   ✅ Window complete in {window_time:.2f}s")
        
        # Combine predictions
        if all_predictions:
            oof_predictions = pd.concat(all_predictions, axis=0).sort_index()
        else:
            oof_predictions = pd.DataFrame()
        
        total_time = time.time() - start_time
        
        # Save final best params
        self._save_best_params(self._best_params)
        
        if verbose:
            logger.info("=" * 80)
            logger.info(f"✅ Incremental Training Complete in {total_time:.2f}s")
            logger.info(f"   Total windows: {len(all_metadata)}")
            logger.info(f"   Total OOF predictions: {len(oof_predictions)}")
            logger.info("=" * 80)
        
        return IncrementalTrainingResults(
            oof_predictions=oof_predictions,
            final_model=self._current_model,
            window_metadata=all_metadata,
            hpo_history=hpo_history,
            best_params=self._best_params,
            training_time=total_time
        )
    
    @abstractmethod
    def _train_initial(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """Train initial model on burn-in data."""
        pass
    
    @abstractmethod
    def _train_incremental(
        self,
        new_data: pd.DataFrame,
        full_train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """Incrementally train model with new data."""
        pass
    
    @abstractmethod
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def _run_incremental_hpo(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Run incremental HPO with neighborhood search."""
        pass
    
    def _get_feature_cols(self, data: pd.DataFrame) -> List[str]:
        """Get feature column names (exclude __ prefixed columns)."""
        all_features = [c for c in data.columns if not c.startswith('__')]

        # Filter for specialist features if configured
        if self.model_config.get('use_specialist_outputs_only', False):
            if self._specialist_feature_names:
                specialist_set = set(self._specialist_feature_names)
                filtered_features = [f for f in all_features if f in specialist_set]
                if not filtered_features:
                    logger.warning("Feature filtering enabled but no specialist features found in data! Using all features.")
                    return all_features
                return filtered_features
            else:
                logger.warning("Feature filtering enabled but no specialist feature names provided! Using all features.")
                return all_features

        return all_features


# ============================================================================
# Incremental LightGBM Trainer
# ============================================================================

class IncrementalLGBMTrainer(BaseIncrementalTrainer):
    """
    Incremental LightGBM trainer using init_model for warm starting.
    """
    
    def __init__(
        self,
        model_id: str,
        config: Optional[IncrementalTrainingConfig] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for IncrementalLGBMTrainer")
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default LGBM parameters."""
        params = {
            'objective': 'regression' if self.config.task_type == 'regression' else 'binary',
            'metric': 'rmse' if self.config.task_type == 'regression' else 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'feature_fraction': 0.8,
        }
        # Override with config params if available
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params
    
    def _train_initial(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> lgb.Booster:
        """Train initial LGBM model."""
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        # Merge default params with cached best params
        params = self._get_default_params()
        params.update(self._best_params)
        
        n_estimators = params.pop('n_estimators', 500)
        
        # Train model
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
        
        if verbose:
            logger.info(f"   Initial LGBM trained: {model.num_trees()} trees")
        
        return model
    
    def _train_incremental(
        self,
        new_data: pd.DataFrame,
        full_train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> lgb.Booster:
        """Incrementally train LGBM using init_model."""
        X = full_train_data[self._feature_cols].values.astype(np.float32)
        y = full_train_data['__target__'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        # Continue training from previous model using init_model
        additional_rounds = max(50, min(100, len(new_data) // 10))
        
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=additional_rounds,
            valid_sets=[train_dataset, val_dataset],
            valid_names=['train', 'valid'],
            init_model=self._current_model,  # Resume from previous model
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds // 2),
                lgb.log_evaluation(period=0)
            ]
        )
        
        if verbose:
            logger.info(f"   Incremental LGBM: +{additional_rounds} rounds → {model.num_trees()} total trees")
        
        return model
    
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """Generate LGBM predictions."""
        X = pred_data[self._feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            return pd.DataFrame({
                'prediction': (predictions > 0.5).astype(int),
                'probability': predictions
            }, index=pred_data.index)
        else:
            return pd.DataFrame({
                'prediction': predictions
            }, index=pred_data.index)
    
    def _run_incremental_hpo(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Run incremental HPO with neighborhood search around current best."""
        if not OPTUNA_AVAILABLE:
            return None
        
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            
            # Neighborhood search around current best
            for param_name in self.config.sensitive_params_lgbm:
                current_val = current_best.get(param_name)
                
                if param_name == 'learning_rate':
                    base = current_val or 0.05
                    params['learning_rate'] = trial.suggest_float(
                        'learning_rate',
                        max(0.001, base * (1 - radius)),
                        min(0.3, base * (1 + radius)),
                        log=True
                    )
                elif param_name == 'num_leaves':
                    base = current_val or 31
                    params['num_leaves'] = trial.suggest_int(
                        'num_leaves',
                        max(8, int(base * (1 - radius))),
                        min(128, int(base * (1 + radius)))
                    )
                elif param_name == 'max_depth':
                    base = current_val or 6
                    params['max_depth'] = trial.suggest_int(
                        'max_depth',
                        max(3, int(base * (1 - radius))),
                        min(12, int(base * (1 + radius)))
                    )
                elif param_name in ['reg_alpha', 'reg_lambda']:
                    base = current_val or 0.1
                    params[param_name] = trial.suggest_float(
                        param_name,
                        max(0.0, base * (1 - radius)),
                        min(5.0, base * (1 + radius))
                    )
            
            train_dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
            
            try:
                model = lgb.train(
                    params,
                    train_dataset,
                    num_boost_round=200,
                    valid_sets=[val_dataset],
                    callbacks=[lgb.early_stopping(stopping_rounds=20)]
                )
                
                preds = model.predict(X_val)
                
                if self.config.task_type == 'regression':
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y_val, preds)
                else:
                    from sklearn.metrics import log_loss
                    return -log_loss(y_val, preds)
            except Exception:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials_per_round,
            timeout=self.config.hpo_timeout_per_round,
            show_progress_bar=verbose
        )
        
        # Update best params if improved
        if study.best_trial is not None:
            self._best_params.update(study.best_params)
            if verbose:
                logger.info(f"   HPO: Updated params - {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }


# ============================================================================
# Incremental NGBoost Trainer
# ============================================================================

class IncrementalNGBoostTrainer(BaseIncrementalTrainer):
    """
    Incremental NGBoost trainer using warm_start.
    """
    
    def __init__(
        self,
        model_id: str,
        config: Optional[IncrementalTrainingConfig] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        if not NGBOOST_AVAILABLE:
            raise ImportError("NGBoost is required for IncrementalNGBoostTrainer")
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
        self._accumulated_X: Optional[np.ndarray] = None
        self._accumulated_y: Optional[np.ndarray] = None
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default NGBoost parameters."""
        params = {
            'n_estimators': 500,
            'learning_rate': 0.01,
            'minibatch_frac': 0.5,
            'verbose': False,
        }
        # Override with config params if available
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params
    
    def _train_initial(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """Train initial NGBoost model."""
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store accumulated data
        self._accumulated_X = X.copy()
        self._accumulated_y = y.copy()
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        if self.config.task_type == 'classification':
            model = NGBClassifier(Dist=Normal, **params)
        else:
            model = NGBRegressor(Dist=Normal, **params)
        
        model.fit(
            X_train, y_train,
            X_val=X_val, Y_val=y_val,
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        if verbose:
            n_est = getattr(model, 'n_estimators', 'N/A')
            logger.info(f"   Initial NGBoost trained: {n_est} estimators")
        
        return model
    
    def _train_incremental(
        self,
        new_data: pd.DataFrame,
        full_train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """
        Incrementally train NGBoost.
        
        NGBoost doesn't have native incremental training with warm_start.
        To prevent drift in probabilistic predictions, we:
        1. Retrain on ALL accumulated data with CONSISTENT hyperparameters
        2. Use the same n_estimators as initial training
        3. Track distribution parameters across windows for drift monitoring
        """
        X = full_train_data[self._feature_cols].values.astype(np.float64)
        y = full_train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update accumulated data
        self._accumulated_X = X.copy()
        self._accumulated_y = y.copy()
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # CRITICAL: Use SAME parameters as initial training to prevent drift
        # Only update learning-related params, not structural ones
        params = self._get_default_params()
        params.update(self._best_params)
        
        # Keep n_estimators consistent - don't reduce it
        # This ensures probabilistic calibration remains stable
        # Early stopping will naturally limit actual iterations
        
        if self.config.task_type == 'classification':
            model = NGBClassifier(Dist=Normal, **params)
        else:
            model = NGBRegressor(Dist=Normal, **params)
        
        # Store previous model's distribution stats for drift detection
        prev_dist_stats = None
        if self._current_model is not None:
            try:
                # Sample some predictions from previous model for comparison
                sample_X = X_val[:min(100, len(X_val))]
                prev_dist = self._current_model.pred_dist(sample_X)
                if hasattr(prev_dist, 'params'):
                    prev_dist_stats = {
                        'mean_loc': float(np.mean(prev_dist.params.get('loc', [0]))),
                        'mean_scale': float(np.mean(prev_dist.params.get('scale', [1])))
                    }
            except Exception:
                pass
        
        model.fit(
            X_train, y_train,
            X_val=X_val, Y_val=y_val,
            early_stopping_rounds=self.config.early_stopping_rounds
        )
        
        # Check for drift in probabilistic predictions
        if prev_dist_stats is not None:
            try:
                sample_X = X_val[:min(100, len(X_val))]
                new_dist = model.pred_dist(sample_X)
                if hasattr(new_dist, 'params'):
                    new_mean_scale = float(np.mean(new_dist.params.get('scale', [1])))
                    scale_drift = abs(new_mean_scale - prev_dist_stats['mean_scale']) / max(prev_dist_stats['mean_scale'], 1e-6)
                    if scale_drift > 0.5:  # 50% drift threshold
                        logger.warning(
                            f"   ⚠️ NGBoost scale drift detected: {scale_drift:.2%}. "
                            f"Previous mean_scale: {prev_dist_stats['mean_scale']:.4f}, "
                            f"New mean_scale: {new_mean_scale:.4f}"
                        )
            except Exception as e:
                logger.debug(f"Could not check NGBoost drift: {e}")
        
        n_est = getattr(model, 'n_estimators', 'N/A')
        if verbose:
            logger.info(f"   Incremental NGBoost retrained on {len(X_train)} samples: {n_est} estimators")
        
        return model
    
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """Generate NGBoost predictions."""
        X = pred_data[self._feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            probs = self._current_model.predict_proba(X)
            return pd.DataFrame({
                'prediction': predictions,
                'probability': probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            }, index=pred_data.index)
        else:
            # Get distribution for uncertainty
            dist = self._current_model.pred_dist(X)
            std = dist.params.get('scale', np.zeros_like(predictions)) if hasattr(dist, 'params') else np.zeros_like(predictions)
            
            return pd.DataFrame({
                'prediction': predictions,
                'std': std
            }, index=pred_data.index)
    
    def _run_incremental_hpo(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Run incremental HPO for NGBoost."""
        if not OPTUNA_AVAILABLE:
            return None
        
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            
            for param_name in self.config.sensitive_params_ngboost:
                current_val = current_best.get(param_name)
                
                if param_name == 'learning_rate':
                    base = current_val or 0.01
                    params['learning_rate'] = trial.suggest_float(
                        'learning_rate',
                        max(0.001, base * (1 - radius)),
                        min(0.1, base * (1 + radius)),
                        log=True
                    )
                elif param_name == 'n_estimators':
                    base = current_val or 500
                    params['n_estimators'] = trial.suggest_int(
                        'n_estimators',
                        max(100, int(base * (1 - radius))),
                        min(1000, int(base * (1 + radius)))
                    )
                elif param_name == 'minibatch_frac':
                    base = current_val or 0.5
                    params['minibatch_frac'] = trial.suggest_float(
                        'minibatch_frac',
                        max(0.1, base * (1 - radius)),
                        min(1.0, base * (1 + radius))
                    )
            
            try:
                if self.config.task_type == 'classification':
                    model = NGBClassifier(Dist=Normal, **params)
                else:
                    model = NGBRegressor(Dist=Normal, **params)
                
                model.fit(
                    X_train, y_train,
                    X_val=X_val, Y_val=y_val,
                    early_stopping_rounds=20
                )
                
                preds = model.predict(X_val)
                
                if self.config.task_type == 'regression':
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y_val, preds)
                else:
                    from sklearn.metrics import log_loss
                    probs = model.predict_proba(X_val)
                    return -log_loss(y_val, probs)
            except Exception:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials_per_round,
            timeout=self.config.hpo_timeout_per_round,
            show_progress_bar=verbose
        )
        
        if study.best_trial is not None:
            self._best_params.update(study.best_params)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }


# ============================================================================
# Incremental KNN Trainer
# ============================================================================

class IncrementalKNNTrainer(BaseIncrementalTrainer):
    """
    Incremental KNN trainer with point accumulation.
    
    KNN doesn't require retraining - we just accumulate points.
    """
    
    def __init__(
        self,
        model_id: str,
        config: Optional[IncrementalTrainingConfig] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for IncrementalKNNTrainer")
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
        self._accumulated_X: List[np.ndarray] = []
        self._accumulated_y: List[np.ndarray] = []
        self._scaler: Optional[StandardScaler] = None
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default KNN parameters."""
        params = {
            'n_neighbors': 15,
            'weights': 'distance',
            'algorithm': 'ball_tree',
            'leaf_size': 30,
            'metric': 'minkowski',
            'p': 2,
            'n_jobs': -1
        }
        # Override with config params if available
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params
    
    def _train_initial(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """Train initial KNN model."""
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features (important for KNN)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Store accumulated data
        self._accumulated_X = [X_scaled]
        self._accumulated_y = [y]
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        if self.config.task_type == 'classification':
            model = KNeighborsClassifier(**params)
        else:
            model = KNeighborsRegressor(**params)
        
        model.fit(X_scaled, y)
        
        if verbose:
            logger.info(f"   Initial KNN trained: {len(y)} points, k={params['n_neighbors']}")
        
        return model
    
    def _train_incremental(
        self,
        new_data: pd.DataFrame,
        full_train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """
        Incrementally add points to KNN.
        
        KNN is lazy - we just accumulate points and refit.
        """
        X = full_train_data[self._feature_cols].values.astype(np.float32)
        y = full_train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update scaler with all data
        X_scaled = self._scaler.fit_transform(X)
        
        # Update accumulated data
        self._accumulated_X = [X_scaled]
        self._accumulated_y = [y]
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        # Adjust n_neighbors based on data size
        n_samples = len(y)
        max_neighbors = max(5, int(np.sqrt(n_samples)))
        params['n_neighbors'] = min(params.get('n_neighbors', 15), max_neighbors)
        
        if self.config.task_type == 'classification':
            model = KNeighborsClassifier(**params)
        else:
            model = KNeighborsRegressor(**params)
        
        model.fit(X_scaled, y)
        
        if verbose:
            logger.info(f"   Incremental KNN: {len(y)} points, k={params['n_neighbors']}")
        
        return model
    
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """Generate KNN predictions."""
        X = pred_data[self._feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)
        
        predictions = self._current_model.predict(X_scaled)
        
        if self.config.task_type == 'classification':
            probs = self._current_model.predict_proba(X_scaled)
            return pd.DataFrame({
                'prediction': predictions,
                'probability': probs.max(axis=1)
            }, index=pred_data.index)
        else:
            return pd.DataFrame({
                'prediction': predictions
            }, index=pred_data.index)
    
    def _run_incremental_hpo(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Run incremental HPO for KNN."""
        if not OPTUNA_AVAILABLE:
            return None
        
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        max_k = max(5, int(np.sqrt(len(y))))
        
        def objective(trial):
            params = self._get_default_params()
            
            for param_name in self.config.sensitive_params_knn:
                current_val = current_best.get(param_name)
                
                if param_name == 'n_neighbors':
                    base = current_val or 15
                    params['n_neighbors'] = trial.suggest_int(
                        'n_neighbors',
                        max(3, int(base * (1 - radius))),
                        min(max_k, int(base * (1 + radius)))
                    )
                elif param_name == 'leaf_size':
                    base = current_val or 30
                    params['leaf_size'] = trial.suggest_int(
                        'leaf_size',
                        max(10, int(base * (1 - radius))),
                        min(100, int(base * (1 + radius)))
                    )
            
            try:
                from sklearn.model_selection import cross_val_score
                
                if self.config.task_type == 'classification':
                    model = KNeighborsClassifier(**params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                else:
                    model = KNeighborsRegressor(**params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
                
                return scores.mean()
            except Exception:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials_per_round,
            timeout=self.config.hpo_timeout_per_round,
            show_progress_bar=verbose
        )
        
        if study.best_trial is not None:
            self._best_params.update(study.best_params)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }


# ============================================================================
# Incremental BayesianRidge Trainer
# ============================================================================

class IncrementalBayesianRidgeTrainer(BaseIncrementalTrainer):
    """
    Incremental BayesianRidge trainer using partial_fit-like pattern.
    
    Note: sklearn's BayesianRidge doesn't have partial_fit, but we can
    warm-start by using the previous coefficients as prior information.
    """
    
    def __init__(
        self,
        model_id: str,
        config: Optional[IncrementalTrainingConfig] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for IncrementalBayesianRidgeTrainer")
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._prev_coef: Optional[np.ndarray] = None
        self._prev_alpha: Optional[float] = None
        self._prev_lambda: Optional[float] = None
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default BayesianRidge parameters."""
        params = {
            'max_iter': 300,  # n_iter renamed to max_iter in newer sklearn
            'tol': 1e-3,
            'alpha_1': 1e-6,
            'alpha_2': 1e-6,
            'lambda_1': 1e-6,
            'lambda_2': 1e-6,
            'compute_score': True,
            'fit_intercept': True
        }
        # Override with config params if available
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params
    
    def _train_initial(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """Train initial BayesianRidge model."""
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        model = BayesianRidge(**params)
        model.fit(X_scaled, y)
        
        # Store for warm starting
        self._prev_coef = model.coef_.copy()
        self._prev_alpha = model.alpha_
        self._prev_lambda = model.lambda_
        
        if verbose:
            logger.info(f"   Initial BayesianRidge trained: alpha={model.alpha_:.4f}, lambda={model.lambda_:.4f}")
        
        return model
    
    def _train_incremental(
        self,
        new_data: pd.DataFrame,
        full_train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Any:
        """
        Incrementally train BayesianRidge.
        
        We use the previous model's learned parameters to initialize
        the prior for the new model.
        """
        X = full_train_data[self._feature_cols].values.astype(np.float64)
        y = full_train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update scaler
        X_scaled = self._scaler.fit_transform(X)
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        # Use previous alpha/lambda as starting point for prior
        if self._prev_alpha is not None:
            params['alpha_init'] = self._prev_alpha
        if self._prev_lambda is not None:
            params['lambda_init'] = self._prev_lambda
        
        model = BayesianRidge(**params)
        model.fit(X_scaled, y)
        
        # Update for next round
        self._prev_coef = model.coef_.copy()
        self._prev_alpha = model.alpha_
        self._prev_lambda = model.lambda_
        
        if verbose:
            logger.info(f"   Incremental BayesianRidge: alpha={model.alpha_:.4f}, lambda={model.lambda_:.4f}")
        
        return model
    
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """Generate BayesianRidge predictions."""
        X = pred_data[self._feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)
        
        predictions, std = self._current_model.predict(X_scaled, return_std=True)
        
        return pd.DataFrame({
            'prediction': predictions,
            'std': std
        }, index=pred_data.index)
    
    def _run_incremental_hpo(
        self,
        train_data: pd.DataFrame,
        window: IncrementalTrainingWindow,
        verbose: bool
    ) -> Optional[Dict[str, Any]]:
        """Run incremental HPO for BayesianRidge."""
        if not OPTUNA_AVAILABLE:
            return None
        
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            
            for param_name in self.config.sensitive_params_bayesian:
                current_val = current_best.get(param_name)
                
                if param_name in ['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']:
                    base = current_val or 1e-6
                    params[param_name] = trial.suggest_float(
                        param_name,
                        max(1e-10, base * (1 - radius)),
                        min(1e-2, base * (1 + radius)),
                        log=True
                    )
            
            try:
                model = BayesianRidge(**params)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_val)
                
                from sklearn.metrics import mean_squared_error
                return -mean_squared_error(y_val, preds)
            except Exception:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=self.config.hpo_n_trials_per_round,
            timeout=self.config.hpo_timeout_per_round,
            show_progress_bar=verbose
        )
        
        if study.best_trial is not None:
            self._best_params.update(study.best_params)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }


# ============================================================================
# Unified Incremental Analyst Trainer
# ============================================================================

class IncrementalAnalystTrainer:
    """
    Unified incremental trainer that trains all analyst base models.
    
    Trains: LGBM, NGBoost, KNN, BayesianRidge
    """
    
    def __init__(
        self,
        model_id: str,
        execution_mode: str = "blank",
        task_type: str = "regression",
        enable_incremental_hpo: bool = True,
        model_configs: Optional[Dict[str, Any]] = None
    ):
        self.model_id = model_id
        self.execution_mode = execution_mode
        self.task_type = task_type
        self.model_configs = model_configs or {}
        
        # Create config
        self.config = IncrementalTrainingConfig(
            model_id=model_id,
            execution_mode=execution_mode,
            task_type=task_type,
            enable_incremental_hpo=enable_incremental_hpo
        )
        
        # Initialize trainers
        self.trainers: Dict[str, BaseIncrementalTrainer] = {}
        
        if LIGHTGBM_AVAILABLE:
            self.trainers['lgbm'] = IncrementalLGBMTrainer(model_id, self.config, self.model_configs.get('lgbm'))
        else:
            logger.warning("LightGBM not available, skipping LGBM trainer")
        
        if NGBOOST_AVAILABLE:
            self.trainers['ngboost'] = IncrementalNGBoostTrainer(model_id, self.config, self.model_configs.get('ngboost'))
        else:
            logger.warning("NGBoost not available, skipping NGBoost trainer")
        
        if SKLEARN_AVAILABLE:
            self.trainers['knn'] = IncrementalKNNTrainer(model_id, self.config, self.model_configs.get('knn'))
            self.trainers['bayesianridge'] = IncrementalBayesianRidgeTrainer(model_id, self.config, self.model_configs.get('bayesianridge'))
        else:
            logger.warning("scikit-learn not available, skipping KNN and BayesianRidge trainers")
        
        logger.info(f"IncrementalAnalystTrainer initialized with {len(self.trainers)} models: {list(self.trainers.keys())}")
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data_start: datetime,
        data_end: datetime,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True,
        specialist_feature_names: Optional[List[str]] = None
    ) -> Dict[str, IncrementalTrainingResults]:
        """
        Train all enabled models incrementally.
        
        Returns:
            Dictionary mapping model names to their training results
        """
        results = {}
        
        for model_name, trainer in self.trainers.items():
            if verbose:
                logger.info(f"\n{'='*80}")
                logger.info(f"Training {model_name.upper()}")
                logger.info(f"{'='*80}")
            
            try:
                result = trainer.train_and_predict(
                    X=X,
                    y=y,
                    data_start=data_start,
                    data_end=data_end,
                    sample_weight=sample_weight,
                    verbose=verbose,
                    specialist_feature_names=specialist_feature_names
                )
                results[model_name] = result
                
                if verbose:
                    logger.info(f"✅ {model_name}: {len(result.oof_predictions)} OOF predictions")
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def get_combined_oof_predictions(
        self,
        results: Dict[str, IncrementalTrainingResults]
    ) -> pd.DataFrame:
        """Combine OOF predictions from all models into a single DataFrame."""
        combined = {}
        
        for model_name, result in results.items():
            if result.oof_predictions is not None and not result.oof_predictions.empty:
                for col in result.oof_predictions.columns:
                    combined[f"{model_name}_{col}"] = result.oof_predictions[col]
        
        if combined:
            return pd.DataFrame(combined)
        return pd.DataFrame()
