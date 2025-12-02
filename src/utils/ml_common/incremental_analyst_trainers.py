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
- CALIBRATION: All models are wrapped with Isotonic Regression calibration

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
    from sklearn.isotonic import IsotonicRegression
    from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
    from sklearn.model_selection import train_test_split, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KNeighborsClassifier = None
    KNeighborsRegressor = None
    BayesianRidge = None
    StandardScaler = None
    IsotonicRegression = None
    BaseEstimator = None
    RegressorMixin = None
    ClassifierMixin = None
    clone = None
    train_test_split = None
    cross_val_score = None

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
        'learning_rate', 'n_estimators', 'minibatch_frac', 'base_learner_max_depth'
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
# Calibration Wrapper
# ============================================================================

class CalibratedModel(BaseEstimator, RegressorMixin):
    """
    Wrapper that applies Isotonic Regression calibration to any base model.
    Handles both regression and classification (via probability).
    """

    def __init__(self, base_model, task_type='regression'):
        self.base_model = base_model
        self.task_type = task_type
        self.calibrator = IsotonicRegression(out_of_bounds='clip', increasing='auto')
        self.is_calibrated = False
        self._fitted_scaler = None  # For models that need scaling inside

    def fit_calibration(self, X_val, y_val):
        """Fit the calibrator using validation data."""
        if len(X_val) < 10:
            return  # Not enough data to calibrate

        # Get base predictions
        try:
            if self.task_type == 'classification' and hasattr(self.base_model, 'predict_proba'):
                raw_preds = self.base_model.predict_proba(X_val)[:, 1]
            else:
                raw_preds = self.base_model.predict(X_val)
        except Exception as e:
            logger.warning(f"Failed to get base predictions for calibration: {e}")
            return

        # Fit calibrator (maps raw_preds -> y_val)
        try:
            self.calibrator.fit(raw_preds, y_val)
            self.is_calibrated = True
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            self.is_calibrated = False

    def predict(self, X):
        """Predict with calibration."""
        if self.task_type == 'classification' and hasattr(self.base_model, 'predict_proba'):
            raw_preds = self.base_model.predict_proba(X)[:, 1]
        else:
            raw_preds = self.base_model.predict(X)

        if self.is_calibrated:
            return self.calibrator.predict(raw_preds)
        return raw_preds

    def predict_proba(self, X):
        """For classification, return calibrated probabilities."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")

        calibrated_preds = self.predict(X)
        # Return as (n_samples, 2) array [p0, p1]
        return np.vstack([1 - calibrated_preds, calibrated_preds]).T

    def __getattr__(self, name):
        """Delegate other attributes to base model."""
        return getattr(self.base_model, name)


# ============================================================================
# Incremental Training Windows Generator
# ============================================================================

class IncrementalWindowGenerator:
    """
    Generates training windows for incremental OOF prediction generation.
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
        
        # First window: burn-in training
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
            
            # Move to next window
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
        self._current_model = None  # Will hold a CalibratedModel instance
        self._best_params: Dict[str, Any] = {}
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
        """Train model incrementally with rolling OOF predictions."""
        start_time = time.time()
        self._specialist_feature_names = specialist_feature_names
        
        if verbose:
            logger.info("=" * 80)
            logger.info(f"🚀 Starting Incremental {self.__class__.__name__} Training")
            logger.info("=" * 80)
            logger.info(f"Burn-in period: {self.config.burn_in_days} days")
        
        self._best_params = self._load_best_params()
        
        window_generator = IncrementalWindowGenerator(
            config=self.config,
            data_start=data_start,
            data_end=data_end
        )
        
        # Prepare aligned data
        aligned_data = pd.DataFrame(index=X.index)
        for col in X.columns:
            aligned_data[col] = X[col]
        aligned_data['__target__'] = y
        if sample_weight is not None:
            aligned_data['__weight__'] = sample_weight
        
        all_predictions = []
        all_metadata = []
        hpo_history = {}
        previous_training_end = None
        
        for window in window_generator.windows:
            if verbose:
                logger.info(f"\n🔄 Window {window.window_id + 1}/{len(window_generator.windows)}")
            
            window_start_time = time.time()
            
            # Data selection
            train_mask = (aligned_data.index >= window.training_start) & (aligned_data.index < window.training_end)
            train_data = aligned_data.loc[train_mask].copy()
            
            pred_mask = (aligned_data.index >= window.prediction_start) & (aligned_data.index < window.prediction_end)
            pred_data = aligned_data.loc[pred_mask].copy()
            
            # Validation check
            if len(train_data) > 0 and len(pred_data) > 0:
                train_max_ts = train_data.index.max()
                pred_min_ts = pred_data.index.min()
                if train_max_ts >= pred_min_ts:
                    logger.error(f"⚠️ LEAKAGE: Train max {train_max_ts} >= Pred min {pred_min_ts}. Skipping.")
                    previous_training_end = window.training_end
                    continue
            
            if len(train_data) < 100:
                logger.warning(f"Window {window.window_id}: Insufficient samples. Skipping.")
                previous_training_end = window.training_end
                continue
            
            # Initial HPO if burn-in and enabled
            if window.is_burn_in and self.config.enable_incremental_hpo:
                if verbose: logger.info("   Running initial HPO on burn-in data...")
                hpo_result = self._run_incremental_hpo(train_data, window, verbose)
                if hpo_result:
                    hpo_history[f"window_{window.window_id}_pre"] = hpo_result

            # Train/Update model
            if window.is_burn_in or self._current_model is None:
                self._current_model = self._train_initial(train_data, window, verbose)
            else:
                if previous_training_end is not None:
                    new_data_mask = (aligned_data.index >= previous_training_end) & (aligned_data.index < window.training_end)
                    new_data = aligned_data.loc[new_data_mask].copy()
                    
                    if len(new_data) > 0:
                        self._current_model = self._train_incremental(new_data, train_data, window, verbose)
                    else:
                        logger.info(f"   No new data, using current model")
                else:
                    self._current_model = self._train_initial(train_data, window, verbose)
            
            previous_training_end = window.training_end
            
            # Predict
            if len(pred_data) > 0:
                predictions = self._predict(pred_data)
                all_predictions.append(predictions)
            
            # HPO for next rounds
            if self.config.enable_incremental_hpo and not window.is_burn_in:
                hpo_result = self._run_incremental_hpo(train_data, window, verbose)
                if hpo_result:
                    hpo_history[f"window_{window.window_id}"] = hpo_result
            
            # Metadata
            metadata = {
                'window_id': window.window_id,
                'training_samples': len(train_data),
                'prediction_samples': len(pred_data),
                'training_time': time.time() - window_start_time,
                'is_burn_in': window.is_burn_in,
                **window.to_dict()
            }
            if hasattr(self._current_model, 'is_calibrated'):
                 metadata['calibration_status'] = self._current_model.is_calibrated

            all_metadata.append(metadata)
        
        # Combine
        oof_predictions = pd.concat(all_predictions, axis=0).sort_index() if all_predictions else pd.DataFrame()
        self._save_best_params(self._best_params)
        
        return IncrementalTrainingResults(
            oof_predictions=oof_predictions,
            final_model=self._current_model,
            window_metadata=all_metadata,
            hpo_history=hpo_history,
            best_params=self._best_params,
            training_time=time.time() - start_time
        )
    
    @abstractmethod
    def _train_initial(self, train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> Any:
        pass
    
    @abstractmethod
    def _train_incremental(self, new_data: pd.DataFrame, full_train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> Any:
        pass
    
    @abstractmethod
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def _run_incremental_hpo(self, train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> Optional[Dict[str, Any]]:
        pass
    
    def _get_feature_cols(self, data: pd.DataFrame) -> List[str]:
        all_features = [c for c in data.columns if not c.startswith('__')]
        if self.model_config.get('use_specialist_outputs_only', False) and self._specialist_feature_names:
            specialist_set = set(self._specialist_feature_names)
            return [f for f in all_features if f in specialist_set]
        return all_features


# ============================================================================
# Incremental LGBM Trainer
# ============================================================================

class IncrementalLGBMTrainer(BaseIncrementalTrainer):
    """Incremental LightGBM trainer with calibration."""
    
    def __init__(self, model_id: str, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
    
    def _get_default_params(self) -> Dict[str, Any]:
        params = {
            'objective': 'regression' if self.config.task_type == 'regression' else 'binary',
            'metric': 'rmse' if self.config.task_type == 'regression' else 'binary_logloss',
            'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1,
            'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6,
            'reg_alpha': 0.1, 'reg_lambda': 0.1
        }
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params
    
    def _train_initial(self, train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> CalibratedModel:
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        
        params = self._get_default_params()
        params.update(self._best_params)
        n_est = params.pop('n_estimators', 500)
        
        model = lgb.train(
            params, train_ds, num_boost_round=n_est,
            valid_sets=[train_ds, val_ds],
            callbacks=[lgb.early_stopping(35), lgb.log_evaluation(0)]
        )
        
        # Wrap and calibrate
        calibrated_model = CalibratedModel(model, self.config.task_type)
        calibrated_model.fit_calibration(X_val, y_val)

        if verbose:
            logger.info(f"   Initial LGBM: {model.num_trees()} trees, calibrated={calibrated_model.is_calibrated}")
        
        return calibrated_model
    
    def _train_incremental(self, new_data: pd.DataFrame, full_train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> CalibratedModel:
        X = full_train_data[self._feature_cols].values.astype(np.float32)
        y = full_train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        # Access base model from wrapper
        prev_lgbm = self._current_model.base_model
        
        model = lgb.train(
            params, train_ds, num_boost_round=100,
            valid_sets=[train_ds, val_ds],
            init_model=prev_lgbm,
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        calibrated_model = CalibratedModel(model, self.config.task_type)
        calibrated_model.fit_calibration(X_val, y_val)
        
        return calibrated_model
    
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        X = pred_data[self._feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        
        # Predict using calibrated wrapper
        preds = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            return pd.DataFrame({'prediction': (preds > 0.5).astype(int), 'probability': preds}, index=pred_data.index)
        else:
            return pd.DataFrame({'prediction': preds}, index=pred_data.index)

    def _run_incremental_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            for param_name in self.config.sensitive_params_lgbm:
                current_val = current_best.get(param_name)
                if param_name == 'learning_rate':
                    base = current_val or 0.05
                    params['learning_rate'] = trial.suggest_float('learning_rate', max(0.001, base*(1-radius)), min(0.3, base*(1+radius)), log=True)
                elif param_name == 'num_leaves':
                    base = current_val or 31
                    params['num_leaves'] = trial.suggest_int('num_leaves', max(8, int(base*(1-radius))), min(128, int(base*(1+radius))))
                elif param_name == 'max_depth':
                    base = current_val or 6
                    params['max_depth'] = trial.suggest_int('max_depth', max(3, int(base*(1-radius))), min(12, int(base*(1+radius))))
                elif param_name in ['reg_alpha', 'reg_lambda']:
                    base = current_val or 0.1
                    params[param_name] = trial.suggest_float(param_name, max(0.0, base*(1-radius)), min(5.0, base*(1+radius)))
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            try:
                model = lgb.train(params, dtrain, num_boost_round=200, valid_sets=[dval], callbacks=[lgb.early_stopping(20)])
                preds = model.predict(X_val)
                if self.config.task_type == 'regression':
                    return -((y_val - preds)**2).mean()
                else:
                    from sklearn.metrics import log_loss
                    return -log_loss(y_val, preds)
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}


# ============================================================================
# Incremental NGBoost Trainer
# ============================================================================

class IncrementalNGBoostTrainer(BaseIncrementalTrainer):
    """Incremental NGBoost trainer with calibration."""
    
    def __init__(self, model_id: str, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
    
    def _get_default_params(self):
        params = {'n_estimators': 500, 'learning_rate': 0.01, 'minibatch_frac': 0.5, 'verbose': False}
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params
    
    def _create_base_learner(self, params):
        from ngboost.learners import default_tree_learner
        tree_params = dict(default_tree_learner.get_params(deep=True))
        tree_params['max_depth'] = int(params.get('base_learner_max_depth', 3))
        if 'base_learner_min_samples_leaf' in params:
            tree_params['min_samples_leaf'] = int(params['base_learner_min_samples_leaf'])
        return default_tree_learner.__class__(**tree_params)

    def _train_initial(self, train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> CalibratedModel:
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        params = self._get_default_params()
        params.update(self._best_params)
        base_learner = self._create_base_learner(params)
        ng_params = {k: v for k, v in params.items() if k != 'base_learner_max_depth' and k != 'base_learner_min_samples_leaf'}
        
        if self.config.task_type == 'classification':
            model = NGBClassifier(Dist=Normal, Base=base_learner, **ng_params)
        else:
            model = NGBRegressor(Dist=Normal, Base=base_learner, **ng_params)

        model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=35)
        
        calibrated_model = CalibratedModel(model, self.config.task_type)
        calibrated_model.fit_calibration(X_val, y_val)
        
        if verbose:
            logger.info(f"   Initial NGBoost trained, calibrated={calibrated_model.is_calibrated}")
        
        return calibrated_model

    def _train_incremental(self, new_data, full_train_data, window, verbose) -> CalibratedModel:
        # Re-train on full accumulated data (NGBoost doesn't support true partial_fit)
        # Using consistent params to minimize drift
        return self._train_initial(full_train_data, window, verbose)

    def _predict(self, pred_data) -> pd.DataFrame:
        X = pred_data[self._feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        preds = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            # Calibrated probabilities
            probs = self._current_model.predict_proba(X) # Returns [p0, p1] from calibrated wrapper
            return pd.DataFrame({'prediction': (preds > 0.5).astype(int), 'probability': probs[:, 1]}, index=pred_data.index)
        else:
            # Get uncertainty from base model
            try:
                dist = self._current_model.base_model.pred_dist(X)
                std = dist.params.get('scale', np.zeros_like(preds))
            except:
                std = np.zeros_like(preds)
            return pd.DataFrame({'prediction': preds, 'std': std}, index=pred_data.index)

    def _run_incremental_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            for param_name in self.config.sensitive_params_ngboost:
                current_val = current_best.get(param_name)
                if param_name == 'learning_rate':
                    base = current_val or 0.01
                    params['learning_rate'] = trial.suggest_float('learning_rate', max(0.01, base*(1-radius)), min(0.05, base*(1+radius)), log=True)
                elif param_name == 'n_estimators':
                    base = current_val or 500
                    params['n_estimators'] = trial.suggest_int('n_estimators', max(100, int(base*(1-radius))), min(1000, int(base*(1+radius))))
                elif param_name == 'minibatch_frac':
                    base = current_val or 0.5
                    params['minibatch_frac'] = trial.suggest_float('minibatch_frac', max(0.1, base*(1-radius)), min(1.0, base*(1+radius)))
                elif param_name == 'base_learner_max_depth':
                    params['base_learner_max_depth'] = trial.suggest_int('base_learner_max_depth', 3, 5)
            
            try:
                base_learner = self._create_base_learner(params)
                ng_params = {k: v for k, v in params.items() if k != 'base_learner_max_depth' and k != 'base_learner_min_samples_leaf'}
                if self.config.task_type == 'classification':
                    model = NGBClassifier(Dist=Normal, Base=base_learner, **ng_params)
                else:
                    model = NGBRegressor(Dist=Normal, Base=base_learner, **ng_params)
                model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=35)
                preds = model.predict(X_val)
                if self.config.task_type == 'regression':
                    return -((y_val - preds)**2).mean()
                else:
                    from sklearn.metrics import log_loss
                    probs = model.predict_proba(X_val)
                    return -log_loss(y_val, probs)
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}


# ============================================================================
# Incremental KNN Trainer
# ============================================================================

class IncrementalKNNTrainer(BaseIncrementalTrainer):
    def __init__(self, model_id, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._scaler = None

    def _get_default_params(self):
        params = {'n_neighbors': 15, 'weights': 'distance', 'n_jobs': -1}
        if self.model_config: params.update(self.model_config.get('params', {}))
        return params

    def _train_initial(self, train_data, window, verbose) -> CalibratedModel:
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Split raw data first
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale only training data for fit
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_raw)
        
        params = self._get_default_params()
        # Cap neighbors
        n_samples = len(y_train)
        max_k = max(5, int(np.sqrt(n_samples)))
        if 'n_neighbors' in params: params['n_neighbors'] = min(params['n_neighbors'], max_k)
        
        if self.config.task_type == 'classification':
            model = KNeighborsClassifier(**params)
        else:
            model = KNeighborsRegressor(**params)

        model.fit(X_train_scaled, y_train)
        
        # Wrapper that handles scaling
        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))
            def predict_proba(self, X):
                return self.model.predict_proba(self.scaler.transform(X))

        # Calibrate using raw validation data (wrapper handles scaling)
        calibrated_model = CalibratedModel(ScaledModel(model, self._scaler), self.config.task_type)
        calibrated_model.fit_calibration(X_val_raw, y_val)
        
        return calibrated_model

    def _train_incremental(self, new_data, full_train_data, window, verbose):
        # KNN just refits on all data
        return self._train_initial(full_train_data, window, verbose)

    def _predict(self, pred_data) -> pd.DataFrame:
        X = pred_data[self._feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        
        preds = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            probs = self._current_model.predict_proba(X)
            return pd.DataFrame({'prediction': preds, 'probability': probs[:, 1]}, index=pred_data.index)
        else:
            return pd.DataFrame({'prediction': preds}, index=pred_data.index)

    def _run_incremental_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)

        # We need scaling for HPO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        max_k = max(5, int(np.sqrt(len(y))))
        
        def objective(trial):
            params = self._get_default_params()
            for param_name in self.config.sensitive_params_knn:
                current_val = current_best.get(param_name)
                if param_name == 'n_neighbors':
                    base = current_val or 15
                    params['n_neighbors'] = trial.suggest_int('n_neighbors', max(3, int(base*(1-radius))), min(max_k, int(base*(1+radius))))
                elif param_name == 'leaf_size':
                    base = current_val or 30
                    params['leaf_size'] = trial.suggest_int('leaf_size', max(10, int(base*(1-radius))), min(100, int(base*(1+radius))))
            
            try:
                if self.config.task_type == 'classification':
                    model = KNeighborsClassifier(**params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                else:
                    model = KNeighborsRegressor(**params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
                return scores.mean()
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}


# ============================================================================
# Incremental BayesianRidge Trainer
# ============================================================================

class IncrementalBayesianRidgeTrainer(BaseIncrementalTrainer):
    def __init__(self, model_id, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._scaler = None
        self._prev_params = {}

    def _get_default_params(self):
        params = {'max_iter': 300, 'tol': 1e-3}
        if self.model_config: params.update(self.model_config.get('params', {}))
        return params

    def _train_initial(self, train_data, window, verbose) -> CalibratedModel:
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Split raw
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self._scaler = StandardScaler()
        X_tr_scaled = self._scaler.fit_transform(X_tr)
        
        params = self._get_default_params()
        params.update(self._best_params)
        model = BayesianRidge(**params)
        model.fit(X_tr_scaled, y_tr)
        
        # Store for warm start
        self._prev_params = {'alpha_init': model.alpha_, 'lambda_init': model.lambda_}
        
        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))
        
        calibrated_model = CalibratedModel(ScaledModel(model, self._scaler), self.config.task_type)
        calibrated_model.fit_calibration(X_va, y_va)
        
        return calibrated_model

    def _train_incremental(self, new_data, full_train_data, window, verbose):
        # Similar logic but using warm start params
        X = full_train_data[self._feature_cols].values.astype(np.float64)
        y = full_train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

        # Update scaler with all data
        self._scaler.fit(X_tr)
        X_tr_scaled = self._scaler.transform(X_tr)
        
        params = self._get_default_params()
        params.update(self._best_params)
        params.update(self._prev_params) # Warm start
        
        model = BayesianRidge(**params)
        model.fit(X_tr_scaled, y_tr)
        self._prev_params = {'alpha_init': model.alpha_, 'lambda_init': model.lambda_}
        
        # Re-calibrate
        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))

        calibrated_model = CalibratedModel(ScaledModel(model, self._scaler), self.config.task_type)
        calibrated_model.fit_calibration(X_va, y_va)
        
        return calibrated_model

    def _predict(self, pred_data) -> pd.DataFrame:
        X = pred_data[self._feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        preds = self._current_model.predict(X)
        # Base model std?
        try:
            _, std = self._current_model.base_model.model.predict(self._current_model.base_model.scaler.transform(X), return_std=True)
        except:
            std = np.zeros_like(preds)

        return pd.DataFrame({'prediction': preds, 'std': std}, index=pred_data.index)

    def _run_incremental_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None
        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            for param_name in self.config.sensitive_params_bayesian:
                current_val = current_best.get(param_name)
                if param_name in ['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']:
                    base = current_val or 1e-6
                    params[param_name] = trial.suggest_float(param_name, max(1e-10, base*(1-radius)), min(1e-2, base*(1+radius)), log=True)
            
            try:
                model = BayesianRidge(**params)
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
                return scores.mean()
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}

# ============================================================================
# Unified Incremental Analyst Trainer
# ============================================================================

class IncrementalAnalystTrainer:
    """Unified incremental trainer that trains all analyst base models."""
    
    def __init__(self, model_id: str, execution_mode="blank", task_type="regression", enable_incremental_hpo=True, model_configs=None):
        self.model_id = model_id
        self.config = IncrementalTrainingConfig(model_id, execution_mode, oof_batch_days=14, task_type=task_type, enable_incremental_hpo=enable_incremental_hpo)
        self.model_configs = model_configs or {}
        
        self.trainers = {}
        if LIGHTGBM_AVAILABLE:
            self.trainers['lgbm'] = IncrementalLGBMTrainer(model_id, self.config, self.model_configs.get('lgbm'))
        if NGBOOST_AVAILABLE:
            self.trainers['ngboost'] = IncrementalNGBoostTrainer(model_id, self.config, self.model_configs.get('ngboost'))
        if SKLEARN_AVAILABLE:
            self.trainers['knn'] = IncrementalKNNTrainer(model_id, self.config, self.model_configs.get('knn'))
            self.trainers['bayesianridge'] = IncrementalBayesianRidgeTrainer(model_id, self.config, self.model_configs.get('bayesianridge'))
            
    def train_all_models(self, X, y, data_start, data_end, sample_weight=None, verbose=True, specialist_feature_names=None):
        results = {}
        for name, trainer in self.trainers.items():
            if verbose: logger.info(f"Training {name}")
            results[name] = trainer.train_and_predict(X, y, data_start, data_end, sample_weight, verbose, specialist_feature_names)
        return results

    def get_combined_oof_predictions(self, results):
        combined = {}
        for name, res in results.items():
            if not res.oof_predictions.empty:
                for col in res.oof_predictions.columns:
                    combined[f"{name}_{col}"] = res.oof_predictions[col]
        return pd.DataFrame(combined)
