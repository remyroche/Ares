"""
Incremental Analyst Model Trainers with Rolling OOF Predictions

This module provides incremental training for Analyst base models:
- LGBM: incremental training via init_model
- NGBoost: incremental training via warm_start/init_model
- KNN: incremental addition of points
- RidgeClassifier (replaces BayesianRidge): incremental training via warm_start/partial_fit simulation

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
        task_type="classification" # Now defaults to classification for Analyst models
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
from scipy.special import expit, erfinv
from src.utils.ml_common.oof_probability_calibration import get_recommended_calibration_method

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
    from ngboost.distns import Normal, Bernoulli
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    ngboost = None
    NGBRegressor = None
    NGBClassifier = None
    Normal = None
    Bernoulli = None

try:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import BayesianRidge, RidgeClassifier, LogisticRegression
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
    RidgeClassifier = None
    LogisticRegression = None
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
    # Suppress verbose Optuna trial logs (INFO level) to reduce log pollution
    optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    task_type: str = "classification"  # Defaulting to classification as per request
    calibration_method: str = "isotonic"  # "isotonic", "platt", "auto"
    
    # HPO configuration - runs ONCE at burn-in only (NOT during incremental windows)
    # This finds good hyperparameters on the initial data, then warm-starts from there
    enable_burnin_hpo: bool = True  # Run HPO once at burn-in
    hpo_n_trials: int = 10  # Number of trials for burn-in HPO
    hpo_timeout: int = 300  # 5 minutes max for burn-in HPO
    early_stopping_rounds: int = 35
    
    # Hyperparameters to tune during burn-in HPO
    sensitive_params_lgbm: List[str] = field(default_factory=lambda: [
        'learning_rate', 'num_leaves', 'max_depth', 'reg_alpha', 'reg_lambda'
    ])
    sensitive_params_ngboost: List[str] = field(default_factory=lambda: [
        'learning_rate', 'n_estimators', 'minibatch_frac', 'base_learner_max_depth'
    ])
    sensitive_params_knn: List[str] = field(default_factory=lambda: [
        'n_neighbors', 'leaf_size'
    ])
    sensitive_params_ridge: List[str] = field(default_factory=lambda: [
        'alpha', 'tol'
    ])
    
    # Neighborhood search radius for HPO (percentage of default value)
    hpo_neighborhood_radius: float = 0.3  # 30% around default

    # Backward compatibility aliases
    @property
    def enable_incremental_hpo(self) -> bool:
        """Deprecated: Use enable_burnin_hpo instead."""
        return self.enable_burnin_hpo

    @property
    def hpo_n_trials_per_round(self) -> int:
        """Deprecated: Use hpo_n_trials instead."""
        return self.hpo_n_trials

    @property
    def hpo_timeout_per_round(self) -> int:
        """Deprecated: Use hpo_timeout instead."""
        return self.hpo_timeout
    
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
        """Calculate burn-in period as full_period - 4 months (or 8 months for full mode)."""
        buffer_days = 240 if self.execution_mode == "full" else BURN_IN_BUFFER_DAYS
        return max(30, self.full_period_days - buffer_days)


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

    def __init__(self, base_model, task_type='classification', calibration_method='isotonic'):
        self.base_model = base_model
        self.task_type = task_type
        self.calibration_method = calibration_method
        self.calibrator = None
        self.is_calibrated = False
        self._fitted_scaler = None  # For models that need scaling inside
        self._resolved_method = None

    def fit_calibration(self, X_val, y_val):
        """Fit the calibrator using validation data."""
        if len(X_val) < 10:
            return  # Not enough data to calibrate

        # Get base predictions
        try:
            if self.task_type == 'classification':
                if hasattr(self.base_model, 'predict_proba'):
                    raw_preds = self.base_model.predict_proba(X_val)[:, 1]
                elif hasattr(self.base_model, 'decision_function'):
                    raw_preds = self.base_model.decision_function(X_val)
                    if raw_preds.ndim > 1:
                        raw_preds = raw_preds[:, 0]
                else:
                    raw_preds = self.base_model.predict(X_val)
            else:
                raw_preds = self.base_model.predict(X_val)
        except Exception as e:
            logger.warning(f"Failed to get base predictions for calibration: {e}")
            return

        method = self.calibration_method or 'isotonic'
        if method == 'auto':
            try:
                recommended = get_recommended_calibration_method(len(X_val))
            except Exception:
                recommended = 'isotonic'
            if recommended == 'platt':
                method = 'platt'
            else:
                method = 'isotonic'

        if self.task_type != 'classification':
            method = 'isotonic'

        try:
            if method == 'platt' and SKLEARN_AVAILABLE and LogisticRegression is not None:
                self.calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
                self.calibrator.fit(np.asarray(raw_preds).reshape(-1, 1), y_val)
                self._resolved_method = 'platt'
                self.is_calibrated = True
            else:
                if IsotonicRegression is None:
                    self.is_calibrated = False
                    return
                self.calibrator = IsotonicRegression(out_of_bounds='clip', increasing='auto')
                self.calibrator.fit(raw_preds, y_val)
                self._resolved_method = 'isotonic'
                self.is_calibrated = True
        except Exception as e:
            logger.warning(f"Calibration failed: {e}")
            self.is_calibrated = False

    def predict(self, X):
        """Predict with calibration."""
        try:
            if self.task_type == 'classification':
                if hasattr(self.base_model, 'predict_proba'):
                    raw_preds = self.base_model.predict_proba(X)[:, 1]
                elif hasattr(self.base_model, 'decision_function'):
                    raw_preds = self.base_model.decision_function(X)
                    if raw_preds.ndim > 1: raw_preds = raw_preds[:, 0]
                else:
                    raw_preds = self.base_model.predict(X)
            else:
                raw_preds = self.base_model.predict(X)
        except Exception as e:
            # Last resort fallback
            logger.error(f"Prediction failed in wrapper: {e}")
            return np.zeros(len(X))

        if self.is_calibrated and self.calibrator is not None:
            if getattr(self, '_resolved_method', None) == 'platt' and hasattr(self.calibrator, 'predict_proba'):
                try:
                    return self.calibrator.predict_proba(np.asarray(raw_preds).reshape(-1, 1))[:, 1]
                except Exception:
                    pass
            return self.calibrator.predict(raw_preds)

        # If not calibrated and it's classification decision function,
        # squeeze to 0-1 range roughly using sigmoid if needed,
        # or just return raw if expected.
        # But predict() usually returns the target variable.
        # For classification, this wrapper `predict` returns PROBABILITY.

        if self.task_type == 'classification' and hasattr(self.base_model, 'decision_function') and not hasattr(self.base_model, 'predict_proba'):
             # Simple sigmoid fallback if no calibration
             return 1 / (1 + np.exp(-raw_preds))

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
            logger.info(f"   Task Type: {self.config.task_type}")
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
            
            # HPO runs ONCE at burn-in only - finds good hyperparameters on initial data
            # Subsequent incremental windows use warm-start with these params (no re-tuning)
            if window.is_burn_in and self.config.enable_burnin_hpo:
                if verbose: logger.info("   Running burn-in HPO (one-time optimization)...")
                hpo_result = self._run_burnin_hpo(train_data, window, verbose)
                if hpo_result:
                    hpo_history[f"burnin_hpo"] = hpo_result

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
    def _run_burnin_hpo(self, train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> Optional[Dict[str, Any]]:
        """Run HPO once at burn-in to find good hyperparameters. NOT called during incremental windows."""
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

        if not self._feature_cols:
            logger.warning(
                "IncrementalLGBMTrainer: no specialist features found; "
                "using dummy feature to prevent crash."
            )
            train_data = train_data.copy()
            train_data['__dummy_const__'] = 0.0
            self._feature_cols = ['__dummy_const__']

        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        sw = None
        if '__weight__' in train_data.columns:
            sw = train_data['__weight__'].values
        
        if sw is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(X, y, sw, test_size=0.2, shuffle=False)
            train_ds = lgb.Dataset(X_train, label=y_train, weight=sw_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, weight=sw_val)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        
        params = self._get_default_params()
        params.update(self._best_params)
        n_est = params.pop('n_estimators', 500)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        model = lgb.train(
            params, train_ds, num_boost_round=n_est,
            valid_sets=[train_ds, val_ds],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        # Wrap and calibrate
        calibrated_model = CalibratedModel(model, self.config.task_type, self.config.calibration_method)
        calibrated_model.fit_calibration(X_val, y_val)

        if verbose:
            logger.info(f"   Initial LGBM: {model.num_trees()} trees, calibrated={calibrated_model.is_calibrated}")
        
        return calibrated_model
    
    def _train_incremental(self, new_data: pd.DataFrame, full_train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> CalibratedModel:
        if not self._feature_cols:
             self._feature_cols = self._get_feature_cols(full_train_data)
             if not self._feature_cols:
                 full_train_data = full_train_data.copy()
                 full_train_data['__dummy_const__'] = 0.0
                 self._feature_cols = ['__dummy_const__']

        X = full_train_data[self._feature_cols].values.astype(np.float32)
        y = full_train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        sw = None
        if '__weight__' in full_train_data.columns:
            sw = full_train_data['__weight__'].values
        
        if sw is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(X, y, sw, test_size=0.2, shuffle=False)
            train_ds = lgb.Dataset(X_train, label=y_train, weight=sw_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, weight=sw_val)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        
        params = self._get_default_params()
        params.update(self._best_params)
        early_stopping_rounds = params.pop('early_stopping_rounds', 20)
        
        # Access base model from wrapper
        prev_lgbm = self._current_model.base_model
        
        model = lgb.train(
            params, train_ds, num_boost_round=100,
            valid_sets=[train_ds, val_ds],
            init_model=prev_lgbm,
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        calibrated_model = CalibratedModel(model, self.config.task_type, self.config.calibration_method)
        calibrated_model.fit_calibration(X_val, y_val)
        
        return calibrated_model
    
    def _predict(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        if not self._feature_cols:
             return pd.DataFrame({'prediction': 0}, index=pred_data.index)

        if '__dummy_const__' in self._feature_cols and '__dummy_const__' not in pred_data:
            pred_data = pred_data.copy()
            pred_data['__dummy_const__'] = 0.0

        X = pred_data[self._feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        
        preds = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            return pd.DataFrame({'prediction': (preds > 0.5).astype(int), 'probability': preds}, index=pred_data.index)
        else:
            return pd.DataFrame({'prediction': preds}, index=pred_data.index)

    def _run_burnin_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None

        if not self._feature_cols:
             self._feature_cols = self._get_feature_cols(train_data)
             if not self._feature_cols:
                 train_data = train_data.copy()
                 train_data['__dummy_const__'] = 0.0
                 self._feature_cols = ['__dummy_const__']

        if '__dummy_const__' in self._feature_cols and '__dummy_const__' not in train_data:
            train_data = train_data.copy()
            train_data['__dummy_const__'] = 0.0

        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
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
                model = lgb.train(
                    params, dtrain, num_boost_round=200, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]
                )
                preds = model.predict(X_val)
                if self.config.task_type == 'regression':
                    return -((y_val - preds)**2).mean()
                else:
                    from sklearn.metrics import log_loss
                    return -log_loss(y_val, preds)
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round, show_progress_bar=False)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}


"""Incremental bagged LightGBM classifier utilities.

This section defines the bagged meta-model used by the Analyst incremental
trainer. Historically the wrapper was named ``BaggedLGBMRegressor`` even
though it operates in a classification setting (probabilities for class 1).

To make the intent clearer we now expose it as ``BaggedLGBMClassifier`` while
keeping a thin alias ``BaggedLGBMRegressor`` for backward compatibility so
that any persisted artifacts or older code paths continue to function.
"""

# ============================================================================
# Incremental Bagged LGBM Trainer (Bag-Lower)
# ============================================================================


class BaggedLGBMClassifier:
    """Bagged LightGBM classifier with Diversity Defense support.

    All underlying models are expected to behave like ``LGBMClassifier``
    instances and return class-1 probabilities via ``predict_proba``. The
    wrapper exposes several aggregation views over the per-bag probabilities
    (mean/std/lower and, when Diversity Defense is enabled, MAD-based
    consensus signals).
    """

    def __init__(
        self,
        models: List[Any],
        n_features: int,
        specialist_types: Optional[List[Any]] = None,
        use_diversity_defense: bool = False,
        diversity_defense_config: Optional[Any] = None,
    ):
        self.models = models
        self.n_features = n_features
        self.specialist_types = specialist_types or []
        self.use_diversity_defense = use_diversity_defense
        self.diversity_defense_config = diversity_defense_config

        # Initialize aggregator for diversity defense
        self._aggregator = None
        if use_diversity_defense:
            try:
                from src.utils.ml_common.optimization.diversity_defense_objectives import (
                    DiversityDefenseAggregator,
                    DiversityDefenseConfig,
                )
                if diversity_defense_config is not None:
                    self._aggregator = DiversityDefenseAggregator(diversity_defense_config)
                else:
                    self._aggregator = DiversityDefenseAggregator()
            except ImportError:
                logger.warning("Diversity Defense not available, falling back to standard aggregation")
                self.use_diversity_defense = False

    def predict_components(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (mean, std, lower) or classifier metrics across bags.

        For Classification (Standard Bagging):
        Output = (N_up * AvgConf_up - N_down * AvgConf_down) / Total_N
        This replaces the simple mean-std calculation.
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            # Best-effort fallback: clip to common number of features
            n_common = min(self.n_features, X.shape[1])
            X = X[:, :n_common]
        if not self.models:
            n_samples = X.shape[0]
            zeros = np.zeros(n_samples, dtype=float)
            return zeros, zeros, zeros

        classification_mode = False
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                classification_mode = True
                break

        bag_preds = []
        for model in self.models:
            try:
                if classification_mode and hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[:, 1]
                else:
                    raw = model.predict(X)
                    raw = np.asarray(raw, dtype=float).reshape(-1)
                    if raw.size and (np.nanmin(raw) < 0.0 or np.nanmax(raw) > 1.0):
                        prob = expit(raw)
                    else:
                        prob = raw

                bag_preds.append(prob)
            except Exception:
                continue

        if not bag_preds:
            n_samples = X.shape[0]
            zeros = np.zeros(n_samples, dtype=float)
            return zeros, zeros, zeros

        preds_mat = np.vstack(bag_preds).T  # (n_samples, n_bags)

        mean = preds_mat.mean(axis=1)
        std = preds_mat.std(axis=1)

        if not classification_mode:
            lower = mean - std
            return mean, std, lower

        up_mask = preds_mat > 0.5
        down_mask = ~up_mask

        n_up = up_mask.sum(axis=1)
        n_down = down_mask.sum(axis=1)
        n_total = preds_mat.shape[1]

        sum_p_up = np.where(up_mask, preds_mat, 0.0).sum(axis=1)
        avg_conf_up = np.divide(sum_p_up, n_up, out=np.zeros_like(sum_p_up), where=n_up != 0)

        sum_p_down = np.where(down_mask, 1.0 - preds_mat, 0.0).sum(axis=1)
        avg_conf_down = np.divide(sum_p_down, n_down, out=np.zeros_like(sum_p_down), where=n_down != 0)

        score = (n_up * avg_conf_up - n_down * avg_conf_down) / n_total
        return mean, std, score

    def predict_diversity_defense(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Return diversity defense predictions with MAD-based consensus.

        Returns:
            Dict with keys:
                - 'mean': Mean prediction
                - 'std': Standard deviation
                - 'lower': Mean - std
                - 'median': Median prediction (more robust)
                - 'mad': Median Absolute Deviation (disagreement measure)
                - 'consensus': MAD-weighted consensus signal
                - 'raw_preds': Raw predictions from all specialists
        """
        if X.ndim != 2 or X.shape[1] != self.n_features:
            n_common = min(self.n_features, X.shape[1])
            X = X[:, :n_common]

        if not self.models:
            n_samples = X.shape[0]
            zeros = np.zeros(n_samples, dtype=float)
            return {
                'mean': zeros, 'std': zeros, 'lower': zeros,
                'median': zeros, 'mad': zeros, 'consensus': zeros,
                'raw_preds': np.zeros((n_samples, 1))
            }

        classification_mode = False
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                classification_mode = True
                break

        bag_preds = []
        for model in self.models:
            try:
                if classification_mode and hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[:, 1]
                elif classification_mode:
                    try:
                        prob = model.predict_proba(X)[:, 1]
                    except Exception:
                        raw = model.predict(X, raw_score=True)
                        prob = expit(raw)
                else:
                    raw = model.predict(X)
                    raw = np.asarray(raw, dtype=float).reshape(-1)
                    if raw.size and (np.nanmin(raw) < 0.0 or np.nanmax(raw) > 1.0):
                        prob = expit(raw)
                    else:
                        prob = raw

                bag_preds.append(prob)
            except Exception:
                continue

        if not bag_preds:
            n_samples = X.shape[0]
            zeros = np.zeros(n_samples, dtype=float)
            return {
                'mean': zeros, 'std': zeros, 'lower': zeros,
                'median': zeros, 'mad': zeros, 'consensus': zeros,
                'raw_preds': np.zeros((n_samples, 1))
            }

        preds_mat = np.vstack(bag_preds).T  # (n_samples, n_bags)

        mean = preds_mat.mean(axis=1)
        std = preds_mat.std(axis=1)
        lower = mean - std

        median = np.median(preds_mat, axis=1)
        mad_raw = np.median(np.abs(preds_mat - median[:, np.newaxis]), axis=1)

        if not classification_mode:
            return {
                'mean': mean,
                'std': std,
                'lower': lower,
                'median': median,
                'mad': mad_raw,
                'consensus': lower,
                'raw_preds': preds_mat,
            }

        preds_signal = 2.0 * preds_mat - 1.0
        median_signal = np.median(preds_signal, axis=1)
        mad = np.median(np.abs(preds_signal - median_signal[:, np.newaxis]), axis=1)

        consensus = None
        if self._aggregator is not None:
            try:
                preds_matrix = np.vstack(bag_preds)
                agg_result = self._aggregator.aggregate(preds_matrix)
                consensus = agg_result.get('final_signal')
            except Exception as e:
                logger.warning(
                    "Diversity Defense aggregation failed, falling back to simple MAD consensus: %s",
                    e,
                )

        if consensus is None:
            mad_floor = 0.25
            if self.diversity_defense_config is not None:
                mad_floor = getattr(self.diversity_defense_config, 'mad_floor', 0.25)
            mad_eff = np.maximum(mad, mad_floor)
            signal_strength = np.abs(median_signal)
            raw_score = signal_strength - 1.1 * mad_eff
            consensus = np.clip(raw_score, 0.0, 1.0)

        return {
            'mean': mean,
            'std': std,
            'lower': lower,
            'median': median,
            'mad': mad,
            'consensus': consensus,
            'raw_preds': preds_mat,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return consensus signal or bagging score.

        This mirrors the scikit-learn ``predict`` API, but instead of hard
        class labels we return a continuous score in ``[0, 1]`` that can be
        interpreted as a probability-like meta-signal (class-1 propensity).
        """
        if self.use_diversity_defense:
            result = self.predict_diversity_defense(X)
            return result['consensus']
        _, _, score = self.predict_components(X)
        return score


# Backward-compatible alias (old name used in some artifacts / configs)
BaggedLGBMRegressor = BaggedLGBMClassifier


class IncrementalLGBMBaggedTrainer(BaseIncrementalTrainer):
    """
    Incremental bagged LightGBM trainer with Diversity Defense support.

    Produces bag-lower predictions with calibration. When diversity defense
    is enabled, trains specialist models with different objectives:
    - 3x Sharpe models (vol-normalized, risk-adjusted)
    - 3x Tanh models (directional accuracy)
    - 4x Huber models (2 standard + 2 asymmetric)

    Feature diversity is controlled via colsample_bytree (NOT external loops).
    The optimal value is determined via DiversitySweep at the start of training.
    """

    def __init__(self, model_id: str, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
        self._n_bags = int((model_config or {}).get('n_bags', 10))
        self._sample_fraction = float((model_config or {}).get('bagging_sample_fraction', 0.7))
        self._bagged_boosters = []  # Store boosters for warm start

        scheme = (model_config or {}).get('specialist_scheme', None)
        self._specialist_scheme = str(scheme).lower().strip() if scheme is not None else None

        # Diversity Defense configuration
        self._use_diversity_defense = bool((model_config or {}).get('use_diversity_defense', True))
        self._run_diversity_sweep = bool((model_config or {}).get('run_diversity_sweep', True))
        self._optimal_colsample_bytree = float((model_config or {}).get('colsample_bytree', 0.5))
        self._diversity_sweep_done = False
        self._dd_config = None
        self._dd_objectives = None

        if self._use_diversity_defense:
            try:
                from src.utils.ml_common.optimization.diversity_defense_objectives import (
                    DiversityDefenseConfig,
                    DiversityDefenseObjectives,
                )
                dd_config_dict = (model_config or {}).get('diversity_defense_config', {})
                base_dd = DiversityDefenseConfig()
                self._dd_config = DiversityDefenseConfig(
                    sharpe_count=dd_config_dict.get('sharpe_count', base_dd.sharpe_count),
                    tanh_count=dd_config_dict.get('tanh_count', base_dd.tanh_count),
                    huber_standard_count=dd_config_dict.get('huber_standard_count', base_dd.huber_standard_count),
                    huber_asymmetric_count=dd_config_dict.get('huber_asymmetric_count', base_dd.huber_asymmetric_count),
                    sharpe_lambdas=dd_config_dict.get('sharpe_lambdas', base_dd.sharpe_lambdas),
                    huber_delta=dd_config_dict.get('huber_delta', base_dd.huber_delta),
                    asymmetric_penalty=dd_config_dict.get('asymmetric_penalty', base_dd.asymmetric_penalty),
                    lr_sharpe=dd_config_dict.get('lr_sharpe', base_dd.lr_sharpe),
                    lr_tanh=dd_config_dict.get('lr_tanh', base_dd.lr_tanh),
                    lr_huber=dd_config_dict.get('lr_huber', base_dd.lr_huber),
                    sample_fraction=self._sample_fraction,
                    colsample_bytree=self._optimal_colsample_bytree,
                    z_score_window=dd_config_dict.get('z_score_window', base_dd.z_score_window),
                    mad_floor=dd_config_dict.get('mad_floor', base_dd.mad_floor),
                    noise_threshold=dd_config_dict.get('noise_threshold', base_dd.noise_threshold),
                    cap_threshold=dd_config_dict.get('cap_threshold', base_dd.cap_threshold),
                    focal_gamma_standard=dd_config_dict.get('focal_gamma_standard', base_dd.focal_gamma_standard),
                    focal_alpha_standard=dd_config_dict.get('focal_alpha_standard', base_dd.focal_alpha_standard),
                    focal_gamma_asymmetric=dd_config_dict.get('focal_gamma_asymmetric', base_dd.focal_gamma_asymmetric),
                    focal_alpha_asymmetric=dd_config_dict.get('focal_alpha_asymmetric', base_dd.focal_alpha_asymmetric),
                )
                self._dd_objectives = DiversityDefenseObjectives(self._dd_config)
                logger.info(f"Diversity Defense enabled for {model_id}")
            except ImportError as e:
                logger.warning(f"Diversity Defense not available: {e}, using standard bagging")
                self._use_diversity_defense = False

        if self._specialist_scheme in ('abc', 'abc_v1'):
            self._use_diversity_defense = False

    def _run_colsample_optimization(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> float:
        """
        Run DiversitySweep to find optimal colsample_bytree.

        Args:
            X: Feature matrix (numpy array)
            y: Target array
            verbose: Print progress

        Returns:
            Optimal colsample_bytree value
        """
        if self._diversity_sweep_done:
            return self._optimal_colsample_bytree

        try:
            from src.utils.ml_common.optimization.diversity_defense_objectives import DiversitySweep

            # Need minimum samples for meaningful sweep
            if len(X) < 500:
                logger.info(f"Insufficient data for DiversitySweep ({len(X)} samples), using default colsample_bytree=0.5")
                self._diversity_sweep_done = True
                return 0.5

            if verbose:
                logger.info("[DIVERSITY_SWEEP] Finding optimal colsample_bytree...")

            # Use a sample for faster sweep if dataset is large
            sweep_sample_size = min(5000, len(X))
            if len(X) > sweep_sample_size:
                rng = np.random.RandomState(42)
                sweep_indices = rng.choice(len(X), size=sweep_sample_size, replace=False)
                sweep_indices.sort()
                X_sweep = pd.DataFrame(X[sweep_indices])
                y_sweep = y[sweep_indices]
            else:
                X_sweep = pd.DataFrame(X)
                y_sweep = y

            sweep = DiversitySweep(
                X=X_sweep,
                y=y_sweep,
                colsample_settings=[0.7, 0.6, 0.5, 0.4, 0.3],
                n_splits=2,  # Fast sweep
                n_models=5,  # Mini-ensemble for speed
            )

            df_sweep = sweep.run(verbose=verbose)
            optimal = sweep.get_optimal_fraction()

            if verbose:
                logger.info(f"[DIVERSITY_SWEEP] ✅ Optimal colsample_bytree: {optimal:.2f}")
                try:
                    if df_sweep is not None and not df_sweep.empty:
                        best_idx = df_sweep['ESR_Score'].idxmax()
                        best_row = df_sweep.loc[best_idx]
                        logger.info(
                            "[DIVERSITY_SWEEP] Summary -> fraction=%.2f, Sharpe=%.4f, AvgCorr=%.4f, ESR=%.4f",
                            best_row.get('Fraction', float('nan')),
                            best_row.get('Sharpe', float('nan')),
                            best_row.get('Avg_Corr', float('nan')),
                            best_row.get('ESR_Score', float('nan')),
                        )
                except Exception as e:
                    logger.warning("[DIVERSITY_SWEEP] Failed to log summary statistics: %s", e)

            self._diversity_sweep_done = True
            self._optimal_colsample_bytree = optimal  # Cache the result
            return optimal

        except Exception as e:
            logger.warning(f"[DIVERSITY_SWEEP] Failed: {e}, using default colsample_bytree=0.5")
            self._diversity_sweep_done = True
            self._optimal_colsample_bytree = 0.5  # Cache the default
            return 0.5

    def _get_default_params(self) -> Dict[str, Any]:
        params = {
            'objective': 'binary', # Default to binary for Classifier
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        if self.model_config and 'params' in self.model_config:
            params.update(self.model_config['params'])
        return params

    def _train_bagged_model(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray], verbose: bool) -> BaggedLGBMClassifier:
        n_samples, n_features = X.shape
        if n_samples == 0 or n_features == 0:
            return BaggedLGBMClassifier([], n_features)

        base_params = self._get_default_params()
        base_params.update(self._best_params)

        # Drop early-stopping parameters that are only valid for the low-level
        # lgb.train API. Passing early_stopping_rounds into the scikit-learn
        # LGBMRegressor constructor without an eval_set causes LightGBM to
        # raise "For early stopping, at least one dataset and eval metric is
        # required for evaluation". Bag-level models are trained on simple
        # bootstrap subsets, so we disable built-in early stopping here and
        # rely on global incremental training/HPO instead.
        base_params.pop("early_stopping_rounds", None)

        # ============================================================
        # STEP 0: Run DiversitySweep to optimize colsample_bytree
        # This is run ONCE at the start of training (first call only)
        # ============================================================
        if self._run_diversity_sweep and not self._diversity_sweep_done:
            self._optimal_colsample_bytree = self._run_colsample_optimization(X, y, verbose=verbose)

            # Update DD config with optimized value
            if self._use_diversity_defense and self._dd_config is not None:
                # Create new config with optimized colsample_bytree
                from src.utils.ml_common.optimization.diversity_defense_objectives import DiversityDefenseConfig
                self._dd_config = DiversityDefenseConfig(
                    sharpe_count=self._dd_config.sharpe_count,
                    tanh_count=self._dd_config.tanh_count,
                    huber_standard_count=self._dd_config.huber_standard_count,
                    huber_asymmetric_count=self._dd_config.huber_asymmetric_count,
                    sharpe_lambdas=self._dd_config.sharpe_lambdas,
                    huber_delta=self._dd_config.huber_delta,
                    asymmetric_penalty=self._dd_config.asymmetric_penalty,
                    lr_sharpe=self._dd_config.lr_sharpe,
                    lr_tanh=self._dd_config.lr_tanh,
                    lr_huber=self._dd_config.lr_huber,
                    sample_fraction=self._dd_config.sample_fraction,
                    colsample_bytree=self._optimal_colsample_bytree,
                    z_score_window=self._dd_config.z_score_window,
                    mad_floor=self._dd_config.mad_floor,
                    noise_threshold=self._dd_config.noise_threshold,
                    cap_threshold=self._dd_config.cap_threshold,
                    # Pass new params
                    focal_gamma_standard=self._dd_config.focal_gamma_standard,
                    focal_alpha_standard=self._dd_config.focal_alpha_standard,
                    focal_gamma_asymmetric=self._dd_config.focal_gamma_asymmetric,
                    focal_alpha_asymmetric=self._dd_config.focal_alpha_asymmetric,
                )
                logger.info(f"[DIVERSITY_DEFENSE] Updated colsample_bytree to {self._optimal_colsample_bytree:.2f}")

        # Feature diversity via colsample_bytree (NOT external loop)
        # Optimal value from DiversitySweep, or default 0.5
        if self._use_diversity_defense and self._dd_config is not None:
            base_params['colsample_bytree'] = self._dd_config.colsample_bytree
        else:
            base_params['colsample_bytree'] = self._optimal_colsample_bytree

        sample_frac = min(max(self._sample_fraction, 0.1), 1.0)

        if verbose:
            try:
                logger.info(
                    "[BAGGED_LGBM] Training %s bags with colsample_bytree=%.2f, sample_fraction=%.2f, use_diversity_defense=%s",
                    self._n_bags,
                    float(base_params.get('colsample_bytree', 0.0)),
                    float(sample_frac),
                    self._use_diversity_defense,
                )
            except Exception:
                # Logging must never break training
                pass

        rng = np.random.RandomState(42)
        models: List[Any] = []
        specialist_types: List[Any] = []
        new_boosters = []

        # Compute volatility for Sharpe objectives (needed for Diversity Defense)
        if self._use_diversity_defense and self._dd_config is not None and self._dd_objectives is not None:
            # We assume y is continuous returns.
            # Use diversity defense helper to create ensemble
            from src.utils.ml_common.optimization.diversity_defense_objectives import (
                create_diversity_defense_ensemble,
                get_sharpe_weights
            )

            # --- CRITICAL: Calculate Sharpe Weights BEFORE Bagging ---
            # Weights depend on time-series rolling volatility.
            # Bagging shuffles data, destroying this structure.
            # We must calculate weights on the full time-ordered `y` vector here.

            # Convert y to series for rolling calc if needed
            y_series = pd.Series(y)

            # Calculate weights for 12h and 4d windows (assuming 15m data for now as per user prompt context)
            # User specified: 12h_Volatility, 4d_Volatility
            # Windows: 12h = 48 periods, 4d = 384 periods (at 15m)
            # The get_sharpe_weights helper handles window conversion if we pass hours.

            # Note: get_sharpe_weights returns numpy array aligned with input
            w_sharpe_12h = get_sharpe_weights(y_series, window_hours=12)
            w_sharpe_4d = get_sharpe_weights(y_series, window_hours=96) # 4 days = 96 hours

            # Downside Deviation (User requested "Standard Rolling Deviation" as proxy for efficiency)
            # We'll use a medium window, e.g., 24h
            w_sharpe_dd = get_sharpe_weights(y_series, window_hours=24)

            specialist_configs = self._dd_config.get_specialist_configs()
            n_bags = len(specialist_configs)

            # Check if we have previous boosters for warm start
            has_warm_start = len(self._bagged_boosters) == n_bags

            for spec_idx, spec_config in enumerate(specialist_configs):
                # Prepare params
                params = dict(base_params)
                params['random_state'] = int(params.get('random_state', 42)) + spec_idx

                # Learning rate
                lr = getattr(self._dd_config, spec_config.learning_rate_key, 0.03)
                params['learning_rate'] = lr

                # Row sampling (Bootstrap)
                n_rows_sub = max(10, int(round(sample_frac * n_samples)))
                row_idx = np.sort(rng.choice(n_samples, size=n_rows_sub, replace=False))

                # Subset Data
                # Note: X is numpy array here usually
                X_bag = X[row_idx]

                # Prepare Targets & Weights based on Specialist Type
                y_bag_raw = y[row_idx] # Continuous returns
                y_bag_target = None
                sample_weight_bag = None
                fobj = None

                if spec_config.specialist_type == self._dd_objectives.config.SpecialistType.SHARPE:
                    # Target: Binary (>0)
                    y_bag_target = (y_bag_raw > 0).astype(int)

                    # Weights: Select based on model index (0, 1, 2)
                    # We need to map spec_idx to specific Sharpe role.
                    # Assuming order: 3 Sharpe, then Tanh...
                    sharpe_idx = sum(1 for c in specialist_configs[:spec_idx] if c.specialist_type == self._dd_objectives.config.SpecialistType.SHARPE)

                    if sharpe_idx == 0:
                        weights_full = w_sharpe_12h
                    elif sharpe_idx == 1:
                        weights_full = w_sharpe_4d
                    else:
                        weights_full = w_sharpe_dd

                    # Slice weights for this bag
                    sample_weight_bag = weights_full[row_idx]

                    # Use standard binary objective
                    params['objective'] = 'binary'
                    params['metric'] = 'binary_logloss'

                elif spec_config.specialist_type == self._dd_objectives.config.SpecialistType.TANH:
                    # Target: Binary (> Threshold)
                    # Thresholds: 0.3%, 0.6%, 0.9%
                    tanh_idx = sum(1 for c in specialist_configs[:spec_idx] if c.specialist_type == self._dd_objectives.config.SpecialistType.TANH)
                    threshold = [0.003, 0.006, 0.009][tanh_idx % 3]
                    y_bag_target = (y_bag_raw > threshold).astype(int)

                    # Use standard binary objective
                    params['objective'] = 'binary'
                    params['metric'] = 'binary_logloss'

                elif spec_config.specialist_type in (self._dd_objectives.config.SpecialistType.HUBER, self._dd_objectives.config.SpecialistType.ASYMMETRIC_HUBER):
                    # Target: Binary (>0)
                    y_bag_target = (y_bag_raw > 0).astype(int)

                    # Custom Focal Loss Objective
                    fobj = self._dd_objectives.get_objective_for_specialist(spec_config)
                    # Remove standard objective param to avoid conflict
                    if 'objective' in params: del params['objective']
                    if 'metric' in params: del params['metric']

                # Fallback if no target set
                if y_bag_target is None:
                    y_bag_target = (y_bag_raw > 0).astype(int)

                try:
                    # Initialize Model
                    if fobj is not None:
                        model = lgb.LGBMClassifier(objective=fobj, **params)
                    else:
                        model = lgb.LGBMClassifier(**params)

                    # Warm start
                    init_model = self._bagged_boosters[spec_idx] if has_warm_start else None

                    # Fit
                    if sample_weight_bag is not None:
                        model.fit(X_bag, y_bag_target, sample_weight=sample_weight_bag, init_model=init_model)
                    else:
                        model.fit(X_bag, y_bag_target, init_model=init_model)

                    models.append(model)
                    specialist_types.append(spec_config.specialist_type)

                    # Store booster
                    if hasattr(model, 'booster_'):
                        new_boosters.append(model.booster_)

                except Exception as e:
                    logger.warning(f"Specialist {spec_idx} ({spec_config.role_description}) failed: {e}")
                    continue

        else:
            # ============================================================
            # STANDARD MODE: Train identical models with different seeds
            # ============================================================
            n_bags = max(1, int(self._n_bags))

            # Check if we have previous boosters for warm start
            has_warm_start = len(self._bagged_boosters) == n_bags

            # Use Focal Loss for Standard Bagging
            # alpha=0.25, gamma=2.0 (Default)
            from src.utils.ml_common.optimization.diversity_defense_objectives import get_focal_loss
            focal_loss_obj = get_focal_loss(alpha=0.25, gamma=2.0)

            # Target is Binary (>0)
            y_binary = (y > 0).astype(int)

            for bag_idx in range(n_bags):
                params = dict(base_params)
                params['random_state'] = int(params.get('random_state', 42)) + bag_idx

                # Explicitly set AUC metric as requested
                # Note: 'metric' is used for validation/early stopping.
                # Since we don't have eval_set here (bootstrap), it mostly affects log.
                # But we set it for correctness.
                params['metric'] = 'auc'

                # Remove standard objective since we're using custom
                if 'objective' in params: del params['objective']

                # Row sampling
                n_rows_sub = max(10, int(round(sample_frac * n_samples)))
                n_rows_sub = min(n_rows_sub, n_samples)
                row_idx = np.sort(rng.choice(n_samples, size=n_rows_sub, replace=False))

                # Subset
                X_bag = X[row_idx]
                y_bag = y_binary[row_idx]
                if sample_weight is not None:
                    sw_bag = sample_weight[row_idx]
                else:
                    sw_bag = None

                try:
                    model = lgb.LGBMClassifier(objective=focal_loss_obj, **params)

                    # Use warm start if available
                    init_model = self._bagged_boosters[bag_idx] if has_warm_start else None

                    if sw_bag is not None:
                        model.fit(X_bag, y_bag, sample_weight=sw_bag, init_model=init_model)
                    else:
                        model.fit(X_bag, y_bag, init_model=init_model)

                    models.append(model)

                    # Store booster for next round
                    if hasattr(model, 'booster_'):
                        new_boosters.append(model.booster_)

                except Exception as e:
                    logger.warning(f"Bag {bag_idx} failed during training: {e}")
                    continue

        # Update stored boosters if we successfully trained
        if len(new_boosters) == len(models):
            self._bagged_boosters = new_boosters

        return BaggedLGBMClassifier(
            models=models,
            n_features=n_features,
            specialist_types=specialist_types if self._use_diversity_defense else None,
            use_diversity_defense=self._use_diversity_defense,
            diversity_defense_config=self._dd_config,
        )
# Incremental NGBoost Trainer
# ============================================================================

class IncrementalNGBoostTrainer(BaseIncrementalTrainer):
    """Incremental NGBoost trainer with calibration. Now defaulting to Classifier."""
    
    def __init__(self, model_id: str, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._feature_cols: List[str] = []
    
    def _get_default_params(self):
        params = {'n_estimators': 300, 'learning_rate': 0.01, 'minibatch_frac': 0.5, 'verbose': False}
        if self.model_config and 'params' in self.model_config:
            cfg_params = dict(self.model_config['params'])
            base_cfg = cfg_params.pop('base_learner_params', None) or {}
            if isinstance(base_cfg, dict):
                if 'max_depth' in base_cfg and 'base_learner_max_depth' not in cfg_params:
                    cfg_params['base_learner_max_depth'] = base_cfg['max_depth']
                if 'min_samples_leaf' in base_cfg and 'base_learner_min_samples_leaf' not in cfg_params:
                    cfg_params['base_learner_min_samples_leaf'] = base_cfg['min_samples_leaf']
            params.update(cfg_params)
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
        y = train_data['__target__'].values.astype(int if self.config.task_type == 'classification' else np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        sw = None
        if '__weight__' in train_data.columns:
            sw = train_data['__weight__'].values

        if sw is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(X, y, sw, test_size=0.2, shuffle=False)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            sw_train = None
        
        params = self._get_default_params()
        params.update(self._best_params)
        base_learner = self._create_base_learner(params)
        ng_params = {
            k: v
            for k, v in params.items()
            if k not in ('base_learner_max_depth', 'base_learner_min_samples_leaf', 'base_learner_params')
        }
        
        early_stopping_rounds = params.pop('early_stopping_rounds', 35)

        if self.config.task_type == 'classification':
            model = NGBClassifier(Dist=Bernoulli, Base=base_learner, **ng_params)
        else:
            model = NGBRegressor(Dist=Normal, Base=base_learner, **ng_params)

        model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, sample_weight=sw_train, early_stopping_rounds=early_stopping_rounds)
        
        calibrated_model = CalibratedModel(model, self.config.task_type, self.config.calibration_method)
        calibrated_model.fit_calibration(X_val, y_val)
        
        if verbose:
            logger.info(f"   Initial NGBoost trained, calibrated={calibrated_model.is_calibrated}")
        
        return calibrated_model

    def _train_incremental(self, new_data, full_train_data, window, verbose) -> CalibratedModel:
        max_history = 10000
        if len(full_train_data) > max_history:
            if verbose: logger.info(f"   NGBoost: Truncating history to last {max_history} samples for speed")
            train_data_window = full_train_data.iloc[-max_history:].copy()
        else:
            train_data_window = full_train_data

        return self._train_initial(train_data_window, window, verbose)

    def _predict(self, pred_data) -> pd.DataFrame:
        if self._feature_cols:
            X = pred_data[self._feature_cols].values.astype(np.float64)
        else:
            X = np.zeros((len(pred_data), 1), dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        preds = self._current_model.predict(X)
        
        if self.config.task_type == 'classification':
            probs = self._current_model.predict_proba(X)
            return pd.DataFrame({'prediction': (probs[:, 1] > 0.5).astype(int), 'probability': probs[:, 1]}, index=pred_data.index)
        else:
            try:
                dist = self._current_model.base_model.pred_dist(X)
                std = dist.params.get('scale', np.zeros_like(preds))
            except:
                std = np.zeros_like(preds)
            return pd.DataFrame({'prediction': preds, 'std': std}, index=pred_data.index)

    def _run_burnin_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None

        if not self._feature_cols:
            self._feature_cols = self._get_feature_cols(train_data)

        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values.astype(int if self.config.task_type == 'classification' else np.float64)
        X = np.nan_to_num(X, nan=0.0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
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
                ng_params = {
                    k: v
                    for k, v in params.items()
                    if k not in ('base_learner_max_depth', 'base_learner_min_samples_leaf', 'base_learner_params')
                }
                if self.config.task_type == 'classification':
                    model = NGBClassifier(Dist=Bernoulli, Base=base_learner, **ng_params)
                else:
                    model = NGBRegressor(Dist=Normal, Base=base_learner, **ng_params)
                model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=35)

                if self.config.task_type == 'regression':
                    preds = model.predict(X_val)
                    return -((y_val - preds)**2).mean()
                else:
                    from sklearn.metrics import log_loss
                    probs = model.predict_proba(X_val)
                    return -log_loss(y_val, probs)
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round, show_progress_bar=False)
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
        self._feature_cols: List[str] = []

    def _get_default_params(self):
        params = {'n_neighbors': 15, 'weights': 'distance', 'n_jobs': -1}
        if self.model_config: params.update(self.model_config.get('params', {}))
        return params

    def _train_initial(self, train_data, window, verbose) -> CalibratedModel:
        self._feature_cols = self._get_feature_cols(train_data)
        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        
        params = self._get_default_params()
        n_samples = len(y_train)
        max_k = max(5, int(np.sqrt(n_samples)))
        if 'n_neighbors' in params: params['n_neighbors'] = min(params['n_neighbors'], max_k)
        
        if self.config.task_type == 'classification':
            y_train = y_train.astype(int)
            y_val = y_val.astype(int)
            model = KNeighborsClassifier(**params)
        else:
            model = KNeighborsRegressor(**params)

        model.fit(X_train_scaled, y_train)
        
        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))
            def predict_proba(self, X):
                return self.model.predict_proba(self.scaler.transform(X))

        calibrated_model = CalibratedModel(ScaledModel(model, self._scaler), self.config.task_type, self.config.calibration_method)
        calibrated_model.fit_calibration(X_val, y_val)
        
        return calibrated_model

    def _train_incremental(self, new_data, full_train_data, window, verbose):
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

    def _run_burnin_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None
        if not self._feature_cols:
            self._feature_cols = self._get_feature_cols(train_data)

        X = train_data[self._feature_cols].values.astype(np.float32)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)

        if self.config.task_type == 'classification':
            y = y.astype(int)

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
                    model = KNeighborsClassifier(n_jobs=1, **params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy', n_jobs=1)
                else:
                    model = KNeighborsRegressor(n_jobs=1, **params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
                return scores.mean()
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round, show_progress_bar=False)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}


# ============================================================================
# Incremental Ridge Trainer (Replacing BayesianRidge)
# ============================================================================

class IncrementalBayesianRidgeTrainer(BaseIncrementalTrainer):
    """
    Renamed to RidgeClassifier under the hood, but keeping class name for compatibility.
    Uses RidgeClassifier for classification tasks.
    """
    def __init__(self, model_id, config=None, model_config=None):
        super().__init__(model_id, config, model_config)
        self._scaler = None
        self._feature_cols: List[str] = []

    def _get_default_params(self):
        params = {'alpha': 1.0, 'tol': 1e-3, 'class_weight': 'balanced'}
        if self.model_config:
            params.update(self.model_config.get('params', {}))
        for k in ['n_iter', 'lambda_1', 'lambda_2', 'alpha_1', 'alpha_2']:
            params.pop(k, None)
        return params

    def _train_initial(self, train_data: pd.DataFrame, window: IncrementalTrainingWindow, verbose: bool) -> CalibratedModel:
        self._feature_cols = self._get_feature_cols(train_data)
        if not self._feature_cols:
            logger.warning("IncrementalBayesianRidgeTrainer: no specialist features found; using constant feature.")
            train_data = train_data.copy()
            train_data['__dummy_const__'] = 0.0
            self._feature_cols = ['__dummy_const__']

        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        sw = None
        if '__weight__' in train_data.columns:
            sw = train_data['__weight__'].values

        if sw is not None:
            X_tr, X_va, y_tr, y_va, sw_tr, sw_va = train_test_split(X, y, sw, test_size=0.2, shuffle=False)
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, shuffle=False)
            sw_tr = None
        
        self._scaler = StandardScaler()
        X_tr_scaled = self._scaler.fit_transform(X_tr)
        
        params = self._get_default_params()
        params.update(self._best_params)
        
        if self.config.task_type == 'classification':
            y_tr = y_tr.astype(int)
            y_va = y_va.astype(int)
            model = RidgeClassifier(**params)
        else:
            logger.warning("Regression task requested for RidgeClassifier trainer. Falling back to BayesianRidge.")
            model = BayesianRidge()

        if sw_tr is not None:
            model.fit(X_tr_scaled, y_tr, sample_weight=sw_tr)
        else:
            model.fit(X_tr_scaled, y_tr)
        
        class ScaledModel:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))
            def decision_function(self, X):
                if hasattr(self.model, 'decision_function'):
                    return self.model.decision_function(self.scaler.transform(X))
                return self.model.predict(self.scaler.transform(X))

        calibrated_model = CalibratedModel(ScaledModel(model, self._scaler), self.config.task_type, self.config.calibration_method)
        calibrated_model.fit_calibration(X_va, y_va)
        
        return calibrated_model

    def _train_incremental(self, new_data, full_train_data, window, verbose):
        return self._train_initial(full_train_data, window, verbose)

    def _predict(self, pred_data) -> pd.DataFrame:
        if not self._feature_cols:
             return pd.DataFrame({'prediction': 0, 'probability': 0}, index=pred_data.index)

        if '__dummy_const__' in self._feature_cols and '__dummy_const__' not in pred_data:
            pred_data = pred_data.copy()
            pred_data['__dummy_const__'] = 0.0

        X = pred_data[self._feature_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)
        
        if self.config.task_type == 'classification':
            probs = self._current_model.predict_proba(X)
            preds = (probs[:, 1] > 0.5).astype(int)
            return pd.DataFrame({'prediction': preds, 'probability': probs[:, 1]}, index=pred_data.index)
        else:
            preds = self._current_model.predict(X)
            return pd.DataFrame({'prediction': preds}, index=pred_data.index)

    def _run_burnin_hpo(self, train_data, window, verbose):
        if not OPTUNA_AVAILABLE: return None

        if not self._feature_cols:
            self._feature_cols = self._get_feature_cols(train_data)

        X = train_data[self._feature_cols].values.astype(np.float64)
        y = train_data['__target__'].values
        X = np.nan_to_num(X, nan=0.0)
        
        if self.config.task_type == 'classification':
            y = y.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        current_best = self._best_params.copy()
        radius = self.config.hpo_neighborhood_radius
        
        def objective(trial):
            params = self._get_default_params()
            for param_name in self.config.sensitive_params_ridge:
                current_val = current_best.get(param_name)
                if param_name == 'alpha':
                    base = current_val or 1.0
                    params['alpha'] = trial.suggest_float('alpha', max(0.01, base*(1-radius)), min(10.0, base*(1+radius)), log=True)
                elif param_name == 'tol':
                    base = current_val or 1e-3
                    params['tol'] = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
            
            try:
                if self.config.task_type == 'classification':
                    model = RidgeClassifier(**params)
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy', n_jobs=1)
                else:
                    model = BayesianRidge()
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
                return scores.mean()
            except:
                return float('-inf')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.hpo_n_trials_per_round, timeout=self.config.hpo_timeout_per_round, show_progress_bar=False)
        if study.best_trial:
            self._best_params.update(study.best_params)
        return {'best_params': study.best_params, 'best_value': study.best_value}

# ============================================================================
# Unified Incremental Analyst Trainer
# ============================================================================

class IncrementalAnalystTrainer:
    """Unified incremental trainer that trains all analyst base models."""
    
    def __init__(self, model_id: str, execution_mode="blank", task_type="classification", enable_burnin_hpo=True, model_configs=None):
        self.model_id = model_id
        oof_batch_days = 28 if execution_mode == "full" else 14
        self.config = IncrementalTrainingConfig(
            model_id,
            execution_mode,
            oof_batch_days=oof_batch_days,
            task_type=task_type,
            enable_burnin_hpo=enable_burnin_hpo,
        )
        self.model_configs = model_configs or {}
        
        self.trainers = {}
        if LIGHTGBM_AVAILABLE:
            lgbm_cfg = self.model_configs.get('lgbm') or {}
            if lgbm_cfg.get('enabled', True):
                self.trainers['lgbm'] = IncrementalLGBMTrainer(model_id, self.config, lgbm_cfg)

            bag_cfg = self.model_configs.get('lgbm_bag_lower') or {}
            if bag_cfg.get('enabled', False):
                self.trainers['lgbm_bag_lower'] = IncrementalLGBMBaggedTrainer(model_id, self.config, bag_cfg)

        if NGBOOST_AVAILABLE:
            self.trainers['ngboost'] = IncrementalNGBoostTrainer(model_id, self.config, self.model_configs.get('ngboost'))
        if SKLEARN_AVAILABLE:
            knn_cfg = self.model_configs.get('knn') or {}
            if knn_cfg.get('enabled', True):
                self.trainers['knn'] = IncrementalKNNTrainer(model_id, self.config, knn_cfg)
            bayes_cfg = self.model_configs.get('bayesianridge') or {}
            if bayes_cfg.get('enabled', True):
                self.trainers['bayesianridge'] = IncrementalBayesianRidgeTrainer(model_id, self.config, bayes_cfg)
            
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
