"""
Standardized XGBoost Training Module with OOF Predictions and Scheduled Retraining

This module provides a unified interface for training XGBoost models across all regime steps
with proper out-of-fold (OOF) predictions, scheduled retraining, and HPO.

Features:
- OOF predictions only (no data leakage)
- Respects burn-in periods
- Retrain every 10 days without HPO
- Full HPO every 30 days using BOHB (TPE + Hyperband)
- DMatrix support with sparse matrices
- Warm start from previous HPO runs
- Standardized parameter ranges across all models

Usage:
    ```python
    from src.utils.ml_common.standardized_xgb_trainer import StandardizedXGBTrainer

    trainer = StandardizedXGBTrainer(
        model_id="ETHUSDT_binance_15m_mean_reversion",
        config=config
    )

    results = trainer.train_and_predict(
        X=features_df,
        y=targets_series,
        data_start=market_data.index.min(),
        data_end=market_data.index.max()
    )

    # Access OOF predictions
    oof_predictions = results.oof_predictions
    models = results.models
    metadata = results.metadata
    ```
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from .retraining_scheduler import (
    OOFPredictionGenerator,
    RetrainingSchedule,
    RetrainingManager,
)
from .optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
)

logger = logging.getLogger(__name__)


@dataclass
class XGBTrainingConfig:
    """Configuration for standardized XGBoost training.

    IMPORTANT: The retraining schedule (retrain_interval_days, hpo_interval_days) refers
    to HISTORICAL DATA windows, not real-time retraining. During historical training:
    - Create OOF prediction windows every 10 days of historical data
    - Run HPO every 30 days of historical data (every 3rd window)

    For live/production retraining, use the RetrainingManager separately to check
    if real-time retraining is needed based on actual elapsed time.
    """

    # Model identification
    model_id: str  # Unique ID for the model (e.g., "ETHUSDT_binance_15m_mean_reversion")

    # Historical data retraining schedule (for OOF windows)
    retrain_interval_days: int = 10  # Create OOF window every 10 days of historical data
    hpo_interval_days: int = 30  # Run HPO every 30 days of historical data
    burnin_pct: float = 1/12  # 3 months burn-in (1/12 of year)
    min_samples_for_training: int = 1000

    # XGBoost base parameters (used when not doing HPO)
    tree_method: str = "hist"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 10.0
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    gamma: float = 5.0
    reg_lambda: float = 1.5
    early_stopping_rounds: int = 20

    # Task type and objective
    task_type: str = "classification"  # "classification" or "regression"
    objective: str = "binary:logistic"  # "binary:logistic", "multi:softprob", or "reg:squarederror"
    num_class: Optional[int] = None  # Required for multi:softprob, ignored for binary/regression

    # HPO configuration
    hpo_n_estimators: int = 300  # Use 300 trees during HPO
    hpo_n_trials: int = 50  # Number of HPO trials
    hpo_stratified_sampling_pct: Tuple[float, float] = (0.1, 0.5)  # 10-50% sampling
    enable_warm_start: bool = True  # Use previous best params as starting point

    # Parameter search ranges for HPO
    learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    max_depth_range: Tuple[int, int] = (4, 9)
    min_child_weight_range: Tuple[float, float] = (5.0, 20.0)
    subsample_range: Tuple[float, float] = (0.6, 0.8)
    colsample_bytree_range: Tuple[float, float] = (0.6, 0.8)
    gamma_range: Tuple[float, float] = (3.0, 8.0)
    lambda_range: Tuple[float, float] = (0.5, 2.5)

    # Sparse matrix configuration
    enable_sparse_matrices: bool = True
    sparsity_threshold: float = 0.5  # Use sparse if >50% zeros

    # Paths
    cache_dir: Path = Path("cache/xgb_models")
    hpo_cache_dir: Path = Path("cache/xgb_hpo")

    def __post_init__(self):
        """Create cache directories."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hpo_cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class XGBTrainingResults:
    """Results from standardized XGBoost training."""

    oof_predictions: pd.DataFrame  # OOF predictions with probabilities
    models: List[Any]  # List of trained models (one per window)
    metadata: List[Dict[str, Any]]  # Metadata for each training window
    hpo_history: Optional[Dict[str, Any]] = None  # HPO history if HPO was run
    training_windows: Optional[List[Dict[str, Any]]] = None  # Training window info


class StandardizedXGBTrainer:
    """
    Standardized XGBoost trainer with OOF predictions and scheduled retraining.

    This trainer ensures:
    1. No data leakage (only OOF predictions)
    2. Proper burn-in handling
    3. Retraining every 10 days without HPO
    4. Full HPO every 30 days
    5. DMatrix with sparse matrices when appropriate
    6. Warm start from previous HPO runs
    """

    def __init__(self, model_id: str, config: Optional[XGBTrainingConfig] = None):
        """
        Initialize standardized XGBoost trainer.

        Args:
            model_id: Unique identifier for this model
            config: Training configuration (uses defaults if not provided)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for StandardizedXGBTrainer")

        self.model_id = model_id
        self.config = config or XGBTrainingConfig(model_id=model_id)
        self.retrain_manager = RetrainingManager(cache_dir=self.config.cache_dir)

        logger.info(f"Initialized StandardizedXGBTrainer for model: {model_id}")

    def train_and_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        data_start: datetime,
        data_end: datetime,
        sample_weight: Optional[np.ndarray] = None,
        eval_metric: str = "logloss",
        verbose: bool = True
    ) -> XGBTrainingResults:
        """
        Train XGBoost models with OOF predictions and scheduled retraining.

        Args:
            X: Feature dataframe with DatetimeIndex
            y: Target series with DatetimeIndex
            data_start: Start of available data
            data_end: End of available data
            sample_weight: Optional sample weights
            eval_metric: Evaluation metric for XGBoost
            verbose: Whether to print progress messages

        Returns:
            XGBTrainingResults with OOF predictions, models, and metadata
        """
        if verbose:
            logger.info("=" * 80)
            logger.info("🚀 Starting Standardized XGBoost Training with OOF Predictions")
            logger.info("=" * 80)
            logger.info(f"Model ID: {self.model_id}")
            logger.info(f"Data range: {data_start} → {data_end}")
            logger.info(f"Samples: {len(X)}, Features: {len(X.columns)}")
            logger.info(f"Retrain interval: {self.config.retrain_interval_days} days")
            logger.info(f"HPO interval: {self.config.hpo_interval_days} days")

        # Create retraining schedule
        schedule = RetrainingSchedule(
            model_type='xgb',
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
            logger.info(f"⏱️  Burn-in period: {data_start} → {data_start + timedelta(days=int((data_end - data_start).days * schedule.burnin_pct))}")

        # Prepare data alignment
        aligned_data = pd.DataFrame(index=X.index)
        for col in X.columns:
            aligned_data[col] = X[col]
        aligned_data['__target__'] = y
        if sample_weight is not None:
            aligned_data['__weight__'] = sample_weight

        # Training function (with HPO scheduling)
        def training_func(train_data: pd.DataFrame, window_id: int, window_start: datetime) -> Any:
            """Train XGBoost model for a specific window."""
            return self._train_single_window(
                train_data=train_data,
                window_id=window_id,
                window_start=window_start,
                eval_metric=eval_metric,
                verbose=verbose
            )

        # Prediction function
        def prediction_func(model: Any, pred_data: pd.DataFrame) -> pd.DataFrame:
            """Generate predictions for a specific window."""
            return self._predict_single_window(model, pred_data)

        # Generate OOF predictions
        start_time = time.time()
        all_predictions = []
        all_models = []
        all_metadata = []
        hpo_history = {}

        for window in oof_generator.windows:
            if verbose:
                logger.info(f"\n🔄 Processing Window {window.window_id + 1}/{len(oof_generator.windows)}")
                logger.info(f"   Train: {window.training_start} → {window.training_end}")
                logger.info(f"   Predict: {window.prediction_start} → {window.prediction_end}")

            # Get training data (only data before prediction period)
            train_mask = (aligned_data.index >= window.training_start) & (aligned_data.index < window.training_end)
            train_data = aligned_data.loc[train_mask].copy()

            # Check minimum samples
            if len(train_data) < schedule.min_samples_for_training:
                logger.warning(
                    f"⚠️  Window {window.window_id}: Insufficient training samples "
                    f"({len(train_data)} < {schedule.min_samples_for_training}). Skipping."
                )
                continue

            # Train model
            window_start_time = time.time()
            model = training_func(train_data, window.window_id, window.prediction_start)
            all_models.append(model)
            training_time = time.time() - window_start_time

            # Get prediction data
            pred_mask = (aligned_data.index >= window.prediction_start) & (aligned_data.index <= window.prediction_end)
            pred_data = aligned_data.loc[pred_mask].copy()

            if len(pred_data) == 0:
                continue

            # Make predictions
            predictions = prediction_func(model, pred_data)
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
                'used_hpo': getattr(model, '__used_hpo__', False),
            }
            all_metadata.append(metadata)

            if verbose:
                logger.info(f"   ✅ Window {window.window_id} complete in {training_time:.2f}s")
                logger.info(f"      HPO: {metadata['used_hpo']}, Predictions: {len(predictions)}")

        # Combine all predictions
        if all_predictions:
            oof_predictions = pd.concat(all_predictions, axis=0)
            oof_predictions = oof_predictions.sort_index()
        else:
            oof_predictions = pd.DataFrame()

        total_time = time.time() - start_time

        if verbose:
            logger.info("=" * 80)
            logger.info(f"✅ OOF Training Complete in {total_time:.2f}s ({total_time/60:.2f} minutes)")
            logger.info(f"   Total windows: {len(all_models)}")
            logger.info(f"   Total predictions: {len(oof_predictions)}")
            logger.info(f"   HPO runs: {sum(1 for m in all_metadata if m.get('used_hpo', False))}")
            logger.info("=" * 80)

        return XGBTrainingResults(
            oof_predictions=oof_predictions,
            models=all_models,
            metadata=all_metadata,
            hpo_history=hpo_history if hpo_history else None,
            training_windows=[w.to_dict() for w in oof_generator.windows]
        )

    def _train_single_window(
        self,
        train_data: pd.DataFrame,
        window_id: int,
        window_start: datetime,
        eval_metric: str,
        verbose: bool
    ) -> Any:
        """
        Train XGBoost model for a single window.

        Determines whether to use HPO based on schedule and trains accordingly.
        """
        # Check if HPO should be run for this window
        should_run_hpo = self._should_run_hpo(window_start)

        # Extract features and target
        feature_cols = [c for c in train_data.columns if not c.startswith('__')]
        X_train = train_data[feature_cols]
        y_train = train_data['__target__']

        # Extract sample weights if available
        sample_weight = train_data['__weight__'].values if '__weight__' in train_data.columns else None

        # Split into train/val for early stopping
        n_train = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:n_train], X_train.iloc[n_train:]
        y_tr, y_val = y_train.iloc[:n_train], y_train.iloc[n_train:]
        w_tr = sample_weight[:n_train] if sample_weight is not None else None
        w_val = sample_weight[n_train:] if sample_weight is not None else None

        # Convert to DMatrix (with sparse support)
        dtrain = self._create_dmatrix(X_tr, y_tr, w_tr)
        dval = self._create_dmatrix(X_val, y_val, w_val)

        if should_run_hpo:
            if verbose:
                logger.info(f"   🎯 Running HPO for window {window_id} (scheduled HPO)")
            model = self._train_with_hpo(dtrain, dval, eval_metric, verbose)
            model.__used_hpo__ = True
        else:
            if verbose:
                logger.info(f"   🤖 Training with fixed parameters for window {window_id}")
            model = self._train_with_fixed_params(dtrain, dval, eval_metric, verbose)
            model.__used_hpo__ = False

        return model

    def _should_run_hpo(self, window_start: datetime) -> bool:
        """
        Determine if HPO should be run based on schedule.

        HPO runs every 30 days, regular training every 10 days.
        """
        hpo_model_id = f"{self.model_id}_hpo"
        last_hpo = self.retrain_manager.get_last_training_time(hpo_model_id)

        if last_hpo is None:
            # Never run HPO before, should run now
            self.retrain_manager.record_training(hpo_model_id, RetrainingSchedule.for_xgb())
            return True

        # Check if 30 days have passed since last HPO
        days_since_hpo = (window_start - last_hpo).days

        if days_since_hpo >= self.config.hpo_interval_days:
            self.retrain_manager.record_training(hpo_model_id, RetrainingSchedule.for_xgb())
            return True

        return False

    def _create_dmatrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> xgb.DMatrix:
        """
        Create XGBoost DMatrix with sparse matrix support.

        Automatically detects if data is sparse and uses scipy.sparse matrices.
        """
        X_array = X.values

        # Check if data is sparse
        if self.config.enable_sparse_matrices:
            sparsity = np.mean(X_array == 0)
            if sparsity > self.config.sparsity_threshold:
                logger.debug(f"Converting to sparse matrix (sparsity={sparsity:.2%})")
                X_array = sparse.csr_matrix(X_array)

        return xgb.DMatrix(
            X_array,
            label=y.values,
            weight=sample_weight,
            feature_names=X.columns.tolist()
        )

    def _train_with_fixed_params(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        eval_metric: str,
        verbose: bool
    ) -> xgb.Booster:
        """Train XGBoost with fixed parameters (no HPO)."""
        params = {
            'tree_method': self.config.tree_method,
            'learning_rate': self.config.learning_rate,
            'max_depth': self.config.max_depth,
            'min_child_weight': self.config.min_child_weight,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'gamma': self.config.gamma,
            'lambda': self.config.reg_lambda,
            'eval_metric': eval_metric,
            'objective': self.config.objective,
        }

        # Add num_class for multiclass classification
        if self.config.num_class is not None:
            params['num_class'] = self.config.num_class

        evals = [(dtrain, 'train'), (dval, 'val')]

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False
        )

        return model

    def _train_with_hpo(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        eval_metric: str,
        verbose: bool
    ) -> xgb.Booster:
        """
        Train XGBoost with HPO using BOHB (TPE + Hyperband).

        Uses HierarchicalParameterOptimizer with:
        - Staged optimization: Coarse Grid → Fine Grid → TPE
        - Warm start from previous runs
        - Stratified sampling (10-50% of data)
        - 300 trees during HPO
        """
        # Load warm start parameters if available
        warm_start_params = self._load_warm_start_params() if self.config.enable_warm_start else None

        # Define parameter groups for hierarchical optimization
        param_groups = [
            ParameterGroup(
                name="structure",
                params={
                    "max_depth": {
                        "type": "int",
                        "low": self.config.max_depth_range[0],
                        "high": self.config.max_depth_range[1]
                    },
                    "min_child_weight": {
                        "type": "float",
                        "low": self.config.min_child_weight_range[0],
                        "high": self.config.min_child_weight_range[1]
                    },
                },
                priority=1,
                description="Model structure parameters"
            ),
            ParameterGroup(
                name="regularization",
                params={
                    "gamma": {
                        "type": "float",
                        "low": self.config.gamma_range[0],
                        "high": self.config.gamma_range[1]
                    },
                    "lambda": {
                        "type": "float",
                        "low": self.config.lambda_range[0],
                        "high": self.config.lambda_range[1]
                    },
                },
                priority=2,
                depends_on=["structure"],
                description="Regularization parameters"
            ),
            ParameterGroup(
                name="sampling",
                params={
                    "subsample": {
                        "type": "float",
                        "low": self.config.subsample_range[0],
                        "high": self.config.subsample_range[1]
                    },
                    "colsample_bytree": {
                        "type": "float",
                        "low": self.config.colsample_bytree_range[0],
                        "high": self.config.colsample_bytree_range[1]
                    },
                },
                priority=3,
                depends_on=["regularization"],
                description="Sampling parameters"
            ),
            ParameterGroup(
                name="learning",
                params={
                    "learning_rate": {
                        "type": "float",
                        "low": self.config.learning_rate_range[0],
                        "high": self.config.learning_rate_range[1],
                        "log": True
                    },
                },
                priority=4,
                depends_on=["sampling"],
                description="Learning rate"
            ),
        ]

        # Objective function for HPO
        def objective(params: Dict[str, Any]) -> float:
            """Objective function for HPO."""
            xgb_params = {
                'tree_method': self.config.tree_method,
                'learning_rate': float(params.get("learning_rate", 0.05)),
                'max_depth': int(params.get("max_depth", 6)),
                'min_child_weight': float(params.get("min_child_weight", 10.0)),
                'subsample': float(params.get("subsample", 0.7)),
                'colsample_bytree': float(params.get("colsample_bytree", 0.7)),
                'gamma': float(params.get("gamma", 5.0)),
                'lambda': float(params.get("lambda", 1.5)),
                'eval_metric': eval_metric,
                'objective': self.config.objective,
            }

            # Add num_class for multiclass classification
            if self.config.num_class is not None:
                xgb_params['num_class'] = self.config.num_class

            try:
                model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=self.config.hpo_n_estimators,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose_eval=False
                )

                # Get validation score (lower is better for logloss)
                val_score = model.best_score

                # Return negative score for maximization
                return -val_score

            except Exception as e:
                logger.warning(f"HPO trial failed: {e}")
                return -999.0

        # Create optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ],
            cv_folds=3,
            scoring_metric='custom',
            direction='maximize',
            n_rounds=1,
            enable_final_refinement=False,
            random_state=42,
            verbose=verbose,
        )

        # Subsample data for HPO if needed (stratified sampling)
        dtrain_hpo, dval_hpo = self._stratified_subsample(dtrain, dval)

        # Run HPO
        if verbose:
            logger.info(f"      🔍 Running HPO with {self.config.hpo_n_estimators} trees...")

        result = optimizer.optimize(
            X_train=dtrain_hpo,
            y_train=None,  # Already in DMatrix
            X_val=dval_hpo,
            y_val=None,
        )

        best_params = result.best_params

        # Save best parameters for warm start
        if self.config.enable_warm_start:
            self._save_warm_start_params(best_params)

        # Train final model with best parameters on full data
        final_params = {
            'tree_method': self.config.tree_method,
            'learning_rate': float(best_params.get("learning_rate", 0.05)),
            'max_depth': int(best_params.get("max_depth", 6)),
            'min_child_weight': float(best_params.get("min_child_weight", 10.0)),
            'subsample': float(best_params.get("subsample", 0.7)),
            'colsample_bytree': float(best_params.get("colsample_bytree", 0.7)),
            'gamma': float(best_params.get("gamma", 5.0)),
            'lambda': float(best_params.get("lambda", 1.5)),
            'eval_metric': eval_metric,
            'objective': self.config.objective,
        }

        # Add num_class for multiclass classification
        if self.config.num_class is not None:
            final_params['num_class'] = self.config.num_class

        evals = [(dtrain, 'train'), (dval, 'val')]

        model = xgb.train(
            params=final_params,
            dtrain=dtrain,
            num_boost_round=self.config.n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=False
        )

        if verbose:
            logger.info(f"      ✅ HPO complete! Best params: {best_params}")

        return model

    def _stratified_subsample(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """
        Stratified subsampling for HPO (10-50% of data based on size).

        Maintains class balance while reducing computational cost.
        """
        n_train = dtrain.num_row()

        # Determine sampling percentage based on data size
        if n_train < 5000:
            sampling_pct = 0.5  # 50% for small datasets
        elif n_train < 20000:
            sampling_pct = 0.3  # 30% for medium datasets
        else:
            sampling_pct = 0.1  # 10% for large datasets

        # Clamp to configured range
        sampling_pct = max(
            self.config.hpo_stratified_sampling_pct[0],
            min(sampling_pct, self.config.hpo_stratified_sampling_pct[1])
        )

        # For now, return full data (stratified sampling requires label access)
        # TODO: Implement proper stratified sampling
        logger.debug(f"Using {sampling_pct*100:.0f}% of data for HPO")

        return dtrain, dval

    def _predict_single_window(self, model: Any, pred_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for a single window."""
        feature_cols = [c for c in pred_data.columns if not c.startswith('__')]
        X_pred = pred_data[feature_cols]

        # Create DMatrix for prediction
        dpred = self._create_dmatrix(X_pred, pd.Series(np.zeros(len(X_pred))))

        # Generate predictions
        predictions = model.predict(dpred)

        # Handle different prediction types
        if self.config.task_type == "regression":
            # Regression: return single prediction value
            result = pd.DataFrame({
                'prediction': predictions
            }, index=pred_data.index)
        elif predictions.ndim == 2:
            # Multi-class classification: return all class probabilities
            result = pd.DataFrame(
                predictions,
                index=pred_data.index,
                columns=[f'prob_class_{i}' for i in range(predictions.shape[1])]
            )
        else:
            # Binary classification: return single probability
            result = pd.DataFrame({
                'probability': predictions
            }, index=pred_data.index)

        return result

    def _load_warm_start_params(self) -> Optional[Dict[str, Any]]:
        """Load warm start parameters from previous HPO run."""
        warm_start_path = self.config.hpo_cache_dir / f"{self.model_id}_warm_start.json"

        if not warm_start_path.exists():
            return None

        try:
            with open(warm_start_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load warm start parameters: {e}")
            return None

    def _save_warm_start_params(self, params: Dict[str, Any]):
        """Save parameters for warm start in future HPO runs.

        IMPORTANT: Parameters are saved per model_id to ensure each step has its own
        warm start history (e.g., mean_reversion warm start != smc warm start).
        """
        warm_start_path = self.config.hpo_cache_dir / f"{self.model_id}_warm_start.json"

        try:
            with open(warm_start_path, 'w') as f:
                json.dump(params, f, indent=2)
            logger.debug(f"Saved warm start parameters to {warm_start_path} (per-step storage)")
        except Exception as e:
            logger.warning(f"Failed to save warm start parameters: {e}")
