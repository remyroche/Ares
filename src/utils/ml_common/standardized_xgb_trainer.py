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

HPO Optimization Strategy:
- For REGULARIZATION parameters (gamma, min_child_weight, lambda, alpha, colsample_bytree, subsample):
  Uses IC-SNR (Information Coefficient × Signal-to-Noise Ratio) objective:
    score = median_IC × SNR_factor − IC_volatility_penalty + stability_bonus
  
  Where:
  - IC: Spearman correlation between predictions and actuals (computed on purged+embargoed OOF)
  - SNR: mean_IC / std_IC (Signal-to-Noise Ratio)
  - IC_volatility_penalty: penalizes high variance in IC across folds
  - stability_bonus: rewards consistent IC across random subsamples
  
  This favors regularization settings that:
  1. Maintain consistent predictive power (high median IC)
  2. Have robust signal (high SNR)
  3. Are stable across validation chunks (low volatility)

- For OTHER parameters (max_depth, learning_rate):
  Uses standard loss-based objective (e.g., logloss, RMSE)

Regularization Parameter Ranges (IC-SNR optimized):
- gamma: 1-5 (min loss reduction)
- min_child_weight: 20-40 (min sum of instance weight)
- lambda: 1-5 (L2 regularization)
- alpha: 0.2-1 (L1 regularization)
- colsample_bytree: 0.6-0.8 (column sampling)
- subsample: 0.7-0.9 (row sampling)

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

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

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
from .optimization.ic_snr_objective import (
    ICSNRConfig,
    ICSNRObjective,
    compute_ic_metrics_purged,
    compute_stability_across_subsamples,
    create_ic_snr_objective_for_xgb,
    DEFAULT_REGULARIZATION_RANGES,
    is_regularization_param,
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
    retrain_interval_days: int = 21  # Create OOF window every 21 days of historical data
    hpo_interval_days: int = 30  # Run HPO every 30 days of historical data
    burnin_pct: float = 1/12  # 3 months burn-in (1/12 of year)
    min_samples_for_training: int = 1000

    # XGBoost base parameters (used when not doing HPO)
    tree_method: str = "hist"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 30.0  # Higher for regularization
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    gamma: float = 3.0  # Updated for IC-SNR optimized regularization
    reg_lambda: float = 3.0  # L2 regularization
    reg_alpha: float = 0.5  # L1 regularization
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
    
    # Regularization parameter ranges (optimized with IC-SNR objective)
    # These ranges are specifically tuned for robust feature suppression
    min_child_weight_range: Tuple[float, float] = (20.0, 40.0)  # Higher = more conservative
    subsample_range: Tuple[float, float] = (0.7, 0.9)  # Row sampling
    colsample_bytree_range: Tuple[float, float] = (0.6, 0.8)  # Column sampling
    gamma_range: Tuple[float, float] = (1.0, 5.0)  # Min loss reduction
    lambda_range: Tuple[float, float] = (1.0, 5.0)  # L2 regularization (reg_lambda)
    alpha_range: Tuple[float, float] = (0.2, 1.0)  # L1 regularization (reg_alpha)
    
    # IC-SNR HPO configuration for regularization parameters
    # When True, uses IC×SNR − volatility_penalty objective for regularization HPO
    use_ic_snr_for_regularization: bool = True
    ic_snr_n_folds: int = 5  # Folds for purged CV IC computation
    ic_snr_purge_minutes: int = 30  # Purge window before validation
    ic_snr_embargo_minutes: int = 15  # Embargo window after validation
    ic_snr_weight: float = 1.0  # Weight for SNR factor
    ic_snr_volatility_penalty: float = 0.5  # Penalty for IC volatility
    ic_snr_subsample_chunks: int = 3  # Subsamples for stability check

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

        base_score = self._compute_base_score(y_tr)

        # Convert to DMatrix (with sparse support)
        dtrain = self._create_dmatrix(X_tr, y_tr, w_tr)
        dval = self._create_dmatrix(X_val, y_val, w_val)

        if should_run_hpo:
            if verbose:
                logger.info(f"   🎯 Running HPO for window {window_id} (scheduled HPO)")
            model = self._train_with_hpo(dtrain, dval, eval_metric, verbose, base_score=base_score)
            model.__used_hpo__ = True
        else:
            if verbose:
                logger.info(f"   🤖 Training with fixed parameters for window {window_id}")
            model = self._train_with_fixed_params(dtrain, dval, eval_metric, verbose, base_score=base_score)
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

    def _compute_base_score(self, y: pd.Series) -> Optional[float]:
        try:
            if y is None or len(y) == 0:
                return None
            y_array = y.values
            if self.config.task_type == "regression":
                return float(np.nanmean(y_array))
            if self.config.task_type == "classification":
                pos_rate = float(np.mean(y_array == 1))
                eps = 1e-6
                if pos_rate <= 0.0:
                    return eps
                if pos_rate >= 1.0:
                    return 1.0 - eps
                return pos_rate
        except Exception:
            return None
        return None

    def _train_with_fixed_params(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        eval_metric: str,
        verbose: bool,
        base_score: Optional[float] = None
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
            'alpha': self.config.reg_alpha,
            'eval_metric': eval_metric,
            'objective': self.config.objective,
        }

        if base_score is not None:
            params['base_score'] = float(base_score)

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
        verbose: bool,
        base_score: Optional[float] = None
    ) -> xgb.Booster:
        """
        Train XGBoost with HPO using BOHB (TPE + Hyperband).

        Uses HierarchicalParameterOptimizer with:
        - Staged optimization: Coarse Grid → Fine Grid → TPE
        - Warm start from previous runs
        - Stratified sampling (10-50% of data)
        - 300 trees during HPO
        
        For REGULARIZATION parameters (gamma, min_child_weight, lambda, alpha,
        colsample_bytree, subsample), uses IC-SNR objective:
            score = median_IC × SNR_factor − IC_volatility_penalty
            
        For OTHER parameters (max_depth, learning_rate), uses standard loss objective.
        """
        # Load warm start parameters if available
        warm_start_params = self._load_warm_start_params() if self.config.enable_warm_start else None

        # Subsample data for HPO if needed (stratified sampling)
        dtrain_hpo, dval_hpo = self._stratified_subsample(dtrain, dval)
        
        # Create IC-SNR config if enabled for regularization
        ic_snr_config = None
        if self.config.use_ic_snr_for_regularization:
            ic_snr_config = ICSNRConfig(
                n_folds=self.config.ic_snr_n_folds,
                purge_minutes=self.config.ic_snr_purge_minutes,
                embargo_minutes=self.config.ic_snr_embargo_minutes,
                snr_weight=self.config.ic_snr_weight,
                volatility_penalty_weight=self.config.ic_snr_volatility_penalty,
                min_samples_per_fold=100,
                use_median_ic=True,
                subsample_chunks=self.config.ic_snr_subsample_chunks
            )
        
        # =====================================================================
        # Phase 1: Optimize STRUCTURE parameters with standard loss objective
        # (max_depth only - min_child_weight is a regularization param)
        # =====================================================================
        structure_param_groups = [
            ParameterGroup(
                name="structure",
                params={
                    "max_depth": {
                        "type": "int",
                        "low": self.config.max_depth_range[0],
                        "high": self.config.max_depth_range[1]
                    },
                },
                priority=1,
                description="Model structure parameters (standard loss objective)"
            ),
        ]

        # Standard loss-based objective for structure parameters
        def structure_objective(params: Dict[str, Any]) -> float:
            """Standard loss objective for structure parameters."""
            xgb_params = {
                'tree_method': self.config.tree_method,
                'learning_rate': self.config.learning_rate,  # Fixed
                'max_depth': int(params.get("max_depth", 6)),
                'min_child_weight': self.config.min_child_weight,  # Fixed
                'subsample': self.config.subsample,  # Fixed
                'colsample_bytree': self.config.colsample_bytree,  # Fixed
                'gamma': self.config.gamma,  # Fixed
                'lambda': self.config.reg_lambda,  # Fixed
                'alpha': self.config.reg_alpha,  # Fixed
                'eval_metric': eval_metric,
                'objective': self.config.objective,
            }

            if base_score is not None:
                xgb_params['base_score'] = float(base_score)

            if self.config.num_class is not None:
                xgb_params['num_class'] = self.config.num_class

            try:
                model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain_hpo,
                    num_boost_round=self.config.hpo_n_estimators,
                    evals=[(dval_hpo, 'val')],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose_eval=False
                )
                # Return negative loss for maximization
                return -model.best_score
            except Exception as e:
                logger.warning(f"Structure HPO trial failed: {e}")
                return -999.0

        if verbose:
            logger.info(f"      🔍 Phase 1: Optimizing structure params (max_depth)...")

        structure_optimizer = HierarchicalParameterOptimizer(
            param_groups=structure_param_groups,
            objective_func=structure_objective,
            stages=[OptimizationStage.COARSE_GRID, OptimizationStage.TPE],
            cv_folds=3,
            scoring_metric='custom',
            direction='maximize',
            n_rounds=1,
            enable_final_refinement=False,
            random_state=42,
            verbose=verbose,
        )

        structure_result = structure_optimizer.optimize(
            X_train=dtrain_hpo, y_train=None, X_val=dval_hpo, y_val=None
        )
        best_max_depth = int(structure_result.best_params.get("max_depth", 6))

        # =====================================================================
        # Phase 2: Optimize REGULARIZATION parameters with IC-SNR objective
        # (gamma, min_child_weight, lambda, alpha, colsample_bytree, subsample)
        # =====================================================================
        if verbose:
            logger.info(f"      🎯 Phase 2: Optimizing regularization params with IC-SNR objective...")
            logger.info(f"         Goal: score = median_IC × SNR_factor − IC_volatility_penalty")

        best_reg_params = self._optimize_regularization_with_ic_snr(
            dtrain=dtrain_hpo,
            dval=dval_hpo,
            best_max_depth=best_max_depth,
            eval_metric=eval_metric,
            base_score=base_score,
            ic_snr_config=ic_snr_config,
            verbose=verbose
        )

        # =====================================================================
        # Phase 3: Optimize LEARNING RATE with standard loss objective
        # (using best structure and regularization params)
        # =====================================================================
        learning_param_groups = [
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
                priority=1,
                description="Learning rate (standard loss objective)"
            ),
        ]

        def learning_objective(params: Dict[str, Any]) -> float:
            """Standard loss objective for learning rate."""
            xgb_params = {
                'tree_method': self.config.tree_method,
                'learning_rate': float(params.get("learning_rate", 0.05)),
                'max_depth': best_max_depth,
                'min_child_weight': best_reg_params['min_child_weight'],
                'subsample': best_reg_params['subsample'],
                'colsample_bytree': best_reg_params['colsample_bytree'],
                'gamma': best_reg_params['gamma'],
                'lambda': best_reg_params['lambda'],
                'alpha': best_reg_params['alpha'],
                'eval_metric': eval_metric,
                'objective': self.config.objective,
            }

            if base_score is not None:
                xgb_params['base_score'] = float(base_score)

            if self.config.num_class is not None:
                xgb_params['num_class'] = self.config.num_class

            try:
                model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain_hpo,
                    num_boost_round=self.config.hpo_n_estimators,
                    evals=[(dval_hpo, 'val')],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose_eval=False
                )
                return -model.best_score
            except Exception as e:
                logger.warning(f"Learning rate HPO trial failed: {e}")
                return -999.0

        if verbose:
            logger.info(f"      🔍 Phase 3: Optimizing learning rate...")

        learning_optimizer = HierarchicalParameterOptimizer(
            param_groups=learning_param_groups,
            objective_func=learning_objective,
            stages=[OptimizationStage.COARSE_GRID, OptimizationStage.TPE],
            cv_folds=3,
            scoring_metric='custom',
            direction='maximize',
            n_rounds=1,
            enable_final_refinement=False,
            random_state=42,
            verbose=verbose,
        )

        learning_result = learning_optimizer.optimize(
            X_train=dtrain_hpo, y_train=None, X_val=dval_hpo, y_val=None
        )
        best_learning_rate = float(learning_result.best_params.get("learning_rate", 0.05))

        # =====================================================================
        # Combine all best parameters
        # =====================================================================
        best_params = {
            'max_depth': best_max_depth,
            'learning_rate': best_learning_rate,
            **best_reg_params
        }

        # Save best parameters for warm start
        if self.config.enable_warm_start:
            self._save_warm_start_params(best_params)

        # Train final model with best parameters on full data
        final_params = {
            'tree_method': self.config.tree_method,
            'learning_rate': best_learning_rate,
            'max_depth': best_max_depth,
            'min_child_weight': best_reg_params['min_child_weight'],
            'subsample': best_reg_params['subsample'],
            'colsample_bytree': best_reg_params['colsample_bytree'],
            'gamma': best_reg_params['gamma'],
            'lambda': best_reg_params['lambda'],
            'alpha': best_reg_params['alpha'],
            'eval_metric': eval_metric,
            'objective': self.config.objective,
        }

        if base_score is not None:
            final_params['base_score'] = float(base_score)

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

    def _optimize_regularization_with_ic_snr(
        self,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        best_max_depth: int,
        eval_metric: str,
        base_score: Optional[float],
        ic_snr_config: Optional[ICSNRConfig],
        verbose: bool
    ) -> Dict[str, float]:
        """
        Optimize regularization parameters using IC-SNR objective.
        
        Regularization parameters optimized:
        - gamma: Min loss reduction (1-5)
        - min_child_weight: Min sum of instance weight (20-40)
        - lambda: L2 regularization (1-5)
        - alpha: L1 regularization (0.2-1)
        - colsample_bytree: Column sampling (0.6-0.8)
        - subsample: Row sampling (0.7-0.9)
        
        Objective: score = median_IC × SNR_factor − IC_volatility_penalty
        
        Args:
            dtrain: Training DMatrix
            dval: Validation DMatrix
            best_max_depth: Best max_depth from structure optimization
            eval_metric: XGBoost evaluation metric
            base_score: Base score for XGBoost
            ic_snr_config: IC-SNR configuration
            verbose: Verbose output
            
        Returns:
            Dict with best regularization parameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default regularization params")
            return {
                'gamma': self.config.gamma,
                'min_child_weight': self.config.min_child_weight,
                'lambda': self.config.reg_lambda,
                'alpha': self.config.reg_alpha,
                'colsample_bytree': self.config.colsample_bytree,
                'subsample': self.config.subsample,
            }
        
        # Extract data from DMatrix for IC computation
        X_val = dval.get_data()
        y_val = dval.get_label()
        
        # Handle sparse matrices
        if hasattr(X_val, 'toarray'):
            X_val_array = X_val.toarray()
        else:
            X_val_array = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
        
        # Create IC-SNR objective
        if ic_snr_config is None:
            ic_snr_config = ICSNRConfig()
        
        # Track best result
        best_score = -np.inf
        best_params = {
            'gamma': self.config.gamma,
            'min_child_weight': self.config.min_child_weight,
            'lambda': self.config.reg_lambda,
            'alpha': self.config.reg_alpha,
            'colsample_bytree': self.config.colsample_bytree,
            'subsample': self.config.subsample,
        }
        
        def ic_snr_objective(trial: optuna.Trial) -> float:
            """IC-SNR objective for regularization parameters."""
            nonlocal best_score, best_params
            
            # Sample regularization parameters
            params = {
                'gamma': trial.suggest_float('gamma', 
                    self.config.gamma_range[0], self.config.gamma_range[1]),
                'min_child_weight': trial.suggest_float('min_child_weight',
                    self.config.min_child_weight_range[0], self.config.min_child_weight_range[1]),
                'lambda': trial.suggest_float('lambda',
                    self.config.lambda_range[0], self.config.lambda_range[1]),
                'alpha': trial.suggest_float('alpha',
                    self.config.alpha_range[0], self.config.alpha_range[1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree',
                    self.config.colsample_bytree_range[0], self.config.colsample_bytree_range[1]),
                'subsample': trial.suggest_float('subsample',
                    self.config.subsample_range[0], self.config.subsample_range[1]),
            }
            
            # Build XGBoost params
            xgb_params = {
                'tree_method': self.config.tree_method,
                'learning_rate': self.config.learning_rate,  # Fixed during regularization HPO
                'max_depth': best_max_depth,
                'min_child_weight': params['min_child_weight'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'gamma': params['gamma'],
                'lambda': params['lambda'],
                'alpha': params['alpha'],
                'eval_metric': eval_metric,
                'objective': self.config.objective,
            }
            
            if base_score is not None:
                xgb_params['base_score'] = float(base_score)
            
            if self.config.num_class is not None:
                xgb_params['num_class'] = self.config.num_class
            
            try:
                # Train model
                model = xgb.train(
                    params=xgb_params,
                    dtrain=dtrain,
                    num_boost_round=self.config.hpo_n_estimators,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose_eval=False
                )
                
                # Get predictions
                y_pred = model.predict(dval)
                
                # Compute IC metrics with purged cross-validation
                y_val_series = pd.Series(y_val)
                y_pred_series = pd.Series(y_pred, index=y_val_series.index)
                
                ic_metrics = compute_ic_metrics_purged(
                    y_true=y_val_series,
                    y_pred=y_pred_series,
                    config=ic_snr_config
                )
                
                # Compute stability across subsamples. Reuse the original
                # feature names from the validation DMatrix so that
                # xgboost.DMatrix does not emit feature-name mismatch
                # warnings when working with NumPy subsamples.
                feature_names = getattr(dval, 'feature_names', None)

                def predict_func(X):
                    if feature_names is not None:
                        d = xgb.DMatrix(X, feature_names=feature_names)
                    else:
                        d = xgb.DMatrix(X)
                    return model.predict(d)
                
                stability_score, _ = compute_stability_across_subsamples(
                    predict_func=predict_func,
                    X=X_val_array,
                    y=y_val,
                    n_subsamples=ic_snr_config.subsample_chunks
                )
                
                # Final score: IC-SNR + stability bonus
                # score = median_IC × SNR_factor − IC_volatility_penalty + stability_bonus
                stability_bonus = 0.1 * stability_score
                final_score = ic_metrics.final_score + stability_bonus
                
                # Track best
                if final_score > best_score:
                    best_score = final_score
                    best_params = params.copy()
                
                if verbose:
                    logger.debug(
                        f"IC-SNR Trial: median_IC={ic_metrics.median_ic:.4f}, "
                        f"SNR={ic_metrics.snr:.4f}, stability={stability_score:.4f}, "
                        f"score={final_score:.4f}"
                    )
                
                return final_score
                
            except Exception as e:
                logger.warning(f"IC-SNR regularization trial failed: {e}")
                return -999.0
        
        # Run Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        n_train = dtrain.num_row()
        base_max_trials = min(self.config.hpo_n_trials, 30)
        if n_train < 10000:
            n_trials = base_max_trials
        elif n_train < 50000:
            n_trials = max(5, int(0.7 * base_max_trials))
        elif n_train < 100000:
            n_trials = max(5, int(0.5 * base_max_trials))
        else:
            n_trials = max(5, int(0.3 * base_max_trials))
        
        study.optimize(
            ic_snr_objective,
            n_trials=n_trials,
            show_progress_bar=verbose,
            timeout=600  # 10 minute timeout
        )
        
        if verbose:
            logger.info(
                f"         Best IC-SNR score: {study.best_value:.4f}, "
                f"Best params: gamma={best_params['gamma']:.2f}, "
                f"mcw={best_params['min_child_weight']:.1f}, "
                f"lambda={best_params['lambda']:.2f}, "
                f"alpha={best_params['alpha']:.2f}"
            )
        
        return best_params

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
        n_val = dval.num_row()

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

        if sampling_pct >= 0.999:
            # For now, return full data (stratified sampling requires label access)
            # TODO: Implement proper stratified sampling
            logger.debug(f"Using {sampling_pct*100:.0f}% of data for HPO")
            return dtrain, dval

        rng = np.random.RandomState(42)

        y_train = dtrain.get_label()
        y_val = dval.get_label()

        def _sample_indices(labels: np.ndarray, n_samples: int) -> np.ndarray:
            if self.config.task_type == "classification" and len(np.unique(labels)) > 1:
                indices_list = []
                unique_labels, counts = np.unique(labels, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    label_idx = np.where(labels == label)[0]
                    n_label = max(1, int(round(count * sampling_pct)))
                    n_label = min(n_label, len(label_idx))
                    chosen = rng.choice(label_idx, size=n_label, replace=False)
                    indices_list.append(chosen)
                all_indices = np.concatenate(indices_list)
                rng.shuffle(all_indices)
                return all_indices
            n_sample = max(1, int(round(n_samples * sampling_pct)))
            n_sample = min(n_sample, n_samples)
            return rng.choice(n_samples, size=n_sample, replace=False)

        train_indices = _sample_indices(y_train, n_train)
        val_indices = _sample_indices(y_val, n_val)

        X_train = dtrain.get_data()
        X_val = dval.get_data()

        X_train_sub = X_train[train_indices]
        X_val_sub = X_val[val_indices]

        w_train = dtrain.get_weight()
        w_val = dval.get_weight()

        w_train_sub = w_train[train_indices] if w_train is not None and len(w_train) > 0 else None
        w_val_sub = w_val[val_indices] if w_val is not None and len(w_val) > 0 else None

        feature_names_train = dtrain.feature_names
        feature_names_val = dval.feature_names

        dtrain_sub = xgb.DMatrix(
            X_train_sub,
            label=y_train[train_indices],
            weight=w_train_sub,
            feature_names=feature_names_train,
        )

        dval_sub = xgb.DMatrix(
            X_val_sub,
            label=y_val[val_indices],
            weight=w_val_sub,
            feature_names=feature_names_val,
        )

        logger.debug(f"Using {sampling_pct*100:.0f}% of data for HPO (train={len(train_indices)}, val={len(val_indices)})")

        return dtrain_sub, dval_sub

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
