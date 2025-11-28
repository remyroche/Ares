"""
IC-SNR Based HPO Objective for Regularization Parameters

This module provides specialized objective functions for hyperparameter optimization 
of regularization parameters in XGBoost models, using Information Coefficient (IC) 
and Signal-to-Noise Ratio (SNR) metrics.

The key objective for regularization HPO is:
    score = median_IC × SNR_factor − IC_volatility_penalty

Where:
- IC: Spearman correlation between predictions and actual values
- SNR: mean_IC / std_IC (Signal-to-Noise Ratio)
- IC_volatility_penalty: penalizes high variance in IC across folds

This approach favors regularization settings that:
1. Maintain consistent predictive power (high median IC)
2. Have robust signal (high SNR)
3. Are stable across validation chunks (low volatility)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from ...purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False
    PurgedKFoldTime = None

logger = logging.getLogger(__name__)


@dataclass
class ICSNRConfig:
    """Configuration for IC-SNR based HPO objective.
    
    Attributes:
        n_folds: Number of folds for cross-validation IC computation
        purge_minutes: Minutes to purge before validation fold
        embargo_minutes: Minutes to embargo after validation fold
        snr_weight: Weight for SNR factor in final score (default 1.0)
        volatility_penalty_weight: Weight for IC volatility penalty (default 0.5)
        min_samples_per_fold: Minimum samples required per fold
        use_median_ic: Use median IC vs mean IC (median is more robust)
        subsample_chunks: Number of subsample chunks for stability check
    """
    n_folds: int = 5
    purge_minutes: int = 30
    embargo_minutes: int = 15
    snr_weight: float = 1.0
    volatility_penalty_weight: float = 0.5
    min_samples_per_fold: int = 100
    use_median_ic: bool = True
    subsample_chunks: int = 3


@dataclass 
class ICMetrics:
    """Information Coefficient metrics from cross-validation."""
    ic_values: List[float]  # IC per fold
    mean_ic: float
    median_ic: float
    std_ic: float
    snr: float  # mean_ic / std_ic
    ic_sharpe: float  # mean_ic / std_ic (same as SNR, for clarity)
    volatility_penalty: float
    final_score: float
    n_valid_folds: int


def compute_spearman_ic(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    handle_constant: bool = True
) -> float:
    """Compute Spearman Information Coefficient between predictions and actuals.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted values or probabilities
        handle_constant: If True, return 0 for constant arrays
        
    Returns:
        Spearman correlation coefficient (IC)
    """
    # Handle edge cases
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    # Check for constant arrays
    if handle_constant:
        if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
            return 0.0
    
    try:
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if np.sum(mask) < 2:
            return 0.0
            
        ic, _ = spearmanr(y_true[mask], y_pred[mask])
        
        # Handle NaN result
        if np.isnan(ic):
            return 0.0
            
        return float(ic)
    except Exception as e:
        logger.warning(f"Error computing Spearman IC: {e}")
        return 0.0


def compute_ic_metrics_purged(
    y_true: pd.Series,
    y_pred: pd.Series,
    config: ICSNRConfig = ICSNRConfig(),
    datetime_index: Optional[pd.DatetimeIndex] = None
) -> ICMetrics:
    """Compute IC metrics using purged + embargoed cross-validation.
    
    Args:
        y_true: Series of actual target values with DatetimeIndex
        y_pred: Series of predicted values with matching index
        config: IC-SNR configuration
        datetime_index: Optional explicit datetime index (uses y_true.index if not provided)
        
    Returns:
        ICMetrics with all computed metrics
    """
    # Align series
    common_index = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[common_index]
    y_pred = y_pred.loc[common_index]
    
    n_samples = len(y_true)
    if n_samples < config.n_folds * config.min_samples_per_fold:
        logger.warning(f"Insufficient samples ({n_samples}) for {config.n_folds} folds")
        return ICMetrics(
            ic_values=[0.0],
            mean_ic=0.0,
            median_ic=0.0,
            std_ic=1.0,
            snr=0.0,
            ic_sharpe=0.0,
            volatility_penalty=1.0,
            final_score=-1.0,
            n_valid_folds=0
        )
    
    # Use datetime index for purged/embargoed splits
    index = datetime_index if datetime_index is not None else y_true.index
    
    ic_values = []
    
    if PURGED_KFOLD_AVAILABLE and isinstance(index, pd.DatetimeIndex):
        # Use purged + embargoed CV
        splitter = PurgedKFoldTime(
            n_splits=config.n_folds,
            purge=pd.Timedelta(minutes=config.purge_minutes),
            embargo=pd.Timedelta(minutes=config.embargo_minutes)
        )
        
        # Create DataFrame for splitting
        X_dummy = pd.DataFrame(index=index, data={'dummy': np.zeros(len(index))})
        
        for train_idx, val_idx in splitter.split(X_dummy):
            if len(val_idx) < config.min_samples_per_fold:
                continue
                
            y_true_fold = y_true.iloc[val_idx].values
            y_pred_fold = y_pred.iloc[val_idx].values
            
            ic = compute_spearman_ic(y_true_fold, y_pred_fold)
            ic_values.append(ic)
    else:
        # Fall back to sequential splits without purging
        fold_size = n_samples // config.n_folds
        for i in range(config.n_folds):
            start = i * fold_size
            end = start + fold_size if i < config.n_folds - 1 else n_samples
            
            if end - start < config.min_samples_per_fold:
                continue
                
            y_true_fold = y_true.iloc[start:end].values
            y_pred_fold = y_pred.iloc[start:end].values
            
            ic = compute_spearman_ic(y_true_fold, y_pred_fold)
            ic_values.append(ic)
    
    # Compute aggregate metrics
    if len(ic_values) == 0:
        return ICMetrics(
            ic_values=[0.0],
            mean_ic=0.0,
            median_ic=0.0,
            std_ic=1.0,
            snr=0.0,
            ic_sharpe=0.0,
            volatility_penalty=1.0,
            final_score=-1.0,
            n_valid_folds=0
        )
    
    mean_ic = float(np.mean(ic_values))
    median_ic = float(np.median(ic_values))
    std_ic = float(np.std(ic_values)) if len(ic_values) > 1 else 0.01  # Avoid division by zero
    
    # SNR = mean IC / std IC (or use median IC)
    ic_for_snr = median_ic if config.use_median_ic else mean_ic
    snr = ic_for_snr / max(std_ic, 1e-6)
    ic_sharpe = mean_ic / max(std_ic, 1e-6)
    
    # Volatility penalty = std / (|median| + epsilon)
    volatility_penalty = std_ic / (abs(median_ic) + 0.01)
    
    # Final score: median_IC × SNR_factor − IC_volatility_penalty
    # SNR factor is log-scaled to reduce extreme values
    snr_factor = np.sign(snr) * np.log1p(abs(snr)) * config.snr_weight
    final_score = median_ic * (1 + snr_factor) - volatility_penalty * config.volatility_penalty_weight
    
    return ICMetrics(
        ic_values=ic_values,
        mean_ic=mean_ic,
        median_ic=median_ic,
        std_ic=std_ic,
        snr=snr,
        ic_sharpe=ic_sharpe,
        volatility_penalty=volatility_penalty,
        final_score=float(final_score),
        n_valid_folds=len(ic_values)
    )


def compute_stability_across_subsamples(
    predict_func: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    n_subsamples: int = 3,
    subsample_frac: float = 0.7,
    random_state: int = 42
) -> Tuple[float, List[float]]:
    """Compute IC stability across random subsamples.
    
    Args:
        predict_func: Function that takes X and returns predictions
        X: Feature array
        y: Target array
        n_subsamples: Number of subsamples to evaluate
        subsample_frac: Fraction of data to use in each subsample
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (stability_score, ic_per_subsample)
        - stability_score: 1 / (1 + coefficient of variation of ICs)
        - ic_per_subsample: List of IC values for each subsample
    """
    np.random.seed(random_state)
    n_samples = len(X)
    subsample_size = int(n_samples * subsample_frac)
    
    ic_values = []
    
    for i in range(n_subsamples):
        # Random subsample
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
        
        try:
            # Get predictions
            preds = predict_func(X_sub)
            
            # Compute IC
            ic = compute_spearman_ic(y_sub, preds)
            ic_values.append(ic)
        except Exception as e:
            logger.warning(f"Subsample {i} prediction failed: {e}")
            ic_values.append(0.0)
    
    if len(ic_values) == 0:
        return 0.0, []
    
    mean_ic = np.mean(ic_values)
    std_ic = np.std(ic_values) if len(ic_values) > 1 else 0.01
    
    # Coefficient of variation (CV) = std / |mean|
    cv = std_ic / (abs(mean_ic) + 1e-6)
    
    # Stability score: higher is better (1 = perfect stability)
    stability_score = 1.0 / (1.0 + cv)
    
    return float(stability_score), ic_values


class ICSNRObjective:
    """IC-SNR based objective function for regularization HPO.
    
    This objective is specifically designed for optimizing regularization
    parameters (gamma, min_child_weight, lambda, alpha, colsample_bytree, subsample)
    where the goal is to find settings that produce robust, consistent predictions.
    
    The objective function is:
        score = median_IC × SNR_factor − IC_volatility_penalty + stability_bonus
    
    Usage:
        objective = ICSNRObjective(dtrain, dval, config)
        score = objective.evaluate(params)
    """
    
    # Regularization parameters that should use IC-SNR objective
    REGULARIZATION_PARAMS = {
        'gamma', 'min_child_weight', 'lambda', 'alpha', 
        'colsample_bytree', 'subsample', 'reg_lambda', 'reg_alpha'
    }
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        datetime_index: Optional[pd.DatetimeIndex] = None,
        config: ICSNRConfig = ICSNRConfig(),
        base_params: Optional[Dict[str, Any]] = None,
        model_builder: Optional[Callable] = None
    ):
        """Initialize IC-SNR objective.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            datetime_index: Datetime index for purged CV (optional)
            config: IC-SNR configuration
            base_params: Base XGBoost parameters (non-regularization)
            model_builder: Optional custom model builder function
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.datetime_index = datetime_index
        self.config = config
        self.base_params = base_params or {}
        self.model_builder = model_builder
        
        # Track evaluation history
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate(
        self,
        params: Dict[str, Any],
        n_estimators: int = 300,
        early_stopping_rounds: int = 20
    ) -> float:
        """Evaluate IC-SNR objective for given parameters.
        
        Args:
            params: Hyperparameters to evaluate (including regularization)
            n_estimators: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            IC-SNR score (higher is better)
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not available")
            return -999.0
        
        # Merge with base params
        full_params = {**self.base_params, **params}
        
        # Create DMatrix
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        try:
            # Train model
            model = xgb.train(
                params=full_params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'val')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            # Get predictions on validation set
            y_pred = model.predict(dval)
            
            # Convert to pandas for IC computation
            y_val_series = pd.Series(self.y_val)
            y_pred_series = pd.Series(y_pred, index=y_val_series.index)
            
            # Compute IC metrics using purged CV
            ic_metrics = compute_ic_metrics_purged(
                y_true=y_val_series,
                y_pred=y_pred_series,
                config=self.config,
                datetime_index=self.datetime_index
            )
            
            # Compute stability across subsamples
            def predict_func(X):
                d = xgb.DMatrix(X)
                return model.predict(d)
            
            stability_score, subsample_ics = compute_stability_across_subsamples(
                predict_func=predict_func,
                X=self.X_val,
                y=self.y_val,
                n_subsamples=self.config.subsample_chunks
            )
            
            # Final score with stability bonus
            # stability_score is in [0, 1], so multiply by small factor
            stability_bonus = 0.1 * stability_score
            final_score = ic_metrics.final_score + stability_bonus
            
            # Log evaluation
            eval_record = {
                'params': params,
                'median_ic': ic_metrics.median_ic,
                'mean_ic': ic_metrics.mean_ic,
                'std_ic': ic_metrics.std_ic,
                'snr': ic_metrics.snr,
                'volatility_penalty': ic_metrics.volatility_penalty,
                'stability_score': stability_score,
                'final_score': final_score
            }
            self.evaluation_history.append(eval_record)
            
            logger.debug(
                f"IC-SNR Eval: median_IC={ic_metrics.median_ic:.4f}, "
                f"SNR={ic_metrics.snr:.4f}, stability={stability_score:.4f}, "
                f"score={final_score:.4f}"
            )
            
            return float(final_score)
            
        except Exception as e:
            logger.warning(f"IC-SNR evaluation failed: {e}")
            return -999.0
    
    def get_best_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get the best evaluation from history."""
        if not self.evaluation_history:
            return None
        return max(self.evaluation_history, key=lambda x: x['final_score'])


def create_ic_snr_objective_for_xgb(
    dtrain,  # xgb.DMatrix
    dval,    # xgb.DMatrix
    config: ICSNRConfig = ICSNRConfig(),
    base_params: Optional[Dict[str, Any]] = None,
    datetime_index: Optional[pd.DatetimeIndex] = None
) -> Callable[[Dict[str, Any]], float]:
    """Create an IC-SNR objective function for XGBoost HPO.
    
    This is a convenience function to create an objective function
    that can be used directly with Optuna or other HPO frameworks.
    
    Args:
        dtrain: XGBoost DMatrix for training
        dval: XGBoost DMatrix for validation
        config: IC-SNR configuration
        base_params: Base XGBoost parameters
        datetime_index: Optional datetime index for purged CV
        
    Returns:
        Callable objective function that takes params dict and returns score
        
    Example:
        objective = create_ic_snr_objective_for_xgb(dtrain, dval)
        
        def optuna_objective(trial):
            params = {
                'gamma': trial.suggest_float('gamma', 1, 5),
                'min_child_weight': trial.suggest_float('min_child_weight', 20, 40),
                ...
            }
            return objective(params)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost required for IC-SNR objective")
    
    # Extract arrays from DMatrix
    X_train = dtrain.get_data()
    y_train = dtrain.get_label()
    X_val = dval.get_data()
    y_val = dval.get_label()
    
    # Handle sparse matrices
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    if hasattr(X_val, 'toarray'):
        X_val = X_val.toarray()
    
    objective = ICSNRObjective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        datetime_index=datetime_index,
        config=config,
        base_params=base_params
    )
    
    return objective.evaluate


# Default regularization parameter ranges for IC-SNR HPO
DEFAULT_REGULARIZATION_RANGES = {
    'gamma': {'low': 1.0, 'high': 5.0, 'type': 'float'},
    'min_child_weight': {'low': 20.0, 'high': 40.0, 'type': 'float'},
    'lambda': {'low': 1.0, 'high': 5.0, 'type': 'float'},  # reg_lambda in XGBoost
    'alpha': {'low': 0.2, 'high': 1.0, 'type': 'float'},   # reg_alpha in XGBoost
    'colsample_bytree': {'low': 0.6, 'high': 0.8, 'type': 'float'},
    'subsample': {'low': 0.7, 'high': 0.9, 'type': 'float'},
}


def is_regularization_param(param_name: str) -> bool:
    """Check if a parameter is a regularization parameter.
    
    Args:
        param_name: Name of the parameter
        
    Returns:
        True if it's a regularization parameter that should use IC-SNR objective
    """
    return param_name in ICSNRObjective.REGULARIZATION_PARAMS


__all__ = [
    'ICSNRConfig',
    'ICMetrics',
    'compute_spearman_ic',
    'compute_ic_metrics_purged',
    'compute_stability_across_subsamples',
    'ICSNRObjective',
    'create_ic_snr_objective_for_xgb',
    'DEFAULT_REGULARIZATION_RANGES',
    'is_regularization_param',
]
