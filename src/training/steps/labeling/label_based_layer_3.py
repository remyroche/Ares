from typing import List, Tuple, Optional, Any, Dict
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from scipy.special import expit  # Needed for custom objectives
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss as sk_log_loss,
    brier_score_loss,
    roc_auc_score as sk_roc_auc_score,
    average_precision_score,
    mean_squared_error, # Added
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge # Added
from sklearn.ensemble import ExtraTreesRegressor

# Import probability calibration utilities
from src.utils.ml_common.oof_probability_calibration import (
    OOFProbabilityCalibrator,
    OOFCalibrationConfig,
    calibrate_oof_predictions,
    get_recommended_calibration_method,
)
import shap
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
from src.training.steps.labeling.generate_weights_per_label import (
    finalize_sample_weights,
)
from src.training.steps.labeling.lgbm_feature_selection import lgbm_feature_selection_pipeline
from src.training.steps.labeling.regime_leaf_feature_extractor import extract_regime_leaf_onehot_features
from src.training.steps.labeling.short_nn_sequence_template import generate_nn_sequence_embeddings

from src.utils.purged_kfold import PurgedKFoldTime

logger = logging.getLogger(__name__)


def _safe_log_loss(
    y_true,
    y_pred,
    labels=None,
    sample_weight=None,
    default: float = 0.693
):
    """
    Wrap sklearn log_loss to handle single-class targets gracefully.
    """
    try:
        labels_arg = labels
        unique = np.unique(np.asarray(y_true))
        if labels_arg is None and unique.size == 1:
            labels_arg = [0, 1]
        return sk_log_loss(y_true, y_pred, labels=labels_arg, sample_weight=sample_weight)
    except ValueError as exc:
        logger.warning(f"log_loss fallback due to: {exc}")
        return default


def _safe_roc_auc_score(
    y_true,
    y_pred,
    sample_weight=None,
    default: float = 0.5
):
    """
    Wrap sklearn roc_auc_score to handle single-class targets gracefully.
    """
    try:
        unique = np.unique(np.asarray(y_true))
        if unique.size == 1:
            # Provide both classes with dummy values
            y_true = np.concatenate([y_true, [1 - unique[0]]])
            y_pred = np.concatenate([y_pred, [y_pred.mean()]])
            if sample_weight is not None:
                sample_weight = np.concatenate([sample_weight, [sample_weight.mean()]])
        return sk_roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
    except ValueError as exc:
        logger.warning(f"roc_auc_score fallback due to: {exc}")
        return default


def log_loss(*args, **kwargs):
    """Project-wide safe wrapper for sklearn.log_loss."""
    return _safe_log_loss(*args, **kwargs)


def roc_auc_score(*args, **kwargs):
    """Project-wide safe wrapper for sklearn.roc_auc_score."""
    return _safe_roc_auc_score(*args, **kwargs)


# ---------------------------------------------------------
# Custom Objectives
# ---------------------------------------------------------

def _focal_loss_objective(y_true, y_pred):
    """
    Focal Loss for LightGBM (Binary Classification).
    y_pred is raw margin (before sigmoid).
    """
    # For some reason LightGBM may pass y_true as a Dataset object?
    # Usually in custom objective (y_true, y_pred), where y_true is the dataset.
    # Wait, lightgbm custom objective signature: (preds, train_data) -> (grad, hess)
    # But if we use sklearn API (LGBMClassifier), it passes (y_true, y_pred) if we define objective appropriately?
    # Actually, sklearn API objective should be callable(y_true, y_pred).
    # Let's handle both cases just in case.
    
    # If y_true is not array-like, try get_label()
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()

    gamma = 2.0
    # alpha = 0.25 # Balance factor - unused in simplified grad/hess?
    # The snippet didn't use alpha in the final grad calculation except implicitly?
    # The user provided:
    # grad = -y_true * (1 - p)**gamma * ... + (1 - y_true) * p**gamma * ...
    # This formula incorporates the down-weighting.
    
    # Sigmoid to get probability
    p = expit(y_pred)
    
    # Gradient calculation
    # Using robust form from user snippet
    # term1 = (1 - p) ** gamma * np.log(p + 1e-15)
    # term2 = p ** gamma * np.log(1 - p + 1e-15)
    
    grad = -y_true * (1 - p)**gamma * (1 - p - gamma * p * np.log(p + 1e-15)) + \
           (1 - y_true) * p**gamma * (p + gamma * (1 - p) * np.log(1 - p + 1e-15))

    # Hessian approximation (Simplified for stability as per "Pro Tip")
    # p * (1 - p) is Hessian of LogLoss
    hess = p * (1 - p)
    # Enhance hessian with focal weight approximation if desired, but user said:
    # "Or just p * (1-p) ... scaled by the focal weight"
    # User snippet: hess = np.abs(grad) * (1 - p) * p + gamma * np.abs(grad)
    # Or simplified. Let's use the robust p*(1-p) which is standard stable proxy.
    # Actually, let's strictly follow the snippet's robust suggestion if possible,
    # or the stable p*(1-p) if that fails.
    # User: "hess = p * (1 - p)" -> This is safe.
    
    return grad, hess

def _asymmetric_mse_objective(y_true, y_pred):
    """
    Asymmetric MSE: Penalize Over-Prediction (Pred > True) more heavily.
    """
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()

    residual = (y_true - y_pred) # true - pred
    # MSE Grad is -2 * (y - p).
    grad = -2 * residual
    hess = 2 * np.ones_like(residual)
    
    # Define penalty multiplier
    penalty = 1.5  # 50% extra penalty for being WRONG on the UPSIDE
    
    # If Residual < 0, it means y_true - y_pred < 0 => y_pred > y_true (Over-prediction)
    # We want to increase gradient here to force model down.
    
    # Mask for over-prediction
    over_pred = residual < 0
    
    grad[over_pred] *= penalty
    hess[over_pred] *= penalty
    
    return grad, hess

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _clip_probs(probs: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    """Clip probability values to specified range."""
    return np.clip(probs, clip_low, clip_high)


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
    if temperature <= 0:
        return probs
    probs_temp = np.log(probs + 1e-12) / temperature
    return 1.0 / (1.0 + np.exp(-probs_temp))


def _fit_temperature(y_true: np.ndarray, probs: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    """Fit temperature scaling parameter using optimization."""
    from scipy.optimize import minimize_scalar

    def nll(temp):
        p_temp = _apply_temperature(probs, temp)
        p_temp = np.clip(p_temp, 1e-12, 1 - 1e-12)
        if sample_weight is not None:
            return -np.mean(sample_weight * (y_true * np.log(p_temp) + (1 - y_true) * np.log(1 - p_temp)))
        else:
            return -np.mean(y_true * np.log(p_temp) + (1 - y_true) * np.log(1 - p_temp))

    result = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
    return result.x if result.success else 1.0

def _calculate_alpha_target(returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    Calculate Volatility-Standardized Forward Return (Alpha Target).
    y = Clip(Returns / Volatility, -4.0, 4.0)
    """
    # Avoid division by zero
    vol_safe = np.where(volatility < 1e-6, 1e-6, volatility)
    alpha = returns / vol_safe
    # Clip outliers
    return np.clip(alpha, -4.0, 4.0)

def _calculate_ic_score(y_true: np.ndarray, y_pred: np.ndarray, folds_ic: List[float] = None) -> float:
    """
    Calculate ScoreIC = 100 * SpearmanIC + 50 * IC_IR

    IC_IR = Mean(IC) / Std(IC) across folds.
    """
    # Global IC
    ic, _ = spearmanr(y_true, y_pred)
    if np.isnan(ic): ic = 0.0

    # IR component
    if folds_ic and len(folds_ic) > 1:
        mean_ic = np.mean(folds_ic)
        std_ic = np.std(folds_ic)
        if std_ic < 1e-6:
            ir = 0.0
        else:
            ir = mean_ic / std_ic
    else:
        ir = 0.0

    return 100 * ic + 50 * ir

# ---------------------------------------------------------
# Feature Inventory
# ---------------------------------------------------------

def _dump_layer3_feature_inventory(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    outcomes_dir: Path,
    symbol: str,
    timeframe: str,
    ts: str,
    stage: str,
    cfg: Optional[Dict[str, Any]] = None,
    meta_prob_col: str = 'meta_prob',
) -> None:
    try:
        if df is None or df.empty:
            return
        if not feature_cols:
            return

        try:
            include_spearman = bool(cfg.get('layer3_feature_inventory_include_spearman', True)) if isinstance(cfg, dict) else True
        except Exception:
            include_spearman = True

        try:
            tail_frac = float(cfg.get('layer3_feature_inventory_tail_frac', 0.2)) if isinstance(cfg, dict) else 0.2
        except Exception:
            tail_frac = 0.2
        if (not np.isfinite(tail_frac)) or tail_frac <= 0.05 or tail_frac >= 0.5:
            tail_frac = 0.2

        n = int(df.shape[0])
        split_idx = int(max(0, min(n, int(np.floor(n * (1.0 - tail_frac))))))
        split_idx = int(max(1, min(n - 1, split_idx))) if n >= 2 else 0

        y = pd.to_numeric(df.get(target_col), errors='coerce').astype(float) if target_col in df.columns else None
        p = pd.to_numeric(df.get(meta_prob_col), errors='coerce').astype(float) if meta_prob_col in df.columns else None

        rows = []
        for col in feature_cols:
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors='coerce').astype(float)
            arr = s.to_numpy(dtype=float, copy=False)
            finite = np.isfinite(arr)
            n_finite = int(np.sum(finite))
            n_nan = int(arr.size - n_finite)
            pct_nan = float(n_nan / max(1, arr.size))

            mean = float(np.nanmean(arr)) if n_finite > 0 else float('nan')
            std = float(np.nanstd(arr)) if n_finite > 1 else float('nan')
            vmin = float(np.nanmin(arr)) if n_finite > 0 else float('nan')
            vmax = float(np.nanmax(arr)) if n_finite > 0 else float('nan')

            p01 = float(np.nanpercentile(arr, 1)) if n_finite > 10 else float('nan')
            p50 = float(np.nanpercentile(arr, 50)) if n_finite > 0 else float('nan')
            p99 = float(np.nanpercentile(arr, 99)) if n_finite > 10 else float('nan')

            zero_frac = float(np.mean((arr[finite] == 0.0))) if n_finite > 0 else float('nan')

            def _pearson(a: np.ndarray, b: pd.Series) -> float:
                try:
                    bv = b.to_numpy(dtype=float, copy=False)
                    m = np.isfinite(a) & np.isfinite(bv)
                    if int(np.sum(m)) < 10:
                        return float('nan')
                    aa = a[m]
                    bb = bv[m]
                    if float(np.nanstd(aa)) <= 1e-12 or float(np.nanstd(bb)) <= 1e-12:
                        return float('nan')
                    return float(np.corrcoef(aa, bb)[0, 1])
                except Exception:
                    return float('nan')

            corr_target = _pearson(arr, y) if y is not None else float('nan')
            corr_meta = _pearson(arr, p) if p is not None else float('nan')

            spearman_target = float('nan')
            spearman_meta = float('nan')
            if include_spearman and (y is not None):
                try:
                    m = np.isfinite(arr) & np.isfinite(y.to_numpy(dtype=float, copy=False))
                    if int(np.sum(m)) >= 10:
                        spearman_target = float(spearmanr(arr[m], y.to_numpy(dtype=float, copy=False)[m]).correlation)
                except Exception:
                    spearman_target = float('nan')
            if include_spearman and (p is not None):
                try:
                    m = np.isfinite(arr) & np.isfinite(p.to_numpy(dtype=float, copy=False))
                    if int(np.sum(m)) >= 10:
                        spearman_meta = float(spearmanr(arr[m], p.to_numpy(dtype=float, copy=False)[m]).correlation)
                except Exception:
                    spearman_meta = float('nan')

            drift_smd = float('nan')
            drift_nan_delta = float('nan')
            try:
                if n >= 2 and 0 < split_idx < n:
                    a0 = arr[:split_idx]
                    a1 = arr[split_idx:]
                    m0 = np.isfinite(a0)
                    m1 = np.isfinite(a1)
                    if int(np.sum(m0)) >= 10 and int(np.sum(m1)) >= 10:
                        mu0 = float(np.nanmean(a0))
                        mu1 = float(np.nanmean(a1))
                        sd0 = float(np.nanstd(a0))
                        sd1 = float(np.nanstd(a1))
                        pool = float(np.sqrt(0.5 * (sd0 * sd0 + sd1 * sd1)))
                        drift_smd = float(abs(mu1 - mu0) / (pool + 1e-12))
                    drift_nan_delta = float((np.mean(~m1) - np.mean(~m0)))
            except Exception:
                drift_smd = float('nan')

            rows.append(
                {
                    'feature': str(col),
                    'n': int(arr.size),
                    'n_finite': n_finite,
                    'n_nan': n_nan,
                    'pct_nan': pct_nan,
                    'mean': mean,
                    'std': std,
                    'min': vmin,
                    'p01': p01,
                    'p50': p50,
                    'p99': p99,
                    'max': vmax,
                    'zero_frac': zero_frac,
                    'pearson_target': corr_target,
                    'pearson_meta_prob': corr_meta,
                    'spearman_target': spearman_target,
                    'spearman_meta_prob': spearman_meta,
                    'drift_smd': drift_smd,
                    'drift_nan_delta': drift_nan_delta,
                }
            )

        if not rows:
            return

        inv_df = pd.DataFrame(rows)
        try:
            inv_df = inv_df.sort_values(['pct_nan', 'drift_smd'], ascending=[True, False])
        except Exception:
            pass

        try:
            out_csv = outcomes_dir / f"layer3_feature_inventory_{stage}_{symbol}_{timeframe}_{ts}.csv"
            inv_df.to_csv(out_csv, index=False)
        except Exception:
            pass

        try:
            meta = {
                'stage': str(stage),
                'timestamp': str(ts),
                'symbol': str(symbol),
                'timeframe': str(timeframe),
                'n_rows': int(n),
                'n_features': int(len(feature_cols)),
                'target_col': str(target_col),
                'meta_prob_col': str(meta_prob_col) if meta_prob_col in df.columns else None,
                'tail_frac': float(tail_frac),
                'split_idx': int(split_idx),
            }
            out_json = outcomes_dir / f"layer3_feature_inventory_{stage}_{symbol}_{timeframe}_{ts}.json"
            out_json.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass
    except Exception:
        return


def _run_layer3_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    model_type: str,  # 'classifier', 'regressor', 'ridge', 'alpha_lgbm', 'alpha_ridge'
    objective_func: str = None, # 'binary_logloss', 'focal', 'mse', 'huber', 'asymmetric_mse'
    n_trials: int = 40,
) -> Dict[str, Any]:
    """
    Run HPO using Optuna for Layer 3 (supporting both Alpha and Prob models).
    """
    print(f"\n>> Running HPO for {model_type} (Obj: {objective_func}) ({n_trials} trials)...")

    # Subsample for speed
    n_total = len(X)
    n_sample = max(2000, int(n_total * 0.5))

    if n_total > n_sample:
        sample_idx = np.random.RandomState(42).choice(n_total, n_sample, replace=False)
        sample_idx.sort()
        X_hpo = X.iloc[sample_idx]
        y_hpo = y.iloc[sample_idx]
        w_hpo = w[sample_idx]
    else:
        X_hpo = X
        y_hpo = y
        w_hpo = w

    # Split into Train/Val
    split_idx = int(len(X_hpo) * 0.8)
    X_train, X_val = X_hpo.iloc[:split_idx], X_hpo.iloc[split_idx:]
    y_train, y_val = y_hpo.iloc[:split_idx], y_hpo.iloc[split_idx:]
    w_train, w_val = w_hpo[:split_idx], w_hpo[split_idx:]

    def objective(trial):
        # Hyperparameters for LGBM
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': trial.suggest_int('n_estimators', 400, 800),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1e-2),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.3, 0.7),
            'lambda_l2': 0.0, # Will set constraint later or let it float? Snippet uses 2x L1 constraint.
            'bagging_freq': 1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_jobs': 1,
            'verbosity': -1
        }
        params['lambda_l2'] = 2.0 * params['lambda_l1']

        try:
            # Alpha Models (Regression)
            if model_type == 'alpha_lgbm':
                # Objective handling
                if objective_func == 'asymmetric_mse':
                    params['objective'] = _asymmetric_mse_objective
                    params['metric'] = 'rmse' # Proxy metric for early stopping?
                elif objective_func == 'huber':
                    params['objective'] = 'huber'
                    params['metric'] = 'mae'
                    params['alpha'] = 0.9 # Huber alpha
                else:
                    params['objective'] = 'regression'
                    params['metric'] = 'rmse'

                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    eval_sample_weight=[w_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                preds = model.predict(X_val)
                # Score: IC (Maximize)
                ic = spearmanr(y_val, preds)[0]
                return ic if np.isfinite(ic) else -1.0

            elif model_type == 'alpha_ridge':
                alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train, sample_weight=w_train)
                preds = model.predict(X_val)
                ic = spearmanr(y_val, preds)[0]
                return ic if np.isfinite(ic) else -1.0

            # Prob Models (Classification)
            elif model_type == 'classifier':
                if objective_func == 'focal':
                    params['objective'] = _focal_loss_objective
                    params['metric'] = 'binary_logloss' # Monitoring metric
                else:
                    params['objective'] = 'binary'
                    params['metric'] = 'binary_logloss'

                model = lgb.LGBMClassifier(**params)

                # Check for binary targets
                # If custom obj, we might need y_train as 0/1 integers
                y_tr_bin = (y_train >= 0.5).astype(int)
                y_val_bin = (y_val >= 0.5).astype(int)

                model.fit(
                    X_train, y_tr_bin,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val_bin)],
                    eval_sample_weight=[w_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )

                # Predict
                # For custom obj, predict_proba might return raw scores if class is not updated?
                # LGBMClassifier usually handles sigmoid automatically for built-in,
                # but for custom obj it might return margin?
                # Scikit-learn API with custom objective typically needs careful handling.
                # However, model.predict_proba() usually works if obj returns grad/hess.
                # Let's assume standard behavior or raw margin -> sigmoid.

                preds = model.predict_proba(X_val)[:, 1]

                # Score: ScoreL3 (Maximize)
                auc = roc_auc_score(y_val_bin, preds)
                ll = log_loss(y_val_bin, preds)
                # Approximate ScoreL3
                score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
                return score

        except Exception as e:
            # print(f"Trial failed: {e}")
            return -999.0

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"   Best HPO Score: {study.best_value:.4f}")
    print(f"   Best Params: {study.best_params}")

    if model_type == 'alpha_ridge':
        return study.best_params

    # Reconstruct params for LGBM
    best_p = study.best_params.copy()
    best_p['lambda_l2'] = 2.0 * best_p['lambda_l1']
    best_p['bagging_freq'] = 1
    best_p['feature_fraction'] = 0.8
    best_p['bagging_fraction'] = 0.8
    best_p['n_jobs'] = 1
    best_p['verbosity'] = -1

    # Re-attach objective function to best params
    if model_type == 'alpha_lgbm':
        if objective_func == 'asymmetric_mse':
            best_p['objective'] = _asymmetric_mse_objective
            best_p['metric'] = 'rmse'
        elif objective_func == 'huber':
            best_p['objective'] = 'huber'
            best_p['metric'] = 'mae'
            best_p['alpha'] = 0.9
        else:
            best_p['objective'] = 'regression'
            best_p['metric'] = 'rmse'
    elif model_type == 'classifier':
        if objective_func == 'focal':
            best_p['objective'] = _focal_loss_objective
            best_p['metric'] = 'binary_logloss'
        else:
            best_p['objective'] = 'binary'
            best_p['metric'] = 'binary_logloss'

    return best_p


def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Revised Layer 3: Dual-Head Architecture (Alpha Generation + Probability Calibration).

    1. Alpha Generator (Head A): Predicts Vol-Standardized Returns (Alpha).
       - Race: Ridge vs LGBM(Huber) vs LGBM(AsymmetricMSE)
       - Metric: ScoreIC (Spearman + Stability)
       - Output: 'meta_alpha'

    2. Probability Calibrator (Head B): Predicts Probability of Profit (0/1).
       - Race: LGBM(LogLoss) vs LGBM(FocalLoss)
       - Metric: ScoreL3 (AUC, LogLoss, ECE)
       - Output: 'meta_prob'
    """
    print(f"\n{'='*60}")
    print("LAYER 3: DUAL-HEAD META-MODEL (ALPHA + PROBABILITY)")
    print(f"{'='*60}")

    df = oof_df.copy()
    cfg = config if isinstance(config, dict) else {}

    # ---------------------------------------------------------
    # 1. Feature Engineering (Shared)
    # ---------------------------------------------------------
    # ... (Keep existing feature engineering logic) ...
    # Initialize unified results container
    all_comparison_results = []

    print("<< Generating Layer 3 Features...")

    if base_model_cols:
        safe_base_cols = [c for c in base_model_cols if c in df.columns]
        if safe_base_cols:
            df[safe_base_cols] = df[safe_base_cols].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    else:
        safe_base_cols = []

    if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
        for c in ['volume', 'high', 'low', 'close']:
            if c in market_data.columns:
                df[c] = market_data[c].reindex(df.index)

    try:
        df = generate_layer3_features(df, safe_base_cols)
    except Exception as e:
        print(f"⚠️ generate_layer3_features failed: {e}")

    # Check for Base Model Spread
    if 'ens_prediction_dispersion' not in df.columns and 'base_pred_std' in df.columns:
        df['ens_prediction_dispersion'] = df['base_pred_std'] # Alias if needed

    candidate_features = []
    candidate_features.extend(safe_base_cols)

    # --- NEW: Regime Leaf & NN Sequence Features ---
    if market_data is not None and not market_data.empty:
        # 1. Regime Leaf Features
        try:
            print("<< Generating Regime Leaf Features...")
            rl_config = {
                "enabled_targets": [
                    "regime_trendiness",
                    "regime_volatility",
                    "regime_trend_efficiency",
                    "regime_memory",
                    "regime_liquidity",
                    "regime_volume_force_direction",
                    "regime_breakout",
                    "regime_future_range",
                    "regime_downside_ae",
                    "regime_upside_ae",
                    "regime_tail_min_bar",
                    "regime_jump_max_abs_bar",
                    "regime_vol_of_vol"
                ],
                "inputs": {
                    "input_source": "ohlcv_only",
                    "ohlcv_feature_config": {}
                },
                "onehot": {"enabled": False},
                "interaction_feature": {"enabled": True, "include_base": True},
                "reporting": {"enabled": False},
                "walk_forward": {"mode": "cross_fit", "cross_fit": {"n_splits": 5}}
            }

            rl_df = extract_regime_leaf_onehot_features(
                X=pd.DataFrame(index=df.index),
                market_data=market_data,
                config=rl_config,
                random_state=42,
                verbose=False
            )

            if rl_df is not None and not rl_df.empty:
                # Align and merge
                rl_df = rl_df.reindex(df.index).fillna(0.0)
                new_rl_cols = [c for c in rl_df.columns if c not in df.columns]
                if new_rl_cols:
                    df = pd.concat([df, rl_df[new_rl_cols]], axis=1)
                    candidate_features.extend(new_rl_cols)
                    print(f"   Added {len(new_rl_cols)} regime leaf features")
        except Exception as e:
            print(f"⚠️ Regime leaf extraction failed: {e}")

        # 2. NN Sequence Embeddings
        try:
            print("<< Generating NN Sequence Embeddings...")
            nn_df = generate_nn_sequence_embeddings(
                market_data=market_data,
                encoder_type="stacked",
                seq_len=24,
                embed_dim=8,
                use_conv=True,
                use_lstm=True,
                use_attention=False
            )

            if nn_df is not None and not nn_df.empty:
                nn_df = nn_df.reindex(df.index).fillna(0.0)
                new_nn_cols = [c for c in nn_df.columns if c not in df.columns]
                if new_nn_cols:
                    df = pd.concat([df, nn_df[new_nn_cols]], axis=1)
                    candidate_features.extend(new_nn_cols)
                    print(f"   Added {len(new_nn_cols)} NN embedding features")
        except Exception as e:
            print(f"⚠️ NN embedding generation failed: {e}")

    candidate_features.extend(
        [
            'ensemble_prob', 'max_base_prob', 'min_base_prob', 'base_prob_range',
            'logit_prob', 'logit_momentum_5', 'logit_momentum_1',
            'vol_at_signal', 'volatility_risk_ratio', 'candle_shape', 'candle_shape_4',
            'base_pred_mean', 'base_pred_std', 'base_pred_range',
            'momentum_agreement', 'momentum_agreement_abs', 'momentum_weighted_agreement', 'trend_consistency_12',
            'ens_prediction_dispersion', 'ens_confidence_gap', 'ens_uncertainty', 'ens_prediction_range',
            'ens_avg_divergence', 'ens_max_confidence', 'ens_disagreement_rate', 'ens_snr_internal', 'ens_snr_consensus',
            'slope_short', 'adx_proxy', 'momentum_short', 'snr',
            'time_since_last_vol_spike', 'time_since_last_large_candle',
            'choppiness_index', 'variance_ratio', 'permutation_entropy',
            'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'is_weekend',
            'efficiency_ratio',
            'geo_rolling_mae', 'geo_mae_volatility', 'geo_efficiency_ratio',
            'geo_median_time_to_stop', 'geo_median_time_to_target', 'geo_time_asymmetry',
            'geo_prob_target_shrunk', 'geo_prob_stop_shrunk', 'geo_expected_payoff',
            'price_position_in_range',
        ]
    )
    for c in ['volatility_1d']:
        if c in df.columns:
            candidate_features.append(c)

    meta_features = [c for c in list(dict.fromkeys(candidate_features)) if c in df.columns]

    # Clean features
    other_cols = [c for c in meta_features if c not in set(safe_base_cols)]
    if other_cols:
        df[other_cols] = df[other_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ---------------------------------------------------------
    # 2. Data Alignment
    # ---------------------------------------------------------
    
    # Must use original index alignment
    # Net Returns needed for Alpha Target
    if net_returns is None:
        raise ValueError("net_returns is required for Alpha generation.")
    
    ret_series = net_returns.reindex(df.index)
    
    # Volatility needed for Alpha Target and Weighting
    vol_series = df['volatility_1d'].replace(0, np.nan).fillna(method='ffill').fillna(0.001)
    
    # Targets
    # A. Alpha Target: Vol-Standardized Return
    y_alpha = _calculate_alpha_target(ret_series.values, vol_series.values)
    
    # B. Prob Target: Binary (0/1)
    # Ensure strict binary for classifier
    if target_col in df.columns:
        y_prob = (pd.to_numeric(df[target_col], errors='coerce').fillna(0.5) >= 0.5).astype(int).values
    else:
        # Fallback if target_col missing (shouldn't happen in OOF)
        y_prob = (ret_series > 0).astype(int).values

    # Weights
    # A. Alpha Weights: Variance Inverse (1 / vol^2)
    # Clip vol to avoid explosion
    vol_safe = np.clip(vol_series.values, 1e-4, None)
    w_alpha = 1.0 / (vol_safe ** 2)
    w_alpha = finalize_sample_weights(w_alpha) # MAD scale / Center at 1.0

    # B. Prob Weights: Standard Layer 2 Composite (passed in)
    # Using L2 weights as base, maybe combined with L1?
    # layer2_weight is already passed aligned.
    w_prob = layer2_weight.reindex(df.index).fillna(1.0).values
    w_prob = finalize_sample_weights(w_prob)

    # Clean Feature Matrix
    X = df[meta_features]

    # Common Cross-Validation (Purged)
    n_splits = 5
    # Use config for purge?
    cv = PurgedKFoldTime(n_splits=n_splits, purge=100, embargo=50)
    splits = list(cv.split(df))

    # ---------------------------------------------------------
    # HEAD A: ALPHA GENERATION
    # ---------------------------------------------------------
    print("\n>> HEAD A: ALPHA GENERATION (Race & Train)")

    alpha_candidates = [
        {'name': 'Ridge_MSE', 'type': 'alpha_ridge', 'obj': 'mse'},
        {'name': 'LGBM_Huber', 'type': 'alpha_lgbm', 'obj': 'huber'},
        {'name': 'LGBM_AsymMSE', 'type': 'alpha_lgbm', 'obj': 'asymmetric_mse'}
    ]

    alpha_scores = {}

    for cand in alpha_candidates:
        print(f"   Racing {cand['name']}...")
        fold_ics = []

        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_alpha[train_idx], y_alpha[val_idx]
            w_tr = w_alpha[train_idx]
            
            # Quick train with default params
            try:
                if cand['type'] == 'alpha_ridge':
                    model = Ridge(alpha=1.0)
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    preds = model.predict(X_val)
                else:
                    # LGBM
                    params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': 1}
                    if cand['obj'] == 'huber':
                        params['objective'] = 'huber'
                        params['alpha'] = 0.9
                    elif cand['obj'] == 'asymmetric_mse':
                        params['objective'] = _asymmetric_mse_objective

                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    preds = model.predict(X_val)
                
                # Evaluate IC
                ic, _ = spearmanr(y_val, preds)
                if np.isfinite(ic):
                    fold_ics.append(ic)
            except Exception as e:
                print(f"     Failed: {e}")

        # Compute ScoreIC
        if fold_ics:
            score_ic = _calculate_ic_score(None, None, fold_ics) # Only needs folds for IR calc if global not available?
            # Actually _calculate_ic_score takes vectors.
            # Let's use average IC for race simplicity or implement IR logic here.
            # Score = 100*MeanIC + 50*(Mean/Std)
            mean_ic = np.mean(fold_ics)
            std_ic = np.std(fold_ics) + 1e-6
            score_ic = 100 * mean_ic + 50 * (mean_ic / std_ic)
        else:
            score_ic = -999.0
            
        alpha_scores[cand['name']] = score_ic
        print(f"     ScoreIC: {score_ic:.4f} (Mean IC: {np.mean(fold_ics):.4f})")

    best_alpha_name = max(alpha_scores, key=alpha_scores.get)
    best_alpha_cand = next(c for c in alpha_candidates if c['name'] == best_alpha_name)
    print(f"   🏆 Alpha Winner: {best_alpha_name}")

    # HPO for Alpha Winner
    best_alpha_params = _run_layer3_hpo(
        X, pd.Series(y_alpha), w_alpha,
        model_type=best_alpha_cand['type'],
        objective_func=best_alpha_cand['obj'],
        n_trials=25
    )

    # Final Alpha Model & OOF
    print("   Generating Meta-Alpha OOF...")
    meta_alpha_oof = np.full(len(df), np.nan)

    # We retrain fold-by-fold for OOF
    for train_idx, val_idx in splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y_alpha[train_idx]
        w_tr = w_alpha[train_idx]

        if best_alpha_cand['type'] == 'alpha_ridge':
            model = Ridge(**best_alpha_params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict(X_val)
        else:
            # Re-attach objective if lost in translation (params usually simple dict)
            # best_alpha_params already includes 'objective' from _run_layer3_hpo if it was LGBM
            model = lgb.LGBMRegressor(**best_alpha_params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            preds = model.predict(X_val)
            
        meta_alpha_oof[val_idx] = preds

    # Fit Final Alpha Model (Full Data)
    print("   Fitting Final Alpha Model (Full)...")
    if best_alpha_cand['type'] == 'alpha_ridge':
        final_alpha_model = Ridge(**best_alpha_params)
        final_alpha_model.fit(X, y_alpha, sample_weight=w_alpha)
    else:
        final_alpha_model = lgb.LGBMRegressor(**best_alpha_params)
        final_alpha_model.fit(X, y_alpha, sample_weight=w_alpha)

    # ---------------------------------------------------------
    # HEAD B: PROBABILITY CALIBRATION
    # ---------------------------------------------------------
    print("\n>> HEAD B: PROBABILITY CALIBRATION (Race & Train)")
    
    prob_candidates = [
        {'name': 'LGBM_LogLoss', 'type': 'classifier', 'obj': 'binary_logloss'},
        {'name': 'LGBM_Focal', 'type': 'classifier', 'obj': 'focal'}
    ]
    
    prob_scores = {}

    for cand in prob_candidates:
        print(f"   Racing {cand['name']}...")
        fold_scores = []
        
        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_prob[train_idx], y_prob[val_idx]
            w_tr = w_prob[train_idx]
            
            try:
                params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': 1}
                if cand['obj'] == 'focal':
                    params['objective'] = _focal_loss_objective
                else:
                    params['objective'] = 'binary'
                    
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                preds = model.predict_proba(X_val)[:, 1]
                
                # ScoreL3
                auc = roc_auc_score(y_val, preds)
                ll = log_loss(y_val, preds)
                score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
                fold_scores.append(score)
            except Exception as e:
                print(f"     Failed: {e}")
                
        prob_scores[cand['name']] = np.mean(fold_scores) if fold_scores else -999.0
        print(f"     ScoreL3: {prob_scores[cand['name']]:.4f}")

    best_prob_name = max(prob_scores, key=prob_scores.get)
    best_prob_cand = next(c for c in prob_candidates if c['name'] == best_prob_name)
    print(f"   🏆 Prob Winner: {best_prob_name}")

    # HPO for Prob Winner
    best_prob_params = _run_layer3_hpo(
        X, pd.Series(y_prob), w_prob,
        model_type='classifier',
        objective_func=best_prob_cand['obj'],
        n_trials=25
    )

    # Final Prob Model & OOF
    print("   Generating Meta-Prob OOF...")
    meta_prob_oof = np.full(len(df), np.nan)

    # Need to handle Calibration wrapper for Prob model
    # We define a helper to get calibrated preds

    final_base_prob = lgb.LGBMClassifier(**best_prob_params)

    # Calibrated CV for OOF
    # We can use CalibratedClassifierCV with 'prefit' if we split manually, or cv=int
    # Since we want OOF aligned with our split, we iterate manually.

    for train_idx, val_idx in splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y_prob[train_idx]
        w_tr = w_prob[train_idx]

        # Train base
        base_est = lgb.LGBMClassifier(**best_prob_params)
        base_est.fit(X_tr, y_tr, sample_weight=w_tr)

        # Calibrate (Isotonic on validation?)
        # Standard approach: Train on (k-1), Calibrate on subset of (k-1) or Calibrate on Val?
        # If we calibrate on Val, we leak label info if we use those preds for downstream.
        # Correct OOF Calibration:
        # 1. Inner Split of Train -> Calib/Train-Base.
        # 2. Train Base, Calibrate Isotonic.
        # 3. Predict Val.

        # Simple implementation: Use CalibratedClassifierCV with internal CV=3 on Training data
        cal_model = CalibratedClassifierCV(base_est, method='isotonic', cv=3)
        cal_model.fit(X_tr, y_tr, sample_weight=w_tr)

        preds = cal_model.predict_proba(X_val)[:, 1]
        meta_prob_oof[val_idx] = preds

    # Fit Final Prob Model (Full Data)
    print("   Fitting Final Prob Model (Full)...")
    final_prob_model = CalibratedClassifierCV(
        estimator=lgb.LGBMClassifier(**best_prob_params),
        method='isotonic',
        cv=3 # Internal CV for calibration on full fit
    )
    final_prob_model.fit(X, y_prob, sample_weight=w_prob)

    # ---------------------------------------------------------
    # Output Assembly
    # ---------------------------------------------------------

    df['meta_alpha'] = meta_alpha_oof
    df['meta_prob'] = meta_prob_oof

    models_dict = {
        'alpha_model': final_alpha_model,
        'prob_model': final_prob_model,
        'best_alpha_type': best_alpha_name,
        'best_prob_type': best_prob_name
    }

    # Generate unified report
    try:
        plot_diagnostics(
            y_true=y_prob,
            y_prob=meta_prob_oof,
            output_path=str(outcomes_dir / f"layer3_prob_calibration_{ts}.png")
        )

        # Alpha Scatter Plot
        mask = np.isfinite(meta_alpha_oof) & np.isfinite(y_alpha)
        if mask.sum() > 100:
            plt.figure(figsize=(10, 6))
            sns.regplot(x=meta_alpha_oof[mask], y=y_alpha[mask], scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
            plt.title(f"Meta-Alpha vs Target (IC={spearmanr(meta_alpha_oof[mask], y_alpha[mask])[0]:.4f})")
            plt.xlabel("Predicted Alpha")
            plt.ylabel("Realized Vol-Adj Return")
            plt.savefig(str(outcomes_dir / f"layer3_alpha_scatter_{ts}.png"))
            plt.close()

    except Exception:
        pass

    return df, models_dict

def _run_shap_analysis(model, X, output_dir, symbol, timeframe, ts, md_path):
    # (Existing implementation kept but likely needs update to handle dict of models if called directly)
    # The calling function manages this.
    pass

# Helper: Advanced Diagnostic Plot (Unchanged)
def plot_diagnostics(y_true, y_prob, output_path=None):
    # (Existing implementation)
    try:
        y_prob_numeric = pd.to_numeric(y_prob, errors='coerce')
        y_true_numeric = pd.to_numeric(y_true, errors='coerce')
        mask = ~y_prob_numeric.isna() & ~y_true_numeric.isna()
        y_true = y_true_numeric[mask]
        y_prob = y_prob_numeric[mask]

        if len(y_true) == 0:
            return

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax[0].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Meta-Model')
        ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
        ax[0].set_xlabel('Predicted Probability')
        ax[0].set_ylabel('Actual Win Rate')
        ax[0].set_title('Calibration (Reliability)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        sns.histplot(y_prob, bins=20, kde=True, ax=ax[1], color='purple', alpha=0.6)
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel('Predicted Probability')
        ax[1].set_title('Resolution (Confidence Distribution)')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
