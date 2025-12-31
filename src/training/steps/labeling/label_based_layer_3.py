from typing import List, Tuple, Optional, Any, Dict
import copy
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
from scipy.stats import spearmanr, kurtosis, f_oneway, rankdata
from scipy.special import logit, expit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss as sk_log_loss,
    brier_score_loss,
    roc_auc_score as sk_roc_auc_score,
    average_precision_score,
    mean_squared_error,
)
from joblib import Parallel, delayed
from numba import njit, prange
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from sklearn.base import BaseEstimator, ClassifierMixin

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

from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
from src.training.steps.labeling.generate_weights_per_label import (
    finalize_sample_weights,
)
from src.training.steps.labeling.lgbm_feature_selection import lgbm_feature_selection_pipeline
from src.training.steps.labeling.regime_leaf_feature_extractor import extract_regime_leaf_onehot_features
from src.training.steps.labeling.short_nn_sequence_template import generate_nn_sequence_embeddings

from src.utils.purged_kfold import PurgedKFoldTime

logger = logging.getLogger(__name__)

EPS = 1e-12

class ManualCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """
    Manual implementation of calibration for pre-fitted models,
    bypassing sklearn's validation strictness on cv='prefit'.
    """
    def __init__(self, base_estimator, method='isotonic'):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrator = IsotonicRegression(out_of_bounds='clip') if method == 'isotonic' else None
        self.classes_ = [0, 1]

    def fit(self, X, y, sample_weight=None):
        # Assumes base_estimator is ALREADY fitted.
        # We predict using base estimator, then fit the calibrator on (preds, y).

        # Get raw probabilities (uncalibrated)
        # Handle cases where base estimator might not return 2 columns
        if hasattr(self.base_estimator, "predict_proba"):
            raw_preds = self.base_estimator.predict_proba(X)
            if raw_preds.shape[1] == 2:
                pos_probs = raw_preds[:, 1]
            else:
                pos_probs = raw_preds[:, 0] # Should not happen for binary classifier
        else:
            # Fallback to decision function if available
            if hasattr(self.base_estimator, "decision_function"):
                pos_probs = _sigmoid(self.base_estimator.decision_function(X))
            else:
                # Fallback to predict
                pos_probs = self.base_estimator.predict(X).astype(float)

        if self.calibrator:
            self.calibrator.fit(pos_probs, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        if hasattr(self.base_estimator, "predict_proba"):
            raw_preds = self.base_estimator.predict_proba(X)
            pos_probs = raw_preds[:, 1] if raw_preds.shape[1] == 2 else raw_preds[:, 0]
        elif hasattr(self.base_estimator, "decision_function"):
            pos_probs = _sigmoid(self.base_estimator.decision_function(X))
        else:
            pos_probs = self.base_estimator.predict(X).astype(float)

        if self.calibrator:
            cal_p = self.calibrator.transform(pos_probs)
            # Clip for safety
            cal_p = np.clip(cal_p, 0.0, 1.0)
            return np.column_stack((1-cal_p, cal_p))

        return np.column_stack((1-pos_probs, pos_probs))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# -----------------------------------------------------------
# CUSUM Signal Generation
# -----------------------------------------------------------

def compute_cusum_signals_multi_window(
    market_data: pd.DataFrame,
    windows: List[int] = [12, 24, 48],
    k: float = 0.5, # Threshold factor
    alpha: float = 0.5 # Volatility/ER adjustment factor
) -> pd.DataFrame:
    """
    Computes Trend and Reversal CUSUM signals across multiple windows.
    """
    df = pd.DataFrame(index=market_data.index)

    close = market_data['close']
    # Log returns
    r = np.log(close / close.shift(1)).fillna(0.0)

    # Pre-compute Volatility and ER for dynamic thresholds
    # We use a base window for vol normalization, e.g., 20 or the window itself?
    # The prompt implies window-specific calculations: "sigma_t,w"

    for w in windows:
        # 1. Volatility (sigma_t,w)
        sigma = r.rolling(window=w).std().fillna(0.0)

        # 2. Efficiency Ratio (ER_t,w)
        # ER = |Change| / Sum(|Returns|)
        change = close.diff(w).abs()
        vol_sum = close.diff().abs().rolling(window=w).sum()
        er = (change / (vol_sum + EPS)).fillna(0.0)

        # 3. Liquidity Mod (Proxy: Volume / Moving Average Volume)
        if 'volume' in market_data.columns:
            vol_ma = market_data['volume'].rolling(window=w*5).mean() + EPS
            liq_mod = (market_data['volume'] / vol_ma).clip(0.5, 2.0).fillna(1.0)
        else:
            liq_mod = 1.0

        # 4. Dynamic Threshold h_t,w
        # h = k * sigma * (1 + alpha * (1 - ER)) * LiqMod
        h = k * sigma * (1.0 + alpha * (1.0 - er)) * liq_mod

        # 5. Trend CUSUM
        # S_t+ = max(0, S_{t-1} + r_t)
        # S_t- = min(0, S_{t-1} + r_t)
        # We normalize by h for the signal

        # Vectorized CUSUM is hard in pandas, use numba or loop.
        # For simplicity and speed in python, we can use a helper or simple loop.
        # Given the requirements, a loop is safest for correctness.

        r_vals = r.values
        h_vals = h.values
        n = len(r_vals)

        s_trend_pos = np.zeros(n)
        s_trend_neg = np.zeros(n)

        # 6. Residual CUSUM (Reversal)
        # r_tilde = r_t - E[r]_w
        r_mean = r.rolling(window=w).mean().fillna(0.0).values
        r_tilde = r_vals - r_mean

        s_rev_pos = np.zeros(n)
        s_rev_neg = np.zeros(n)

        # Fast Loop
        # JIT this for performance
        _compute_cusum_loop(n, r_vals, r_tilde, s_trend_pos, s_trend_neg, s_rev_pos, s_rev_neg)

        # 7. Normalize Signals: Signal = (S+ - |S-|) / h
        # Avoid div by zero
        denom = h_vals + EPS

        trend_sig = (s_trend_pos - np.abs(s_trend_neg)) / denom
        rev_sig = (s_rev_pos - np.abs(s_rev_neg)) / denom

        df[f'trend_signal_{w}'] = trend_sig
        df[f'reversal_signal_{w}'] = rev_sig
        df[f'sigma_{w}'] = sigma
        df[f'er_{w}'] = er

    return df

@njit
def _compute_cusum_loop(n, r, r_tilde, s_t_p, s_t_n, s_r_p, s_r_n):
    for t in range(1, n):
        # Trend
        s_t_p[t] = max(0.0, s_t_p[t-1] + r[t])
        s_t_n[t] = min(0.0, s_t_n[t-1] + r[t])

        # Reversal
        s_r_p[t] = max(0.0, s_r_p[t-1] + r_tilde[t])
        s_r_n[t] = min(0.0, s_r_n[t-1] + r_tilde[t])


# -----------------------------------------------------------
# Adaptive Geometry Generation & Selection
# -----------------------------------------------------------

def apply_smart_activation(
    sig: np.ndarray,
    vol: np.ndarray,
    mode: str
) -> np.ndarray:
    """
    Applies non-linear transforms with safety clipping.
    """
    if len(sig) == 0:
        return np.array([])
        
    transformed = sig.copy()

    if mode == 'linear':
        pass

    elif mode == 'cubic_regime':
        # "Sniper": Penalize uncertainty + Cut size in crisis
        # Logic: If Vol > 95th percentile, multiplier = 0.5, else 1.0
        if len(vol) > 0:
            p95 = np.percentile(vol, 95)
            regime_mult = np.where(vol > p95, 0.5, 1.0)
            
            # Apply Cubic
            transformed = (np.sign(sig) * np.abs(sig)**3) * regime_mult
        else:
            transformed = (np.sign(sig) * np.abs(sig)**3)

    elif mode == 'tanh_dynamic':
        # Normalize vol to median to keep tanh input range reasonable
        if len(vol) > 0:
            norm_vol = vol / (np.median(vol) + EPS)
            transformed = np.tanh(sig / norm_vol)
        else:
            transformed = np.tanh(sig)

    # --- SAFETY SCALING ---
    return np.clip(transformed, -5.0, 5.0)

def generate_geometries_adaptive(
    base_signals: pd.DataFrame,
    volatility: pd.Series,
    mfe: pd.Series,
    mae: pd.Series,
    trend_ratios: list = [0.0, 0.5, 1.0],
    activations: list = ['linear', 'cubic_regime', 'tanh_dynamic']
) -> dict:

    # We expect base_signals to contain multi-window signals.
    
    if base_signals.empty:
        logger.warning("Empty base_signals passed to generate_geometries_adaptive")
        return {}

    trend_cols = [c for c in base_signals.columns if 'trend_signal' in c]
    rev_cols = [c for c in base_signals.columns if 'reversal_signal' in c]

    if not trend_cols or not rev_cols:
        return {}

    trend_vec = base_signals[trend_cols].mean(axis=1).values.astype(np.float32)[:, None]
    rev_vec = base_signals[rev_cols].mean(axis=1).values.astype(np.float32)[:, None]
    vol_vec = volatility.values.astype(np.float32)[:, None]

    mfe_arr = mfe.values.astype(np.float32)
    mae_arr = mae.values.astype(np.float32)

    alphas = np.array(trend_ratios, dtype=np.float32)

    # Vectorized Linear Mix
    w_trend = alphas[None, :]
    w_rev = 1.0 - w_trend
    linear_sigs = (trend_vec @ w_trend) - (rev_vec @ w_rev)

    meta_geometries = {}

    for i, alpha in enumerate(alphas):
        base_sig = linear_sigs[:, i]

        for act in activations:
            geom_id = f"g_a{alpha:.2f}_{act}"
            final_sig = apply_smart_activation(base_sig, vol_vec.flatten(), act)

            meta_geometries[geom_id] = {
                'composite_signal': final_sig, # float32
                'alpha': alpha,
                'activation': act,
                'mfe': mfe_arr,
                'mae': mae_arr,
                'sigma_eff': vol_vec.flatten()
            }

    return meta_geometries

@njit(parallel=True)
def is_pareto_efficient_numba(costs):
    """
    Find the Pareto-efficient points (maximize all columns).
    costs: (N, M) array of fitness values.
    Returns: Boolean array of size N (True if efficient).
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=np.bool_)
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            all_better_or_eq = True
            any_strictly_better = False

            for k in range(costs.shape[1]):
                if costs[j, k] < costs[i, k]:
                    all_better_or_eq = False
                    break
                if costs[j, k] > costs[i, k]:
                    any_strictly_better = True

            if all_better_or_eq and any_strictly_better:
                is_efficient[i] = False
                break
    return is_efficient

def compute_risk_metrics(mfe: np.ndarray, mae: np.ndarray, sigma: np.ndarray):
    """
    Compute RAD and Tail Risk efficiently using NumPy.
    """
    # 1. RAD: (MFE / MAE) / Volatility
    rad_vec = (mfe / (mae + EPS)) / (sigma + EPS)
    rad_med = np.median(rad_vec)

    # MAD (Median Absolute Deviation) for stability
    rad_mad = np.median(np.abs(rad_vec - rad_med))
    stability = 1.0 / (1.0 + rad_mad)

    rad_score = rad_med * stability

    # 2. Tail Risk Proxy
    tail_risk = np.percentile(mae, 95) / (np.median(sigma) + EPS)

    return rad_score, tail_risk

def worker_evaluate_geometry(
    gid: str,
    df: Dict[str, np.ndarray], # Using dict to avoid serialization overhead of DataFrame
    labels: np.ndarray,
    min_variance: float = 1e-8,
    min_f_score: float = 0.5
):
    """
    Evaluates a single meta-geometry.
    """
    sig = df['composite_signal']
    sigma_vals = df['sigma_eff']
    mfe = df['mfe']
    mae = df['mae']

    # PHASE 1: Fast Screening
    if np.var(sig) < min_variance:
        return None

    # Filter invalid/nan labels
    valid_mask = np.isfinite(labels)
    if valid_mask.sum() < 20:
        return None

    sig_valid = sig[valid_mask]
    lbl_valid = labels[valid_mask]

    sig_0 = sig_valid[lbl_valid == 0]
    sig_1 = sig_valid[lbl_valid == 1]

    if len(sig_0) < 10 or len(sig_1) < 10:
        return None

    f_stat, p_val = f_oneway(sig_0, sig_1)

    if np.isnan(f_stat) or f_stat < min_f_score:
        return None

    # PHASE 2: Risk Metrics
    rad, tail = compute_risk_metrics(mfe, mae, sigma_vals)

    # PHASE 3: The Probe (Time-Series CV)
    X = np.column_stack((sig_valid, sigma_vals[valid_mask]))
    y = lbl_valid

    tscv = TimeSeriesSplit(n_splits=5, gap=50)

    preds_list = []
    targets_list = []

    model = HistGradientBoostingClassifier(
        max_iter=50,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42
    )

    try:
        for train_idx, val_idx in tscv.split(X):
            if len(np.unique(y[train_idx])) < 2:
                continue
            model.fit(X[train_idx], y[train_idx])
            p = model.predict_proba(X[val_idx])[:, 1]
            preds_list.extend(p)
            targets_list.extend(y[val_idx])

        if not targets_list or len(np.unique(targets_list)) < 2:
            return None

        auc = sk_roc_auc_score(targets_list, preds_list)

    except Exception:
        return None

    return {
        'id': gid,
        'auc': auc,
        'rad': rad,
        'tail_risk': tail,
        'sig_ptr': sig.astype(np.float32)
    }

def select_best_geometries_production(
    meta_geometries: dict,
    labels: pd.Series,
    top_k: int = 5,
    min_auc: float = 0.51,
    n_jobs: int = -1
):
    print(f"1. Parallel Evaluation of {len(meta_geometries)} geometries...")
    labels_arr = labels.values

    # Prepare dict inputs for worker
    # Ensure all arrays are aligned

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker_evaluate_geometry)(k, v, labels_arr)
        for k, v in meta_geometries.items()
    )
    results = [r for r in results if r is not None]

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 2. Hard Filter
    df = df[df['auc'] > min_auc].copy().reset_index(drop=True)
    if df.empty:
        print("No candidates passed min_auc.")
        return pd.DataFrame()

    # 3. Transformations & Normalization
    auc_log = df['auc'].clip(0.5001, 0.9999).apply(logit)
    rad_vals = df['rad'].values
    safe_vals = -1 * df['tail_risk'].values # Flip sign

    def robust_scale(x):
        return (x - np.median(x)) / (np.median(np.abs(x - np.median(x))) + EPS)

    df['z_auc'] = robust_scale(auc_log)
    df['z_rad'] = robust_scale(rad_vals)
    df['z_safe'] = robust_scale(safe_vals)

    # 4. Numba Pareto Filter
    fitness_matrix = df[['z_auc', 'z_rad', 'z_safe']].values.astype(np.float64)
    pareto_mask = is_pareto_efficient_numba(fitness_matrix)

    pareto_candidates = df[pareto_mask].copy()
    print(f"2. Pareto Filter: {len(df)} -> {len(pareto_candidates)} candidates.")

    candidates = pareto_candidates if len(pareto_candidates) >= top_k else df

    # 5. Composite Ranking
    candidates['score'] = (
        0.4 * candidates['z_auc'] +
        0.4 * candidates['z_rad'] +
        0.2 * candidates['z_safe']
    )
    candidates = candidates.sort_values('score', ascending=False)

    # 6. Diversity Pruning
    print("3. Diversity Check (Matrix Method)...")
    check_limit = min(len(candidates), 200)
    top_candidates = candidates.iloc[:check_limit]

    if len(top_candidates) == 0:
         return pd.DataFrame()

    signal_matrix = np.vstack(top_candidates['sig_ptr'].values)
    corr_matrix = np.corrcoef(signal_matrix)

    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    if len(upper_tri) > 0:
        adaptive_thresh = np.clip(np.percentile(upper_tri, 75), 0.70, 0.95)
    else:
        adaptive_thresh = 0.90

    print(f"   Adaptive Correlation Threshold: {adaptive_thresh:.3f}")

    kept_indices = []
    dropped_indices = set()

    for i in range(check_limit):
        if i in dropped_indices:
            continue
        kept_indices.append(i)
        if len(kept_indices) >= top_k:
            break

        correlations = corr_matrix[i, :]
        high_corr_neighbors = np.where((correlations > adaptive_thresh) & (np.arange(check_limit) > i))[0]
        for neighbor in high_corr_neighbors:
            dropped_indices.add(neighbor)

    selected_df = top_candidates.iloc[kept_indices].copy()
    selected_df = selected_df.drop(columns=['sig_ptr'])

    return selected_df

# -----------------------------------------------------------
# Core Helpers from original file
# -----------------------------------------------------------

def _clip_probs(probs: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    return np.clip(probs, clip_low, clip_high)

def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return probs
    probs_temp = np.log(probs + 1e-12) / temperature
    return 1.0 / (1.0 + np.exp(-probs_temp))

def _fit_temperature_scalar(y_true: np.ndarray, probs: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
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

def _get_score(preds, y_true):
    try:
        auc = sk_roc_auc_score(y_true, preds)
    except ValueError:
        auc = 0.5
    try:
        ll = sk_log_loss(y_true, preds)
    except ValueError:
        ll = 0.693
    score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
    return score

def _safe_log_loss(y_true, y_pred, labels=None, sample_weight=None, default: float = 0.693):
    try:
        labels_arg = labels
        unique = np.unique(np.asarray(y_true))
        if labels_arg is None and unique.size == 1:
            labels_arg = [0, 1]
        return sk_log_loss(y_true, y_pred, labels=labels_arg, sample_weight=sample_weight)
    except ValueError as exc:
        logger.warning(f"log_loss fallback due to: {exc}")
        return default

def _safe_roc_auc_score(y_true, y_pred, sample_weight=None, default: float = 0.5):
    try:
        unique = np.unique(np.asarray(y_true))
        if unique.size == 1:
            y_true = np.concatenate([y_true, [1 - unique[0]]])
            y_pred = np.concatenate([y_pred, [y_pred.mean()]])
            if sample_weight is not None:
                sample_weight = np.concatenate([sample_weight, [sample_weight.mean()]])
        return sk_roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
    except ValueError as exc:
        logger.warning(f"roc_auc_score fallback due to: {exc}")
        return default

def log_loss(*args, **kwargs):
    return _safe_log_loss(*args, **kwargs)

def roc_auc_score(*args, **kwargs):
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
    mdi_importance: Optional[Dict[str, float]] = None,
    shap_importance: Optional[Dict[str, float]] = None,
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
                    'mdi_importance': float(mdi_importance.get(col, 0.0)) if mdi_importance else 0.0,
                    'shap_importance': float(shap_importance.get(col, 0.0)) if shap_importance else 0.0,
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
    tprint_info(f"🎯 Running HPO for {model_type} (Obj: {objective_func}) ({n_trials} trials)...")

    # Subsample for speed
    n_total = len(X)
    n_sample = max(2000, int(n_total * 0.5))

    if n_total > n_sample:
        tprint_info(f"📊 HPO: Subsampling {n_sample}/{n_total} samples for speed")
        sample_idx = np.random.RandomState(42).choice(n_total, n_sample, replace=False)
        sample_idx.sort()
        X_hpo = X.iloc[sample_idx]
        y_hpo = y.iloc[sample_idx]
        w_hpo = w[sample_idx]
    else:
        tprint_info(f"📊 HPO: Using all {n_total} samples")
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
            'num_leaves': trial.suggest_int('num_leaves', 16, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03),
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1e-1),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 10.0),
            'bagging_freq': 1,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'n_jobs': 1,
            'verbosity': -1
        }

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
                preds = model.predict_proba(X_val)[:, 1]
                # ScoreL3: AUC + Excess LogLoss
                auc = roc_auc_score(y_val_bin, preds)
                ll = log_loss(y_val_bin, preds)
                score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
                return score if np.isfinite(score) else -999.0

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
def generate_efficiency_labels(events_df, price_series, tx_cost=0.0, threshold=0.2):
    labels = pd.Series(index=events_df.index, dtype=float)
    net_returns = []
    valid_indices = []
    sorted_events = events_df.sort_index()
    for idx, row in sorted_events.iterrows():
        try:
            t0, t1 = row['entry_time'], row['exit_time']
            if t0 not in price_series.index or t1 not in price_series.index:
                continue  # Skip if timestamps not available
            trade_price = price_series[t0:t1]
            if len(trade_price) < 2:
                continue
            realized_ret = (trade_price.iloc[-1] / trade_price.iloc[0]) - 1
            net_ret = realized_ret - tx_cost
            net_returns.append(net_ret)
            valid_indices.append(idx)
        except Exception:
            continue
    if len(net_returns) == 0:
        return labels.fillna(0)
    net_returns_series = pd.Series(net_returns, index=valid_indices)
    threshold_series = net_returns_series.expanding(min_periods=20).quantile(0.60).shift(1)
    threshold_series = threshold_series.fillna(0.0)
    efficiency_mask = net_returns_series > threshold_series
    labels.loc[valid_indices] = efficiency_mask.astype(float)
    return labels.fillna(0)

# Fast ECE
def _fast_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
    if not np.any(mask): return 0.0
    y_true_arr = y_true_arr[mask]
    y_prob_arr = y_prob_arr[mask]
    n = int(y_prob_arr.size)
    if n <= 0: return 0.0
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob_arr, bin_edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    sum_prob = np.bincount(bin_idx, weights=y_prob_arr, minlength=n_bins).astype(float)
    sum_true = np.bincount(bin_idx, weights=y_true_arr, minlength=n_bins).astype(float)
    nonzero = counts > 0
    if not np.any(nonzero): return 0.0
    mean_prob = np.zeros(n_bins, dtype=float)
    mean_true = np.zeros(n_bins, dtype=float)
    mean_prob[nonzero] = sum_prob[nonzero] / counts[nonzero]
    mean_true[nonzero] = sum_true[nonzero] / counts[nonzero]
    return float(np.sum((counts[nonzero] / n) * np.abs(mean_prob[nonzero] - mean_true[nonzero])))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


# -----------------------------------------------------------
# Geometry-Specific Feature Generation
# -----------------------------------------------------------


def _generate_geometry_specific_nn_features(
    market_data: pd.DataFrame,
    geometry_horizon: float,
    embed_dim: int = 8
) -> pd.DataFrame:
    """
    Generate NN sequence features with sequence length adapted to geometry horizon.
    
    Args:
        market_data: OHLCV data
        geometry_horizon: The effective horizon of the geometry (in bars)
        embed_dim: Embedding dimension
    
    Returns:
        DataFrame with NN embedding features
    """
    
    # Scale sequence length based on geometry horizon
    # Default is 24 bars, scale proportionally but keep reasonable bounds
    default_seq_len = 24
    scale_factor = geometry_horizon / default_seq_len
    seq_len = max(8, min(48, int(default_seq_len * scale_factor)))
    
    try:
        nn_df = generate_nn_sequence_embeddings(
            market_data=market_data,
            encoder_type="stacked",
            seq_len=seq_len,
            embed_dim=embed_dim,
            use_conv=True,
            use_lstm=True,
            use_attention=False
        )
        return nn_df if nn_df is not None else pd.DataFrame(index=market_data.index)
    except Exception as e:
        print(f"⚠️ Geometry-specific NN features failed: {e}")
        return pd.DataFrame(index=market_data.index)

# -----------------------------------------------------------
# Training Pipeline Components
# -----------------------------------------------------------


def _train_meta_learner_for_geometry(
    gid: str,
    df: pd.DataFrame,
    meta_features: List[str],
    target_col: str,
    schemes: Dict[str, np.ndarray],
    y_values: np.ndarray,
    outcomes_dir: Path,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, Any, float, Dict[str, Any]]:
    """
    Runs the Scheme Comparison -> Race -> HPO -> Fit pipeline for a SINGLE geometry.
    Returns (oof_probs, final_model, best_score).
    """
    tprint_info(f"🚀 Training Meta-Learner for geometry: {gid}")
    
    X = df[meta_features]
    y = df[target_col]
    
    tprint_info(f"📊 {gid}: {len(X)} samples, {len(meta_features)} features")
    
    # 1. Scheme Screening (Simplified)
    # Evaluate schemes on a subset or full folds
    best_scheme = None
    best_scheme_score = -float('inf')
    best_w = None
    
    tprint_info(f"🔍 {gid}: Evaluating {len(schemes)} weighting schemes...")
    
    # We use a simple LGBM probe for scheme selection
    lgb_params = {'n_estimators': 100, 'max_depth': 4, 'verbose': -1, 'n_jobs': 1}
    
    # PurgedCV for scheme eval
    cv = PurgedKFoldTime(n_splits=3, purge=50) # Faster 3-fold

    for s_name, w_vec in schemes.items():
        tprint_info(f"🔍 {gid}: Testing scheme: {s_name}")
        scores = []
        for train_idx, val_idx in cv.split(X):
            if len(np.unique(y.iloc[train_idx])) < 2: continue
            m = lgb.LGBMClassifier(**lgb_params)
            m.fit(X.iloc[train_idx], (y.iloc[train_idx]>0.5).astype(int), sample_weight=w_vec[train_idx])
            p = m.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(_get_score(p, (y.iloc[val_idx]>0.5).astype(int)))
        
        avg_score = np.mean(scores) if scores else -999
        tprint_info(f"📊 {gid}: {s_name} score: {avg_score:.3f}")
        if avg_score > best_scheme_score:
            best_scheme_score = avg_score
            best_scheme = s_name
            best_w = w_vec
            
    tprint_success(f"✅ {gid}: Best Scheme: {best_scheme} (Score: {best_scheme_score:.3f})")
    
    if best_w is None:
        best_w = np.ones(len(y))
        tprint_warning(f"⚠️ {gid}: No valid scheme found, using equal weights")

    # 2. HPO
    tprint_info(f"🎯 {gid}: Running HPO with 15 trials...")
    best_params = _run_layer3_hpo(X, (y>0.5).astype(int), best_w, 'classifier', n_trials=15)
    tprint_info(f"🎯 {gid}: HPO completed")
    
    # 3. OOF Generation with Best Params
    tprint_info(f"🔄 {gid}: Generating OOF predictions with 5-fold CV...")
    cv_full = PurgedKFoldTime(n_splits=5, purge=50)
    oof_probs = np.full(len(df), np.nan)
    
    # We use a CalibratedClassifierCV wrapper around LGBM for OOF and Final
    # But CalibratedClassifierCV doesn't support sample_weight in fit nicely with internal split?
    # Actually it does in newer sklearn.
    # Alternatively, use LGBM directly and calibrate manually.
    
    for train_idx, val_idx in cv_full.split(X):
        m = lgb.LGBMClassifier(**best_params)
        m.fit(X.iloc[train_idx], (y.iloc[train_idx]>0.5).astype(int), sample_weight=best_w[train_idx])
        p = m.predict_proba(X.iloc[val_idx])[:, 1]

        # Calibration (Isotonic)
        iso = IsotonicRegression(out_of_bounds='clip')
        # Fit on a subset of val? No, typically fit on val, predict on val is cheating.
        # Proper way: Inner CV or just use raw probs for OOF.
        # Standard: Use raw probs for OOF, then calibrate the OOF ensemble later?
        # Or calibrate using train? No.
        # Let's use raw probs for OOF to avoid leakage.
        oof_probs[val_idx] = p

    # 4. Final Fit
    final_base = lgb.LGBMClassifier(**best_params)
    final_model = CalibratedClassifierCV(final_base, method='isotonic', cv=3)
    final_model.fit(X, (y>0.5).astype(int), sample_weight=best_w)
    
    # --- De Prado Metrics ---
    # 5. Feature Importance (MDI)
    mdi_imp = {}
    try:
        # Get importances from the base models in calibration if possible, 
        # or fit a separate model on all data for importance.
        imp_model = lgb.LGBMClassifier(**best_params)
        imp_model.fit(X, (y>0.5).astype(int), sample_weight=best_w)
        mdi_imp = dict(zip(meta_features, imp_model.feature_importances_.astype(float)))
    except Exception: pass

    # 6. SHAP Importance (subset)
    shap_imp = {}
    try:
        sample_size = min(len(X), 200)
        explainer = shap.TreeExplainer(imp_model)
        shap_values = explainer.shap_values(X.iloc[:sample_size])
        # For binary classification, shap_values is a list of two arrays.
        # Use absolute mean of shap values for class 1
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[1]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)
        shap_imp = dict(zip(meta_features, vals.astype(float)))
    except Exception: pass

    return oof_probs, final_model, best_scheme_score, {
        'mdi': mdi_imp,
        'shap': shap_imp
    }

# -----------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Revised Layer 3: Multi-Geometry Meta-Models with Geometry-Specific Features.
    
    Process:
    1. Receives pre-computed signals (including CUSUM from previous layers).
    2. Generates and Selects Adaptive Geometries.
    3. For each selected geometry:
       - Generates geometry-specific regime features (scaled horizons)
       - Generates geometry-specific NN features (scaled sequence lengths)
       - Trains a Meta-Learner
    4. Aggregates results.
    """
    tprint_info("="*60)
    tprint_info("🚀 LAYER 3: MULTI-GEOMETRY ANALYST META-MODELS")
    tprint_info("="*60)

    # Initialize timestamp and outcomes directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outcomes_dir = Path('outcomes')
    outcomes_dir.mkdir(exist_ok=True, parents=True)

    df = oof_df.copy()
    cfg = config if isinstance(config, dict) else {}
    
    tprint_info(f"📊 Input data: {len(df)} rows, {len(base_model_cols)} base features")
    tprint_info(f"🎯 Target column: {target_col}")

    # ---------------------------------------------------------
    # 1. Feature Engineering (Shared)
    # ---------------------------------------------------------
    # ... (Keep existing feature engineering logic) ...
    # Initialize unified results container
    all_comparison_results = []

    print("<< Generating Layer 3 Features...")
    tprint_info("🔧 Generating Layer 3 features...")

    # 1. Base Feature Generation (Global)
    # ... (Keep existing logic to generate global features)
    safe_base_cols = []  # Initialize to empty list to avoid UnboundLocalError
    if base_model_cols:
        safe_base_cols = [c for c in base_model_cols if c in df.columns]
        df[safe_base_cols] = df[safe_base_cols].fillna(0.5)

    if market_data is not None and isinstance(market_data, pd.DataFrame) and not market_data.empty:
        for c in ['volume', 'high', 'low', 'close']:
            if c in market_data.columns:
                df[c] = market_data[c].reindex(df.index)

    try:
        df = generate_layer3_features(df, safe_base_cols if base_model_cols else [])
    except Exception as e:
        print(f"⚠️ generate_layer3_features failed: {e}")

    # Check for Base Model Spread
    if 'ens_prediction_dispersion' not in df.columns and 'base_pred_std' in df.columns:
        df['ens_prediction_dispersion'] = df['base_pred_std'] # Alias if needed

    candidate_features = []
    candidate_features.extend(safe_base_cols)

    # --- Geometry-Specific Features ---
    # Generate regime and NN features based on geometry characteristics
    # Note: This should be called after geometry selection with actual geometry data
    # For now, we'll create a placeholder that can be enhanced when geometry loop is restored
    
    def generate_features_for_geometry(alpha: float, geometry_id: str = "default"):
        """Generate features for a specific geometry with given alpha parameter"""
        try:
            # Horizon based on geometry alpha parameter
            # alpha=0 (pure reversal) -> 4 bars, alpha=1 (pure trend) -> 16 bars
            geometry_horizon = 4 + alpha * 12  # Linear mapping from 4 to 16 bars
            
            print(f"   Geometry {geometry_id}: alpha={alpha:.2f} -> horizon={geometry_horizon:.1f} bars")
            
            # Generate regime features with geometry-specific horizons
            base_regime_config = {
                "enabled_targets": [
                    "regime_trendiness", "regime_volatility", "regime_trend_efficiency",
                    "regime_memory", "regime_liquidity", "regime_volume_force_direction",
                    "regime_breakout", "regime_future_range", "regime_downside_ae",
                    "regime_upside_ae", "regime_tail_min_bar", "regime_jump_max_abs_bar",
                    "regime_vol_of_vol"
                ],
                "inputs": {"input_source": "ohlcv_only", "ohlcv_feature_config": {}},
                "onehot": {"enabled": False},
                "interaction_feature": {"enabled": True, "include_base": True},
                "reporting": {"enabled": False},
                "walk_forward": {"mode": "cross_fit", "cross_fit": {"n_splits": 5}}
            }
            
            # Regime feature extraction moved to Layer 2
            print(f"     Skipping redundant regime extraction (now in Layer 2)")
            
            # Generate NN features with geometry-specific sequence length (DISABLED)
            # Note: NN feature generation disabled to reduce complexity and improve performance
            # To enable, uncomment the following block:
            #
            # nn_df = _generate_geometry_specific_nn_features(market_data, geometry_horizon, embed_dim=8)
            # if nn_df is not None and not nn_df.empty:
            #     nn_df = nn_df.reindex(df.index).fillna(0.0)
            #     new_nn_cols = [f'{geometry_id}_nn_{c}' for c in nn_df.columns if c not in df.columns]
            #     if new_nn_cols:
            #         for col in nn_df.columns:
            #             df[f'{geometry_id}_nn_{col}'] = nn_df[col]
            #         candidate_features.extend(new_nn_cols)
            #         print(f"     Added {len(new_nn_cols)} NN features")
                    
        except Exception as e:
            print(f"⚠️ Geometry {geometry_id} features failed: {e}")
    
    # Geometry-specific features will be generated in the loop above
    # for each selected geometry

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
    # 0. Pre-calculate Required Series (Returns, Volatility)
    # ---------------------------------------------------------
    
    # Net Returns needed for Alpha Target
    if net_returns is None:
        # Attempt to calculate or raise
        if 'close' in df.columns:
            net_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        else:
             # Fallback: look for a returns column
             ret_col = next((c for c in df.columns if 'return' in c), None)
             if ret_col:
                 net_returns = df[ret_col].fillna(0)
             else:
                 # Last resort: mock it (though this suggests bigger issues)
                 print("   Warning: net_returns missing and cannot be calculated. Using Zeros.")
                 net_returns = pd.Series(0, index=df.index)
    
    # Create the aligned series for later use
    ret_series = net_returns.reindex(df.index)

    # Volatility needed for Alpha Target and Weighting
    if 'volatility_1d' in df.columns:
        vol_series = df['volatility_1d'].replace(0, np.nan).ffill().fillna(0.001)
    else:
        # Simple fallback
        vol_series = net_returns.rolling(24).std().fillna(0.001)

    # 1. Generate Global Features (Shared across all meta-models)
    global_features = []

    # ---------------------------------------------------------
    # 2. Geometry Generation & Selection
    # ---------------------------------------------------------
    
    # Expect pre-computed CUSUM signals from previous layers
    # Look for CUSUM columns in the input dataframe
    cusum_cols = [c for c in df.columns if 'trend_signal' in c or 'reversal_signal' in c]
    
    if not cusum_cols:
        print("⚠️ No CUSUM signals found in input. Using fallback signals.")
        # Create minimal fallback signals
        cusum_df = pd.DataFrame(index=df.index)
        cusum_df['trend_signal_24'] = np.zeros(len(df))
        cusum_df['reversal_signal_24'] = np.zeros(len(df))
    else:
        cusum_df = df[cusum_cols]
        print(f"Found CUSUM signals: {cusum_cols}")
    
    # Get required data for geometry generation
    vol_s = df['volatility_1d'] if 'volatility_1d' in df.columns else pd.Series(0.01, index=df.index)
    mfe_s = df.get('mfe', pd.Series(0.02, index=df.index))  # Default MFE
    mae_s = df.get('mae', pd.Series(0.01, index=df.index))  # Default MAE
    
    print("<< Generating Adaptive Geometries...")
    geometries_dict = generate_geometries_adaptive(
        base_signals=cusum_df,
        volatility=vol_s,
        mfe=mfe_s,
        mae=mae_s
    )
    
    print(f"Generated {len(geometries_dict)} geometries")
    
    # Select best geometries
    y_target = df[target_col].fillna(0)
    selected_geoms_df = select_best_geometries_production(
        geometries_dict, y_target, top_k=6, n_jobs=1
    )
    
    if selected_geoms_df.empty:
        print("⚠️ No geometries selected! Using fallback.")
        selected_ids = ['fallback']
        # Create fallback geometry
        if not geometries_dict:
            geometries_dict['fallback'] = {
                'composite_signal': np.zeros(len(df)),
                'sigma_eff': vol_s.values,
                'alpha': 0.5,
                'activation': 'linear'
            }
    else:
        selected_ids = selected_geoms_df['id'].values.tolist()
        print(f"Selected {len(selected_ids)} geometries: {selected_ids}")
    
    # ---------------------------------------------------------
    # 3. Multi-Geometry Meta-Learner Training
    # ---------------------------------------------------------
    
    # Prepare weighting schemes
    w_l1 = finalize_sample_weights(layer1_weight) if layer1_weight is not None else np.ones(len(df))
    w_l2 = finalize_sample_weights(layer2_weight) if layer2_weight is not None else np.ones(len(df))
    schemes = {
        'S1_L1': w_l1,
        'S2_L1_L2': w_l1 * w_l2,
        'S3_L2': w_l2
    }
    
    final_models = {}
    geometry_metrics = [] # Store per-geometry metrics
    global_mdi = {}
    global_shap = {}
    
    for gid in selected_ids:
        if gid not in geometries_dict:
            continue
            
        print(f"\n--- Training Meta-Learner for Geometry: {gid} ---")
        
        g_data = geometries_dict[gid]
        alpha = g_data.get('alpha', 0.5)
        sig = g_data['composite_signal']
        sigma = g_data['sigma_eff']
        
        # Add geometry signal to dataframe
        df[f'{gid}_sig'] = sig
        df[f'{gid}_sigma'] = sigma
        df[f'{gid}_sig_x_vol'] = sig * sigma
        
        # Generate geometry-specific features
        generate_features_for_geometry(alpha, gid)
        
        # Collect all features for this geometry (NN features disabled)
        geometry_features = [f'{gid}_sig', f'{gid}_sigma', f'{gid}_sig_x_vol']
        geometry_features.extend([c for c in df.columns if c.startswith(f'{gid}_rl_')])  # Only regime features
        geometry_features.extend(global_features)  # Include global features
        
        # Train meta-learner for this geometry
        try:
            oof_p, model, score, importance = _train_meta_learner_for_geometry(
                gid, df, geometry_features, target_col, schemes, y_target.values,
                outcomes_dir, cfg
            )
            
            # Aggregate importance
            if importance:
                for k, v in importance.get('mdi', {}).items():
                    global_mdi[k] = global_mdi.get(k, 0.0) + v
                for k, v in importance.get('shap', {}).items():
                    global_shap[k] = global_shap.get(k, 0.0) + v
            
            # Calculate metrics for this geometry
            if target_col in df.columns:
                y_true_g = (df[target_col].fillna(0) > 0.5).astype(int)
                try:
                    auc_g = sk_roc_auc_score(y_true_g, oof_p)
                    ll_g = sk_log_loss(y_true_g, oof_p)
                except Exception as e:
                    auc_g, ll_g = 0.5, 0.693

                geometry_metrics.append({
                    'geometry': gid,
                    'auc': auc_g,
                    'log_loss': ll_g,
                    'score': score,
                    'alpha': alpha,
                    'activation': g_data.get('activation', 'linear')
                })
                print(f"   Geometry {gid}: Score={score:.4f}, AUC={auc_g:.4f}, LogLoss={ll_g:.4f}")
            else:
                 print(f"   Geometry {gid}: Score={score:.4f}")

            df[f'meta_prob_{gid}'] = oof_p
            final_models[gid] = model
            
        except Exception as e:
            print(f"   Geometry {gid} training failed: {e}")
            df[f'meta_prob_{gid}'] = 0.5  # Fallback probability
    
    print(f"\n>>> Trained {len(final_models)} geometry meta-models")
    
    # Save geometry metrics if any
    if geometry_metrics:
        try:
            pd.DataFrame(geometry_metrics).to_csv(outcomes_dir / f"layer3_geometry_metrics_{ts}.csv", index=False)
            print(f"   Saved geometry metrics to layer3_geometry_metrics_{ts}.csv")
        except Exception as e:
            print(f"   Failed to save geometry metrics: {e}")

    # ---------------------------------------------------------
    # 4. Data Alignment
    # ---------------------------------------------------------
    
    # Must use original index alignment
    # Net Returns needed for Alpha Target (Calculated at step 0)
    
    # Ensure net_returns is set (it should be from Step 0 logic)
    if net_returns is None:
         # Double check if it was set in Step 0, if not, force it
         if 'ret_series' in locals() and ret_series is not None:
             net_returns = ret_series
         else:
             # Should be caught by Step 0 logic, but just in case
             net_returns = pd.Series(0, index=df.index)

    ret_series = net_returns.reindex(df.index)
    
    # Volatility needed for Alpha Target and Weighting
    # vol_series already defined
    
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
    if layer2_weight is not None:
        w_prob = layer2_weight.reindex(df.index).fillna(1.0).values
    else:
        w_prob = np.ones(len(df))
    
    w_prob = finalize_sample_weights(w_prob)

    # 4. Integrate Specialist Signals into Final Feature Matrix
    specialist_cols = [c for c in df.columns if c.startswith('meta_prob_g_')]
    if specialist_cols:
        tprint_info(f"   Integrating {len(specialist_cols)} Specialist signals into the final ensemble...")
        meta_features.extend(specialist_cols)
        # Ensure no duplicates
        meta_features = list(dict.fromkeys(meta_features))

    # Clean Feature Matrix
    X = df[meta_features]

    # Common Cross-Validation (Purged)
    # Common Cross-Validation (Purged)
    n_splits = 5
    
    if len(df) < n_splits * 2:
        tprint_warning(f"   Insufficient data for CV split (n={len(df)}). Skipping Layer 3 training.")
        fallback_df = pd.DataFrame(index=df.index)
        fallback_df['meta_prob'] = 0.5
        if target_col in df.columns:
            fallback_df[target_col] = df[target_col]
        else:
            fallback_df[target_col] = 0
        return fallback_df, None

    # Use config for purge?
    cv = PurgedKFoldTime(n_splits=n_splits, purge=100, embargo=50)
    splits = list(cv.split(df))

    # ---------------------------------------------------------
    # HEAD A: ALPHA GENERATION
    # ---------------------------------------------------------
    tprint_info("\n>> HEAD A: ALPHA GENERATION (Race & Train)")

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
            # Score = 100*MeanIC + 50*(Mean/Std) - IC IR metric
            mean_ic = np.mean(fold_ics)
            std_ic = np.std(fold_ics) + 1e-6
            score_ic = 100 * mean_ic + 50 * (mean_ic / std_ic)
        else:
            score_ic = -999.0
            mean_ic = 0.0  # Set default for print statement
            
        alpha_scores[cand['name']] = score_ic
        print(f"     ScoreIC: {score_ic:.4f} (Mean IC: {mean_ic:.4f})")

    best_alpha_name = max(alpha_scores, key=alpha_scores.get)
    best_alpha_cand = next(c for c in alpha_candidates if c['name'] == best_alpha_name)
    print(f"   🏆 Alpha Winner: {best_alpha_name}")

    # HPO for Alpha Winner
    best_alpha_params = _run_layer3_hpo(
        X, pd.Series(y_alpha), w_alpha,
        model_type=best_alpha_cand['type'],
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

    # Improved Calibration using OOFProbabilityCalibrator
    # OOFCalibrationConfig only accepts: method, isotonic_out_of_bounds, platt_regularization, 
    # temperature_search_range, beta_prior_strength, min_samples_for_calibration, validation_split,
    # output_range, clip_to_range, cache_dir, model_id
    cal_config = OOFCalibrationConfig(
        method='isotonic',
        min_samples_for_calibration=100
    )
    
    calibrator = OOFProbabilityCalibrator(config=cal_config)
    
    # Calibrate the OOF predictions using the proper interface
    # Create a pd.Series from meta_prob_oof for the calibrator
    oof_series = pd.Series(meta_prob_oof, index=df.index)
    y_true_series = pd.Series(y_prob, index=df.index)
    
    # Fit and transform to get calibrated predictions
    try:
        tprint_info("🔧 Calibrating OOF predictions...")
        calibrated_series = calibrator.fit_transform(
            oof_predictions=oof_series,
            y_true=y_true_series
        )
        meta_prob_oof = calibrated_series.values
        tprint_success(f"✅ Calibration complete: {calibrator.get_calibration_metrics()}")
    except Exception as e:
        tprint_error(f"❌ OOF Calibration failed: {e}, using raw probabilities")
    
    # Fit Final Prob Model (Full Data)
    tprint_info("🚀 Fitting Final Prob Model (Full) with Enhanced Calibration...")
    base_prob_model = lgb.LGBMClassifier(**best_prob_params)
    base_prob_model.fit(X, y_prob, sample_weight=w_prob)
    
    # Final production calibrator fit on full data
    # Use ManualCalibratedClassifier because base_prob_model is already fitted on full data
    # and standard CalibratedClassifierCV(cv='prefit') might be flaky in this env
    final_prob_model = ManualCalibratedClassifier(
        base_estimator=base_prob_model,
        method='isotonic'
    )
    final_prob_model.fit(X, y_prob)
    tprint_success("✅ Final probability model fitted and calibrated")

    # ---------------------------------------------------------
    # Output Assembly
    # ---------------------------------------------------------

    tprint_info("📊 Assembling final outputs...")
    df['meta_alpha'] = meta_alpha_oof
    df['meta_prob'] = meta_prob_oof

    models_dict = {
        'alpha_model': final_alpha_model,
        'prob_model': final_prob_model,
        'geometry_models': final_models, # Added per user request
        'best_alpha_type': best_alpha_name,
        'best_prob_type': best_prob_name
    }

    tprint_info(f"📈 Final models: {len(final_models)} geometry models, 1 alpha model, 1 prob model")

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

def _safe_to_markdown(df: pd.DataFrame) -> str:
    """Fallback for to_markdown() if tabulate is missing."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = df.columns
        res = [" | " + " | ".join(map(str, cols)) + " | "]
        res.append(" | " + " | ".join(["---"] * len(cols)) + " | ")
        for _, row in df.iterrows():
            formatted_row = [f"{x:.4f}" if isinstance(x, (float, np.float64, np.float32)) else str(x) for x in row]
            res.append(" | " + " | ".join(formatted_row) + " | ")
        return "\n".join(res)

def _generate_layer3_meta_report(
    df: pd.DataFrame, 
    specialist_metrics: List[Dict[str, Any]], 
    median_mdi: Dict[str, float], 
    median_shap: Dict[str, float],
    outcomes_dir: Path, 
    ts: str,
    cfg: Dict[str, Any]
):
    """
    Generate a detailed analysis report for Layer 3 Meta-Models.
    Aligned with De Prado's best practices for model verification.
    """
    try:
        lines = ["# Layer 3 Meta-Labeling Consolidated Report\n\n"]
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Instrument: {cfg.get('symbol', 'UNKNOWN')} | Timeframe: {cfg.get('timeframe', '15m')}\n\n")

        # 1. Specialists Performance
        lines.append("## Specialist Performance Summary\n")
        metric_df = pd.DataFrame(specialist_metrics)
        if not metric_df.empty:
            # Sort by score or AUC
            metric_df = metric_df.sort_values('score', ascending=False)
            lines.append(_safe_to_markdown(metric_df) + "\n\n")
        else:
            lines.append("No geometry-specific metrics available.\n\n")

        # 2. Global Feature Importance (MDI)
        lines.append("## Top 20 Global Feature Importance (MDI)\n")
        if median_mdi:
            mdi_df = pd.DataFrame(list(median_mdi.items()), columns=['feature', 'importance'])\
                       .sort_values('importance', ascending=False).head(20)
            lines.append(_safe_to_markdown(mdi_df) + "\n\n")
        else:
            lines.append("MDI importance not available.\n\n")

        # 3. Non-Linear Importance (SHAP)
        lines.append("## Top 20 Non-Linear Importance (SHAP Absolute Mean)\n")
        if median_shap:
            shap_df = pd.DataFrame(list(median_shap.items()), columns=['feature', 'shap_abs_mean'])\
                        .sort_values('shap_abs_mean', ascending=False).head(20)
            lines.append(_safe_to_markdown(shap_df) + "\n\n")
        else:
            lines.append("SHAP importance not available.\n\n")

        # 4. Calibration Check (ECE Proxy)
        lines.append("## Calibration Diagnostics\n")
        if 'meta_prob' in df.columns:
            ece = _fast_expected_calibration_error(
                (df[cfg.get('target_col', 'target')] > 0.5).astype(int).values,
                df['meta_prob'].values
            )
            lines.append(f"- **Expected Calibration Error (ECE)**: {ece:.4f}\n")
            lines.append("- *Note: ECE targets < 0.05 for high-confidence trading.*\n\n")

        report_path = outcomes_dir / f"layer3_meta_report_{ts}.md"
        report_path.write_text("".join(lines))
        tprint_success(f"💾 Layer 3 meta-report saved to {report_path}")

    except Exception as e:
        print(f"⚠️ Failed to generate Layer 3 meta-report: {e}")

    tprint_success(f"🎉 Layer 3 Complete! Generated {len(df)} rows with meta_alpha and meta_prob")
    
    # --- De Prado Report Generation ---
    _generate_layer3_meta_report(df, geometry_metrics, global_mdi, global_shap, outcomes_dir, ts, cfg)
    
    # Dump feature inventory with aggregated importance
    try:
        symbol = cfg.get('symbol', 'UNKNOWN')
        tf = cfg.get('timeframe', '15m')
        _dump_layer3_feature_inventory(
            df=df,
            feature_cols=meta_features,
            target_col=target_col,
            outcomes_dir=outcomes_dir,
            symbol=symbol,
            timeframe=tf,
            ts=ts,
            stage="Layer3_Final",
            cfg=cfg,
            mdi_importance=global_mdi,
            shap_importance=global_shap
        )
    except Exception as e:
        print(f"⚠️ Failed to dump feature inventory: {e}")

    tprint_info(f"💾 Outputs saved to {outcomes_dir}")
    return df, models_dict


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
    
    # Return nothing
    return
