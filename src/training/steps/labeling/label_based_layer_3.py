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

EPS = 1e-12

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
        _compute_cusum_loop(n, r_vals, r_tilde, s_trend_pos, s_trend_neg, s_rev_pos, s_rev_neg)

        # 7. Normalize Signals: Signal = (S+ - |S-|) / h
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
    transformed = sig.copy()

    if mode == 'linear':
        pass

    elif mode == 'cubic_regime':
        # "Sniper": Penalize uncertainty + Cut size in crisis
        p95 = np.percentile(vol, 95)
        regime_mult = np.where(vol > p95, 0.5, 1.0)
        transformed = (np.sign(sig) * np.abs(sig)**3) * regime_mult

    elif mode == 'tanh_dynamic':
        norm_vol = vol / (np.median(vol) + EPS)
        transformed = np.tanh(sig / norm_vol)

    return np.clip(transformed, -5.0, 5.0)

def generate_geometries_adaptive(
    base_signals: pd.DataFrame,
    volatility: pd.Series,
    mfe: pd.Series,
    mae: pd.Series,
    trend_ratios: list = [0.0, 0.5, 1.0],
    activations: list = ['linear', 'cubic_regime', 'tanh_dynamic']
) -> dict:
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
    rad_vec = (mfe / (mae + EPS)) / (sigma + EPS)
    rad_med = np.median(rad_vec)
    rad_mad = np.median(np.abs(rad_vec - rad_med))
    stability = 1.0 / (1.0 + rad_mad)
    rad_score = rad_med * stability
    tail_risk = np.percentile(mae, 95) / (np.median(sigma) + EPS)
    return rad_score, tail_risk

def worker_evaluate_geometry(
    gid: str,
    df: Dict[str, np.ndarray],
    labels: np.ndarray,
    min_variance: float = 1e-8,
    min_f_score: float = 0.5
):
    sig = df['composite_signal']
    sigma_vals = df['sigma_eff']
    mfe = df['mfe']
    mae = df['mae']

    if np.var(sig) < min_variance:
        return None

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

    rad, tail = compute_risk_metrics(mfe, mae, sigma_vals)

    X = np.column_stack((sig_valid, sigma_vals[valid_mask]))
    y = lbl_valid

    tscv = TimeSeriesSplit(n_splits=5, gap=50)
    preds_list = []
    targets_list = []

    model = HistGradientBoostingClassifier(
        max_iter=50, max_depth=3, learning_rate=0.1, min_samples_leaf=30,
        l2_regularization=1.0, early_stopping=False, random_state=42
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

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker_evaluate_geometry)(k, v, labels_arr)
        for k, v in meta_geometries.items()
    )
    results = [r for r in results if r is not None]

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    df = df[df['auc'] > min_auc].copy().reset_index(drop=True)
    if df.empty:
        print("No candidates passed min_auc.")
        return pd.DataFrame()

    auc_log = df['auc'].clip(0.5001, 0.9999).apply(logit)
    rad_vals = df['rad'].values
    safe_vals = -1 * df['tail_risk'].values

    def robust_scale(x):
        return (x - np.median(x)) / (np.median(np.abs(x - np.median(x))) + EPS)

    df['z_auc'] = robust_scale(auc_log)
    df['z_rad'] = robust_scale(rad_vals)
    df['z_safe'] = robust_scale(safe_vals)

    fitness_matrix = df[['z_auc', 'z_rad', 'z_safe']].values.astype(np.float64)
    pareto_mask = is_pareto_efficient_numba(fitness_matrix)

    pareto_candidates = df[pareto_mask].copy()
    print(f"2. Pareto Filter: {len(df)} -> {len(pareto_candidates)} candidates.")

    candidates = pareto_candidates if len(pareto_candidates) >= top_k else df

    candidates['score'] = (
        0.4 * candidates['z_auc'] +
        0.4 * candidates['z_rad'] +
        0.2 * candidates['z_safe']
    )
    candidates = candidates.sort_values('score', ascending=False)

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
# Core Helpers & Objectives
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

def _focal_loss_objective(y_true, y_pred):
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()
    gamma = 2.0
    p = expit(y_pred)
    grad = -y_true * (1 - p)**gamma * (1 - p - gamma * p * np.log(p + 1e-15)) + \
           (1 - y_true) * p**gamma * (p + gamma * (1 - p) * np.log(1 - p + 1e-15))
    hess = p * (1 - p)
    return grad, hess

def _asymmetric_mse_objective(y_true, y_pred):
    if hasattr(y_true, 'get_label'):
        y_true = y_true.get_label()
    residual = (y_true - y_pred)
    grad = -2 * residual
    hess = 2 * np.ones_like(residual)
    penalty = 1.5
    over_pred = residual < 0
    grad[over_pred] *= penalty
    hess[over_pred] *= penalty
    return grad, hess

def _calculate_alpha_target(returns: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    vol_safe = np.where(volatility < 1e-6, 1e-6, volatility)
    alpha = returns / vol_safe
    return np.clip(alpha, -4.0, 4.0)

def _calculate_ic_score(y_true: np.ndarray, y_pred: np.ndarray, folds_ic: List[float] = None) -> float:
    ic, _ = spearmanr(y_true, y_pred)
    if np.isnan(ic): ic = 0.0
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

# ---------------------------------------------------------
# HPO
# ---------------------------------------------------------

def _run_layer3_hpo(
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    model_type: str,  # 'classifier', 'regressor', 'ridge', 'alpha_lgbm', 'alpha_ridge'
    objective_func: str = None, # 'binary_logloss', 'focal', 'mse', 'huber', 'asymmetric_mse'
    n_trials: int = 40,
) -> Dict[str, Any]:
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

    split_idx = int(len(X_hpo) * 0.8)
    X_train, X_val = X_hpo.iloc[:split_idx], X_hpo.iloc[split_idx:]
    y_train, y_val = y_hpo.iloc[:split_idx], y_hpo.iloc[split_idx:]
    w_train, w_val = w_hpo[:split_idx], w_hpo[split_idx:]

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'n_estimators': trial.suggest_int('n_estimators', 400, 800),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 50),
            'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 1e-2),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.3, 0.7),
            'lambda_l2': 0.0,
            'bagging_freq': 1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_jobs': 1,
            'verbosity': -1
        }
        params['lambda_l2'] = 2.0 * params['lambda_l1']

        try:
            if model_type == 'alpha_lgbm':
                if objective_func == 'asymmetric_mse':
                    params['objective'] = _asymmetric_mse_objective
                    params['metric'] = 'rmse'
                elif objective_func == 'huber':
                    params['objective'] = 'huber'
                    params['metric'] = 'mae'
                    params['alpha'] = 0.9
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
                ic = spearmanr(y_val, preds)[0]
                return ic if np.isfinite(ic) else -1.0

            elif model_type == 'alpha_ridge':
                alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train, sample_weight=w_train)
                preds = model.predict(X_val)
                ic = spearmanr(y_val, preds)[0]
                return ic if np.isfinite(ic) else -1.0

            elif model_type == 'classifier':
                if objective_func == 'focal':
                    params['objective'] = _focal_loss_objective
                    params['metric'] = 'binary_logloss'
                else:
                    params['objective'] = 'binary'
                    params['metric'] = 'binary_logloss'

                model = lgb.LGBMClassifier(**params)
                y_tr_bin = (y_train >= 0.5).astype(int)
                y_val_bin = (y_val >= 0.5).astype(int)

                model.fit(
                    X_train, y_tr_bin,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val_bin)],
                    eval_sample_weight=[w_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                preds = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val_bin, preds)
                ll = log_loss(y_val_bin, preds)
                score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
                return score

        except Exception as e:
            return -999.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=HyperbandPruner())
    study.optimize(objective, n_trials=n_trials)

    print(f"   Best HPO Score: {study.best_value:.4f}")

    if model_type == 'alpha_ridge':
        return study.best_params

    best_p = study.best_params.copy()
    best_p['lambda_l2'] = 2.0 * best_p['lambda_l1']
    best_p['bagging_freq'] = 1
    best_p['feature_fraction'] = 0.8
    best_p['bagging_fraction'] = 0.8
    best_p['n_jobs'] = 1
    best_p['verbosity'] = -1

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

# -----------------------------------------------------------
# Training Pipeline Components
# -----------------------------------------------------------

def _train_dual_head_for_geometry(
    gid: str,
    X: pd.DataFrame,
    y_alpha: np.ndarray,
    y_prob: np.ndarray,
    w_alpha: np.ndarray,
    w_prob: np.ndarray,
    outcomes_dir: Path,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Trains Dual-Head (Alpha + Prob) model for a single geometry.
    Returns (meta_alpha_oof, meta_prob_oof, alpha_model, prob_model)
    """
    print(f"\n[{gid}] Training Dual-Head Meta-Learner...")
    
    # Common Cross-Validation
    n_splits = 5
    cv = PurgedKFoldTime(n_splits=n_splits, purge=100, embargo=50)
    splits = list(cv.split(X))

    # --- HEAD A: ALPHA GENERATION ---
    print(f"   [{gid}] Head A: Alpha Generation")
    alpha_candidates = [
        {'name': 'Ridge_MSE', 'type': 'alpha_ridge', 'obj': 'mse'},
        {'name': 'LGBM_Huber', 'type': 'alpha_lgbm', 'obj': 'huber'},
        {'name': 'LGBM_AsymMSE', 'type': 'alpha_lgbm', 'obj': 'asymmetric_mse'}
    ]
    alpha_scores = {}

    for cand in alpha_candidates:
        fold_ics = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_alpha[train_idx], y_alpha[val_idx]
            w_tr = w_alpha[train_idx]
            
            try:
                if cand['type'] == 'alpha_ridge':
                    model = Ridge(alpha=1.0)
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    preds = model.predict(X_val)
                else:
                    params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': 1}
                    if cand['obj'] == 'huber':
                        params['objective'] = 'huber'; params['alpha'] = 0.9
                    elif cand['obj'] == 'asymmetric_mse':
                        params['objective'] = _asymmetric_mse_objective
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    preds = model.predict(X_val)
                
                ic, _ = spearmanr(y_val, preds)
                if np.isfinite(ic): fold_ics.append(ic)
            except Exception: pass

        if fold_ics:
            mean_ic = np.mean(fold_ics)
            std_ic = np.std(fold_ics) + 1e-6
            score_ic = 100 * mean_ic + 50 * (mean_ic / std_ic)
        else:
            score_ic = -999.0
        alpha_scores[cand['name']] = score_ic

    best_alpha_name = max(alpha_scores, key=alpha_scores.get)
    best_alpha_cand = next(c for c in alpha_candidates if c['name'] == best_alpha_name)
    print(f"     Winner: {best_alpha_name}")

    best_alpha_params = _run_layer3_hpo(
        X, pd.Series(y_alpha), w_alpha,
        model_type=best_alpha_cand['type'], objective_func=best_alpha_cand['obj'], n_trials=10
    )

    meta_alpha_oof = np.full(len(X), np.nan)
    for train_idx, val_idx in splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y_alpha[train_idx]
        w_tr = w_alpha[train_idx]
        if best_alpha_cand['type'] == 'alpha_ridge':
            model = Ridge(**best_alpha_params)
        else:
            model = lgb.LGBMRegressor(**best_alpha_params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        meta_alpha_oof[val_idx] = model.predict(X_val)

    if best_alpha_cand['type'] == 'alpha_ridge':
        final_alpha_model = Ridge(**best_alpha_params)
    else:
        final_alpha_model = lgb.LGBMRegressor(**best_alpha_params)
    final_alpha_model.fit(X, y_alpha, sample_weight=w_alpha)


    # --- HEAD B: PROBABILITY CALIBRATION ---
    print(f"   [{gid}] Head B: Probability Calibration")
    prob_candidates = [
        {'name': 'LGBM_LogLoss', 'type': 'classifier', 'obj': 'binary_logloss'},
        {'name': 'LGBM_Focal', 'type': 'classifier', 'obj': 'focal'}
    ]
    prob_scores = {}

    for cand in prob_candidates:
        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_prob[train_idx], y_prob[val_idx]
            w_tr = w_prob[train_idx]
            try:
                params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': 1}
                if cand['obj'] == 'focal': params['objective'] = _focal_loss_objective
                else: params['objective'] = 'binary'
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                preds = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, preds)
                ll = log_loss(y_val, preds)
                score = 100 * (auc - 0.5) + 50 * (0.693 - ll)
                fold_scores.append(score)
            except Exception: pass
        prob_scores[cand['name']] = np.mean(fold_scores) if fold_scores else -999.0

    best_prob_name = max(prob_scores, key=prob_scores.get)
    best_prob_cand = next(c for c in prob_candidates if c['name'] == best_prob_name)
    print(f"     Winner: {best_prob_name}")

    best_prob_params = _run_layer3_hpo(
        X, pd.Series(y_prob), w_prob,
        model_type='classifier', objective_func=best_prob_cand['obj'], n_trials=10
    )

    meta_prob_oof = np.full(len(X), np.nan)
    for train_idx, val_idx in splits:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, w_tr = y_prob[train_idx], w_prob[train_idx]
        base_est = lgb.LGBMClassifier(**best_prob_params)
        cal_model = CalibratedClassifierCV(base_est, method='isotonic', cv=3)
        cal_model.fit(X_tr, y_tr, sample_weight=w_tr)
        meta_prob_oof[val_idx] = cal_model.predict_proba(X_val)[:, 1]

    final_prob_model = CalibratedClassifierCV(
        estimator=lgb.LGBMClassifier(**best_prob_params),
        method='isotonic', cv=3
    )
    final_prob_model.fit(X, y_prob, sample_weight=w_prob)

    return meta_alpha_oof, meta_prob_oof, final_alpha_model, final_prob_model

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
    Revised Layer 3: Multi-Geometry Dual-Head Architecture.

    1. Selects best geometries.
    2. Generates Regime Leaf & NN features ONCE (post-selection).
    3. For EACH geometry:
       - Trains Dual-Head Meta-Learner (Alpha + Prob).
    4. Aggregates results.
    """
    print(f"\n{'='*60}")
    print("LAYER 3: MULTI-GEOMETRY DUAL-HEAD META-MODELS")
    print(f"{'='*60}")

    df = oof_df.copy()
    cfg = config if isinstance(config, dict) else {}

    # ---------------------------------------------------------
    # 1. Feature Engineering (Shared/Global)
    # ---------------------------------------------------------
    print("<< Generating Global Layer 3 Features...")
    if base_model_cols:
        safe_base_cols = [c for c in base_model_cols if c in df.columns]
        df[safe_base_cols] = df[safe_base_cols].fillna(0.5)

    if market_data is not None and not market_data.empty:
        for c in ['volume', 'high', 'low', 'close']:
            if c in market_data.columns:
                df[c] = market_data[c].reindex(df.index)

    try:
        df = generate_layer3_features(df, safe_base_cols if base_model_cols else [])
    except Exception as e:
        print(f"⚠️ generate_layer3_features failed: {e}")

    # Check for Base Model Spread
    if 'ens_prediction_dispersion' not in df.columns and 'base_pred_std' in df.columns:
        df['ens_prediction_dispersion'] = df['base_pred_std']

    # Identify Global Features
    global_features = [c for c in df.columns if c not in ['target', target_col] and 'meta_prob' not in c]
    
    # ---------------------------------------------------------
    # 2. Geometry Selection
    # ---------------------------------------------------------
    if market_data is not None:
        print("<< Computing CUSUM Signals (Trend/Reversal)...")
        cusum_df = compute_cusum_signals_multi_window(market_data, windows=[12, 24, 48])
        cusum_df = cusum_df.reindex(df.index).fillna(0.0)
    else:
        print("⚠️ No market_data, skipping CUSUM generation.")
        cusum_df = pd.DataFrame(index=df.index)

    if 'mfe' in df.columns and 'mae' in df.columns:
        mfe_s, mae_s = df['mfe'], df['mae']
    else:
        mfe_s = pd.Series(0, index=df.index)
        mae_s = pd.Series(1, index=df.index)

    vol_s = df['volatility_1d'] if 'volatility_1d' in df.columns else pd.Series(0.01, index=df.index)
    
    print("<< Generating Meta-Geometries...")
    geometries_dict = generate_geometries_adaptive(
        base_signals=cusum_df, volatility=vol_s, mfe=mfe_s, mae=mae_s
    )
    
    print("<< Selecting Best Geometries...")
    y_target = df[target_col].fillna(0)
    
    selected_geoms_df = select_best_geometries_production(
        geometries_dict, y_target, top_k=5, n_jobs=1
    )
    
    if selected_geoms_df.empty:
        print("⚠️ No geometries selected! Using fallback.")
        selected_ids = ['default']
        if not geometries_dict:
             geometries_dict['default'] = {
                 'composite_signal': np.zeros(len(df)), 'sigma_eff': vol_s.values,
                 'z_auc': 0, 'z_stab': 0, 'z_rad': 0, 'z_safe': 0
             }
        else:
             selected_ids = [list(geometries_dict.keys())[0]]
    else:
        selected_ids = selected_geoms_df['id'].values.tolist()
        print(f"   Selected {len(selected_ids)}: {selected_ids}")

    # ---------------------------------------------------------
    # 3. Generate Expensive Features (Once, Post-Selection)
    # ---------------------------------------------------------
    # We generate these ONLY if we have valid geometries selected.
    # We use a fixed horizon of 48 to match the max geometry window.

    advanced_features_df = pd.DataFrame(index=df.index)

    if market_data is not None:
        # Regime Leaf Features
        try:
            print("<< Generating Regime Leaf Features (Horizon=48)...")
            rl_config = {
                "targets": {"macro_trend_horizons": [48]}, # Explicitly set horizon
                "enabled_targets": [
                    "regime_trendiness", "regime_volatility", "regime_trend_efficiency",
                    "regime_memory", "regime_liquidity", "regime_volume_force_direction",
                    "regime_breakout", "regime_future_range"
                ],
                "inputs": {"input_source": "ohlcv_only", "ohlcv_feature_config": {}},
                "onehot": {"enabled": False},
                "interaction_feature": {"enabled": True, "include_base": True},
                "reporting": {"enabled": False},
                "walk_forward": {"mode": "cross_fit", "cross_fit": {"n_splits": 5}}
            }
            rl_df = extract_regime_leaf_onehot_features(
                X=pd.DataFrame(index=df.index), market_data=market_data,
                config=rl_config, random_state=42, verbose=False
            )
            if rl_df is not None and not rl_df.empty:
                rl_df = rl_df.reindex(df.index).fillna(0.0)
                advanced_features_df = pd.concat([advanced_features_df, rl_df], axis=1)
        except Exception as e:
            print(f"⚠️ Regime leaf generation failed: {e}")

        # NN Sequence Embeddings
        try:
            print("<< Generating NN Embeddings...")
            nn_df = generate_nn_sequence_embeddings(
                market_data=market_data, encoder_type="stacked", seq_len=24, embed_dim=8,
                use_conv=True, use_lstm=True, use_attention=False
            )
            if nn_df is not None and not nn_df.empty:
                nn_df = nn_df.reindex(df.index).fillna(0.0)
                advanced_features_df = pd.concat([advanced_features_df, nn_df], axis=1)
        except Exception as e:
            print(f"⚠️ NN embeddings failed: {e}")

    # ---------------------------------------------------------
    # 4. Loop: Train Per Geometry
    # ---------------------------------------------------------
    final_models = {}
    
    # Prepare Shared Targets & Weights
    if net_returns is None: raise ValueError("net_returns required")
    ret_series = net_returns.reindex(df.index)
    vol_series = df['volatility_1d'].replace(0, np.nan).fillna(method='ffill').fillna(0.001)

    y_alpha = _calculate_alpha_target(ret_series.values, vol_series.values)
    y_prob = (pd.to_numeric(df[target_col], errors='coerce').fillna(0.5) >= 0.5).astype(int).values

    vol_safe = np.clip(vol_series.values, 1e-4, None)
    w_alpha = finalize_sample_weights(1.0 / (vol_safe ** 2))
    w_prob = finalize_sample_weights(layer2_weight.reindex(df.index).fillna(1.0).values)

    for gid in selected_ids:
        print(f"\n>> Processing Geometry: {gid}")
        g_data = geometries_dict[gid]
        
        # 1. Construct Feature Matrix
        geo_df = df[global_features].copy()

        # Add Geometry-Specific Signals
        geo_df[f'{gid}_sig'] = g_data['composite_signal']
        geo_df[f'{gid}_sigma'] = g_data['sigma_eff']
        geo_df[f'{gid}_sig_x_vol'] = g_data['composite_signal'] * g_data['sigma_eff']

        # Attach Pre-Calculated Advanced Features
        if not advanced_features_df.empty:
            geo_df = pd.concat([geo_df, advanced_features_df], axis=1)

        # Clean
        geo_df = geo_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2. Train Dual-Head Model
        alpha_oof, prob_oof, alpha_model, prob_model = _train_dual_head_for_geometry(
            gid, geo_df, y_alpha, y_prob, w_alpha, w_prob, Path('outcomes'), cfg
        )

        # Store
        df[f'meta_prob_{gid}'] = prob_oof
        df[f'meta_alpha_{gid}'] = alpha_oof

        final_models[gid] = {
            'alpha_model': alpha_model,
            'prob_model': prob_model,
            'features': list(geo_df.columns) # Important for inference
        }

    # 5. Aggregate (Simple Average for now)
    prob_cols = [c for c in df.columns if c.startswith('meta_prob_') and c != 'meta_prob']
    alpha_cols = [c for c in df.columns if c.startswith('meta_alpha_') and c != 'meta_alpha']

    if prob_cols:
        df['meta_prob'] = df[prob_cols].mean(axis=1)
    if alpha_cols:
        df['meta_alpha'] = df[alpha_cols].mean(axis=1)

    # Helper: Diagnostic Plot (Unchanged logic)
    def plot_diagnostics(y_true, y_prob, output_path=None):
        try:
            y_prob_numeric = pd.to_numeric(y_prob, errors='coerce')
            mask = ~y_prob_numeric.isna()
            y_true = y_true[mask]
            y_prob = y_prob_numeric[mask]
            if len(y_true) == 0: return

            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax[0].plot(prob_pred, prob_true, marker='o', label='Meta-Model')
            ax[0].plot([0, 1], [0, 1], '--', color='gray')
            sns.histplot(y_prob, bins=20, ax=ax[1])
            if output_path: plt.savefig(output_path)
            plt.close(fig)
        except Exception: pass

    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        plot_diagnostics(
            y_prob, df['meta_prob'],
            output_path=f"outcomes/layer3_prob_calibration_{ts}.png"
        )
    except Exception: pass

    return df, final_models

def _run_shap_analysis(model, X, output_dir, symbol, timeframe, ts, md_path):
    pass
