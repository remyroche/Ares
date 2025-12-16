"""
Post-HPO Model Evaluation and Backtesting Module.

After HPO completes, this module trains multiple ML models for SNR diagnostics
and backtesting with extensive metrics and reporting.

Models trained:
1. Simple LGBM (baseline)
2. Logistic Regression (linear benchmark)
3. LGBM Bagged with Diversity Defense

Metrics computed:
- OOF AUC, Precision, Recall, F1
- Calibration metrics (ECE, MCE, Brier)
- Sharpe ratio, profit factor, win rate
- Signal-to-Noise Ratio (SNR) diagnostics
- Parameter-outcome correlation matrix
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr

import lightgbm as lgb

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.pipeline_standards import PipelineStandards
from src.utils.ml_common.validation.thresholding import optimize_threshold

logger = logging.getLogger(__name__)


def _safe_to_datetime_index(idx: Any) -> Optional[pd.DatetimeIndex]:
    try:
        if isinstance(idx, pd.DatetimeIndex):
            return idx
        if isinstance(idx, pd.Index) and getattr(idx, "dtype", None) is not None:
            return pd.DatetimeIndex(idx)
        arr = np.asarray(idx)
        return pd.DatetimeIndex(arr)
    except Exception:
        return None


def _build_purged_embargo_splits(
    t0_index: pd.Index,
    t1: pd.Series,
    n_splits: int,
    embargo: Optional[pd.Timedelta] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx_dt = _safe_to_datetime_index(t0_index)
    t1_dt = None
    try:
        t1_dt = pd.to_datetime(t1)
    except Exception:
        t1_dt = None

    if idx_dt is None or t1_dt is None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return [(tr, te) for tr, te in tscv.split(np.arange(len(t0_index)))]

    if embargo is None:
        try:
            deltas = (t1_dt.values - idx_dt.values).astype('timedelta64[ns]')
            deltas = deltas[np.isfinite(deltas.astype('int64'))]
            if deltas.size:
                emb = pd.to_timedelta(np.median(deltas))
                if emb <= pd.Timedelta(0):
                    emb = pd.Timedelta(0)
                embargo = emb
            else:
                embargo = pd.Timedelta(0)
        except Exception:
            embargo = pd.Timedelta(0)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    t0_vals = idx_dt.values
    t1_vals = pd.DatetimeIndex(t1_dt.reindex(t0_index).values).values

    for tr_raw, te_raw in tscv.split(np.arange(len(t0_index))):
        if te_raw.size == 0:
            continue

        te_start = t0_vals[te_raw.min()]
        te_end = t0_vals[te_raw.max()]

        tr_mask = np.ones(len(tr_raw), dtype=bool)
        tr_indices = tr_raw

        try:
            t0_tr = t0_vals[tr_indices]
            t1_tr = t1_vals[tr_indices]
            overlap = (t0_tr <= te_end) & (t1_tr >= te_start)
            tr_mask &= ~overlap
        except Exception:
            pass

        if embargo is not None and embargo > pd.Timedelta(0):
            try:
                embargo_end = te_end + embargo
                t0_tr = t0_vals[tr_indices]
                emb = (t0_tr > te_end) & (t0_tr <= embargo_end)
                tr_mask &= ~emb
            except Exception:
                pass

        tr_final = tr_indices[tr_mask]
        splits.append((tr_final, te_raw))

    return splits


def _causal_impute_train_val(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_combined = pd.concat([X_train, X_val], axis=0)
    X_combined = X_combined.replace([np.inf, -np.inf], np.nan)
    X_combined = X_combined.ffill()

    X_train_filled = X_combined.iloc[: len(X_train)].copy()
    X_val_filled = X_combined.iloc[len(X_train) :].copy()

    med = X_train_filled.median(numeric_only=True)
    if isinstance(med, pd.Series) and not med.empty:
        X_train_filled = X_train_filled.fillna(med)
        X_val_filled = X_val_filled.fillna(med)

    X_train_filled = X_train_filled.fillna(0.0)
    X_val_filled = X_val_filled.fillna(0.0)
    return X_train_filled, X_val_filled


# ============================================================================
# CALIBRATION METRICS
# ============================================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.
    
    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
    
    Returns:
        Dict with ECE, MCE, Brier score, log loss
    """
    tprint_info("   Computing calibration metrics...")
    
    # Filter valid predictions
    valid_mask = np.isfinite(y_prob) & np.isfinite(y_true)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]
    
    if len(y_true) < 50:
        return {"ece": np.nan, "mce": np.nan, "brier": np.nan, "log_loss": np.nan}
    
    # Clip probabilities for numerical stability
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    
    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Log loss
    try:
        ll = log_loss(y_true, y_prob)
    except Exception:
        ll = np.nan
    
    # Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        
        # Bin weights (proportion of samples in each bin)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_weights = bin_counts / len(y_prob)
        
        # ECE: weighted average of |accuracy - confidence| per bin
        calibration_errors = np.abs(prob_true - prob_pred)
        ece = np.sum(bin_weights[:len(calibration_errors)] * calibration_errors)
        
        # MCE: maximum calibration error
        mce = np.max(calibration_errors) if len(calibration_errors) > 0 else np.nan
    except Exception:
        ece = np.nan
        mce = np.nan
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "log_loss": float(ll),
    }


# ============================================================================
# SNR DIAGNOSTICS
# ============================================================================

def compute_snr_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Signal-to-Noise Ratio diagnostics.
    
    SNR = (Mean positive return when prob > threshold) / (Std of returns when prob > threshold)
    
    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        returns: Realized returns per event
        threshold: Probability threshold for signal
    
    Returns:
        Dict with SNR metrics
    """
    tprint_info("   Computing SNR diagnostics...")
    
    valid_mask = np.isfinite(y_prob) & np.isfinite(returns)
    y_prob = y_prob[valid_mask]
    returns = returns[valid_mask]
    y_true = y_true[valid_mask]

    if len(returns) < 50:
        return {"snr": np.nan, "snr_positive": np.nan, "snr_negative": np.nan}
    
    # Signal mask: predictions above threshold
    signal_mask = y_prob >= threshold
    noise_mask = y_prob < threshold
    
    # Returns when signal is positive
    signal_returns = returns[signal_mask]
    noise_returns = returns[noise_mask]
    
    # SNR calculations
    if len(signal_returns) > 10 and np.std(signal_returns) > 1e-9:
        snr_positive = np.mean(signal_returns) / np.std(signal_returns)
    else:
        snr_positive = np.nan
    
    if len(noise_returns) > 10 and np.std(noise_returns) > 1e-9:
        snr_negative = np.mean(noise_returns) / np.std(noise_returns)
    else:
        snr_negative = np.nan
    
    # Overall SNR
    if np.std(returns) > 1e-9:
        snr_overall = np.mean(returns) / np.std(returns)
    else:
        snr_overall = np.nan
    
    # Information Coefficient (IC): Spearman correlation between prob and returns
    try:
        ic, _ = spearmanr(y_prob, returns)
    except Exception:
        ic = np.nan
    
    # Hit rate at different thresholds
    hit_rates = {}
    for thr in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = y_prob >= thr
        if mask.sum() > 10:
            hit_rates[f"hit_rate_{int(thr*100)}"] = float((returns[mask] > 0).mean())
        else:
            hit_rates[f"hit_rate_{int(thr*100)}"] = np.nan
    
    return {
        "snr_overall": float(snr_overall) if np.isfinite(snr_overall) else np.nan,
        "snr_positive": float(snr_positive) if np.isfinite(snr_positive) else np.nan,
        "snr_negative": float(snr_negative) if np.isfinite(snr_negative) else np.nan,
        "information_coefficient": float(ic) if np.isfinite(ic) else np.nan,
        "n_signals": int(signal_mask.sum()),
        "n_noise": int(noise_mask.sum()),
        **hit_rates,
    }


# ============================================================================
# BACKTESTING METRICS
# ============================================================================

def compute_backtest_metrics(
    y_prob: np.ndarray,
    returns: np.ndarray,
    threshold: float = 0.5,
    risk_free_rate: float = 0.0,
    transaction_cost: float = 0.0,
    direction: str = "long",
    event_times: Optional[Any] = None,
    returns_are_net: bool = True,
    annualize: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive backtesting metrics.
    
    Args:
        y_prob: Predicted probabilities
        returns: Realized returns per event
        threshold: Probability threshold for taking trades
        risk_free_rate: Annual risk-free rate (currently unused for event-level sharpe)
        transaction_cost: Transaction cost per trade (only applied when returns_are_net=False)
        direction: Trading direction (currently unused; returns are assumed direction-consistent)
        event_times: Optional event timestamps (used to estimate trades/year)
        returns_are_net: If True, assumes returns already include transaction costs
        annualize: If True and event_times available, annualize Sharpe/Calmar using trade frequency
    
    Returns:
        Dict with backtest metrics including cost-adjusted returns
    """
    if verbose:
        tprint_info("   Computing backtest metrics...")
    
    valid_mask = np.isfinite(y_prob) & np.isfinite(returns)
    y_prob = y_prob[valid_mask]
    returns = returns[valid_mask]

    event_times_filtered = None
    if event_times is not None:
        try:
            if isinstance(event_times, pd.Index):
                event_times_filtered = event_times[valid_mask]
            else:
                event_times_filtered = np.asarray(event_times)[valid_mask]
        except Exception:
            event_times_filtered = None
    
    if len(returns) < 50:
        return {
            "sharpe_ratio": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "mean_return": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
        }
    
    # Filter to trades taken (prob >= threshold)
    trade_mask = y_prob >= threshold
    trade_returns = returns[trade_mask]

    trade_times = None
    if event_times_filtered is not None:
        try:
            trade_times = event_times_filtered[trade_mask]
        except Exception:
            trade_times = None
    
    if len(trade_returns) < 10:
        return {
            "sharpe_ratio": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "mean_return": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
            "n_trades": int(trade_mask.sum()),
            "cost_adjusted_sharpe": np.nan,
            "cost_adjusted_return": np.nan,
        }
    
    n_trades = int(trade_mask.sum())

    trades_per_year = None
    trades_per_day = None
    span_days = None
    try:
        et_idx = _safe_to_datetime_index(event_times_filtered)
        if et_idx is not None and et_idx.size >= 2:
            span_days = float((et_idx.max() - et_idx.min()).total_seconds()) / 86400.0
            if span_days > 0:
                trades_per_day = float(n_trades) / span_days
                trades_per_year = trades_per_day * 365.0
    except Exception:
        trades_per_year = None

    apply_costs = (not returns_are_net) and (transaction_cost is not None) and (float(transaction_cost) > 0)
    if apply_costs:
        cost_adjusted_returns = trade_returns - float(transaction_cost)
        total_cost = float(n_trades) * float(transaction_cost)
    else:
        cost_adjusted_returns = trade_returns.copy()
        total_cost = 0.0
    
    # Basic metrics (gross)
    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    
    # Cost-adjusted metrics
    mean_return_cost_adj = np.mean(cost_adjusted_returns)
    std_return_cost_adj = np.std(cost_adjusted_returns)
    
    # Win rate
    win_rate = (trade_returns > 0).mean()
    win_rate_cost_adj = (cost_adjusted_returns > 0).mean()
    
    sharpe_ratio = np.nan
    cost_adjusted_sharpe = np.nan

    if std_return > 1e-9:
        sharpe_per_trade = (mean_return - 0.0) / std_return
        if annualize and trades_per_year is not None and trades_per_year > 0:
            sharpe_ratio = sharpe_per_trade * np.sqrt(trades_per_year)
        else:
            sharpe_ratio = sharpe_per_trade

    if std_return_cost_adj > 1e-9:
        sharpe_cost_per_trade = (mean_return_cost_adj - 0.0) / std_return_cost_adj
        if annualize and trades_per_year is not None and trades_per_year > 0:
            cost_adjusted_sharpe = sharpe_cost_per_trade * np.sqrt(trades_per_year)
        else:
            cost_adjusted_sharpe = sharpe_cost_per_trade
    
    # Profit factor
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    if gross_loss > 1e-9:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf if gross_profit > 0 else np.nan
    
    # Cost-adjusted profit factor
    cost_adj_profit = cost_adjusted_returns[cost_adjusted_returns > 0].sum()
    cost_adj_loss = abs(cost_adjusted_returns[cost_adjusted_returns < 0].sum())
    if cost_adj_loss > 1e-9:
        profit_factor_cost_adj = cost_adj_profit / cost_adj_loss
    else:
        profit_factor_cost_adj = np.inf if cost_adj_profit > 0 else np.nan
    
    max_drawdown = np.nan
    calmar_ratio = np.nan
    try:
        r_for_eq = np.asarray(cost_adjusted_returns, dtype=float)
        tt_idx = _safe_to_datetime_index(trade_times)
        if tt_idx is not None and tt_idx.size == r_for_eq.size:
            try:
                order = np.argsort(tt_idx.view("int64"))
                r_for_eq = r_for_eq[order]
            except Exception:
                pass
        r_for_eq = np.nan_to_num(r_for_eq, nan=0.0, posinf=0.0, neginf=0.0)
        r_for_eq = np.clip(r_for_eq, -0.999, None)
        equity = np.cumprod(1.0 + r_for_eq)
        running_max = np.maximum.accumulate(equity)
        dd = 1.0 - (equity / (running_max + 1e-12))
        max_drawdown = float(np.max(dd)) if dd.size else np.nan

        if max_drawdown > 1e-12 and n_trades > 0:
            ann_return = None
            if annualize and span_days is not None and span_days > 0:
                try:
                    span_years = float(span_days) / 365.0
                    if span_years > 0:
                        ann_return = float(equity[-1] ** (1.0 / span_years) - 1.0)
                except Exception:
                    ann_return = None
            if ann_return is None and annualize and trades_per_year is not None and trades_per_year > 0:
                try:
                    ann_return = float(equity[-1] ** (trades_per_year / float(n_trades)) - 1.0)
                except Exception:
                    ann_return = float(mean_return_cost_adj) * float(trades_per_year)
            if ann_return is not None:
                calmar_ratio = float(ann_return) / float(max_drawdown)
    except Exception:
        max_drawdown = np.nan
        calmar_ratio = np.nan
    
    # Additional metrics
    avg_win = np.mean(trade_returns[trade_returns > 0]) if (trade_returns > 0).any() else 0
    avg_loss = np.mean(trade_returns[trade_returns < 0]) if (trade_returns < 0).any() else 0
    
    return {
        "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.nan,
        "win_rate": float(win_rate),
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio) if np.isfinite(calmar_ratio) else np.nan,
        "n_trades": n_trades,
        "trades_per_day": float(trades_per_day) if trades_per_day is not None and np.isfinite(trades_per_day) else np.nan,
        "trades_per_year": float(trades_per_year) if trades_per_year is not None and np.isfinite(trades_per_year) else np.nan,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_return": float(np.sum(trade_returns)),
        "cost_adjusted_sharpe": float(cost_adjusted_sharpe) if np.isfinite(cost_adjusted_sharpe) else np.nan,
        "cost_adjusted_return": float(mean_return_cost_adj),
        "cost_adjusted_profit_factor": float(profit_factor_cost_adj) if np.isfinite(profit_factor_cost_adj) else np.nan,
        "cost_adjusted_win_rate": float(win_rate_cost_adj),
        "total_transaction_cost": float(total_cost),
    }


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_simple_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
    enable_calibration: bool = False,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Train a simple LightGBM classifier with time-series CV.
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Optional sample weights
        n_splits: Number of CV splits
        enable_calibration: Whether to apply probability calibration
    
    Returns:
        Tuple of (OOF predictions, trained model (possibly calibrated), training metrics)
    """
    tprint_info("🌲 Training Simple LGBM...")
    
    params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42,
    }
    
    oof_probs = np.full(len(y), np.nan)
    fold_aucs = []
    
    tscv = TimeSeriesSplit(n_splits=n_splits) if cv_splits is None else None
    splits_iter = list(tscv.split(X)) if tscv is not None else list(cv_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(splits_iter):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train, X_val = _causal_impute_train_val(X_train, X_val)
        
        if len(np.unique(y_train.dropna())) < 2:
            continue
        
        model = lgb.LGBMClassifier(**params)
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        
        if len(np.unique(y_val)) >= 2:
            auc = roc_auc_score(y_val, probs)
            fold_aucs.append(auc)
            tprint_info(f"      Fold {fold_idx + 1}: AUC = {auc:.4f}")
    
    # Train final model on all data (use causal fill + train-fitted median; no bfill)
    X_fit = X.replace([np.inf, -np.inf], np.nan).ffill()
    med_fit = X_fit.median(numeric_only=True)
    if isinstance(med_fit, pd.Series) and not med_fit.empty:
        X_fit = X_fit.fillna(med_fit)
    X_fit = X_fit.fillna(0.0)

    base_model = lgb.LGBMClassifier(**params)
    if sample_weights is not None:
        base_model.fit(X_fit, y, sample_weight=sample_weights)
    else:
        base_model.fit(X_fit, y)
    
    # Apply calibration if requested
    final_model = base_model
    if enable_calibration:
        try:
            tprint_info("   Applying probability calibration...")
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=min(3, n_splits))
            calibrated_model.fit(X_fit, y)
            final_model = calibrated_model
            tprint_info("   ✅ Calibration applied")
        except Exception as e:
            tprint_warning(f"   ⚠️ Calibration failed: {e}, using uncalibrated model")
            final_model = base_model
    
    metrics = {
        "model_type": "simple_lgbm",
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
        "calibrated": enable_calibration and final_model != base_model,
    }
    
    tprint_success(f"   ✅ Simple LGBM: Mean AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
    
    return oof_probs, final_model, metrics


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
    enable_calibration: bool = False,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Train a Logistic Regression classifier with time-series CV.
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Optional sample weights
        n_splits: Number of CV splits
        enable_calibration: Whether to apply probability calibration
    
    Returns:
        Tuple of (OOF predictions, trained model (possibly calibrated), training metrics)
    """
    tprint_info("📈 Training Logistic Regression...")
    
    oof_probs = np.full(len(y), np.nan)
    fold_aucs = []
    
    tscv = TimeSeriesSplit(n_splits=n_splits) if cv_splits is None else None
    splits_iter = list(tscv.split(X)) if tscv is not None else list(cv_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(splits_iter):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train, X_val = _causal_impute_train_val(X_train, X_val)
        
        if len(np.unique(y_train.dropna())) < 2:
            continue
        
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1,
        )
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
            model.fit(X_train, y_train, sample_weight=w_train)
        else:
            model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        
        if len(np.unique(y_val)) >= 2:
            auc = roc_auc_score(y_val, probs)
            fold_aucs.append(auc)
            tprint_info(f"      Fold {fold_idx + 1}: AUC = {auc:.4f}")
    
    # Train final model on all data (use causal fill + train-fitted median; no bfill)
    X_fit = X.replace([np.inf, -np.inf], np.nan).ffill()
    med_fit = X_fit.median(numeric_only=True)
    if isinstance(med_fit, pd.Series) and not med_fit.empty:
        X_fit = X_fit.fillna(med_fit)
    X_fit = X_fit.fillna(0.0)

    base_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1)
    if sample_weights is not None:
        base_model.fit(X_fit, y, sample_weight=sample_weights)
    else:
        base_model.fit(X_fit, y)
    
    # Apply calibration if requested
    final_model = base_model
    if enable_calibration:
        try:
            tprint_info("   Applying probability calibration...")
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=min(3, n_splits))
            calibrated_model.fit(X_fit, y)
            final_model = calibrated_model
            tprint_info("   ✅ Calibration applied")
        except Exception as e:
            tprint_warning(f"   ⚠️ Calibration failed: {e}, using uncalibrated model")
            final_model = base_model
    
    metrics = {
        "model_type": "logistic_regression",
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
        "calibrated": enable_calibration and final_model != base_model,
    }
    
    tprint_success(f"   ✅ Logistic Regression: Mean AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
    
    return oof_probs, final_model, metrics


def train_lgbm_bagged_diversity(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
    n_bags: int = 10,
    feature_fraction_range: Tuple[float, float] = (0.5, 0.8),
    depth_range: Tuple[int, int] = (3, 7),
    bag_parallelism: int = -1,
    enable_calibration: bool = False,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Train LGBM Bagged ensemble with diversity defense.
    
    Diversity is achieved through:
    1. Bootstrap sampling
    2. Random feature subsets
    3. Varying tree depths
    4. Different random seeds
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Optional sample weights
        n_splits: Number of CV splits
        n_bags: Number of bagged estimators
        feature_fraction_range: Range for feature fraction
        depth_range: Range for max_depth
        bag_parallelism: Parallelism for bag training
        enable_calibration: Whether to apply probability calibration
    
    Returns:
        Tuple of (OOF predictions, trained ensemble (models list or calibrated wrapper), training metrics)
    """
    tprint_info(f"🎒 Training LGBM Bagged with Diversity Defense ({n_bags} bags)...")
    
    oof_probs = np.full(len(y), np.nan)
    all_models = []
    fold_aucs = []
    diversity_metrics = []
    
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def _train_single_bag(
        bag_idx: int,
        fold_idx: int,
        X_train_fold: pd.DataFrame,
        y_train_fold: pd.Series,
        X_val_fold: pd.DataFrame,
        sample_weights_fold: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[lgb.LGBMClassifier]]:
        # Diversity: vary hyperparameters per bag
        rng = np.random.RandomState(42 + fold_idx * 100 + bag_idx)

        feature_fraction = rng.uniform(*feature_fraction_range)
        max_depth = rng.randint(*depth_range)

        params = {
            'n_estimators': 150,
            'max_depth': max_depth,
            'learning_rate': 0.05,
            'num_leaves': min(2 ** max_depth, 64),
            'subsample': 0.8,
            'colsample_bytree': feature_fraction,
            'reg_alpha': rng.uniform(0.01, 0.2),
            'reg_lambda': rng.uniform(0.01, 0.2),
            # Single-threaded models when running bags in parallel
            'n_jobs': 1,
            'verbose': -1,
            'random_state': 42 + bag_idx,
        }

        # Bootstrap sample
        n_train = len(X_train_fold)
        boot_idx = rng.choice(n_train, size=n_train, replace=True)

        X_boot = X_train_fold.iloc[boot_idx]
        y_boot = y_train_fold.iloc[boot_idx]

        model = lgb.LGBMClassifier(**params)

        try:
            if sample_weights_fold is not None:
                w_boot = sample_weights_fold[boot_idx]
                model.fit(X_boot, y_boot, sample_weight=w_boot)
            else:
                model.fit(X_boot, y_boot)

            probs = model.predict_proba(X_val_fold)[:, 1]
            return probs, model
        except Exception as e:
            tprint_warning(f"      Bag {bag_idx} failed: {e}")
            return None, None

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if len(np.unique(y_train.dropna())) < 2:
            continue

        if sample_weights is not None:
            sample_weights_fold = sample_weights[train_idx]
        else:
            sample_weights_fold = None

        # Train bags in parallel within this fold
        bag_results = Parallel(n_jobs=bag_parallelism)(
            delayed(_train_single_bag)(
                bag_idx,
                fold_idx,
                X_train,
                y_train,
                X_val,
                sample_weights_fold,
            )
            for bag_idx in range(n_bags)
        )

        fold_probs = []
        fold_models = []

        for probs, model in bag_results:
            if probs is None or model is None:
                continue
            fold_probs.append(probs)
            fold_models.append(model)

        if fold_probs:
            # Average predictions across bags
            mean_probs = np.mean(fold_probs, axis=0)
            oof_probs[val_idx] = mean_probs

            # Compute diversity metric (average pairwise disagreement)
            diversity = np.nan
            if len(fold_probs) >= 2:
                pairwise_corrs = []
                for i in range(len(fold_probs)):
                    for j in range(i + 1, len(fold_probs)):
                        corr, _ = pearsonr(fold_probs[i], fold_probs[j])
                        pairwise_corrs.append(corr)
                if pairwise_corrs:
                    diversity = 1 - np.mean(pairwise_corrs)  # Higher = more diverse
                    diversity_metrics.append(diversity)

            if len(np.unique(y_val)) >= 2:
                auc = roc_auc_score(y_val, mean_probs)
                fold_aucs.append(auc)
                tprint_info(
                    f"      Fold {fold_idx + 1}: AUC = {auc:.4f} (diversity={diversity:.3f})"
                )

        all_models.extend(fold_models)
    
    # Train final ensemble on all data for inference (use causal fill + train-fitted median; no bfill)
    X_fit = X.replace([np.inf, -np.inf], np.nan).ffill()
    med_fit = X_fit.median(numeric_only=True)
    if isinstance(med_fit, pd.Series) and not med_fit.empty:
        X_fit = X_fit.fillna(med_fit)
    X_fit = X_fit.fillna(0.0)

    final_ensemble_models = []
    if all_models:
        tprint_info("   Training final ensemble on all data...")
        for bag_idx in range(min(n_bags, len(all_models))):
            rng = np.random.RandomState(42 + bag_idx)
            feature_fraction = rng.uniform(*feature_fraction_range)
            max_depth = rng.randint(*depth_range)
            
            params = {
                'n_estimators': 150,
                'max_depth': max_depth,
                'learning_rate': 0.05,
                'num_leaves': min(2 ** max_depth, 64),
                'subsample': 0.8,
                'colsample_bytree': feature_fraction,
                'reg_alpha': rng.uniform(0.01, 0.2),
                'reg_lambda': rng.uniform(0.01, 0.2),
                'n_jobs': 1,
                'verbose': -1,
                'random_state': 42 + bag_idx,
            }
            
            # Bootstrap sample
            n_train = len(X_fit)
            boot_idx = rng.choice(n_train, size=n_train, replace=True)
            X_boot = X_fit.iloc[boot_idx]
            y_boot = y.iloc[boot_idx]
            
            model = lgb.LGBMClassifier(**params)
            try:
                if sample_weights is not None:
                    w_boot = sample_weights[boot_idx]
                    model.fit(X_boot, y_boot, sample_weight=w_boot)
                else:
                    model.fit(X_boot, y_boot)
                final_ensemble_models.append(model)
            except Exception as e:
                tprint_warning(f"      Final ensemble bag {bag_idx} failed: {e}")
        
        if not final_ensemble_models:
            tprint_warning("   ⚠️ No final ensemble models trained, using CV models")
            final_ensemble_models = all_models[:n_bags] if all_models else []
    
    # Wrap ensemble for calibration if requested
    final_ensemble = final_ensemble_models
    if enable_calibration and final_ensemble_models:
        try:
            tprint_info("   Applying probability calibration to ensemble...")
            # Create a sklearn-compatible wrapper that averages predictions from all models
            class EnsembleWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, models):
                    self.models = models
                    self.classes_ = np.array([0, 1])  # Binary classification
                
                def fit(self, X, y):
                    # Models are already trained, just store for predict_proba
                    return self
                
                def predict_proba(self, X):
                    probs = np.array([m.predict_proba(X)[:, 1] for m in self.models])
                    mean_probs = probs.mean(axis=0)
                    return np.column_stack([1 - mean_probs, mean_probs])
                
                def predict(self, X):
                    proba = self.predict_proba(X)
                    return (proba[:, 1] >= 0.5).astype(int)
            
            base_ensemble = EnsembleWrapper(final_ensemble_models)
            calibrated_ensemble = CalibratedClassifierCV(base_ensemble, method='isotonic', cv=min(3, n_splits))
            calibrated_ensemble.fit(X_fit, y)
            final_ensemble = calibrated_ensemble
            tprint_info("   ✅ Ensemble calibration applied")
        except Exception as e:
            tprint_warning(f"   ⚠️ Ensemble calibration failed: {e}, using uncalibrated ensemble")
            final_ensemble = final_ensemble_models
    
    metrics = {
        "model_type": "lgbm_bagged_diversity",
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
        "n_bags": n_bags,
        "mean_diversity": float(np.mean(diversity_metrics)) if diversity_metrics else np.nan,
        "total_models": len(all_models),
        "final_ensemble_size": len(final_ensemble_models),
        "calibrated": enable_calibration and final_ensemble != final_ensemble_models,
    }
    
    tprint_success(
        f"   ✅ LGBM Bagged: Mean AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}, "
        f"Diversity = {metrics['mean_diversity']:.3f}"
    )
    
    return oof_probs, final_ensemble, metrics


# ============================================================================
# COMPREHENSIVE POST-HPO EVALUATION
# ============================================================================

def run_post_hpo_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    t1: Optional[pd.Series] = None,
    n_splits: int = 5,
    n_bags: int = 10,
    probability_threshold: float = 0.5,
    symbol: str = "UNKNOWN",
    exchange: str = "UNKNOWN",
    timeframe: str = "15m",
    direction: str = "long",
    save_artifacts: bool = True,
    optimize_thresholds: bool = True,
    enable_calibration: bool = True,
    transaction_cost: float = 0.001,
    embargo: Optional[pd.Timedelta] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive post-HPO model evaluation.
    
    Trains multiple models, computes SNR diagnostics, backtesting metrics,
    and generates comparison reports.
    
    Args:
        X: Feature matrix
        y: Binary labels
        realized_returns: Realized returns per event
        sample_weights: Optional sample weights
        n_splits: Number of CV splits
        n_bags: Number of bags for diversity ensemble
        probability_threshold: Default threshold for backtest signal (may be optimized)
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        direction: Trading direction
        save_artifacts: Whether to save artifacts to disk
        optimize_thresholds: Whether to optimize thresholds via CV
        enable_calibration: Whether to apply probability calibration
        transaction_cost: Transaction cost rate (e.g., 0.001 = 0.1%)
    
    Returns:
        Dict with all evaluation results
    """
    tprint_info("=" * 60)
    tprint_info("🔬 POST-HPO MODEL EVALUATION")
    tprint_info("=" * 60)
    tprint_info(f"   Symbol: {symbol} | Timeframe: {timeframe} | Direction: {direction}")
    tprint_info(f"   Samples: {len(y)} | Features: {X.shape[1]}")
    tprint_info("=" * 60)
    
    results = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "timestamp": datetime.utcnow().isoformat(),
        "models": {},
    }
    
    # Align indices
    common_idx = X.index.intersection(y.index).intersection(realized_returns.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    returns_arr = realized_returns.loc[common_idx].values
    
    # Validate and align sample weights
    if sample_weights is not None:
        if isinstance(sample_weights, pd.Series):
            sample_weights = sample_weights.loc[common_idx].values
        elif isinstance(sample_weights, np.ndarray):
            if len(sample_weights) != len(common_idx):
                tprint_warning(f"   ⚠️ Sample weights length ({len(sample_weights)}) doesn't match data ({len(common_idx)}), ignoring weights")
                sample_weights = None
        else:
            tprint_warning(f"   ⚠️ Invalid sample weights type, ignoring weights")
            sample_weights = None
        
        if sample_weights is not None:
            # Validate weights: non-negative and finite
            invalid_mask = ~np.isfinite(sample_weights) | (sample_weights < 0)
            if invalid_mask.any():
                n_invalid = invalid_mask.sum()
                tprint_warning(f"   ⚠️ Found {n_invalid} invalid weights (non-finite or negative), setting to 0")
                sample_weights[invalid_mask] = 0.0
            
            # Normalize weights to prevent numerical issues
            weight_sum = sample_weights.sum()
            if weight_sum > 0:
                sample_weights = sample_weights / weight_sum * len(sample_weights)
            else:
                tprint_warning(f"   ⚠️ All weights are zero, ignoring weights")
                sample_weights = None

    # Clean data (imputation is performed fold-wise to avoid leakage)
    valid_mask = ~y.isna()
    X_clean = X.loc[valid_mask].copy()
    y_clean = y.loc[valid_mask]
    returns_clean = returns_arr[valid_mask.values]

    cv_splits = None
    if t1 is not None:
        try:
            t1_aligned = t1.loc[common_idx]
            t1_clean = t1_aligned.loc[X_clean.index]
            if len(t1_clean) == len(X_clean):
                cv_splits = _build_purged_embargo_splits(
                    t0_index=X_clean.index,
                    t1=t1_clean,
                    n_splits=n_splits,
                    embargo=embargo,
                )
        except Exception:
            cv_splits = None

    if sample_weights is not None:
        weights_clean = sample_weights[valid_mask.values]
    else:
        weights_clean = None

    tprint_info(f"   Clean samples: {len(y_clean)}")
    
    # =========================================================================
    # Train Models
    # =========================================================================
    
    # 1. Simple LGBM
    try:
        lgbm_probs, lgbm_model, lgbm_metrics = train_simple_lgbm(
            X_clean, y_clean, weights_clean, n_splits, enable_calibration, cv_splits=cv_splits
        )
        results["models"]["simple_lgbm"] = {
            "oof_probs": lgbm_probs,
            "training_metrics": lgbm_metrics,
            "final_model": lgbm_model,  # Store for inference
        }
    except Exception as e:
        tprint_error(f"   ❌ Simple LGBM failed: {e}")
        lgbm_probs = None
        lgbm_model = None
    
    # 2. Logistic Regression
    try:
        lr_probs, lr_model, lr_metrics = train_logistic_regression(
            X_clean, y_clean, weights_clean, n_splits, enable_calibration, cv_splits=cv_splits
        )
        results["models"]["logistic_regression"] = {
            "oof_probs": lr_probs,
            "training_metrics": lr_metrics,
            "final_model": lr_model,  # Store for inference
        }
    except Exception as e:
        tprint_error(f"   ❌ Logistic Regression failed: {e}")
        lr_probs = None
        lr_model = None
    
    # 3. LGBM Bagged with Diversity
    try:
        bagged_probs, bagged_ensemble, bagged_metrics = train_lgbm_bagged_diversity(
            X_clean,
            y_clean,
            weights_clean,
            n_splits,
            n_bags,
            enable_calibration=enable_calibration,
            cv_splits=cv_splits,
        )
        results["models"]["lgbm_bagged_diversity"] = {
            "oof_probs": bagged_probs,
            "training_metrics": bagged_metrics,
            "final_ensemble": bagged_ensemble,  # Store for inference
        }
    except Exception as e:
        tprint_error(f"   ❌ LGBM Bagged failed: {e}")
        bagged_probs = None
        bagged_ensemble = None
    
    # =========================================================================
    # Compute Metrics for Each Model
    # =========================================================================
    
    tprint_info("\n" + "=" * 60)
    tprint_info("📊 COMPUTING COMPREHENSIVE METRICS")
    tprint_info("=" * 60)
    
    for model_name, model_data in results["models"].items():
        oof_probs = model_data.get("oof_probs")
        if oof_probs is None:
            continue
        
        tprint_info(f"\n--- {model_name.upper()} ---")
        
        # Optimize threshold if requested
        optimal_threshold = probability_threshold
        threshold_optimization_result = None
        if optimize_thresholds:
            try:
                tprint_info("   Optimizing threshold...")
                # Use Sharpe ratio as optimization metric for trading
                valid_probs_mask = ~np.isnan(oof_probs)
                if valid_probs_mask.sum() > 100:
                    # Optimize threshold based on cost-adjusted Sharpe
                    thresholds_to_test = np.linspace(0.3, 0.8, 21)
                    best_sharpe = -np.inf
                    best_thresh = probability_threshold
                    
                    for thresh in thresholds_to_test:
                        backtest_test = compute_backtest_metrics(
                            oof_probs[valid_probs_mask],
                            returns_clean[valid_probs_mask],
                            thresh,
                            transaction_cost=transaction_cost,
                            direction=direction,
                            event_times=X_clean.index[valid_probs_mask],
                            returns_are_net=True,
                        )
                        test_sharpe = backtest_test.get("cost_adjusted_sharpe", np.nan)
                        if np.isfinite(test_sharpe) and test_sharpe > best_sharpe:
                            best_sharpe = test_sharpe
                            best_thresh = thresh
                    
                    optimal_threshold = best_thresh
                    threshold_optimization_result = {
                        "optimal_threshold": float(optimal_threshold),
                        "optimal_sharpe": float(best_sharpe),
                        "default_threshold": probability_threshold,
                    }
                    tprint_info(f"   ✅ Optimal threshold: {optimal_threshold:.3f} (Sharpe: {best_sharpe:.4f})")
            except Exception as e:
                tprint_warning(f"   ⚠️ Threshold optimization failed: {e}, using default")
                optimal_threshold = probability_threshold
        
        # Calibration metrics
        calib_metrics = compute_calibration_metrics(y_clean.values, oof_probs)
        model_data["calibration_metrics"] = calib_metrics
        
        # SNR diagnostics (using optimal threshold)
        snr_metrics = compute_snr_diagnostics(
            y_clean.values, oof_probs, returns_clean, optimal_threshold
        )
        model_data["snr_metrics"] = snr_metrics
        
        # Backtest metrics (using optimal threshold and transaction costs)
        backtest_metrics = compute_backtest_metrics(
            oof_probs, returns_clean, optimal_threshold,
            transaction_cost=transaction_cost,
            direction=direction,
            event_times=X_clean.index,
            returns_are_net=True,
        )
        model_data["backtest_metrics"] = backtest_metrics
        
        # Store threshold optimization results
        if threshold_optimization_result:
            model_data["threshold_optimization"] = threshold_optimization_result
            model_data["used_threshold"] = optimal_threshold
        else:
            model_data["used_threshold"] = probability_threshold
        
        # Classification metrics (using optimal threshold)
        valid_probs = ~np.isnan(oof_probs)
        if valid_probs.sum() > 50:
            y_pred = (oof_probs[valid_probs] >= optimal_threshold).astype(int)
            y_true_valid = y_clean.values[valid_probs]
            
            model_data["classification_metrics"] = {
                "accuracy": float(accuracy_score(y_true_valid, y_pred)),
                "precision": float(precision_score(y_true_valid, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_valid, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_valid, y_pred, zero_division=0)),
                "threshold_used": float(optimal_threshold),
            }
        
        # Print summary
        train_auc = model_data["training_metrics"].get("mean_auc", np.nan)
        sharpe = backtest_metrics.get("cost_adjusted_sharpe", backtest_metrics.get("sharpe_ratio", np.nan))
        snr = snr_metrics.get("snr_positive", np.nan)
        ic = snr_metrics.get("information_coefficient", np.nan)
        
        tprint_info(f"   AUC: {train_auc:.4f} | Sharpe (cost-adj): {sharpe:.4f} | SNR: {snr:.4f} | IC: {ic:.4f} | Threshold: {optimal_threshold:.3f}")
    
    # =========================================================================
    # Generate Comparison Report
    # =========================================================================
    
    tprint_info("\n" + "=" * 60)
    tprint_info("📋 MODEL COMPARISON SUMMARY")
    tprint_info("=" * 60)
    
    comparison_data = []
    for model_name, model_data in results["models"].items():
        row = {
            "Model": model_name,
            "AUC": model_data.get("training_metrics", {}).get("mean_auc", np.nan),
            "AUC_Std": model_data.get("training_metrics", {}).get("std_auc", np.nan),
            "Sharpe": model_data.get("backtest_metrics", {}).get("sharpe_ratio", np.nan),
            "Win_Rate": model_data.get("backtest_metrics", {}).get("win_rate", np.nan),
            "Profit_Factor": model_data.get("backtest_metrics", {}).get("profit_factor", np.nan),
            "SNR": model_data.get("snr_metrics", {}).get("snr_positive", np.nan),
            "IC": model_data.get("snr_metrics", {}).get("information_coefficient", np.nan),
            "ECE": model_data.get("calibration_metrics", {}).get("ece", np.nan),
            "Brier": model_data.get("calibration_metrics", {}).get("brier", np.nan),
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    results["model_comparison"] = comparison_df.to_dict(orient='records')
    
    # Print comparison table
    tprint_info("\n" + comparison_df.to_string(index=False))
    
    # =========================================================================
    # Save Artifacts
    # =========================================================================
    
    if save_artifacts:
        tprint_info("\n💾 Saving artifacts...")
        
        # Use PipelineStandards for path construction
        try:
            base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
            outcomes_dir = Path(base_dir) / "post_hpo_evaluation"
            outcomes_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to build standardized path: {e}, using fallback")
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison CSV
        csv_path = outcomes_dir / f"post_hpo_model_comparison_{symbol}_{timeframe}_{timestamp}.csv"
        comparison_df.to_csv(csv_path, index=False)
        tprint_success(f"   ✅ Saved comparison to {csv_path}")
        
        # Save detailed JSON report
        json_path = outcomes_dir / f"post_hpo_evaluation_{symbol}_{timeframe}_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = {
            "symbol": results["symbol"],
            "timeframe": results["timeframe"],
            "direction": results["direction"],
            "n_samples": results["n_samples"],
            "n_features": results["n_features"],
            "timestamp": results["timestamp"],
            "model_comparison": results["model_comparison"],
            "models": {},
        }
        
        for model_name, model_data in results["models"].items():
            json_results["models"][model_name] = {
                "training_metrics": model_data.get("training_metrics", {}),
                "calibration_metrics": model_data.get("calibration_metrics", {}),
                "snr_metrics": model_data.get("snr_metrics", {}),
                "backtest_metrics": model_data.get("backtest_metrics", {}),
                "classification_metrics": model_data.get("classification_metrics", {}),
                "threshold_optimization": model_data.get("threshold_optimization", {}),
                "used_threshold": model_data.get("used_threshold", probability_threshold),
            }
            # Note: final_model/final_ensemble are not JSON-serializable, store paths if needed
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        tprint_success(f"   ✅ Saved detailed report to {json_path}")
        
        results["artifacts"] = {
            "comparison_csv": str(csv_path),
            "detailed_json": str(json_path),
        }
    
    tprint_success("\n✅ Post-HPO Evaluation Complete!")
    
    return results


# ============================================================================
# PARAMETER-OUTCOME CORRELATION ANALYSIS
# ============================================================================

def compute_parameter_outcome_correlations(
    candidate_pool: List[Dict[str, Any]],
    param_keys: Optional[List[str]] = None,
    outcome_keys: Optional[List[str]] = None,
    backtest_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix between HPO parameters and outcome metrics.
    
    Args:
        candidate_pool: List of candidate configurations from HPO
        param_keys: List of parameter keys to include
        outcome_keys: List of outcome keys to include
        backtest_metrics: Optional dict mapping config_id or params hash to backtest metrics
    
    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    tprint_info("📊 Computing parameter-outcome correlations...")
    
    if not candidate_pool:
        tprint_warning("   ⚠️ Empty candidate pool")
        return pd.DataFrame(), pd.DataFrame()
    
    # Extract data
    data = []
    for c in candidate_pool:
        row = {}
        
        # Extract parameters
        params = c.get('params', {})
        if isinstance(params, dict):
            for k, v in params.items():
                try:
                    row[f"param_{k}"] = float(v)
                except (ValueError, TypeError):
                    continue
        
        # Extract outcomes - comprehensive list including all available metrics
        outcome_keys_to_extract = [
            # Primary objectives
            'edge', 'combined', 'edge_scaled',
            # Model performance
            'auc', 'mean_auc', 'learnability', 'profitability',
            # Risk-adjusted metrics
            'sharpe_pos', 'ic', 'calibration_score', 'fidelity_score',
            # Event statistics
            'n_events', 'n_raw_events', 'n_vol_scaled_events', 'trades_per_day',
            # Return statistics
            'mean_pos', 'mean_neg', 'balance_score',
            # Signal quality
            'snr_pre', 'snr_post', 'retention_total', 'retention_pos', 'retention_neg',
            'effect_size_pre', 'effect_size_post',
            # Time-to-outcome
            'mean_tto', 'timeout_rate', 'tto_penalty',
            # Statistical power
            'n_required_80pct_power', 'n_pre_events',
            # Learnability robustness
            'learnability_worst_fold_auc', 'learnability_auc_cv_std',
        ]
        
        for k in outcome_keys_to_extract:
            if k in c:
                try:
                    val = c[k]
                    # Handle NaN and None values
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    row[f"outcome_{k}"] = float(val)
                except (ValueError, TypeError):
                    continue
        
        # Add backtest metrics if available (match by config_id or params hash)
        if backtest_metrics:
            config_id = c.get('config_id')
            if config_id and config_id in backtest_metrics:
                bt_metrics = backtest_metrics[config_id]
                for k, v in bt_metrics.items():
                    try:
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            row[f"outcome_backtest_{k}"] = float(v)
                    except (ValueError, TypeError):
                        continue
            else:
                # Try matching by params hash as fallback
                try:
                    import hashlib
                    params_str = json.dumps(c.get('params', {}), sort_keys=True)
                    params_hash = hashlib.md5(params_str.encode()).hexdigest()
                    if params_hash in backtest_metrics:
                        bt_metrics = backtest_metrics[params_hash]
                        for k, v in bt_metrics.items():
                            try:
                                if isinstance(v, (int, float)) and not np.isnan(v):
                                    row[f"outcome_backtest_{k}"] = float(v)
                            except (ValueError, TypeError):
                                continue
                except Exception:
                    pass  # Skip hash matching if it fails
        
        if row:
            data.append(row)
    
    if not data:
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Filter columns
    param_cols = [c for c in df.columns if c.startswith('param_')]
    outcome_cols = [c for c in df.columns if c.startswith('outcome_')]
    
    if param_keys:
        param_cols = [c for c in param_cols if c.replace('param_', '') in param_keys]
    if outcome_keys:
        outcome_cols = [c for c in outcome_cols if c.replace('outcome_', '') in outcome_keys]
    
    if not param_cols or not outcome_cols:
        return pd.DataFrame(), pd.DataFrame()
    
    # Compute correlation matrix
    corr_data = {}
    pval_data = {}
    
    for param in param_cols:
        param_name = param.replace('param_', '')
        corr_data[param_name] = {}
        pval_data[param_name] = {}
        
        for outcome in outcome_cols:
            outcome_name = outcome.replace('outcome_', '')
            
            # Filter valid pairs
            mask = df[param].notna() & df[outcome].notna()
            if mask.sum() < 10:
                corr_data[param_name][outcome_name] = np.nan
                pval_data[param_name][outcome_name] = np.nan
                continue
            
            try:
                corr, pval = spearmanr(df.loc[mask, param], df.loc[mask, outcome])
                corr_data[param_name][outcome_name] = corr
                pval_data[param_name][outcome_name] = pval
            except Exception:
                corr_data[param_name][outcome_name] = np.nan
                pval_data[param_name][outcome_name] = np.nan
    
    corr_df = pd.DataFrame(corr_data).T
    pval_df = pd.DataFrame(pval_data).T
    
    tprint_info(f"   Computed {len(param_cols)} x {len(outcome_cols)} correlation matrix")
    
    return corr_df, pval_df


def generate_correlation_report(
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    significance_threshold: float = 0.05,
) -> str:
    """
    Generate a formatted correlation report.
    
    Args:
        corr_df: Correlation matrix
        pval_df: P-value matrix
        significance_threshold: P-value threshold for significance
    
    Returns:
        Formatted report string
    """
    if corr_df.empty:
        return "No correlation data available."
    
    report = []
    report.append("=" * 60)
    report.append("PARAMETER-OUTCOME CORRELATION MATRIX")
    report.append("=" * 60)
    report.append("")
    
    # Format correlation matrix with significance markers
    formatted_corr = corr_df.copy()
    for col in formatted_corr.columns:
        for idx in formatted_corr.index:
            corr_val = formatted_corr.loc[idx, col]
            pval = pval_df.loc[idx, col] if idx in pval_df.index and col in pval_df.columns else np.nan
            
            if np.isnan(corr_val):
                formatted_corr.loc[idx, col] = "N/A"
            elif pval < significance_threshold:
                formatted_corr.loc[idx, col] = f"{corr_val:+.3f}*"
            else:
                formatted_corr.loc[idx, col] = f"{corr_val:+.3f}"
    
    report.append(formatted_corr.to_string())
    report.append("")
    report.append(f"* = significant at p < {significance_threshold}")
    report.append("")
    
    # Highlight key findings
    report.append("-" * 40)
    report.append("KEY FINDINGS:")
    report.append("-" * 40)
    
    # Find strongest correlations with 'edge' outcome
    if 'edge' in corr_df.columns:
        edge_corrs = corr_df['edge'].dropna().sort_values(key=abs, ascending=False)
        report.append("\nStrongest correlations with EDGE:")
        for param, corr in edge_corrs.head(5).items():
            direction = "↑" if corr > 0 else "↓"
            report.append(f"  {param}: {corr:+.3f} {direction}")
    
    return "\n".join(report)
