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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr, pearsonr

import lightgbm as lgb

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

logger = logging.getLogger(__name__)


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
    periods_per_year: int = 365 * 24 * 4,  # 15m bars
) -> Dict[str, float]:
    """
    Compute comprehensive backtesting metrics.
    
    Args:
        y_prob: Predicted probabilities
        returns: Realized returns per event
        threshold: Probability threshold for taking trades
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Dict with backtest metrics
    """
    tprint_info("   Computing backtest metrics...")
    
    valid_mask = np.isfinite(y_prob) & np.isfinite(returns)
    y_prob = y_prob[valid_mask]
    returns = returns[valid_mask]
    
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
    
    if len(trade_returns) < 10:
        return {
            "sharpe_ratio": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "mean_return": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
            "n_trades": int(trade_mask.sum()),
        }
    
    # Basic metrics
    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    
    # Win rate
    win_rate = (trade_returns > 0).mean()
    
    # Sharpe ratio (annualized)
    if std_return > 1e-9:
        sharpe_ratio = (mean_return - risk_free_rate / periods_per_year) / std_return
        sharpe_ratio *= np.sqrt(periods_per_year)  # Annualize
    else:
        sharpe_ratio = np.nan
    
    # Profit factor
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss = abs(trade_returns[trade_returns < 0].sum())
    if gross_loss > 1e-9:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf if gross_profit > 0 else np.nan
    
    # Max drawdown
    cumulative_returns = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Calmar ratio
    if max_drawdown > 1e-9:
        calmar_ratio = (mean_return * periods_per_year) / max_drawdown
    else:
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
        "n_trades": int(trade_mask.sum()),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_return": float(np.sum(trade_returns)),
    }


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_simple_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> Tuple[np.ndarray, lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Train a simple LightGBM classifier with time-series CV.
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Optional sample weights
        n_splits: Number of CV splits
    
    Returns:
        Tuple of (OOF predictions, trained model, training metrics)
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
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
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
    
    # Train final model on all data
    final_model = lgb.LGBMClassifier(**params)
    if sample_weights is not None:
        final_model.fit(X, y, sample_weight=sample_weights)
    else:
        final_model.fit(X, y)
    
    metrics = {
        "model_type": "simple_lgbm",
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
    }
    
    tprint_success(f"   ✅ Simple LGBM: Mean AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
    
    return oof_probs, final_model, metrics


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
) -> Tuple[np.ndarray, LogisticRegression, Dict[str, Any]]:
    """
    Train a Logistic Regression classifier with time-series CV.
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Optional sample weights
        n_splits: Number of CV splits
    
    Returns:
        Tuple of (OOF predictions, trained model, training metrics)
    """
    tprint_info("📈 Training Logistic Regression...")
    
    oof_probs = np.full(len(y), np.nan)
    fold_aucs = []
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
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
    
    # Train final model on all data
    final_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1)
    if sample_weights is not None:
        final_model.fit(X, y, sample_weight=sample_weights)
    else:
        final_model.fit(X, y)
    
    metrics = {
        "model_type": "logistic_regression",
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
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
) -> Tuple[np.ndarray, List[lgb.LGBMClassifier], Dict[str, Any]]:
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
    
    Returns:
        Tuple of (OOF predictions, trained models list, training metrics)
    """
    tprint_info(f"🎒 Training LGBM Bagged with Diversity Defense ({n_bags} bags)...")
    
    oof_probs = np.full(len(y), np.nan)
    all_models = []
    fold_aucs = []
    diversity_metrics = []
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if len(np.unique(y_train.dropna())) < 2:
            continue
        
        fold_probs = []
        fold_models = []
        
        for bag_idx in range(n_bags):
            # Diversity: vary hyperparameters
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
                'n_jobs': -1,
                'verbose': -1,
                'random_state': 42 + bag_idx,
            }
            
            # Bootstrap sample
            n_train = len(X_train)
            boot_idx = rng.choice(n_train, size=n_train, replace=True)
            
            X_boot = X_train.iloc[boot_idx]
            y_boot = y_train.iloc[boot_idx]
            
            model = lgb.LGBMClassifier(**params)
            
            try:
                if sample_weights is not None:
                    w_boot = sample_weights[train_idx][boot_idx]
                    model.fit(X_boot, y_boot, sample_weight=w_boot)
                else:
                    model.fit(X_boot, y_boot)
                
                probs = model.predict_proba(X_val)[:, 1]
                fold_probs.append(probs)
                fold_models.append(model)
            except Exception as e:
                tprint_warning(f"      Bag {bag_idx} failed: {e}")
                continue
        
        if fold_probs:
            # Average predictions across bags
            mean_probs = np.mean(fold_probs, axis=0)
            oof_probs[val_idx] = mean_probs
            
            # Compute diversity metric (average pairwise disagreement)
            if len(fold_probs) >= 2:
                pairwise_corrs = []
                for i in range(len(fold_probs)):
                    for j in range(i + 1, len(fold_probs)):
                        corr, _ = pearsonr(fold_probs[i], fold_probs[j])
                        pairwise_corrs.append(corr)
                diversity = 1 - np.mean(pairwise_corrs)  # Higher = more diverse
                diversity_metrics.append(diversity)
            
            if len(np.unique(y_val)) >= 2:
                auc = roc_auc_score(y_val, mean_probs)
                fold_aucs.append(auc)
                tprint_info(f"      Fold {fold_idx + 1}: AUC = {auc:.4f} (diversity={diversity:.3f})")
        
        all_models.extend(fold_models)
    
    metrics = {
        "model_type": "lgbm_bagged_diversity",
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else np.nan,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else np.nan,
        "n_folds": len(fold_aucs),
        "n_bags": n_bags,
        "mean_diversity": float(np.mean(diversity_metrics)) if diversity_metrics else np.nan,
        "total_models": len(all_models),
    }
    
    tprint_success(
        f"   ✅ LGBM Bagged: Mean AUC = {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}, "
        f"Diversity = {metrics['mean_diversity']:.3f}"
    )
    
    return oof_probs, all_models, metrics


# ============================================================================
# COMPREHENSIVE POST-HPO EVALUATION
# ============================================================================

def run_post_hpo_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    sample_weights: Optional[np.ndarray] = None,
    n_splits: int = 5,
    n_bags: int = 10,
    probability_threshold: float = 0.5,
    symbol: str = "UNKNOWN",
    timeframe: str = "15m",
    direction: str = "long",
    save_artifacts: bool = True,
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
        probability_threshold: Threshold for backtest signal
        symbol: Trading symbol
        timeframe: Timeframe
        direction: Trading direction
        save_artifacts: Whether to save artifacts to disk
    
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
    
    if sample_weights is not None:
        # Align weights
        if len(sample_weights) == len(y):
            pass
        else:
            sample_weights = None
    
    # Clean data
    valid_mask = ~y.isna()
    X_clean = X.loc[valid_mask].fillna(0)
    y_clean = y.loc[valid_mask]
    returns_clean = returns_arr[valid_mask.values]
    
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
            X_clean, y_clean, weights_clean, n_splits
        )
        results["models"]["simple_lgbm"] = {
            "oof_probs": lgbm_probs,
            "training_metrics": lgbm_metrics,
        }
    except Exception as e:
        tprint_error(f"   ❌ Simple LGBM failed: {e}")
        lgbm_probs = None
    
    # 2. Logistic Regression
    try:
        lr_probs, lr_model, lr_metrics = train_logistic_regression(
            X_clean, y_clean, weights_clean, n_splits
        )
        results["models"]["logistic_regression"] = {
            "oof_probs": lr_probs,
            "training_metrics": lr_metrics,
        }
    except Exception as e:
        tprint_error(f"   ❌ Logistic Regression failed: {e}")
        lr_probs = None
    
    # 3. LGBM Bagged with Diversity
    try:
        bagged_probs, bagged_models, bagged_metrics = train_lgbm_bagged_diversity(
            X_clean, y_clean, weights_clean, n_splits, n_bags
        )
        results["models"]["lgbm_bagged_diversity"] = {
            "oof_probs": bagged_probs,
            "training_metrics": bagged_metrics,
        }
    except Exception as e:
        tprint_error(f"   ❌ LGBM Bagged failed: {e}")
        bagged_probs = None
    
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
        
        # Calibration metrics
        calib_metrics = compute_calibration_metrics(y_clean.values, oof_probs)
        model_data["calibration_metrics"] = calib_metrics
        
        # SNR diagnostics
        snr_metrics = compute_snr_diagnostics(
            y_clean.values, oof_probs, returns_clean, probability_threshold
        )
        model_data["snr_metrics"] = snr_metrics
        
        # Backtest metrics
        backtest_metrics = compute_backtest_metrics(
            oof_probs, returns_clean, probability_threshold
        )
        model_data["backtest_metrics"] = backtest_metrics
        
        # Classification metrics
        valid_probs = ~np.isnan(oof_probs)
        if valid_probs.sum() > 50:
            y_pred = (oof_probs[valid_probs] >= probability_threshold).astype(int)
            y_true_valid = y_clean.values[valid_probs]
            
            model_data["classification_metrics"] = {
                "accuracy": float(accuracy_score(y_true_valid, y_pred)),
                "precision": float(precision_score(y_true_valid, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_valid, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_valid, y_pred, zero_division=0)),
            }
        
        # Print summary
        train_auc = model_data["training_metrics"].get("mean_auc", np.nan)
        sharpe = backtest_metrics.get("sharpe_ratio", np.nan)
        snr = snr_metrics.get("snr_positive", np.nan)
        ic = snr_metrics.get("information_coefficient", np.nan)
        
        tprint_info(f"   AUC: {train_auc:.4f} | Sharpe: {sharpe:.4f} | SNR: {snr:.4f} | IC: {ic:.4f}")
    
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
            }
        
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix between HPO parameters and outcome metrics.
    
    Args:
        candidate_pool: List of candidate configurations from HPO
        param_keys: List of parameter keys to include
        outcome_keys: List of outcome keys to include
    
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
        
        # Extract outcomes
        for k in ['edge', 'auc', 'mean_auc', 'learnability', 'profitability', 
                  'sharpe_pos', 'combined', 'ic', 'calibration_score']:
            if k in c:
                try:
                    row[f"outcome_{k}"] = float(c[k])
                except (ValueError, TypeError):
                    continue
        
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
