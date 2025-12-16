"""Meta-Labeling HPO Layer 3: Model Hyperparameters Optimization.

This module handles Layer 3 of the hierarchical HPO process:
- Optimizes LightGBM model hyperparameters
- Uses cross-validation with time-series aware splits
- Supports isotonic calibration for probability outputs
- Computes Sharpe-based utility with various penalties

Layer 3 focuses on:
1. LightGBM hyperparameters (n_estimators, learning_rate, depth, etc.)
2. Regularization parameters (reg_alpha, reg_lambda, min_child_samples)
3. Calibration and density gates
4. Recency weighting
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# Import shared utilities
try:
    from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
        _write_hpo_stage_report,
        _sanitize_json_value,
        _soft_sharpe_scale,
        calculate_hpo_utility,
        tprint_info,
        tprint_success,
        tprint_warning,
    )
except ImportError:
    # Fallback for standalone testing
    def tprint_info(msg: str) -> None:
        print(f"[INFO] {msg}")
    def tprint_success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")
    def tprint_warning(msg: str) -> None:
        print(f"[WARNING] {msg}")
    def _write_hpo_stage_report(**kwargs) -> Dict[str, Any]:
        return {}
    def _sanitize_json_value(obj: Any) -> Any:
        return obj
    def _soft_sharpe_scale(raw_sharpe: float, scale: float = 30.0) -> float:
        return float(np.arcsinh(raw_sharpe / scale) * scale)
    def calculate_hpo_utility(**kwargs) -> Dict[str, Any]:
        return {"utility": 0.0}


def get_layer3_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get the Layer 3 search space for model hyperparameters.
    
    Args:
        config: HPO configuration
        
    Returns:
        Dictionary defining the search space for Layer 3
    """
    # Get config-based limits
    try:
        n_estimators_max = int(config.get("layer3_n_estimators_max", 500))
    except Exception:
        n_estimators_max = 500
    
    try:
        max_depth_max = int(config.get("layer3_max_depth_max", 10))
    except Exception:
        max_depth_max = 10
    
    return {
        # Core LightGBM parameters
        "n_estimators": {"type": "int", "low": 50, "high": n_estimators_max},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": max_depth_max},
        "num_leaves": {"type": "int", "low": 8, "high": 128},
        "min_child_samples": {"type": "int", "low": 10, "high": 200},
        
        # Regularization parameters
        "reg_alpha": {"type": "float", "low": 1e-6, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-6, "high": 10.0, "log": True},
        "min_split_gain": {"type": "float", "low": 0.0, "high": 1.0},
        
        # Sampling parameters
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        
        # Recency weighting (if enabled)
        "recency_decay_lambda": {"type": "float", "low": 0.0, "high": 0.02},
    }


def get_lgbm_params_from_trial(
    trial_params: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract LightGBM parameters from trial parameters.
    
    Args:
        trial_params: Trial parameters from optimizer
        config: HPO configuration
        
    Returns:
        Dictionary of LightGBM parameters
    """
    lgbm_keys = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "subsample",
        "colsample_bytree",
    }
    
    lgbm_params = {
        k: v for k, v in trial_params.items() 
        if k in lgbm_keys
    }
    
    # Add fixed parameters
    lgbm_params.update({
        "boosting_type": "gbdt",
        "objective": "binary",
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    })
    
    return lgbm_params


def compute_layer3_cv_metrics(
    *,
    model_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    returns: np.ndarray,
    n_cv_folds: int = 5,
    prob_threshold: float = 0.5,
    direction: str = "long",
    days_span: float = 365.0,
    config: Dict[str, Any] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute Layer 3 cross-validated metrics.
    
    Args:
        model_params: LightGBM model parameters
        X: Feature matrix
        y: Target labels
        sample_weights: Sample weights
        returns: Realized returns
        n_cv_folds: Number of CV folds
        prob_threshold: Probability threshold for trading
        direction: Trading direction
        days_span: Number of days in dataset
        config: HPO configuration
        
    Returns:
        Tuple of (utility, metrics_dict)
    """
    if lgb is None:
        return 0.0, {"fail_reason": "lightgbm_not_available"}
    
    if config is None:
        config = {}
    
    try:
        # Create LightGBM model
        lgbm_params = get_lgbm_params_from_trial(model_params, config)
        model = lgb.LGBMClassifier(**lgbm_params)
        
        # Create CV splits
        kf = TimeSeriesSplit(n_splits=n_cv_folds)
        splits = list(kf.split(X))
        
        fold_aucs: List[float] = []
        fold_sharpes: List[float] = []
        oof_predictions = np.full(len(y), np.nan, dtype=float)
        
        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            w_tr = sample_weights[tr_idx].copy()
            
            # Check for single-class train split
            y_tr_unique = np.unique(y_tr.values)
            if len(y_tr_unique) < 2:
                # Use prior-based predictions
                prior = float(np.mean(y_tr.values.astype(float)))
                prior = float(np.clip(prior, 0.0, 1.0)) if np.isfinite(prior) else 0.5
                preds = np.full(len(X_te), prior, dtype=float)
                oof_predictions[te_idx] = preds
                continue
            
            # Train model
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            
            # Get predictions
            proba = model.predict_proba(X_te)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                preds = proba[:, 1]
            elif proba.ndim == 2 and proba.shape[1] == 1:
                preds = proba[:, 0]
            else:
                preds = proba
            
            preds = np.asarray(preds, dtype=float)
            oof_predictions[te_idx] = preds
            
            # Compute fold AUC
            try:
                if len(np.unique(y_te.values)) >= 2:
                    fold_auc = float(roc_auc_score(y_te, preds))
                    fold_aucs.append(fold_auc)
            except Exception:
                pass
            
            # Compute fold Sharpe
            try:
                take_mask = preds > prob_threshold
                ret_te = returns[te_idx]
                taken_returns = ret_te[take_mask]
                if len(taken_returns) >= 10:
                    mean_ret = float(np.mean(taken_returns))
                    std_ret = float(np.std(taken_returns))
                    if std_ret > 1e-8:
                        sharpe = mean_ret / std_ret * np.sqrt(252)
                        fold_sharpes.append(_soft_sharpe_scale(sharpe))
            except Exception:
                pass
        
        # Compute aggregate metrics
        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        std_auc = float(np.std(fold_aucs)) if len(fold_aucs) > 1 else 0.0
        mean_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        std_sharpe = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
        
        # Compute overall statistics
        valid_oof = ~np.isnan(oof_predictions)
        n_valid = int(np.sum(valid_oof))
        
        if n_valid < 50:
            return 0.0, {"fail_reason": "too_few_valid_oof_predictions", "n_valid": n_valid}
        
        # Compute trade statistics
        take_mask = oof_predictions > prob_threshold
        n_trades = int(np.sum(take_mask))
        trades_per_day = float(n_trades) / max(days_span, 1.0)
        
        if n_trades < 10:
            return 0.0, {
                "fail_reason": "too_few_trades",
                "n_trades": n_trades,
                "mean_auc": mean_auc,
            }
        
        # Compute taken trade returns
        taken_returns = returns[take_mask]
        mean_return = float(np.mean(taken_returns))
        
        # Compute utility using Sharpe-based formula
        utility_result = calculate_hpo_utility(
            mean_auc=mean_auc,
            sharpe_mean=mean_sharpe,
            sharpe_std=std_sharpe,
            trades_per_day=trades_per_day,
            win_rate=float(np.mean(taken_returns > 0)),
            mean_return=mean_return,
            config=config,
        )
        
        utility = float(utility_result.get("utility", 0.0))
        
        metrics = {
            "utility": utility,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "sharpe_mean": mean_sharpe,
            "sharpe_std": std_sharpe,
            "n_trades": n_trades,
            "trades_per_day": trades_per_day,
            "mean_return": mean_return,
            "n_folds": len(fold_aucs),
        }
        
        return utility, metrics
        
    except Exception as e:
        return 0.0, {"fail_reason": str(e)}


def save_layer3_results(
    *,
    best_model_params: Dict[str, Any],
    best_l3_utility: float,
    l3_metrics: Dict[str, Any],
    layer3_search_space: Dict[str, Any],
    l3_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    outcomes_dir: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
) -> Tuple[Optional[Path], Optional[Path], Dict[str, Any]]:
    """Save Layer 3 optimization results.
    
    Args:
        best_model_params: Best model parameters
        best_l3_utility: Best Layer 3 utility
        l3_metrics: Layer 3 metrics dictionary
        layer3_search_space: Search space definition
        l3_history: Optimization history
        config: HPO configuration
        outcomes_dir: Directory for saving results
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe string
        direction: Trading direction
        
    Returns:
        Tuple of (params_path, history_path, stage_report)
    """
    timestamp = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Persist Layer 3 params
    l3_path: Optional[Path] = None
    try:
        l3_path = outcomes_dir / f"hpo_layer3_best_params_{symbol}_{timeframe}_{timestamp}.json"
        l3_payload = {
            "best_params": _sanitize_json_value(best_model_params),
            "best_utility": best_l3_utility,
            "timestamp": timestamp,
        }
        l3_path.parent.mkdir(parents=True, exist_ok=True)
        with open(l3_path, "w") as f:
            json.dump(l3_payload, f, indent=2, default=str)
        tprint_info(f"   💾 Saved Layer 3 best params to {l3_path}")
    except Exception as l3_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 3 params: {l3_exc}")
    
    # Save Layer 3 History
    l3_history_path: Optional[Path] = None
    try:
        l3_history_path = outcomes_dir / f"hpo_layer3_history_{symbol}_{timeframe}_{timestamp}.json"
        with open(l3_history_path, "w") as f:
            json.dump(_sanitize_json_value(l3_history), f, indent=2, default=str)
        tprint_info(f"   💾 Saved Layer 3 history to {l3_history_path}")
    except Exception as e:
        tprint_warning(f"   ⚠️ Failed to save Layer 3 history: {e}")
    
    # Save Layer 3 trials CSV
    l3_trials_path: Optional[Path] = None
    try:
        if l3_history and len(l3_history) > 0:
            l3_trials_path = outcomes_dir / f"hpo_layer3_trials_{symbol}_{timeframe}_{timestamp}.csv"
            trial_rows = []
            for trial in l3_history:
                if isinstance(trial, dict):
                    row = {
                        "trial_number": trial.get("trial_number"),
                        "value": trial.get("value"),
                        **trial.get("params", {}),
                    }
                    trial_rows.append(row)
            if trial_rows:
                pd.DataFrame(trial_rows).to_csv(l3_trials_path, index=False)
                tprint_info(f"   💾 Saved Layer 3 trial metrics to {l3_trials_path}")
    except Exception as l3_trials_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 3 trial metrics: {l3_trials_exc}")
    
    # Write stage report
    stage_report: Dict[str, Any] = {}
    try:
        stage_report = _write_hpo_stage_report(
            outcomes_dir=outcomes_dir,
            run_timestamp=timestamp,
            stage_id="layer3_model",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            best_params=_sanitize_json_value(dict(best_model_params)) if isinstance(best_model_params, dict) else {},
            metrics={
                "best_utility": best_l3_utility,
                **(l3_metrics if isinstance(l3_metrics, dict) else {}),
            },
            search_space=layer3_search_space,
            trials_csv_path=l3_trials_path,
            history_json_path=l3_history_path,
        )
    except Exception as l3_report_exc:
        tprint_warning(f"   ⚠️ Failed to write Layer 3 report: {l3_report_exc}")
    
    return l3_path, l3_history_path, stage_report
