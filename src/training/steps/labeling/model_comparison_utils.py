"""
Model Comparison Utilities for Meta-Labeling.

Provides side-by-side comparison of LGBM vs XGBoost (and other models)
with focus on calibration quality, not just AUC.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import warnings

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success
except ImportError:
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)


def compare_models_side_by_side(
    X: pd.DataFrame,
    y: pd.Series,
    returns: Optional[pd.Series] = None,
    cv_folds: int = 5,
    transaction_cost: float = 0.003,
) -> Dict[str, Any]:
    """
    Compare LGBM, XGBoost, and LogReg on the same data.
    
    Evaluates models on:
    - AUC (discrimination)
    - Brier score (calibration)
    - ECE (Expected Calibration Error)
    - Monotonicity (higher prob → higher return)
    
    Args:
        X: Feature DataFrame
        y: Binary labels
        returns: Optional realized returns for monotonicity check
        cv_folds: Number of time-series CV folds
        transaction_cost: Transaction cost for net return calculations
        
    Returns:
        Dictionary with per-model metrics and recommendation.
    """
    if not SKLEARN_AVAILABLE:
        tprint_warning("sklearn not available")
        return {"error": "sklearn_not_available"}
    
    # Clean data
    valid_mask = ~y.isna()
    X_clean = X.loc[valid_mask].fillna(0)
    y_clean = y.loc[valid_mask]
    
    if returns is not None:
        returns_clean = returns.loc[valid_mask]
    else:
        returns_clean = None
    
    if len(y_clean) < 100:
        tprint_warning(f"Insufficient samples: {len(y_clean)}")
        return {"error": "insufficient_samples"}
    
    # Define models
    models = {}
    
    if LGBM_AVAILABLE:
        models["lgbm"] = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=5,
            n_estimators=100,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
            n_jobs=-1,
        )
    
    if XGB_AVAILABLE:
        models["xgboost"] = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=5,
            n_estimators=100,
            learning_rate=0.05,
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
    
    models["logreg"] = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    
    models["random_forest"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    
    # Time series CV
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    
    results = {}
    
    for model_name, model in models.items():
        tprint_info(f"Evaluating {model_name}...")
        
        oof_probs = np.zeros(len(y_clean))
        fold_aucs = []
        fold_briers = []
        
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                
                # Skip if only one class
                if len(np.unique(y_train)) < 2:
                    continue
                
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_val)[:, 1]
                oof_probs[val_idx] = probs
                
                # Fold metrics
                if len(np.unique(y_val)) >= 2:
                    fold_aucs.append(roc_auc_score(y_val, probs))
                    fold_briers.append(brier_score_loss(y_val, probs))
            
            # Overall metrics
            valid_oof = oof_probs > 0
            if valid_oof.sum() < 50:
                continue
            
            y_eval = y_clean.iloc[valid_oof]
            probs_eval = oof_probs[valid_oof]
            
            auc = roc_auc_score(y_eval, probs_eval)
            brier = brier_score_loss(y_eval, probs_eval)
            
            # ECE
            try:
                fraction_pos, mean_pred = calibration_curve(y_eval, probs_eval, n_bins=10)
                ece = np.mean(np.abs(fraction_pos - mean_pred))
            except Exception:
                ece = np.nan
            
            # Monotonicity check
            monotonicity_score = np.nan
            if returns_clean is not None:
                try:
                    returns_eval = returns_clean.iloc[valid_oof].values
                    monotonicity_score = _compute_monotonicity_score(probs_eval, returns_eval)
                except Exception:
                    pass
            
            results[model_name] = {
                "auc": float(auc),
                "auc_std": float(np.std(fold_aucs)) if fold_aucs else np.nan,
                "brier": float(brier),
                "brier_std": float(np.std(fold_briers)) if fold_briers else np.nan,
                "ece": float(ece) if not np.isnan(ece) else None,
                "monotonicity_score": float(monotonicity_score) if not np.isnan(monotonicity_score) else None,
                "n_folds": len(fold_aucs),
            }
            
            tprint_success(f"  {model_name}: AUC={auc:.3f}, Brier={brier:.3f}, ECE={ece:.3f if not np.isnan(ece) else 'N/A'}")
            
        except Exception as e:
            tprint_warning(f"  {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}
    
    # Select best model based on weighted score
    # Prioritize: Brier (calibration) > Monotonicity > AUC
    best_model = _select_best_model(results, transaction_cost)
    
    return {
        "models": results,
        "best_model": best_model,
        "n_samples": len(y_clean),
        "recommendation": _generate_recommendation(results, best_model),
    }


def _compute_monotonicity_score(probs: np.ndarray, returns: np.ndarray, n_bins: int = 10) -> float:
    """Compute monotonicity score: 1.0 = perfect monotonicity."""
    try:
        bin_edges = np.percentile(probs, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        bin_indices = np.digitize(probs, bin_edges[1:])
        
        bin_returns = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() >= 2:
                bin_returns.append(returns[mask].mean())
            else:
                bin_returns.append(np.nan)
        
        # Count violations
        valid_returns = [r for r in bin_returns if not np.isnan(r)]
        if len(valid_returns) < 2:
            return 0.0
        
        violations = sum(1 for i in range(len(valid_returns) - 1) if valid_returns[i + 1] < valid_returns[i])
        return 1.0 - (violations / (len(valid_returns) - 1))
    except Exception:
        return 0.0


def _select_best_model(results: Dict, transaction_cost: float) -> str:
    """Select best model prioritizing calibration over discrimination."""
    scores = {}
    
    for model_name, metrics in results.items():
        if "error" in metrics:
            continue
        
        auc = metrics.get("auc", 0.5)
        brier = metrics.get("brier", 0.25)
        mono = metrics.get("monotonicity_score", 0.5) or 0.5
        
        # Weighted score (higher is better)
        # - Brier: lower is better, so use (0.25 - brier) normalized
        # - Mono: direct
        # - AUC: direct
        brier_score = max(0, (0.25 - brier) / 0.25)  # 0.25 is baseline
        auc_score = (auc - 0.5) / 0.5  # Normalize AUC contribution
        
        # Weights: Calibration 50%, Monotonicity 30%, AUC 20%
        composite = 0.5 * brier_score + 0.3 * mono + 0.2 * auc_score
        scores[model_name] = composite
    
    if not scores:
        return "lgbm"  # Fallback
    
    return max(scores.items(), key=lambda x: x[1])[0]


def _generate_recommendation(results: Dict, best_model: str) -> str:
    """Generate human-readable recommendation."""
    if best_model not in results or "error" in results.get(best_model, {}):
        return "Unable to determine best model. Use LGBM as default."
    
    metrics = results[best_model]
    auc = metrics.get("auc", 0)
    brier = metrics.get("brier", 1)
    mono = metrics.get("monotonicity_score", 0)
    
    rec = f"Recommended: {best_model.upper()} (AUC={auc:.3f}, Brier={brier:.3f}"
    if mono:
        rec += f", Mono={mono:.2f}"
    rec += ")"
    
    # Add comparison note
    for name, m in results.items():
        if name != best_model and "error" not in m:
            other_auc = m.get("auc", 0)
            other_brier = m.get("brier", 1)
            if other_auc > auc + 0.02 and other_brier > brier:
                rec += f"\n  Note: {name} has higher AUC but worse calibration."
    
    return rec
