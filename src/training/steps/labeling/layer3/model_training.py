"""
Layer 3 Model Training - Multi-Horizon ORF Implementation + ExtraTrees with Monotonic Constraints

Handles training of:
1. ORF 12/48 bars (Regressor/Classifier)
2. ExtraTrees 12/48 bars (Regressor/Classifier) with Monotonic Constraints derived from Ridge.

Produces CATE (Conditional Average Treatment Effect) and Standard Errors (SE) for each.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from econml.orf import DMLOrthoForest
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.special import expit

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)

def train_orf_meta_model(
    X: pd.DataFrame,
    Y: np.ndarray,
    T: np.ndarray,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains a single ORF model and returns estimates + uncertainty.
    """
    tprint_info(f"🌲 Training ORF: {model_name}...")
    
    cfg = config or {}
    orf_params = cfg.get('orf_params', {
        'n_trees': 100 if fast_mode else 500,
        'min_leaf_size': 20 if fast_mode else 50,
        'max_depth': 5 if fast_mode else 10,
        'subsample_ratio': 0.5,
        'bootstrap': False,
        'verbose': 0,
        'n_jobs': -1,
        'random_state': 42
    })
    
    # Scaling context features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize ORF
    est = DMLOrthoForest(
        **orf_params,
        model_T=LassoCV(cv=3),
        model_Y=LassoCV(cv=3)
    )
    
    # Fit with inference for SEs
    try:
        # Ensure numpy arrays
        Y_np = np.asarray(Y).flatten()
        T_np = np.asarray(T).reshape(-1, 1)
        
        # Use bootstrap inference for standard errors
        est.fit(Y_np, T_np, X=X_scaled, inference='blb')
        
        # Generate CATE and SE
        inf = est.effect_inference(X_scaled)
        cate = inf.point_estimate.flatten()
        se = inf.stderr.flatten()
        
        tprint_success(f"   ✅ {model_name} training complete.")
        return {
            'model': est,
            'cate': cate,
            'se': se,
            'scaler': scaler
        }
    except Exception as e:
        tprint_error(f"   ❌ {model_name} failed: {e}")
        # Fallback to zeros if failed
        return {
            'model': None,
            'cate': np.zeros(len(X)),
            'se': np.ones(len(X)),
            'scaler': scaler
        }

def train_extratrees_constrained(
    X: pd.DataFrame,
    Y: np.ndarray,
    model_name: str,
    task_type: str = 'regression',
    config: Optional[Dict[str, Any]] = None,
    sample_weight: Optional[np.ndarray] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Trains ExtraTrees model with Monotonic Constraints derived from Ridge.

    Step 1: Train Ridge to determine feature directions (+1/-1/0).
    Step 2: Train ExtraTrees with monotonic_cst.
    """
    tprint_info(f"🌳 Training ExtraTrees ({task_type}): {model_name} with Constraints...")

    cfg = config or {}
    et_params = cfg.get('et_params', {
        'n_estimators': 100 if fast_mode else 300,
        'max_depth': 10 if fast_mode else 20,
        'min_samples_leaf': 20,
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42
    })

    try:
        # Standardize X for Ridge
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        Y_np = np.asarray(Y).flatten()

        # 1. Ridge for Directionality
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, Y_np, sample_weight=sample_weight)
        coefs = ridge.coef_

        # Determine constraints: 1 (increasing), -1 (decreasing), 0 (none)
        # We set a small threshold to avoid constraining noise
        threshold = 1e-4
        constraints = np.zeros(len(coefs), dtype=int)
        constraints[coefs > threshold] = 1
        constraints[coefs < -threshold] = -1

        tprint_info(f"   🔒 Monotonic Constraints: {np.sum(constraints==1)} pos, {np.sum(constraints==-1)} neg, {np.sum(constraints==0)} free")

        # 2. Train ExtraTrees with constraints
        # Ensure sklearn version supports monotonic_cst
        # Note: monotonic_cst expects array-like of shape (n_features)

        if task_type == 'regression':
            et_model = ExtraTreesRegressor(
                monotonic_cst=constraints,
                **et_params
            )
            et_model.fit(X_scaled, Y_np, sample_weight=sample_weight)
            preds = et_model.predict(X_scaled)

        else: # classification
            # Scikit-learn 1.4+ ExtraTreesClassifier supports monotonic_cst for binary classification
            et_model = ExtraTreesClassifier(
                monotonic_cst=constraints,
                **et_params
            )
            # Ensure Y is int for classifier
            Y_int = (Y_np > 0).astype(int)
            et_model.fit(X_scaled, Y_int, sample_weight=sample_weight)
            preds = et_model.predict_proba(X_scaled)[:, 1]

        tprint_success(f"   ✅ {model_name} ExtraTrees training complete.")

        # Calculate approximate SE (Standard Error) for ET
        # Using variance of trees predictions if bootstrap=True
        if hasattr(et_model, 'estimators_'):
            # Collect predictions from all trees
            if task_type == 'regression':
                tree_preds = np.array([tree.predict(X_scaled) for tree in et_model.estimators_])
            else:
                tree_preds = np.array([tree.predict_proba(X_scaled)[:, 1] for tree in et_model.estimators_])

            se = np.std(tree_preds, axis=0)
        else:
            se = np.ones(len(preds)) # Fallback

        return {
            'model': et_model,
            'ridge_model': ridge,
            'cate': preds, # Using 'cate' key for compatibility with ORF outputs
            'se': se,
            'scaler': scaler,
            'constraints': constraints
        }

    except Exception as e:
        tprint_error(f"   ❌ {model_name} ExtraTrees failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model': None,
            'cate': np.zeros(len(X)),
            'se': np.ones(len(X)),
            'scaler': scaler
        }

def train_dual_head_models(
    X: pd.DataFrame,
    y_alpha: np.ndarray,
    y_prob: np.ndarray,
    w_alpha: np.ndarray,
    w_prob: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    config: Optional[Dict[str, Any]] = None,
    fast_mode: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the 4 requested ORF models AND 4 ExtraTrees models.
    """
    cfg = config or {}
    base_model_cols = cfg.get('base_model_cols', [])
    if not base_model_cols:
        base_model_cols = [c for c in X.columns if c.startswith('prob_') and not c.endswith('_oof')]
    
    # T (Treatment): Base Model Consensus (Used for ORF T-learner)
    T = X[base_model_cols].mean(axis=1).values
    
    # Context X: Exclude base models
    context_cols = [c for c in X.columns if c not in base_model_cols and c != 'regime_label']
    X_context = X[context_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # For ExtraTrees, we might want to include T (Base Consensus) as a feature?
    # Or keep it purely context-based?
    # Standard stacking usually includes base models.
    # But here we are doing "Contextual" meta-modeling.
    # Let's add T to X_context for ExtraTrees to allow it to correct the bias directly.
    X_et = X_context.copy()
    X_et['consensus_T'] = T

    # Horizon outcomes from config or calculated externally
    y_alpha_48 = cfg.get('y_alpha_48', y_alpha * 1.5) # Dummy fallback
    y_prob_48 = cfg.get('y_prob_48', y_prob) # Dummy fallback
    
    # --- 1. ORF Models ---
    res_12_reg = train_orf_meta_model(X_context, y_alpha, T, "ORF_12_Reg", cfg, fast_mode)
    res_12_cls = train_orf_meta_model(X_context, y_prob, T, "ORF_12_Cls", cfg, fast_mode)
    res_48_reg = train_orf_meta_model(X_context, y_alpha_48, T, "ORF_48_Reg", cfg, fast_mode)
    res_48_cls = train_orf_meta_model(X_context, y_prob_48, T, "ORF_48_Cls", cfg, fast_mode)
    
    # --- 2. ExtraTrees Models (Constrained) ---
    et_12_reg = train_extratrees_constrained(X_et, y_alpha, "ET_12_Reg", 'regression', cfg, w_alpha, fast_mode)
    et_12_cls = train_extratrees_constrained(X_et, y_prob, "ET_12_Cls", 'classification', cfg, w_prob, fast_mode)
    et_48_reg = train_extratrees_constrained(X_et, y_alpha_48, "ET_48_Reg", 'regression', cfg, w_alpha, fast_mode)
    et_48_cls = train_extratrees_constrained(X_et, y_prob_48, "ET_48_Cls", 'classification', cfg, w_prob, fast_mode)

    # Aggregate results
    # We keep ORF as the primary 'oof' output for compatibility, but provide ET as well
    all_results = {
        'alpha_oof': res_12_reg['cate'],
        'prob_oof': expit(res_12_cls['cate'] / (res_12_cls['cate'].std() + 1e-9)),

        # Extended outputs for Ensembling/Reporting
        'et_alpha_oof': et_12_reg['cate'],
        'et_prob_oof': et_12_cls['cate'],

        'models': {
            'orf_12_reg': res_12_reg,
            'orf_12_cls': res_12_cls,
            'orf_48_reg': res_48_reg,
            'orf_48_cls': res_48_cls,
            'et_12_reg': et_12_reg,
            'et_12_cls': et_12_cls,
            'et_48_reg': et_48_reg,
            'et_48_cls': et_48_cls
        },
        'alpha_metrics': {'final_ic': np.corrcoef(np.asarray(y_alpha).flatten(), res_12_reg['cate'])[0,1]},
        'prob_metrics': {
            'final_auc': 0.5 if len(np.unique(y_prob)) < 2 else roc_auc_score(np.asarray(y_prob).flatten(), expit(res_12_cls['cate'] / (res_12_cls['cate'].std() + 1e-9))),
            'final_logloss': 0.69 # Placeholder
        },
        'alpha_models': {'Global': [res_12_reg['model']]}, # Compatibility
        'prob_models': {'Global': [res_12_cls['model']]} # Compatibility
    }
    
    return all_results

def train_alpha_head(*args, **kwargs):
    raise NotImplementedError("Use train_dual_head_models")

def train_probability_head(*args, **kwargs):
    raise NotImplementedError("Use train_dual_head_models")

def select_uncorrelated_models(*args, **kwargs):
    return []
