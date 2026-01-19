"""
Layer 2.5 Chaser - Enhanced Regime-Aware Residual Learner

This module implements the Layer 2.5 Chaser model, designed to capture non-linear alpha
in the residuals of the causal anchor models. It utilizes a Teacher-Student architecture
where a robust linear teacher (BayesianRidge/Huber) provides a baseline, and non-linear
students (XGBoost, CatBoost, LightGBM, ExtraTrees) learn to correct the errors (residuals).

Key Features:
1. Regime-Aware Training: Trains separate specialist ensembles per GMM regime (soft-weighted).
2. Dual-Feature Engineering: Uses both FracDiff (stationary) and Residualized (detrended) features.
3. Robust Teacher-Student: BayesianRidge OOF teacher -> Residual/Margin-correcting Students.
4. Volatility-Normalized Labels: Integrates with AFML labeling standards.
5. Multi-Model Ensemble: Diversified student pool with correlation-based pruning.
6. Weak Huber Constraints: Applies weak monotonic/interaction constraints from Huber analysis.
7. Strong Regularization: User-specified regularization parameters for robust training.
8. Meta-Learner Ready: Outputs teacher baseline and chaser correction signals for stacking.
9. De Prado Feature Selection: Reduces feature set and collinearity using MDI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import logging
import gc

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# Import PurgedKFold
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    PURGED_KFOLD_AVAILABLE = True
except ImportError:
    PURGED_KFOLD_AVAILABLE = False
    from sklearn.model_selection import TimeSeriesSplit

# Import Huber constraint utilities
try:
    from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs, get_huber_tier_config
    HUBER_AVAILABLE = True
except ImportError:
    HUBER_AVAILABLE = False

import xgboost as xgb

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.fracdiff import FracDiffTransformer
from src.training.steps.labeling.feature_engineering_utils import apply_layer2_price_processing
from src.training.steps.labeling.de_prado_feature_engine import DePradoFeatureEngine

# -----------------------------
# Utilities
# -----------------------------
def robust_stats(x: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Calculate median and robust sigma (MAD-based)."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    sigma = 1.4826 * mad
    return med, sigma

def clip_with_stats(x: np.ndarray, med: float, sigma: float, k: float = 3.0) -> np.ndarray:
    """Clip data based on provided robust stats."""
    return np.clip(x, med - k * sigma, med + k * sigma)

def robust_sigma(x: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate robust standard deviation using MAD."""
    _, s = robust_stats(x, eps)
    return s

def winsorize(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Winsorize data based on robust sigma."""
    med, s = robust_stats(x)
    return clip_with_stats(x, med, s, k)

def normalize_weights(w: np.ndarray, clip_percentile: float = 0.99) -> np.ndarray:
    """
    Normalize weights to mean 1.0 with robust clipping to prevent long tails.
    1. Clip at high percentile to remove extreme outliers.
    2. Normalize to mean 1.0.
    """
    w = np.asarray(w, dtype=np.float64)
    # Clip extreme weights to prevent optimizer instability
    if clip_percentile < 1.0:
        limit = np.quantile(w, clip_percentile)
        w = np.minimum(w, limit)

    m = np.mean(w)
    return w if (not np.isfinite(m) or m <= 0) else w / m

def uncertainty_to_chaser_weight(std: np.ndarray, clip=(0.5, 2.0)) -> np.ndarray:
    """Convert prediction uncertainty to sample weights (upweight uncertain samples)."""
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
    med = np.median(std)
    w = np.sqrt(std / (med + 1e-12))
    w = np.clip(w, clip[0], clip[1])
    return normalize_weights(w)

def sanity_check_uncertainty(y: np.ndarray, p_teacher_oof: np.ndarray, std_oof: np.ndarray) -> float:
    """Check: higher teacher uncertainty should correlate with larger errors."""
    y = np.asarray(y, dtype=np.int32)
    p = np.clip(np.asarray(p_teacher_oof, dtype=np.float64), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))  # per-sample logloss
    corr = np.corrcoef(ll, std_oof)[0, 1]
    return float(corr if np.isfinite(corr) else 0.0)

def combine_weights(base_weight: np.ndarray | None, extra: np.ndarray) -> np.ndarray:
    """Combine base sample weights with extra weights."""
    if base_weight is None:
        return normalize_weights(extra)
    return normalize_weights(np.asarray(base_weight, dtype=np.float64) * extra)

def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probability to log-odds (logit)."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))

# -----------------------------
# Teacher (BayesianRidge) OOF
# -----------------------------
@dataclass
class TeacherOOF:
    scaler: RobustScaler
    mu_oof: np.ndarray          # (n,)
    std_oof: np.ndarray         # (n,)
    # For classifier mode
    calibrator: LogisticRegression | None = None
    p_oof: np.ndarray | None = None
    margin_oof: np.ndarray | None = None
    
    # Store the fitted model for production inference
    model: BayesianRidge | None = None

def fit_bayes_teacher_oof(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_splits: int = 5,
    winsor_k: float = 4.0,
    is_classifier: bool = False,
    index: pd.Index | None = None
) -> TeacherOOF:
    """
    OOF BayesianRidge teacher for:
      - regression: mu/std directly
      - classification: mu/std score -> calibrate mu->p (Platt) -> margin
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X).astype(np.float64, copy=False)

    # Use PurgedKFold if available and index is provided/compatible
    if PURGED_KFOLD_AVAILABLE and index is not None:
        cv = PurgedKFoldTime(n_splits=n_splits)
        splitter = cv.split(pd.DataFrame(Xs, index=index))
    else:
        cv = TimeSeriesSplit(n_splits=n_splits)
        splitter = cv.split(Xs)

    mu_oof = np.full(Xs.shape[0], np.nan, dtype=np.float64)
    std_oof = np.full(Xs.shape[0], np.nan, dtype=np.float64)

    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    # Walk-forward CV
    for tr, va in splitter:
        X_tr, X_va = Xs[tr], Xs[va]
        y_tr = y[tr].astype(np.float64)

        # If regression on heavy-tailed targets, winsorize for the teacher
        if not is_classifier:
            # Winsorize using stats from training fold only (no leakage)
            med_tr, s_tr = robust_stats(y_tr)
            y_tr = clip_with_stats(y_tr, med_tr, s_tr, k=winsor_k)

        model = BayesianRidge()
        if w is None:
            model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr, sample_weight=w[tr])

        mu, std = model.predict(X_va, return_std=True)
        mu_oof[va] = mu
        std_oof[va] = np.maximum(std, 1e-12)

    # Fill NaNs (start of OOF) with first valid prediction or mean
    valid_mask = np.isfinite(mu_oof)
    if not valid_mask.all():
        mu_mean = np.nanmean(mu_oof) if np.any(valid_mask) else 0.0
        std_mean = np.nanmean(std_oof) if np.any(valid_mask) else 1.0
        mu_oof[~valid_mask] = mu_mean
        std_oof[~valid_mask] = std_mean

    # Train final model on full data for production
    final_model = BayesianRidge()
    y_full = y.astype(np.float64)
    if not is_classifier:
        y_full = winsorize(y_full, k=winsor_k)

    if w is None:
        final_model.fit(Xs, y_full)
    else:
        final_model.fit(Xs, y_full, sample_weight=w)

    out = TeacherOOF(scaler=scaler, mu_oof=mu_oof, std_oof=std_oof, model=final_model)

    if is_classifier:
        ok = np.isfinite(mu_oof)
        calib = LogisticRegression(solver="lbfgs", max_iter=2000)
        y_bin = y.astype(np.int32)

        if w is None:
            calib.fit(mu_oof[ok].reshape(-1, 1), y_bin[ok])
        else:
            calib.fit(mu_oof[ok].reshape(-1, 1), y_bin[ok], sample_weight=w[ok])

        p_oof = np.full_like(mu_oof, np.nan)
        p_oof[ok] = calib.predict_proba(mu_oof[ok].reshape(-1, 1))[:, 1]

        # Fill NaNs in p_oof
        p_mean = np.nanmean(p_oof) if np.any(np.isfinite(p_oof)) else 0.5
        p_oof[~ok] = p_mean

        margin_oof = prob_to_logit(p_oof)

        out.calibrator = calib
        out.p_oof = p_oof
        out.margin_oof = margin_oof

    return out

# -----------------------------
# Student Chasers
# -----------------------------
def train_chaser_student(
    X: np.ndarray,
    y: np.ndarray,
    teacher: TeacherOOF,
    base_weight: np.ndarray | None = None,
    mode: str = "regression",  # "regression" or "classification"
    winsor_resid_k: float = 3.0,
    model_type: str = "xgb", # "xgb", "lgb", "cat", "et"
    model_params: dict | None = None,
    num_boost_round: int = 1000, # Default increased to 1000
    # New parameters for weak constraints
    monotone_constraints_weak: dict | None = None,
    interaction_constraints_weak: list[list[str]] | None = None,
    huber_teacher_mu: np.ndarray | None = None,  # For teacher disagreement features
):
    """
    Train a student model (Chaser) on residuals or with margin correction.
    Uses aggressive early stopping (30 rounds) with an internal validation split.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    # Teacher-uncertainty weights (chasers upweight uncertainty, bounded)
    w_chase = uncertainty_to_chaser_weight(teacher.std_oof, clip=(0.5, 2.0))
    w_final = combine_weights(base_weight, w_chase)

    # Robust normalization of final weights (clip + mean=1.0)
    w_final = normalize_weights(w_final, clip_percentile=0.99)

    init_score = None
    baseline = None

    if mode == "regression":
        # Residual target
        # Calculate raw residuals
        r = y.astype(np.float64) - teacher.mu_oof

        # --- Internal Validation Split ---
        n_samples = len(X)
        split_idx = int(n_samples * 0.85)

        r_train = r[:split_idx]
        r_valid = r[split_idx:]

        # Calculate robust stats on TRAINING split only
        med_train, s_train = robust_stats(r_train)

        # Apply winsorization using training stats
        r_train_wins = clip_with_stats(r_train, med_train, s_train, k=winsor_resid_k)
        r_valid_wins = clip_with_stats(r_valid, med_train, s_train, k=winsor_resid_k)

        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = r_train_wins, r_valid_wins
        w_train, w_valid = w_final[:split_idx], w_final[split_idx:]

    elif mode == "classification":
        if teacher.margin_oof is None:
            raise ValueError("For classification mode, teacher.margin_oof must be available.")
        target = y.astype(np.int32)
        init_score = teacher.margin_oof.astype(np.float64)
        baseline = teacher.margin_oof.astype(np.float64)

        n_samples = len(X)
        split_idx = int(n_samples * 0.85)

        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = target[:split_idx], target[split_idx:]
        w_train, w_valid = w_final[:split_idx], w_final[split_idx:]
    else:
        raise ValueError("mode must be 'regression' or 'classification'.")

    init_score_train = init_score[:split_idx] if init_score is not None else None
    init_score_valid = init_score[split_idx:] if init_score is not None else None
    baseline_train = baseline[:split_idx] if baseline is not None else None
    baseline_valid = baseline[split_idx:] if baseline is not None else None

    # Reconstruct full target for models trained on full dataset (like ET)
    # y_train and y_valid are already winsorized/processed
    target_full = np.concatenate([y_train, y_valid])

    # --- XGBoost ---
    if model_type == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid)

        if mode == "classification":
            dtrain.set_base_margin(init_score_train)
            dvalid.set_base_margin(init_score_valid)

        default_params = {
            "eta": 0.03,
            "max_depth": 5, # Increased to 5
            "min_child_weight": 10,
            "subsample": 0.6,
            "colsample_bytree": 0.7,
            "colsample_bynode": 0.4,
            "reg_lambda": 25.0, # Decreased to 25
            "reg_alpha": 0.0,
            "gamma": 0.7, # Decreased to 0.7
            "num_parallel_tree": 15, # Random Forest behavior
            "n_jobs": -1
        }
        if mode == "regression":
            default_params["objective"] = "reg:squarederror"
            default_params["eval_metric"] = "mae"
        else:
            default_params["objective"] = "binary:logistic"
            default_params["eval_metric"] = "logloss"

        params = default_params.copy()
        if model_params:
            params.update(model_params)

        # Add weak constraints if available
        if monotone_constraints_weak is not None and mode == "classification":
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            mono_tuple = tuple(monotone_constraints_weak.get(f"f{i}", 0) for i in range(X.shape[1]))
            params["monotone_constraints"] = mono_tuple
            
        if interaction_constraints_weak is not None and mode == "classification":
            params["interaction_constraints"] = interaction_constraints_weak

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=30, # Aggressive early stopping
            verbose_eval=False
        )
        return {"model": bst, "mode": mode, "type": "xgb", "params": params}

    # --- LightGBM ---
    elif model_type == "lgb":
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=train_data)

        if mode == "classification":
            train_data.set_init_score(init_score_train)
            valid_data.set_init_score(init_score_valid)

        default_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.7,
            "subsample_freq": 1, # Added
            "colsample_bytree": 0.7,
            "colsample_bynode": 0.7, # Added feature_fraction_bynode
            "reg_lambda": 10.0,
            "min_split_gain": 0.005, # Added min_gain_to_split
            "linear_tree": True, # Added linear=true
            "path_smooth": 20,
            "extra_trees": True,
            "n_jobs": -1,
            "verbose": -1
        }
        if mode == "regression":
            default_params["objective"] = "regression"
            default_params["metric"] = "l1"
        else:
            default_params["objective"] = "binary"
            default_params["metric"] = "binary_logloss"

        params = default_params.copy()
        if model_params:
            params.update(model_params)

        # Add weak constraints if available
        if monotone_constraints_weak is not None and mode == "classification":
            params["monotone_constraints"] = monotone_constraints_weak
            
        if interaction_constraints_weak is not None and mode == "classification":
            params["interaction_constraints"] = interaction_constraints_weak

        # Callbacks for early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

        bst = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=callbacks
        )
        return {"model": bst, "mode": mode, "type": "lgb", "params": params}

    # --- CatBoost ---
    elif model_type == "cat":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")

        train_pool = cb.Pool(X_train, label=y_train, weight=w_train)
        valid_pool = cb.Pool(X_valid, label=y_valid, weight=w_valid)

        if mode == "classification":
            train_pool.set_baseline(baseline_train)
            valid_pool.set_baseline(baseline_valid)

        default_params = {
            "iterations": num_boost_round,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 20.0,
            "subsample": 0.6,
            "rsm": 0.8, # Added rsm
            "bagging_temperature": 1, # Added bagging_temperature
            "random_strength": 5.0,
            "verbose": False,
            "allow_writing_files": False,
            "early_stopping_rounds": 30 # Native parameter
        }
        if mode == "regression":
            default_params["loss_function"] = "MAE"
        else:
            default_params["loss_function"] = "Logloss"

        params = default_params.copy()
        if model_params:
            params.update(model_params)

        # Add weak constraints if available
        if monotone_constraints_weak is not None and mode == "classification":
            params["monotone_constraints"] = monotone_constraints_weak

        model = cb.CatBoost(params)
        model.fit(train_pool, eval_set=valid_pool)
        return {"model": model, "mode": mode, "type": "cat", "params": params}

    # --- ExtraTrees ---
    elif model_type == "et":
        # ExtraTrees doesn't support early stopping in the same way (sklearn)
        # We train on full set (X, y) or we could simulate it but ET is fast anyway
        # We stick to full training for ET but update parameters

        # Determine estimators from num_boost_round if passed, but cap/set to 500
        n_estimators = 500 # Reduced to 500

        if mode == "classification":
            # "residual_prob" subtype
            res_target = y.astype(float) - teacher.p_oof
            et_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                n_jobs=-1,
                max_depth=6, # Reduced to 6
                min_samples_leaf=20, # Increased to 20
                random_state=42
            )
            if model_params:
                et_model.set_params(**model_params)

            et_model.fit(X, res_target, sample_weight=w_final)
            return {"model": et_model, "mode": mode, "type": "et", "subtype": "residual_prob"}
        else:
            # Regression on residuals
            et_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                n_jobs=-1,
                max_depth=6, # Reduced to 6
                min_samples_leaf=20, # Increased to 20
                random_state=42
            )
            if model_params:
                et_model.set_params(**model_params)
            et_model.fit(X, target_full, sample_weight=w_final)
            return {"model": et_model, "mode": mode, "type": "et"}

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def predict_chaser_student(
    X: np.ndarray,
    teacher_mu: np.ndarray | None,
    teacher_margin: np.ndarray | None,
    student_artifact: dict,
) -> np.ndarray:
    """
    Make predictions using the student model and teacher baseline.
    """
    X = np.asarray(X, dtype=np.float32)
    model = student_artifact["model"]
    mode = student_artifact["mode"]
    m_type = student_artifact["type"]

    if mode == "regression":
        if teacher_mu is None:
            raise ValueError("teacher_mu required for regression prediction")

        if m_type == "xgb":
            d = xgb.DMatrix(X)
            r_hat = model.predict(d)
        elif m_type == "lgb":
            r_hat = model.predict(X)
        elif m_type == "cat":
            r_hat = model.predict(X)
        elif m_type == "et":
            r_hat = model.predict(X)
        else:
            raise ValueError(f"Unknown model type {m_type}")

        return teacher_mu + r_hat

    elif mode == "classification":
        if teacher_margin is None:
            raise ValueError("teacher_margin required for classification prediction")
            
        if m_type == "xgb":
            d = xgb.DMatrix(X)
            d.set_base_margin(teacher_margin)
            p_hat = model.predict(d) # Returns sigmoid(margin + delta)
        elif m_type == "lgb":
            # predict returns raw scores if raw_score=True, else probabilities
            # With init_score in predict? LGBM python API doesn't support init_score in predict easily.
            # We predict raw margin correction, add to baseline, then sigmoid.
            margin_delta = model.predict(X, raw_score=True)
            # Note: LGBM trained with init_score learns the residual margin.
            p_hat = sigmoid(teacher_margin + margin_delta)
        elif m_type == "cat":
            # CatBoost predict with prediction_type='RawFormulaVal' gives delta
            margin_delta = model.predict(X, prediction_type='RawFormulaVal')
            p_hat = sigmoid(teacher_margin + margin_delta)
        elif m_type == "et":
            # "residual_prob" subtype
            # Predicts (y - p_teacher)
            # Final p = p_teacher + delta
            # We need p_teacher, but we passed margin.
            p_teacher = sigmoid(teacher_margin)
            delta = model.predict(X)
            p_hat = np.clip(p_teacher + delta, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown model type {m_type}")

        return p_hat

    raise ValueError("Unknown mode.")


class Layer25Chaser:
    """
    Layer 2.5 Chaser: Enhanced Regime-Aware Residual Learner.
    """
    
    def __init__(
        self,
        mode: str = "regression",
        regime_split: bool = True,
        feature_engineering: bool = True,
        correlation_threshold: float = 0.7,
        verbose: bool = True,
        models_to_train: List[str] = None,
        # New parameters for weak constraints
        use_huber_constraints: bool = True,
        constraint_tier: str = "stronger",
    ):
        self.mode = mode
        self.regime_split = regime_split
        self.feature_engineering = feature_engineering
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
        self.models_to_train = models_to_train or ["xgb", "lgb", "cat", "et"]
        
        # New constraint parameters
        self.use_huber_constraints = use_huber_constraints
        self.constraint_tier = constraint_tier

        self.regime_models: Dict[int, Dict[str, Any]] = {}
        self.global_models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.fracdiff_transformer = FracDiffTransformer()

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create FracDiff, Residualized, and Anti-Explosion features.
        Residualized = Detrended (x - LinearTrend).
        """
        if not self.feature_engineering:
            return X.copy()
            
        X_eng = pd.DataFrame(index=X.index)

        # 1. Apply Anti-Explosion Features if price is available
        # Check for 'close' or 'Close'
        price_col = None
        if 'close' in X.columns:
            price_col = 'close'
        elif 'Close' in X.columns:
            price_col = 'Close'

        if price_col:
            try:
                # This generates ~16 features including FracDiff on price
                processed = apply_layer2_price_processing(X, price_col=price_col, enable_price_features=True)
                # We only want the new features, X might have many columns
                # processed has X columns + new ones.
                new_cols = [c for c in processed.columns if c not in X.columns]
                if new_cols:
                    X_eng = pd.concat([X_eng, processed[new_cols]], axis=1)
                    if self.verbose:
                        tprint_info(f"   ✨ Added {len(new_cols)} Anti-Explosion features")
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Anti-Explosion feature generation failed: {e}")

        # 2. Process specific columns for Residualization and FracDiff (if not price)
        # We iterate columns to generate _res and _fd for NON-price columns or just all?
        # The original code did it for ALL columns in X.
        # If X has 100 features, this doubles it to 300.
        # "Anti-explosion" suggests we should be careful.
        # But to preserve behavior for non-price features, we keep it,
        # maybe skipping 'close' if handled above?
        # Actually 'fracdiff_price' is generated by apply_layer2_price_processing.

        for col in X.columns:
            # Skip if it's the price col and we already generated fracdiff_log_price
            if price_col and col == price_col and 'fracdiff_log_price' in X_eng.columns:
                continue

            # 1. FracDiff (Stationary memory)
            # Use cached optimal d if possible, else find it
            try:
                fd_series, _ = self.fracdiff_transformer.fracdiff_series(
                    X[col], method='binary_search', tolerance=0.1 # Fast search
                )
                X_eng[f"{col}_fd"] = fd_series
            except Exception:
                X_eng[f"{col}_fd"] = X[col] # Fallback
            
            # 2. Residualized (Detrended)
            # Linear Model Residualisation: X - LinearTrend(X)
            lr = LinearRegression()
            # Handle NaNs in column before fitting
            col_data = X[col].fillna(0)
            time_idx = np.arange(len(col_data))
            lr.fit(time_idx.reshape(-1, 1), col_data.values)
            fitted = lr.predict(time_idx.reshape(-1, 1))
            X_eng[f"{col}_res"] = col_data - fitted

        # Merge original X? No, original returns X_eng which contained only new features?
        # Wait, original X_eng started empty: `X_eng = pd.DataFrame(index=X.index)`
        # Then `X_eng` only contained `_fd` and `_res`.
        # It did NOT return `X`.
        # `fit` calls `X_proc = self._engineer_features(X)`.
        # If `_engineer_features` returns ONLY engineered features, then original features are LOST?
        # Let's check `fit` usage. `self.feature_names = list(X_proc.columns)`.
        # Yes, it seems it replaced features.
        # BUT: `X_eng` construction loop iterates `col in X.columns`.
        # So it transforms EVERY feature.
        # The user said "Dual-Feature Engineering: Uses both FracDiff... and Residualized...".
        # So likely it replaces raw with transformed.

        # With Anti-Explosion, we add price features. We should also keep the transformed features.
        # If I return X_eng (which has anti-explosion + transformed old features), it should be fine.

        # Fill NaNs created by differencing/fracdiff
        X_eng = X_eng.fillna(0.0)
        return X_eng

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_probs: pd.DataFrame | None = None,
        sample_weight: pd.Series | None = None
    ):
        """
        Fit the Chaser model(s).

        Args:
            X: Features
            y: Targets (Residuals for regression, Binary for classification)
            regime_probs: GMM probabilities (N, K). If None, global model only.
            sample_weight: Base sample weights.
        """
        if self.verbose:
            tprint_info("🚀 Training Layer 2.5 Chaser...")

        # Feature Engineering
        if self.feature_engineering:
            if self.verbose: tprint_info("   🛠️ Engineering features (FracDiff + Residuals)...")
            X_proc = self._engineer_features(X)
            if self.verbose:
                tprint_info(f"   ✅ Features ready: {X_proc.shape[1]} columns (FracDiff/Residualized)")
        else:
            X_proc = X.copy()

        # --- MDI Feature Selection (De Prado) ---
        if self.verbose:
            tprint_info("   🔍 Running De Prado MDI Feature Selection...")

        try:
            # Initialize selection engine
            # We use moderate params to reduce feature set size and collinearity
            selector = DePradoFeatureEngine(
                n_estimators=500,
                max_clusters=12,
                random_state=42,
                topk_freq_threshold=0.4, # Hardening
                use_lgbm=LGBM_AVAILABLE,
                is_regression=(self.mode == "regression")
            )

            # Run selection
            # Note: DePrado engine takes X and y. It doesn't explicitly use weights,
            # but usually it's fine for structure discovery.
            selected_features = selector.run_selection(X_proc, y)

            # Apply selection
            n_orig = X_proc.shape[1]
            X_proc = X_proc[selected_features]
            n_new = X_proc.shape[1]

            if self.verbose:
                tprint_success(f"   ✅ Selected {n_new} features from {n_orig} (Reduction: {100*(1 - n_new/n_orig):.1f}%)")

        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ De Prado Feature Selection failed: {e}. Using all features.")

        self.feature_names = list(X_proc.columns)
        X_np = X_proc.values.astype(np.float32)
        y_np = y.values
        w_np = sample_weight.values if sample_weight is not None else None

        # Get weak constraints from Huber if available
        monotone_constraints_weak = None
        interaction_constraints_weak = None
        
        if self.use_huber_constraints and HUBER_AVAILABLE:
            try:
                # Use weak tier constraints for chasers
                huber_config = get_huber_tier_config(self.constraint_tier)
                huber_results = prepare_huber_teacher_outputs(
                    X_train=pd.DataFrame(X_np, columns=self.feature_names),
                    y_train=y_np,
                    sample_weight=w_np,
                    config=huber_config,
                    tier=self.constraint_tier
                )
                monotone_constraints_weak = huber_results.get('monotonic_constraints', {})
                interaction_constraints_weak = huber_results.get('interaction_constraints', [])
                
                if self.verbose:
                    tprint_info(f"   🔗 Applied {len(monotone_constraints_weak)} monotone constraints from Huber")
                    tprint_info(f"   🔄 Applied {len(interaction_constraints_weak)} interaction constraint groups")
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Failed to get Huber constraints: {e}")

        # Helper to train a set of models for a specific weight vector
        def train_ensemble(weights, prefix=""):
            if self.verbose:
                tprint_info(f"   🎓 Training Teacher (BayesianRidge) for {prefix}...")

            # 1. Teacher
            teacher = fit_bayes_teacher_oof(
                X_np, y_np, sample_weight=weights, n_splits=5, is_classifier=(self.mode == "classification"),
                index=X.index # Pass index for PurgedKFold
            )
            
            if self.verbose:
                if teacher.p_oof is not None:
                    acc = np.mean((teacher.p_oof > 0.5) == y_np)
                    tprint_info(f"      - Teacher Accuracy: {acc:.4f}")
                elif teacher.mu_oof is not None:
                    # Simple R2 proxy
                    r2 = 1 - np.sum((y_np - teacher.mu_oof)**2) / np.sum((y_np - np.mean(y_np))**2)
                    tprint_info(f"      - Teacher R2: {r2:.4f}")

            # Sanity check uncertainty signal
            if self.mode == "classification" and teacher.p_oof is not None:
                uncertainty_corr = sanity_check_uncertainty(y_np, teacher.p_oof, teacher.std_oof)
                if self.verbose:
                    tprint_info(f"   📊 Uncertainty-error correlation: {uncertainty_corr:.3f}")

            # 2. Students
            students = {}
            model_scores = {}
            temp_preds = {}

            if self.verbose:
                tprint_info(f"   👨‍🎓 Training Students ({len(self.models_to_train)} types) for {prefix}...")

            for m_type in self.models_to_train:
                if m_type == "lgb" and not LGBM_AVAILABLE: continue
                if m_type == "cat" and not CATBOOST_AVAILABLE: continue

                try:
                    student = train_chaser_student(
                        X_np, y_np, teacher, 
                        base_weight=weights, 
                        mode=self.mode, 
                        model_type=m_type,
                        monotone_constraints_weak=monotone_constraints_weak,
                        interaction_constraints_weak=interaction_constraints_weak,
                        huber_teacher_mu=teacher.mu_oof
                    )

                    # Generate OOF preds for correlation check (on training data)
                    # For simplicity, we just predict on full training data here
                    if self.mode == "regression":
                        pred = predict_chaser_student(X_np, teacher.mu_oof, None, student)
                        # Score: IC (Spearman)
                        score = float(spearmanr(pred, y_np)[0])
                    else:
                        pred = predict_chaser_student(X_np, None, teacher.margin_oof, student)
                        # Score: AUC
                        try:
                            score = roc_auc_score(y_np, pred, sample_weight=weights)
                        except Exception:
                            score = 0.5

                    temp_preds[m_type] = pred
                    students[m_type] = student
                    model_scores[m_type] = score

                    if self.verbose:
                        tprint_info(f"   📝 {m_type.upper()} Score: {score:.4f}")

                    if self.verbose:
                        tprint_success(f"      ✅ {m_type} trained")

                except Exception as e:
                    tprint_warning(f"   ⚠️ Failed to train {m_type} student: {e}")

            # 3. Prune redundant models
            # 1. Rank by score (IC or AUC)
            sorted_models = sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)

            kept = []
            if sorted_models:
                kept.append(sorted_models[0])  # Always keep best

                for i in range(1, len(sorted_models)):
                    curr = sorted_models[i]
                    is_redundant = False

                    # Check correlation with higher-ranked kept models
                    for existing in kept:
                        corr = np.corrcoef(temp_preds[curr], temp_preds[existing])[0, 1]
                        if corr > self.correlation_threshold:
                            is_redundant = True
                            if self.verbose:
                                tprint_info(f"   ✂️ Pruning {curr} (corr {corr:.3f} with {existing})")
                            break

                    if not is_redundant:
                        kept.append(curr)

                # Filter students dictionary
                students = {k: v for k, v in students.items() if k in kept}
                if self.verbose and len(kept) < len(sorted_models):
                    tprint_info(f"   🏁 Final Ensemble: {kept} (Best: {sorted_models[0]})")

            return {"teacher": teacher, "students": students}

        # Training Loop
        if self.regime_split and regime_probs is not None:
            n_regimes = regime_probs.shape[1]
            if self.verbose: tprint_info(f"   🔄 Training {n_regimes} regime-specific ensembles...")

            for k in range(n_regimes):
                # Calculate regime-weighted sample weights
                # w_k = w_base * P(regime_k)
                regime_w = regime_probs.iloc[:, k].values
                # Normalize regime prob to avoid vanishing weights?
                # No, we want to downweight samples where regime is not active.

                if w_np is not None:
                    final_w = w_np * regime_w
                else:
                    final_w = regime_w

                # Skip if regime mass is too small
                if np.sum(final_w) < 10:
                    tprint_warning(f"   ⚠️ Regime {k} has insufficient mass, skipping.")
                    continue

                self.regime_models[k] = train_ensemble(final_w, prefix=f"Regime {k}")

        else:
            # Global model
            if self.verbose: tprint_info("   🌍 Training global ensemble...")
            self.global_models = train_ensemble(w_np, prefix="Global")

        if self.verbose: tprint_success("✅ Chaser training complete.")

    def predict(
        self,
        X: pd.DataFrame,
        regime_probs: pd.DataFrame | None = None,
        return_individual: bool = False,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the ensemble.

        Args:
            X: Input features
            regime_probs: Regime probabilities for weighting
            return_individual: If True, returns dict of {model_name: prediction}
            return_confidence: If True, returns (prediction, confidence) tuple
        """
        # Feature Engineering
        if self.feature_engineering:
            X_proc = self._engineer_features(X)
        else:
            X_proc = X.copy()

        # Ensure we use the selected features from training
        if hasattr(self, 'feature_names') and self.feature_names:
            # Check if features are missing
            missing = [f for f in self.feature_names if f not in X_proc.columns]
            if missing:
                # If we engineer features correctly, this shouldn't happen unless input X is different schema
                # Just fill 0 for safety or raise warning
                if self.verbose:
                    tprint_warning(f"   ⚠️ Missing {len(missing)} features in predict. Filling 0.")
                for m in missing:
                    X_proc[m] = 0.0

            X_proc = X_proc[self.feature_names]

        X_np = X_proc.values.astype(np.float32)
        n_samples = len(X)

        # Helper to predict with an ensemble
        def predict_ensemble(ensemble, x_data):
            teacher = ensemble["teacher"]
            students = ensemble["students"]

            # 1. Teacher Prediction
            teacher_X = teacher.scaler.transform(x_data)
            teacher_mu = teacher.model.predict(teacher_X)

            teacher_margin = None
            if self.mode == "classification":
                # Convert mu to margin
                # Using the calibrator trained on OOF
                if teacher.calibrator:
                    # Prob
                    p = teacher.calibrator.predict_proba(teacher_mu.reshape(-1, 1))[:, 1]
                    # Margin
                    teacher_margin = prob_to_logit(p)
                else:
                    # Fallback (shouldn't happen if trained correctly)
                    teacher_margin = teacher_mu

            # 2. Student Predictions
            preds_dict = {}
            preds_list = []

            for name, student in students.items():
                p = predict_chaser_student(x_data, teacher_mu, teacher_margin, student)
                preds_dict[name] = p
                preds_list.append(p)

            # Average student predictions
            if not preds_list:
                # Fallback to teacher
                baseline = sigmoid(teacher_margin) if self.mode == "classification" else teacher_mu
                return baseline, {}, np.zeros(len(x_data))

            ensemble_mean = np.mean(preds_list, axis=0)

            # Calculate confidence (std dev of ensemble members)
            # Lower std = higher confidence agreement
            # We invert it: 1 / (1 + std) or similar
            if len(preds_list) > 1:
                ensemble_std = np.std(preds_list, axis=0)
                confidence = 1.0 / (1.0 + ensemble_std)
            else:
                confidence = np.ones(len(x_data)) # Single model = full confidence (relative to self)

            return ensemble_mean, preds_dict, confidence

        # Initialize outputs
        final_pred = np.zeros(n_samples)
        total_weight = np.zeros(n_samples)
        final_confidence = np.zeros(n_samples)
        all_individual_preds = {} # {model_name: np.zeros(n_samples)}

        # 1. Regime-based Prediction
        if self.regime_split and regime_probs is not None and self.regime_models:

            for k, ensemble in self.regime_models.items():
                if k < regime_probs.shape[1]:
                    prob_k = regime_probs.iloc[:, k].values
                    mask = prob_k > 0.001

                    if np.any(mask):
                        pred_k, ind_k, conf_k = predict_ensemble(ensemble, X_np[mask])

                        # Accumulate weighted average
                        final_pred[mask] += pred_k * prob_k[mask]
                        final_confidence[mask] += conf_k * prob_k[mask]
                        total_weight[mask] += prob_k[mask]

                        # Store individual predictions (weighted)
                        # Naming: "regime_{k}_{model}"
                        if return_individual:
                            for model_name, model_pred in ind_k.items():
                                key = f"regime{k}_{model_name}"
                                if key not in all_individual_preds:
                                    all_individual_preds[key] = np.zeros(n_samples)
                                all_individual_preds[key][mask] = model_pred

            # Normalize
            nonzero = total_weight > 0
            final_pred[nonzero] /= total_weight[nonzero]
            final_confidence[nonzero] /= total_weight[nonzero]
            
            # Fill gaps with global or 0
            if not np.all(nonzero):
                if self.global_models:
                    g_pred, g_ind, g_conf = predict_ensemble(self.global_models, X_np[~nonzero])
                    final_pred[~nonzero] = g_pred
                    final_confidence[~nonzero] = g_conf
                    if return_individual:
                        for m_name, m_pred in g_ind.items():
                            key = f"global_{m_name}"
                            if key not in all_individual_preds:
                                all_individual_preds[key] = np.zeros(n_samples)
                            all_individual_preds[key][~nonzero] = m_pred

        # 2. Global Prediction
        elif self.global_models:
            final_pred, ind_preds, final_confidence = predict_ensemble(self.global_models, X_np)
            if return_individual:
                for m_name, m_pred in ind_preds.items():
                    all_individual_preds[f"global_{m_name}"] = m_pred

        else:
            raise ValueError("No trained models available.")

        # Return Logic
        if return_individual:
            # We also return the average as "ensemble_mean"
            all_individual_preds["ensemble_mean"] = final_pred
            if return_confidence:
                return all_individual_preds, final_confidence
            return all_individual_preds
        elif return_confidence:
            return final_pred, final_confidence
        else:
            return final_pred

# Convenience functions
def create_chaser(**kwargs):
    return Layer25Chaser(**kwargs)
