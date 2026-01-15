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
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone

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

# -----------------------------
# Utilities
# -----------------------------
def robust_sigma(x: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate robust standard deviation using MAD."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return 1.4826 * mad

def winsorize(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Winsorize data based on robust sigma."""
    s = robust_sigma(x)
    med = np.median(x)
    return np.clip(x, med - k * s, med + k * s)

def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize weights to mean 1.0."""
    w = np.asarray(w, dtype=np.float64)
    m = np.mean(w)
    return w if (not np.isfinite(m) or m <= 0) else w / m

def uncertainty_to_chaser_weight(std: np.ndarray, clip=(0.5, 2.0)) -> np.ndarray:
    """Convert prediction uncertainty to sample weights (upweight uncertain samples)."""
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
    med = np.median(std)
    w = np.sqrt(std / (med + 1e-12))
    w = np.clip(w, clip[0], clip[1])
    return normalize_weights(w)

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
    is_classifier: bool = False
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

    tscv = TimeSeriesSplit(n_splits=n_splits)
    mu_oof = np.full(Xs.shape[0], np.nan, dtype=np.float64)
    std_oof = np.full(Xs.shape[0], np.nan, dtype=np.float64)

    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    # Walk-forward CV
    for tr, va in tscv.split(Xs):
        X_tr, X_va = Xs[tr], Xs[va]
        y_tr = y[tr].astype(np.float64)

        # If regression on heavy-tailed targets, winsorize for the teacher
        if not is_classifier:
            y_tr = winsorize(y_tr, k=winsor_k)

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
    num_boost_round: int = 800,
):
    """
    Train a student model (Chaser) on residuals or with margin correction.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    # Teacher-uncertainty weights (chasers upweight uncertainty, bounded)
    w_chase = uncertainty_to_chaser_weight(teacher.std_oof, clip=(0.5, 2.0))
    w_final = combine_weights(base_weight, w_chase)

    if mode == "regression":
        # Residual target
        r = y.astype(np.float64) - teacher.mu_oof
        r = winsorize(r, k=winsor_resid_k)
        target = r
        init_score = None # For models that support it
        baseline = None   # For CatBoost
    elif mode == "classification":
        if teacher.margin_oof is None:
            raise ValueError("For classification mode, teacher.margin_oof must be available.")
        target = y.astype(np.int32)
        init_score = teacher.margin_oof.astype(np.float64)
        baseline = teacher.margin_oof.astype(np.float64)
    else:
        raise ValueError("mode must be 'regression' or 'classification'.")

    # --- XGBoost ---
    if model_type == "xgb":
        dtrain = xgb.DMatrix(X, label=target, weight=w_final)
        if mode == "classification":
            dtrain.set_base_margin(init_score)

        default_params = {
            "eta": 0.05,
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
            "gamma": 0.05,
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

        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        return {"model": bst, "mode": mode, "type": "xgb", "params": params}

    # --- LightGBM ---
    elif model_type == "lgb":
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        train_data = lgb.Dataset(X, label=target, weight=w_final)
        if mode == "classification":
            train_data.set_init_score(init_score)

        default_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
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

        bst = lgb.train(params, train_data, num_boost_round=num_boost_round)
        return {"model": bst, "mode": mode, "type": "lgb", "params": params}

    # --- CatBoost ---
    elif model_type == "cat":
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed")

        train_pool = cb.Pool(X, label=target, weight=w_final)
        if mode == "classification":
            train_pool.set_baseline(baseline)

        default_params = {
            "iterations": num_boost_round,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "verbose": False,
            "allow_writing_files": False
        }
        if mode == "regression":
            default_params["loss_function"] = "MAE"
        else:
            default_params["loss_function"] = "Logloss"

        params = default_params.copy()
        if model_params:
            params.update(model_params)

        model = cb.CatBoost(params)
        model.fit(train_pool)
        return {"model": model, "mode": mode, "type": "cat", "params": params}

    # --- ExtraTrees ---
    elif model_type == "et":
        # ExtraTrees doesn't support margin/init_score easily for classification
        # We will use it only for regression on residuals
        if mode == "classification":
            # Fallback: Train on raw binary target with sample weights
            # This ignores the teacher margin, making it a pure student
            # Or train on probability residuals? No.
            # We skip margin correction for ET Classifier here.
            # print("Warning: ExtraTrees classifier does not support margin correction. Training standard model.")
            et_model = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, max_depth=10, min_samples_leaf=5)
            # Train regression on the binary target? Or use Classifier?
            # Using Regressor on binary target is sometimes robust.
            # But let's use the residual logic: y_bin - p_oof
            # This is "regression on probability residuals".
            res_target = y.astype(float) - teacher.p_oof
            et_model.fit(X, res_target, sample_weight=w_final)
            # Store prediction type as "residual_prob"
            return {"model": et_model, "mode": mode, "type": "et", "subtype": "residual_prob"}
        else:
            # Regression on residuals
            et_model = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, max_depth=10, min_samples_leaf=5, random_state=42)
            if model_params:
                et_model.set_params(**model_params)
            et_model.fit(X, target, sample_weight=w_final)
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
        models_to_train: List[str] = None
    ):
        self.mode = mode
        self.regime_split = regime_split
        self.feature_engineering = feature_engineering
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose
        self.models_to_train = models_to_train or ["xgb", "lgb", "cat", "et"]

        self.regime_models: Dict[int, Dict[str, Any]] = {}
        self.global_models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.fracdiff_transformer = FracDiffTransformer()

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create FracDiff and Residualized features.
        Residualized = Detrended via rolling linear regression against time.
        """
        if not self.feature_engineering:
            return X.copy()
            
        X_eng = pd.DataFrame(index=X.index)

        # Import numba functions locally
        from src.utils.numba_funcs import _numba_rolling_slope, _numba_rolling_mean

        window = 20

        # We process each column
        for col in X.columns:
            # 1. FracDiff (Stationary memory)
            try:
                fd_series, _ = self.fracdiff_transformer.fracdiff_series(
                    X[col], method='binary_search', tolerance=0.1
                )
                X_eng[f"{col}_fd"] = fd_series
            except Exception:
                X_eng[f"{col}_fd"] = X[col]
            
            # 2. Residualized (Linear Detrending)
            # x_t = alpha + beta * t + epsilon_t
            # epsilon_t = x_t - (alpha + beta * t)
            # Using rolling window linear regression

            y_vals = X[col].values.astype(np.float64)
            slope = _numba_rolling_slope(y_vals, window)
            mean = _numba_rolling_mean(y_vals, window)

            # Prediction at the end of the window (relative time t = window - 1)
            # mean_x for 0..window-1 is (window-1)/2
            # pred = mean_y + slope * (t - mean_x)
            pred = mean + slope * (window - 1.0) / 2.0

            # Residual
            residuals = y_vals - pred

            # Mask first window-1 elements (where slope/mean are not valid)
            # _numba_rolling_* returns 0.0 for first window-1 elements.
            # We set them to NaN to be handled by fillna(0.0) later or handle explicitely.
            if len(residuals) >= window:
                residuals[:window-1] = np.nan

            X_eng[f"{col}_res"] = residuals

        # Fill NaNs created by differencing/fracdiff/rolling
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
        else:
            X_proc = X.copy()

        self.feature_names = list(X_proc.columns)
        X_np = X_proc.values.astype(np.float32)
        y_np = y.values
        w_np = sample_weight.values if sample_weight is not None else None

        # Helper to train a set of models for a specific weight vector
        def train_ensemble(weights, prefix=""):
            # 1. Teacher
            teacher = fit_bayes_teacher_oof(
                X_np, y_np, sample_weight=weights, n_splits=5, is_classifier=(self.mode == "classification")
            )

            # 2. Students
            students = {}
            valid_models = []

            # Temporary storage for pruning
            temp_preds = {}

            for m_type in self.models_to_train:
                if m_type == "lgb" and not LGBM_AVAILABLE: continue
                if m_type == "cat" and not CATBOOST_AVAILABLE: continue

                try:
                    student = train_chaser_student(
                        X_np, y_np, teacher, base_weight=weights, mode=self.mode, model_type=m_type
                    )

                    # Generate OOF preds for correlation check (on training data)
                    # For simplicity, we just predict on full training data here
                    # (Note: this is biased for pruning but fast. Proper way is OOF predictions during training)
                    # The student function trains on full data.
                    # We accept slight bias for pruning redundant models.

                    if self.mode == "regression":
                        pred = predict_chaser_student(X_np, teacher.mu_oof, None, student)
                    else:
                        pred = predict_chaser_student(X_np, None, teacher.margin_oof, student)

                    temp_preds[m_type] = pred
                    students[m_type] = student
                    valid_models.append(m_type)

                except Exception as e:
                    tprint_warning(f"   ⚠️ Failed to train {m_type} student: {e}")

            # 3. Prune redundant models
            # Greedy selection: Pick best? Or just check correlations.
            # We keep all if correlation < threshold.
            if len(valid_models) > 1:
                kept = [valid_models[0]]
                for i in range(1, len(valid_models)):
                    curr = valid_models[i]
                    is_redundant = False
                    for existing in kept:
                        corr = np.corrcoef(temp_preds[curr], temp_preds[existing])[0, 1]
                        if corr > self.correlation_threshold:
                            is_redundant = True
                            break
                    if not is_redundant:
                        kept.append(curr)

                # Filter students dictionary
                students = {k: v for k, v in students.items() if k in kept}
                if self.verbose and len(kept) < len(valid_models):
                    tprint_info(f"   ✂️ Pruned redundant models in {prefix}: {len(valid_models)} -> {len(kept)}")

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
        regime_probs: pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Predict using the ensemble.
        """
        # Feature Engineering
        if self.feature_engineering:
            X_proc = self._engineer_features(X)
        else:
            X_proc = X.copy()

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
            preds = []
            for name, student in students.items():
                p = predict_chaser_student(x_data, teacher_mu, teacher_margin, student)
                preds.append(p)

            # Average student predictions
            if not preds:
                return sigmoid(teacher_margin) if self.mode == "classification" else teacher_mu

            return np.mean(preds, axis=0)

        # 1. Regime-based Prediction
        if self.regime_split and regime_probs is not None and self.regime_models:
            final_pred = np.zeros(n_samples)
            total_weight = np.zeros(n_samples)

            for k, ensemble in self.regime_models.items():
                # Get regime probability for these samples
                # Assuming regime_probs aligns with X
                if k < regime_probs.shape[1]:
                    prob_k = regime_probs.iloc[:, k].values

                    # Optimization: only predict for samples with non-zero prob
                    # But for simplicity with vectorization, we compute all (or mask)
                    # Using mask for speed
                    mask = prob_k > 0.001
                    if np.any(mask):
                        pred_k = predict_ensemble(ensemble, X_np[mask])
                        final_pred[mask] += pred_k * prob_k[mask]
                        total_weight[mask] += prob_k[mask]
            
            # Normalize by total weight (in case sum probs != 1 or missing regimes)
            # Avoid div by zero
            nonzero = total_weight > 0
            final_pred[nonzero] /= total_weight[nonzero]
            
            # If any sample had 0 weight (no active regime model), fall back to global or 0
            if not np.all(nonzero):
                if self.global_models:
                    global_pred = predict_ensemble(self.global_models, X_np)
                    final_pred[~nonzero] = global_pred[~nonzero]
            
            return final_pred

        # 2. Global Prediction
        elif self.global_models:
            return predict_ensemble(self.global_models, X_np)

        else:
            raise ValueError("No trained models available.")

# Convenience functions
def create_chaser(**kwargs):
    return Layer25Chaser(**kwargs)
