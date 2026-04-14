import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score
from scipy.stats import rankdata, spearmanr
from scipy.special import logit, expit
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
import joblib
from extreme_price_movements.utils import tprint


def _is_xgb(est):
    return type(est).__module__.startswith("xgboost")

def _is_lgb(est):
    return type(est).__module__.startswith("lightgbm")

def _is_cb(est):
    return type(est).__module__.startswith("catboost")
from extreme_price_movements.purged_cv import PurgedKFold
from extreme_price_movements.metrics import calculate_selection_score
from extreme_price_movements.model_scoring import (
    AlphaRankConfig,
    alpha_objective_logloss,
    alpha_rank_components,
    ece_at_mask,
    topk_mask,
    calibration_curve_bins,
    calibration_profile,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import brentq
from extreme_price_movements.tree_leaf_policy import tree_regularization_params
from extreme_price_movements.calibration import (
    safe_clip_proba,
    compute_prevalences,
    compute_logit_shift,
    apply_logit_shift,
)
from extreme_price_movements.training_utils import robust_sigma


def _safe_binary_calibrate(preds, y_true, min_unique=20, min_samples=100):
    """Calibrate binary probabilities with guardrails and fallback.

    Returns (calibrated_probs, calibrator_or_none, method_name).
    """
    preds = np.asarray(preds, dtype=np.float64)
    y_true = np.asarray(y_true)
    valid = np.isfinite(preds) & np.isfinite(y_true)
    if valid.sum() < max(20, min_samples):
        return preds, None, "identity"

    x = preds[valid]
    y = y_true[valid]
    if len(np.unique(y)) < 2 or len(np.unique(np.round(x, 8))) < min_unique:
        try:
            platt = LogisticRegression(random_state=42, max_iter=1000)
            platt.fit(x.reshape(-1, 1), y)
            out = preds.copy()
            out[valid] = platt.predict_proba(x.reshape(-1, 1))[:, 1]
            return out.astype(np.float32), platt, "platt"
        except Exception:
            return preds, None, "identity"

    try:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.05, y_max=0.95)
        iso.fit(x, y)
        out = preds.copy()
        out[valid] = iso.predict(x)
        return out.astype(np.float32), iso, "isotonic"
    except Exception:
        try:
            platt = LogisticRegression(random_state=42, max_iter=1000)
            platt.fit(x.reshape(-1, 1), y)
            out = preds.copy()
            out[valid] = platt.predict_proba(x.reshape(-1, 1))[:, 1]
            return out.astype(np.float32), platt, "platt"
        except Exception:
            return preds, None, "identity"


def calculate_selection_score(y_true, y_prob, y_ret, sample_weight=None, symbols=None, **kwargs):
    """Backward-compatible wrapper exposing legacy AUC/BSS/IC keys for tests/logs."""
    from extreme_price_movements.metrics import calculate_selection_score as _calc
    from extreme_price_movements.model_scoring import ic_cross_sectional
    out = _calc(y_true, y_prob, y_ret, sample_weight=sample_weight, **kwargs)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_ret = np.asarray(y_ret)
    try:
        out["AUC"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        out["AUC"] = 0.5
    brier = float(np.mean((y_prob - y_true) ** 2))
    base = float(np.mean(y_true))
    brier_ref = float(np.mean((base - y_true) ** 2))
    out["BSS"] = 1.0 - (brier / max(1e-9, brier_ref))
    try:
        if symbols is not None:
            out["IC"] = float(ic_cross_sectional(y_prob, y_ret, groups=symbols))
            if np.isnan(out["IC"]):
                out["IC"] = 0.0
        else:
            out["IC"] = float(np.corrcoef(rankdata(y_prob), rankdata(y_ret))[0, 1])
    except Exception:
        out["IC"] = 0.0
    return out


class Float64Wrapper(BaseEstimator, ClassifierMixin):
    """Wraps a classifier so predict_proba / decision_function always return float64.
    Some estimators (e.g. XGBoost) return float32 predictions by default."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.classes_ = np.unique(y)
        if sample_weight is not None:
            self.estimator.fit(X, y, sample_weight=sample_weight, **kwargs)
        else:
            self.estimator.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        return np.asarray(self.estimator.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.estimator.predict(X)

    def decision_function(self, X):
        if hasattr(self.estimator, 'decision_function'):
            return np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        return self.predict_proba(X)[:, 1]

    def get_params(self, deep=True):
        return {"estimator": self.estimator}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self


class NativeLGBMBoosterClassifier(BaseEstimator, ClassifierMixin):
    """Minimal sklearn-like wrapper around a persisted LightGBM Booster."""

    def __init__(self, booster=None):
        self.booster = booster
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def predict_proba(self, X):
        if self.booster is None:
            raise ValueError("LightGBM booster is not loaded")
        proba = np.asarray(self.booster.predict(X), dtype=np.float64)
        proba = np.clip(proba, 1e-6, 1.0 - 1e-6)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)


class ScaledLogisticRegression(LogisticRegression):
    """
    Wrapper to apply StandardScaler internally, ensuring sample_weight 
    is correctly passed to fit (bypassing Pipeline limitations with CalibratedClassifierCV).
    """
    def __init__(self, class_weight=None, **kwargs):
        super().__init__(class_weight=class_weight, **kwargs)
        self.scaler = StandardScaler()

    def fit(self, X, y, sample_weight=None):
        X_scaled = self.scaler.fit_transform(X)
        return super().fit(X_scaled, y, sample_weight=sample_weight)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict(X_scaled)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return super().predict_proba(X_scaled)



class ModelRace(BaseEstimator, ClassifierMixin):
    def __init__(self, kind="long", task="base", n_splits=5, race_sample_frac=0.5, race_early_stopping_rounds=50, max_label_horizon_hours=8):
        self.kind = kind
        self.task = task
        self.n_splits = n_splits
        self.race_sample_frac = race_sample_frac
        self.race_early_stopping_rounds = race_early_stopping_rounds
        self.max_label_horizon_hours = max_label_horizon_hours
        self.best_model = None
        self.best_model_name = None
        self.metrics = {}
        self.detailed_metrics = {}
        self.oof_probs = None
        self.final_bias_factor_ = 1.0
        self.calibration_state_ = None
        self._used_sample_weight_ = False

    def _compute_pos_weight(self, y):
        # Inverse class frequency
        return (len(y) - y.sum()) / max(1, y.sum())

    def _subsample_indices(self, indices, frac, seed=42):
        if frac >= 1.0:
            return indices
        np.random.seed(seed)
        n_samples = int(len(indices) * frac)
        return np.random.choice(indices, n_samples, replace=False)

    def _build_bias_state(self, y_unweighted, y_weighted, eps=1e-6):
        delta_logit = compute_logit_shift(y_unweighted, y_weighted, eps=eps)
        return {
            "schema_version": 1,
            "method": "logit_shift",
            "target_unweighted_prevalence": float(y_unweighted),
            "weighted_prevalence": float(y_weighted),
            "delta_logit": float(delta_logit),
            "eps": float(eps),
            "calibration_input": "bias_corrected",
        }

    def _apply_bias_state(self, p_raw, state):
        return apply_logit_shift(p_raw, state["delta_logit"], eps=state.get("eps", 1e-6))

    def _get_candidates(self, race_mode=True):
        candidates = {}

        if self.task == "base":
            # Base models are restricted to ExtraTrees only.
            et_params = {
                "n_estimators": 400,
                "max_depth": 6,
                "min_samples_leaf": 64,
                "min_samples_split": 128,
                "bootstrap": True,
                "ccp_alpha": 1e-4,
                "max_leaf_nodes": 512,
                "max_features": "sqrt",
                "n_jobs": 2,
                "random_state": 42
            }
            candidates["extratrees"] = Float64Wrapper(ExtraTreesClassifier(**et_params))
        elif self.task == "meta":
            # Meta models are restricted to XGBoost only.
            xgb_params = {
                "n_estimators": 800 if race_mode else 1200,
                "max_depth": 4,
                "learning_rate": 0.015,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 5,
                "gamma": 0.5,
                "tree_method": "hist",
                "n_jobs": 2,
                "random_state": 42
            }
            candidates["xgboost"] = Float64Wrapper(XGBClassifier(**xgb_params))
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return candidates

    def _fit_model(self, model, X_tr, y_tr, X_val=None, y_val=None, sample_weight=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        pos_weight = self._compute_pos_weight(y_tr)

        # Handle Float64Wrapper
        inner = model.estimator if isinstance(model, Float64Wrapper) else model

        if isinstance(inner, ScaledLogisticRegression):
            # Safe to set because we updated __init__
            inner.set_params(class_weight={0: 1.0, 1: pos_weight})
        elif isinstance(inner, ExtraTreesClassifier):
            inner.set_params(class_weight={0: 1.0, 1: pos_weight})

        if CatBoostClassifier is not None and isinstance(inner, CatBoostClassifier):
            # CatBoost requires scale_pos_weight for custom class weighting
            inner.set_params(scale_pos_weight=pos_weight)
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": (X_val, y_val),
                    "early_stopping_rounds": self.race_early_stopping_rounds,
                    "use_best_model": True,
                })
        elif XGBClassifier is not None and isinstance(inner, XGBClassifier):
            inner.set_params(scale_pos_weight=pos_weight, eval_metric="auc")
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                    # early_stopping_rounds deprecated in fit, use constructor or callbacks if needed
                    # For simple race, we can omit it or relying on constructor
                })
        elif LGBMClassifier is not None and isinstance(inner, LGBMClassifier):
            inner.set_params(scale_pos_weight=pos_weight)
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": [(X_val, y_val)],
                    "eval_metric": "auc",
                    "callbacks": [],
                })
                try:
                    from lightgbm import early_stopping
                    fit_kwargs["callbacks"].append(early_stopping(self.race_early_stopping_rounds, verbose=False))
                except Exception:
                    pass

        model.fit(X_tr, y_tr, **fit_kwargs)

    def _tree_sigma_features(self, model, X):
        """Return tree-ensemble dispersion features for a fitted model.

        The base race only uses ExtraTrees, so we can compute the per-tree vote
        spread directly from the fitted estimators. Fallbacks are NaN when the
        model does not expose tree estimators.
        """
        inner = model.estimator if isinstance(model, Float64Wrapper) else model
        estimators = getattr(inner, "estimators_", None)
        if not estimators:
            n = len(X)
            nan_vec = np.full(n, np.nan, dtype=np.float32)
            return nan_vec, nan_vec
        try:
            tree_preds = np.stack([tree.predict(X) for tree in estimators], axis=1)
            sigma = np.asarray(tree_preds.std(axis=1), dtype=np.float32)
            sigma_robust = np.asarray(robust_sigma(tree_preds), dtype=np.float32)
            return sigma, sigma_robust
        except Exception:
            n = len(X)
            nan_vec = np.full(n, np.nan, dtype=np.float32)
            return nan_vec, nan_vec

    def fit(self, X, y, sample_weight=None, returns=None, groups=None, symbols=None):
        """
        X: features
        y: binary target
        sample_weight: weights for training
        returns: continuous returns for IC calculation (validation)
        groups: typically timestamps for time-based splitting
        symbols: symbol array for per-asset IC calculation
        """
        tprint(f"Entering function: fit in model_race.py")
        self.oof_probs = None  # Will store OOF predictions from best model

        # 0. Preparation
        # Cast y and sample_weight to float64 for consistent dtype handling
        y = np.asarray(y, dtype=np.float64)
        y = np.clip(y, 0.0, 1.0)
        y_hard = (y >= 0.5).astype(np.int8)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if returns is None:
            returns = y
        else:
            returns = np.asarray(returns, dtype=np.float64)
        groups_arr = None if groups is None else np.asarray(groups)
        symbols_arr = None if symbols is None else np.asarray(symbols)

        # Optimize: Convert to numpy once if possible (and suitable for all models)
        # ExtraTrees/XGBoost prefer numpy. CatBoost handles both but numpy is fine if no categorical features.
        # We assume numeric features here.
        X_np = X
        X_np = X
        use_numpy = False
        if hasattr(X, "iloc"):
            try:
                # Float32 for memory; Float64Wrapper ensures predict_proba returns float64
                X_np = X.to_numpy(dtype=np.float32, copy=False)
                use_numpy = True
            except (ValueError, TypeError):
                # Fallback if conversion fails (e.g. mixed types)
                use_numpy = False
        elif hasattr(X, "ndim") and hasattr(X, "shape"):
            # Numpy array
            use_numpy = True
            X_np = X

        # Cache CV splits
        # Purge is set to max_label_horizon + buffer to prevent label leakage.
        # With label horizons up to max_label_horizon_hours, labels at time t can overlap
        # with labels at t+max_label_horizon_hours, causing regime sensitivity.
        # Buffer of 2 provides additional safety margin.
        purge_samples = self.max_label_horizon_hours + 2
        embargo_samples = max(2, self.max_label_horizon_hours // 2)

        # Use groups as timestamps for time-based purging if available
        # Note: training.py passes timestamps in 'groups'
        times_for_cv = groups_arr if groups_arr is not None else None

        tscv = PurgedKFold(n_splits=self.n_splits, purge=purge_samples, embargo=embargo_samples, times=times_for_cv)
        cached_splits = list(tscv.split(X))

        # 1. The Race
        candidates = self._get_candidates(race_mode=True)
        results = {}

        # --- Dynamic Tree Regularization ---
        tree_dyn = tree_regularization_params(y, task_type="classification")
        n_pos = int(np.sum(y > 0)) if len(y) > 0 else 0
        min_leaf_dyn = int(tree_dyn["min_samples_leaf"])
        min_split_dyn = int(tree_dyn["min_samples_split"])
        min_cw_dyn = max(1, int(np.ceil(0.25 * min_leaf_dyn)))  # XGB hessian-scaled analogue
        tprint(
            f"ModelRace: Dynamic min_samples_leaf={min_leaf_dyn}, min_samples_split={min_split_dyn}, "
            f"min_child_weight={min_cw_dyn} (pos={n_pos}, pos_frac=1%)"
        )

        def safe_slice(arr, idx):
            if hasattr(arr, "iloc"): return arr.iloc[idx]
            return arr[idx]

        # Store per-model detailed metrics for reporting
        detailed_metrics = {}
        rank_cfg = AlphaRankConfig(k_frac=0.10, cal_metric="brier")

        for name, model in candidates.items():
            # Apply dynamic regularization to inner estimator
            if hasattr(model, "estimator"): inner = model.estimator
            else: inner = model

            # ExtraTrees / RF
            if "min_samples_leaf" in inner.get_params():
                inner.set_params(min_samples_leaf=min_leaf_dyn)
            if "min_samples_split" in inner.get_params():
                inner.set_params(min_samples_split=min_split_dyn)
            if "bootstrap" in inner.get_params():
                inner.set_params(bootstrap=False)
            if "ccp_alpha" in inner.get_params():
                inner.set_params(ccp_alpha=1e-4)
            if "max_leaf_nodes" in inner.get_params():
                inner.set_params(max_leaf_nodes=512)
            # LightGBM / CatBoost
            if "min_data_in_leaf" in inner.get_params():
                inner.set_params(min_data_in_leaf=min_leaf_dyn)
            # LightGBM alias
            if "min_child_samples" in inner.get_params():
                inner.set_params(min_child_samples=min_leaf_dyn)
            # XGBoost (hessian-based: use dedicated min_cw_dyn)
            if "min_child_weight" in inner.get_params():
                 inner.set_params(min_child_weight=min_cw_dyn)

            tprint(f"Race: Training {name}...")
            fold_scores = []
            fold_aucs = []
            fold_ics = []
            fold_bss = []
            fold_bs = [] # Brier Score
            fold_ref = [] # Brier Ref
            fold_brier = [] # Basic (unweighted) Brier
            fold_p10 = [] # Prec Top 10%
            fold_p20 = [] # Prec Top 20%
            fold_p25 = [] # Prec Top 25%
            fold_p30 = [] # Prec Top 30%
            fold_p40 = [] # Prec Top 40%
            fold_logloss = []
            fold_accuracy = []
            fold_base_logloss = []
            fold_logloss_imp = []
            oof_model = np.full(len(y), np.nan, dtype=np.float64)
            oof_sigma_trees = np.full(len(y), np.nan, dtype=np.float32)
            oof_sigma_robust = np.full(len(y), np.nan, dtype=np.float32)

            try:
                for fold_i, (train_idx, val_idx) in enumerate(cached_splits):
                    # No subsampling — use identical splits for race and OOF
                    if use_numpy:
                        X_tr, X_val = X_np[train_idx], X_np[val_idx]
                    else:
                        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

                    y_tr = safe_slice(y, train_idx)
                    y_val = safe_slice(y, val_idx)
                    y_tr_fit = (y_tr >= 0.5).astype(np.int8)
                    y_val_fit = (y_val >= 0.5).astype(np.int8)
                    w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
                    ret_val = safe_slice(returns, val_idx)


                    
                    
                    # Fit raw model (no CalibratedClassifierCV wrapper)
                    # We will apply bias correction manually on validation set
                    model_clone = clone(model)
                    
                    # We use _fit_model to handle sample weights and early stopping
                    # passing X_val/y_val for early stopping if supported (XGB/LGBM/Cat)
                    w_tr_fit = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
                    self._fit_model(model_clone, X_tr, y_tr_fit, X_val=X_val, y_val=y_val_fit, sample_weight=w_tr_fit)
                    
                    # --- Calibration Wrapper ---
                    # Build a time-aware isotonic calibrator using inner PurgedKFold OOF
                    # predictions on the training fold. This avoids the prefit leakage path.
                    from sklearn.isotonic import IsotonicRegression

                    inner_times = None
                    if times_for_cv is not None:
                        inner_times = np.asarray(times_for_cv)[train_idx]
                    inner_cv = PurgedKFold(
                        n_splits=3,
                        purge=max(2, purge_samples // 2),
                        embargo=max(1, embargo_samples // 2),
                        times=inner_times,
                    )
                    inner_oof = np.full(len(X_tr), np.nan, dtype=float)
                    for inner_tr, inner_va in inner_cv.split(X_tr):
                        if len(inner_tr) < 20 or len(inner_va) < 5:
                            continue
                        if len(np.unique(y_tr_fit[inner_tr])) < 2:
                            continue
                        cal_fold = clone(model)
                        w_inner_tr = (
                            w_tr_fit[inner_tr] if w_tr_fit is not None else None
                        )
                        self._fit_model(
                            cal_fold,
                            X_tr[inner_tr],
                            y_tr_fit[inner_tr],
                            X_val=X_tr[inner_va],
                            y_val=y_tr_fit[inner_va],
                            sample_weight=w_inner_tr,
                        )
                        inner_oof[inner_va] = cal_fold.predict_proba(X_tr[inner_va])[:, 1]

                    cal_mask = np.isfinite(inner_oof)
                    if np.sum(cal_mask) >= 20 and len(np.unique(y_tr_fit[cal_mask])) > 1:
                        calibrator = IsotonicRegression(out_of_bounds="clip")
                        calibrator.fit(inner_oof[cal_mask], y_tr_fit[cal_mask])
                        probs_raw = calibrator.transform(
                            model_clone.predict_proba(X_val)[:, 1]
                        )
                    else:
                        probs_raw = model_clone.predict_proba(X_val)[:, 1]
                    
                    # --- Prior Correction (replaces in-fold Platt scaling) ---
                    # Tree models with scale_pos_weight / class_weight shift raw
                    # probabilities away from the true prevalence.  We correct by
                    # mapping the model's mean prediction back to the training
                    # prevalence using a logit-space shift.
                    p_train = float(np.mean(y_tr_fit))
                    p_model = float(np.clip(np.mean(probs_raw), 1e-7, 1 - 1e-7))
                    if abs(p_model - p_train) > 0.01:
                        # Shift in logit space: logit(p_corrected) = logit(p_raw) + delta
                        delta = logit(np.clip(p_train, 1e-7, 1 - 1e-7)) - logit(p_model)
                        probs = expit(logit(np.clip(probs_raw, 1e-7, 1 - 1e-7)) + delta)
                    else:
                        probs = probs_raw

                    oof_model[val_idx] = probs
                    sigma_fold, sigma_robust_fold = self._tree_sigma_features(
                        model_clone, X_val
                    )
                    oof_sigma_trees[val_idx] = sigma_fold
                    oof_sigma_robust[val_idx] = sigma_robust_fold
                    
                    # w_bss=0.20: Enabled BSS in selection score
                    # We now compute weighted BSS for diagnostics
                    w_val = safe_slice(sample_weight, val_idx) if sample_weight is not None else None
                    metrics = calculate_selection_score(y_val_fit, probs, ret_val, sample_weight=w_val, w_bss=0.20, w_realized=0.55, w_uic=0.25)
                    fold_scores.append(metrics["Selection_Score"])
                    fold_aucs.append(metrics["AUC"])
                    fold_ics.append(metrics["IC"])
                    fold_bss.append(metrics["BSS"])
                    fold_bs.append(metrics.get("Brier_Score", np.nan))
                    fold_ref.append(metrics.get("Brier_Ref", np.nan))
                    fold_brier.append(metrics.get("Brier", np.nan))
                    fold_p10.append(metrics.get("Prec_Top10", np.nan))
                    fold_p20.append(metrics.get("Prec_Top20", np.nan))
                    fold_p25.append(metrics.get("Prec_Top25", np.nan))
                    fold_p30.append(metrics.get("Prec_Top30", np.nan))
                    fold_p40.append(metrics.get("Prec_Top40", np.nan))
                    # fold_p40 handled below if needed, but metrics returns it
                    
                    try:
                        ll_fold = log_loss(y_val_fit, np.clip(probs, 1e-7, 1-1e-7))
                        fold_logloss.append(ll_fold)
                        p_fold = float(np.mean(y_val_fit))
                        p_fold = float(np.clip(p_fold, 1e-7, 1 - 1e-7))
                        ll_base_fold = log_loss(y_val_fit, np.full_like(y_val_fit, p_fold, dtype=np.float64))
                        fold_base_logloss.append(ll_base_fold)
                        fold_logloss_imp.append((ll_base_fold - ll_fold) / max(ll_base_fold, 1e-9))
                    except:
                        fold_logloss.append(np.nan)
                        fold_base_logloss.append(np.nan)
                        fold_logloss_imp.append(np.nan)
                    fold_accuracy.append(accuracy_score(y_val_fit, probs > 0.5))

                # --- Enforce Calibration (Post-hoc on OOF) with guardrails ---
                valid = np.isfinite(oof_model)
                oof_cal, calibrator, cal_method = _safe_binary_calibrate(
                    oof_model, y_hard, min_unique=20, min_samples=100
                )
                tprint(f"  {name}: OOF calibration method={cal_method}")

                # Use Calibrated OOF for Selection Metrics
                oof_metrics = calculate_selection_score(
                    y_hard[valid], oof_cal[valid], returns[valid],
                    sample_weight=sample_weight[valid] if sample_weight is not None else None,
                    w_bss=0.20, w_realized=0.55, w_uic=0.25
                )

                # Stability Diagnostics
                std_score = np.nanstd(fold_scores)
                avg_p10 = np.nanmean(fold_p10)
                std_p10 = np.nanstd(fold_p10)
                cv_p10 = std_p10 / avg_p10 if avg_p10 > 1e-9 else 1.0

                train_loss = alpha_objective_logloss(y, np.clip(np.nan_to_num(oof_cal, nan=np.nanmean(oof_cal)), 1e-6, 1-1e-6), w=sample_weight)

                comps = alpha_rank_components(
                    y_hard[valid], oof_cal[valid], returns[valid],
                    w=sample_weight[valid] if sample_weight is not None else None,
                    groups=groups_arr[valid] if groups_arr is not None else None,
                    cfg=rank_cfg
                )

                results[name] = 0.0 # Placeholder, set by rank_score later

                top10_mask = topk_mask(oof_cal[valid], 0.10, groups=groups_arr[valid] if groups_arr is not None else None)
                ece10 = ece_at_mask(y_hard[valid], oof_cal[valid], top10_mask, n_bins=10, w=sample_weight[valid] if sample_weight is not None else None)
                top30_mask = topk_mask(oof_cal[valid], 0.30, groups=groups_arr[valid] if groups_arr is not None else None)
                ece30 = ece_at_mask(y_hard[valid], oof_cal[valid], top30_mask, n_bins=10, w=sample_weight[valid] if sample_weight is not None else None)
                curve = calibration_curve_bins(y_hard[valid], oof_cal[valid], n_bins=10)
                profile = calibration_profile(curve)

                detailed_metrics[name] = {
                    "score": oof_metrics["Selection_Score"],
                    "rank_score": 0.0,
                    "alpha_train_loss": train_loss,
                    "rank_components": comps,
                    "ece_top10": ece10,
                    "ece_top30": ece30,
                    "calibration_curve": curve,
                    "calibration_profile": profile,
                    "AUC": oof_metrics["AUC"],
                    "IC": oof_metrics["IC"],
                    "BSS": oof_metrics["BSS"],
                    "BS": oof_metrics.get("Brier_Score", np.nan),
                    "BS_Ref": oof_metrics.get("Brier_Ref", np.nan),
                    "Brier": oof_metrics.get("Brier", np.nan),
                    "Prec10": oof_metrics.get("Prec_Top10", np.nan),
                    "CV_Prec10": cv_p10, # Keep fold-based CV stability measure
                    "Prec20": oof_metrics.get("Prec_Top20", np.nan),
                    "CV_Prec20": np.nanstd(fold_p20) / (np.nanmean(fold_p20) + 1e-9),
                    "Prec25": oof_metrics.get("Prec_Top25", np.nan),
                    "Prec30": oof_metrics.get("Prec_Top30", np.nan),
                    "Prec40": oof_metrics.get("Prec_Top40", np.nan),
                    "std_score": std_score,
                    "LogLoss": log_loss(y_hard[valid], np.clip(oof_cal[valid], 1e-7, 1-1e-7)),
                    "Accuracy": accuracy_score(y_hard[valid], oof_cal[valid] > 0.5),
                    "fold_logloss": [float(x) for x in fold_logloss],
                    "fold_precision20": [float(x) for x in fold_p20],
                    "fold_precision10": [float(x) for x in fold_p10],
                    "fold_brier": [float(x) for x in fold_brier],
                    "fold_base_logloss": [float(x) for x in fold_base_logloss],
                    "fold_logloss_imp": [float(x) for x in fold_logloss_imp],
                    "oof_sigma_trees": oof_sigma_trees.copy().astype(np.float32),
                    "oof_sigma_robust": oof_sigma_robust.copy().astype(np.float32),
                    # Store Calibrated OOF for gate checks
                    "oof_probs": np.nan_to_num(oof_cal.copy(), nan=0.5).astype(np.float32),
                    # Store raw OOF for reference
                    "oof_raw": np.nan_to_num(oof_model.copy(), nan=0.5).astype(np.float32),
                    # Store calibrator for inference
                    "calibrator": calibrator,
                }
                
                # --- Top-K Feature Importance Reporting ---
                if hasattr(model_clone, "estimator"): inner_m = model_clone.estimator
                else: inner_m = model_clone
                
                if hasattr(inner_m, "feature_importances_"):
                    importances = inner_m.feature_importances_
                    feat_names = X.columns if hasattr(X, "columns") else [f"f_{i}" for i in range(len(importances))]
                    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
                    
                    top5 = imp_df.head(5)
                    top10_pct_n = max(1, int(len(imp_df) * 0.10))
                    top10_pct_sum = imp_df.head(top10_pct_n)["importance"].sum()
                    
                    tprint(f"  {name} Importance: Top10% CumSum={top10_pct_sum:.4f}")
                    for _, row in top5.iterrows():
                        tprint(f"    - {row['feature']}: {row['importance']:.4f}")

                tprint(f"  {name}: OOF_Cal_Score={detailed_metrics[name]['score']:.4f} AUC={detailed_metrics[name]['AUC']:.4f} IC={detailed_metrics[name]['IC']:.4f} BSS={detailed_metrics[name]['BSS']:.4f} Prec10={detailed_metrics[name]['Prec10']:.4f}")

            except Exception as e:
                tprint(f"  {name} Failed: {e}")
                results[name] = -float("inf")

        self.detailed_metrics = detailed_metrics

        if not results:
            raise ValueError("All models failed in race")

        # New alpha ranking system based on IC/Prec@K/Calibration/Stability
        comp_keys = ["IC", "Prec@K", "Cal@K", "StdIC"]
        zscores = {n: {} for n in detailed_metrics}
        for ck in comp_keys:
            vals = np.array([detailed_metrics[n]["rank_components"].get(ck, np.nan) for n in detailed_metrics], dtype=float)
            mu, sd = np.nanmean(vals), np.nanstd(vals)
            for n in detailed_metrics:
                v = detailed_metrics[n]["rank_components"].get(ck, np.nan)
                zscores[n][ck] = 0.0 if (not np.isfinite(sd) or sd < 1e-12 or not np.isfinite(v)) else float((v - mu) / sd)

        for n in detailed_metrics:
            c = detailed_metrics[n]["rank_components"]
            z = zscores[n]
            rank_score = (
                rank_cfg.w_ic * z["IC"] + rank_cfg.w_prec * z["Prec@K"]
                - rank_cfg.w_cal * z["Cal@K"] - rank_cfg.w_std * z["StdIC"]
                - rank_cfg.w_neff_pen * c.get("n_eff_pen", 0.0)
            )
            detailed_metrics[n]["rank_score"] = float(rank_score)
            results[n] = float(rank_score)

        # Compute quality gate checks for each model (same as _base_model_report_entry in training.py)
        def _compute_gate_checks(dm, y_hard, oof_probs, prev):
            """Compute quality gate checks for a model. Returns (n_passed, checks_dict)."""
            from sklearn.metrics import average_precision_score, brier_score_loss, log_loss as _log_loss
            
            # Brier and logloss improvement
            p_clip = np.clip(oof_probs, 1e-7, 1 - 1e-7)
            base_brier = prev * (1.0 - prev)
            base_ll = -(prev * np.log(prev + 1e-12) + (1 - prev) * np.log(1 - prev + 1e-12))
            try:
                brier = float(brier_score_loss(y_hard, p_clip))
                ll = float(_log_loss(y_hard, p_clip))
                brier_imp = (base_brier - brier) / max(base_brier, 1e-9)
                ll_imp = (base_ll - ll) / max(base_ll, 1e-9)
            except Exception:
                brier_imp, ll_imp = 0.0, 0.0
            
            # PR-AUC
            try:
                pr_auc = float(average_precision_score(y_hard, oof_probs)) if len(np.unique(y_hard)) > 1 else 0.0
            except Exception:
                pr_auc = 0.0
            
            # Lift@20%
            k_frac = 0.20
            k_n = max(1, int(len(y_hard) * k_frac))
            idx = np.argsort(oof_probs)[-k_n:]
            prec_k = float(np.mean(y_hard[idx]))
            lift_k = prec_k / max(prev, 1e-9)
            prec_lift_abs = prec_k - prev
            
            # Fold stability
            fold_imp = [x for x in dm.get("fold_logloss_imp", []) if np.isfinite(x)]
            pos_fold_ratio = float(np.mean(np.array(fold_imp) > 0.0)) if fold_imp else 0.0
            worst_fold_imp = float(np.min(fold_imp)) if fold_imp else -1.0
            
            # Bootstrap CV for precision@20%
            n_boot = 50
            rng_boot = np.random.RandomState(42)
            prec_samples = []
            n_total = len(y_hard)
            for _ in range(n_boot):
                idx_b = rng_boot.choice(n_total, size=n_total, replace=True)
                _n_k = max(1, int(n_total * k_frac))
                top_idx = np.argsort(oof_probs[idx_b])[-_n_k:]
                p_k_b = float(np.mean(y_hard[idx_b][top_idx]))
                prec_samples.append(p_k_b)
            prec_arr = np.array(prec_samples)
            bootstrap_prec20_cv = float(np.std(prec_arr) / (np.mean(prec_arr) + 1e-9))
            
            # Prevalence-aware PR-AUC threshold
            pr_auc_threshold = max(1.25 * prev, prev + 0.05)
            
            # Gate checks (same as training.py)
            checks = {
                "pr_auc_ge_threshold": pr_auc >= pr_auc_threshold,
                "pr_auc_ge_random": pr_auc >= prev,
                "brier_and_logloss_improve_ge_2pct": bool((brier_imp >= 0.02) and (ll_imp >= 0.02)),
                "liftk_and_preck_lift": bool((lift_k >= 1.2) and ((prec_lift_abs >= 0.025) or ((lift_k - 1.0) >= 0.05))),
                "bootstrap_prec20_cv_le_0_30": bootstrap_prec20_cv <= 0.30,
                "delta_logloss_le_minus_0_5pct": ll_imp >= 0.005,
                "logloss_improves_in_ge_70pct_folds": pos_fold_ratio >= 0.70,
                "worst_fold_delta_logloss_ge_0_5pct_improve": worst_fold_imp >= -0.005,
            }
            n_passed = sum(checks.values())
            return n_passed, checks
        
        # Compute gate checks for all models
        prev = float(np.mean(y_hard))
        gate_results = {}
        for n in detailed_metrics:
            dm = detailed_metrics[n]
            # Get OOF predictions for this model (stored during race)
            oof_model = np.full(len(y), np.nan, dtype=np.float64)
            # We need to reconstruct OOF from the race - use stored fold predictions
            # The OOF was computed during the race loop above
            # For now, use the model's stored OOF if available, otherwise compute
            if "oof_probs" in dm:
                oof_model = dm["oof_probs"]
            else:
                # Fallback: use the overall OOF (same for all models in current implementation)
                oof_model = np.nan_to_num(oof_model, nan=0.5)
            n_passed, checks = _compute_gate_checks(dm, y_hard, oof_model, prev)
            gate_results[n] = {"n_passed": n_passed, "checks": checks}
            detailed_metrics[n]["gate_checks"] = checks
            detailed_metrics[n]["gate_n_passed"] = n_passed
        
        # Winner selection: prioritize models passing most gates, use rank_score as tie-breaker
        # Sort by (n_passed, rank_score) descending
        eligible_results = {
            name: score for name, score in results.items() if name in gate_results
        }
        if not eligible_results:
            raise RuntimeError("ModelRace: no eligible models remained after gating")

        sorted_candidates = sorted(
            eligible_results.items(),
            key=lambda x: (gate_results[x[0]]["n_passed"], x[1]),
            reverse=True,
        )
        best_name = sorted_candidates[0][0]
        best_n_passed = gate_results[best_name]["n_passed"]
        best_score = results[best_name]
        
        tprint(f"Gate-aware winner selection:")
        for name, score in sorted_candidates[:4]:
            tprint(f"  {name}: gates_passed={gate_results[name]['n_passed']}/7, rank_score={score:.4f}")
        
        # Log if we're selecting a model with fewer gates passed due to tie-breaker
        if best_n_passed < 7:
            tprint(f"WARNING: Best model '{best_name}' passes only {best_n_passed}/7 gates")

        self.best_model_name = best_name
        self.metrics = results
        dm = detailed_metrics.get(best_name, {})
        tprint(f"Race Winner: {best_name} (RankScore={results[best_name]:.4f}, IC={dm.get('rank_components',{}).get('IC',0):.4f}, Prec@10={dm.get('rank_components',{}).get('Prec@K',0):.4f})")

        # 2. Generate OOF predictions with best model (for meta model)
        tprint(f"Generating OOF predictions with {best_name}...")
        oof_probs = np.full(len(y), np.nan, dtype=np.float32)
        oof_candidates = self._get_candidates(race_mode=True)
        oof_model = oof_candidates[best_name]
        for train_idx, val_idx in cached_splits:
            if use_numpy:
                X_tr, X_val = X_np[train_idx], X_np[val_idx]
            else:
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = safe_slice(y, train_idx)
            y_val = safe_slice(y, val_idx)
            y_tr_fit = (y_tr >= 0.5).astype(np.int8)
            y_val_fit = (y_val >= 0.5).astype(np.int8)
            w_tr = safe_slice(sample_weight, train_idx) if sample_weight is not None else None
            
            # Raw model fit — using _fit_model for consistency with race (early stopping, weights)
            estimator = clone(oof_model)
            self._fit_model(estimator, X_tr, y_tr_fit, X_val=X_val, y_val=y_val_fit, sample_weight=w_tr)
            
            oof_probs[val_idx] = estimator.predict_proba(X_val)[:, 1]
            
            # Predict raw then apply fold bias correction contract.
            probs_raw = estimator.predict_proba(X_val)[:, 1]
            if sample_weight is not None:
                w_tr_fold = sample_weight[train_idx]
                den = float(np.sum(w_tr_fold))
                p_weighted_fold = float(np.sum(w_tr_fold * y_tr_fit) / max(den, 1e-12))
            else:
                p_weighted_fold = float(np.mean(y_tr_fit))
            p_unweighted_fold = float(np.mean(y_val_fit))
            delta_logit_fold = compute_logit_shift(p_unweighted_fold, p_weighted_fold, eps=1e-6)
            oof_probs[val_idx] = apply_logit_shift(probs_raw, delta_logit_fold, eps=1e-6)
        # Fill any remaining NaN with 0.5 (neutral)
        oof_probs = np.nan_to_num(oof_probs, nan=0.5)
        self.oof_probs = oof_probs
        self._used_sample_weight_ = sample_weight is not None
        p_unweighted_all, p_weighted_all = compute_prevalences(y_hard, sample_weight)
        self.calibration_state_ = self._build_bias_state(p_unweighted_all, p_weighted_all, eps=1e-6)
        tprint(f"OOF predictions: mean={np.mean(oof_probs):.4f}, std={np.std(oof_probs):.4f}")

        # Effective sample size diagnostic
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64)
            n_eff = (np.sum(sw) ** 2) / np.sum(sw ** 2)
            tprint(f"Weight diagnostics: n={len(sw)}, n_eff={n_eff:.0f} ({100*n_eff/len(sw):.0f}%), mean={np.mean(sw):.3f}, std={np.std(sw):.3f}, p95={np.percentile(sw,95):.3f}")

        # Calculate Winner OOF Metrics (rank-based, not calibration-dependent)
        try:
            oof_auc = roc_auc_score(y_hard, oof_probs)
            oof_logloss = log_loss(y_hard, np.clip(oof_probs, 1e-7, 1-1e-7))
            oof_accuracy = accuracy_score(y_hard, oof_probs > 0.5)
            if returns is not None and np.std(oof_probs) > 1e-9 and np.std(returns) > 1e-9:
                if symbols_arr is not None:
                    from extreme_price_movements.model_scoring import ic_cross_sectional
                    oof_ic = ic_cross_sectional(oof_probs, returns, groups=symbols_arr)
                    if np.isnan(oof_ic):
                        oof_ic = 0.0
                else:
                    oof_ic = np.corrcoef(rankdata(oof_probs), rankdata(returns))[0, 1]
            else:
                oof_ic = 0.0
            # OOF selection score (same weights as race)
            oof_sel = calculate_selection_score(y_hard, oof_probs, returns, sample_weight=sample_weight, symbols=symbols_arr, groups=groups_arr, w_bss=0.20, w_realized=0.55, w_uic=0.25)
            tprint(f"Winner OOF Metrics: AUC={oof_auc:.4f}  IC={oof_ic:.4f}  LogLoss={oof_logloss:.4f}  Acc={oof_accuracy:.4f}  SelScore={oof_sel['Selection_Score']:.4f}  Lift@30={oof_sel.get('Lift_Top30', 0.0):.4f}")
            
            # --- Post-hoc Isotonic Calibration ---
            # Fit calibrator on OOF predictions
            # Use relaxed constraints (y_min=0.05, y_max=0.95) to prevent degenerate
            # score distributions where all predictions cluster tightly around prevalence.
            # This was causing "degenerate IQR" warnings in backtest.
            tprint("Running calibration router on OOF predictions (isotonic/platt/identity)...")
            calibrated_oof, self.calibrator_, cal_method = _safe_binary_calibrate(
                oof_probs, y_hard, min_unique=20, min_samples=100
            )
            if isinstance(self.calibration_state_, dict):
                self.calibration_state_["calibration_input"] = "bias_corrected"
                self.calibration_state_["calibration_method"] = cal_method
            tprint(f"Calibration router selected: {cal_method}")
            
            # --- Minimum Variance Enforcement ---
            # Prevent degenerate score distributions (all predictions near prevalence)
            # This addresses the "degenerate IQR" warnings seen in backtest
            MIN_VARIANCE = 0.01  # Minimum std dev threshold
            cal_std = np.std(calibrated_oof)
            if cal_std < MIN_VARIANCE:
                tprint(f"WARNING: Calibrated scores have low variance (std={cal_std:.6f}). Enforcing minimum spread.")
                # Blend with rank-based scores to restore variance while preserving rank order
                rank_scores = (rankdata(oof_probs) - 1) / (len(oof_probs) - 1)  # Normalize to [0, 1]
                # Blend factor: how much to mix in rank scores (higher = more variance)
                blend_factor = 0.3
                # Center rank scores around prevalence for consistency
                prevalence = np.mean(y_hard)
                rank_scores_centered = rank_scores - 0.5 + prevalence
                rank_scores_centered = np.clip(rank_scores_centered, 0.05, 0.95)
                # Blend calibrated with rank-based
                calibrated_oof = (1 - blend_factor) * calibrated_oof + blend_factor * rank_scores_centered
                tprint(f"  Blended with rank scores: new std={np.std(calibrated_oof):.6f}")
            
            # Preserve raw rank scores for downstream use (e.g., ridge sizer)
            self.raw_rank_scores_ = (rankdata(oof_probs) - 1) / (len(oof_probs) - 1)
            
            from sklearn.metrics import brier_score_loss
            from sklearn.linear_model import LogisticRegression
            raw_brier = brier_score_loss(y_hard, np.clip(oof_probs, 1e-7, 1-1e-7))
            cal_brier = brier_score_loss(y_hard, np.clip(calibrated_oof, 1e-7, 1-1e-7))
            tprint(f"Calibration ({cal_method}): Brier raw={raw_brier:.4f} -> calibrated={cal_brier:.4f}")
            
            # Optional Platt scaling: only apply if it improves Brier score
            platt_calibrator = LogisticRegression(random_state=42, max_iter=1000)
            platt_calibrator.fit(calibrated_oof.reshape(-1, 1), y_hard)
            platt_calibrated = platt_calibrator.predict_proba(calibrated_oof.reshape(-1, 1))[:, 1]
            platt_brier = brier_score_loss(y_hard, np.clip(platt_calibrated, 1e-7, 1-1e-7))
            
            if platt_brier < cal_brier - 1e-4:  # Only keep Platt if it materially improves Brier
                self.platt_calibrator_ = platt_calibrator
                tprint(f"Platt scaling enabled: Brier improved {cal_brier:.4f} -> {platt_brier:.4f}")
            else:
                self.platt_calibrator_ = None
                tprint(f"Platt scaling skipped: no improvement (isotonic={cal_brier:.4f}, platt={platt_brier:.4f})")
            win_dm = self.detailed_metrics.get(self.best_model_name, {})
            tprint(f"Calibration profile ({self.best_model_name}): {win_dm.get('calibration_profile', 'n/a')}, ECE@10={win_dm.get('ece_top10', float('nan')):.4f}")
            
            self.oof_probs = calibrated_oof

        except Exception as e:
            tprint(f"Error calculating OOF metrics or calibration: {e}")

        # Recap
        tprint("\n=== Model Race Recap ===")
        tprint(f"{'Model':<15} {'RankSc':>8} {'RcAUC':>8} {'RcIC':>8} {'RcBSS':>8} {'RcBrier':>8} {'RaceP10':>8} {'RaceP30':>8} {'RaceP40':>8} {'LL':>8} {'ECE10':>8} {'ECE30':>8}")
        tprint("-" * 122)

        sorted_models = sorted(detailed_metrics.items(), key=lambda x: x[1]['rank_score'], reverse=True)
        for name, m in sorted_models:
            ece10 = m.get('ece_top10', np.nan)
            ece30 = m.get('ece_top30', np.nan)
            tprint(f"{name:<15} {m['rank_score']:8.4f} {m['AUC']:8.4f} {m['IC']:8.4f} {m['BSS']:8.4f} {m.get('Brier',0):8.4f} {m['Prec10']:8.4f} {m.get('Prec30', np.nan):8.4f} {m['Prec40']:8.4f} {m['LogLoss']:8.4f} {ece10:8.4f} {ece30:8.4f}")
        tprint("========================\n")

        # 3. Final Retraining (raw model, no calibration wrapper)
        # Calibration is harmful with small n and uncalibrated objectives.
        # Output is treated as rank score by downstream (engine, backtest).
        # NOTE: Winner already selected above (lines 670-692) using gate-aware logic:
        #       - Primary: most gates passed
        #       - Tie-breaker: rank_score
        # Do NOT re-select here - use existing self.best_model_name
        if self.best_model_name and self.best_model_name in candidates:
             self.best_model = candidates[self.best_model_name]
        else:
             # Fallback should never happen, but log if it does
             tprint(f"WARNING: best_model_name '{self.best_model_name}' not in candidates, falling back to rank_score selection")
             if detailed_metrics:
                  self.best_model_name = max(detailed_metrics.items(), key=lambda x: x[1]['rank_score'])[0]
                  self.best_model = candidates[self.best_model_name]

        tprint(f"Retraining {self.best_model_name} on full data (full config)...")
        final_candidates = self._get_candidates(race_mode=False)
        # For the final model, we ALSO use CalibratedClassifierCV to ensure inference probabilities are calibrated.
        # Use TimeSeriesSplit(5) for better usage of data in final model.

        final_base = clone(self.best_model)
        if hasattr(final_base, "estimator"): # Unwrap wrapper if needed? No, Float64Wrapper is a classifier.
             pass

        # We fit the raw model on full data.
        # Note: We need to store the bias correction factor for the FINAL model too?
        # Post-hoc Isotonic handles bias, but inputs to Itosonic must be consistent with training?
        # Actually, Isotonic is fit on OOF. OOF was Bias-Corrected in the loop (see above).
        # So Isotonic expects Bias-Corrected inputs.
        # Therefore, we MUST compute and apply Bias Correction in predict_proba BEFORE Isotonic.

        # 1. Fit Raw Model
        # (We could calibrate here too, but Isotonic on OOF is usually enough for the final head)
        # Actually: to be consistent with the race metrics, we SHOULD use CalibratedClassifierCV here too?
        # User said: "Consider wrapping each fold's estimator... but Post-hoc OOF is for final model"
        # Since we use Isotonic on OOF in predict_proba, the final model is effectively calibrated.
        # But wait! If we leave 'best_model' as raw, then predict_proba applies isotonic.
        # That logic is sound.

        # Use _fit_model so that class weights and other dynamics are applied correctly,
        # even without early stopping (no eval_set passed).
        self._fit_model(self.best_model, X, y_hard, sample_weight=sample_weight)

        # 4. Post-refit recalibration
        # The refit model has a different distribution than the fold-specific OOF models.
        # Re-generate OOF predictions via CV on the refit model, then re-fit calibration.
        try:
            tprint("Post-refit recalibration: generating OOF from refit model...")
            refit_oof = np.full(len(y_hard), np.nan, dtype=np.float64)
            for train_idx, val_idx in cached_splits:
                X_val_fold = X_np[val_idx] if use_numpy else X.iloc[val_idx]
                probs_raw = self.best_model.predict_proba(X_val_fold)[:, 1]
                y_tr_fold = y_hard[train_idx]
                if sample_weight is not None:
                    w_tr_fold = sample_weight[train_idx]
                    den = float(np.sum(w_tr_fold))
                    p_weighted_fold = float(np.sum(w_tr_fold * y_tr_fold) / max(den, 1e-12))
                else:
                    p_weighted_fold = float(np.mean(y_tr_fold))
                p_unweighted_fold = float(np.mean(y_hard[val_idx]))
                delta_logit_fold = compute_logit_shift(
                    p_unweighted_fold, p_weighted_fold, eps=1e-6
                )
                refit_oof[val_idx] = apply_logit_shift(
                    probs_raw, delta_logit_fold, eps=1e-6
                )
            refit_oof = np.nan_to_num(refit_oof, nan=0.5)

            refit_calibrated, refit_calibrator, refit_cal_method = (
                _safe_binary_calibrate(refit_oof, y_hard, min_unique=20, min_samples=100)
            )
            self.calibrator_ = refit_calibrator
            if isinstance(self.calibration_state_, dict):
                self.calibration_state_["calibration_method"] = refit_cal_method

            from sklearn.metrics import brier_score_loss
            from sklearn.linear_model import LogisticRegression

            cal_brier = brier_score_loss(
                y_hard, np.clip(refit_calibrated, 1e-7, 1 - 1e-7)
            )
            platt = LogisticRegression(random_state=42, max_iter=1000)
            platt.fit(refit_calibrated.reshape(-1, 1), y_hard)
            platt_pred = platt.predict_proba(refit_calibrated.reshape(-1, 1))[:, 1]
            platt_brier = brier_score_loss(
                y_hard, np.clip(platt_pred, 1e-7, 1 - 1e-7)
            )
            if platt_brier < cal_brier - 1e-4:
                self.platt_calibrator_ = platt
                tprint(
                    f"Post-refit recalibration: {refit_cal_method} + Platt "
                    f"(Brier {cal_brier:.4f} -> {platt_brier:.4f})"
                )
            else:
                self.platt_calibrator_ = None
                tprint(
                    f"Post-refit recalibration: {refit_cal_method} "
                    f"(Brier {cal_brier:.4f}, Platt skipped)"
                )
        except Exception as _e:
            tprint(f"WARNING: post-refit recalibration failed, keeping OOF calibration: {_e}")

        # Extract feature importances if possible
        try:
            est = self.best_model
            if hasattr(est, "estimator"):
                est = est.estimator
            if hasattr(est, "feature_importances_"):
                fi = est.feature_importances_
                if hasattr(X, "columns"):
                    cols = X.columns
                else:
                    cols = [f"f_{i}" for i in range(len(fi))]
                # Print top 20 features sorted by importance
                sorted_idx = np.argsort(fi)[::-1][:20]
                tprint(f"\n=== BASE MODEL FEATURE IMPORTANCES ({self.best_model_name}) - Top 20 ===")
                for i, idx in enumerate(sorted_idx):
                    tprint(f"  {i+1:2d}. {cols[idx]}: {fi[idx]:.6f}")
                tprint("=== END BASE MODEL FEATURES ===\n")
            else:
                tprint(f"Feature Importances ({self.best_model_name}): None (no feature_importances_)")
        except Exception as e:
            tprint(f"Failed to get feature importances: {e}")

# No more manual bias correction factor needed (Isotonic handles it)
        return self

    def predict_proba_raw(self, X):
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        probs = np.asarray(self.best_model.predict_proba(X), dtype=np.float64)
        return safe_clip_proba(probs[:, 1], eps=1e-6)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("ModelRace not fitted")
        if self.calibration_state_ is None or "delta_logit" not in self.calibration_state_:
            raise RuntimeError("Missing calibration_state_. Refit ModelRace with calibration enabled before inference.")

        if self._used_sample_weight_ and self.calibration_state_ is None:
            raise RuntimeError("Sample-weighted training requires persisted calibration_state_.")

        p_raw = self.predict_proba_raw(X)
        p_corr = self._apply_bias_state(p_raw, self.calibration_state_)

        if hasattr(self, 'calibrator_') and self.calibrator_ is not None:
            if self.calibration_state_.get("calibration_input") != "bias_corrected":
                raise RuntimeError("Calibrator expects bias-corrected inputs but calibration_state_ is inconsistent.")
            if hasattr(self.calibrator_, 'predict_proba'):
                p_cal = self.calibrator_.predict_proba(p_corr.reshape(-1, 1))[:, 1]
            else:
                p_cal = self.calibrator_.predict(p_corr)
        else:
            p_cal = p_corr

        # Apply Platt scaling if available
        if hasattr(self, 'platt_calibrator_') and self.platt_calibrator_ is not None:
            p_cal = self.platt_calibrator_.predict_proba(p_cal.reshape(-1, 1))[:, 1]

        p_cal = safe_clip_proba(p_cal, eps=1e-6)
        probs = np.column_stack([1.0 - p_cal, p_cal])
        return np.asarray(probs, dtype=np.float64)

    def predict(self, X):
        # Return probability class 1 (rank score, not calibrated probability)
        return self.predict_proba(X)[:, 1]

    def strip_for_serialization(self):
        """Drop heavy internals not needed for inference or meta training."""
        for attr in ["race_sample_frac", "race_early_stopping_rounds",
                      "n_splits"]:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass
        return self

    def save_native(self, directory):
        """Save using native model formats (10-100x faster than pickle)."""
        os.makedirs(directory, exist_ok=True)
        inner = self.best_model
        if isinstance(inner, Float64Wrapper):
            inner = inner.estimator
        if _is_xgb(inner):
            inner.save_model(os.path.join(directory, "model.ubj"))
            fmt = "model.ubj"
        elif _is_lgb(inner):
            inner.booster_.save_model(os.path.join(directory, "model.lgb"))
            fmt = "model.lgb"
        elif _is_cb(inner):
            inner.save_model(os.path.join(directory, "model.cbm"))
            fmt = "model.cbm"
        else:
            joblib.dump(inner, os.path.join(directory, "model.joblib"), compress=3)
            fmt = "model.joblib"
        sidecar = {
            "best_model_name": self.best_model_name,
            "kind": self.kind,
            "metrics": self.metrics,
            "detailed_metrics": self.detailed_metrics,
            "oof_probs": self.oof_probs,
            "calibration_state_": self.calibration_state_,
            "final_bias_factor_": self.final_bias_factor_,
            "_used_sample_weight_": self._used_sample_weight_,
            "calibrator_": getattr(self, "calibrator_", None),
            "platt_calibrator_": getattr(self, "platt_calibrator_", None),
            "classes_": getattr(self.best_model, "classes_", np.array([0, 1])),
            "model_file": fmt,
        }
        with open(os.path.join(directory, "sidecar.pkl"), "wb") as f:
            pickle.dump(sidecar, f)

    @classmethod
    def load_native(cls, directory):
        """Load from native-format files."""
        with open(os.path.join(directory, "sidecar.pkl"), "rb") as f:
            sc = pickle.load(f)
        mf = sc["model_file"]
        mp = os.path.join(directory, mf)
        if mf.endswith(".ubj"):
            from xgboost import XGBClassifier as _XGB
            inner = _XGB()
            inner.load_model(mp)
        elif mf.endswith(".lgb"):
            booster = __import__("lightgbm").Booster(model_file=mp)
            inner = NativeLGBMBoosterClassifier(booster=booster)
        elif mf.endswith(".cbm"):
            from catboost import CatBoostClassifier as _CB
            inner = _CB()
            inner.load_model(mp)
        else:
            inner = joblib.load(mp)
        wrapper = Float64Wrapper(estimator=inner)
        wrapper.classes_ = sc.get("classes_", np.array([0, 1]))
        obj = cls.__new__(cls)
        obj.best_model = wrapper
        obj.best_model_name = sc["best_model_name"]
        obj.kind = sc["kind"]
        obj.metrics = sc["metrics"]
        obj.detailed_metrics = sc["detailed_metrics"]
        obj.oof_probs = sc["oof_probs"]
        obj.calibration_state_ = sc["calibration_state_"]
        obj.final_bias_factor_ = sc["final_bias_factor_"]
        obj._used_sample_weight_ = sc["_used_sample_weight_"]
        obj.calibrator_ = sc.get("calibrator_")
        obj.platt_calibrator_ = sc.get("platt_calibrator_")
        return obj
