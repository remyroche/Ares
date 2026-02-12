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
from extreme_price_movements.calibration import (
    safe_clip_proba,
    compute_prevalences,
    compute_logit_shift,
    apply_logit_shift,
)


def calculate_selection_score(y_true, y_prob, y_ret, sample_weight=None, **kwargs):
    """Backward-compatible wrapper exposing legacy AUC/BSS/IC keys for tests/logs."""
    from extreme_price_movements.metrics import calculate_selection_score as _calc
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
        out["IC"] = float(np.corrcoef(rankdata(y_prob), rankdata(y_ret))[0, 1])
    except Exception:
        out["IC"] = 0.0
    return out


class Float64Wrapper(BaseEstimator, ClassifierMixin):
    """Wraps a classifier so predict_proba / decision_function always return float64.
    Some estimators (e.g. XGBoost) return float32 predictions by default."""
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        if sample_weight is not None:
            self.estimator.fit(X, y, sample_weight=sample_weight)
        else:
            self.estimator.fit(X, y)
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
    def __init__(self, kind="long", n_splits=5, race_sample_frac=0.5, race_early_stopping_rounds=50, max_label_horizon_hours=8):
        self.kind = kind
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
        # Configure models
        # ExtraTrees, XGBoost, LightGBM, CatBoost
        
        candidates = {}
        
        # 1. ExtraTrees
        et_params = {
            "n_estimators": 200 if race_mode else 800,
            "max_depth": 7,
            "min_samples_leaf": 50,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42
        }
        candidates["extratrees"] = Float64Wrapper(ExtraTreesClassifier(**et_params))

        # 2. XGBoost
        xgb_params = {
            "n_estimators": 5 if race_mode else 10,  # Increase from 3/5
            "num_parallel_tree": 150 if race_mode else 400,
            "max_depth": 5,  # Increased from 4 for better separation
            "learning_rate": 0.05,
            "reg_lambda": 5.0,              # L2 (default=1 is often too weak)
            "reg_alpha": 0.0,               # keep 0 initially
            "min_child_weight": 20,
            "tree_method": "hist",
            "gamma": 1.0,                   
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "enable_categorical": False,
            "eval_metric": ["auc", "aucpr"],  # Track PR-AUC for ranking quality
        }
        if XGBClassifier is not None:
            candidates["xgboost"] = Float64Wrapper(XGBClassifier(**xgb_params))

        # 3. LightGBM
        lgb_params = {
            "n_estimators": 300 if race_mode else 1000,  # Increase from 200/800
            "max_depth": 6,  # Increased from 4 for better separation
            "learning_rate": 0.05,
            "subsample": 0.8,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "lambda_l1": 0.0,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
            "is_unbalance": True,  # Handle class imbalance for better PR-AUC
            "objective": "xentropy",
            "metric": ["auc", "average_precision"],
        }
        if LGBMClassifier is not None:
            candidates["lightgbm"] = Float64Wrapper(LGBMClassifier(**lgb_params))

        # 4. CatBoost
        cb_params = {
            "iterations": 200 if race_mode else 800,
            "l2_leaf_reg": 10.0,        
            "random_strength": 1.0,     
            "bagging_temperature": 1.0,         
            "depth": 5,  # Increased from 4 for better separation
            "learning_rate": 0.05,
            "verbose": 0,
            "thread_count": -1,
            "random_seed": 42,
            "allow_writing_files": False,
            "eval_metric": "PRAUC",  # Direct PR-AUC optimization
            "auto_class_weights": "Balanced",  # Handle class imbalance
        }
        if CatBoostClassifier is not None:
            candidates["catboost"] = Float64Wrapper(CatBoostClassifier(**cb_params))
        
        return candidates

    def _fit_model(self, model, X_tr, y_tr, X_val=None, y_val=None, sample_weight=None):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        pos_weight = self._compute_pos_weight(y_tr)

        if isinstance(model, ScaledLogisticRegression):
            # Safe to set because we updated __init__
            model.set_params(class_weight={0: 1.0, 1: pos_weight})
        elif isinstance(model, ExtraTreesClassifier):
            model.set_params(class_weight={0: 1.0, 1: pos_weight})

        if CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
            model.set_params(scale_pos_weight=pos_weight)
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": (X_val, y_val),
                    "early_stopping_rounds": self.race_early_stopping_rounds,
                    "use_best_model": True,
                })
        elif XGBClassifier is not None and isinstance(model, XGBClassifier):
            model.set_params(scale_pos_weight=pos_weight, eval_metric="auc")
            if X_val is not None and y_val is not None:
                fit_kwargs.update({
                    "eval_set": [(X_val, y_val)],
                    "verbose": False,
                    # early_stopping_rounds deprecated in fit, use constructor or callbacks if needed
                    # For simple race, we can omit it or relying on constructor
                })
        elif LGBMClassifier is not None and isinstance(model, LGBMClassifier):
            model.set_params(scale_pos_weight=pos_weight)
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

    def fit(self, X, y, sample_weight=None, returns=None, groups=None):
        """
        X: features
        y: binary target
        sample_weight: weights for training
        returns: continuous returns for IC calculation (validation)
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
        tscv = PurgedKFold(n_splits=self.n_splits, purge=purge_samples, embargo=embargo_samples)
        cached_splits = list(tscv.split(X))

        # 1. The Race
        candidates = self._get_candidates(race_mode=True)
        results = {}

        # --- Dynamic Regularization (User Requested) ---
        n_samples = len(y)
        min_leaf_dyn = max(75, int(n_samples / 50))
        tprint(f"ModelRace: Dynamic min_samples_leaf set to {min_leaf_dyn} (n={n_samples})")

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

            # ExtraTrees
            if "min_samples_leaf" in inner.get_params():
                inner.set_params(min_samples_leaf=min_leaf_dyn)
            # LightGBM / CatBoost
            if "min_data_in_leaf" in inner.get_params():
                inner.set_params(min_data_in_leaf=min_leaf_dyn)
            # LightGBM alias
            if "min_child_samples" in inner.get_params():
                inner.set_params(min_child_samples=min_leaf_dyn)
            # XGBoost (approximate mapping: hessian ~ 0.25 * count)
            if "min_child_weight" in inner.get_params():
                 inner.set_params(min_child_weight=max(1, int(min_leaf_dyn / 4)))

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
            fold_p40 = [] # Prec Top 40%
            fold_logloss = []
            fold_accuracy = []
            fold_base_logloss = []
            fold_logloss_imp = []
            oof_model = np.full(len(y), np.nan, dtype=np.float64)

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
                    
                    # Predict raw (biased) probabilities on validation
                    probs_raw = model_clone.predict_proba(X_val)[:, 1]
                    
                    # --- In-Fold Platt Scaling (User Requested) ---
                    # Maps raw score (which might be overconfident/biased) to well-calibrated prob
                    # using Logistic Regression on logits.
                    p_clip = np.clip(probs_raw, 1e-6, 1 - 1e-6)
                    logits_raw = logit(p_clip).reshape(-1, 1)

                    try:
                        # High C (100) to allow fitting the curve without over-regularizing the scaler itself
                        platt = LogisticRegression(C=100.0, solver='lbfgs', random_state=42)
                        platt.fit(logits_raw, y_val_fit)
                        probs = platt.predict_proba(logits_raw)[:, 1]
                    except Exception:
                        # Fallback if single-class in validation or convergence fail
                        probs = probs_raw

                    oof_model[val_idx] = probs
                    
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

                avg_score = np.nanmean(fold_scores)
                avg_auc = np.nanmean(fold_aucs)
                avg_ic = np.nanmean(fold_ics)
                avg_bss_val = np.nanmean(fold_bss)
                avg_bs = np.nanmean(fold_bs)
                avg_ref = np.nanmean(fold_ref)
                avg_brier = np.nanmean(fold_brier)
                avg_p10 = np.nanmean(fold_p10)
                std_p10 = np.nanstd(fold_p10)
                cv_p10 = std_p10 / avg_p10 if avg_p10 > 1e-9 else 1.0

                avg_p20 = np.nanmean(fold_p20)
                std_p20 = np.nanstd(fold_p20)
                cv_p20 = std_p20 / avg_p20 if avg_p20 > 1e-9 else 1.0

                avg_p25 = np.nanmean(fold_p25)
                avg_p40 = np.nanmean(fold_p40)
                std_score = np.nanstd(fold_scores)
                avg_logloss = np.nanmean(fold_logloss)
                avg_accuracy = np.nanmean(fold_accuracy)

                train_loss = alpha_objective_logloss(y, np.clip(np.nan_to_num(oof_model, nan=np.nanmean(oof_model)), 1e-6, 1-1e-6), w=sample_weight)
                valid = np.isfinite(oof_model)
                comps = alpha_rank_components(y_hard[valid], oof_model[valid], returns[valid], sample_weight[valid] if sample_weight is not None else None, groups_arr[valid] if groups_arr is not None else None, rank_cfg)
                results[name] = 0.0
                top10_mask = topk_mask(oof_model[valid], 0.10, groups=groups_arr[valid] if groups_arr is not None else None)
                ece10 = ece_at_mask(y_hard[valid], oof_model[valid], top10_mask, n_bins=10, w=sample_weight[valid] if sample_weight is not None else None)
                curve = calibration_curve_bins(y_hard[valid], oof_model[valid], n_bins=10)
                profile = calibration_profile(curve)
                detailed_metrics[name] = {
                    "score": avg_score,
                    "rank_score": 0.0,
                    "alpha_train_loss": train_loss,
                    "rank_components": comps,
                    "ece_top10": ece10,
                    "calibration_curve": curve,
                    "calibration_profile": profile,
                    "AUC": avg_auc,
                    "IC": avg_ic,
                    "BSS": avg_bss_val,
                    "BS": avg_bs,
                    "BS_Ref": avg_ref,
                    "Brier": avg_brier,
                    "Prec10": avg_p10,
                    "CV_Prec10": cv_p10,
                    "Prec20": avg_p20,
                    "CV_Prec20": cv_p20,
                    "Prec25": avg_p25,
                    "Prec40": avg_p40,
                    "std_score": std_score,
                    "LogLoss": avg_logloss,
                    "Accuracy": avg_accuracy,
                    "fold_logloss": [float(x) for x in fold_logloss],
                    "fold_precision20": [float(x) for x in fold_p20],
                    "fold_precision10": [float(x) for x in fold_p10],
                    "fold_brier": [float(x) for x in fold_brier],
                    "fold_base_logloss": [float(x) for x in fold_base_logloss],
                    "fold_logloss_imp": [float(x) for x in fold_logloss_imp],
                    # Store per-model OOF predictions for gate checks
                    "oof_probs": np.nan_to_num(oof_model.copy(), nan=0.5).astype(np.float32),
                }
                tprint(f"  {name}: LegacyScore={avg_score:.4f} AUC={avg_auc:.4f} IC={avg_ic:.4f} BSS={avg_bss_val:.4f} Brier={avg_brier:.4f} Prec10={avg_p10:.4f} Prec40={avg_p40:.4f} LogLoss={avg_logloss:.4f} TrainLoss={train_loss:.4f}")

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
        sorted_candidates = sorted(
            results.items(),
            key=lambda x: (gate_results[x[0]]["n_passed"], x[1]),
            reverse=True
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
                oof_ic = np.corrcoef(rankdata(oof_probs), rankdata(returns))[0, 1]
            else:
                oof_ic = 0.0
            # OOF selection score (same weights as race)
            oof_sel = calculate_selection_score(y_hard, oof_probs, returns, sample_weight=sample_weight, w_bss=0.20, w_realized=0.55, w_uic=0.25)
            tprint(f"Winner OOF Metrics: AUC={oof_auc:.4f}  IC={oof_ic:.4f}  LogLoss={oof_logloss:.4f}  Acc={oof_accuracy:.4f}  SelScore={oof_sel['Selection_Score']:.4f}")
            
            # --- Post-hoc Isotonic Calibration ---
            # Fit calibrator on OOF predictions
            tprint("Fitting IsotonicRegression on OOF predictions (unweighted)...")
            self.calibrator_ = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            # IMPORTANT: Fit calibrator WITHOUT sample weights.
            # Weights upweight minority class, making calibrator target weighted
            # prevalence (~0.5) instead of actual prevalence (~0.31).
            self.calibrator_.fit(oof_probs, y_hard)
            if isinstance(self.calibration_state_, dict):
                self.calibration_state_["calibration_input"] = "bias_corrected"
                
            # Re-calibrate the stored OOF probs
            calibrated_oof = self.calibrator_.predict(oof_probs).astype(np.float32)
            
            raw_brier = brier_score_loss(y_hard, np.clip(oof_probs, 1e-7, 1-1e-7))
            cal_brier = brier_score_loss(y_hard, np.clip(calibrated_oof, 1e-7, 1-1e-7))
            tprint(f"Isotonic calibration: Brier raw={raw_brier:.4f} -> calibrated={cal_brier:.4f}")
            
            # Optional Platt scaling: only apply if it improves Brier score
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import brier_score_loss
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
        tprint(f"{'Model':<15} {'RankSc':>8} {'RcAUC':>8} {'RcIC':>8} {'RcBSS':>8} {'RcBrier':>8} {'RaceP10':>8} {'RaceP40':>8} {'LL':>8} {'ECE10':>8}")
        tprint("-" * 112)

        sorted_models = sorted(detailed_metrics.items(), key=lambda x: x[1]['rank_score'], reverse=True)
        for name, m in sorted_models:
            ece10 = m.get('ece_top10', np.nan)
            tprint(f"{name:<15} {m['rank_score']:8.4f} {m['AUC']:8.4f} {m['IC']:8.4f} {m['BSS']:8.4f} {m.get('Brier',0):8.4f} {m['Prec10']:8.4f} {m['Prec40']:8.4f} {m['LogLoss']:8.4f} {ece10:8.4f}")
        tprint("========================\n")

        # 3. Final Retraining (raw model, no calibration wrapper)
        # Calibration is harmful with small n and uncalibrated objectives.
        # Output is treated as rank score by downstream (engine, backtest).
        if detailed_metrics:
             best_name = max(detailed_metrics.items(), key=lambda x: x[1]['rank_score'])[0]
             self.best_model_name = best_name
             self.best_model = candidates[best_name]

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
        
        if sample_weight is not None:
             self.best_model.fit(X, y_hard, sample_weight=sample_weight)
        else:
             self.best_model.fit(X, y_hard)

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

        if hasattr(self, 'calibrator_'):
            if self.calibration_state_.get("calibration_input") != "bias_corrected":
                raise RuntimeError("Calibrator expects bias-corrected inputs but calibration_state_ is inconsistent.")
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
            from lightgbm import LGBMClassifier as _LGB
            inner = _LGB()
            inner._Booster = __import__("lightgbm").Booster(model_file=mp)
            inner._fitted = True
            inner._n_classes = 2
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
