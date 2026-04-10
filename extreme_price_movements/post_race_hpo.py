"""
Optuna HPO for model-race winners (ExtraTrees, XGBoost, LightGBM, CatBoost).

Crypto-noise-tuned search spaces: conservative depth/leaves, strong regularisation,
high min-child constraints.  Objective = mean_auc - 0.5 * std_auc across folds
(stability penalty).  Multi-seed evaluation (2 seeds averaged) to reduce noise
sensitivity.  Early stopping for boosters, Optuna MedianPruner with warm-up.

Designed to be called *after* ModelRace picks a winner:
    winner_name = race.best_model_name   # e.g. "lightgbm"
    result = run_hpo(X, y, ..., model_name=winner_name)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

from extreme_price_movements.utils import tprint

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import lightgbm as _lgb_module
except Exception:
    _lgb_module = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


# ---------------------------
# Purged K-Fold (count-based)
# ---------------------------
class PurgedKFold:
    def __init__(
        self,
        n_splits: int = 5,
        purge: int = 5,
        embargo: int = 0,
        min_train_size: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >=2")
        self.n_splits = int(n_splits)
        self.purge = int(purge)
        self.embargo = int(embargo)
        self.min_train_size = None if min_train_size is None else int(min_train_size)

    def split(self, X) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = X.shape[0]
        idx = np.arange(n, dtype=np.int32)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=np.int32)
        fold_sizes[: n % self.n_splits] += 1
        bounds = np.r_[0, fold_sizes.cumsum()]

        out = []
        for k in range(self.n_splits):
            test_start = int(bounds[k])
            test_end = int(bounds[k + 1])

            pre_end = max(0, test_start - self.purge)
            post_start = min(n, test_end + self.embargo)

            train = np.r_[0:pre_end, post_start:n].astype(np.int32, copy=False)
            test = idx[test_start:test_end]

            if self.min_train_size is not None and train.size < self.min_train_size:
                continue
            if train.size == 0 or test.size == 0:
                continue

            out.append((train, test))
        return out


# ---------------------------
# Metrics
# ---------------------------
def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    if np.unique(y).size < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, y_score))
    except Exception:
        return 0.5


# ---------------------------
# Utilities
# ---------------------------
def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    return X


def _as_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    return y


def get_class_weight_balanced(y01: np.ndarray) -> Dict[int, float]:
    y = np.asarray(y01, dtype=np.int32)
    n = y.size
    c0 = max(1, int((y == 0).sum()))
    c1 = max(1, int((y == 1).sum()))
    w0 = n / (2.0 * c0)
    w1 = n / (2.0 * c1)
    return {0: float(w0), 1: float(w1)}


# ---------------------------
# Model builders
# ---------------------------
def build_extratrees(params: Dict[str, Any]) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**params)


def build_xgboost(params: Dict[str, Any]) -> Any:
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed")
    return XGBClassifier(**params)


def build_lightgbm(params: Dict[str, Any]) -> Any:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm not installed")
    return LGBMClassifier(**params)


def build_catboost(params: Dict[str, Any]) -> Any:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost not installed")
    return CatBoostClassifier(**params)


# ---------------------------
# Model-name → suggest / build dispatch
# ---------------------------
_SUGGEST_FN = {}  # populated after function definitions below
_BUILD_FN = {
    "extratrees": build_extratrees,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
}


# ---------------------------
# Optuna search spaces (crypto-noise tuned)
# ---------------------------
def suggest_extratrees(
    trial: optuna.Trial, *, base_random_state: int = 42, n_samples: int = 10000
) -> Dict[str, Any]:
    """ExtraTrees — constrain hard to avoid overfitting noise."""
    n_estimators = trial.suggest_int("n_estimators", 300, 2000, step=200)
    max_depth = trial.suggest_int("max_depth", 3, 10)

    # Dynamic Regularization based on POSITIVE samples
    n_pos = int(n_samples)  # n_samples is passed as n_pos from make_objective
    min_leaf_dyn = max(75, int(n_pos * 0.015))  # Increased to 1.5%

    min_samples_leaf = trial.suggest_int(
        "min_samples_leaf", min_leaf_dyn, max(400, int(n_pos * 0.05)), log=True
    )
    min_samples_split = trial.suggest_int(
        "min_samples_split", min_leaf_dyn * 2, max(800, int(n_pos * 0.10)), log=True
    )

    max_feat_mode = trial.suggest_categorical(
        "max_features_mode", ["sqrt", "log2", "frac"]
    )
    if max_feat_mode == "frac":
        max_features = trial.suggest_float("max_features_frac", 0.2, 0.8)
    else:
        max_features = max_feat_mode

    bootstrap = trial.suggest_categorical("bootstrap", [False, True])
    use_oob = bootstrap and trial.suggest_categorical("use_oob", [False, False, True])

    criterion = trial.suggest_categorical("criterion", ["gini", "log_loss"])
    class_weight_mode = trial.suggest_categorical(
        "class_weight_mode", ["none", "balanced"]
    )
    class_weight = None if class_weight_mode == "none" else "balanced"

    max_samples = None
    if bootstrap:
        max_samples = trial.suggest_float("max_samples", 0.5, 0.95)

    min_impurity_decrease = trial.suggest_float(
        "min_impurity_decrease", 1e-6, 1e-2, log=True
    )
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 1e-2)

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "oob_score": bool(use_oob),
        "max_samples": max_samples,
        "criterion": criterion,
        "class_weight": class_weight,
        "min_impurity_decrease": min_impurity_decrease,
        "ccp_alpha": ccp_alpha,
        "n_jobs": 2,
        "random_state": base_random_state,
    }


def suggest_xgboost(
    trial: optuna.Trial, *, base_random_state: int = 42, n_samples: int = 10000
) -> Dict[str, Any]:
    """XGBoost — conservative for noisy labels: high min_child_weight, gamma, reg_lambda."""
    n_estimators = trial.suggest_int("n_estimators", 600, 6000, step=200)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 5)

    # Dynamic Regularization for XGBoost (hessian based) based on POSITIVE samples
    n_pos = int(n_samples)  # n_samples is passed as n_pos from make_objective
    min_cw_dyn = max(75, int(n_pos * 0.015 * 0.25))

    min_child_weight = trial.suggest_float(
        "min_child_weight",
        float(min_cw_dyn),
        float(max(500, int(n_pos * 0.05 * 0.25))),
        log=True,
    )
    gamma = trial.suggest_float("gamma", 0.5, 20.0, log=True)

    subsample = trial.suggest_float("subsample", 0.6, 0.9)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.9)
    colsample_bynode = trial.suggest_float("colsample_bynode", 0.5, 0.9)

    reg_lambda = trial.suggest_float("reg_lambda", 15.0, 500.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0)

    max_delta_step = trial.suggest_float("max_delta_step", 0.0, 5.0)
    tree_method = "hist"
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    max_leaves = None
    if grow_policy == "lossguide":
        max_leaves = trial.suggest_int("max_leaves", 16, 256, log=True)

    use_spw = trial.suggest_categorical("use_scale_pos_weight", [False, True])

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "colsample_bynode": colsample_bynode,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "max_delta_step": max_delta_step,
        "tree_method": tree_method,
        "grow_policy": grow_policy,
        "n_jobs": 2,
        "random_state": base_random_state,
        "eval_metric": "auc",
        "verbosity": 0,
        "enable_categorical": False,
    }
    if max_leaves is not None:
        params["max_leaves"] = max_leaves

    params["_use_scale_pos_weight"] = bool(use_spw)
    return params


def suggest_lightgbm(
    trial: optuna.Trial, *, base_random_state: int = 42, n_samples: int = 10000
) -> Dict[str, Any]:
    """LightGBM — cap leaves aggressively, raise min_child_samples, strong L1/L2."""
    n_estimators = trial.suggest_int("n_estimators", 800, 8000, step=200)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 6)

    num_leaves = trial.suggest_int("num_leaves", 8, 96, log=True)
    if max_depth > 0:
        num_leaves = min(num_leaves, 2**max_depth)

    subsample = trial.suggest_float("subsample", 0.6, 0.9)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.9)

    # Dynamic Regularization based on POSITIVE samples
    n_pos = int(n_samples)  # n_samples is passed as n_pos from make_objective
    min_leaf_dyn = max(75, int(n_pos * 0.015))

    min_child_samples = trial.suggest_int(
        "min_child_samples", min_leaf_dyn, max(600, int(n_pos * 0.05)), log=True
    )
    min_child_weight = trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True)

    lambda_l2 = trial.suggest_float("lambda_l2", 15.0, 500.0, log=True)
    lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 50.0)

    min_split_gain = trial.suggest_float("min_split_gain", 0.0, 5.0)

    feature_fraction = trial.suggest_float("feature_fraction", 0.5, 0.9)
    bagging_fraction = subsample
    bagging_freq = trial.suggest_int("bagging_freq", 0, 10)

    imbalance_mode = trial.suggest_categorical(
        "imbalance_mode", ["none", "scale_pos_weight"]
    )
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 20.0, log=True)

    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "min_child_samples": min_child_samples,
        "min_child_weight": min_child_weight,
        "min_split_gain": min_split_gain,
        "lambda_l2": lambda_l2,
        "lambda_l1": lambda_l1,
        "objective": "binary",
        "metric": "auc",
        "random_state": base_random_state,
        "n_jobs": 2,
        "verbose": -1,
    }
    params["_imbalance_mode"] = imbalance_mode
    params["_scale_pos_weight"] = float(scale_pos_weight)
    return params


def suggest_catboost(
    trial: optuna.Trial, *, base_random_state: int = 42, n_samples: int = 10000
) -> Dict[str, Any]:
    """CatBoost — keep depth small, strong regularisation."""
    iterations = trial.suggest_int("iterations", 800, 8000, step=200)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    depth = trial.suggest_int("depth", 2, 5)

    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 15.0, 500.0, log=True)
    random_strength = trial.suggest_float("random_strength", 0.0, 3.0)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.5)

    rsm = trial.suggest_float("rsm", 0.5, 0.9)
    border_count = trial.suggest_int("border_count", 32, 255)
    od_type = trial.suggest_categorical("od_type", ["IncToDec", "Iter"])
    use_class_weights = trial.suggest_categorical("use_class_weights", [False, True])

    return {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "random_strength": random_strength,
        "bagging_temperature": bagging_temperature,
        "rsm": rsm,
        "border_count": border_count,
        "od_type": od_type,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": base_random_state,
        "thread_count": -1,
        "verbose": 0,
        "allow_writing_files": False,
        "_use_class_weights": bool(use_class_weights),
    }


_SUGGEST_FN = {
    "extratrees": suggest_extratrees,
    "xgboost": suggest_xgboost,
    "lightgbm": suggest_lightgbm,
    "catboost": suggest_catboost,
}


def suggest_extratrees_base(
    trial: optuna.Trial,
    *,
    base_random_state: int = 42,
    n_pos: int = 1000,
) -> Dict[str, Any]:
    """Narrow ExtraTrees search space for base-model HPO.

    Search space (user-specified):
        max_depth          ∈ {5, 6, 7, 8}
        min_samples_leaf   ∈ {0.5%, 1%, 2%} of positive samples
        min_samples_split  = 2 × min_samples_leaf  (deterministic)
        max_features       ∈ {"sqrt", 0.25, 0.33, 0.5}
        ccp_alpha          ∈ {1e-5, 1e-6}
        min_impurity_decrease ∈ {1e-5, 1e-4}
    """
    max_depth = trial.suggest_categorical("max_depth", [5, 6, 7, 8])

    leaf_frac = trial.suggest_categorical("min_samples_leaf_frac", [0.005, 0.01, 0.02])
    min_samples_leaf = max(20, int(np.ceil(n_pos * leaf_frac)))
    min_samples_split = max(2, 2 * min_samples_leaf)

    max_features = trial.suggest_categorical("max_features", ["sqrt", 0.25, 0.33, 0.5])
    ccp_alpha = trial.suggest_categorical("ccp_alpha", [1e-5, 1e-6])
    min_impurity_decrease = trial.suggest_categorical(
        "min_impurity_decrease", [1e-5, 1e-4]
    )

    return {
        "n_estimators": 400,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "bootstrap": False,
        "oob_score": False,
        "max_samples": None,
        "criterion": "gini",
        "class_weight": None,
        "min_impurity_decrease": min_impurity_decrease,
        "ccp_alpha": ccp_alpha,
        "max_leaf_nodes": 512,
        "n_jobs": 2,
        "random_state": base_random_state,
    }


# ---------------------------
# CV objective with early stopping + pruning
# ---------------------------
@dataclass
class HPOConfig:
    model_name: str  # extratrees | xgboost | lightgbm | catboost
    n_trials: int = 150
    timeout_sec: Optional[int] = None
    n_splits: int = 3
    purge: int = 12
    embargo: int = 2
    random_state: int = 42

    # Early stopping (boosters only)
    early_stopping_rounds: int = 50

    # Pruning
    pruner_warmup_steps: int = 4  # default to n_splits, will be overridden

    # Multi-seed: average over this many seeds per trial to reduce noise sensitivity
    n_seeds: int = 1


def _prepare_lgbm_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract private keys from LightGBM params and apply imbalance settings."""
    p = dict(params)
    imbalance_mode = str(p.pop("_imbalance_mode", "none"))
    spw = float(p.pop("_scale_pos_weight", 1.0))
    if imbalance_mode == "scale_pos_weight":
        p["scale_pos_weight"] = spw
    return p


def _lgbm_callbacks(early_stopping_rounds: int) -> list:
    """Build LightGBM callbacks list with early stopping if available."""
    cbs = []
    if _lgb_module is not None:
        try:
            cbs.append(_lgb_module.early_stopping(early_stopping_rounds, verbose=False))
        except Exception:
            pass
        try:
            cbs.append(_lgb_module.log_evaluation(period=-1))
        except Exception:
            pass
    return cbs


def _fit_predict_fold(
    model_name: str,
    params: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    sample_weight_tr: Optional[np.ndarray],
    config: HPOConfig,
) -> np.ndarray:
    """
    Fit one fold, return validation probabilities (class 1).
    Uses early stopping for boosting models.
    """
    if model_name == "extratrees":
        clf = build_extratrees(params)
        clf.fit(X_tr, y_tr, sample_weight=sample_weight_tr)
        return clf.predict_proba(X_va)[:, 1].astype(np.float32)

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed")
        p = dict(params)
        use_spw = bool(p.pop("_use_scale_pos_weight", False))
        if use_spw:
            pos = max(1, int((y_tr == 1).sum()))
            neg = max(1, int((y_tr == 0).sum()))
            p["scale_pos_weight"] = float(neg / pos)

        clf = build_xgboost(p)
        clf.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=config.early_stopping_rounds,
        )
        return clf.predict_proba(X_va)[:, 1].astype(np.float32)

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm not installed")
        p = _prepare_lgbm_params(params)
        clf = build_lightgbm(p)
        clf.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=_lgbm_callbacks(config.early_stopping_rounds),
        )
        return clf.predict_proba(X_va)[:, 1].astype(np.float32)

    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost not installed")
        p = dict(params)
        use_cw = bool(p.pop("_use_class_weights", False))
        if use_cw:
            cw = get_class_weight_balanced(y_tr)
            p["class_weights"] = [cw[0], cw[1]]

        clf = build_catboost(p)
        clf.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=False,
        )
        return clf.predict_proba(X_va)[:, 1].astype(np.float32)

    raise ValueError(f"Unknown model_name={model_name}")


def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    config: HPOConfig,
):
    """
    Build Optuna objective.
    Objective = mean_auc - 0.5 * std_auc  (stability penalty).
    Multi-seed: each trial is evaluated with `config.n_seeds` seeds and averaged.
    """
    X = _as_2d(X)
    y = _as_1d(y).astype(np.int32)
    sw = None if sample_weight is None else _as_1d(sample_weight).astype(np.float32)

    cv = PurgedKFold(
        n_splits=config.n_splits, purge=config.purge, embargo=config.embargo
    )
    splits = cv.split(X)
    if len(splits) < 2:
        raise ValueError(
            "Not enough splits produced. Reduce purge/embargo or increase n_splits."
        )

    suggest_fn = _SUGGEST_FN[config.model_name]

    def objective(trial: optuna.Trial) -> float:
        # Suggest params once (seed-independent hypers)
        # Use positive sample count for dynamic regularization bounds
        n_pos = int(np.sum(y > 0)) if len(y) > 0 else 0
        params_base = suggest_fn(
            trial, base_random_state=config.random_state, n_samples=n_pos
        )

        all_seed_ics: List[List[float]] = []  # [seed][fold]

        running_ics = []  # For pruning reports

        for seed_offset in range(config.n_seeds):
            seed = config.random_state + seed_offset
            # Clone params and override seed
            params = dict(params_base)
            if config.model_name == "extratrees":
                params["random_state"] = seed
            elif config.model_name == "xgboost":
                params["random_state"] = seed
            elif config.model_name == "lightgbm":
                params["random_state"] = seed
            elif config.model_name == "catboost":
                params["random_seed"] = seed

            fold_ics: List[float] = []
            for fold_i, (tr, va) in enumerate(splits):
                X_tr, y_tr = X[tr], y[tr]
                X_va, y_va = X[va], y[va]
                sw_tr = None if sw is None else sw[tr]

                y_score = _fit_predict_fold(
                    config.model_name,
                    params,
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    sample_weight_tr=sw_tr,
                    config=config,
                )

                # Calculate Top-30% IC instead of Global AUC
                if len(y_score) > 10:
                    ranks = rankdata(y_score) / len(y_score)
                    t30 = ranks >= 0.70
                    if t30.sum() > 2:
                        ic_t30 = float(spearmanr(y_score[t30], y_va[t30]).statistic)
                    else:
                        ic_t30 = 0.0
                else:
                    ic_t30 = 0.0

                if not np.isfinite(ic_t30):
                    ic_t30 = 0.0

                fold_ics.append(ic_t30)
                running_ics.append(ic_t30)

                step = len(running_ics) - 1
                trial.report(float(np.mean(running_ics)), step=step)

            all_seed_ics.append(fold_ics)

        # Flatten all fold ICs across seeds
        flat = [a for sa in all_seed_ics for a in sa]
        mean_ic = float(np.mean(flat))
        std_ic = float(np.std(flat))
        # Stability-penalised objective for Top-30% IC
        return mean_ic - 1.0 * std_ic

    return objective


# ---------------------------
# Reconstruct model params from Optuna best_params
# ---------------------------
def _reconstruct_params(
    model_name: str, raw_params: Dict[str, Any], random_state: int
) -> Dict[str, Any]:
    """Convert Optuna best_params (flat dict with helper keys) into constructor-ready params."""
    p = dict(raw_params)

    if model_name == "extratrees":
        mf_mode = p.pop("max_features_mode", "sqrt")
        if mf_mode == "frac":
            p["max_features"] = p.pop("max_features_frac")
        else:
            p["max_features"] = mf_mode
            p.pop("max_features_frac", None)

        cw_mode = p.pop("class_weight_mode", "none")
        p["class_weight"] = None if cw_mode == "none" else "balanced"

        use_oob = p.pop("use_oob", False)
        p["oob_score"] = bool(use_oob) and bool(p.get("bootstrap", False))

        # max_samples only valid when bootstrap=True
        if not p.get("bootstrap", False):
            p.pop("max_samples", None)

        p.update({"n_jobs": 2, "random_state": random_state})

    elif model_name == "xgboost":
        p.pop("use_scale_pos_weight", None)  # handled at fit time
        p.update(
            {
                "n_jobs": 2,
                "random_state": random_state,
                "enable_categorical": False,
                "eval_metric": "auc",
                "verbosity": 0,
                "tree_method": "hist",
            }
        )

    elif model_name == "lightgbm":
        p.pop("imbalance_mode", None)
        p.pop("scale_pos_weight", None)
        p.update(
            {
                "objective": "binary",
                "metric": "auc",
                "random_state": random_state,
                "n_jobs": 2,
                "verbose": -1,
            }
        )

    elif model_name == "catboost":
        p.pop("use_class_weights", None)
        p.update(
            {
                "random_seed": random_state,
                "thread_count": -1,
                "verbose": 0,
                "allow_writing_files": False,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
            }
        )

    return p


# ---------------------------
# OOF prediction generation
# ---------------------------
def _generate_oof(
    model_name: str,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    sw: Optional[np.ndarray],
    config: HPOConfig,
) -> np.ndarray:
    """Generate out-of-fold predictions using best params (for metrics reporting + downstream)."""
    cv = PurgedKFold(
        n_splits=config.n_splits, purge=config.purge, embargo=config.embargo
    )
    splits = cv.split(X)
    oof = np.full(len(y), np.nan, dtype=np.float64)

    for tr, va in splits:
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        sw_tr = None if sw is None else sw[tr]

        proba = _fit_predict_fold(
            model_name,
            params,
            X_tr,
            y_tr,
            X_va,
            y_va,
            sample_weight_tr=sw_tr,
            config=config,
        )
        oof[va] = proba

    oof = np.nan_to_num(oof, nan=0.5)
    return oof


# ---------------------------
# Run HPO
# ---------------------------
def run_hpo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    model_name: str,
    n_trials: int = 150,
    timeout_sec: Optional[int] = None,
    n_splits: int = 4,
    purge: int = 12,
    embargo: int = 2,
    random_state: int = 42,
    early_stopping_rounds: int = 200,
    n_seeds: int = 1,
    study_name: str = "hpo_study",
    storage: Optional[str] = None,
    out_dir: str = "./hpo_out",
) -> Dict[str, Any]:
    """
    Run Optuna HPO for the given model.

    Returns dict with:
      - best_params, best_value (stability-penalised AUC)
      - oof_probs: OOF predictions with best params
      - model: refitted model on full data
    """
    os.makedirs(out_dir, exist_ok=True)

    if model_name not in _SUGGEST_FN:
        raise ValueError(
            f"Unknown model_name={model_name}. Choose from {list(_SUGGEST_FN)}"
        )

    config = HPOConfig(
        model_name=model_name,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        n_splits=n_splits,
        purge=purge,
        embargo=embargo,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        n_seeds=n_seeds,
    )
    config.pruner_warmup_steps = n_splits

    sampler = TPESampler(seed=random_state, multivariate=True, group=True)
    pruner = MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=config.pruner_warmup_steps,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(X, y, sample_weight, config)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # --- Results ---
    best_raw_params = dict(study.best_params)
    best_value = float(study.best_value)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "best_value": best_value,
        "best_raw_params": best_raw_params,
        "n_trials": len(study.trials),
    }

    # Save raw results JSON
    with open(os.path.join(out_dir, f"{study_name}_{model_name}_best.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    # --- Reconstruct best params for model construction ---
    X = _as_2d(X)
    y = _as_1d(y).astype(np.int32)
    sw = None if sample_weight is None else _as_1d(sample_weight).astype(np.float32)

    # For the suggest functions, we need to re-call them with the best trial to get
    # the full params dict (including private keys like _use_scale_pos_weight).
    # Easiest: re-suggest from best trial.
    best_trial = study.best_trial
    suggest_fn = _SUGGEST_FN[model_name]
    full_params = suggest_fn(best_trial, base_random_state=random_state)

    # --- Generate OOF predictions with best params ---
    oof_probs = _generate_oof(model_name, full_params, X, y, sw, config)
    result["oof_probs"] = oof_probs

    # OOF metrics
    oof_auc = auc_safe(y, oof_probs)
    result["oof_auc"] = oof_auc

    # Save OOF predictions
    np.save(os.path.join(out_dir, f"{study_name}_{model_name}_oof.npy"), oof_probs)

    # --- Refit on full data ---
    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
    splits = cv.split(X)
    tr, va = splits[-1]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    sw_tr = None if sw is None else sw[tr]

    # Use full_params for refit, override random_state
    refit_params = dict(full_params)
    if model_name == "extratrees":
        refit_params["random_state"] = random_state
        model = build_extratrees(refit_params)
        model.fit(X, y, sample_weight=sw)

    elif model_name == "xgboost":
        p = dict(refit_params)
        use_spw = bool(p.pop("_use_scale_pos_weight", False))
        if use_spw:
            pos = max(1, int((y == 1).sum()))
            neg = max(1, int((y == 0).sum()))
            p["scale_pos_weight"] = float(neg / pos)
        p["random_state"] = random_state
        model = build_xgboost(p)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

    elif model_name == "lightgbm":
        p = dict(refit_params)
        imb = str(p.pop("_imbalance_mode", "none"))
        spw = float(p.pop("_scale_pos_weight", 1.0))
        if imb == "scale_pos_weight":
            p["scale_pos_weight"] = spw
        p["random_state"] = random_state
        model = build_lightgbm(p)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=_lgbm_callbacks(early_stopping_rounds),
        )

    elif model_name == "catboost":
        p = dict(refit_params)
        use_cw = bool(p.pop("_use_class_weights", False))
        if use_cw:
            cw = get_class_weight_balanced(y)
            p["class_weights"] = [cw[0], cw[1]]
        p["random_seed"] = random_state
        model = build_catboost(p)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=(X_va, y_va),
            use_best_model=True,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    result["model"] = model

    # Save fitted model
    try:
        import joblib

        joblib.dump(
            model, os.path.join(out_dir, f"{study_name}_{model_name}_best_model.joblib")
        )
    except Exception:
        pass

    result["out_dir"] = out_dir
    return result


# ---------------------------
# Base ExtraTrees HPO (narrow search, scope-aware JSON warm-start)
# ---------------------------
_BASE_HPO_JSON = "best_base_extratrees_params.json"


def _scope_suffix(scope_key: Optional[str]) -> str:
    if not scope_key:
        return ""
    safe = str(scope_key).strip().replace(os.sep, "_").replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe)
    return f"__{safe}" if safe else ""


def _base_hpo_json_path(out_dir: str, scope_key: Optional[str] = None) -> str:
    suffix = _scope_suffix(scope_key)
    stem, ext = os.path.splitext(_BASE_HPO_JSON)
    return os.path.join(out_dir, f"{stem}{suffix}{ext}")


def _load_base_hpo_json(
    out_dir: str, scope_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    path = _base_hpo_json_path(out_dir, scope_key=scope_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_base_hpo_json(
    out_dir: str, payload: Dict[str, Any], scope_key: Optional[str] = None
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = _base_hpo_json_path(out_dir, scope_key=scope_key)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_best_base_extratrees_params(
    out_dir: str = "./hpo_out",
    scope_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Public API for training.py to load previously saved best params."""
    data = _load_base_hpo_json(out_dir, scope_key=scope_key)
    if data is None:
        return None
    return data.get("best_params")


def _subsample_diverse(
    X: np.ndarray,
    y: np.ndarray,
    symbols: Optional[np.ndarray],
    max_rows: int = 5000,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample up to *max_rows* rows, balanced across symbols and time."""
    n = X.shape[0]
    if n <= max_rows:
        return X, y

    rng = np.random.RandomState(rng_seed)

    if symbols is not None and len(symbols) == n:
        unique_syms = np.unique(symbols)
        per_sym = max(1, max_rows // len(unique_syms))
        idx_parts: list[np.ndarray] = []
        for sym in unique_syms:
            sym_idx = np.flatnonzero(symbols == sym)
            if len(sym_idx) <= per_sym:
                idx_parts.append(sym_idx)
            else:
                idx_parts.append(rng.choice(sym_idx, size=per_sym, replace=False))
        idx = np.concatenate(idx_parts)
        if len(idx) > max_rows:
            idx = rng.choice(idx, size=max_rows, replace=False)
        idx.sort()
    else:
        idx = rng.choice(n, size=max_rows, replace=False)
        idx.sort()

    return X[idx], y[idx]


def run_base_extratrees_hpo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    symbols: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_trials: int = 40,
    n_splits: int = 3,
    purge: int = 12,
    embargo: int = 2,
    max_rows: int = 5000,
    random_state: int = 42,
    out_dir: str = "./hpo_out",
    study_name: str = "base_extratrees_hpo",
    scope_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run HPO for the base ExtraTrees classifier with a narrow search space.

    - 40 Optuna trials, MedianPruner (prune after 30 % of CV rounds)
    - Max 5 000 samples, sourced from diverse periods / assets
    - Saves best params for later training reuse
    """
    X = _as_2d(X)
    y_raw = _as_1d(y)
    y_hard = (y_raw >= 0.5).astype(np.int32)
    sw = None if sample_weight is None else _as_1d(sample_weight).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)

    X, y_hard = _subsample_diverse(
        X, y_hard, symbols, max_rows=max_rows, rng_seed=random_state
    )
    if sw is not None and sw.shape[0] != y_hard.shape[0]:
        sw = None
    tprint(
        f"Base HPO: {X.shape[0]} rows, {X.shape[1]} features after diverse subsample"
    )

    n_pos = int(np.sum(y_hard > 0))
    tprint(f"Base HPO: n_positive={n_pos}, n_negative={int(np.sum(y_hard == 0))}")

    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
    splits = cv.split(X)
    if len(splits) < 2:
        raise ValueError(
            "Not enough CV splits — reduce purge/embargo or increase n_splits."
        )

    total_rounds = n_splits
    warmup_steps = max(1, int(np.ceil(0.30 * total_rounds)))

    sampler = TPESampler(seed=random_state, multivariate=True, group=True)
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=warmup_steps,
        interval_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_extratrees_base(
            trial, base_random_state=random_state, n_pos=n_pos
        )
        params["random_state"] = random_state

        fold_ics: List[float] = []
        for fold_i, (tr, va) in enumerate(splits):
            X_tr, y_tr = X[tr], y_hard[tr]
            X_va, y_va = X[va], y_hard[va]
            sw_tr = None if sw is None else sw[tr]

            clf = build_extratrees(params)
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            proba = clf.predict_proba(X_va)[:, 1].astype(np.float32)

            if len(proba) > 10:
                ranks = rankdata(proba) / len(proba)
                t30 = ranks >= 0.70
                if t30.sum() > 2:
                    ic = float(spearmanr(proba[t30], y_va[t30]).statistic)
                else:
                    ic = 0.0
            else:
                ic = 0.0
            if not np.isfinite(ic):
                ic = 0.0

            fold_ics.append(ic)
            trial.report(ic, step=fold_i)

        mean_ic = float(np.mean(fold_ics))
        std_ic = float(np.std(fold_ics))
        return mean_ic - 1.0 * std_ic

    study.optimize(
        objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=True
    )

    best_value = float(study.best_value)
    best_trial = study.best_trial
    best_params = suggest_extratrees_base(
        best_trial, base_random_state=random_state, n_pos=n_pos
    )
    best_params.pop("n_jobs", None)
    best_params.pop("random_state", None)

    payload = {
        "best_params": best_params,
        "n_trials_completed": len(study.trials),
        "n_pos_at_optimisation": n_pos,
        "search_space": "base_narrow",
    }

    _save_base_hpo_json(out_dir, payload, scope_key=scope_key)
    save_name = os.path.basename(_base_hpo_json_path(out_dir, scope_key=scope_key))
    tprint(f"Base HPO: saved params from current run to {save_name}")

    payload["best_value"] = best_value
    payload["out_dir"] = out_dir
    return payload


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 50)).astype(np.float32)
    y = (rng.random(1000) > 0.6).astype(np.int32)

    # Suppose ModelRace picked "lightgbm"
    winner = "lightgbm"

    result = run_hpo(
        X,
        y,
        sample_weight=None,
        model_name=winner,
        n_trials=50,
        n_splits=4,
        purge=12,
        embargo=2,
        random_state=42,
        early_stopping_rounds=200,
        n_seeds=1,
        study_name="demo_hpo",
        out_dir="./hpo_out",
    )

    print(f"Best value (mean_auc - 0.5*std_auc): {result['best_value']:.4f}")
    print(f"OOF AUC: {result['oof_auc']:.4f}")
    print(f"Best params: {json.dumps(result['best_raw_params'], indent=2)}")
