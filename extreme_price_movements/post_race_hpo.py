"""
Optuna HPO for model-race winners (ExtraTrees, XGBoost, LightGBM, CatBoost).

Crypto-noise-tuned search spaces: conservative depth/leaves, strong regularisation,
high min-child constraints.

Objective (composite z-score):
    score = 0.5 * Zscore_PR_AUC_Lift
          + 0.3 * Zscore_Top30_Abs_Ret_Lift
          + 0.2 * Zscore_Median_IC_T30
          - 0.2 * Zscore_Std_IC_T30
          - 0.1 * Zscore_Brier

Multi-seed evaluation (2 seeds averaged) to reduce noise sensitivity.
Early stopping for boosters, Optuna MedianPruner with warm-up.

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
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from extreme_price_movements.utils import tprint


# ---------------------------
# Running statistics for z-score normalization across trials
# ---------------------------
class _RunningStats:
    """Online mean and variance tracking for z-score computation."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        """Update with new value using Welford's algorithm."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def get_mean(self) -> float:
        return self.mean if self.n > 0 else 0.0

    def get_std(self) -> float:
        return (self.m2 / self.n) ** 0.5 if self.n > 0 else 1.0

    def zscore(self, x: float) -> float:
        """Compute z-score for a value."""
        std = self.get_std()
        if std < 1e-9:
            return 0.0
        return (x - self.get_mean()) / std


# Global running stats for base HPO metrics (populated across trials)
_running_stats_base = {
    "pr_auc_lift": _RunningStats(),
    "top30_abs_ret_lift": _RunningStats(),
    "median_ic_t30": _RunningStats(),
    "std_ic_t30": _RunningStats(),
    "brier": _RunningStats(),
}

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
        "n_jobs": 3,
        "random_state": base_random_state,
    }


def suggest_xgboost(
    trial: optuna.Trial, *, base_random_state: int = 42, n_samples: int = 10000
) -> Dict[str, Any]:
    """XGBoost — conservative for noisy labels: high min_child_weight, gamma, reg_lambda."""
    n_estimators = trial.suggest_int("n_estimators", 400, 1500, step=100)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 6)

    # Dynamic Regularization for XGBoost (hessian based) based on POSITIVE samples
    n_pos = int(n_samples)  # n_samples is passed as n_pos from make_objective
    min_cw_dyn = max(75, int(n_pos * 0.015 * 0.25))

    min_child_weight = trial.suggest_float(
        "min_child_weight",
        float(min_cw_dyn),
        float(max(500, int(n_pos * 0.05 * 0.25))),
        log=True,
    )
    gamma = trial.suggest_float("gamma", 0.5, 5.0, log=True)

    subsample = trial.suggest_float("subsample", 0.6, 0.9)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 0.8)
    colsample_bynode = trial.suggest_float("colsample_bynode", 0.6, 0.8)

    reg_lambda = trial.suggest_float("reg_lambda", 1.0, 100.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)

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

    params["objective"] = "binary:logistic"
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
    max_depth = trial.suggest_categorical("max_depth", [6, 7, 8, 9, 10, 11, 12, 13])

    leaf_frac = trial.suggest_categorical("min_samples_leaf_frac", [0.005, 0.01, 0.02])
    min_samples_leaf = max(20, int(np.ceil(n_pos * leaf_frac)))
    split_mult = trial.suggest_categorical("min_samples_split_mult", [2, 3])
    min_samples_split = max(2, split_mult * min_samples_leaf)

    max_features = trial.suggest_categorical("max_features", ["sqrt", 0.25, 0.33, 0.5])
    ccp_alpha = trial.suggest_categorical("ccp_alpha", [1e-5, 1e-6])
    min_impurity_decrease = trial.suggest_categorical(
        "min_impurity_decrease", [1e-5, 1e-4]
    )

    n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 400, 500, 600])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [256, 512, 1024])

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "bootstrap": False,
        "oob_score": False,
        "max_samples": None,
        "criterion": criterion,
        "class_weight": None,
        "min_impurity_decrease": min_impurity_decrease,
        "ccp_alpha": ccp_alpha,
        "max_leaf_nodes": max_leaf_nodes,
        "n_jobs": 3,
        "random_state": base_random_state,
    }


def _fallback_extratrees_base_params(
    *, base_random_state: int = 42, n_pos: int = 1000
) -> Dict[str, Any]:
    """Conservative fallback ExtraTrees params when Optuna yields no completed trial."""
    min_samples_leaf = max(20, int(np.ceil(float(n_pos) * 0.01)))
    min_samples_split = max(2, 2 * min_samples_leaf)
    return {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": "sqrt",
        "bootstrap": False,
        "oob_score": False,
        "max_samples": None,
        "criterion": "gini",
        "class_weight": None,
        "min_impurity_decrease": 1e-4,
        "ccp_alpha": 1e-5,
        "max_leaf_nodes": 512,
        "n_jobs": 3,
        "random_state": base_random_state,
    }


def suggest_extratrees_base_reg(
    trial: optuna.Trial,
    *,
    base_random_state: int = 42,
    n_pos: int = 1000,
) -> Dict[str, Any]:
    max_depth = trial.suggest_categorical("max_depth", [6, 7, 8, 9, 10, 11, 12, 13])
    leaf_frac = trial.suggest_categorical("min_samples_leaf_frac", [0.005, 0.01, 0.02])
    min_samples_leaf = max(20, int(np.ceil(n_pos * leaf_frac)))
    split_mult = trial.suggest_categorical("min_samples_split_mult", [2, 3])
    min_samples_split = max(2, split_mult * min_samples_leaf)
    max_features = trial.suggest_categorical("max_features", ["sqrt", 0.25, 0.33, 0.5])
    ccp_alpha = trial.suggest_categorical("ccp_alpha", [1e-5, 1e-6])
    min_impurity_decrease = trial.suggest_categorical(
        "min_impurity_decrease", [1e-5, 1e-4]
    )
    n_estimators = trial.suggest_categorical("n_estimators", [200, 300, 400, 500, 600])
    max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [256, 512, 1024])
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "bootstrap": False,
        "min_impurity_decrease": min_impurity_decrease,
        "ccp_alpha": ccp_alpha,
        "max_leaf_nodes": max_leaf_nodes,
        "criterion": "absolute_error",
        "n_jobs": 3,
        "random_state": base_random_state,
    }


def _fallback_extratrees_base_reg_params(
    *, base_random_state: int = 42, n_pos: int = 1000
) -> Dict[str, Any]:
    min_samples_leaf = max(20, int(np.ceil(float(n_pos) * 0.01)))
    min_samples_split = max(2, 2 * min_samples_leaf)
    return {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_features": "sqrt",
        "bootstrap": False,
        "min_impurity_decrease": 1e-4,
        "ccp_alpha": 1e-5,
        "max_leaf_nodes": 512,
        "criterion": "absolute_error",
        "n_jobs": 3,
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
    optuna_patience_trials: int = 30
    optuna_min_trials_before_stop: int = 50
    optuna_meaningful_improvement_pct: float = 0.005

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


def _make_optuna_patience_callback(
    *,
    patience: int,
    label: str,
    min_delta: float = 0.0,
    min_trials_before_stop: int = 0,
    meaningful_improvement_pct: float = 0.005,
):
    best_value = float("-inf")
    best_trial_number = -1
    last_meaningful_improvement_trial = -1

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_value, best_trial_number, last_meaningful_improvement_trial
        if trial.value is None or not np.isfinite(trial.value):
            return
        current_value = float(trial.value)
        if current_value > (best_value + float(min_delta)):
            prev_best = best_value
            best_value = current_value
            best_trial_number = int(trial.number)
            if prev_best == float("-inf"):
                last_meaningful_improvement_trial = int(trial.number)
            else:
                denom = max(abs(prev_best), 1e-12)
                rel_gain = (current_value - prev_best) / denom
                if rel_gain > float(meaningful_improvement_pct):
                    last_meaningful_improvement_trial = int(trial.number)
            return
        if int(trial.number) < int(min_trials_before_stop):
            return
        if best_trial_number >= 0 and (int(trial.number) - best_trial_number) >= int(patience):
            tprint(
                f"{label}: early stopping after {patience} trials without improvement "
                f"(best={best_value:.6f}, last_improved_trial={best_trial_number})"
            )
            study.stop()
            return
        extended_patience = int(np.ceil(float(patience) * 1.5))
        if (
            last_meaningful_improvement_trial >= 0
            and (int(trial.number) - last_meaningful_improvement_trial)
            >= extended_patience
        ):
            tprint(
                f"{label}: early stopping after {extended_patience} trials without meaningful improvement "
                f"(>{100.0 * float(meaningful_improvement_pct):.2f}% gain, best={best_value:.6f}, "
                f"last_meaningful_trial={last_meaningful_improvement_trial})"
            )
            study.stop()

    return _callback


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



def _compute_lift_top30(y_score: np.ndarray, y_true: np.ndarray) -> float:
    """Compute lift in positive rate for top 30% predictions vs baseline positive rate."""
    y_score = np.asarray(y_score, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if len(y_score) < 10 or len(y_true) < 10:
        return 0.0

    ranks = rankdata(y_score) / len(y_score)
    t30_mask = ranks >= 0.70

    if t30_mask.sum() < 2:
        return 0.0

    top30_pos_rate = float(np.mean(y_true[t30_mask]))
    baseline = float(np.mean(y_true))

    if baseline < 1e-9:
        return 0.0

    return (top30_pos_rate - baseline) / baseline


def _compute_precision_top10(y_score: np.ndarray, y_true: np.ndarray) -> float:
    """Compute precision for the top 10% predictions."""
    y_score = np.asarray(y_score, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if len(y_score) < 10 or len(y_true) < 10:
        return 0.0

    ranks = rankdata(y_score) / len(y_score)
    t10_mask = ranks >= 0.90

    if t10_mask.sum() < 1:
        return 0.0

    return float(np.mean(y_true[t10_mask]))


def _compute_pr_auc_top30(y_score: np.ndarray, y_true: np.ndarray) -> float:
    """Compute PR AUC specifically for the top 30% of predictions."""
    from sklearn.metrics import precision_recall_curve, auc
    y_score = np.asarray(y_score, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    if len(y_score) < 10 or len(y_true) < 10:
        return 0.0

    ranks = rankdata(y_score) / len(y_score)
    t30_mask = ranks >= 0.70

    if t30_mask.sum() < 2:
        return 0.0

    y_score_top = y_score[t30_mask]
    y_true_top = y_true[t30_mask]

    baseline = float(np.mean(y_true_top > 0)) if len(y_true_top) > 0 else 0.5
    if len(np.unique(y_true_top)) < 2:
        return baseline

    precision, recall, _ = precision_recall_curve(y_true_top, y_score_top)
    pr_auc = float(auc(recall, precision)) if len(recall) > 1 else baseline
    return pr_auc


def _compute_pr_auc_lift(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR AUC lift over baseline (random) PR AUC."""
    from sklearn.metrics import precision_recall_curve, auc

    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    # Baseline: PR AUC of a random classifier = positive rate
    baseline = float(np.mean(y_true > 0)) if len(y_true) > 0 else 0.5

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = float(auc(recall, precision)) if len(recall) > 1 else baseline

    # Lift: how much better than baseline
    if baseline < 1e-9:
        return 0.0
    return (pr_auc - baseline) / max(baseline, 1e-9)


def _compute_top30_abs_ret_lift(y_score: np.ndarray, y_ret: np.ndarray) -> float:
    """Compute lift in absolute returns for top 30% predictions vs baseline."""
    y_score = np.asarray(y_score, dtype=np.float64)
    y_ret = np.asarray(y_ret, dtype=np.float64)

    if len(y_score) < 10 or len(y_ret) < 10:
        return 0.0

    ranks = rankdata(y_score) / len(y_score)
    t30_mask = ranks >= 0.70

    if t30_mask.sum() < 2:
        return 0.0

    abs_ret_t30 = float(np.mean(np.abs(y_ret[t30_mask])))
    abs_ret_all = float(np.mean(np.abs(y_ret)))

    if abs_ret_all < 1e-9:
        return 0.0

    return (abs_ret_t30 - abs_ret_all) / abs_ret_all


def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    config: HPOConfig,
):
    """
    Build Optuna objective.

    New composite objective with z-scored metrics:
    score = 0.5 * Zscore_PR_AUC_Lift
          + 0.3 * Zscore_Top30_Abs_Ret_Lift
          + 0.2 * Zscore_Median_IC_T30
          - 0.2 * Zscore_Std_IC_T30
          - 0.1 * Zscore_Brier
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
        n_pos = int(np.sum(y > 0)) if len(y) > 0 else 0
        params_base = suggest_fn(
            trial, base_random_state=config.random_state, n_samples=n_pos
        )

        # Metrics per fold: [pr_auc_lift, top30_abs_ret_lift, ic_t30, brier]
        fold_metrics: List[Dict[str, float]] = []

        for seed_offset in range(config.n_seeds):
            seed = config.random_state + seed_offset
            params = dict(params_base)
            if config.model_name == "extratrees":
                params["random_state"] = seed
            elif config.model_name == "xgboost":
                params["random_state"] = seed
            elif config.model_name == "lightgbm":
                params["random_state"] = seed
            elif config.model_name == "catboost":
                params["random_seed"] = seed

            for fold_i, (tr, va) in enumerate(splits):
                X_tr, y_tr = X[tr], y[tr]
                X_va, y_va = X[va], y[va]
                y_va_ret = y_va.astype(np.float64)  # Use raw returns for ret metrics
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

                # Compute metrics for this fold
                # 1. PR AUC Lift
                pr_auc_lift = _compute_pr_auc_lift(y_va, y_score)

                # 2. Top-30% Absolute Returns Lift
                top30_abs_ret_lift = _compute_top30_abs_ret_lift(y_score, y_va_ret)

                # 3. IC T30 (Spearman in top 30% of predictions)
                ic_t30 = 0.0
                if len(y_score) > 10:
                    ranks = rankdata(y_score) / len(y_score)
                    t30 = ranks >= 0.70
                    if t30.sum() > 2:
                        ic_val = float(spearmanr(y_score[t30], y_va_ret[t30]).statistic)
                        ic_t30 = ic_val if np.isfinite(ic_val) else 0.0

                # 4. Brier Score (for binary calibration)
                brier = 0.0
                try:
                    y_binary = (y_va > 0).astype(np.int32)
                    brier = float(brier_score_loss(y_binary, y_score))
                except Exception:
                    brier = 1.0  # Worst case

                fold_metrics.append({
                    "pr_auc_lift": pr_auc_lift,
                    "top30_abs_ret_lift": top30_abs_ret_lift,
                    "ic_t30": ic_t30,
                    "brier": brier,
                })

        # Aggregate metrics across all folds/seeds
        if not fold_metrics:
            return -1e9

        # Compute summary statistics
        pr_auc_lifts = [m["pr_auc_lift"] for m in fold_metrics]
        top30_abs_ret_lifts = [m["top30_abs_ret_lift"] for m in fold_metrics]
        ic_t30s = [m["ic_t30"] for m in fold_metrics]
        briers = [m["brier"] for m in fold_metrics]

        # Trial-level aggregates
        trial_pr_auc_lift = float(np.median(pr_auc_lifts))
        trial_top30_abs_ret_lift = float(np.median(top30_abs_ret_lifts))
        trial_median_ic_t30 = float(np.median(ic_t30s))
        trial_std_ic_t30 = float(np.std(ic_t30s))
        trial_brier = float(np.mean(briers))

        # Update running stats for z-score computation
        stats_pr = _running_stats_base["pr_auc_lift"]
        stats_top30 = _running_stats_base["top30_abs_ret_lift"]
        stats_median_ic = _running_stats_base["median_ic_t30"]
        stats_std_ic = _running_stats_base["std_ic_t30"]
        stats_brier = _running_stats_base["brier"]

        # Compute z-scores (using population stats if available, else use trial as reference)
        z_pr = stats_pr.zscore(trial_pr_auc_lift) if stats_pr.n >= 10 else trial_pr_auc_lift * 5.0
        z_top30 = stats_top30.zscore(trial_top30_abs_ret_lift) if stats_top30.n >= 10 else trial_top30_abs_ret_lift * 5.0
        z_median_ic = stats_median_ic.zscore(trial_median_ic_t30) if stats_median_ic.n >= 10 else trial_median_ic_t30 * 5.0
        z_std_ic = stats_std_ic.zscore(trial_std_ic_t30) if stats_std_ic.n >= 10 else trial_std_ic_t30 * 5.0
        z_brier = stats_brier.zscore(trial_brier) if stats_brier.n >= 10 else (trial_brier - 0.25) * 5.0

        # Update running stats with this trial's values for future trials
        stats_pr.update(trial_pr_auc_lift)
        stats_top30.update(trial_top30_abs_ret_lift)
        stats_median_ic.update(trial_median_ic_t30)
        stats_std_ic.update(trial_std_ic_t30)
        stats_brier.update(trial_brier)

        # Composite score: 0.5*Z_PR + 0.3*Z_Top30 + 0.2*Z_IC - 0.2*Z_Std_IC - 0.1*Z_Brier
        composite_score = (
            0.5 * z_pr
            + 0.3 * z_top30
            + 0.2 * z_median_ic
            - 0.2 * z_std_ic  # Penalize high std
            - 0.1 * z_brier   # Penalize high Brier
        )

        # Report intermediate value for pruning
        trial.report(composite_score, step=len(fold_metrics))

        # Store raw metrics for analysis
        trial.set_user_attr("pr_auc_lift", trial_pr_auc_lift)
        trial.set_user_attr("top30_abs_ret_lift", trial_top30_abs_ret_lift)
        trial.set_user_attr("median_ic_t30", trial_median_ic_t30)
        trial.set_user_attr("std_ic_t30", trial_std_ic_t30)
        trial.set_user_attr("brier", trial_brier)
        trial.set_user_attr("z_pr", z_pr)
        trial.set_user_attr("z_top30", z_top30)
        trial.set_user_attr("z_median_ic", z_median_ic)
        trial.set_user_attr("z_std_ic", z_std_ic)
        trial.set_user_attr("z_brier", z_brier)

        return float(composite_score)

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

        p.update({"n_jobs": 3, "random_state": random_state})

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


def load_best_base_extratrees_payload(
    out_dir: str = "./hpo_out",
    scope_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load the full saved base HPO payload for a given scope."""
    return _load_base_hpo_json(out_dir, scope_key=scope_key)


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
    selected_features: Optional[List[str]] = None,
    symbols: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_trials: int = 150,
    n_splits: int = 3,
    purge: int = 12,
    embargo: int = 2,
    max_rows: int = 6500,
    random_state: int = 42,
    out_dir: str = "./hpo_out",
    study_name: str = "base_extratrees_hpo",
    scope_key: Optional[str] = None,
    optuna_patience_trials: int = 30,
    optuna_min_trials_before_stop: int = 50,
    optuna_meaningful_improvement_pct: float = 0.005,
) -> Dict[str, Any]:
    """Run HPO for the base ExtraTrees classifier with a narrow search space.

    - 150 Optuna trials, MedianPruner (prune after 30 % of CV rounds)
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
    warmup_steps = max(1, int(np.ceil(0.20 * total_rounds)))

    sampler = TPESampler(seed=random_state, multivariate=True, group=True)
    pruner = SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2,
        min_early_stopping_rate=0,
        bootstrap_count=5,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    def _log_best_trial(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial) -> None:
        completed = [
            t
            for t in study_obj.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and np.isfinite(float(t.value))
        ]
        if not completed:
            return
        best_trial_local = max(completed, key=lambda t: float(t.value))
        if best_trial_local.number != trial_obj.number:
            return
        best_params_local = suggest_extratrees_base(
            best_trial_local, base_random_state=random_state, n_pos=n_pos
        )
        best_params_local.pop("n_jobs", None)
        best_params_local.pop("random_state", None)
        scope_name = scope_key or study_name
        tprint(
            "BASE HPO[{}] NEW BEST: trial={} value={:.6f} params={}".format(
                scope_name,
                trial_obj.number,
                float(best_trial_local.value),
                json.dumps(best_params_local, sort_keys=True),
            )
        )

    scope_name = scope_key or study_name
    patience_callback = _make_optuna_patience_callback(
        patience=int(optuna_patience_trials),
        label=f"BASE HPO[{scope_name}]",
        min_delta=0.0,
        min_trials_before_stop=int(optuna_min_trials_before_stop),
        meaningful_improvement_pct=float(optuna_meaningful_improvement_pct),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_extratrees_base(
            trial, base_random_state=random_state, n_pos=n_pos
        )
        params["random_state"] = random_state

        fold_Qs: List[float] = []
        for fold_i, (tr, va) in enumerate(splits):
            X_tr, y_tr = X[tr], y_hard[tr]
            X_va, y_va = X[va], y_hard[va]
            sw_tr = None if sw is None else sw[tr]

            clf = build_extratrees(params)
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            proba = clf.predict_proba(X_va)[:, 1].astype(np.float32)

            lift_top30 = _compute_lift_top30(proba, y_va)
            prec_top10 = _compute_precision_top10(proba, y_va)
            pr_auc_top30 = _compute_pr_auc_top30(proba, y_va)

            Q_f = 0.30 * lift_top30 + 0.35 * prec_top10 + 0.35 * pr_auc_top30
            if not np.isfinite(Q_f):
                Q_f = 0.0

            fold_Qs.append(Q_f)
            interim_mean_Q = float(np.mean(fold_Qs))
            interim_std_Q = float(np.std(fold_Qs))
            interim_score = interim_mean_Q - 1.0 * interim_std_Q
            trial.report(interim_score, step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_Q = float(np.mean(fold_Qs))
        std_Q = float(np.std(fold_Qs))
        trial_score = mean_Q - 1.0 * std_Q
        trial.set_user_attr("mean_Q", mean_Q)
        trial.set_user_attr("std_Q", std_Q)
        trial.set_user_attr("fold_Qs", [float(v) for v in fold_Qs])
        trial.set_user_attr("objective", trial_score)
        return trial_score

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[_log_best_trial, patience_callback],
    )

    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))
    ]
    if completed_trials:
        best_trial = max(completed_trials, key=lambda t: float(t.value))
        best_value = float(best_trial.value)
        best_params = suggest_extratrees_base(
            best_trial, base_random_state=random_state, n_pos=n_pos
        )
        fallback_used = False
    else:
        tprint(
            "WARNING: Base HPO completed with zero finished trials; using conservative fallback params."
        )
        best_trial = None
        best_value = float("nan")
        best_params = _fallback_extratrees_base_params(
            base_random_state=random_state, n_pos=n_pos
        )
        fallback_used = True
    best_params.pop("n_jobs", None)
    best_params.pop("random_state", None)

    payload = {
        "best_value": best_value,
        "best_params": best_params,
        "selected_features": [str(v) for v in (selected_features or []) if isinstance(v, str)],
        "n_trials_completed": len(study.trials),
        "n_trials_finished": int(len(completed_trials)),
        "n_pos_at_optimisation": n_pos,
        "search_space": "base_narrow",
        "fallback_used": bool(fallback_used),
        "best_trial_metrics": {
            "mean_Q": float(best_trial.user_attrs.get("mean_Q", best_value))
            if best_trial is not None
            else float("nan"),
            "std_Q": float(best_trial.user_attrs.get("std_Q", 0.0))
            if best_trial is not None
            else float("nan"),
            "fold_Qs": [
                float(v) for v in (best_trial.user_attrs.get("fold_Qs", []) if best_trial is not None else [])
            ],
            "objective": float(best_trial.user_attrs.get("objective", best_value))
            if best_trial is not None
            else float("nan"),
        },
    }

    _save_base_hpo_json(out_dir, payload, scope_key=scope_key)
    save_name = os.path.basename(_base_hpo_json_path(out_dir, scope_key=scope_key))
    tprint(f"Base HPO: saved params from current run to {save_name}")

    payload["out_dir"] = out_dir
    return payload


def run_base_extratrees_reg_hpo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    selected_features: Optional[List[str]] = None,
    symbols: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    n_trials: int = 150,
    n_splits: int = 3,
    purge: int = 12,
    embargo: int = 2,
    max_rows: int = 6500,
    random_state: int = 42,
    out_dir: str = "./hpo_out",
    study_name: str = "base_extratrees_reg_hpo",
    scope_key: Optional[str] = None,
    optuna_patience_trials: int = 30,
    optuna_min_trials_before_stop: int = 50,
    optuna_meaningful_improvement_pct: float = 0.005,
) -> Dict[str, Any]:
    X = _as_2d(X)
    y_reg = _as_1d(y).astype(np.float32, copy=False)
    sw = None if sample_weight is None else _as_1d(sample_weight).astype(np.float32, copy=False)

    os.makedirs(out_dir, exist_ok=True)
    X, y_reg = _subsample_diverse(
        X, y_reg, symbols, max_rows=max_rows, rng_seed=random_state
    )
    if sw is not None:
        _, sw = _subsample_diverse(
            np.zeros((len(sw), 1), dtype=np.float32),
            sw,
            symbols,
            max_rows=max_rows,
            rng_seed=random_state,
        )
    finite_mask = np.isfinite(y_reg)
    X = X[finite_mask]
    y_reg = y_reg[finite_mask]
    if sw is not None and len(sw) == len(finite_mask):
        sw = sw[finite_mask]
    if X.shape[0] < 200:
        raise ValueError("Base reg HPO: insufficient finite rows after filtering.")

    n_positive = int(np.sum(y_reg > 0.0))
    tprint(
        f"Base reg HPO: {X.shape[0]} rows, {X.shape[1]} features after subsample; "
        f"positive_target={n_positive}, zero_target={int(np.sum(y_reg <= 0.0))}"
    )

    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
    splits = cv.split(X)
    if len(splits) < 2:
        raise ValueError(
            "Not enough CV splits — reduce purge/embargo or increase n_splits."
        )

    sampler = TPESampler(seed=random_state, multivariate=True, group=True)
    pruner = SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2,
        min_early_stopping_rate=0,
        bootstrap_count=1,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    def _pseudo_huber(err: np.ndarray, delta: float = 1.0) -> float:
        e = np.asarray(err, dtype=np.float64)
        return float(
            np.mean((delta**2) * (np.sqrt(1.0 + np.square(e / delta)) - 1.0))
        )

    def _spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 5:
            return 0.0
        try:
            val = float(spearmanr(a, b).statistic)
        except Exception:
            val = 0.0
        return 0.0 if not np.isfinite(val) else val

    def _top_metric(pred: np.ndarray, target: np.ndarray, frac: float) -> float:
        n_top = max(5, int(np.ceil(len(pred) * float(frac))))
        idx = np.argsort(pred)[-n_top:]
        return _spearman_safe(pred[idx], target[idx])

    def _bucket_monotonicity(pred: np.ndarray, target: np.ndarray) -> float:
        if len(pred) < 50:
            return 0.0
        order = np.argsort(pred)
        buckets = np.array_split(order, 10)
        means = np.array(
            [np.nanmean(target[idx]) if len(idx) else np.nan for idx in buckets],
            dtype=np.float64,
        )
        valid = np.isfinite(means)
        if int(valid.sum()) < 3:
            return 0.0
        return _spearman_safe(np.arange(len(means), dtype=np.float64)[valid], means[valid])

    scope_name = scope_key or study_name
    patience_callback = _make_optuna_patience_callback(
        patience=int(optuna_patience_trials),
        label=f"BASE REG HPO[{scope_name}]",
        min_delta=0.0,
        min_trials_before_stop=int(optuna_min_trials_before_stop),
        meaningful_improvement_pct=float(optuna_meaningful_improvement_pct),
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_extratrees_base_reg(
            trial, base_random_state=random_state, n_pos=max(n_positive, 1)
        )
        fold_scores: list[float] = []
        fold_ics: list[float] = []
        fold_mae: list[float] = []
        fold_huber: list[float] = []
        fold_mono: list[float] = []
        for fold_i, (tr, va) in enumerate(splits):
            X_tr, y_tr = X[tr], y_reg[tr]
            X_va, y_va = X[va], y_reg[va]
            sw_tr = None if sw is None else sw[tr]
            model = ExtraTreesRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            pred = model.predict(X_va).astype(np.float32, copy=False)
            ic = _spearman_safe(pred, y_va)
            ic_top20 = _top_metric(pred, y_va, 0.20)
            ic_top10 = _top_metric(pred, y_va, 0.10)
            mono = _bucket_monotonicity(pred, y_va)
            mae = float(np.mean(np.abs(pred - y_va)))
            huber = _pseudo_huber(pred - y_va, delta=1.0)
            score = float(
                0.35 * ic_top10
                + 0.25 * ic_top20
                + 0.20 * ic
                + 0.10 * mono
                - 0.06 * mae
                - 0.04 * huber
            )
            fold_scores.append(score)
            fold_ics.append(ic)
            fold_mae.append(mae)
            fold_huber.append(huber)
            fold_mono.append(mono)
            trial.report(float(np.mean(fold_scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial_score = float(np.mean(fold_scores) - 0.50 * np.std(fold_scores))
        trial.set_user_attr("mean_ic", float(np.mean(fold_ics)))
        trial.set_user_attr("mean_mae", float(np.mean(fold_mae)))
        trial.set_user_attr("mean_huber", float(np.mean(fold_huber)))
        trial.set_user_attr("mean_mono", float(np.mean(fold_mono)))
        trial.set_user_attr("objective", trial_score)
        trial.set_user_attr("fold_scores", [float(v) for v in fold_scores])
        return trial_score

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=None,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[patience_callback],
    )

    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))
    ]
    if completed_trials:
        best_trial = max(completed_trials, key=lambda t: float(t.value))
        best_value = float(best_trial.value)
        best_params = suggest_extratrees_base_reg(
            best_trial, base_random_state=random_state, n_pos=max(n_positive, 1)
        )
        fallback_used = False
    else:
        tprint(
            "WARNING: Base reg HPO completed with zero finished trials; using conservative fallback params."
        )
        best_trial = None
        best_value = float("nan")
        best_params = _fallback_extratrees_base_reg_params(
            base_random_state=random_state, n_pos=max(n_positive, 1)
        )
        fallback_used = True
    best_params.pop("n_jobs", None)
    best_params.pop("random_state", None)

    payload = {
        "best_value": best_value,
        "best_params": best_params,
        "selected_features": [
            str(v) for v in (selected_features or []) if isinstance(v, str)
        ],
        "n_trials_completed": len(study.trials),
        "n_trials_finished": int(len(completed_trials)),
        "n_pos_at_optimisation": n_positive,
        "search_space": "base_reg_narrow",
        "fallback_used": bool(fallback_used),
        "best_trial_metrics": {
            "mean_ic": float(best_trial.user_attrs.get("mean_ic", best_value))
            if best_trial is not None
            else float("nan"),
            "mean_mae": float(best_trial.user_attrs.get("mean_mae", np.nan))
            if best_trial is not None
            else float("nan"),
            "mean_huber": float(best_trial.user_attrs.get("mean_huber", np.nan))
            if best_trial is not None
            else float("nan"),
            "mean_mono": float(best_trial.user_attrs.get("mean_mono", np.nan))
            if best_trial is not None
            else float("nan"),
            "objective": float(best_trial.user_attrs.get("objective", best_value))
            if best_trial is not None
            else float("nan"),
            "fold_scores": [
                float(v)
                for v in (
                    best_trial.user_attrs.get("fold_scores", [])
                    if best_trial is not None
                    else []
                )
            ],
        },
    }
    _save_base_hpo_json(out_dir, payload, scope_key=scope_key)
    save_name = os.path.basename(_base_hpo_json_path(out_dir, scope_key=scope_key))
    tprint(f"Base reg HPO: saved params from current run to {save_name}")
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
