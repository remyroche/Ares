from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import lightgbm as lgb


# ============================================================
# 1) MAIN MINER PARAMS SPEC
#    Copy your main miner params here EXACTLY.
#    Only learning_rate will be changed by HPO wrapper.
# ============================================================

MAIN_MINER_PARAMS: Dict[str, Any] = {
    # ----- copy these from your main LGBM miner -----
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.03,          # HPO wrapper will double this -> 0.06
    "num_leaves": 64,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 10.0,
    "min_gain_to_split": 0.002,     # optimized in HPO
    "min_data_in_leaf": 100,        # optimized in HPO (from subsample)
    "verbosity": -1,
    "extra_trees": True,
    "random_state": 42,
    "n_jobs": 3,
}

# HPO grids
ALPHA_GRID = (0.80, 0.825, 0.85, 0.875, 0.90)
MIN_GAIN_GRID = (0.001, 0.002, 0.003)
MIN_LEAF_FRAC_GRID = (0.001, 0.0015, 0.002)

# Search controls
SUPPORT_MIN = 0.05
SUPPORT_MAX = 0.10
TARGET_SUPPORT = 0.06

SUBSAMPLE_FRAC = 0.30
MAX_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 25
MIN_LEAF_FLOOR = 25

RANDOM_SEED = 42


@dataclass(frozen=True)
class HPOConfig:
    alpha: float
    min_gain_to_split: float
    min_leaf_frac: float


@dataclass
class EvalResult:
    cfg: HPOConfig
    score: float
    total_support: float
    weighted_incremental_return: float
    weighted_mask_score: float
    n_kept_rules: int
    best_iteration: int
    valid: bool
    reason: str


# ============================================================
# 2) PARAM HANDLING
# ============================================================

def build_hpo_params(main_params: Dict[str, Any], cfg: HPOConfig, n_train_subsample: int) -> Dict[str, Any]:
    """
    Exact same params as the main miner, except:
    - learning_rate is doubled
    - alpha / min_gain_to_split / min_data_in_leaf are replaced by HPO values
    - min_data_in_leaf is computed from TRAIN SUBSAMPLE SIZE, not full dataset
    """
    params = dict(main_params)

    # Same as main miner, except LR is doubled
    params["learning_rate"] = float(main_params["learning_rate"]) * 2.0

    # HPO dimensions
    params["alpha"] = cfg.alpha
    params["min_gain_to_split"] = cfg.min_gain_to_split
    params["min_data_in_leaf"] = max(MIN_LEAF_FLOOR, int(round(cfg.min_leaf_frac * n_train_subsample)))

    # Ensure objective/metric remain consistent
    params["objective"] = "quantile"
    params["metric"] = "quantile"

    return params


# ============================================================
# 3) SUBSAMPLING / SPLITTING
# ============================================================

def block_subsample_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """
    Fast contiguous block subsample to preserve time structure better than iid sampling.
    """
    m = max(1, int(round(n * frac)))
    if m >= n:
        return np.arange(n, dtype=np.int32)

    start = int(rng.integers(0, n - m + 1))
    return np.arange(start, start + m, dtype=np.int32)


def train_val_split_time_ordered(n: int, val_frac: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple recent-block validation split.
    """
    n_val = max(1, int(round(n * val_frac)))
    split = n - n_val
    train_idx = np.arange(split, dtype=np.int32)
    val_idx = np.arange(split, n, dtype=np.int32)
    return train_idx, val_idx


# ============================================================
# 4) MODEL TRAINING
# ============================================================

def train_lgbm_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> lgb.Booster:
    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=True)

    booster = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_val],
        valid_names=["valid"],
        num_boost_round=MAX_BOOST_ROUNDS,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster


# ============================================================
# 5) RULE EXTRACTION HOOK
# ============================================================

def build_candidate_rule_matrix(model: lgb.Booster, X_val: np.ndarray) -> np.ndarray:
    """
    IMPORTANT: replace this with your own miner's exact leaf/rule extraction logic.

    Must return:
        rule_matrix: bool array of shape [n_val, n_rules]
    """
    # Vectorized leaf prediction
    leaf_idx = model.predict(X_val, pred_leaf=True)  # shape: [n_val, n_trees]
    if leaf_idx.ndim == 1:
        leaf_idx = leaf_idx[:, None]

    # Unique combinations across all trees define the rules (regimes)
    _, inv = np.unique(leaf_idx, axis=0, return_inverse=True)

    n_val = X_val.shape[0]
    n_rules = inv.max() + 1

    rule_matrix = np.zeros((n_val, n_rules), dtype=bool)
    rule_matrix[np.arange(n_val), inv] = True
    return rule_matrix


# ============================================================
# 6) VECTORIZED MODEL-LEVEL SCORING
# ============================================================

def score_rule_matrix_vectorized(
    y_val: np.ndarray,
    rule_matrix: np.ndarray,
    support_min: float = SUPPORT_MIN,
    support_max: float = SUPPORT_MAX,
    target_support: float = TARGET_SUPPORT,
) -> Tuple[float, Dict[str, Any]]:
    """
    Fully vectorized scoring from raw validation outcomes + boolean rule matrix.
    """

    n = y_val.shape[0]
    if n == 0 or rule_matrix.size == 0:
        return -np.inf, {"reason": "empty_rule_matrix"}

    M = np.asarray(rule_matrix, dtype=np.uint8)   # [n, k]
    k = M.shape[1]
    if k == 0:
        return -np.inf, {"reason": "no_rules"}

    counts = M.sum(axis=0).astype(np.float64)     # [k]
    valid_count_mask = counts > 0
    if not np.any(valid_count_mask):
        return -np.inf, {"reason": "all_rules_empty"}

    M = M[:, valid_count_mask]
    counts = counts[valid_count_mask]
    supports = counts / float(n)

    baseline_mean = float(y_val.mean())
    baseline_std = float(y_val.std(ddof=1)) + 1e-12

    sum_y = M.T @ y_val                             # [k]
    mean_return = sum_y / counts                    # [k]

    incremental = mean_return - baseline_mean       # [k]
    mask_score = incremental / baseline_std         # [k]

    keep = incremental > 0.0
    if not np.any(keep):
        return -np.inf, {"reason": "no_positive_rules"}

    M_keep = M[:, keep]
    supports_keep = supports[keep]
    incremental_keep = incremental[keep]
    mask_keep = mask_score[keep]

    union_mask = M_keep.any(axis=1)
    total_support = float(union_mask.mean())

    if total_support < support_min:
        return -np.inf, {
            "reason": "support_too_low",
            "support": total_support,
            "n_kept_rules": int(keep.sum()),
        }

    if total_support > support_max:
        return -np.inf, {
            "reason": "support_too_high",
            "support": total_support,
            "n_kept_rules": int(keep.sum()),
        }

    w = supports_keep
    w_sum = float(w.sum()) + 1e-12
    weighted_incremental = float(np.dot(incremental_keep, w) / w_sum)
    weighted_mask = float(np.dot(mask_keep, w) / w_sum)

    support_penalty = 1.0 - abs(total_support - target_support)
    score = weighted_incremental * weighted_mask * support_penalty

    diagnostics = {
        "support": total_support,
        "weighted_incremental": weighted_incremental,
        "weighted_mask": weighted_mask,
        "n_kept_rules": int(keep.sum()),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
    }
    return score, diagnostics


# ============================================================
# 7) SINGLE CONFIG EVALUATION
# ============================================================

def evaluate_config(
    cfg: HPOConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    main_params: Dict[str, Any],
) -> EvalResult:
    params = build_hpo_params(main_params, cfg, n_train_subsample=len(X_train))

    model = train_lgbm_quantile(X_train, y_train, X_val, y_val, params)

    rule_matrix = build_candidate_rule_matrix(model, X_val)
    score, d = score_rule_matrix_vectorized(
        y_val=y_val,
        rule_matrix=rule_matrix,
        support_min=SUPPORT_MIN,
        support_max=SUPPORT_MAX,
        target_support=TARGET_SUPPORT,
    )

    valid = np.isfinite(score)
    return EvalResult(
        cfg=cfg,
        score=float(score),
        total_support=float(d.get("support", np.nan)),
        weighted_incremental_return=float(d.get("weighted_incremental", np.nan)),
        weighted_mask_score=float(d.get("weighted_mask", np.nan)),
        n_kept_rules=int(d.get("n_kept_rules", 0)),
        best_iteration=int(getattr(model, "best_iteration", 0) or 0),
        valid=bool(valid),
        reason=str(d.get("reason", "ok" if valid else "invalid")),
    )


# ============================================================
# 8) TWO-STAGE SHORT HPO
# ============================================================

def run_short_hpo_for_target_horizon(
    X: np.ndarray,
    y: np.ndarray,
    main_params: Dict[str, Any] = MAIN_MINER_PARAMS,
    subsample_frac: float = SUBSAMPLE_FRAC,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Main entry point for dynamic HPO integration.
    """
    rng = np.random.default_rng(seed)

    if len(y) < 100:
        # Fallback for very small datasets
        return {
            "best_alpha_result": EvalResult(
                cfg=HPOConfig(alpha=0.95, min_gain_to_split=0.002, min_leaf_frac=0.0015),
                score=0.0,
                total_support=0.0,
                weighted_incremental_return=0.0,
                weighted_mask_score=0.0,
                n_kept_rules=0,
                best_iteration=0,
                valid=True,
                reason="small_data_fallback",
            ),
            "best_final_result": EvalResult(
                cfg=HPOConfig(alpha=0.95, min_gain_to_split=0.002, min_leaf_frac=0.0015),
                score=0.0,
                total_support=0.0,
                weighted_incremental_return=0.0,
                weighted_mask_score=0.0,
                n_kept_rules=0,
                best_iteration=0,
                valid=True,
                reason="small_data_fallback",
            ),
            "stage1_results": [],
            "stage2_results": [],
            "subsample_n": len(y),
            "train_n": len(y),
            "val_n": 0,
        }

    sub_idx = block_subsample_indices(len(y), frac=subsample_frac, rng=rng)
    X_sub = X[sub_idx]
    y_sub = y[sub_idx]

    tr_idx, va_idx = train_val_split_time_ordered(len(y_sub), val_frac=0.25)
    X_train, y_train = X_sub[tr_idx], y_sub[tr_idx]
    X_val, y_val = X_sub[va_idx], y_sub[va_idx]

    # Stage 1: alpha screen
    stage1_cfgs = [
        HPOConfig(alpha=a, min_gain_to_split=0.002, min_leaf_frac=0.0015)
        for a in ALPHA_GRID
    ]
    stage1_results = [
        evaluate_config(cfg, X_train, y_train, X_val, y_val, main_params)
        for cfg in stage1_cfgs
    ]

    valid_stage1 = [r for r in stage1_results if r.valid]
    if valid_stage1:
        best_alpha_result = max(valid_stage1, key=lambda r: r.score)
    else:
        best_alpha_result = max(stage1_results, key=lambda r: r.score)

    best_alpha = best_alpha_result.cfg.alpha

    # Stage 2: local grid
    stage2_cfgs = [
        HPOConfig(alpha=best_alpha, min_gain_to_split=g, min_leaf_frac=lf)
        for g, lf in product(MIN_GAIN_GRID, MIN_LEAF_FRAC_GRID)
    ]
    stage2_results = [
        evaluate_config(cfg, X_train, y_train, X_val, y_val, main_params)
        for cfg in stage2_cfgs
    ]

    valid_stage2 = [r for r in stage2_results if r.valid]
    if valid_stage2:
        best_final = max(valid_stage2, key=lambda r: r.score)
    else:
        best_final = max(stage2_results, key=lambda r: r.score)

    return {
        "best_alpha_result": best_alpha_result,
        "best_final_result": best_final,
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "subsample_n": len(y_sub),
        "train_n": len(y_train),
        "val_n": len(y_val),
    }


def run_hpo_all(
    datasets: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    main_params: Dict[str, Any] = MAIN_MINER_PARAMS,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (target, horizon), (X, y) in datasets.items():
        out[(target, horizon)] = run_short_hpo_for_target_horizon(
            X=X,
            y=y,
            main_params=main_params,
            subsample_frac=SUBSAMPLE_FRAC,
            seed=RANDOM_SEED,
        )
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    X_demo = rng.normal(size=(10000, 20)).astype(np.float32)
    y_demo = (0.001 * X_demo[:, 0] - 0.0007 * X_demo[:, 1] + rng.normal(scale=0.01, size=10000)).astype(np.float32)

    datasets = {
        ("target_a", "3h"): (X_demo, y_demo),
    }

    results = run_hpo_all(datasets, main_params=MAIN_MINER_PARAMS)

    for key, res in results.items():
        best = res["best_final_result"]
        print(f"\n{key}")
        print("best cfg:", best.cfg)
        print("score:", best.score)
        print("support:", best.total_support)
        print("valid:", best.valid)
        print("reason:", best.reason)
