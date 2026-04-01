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
MIN_GAIN_GRID = (1e-4, 5e-4, 1e-3, 2e-3)
MIN_LEAF_FRAC_GRID = (0.0005, 0.0010, 0.0015)

# Search controls
TARGET_SUPPORT = 0.15
SUPPORT_BAND_WIDTH = 0.05
FAVORED_BAND_WIDTH = 0.025

SUPPORT_MIN = TARGET_SUPPORT - SUPPORT_BAND_WIDTH
SUPPORT_MAX = TARGET_SUPPORT + SUPPORT_BAND_WIDTH
PREFERRED_SUPPORT_MIN = TARGET_SUPPORT - FAVORED_BAND_WIDTH
PREFERRED_SUPPORT_MAX = TARGET_SUPPORT + FAVORED_BAND_WIDTH
MAX_OUTSIDE_SUPPORT_WEIGHT = 0.35

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


def _support_quality_score(
    supports: np.ndarray,
    weights: np.ndarray,
    *,
    target_support: float = TARGET_SUPPORT,
    preferred_min: float = PREFERRED_SUPPORT_MIN,
    preferred_max: float = PREFERRED_SUPPORT_MAX,
) -> float:
    """Weighted support quality centered on target_support with bonus in preferred band."""
    supports = np.asarray(supports, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if supports.size == 0 or weights.size == 0:
        return -np.inf

    w_sum = float(weights.sum())
    if w_sum <= 0.0:
        return -np.inf

    dist = np.abs(supports - target_support)
    base = np.clip(1.0 - (dist / max(target_support, 1e-9)), 0.0, 1.0)
    preferred_bonus = np.where(
        (supports >= preferred_min) & (supports <= preferred_max), 1.0, 0.0
    )
    quality = 0.7 * base + 0.3 * preferred_bonus
    return float(np.dot(quality, weights) / w_sum)


def _support_preference_score_scalar(
    support_pct: float,
    *,
    target_pct: float = TARGET_SUPPORT,
    preferred_low_pct: float = PREFERRED_SUPPORT_MIN,
    preferred_high_pct: float = PREFERRED_SUPPORT_MAX,
) -> float:
    dist = abs(float(support_pct) - float(target_pct))
    base = max(0.0, 1.0 - dist / max(float(target_pct), 1e-9))
    preferred = 1.0 if preferred_low_pct <= support_pct <= preferred_high_pct else 0.0
    return float(0.7 * base + 0.3 * preferred)


def make_support_preference_weights(
    X: np.ndarray,
    *,
    target_pct: float = TARGET_SUPPORT,
    preferred_low_pct: float = PREFERRED_SUPPORT_MIN,
    preferred_high_pct: float = PREFERRED_SUPPORT_MAX,
    strength: float = 0.20,
    w_min: float = 0.85,
    w_max: float = 1.25,
) -> np.ndarray:
    if X.size == 0:
        return np.empty(X.shape[0], dtype=np.float32)

    X_bin = np.asarray(X > 0.5, dtype=np.float32)
    if X_bin.shape[1] == 0:
        return np.ones(X.shape[0], dtype=np.float32)

    support_pct = np.mean(X_bin, axis=0, dtype=np.float64)
    feature_pref = np.array(
        [
            _support_preference_score_scalar(
                float(p),
                target_pct=target_pct,
                preferred_low_pct=preferred_low_pct,
                preferred_high_pct=preferred_high_pct,
            )
            for p in support_pct
        ],
        dtype=np.float32,
    )
    global_pref = float(np.mean(feature_pref)) if feature_pref.size > 0 else 0.0
    active_count = np.sum(X_bin, axis=1, dtype=np.float32)
    active_pref_sum = X_bin @ feature_pref
    row_pref = np.where(
        active_count > 0.0,
        active_pref_sum / np.maximum(active_count, 1.0),
        global_pref,
    )
    centered = row_pref - global_pref
    weights = 1.0 + float(strength) * centered
    return np.clip(weights, w_min, w_max).astype(np.float32, copy=False)


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
    min_leaf_from_frac = int(round(cfg.min_leaf_frac * n_train_subsample))
    max_depth = int(main_params.get("max_depth", 6))
    num_leaves = int(main_params.get("num_leaves", 64))
    depth_leaf_budget = (
        2 ** max(max_depth, 1) if max_depth > 0 else num_leaves
    )
    effective_leaf_budget = max(1, min(num_leaves, depth_leaf_budget))
    depth_aware_floor = int(np.ceil(n_train_subsample / (effective_leaf_budget * 8.0)))
    params["min_data_in_leaf"] = max(
        MIN_LEAF_FLOOR,
        min_leaf_from_frac,
        depth_aware_floor,
    )

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
    train_weight: Optional[np.ndarray] = None,
) -> lgb.Booster:
    lgb_train = lgb.Dataset(
        X_train, label=y_train, weight=train_weight, free_raw_data=True
    )
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
    vol_val: np.ndarray,
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

    # Volatility normalization of returns using local volatility (vol_val)
    vol_safe = np.maximum(vol_val, 1e-12)
    y_val_norm = y_val / vol_safe

    baseline_mean_norm = float(y_val_norm.mean())

    sum_y_norm = M.T @ y_val_norm                     # [k]
    mean_return_norm = sum_y_norm / counts            # [k]

    incremental = mean_return_norm - baseline_mean_norm # [k]

    # Alternative to IC: Difference in Means (Cohen's d-like or direct delta)
    # We already have mean_return_norm. Let's calculate the inactive mean return
    # to compute the difference in means between active and inactive states.
    # inactive counts = n - counts
    sum_y_inactive = y_val_norm.sum() - sum_y_norm
    inactive_counts = n - counts
    mean_return_inactive = np.zeros_like(sum_y_inactive)
    valid_inactive = inactive_counts > 0
    mean_return_inactive[valid_inactive] = sum_y_inactive[valid_inactive] / inactive_counts[valid_inactive]

    diff_in_means = mean_return_norm - mean_return_inactive

    # Sortino ratio approximation per rule
    # We need the standard deviation of *negative* returns when the rule is active
    # y_val_norm_negative is only the downside parts
    y_val_norm_neg = np.minimum(y_val_norm, 0.0)
    sum_y_norm_neg_sq = M.T @ (y_val_norm_neg ** 2)

    # Downside variance: Average of squared negative returns
    downside_var = sum_y_norm_neg_sq / counts
    downside_std = np.sqrt(downside_var + 1e-12)

    # Use incremental return in numerator for Sortino to ensure positive
    # values for valid rules (since we only keep rules with incremental > 0)
    # This prevents the negative multiplication bug when integrating metrics.
    sortino = incremental / downside_std # [k]

    # We only keep rules that have positive incremental return
    keep = incremental > 0.0
    if not np.any(keep):
        return -np.inf, {"reason": "no_positive_rules"}

    M_keep = M[:, keep]
    supports_keep = supports[keep]
    incremental_keep = incremental[keep]
    diff_in_means_keep = diff_in_means[keep]
    sortino_keep = sortino[keep]

    support_weights = supports_keep.copy()
    support_w_sum = float(support_weights.sum()) + 1e-12
    outside_support_mask = (supports_keep < support_min) | (supports_keep > support_max)
    outside_support_weight = float(
        np.dot(outside_support_mask.astype(np.float64), support_weights) / support_w_sum
    )
    if outside_support_weight > MAX_OUTSIDE_SUPPORT_WEIGHT:
        return -np.inf, {
            "reason": "too_many_rules_outside_support_band",
            "outside_support_weight": outside_support_weight,
            "n_kept_rules": int(keep.sum()),
        }

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

    w = support_weights
    w_sum = support_w_sum
    weighted_incremental = float(np.dot(incremental_keep, w) / w_sum)

    # Clip diff_in_means so we don't multiply by negatives either,
    # though it should strictly be positive since incremental > 0.
    weighted_diff_in_means = max(1e-12, float(np.dot(diff_in_means_keep, w) / w_sum))
    weighted_sortino = max(1e-12, float(np.dot(sortino_keep, w) / w_sum))

    support_penalty = 1.0 - abs(total_support - target_support)
    support_quality = _support_quality_score(
        supports_keep,
        w,
        target_support=TARGET_SUPPORT,
        preferred_min=PREFERRED_SUPPORT_MIN,
        preferred_max=PREFERRED_SUPPORT_MAX,
    )
    score = (
        weighted_incremental
        * weighted_diff_in_means
        * weighted_sortino
        * max(support_penalty, 1e-6)
        * max(support_quality, 1e-6)
    )

    diagnostics = {
        "support": total_support,
        "outside_support_weight": outside_support_weight,
        "support_quality": support_quality,
        "weighted_incremental": weighted_incremental,
        "weighted_diff_in_means": weighted_diff_in_means,
        "weighted_sortino": weighted_sortino,
        "n_kept_rules": int(keep.sum()),
        "baseline_mean_norm": baseline_mean_norm,
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
    vol_val: np.ndarray,
    main_params: Dict[str, Any],
) -> EvalResult:
    params = build_hpo_params(main_params, cfg, n_train_subsample=len(X_train))

    train_weight = make_support_preference_weights(X_train)
    model = train_lgbm_quantile(
        X_train, y_train, X_val, y_val, params, train_weight=train_weight
    )

    rule_matrix = build_candidate_rule_matrix(model, X_val)
    score, d = score_rule_matrix_vectorized(
        y_val=y_val,
        vol_val=vol_val,
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
    vol: Optional[np.ndarray] = None,
    main_params: Dict[str, Any] = MAIN_MINER_PARAMS,
    subsample_frac: float = SUBSAMPLE_FRAC,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Main entry point for dynamic HPO integration.
    """
    rng = np.random.default_rng(seed)

    if vol is None:
        vol = np.ones_like(y)

    if len(y) < 100:
        # Fallback for very small datasets
        return {
            "best_alpha_result": EvalResult(
                cfg=HPOConfig(alpha=0.90, min_gain_to_split=0.001, min_leaf_frac=0.0010),
                score=np.nan,
                total_support=np.nan,
                weighted_incremental_return=np.nan,
                weighted_mask_score=np.nan,
                n_kept_rules=0,
                best_iteration=0,
                valid=False,
                reason="small_data_fallback_n<100",
            ),
            "best_final_result": EvalResult(
                cfg=HPOConfig(alpha=0.90, min_gain_to_split=0.001, min_leaf_frac=0.0010),
                score=np.nan,
                total_support=np.nan,
                weighted_incremental_return=np.nan,
                weighted_mask_score=np.nan,
                n_kept_rules=0,
                best_iteration=0,
                valid=False,
                reason="small_data_fallback_n<100",
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
    vol_sub = vol[sub_idx]

    tr_idx, va_idx = train_val_split_time_ordered(len(y_sub), val_frac=0.25)
    X_train, y_train = X_sub[tr_idx], y_sub[tr_idx]
    X_val, y_val = X_sub[va_idx], y_sub[va_idx]
    vol_val = vol_sub[va_idx]

    # Stage 1: alpha screen
    stage1_cfgs = [
        HPOConfig(alpha=a, min_gain_to_split=0.001, min_leaf_frac=0.0010)
        for a in ALPHA_GRID
    ]
    stage1_results = [
        evaluate_config(cfg, X_train, y_train, X_val, y_val, vol_val, main_params)
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
        evaluate_config(cfg, X_train, y_train, X_val, y_val, vol_val, main_params)
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
            vol=np.ones_like(y), # Default to 1 if not provided in tests
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
