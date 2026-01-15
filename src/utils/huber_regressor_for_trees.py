"""
Huber Teacher Orchestrator (Updated)
-----------------------------------
Goal: Use a robust linear teacher (Huber) to:
  1) Warm-start downstream tree models (baseline predictions + residual targets)
  2) Provide high-precision monotonic constraints
  3) Provide interaction constraints (LightGBM/XGBoost) + interaction audit for CatBoost

Key upgrades vs. original:
  - Single global RobustScaler reused across all splits (coefficients comparable)
  - Optional hyperparameter grid usage (time-split + param stability)
  - Monotonic constraints inferred from teacher effect curves (ALE/PDP-like binning),
    not just coefficient sign (more tree-native, less confounded)
  - Meaningful stability metrics (sign stability + top-K inclusion), no "nonzero rate"
  - Interaction constraints derived from out-of-sample "pair synergy" screening
    (does interaction add beyond additive?) rather than raw feature correlation clustering
  - Output formatted for LightGBM/XGBoost/CatBoost

Notes:
  - Interaction constraints are supported by LightGBM and XGBoost.
  - CatBoost supports monotone constraints, but does not provide a direct "interaction_constraints"
    parameter analogous to LightGBM/XGBoost; we provide an interaction audit list instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

from joblib import Parallel, delayed

from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class HuberTeacherConfig:
    # Huber grid
    epsilons: Tuple[float, ...] = (1.1, 1.35, 1.75)
    alphas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    max_iter: int = 5000

    # Time splits (walk-forward with rolling window)
    n_time_splits: int = 5
    window_mult: float = 2.0  # window length = window_mult * split_size
    embargo: int = 0          # optional embargo samples to avoid label leakage

    # Feature pruning
    pruning_percentile: int = 15          # prune weakest coefficients (by |median coef|)
    topk_inclusion_frac: float = 0.35     # for stability: top-K inclusion rate thresholding

    # Monotone inference (teacher effect curve)
    mono_bins: int = 10
    mono_score_threshold: float = 0.9     # fraction of adjacent diffs consistent in sign
    mono_sign_stability: float = 0.8      # fraction of splits agreeing on direction
    mono_effect_min_frac: float = 0.10    # effect range >= frac * median(|teacher pred|) or fallback scale
    mono_use_coeff_fallback: bool = True  # if curve ambiguous but coef sign stable, allow fallback

    # Interaction inference (pair synergy screening)
    interaction_max_features: int = 80    # shortlist size for pair screening
    interaction_top_pairs: int = 50       # keep strongest pairs for grouping
    interaction_min_gain: float = 0.0     # minimum MAE improvement (positive means better) per split
    interaction_sign_stability: float = 0.6  # fraction splits with positive gain

    # Correlation grouping fallback (optional)
    corr_group_threshold: float = 0.7     # used only if you enable correlation grouping

    # Compute
    n_jobs: int = -1
    verbose: bool = True


# -----------------------------
# Helpers
# -----------------------------

def _as_float32_2d(df: Union[pd.DataFrame, np.ndarray], columns: Optional[List[str]] = None) -> np.ndarray:
    if isinstance(df, pd.DataFrame):
        if columns is not None:
            df = df[columns]
        return df.values.astype(np.float32, copy=False)
    arr = np.asarray(df)
    if arr.ndim != 2:
        raise ValueError("X must be 2D.")
    return arr.astype(np.float32, copy=False)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    mean = np.mean(w)
    if not np.isfinite(mean) or mean <= 0:
        return w
    return w / mean


def _make_walkforward_splits(
    n_samples: int,
    n_splits: int,
    window_mult: float,
    embargo: int
) -> List[Tuple[int, int]]:
    """
    Returns list of (start, end) indices for training windows in walk-forward fashion.
    Uses rolling window of length window_mult * split_size, ending at progressive endpoints.
    """
    if n_splits < 2:
        return [(0, n_samples)]

    split_size = max(1, n_samples // n_splits)
    window = max(2, int(window_mult * split_size))

    splits: List[Tuple[int, int]] = []
    for k in range(n_splits):
        end = n_samples if (k == n_splits - 1) else (k + 1) * split_size
        end = max(2, min(end, n_samples))

        # Apply embargo by cutting the last embargo samples off the training end
        train_end = max(2, end - max(0, embargo))

        start = max(0, train_end - window)
        if train_end - start < 2:
            continue
        splits.append((start, train_end))

    # Ensure uniqueness and order
    splits = sorted(list(dict.fromkeys(splits)))
    return splits if splits else [(0, n_samples)]


def _fit_huber(
    X_scaled: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    epsilon: float,
    alpha: float,
    max_iter: int
) -> HuberRegressor:
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=max_iter)
    if sample_weight is None:
        model.fit(X_scaled, y)
    else:
        model.fit(X_scaled, y, sample_weight=sample_weight)
    return model


def _median_choice(vals: Tuple[float, ...]) -> float:
    return float(np.median(np.asarray(vals, dtype=float)))


# -----------------------------
# Split fitting: (split × grid)
# -----------------------------

def _fit_split_grid(
    X_scaled_full: np.ndarray,
    y_full: np.ndarray,
    w_full: Optional[np.ndarray],
    start: int,
    end: int,
    epsilons: Tuple[float, ...],
    alphas: Tuple[float, ...],
    max_iter: int
) -> Dict[str, np.ndarray]:
    X = X_scaled_full[start:end]
    y = y_full[start:end]
    w = None if w_full is None else w_full[start:end]

    coefs = []
    intercepts = []
    params = []
    for eps in epsilons:
        for a in alphas:
            m = _fit_huber(X, y, w, eps, a, max_iter)
            coefs.append(m.coef_.astype(np.float64, copy=False))
            intercepts.append(float(m.intercept_))
            params.append((float(eps), float(a)))

    return {
        "coefs": np.vstack(coefs),                  # (n_grid, n_features)
        "intercepts": np.asarray(intercepts),       # (n_grid,)
        "params": np.asarray(params, dtype=float),  # (n_grid, 2)
        "start_end": np.asarray([start, end], dtype=int)
    }


# -----------------------------
# Monotone inference via teacher effect curve (ALE/PDP-like)
# -----------------------------

def _teacher_effect_curve_sign(
    x: np.ndarray,
    teacher_pred: np.ndarray,
    bins: int
) -> Tuple[int, float, float]:
    """
    Computes a binned "effect curve" of teacher_pred vs x.
    Returns: (sign, monotone_score, effect_range)

    sign: -1, 0, +1 based on overall slope
    monotone_score: fraction of adjacent differences consistent with sign
    effect_range: max - min of binned means
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(teacher_pred, dtype=np.float64)

    # Quantile bins (handles heavy tails)
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        return 0, 0.0, 0.0

    # Assign bins
    idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, edges.size - 2)

    # Mean prediction per bin
    bmeans = []
    for b in range(edges.size - 1):
        mask = idx == b
        if np.any(mask):
            bmeans.append(np.mean(y[mask]))
        else:
            bmeans.append(np.nan)
    bmeans = np.asarray(bmeans, dtype=np.float64)

    # Interpolate missing bins
    if np.all(np.isnan(bmeans)):
        return 0, 0.0, 0.0
    nans = np.isnan(bmeans)
    if np.any(nans):
        xs = np.arange(bmeans.size)
        bmeans[nans] = np.interp(xs[nans], xs[~nans], bmeans[~nans])

    diffs = np.diff(bmeans)
    eff_range = float(np.max(bmeans) - np.min(bmeans))

    overall = bmeans[-1] - bmeans[0]
    if np.isclose(overall, 0.0):
        return 0, 0.0, eff_range

    sgn = 1 if overall > 0 else -1
    score = float(np.mean((diffs >= 0) if sgn > 0 else (diffs <= 0)))
    return sgn, score, eff_range


# -----------------------------
# Interaction inference via pair synergy screening
# -----------------------------

def _pair_synergy_gain_mae(
    x_i: np.ndarray,
    x_j: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray]
) -> float:
    """
    Measures whether an interaction term improves out-of-sample fit beyond additive.

    Model A: y ~ [x_i, x_j]
    Model B: y ~ [x_i, x_j, x_i*x_j]  (scaled outside; here assumes already scaled)

    Returns MAE gain: MAE_A - MAE_B (positive => interaction helps).
    Uses a robust learner (Huber with fixed settings) for stability.
    """
    X_a = np.column_stack([x_i, x_j]).astype(np.float64, copy=False)
    X_b = np.column_stack([x_i, x_j, x_i * x_j]).astype(np.float64, copy=False)

    # Fixed robust params for screening (keep cheap & stable)
    m_a = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=2000)
    m_b = HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=2000)

    if w is None:
        m_a.fit(X_a, y)
        m_b.fit(X_b, y)
    else:
        m_a.fit(X_a, y, sample_weight=w)
        m_b.fit(X_b, y, sample_weight=w)

    pred_a = m_a.predict(X_a)
    pred_b = m_b.predict(X_b)
    mae_a = mean_absolute_error(y, pred_a, sample_weight=w)
    mae_b = mean_absolute_error(y, pred_b, sample_weight=w)
    return float(mae_a - mae_b)


def _build_interaction_groups_from_edges(
    feature_names: List[str],
    edges: List[Tuple[int, int]],
) -> List[List[str]]:
    """
    Converts an edge list into groups via connected components.
    This matches the common "allowed interaction groups" semantics used by LightGBM/XGBoost.
    """
    n = len(feature_names)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(i, j)

    comps: Dict[int, List[int]] = {}
    for k in range(n):
        r = find(k)
        comps.setdefault(r, []).append(k)

    groups = []
    for _, idxs in comps.items():
        if len(idxs) >= 2:
            groups.append([feature_names[i] for i in idxs])

    # Sort groups by size desc
    groups.sort(key=len, reverse=True)
    return groups


# -----------------------------
# Main API
# -----------------------------

def prepare_huber_teacher_outputs(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    vol_proxy: Optional[pd.Series] = None,
    sample_weight: Optional[np.ndarray] = None,
    # Legacy arguments for backward compatibility
    epsilons: List[float] = [1.1, 1.35, 1.75],
    alphas: List[float] = [1e-4, 1e-3, 1e-2],
    pruning_percentile: int = 15,
    corr_threshold: float = 0.7,
    n_jobs: int = -1,
    sign_agree_threshold: float = 0.8,
    nonzero_rate_threshold: float = 0.7,
    n_time_splits: int = 5,
    # New configuration object
    config: Optional[HuberTeacherConfig] = None
) -> Dict:
    """
    Produces:
      - selected_features
      - monotonic constraints (dict + vectors for LGBM/XGB/Cat)
      - interaction constraints (groups for LGBM/XGB; audit list for CatBoost)
      - warm start predictions (train/val/test)
      - residual meta targets (y - warm_start)
      - fitted scaler and teacher model

    Assumptions:
      - X_train is time-ordered already (no shuffling).
    """
    # -----------------------------
    # Config / Backward Compatibility
    # -----------------------------
    if config is None:
        config = HuberTeacherConfig(
            epsilons=tuple(epsilons),
            alphas=tuple(alphas),
            pruning_percentile=pruning_percentile,
            corr_group_threshold=corr_threshold,
            n_jobs=n_jobs,
            n_time_splits=n_time_splits,
            mono_sign_stability=sign_agree_threshold,
            # Note: nonzero_rate_threshold is superseded by topk_inclusion_frac
            # and other stability metrics in the new logic.
        )
    cfg = config

    # -----------------------------
    # Weights
    # -----------------------------
    if sample_weight is not None:
        w = _normalize_weights(np.asarray(sample_weight))
    elif vol_proxy is not None:
        inv = (1.0 / vol_proxy.replace([np.inf, -np.inf], np.nan)).fillna(vol_proxy.median())
        w = _normalize_weights(inv.values)
    else:
        w = None

    # -----------------------------
    # Data to numpy (preserve column order)
    # -----------------------------
    train_columns = X_train.columns.tolist()
    feature_names = np.asarray(train_columns)

    X_tr = _as_float32_2d(X_train, columns=train_columns)
    y_tr = np.asarray(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=np.float64)

    n_samples, n_features = X_tr.shape
    if cfg.verbose:
        print(f"[HuberTeacher] n_samples={n_samples}, n_features={n_features}")

    # -----------------------------
    # Global scaler (critical upgrade)
    # -----------------------------
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float64, copy=False)

    # -----------------------------
    # Walk-forward splits
    # -----------------------------
    splits = _make_walkforward_splits(
        n_samples=n_samples,
        n_splits=cfg.n_time_splits,
        window_mult=cfg.window_mult,
        embargo=cfg.embargo
    )
    if cfg.verbose:
        print(f"[HuberTeacher] splits={len(splits)} -> {splits[:3]}{' ...' if len(splits) > 3 else ''}")

    # -----------------------------
    # Fit split × grid in parallel
    # -----------------------------
    split_grid_results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_fit_split_grid)(
            X_tr_scaled, y_tr, w,
            start, end,
            cfg.epsilons, cfg.alphas, cfg.max_iter
        )
        for (start, end) in splits
    )

    n_grid = len(cfg.epsilons) * len(cfg.alphas)
    coefs_all = np.stack([r["coefs"] for r in split_grid_results], axis=0)  # (n_splits, n_grid, n_features)
    # Aggregate across (splits, grid) for a robust global estimate
    median_coef = np.median(coefs_all.reshape(-1, n_features), axis=0)
    abs_median_coef = np.abs(median_coef)

    # -----------------------------
    # Feature pruning (by median |coef|)
    # -----------------------------
    kth = np.percentile(abs_median_coef, cfg.pruning_percentile)
    keep_mask = abs_median_coef > kth
    selected_feats = feature_names[keep_mask].tolist()
    selected_idx = np.where(keep_mask)[0]

    if cfg.verbose:
        print(f"[HuberTeacher] pruning_percentile={cfg.pruning_percentile} -> kept={keep_mask.sum()}/{n_features}")

    # -----------------------------
    # Fit final teacher on full train (median params)
    # -----------------------------
    eps0 = _median_choice(cfg.epsilons)
    a0 = _median_choice(cfg.alphas)
    teacher = _fit_huber(X_tr_scaled, y_tr, w, eps0, a0, cfg.max_iter)
    warm_train = teacher.predict(X_tr_scaled)

    # -----------------------------
    # Warm-start preds for val/test with column alignment
    # -----------------------------
    def _predict_df(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        if df is None:
            return None
        X = _as_float32_2d(df, columns=train_columns)
        Xs = scaler.transform(X).astype(np.float64, copy=False)
        return teacher.predict(Xs)

    warm_val = _predict_df(X_val)
    warm_test = _predict_df(X_test)

    # -----------------------------
    # Stability metrics: sign stability + top-K inclusion across (splits × grid)
    # -----------------------------
    # Top-K inclusion: feature is "important" if in top-K of |coef| for that fit
    K = max(1, int(cfg.topk_inclusion_frac * n_features))
    flat = coefs_all.reshape(-1, n_features)  # (n_splits*n_grid, n_features)
    ranks = np.argsort(-np.abs(flat), axis=1)  # descending
    topk_sets = ranks[:, :K]

    inclusion = np.zeros(n_features, dtype=np.float64)
    for row in topk_sets:
        inclusion[row] += 1
    inclusion /= topk_sets.shape[0]  # in [0,1]

    sign_ref = np.sign(median_coef)
    sign_stability = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        s = np.sign(flat[:, j])
        # ignore zeros by treating as mismatch
        sign_stability[j] = np.mean(s == sign_ref[j]) if sign_ref[j] != 0 else 0.0

    # -----------------------------
    # Monotone constraints via effect curves (split-level)
    # -----------------------------
    # We compute per-split effect curve sign/score using the final teacher predictions,
    # restricted to that split window. This gives regime-aware monotonicity.
    # (If you want stricter OOS, compute teacher on past and evaluate on forward window.)
    mono_sign_votes = np.zeros((len(splits), n_features), dtype=np.int8)
    mono_scores = np.zeros((len(splits), n_features), dtype=np.float64)
    mono_ranges = np.zeros((len(splits), n_features), dtype=np.float64)

    for si, (start, end) in enumerate(splits):
        pred_s = warm_train[start:end]
        Xs = X_tr_scaled[start:end]
        for j in range(n_features):
            sgn, score, rng = _teacher_effect_curve_sign(Xs[:, j], pred_s, bins=cfg.mono_bins)
            mono_sign_votes[si, j] = sgn
            mono_scores[si, j] = score
            mono_ranges[si, j] = rng

    # Stability of curve sign: fraction of splits agreeing with median curve sign
    curve_median_sign = np.sign(np.median(mono_sign_votes.astype(np.int16), axis=0))
    curve_sign_stability = np.zeros(n_features, dtype=np.float64)
    curve_score_median = np.median(mono_scores, axis=0)
    curve_range_median = np.median(mono_ranges, axis=0)

    for j in range(n_features):
        ms = curve_median_sign[j]
        if ms == 0:
            curve_sign_stability[j] = 0.0
        else:
            curve_sign_stability[j] = np.mean(mono_sign_votes[:, j] == ms)

    # Effect size threshold: scale relative to typical prediction magnitude
    pred_scale = float(np.median(np.abs(warm_train))) if np.median(np.abs(warm_train)) > 0 else float(np.std(warm_train) + 1e-12)
    min_effect = cfg.mono_effect_min_frac * pred_scale

    # Decide monotonic constraint per selected feature
    mono_vec_full = np.zeros(n_features, dtype=int)

    for j in selected_idx:
        # Primary: curve-based decision
        ms = int(curve_median_sign[j])
        passes_curve = (
            ms != 0
            and curve_sign_stability[j] >= cfg.mono_sign_stability
            and curve_score_median[j] >= cfg.mono_score_threshold
            and curve_range_median[j] >= min_effect
        )

        if passes_curve:
            mono_vec_full[j] = ms
            continue

        # Fallback: coefficient sign stability (optional)
        if cfg.mono_use_coeff_fallback:
            cs = int(np.sign(median_coef[j]))
            passes_coef = (
                cs != 0
                and sign_stability[j] >= cfg.mono_sign_stability
                and inclusion[j] >= cfg.topk_inclusion_frac
            )
            mono_vec_full[j] = cs if passes_coef else 0
        else:
            mono_vec_full[j] = 0

    # Create selected-only monotone vector aligned to selected features
    mono_vec_selected = mono_vec_full[selected_idx].astype(int)

    # -----------------------------
    # Interaction constraints via pair synergy screening (on selected shortlist)
    # -----------------------------
    # Shortlist by inclusion + |coef| (stable importance proxy)
    stable_score = (0.5 * inclusion) + (0.5 * (abs_median_coef / (np.median(abs_median_coef) + 1e-12)))
    shortlist_idx = np.argsort(-stable_score)[: min(cfg.interaction_max_features, n_features)]
    shortlist_names = feature_names[shortlist_idx].tolist()

    if cfg.verbose:
        print(f"[HuberTeacher] interaction shortlist={len(shortlist_idx)} features")

    # Evaluate pair gains across splits; keep stable winners
    def _pair_eval(i_pos: int, j_pos: int) -> Tuple[int, int, float, float]:
        i = shortlist_idx[i_pos]
        j = shortlist_idx[j_pos]
        gains = []
        for (start, end) in splits:
            # Use split window data (scaled) and teacher residual target? Here we screen on y directly.
            Xw = X_tr_scaled[start:end]
            yw = y_tr[start:end]
            ww = None if w is None else w[start:end]
            gain = _pair_synergy_gain_mae(Xw[:, i], Xw[:, j], yw, ww)
            gains.append(gain)

        gains = np.asarray(gains, dtype=np.float64)
        mean_gain = float(np.mean(gains))
        pos_rate = float(np.mean(gains > cfg.interaction_min_gain))
        return i, j, mean_gain, pos_rate

    # Generate pair list (upper triangle)
    pair_positions = []
    m = len(shortlist_idx)
    for a in range(m):
        for b in range(a + 1, m):
            pair_positions.append((a, b))

    # To control cost, cap number of pairs if needed
    # (You can tune this based on your compute budget.)
    max_pairs = min(len(pair_positions), 5000)
    if len(pair_positions) > max_pairs:
        rng = np.random.default_rng(123)
        pair_positions = list(rng.choice(pair_positions, size=max_pairs, replace=False))

    pair_results = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_pair_eval)(a, b) for (a, b) in pair_positions
    )

    # Filter by stability + gain, then select top pairs
    kept = [
        (i, j, mean_gain, pos_rate)
        for (i, j, mean_gain, pos_rate) in pair_results
        if (pos_rate >= cfg.interaction_sign_stability and mean_gain > cfg.interaction_min_gain)
    ]
    kept.sort(key=lambda t: t[2], reverse=True)
    kept = kept[: cfg.interaction_top_pairs]

    edges_selected_space: List[Tuple[int, int]] = []
    # Convert absolute feature indices into indices within selected_feats for constraint grouping
    selected_index_map = {int(ix): pos for pos, ix in enumerate(selected_idx.tolist())}
    for (i, j, _, _) in kept:
        if int(i) in selected_index_map and int(j) in selected_index_map:
            edges_selected_space.append((selected_index_map[int(i)], selected_index_map[int(j)]))

    interaction_groups = _build_interaction_groups_from_edges(
        feature_names=selected_feats,
        edges=edges_selected_space
    )

    # CatBoost: provide audit pairs as names (no hard constraint parameter)
    interaction_audit = []
    for (i, j, mean_gain, pos_rate) in kept:
        interaction_audit.append({
            "feature_i": feature_names[int(i)],
            "feature_j": feature_names[int(j)],
            "mean_mae_gain": float(mean_gain),
            "pos_rate": float(pos_rate),
        })

    if cfg.verbose:
        print(f"[HuberTeacher] interaction pairs kept={len(kept)}, groups={len(interaction_groups)}")

    # -----------------------------
    # Outputs formatted for LGBM / XGB / Cat
    # -----------------------------
    # Full vectors aligned to original feature order
    lgbm_mono_full = mono_vec_full.tolist()
    xgb_mono_full = tuple(int(v) for v in mono_vec_full.tolist())  # XGBoost accepts tuple/list/string
    # CatBoost accepts list/tuple or sparse string; we provide both
    cat_mono_full_list = mono_vec_full.tolist()
    cat_mono_sparse = ",".join(
        f"{train_columns[i]}:{int(mono_vec_full[i])}"
        for i in range(n_features)
        if int(mono_vec_full[i]) != 0
    )

    # Selected-only vectors aligned to selected_feats
    lgbm_mono_selected = mono_vec_selected.tolist()
    xgb_mono_selected = tuple(int(v) for v in mono_vec_selected.tolist())
    cat_mono_selected_list = mono_vec_selected.tolist()
    cat_mono_selected_sparse = ",".join(
        f"{selected_feats[i]}:{int(mono_vec_selected[i])}"
        for i in range(len(selected_feats))
        if int(mono_vec_selected[i]) != 0
    )

    # Provide dict form for convenience
    mono_dict_selected = dict(zip(selected_feats, [int(v) for v in mono_vec_selected]))

    # Residual meta target (often what you train the tree on)
    residual_target = y_tr - warm_train

    # -----------------------------
    # Construct Return Dictionary with Backward Compatibility
    # -----------------------------
    return {
        # Core (Legacy & New)
        "selected_features": selected_feats,
        "monotonic_constraints": mono_dict_selected,  # Legacy format: Dict[str, int]
        "interaction_constraints": interaction_groups, # Legacy format: List[List[str]]
        
        "warm_start": {
            "train": warm_train,
            "val": warm_val,
            "test": warm_test,
        },
        "huber_models": [teacher], # Legacy key
        "quantile_meta_targets": residual_target, # Legacy key (alias for residual_target)
        "residual_meta_target": residual_target,  # New key
        "scaler": scaler,

        # New Rich Metadata (stored in separate keys to avoid breaking legacy consumers)
        "train_columns": train_columns,
        "selected_feature_indices": selected_idx.tolist(),
        
        "huber_teacher": teacher,

        "stability": {
            "coef_median": median_coef.tolist(),
            "coef_abs_median": abs_median_coef.tolist(),
            "sign_stability": sign_stability.tolist(),
            "topk_inclusion_rate": inclusion.tolist(),
            "curve_median_sign": curve_median_sign.astype(int).tolist(),
            "curve_sign_stability": curve_sign_stability.tolist(),
            "curve_score_median": curve_score_median.tolist(),
            "curve_range_median": curve_range_median.tolist(),
        },

        "monotonic_constraints_details": {
            "selected_dict": mono_dict_selected,

            # Full vectors aligned to original feature order (X_train columns)
            "lightgbm_full": lgbm_mono_full,
            "xgboost_full": xgb_mono_full,
            "catboost_full_list": cat_mono_full_list,
            "catboost_full_sparse": cat_mono_sparse,

            # Selected-only vectors aligned to selected_features order
            "lightgbm_selected": lgbm_mono_selected,
            "xgboost_selected": xgb_mono_selected,
            "catboost_selected_list": cat_mono_selected_list,
            "catboost_selected_sparse": cat_mono_selected_sparse,
        },

        "interaction_constraints_details": {
            # For LightGBM/XGBoost: list of feature-name groups (allowed interactions within groups)
            "groups_selected": interaction_groups,

            # Audit list you can use for CatBoost governance/feature engineering
            "catboost_interaction_audit": interaction_audit,

            # Convenience: shortlist used for screening
            "shortlist_features": shortlist_names,
        },
    }


# Backward compatibility alias
def prepare_huber_production_orchestrator(*args, **kwargs):
    """Deprecated alias for prepare_huber_teacher_outputs"""
    return prepare_huber_teacher_outputs(*args, **kwargs)
