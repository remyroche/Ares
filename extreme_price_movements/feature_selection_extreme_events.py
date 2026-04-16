"""
Robust MDI Feature Selection with Quantile-Transformed Correlations v3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import importlib.util
import json
import os

from numba import jit, prange
import numpy as np

from .feature_views import get_feature_view
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.utils import check_random_state
from .utils import tprint, check_inf_nan, clean_dataset
from .sequential_bootstrap import get_ind_matrix, seq_bootstrap

# Optional LightGBM for early-stopping capable selectors
if importlib.util.find_spec("lightgbm") is not None:
    import lightgbm as lgb
else:
    lgb = None

# ======================================================================================
# Purged + Embargoed CV (time series)
# ======================================================================================

def purged_embargoed_splits(
    n_samples: int,
    n_splits: int,
    purge: int = 0,
    embargo: int = 0,
    min_train_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for _k in range(n_splits):
        val_start = start
        val_end = start + fold_sizes[_k]
        start = val_end

        val_idx = indices[val_start:val_end]

        train_end_1 = max(0, val_start - purge)
        train_idx_1 = indices[:train_end_1]

        train_start_2 = val_end + embargo
        train_idx_2 = indices[train_start_2:]

        train_idx = np.concatenate([train_idx_1, train_idx_2])

        if len(train_idx) == 0:
            continue
        if min_train_size is not None and len(train_idx) < min_train_size:
            continue

        splits.append((train_idx, val_idx))

    if not splits:
        # Warning instead of Error? No, CV fails without splits.
        # But for MDI, maybe we can just return what we have or fail gracefully.
        # For now, keep raising error but caller should handle it.
        raise ValueError("No valid splits produced (check min_train_size / purge).")
    return splits


# ======================================================================================
# Fast MDI Metrics
# ======================================================================================

@jit(nopython=True, nogil=True, cache=True)
def _numba_compute_mdi_metrics_tree(
    children_left, children_right, feature, weighted_n_node_samples,
    impurity, gains, root_n, depth_discount, eps,
    freq, mdi_depth, mdi_cov
):
    # Stack: (node_idx, depth)
    # We use a simple list as stack. Numba supports list of tuples if typed,
    # but separate lists are safer/easier in older Numba versions.
    # Actually, Numba handles list of tuples well in nopython mode now.

    stack_nodes = [0]
    stack_depths = [0]

    # Pre-allocate sizes? List append is supported.

    while len(stack_nodes) > 0:
        u = stack_nodes.pop()
        d = stack_depths.pop()

        # Leaf check: children_left[u] == -1
        if u == -1 or children_left[u] == -1:
            continue

        f = feature[u]
        if f < 0: continue

        delta = gains[u]
        node_n = weighted_n_node_samples[u]
        cov = node_n / (root_n + eps)

        freq[f] += 1.0
        # Cast to match array types if needed, but += handles it
        mdi_depth[f] += (depth_discount ** d) * delta
        mdi_cov[f] += cov * delta / (node_n + eps)

        # Push children (Right first for DFS left-first traversal order)
        # Right
        r = children_right[u]
        stack_nodes.append(r)
        stack_depths.append(d + 1)

        # Left
        l = children_left[u]
        stack_nodes.append(l)
        stack_depths.append(d + 1)

def extract_extra_mdi_metrics_fast(
    fitted_forest,
    n_features: int,
    depth_discount: float = 0.85,
    eps: float = 1e-12,
    compute_median: bool = False,
):
    freq = np.zeros(n_features, dtype=np.float32)
    mdi_depth = np.zeros(n_features, dtype=np.float32)
    mdi_cov = np.zeros(n_features, dtype=np.float32)
    gains_by_feature = defaultdict(list) if compute_median else None

    estimators = getattr(fitted_forest, "estimators_", None)
    if estimators is None:
        raise ValueError("Forest must be fitted and have estimators_.")

    for est in estimators:
        t = est.tree_
        weighted_n = t.weighted_n_node_samples.astype(np.float64, copy=False)
        impurity = t.impurity.astype(np.float64, copy=False)
        left, right, features = t.children_left, t.children_right, t.feature
        n_nodes = t.node_count
        is_split = left != -1

        # Vectorized Gain Calculation
        gains = np.zeros(n_nodes, dtype=np.float64)
        split_idx = np.flatnonzero(is_split)
        l, r = left[split_idx], right[split_idx]
        gains[split_idx] = weighted_n[split_idx] * impurity[split_idx] - (
            weighted_n[l] * impurity[l] + weighted_n[r] * impurity[r]
        )

        root_n = weighted_n[0] if n_nodes else 0.0

        if not compute_median:
            _numba_compute_mdi_metrics_tree(
                left, right, features, weighted_n, impurity, gains,
                root_n, depth_discount, eps,
                freq, mdi_depth, mdi_cov
            )
        else:
            stack = [(0, 0)] # (node, depth)

            while stack:
                u, d = stack.pop()
                if u == -1 or not is_split[u]: continue

                f = features[u]
                if f < 0: continue # Guard for robust leaf identification

                delta, node_n = gains[u], weighted_n[u]
                cov = node_n / (root_n + eps)

                freq[f] += 1.0
                mdi_depth[f] += np.float32((depth_discount ** d) * delta)
                mdi_cov[f] += np.float32(cov * (delta / (node_n + eps)))

                if compute_median:
                    gains_by_feature[f].append(float(delta))

                stack.append((left[u], d + 1))
                stack.append((right[u], d + 1))

    median_gain = np.zeros(n_features, dtype=np.float32)
    if compute_median:
        for f, gs in gains_by_feature.items():
            median_gain[f] = np.float32(np.median(gs))

    return freq, mdi_depth, mdi_cov, median_gain


def _is_xgb_model(model) -> bool:
    """Check if model is XGBoost (XGBRegressor/XGBClassifier)."""
    return xgb is not None and hasattr(model, "get_booster")


def _is_lgb_model(model) -> bool:
    """Check if model is LightGBM (LGBMRegressor/LGBMClassifier)."""
    return lgb is not None and hasattr(model, "booster_")


def extract_xgb_mdi_metrics(
    model,
    feature_names: List[str],
    depth_discount: float = 0.85,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract rich MDI metrics from XGBoost model trees.

    Returns (freq, mdi_depth, mdi_cov, gain) with same semantics as sklearn.
    """
    freq = np.zeros(len(feature_names), dtype=np.float32)
    mdi_depth = np.zeros(len(feature_names), dtype=np.float32)
    mdi_cov = np.zeros(len(feature_names), dtype=np.float32)
    gain_sum = np.zeros(len(feature_names), dtype=np.float32)

    booster = model.get_booster()
    # Get feature name mapping
    fmap = {f: i for i, f in enumerate(feature_names)}

    # Dump trees as JSON and parse
    try:
        trees_json = booster.get_dump(dump_format="json")
    except Exception:
        # Fallback to text dump
        trees_json = []

    if not trees_json:
        return freq, mdi_depth, mdi_cov, gain_sum

    import json

    for tree_str in trees_json:
        try:
            tree = json.loads(tree_str)
        except json.JSONDecodeError:
            continue

        # DFS traverse tree with (node, depth) stack
        stack = [(tree, 0)]
        # Get root samples for coverage calculation
        root_n = float(tree.get("cover", 1.0)) if "cover" in tree else 1.0

        while stack:
            node, depth = stack.pop()

            # Check if split node
            split_feat = node.get("split", None)
            if split_feat is None:
                continue  # Leaf

            # Map feature name to index
            feat_idx = fmap.get(split_feat, None)
            if feat_idx is None:
                # Try parsing as f{index} format
                if isinstance(split_feat, str) and split_feat.startswith("f"):
                    try:
                        feat_idx = int(split_feat[1:])
                        if feat_idx >= len(feature_names):
                            continue
                    except ValueError:
                        continue
                else:
                    continue

            # Get gain and cover
            gain = float(node.get("gain", 0.0))
            cover = float(node.get("cover", 1.0))

            freq[feat_idx] += 1.0
            mdi_depth[feat_idx] += np.float32((depth_discount ** depth) * gain)
            cov_weight = cover / (root_n + eps)
            mdi_cov[feat_idx] += np.float32(cov_weight * gain)
            gain_sum[feat_idx] += np.float32(gain)

            # Push children
            left = node.get("yes", None)
            right = node.get("no", None)
            missing = node.get("missing", None)

            if left is not None:
                stack.append((left, depth + 1))
            if right is not None and right != left:
                stack.append((right, depth + 1))

    return freq, mdi_depth, mdi_cov, gain_sum


def extract_lgb_mdi_metrics(
    model,
    feature_names: List[str],
    depth_discount: float = 0.85,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract rich MDI metrics from LightGBM model trees.

    Returns (freq, mdi_depth, mdi_cov, gain) with same semantics as sklearn.
    """
    freq = np.zeros(len(feature_names), dtype=np.float32)
    mdi_depth = np.zeros(len(feature_names), dtype=np.float32)
    mdi_cov = np.zeros(len(feature_names), dtype=np.float32)
    gain_sum = np.zeros(len(feature_names), dtype=np.float32)

    booster = model.booster_
    if booster is None:
        return freq, mdi_depth, mdi_cov, gain_sum

    # Get model dump
    try:
        tree_strs = booster.dump_model()["tree_info"]
    except Exception:
        return freq, mdi_depth, mdi_cov, gain_sum

    # Map feature names to indices
    fmap = {f: i for i, f in enumerate(feature_names)}

    for tree_info in tree_strs:
        tree = tree_info.get("tree_structure", {})
        if not tree:
            continue

        # Get root samples
        root_n = float(tree.get("internal_count", tree.get("leaf_count", 1.0)))

        # DFS traverse with (node_dict, depth) stack
        stack = [(tree, 0)]

        while stack:
            node, depth = stack.pop()

            # Check if internal node
            split_feat = node.get("split_feature", None)
            if split_feat is None:
                continue  # Leaf

            # Map feature to index
            if isinstance(split_feat, str):
                feat_idx = fmap.get(split_feat, None)
            else:
                feat_idx = int(split_feat)

            if feat_idx is None or feat_idx >= len(feature_names):
                continue

            # Get gain and count
            gain = float(node.get("split_gain", 0.0))
            count = float(node.get("internal_count", 1.0))

            freq[feat_idx] += 1.0
            mdi_depth[feat_idx] += np.float32((depth_discount ** depth) * gain)
            cov_weight = count / (root_n + eps)
            mdi_cov[feat_idx] += np.float32(cov_weight * gain)
            gain_sum[feat_idx] += np.float32(gain)

            # Push children
            left = node.get("left_child", None)
            right = node.get("right_child", None)

            if left is not None:
                stack.append((left, depth + 1))
            if right is not None:
                stack.append((right, depth + 1))

    return freq, mdi_depth, mdi_cov, gain_sum


def extract_gbdt_mdi_metrics(
    model,
    feature_names: List[str],
    depth_discount: float = 0.85,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified interface to extract MDI metrics from any GBDT model.

    Auto-detects XGB/LGB and delegates to appropriate extractor.
    Returns (freq, mdi_depth, mdi_cov, gain_sum).
    """
    if _is_xgb_model(model):
        return extract_xgb_mdi_metrics(model, feature_names, depth_discount, eps)
    elif _is_lgb_model(model):
        return extract_lgb_mdi_metrics(model, feature_names, depth_discount, eps)
    else:
        raise ValueError(f"Unknown GBDT model type: {type(model)}")


# ======================================================================================
# Main Production Pipeline
# ======================================================================================

def linear_prescreen_enet(
    X: pd.DataFrame,
    y: np.ndarray,
    n_select: int,
    multiplier: int = 4,
    max_drop_frac: float = 0.25,
    l1_ratio: float = 0.2,
    alpha_lo: float = 0.01,
    alpha_hi: float = 1e1,
    max_iter: int = 5000,
    tol: float = 1e-3,
    max_steps: int = 25,
    tol_frac: float = 0.15,
    random_state: int = 42,
    loss: str = "huber",
    huber_epsilon: float = 1.35,
) -> list[str]:
    """
    Drop-in ElasticNet-penalized linear pre-screen targeting keep-count ~= multiplier * n_select.

    Returns a list of feature names to keep (exactly target_keep if possible,
    otherwise the closest solution trimmed by |coef|).

    Notes:
    - Uses RobustScaler on X (handles outlier meta-features better than StandardScaler)
    - y is used as-is (caller is responsible for any target transform)
    - Searches alpha on a log scale to hit the target sparsity
    - Loss is configurable (default: Huber), with ElasticNet penalty via SGDRegressor
    """
    if X is None or X.empty:
        return []
    p = X.shape[1]
    target_keep_raw = int(np.clip(multiplier * int(n_select), 1, p))
    min_keep_from_cap = int(np.ceil((1.0 - float(np.clip(max_drop_frac, 0.0, 0.95))) * p))
    target_keep = int(np.clip(max(target_keep_raw, min_keep_from_cap), 1, p))

    y = np.asarray(y, dtype=float)
    # Scale target to unit variance for stable ElasticNet convergence
    y_std = float(np.nanstd(y))
    y_t = y / max(y_std, 1e-9)

    def fit_abscoef(alpha: float) -> np.ndarray:
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("enet", SGDRegressor(
                loss=loss,
                alpha=alpha,
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
                epsilon=huber_epsilon,
            ))
        ])
        pipe.fit(X, y_t)
        return np.abs(pipe.named_steps["enet"].coef_)

    # Count "selected" coefficients; pre-screen is ranking-based, so threshold is just for guidance.
    def nnz(abscoef: np.ndarray) -> int:
        return int(np.sum(abscoef > 1e-12))

    # Bracket alphas so that:
    # - alpha_lo keeps >= target_keep
    # - alpha_hi keeps <= target_keep (if possible)
    abs_lo = fit_abscoef(alpha_lo)
    n_lo = nnz(abs_lo)

    # If even very small alpha yields too few selected, fall back to top-|coef|
    if n_lo <= target_keep:
        idx = np.argsort(abs_lo)[::-1][:target_keep]
        return X.columns[idx].tolist()

    abs_hi = fit_abscoef(alpha_hi)
    n_hi = nnz(abs_hi)

    # Increase alpha_hi until we get <= target_keep (or give up)
    grow = 0
    while n_hi > target_keep and alpha_hi < 1e6 and grow < 20:
        alpha_hi *= 10.0
        abs_hi = fit_abscoef(alpha_hi)
        n_hi = nnz(abs_hi)
        grow += 1

    # If we can't sparsify enough, just take top-|coef| from the strongest-regularized fit we have
    if n_hi > target_keep:
        idx = np.argsort(abs_hi)[::-1][:target_keep]
        return X.columns[idx].tolist()

    # Binary search on log10(alpha)
    lo, hi = np.log10(alpha_lo), np.log10(alpha_hi)
    best_abs = abs_hi
    best_n = n_hi
    best_gap = abs(best_n - target_keep)

    for _ in range(max_steps):
        mid = (lo + hi) / 2.0
        alpha_mid = 10 ** mid
        abs_mid = fit_abscoef(alpha_mid)
        n_mid = nnz(abs_mid)

        gap = abs(n_mid - target_keep)
        if gap < best_gap:
            best_gap, best_abs, best_n = gap, abs_mid, n_mid

        # Close enough
        if gap <= max(1, int(target_keep * tol_frac)):
            best_abs, best_n = abs_mid, n_mid
            break

        # Too many kept => alpha too small => increase alpha
        if n_mid > target_keep:
            lo = mid
        else:
            hi = mid

    # Enforce exact keep-count by coefficient magnitude
    idx = np.argsort(best_abs)[::-1][:target_keep]
    return X.columns[idx].tolist()

def suggest_depth(p: int, n_samples: int) -> int:
    # Interaction target: log2(p) * 0.8
    # Ensure p > 0 to avoid log2(0)
    p = max(1, p)
    depth = int(np.clip(np.round(np.log2(p) * 0.8), 3, 8))
    if n_samples < 3000:
        depth = min(depth, 4)
    return depth

@dataclass
class MDISelectionResult:
    metrics_table: pd.DataFrame
    selected_features: List[str]
    kept_after_dedupe: List[str]
    top30_metric_table: Optional[pd.DataFrame] = None
    stability_table: Optional[pd.DataFrame] = None
    interaction_table: Optional[pd.DataFrame] = None
    final_score_table: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, Any]] = None


def _emit_mdi_noise_diagnostics(
    X_np: np.ndarray,
    y_np: np.ndarray,
    metrics_df_sorted: pd.DataFrame,
    mean_model_fit_score: float,
) -> None:
    """Log root-cause diagnostics when MDI importances collapse to near-zero."""
    # Option 1: Poor features
    feature_var = np.nanvar(X_np, axis=0)
    near_constant_ratio = float(np.mean(feature_var < 1e-8)) if feature_var.size else 1.0

    # Option 2: Poor importance measurement
    mean_share = float(np.nanmean(metrics_df_sorted.get("share_mu", np.array([0.0]))))
    mean_share_std = float(np.nanmean(metrics_df_sorted.get("share_std", np.array([0.0]))))
    high_noise_ratio = float(
        np.mean(
            metrics_df_sorted.get("share_std", pd.Series(dtype=float)).to_numpy()
            > metrics_df_sorted.get("share_mu", pd.Series(dtype=float)).to_numpy()
        )
    ) if len(metrics_df_sorted) else 1.0

    # Option 3: Poor target
    target_std = float(np.nanstd(y_np))
    target_iqr = float(np.nanpercentile(y_np, 75) - np.nanpercentile(y_np, 25))
    unique_target_ratio = float(np.unique(np.round(y_np, 8)).size / max(len(y_np), 1))

    # Option 4: Poor model fit (LGBM or any tree regressor used for MDI)
    tprint(
        "MDI diagnostics (near-zero effective mass): "
        f"features[near_const={near_constant_ratio:.1%}] | "
        f"importance[mean_share={mean_share:.2e}, mean_std={mean_share_std:.2e}, std>mu={high_noise_ratio:.1%}] | "
        f"target[std={target_std:.3e}, iqr={target_iqr:.3e}, unique_ratio={unique_target_ratio:.1%}] | "
        f"model_fit[oof_r2={mean_model_fit_score:.4f}]"
    )

    if near_constant_ratio > 0.35:
        tprint("MDI diagnostics: Feature quality risk — high near-constant feature share.")
    if high_noise_ratio > 0.7:
        tprint("MDI diagnostics: Importance measurement risk — fold-to-fold variance dominates mean importance.")
    if target_std < 1e-5 or target_iqr < 1e-5:
        tprint("MDI diagnostics: Target quality risk — target has very low dispersion.")
    if np.isfinite(mean_model_fit_score) and mean_model_fit_score < -0.05:
        tprint("MDI diagnostics: Model quality risk — negative OOF R² suggests model is mostly fitting noise.")


def _robust_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    lo = float(np.min(v)) if v.size else 0.0
    hi = float(np.max(v)) if v.size else 1.0
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo)


def _safe_spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 5:
        return 0.0
    aa = a[m]
    bb = b[m]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return 0.0
    ra = rankdata(aa, method="average")
    rb = rankdata(bb, method="average")
    aa0 = ra - float(np.mean(ra))
    bb0 = rb - float(np.mean(rb))
    den = float(np.sqrt(np.sum(aa0 * aa0) * np.sum(bb0 * bb0)))
    if den < 1e-12:
        return 0.0
    return float(np.sum(aa0 * bb0) / den)


def _subset_top_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    top_metric: str,
    top_frac: float,
    tail_q: float,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(m) < 8:
        return 0.0
    yt = y_true[m]
    yp = y_pred[m]
    n = len(yt)
    k = max(1, int(np.ceil(float(top_frac) * n)))
    idx_top = np.argpartition(yp, -k)[-k:]

    if top_metric == "top30_mean_utility":
        return float(np.mean(yt[idx_top])) if len(idx_top) else 0.0
    if target in {"classification", "binary", "clf"} or top_metric == "precision_top":
        y_bin = (yt >= 0.5).astype(float)
        return float(np.mean(y_bin[idx_top])) if len(idx_top) else 0.0
    if top_metric == "weighted_tail":
        q = float(np.nanquantile(yt, tail_q))
        w = np.clip(yt - q, 0.0, None)
        if np.sum(w) < 1e-12:
            return _safe_spearman_np(yp[idx_top], yt[idx_top])
        return float(np.sum(w[idx_top] * yt[idx_top]) / max(np.sum(w[idx_top]), 1e-12))
    # default regression/quantile contract: Spearman in top segment.
    return _safe_spearman_np(yp[idx_top], yt[idx_top])


def _restricted_permutation_importance(
    model,
    X_subset: np.ndarray,
    y_subset: np.ndarray,
    feature_idx: np.ndarray,
    metric_fn,
    rng: np.random.RandomState,
) -> np.ndarray:
    p = X_subset.shape[1]
    out = np.zeros(p, dtype=float)
    if X_subset.shape[0] < 8 or len(feature_idx) == 0:
        return out
    base_pred = model.predict(X_subset)
    base_score = float(metric_fn(y_subset, base_pred))
    xw = np.array(X_subset, copy=True)
    n = xw.shape[0]
    for j in feature_idx:
        perm_idx = rng.permutation(n)
        col = xw[:, j].copy()
        xw[:, j] = col[perm_idx]
        s = float(metric_fn(y_subset, model.predict(xw)))
        out[j] = max(0.0, base_score - s)
        xw[:, j] = col
    return out


def _extract_top_path_pair_scores(
    model,
    X_top: np.ndarray,
    feature_names: Sequence[str],
    rng: np.random.RandomState,
    max_trees: int = 64,
    max_rows: int = 512,
) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if X_top.shape[0] < 5 or not hasattr(model, "estimators_"):
        return out
    if X_top.shape[0] > max_rows:
        ridx = rng.choice(X_top.shape[0], size=max_rows, replace=False)
        X_top = X_top[ridx]
    ests = list(getattr(model, "estimators_", []) or [])
    if len(ests) == 0:
        return out
    if len(ests) > max_trees:
        pick = rng.choice(len(ests), size=max_trees, replace=False)
        ests = [ests[int(i)] for i in pick]

    feat_occ: Dict[str, float] = defaultdict(float)
    pair_occ: Dict[Tuple[str, str], float] = defaultdict(float)
    n_paths = 0.0
    for est in ests:
        try:
            path = est.decision_path(X_top)
            tree_feat = est.tree_.feature
        except Exception:
            continue
        for r in range(X_top.shape[0]):
            st = int(path.indptr[r])
            en = int(path.indptr[r + 1])
            nodes = path.indices[st:en]
            feats = sorted({int(tree_feat[n]) for n in nodes if int(tree_feat[n]) >= 0 and int(tree_feat[n]) < len(feature_names)})
            if not feats:
                continue
            n_paths += 1.0
            names = [feature_names[f] for f in feats]
            for a in names:
                feat_occ[a] += 1.0
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    pair_occ[(a, b)] += 1.0

    if n_paths < 1:
        return out
    for (a, b), c_ab in pair_occ.items():
        p_ab = c_ab / n_paths
        p_a = feat_occ[a] / n_paths
        p_b = feat_occ[b] / n_paths
        lift = p_ab / max(p_a * p_b, 1e-9)
        out[(a, b)] = float(max(lift, 0.0))
    return out


def _feature_family(name: str, family_map: Optional[Dict[str, str]] = None) -> str:
    if isinstance(family_map, dict) and name in family_map:
        return str(family_map[name])
    return str(name).split("_")[0]


def _apply_overlap_hysteresis(
    selected: List[str],
    ranked_all: List[str],
    score_map: Dict[str, float],
    anchors: Sequence[str],
    prev_selected: Sequence[str],
    min_overlap: float,
    margin: float,
) -> Tuple[List[str], Dict[str, float]]:
    sel = list(selected)
    prev = [f for f in prev_selected if f in ranked_all]
    if len(prev) == 0 or len(sel) == 0:
        return sel, {"overlap_before": 1.0, "overlap_after": 1.0, "swaps": 0.0}
    n = len(sel)
    anchors_set = set(anchors)
    overlap_before = len(set(sel) & set(prev)) / max(1, len(set(prev)))

    swaps = 0
    newcomers = [f for f in sel if f not in prev and f not in anchors_set]
    prev_out = [f for f in prev if f not in sel]
    newcomers = sorted(newcomers, key=lambda z: score_map.get(z, -1e18))
    prev_out = sorted(prev_out, key=lambda z: score_map.get(z, -1e18), reverse=True)
    for old in prev_out:
        if not newcomers:
            break
        new = newcomers[0]
        s_old = score_map.get(old, -1e18)
        s_new = score_map.get(new, -1e18)
        if s_new < s_old * (1.0 + float(margin)):
            sel.remove(new)
            if old not in sel:
                sel.append(old)
            newcomers.pop(0)
            swaps += 1

    overlap_now = len(set(sel) & set(prev)) / max(1, len(set(prev)))
    if overlap_now < float(min_overlap):
        need = int(np.ceil(float(min_overlap) * len(set(prev)))) - len(set(sel) & set(prev))
        for old in prev_out:
            if need <= 0:
                break
            if old in sel:
                continue
            drop_candidates = [x for x in sel if x not in anchors_set and x not in prev]
            if not drop_candidates:
                break
            drop = min(drop_candidates, key=lambda z: score_map.get(z, -1e18))
            sel.remove(drop)
            sel.append(old)
            need -= 1
            swaps += 1

    sel = [f for f in ranked_all if f in set(sel)][:n]
    overlap_after = len(set(sel) & set(prev)) / max(1, len(set(prev)))
    return sel, {"overlap_before": float(overlap_before), "overlap_after": float(overlap_after), "swaps": float(swaps)}

def mdi_feature_selection_v3(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    base_model=None,
    n_splits: int = 6,
    purge: int = 5,
    min_samples_leaf: int = 64,
    min_samples_leaf_pct: float = 0.015,
    min_impurity_decrease: float = 0.0,
    analysis_n_estimators: int = 120,
    analysis_max_samples: int = 3000,
    selector_max_missing_frac: float = 0.15,
    selector_near_constant_dominance: float = 0.999,

    pre_dedupe_threshold: float = 0.95, # Relaxed from 0.98 to 0.95 per plan
    random_state: int = 0,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    end_features: Optional[int] = None,
    cumulative_cap: float = 0.98,
    min_share: float = 0.001, # Threshold for noise
    min_features: int = 5,    # Hard floor
    max_features_pct: float = 0.5, # Hard ceiling (fraction of input)
    selector_target: str = "regression",
    selector_loss: Optional[str] = None,
    selector_alpha: Optional[float] = None,
    composite_weights: Optional[Dict[str, float]] = None,
    selector_y: Optional[Union[pd.Series, np.ndarray]] = None,
    candidate_cols: Optional[Sequence[str]] = None,
    cv_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    selector_head_name: Optional[str] = None,
    selector_focus_top_frac: float = 0.30,
    selector_top_metric: Optional[str] = None,
    selector_top_tail_quantile: float = 0.70,
    selector_frequency_hit_mode: str = "relative",
    selector_frequency_hit_quantile: float = 0.80,
    selector_frequency_hit_abs: float = 1e-6,
    selector_interaction_mode: str = "tree_path_lift",
    selector_interaction_topk_pairs: int = 100,
    selector_interaction_max_pairs_per_feature: int = 8,
    selector_interaction_corr_penalty: bool = True,
    selector_family_penalty: bool = True,
    selector_emit_report: bool = True,
    selector_report_dir: Optional[str] = None,
    selector_anchor_features: Optional[Sequence[str]] = None,
    selector_prev_selected: Optional[Sequence[str]] = None,
    selector_hysteresis_margin: float = 0.05,
    selector_min_overlap: float = 0.70,
    selector_family_map: Optional[Dict[str, str]] = None,
) -> MDISelectionResult:
    """
    Robust MDI Feature Selection with Quantile-Transformed Correlations v3.

    This function performs feature selection using Mean Decrease Impurity (MDI) from tree ensembles,
    robustified by Clustered Feature Importance (CFI) via hierarchical clustering or simpler
    deduplication, and recursive feature elimination (RFE).

    Algorithm Overview (MDI Feature Selection):
    1.  **Data Cleaning**: Rows with NaNs/Infs are dropped.
    2.  **Splitting**: Purged K-Fold Cross-Validation is used to respect time-series causality.
    3.  **Feature Deduplication (Pre-screening)**:
        -   Features are Quantile Transformed to Normal distribution.
        -   Greedy removal of highly correlated features (> 0.95 correlation) to reduce multicollinearity.
    4.  **ElasticNet Pre-screening**:
        -   If feature count is high (> 5 * end_features), L1+L2 regularization (ElasticNet) is used to
            quickly select a candidate subset (approx 5x the target count).
    5.  **Recursive Feature Elimination (RFE) Loop**:
        -   In each iteration, `ExtraTreesRegressor` is trained on CV folds.
        -   MDI metrics (Share, Frequency, Depth-weighted, Coverage-weighted) are extracted and aggregated.
        -   Features are ranked by a composite score of Stability-Weighted Importance (Stability = Mean * (HitRate / CV)).
        -   Bottom ~25% of features are dropped until `end_features` is reached.

    Final Feature Count Determination:
    -   The RFE loop aims to reach `end_features` (default: min(60, 1% of samples)).
    -   After the loop, the **Final Feature Selection** applies a "Cumulative Effective Importance" logic:
        1.  Calculate **Effective Importance** for each feature: `Imp_eff = Mean_Imp - 0.5 * Std_Imp`.
            This penalizes features with unstable importance across folds.
        2.  Normalize Effective Importance to sum to 1.0.
        3.  Accumulate features (sorted by rank) until the cumulative sum reaches `cumulative_cap` (default 0.98 or 98%).
        4.  This count is then clamped by `min_features` (floor) and `max_features_pct` (ceiling).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if candidate_cols is not None:
        candidate_cols = [c for c in candidate_cols if c in X.columns]
        if len(candidate_cols) == 0:
            return MDISelectionResult(pd.DataFrame(), [], [])
        X = X[list(candidate_cols)].copy()

    # Label artifacts and other internal bookkeeping columns must never reach
    # MDI. They are not predictive features and can cause leakage if exposed.
    leak_cols = [c for c in X.columns if str(c).startswith("__")]
    if leak_cols:
        tprint(f"MDI: Dropping {len(leak_cols)} internal columns before selection.")
        X = X.drop(columns=leak_cols, errors="ignore")
        if X.empty:
            return MDISelectionResult(pd.DataFrame(), [], [])

    # MDI is only defined on numeric feature blocks. Keep this filter early so
    # timestamp / identifier columns do not leak into variance checks or tree
    # fitting, which can otherwise surface as Timestamp arithmetic errors.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in X.columns if c not in set(numeric_cols)]
    if non_numeric_cols:
        tprint(
            f"MDI: Dropping {len(non_numeric_cols)} non-numeric features before selection."
        )
        X = X[numeric_cols].copy()
        if X.empty:
            return MDISelectionResult(pd.DataFrame(), [], [])

    missing_frac = X.isna().mean(axis=0)
    drop_missing_cols = missing_frac[missing_frac > float(selector_max_missing_frac)].index.tolist()
    if drop_missing_cols:
        tprint(
            f"MDI: Dropping {len(drop_missing_cols)} high-missingness features "
            f"(threshold={float(selector_max_missing_frac):.2f})."
        )
        X = X.drop(columns=drop_missing_cols, errors="ignore")
        if X.empty:
            return MDISelectionResult(pd.DataFrame(), [], [])

    dominant_rate = X.apply(
        lambda s: float(s.value_counts(dropna=False, normalize=True).iloc[0])
        if len(s) > 0 else 1.0,
        axis=0,
    )
    drop_dom_cols = dominant_rate[dominant_rate >= float(selector_near_constant_dominance)].index.tolist()
    if drop_dom_cols:
        tprint(
            f"MDI: Dropping {len(drop_dom_cols)} dominant-value features "
            f"(threshold={float(selector_near_constant_dominance):.3f})."
        )
        X = X.drop(columns=drop_dom_cols, errors="ignore")
        if X.empty:
            return MDISelectionResult(pd.DataFrame(), [], [])

    # Robust cleaning using shared utility.
    # Feature selection can optimize against caller-provided target (`selector_y`),
    # defaulting to `y` when not provided.
    y_for_selector = selector_y if selector_y is not None else y
    X, y_for_selector, sample_weight = clean_dataset(X, y_for_selector, sample_weight, name="X_mdi")

    if X.empty:
        tprint("MDI: X became empty after cleaning. Returning empty result.")
        return MDISelectionResult(pd.DataFrame(), [], [])

    near_const_std = np.nanstd(X.to_numpy(copy=False), axis=0)
    near_const_mask = near_const_std < 1e-9
    if near_const_mask.any():
        near_const_cols = X.columns[near_const_mask].tolist()
        interaction_cols = [
            c for c in near_const_cols if "_G_VOL_" in c or "_G_TREND_" in c
        ]
        base_cols = [c for c in near_const_cols if c not in interaction_cols]
        tprint(
            f"MDI: Dropping {len(near_const_cols)} near-constant features before dedupe "
            f"({len(base_cols)} base, {len(interaction_cols)} interaction)."
        )
        X = X.drop(columns=near_const_cols, errors="ignore")
        if X.empty:
            tprint("MDI: X became empty after near-constant pruning. Returning empty result.")
            return MDISelectionResult(pd.DataFrame(), [], [])

    # Determine end_features if not provided
    if end_features is None:
        n_samples_full = len(X)
        end_features = min(60, max(1, n_samples_full // 100))
        # Ensure we target at least the floor
        end_features = max(end_features, min_features)
        tprint(f"MDI: end_features not provided. Auto-setting to {end_features} based on sample size {n_samples_full} and min_features {min_features}")
    elif end_features < min_features:
        tprint(f"MDI: end_features {end_features} < min_features {min_features}. Boosting end_features.")
        end_features = min_features

    if base_model is None:
        default_leaf = max(int(min_samples_leaf), int(np.ceil(min_samples_leaf_pct * len(X))))
        base_model = ExtraTreesRegressor(
            n_estimators=int(analysis_n_estimators),
            max_depth=None,
            min_samples_leaf=default_leaf,
            min_samples_split=max(2, 3 * default_leaf),
            max_features='sqrt',
            n_jobs=2,
            random_state=42
        )

    rng = check_random_state(random_state)

    # Fast initial conversion
    # Ensure we catch overflow here too
    try:
        X_np_full = np.ascontiguousarray(X.values, dtype=np.float32)
    except Exception as e:
        tprint(f"MDI: Error converting X to float32: {e}")
        # Fallback to float64 but ExtraTrees might be slower or it might be needed for precision
        # But if it failed, likely string/object. clean_dataset should have caught it.
        # If valid float64 > float32 max, it converts to inf.
        X_np_full = np.ascontiguousarray(X.values, dtype=np.float32) # will produce infs

    # SAFETY CHECK: Conversion to float32 might have introduced Inf/NaN (overflow)
    finite_rows = np.isfinite(X_np_full).all(axis=1)
    if not finite_rows.all():
        n_dropped = (~finite_rows).sum()
        tprint(f"MDI: Found non-finite values after float32 conversion (likely overflow). Dropping {n_dropped} rows.")
        X_np_full = X_np_full[finite_rows]

        # Sync X dataframe
        X = X.iloc[finite_rows]

        if hasattr(y_for_selector, 'values'): y_vals = y_for_selector.values
        else: y_vals = np.asarray(y_for_selector)
        y_np = y_vals[finite_rows]

        if sample_weight is not None:
            if hasattr(sample_weight, 'values'): sw_vals = sample_weight.values
            else: sw_vals = np.asarray(sample_weight)
            sw_np = sw_vals[finite_rows]
        else:
            sw_np = None
    else:
        y_np = y_for_selector.values if hasattr(y_for_selector, 'values') else np.asarray(y_for_selector)
        sw_np = None
        if sample_weight is not None:
            sw_np = sample_weight.to_numpy() if isinstance(sample_weight, pd.Series) else np.asarray(sample_weight)

    # Update N after potentially dropping rows
    N = X_np_full.shape[0]

    feature_names_full = list(X.columns)
    name_to_idx = {name: i for i, name in enumerate(feature_names_full)}

    # Check if empty again
    if N < 5:
        tprint("MDI: Too few samples after cleaning. Returning empty result.")
        return MDISelectionResult(pd.DataFrame(), [], [])

    # Initial Splits for Dedupe
    if cv_splits is not None:
        splits_full = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in cv_splits]
    else:
        try:
            splits_full = purged_embargoed_splits(N, n_splits, purge=purge)
        except ValueError as e:
            tprint(f"MDI: Split generation failed: {e}")
            return MDISelectionResult(pd.DataFrame(), [], [])

    # 1. Anchored Pre-Dedupe (Train Window 0)
    train_idx0, _ = splits_full[0]

    # Only run dedupe if we have data
    if len(train_idx0) > 0:
        X_tr0 = X_np_full[train_idx0]

        # Quantile Transform for Robust Pearson
        # Optimize: Limit quantiles and subsample
        qt_subsample = min(len(X_tr0), 100000)
        qt_quantiles = min(len(X_tr0), 256)
        qt = QuantileTransformer(n_quantiles=qt_quantiles, output_distribution='normal', random_state=random_state, subsample=qt_subsample)
        try:
            X_tr0_qt = qt.fit_transform(X_tr0)
        except Exception as e:
            tprint(f"MDI: QuantileTransformer failed: {e}")
            return MDISelectionResult(pd.DataFrame(), [], [])

        # Greedy Streaming Dedupe without full matrix
        x_mean = X_tr0_qt.mean(axis=0)
        x_std = X_tr0_qt.std(axis=0)
        x_std[x_std == 0] = 1.0
        X_z = (X_tr0_qt - x_mean) / x_std

        n_tr0 = X_z.shape[0]
        keep_mask = np.ones(len(feature_names_full), dtype=bool)
        CHUNK_SIZE = 1000

        for i in range(len(feature_names_full)):
            if not keep_mask[i]: continue

            # Candidates j > i
            candidates = np.where(keep_mask)[0]
            candidates = candidates[candidates > i]

            if len(candidates) == 0:
                continue

            v_i = X_z[:, i]

            for k in range(0, len(candidates), CHUNK_SIZE):
                chunk_indices = candidates[k:k+CHUNK_SIZE]
                chunk_data = X_z[:, chunk_indices]

                # Dot product (correlation)
                corrs = v_i @ chunk_data / n_tr0

                redundant_local = np.abs(corrs) >= pre_dedupe_threshold

                if redundant_local.any():
                    drop_indices = chunk_indices[redundant_local]
                    keep_mask[drop_indices] = False
    else:
        keep_mask = np.ones(len(feature_names_full), dtype=bool)

    kept_features = [feature_names_full[i] for i in range(len(keep_mask)) if keep_mask[i]]
    kept_features_indices = [i for i, kept in enumerate(keep_mask) if kept]

    tprint(f"MDI: Starting with {len(kept_features)} features after dedupe (target: {end_features}).")

    current_features = kept_features

    _sel_target = str(selector_target).lower().strip()
    _sel_loss = selector_loss
    _sel_alpha = selector_alpha
    if _sel_loss is None:
        # Caller-provided target controls default selector loss.
        _sel_loss = "huber" if _sel_target in {"classification", "binary", "clf"} else "huber"

    if _sel_target == "quantile":
        assert _sel_alpha is not None, "selector_alpha must be provided when selector_target='quantile'"

    # Composite ranking weights can be caller-provided, otherwise adapt by selector semantics.
    # Keys: share, depth, cov; values must sum > 0.
    _w_default = {"share": 0.50, "depth": 0.30, "cov": 0.20}
    _loss_hint = str(_sel_loss or "").lower()
    if _sel_target in {"classification", "binary", "clf"}:
        _w_default = {"share": 0.60, "depth": 0.20, "cov": 0.20}
    elif _sel_target == "quantile" or "quantile" in _loss_hint:
        _w_default = {"share": 0.35, "depth": 0.25, "cov": 0.40}
    elif "huber" in _loss_hint:
        _w_default = {"share": 0.45, "depth": 0.35, "cov": 0.20}
    _w = dict(_w_default)
    if isinstance(composite_weights, dict):
        for _k in ("share", "depth", "cov"):
            if _k in composite_weights:
                try:
                    _w[_k] = float(composite_weights[_k])
                except Exception:
                    pass
    _sum_w = max(float(_w["share"] + _w["depth"] + _w["cov"]), 1e-12)
    _w = {k: float(v) / _sum_w for k, v in _w.items()}

    _head_name = str(selector_head_name or "default")
    _target_name = str(_sel_target).lower()
    _top_frac = float(np.clip(selector_focus_top_frac, 0.05, 0.80))
    _top_metric = str(selector_top_metric).lower().strip() if selector_top_metric else ""
    if not _top_metric:
        _top_metric = "precision_top" if _target_name in {"classification", "binary", "clf"} else "ic_top"
    _tail_q = float(np.clip(selector_top_tail_quantile, 0.50, 0.99))
    _is_utility_head = ("utility" in _head_name.lower()) or (_top_metric == "top30_mean_utility")

    # Final selection weights (phase-B defaults + utility conservative override)
    _final_w = {
        "top30": 0.35,
        "global": 0.20,
        "stability": 0.25,
        "frequency": 0.15,
        "interaction": 0.05,
    }
    if _is_utility_head:
        _final_w = {
            "top30": 0.30,
            "global": 0.15,
            "stability": 0.30,
            "frequency": 0.20,
            "interaction": 0.05,
        }
    if isinstance(composite_weights, dict):
        if any(k in composite_weights for k in _final_w):
            for k in _final_w:
                if k in composite_weights:
                    try:
                        _final_w[k] = float(composite_weights[k])
                    except Exception:
                        pass
            _sw = max(sum(_final_w.values()), 1e-12)
            _final_w = {k: float(v) / _sw for k, v in _final_w.items()}

    if len(current_features) > 4 * end_features:
        # SGDRegressor only accepts regression losses ('huber', 'squared_error', etc.)
        # If the outer selector target is classification, we use huber as a robust linear proxy.
        _prescreen_loss = str(_sel_loss).lower()
        if _prescreen_loss in {"binary_logloss", "log_loss", "cross_entropy", "binary"}:
            _prescreen_loss = "huber"
        if _sel_target in {"classification", "binary", "clf"}:
            _prescreen_l1_ratio = 0.65
            _prescreen_alpha_lo = 1e-4
            _prescreen_alpha_hi = 5.0
        elif _sel_target == "quantile" or "quantile" in str(_sel_loss).lower():
            _prescreen_l1_ratio = 0.45
            _prescreen_alpha_lo = 5e-4
            _prescreen_alpha_hi = 10.0
        else:
            _prescreen_l1_ratio = 0.35
            _prescreen_alpha_lo = 1e-4
            _prescreen_alpha_hi = 10.0
            
        prescreened_features = linear_prescreen_enet(
            X[current_features],
            y_np,
            n_select=end_features,
            multiplier=5,
            max_drop_frac=0.15,
            l1_ratio=_prescreen_l1_ratio,
            alpha_lo=_prescreen_alpha_lo,
            alpha_hi=_prescreen_alpha_hi,
            loss=_prescreen_loss,
        )
        tprint(f"MDI: ElasticNet prescreen reduced features from {len(current_features)} to {len(prescreened_features)}")
        current_features = prescreened_features

    metrics_df_sorted = pd.DataFrame()
    mean_model_fit_score = float("nan")
    last_fold_share: List[np.ndarray] = []
    last_fold_top_imp: List[np.ndarray] = []
    last_fold_rest_imp: List[np.ndarray] = []
    last_pair_scores: List[Dict[Tuple[str, str], float]] = []
    last_top_metrics: List[Dict[str, float]] = []
    last_features: List[str] = []
    last_metrics_df: Optional[pd.DataFrame] = None

    # Cache supported params once
    base_params = base_model.get_params() if hasattr(base_model, 'get_params') else {}
    supported_params = set(base_params.keys())

    while True:
        p = len(current_features)

        # Subsampling Logic - Limit to 5K events max for MDI
        n_star = min(
            int(max(256, analysis_max_samples)),
            max(1200, 150 * p, 80 * end_features),
        )

        if n_star < N:
             # Systematic sampling
             indices_sub = np.linspace(0, N-1, n_star, dtype=int)

             feat_indices = [name_to_idx[f] for f in current_features]
             X_curr_np = X_np_full[indices_sub][:, feat_indices]
             y_curr_np = y_np[indices_sub]
             sw_curr_np = sw_np[indices_sub] if sw_np is not None else None

             try:
                 splits_curr = purged_embargoed_splits(len(indices_sub), n_splits, purge=purge)
             except ValueError:
                 splits_curr = []
        else:
            feat_indices = [name_to_idx[f] for f in current_features]
            X_curr_np = X_np_full[:, feat_indices]
            y_curr_np = y_np
            sw_curr_np = sw_np
            splits_curr = splits_full

        if not splits_curr:
             tprint("MDI: No splits for current subsample. Breaking.")
             break
        
        import gc
        gc.collect()

        depth = suggest_depth(p, X_curr_np.shape[0])

        # Streaming Aggregation
        sums = {k: np.zeros(p, dtype=np.float64) for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        sq_sums = {k: np.zeros(p, dtype=np.float64) for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        counts = {k: 0 for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        pos_counts = {k: np.zeros(p, dtype=np.float64) for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        fold_share_curr: List[np.ndarray] = []
        fold_top_imp_curr: List[np.ndarray] = []
        fold_rest_imp_curr: List[np.ndarray] = []
        fold_pair_curr: List[Dict[Tuple[str, str], float]] = []
        fold_top_metrics_curr: List[Dict[str, float]] = []

        valid_folds = 0
        fold_fit_scores: List[float] = []

        for train_idx, val_idx in splits_curr:
            m = clone(base_model)
            params = {}
            if "max_depth" in supported_params:
                # For LGBM-style models keep tree depth strictly below 6.
                if "min_data_in_leaf" in supported_params:
                    params["max_depth"] = min(max(2, depth), 5)
                else:
                    params["max_depth"] = depth
            if "n_estimators" in supported_params: params["n_estimators"] = analysis_n_estimators

            train_n = max(1, int(len(train_idx)))
            leaf_samples = max(int(min_samples_leaf), int(np.ceil(min_samples_leaf_pct * train_n)))
            if "min_samples_leaf" in supported_params: params["min_samples_leaf"] = leaf_samples
            if "min_samples_split" in supported_params: params["min_samples_split"] = max(2, 3 * leaf_samples)

            if "min_data_in_leaf" in supported_params:
                params["min_data_in_leaf"] = max(2, int(np.ceil(0.012 * train_n)))
            if "min_data_in_bin" in supported_params:
                params["min_data_in_bin"] = 127
            if "min_gain_to_split" in supported_params:
                base_gain = base_params.get("min_gain_to_split", 0.0)
                params["min_gain_to_split"] = float(base_gain) if float(base_gain) > 0 else 0.0
            if "feature_fraction" in supported_params:
                ff = float(base_params.get("feature_fraction", 0.7))
                params["feature_fraction"] = ff if 0.6 <= ff <= 0.9 else 0.7
            if "bagging_fraction" in supported_params:
                bf = float(base_params.get("bagging_fraction", 0.7))
                params["bagging_fraction"] = bf if 0.6 <= bf <= 0.9 else 0.7

            if "min_impurity_decrease" in supported_params: params["min_impurity_decrease"] = min_impurity_decrease
            if "random_state" in supported_params: params["random_state"] = random_state
            m.set_params(**params)

            sw_tr = sw_curr_np[train_idx] if sw_curr_np is not None else None
            is_lgbm = (m.__class__.__module__.startswith("lightgbm") and lgb is not None)
            if is_lgbm and len(val_idx) > 0:
                # Objective/loss should be caller-driven when provided.
                _obj = str(getattr(m, "objective", None) or base_params.get("objective", "regression")).lower()
                _loss_hint = str(_sel_loss).lower()
                if _sel_target == "quantile":
                    # Enforce quantile objective/metric invariants for quantile intent.
                    if "objective" in supported_params:
                        params["objective"] = "quantile"
                    if "alpha" in supported_params and _sel_alpha is not None:
                        params["alpha"] = float(_sel_alpha)
                    m.set_params(**params)
                if "quantile" in _loss_hint or "quantile" in _obj:
                    _eval_metric = "quantile"
                elif "huber" in _loss_hint or "huber" in _obj:
                    _eval_metric = "huber"
                elif _loss_hint in {"absolute_error", "l1", "epsilon_insensitive"} or "l1" in _obj:
                    _eval_metric = "l1"
                elif _sel_target in {"classification", "binary", "clf"}:
                    _eval_metric = "binary_logloss"
                else:
                    _eval_metric = "l2"
                m.fit(
                    X_curr_np[train_idx],
                    y_curr_np[train_idx],
                    sample_weight=sw_tr,
                    eval_set=[(X_curr_np[val_idx], y_curr_np[val_idx])],
                    eval_metric=_eval_metric,
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            else:
                m.fit(X_curr_np[train_idx], y_curr_np[train_idx], sample_weight=sw_tr)

            if len(val_idx) > 1:
                y_val = y_curr_np[val_idx]
                y_pred = m.predict(X_curr_np[val_idx])
                ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
                if ss_tot > 1e-12:
                    ss_res = float(np.sum((y_val - y_pred) ** 2))
                    fold_fit_scores.append(1.0 - ss_res / ss_tot)

            # C-level raw importance
            # folds_data['share'].append(m.feature_importances_)
            share = m.feature_importances_.astype(np.float64)

            # Fast extra metrics - support sklearn forests, XGB, and LGB
            if hasattr(m, "estimators_"):
                # Sklearn RandomForest/ExtraTrees
                freq, mdi_d, mdi_c, _ = extract_extra_mdi_metrics_fast(m, p)
            elif _is_xgb_model(m) or _is_lgb_model(m):
                # XGBoost or LightGBM - extract full tree structure metrics
                try:
                    freq, mdi_d, mdi_c, gain_sum = extract_gbdt_mdi_metrics(
                        m, current_features, depth_discount=0.85, eps=1e-12
                    )
                    # Use gain_sum as share (more accurate than built-in for GBDT)
                    share = gain_sum.astype(np.float64)
                except Exception as _exc:
                    # Fallback to built-in importances
                    freq = np.zeros(p, dtype=np.float64)
                    mdi_d = np.zeros(p, dtype=np.float64)
                    mdi_c = np.zeros(p, dtype=np.float64)
            else:
                freq = np.zeros(p, dtype=np.float64)
                mdi_d = np.zeros(p, dtype=np.float64)
                mdi_c = np.zeros(p, dtype=np.float64)

            # Ensure float64
            freq = freq.astype(np.float64)
            mdi_d = mdi_d.astype(np.float64)
            mdi_c = mdi_c.astype(np.float64)

            vals = {'share': share, 'freq': freq, 'mdi_depth': mdi_d, 'mdi_cov': mdi_c}

            for k in vals:
                v = vals[k]
                sums[k] += v
                sq_sums[k] += v*v
                pos_counts[k] += (v > 0).astype(np.float64)
                counts[k] += 1

            fold_share_curr.append(share.copy())
            if len(val_idx) >= 12:
                X_val = X_curr_np[val_idx]
                y_val = y_curr_np[val_idx]
                y_pred_val = m.predict(X_val)
                n_top = max(3, int(np.ceil(_top_frac * len(val_idx))))
                idx_top = np.argpartition(y_pred_val, -n_top)[-n_top:]
                idx_rest = np.setdiff1d(np.arange(len(val_idx)), idx_top, assume_unique=False)
                _metric_fn = lambda yy, pp: _subset_top_metric(
                    yy, pp, _target_name, _top_metric, _top_frac, _tail_q
                )
                top_metric_val = _subset_top_metric(y_val, y_pred_val, _target_name, _top_metric, _top_frac, _tail_q)
                fold_top_metrics_curr.append({"metric": float(top_metric_val), "n_top": float(len(idx_top))})

                _perm_topk = min(max(16, int(np.sqrt(p) * 4)), p)
                _cand = np.argsort(share)[-_perm_topk:]
                _rng_fold = np.random.RandomState(int(random_state + valid_folds + p))
                imp_top = _restricted_permutation_importance(
                    m, X_val[idx_top], y_val[idx_top], _cand, _metric_fn, _rng_fold
                )
                if len(idx_rest) >= 8:
                    imp_rest = _restricted_permutation_importance(
                        m, X_val[idx_rest], y_val[idx_rest], _cand, _metric_fn, _rng_fold
                    )
                else:
                    imp_rest = np.zeros(p, dtype=float)
                fold_top_imp_curr.append(np.asarray(imp_top, dtype=float))
                fold_rest_imp_curr.append(np.asarray(imp_rest, dtype=float))

                if str(selector_interaction_mode).lower() != "off":
                    pair_score = _extract_top_path_pair_scores(
                        m,
                        X_val[idx_top],
                        current_features,
                        _rng_fold,
                        max_trees=64,
                    )
                    fold_pair_curr.append(pair_score)

            valid_folds += 1

        if valid_folds == 0:
            break

        mean_model_fit_score = float(np.mean(fold_fit_scores)) if fold_fit_scores else float("nan")

        # 3. Aggregation
        agg = {}
        for metric in sums:
            n = counts[metric]
            mu = sums[metric] / n
            var = (sq_sums[metric] / n) - (mu * mu)
            sd = np.sqrt(np.maximum(var, 0))
            hit = pos_counts[metric] / n

            cv = sd / (mu + 1e-12)
            stab = mu * (hit / (cv + 1e-12))

            # Use float32
            agg[f"{metric}_mu"] = mu.astype(np.float32)
            agg[f"{metric}_std"] = sd.astype(np.float32) # Added for Effective Importance
            agg[f"{metric}_stab"] = stab.astype(np.float32)

        metrics_df = pd.DataFrame(agg, index=current_features)

        # Final Ranking
        metrics_df['composite_rank'] = (
            metrics_df['share_stab'].rank(ascending=False) * float(_w["share"]) +
            metrics_df['mdi_depth_stab'].rank(ascending=False) * float(_w["depth"]) +
            metrics_df['mdi_cov_stab'].rank(ascending=False) * float(_w["cov"])
        ).rank()

        metrics_df_sorted = metrics_df.sort_values('composite_rank')
        last_fold_share = fold_share_curr
        last_fold_top_imp = fold_top_imp_curr
        last_fold_rest_imp = fold_rest_imp_curr
        last_pair_scores = fold_pair_curr
        last_top_metrics = fold_top_metrics_curr
        last_features = list(current_features)
        last_metrics_df = metrics_df.copy()

        # If we reached the target, we are done
        # RFE: Adaptive prune rate with target-aware linear decay + fixed bonus drop.
        if p <= end_features:
            tprint(f"MDI RFE: Reached target {p} features. Stopping.")
            break

        # Adaptive RFE drop schedule:
        # linearly decay drop rate from 50% (at p = 6x target) to 15% (at p = 1x target),
        # then add a fixed +15 bonus drop (bounded by target floor).
        _ratio = float(p) / max(float(end_features), 1.0)
        _ratio_c = float(np.clip(_ratio, 1.0, 6.0))
        _drop_rate = 0.15 + ((_ratio_c - 1.0) * (0.50 - 0.15) / (6.0 - 1.0))
        _bonus_drop = 15
        n_to_drop = int(np.floor(float(p) * _drop_rate)) + _bonus_drop
        n_to_drop = max(1, n_to_drop)
        if p - n_to_drop < end_features:
            n_to_drop = max(0, p - end_features)
        
        if n_to_drop == 0:
            tprint("MDI: Targeted feature count reached.")
            break
            
        tprint(
            f"MDI: Dropping {n_to_drop} features (remaining: {p - n_to_drop}) "
            f"[adaptive rate={_drop_rate:.3f}, p/target={_ratio:.2f}, bonus=+{_bonus_drop}]"
        )
        current_features = metrics_df_sorted.index[:-n_to_drop].tolist()
        
        # Memory cleanup after each RFE iteration
        del X_curr_np
        gc.collect()

    if last_metrics_df is None or last_metrics_df.empty:
        return MDISelectionResult(pd.DataFrame(), [], kept_features)

    metrics_df = last_metrics_df.copy()
    feat_order = list(metrics_df.index)
    p_last = len(feat_order)

    share_mu = metrics_df["share_mu"].to_numpy(dtype=float)
    share_std = metrics_df["share_std"].to_numpy(dtype=float)
    share_eff = np.maximum(0.0, share_mu - 0.5 * share_std)
    total_eff = float(np.sum(share_eff))
    if total_eff <= 1e-12:
        tprint("MDI: Effective importance is zero (high noise). Falling back to share_mu.")
        _emit_mdi_noise_diagnostics(X_np_full, y_np, metrics_df, mean_model_fit_score)
        share_eff = np.maximum(0.0, share_mu)
        total_eff = max(float(np.sum(share_eff)), 1e-12)
    share_norm = share_eff / total_eff
    cumsum = np.cumsum(share_norm)
    cutoff_idx = int(np.searchsorted(cumsum, cumulative_cap))
    cutoff_idx = min(cutoff_idx, len(metrics_df) - 1)
    n_selected_cap = cutoff_idx + 1
    n_total = len(metrics_df)
    n_max_hard = min(max(int(n_total * max_features_pct), min_features), n_total)
    n_min_hard = min(min_features, n_total)
    n_final = min(max(n_selected_cap, n_min_hard), n_max_hard)

    # Fold-level ranking stability and thresholded frequency
    if len(last_fold_share) > 0:
        fs = np.vstack(last_fold_share)
        ranks = np.argsort(np.argsort(-fs, axis=1), axis=1) + 1
        med_rank = np.median(ranks, axis=0)
        mad_rank = np.mean(np.abs(ranks - med_rank), axis=0)
        stability_score = np.clip(1.0 - (mad_rank / max(float(np.max(mad_rank)), 1e-12)), 0.0, 1.0)

        hits = np.zeros(p_last, dtype=float)
        for row in fs:
            row = np.asarray(row, dtype=float)
            nz = row[row > 0]
            thr_rel = float(np.quantile(nz, selector_frequency_hit_quantile)) if nz.size > 0 else np.inf
            thr_abs = float(selector_frequency_hit_abs) * max(float(np.sum(np.abs(row))), 1e-12)
            thr = max(thr_rel, thr_abs) if str(selector_frequency_hit_mode).lower() == "relative" else thr_abs
            if not np.isfinite(thr):
                continue
            hits += (row >= thr).astype(float)
        frequency_score = np.clip(hits / max(float(len(last_fold_share)), 1.0), 0.0, 1.0)
    else:
        med_rank = np.full(p_last, np.nan, dtype=float)
        mad_rank = np.full(p_last, np.nan, dtype=float)
        stability_score = np.zeros(p_last, dtype=float)
        frequency_score = np.zeros(p_last, dtype=float)

    # Top30 attribution
    if len(last_fold_top_imp) > 0:
        imp_top = np.vstack(last_fold_top_imp)
        imp_rest = np.vstack(last_fold_rest_imp) if len(last_fold_rest_imp) == len(last_fold_top_imp) else np.zeros_like(imp_top)
        top_support_raw = np.median(imp_top, axis=0)
        top_lift_raw = np.median(imp_top - imp_rest, axis=0)
    else:
        top_support_raw = np.zeros(p_last, dtype=float)
        top_lift_raw = np.zeros(p_last, dtype=float)

    # Interaction lift aggregation
    pair_agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for d in last_pair_scores:
        for k, v in d.items():
            if np.isfinite(v):
                pair_agg[k].append(float(v))
    pair_rows = []
    interaction_support_raw = np.zeros(p_last, dtype=float)
    idx_map = {f: i for i, f in enumerate(feat_order)}
    # Build cheap correlation proxy once on top candidate interaction features.
    base_pairs = [((a, b), float(np.mean(vs))) for (a, b), vs in pair_agg.items() if len(vs) > 0]
    base_pairs = sorted(base_pairs, key=lambda z: z[1], reverse=True)
    pre_n = max(int(selector_interaction_topk_pairs) * 3, int(selector_interaction_topk_pairs), 64)
    base_pairs = base_pairs[:pre_n]
    corr_lookup: Dict[Tuple[str, str], float] = {}
    if selector_interaction_corr_penalty and len(base_pairs) > 0:
        top_feats = sorted({f for (ab, _) in base_pairs for f in ab if f in idx_map})
        max_corr_feats = max(32, min(512, int(selector_interaction_topk_pairs) * 4))
        top_feats = top_feats[:max_corr_feats]
        if len(top_feats) >= 2:
            cols = [name_to_idx[f] for f in top_feats]
            X_corr = np.nan_to_num(X_np_full[:, cols], nan=0.0)
            cmat = np.corrcoef(X_corr, rowvar=False)
            for i, fa in enumerate(top_feats):
                for j in range(i + 1, len(top_feats)):
                    fb = top_feats[j]
                    c = float(cmat[i, j]) if np.isfinite(cmat[i, j]) else 0.0
                    corr_lookup[(fa, fb)] = c
                    corr_lookup[(fb, fa)] = c

    for (a, b), lift_mean in base_pairs:
        score = float(lift_mean)
        if selector_family_penalty and _feature_family(a, selector_family_map) == _feature_family(b, selector_family_map):
            score *= 0.8
        if selector_interaction_corr_penalty:
            c = float(corr_lookup.get((a, b), 0.0))
            if abs(c) > 0.85:
                score *= 0.7
        pair_rows.append({"feature_a": a, "feature_b": b, "lift": float(lift_mean), "final_pair_score": score})

    if len(pair_rows) > 0:
        pair_df = pd.DataFrame(pair_rows).sort_values("final_pair_score", ascending=False)
        if selector_interaction_topk_pairs > 0:
            pair_df = pair_df.head(int(selector_interaction_topk_pairs)).copy()
        feat_pair_scores: Dict[str, List[float]] = defaultdict(list)

        # Replace O(N) Pandas iterrows with much faster direct array iteration
        f_a_arr = pair_df["feature_a"].to_numpy()
        f_b_arr = pair_df["feature_b"].to_numpy()
        s_arr = pair_df["final_pair_score"].to_numpy(dtype=float)

        for a, b, score in zip(f_a_arr, f_b_arr, s_arr):
            feat_pair_scores[str(a)].append(float(score))
            feat_pair_scores[str(b)].append(float(score))

        for f, vals in feat_pair_scores.items():
            vals_sorted = sorted(vals, reverse=True)[: max(1, int(selector_interaction_max_pairs_per_feature))]
            interaction_support_raw[idx_map[f]] = float(np.sum(vals_sorted))
    else:
        pair_df = pd.DataFrame(columns=["feature_a", "feature_b", "lift", "final_pair_score"])

    # Final score components
    top30_support = _robust_norm(top_support_raw)
    top30_lift = _robust_norm(top_lift_raw)
    global_importance = _robust_norm(share_mu)
    interaction_support = _robust_norm(interaction_support_raw)

    metrics_df["top30_support"] = top30_support.astype(np.float32)
    metrics_df["top30_lift"] = top30_lift.astype(np.float32)
    metrics_df["global_importance"] = global_importance.astype(np.float32)
    metrics_df["stability_score"] = np.asarray(stability_score, dtype=np.float32)
    metrics_df["frequency_score"] = np.asarray(frequency_score, dtype=np.float32)
    metrics_df["interaction_support"] = interaction_support.astype(np.float32)
    metrics_df["median_rank"] = np.asarray(med_rank, dtype=np.float32)
    metrics_df["mean_abs_rank_dev"] = np.asarray(mad_rank, dtype=np.float32)
    metrics_df["final_score"] = (
        float(_final_w["top30"]) * metrics_df["top30_support"]
        + float(_final_w["global"]) * metrics_df["global_importance"]
        + float(_final_w["stability"]) * metrics_df["stability_score"]
        + float(_final_w["frequency"]) * metrics_df["frequency_score"]
        + float(_final_w["interaction"]) * metrics_df["interaction_support"]
    ).astype(np.float32)
    metrics_df["final_rank"] = metrics_df["final_score"].rank(ascending=False, method="average")
    metrics_df_sorted = metrics_df.sort_values("final_score", ascending=False)

    selected = metrics_df_sorted.index[:n_final].tolist()
    anchors = [f for f in (selector_anchor_features or []) if f in metrics_df_sorted.index]
    for f in anchors:
        if f not in selected:
            selected.append(f)
    selected = [f for f in metrics_df_sorted.index if f in set(selected)][: max(n_final, len(anchors))]

    score_map = {f: float(s) for f, s in metrics_df_sorted["final_score"].items()}
    if selector_prev_selected:
        selected, hyst_stats = _apply_overlap_hysteresis(
            selected=selected,
            ranked_all=list(metrics_df_sorted.index),
            score_map=score_map,
            anchors=anchors,
            prev_selected=list(selector_prev_selected),
            min_overlap=float(selector_min_overlap),
            margin=float(selector_hysteresis_margin),
        )
    else:
        hyst_stats = {"overlap_before": 1.0, "overlap_after": 1.0, "swaps": 0.0}

    tprint(
        f"MDI Cap: Selected {len(selected)} features (Cap {cumulative_cap:.0%}, "
        f"Eff. Mass {cumsum[min(max(len(selected)-1, 0), len(cumsum)-1)]:.3f}). "
        f"Constraints: {n_min_hard} <= N <= {n_max_hard}"
    )

    top30_metric_table = pd.DataFrame(last_top_metrics) if len(last_top_metrics) else pd.DataFrame(columns=["metric", "n_top"])
    stability_table = metrics_df_sorted[["median_rank", "mean_abs_rank_dev", "stability_score", "frequency_score"]].copy()
    final_score_table = metrics_df_sorted[
        ["final_score", "top30_support", "top30_lift", "global_importance", "stability_score", "frequency_score", "interaction_support"]
    ].copy()
    summary = {
        "selector_head_name": _head_name,
        "selector_target": _target_name,
        "selector_top_metric": _top_metric,
        "selector_focus_top_frac": _top_frac,
        "weights": {k: float(v) for k, v in _final_w.items()},
        "n_train_features": int(p_last),
        "n_selected": int(len(selected)),
        "anchor_count": int(len(anchors)),
        "selected_anchor_count": int(sum(1 for x in selected if x in set(anchors))),
        "overlap_before": float(hyst_stats.get("overlap_before", 1.0)),
        "overlap_after": float(hyst_stats.get("overlap_after", 1.0)),
        "hysteresis_swaps": float(hyst_stats.get("swaps", 0.0)),
    }

    if selector_emit_report:
        try:
            _base_report_dir = selector_report_dir
            if _base_report_dir is None:
                _base_report_dir = os.path.join("data", "artifacts", "default", "fs_reports")
            _safe_head = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in _head_name)
            _out_dir = os.path.join(_base_report_dir, _safe_head)
            os.makedirs(_out_dir, exist_ok=True)
            metrics_df_sorted.to_csv(os.path.join(_out_dir, "metrics_features.csv"), index=True)
            try:
                metrics_df_sorted.to_parquet(os.path.join(_out_dir, "metrics_features.parquet"), index=True)
            except Exception:
                pass
            pair_df.to_csv(os.path.join(_out_dir, "pairs.csv"), index=False)
            try:
                pair_df.to_parquet(os.path.join(_out_dir, "pairs.parquet"), index=False)
            except Exception:
                pass
            with open(os.path.join(_out_dir, "selected_features.json"), "w", encoding="utf-8") as f:
                json.dump({"selected_features": selected, "anchors": anchors}, f, indent=2)
            with open(os.path.join(_out_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as _e_report:
            tprint(f"MDI: failed to emit selector report for {_head_name}: {_e_report}")

    import gc
    gc.collect()
    return MDISelectionResult(
        metrics_table=metrics_df_sorted,
        selected_features=selected,
        kept_after_dedupe=kept_features,
        top30_metric_table=top30_metric_table,
        stability_table=stability_table,
        interaction_table=pair_df,
        final_score_table=final_score_table,
        summary=summary,
    )

# Backwards compatibility alias if needed, or update call sites
mdi_feature_selection_leakage_safe = mdi_feature_selection_v3


# ======================================================================================
# Top-K Precision Feature Selection (Report 2026-02-11)
# ======================================================================================

@jit(nopython=True, nogil=True, cache=True)
def _compute_spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman correlation using rank computation."""
    n = len(x)
    if n < 3:
        return 0.0
    
    # Compute ranks for x
    x_order = np.argsort(x)
    x_ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        x_ranks[x_order[i]] = float(i)
    
    # Compute ranks for y
    y_order = np.argsort(y)
    y_ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        y_ranks[y_order[i]] = float(i)
    
    # Pearson correlation on ranks
    mean_x = np.mean(x_ranks)
    mean_y = np.mean(y_ranks)
    
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    
    for i in range(n):
        dx = x_ranks[i] - mean_x
        dy = y_ranks[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    
    return num / np.sqrt(den_x * den_y)


@jit(nopython=True, nogil=True, cache=True)
def _bin_and_compute_pos_rates_numba(
    feat_vals: np.ndarray,
    y_binary: np.ndarray,
    sample_weight: np.ndarray,
    n_bins: int,
    percentiles: np.ndarray
) -> tuple:
    """
    Bin feature values and compute positive rates per bin.
    Returns (bin_indices, pos_rates, has_weight).
    """
    n = len(feat_vals)
    has_weight = len(sample_weight) == n
    
    # Assign bins based on percentiles
    bin_labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        bin_labels[i] = 0
        for b in range(n_bins):
            if feat_vals[i] <= percentiles[b]:
                bin_labels[i] = b
                break
        else:
            bin_labels[i] = n_bins - 1
    
    # Compute positive rate per bin using bincount
    # Sum of y per bin
    y_sum = np.zeros(n_bins, dtype=np.float64)
    w_sum = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(n):
        b = bin_labels[i]
        if has_weight:
            y_sum[b] += y_binary[i] * sample_weight[i]
            w_sum[b] += sample_weight[i]
        else:
            y_sum[b] += y_binary[i]
            counts[b] += 1.0
    
    # Compute positive rates
    pos_rates = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        if has_weight:
            if w_sum[b] > 0:
                pos_rates[b] = y_sum[b] / w_sum[b]
        else:
            if counts[b] > 0:
                pos_rates[b] = y_sum[b] / counts[b]
    
    return bin_labels, pos_rates, has_weight


def compute_decile_ranking_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_bins: int = 5,
    n_bootstrap: int = 20,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    n_jobs: int = 2
) -> pd.Series:
    """
    Compute feature importance based on quintile-based monotonic ranking strength.
    
    OPTIMIZED VERSION:
    ==================
    - Uses Numba JIT for core computation
    - Vectorized binning using np.percentile
    - Reduced bootstrap iterations (20 vs 50)
    - Fast Spearman correlation without scipy overhead
    - 5 bins (quintiles) = 20% each, aligning with lift@20% metric
    - Parallel column processing using joblib
    
    PURPOSE:
    ========
    Standard MDI feature selection optimizes for overall impurity reduction,
    which doesn't guarantee that selected features are good at ranking 
    positive examples. This function measures how well each feature 
    exhibits monotonic relationship with the target across quintiles.
    
    ADVANTAGES OVER TOP-K:
    ======================
    1. Monotonic ranking strength: Measures if positive rate increases 
       monotonically across quintiles (not just at one cutoff)
    2. Robust to k choice: No arbitrary 20% cutoff - uses all data
    3. Uses Spearman correlation: Captures ranking quality across entire range
    4. 5 bins (20% each) aligns with lift@20% metric used in training gates
    
    HOW IT WORKS:
    =============
    1. For each feature, bin samples into quintiles (5 equal-sized bins, 20% each)
    2. Compute positive rate per bin
    3. Compute Spearman correlation between bin index and positive rate
    4. Return absolute Spearman correlation as importance
    
    A feature with perfect monotonic ranking would have:
    - Quintile 1 (lowest values): lowest positive rate
    - Quintile 5 (highest values): highest positive rate
    - Spearman correlation ≈ 1.0
    
    Args:
        X: Feature DataFrame (n_samples, n_features)
        y: Binary target (n_samples,)
        n_bins: Number of bins (default 5 for quintiles, max 5)
        n_bootstrap: Number of bootstrap iterations for stability (default 20)
        sample_weight: Optional sample weights
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
    
    Returns:
        pd.Series: Feature importance (Spearman correlation), sorted descending
    """
    # Cap n_bins at 5 (20% each max) to align with lift@20% metric
    n_bins = min(n_bins, 5)
    
    y = np.asarray(y, dtype=np.float64)
    y_binary = (y >= 0.5).astype(np.float64)
    
    n_samples = len(y)
    min_samples_per_bin = max(5, n_bins)
    
    rng = np.random.RandomState(random_state)
    
    # Pre-compute bootstrap indices
    bootstrap_indices = rng.randint(0, n_samples, size=(n_bootstrap, n_samples))
    
    # Prepare sample weight array
    w_arr = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else np.empty(0, dtype=np.float64)
    
    # Convert X to numpy array for faster column access (float32 for memory efficiency)
    X_values = X.values.astype(np.float32)
    col_names = list(X.columns)
    n_cols = len(col_names)
    
    # Pre-allocate results array for parallel processing
    importance_values = np.zeros(n_cols, dtype=np.float64)
    
    def _process_column(col_idx: int) -> float:
        """Process a single column and return its importance score."""
        feat_vals = X_values[:, col_idx]
        
        # Skip features with no variance
        if np.nanstd(feat_vals) < 1e-9:
            return 0.0
        
        # Skip features with too many NaN
        valid_mask = np.isfinite(feat_vals)
        n_valid = valid_mask.sum()
        if n_valid < min_samples_per_bin * n_bins:
            return 0.0
        
        spearman_samples = []
        
        for boot_idx in range(n_bootstrap):
            idx = bootstrap_indices[boot_idx]
            
            feat_sample = feat_vals[idx]
            y_sample = y_binary[idx]
            w_sample = w_arr[idx] if len(w_arr) > 0 else w_arr
            
            # Get valid mask
            valid = np.isfinite(feat_sample)
            n_valid_boot = valid.sum()
            if n_valid_boot < min_samples_per_bin * n_bins:
                continue
            
            feat_valid = feat_sample[valid]
            y_valid = y_sample[valid]
            w_valid = w_sample[valid] if len(w_sample) > 0 else np.empty(0, dtype=np.float64)
            
            # Compute percentiles for binning
            try:
                percentiles = np.percentile(feat_valid, np.linspace(0, 100, n_bins + 1)[1:-1])
                
                # Bin and compute positive rates
                bin_labels, pos_rates, has_weight = _bin_and_compute_pos_rates_numba(
                    feat_valid, y_valid, w_valid, n_bins, percentiles
                )
                
                # Count non-empty bins
                n_actual_bins = 0
                for b in range(n_bins):
                    if pos_rates[b] > 0 or np.sum(bin_labels == b) > 0:
                        n_actual_bins += 1
                
                if n_actual_bins < 3:
                    continue
                
                # Get valid bin indices and positive rates
                bin_indices_arr = np.arange(n_bins, dtype=np.float64)
                
                # Compute Spearman correlation
                corr = _compute_spearman_fast(bin_indices_arr, pos_rates)
                
                if np.isfinite(corr):
                    spearman_samples.append(abs(corr))
                    
            except Exception:
                continue
        
        if spearman_samples:
            return np.mean(spearman_samples)
        else:
            return 0.0
    
    # Use joblib for parallel processing if n_jobs != 1
    if n_jobs != 1 and n_cols > 10:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_process_column)(col_idx) for col_idx in range(n_cols)
            )
            importance_values = np.array(results)
        except ImportError:
            # Fallback to sequential if joblib not available
            for col_idx in range(n_cols):
                importance_values[col_idx] = _process_column(col_idx)
    else:
        # Sequential processing
        for col_idx in range(n_cols):
            importance_values[col_idx] = _process_column(col_idx)
    
    importance = dict(zip(col_names, importance_values))
    return pd.Series(importance).sort_values(ascending=False)


def compute_topk_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    k_frac: float = 0.20,
    n_bootstrap: int = 50,
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42
) -> pd.Series:
    """
    [DEPRECATED] Use compute_decile_ranking_importance instead.
    
    Compute feature importance based on precision@k contribution.
    This is kept for backwards compatibility but decile-based approach
    is preferred for robustness.
    """
    # Delegate to decile-based implementation
    return compute_decile_ranking_importance(
        X, y, n_bins=10, n_bootstrap=n_bootstrap,
        sample_weight=sample_weight, random_state=random_state
    )


def mdi_feature_selection_v4_topk(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    base_model=None,
    sample_weight: Optional[np.ndarray] = None,
    k_frac: float = 0.20,
    topk_weight: float = 0.2,
    **kwargs
) -> MDISelectionResult:
    """
    MDI feature selection v4 with decile-based ranking awareness.
    
    PURPOSE:
    ========
    Combines standard MDI importance with decile-based monotonic ranking strength.
    This addresses models that have good overall metrics but poor 
    concentration of edge at the top (low lift@k).
    
    IMPROVEMENT OVER TOP-K:
    =======================
    Instead of measuring precision at a single k cutoff, this version uses
    decile-based Spearman correlation which:
    1. Measures monotonic ranking strength across ALL deciles
    2. Is robust to the choice of k (no arbitrary 20% cutoff)
    3. Captures whether positive rate increases monotonically with feature value
    
    WEIGHTING:
    ==========
    - MDI importance: (1 - topk_weight) = 0.70 by default
    - Decile ranking importance: topk_weight = 0.20 by default
    
    The decile component ensures features that exhibit monotonic relationship
    with the target get selected, even if they don't have the highest MDI.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        base_model: Base model for MDI (default: ExtraTreesRegressor)
        sample_weight: Optional sample weights
        k_frac: [DEPRECATED] Kept for API compatibility
        topk_weight: Weight for decile ranking component (default 0.20)
        **kwargs: Additional arguments passed to mdi_feature_selection_v3
    
    Returns:
        MDISelectionResult with combined importance ranking
    """
    tprint(f"MDI v4 TopK: Running combined MDI + decile-ranking selection (weight={topk_weight})")
    
    # 1. Get standard MDI result
    mdi_result = mdi_feature_selection_v3(X, y, base_model, sample_weight=sample_weight, **kwargs)
    
    # 2. Get decile-based ranking importance
    decile_imp = compute_decile_ranking_importance(
        X, y, n_bins=10, sample_weight=sample_weight
    )
    
    # 3. Normalize both importance scores
    mdi_imp = mdi_result.metrics_table['share_mu'].copy()
    
    # Normalize to [0, 1]
    mdi_norm = (mdi_imp - mdi_imp.min()) / (mdi_imp.max() - mdi_imp.min() + 1e-9)
    
    decile_aligned = decile_imp.reindex(mdi_imp.index, fill_value=0)
    decile_norm = (decile_aligned - decile_aligned.min()) / (decile_aligned.max() - decile_aligned.min() + 1e-9)
    
    # 4. Combine
    combined = (1 - topk_weight) * mdi_norm + topk_weight * decile_norm
    
    # 5. Re-rank
    combined_sorted = combined.sort_values(ascending=False)
    
    # 6. Select top features (same count as MDI result)
    n_selected = len(mdi_result.selected_features)
    selected = combined_sorted.head(n_selected).index.tolist()
    
    # Create new metrics table with combined score
    new_metrics = mdi_result.metrics_table.copy()
    new_metrics['decile_importance'] = decile_aligned
    new_metrics['combined_score'] = combined
    new_metrics = new_metrics.loc[combined_sorted.index]
    
    tprint(f"MDI v4 TopK: Selected {len(selected)} features with combined importance")
    
    return MDISelectionResult(new_metrics, selected, mdi_result.kept_after_dedupe)
