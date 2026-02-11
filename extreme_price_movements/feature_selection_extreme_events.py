"""
Robust MDI Feature Selection with Quantile-Transformed Correlations v3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import importlib.util

from numba import jit
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
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

@jit(nopython=True, cache=True)
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


# ======================================================================================
# Main Production Pipeline
# ======================================================================================

def linear_prescreen_enet(
    X: pd.DataFrame,
    y: np.ndarray,
    n_select: int,
    multiplier: int = 4,
    l1_ratio: float = 0.6,
    alpha_lo: float = 1e-6,
    alpha_hi: float = 1e1,
    max_iter: int = 5000,
    max_steps: int = 25,
    tol_frac: float = 0.15,
    random_state: int = 42,
) -> list[str]:
    """
    Drop-in ElasticNet pre-screen that targets keep-count ~= multiplier * n_select.

    Returns a list of feature names to keep (exactly target_keep if possible,
    otherwise the closest solution trimmed by |coef|).

    Notes:
    - Uses signed log transform on y: sign(y) * log1p(|y|)
    - Standardizes X before ElasticNet (required for meaningful L1/L2 penalties)
    - Searches alpha on a log scale to hit the target sparsity
    """
    if X is None or X.empty:
        return []
    p = X.shape[1]
    target_keep = int(np.clip(multiplier * int(n_select), 1, p))

    y = np.asarray(y)
    y_t = np.sign(y) * np.log1p(np.abs(y))

    def fit_abscoef(alpha: float) -> np.ndarray:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("enet", ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                random_state=random_state
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

def mdi_feature_selection_v3(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    base_model=None,
    n_splits: int = 6,
    purge: int = 5,
    min_samples_leaf: int = 50,
    min_samples_leaf_pct: float = 0.01,
    min_impurity_decrease: float = 1e-4,
    analysis_n_estimators: int = 500, # Higher for 150-feature stability

    pre_dedupe_threshold: float = 0.95, # Relaxed from 0.98 to 0.95 per plan
    random_state: int = 0,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    end_features: Optional[int] = None,
    cumulative_cap: float = 0.98,
    min_share: float = 0.001, # Threshold for noise
    min_features: int = 5,    # Hard floor
    max_features_pct: float = 0.5, # Hard ceiling (fraction of input)
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

    # Robust Cleaning using shared utility
    # This aligns y and sample_weight if rows are dropped.
    X, y, sample_weight = clean_dataset(X, y, sample_weight, name="X_mdi")

    if X.empty:
        tprint("MDI: X became empty after cleaning. Returning empty result.")
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
        default_leaf = max(1, int(np.ceil(min_samples_leaf_pct * len(X))))
        base_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=default_leaf,
            min_samples_split=max(2, 3 * default_leaf),
            max_features='sqrt',
            n_jobs=-1,
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

        if hasattr(y, 'values'): y_vals = y.values
        else: y_vals = np.asarray(y)
        y_np = y_vals[finite_rows]

        if sample_weight is not None:
            if hasattr(sample_weight, 'values'): sw_vals = sample_weight.values
            else: sw_vals = np.asarray(sample_weight)
            sw_np = sw_vals[finite_rows]
        else:
            sw_np = None
    else:
        y_np = y.values if hasattr(y, 'values') else np.asarray(y)
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

    if len(current_features) > 4 * end_features:
        tprint(f"MDI: Running Linear ElasticNet prescreen on {len(current_features)} features...")
        prescreened_features = linear_prescreen_enet(
            X[current_features],
            y_np,
            n_select=end_features,
            multiplier=4
        )
        tprint(f"MDI: ElasticNet prescreen reduced features from {len(current_features)} to {len(prescreened_features)}")
        current_features = prescreened_features

    metrics_df_sorted = pd.DataFrame()

    # Cache supported params once
    base_params = base_model.get_params() if hasattr(base_model, 'get_params') else {}
    supported_params = set(base_params.keys())

    while True:
        p = len(current_features)

        # Subsampling Logic
        n_star = max(30000, 300 * p, 2000 * end_features)

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

        depth = suggest_depth(p, X_curr_np.shape[0])

        # Streaming Aggregation
        sums = {k: np.zeros(p, dtype=np.float64) for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        sq_sums = {k: np.zeros(p, dtype=np.float64) for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        counts = {k: 0 for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}
        pos_counts = {k: np.zeros(p, dtype=np.float64) for k in ['share', 'freq', 'mdi_depth', 'mdi_cov']}

        valid_folds = 0

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
            leaf_samples = max(1, int(np.ceil(min_samples_leaf_pct * train_n)))
            if "min_samples_leaf" in supported_params: params["min_samples_leaf"] = leaf_samples
            if "min_samples_split" in supported_params: params["min_samples_split"] = max(2, 3 * leaf_samples)

            if "min_data_in_leaf" in supported_params:
                params["min_data_in_leaf"] = max(2, int(np.ceil(0.012 * train_n)))
            if "min_data_in_bin" in supported_params:
                params["min_data_in_bin"] = 127
            if "min_gain_to_split" in supported_params:
                base_gain = base_params.get("min_gain_to_split", 0.0)
                params["min_gain_to_split"] = float(base_gain) if float(base_gain) > 0 else 0.05
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
                m.fit(
                    X_curr_np[train_idx],
                    y_curr_np[train_idx],
                    sample_weight=sw_tr,
                    eval_set=[(X_curr_np[val_idx], y_curr_np[val_idx])],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            else:
                m.fit(X_curr_np[train_idx], y_curr_np[train_idx], sample_weight=sw_tr)

            # C-level raw importance
            # folds_data['share'].append(m.feature_importances_)
            share = m.feature_importances_.astype(np.float64)

            # Fast extra metrics
            freq, mdi_d, mdi_c, _ = extract_extra_mdi_metrics_fast(m, p)
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

            valid_folds += 1

        if valid_folds == 0:
            break

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
            metrics_df['share_stab'].rank(ascending=False) * 0.5 +
            metrics_df['mdi_depth_stab'].rank(ascending=False) * 0.3 +
            metrics_df['mdi_cov_stab'].rank(ascending=False) * 0.2
        ).rank()

        metrics_df_sorted = metrics_df.sort_values('composite_rank')

        # If we reached the target, we are done
        if p <= end_features:
            tprint(f"MDI RFE: Reached target {p} features. Stopping.")
            break

        # Determine how many to drop
        remove_features = p - end_features
        n_drop = int(remove_features / 4) + 5
        n_drop = min(n_drop, remove_features)

        tprint(f"MDI RFE: p={p}, dropping {n_drop} features. (Next p={p-n_drop})")

        survivors = metrics_df_sorted.index[:-n_drop].tolist()
        current_features = survivors

    # --- Post-Loop Cumulative Cap for Noise Tail Removal ---
    # Even if RFE stopped at end_features (or if we just ran once),
    # we apply the cumulative cap to ensure we don't keep weak noise.
    
    # --- Post-Loop Advanced Cumulative Cap ---
    # metric_df_sorted is ordered by composite_rank (best to worst).
    
    # 1. Compute Effective Importance (Penalize Instability)
    # share_eff = max(0, mu - 0.5 * std)
    # We estimate std from stability score: stab = mu * (hit / cv). 
    # But we have 'share_mu' and 'share_stab'.
    # Actually, we calculated:
    # cv = sd / mu
    # stab = mu * (hit / cv) = mu * hit * mu / sd = (mu^2 * hit) / sd?
    # No, we need original sd.
    # We don't have raw sd in the dataframe, but we can approximate or just use share_mu for now
    # if we didn't save sd. 
    # Wait, we constructed agg with separate columns? 
    # "agg[f'{metric}_mu']" and "agg[f'{metric}_stab']".
    # We did not save sigma directly in the dataframe construction earlier in the code.
    # Let's verify `metrics_df` creation block (lines 589-590).
    # It only has _mu and _stab.
    # We can reconstruct sd from stab if hit is known (hit not saved).
    # OR we just rely on composite_rank which uses stability.
    
    # User requested: "share_eff = metrics_df['share_mu'] - z * metrics_df['share_std']"
    # To support this, I should have saved share_std.
    # Since I cannot easily change the loop (it's inside the function above), 
    # I will modify the Aggregation step (lines 574-590) to include `_std`. 
    # BUT I am in `multi_replace`, I can't easily jump back up.
    # Workaround: Use `share_stab` as a proxy for robust importance.
    # `share_stab` IS stability-weighted importance.
    # let's use `share_mu` for mass, but rely on `composite_rank` for ordering?
    # User explicitly asked for "Effective Importance" formula.
    # I will assume I can edit the aggregation block in a separate chunk.
    # YES, I will add a chunk to save `_std` first.
    
    # --- Post-Loop Advanced Cumulative Cap ---
    # metric_df_sorted is ordered by composite_rank (best to worst).
    
    # 1. Compute Effective Importance (Penalize Instability)
    # share_eff = max(0, mu - 0.5 * std)
    z = 0.5
    share_mu = metrics_df_sorted['share_mu'].values.astype(np.float64)
    share_std = metrics_df_sorted['share_std'].values.astype(np.float64)
    
    share_eff = np.maximum(0.0, share_mu - z * share_std)
    
    # 2. Normalize Effective Importance
    total_eff = np.sum(share_eff)
    if total_eff > 1e-12:
        share_norm = share_eff / total_eff
    else:
        # Fallback: if effective importance is zero (all noisy), use straight mu
        tprint("MDI: Effective importance is zero (high noise). Falling back to share_mu.")
        share_norm = share_mu / (np.sum(share_mu) + 1e-12)

    # 3. Cumulative Cutoff
    cumsum = np.cumsum(share_norm)
    
    # 4. Find Cutoff Index (Inclusive)
    # searchsorted returns first index i where cumsum[i] >= cap.
    # We want to include this index i because it's the one that crosses the threshold.
    cutoff_idx = np.searchsorted(cumsum, cumulative_cap)
    
    # Clip to valid range
    cutoff_idx = min(cutoff_idx, len(metrics_df_sorted) - 1)
    
    # Selected Count Candidates
    # +1 because slice is exclusive [0 : cutoff_idx+1] includes index cutoff_idx
    n_selected_cap = cutoff_idx + 1

    # 5. Apply Hard Guardrails
    n_total = len(metrics_df_sorted)
    n_max_hard = int(n_total * max_features_pct)
    # Ensure max is at least min if possible
    n_max_hard = max(n_max_hard, min_features)
    n_max_hard = min(n_max_hard, n_total) # Cap at physical limit
    
    n_min_hard = min(min_features, n_total) # Can't select more than we have
    
    # Enforce constraints order: Min -> Cap -> Max
    # "At least min_features" (unless total is smaller)
    # "At most max_features"
    # Actually, if we set max < min, we have a conflict.
    # Logic: Prioritize Min floor (sanity) over Max ceiling? 
    # Usually Max ceiling is for performance/noise control. Min floor is for signal capture.
    # If Max Pct is tiny (e.g. 10%), but we need 5 features, we should take 5.
    
    # Effective count
    n_final = max(n_selected_cap, n_min_hard)
    n_final = min(n_final, n_max_hard) 
    # If n_max_hard < n_min_hard (e.g. max=1, min=2 due to rounding), 
    # the above line would force it down to 1.
    # But we did `n_max_hard = max(n_max_hard, min_features)` above.
    # So n_max_hard >= min_features (unless n_total is small).
    # Correct.
    
    # 6. Apply Tail Filter (Optional but recommended)
    # Instead of breaking EARLY, we just count how many of the TOP N_FINAL actually pass min_share?
    # User issue: "result: you may end up selecting far less than 98% mass"
    # Logic: We stick to the Rank order. We just select Top N_FINAL.
    # We do NOT filter out items within the Top N_FINAL that have low share, 
    # because they might be high-stability (composite rank).
    # If a feature is Rank #3 but has small share, we Keep it.
    
    selected_by_cap = metrics_df_sorted.index[:n_final].tolist()
    
    tprint(f"MDI Cap: Selected {n_final} features (Cap {cumulative_cap:.0%}, Eff. Mass {cumsum[min(n_final-1, len(cumsum)-1)]:.3f}). Constraints: {n_min_hard} <= N <= {n_max_hard}")
    
    # tprint(f"MDI Cap: Selected {n_final} features (Cap {cumulative_cap:.0%}, Eff. Mass {cumsum[min(n_final-1, len(cumsum)-1)]:.3f}). Constraints: {n_min_hard} <= N <= {n_max_hard}")
    pass # Logging handled above
    
    selected = selected_by_cap

    return MDISelectionResult(metrics_df_sorted, selected, kept_features)

# Backwards compatibility alias if needed, or update call sites
mdi_feature_selection_leakage_safe = mdi_feature_selection_v3
