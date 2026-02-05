"""
Robust MDI Feature Selection with Quantile-Transformed Correlations v3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

from numba import jit
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state
from .utils import tprint, check_inf_nan, clean_dataset
from .sequential_bootstrap import get_ind_matrix, seq_bootstrap

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
    min_impurity_decrease: float = 1e-4,
    analysis_n_estimators: int = 500, # Higher for 150-feature stability
    pre_dedupe_threshold: float = 0.98,
    random_state: int = 0,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    end_features: Optional[int] = None,
) -> MDISelectionResult:
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
        tprint(f"MDI: end_features not provided. Auto-setting to {end_features} based on sample size {n_samples_full}")

    if base_model is None:
        base_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=20,
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

        for train_idx, _ in splits_curr:
            m = clone(base_model)
            params = {}
            if "max_depth" in supported_params: params["max_depth"] = depth
            if "n_estimators" in supported_params: params["n_estimators"] = analysis_n_estimators
            if "min_samples_leaf" in supported_params: params["min_samples_leaf"] = min_samples_leaf
            if "min_impurity_decrease" in supported_params: params["min_impurity_decrease"] = min_impurity_decrease
            if "random_state" in supported_params: params["random_state"] = random_state
            m.set_params(**params)

            sw_tr = sw_curr_np[train_idx] if sw_curr_np is not None else None
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

    selected = metrics_df_sorted.index.tolist()

    return MDISelectionResult(metrics_df_sorted, selected, kept_features)

# Backwards compatibility alias if needed, or update call sites
mdi_feature_selection_leakage_safe = mdi_feature_selection_v3
