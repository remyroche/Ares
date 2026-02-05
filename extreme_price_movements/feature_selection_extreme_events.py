"""
Robust MDI Feature Selection with Quantile-Transformed Correlations v3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
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
        train_end = max(0, val_start - purge)
        train_idx = indices[:train_end]

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
    base_model,
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
    if not np.isfinite(X_np_full).all():
        tprint("MDI: Found non-finite values after float32 conversion (likely overflow). Cleaning numpy array...")
        mask_bad = ~np.isfinite(X_np_full).all(axis=1)
        if mask_bad.any():
            tprint(f"Dropping {mask_bad.sum()} additional rows due to float32 overflow.")
            X_np_full = X_np_full[~mask_bad]

            # Align y and sw
            if hasattr(y, 'values'): y_vals = y.values
            else: y_vals = np.asarray(y)
            y_np = y_vals[~mask_bad]

            if sample_weight is not None:
                if hasattr(sample_weight, 'values'): sw_vals = sample_weight.values
                else: sw_vals = np.asarray(sample_weight)
                sw_np = sw_vals[~mask_bad]
            else:
                sw_np = None

            # Note: We do NOT update X (DataFrame) or feature_names_full (Columns).
            # This logic assumes we iterate feature indices, which are columns.
            # Removing ROWS from numpy array is safe.
        else:
            # Should not happen if all() is false
            y_np = y.values if hasattr(y, 'values') else np.asarray(y)
            sw_np = None
            if sample_weight is not None:
                sw_np = sample_weight.to_numpy() if isinstance(sample_weight, pd.Series) else np.asarray(sample_weight)
    else:
        y_np = y.values if hasattr(y, 'values') else np.asarray(y)
        sw_np = None
        if sample_weight is not None:
            sw_np = sample_weight.to_numpy() if isinstance(sample_weight, pd.Series) else np.asarray(sample_weight)

    feature_names_full = list(X.columns)

    # Check if empty again
    if X_np_full.shape[0] < 5:
        tprint("MDI: Too few samples after cleaning. Returning empty result.")
        return MDISelectionResult(pd.DataFrame(), [], [])

    # Initial Splits for Dedupe
    try:
        splits_full = purged_embargoed_splits(X_np_full.shape[0], n_splits, purge=purge)
    except ValueError as e:
        tprint(f"MDI: Split generation failed: {e}")
        return MDISelectionResult(pd.DataFrame(), [], [])

    # 1. Anchored Pre-Dedupe (Train Window 0)
    train_idx0, _ = splits_full[0]
    X_tr0 = X_np_full[train_idx0]

    # Quantile Transform for Robust Pearson
    # subsample=min(len(X_tr0), 100000) inside QT is default
    qt = QuantileTransformer(n_quantiles=min(len(X_tr0), 1000), output_distribution='normal', random_state=random_state)
    try:
        X_tr0_qt = qt.fit_transform(X_tr0)
    except Exception as e:
        tprint(f"MDI: QuantileTransformer failed: {e}")
        return MDISelectionResult(pd.DataFrame(), [], [])

    corr_p = np.corrcoef(X_tr0_qt, rowvar=False)
    # Fix for scalar result if single feature
    if corr_p.ndim == 0:
        corr_p = corr_p.reshape(1, 1)

    # Greedy Dedupe
    keep_mask = np.ones(len(feature_names_full), dtype=bool)
    for i in range(len(feature_names_full)):
        if not keep_mask[i]: continue
        redundant = np.abs(corr_p[i]) >= pre_dedupe_threshold
        redundant[:i+1] = False
        keep_mask[redundant] = False

    kept_features = [feature_names_full[i] for i in range(len(keep_mask)) if keep_mask[i]]
    kept_features_indices = [i for i, kept in enumerate(keep_mask) if kept]

    tprint(f"MDI: Starting with {len(kept_features)} features after dedupe (target: {end_features}).")

    current_features = kept_features

    # RFE Loop
    metrics_df_sorted = pd.DataFrame() # To hold final result

    while True:
        p = len(current_features)
        K = end_features

        # Check termination condition
        # We run the evaluation even if p <= K to get the final metrics for the survivors
        # But if we just finished a round and reached K, we stop?
        # Actually, the loop condition "while p > K" is usually used, but we need metrics for the final set.
        # So we run evaluation, THEN drop. If we are already <= K, we don't drop, just break after eval.

        # Subsampling Logic
        n_star = max(30000, 300 * p, 2000 * K)
        N = len(X)
        subsample_pct = min(1.0, n_star / N)

        # Subsample Data
        if subsample_pct < 1.0:
            n_sub = int(N * subsample_pct)
            # Sample indices without replacement and sort to preserve time order
            indices_sub = np.sort(rng.choice(N, n_sub, replace=False))

            # Map features indices
            feat_indices = [feature_names_full.index(f) for f in current_features]
            X_curr_np = X_np_full[indices_sub][:, feat_indices]
            y_curr_np = y_np[indices_sub]
            sw_curr_np = sw_np[indices_sub] if sw_np is not None else None

            # Recalculate splits for the smaller dataset
            # Note: purged_embargoed_splits logic assumes contiguous time, but with sorted random subsample
            # we approximate it. It's the best we can do for subsampled RFE on time series without contiguous blocks.
            try:
                splits_curr = purged_embargoed_splits(n_sub, n_splits, purge=purge)
            except ValueError:
                 splits_curr = [] # Should be handled by loop skipping
        else:
            feat_indices = [feature_names_full.index(f) for f in current_features]
            X_curr_np = X_np_full[:, feat_indices]
            y_curr_np = y_np
            sw_curr_np = sw_np
            splits_curr = splits_full # Can reuse if N didn't change (only first iter potentially)

        if not splits_curr:
             tprint("MDI: No splits for current subsample. Breaking.")
             break

        # Suggest depth based on CURRENT p
        depth = suggest_depth(p, X_curr_np.shape[0])

        # 2. Stability Analysis CV
        folds_data = defaultdict(list)

        for train_idx, _ in splits_curr:
            m = clone(base_model)
            # Update params
            if hasattr(m, "set_params"):
                params = {}
                if "max_depth" in m.get_params():
                    params["max_depth"] = depth
                if "n_estimators" in m.get_params():
                    params["n_estimators"] = analysis_n_estimators
                if "min_samples_leaf" in m.get_params():
                    params["min_samples_leaf"] = min_samples_leaf
                if "min_impurity_decrease" in m.get_params():
                    params["min_impurity_decrease"] = min_impurity_decrease
                if "random_state" in m.get_params():
                    params["random_state"] = random_state
                m.set_params(**params)

            sw_tr = sw_curr_np[train_idx] if sw_curr_np is not None else None
            m.fit(X_curr_np[train_idx], y_curr_np[train_idx], sample_weight=sw_tr)

            # C-level raw importance
            folds_data['share'].append(m.feature_importances_)

            # Fast extra metrics
            freq, mdi_d, mdi_c, _ = extract_extra_mdi_metrics_fast(m, p)
            folds_data['freq'].append(freq)
            folds_data['mdi_depth'].append(mdi_d)
            folds_data['mdi_cov'].append(mdi_c)

        # 3. Aggregation
        agg = {}
        for metric, values in folds_data.items():
            arr = np.vstack(values)
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            hit = (arr > 0).mean(axis=0)
            cv = sd / (mu + 1e-12)
            agg[f"{metric}_mu"] = mu
            agg[f"{metric}_stab"] = mu * (hit / (cv + 1e-12))

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
