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
) -> MDISelectionResult:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    rng = check_random_state(random_state)

    # Fast initial conversion
    X_np_full = np.ascontiguousarray(X.values, dtype=np.float32)
    y_np = y.values if hasattr(y, 'values') else np.asarray(y)

    sw_np = None
    if sample_weight is not None:
        sw_np = sample_weight.to_numpy() if isinstance(sample_weight, pd.Series) else np.asarray(sample_weight)

    feature_names_full = list(X.columns)

    splits = purged_embargoed_splits(X_np_full.shape[0], n_splits, purge=purge)

    # 1. Anchored Pre-Dedupe (Train Window 0)
    train_idx0, _ = splits[0]
    X_tr0 = X_np_full[train_idx0]

    # Quantile Transform for Robust Pearson
    # subsample=min(len(X_tr0), 100000) inside QT is default
    qt = QuantileTransformer(n_quantiles=min(len(X_tr0), 1000), output_distribution='normal', random_state=random_state)
    X_tr0_qt = qt.fit_transform(X_tr0)

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
    X_np = X_np_full[:, keep_mask]
    p_reduced = X_np.shape[1]

    # Suggest depth based on REDUCED dimensionality
    depth = suggest_depth(p_reduced, X_np.shape[0])

    # 2. Stability Analysis CV
    folds_data = defaultdict(list)

    for train_idx, _ in splits:
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

        sw_tr = sw_np[train_idx] if sw_np is not None else None
        m.fit(X_np[train_idx], y_np[train_idx], sample_weight=sw_tr)

        # C-level raw importance
        folds_data['share'].append(m.feature_importances_)

        # Fast extra metrics
        freq, mdi_d, mdi_c, _ = extract_extra_mdi_metrics_fast(m, p_reduced)
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

    metrics_df = pd.DataFrame(agg, index=kept_features)

    # Final Ranking
    metrics_df['composite_rank'] = (
        metrics_df['share_stab'].rank(ascending=False) * 0.5 +
        metrics_df['mdi_depth_stab'].rank(ascending=False) * 0.3 +
        metrics_df['mdi_cov_stab'].rank(ascending=False) * 0.2
    ).rank()

    metrics_df_sorted = metrics_df.sort_values('composite_rank')
    selected = metrics_df_sorted.index.tolist()

    return MDISelectionResult(metrics_df_sorted, selected, kept_features)

# Backwards compatibility alias if needed, or update call sites
mdi_feature_selection_leakage_safe = mdi_feature_selection_v3

# Example usage check:
if __name__ == "__main__":
    from sklearn.ensemble import ExtraTreesClassifier
    # p=150, n=50k
    n_rows, n_feats = 1000, 150
    X_dummy = pd.DataFrame(np.random.randn(n_rows, n_feats), columns=[f"f{i}" for i in range(n_feats)])
    # Correlated feature
    X_dummy["f1"] = X_dummy["f0"] * 0.99 + np.random.randn(n_rows)*0.01

    y_dummy = pd.Series((X_dummy['f1'] + X_dummy['f2'] > 0).astype(int))
    w_dummy = np.random.uniform(0.5, 1.5, n_rows)

    model = ExtraTreesClassifier(n_jobs=-1)
    res = mdi_feature_selection_v3(X_dummy, y_dummy, model, n_splits=5, min_samples_leaf=20, sample_weight=w_dummy)
    print(f"Top Features: {res.selected_features[:5]}")
    print(f"Kept after dedupe: {len(res.kept_after_dedupe)}")
