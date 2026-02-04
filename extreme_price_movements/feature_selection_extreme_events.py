"""
Adds a QuantileTransformer step (train-only) before correlation computations so the
correlation matrices (especially Pearson) aren’t overly influenced by heavy tails/outliers.

Notes:
- Correlation is scale-invariant, but Pearson is still sensitive to outliers / tail behavior.
  QuantileTransform -> (approximately) uniform/normal marginals makes Pearson more robust.
- Spearman is rank-based already; QT is redundant in theory, but keeping it consistent can
  reduce numerical weirdness with discretized / spiky features.

Implementation details:
- QT is FIT ONLY on the train subsample used for correlations (anchored train0 or per-fold),
  so no lookahead leakage.
- QT is applied ONLY to the correlation subsample, not the full dataset, to avoid huge cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils import check_random_state

try:
    from scipy.stats import rankdata as _scipy_rankdata  # type: ignore
except Exception:
    _scipy_rankdata = None


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
# Correlation: NumPy Pearson + Spearman (rank then corrcoef) + clustering
# with optional QuantileTransformer on the correlation subsample
# ======================================================================================

def _subsample_rows_np(
    X: np.ndarray,
    sample_rows: Optional[int],
    rng: np.random.RandomState,
) -> np.ndarray:
    if sample_rows is None or X.shape[0] <= sample_rows:
        return X
    idx = rng.choice(X.shape[0], size=sample_rows, replace=False)
    return X[idx]


def _maybe_quantile_transform(
    Xs: np.ndarray,
    use_qt: bool,
    qt_output_distribution: str,
    qt_n_quantiles: int,
    random_state: int,
) -> np.ndarray:
    """
    Fit QT on Xs and transform Xs. Fit is train-only by construction (Xs from train).
    Returns float32.
    """
    if not use_qt:
        return Xs

    n_rows = Xs.shape[0]
    n_quant = int(min(qt_n_quantiles, n_rows))
    if n_quant < 10:
        # too small to be meaningful; skip
        return Xs

    qt = QuantileTransformer(
        n_quantiles=n_quant,
        output_distribution=qt_output_distribution,  # "normal" or "uniform"
        subsample=int(1e9),  # disable internal subsample; we already subsampled rows
        random_state=random_state,
        copy=True,
    )
    Xqt = qt.fit_transform(Xs)  # float64
    return np.ascontiguousarray(Xqt, dtype=np.float32)


def pearson_corrcoef(X: np.ndarray) -> np.ndarray:
    c = np.corrcoef(X, rowvar=False)
    if c.ndim == 0:
        c = c.reshape(1, 1)
    return c


def _rankdata_1d_numpy(a: np.ndarray) -> np.ndarray:
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(1, n + 1, dtype=np.float32)

    sorted_a = a[order]
    diffs = np.diff(sorted_a)
    if diffs.size == 0 or np.all(diffs != 0):
        return ranks

    breaks = np.where(diffs != 0)[0] + 1
    run_starts = np.r_[0, breaks]
    run_ends = np.r_[breaks, n]
    for s, e in zip(run_starts, run_ends):
        if e - s > 1:
            avg = (s + 1 + e) / 2.0
            ranks[order[s:e]] = np.float32(avg)
    return ranks


def _rankdata_1d(a: np.ndarray) -> np.ndarray:
    if _scipy_rankdata is not None:
        return _scipy_rankdata(a, method="average").astype(np.float32, copy=False)
    return _rankdata_1d_numpy(a)


def spearman_corrcoef(X: np.ndarray) -> np.ndarray:
    N, P = X.shape
    R = np.empty((N, P), dtype=np.float32)
    for j in range(P):
        R[:, j] = _rankdata_1d(X[:, j])
    c = np.corrcoef(R, rowvar=False)
    if c.ndim == 0:
        c = c.reshape(1, 1)
    return c


def connected_components_from_adj(adj: np.ndarray) -> List[List[int]]:
    P = adj.shape[0]
    seen = np.zeros(P, dtype=bool)
    comps: List[List[int]] = []
    for i in range(P):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.flatnonzero(adj[u])
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def greedy_dedupe_by_corr(
    corr: np.ndarray,
    feature_names: Sequence[str],
    threshold: float,
) -> List[str]:
    P = corr.shape[0]
    keep = np.ones(P, dtype=bool)
    for i in range(P):
        if not keep[i]:
            continue
        mask = (np.abs(corr[i]) >= threshold)
        mask[: i + 1] = False
        keep[mask] = False
    return [feature_names[i] for i in range(P) if keep[i]]


def clusters_from_corr(
    corr: np.ndarray,
    feature_names: Sequence[str],
    threshold: float,
) -> List[List[str]]:
    adj = (np.abs(corr) >= threshold)
    np.fill_diagonal(adj, False)
    comps = connected_components_from_adj(adj)
    return [[feature_names[i] for i in comp] for comp in comps]


# ======================================================================================
# Extra MDI metrics via shallow tree traversal
# ======================================================================================

@dataclass
class ExtraMDIMetricsFold:
    freq: np.ndarray
    mdi_depth: np.ndarray
    mdi_cov: np.ndarray
    median_gain: np.ndarray
    tree_hit_rate: np.ndarray


def extract_extra_mdi_metrics_from_forest(
    fitted_forest,
    n_features: int,
    depth_discount: float = 0.85,
    eps: float = 1e-12,
) -> ExtraMDIMetricsFold:
    freq = np.zeros(n_features, dtype=np.float32)
    mdi_depth = np.zeros(n_features, dtype=np.float32)
    mdi_cov = np.zeros(n_features, dtype=np.float32)

    gains_per_feature: List[List[float]] = [[] for _ in range(n_features)]
    used_in_tree_counts = np.zeros(n_features, dtype=np.float32)

    estimators = getattr(fitted_forest, "estimators_", None)
    if estimators is None:
        raise ValueError("Forest must be fitted and have estimators_.")

    for est in estimators:
        tree = est.tree_
        feat = tree.feature
        left = tree.children_left
        right = tree.children_right
        impurity = tree.impurity
        w_n = tree.weighted_n_node_samples
        root_samples = float(w_n[0]) if w_n.size else 0.0

        stack = [(0, 0)]
        used_this_tree = np.zeros(n_features, dtype=bool)

        while stack:
            node, depth = stack.pop()
            l = left[node]
            r = right[node]
            f = feat[node]
            if l == -1 or r == -1 or f < 0:
                continue

            N = float(w_n[node])
            NL = float(w_n[l])
            NR = float(w_n[r])

            delta = N * float(impurity[node]) - NL * float(impurity[l]) - NR * float(impurity[r])
            cov = N / (root_samples + eps)

            freq[f] += 1.0
            mdi_depth[f] += np.float32((depth_discount ** depth) * delta)
            mdi_cov[f] += np.float32(cov * (delta / (N + eps)))

            gains_per_feature[f].append(delta)
            used_this_tree[f] = True

            stack.append((l, depth + 1))
            stack.append((r, depth + 1))

        used_in_tree_counts += used_this_tree.astype(np.float32)

    median_gain = np.zeros(n_features, dtype=np.float32)
    for i in range(n_features):
        median_gain[i] = np.float32(np.median(gains_per_feature[i])) if gains_per_feature[i] else np.float32(0.0)

    tree_hit_rate = used_in_tree_counts / np.float32(len(estimators) + eps)

    return ExtraMDIMetricsFold(
        freq=freq,
        mdi_depth=mdi_depth,
        mdi_cov=mdi_cov,
        median_gain=median_gain,
        tree_hit_rate=tree_hit_rate,
    )


# ======================================================================================
# Stability aggregation + selection
# ======================================================================================

def stability_summary(values_by_fold: np.ndarray, eps: float = 1e-12):
    mu = values_by_fold.mean(axis=0)
    sd = values_by_fold.std(axis=0, ddof=0)
    hit = (values_by_fold > 0).mean(axis=0)
    cv = sd / (mu + eps)
    return mu, sd, hit, cv


def stability_score(mu: np.ndarray, hit: np.ndarray, cv: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return mu * (hit / (cv + eps))


@dataclass
class MDISelectionResult:
    metrics_table: pd.DataFrame
    selected_features: List[str]
    clusters: List[List[str]]
    kept_after_dedupe: List[str]


# ======================================================================================
# Main pipeline (leakage-safe) with QuantileTransformer applied to correlation subsamples
# ======================================================================================

def mdi_feature_selection_leakage_safe(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    base_model,
    n_splits: int = 6,
    purge: int = 0,
    embargo: int = 0,
    min_train_size: Optional[int] = None,
    max_depth: int = 4,
    analysis_n_estimators: int = 200,
    depth_discount: float = 0.85,
    pre_dedupe_threshold: float = 0.98,
    pearson_sample_rows: int = 50_000,
    spearman_threshold: float = 0.85,
    spearman_sample_rows: int = 30_000,
    top_n_precluster: int = 200,
    keep_top_per_cluster: int = 1,
    composite_weights: Optional[Dict[str, float]] = None,
    corr_mode: str = "anchored",  # "anchored" or "per_fold_vote"
    # NEW: QuantileTransformer config for correlation computations
    use_quantile_transform_for_corr: bool = True,
    qt_output_distribution: str = "normal",  # "normal" or "uniform"
    qt_n_quantiles: int = 1000,
    random_state: int = 0,
    sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
) -> MDISelectionResult:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if corr_mode not in ("anchored", "per_fold_vote"):
        raise ValueError("corr_mode must be 'anchored' or 'per_fold_vote'.")
    if qt_output_distribution not in ("normal", "uniform"):
        raise ValueError("qt_output_distribution must be 'normal' or 'uniform'.")

    rng = check_random_state(random_state)
    y_np = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

    sw_np = None
    if sample_weight is not None:
        sw_np = sample_weight.to_numpy() if isinstance(sample_weight, pd.Series) else np.asarray(sample_weight)

    feature_names_full = list(X.columns)

    # Downcast to float32 once
    X_np_full = np.ascontiguousarray(X.values, dtype=np.float32)
    n_samples, _p_full = X_np_full.shape

    splits = purged_embargoed_splits(
        n_samples=n_samples,
        n_splits=n_splits,
        purge=purge,
        embargo=embargo,
        min_train_size=min_train_size,
    )

    def make_analysis_model():
        m = clone(base_model)
        if hasattr(m, "set_params"):
            params = {}
            if "max_depth" in m.get_params():
                params["max_depth"] = max_depth
            if "n_estimators" in m.get_params():
                params["n_estimators"] = analysis_n_estimators
            if "random_state" in m.get_params():
                params["random_state"] = random_state
            m.set_params(**params)
        return m

    # ---- Train-only dedupe basis (Pearson) with optional QT ----
    if corr_mode == "anchored":
        train_idx0, _ = splits[0]
        X_train0 = X_np_full[train_idx0]
        Xs = _subsample_rows_np(X_train0, pearson_sample_rows, rng)
        Xs = _maybe_quantile_transform(
            Xs,
            use_qt=use_quantile_transform_for_corr,
            qt_output_distribution=qt_output_distribution,
            qt_n_quantiles=qt_n_quantiles,
            random_state=random_state,
        )
        corr_p = pearson_corrcoef(Xs)
        kept_after_dedupe = greedy_dedupe_by_corr(corr_p, feature_names_full, pre_dedupe_threshold)
    else:
        kept_sets: List[List[str]] = []
        for train_idx, _ in splits:
            Xtr = X_np_full[train_idx]
            Xs = _subsample_rows_np(Xtr, pearson_sample_rows, rng)
            Xs = _maybe_quantile_transform(
                Xs,
                use_qt=use_quantile_transform_for_corr,
                qt_output_distribution=qt_output_distribution,
                qt_n_quantiles=qt_n_quantiles,
                random_state=random_state,
            )
            corr_p = pearson_corrcoef(Xs)
            kept_sets.append(greedy_dedupe_by_corr(corr_p, feature_names_full, pre_dedupe_threshold))
        kept_after_dedupe = sorted(set().union(*kept_sets), key=lambda f: feature_names_full.index(f))

    kept_idx = np.array([feature_names_full.index(f) for f in kept_after_dedupe], dtype=int)
    X_np = X_np_full[:, kept_idx]
    feature_names = kept_after_dedupe
    p = X_np.shape[1]

    # ---- Fit per fold + extract MDI metrics ----
    share_folds, freq_folds, depth_folds, cov_folds, med_folds, thr_folds = [], [], [], [], [], []

    for train_idx, _ in splits:
        m = make_analysis_model()
        Xtr = X_np[train_idx]
        ytr = y_np[train_idx]
        sw_tr = sw_np[train_idx] if sw_np is not None else None

        m.fit(Xtr, ytr, sample_weight=sw_tr)

        share = np.asarray(getattr(m, "feature_importances_", np.zeros(p)), dtype=np.float32)
        extra = extract_extra_mdi_metrics_from_forest(m, n_features=p, depth_discount=depth_discount)

        share_folds.append(share)
        freq_folds.append(extra.freq)
        depth_folds.append(extra.mdi_depth)
        cov_folds.append(extra.mdi_cov)
        med_folds.append(extra.median_gain)
        thr_folds.append(extra.tree_hit_rate)

    share_k = np.vstack(share_folds)
    freq_k = np.vstack(freq_folds)
    depth_k = np.vstack(depth_folds)
    cov_k = np.vstack(cov_folds)
    med_k = np.vstack(med_folds)
    thr_k = np.vstack(thr_folds)

    agg = {}
    for name, arr in [
        ("share", share_k),
        ("freq", freq_k),
        ("mdi_depth", depth_k),
        ("mdi_cov", cov_k),
        ("median_gain", med_k),
        ("tree_hit_rate", thr_k),
    ]:
        mu, sd, hit, cv = stability_summary(arr)
        agg[f"{name}_mu"] = mu
        agg[f"{name}_sd"] = sd
        agg[f"{name}_hit"] = hit
        agg[f"{name}_cv"] = cv
        agg[f"{name}_stab"] = stability_score(mu, hit, cv)

    metrics = pd.DataFrame(agg, index=feature_names)

    if composite_weights is None:
        composite_weights = {
            "share_stab": 0.40,
            "mdi_depth_stab": 0.25,
            "mdi_cov_stab": 0.20,
            "tree_hit_rate_stab": 0.10,
            "median_gain_stab": 0.05,
        }

    rank_df = pd.DataFrame(index=metrics.index)
    for metric_name in composite_weights:
        rank_df[metric_name] = metrics[metric_name].rank(ascending=False, method="average")

    weighted_rank = sum(composite_weights[m] * rank_df[m] for m in composite_weights)
    metrics["composite_rank"] = weighted_rank.rank(ascending=True, method="average")
    metrics = metrics.sort_values("composite_rank")

    shortlist = metrics.head(min(top_n_precluster, len(metrics))).index.tolist()
    shortlist_idx = np.array([feature_names.index(f) for f in shortlist], dtype=int)

    # ---- Train-only Spearman clustering on shortlist with optional QT on the sample ----
    if corr_mode == "anchored":
        train_idx0, _ = splits[0]
        X_train0_short = X_np[train_idx0][:, shortlist_idx]
        Xs = _subsample_rows_np(X_train0_short, spearman_sample_rows, rng)
        Xs = _maybe_quantile_transform(
            Xs,
            use_qt=use_quantile_transform_for_corr,
            qt_output_distribution=qt_output_distribution,
            qt_n_quantiles=qt_n_quantiles,
            random_state=random_state,
        )
        corr_s = spearman_corrcoef(Xs)
        clusters = clusters_from_corr(corr_s, shortlist, spearman_threshold)

        selected: List[str] = []
        for cluster in clusters:
            ranked = metrics.loc[cluster].sort_values("composite_rank").index.tolist()
            selected.extend(ranked[:keep_top_per_cluster])
        selected = sorted(set(selected), key=lambda f: float(metrics.loc[f, "composite_rank"]))

        return MDISelectionResult(metrics_table=metrics, selected_features=selected, clusters=clusters, kept_after_dedupe=kept_after_dedupe)

    else:
        winner_votes: Dict[str, int] = {}
        for train_idx, _ in splits:
            Xtr_short = X_np[train_idx][:, shortlist_idx]
            Xs = _subsample_rows_np(Xtr_short, spearman_sample_rows, rng)
            Xs = _maybe_quantile_transform(
                Xs,
                use_qt=use_quantile_transform_for_corr,
                qt_output_distribution=qt_output_distribution,
                qt_n_quantiles=qt_n_quantiles,
                random_state=random_state,
            )
            corr_s = spearman_corrcoef(Xs)
            fold_clusters = clusters_from_corr(corr_s, shortlist, spearman_threshold)
            for cluster in fold_clusters:
                ranked = metrics.loc[cluster].sort_values("composite_rank").index.tolist()
                for f in ranked[:keep_top_per_cluster]:
                    winner_votes[f] = winner_votes.get(f, 0) + 1

        selected = sorted(winner_votes.keys(), key=lambda f: (-winner_votes[f], float(metrics.loc[f, "composite_rank"])))
        metrics = metrics.assign(votes=pd.Series(winner_votes, dtype=float)).fillna({"votes": 0}).sort_values(
            ["votes", "composite_rank"], ascending=[False, True]
        )

        return MDISelectionResult(metrics_table=metrics, selected_features=selected, clusters=[shortlist], kept_after_dedupe=kept_after_dedupe)


# ======================================================================================
# Example usage
# ======================================================================================
if __name__ == "__main__":
    from sklearn.ensemble import ExtraTreesClassifier

    rng = np.random.RandomState(0)
    n, p = 120_000, 800
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    # Inject heavy tails / outliers in a couple features
    X["f0"] = X["f0"] * 50
    X["f1"] = np.tanh(X["f1"]) * 10

    y = (0.6 * X["f3"] - 0.4 * X["f7"] + 0.2 * rng.randn(n) > 0).astype(int)

    base_model = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=200,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )

    res = mdi_feature_selection_leakage_safe(
        X=X,
        y=y,
        base_model=base_model,
        n_splits=6,
        purge=5,
        min_train_size=40_000,
        max_depth=4,
        analysis_n_estimators=200,
        use_quantile_transform_for_corr=True,
        qt_output_distribution="normal",
        qt_n_quantiles=1000,
        corr_mode="anchored",
        random_state=0,
    )

    print("Selected:", len(res.selected_features))
    print("Top 20:", res.selected_features[:20])
