"""Fold-safe feature pruning + ElasticNet tuning for time-series pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from extreme_price_movements.purged_cv import PurgedKFold


@dataclass
class FoldSelectionResult:
    fold_id: int
    selected_features_fold: List[str]
    alpha: float
    l1_ratio: float
    inner_cv_table_fold: List[Dict[str, float]]


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=np.int32)
    if finite.sum() <= 1:
        return out
    v = x[finite]
    ranks = np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort").astype(np.float64)
    denom = max(len(v) - 1, 1)
    r01 = ranks / denom
    b = np.floor(r01 * float(n_bins)).astype(np.int32)
    b = np.clip(b, 0, n_bins - 1)
    out[finite] = b
    return out


def _mutual_information_discrete(x_bin: np.ndarray, y_bin: np.ndarray, n_x: int, n_y: int, eps: float = 1e-12) -> float:
    x = np.asarray(x_bin, dtype=np.int32)
    y = np.asarray(y_bin, dtype=np.int32)
    valid = (x >= 0) & (x < n_x) & (y >= 0) & (y < n_y)
    if valid.sum() == 0:
        return 0.0
    x = x[valid]
    y = y[valid]
    joint = np.zeros((n_x, n_y), dtype=np.float64)
    np.add.at(joint, (x, y), 1.0)
    joint += eps
    joint /= float(joint.sum())
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    ratio = joint / np.maximum(px @ py, eps)
    return float(np.sum(joint * np.log(np.maximum(ratio, eps))))


def _connected_components_from_corr(corr_abs: np.ndarray, threshold: float) -> List[List[int]]:
    n = corr_abs.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            neigh = np.where((corr_abs[u] >= threshold) & (~seen))[0]
            for v in neigh.tolist():
                seen[v] = True
                stack.append(v)
        comps.append(sorted(comp))
    return comps


def _winsorize_1d(x: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if finite.sum() <= 2:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.quantile(arr[finite], q_low))
    hi = float(np.quantile(arr[finite], q_high))
    return np.clip(arr, lo, hi)


def _to_day_keys(ts: np.ndarray) -> np.ndarray:
    t = np.asarray(ts)
    if t.size == 0:
        return np.asarray([], dtype='datetime64[D]')
    td = t.astype('datetime64[ns]')
    return td.astype('datetime64[D]')


def _topq_daily_aggregated_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q: float,
    times: Optional[np.ndarray] = None,
    clip_score_q: Optional[Tuple[float, float]] = (0.01, 0.99),
) -> float:
    n = len(y_true)
    if n == 0:
        return float('-inf')
    y = _winsorize_1d(np.asarray(y_true, dtype=np.float64), 0.01, 0.99)
    pred = np.asarray(y_pred, dtype=np.float64)
    k = max(1, int(np.ceil(n * float(q))))
    idx = np.argsort(pred)[-k:]
    if times is None:
        v = y[idx]
        if clip_score_q is not None and len(v) > 3:
            lo, hi = clip_score_q
            v = np.clip(v, np.quantile(v, lo), np.quantile(v, hi))
        return float(np.nanmean(v))

    day_keys = _to_day_keys(np.asarray(times)[idx])
    vals = y[idx]
    if len(day_keys) == 0:
        return float(np.nanmean(vals))
    unique_days, inv = np.unique(day_keys, return_inverse=True)
    daily = np.zeros(len(unique_days), dtype=np.float64)
    np.add.at(daily, inv, vals)
    if clip_score_q is not None and len(daily) > 3:
        lo, hi = clip_score_q
        daily = np.clip(daily, np.quantile(daily, lo), np.quantile(daily, hi))
    return float(np.nanmean(daily))


def _deterministic_subsample_idx(n: int, max_n: int = 5000) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    return np.linspace(0, n - 1, max_n, dtype=np.int32)


def _bootstrap_mi_stats(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bins_x: int,
    n_bins_y: int,
    n_bootstrap: int = 5,
    max_bootstrap_samples: int = 3000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mi_mean, mi_std) from bootstrap resamples for feature-stability scoring."""
    Xf = np.asarray(X, dtype=np.float32)
    yf = np.asarray(y, dtype=np.float32)
    n, f = Xf.shape
    if n == 0 or f == 0:
        return np.zeros(f, dtype=np.float64), np.zeros(f, dtype=np.float64)

    n_take = min(int(max_bootstrap_samples), int(n))
    rng = np.random.RandomState(int(random_state))
    mi_boot = np.zeros((max(int(n_bootstrap), 1), f), dtype=np.float64)

    for b in range(mi_boot.shape[0]):
        if n_take < n:
            idx = np.sort(rng.choice(n, size=n_take, replace=False))
        else:
            idx = np.arange(n, dtype=np.int32)
        Xb = Xf[idx]
        yb = yf[idx]
        y_bin = _quantile_bin(yb, n_bins=n_bins_y)
        for j in range(f):
            x_bin = _quantile_bin(Xb[:, j], n_bins=n_bins_x)
            mi_boot[b, j] = _mutual_information_discrete(x_bin, y_bin, n_bins_x, n_bins_y)

    return np.mean(mi_boot, axis=0), np.std(mi_boot, axis=0, ddof=0)


def run_fold_safe_feature_pruning_and_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    timestamps: Optional[np.ndarray] = None,
    *,
    corr_threshold: float = 0.95,
    n_bins_x: int = 16,
    n_bins_y: int = 16,
    l1_ratio_grid: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.6),
    alpha_grid: Optional[Sequence[float]] = None,
    outer_splits: int = 4,
    inner_splits: int = 4,
    top_q: float = 0.10,
    max_samples: int = 5000,
    random_state: int = 42,
    n_bootstrap_mi: int = 5,
    max_bootstrap_samples: int = 3000,
    stability_penalty_weight: float = 0.10,
) -> Dict[str, Any]:
    """Run leakage-safe clustering+MI feature selection and ElasticNet tuning."""
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 1.5, 12)

    Xf = np.asarray(X, dtype=np.float32)
    yf = np.asarray(y, dtype=np.float32)
    n, f = Xf.shape
    if f == 0:
        return {"selected_features": [], "alpha": 1e-2, "l1_ratio": 0.1, "fold_results": []}

    times = None if timestamps is None else np.asarray(timestamps)
    outer = PurgedKFold(
        n_splits=max(2, int(outer_splits)),
        purge=43200 if times is not None else 12,
        embargo=43200 if times is not None else 12,
        times=times,
    )

    fold_results: List[FoldSelectionResult] = []
    feat_to_count: Dict[str, int] = {}
    alphas: List[float] = []
    l1s: List[float] = []

    outer_folds = [(np.asarray(tr, dtype=np.int32), np.asarray(va, dtype=np.int32)) for tr, va in outer.split(Xf)]

    for fold_id, (tr_idx, _va_idx) in enumerate(outer_folds, start=1):
        tr_idx = np.asarray(tr_idx, dtype=np.int32)
        sub_idx = _deterministic_subsample_idx(len(tr_idx), max_n=max_samples)
        tr_idx = tr_idx[sub_idx]

        Xtr = Xf[tr_idx]
        ytr = yf[tr_idx]

        corr = np.nan_to_num(np.corrcoef(np.argsort(np.argsort(Xtr, axis=0), axis=0).astype(np.float64), rowvar=False), nan=0.0)
        corr_abs = np.abs(corr)
        np.fill_diagonal(corr_abs, 1.0)
        clusters = _connected_components_from_corr(corr_abs, threshold=float(corr_threshold))

        mi_vals, mi_std = _bootstrap_mi_stats(
            Xtr,
            ytr,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            n_bootstrap=max(1, int(n_bootstrap_mi)),
            max_bootstrap_samples=max(500, int(max_bootstrap_samples)),
            random_state=int(random_state + fold_id),
        )
        miss_rate = np.mean(~np.isfinite(Xtr), axis=0)

        selected_idx: List[int] = []
        all_idx = np.arange(f, dtype=np.int32)
        for comp in clusters:
            comp_arr = np.asarray(comp, dtype=np.int32)
            outside = np.setdiff1d(all_idx, comp_arr)
            mean_out_corr = np.zeros(len(comp_arr), dtype=np.float64)
            if outside.size > 0:
                mean_out_corr = corr_abs[np.ix_(comp_arr, outside)].mean(axis=1)
            order = sorted(
                range(len(comp_arr)),
                key=lambda i: (
                    -float(mi_vals[comp_arr[i]]),
                    float(mi_std[comp_arr[i]]),
                    float(miss_rate[comp_arr[i]]),
                    float(mean_out_corr[i]),
                    str(feature_names[int(comp_arr[i])]),
                ),
            )
            selected_idx.append(int(comp_arr[order[0]]))

        selected_idx = sorted(set(selected_idx))
        sel_names = [str(feature_names[i]) for i in selected_idx]

        inner_times = times[tr_idx] if times is not None and len(times) == n else None
        inner = PurgedKFold(
            n_splits=max(2, int(inner_splits)),
            purge=43200 if inner_times is not None else 12,
            embargo=43200 if inner_times is not None else 12,
            times=inner_times,
        )

        rows: List[Dict[str, float]] = []
        best_mean = -np.inf
        best_key = None

        Xsel = np.asarray(Xtr[:, selected_idx], dtype=np.float32, order='C')
        inner_folds = [(np.asarray(tr, dtype=np.int32), np.asarray(va, dtype=np.int32)) for tr, va in inner.split(Xsel)]
        fold_payload = []
        for in_tr, in_va in inner_folds:
            fold_payload.append((
                Xsel[in_tr],
                ytr[in_tr],
                Xsel[in_va],
                ytr[in_va],
                (inner_times[in_va] if inner_times is not None and len(inner_times) == len(Xsel) else None),
            ))

        for alpha in alpha_grid:
            for l1r in l1_ratio_grid:
                scores = []
                for x_tr, y_tr, x_va, y_va, fold_times in fold_payload:
                    scaler = StandardScaler()
                    x_tr_s = scaler.fit_transform(x_tr)
                    x_va_s = scaler.transform(x_va)
                    enet = ElasticNet(
                        alpha=float(alpha),
                        l1_ratio=float(l1r),
                        fit_intercept=True,
                        max_iter=10000,
                        tol=1e-4,
                        random_state=random_state,
                    )
                    enet.fit(x_tr_s, y_tr)
                    yhat = enet.predict(x_va_s)
                    scores.append(_topq_daily_aggregated_score(y_va, yhat, q=top_q, times=fold_times, clip_score_q=(0.01, 0.99)))
                n_scores = len(scores)
                if n_scores == 0:
                    continue
                mean_s = float(np.mean(scores)) if n_scores else float('-inf')
                std_s = float(np.std(scores, ddof=1)) if n_scores > 1 else 0.0
                se_s = std_s / np.sqrt(n_scores) if n_scores else 0.0
                adj_s = float(mean_s - float(stability_penalty_weight) * std_s)
                row = {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1r),
                    "mean_score": mean_s,
                    "std_score": std_s,
                    "se_score": se_s,
                    "adjusted_score": adj_s,
                }
                rows.append(row)
                key = (float(alpha), float(l1r))
                if (adj_s > best_mean) or (np.isclose(adj_s, best_mean) and (best_key is None or key > best_key)):
                    best_mean = adj_s
                    best_key = key

        if not rows:
            chosen_alpha, chosen_l1 = float(alpha_grid[0]), float(l1_ratio_grid[0])
            table_out: List[Dict[str, float]] = []
        else:
            best_row = max(rows, key=lambda r: (r["adjusted_score"], r["alpha"], r["l1_ratio"]))
            cutoff = float(best_row["adjusted_score"] - best_row["se_score"])
            eps = 1e-12
            eligible = [r for r in rows if r["adjusted_score"] >= (cutoff - eps)]
            chosen = max(eligible, key=lambda r: (r["alpha"], r["l1_ratio"], r["adjusted_score"]))
            chosen_alpha, chosen_l1 = float(chosen["alpha"]), float(chosen["l1_ratio"])
            table_out = []
            for r in rows:
                rr = dict(r)
                rr["eligible"] = 1.0 if r in eligible else 0.0
                table_out.append(rr)

        fold_results.append(
            FoldSelectionResult(
                fold_id=fold_id,
                selected_features_fold=sel_names,
                alpha=chosen_alpha,
                l1_ratio=chosen_l1,
                inner_cv_table_fold=table_out,
            )
        )
        for nm in sel_names:
            feat_to_count[nm] = feat_to_count.get(nm, 0) + 1
        alphas.append(chosen_alpha)
        l1s.append(chosen_l1)

    if not fold_results:
        selected_final = [str(nm) for nm in feature_names]
        alpha_final = float(np.median(np.asarray(alpha_grid, dtype=np.float64)))
        l1_final = float(np.median(np.asarray(l1_ratio_grid, dtype=np.float64)))
    else:
        min_count = max(1, int(np.ceil(len(fold_results) * 0.5)))
        selected_final = sorted(
            [nm for nm in feature_names if feat_to_count.get(str(nm), 0) >= min_count],
            key=lambda x: (-feat_to_count.get(str(x), 0), str(x)),
        )
        if not selected_final:
            selected_final = [str(min(feature_names))]
        alpha_final = float(np.median(np.asarray(alphas, dtype=np.float64)))
        l1_final = float(np.median(np.asarray(l1s, dtype=np.float64)))

    return {
        "selected_features": selected_final,
        "alpha": alpha_final,
        "l1_ratio": l1_final,
        "stability": {"n_bootstrap_mi": int(n_bootstrap_mi), "stability_penalty_weight": float(stability_penalty_weight)},
        "fold_results": [
            {
                "fold_id": fr.fold_id,
                "selected_features_fold": fr.selected_features_fold,
                "alpha": fr.alpha,
                "l1_ratio": fr.l1_ratio,
                "inner_cv_table_fold": fr.inner_cv_table_fold,
            }
            for fr in fold_results
        ],
    }
