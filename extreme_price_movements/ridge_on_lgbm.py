from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

ALPHA_GRID = (0.05, 0.2, 0.5, 1.0)
L1_RATIO_GRID = (0.6, 0.8, 0.9, 0.95)
LEAF_MODEL_SPECS = (
    {"max_depth": 3, "leaf_frac": 0.02, "prefix": "LGBM_2P_LEAF"},
    {"max_depth": 3, "leaf_frac": 0.04, "prefix": "LGBM_4P_LEAF"},
    {"max_depth": 3, "leaf_frac": 0.06, "prefix": "LGBM_6P_LEAF"},
    {"max_depth": 3, "leaf_frac": 0.08, "prefix": "LGBM_8P_LEAF"},
    {"max_depth": 4, "leaf_frac": 0.02, "prefix": "LGBM_D4_2P_LEAF"},
    {"max_depth": 4, "leaf_frac": 0.05, "prefix": "LGBM_D4_5P_LEAF"},
)


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _stratified_subsample_indices(
    y: np.ndarray, max_n: int = 15000, random_state: int = 42
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int8)
    idx_all = np.arange(len(y_arr), dtype=np.int32)
    if len(y_arr) <= max_n:
        return idx_all
    idx_sub, _ = train_test_split(
        idx_all,
        train_size=max_n,
        stratify=y_arr,
        random_state=random_state,
    )
    return np.asarray(idx_sub, dtype=np.int32)


def top30_boundary_weight(pred: np.ndarray) -> np.ndarray:
    rank_pct = (
        pd.Series(np.asarray(pred, dtype=np.float32))
        .rank(pct=True)
        .to_numpy(dtype=np.float32)
    )
    center = 0.75
    sigma = 0.10
    boundary = np.exp(-((rank_pct - center) ** 2) / (2 * sigma**2))
    topness = np.clip((rank_pct - 0.70) / 0.30, 0.0, 1.0)
    w = 1.0 + 1.5 * boundary + 0.5 * topness
    return np.clip(w, 1.0, 3.0).astype(np.float32)


def _expected_calibration_error(
    y_true: np.ndarray, pred: np.ndarray, n_bins: int = 10
) -> float:
    y = np.asarray(y_true, dtype=np.float32)
    p = np.asarray(pred, dtype=np.float32)
    if len(y) == 0:
        return 0.0
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        if np.any(mask):
            acc = float(np.mean(y[mask]))
            conf = float(np.mean(p[mask]))
            ece += (float(np.sum(mask)) / float(len(y))) * abs(acc - conf)
    return float(ece)


def _metric_pack(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(pred, dtype=np.float64)
    k = max(1, int(0.30 * len(y)))
    idx = np.argsort(p)[-k:]
    top_rate = float(np.mean(y[idx])) if len(idx) else 0.0
    base_rate = float(np.mean(y)) if len(y) else 0.0
    lift30 = top_rate / max(base_rate, 1e-6)
    auc_correct_30 = 0.5
    if len(np.unique(y[idx])) > 1:
        try:
            auc_correct_30 = float(roc_auc_score(y[idx], p[idx]))
        except Exception:
            auc_correct_30 = 0.5

    q = np.quantile(p[idx], np.linspace(0, 1, 6)) if len(idx) >= 10 else None
    if q is None:
        stability30_proxy = 0.0
    else:
        vals = []
        sp = p[idx]
        sy = y[idx]
        for i in range(5):
            m = (sp >= q[i]) & (sp < q[i + 1] if i < 4 else sp <= q[i + 1])
            if np.any(m):
                vals.append(float(np.mean(sy[m])))
        stability30_proxy = float(1.0 / (1.0 + np.std(vals))) if vals else 0.0

    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    auc_random = auc / 0.5
    pr = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else base_rate
    pr_rand = pr / max(base_rate, 1e-6)
    brier = float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    ece = _expected_calibration_error(y, p)
    return {
        "lift30": lift30,
        "auc_correct_30": auc_correct_30,
        "stability30_proxy": stability30_proxy,
        "auc": auc,
        "auc_random": auc_random,
        "pr_auc": pr,
        "pr_random": pr_rand,
        "brier": brier,
        "ece": ece,
        "top30_correctness_rate": top_rate,
        "overall_correctness_rate": base_rate,
        "oof_std": float(np.std(p)),
    }


def _fold_j(metrics: dict[str, float]) -> float:
    return 0.6 * float(metrics["lift30"]) + 0.4 * float(metrics["auc_correct_30"])


def _aggregate_j(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not fold_metrics:
        return {
            "lift30": 0.0,
            "auc_correct_30": 0.0,
            "stability30": 0.0,
            "J_mean": 0.0,
            "J_std": 0.0,
            "J_final": 0.0,
        }

    lift30 = float(np.mean([m["lift30"] for m in fold_metrics]))
    auc_correct_30 = float(np.mean([m["auc_correct_30"] for m in fold_metrics]))
    j_vals = np.array([_fold_j(m) for m in fold_metrics], dtype=float)
    stability30 = float(np.mean(j_vals) - 2.0 * np.std(j_vals))
    j_final = 0.4 * lift30 + 0.2 * auc_correct_30 + 0.4 * stability30

    return {
        "lift30": lift30,
        "auc_correct_30": auc_correct_30,
        "stability30": stability30,
        "J_mean": float(np.mean(j_vals)),
        "J_std": float(np.std(j_vals)),
        "J_final": float(j_final),
    }


def _pruning_thresholds(round_id: int) -> tuple[float, float]:
    if round_id <= 2:
        return 0.60, 0.20
    if round_id <= 4:
        return 0.70, 0.30
    return 0.80, 0.40


def _floor_keep(round_id: int, current_feature_count: int) -> int:
    floor_schedule = {1: 0.80, 2: 0.70, 3: 0.60, 4: 0.50, 5: 0.40, 6: 0.30, 7: 0.20}
    frac = float(floor_schedule.get(round_id, 0.20))
    return int(
        min(current_feature_count, max(40, np.ceil(frac * current_feature_count)))
    )


def _point_line_distance(px, py, x1, y1, x2, y2) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return float(np.sqrt((px - x1) ** 2 + (py - y1) ** 2))
    return float(
        abs(dy * px - dx * py + x2 * y1 - y2 * x1) / np.sqrt(dx * dx + dy * dy)
    )


def _pareto_frontier(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, a in enumerate(cands):
        dominated = False
        for j, b in enumerate(cands):
            if i == j:
                continue
            if (
                float(b["J_final"]) >= float(a["J_final"])
                and float(b["stability30"]) >= float(a["stability30"])
                and (
                    float(b["J_final"]) > float(a["J_final"])
                    or float(b["stability30"]) > float(a["stability30"])
                )
            ):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


def _select_elbow_candidate(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    if len(frontier) == 1:
        return frontier[0]
    if len(frontier) == 2:
        return sorted(frontier, key=lambda z: float(z["J_final"]))[-1]

    perf = np.array([float(c["J_final"]) for c in frontier], dtype=float)
    stab = np.array([float(c["stability30"]) for c in frontier], dtype=float)
    perf_n = (perf - float(np.min(perf))) / max(
        float(np.max(perf) - np.min(perf)), 1e-12
    )
    stab_n = (stab - float(np.min(stab))) / max(
        float(np.max(stab) - np.min(stab)), 1e-12
    )

    order = np.argsort(perf_n)
    perf_n = perf_n[order]
    stab_n = stab_n[order]
    frontier_sorted = [frontier[i] for i in order]

    x1, y1 = float(perf_n[0]), float(stab_n[0])
    x2, y2 = float(perf_n[-1]), float(stab_n[-1])
    dists = np.array(
        [
            _point_line_distance(float(px), float(py), x1, y1, x2, y2)
            for px, py in zip(perf_n, stab_n)
        ],
        dtype=float,
    )
    best_idx = int(np.argmax(dists))
    return frontier_sorted[best_idx]


def _quick_cv_score(
    x_sel: sparse.csr_matrix,
    y: np.ndarray,
    random_state: int,
    alpha: float,
    l1_ratio: float,
    prior_round_pred: np.ndarray | None = None,
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=np.int8)
    prior = (
        np.asarray(prior_round_pred, dtype=np.float32)
        if prior_round_pred is not None
        else np.full(len(y_arr), float(np.mean(y_arr)), dtype=np.float32)
    )
    w = top30_boundary_weight(prior)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []
    for tr_idx, va_idx in cv.split(np.zeros(len(y_arr), dtype=np.int8), y_arr):
        x_tr = x_sel[tr_idx]
        y_tr = y_arr[tr_idx]
        w_tr = w[tr_idx]
        sub_idx = _stratified_subsample_indices(
            y_tr, max_n=15000, random_state=random_state
        )
        lr = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=2000,
            C=1.0 / max(alpha, 1e-6),
            l1_ratio=float(l1_ratio),
            random_state=random_state,
        )
        lr.fit(x_tr[sub_idx], y_tr[sub_idx], sample_weight=w_tr[sub_idx])
        pv = lr.predict_proba(x_sel[va_idx])[:, 1]
        fold_metrics.append(_metric_pack(y_arr[va_idx], pv))
    return _aggregate_j(fold_metrics)


def _feature_abs_spearman_scores(
    x_mat: sparse.csr_matrix,
    y: np.ndarray,
    feat_idx: np.ndarray,
    row_idx: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_mat[row_idx][:, feat_idx].toarray(), dtype=np.float32)
    y_s = np.asarray(y[row_idx], dtype=np.float32)
    if x.size == 0:
        return np.zeros(len(feat_idx), dtype=np.float32)
    x_rank = np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float32)
    y_rank = rankdata(y_s, method="average").astype(np.float32)
    x_rank -= np.mean(x_rank, axis=0, keepdims=True)
    y_rank -= np.mean(y_rank)
    x_std = np.std(x_rank, axis=0) + 1e-12
    y_std = float(np.std(y_rank) + 1e-12)
    corr = np.mean(x_rank * y_rank.reshape(-1, 1), axis=0) / (x_std * y_std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(corr).astype(np.float32)


def _greedy_drop_from_pairs(
    n_feats: int, pairs: np.ndarray, col_strength: np.ndarray
) -> np.ndarray:
    keep = np.ones(n_feats, dtype=bool)
    if len(pairs) == 0:
        return keep
    for i, j in pairs:
        if not (keep[i] and keep[j]):
            continue
        if col_strength[i] <= col_strength[j]:
            keep[j] = False
        else:
            keep[i] = False
    return keep


def _prune_raw_redundancy(
    x_mat: sparse.csr_matrix,
    feat_idx: np.ndarray,
    row_idx: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(feat_idx) <= 1:
        return feat_idx
    x = np.asarray(x_mat[row_idx][:, feat_idx].toarray(), dtype=np.float32)
    x_rank = np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float32)
    corr = np.abs(np.corrcoef(x_rank, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu(corr > threshold, k=1)
    pairs = np.column_stack(np.where(upper))
    strength = np.sum(corr, axis=0)
    keep_mask = _greedy_drop_from_pairs(len(feat_idx), pairs, strength)
    return feat_idx[keep_mask]


def _prune_leaf_overlap_redundancy(
    x_mat: sparse.csr_matrix,
    feat_idx: np.ndarray,
    row_idx: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(feat_idx) <= 1:
        return feat_idx
    x = x_mat[row_idx][:, feat_idx].copy().astype(np.int8)
    x.data = np.ones_like(x.data, dtype=np.int8)
    supports = np.asarray(x.sum(axis=0)).reshape(-1).astype(np.float32)
    inter = (x.T @ x).tocoo()
    valid = inter.row < inter.col
    rows = inter.row[valid]
    cols = inter.col[valid]
    vals = inter.data[valid].astype(np.float32)
    denom = np.minimum(supports[rows], supports[cols]) + 1e-12
    overlap = vals / denom
    sel = overlap > threshold
    pairs = np.column_stack([rows[sel], cols[sel]])
    keep_mask = _greedy_drop_from_pairs(len(feat_idx), pairs, supports)
    return feat_idx[keep_mask]


def _non_overlapping_stratified_buckets(
    y: np.ndarray, n_buckets: int = 5, bucket_size: int = 3000, random_state: int = 42
) -> list[np.ndarray]:
    rng = np.random.RandomState(random_state)
    y_arr = np.asarray(y, dtype=np.int8)
    pos = np.where(y_arr == 1)[0]
    neg = np.where(y_arr == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    p_ratio = float(np.mean(y_arr))
    n_pos_each = int(min(len(pos) // n_buckets, max(1, round(bucket_size * p_ratio))))
    n_neg_each = int(min(len(neg) // n_buckets, max(1, bucket_size - n_pos_each)))
    buckets: list[np.ndarray] = []
    p_ptr = 0
    n_ptr = 0
    for _ in range(n_buckets):
        p_take = pos[p_ptr : p_ptr + n_pos_each]
        n_take = neg[n_ptr : n_ptr + n_neg_each]
        p_ptr += len(p_take)
        n_ptr += len(n_take)
        if len(p_take) + len(n_take) == 0:
            continue
        b = np.concatenate([p_take, n_take]).astype(np.int32, copy=False)
        rng.shuffle(b)
        buckets.append(b)
    return buckets


def _prescreen_features(
    x_mat: sparse.csr_matrix,
    y: np.ndarray,
    feat_idx: np.ndarray,
    random_state: int,
    n_raw_total: int,
) -> np.ndarray:
    n_total = len(feat_idx)
    if n_total <= 600:
        return feat_idx
    row_idx_redund = _stratified_subsample_indices(
        y, max_n=min(3000, len(y)), random_state=random_state + 5
    )
    raw_mask = feat_idx < n_raw_total
    leaf_mask = ~raw_mask
    feat_raw = feat_idx[raw_mask]
    feat_leaf = feat_idx[leaf_mask]
    feat_raw = _prune_raw_redundancy(x_mat, feat_raw, row_idx_redund, threshold=0.98)
    feat_leaf = _prune_leaf_overlap_redundancy(
        x_mat, feat_leaf, row_idx_redund, threshold=0.98
    )
    feat_idx = np.concatenate([feat_raw, feat_leaf]).astype(np.int32, copy=False)
    if len(feat_idx) <= 600:
        return feat_idx
    row_idx_stage1 = _stratified_subsample_indices(
        y, max_n=min(5000, len(y)), random_state=random_state
    )
    scores_1 = _feature_abs_spearman_scores(x_mat, y, feat_idx, row_idx_stage1)
    nkept1 = min(n_total, max(600, int(0.25 * n_total + 200)))
    top1 = np.argsort(scores_1)[-nkept1:]
    feat_stage1 = feat_idx[top1]
    raw_mask_1 = feat_stage1 < n_raw_total
    leaf_mask_1 = ~raw_mask_1
    feat_raw_1 = _prune_raw_redundancy(
        x_mat, feat_stage1[raw_mask_1], row_idx_stage1, threshold=0.97
    )
    feat_leaf_1 = _prune_leaf_overlap_redundancy(
        x_mat, feat_stage1[leaf_mask_1], row_idx_stage1, threshold=0.97
    )
    feat_stage1 = np.concatenate([feat_raw_1, feat_leaf_1]).astype(np.int32, copy=False)
    n1 = len(feat_stage1)
    if n1 <= 1:
        return feat_stage1

    buckets = _non_overlapping_stratified_buckets(
        y, n_buckets=5, bucket_size=3000, random_state=random_state + 17
    )
    if len(buckets) == 0:
        return feat_stage1
    bucket_scores = np.zeros((len(buckets), n1), dtype=np.float32)
    for b_i, b_idx in enumerate(buckets):
        b_idx = b_idx[: min(len(b_idx), 3000)]
        bucket_scores[b_i] = _feature_abs_spearman_scores(x_mat, y, feat_stage1, b_idx)
    mean_score = np.mean(bucket_scores, axis=0)
    std_score = np.std(bucket_scores, axis=0)
    stable_score = mean_score - 0.5 * std_score
    nkept2 = min(n1, max(1, int(0.25 * n1 + 100)))
    top2 = np.argsort(stable_score)[-nkept2:]
    return feat_stage1[top2]


def _stage_a_prune(
    X_combined: sparse.csr_matrix,
    y: np.ndarray,
    random_state: int,
    n_raw_total: int,
    prior_round_pred: np.ndarray | None,
    initial_features: np.ndarray | None = None,
    max_rounds: int = 7,
) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=np.int8)
    n = len(y_arr)
    active_idx = (
        np.asarray(initial_features, dtype=np.int32)
        if initial_features is not None
        else np.arange(X_combined.shape[1], dtype=np.int32)
    )
    active_idx = _prescreen_features(
        X_combined,
        y_arr,
        active_idx,
        random_state=random_state,
        n_raw_total=n_raw_total,
    )

    round_history: list[dict[str, Any]] = []
    last_round_oof = (
        np.asarray(prior_round_pred, dtype=np.float32)
        if prior_round_pred is not None
        else np.full(n, float(np.mean(y_arr)), dtype=np.float32)
    )

    golden_floor_score = None
    golden_floor_se = None

    for round_id in range(1, max_rounds + 1):
        if len(active_idx) <= 40:
            break
        prev_active_idx = np.asarray(active_idx, dtype=np.int32).copy()
        prev_round_oof = np.asarray(last_round_oof, dtype=np.float32).copy()
        min_freq, min_net_support = _pruning_thresholds(round_id)
        x_round = X_combined[:, active_idx]
        w_round = top30_boundary_weight(last_round_oof)

        fold_cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=random_state + round_id
        )
        hp_records: list[dict[str, Any]] = []

        for alpha in ALPHA_GRID:
            for l1_ratio in L1_RATIO_GRID:
                fold_coefs: list[np.ndarray] = []
                fold_metrics: list[dict[str, float]] = []
                round_oof = np.zeros(n, dtype=np.float32)
                ok = True
                for tr_idx, va_idx in fold_cv.split(np.zeros(n, dtype=np.int8), y_arr):
                    x_tr = x_round[tr_idx]
                    y_tr = y_arr[tr_idx]
                    w_tr = w_round[tr_idx]
                    sub_idx = _stratified_subsample_indices(
                        y_tr, max_n=15000, random_state=random_state + round_id
                    )
                    lr = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        max_iter=3000,
                        C=1.0 / max(alpha, 1e-6),
                        l1_ratio=float(l1_ratio),
                        random_state=random_state,
                    )
                    try:
                        lr.fit(
                            x_tr[sub_idx], y_tr[sub_idx], sample_weight=w_tr[sub_idx]
                        )
                        pv = lr.predict_proba(x_round[va_idx])[:, 1]
                    except Exception:
                        ok = False
                        break
                    round_oof[va_idx] = pv.astype(np.float32)
                    fold_metrics.append(_metric_pack(y_arr[va_idx], pv))
                    fold_coefs.append(lr.coef_.reshape(-1).astype(np.float32))
                if not ok or len(fold_metrics) == 0:
                    continue

                agg = _aggregate_j(fold_metrics)
                hp_records.append(
                    {
                        "alpha": float(alpha),
                        "l1_ratio": float(l1_ratio),
                        "oof": round_oof,
                        "fold_coefs": fold_coefs,
                        "fold_metrics": fold_metrics,
                        "lift30": float(agg["lift30"]),
                        "auc_correct_30": float(agg["auc_correct_30"]),
                        "stability30": float(agg["stability30"]),
                        "J_mean": float(agg["J_mean"]),
                        "J_std": float(agg["J_std"]),
                        "J_final": float(agg["J_final"]),
                    }
                )

        if len(hp_records) == 0:
            break

        best = max(hp_records, key=lambda z: float(z["J_final"]))
        all_scores = np.array([float(r["J_final"]) for r in hp_records], dtype=float)
        best_se = (
            float(np.std(all_scores, ddof=1) / max(np.sqrt(len(all_scores)), 1.0))
            if len(all_scores) > 1
            else 0.0
        )
        score_cut = float(best["J_final"]) - best_se
        contenders = [z for z in hp_records if float(z["J_final"]) >= score_cut]

        frontier = _pareto_frontier(contenders)
        print(f"Frontier Analysis: {len(frontier)} non-dominated candidates found.")
        chosen = _select_elbow_candidate(frontier)
        print(
            f"Selected 'Elbow' candidate: Alpha={chosen['alpha']}, "
            f"L1={chosen['l1_ratio']}, Features={len(active_idx)}."
        )

        pooled_coef = np.vstack(
            [coef for cand in contenders for coef in cand["fold_coefs"]]
        )
        total_models_in_pool = max(1, pooled_coef.shape[0])
        active_freq = np.mean(np.abs(pooled_coef) > 1e-6, axis=0)
        pos = np.sum(pooled_coef > 1e-6, axis=0)
        neg = np.sum(pooled_coef < -1e-6, axis=0)
        net_support = np.abs(pos - neg) / float(total_models_in_pool)
        keep_mask = (active_freq >= min_freq) & (net_support >= min_net_support)
        floor_keep = _floor_keep(round_id, len(active_idx))
        if int(np.sum(keep_mask)) < floor_keep:
            rank_score = active_freq * np.maximum(net_support, 1e-8)
            top_k = min(len(rank_score), floor_keep)
            keep_mask = np.zeros(len(rank_score), dtype=bool)
            keep_mask[np.argsort(rank_score)[-top_k:]] = True
        if int(np.sum(keep_mask)) < 40:
            break

        candidate_active = active_idx[np.asarray(keep_mask, dtype=bool)]
        candidate_oof = np.asarray(chosen["oof"], dtype=np.float32)

        if round_id == 1:
            golden_floor_score = float(best["J_final"])
            golden_floor_se = float(best_se)
        else:
            quick = _quick_cv_score(
                X_combined[:, candidate_active],
                y_arr,
                random_state=random_state + round_id,
                alpha=float(chosen["alpha"]),
                l1_ratio=float(chosen["l1_ratio"]),
                prior_round_pred=candidate_oof,
            )
            if (
                golden_floor_score is not None
                and golden_floor_se is not None
                and float(quick["J_final"])
                < float(golden_floor_score - golden_floor_se)
            ):
                round_history.append(
                    {
                        "round": round_id,
                        "stopped_by_golden_floor": True,
                        "candidate_J_final": float(quick["J_final"]),
                        "golden_floor_score": float(golden_floor_score),
                        "golden_floor_se": float(golden_floor_se),
                    }
                )
                active_idx = prev_active_idx
                last_round_oof = prev_round_oof
                break

        active_idx = candidate_active
        last_round_oof = candidate_oof
        round_history.append(
            {
                "round": round_id,
                "n_features_start": int(x_round.shape[1]),
                "n_features_end": int(len(active_idx)),
                "alpha": float(chosen["alpha"]),
                "l1_ratio": float(chosen["l1_ratio"]),
                "J_final": float(chosen["J_final"]),
                "J_mean": float(chosen["J_mean"]),
                "J_std": float(chosen["J_std"]),
                "stability30": float(chosen["stability30"]),
                "min_freq": float(min_freq),
                "min_net_support": float(min_net_support),
                "floor_keep": int(floor_keep),
                "active_freq_mean": float(np.mean(active_freq)),
                "net_support_mean": float(np.mean(net_support)),
            }
        )

    return {
        "selected_indices": np.asarray(active_idx, dtype=np.int32),
        "pruning_history": round_history,
        "last_round_en_oof": np.asarray(last_round_oof, dtype=np.float32),
    }


def _fit_lgbm_leaf_bundle(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray | None,
    random_state: int,
) -> dict[str, Any]:
    models = []
    encoders = []
    train_leaf_parts = []
    eval_leaf_parts = []
    leaf_feature_names: list[str] = []

    for spec in LEAF_MODEL_SPECS:
        leaf_frac = float(spec["leaf_frac"])
        prefix = str(spec["prefix"])
        max_depth = int(spec["max_depth"])
        lgbm = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            max_depth=max_depth,
            min_data_in_leaf=max(50, int(leaf_frac * len(x_train))),
            min_sum_hessian_in_leaf=1e-3,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            min_gain_to_split=0.001,
            max_bin=127,
            n_estimators=500,
            n_jobs=2,
        )
        fit_idx = _stratified_subsample_indices(
            y_train, max_n=15000, random_state=random_state
        )
        fit_kwargs = {}
        if y_eval is not None:
            fit_kwargs = {
                "eval_set": [(x_eval, y_eval)],
                "callbacks": [lgb.early_stopping(25, verbose=False)],
            }
        lgbm.fit(x_train[fit_idx], y_train[fit_idx], **fit_kwargs)

        leaf_train = lgbm.predict(x_train, pred_leaf=True)
        leaf_eval = lgbm.predict(x_eval, pred_leaf=True)
        enc = _make_onehot_encoder()
        enc.fit(leaf_train)
        train_part = enc.transform(leaf_train)
        eval_part = enc.transform(leaf_eval)
        n_cols = int(train_part.shape[1])
        leaf_feature_names.extend([f"{prefix}_{i}" for i in range(n_cols)])
        models.append(lgbm)
        encoders.append(enc)
        train_leaf_parts.append(train_part)
        eval_leaf_parts.append(eval_part)

    train_leaf_matrix = sparse.hstack(train_leaf_parts, format="csr")
    eval_leaf_matrix = sparse.hstack(eval_leaf_parts, format="csr")
    return {
        "models": models,
        "encoders": encoders,
        "train_leaf_matrix": train_leaf_matrix,
        "eval_leaf_matrix": eval_leaf_matrix,
        "leaf_feature_names": leaf_feature_names,
    }


class RidgeOnLGBMModel:
    def __init__(self):
        self.lgb_models = None
        self.leaf_encoders = None
        self.scaler = None
        self.ridge = None
        self.selected_indices = None
        self.selected_feature_names = None
        self.selected_raw_feature_names = None
        self.selected_leaf_feature_names = None
        self.selected_raw_indices = None
        self.selected_leaf_indices = None
        self.raw_feature_names = None
        self.leaf_feature_names = None
        self.combined_feature_names = None
        self.oof_probs = None
        self.uncertainty_features = None
        self.confidence_norm = None
        self.pruning_history = None

    def _build_matrix(self, X):
        x_np = np.asarray(X, dtype=np.float32)
        leaf_parts = []
        for lgbm, enc in zip(self.lgb_models, self.leaf_encoders):
            leaf = lgbm.predict(x_np, pred_leaf=True)
            leaf_parts.append(enc.transform(leaf))
        leaf_oh = sparse.hstack(leaf_parts, format="csr")
        return sparse.hstack([sparse.csr_matrix(x_np), leaf_oh], format="csr")

    def predict_proba(self, X):
        xc = self._build_matrix(X)
        raw_part = xc[:, self.selected_raw_indices]
        leaf_part = xc[:, self.selected_leaf_indices]
        raw_scaled = (
            self.scaler.transform(raw_part) if raw_part.shape[1] > 0 else raw_part
        )
        xs = sparse.hstack([raw_scaled, leaf_part], format="csr")
        score = self.ridge.decision_function(xs)
        p = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return self.predict_proba(X)[:, 1]

    def predict_uncertainty_features(self, X):
        p = self.predict(X)
        conf_raw = np.abs(p - 0.5) * 2.0
        lo = (
            float(self.confidence_norm["p5"])
            if self.confidence_norm
            else float(np.percentile(conf_raw, 5))
        )
        hi = (
            float(self.confidence_norm["p95"])
            if self.confidence_norm
            else float(np.percentile(conf_raw, 95))
        )
        if hi <= lo:
            conf = np.ones(len(conf_raw), dtype=np.float32)
        else:
            conf = 0.7 + 0.6 * np.clip((conf_raw - lo) / (hi - lo), 0.0, 1.0)
        return {
            "ridge_conf_clf_raw": conf_raw.astype(np.float32),
            "ridge_conf_clf": conf.astype(np.float32),
            "prefix_std": np.full(len(conf), np.std(p), dtype=np.float32),
            "leaf_support_q25": np.full(len(conf), np.nan, dtype=np.float32),
            "leaf_target_iqr_mean": np.full(len(conf), np.nan, dtype=np.float32),
        }


def train_ridge_on_lgbm_candidate(X, y, sample_weight=None, random_state=42):
    y_bin = np.asarray(y >= 0.5, dtype=np.int8)
    x_df = X if hasattr(X, "columns") else pd.DataFrame(X)
    x_np = x_df.to_numpy(dtype=np.float32)
    n = len(y_bin)
    if lgb is None or n < 200:
        return None

    w_base = (
        np.ones(n, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )

    race_idx = _stratified_subsample_indices(
        y_bin, max_n=min(50000, n), random_state=random_state + 101
    )
    x_race = x_np[race_idx]
    y_race = y_bin[race_idx]
    w_race = w_base[race_idx]

    outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    oof_race = np.zeros(len(y_race), dtype=np.float32)
    raw_lgbm2p_oof_race = np.zeros(len(y_race), dtype=np.float32)
    ridge_fold_metrics: list[dict[str, float]] = []
    raw_fold_metrics: list[dict[str, float]] = []

    model = RidgeOnLGBMModel()
    for tr_idx, va_idx in outer_cv.split(x_race, y_race):
        x_tr, y_tr = x_race[tr_idx], y_race[tr_idx]
        x_va, y_va = x_race[va_idx], y_race[va_idx]
        bundle = _fit_lgbm_leaf_bundle(
            x_tr, y_tr, x_va, y_va, random_state=random_state
        )
        raw_lgbm2p_oof_race[va_idx] = (
            bundle["models"][0].predict_proba(x_va)[:, 1].astype(np.float32)
        )
        raw_fold_metrics.append(_metric_pack(y_va, raw_lgbm2p_oof_race[va_idx]))
        x_tr_c = sparse.hstack(
            [sparse.csr_matrix(x_tr), bundle["train_leaf_matrix"]], format="csr"
        )
        x_va_c = sparse.hstack(
            [sparse.csr_matrix(x_va), bundle["eval_leaf_matrix"]], format="csr"
        )

        prune = _stage_a_prune(
            x_tr_c,
            y_tr,
            random_state=random_state,
            n_raw_total=x_tr.shape[1],
            prior_round_pred=None,
            initial_features=None,
            max_rounds=7,
        )
        selected = np.asarray(prune["selected_indices"], dtype=np.int32)
        last_en_oof = np.asarray(prune["last_round_en_oof"], dtype=np.float32)

        n_raw_fold = x_tr.shape[1]
        sel_raw = selected[selected < n_raw_fold]
        sel_leaf = selected[selected >= n_raw_fold]
        scaler = RobustScaler(with_centering=False)
        xtr_raw = x_tr_c[:, sel_raw]
        xva_raw = x_va_c[:, sel_raw]
        xtr_raw_s = scaler.fit_transform(xtr_raw) if xtr_raw.shape[1] > 0 else xtr_raw
        xva_raw_s = scaler.transform(xva_raw) if xva_raw.shape[1] > 0 else xva_raw
        xtr_s = sparse.hstack([xtr_raw_s, x_tr_c[:, sel_leaf]], format="csr")
        xva_s = sparse.hstack([xva_raw_s, x_va_c[:, sel_leaf]], format="csr")
        ridge = RidgeClassifier(alpha=1.0, random_state=random_state)
        w_rank_focus = top30_boundary_weight(last_en_oof)
        w_final = np.asarray(w_race[tr_idx], dtype=np.float32) * w_rank_focus
        ridge.fit(xtr_s, y_tr, sample_weight=w_final)
        pred = (1.0 / (1.0 + np.exp(-ridge.decision_function(xva_s)))).astype(
            np.float32
        )
        oof_race[va_idx] = pred
        ridge_fold_metrics.append(_metric_pack(y_va, pred))

    raw_lgbm_metrics = _metric_pack(y_race, raw_lgbm2p_oof_race)
    raw_lgbm_metrics.update(_aggregate_j(raw_fold_metrics))

    bundle_full = _fit_lgbm_leaf_bundle(
        x_np, y_bin, x_np, None, random_state=random_state
    )
    model.lgb_models = bundle_full["models"]
    model.leaf_encoders = bundle_full["encoders"]
    model.leaf_feature_names = bundle_full["leaf_feature_names"]
    x_full_c = sparse.hstack(
        [sparse.csr_matrix(x_np), bundle_full["train_leaf_matrix"]], format="csr"
    )

    prune_full = _stage_a_prune(
        x_full_c,
        y_bin,
        random_state=random_state,
        n_raw_total=x_np.shape[1],
        prior_round_pred=bundle_full["models"][0]
        .predict_proba(x_np)[:, 1]
        .astype(np.float32),
        initial_features=None,
        max_rounds=7,
    )
    model.selected_indices = np.asarray(prune_full["selected_indices"], dtype=np.int32)
    last_round_en_oof = np.asarray(prune_full["last_round_en_oof"], dtype=np.float32)
    model.pruning_history = prune_full["pruning_history"]

    model.scaler = RobustScaler(with_centering=False)
    model.selected_raw_indices = model.selected_indices[
        model.selected_indices < x_np.shape[1]
    ]
    model.selected_leaf_indices = model.selected_indices[
        model.selected_indices >= x_np.shape[1]
    ]
    x_raw = x_full_c[:, model.selected_raw_indices]
    x_raw_s = model.scaler.fit_transform(x_raw) if x_raw.shape[1] > 0 else x_raw
    xs = sparse.hstack(
        [x_raw_s, x_full_c[:, model.selected_leaf_indices]], format="csr"
    )
    model.ridge = RidgeClassifier(alpha=1.0, random_state=random_state)
    w_final_full = w_base * top30_boundary_weight(last_round_en_oof)
    model.ridge.fit(xs, y_bin, sample_weight=w_final_full)

    p_train = (1.0 / (1.0 + np.exp(-model.ridge.decision_function(xs)))).astype(
        np.float32
    )
    oof_full = np.full(n, np.nan, dtype=np.float32)
    oof_full[race_idx] = oof_race
    model.oof_probs = oof_full
    model.raw_feature_names = list(x_df.columns)
    n_raw = len(model.raw_feature_names)
    model.combined_feature_names = model.raw_feature_names + model.leaf_feature_names
    model.selected_feature_names = [
        model.combined_feature_names[i] for i in model.selected_indices
    ]
    model.selected_raw_feature_names = [
        model.combined_feature_names[i] for i in model.selected_indices if i < n_raw
    ]
    model.selected_leaf_feature_names = [
        model.combined_feature_names[i] for i in model.selected_indices if i >= n_raw
    ]

    conf_raw = np.abs(p_train - 0.5) * 2.0
    model.confidence_norm = {
        "p5": float(np.percentile(conf_raw, 5)) if len(conf_raw) else 0.0,
        "p95": float(np.percentile(conf_raw, 95)) if len(conf_raw) else 1.0,
    }
    model.uncertainty_features = model.predict_uncertainty_features(x_np)

    agg_oof = _aggregate_j(ridge_fold_metrics)
    metric_train = _metric_pack(y_bin, p_train)
    agg_train = _aggregate_j([metric_train])
    j_final_train = float(agg_train["J_final"])
    j_final_oof = float(agg_oof["J_final"])
    oof_train_gap = float(
        np.clip((j_final_train - j_final_oof) / max(abs(j_final_train), 1e-8), 0.0, 1.0)
    )
    j_race = float(j_final_oof * (1.0 - oof_train_gap))

    metrics = _metric_pack(y_race, oof_race)
    metrics.update(agg_oof)
    metrics["J_final_train"] = j_final_train
    metrics["J_final_oof"] = j_final_oof
    metrics["OOF_Train_Gap"] = oof_train_gap
    metrics["J_race"] = j_race
    metrics["feature_count"] = int(len(model.selected_indices))
    metrics["n_raw_features_kept"] = int(len(model.selected_raw_feature_names))
    metrics["n_leaf_features_kept"] = int(len(model.selected_leaf_feature_names))
    metrics["n_total_features_kept"] = int(len(model.selected_indices))

    for key in [
        "J_final_train",
        "J_final_oof",
        "OOF_Train_Gap",
        "J_race",
        "lift30",
        "auc_correct_30",
        "stability30",
        "auc",
        "pr_auc",
        "pr_random",
        "brier",
        "ece",
        "top30_correctness_rate",
        "overall_correctness_rate",
        "feature_count",
        "oof_std",
    ]:
        print(f"{key}: {metrics.get(key)}")

    return {
        "model": model,
        "metrics": metrics,
        "raw_lgbm_2p_metrics": raw_lgbm_metrics,
        "raw_lgbm_metrics": raw_lgbm_metrics,
        "oof_probs": oof_full,
        "pruning_history": prune_full["pruning_history"],
        "last_round_en_oof": last_round_en_oof,
        "uncertainty_features": model.uncertainty_features,
    }
