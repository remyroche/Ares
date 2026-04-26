from __future__ import annotations

import gc
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from src.utils.tprint import tprint, tprint_performance

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

ALPHA_GRID = (0.2, 0.5, 1.0)
L1_RATIO_GRID = (0.8, 0.9, 0.95)
EN_PRUNE_N_JOBS = 2
EN_PRUNE_MAX_ITER = 2000
EN_PRUNE_TOL = 1e-3
EN_PRUNE_FOLD_SUBSAMPLE_ROWS = 5000
RACE_MAX_ROWS = 60000
RACE_EVAL_FRACTION = 1.0 / 3.0
LEAF_MODEL_SPECS = (
    {"max_depth": 3, "leaf_frac": 0.02, "prefix": "LGBM_2P_LEAF"},
    {"max_depth": 3, "leaf_frac": 0.04, "prefix": "LGBM_4P_LEAF"},
    {"max_depth": 3, "leaf_frac": 0.06, "prefix": "LGBM_6P_LEAF"},
    {"max_depth": 3, "leaf_frac": 0.08, "prefix": "LGBM_8P_LEAF"},
    {"max_depth": 4, "leaf_frac": 0.02, "prefix": "LGBM_D4_2P_LEAF"},
    {"max_depth": 4, "leaf_frac": 0.05, "prefix": "LGBM_D4_5P_LEAF"},
)

DEFAULT_TREE_FEATURE_CONFIG: dict[str, Any] = {
    "use_boundary_features": True,
    "use_path_features": True,
    "use_soft_features": False,
    "use_raw_features_with_tree_features": True,
    "prune_dense_features": True,
    "dense_prune_max_features": 0,
    "dense_prune_var_threshold": 1e-8,
    "dense_prune_corr_threshold": 0.95,
    "leaf_temperature": 0.5,
    "enable_uncertainty_score_modulation": True,
    "uncertainty_min_improvement": 0.0,
    "uncertainty_correction_weight": 0.2,
}

EN_PRUNE_CV_SPLITS = 2

GLOBAL_TREE_FEATURE_NAMES = (
    "mean_margin_all",
    "mean_margin_interaction",
    "mean_path_depth_all",
    "mean_path_depth_interaction",
    "mean_leaf_entropy_all",
    "mean_leaf_entropy_interaction",
    "mean_leaf_gap_all",
    "mean_leaf_gap_interaction",
)


def _ROUND_THRESHOLDS() -> list[tuple[float, float]]:
    return [
        (0.30, 0.60),
        (0.40, 0.65),
        (0.50, 0.75),
        (0.60, 0.80),
        (0.70, 0.85),
        (0.80, 0.90),
        (0.90, 0.95),
    ]


def _stratified_subsample_indices(
    y: np.ndarray, max_n: int = 10000, random_state: int = 42
) -> np.ndarray:
    n = len(y)
    if n <= max_n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    out: list[np.ndarray] = []
    for c in classes:
        ids = np.where(y == c)[0]
        take = max(1, int(round(max_n * len(ids) / n)))
        take = min(take, len(ids))
        out.append(rng.choice(ids, size=take, replace=False))
    idx = np.sort(np.concatenate(out).astype(np.int32))
    if len(idx) > max_n:
        idx = np.sort(rng.choice(idx, size=max_n, replace=False).astype(np.int32))
    return idx


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
    w = 1.0 + 0.5 * boundary + 0.5 * topness
    return np.clip(w, 1.0, 2.0).astype(np.float32)


def _rank_focus_weight(pred: np.ndarray) -> np.ndarray:
    rank_pct = (
        pd.Series(np.asarray(pred, dtype=np.float32))
        .rank(pct=True)
        .to_numpy(dtype=np.float32)
    )
    return (0.7 + 0.6 * np.sqrt(rank_pct)).astype(np.float32)


def _error_weight_classifier(
    y_true: np.ndarray, pred: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    y_bin = np.asarray(y_true, dtype=np.int8)
    pred_arr = np.asarray(pred, dtype=np.float32)
    predicted_class = (pred_arr >= threshold).astype(np.int8)
    confidence = np.abs(pred_arr - threshold) / max(
        max(np.max(np.abs(pred_arr - threshold)), 1e-6), threshold
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    correct = (predicted_class == y_bin).astype(np.float32)
    w = np.where(
        correct > 0.5,
        np.minimum(1.6, 1.0 + 0.2 * confidence),
        np.minimum(1.6, 1.0 + 0.5 * confidence),
    )
    return w.astype(np.float32)


def _error_weight_regressor(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    resid = np.abs(
        np.asarray(y_true, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
    )
    resid_rank = pd.Series(resid).rank(pct=True).to_numpy(dtype=np.float32)
    return np.minimum(1.6, 1.0 + 0.3 * resid_rank).astype(np.float32)


def _compute_weight_distillation(
    y_true: np.ndarray,
    pred: np.ndarray,
    prev_en_pred: np.ndarray | None,
    is_classifier: bool = True,
) -> np.ndarray:
    w = top30_boundary_weight(pred)
    if is_classifier:
        ew = _error_weight_classifier(y_true, pred)
    else:
        ew = _error_weight_regressor(y_true, pred)
    w = w * ew
    rf = _rank_focus_weight(pred)
    w = w * np.sqrt(rf)
    return w.astype(np.float32)


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


def _top30_feature_contribution_stability(
    pred: np.ndarray,
    coef: np.ndarray,
    y_true: np.ndarray,
    n_slices: int = 5,
) -> float:
    p = np.asarray(pred, dtype=np.float64)
    c = np.asarray(coef, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    k = max(1, int(0.30 * len(p)))
    idx = np.argsort(p)[-k:]
    if len(idx) < 10:
        return 0.0
    top_pred = p[idx]
    top_y = y[idx]
    q = np.quantile(top_pred, np.linspace(0, 1, n_slices + 1))
    slice_contribs = []
    for i in range(n_slices):
        m = (top_pred >= q[i]) & (
            top_pred < q[i + 1] if i < n_slices - 1 else top_pred <= q[i + 1]
        )
        if np.sum(m) < 3:
            continue
        slice_y = top_y[m]
        slice_p = top_pred[m]
        mean_correct = float(np.mean(slice_y)) if len(slice_y) > 0 else 0.0
        slice_contribs.append(mean_correct)
    if len(slice_contribs) < 2:
        return 0.0
    arr = np.asarray(slice_contribs, dtype=np.float64)
    return float(np.mean(arr) - np.std(arr))


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


def _fold_j(m: dict[str, float]) -> float:
    return 0.6 * m.get("lift30", 0.0) + 0.4 * m.get("auc_correct_30", 0.0)


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
    lift30 = float(np.mean([m.get("lift30", 0.0) for m in fold_metrics]))
    auc_c30 = float(np.mean([m.get("auc_correct_30", 0.0) for m in fold_metrics]))
    j_vals = [_fold_j(m) for m in fold_metrics]
    j_arr = np.asarray(j_vals, dtype=np.float64)
    j_mean = float(np.mean(j_arr))
    j_std = float(np.std(j_arr, ddof=1)) if len(j_arr) > 1 else 0.0
    stability30 = float(j_mean - 2.0 * j_std)
    j_final = 0.4 * lift30 + 0.2 * auc_c30 + 0.4 * stability30
    return {
        "lift30": lift30,
        "auc_correct_30": auc_c30,
        "stability30": stability30,
        "J_mean": j_mean,
        "J_std": j_std,
        "J_final": j_final,
    }


def _tree_feature_config(**kwargs) -> dict[str, Any]:
    cfg = dict(DEFAULT_TREE_FEATURE_CONFIG)
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    return cfg


def _feature_scales(x: np.ndarray) -> np.ndarray:
    s = np.std(x, axis=0, dtype=np.float64)
    s[s < 1e-12] = 1.0
    return s.astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1.0 - 1e-6)).astype(np.float32)


def _iter_leaf_paths(node: dict, path: list | None = None) -> list[list[int]]:
    if path is None:
        path = []
    if "leaf_value" in node or "leaf_index" in node:
        return [list(path)]
    results: list[list[int]] = []
    for child in node.get("children", []):
        results.extend(_iter_leaf_paths(child, path + [node.get("split_feature", 0)]))
    if not results and ("leaf_value" not in node):
        for child in node.get("children", []):
            results.extend(_iter_leaf_paths(child, path))
    return results


def _tree_feature_names(tree_index: int, n_leaves: int) -> list[str]:
    return [f"T{tree_index}_L{li}" for li in range(n_leaves)]


def _hard_path_stats_for_tree(
    leaf_ids: np.ndarray, y_train: np.ndarray, train_leaves: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    unique_leaves = np.unique(train_leaves)
    margin_means = np.zeros(len(leaf_ids), dtype=np.float32)
    margin_stds = np.zeros(len(leaf_ids), dtype=np.float32)
    margin_skew = np.zeros(len(leaf_ids), dtype=np.float32)
    margin_min = np.zeros(len(leaf_ids), dtype=np.float32)
    margin_max = np.zeros(len(leaf_ids), dtype=np.float32)
    margin_range = np.zeros(len(leaf_ids), dtype=np.float32)
    depth = np.zeros(len(leaf_ids), dtype=np.float32)
    n_samples = np.zeros(len(leaf_ids), dtype=np.float32)
    pos_rate = np.zeros(len(leaf_ids), dtype=np.float32)
    margin_cv = np.zeros(len(leaf_ids), dtype=np.float32)
    neg_rate = np.zeros(len(leaf_ids), dtype=np.float32)
    for lid in unique_leaves:
        m = train_leaves == lid
        if np.sum(m) < 2:
            continue
        vals = y_train[m]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        mask_eval = leaf_ids == lid
        margin_means[mask_eval] = mean_v
        margin_stds[mask_eval] = std_v
        margin_skew[mask_eval] = (
            float(np.mean(((vals - mean_v) / max(std_v, 1e-6)) ** 3))
            if std_v > 1e-6
            else 0.0
        )
        margin_min[mask_eval] = float(np.min(vals))
        margin_max[mask_eval] = float(np.max(vals))
        margin_range[mask_eval] = float(np.max(vals) - np.min(vals))
        depth[mask_eval] = float(np.log2(max(np.sum(m), 1)))
        n_samples[mask_eval] = float(np.sum(m))
        pos_rate[mask_eval] = float(np.mean(vals > 0.5))
        margin_cv[mask_eval] = std_v / max(abs(mean_v), 1e-6)
        neg_rate[mask_eval] = 1.0 - float(np.mean(vals > 0.5))
    features = np.column_stack(
        [
            margin_means,
            margin_stds,
            margin_skew,
            margin_min,
            margin_max,
            margin_range,
            depth,
            n_samples,
            pos_rate,
            margin_cv,
            neg_rate,
        ]
    ).astype(np.float32)
    names = [
        "mean",
        "std",
        "skew",
        "min",
        "max",
        "range",
        "depth",
        "n_samples",
        "pos_rate",
        "margin_cv",
        "neg_rate",
    ]
    return features, names


def _soft_leaf_probability_matrix(
    x: np.ndarray, model: Any, temperature: float = 0.5
) -> np.ndarray:
    n = x.shape[0]
    booster = model.booster_
    leaves = booster.predict(x, pred_leaf=True)
    leaves = np.asarray(leaves, dtype=np.int64)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    n_trees = leaves.shape[1]
    prob = np.zeros((n, n_trees), dtype=np.float32)
    for t in range(n_trees):
        ids = leaves[:, t]
        unique_ids = np.unique(ids)
        counts = np.array([float(np.sum(ids == lid)) for lid in unique_ids])
        probs_t = counts / float(len(ids))
        for j, lid in enumerate(unique_ids):
            mask = ids == lid
            prob[mask, t] = probs_t[j]
    return prob


def _soft_leaf_stats_for_tree(
    leaf_probs: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    n = leaf_probs.shape[0]
    soft_mean = np.zeros(n, dtype=np.float32)
    soft_var = np.zeros(n, dtype=np.float32)
    soft_entropy = np.zeros(n, dtype=np.float32)
    soft_top1 = np.zeros(n, dtype=np.float32)
    soft_gap = np.zeros(n, dtype=np.float32)
    soft_mass_top2 = np.zeros(n, dtype=np.float32)
    features = np.column_stack(
        [
            soft_mean,
            soft_var,
            soft_entropy,
            soft_top1,
            soft_gap,
            soft_mass_top2,
        ]
    ).astype(np.float32)
    names = [
        "soft_mean",
        "soft_var",
        "soft_entropy",
        "soft_top1",
        "soft_gap",
        "soft_mass_top2",
    ]
    return features, names


def _compute_tree_state_feature_block(
    models: list,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    feature_scales: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_train: list[np.ndarray] = []
    all_eval: list[np.ndarray] = []
    all_names: list[str] = []
    metadata: dict[str, Any] = {
        "n_models": len(models),
        "n_features_raw": 0,
        "n_features_kept": 0,
        "n_dropped_by_pruning": 0,
    }
    y_train_proxy = np.zeros(x_train.shape[0], dtype=np.float32)
    booster = models[0].booster_
    train_leaves_0 = np.asarray(
        booster.predict(x_train, pred_leaf=True), dtype=np.int64
    )
    if train_leaves_0.ndim == 1:
        train_leaves_0 = train_leaves_0.reshape(-1, 1)

    for mi, model in enumerate(models):
        bst = model.booster_
        tr_leaves = np.asarray(bst.predict(x_train, pred_leaf=True), dtype=np.int64)
        if tr_leaves.ndim == 1:
            tr_leaves = tr_leaves.reshape(-1, 1)
        ev_leaves = np.asarray(bst.predict(x_eval, pred_leaf=True), dtype=np.int64)
        if ev_leaves.ndim == 1:
            ev_leaves = ev_leaves.reshape(-1, 1)

        n_trees = tr_leaves.shape[1]
        tree_sample_stride = max(1, n_trees // 20)
        tree_indices = list(range(0, n_trees, tree_sample_stride))[:20]

        for ti in tree_indices:
            suffix = f"T{ti}"
            tr_leaf_ids = tr_leaves[:, ti]
            ev_leaf_ids = ev_leaves[:, ti]
            all_leaf_ids = np.unique(
                np.concatenate([np.unique(tr_leaf_ids), np.unique(ev_leaf_ids)])
            )

            train_stats, stat_names = _hard_path_stats_for_tree(
                tr_leaf_ids, y_train_proxy, tr_leaf_ids
            )
            eval_stats, _ = _hard_path_stats_for_tree(
                ev_leaf_ids, y_train_proxy, tr_leaf_ids
            )

            for lid in all_leaf_ids:
                tr_mask = (tr_leaf_ids == lid).astype(np.float32)
                ev_mask = (ev_leaf_ids == lid).astype(np.float32)
                for si, sn in enumerate(stat_names):
                    all_names.append(f"M{mi}_{suffix}_L{lid}_{sn}")
                    all_train.append(
                        (tr_mask * train_stats[:, si]).astype(np.float32).reshape(-1, 1)
                    )
                    all_eval.append(
                        (ev_mask * eval_stats[:, si]).astype(np.float32).reshape(-1, 1)
                    )

        if config.get("use_soft_features", False):
            train_probs = _soft_leaf_probability_matrix(
                x_train, model, temperature=config.get("leaf_temperature", 0.5)
            )
            eval_probs = _soft_leaf_probability_matrix(
                x_eval, model, temperature=config.get("leaf_temperature", 0.5)
            )
            train_f, f_names = _soft_leaf_stats_for_tree(
                train_probs[:, 0], y_train_proxy
            )
            eval_f, _ = _soft_leaf_stats_for_tree(eval_probs[:, 0], y_train_proxy)
            for ni, nm in enumerate(f_names):
                all_names.append(f"M{mi}_{nm}")
            all_train.append(train_f)
            all_eval.append(eval_f)

    if not all_train:
        z_t = np.zeros((x_train.shape[0], 0), dtype=np.float32)
        z_e = np.zeros((x_eval.shape[0], 0), dtype=np.float32)
        metadata["n_features_raw"] = 0
        return z_t, z_e, metadata

    X_tree = np.hstack(all_train).astype(np.float32)
    X_eval_tree = np.hstack(all_eval).astype(np.float32)
    metadata["n_features_raw"] = X_tree.shape[1]
    if config.get("prune_dense_features", True):
        var_mask = np.var(X_tree, axis=0) > config.get(
            "dense_prune_var_threshold", 1e-6
        )
        X_tree = X_tree[:, var_mask]
        X_eval_tree = X_eval_tree[:, var_mask]
        kept_names = [n for n, m in zip(all_names, var_mask) if m]
        corr_thr = config.get("dense_prune_corr_threshold", 0.98)
        if X_tree.shape[1] > config.get("dense_prune_max_features", 300):
            drop_idx = _greedy_drop_from_pairs(
                X_tree,
                corr_thr,
                max_features=config.get("dense_prune_max_features", 300),
            )
            keep = np.ones(X_tree.shape[1], dtype=bool)
            keep[drop_idx] = False
            X_tree = X_tree[:, keep]
            X_eval_tree = X_eval_tree[:, keep]
            kept_names = [n for n, m in zip(kept_names, keep) if m]
    else:
        kept_names = all_names

    metadata["n_features_kept"] = X_tree.shape[1]
    metadata["n_dropped_by_pruning"] = metadata["n_features_raw"] - X_tree.shape[1]
    metadata["tree_feature_names"] = kept_names
    return X_tree, X_eval_tree, metadata


def _mean_by_suffix(arr: np.ndarray, names: list[str], suffix: str) -> np.ndarray:
    cols = [i for i, n in enumerate(names) if n.endswith(suffix)]
    if not cols:
        return np.zeros(arr.shape[0], dtype=np.float32)
    return np.mean(arr[:, cols], axis=1).astype(np.float32)


def _global_tree_summary_features(
    X_tree: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str]]:
    if X_tree.shape[1] == 0:
        return (
            np.zeros(
                (X_tree.shape[0], len(GLOBAL_TREE_FEATURE_NAMES)), dtype=np.float32
            ),
            list(GLOBAL_TREE_FEATURE_NAMES),
        )
    means = _mean_by_suffix(X_tree, names, "mean")
    stds = _mean_by_suffix(X_tree, names, "std")
    depths = _mean_by_suffix(X_tree, names, "depth")
    entropies = _mean_by_suffix(X_tree, names, "soft_entropy")
    gaps = _mean_by_suffix(X_tree, names, "gap")
    pos_rates = _mean_by_suffix(X_tree, names, "pos_rate")
    features = np.column_stack(
        [
            means,
            means * stds,
            depths,
            depths * stds,
            entropies,
            entropies * stds,
            gaps,
            gaps * stds,
        ]
    ).astype(np.float32)
    return features, list(GLOBAL_TREE_FEATURE_NAMES)


def _fit_dense_tree_feature_pruner(
    X_tree: np.ndarray,
    max_features: int = 300,
    var_thr: float = 1e-6,
    corr_thr: float = 0.98,
) -> dict[str, Any]:
    mask = np.var(X_tree, axis=0) > var_thr
    X = X_tree[:, mask]
    drop_idx = _greedy_drop_from_pairs(X, corr_thr, max_features=max_features)
    keep = np.ones(X.shape[1], dtype=bool)
    keep[drop_idx] = False
    return {"var_mask": mask, "keep_after_corr": keep}


def build_tree_state_features(
    models: list,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    feature_scales: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _tree_feature_config(**(config or {}))
    X_tree, X_eval_tree, meta = _compute_tree_state_feature_block(
        models,
        x_train,
        x_eval,
        feature_scales if feature_scales is not None else _feature_scales(x_train),
        cfg,
    )
    names = meta.get("tree_feature_names", [])
    global_f, global_names = _global_tree_summary_features(X_tree, names)
    global_f_eval, _ = _global_tree_summary_features(X_eval_tree, names)
    return {
        "X_tree_features": X_tree,
        "X_valid_tree_features": X_eval_tree,
        "metadata": meta,
        "tree_feature_names": names,
        "global_tree_features": global_f,
        "global_tree_features_eval": global_f_eval,
        "global_tree_feature_names": global_names,
    }


def _uncertainty_design(
    base_score: np.ndarray, global_features: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            _logit(_sigmoid(base_score)),
            global_features,
        ]
    ).astype(np.float32)


def _fit_uncertainty_scaler(x: np.ndarray) -> dict[str, Any]:
    lo = np.nanpercentile(x, 5, axis=0)
    hi = np.nanpercentile(x, 95, axis=0)
    rng = np.maximum(hi - lo, 1e-6)
    return {"lo": lo.astype(np.float32), "rng": rng.astype(np.float32)}


def _apply_uncertainty_scaler(x: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    return np.clip((x - scaler["lo"]) / scaler["rng"], -3.0, 3.0).astype(np.float32)


def _fit_ridge_uncertainty_modulator(
    base_score: np.ndarray,
    global_tree_features: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    random_state: int = 42,
    min_improvement: float = 0.0,
    correction_weight: float = 0.2,
) -> dict[str, Any]:
    X = _uncertainty_design(base_score, global_tree_features)
    y_arr = np.asarray(y, dtype=np.int8)
    scaler = _fit_uncertainty_scaler(X)
    X_s = _apply_uncertainty_scaler(X, scaler)
    base_j = _fold_j(_metric_pack(y_arr, _sigmoid(base_score)))
    best_j = base_j
    best_lr = None
    base_lift30 = _metric_pack(y_arr, _sigmoid(base_score)).get("lift30", 0.0)
    best_lift30 = base_lift30
    base_stability30 = _aggregate_j([_metric_pack(y_arr, _sigmoid(base_score))]).get(
        "stability30", 0.0
    )
    best_stability30 = base_stability30
    try:
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=random_state)
        sw = (
            sample_weight
            if sample_weight is not None
            else np.ones(len(y_arr), dtype=np.float32)
        )
        lr.fit(X_s, y_arr, sample_weight=sw)
        mod_p = lr.predict_proba(X_s)[:, 1].astype(np.float32)
        mod_j = _fold_j(_metric_pack(y_arr, mod_p))
        mod_metrics = _aggregate_j([_metric_pack(y_arr, mod_p)])
        if mod_j > best_j + min_improvement:
            best_j = mod_j
            best_lr = lr
            best_lift30 = mod_metrics.get("lift30", best_lift30)
            best_stability30 = mod_metrics.get("stability30", best_stability30)
    except Exception:
        pass
    enabled = best_lr is not None
    return {
        "enabled": enabled,
        "scaler": scaler,
        "lr": best_lr,
        "correction_weight": correction_weight,
        "base_j": float(base_j),
        "best_j": float(best_j),
        "base_lift30": float(base_lift30),
        "best_lift30": float(best_lift30),
        "base_stability30": float(base_stability30),
        "best_stability30": float(best_stability30),
        "prob": (
            np.clip(
                (1.0 - correction_weight) * _sigmoid(base_score)
                + correction_weight
                * best_lr.predict_proba(_apply_uncertainty_scaler(X, scaler))[:, 1],
                1e-4,
                1.0 - 1e-4,
            ).astype(np.float32)
            if enabled
            else _sigmoid(base_score)
        ),
    }


def _predict_uncertainty_modulated(
    base_score: np.ndarray, global_tree_features: np.ndarray, modulator: dict[str, Any]
) -> np.ndarray:
    if not modulator.get("enabled", False):
        return _sigmoid(base_score)
    X = _uncertainty_design(base_score, global_tree_features)
    X_s = _apply_uncertainty_scaler(X, modulator["scaler"])
    cw = float(modulator.get("correction_weight", 0.2))
    lr = modulator["lr"]
    return np.clip(
        (1.0 - cw) * _sigmoid(base_score) + cw * lr.predict_proba(X_s)[:, 1],
        1e-4,
        1.0 - 1e-4,
    ).astype(np.float32)


def _pruning_thresholds(round_id: int) -> tuple[float, float]:
    table = _ROUND_THRESHOLDS()
    idx = min(round_id - 1, len(table) - 1)
    return table[idx]


def _floor_keep(round_id: int, n_active: int) -> int:
    table = [120, 100, 80, 60, 50, 45, 40]
    idx = min(round_id - 1, len(table) - 1)
    return max(40, min(table[idx], n_active))


def _point_line_distance(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return float(np.sqrt((x - x1) ** 2 + (y - y1) ** 2))
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
    px = x1 + t * dx
    py = y1 + t * dy
    return float(np.sqrt((x - px) ** 2 + (y - py) ** 2))


def _pareto_frontier(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda z: float(z.get("J_final", 0.0)))
    frontier: list[dict] = []
    max_stab = -np.inf
    for c in sorted_c:
        s = float(c.get("stability30", 0.0))
        if s >= max_stab - 1e-9:
            frontier.append(c)
            max_stab = max(max_stab, s)
    return frontier


def _select_elbow_candidate(frontier: list[dict]) -> dict:
    if len(frontier) <= 1:
        return frontier[0] if frontier else {}
    if len(frontier) == 2:
        return max(frontier, key=lambda z: float(z.get("J_final", 0.0)))
    j_vals = [float(c.get("J_final", 0.0)) for c in frontier]
    s_vals = [float(c.get("stability30", 0.0)) for c in frontier]
    j_min, j_max = min(j_vals), max(j_vals)
    s_min, s_max = min(s_vals), max(s_vals)
    j_range = max(j_max - j_min, 1e-9)
    s_range = max(s_max - s_min, 1e-9)
    distances = []
    for i in range(len(frontier)):
        d = _point_line_distance(
            j_vals[i],
            s_vals[i],
            j_min,
            s_min,
            j_max,
            s_max,
        )
        distances.append(d)
    best_idx = int(np.argmax(distances))
    return frontier[best_idx]


def _fit_en_prune_combo(
    alpha: float,
    l1_ratio: float,
    fold_cache: list[dict[str, Any]],
    y_arr: np.ndarray,
    n: int,
    random_state: int,
) -> dict[str, Any] | None:
    fold_coefs: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []
    round_oof = np.zeros(n, dtype=np.float32)
    convergence_hits = 0
    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=EN_PRUNE_MAX_ITER,
        tol=EN_PRUNE_TOL,
        C=1.0 / max(alpha, 1e-6),
        l1_ratio=float(l1_ratio),
        random_state=random_state,
        warm_start=True,
    )
    for fold in fold_cache:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                lr.fit(
                    fold["x_tr_s"][fold["sub_idx"]],
                    fold["y_tr"][fold["sub_idx"]],
                    sample_weight=fold["w_tr"][fold["sub_idx"]],
                )
            convergence_hits += int(
                any(issubclass(w.category, ConvergenceWarning) for w in caught)
            )
            if hasattr(lr, "n_iter_") and int(np.max(lr.n_iter_)) >= EN_PRUNE_MAX_ITER:
                convergence_hits += 1
            pv = lr.predict_proba(fold["x_va_s"])[:, 1]
        except Exception:
            return None
        va_idx = fold["va_idx"]
        round_oof[va_idx] = pv.astype(np.float32)
        fold_metrics.append(_metric_pack(y_arr[va_idx], pv))
        fold_coefs.append(lr.coef_.reshape(-1).astype(np.float32))
    if not fold_metrics:
        return None
    agg = _aggregate_j(fold_metrics)
    top30_stab = _top30_feature_contribution_stability(
        round_oof, np.mean(np.abs(np.vstack(fold_coefs)), axis=0), y_arr
    )
    agg["top30_stability"] = top30_stab
    return {
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "oof": round_oof,
        "fold_coefs": fold_coefs,
        "fold_metrics": fold_metrics,
        "lift30": float(agg["lift30"]),
        "auc_correct_30": float(agg["auc_correct_30"]),
        "stability30": float(agg["stability30"]),
        "top30_stability": top30_stab,
        "J_mean": float(agg["J_mean"]),
        "J_std": float(agg["J_std"]),
        "J_final": float(agg["J_final"]),
        "convergence_hits": int(convergence_hits),
    }


def _quick_cv_score(
    X: sparse.csr_matrix,
    y: np.ndarray,
    random_state: int,
    alpha: float = 0.5,
    l1_ratio: float = 0.9,
    prior_round_pred: np.ndarray | None = None,
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=np.int8)
    n = len(y_arr)
    fold_cv = StratifiedKFold(
        n_splits=EN_PRUNE_CV_SPLITS, shuffle=True, random_state=random_state
    )
    w = _compute_weight_distillation(
        y_arr,
        prior_round_pred
        if prior_round_pred is not None
        else np.full(n, 0.5, dtype=np.float32),
        prior_round_pred,
        is_classifier=True,
    )
    fold_metrics: list[dict[str, float]] = []
    for tr_idx, va_idx in fold_cv.split(np.zeros(n, dtype=np.int8), y_arr):
        sub_idx = _stratified_subsample_indices(
            y_arr[tr_idx],
            max_n=EN_PRUNE_FOLD_SUBSAMPLE_ROWS,
            random_state=random_state,
        )
        scaler = RobustScaler(with_centering=False)
        x_tr_s = scaler.fit_transform(X[tr_idx])
        x_va_s = scaler.transform(X[va_idx])
        lr = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=EN_PRUNE_MAX_ITER,
            tol=EN_PRUNE_TOL,
            C=1.0 / max(alpha, 1e-6),
            l1_ratio=float(l1_ratio),
            random_state=random_state,
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                lr.fit(
                    x_tr_s[sub_idx],
                    y_arr[tr_idx][sub_idx],
                    sample_weight=w[tr_idx][sub_idx],
                )
            if any(issubclass(warn.category, ConvergenceWarning) for warn in caught):
                tprint("    Quick CV EN convergence warning observed.")
            pv = lr.predict_proba(x_va_s)[:, 1]
            fold_metrics.append(_metric_pack(y_arr[va_idx], pv))
        except Exception:
            pass
    if not fold_metrics:
        return {"J_final": -999.0}
    return _aggregate_j(fold_metrics)


def _feature_abs_spearman_scores(
    X: sparse.csr_matrix, y: np.ndarray, active_idx: np.ndarray, random_state: int
) -> np.ndarray:
    n = X.shape[0]
    sub = _stratified_subsample_indices(
        y, max_n=min(10000, n), random_state=random_state
    )
    yv = np.asarray(y[sub], dtype=np.float64)
    Xs = np.asarray(X[sub][:, active_idx].toarray(), dtype=np.float32)
    if Xs.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    xr = pd.DataFrame(Xs).rank(pct=True).to_numpy(dtype=np.float64)
    yr = pd.Series(yv).rank(pct=True).to_numpy(dtype=np.float64)
    xr -= np.nanmean(xr, axis=0)
    yr -= np.nanmean(yr)
    x_std = np.sqrt(np.nanmean(xr * xr, axis=0))
    y_std = float(np.sqrt(np.nanmean(yr * yr)))
    denom = np.maximum(x_std * max(y_std, 1e-12), 1e-12)
    corr = np.nanmean(xr * yr[:, None], axis=0) / denom
    return np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)).astype(
        np.float32
    )


def _greedy_drop_from_pairs(
    X: np.ndarray, corr_thr: float, max_features: int = 300
) -> np.ndarray:
    if max_features <= 0:
        return np.array([], dtype=np.int32)
    if X.shape[1] <= max_features:
        return np.array([], dtype=np.int32)
    if corr_thr >= 1.0:
        return np.array([], dtype=np.int32)
    n_drop = X.shape[1] - max_features
    sub_n = min(len(X), 5000)
    rng = np.random.default_rng(42)
    idx = (
        rng.choice(len(X), size=sub_n, replace=False)
        if len(X) > sub_n
        else np.arange(len(X))
    )
    Xs = np.asarray(X[idx], dtype=np.float32)
    corr = np.abs(np.corrcoef(Xs.T))
    var = np.var(Xs, axis=0)
    drop_list: list[int] = []
    for _ in range(n_drop):
        np.fill_diagonal(corr, 0.0)
        flat = np.argmax(corr)
        i, j = divmod(flat, corr.shape[1])
        drop_idx = i if var[i] <= var[j] else j
        drop_list.append(drop_idx)
        corr[drop_idx, :] = 0.0
        corr[:, drop_idx] = 0.0
    return np.array(drop_list, dtype=np.int32)


def _prune_raw_redundancy(
    X: sparse.csr_matrix,
    y: np.ndarray,
    active_idx: np.ndarray,
    random_state: int,
    thr: float = 0.98,
) -> np.ndarray:
    n = X.shape[0]
    sub = _stratified_subsample_indices(
        y, max_n=min(8000, n), random_state=random_state
    )
    scores = _feature_abs_spearman_scores(X, y, active_idx, random_state)
    order = np.argsort(scores)[::-1]
    n_f = len(active_idx)
    Xs = X[sub][:, active_idx].toarray().astype(np.float32)
    kept = np.ones(n_f, dtype=bool)
    for rank_i, i in enumerate(order):
        if not kept[i]:
            continue
        col_i = Xs[:, i]
        if np.std(col_i) < 1e-9:
            kept[i] = False
            continue
        for j in order[rank_i + 1 :]:
            if not kept[j]:
                continue
            col_j = Xs[:, j]
            if np.std(col_j) < 1e-9:
                kept[j] = False
                continue
            r = abs(float(np.corrcoef(col_i, col_j)[0, 1]))
            if r > thr:
                kept[j] = False
    return active_idx[kept]


def _prune_leaf_overlap_redundancy(
    X: sparse.csr_matrix, active_idx: np.ndarray, n_raw: int, thr: float = 0.95
) -> np.ndarray:
    tree_idx = active_idx[active_idx >= n_raw]
    if len(tree_idx) <= 10:
        return active_idx
    n = X.shape[0]
    sub = _stratified_subsample_indices(
        np.zeros(n, dtype=np.int8), max_n=min(5000, n), random_state=42
    )
    Xs = np.asarray(X[sub][:, tree_idx].todense())
    binary = (Xs != 0).astype(np.float32)
    to_drop: list[int] = []
    for i in range(len(tree_idx)):
        for j in range(i + 1, len(tree_idx)):
            if tree_idx[i] in to_drop or tree_idx[j] in to_drop:
                continue
            inter = float(np.sum(binary[:, i] * binary[:, j]))
            uni = float(np.sum(binary[:, i] + binary[:, j] > 0))
            if uni > 0 and inter / uni > thr:
                to_drop.append(tree_idx[j])
    if not to_drop:
        return active_idx
    drop_set = set(to_drop)
    return np.array([a for a in active_idx if a not in drop_set], dtype=np.int32)


def _non_overlapping_stratified_buckets(
    y: np.ndarray, n_buckets: int = 5, random_state: int = 42
) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=np.float32)
    order = np.argsort(y_arr, kind="mergesort")
    n = len(y_arr)
    bucket_size = n // n_buckets
    buckets: list[np.ndarray] = []
    idx = order.copy()
    rng.shuffle(idx)
    for b in range(n_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < n_buckets - 1 else n
        buckets.append(idx[start:end])
    return buckets


def _prescreen_features(
    X: sparse.csr_matrix,
    y: np.ndarray,
    active_idx: np.ndarray,
    random_state: int,
    n_raw_total: int,
) -> np.ndarray:
    n = len(y)
    result = np.array(active_idx, dtype=np.int32)

    if len(result) <= 600:
        return result

    t0 = time.perf_counter()
    result = _prune_raw_redundancy(X, y, result, random_state, thr=0.97)
    result = _prune_leaf_overlap_redundancy(X, result, n_raw_total, thr=0.95)
    tprint_performance(
        "  Prescreen raw+tree redundancy prune (stage 0)", time.perf_counter() - t0
    )
    if len(result) <= 600:
        return result

    t0 = time.perf_counter()
    scores = _feature_abs_spearman_scores(X, y, result, random_state)
    thr_val = float(np.percentile(scores, 22.5))
    keep = scores >= thr_val
    result = result[keep]
    tprint_performance(
        "  Prescreen spearman scores (stage 1)", time.perf_counter() - t0
    )
    if len(result) <= 600:
        return result

    t0 = time.perf_counter()
    buckets = _non_overlapping_stratified_buckets(
        y, n_buckets=5, random_state=random_state
    )
    bucket_scores = np.zeros(len(result), dtype=np.float32)
    for bi, b_idx in enumerate(buckets):
        Xb = X[b_idx][:, result].toarray().astype(np.float32)
        yb = np.asarray(y[b_idx], dtype=np.float64)
        for fi in range(len(result)):
            col = Xb[:, fi]
            if np.std(col) < 1e-9:
                continue
            r = np.corrcoef(col, yb)[0, 1]
            bucket_scores[fi] += abs(float(r)) if np.isfinite(r) else 0.0
    bucket_scores /= max(len(buckets), 1)
    thr_val = float(np.percentile(bucket_scores, 22.5))
    keep = bucket_scores >= thr_val
    result = result[keep]
    tprint_performance(
        "  Prescreen bucket stability scores (stage 2)", time.perf_counter() - t0
    )

    tprint(f"  Prescreen final: {len(result)} features from {len(active_idx)}.")
    return result


def _stage_a_prune(
    X_combined: sparse.csr_matrix,
    y: np.ndarray,
    random_state: int,
    n_raw_total: int,
    prior_round_pred: np.ndarray | None,
    initial_features: np.ndarray | None = None,
    max_rounds: int = 5,
) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=np.int8)
    n = len(y_arr)
    active_idx = (
        np.asarray(initial_features, dtype=np.int32)
        if initial_features is not None
        else np.arange(X_combined.shape[1], dtype=np.int32)
    )

    tprint(
        f"Stage-A Prune: {len(active_idx)} initial features, {n} samples, max_rounds={max_rounds}."
    )
    t0_prescreen = time.perf_counter()
    active_idx = _prescreen_features(
        X_combined,
        y_arr,
        active_idx,
        random_state=random_state,
        n_raw_total=n_raw_total,
    )
    tprint_performance("Stage-A prescreen", time.perf_counter() - t0_prescreen)
    tprint(f"Stage-A Prune: {len(active_idx)} features after prescreen.")

    round_history: list[dict[str, Any]] = []
    last_round_oof = (
        np.asarray(prior_round_pred, dtype=np.float32)
        if prior_round_pred is not None
        else np.full(n, float(np.mean(y_arr)), dtype=np.float32)
    )

    _best_j_final = 0.0
    best_prev_j = 0.0
    best_prev_se = 0.0

    for round_id in range(1, max_rounds + 1):
        if len(active_idx) <= 40:
            tprint(
                f"  Round {round_id}: only {len(active_idx)} features left, stopping."
            )
            break
        t0_round = time.perf_counter()
        prev_active_idx = np.asarray(active_idx, dtype=np.int32).copy()
        prev_round_oof = np.asarray(last_round_oof, dtype=np.float32).copy()
        min_freq, min_net_support = _pruning_thresholds(round_id)
        tprint(
            f"  Round {round_id}/{max_rounds}: {len(active_idx)} features, "
            f"min_freq(f)={min_freq:.2f}, min_sign_consistency(s)={min_net_support:.2f}"
        )
        x_round = X_combined[:, active_idx]
        tprint(
            f"    Round {round_id}: feature set frozen at {x_round.shape[1]} "
            "columns for every CV fold and hyperparameter combo."
        )
        w_distillation = _compute_weight_distillation(
            y_arr, last_round_oof, prev_round_oof, is_classifier=True
        )

        fold_cv = StratifiedKFold(
            n_splits=EN_PRUNE_CV_SPLITS,
            shuffle=True,
            random_state=random_state + round_id,
        )
        hp_records: list[dict[str, Any]] = []

        n_hp = len(ALPHA_GRID) * len(L1_RATIO_GRID)
        tprint(
            f"    Evaluating {n_hp} hyperparameter combos across {EN_PRUNE_CV_SPLITS} folds "
            f"(cached fold scaling, subsample={EN_PRUNE_FOLD_SUBSAMPLE_ROWS}, "
            f"n_jobs={EN_PRUNE_N_JOBS}, tol={EN_PRUNE_TOL:g})..."
        )
        t0_hp = time.perf_counter()

        avg_prev_pred = None
        fold_cache: list[dict[str, Any]] = []
        for fold_id, (tr_idx, va_idx) in enumerate(
            fold_cv.split(np.zeros(n, dtype=np.int8), y_arr), start=1
        ):
            x_tr = x_round[tr_idx]
            y_tr = y_arr[tr_idx]
            w_tr = w_distillation[tr_idx]
            scaler = RobustScaler(with_centering=False)
            x_tr_s = scaler.fit_transform(x_tr)
            x_va_s = scaler.transform(x_round[va_idx])
            sub_idx = _stratified_subsample_indices(
                y_tr,
                max_n=EN_PRUNE_FOLD_SUBSAMPLE_ROWS,
                random_state=random_state + round_id + fold_id,
            )
            fold_cache.append(
                {
                    "tr_idx": tr_idx,
                    "va_idx": va_idx,
                    "x_tr_s": x_tr_s,
                    "x_va_s": x_va_s,
                    "y_tr": y_tr,
                    "w_tr": w_tr,
                    "sub_idx": sub_idx,
                }
            )

        hp_grid = [
            (alpha, l1_ratio) for alpha in ALPHA_GRID for l1_ratio in L1_RATIO_GRID
        ]
        hp_results = Parallel(n_jobs=EN_PRUNE_N_JOBS, prefer="threads")(
            delayed(_fit_en_prune_combo)(
                alpha,
                l1_ratio,
                fold_cache,
                y_arr,
                n,
                random_state,
            )
            for alpha, l1_ratio in hp_grid
        )
        hp_records = [r for r in hp_results if r is not None]
        convergence_hits = int(sum(r.get("convergence_hits", 0) for r in hp_records))
        if convergence_hits > 0:
            tprint(
                f"    EN convergence monitor: {convergence_hits} fold fits hit warning/max_iter "
                f"(max_iter={EN_PRUNE_MAX_ITER}, tol={EN_PRUNE_TOL:g})"
            )

        if avg_prev_pred is None and len(hp_records) > 0:
            avg_prev_pred = np.mean(
                np.stack([r["oof"] for r in hp_records], axis=0), axis=0
            )

        tprint_performance(f"    HP grid ({n_hp} combos)", time.perf_counter() - t0_hp)

        if len(hp_records) == 0:
            tprint(f"    No valid HP records in round {round_id}, stopping.")
            break

        for r in hp_records:
            r["J_final"] = (
                0.5 * r["J_final"]
                + 0.2 * r.get("stability30", 0.0)
                + 0.3 * r.get("top30_stability", 0.0)
            )

        best = max(hp_records, key=lambda z: float(z["J_final"]))
        all_scores = np.array(
            [float(r["J_final"]) for r in hp_records], dtype=np.float32
        )
        best_se = (
            float(np.std(all_scores, ddof=1) / max(np.sqrt(len(all_scores)), 1.0))
            if len(all_scores) > 1
            else 0.0
        )
        score_cut = float(best["J_final"]) - best_se
        contenders = [z for z in hp_records if float(z["J_final"]) >= score_cut]

        tprint(
            f"    Best J_final={best['J_final']:.4f}, SE={best_se:.4f}, "
            f"{len(contenders)}/{len(hp_records)} contenders above cut."
        )

        frontier = _pareto_frontier(contenders)
        tprint(f"    Frontier: {len(frontier)} non-dominated candidates.")
        chosen = _select_elbow_candidate(frontier)
        tprint(
            f"    Elbow: Alpha={chosen['alpha']}, L1={chosen['l1_ratio']}, "
            f"J_final={chosen['J_final']:.4f}"
        )

        pooled_coef = np.vstack(
            [coef for cand in contenders for coef in cand["fold_coefs"]]
        )
        del contenders
        total_models_in_pool = max(1, pooled_coef.shape[0])
        active_freq = np.mean(np.abs(pooled_coef) > 1e-6, axis=0)
        pos = np.sum(pooled_coef > 1e-6, axis=0)
        neg = np.sum(pooled_coef < -1e-6, axis=0)
        net_support = np.abs(pos - neg) / float(total_models_in_pool)

        present_mask = np.abs(pooled_coef) > 1e-6
        abs_coef = np.abs(pooled_coef)
        abs_median_magnitude = np.zeros(pooled_coef.shape[1], dtype=np.float32)
        for fi in range(pooled_coef.shape[1]):
            present_vals = abs_coef[present_mask[:, fi], fi]
            if len(present_vals) > 0:
                abs_median_magnitude[fi] = abs(float(np.median(present_vals)))
        del abs_coef, present_mask, pos, neg
        median_abs_magnitude = (
            abs(float(np.median(abs_median_magnitude[abs_median_magnitude > 0])))
            if np.any(abs_median_magnitude > 0)
            else 1.0
        )
        if median_abs_magnitude < 1e-12:
            median_abs_magnitude = 1.0
        rank_importance = abs_median_magnitude / np.float32(median_abs_magnitude)
        rank_importance = np.clip(rank_importance, 0.8, 1.2)
        tprint(
            f"    Rank-importance: median={float(np.median(rank_importance)):.3f}, "
            f"p10={float(np.percentile(rank_importance, 10)):.3f}, "
            f"p90={float(np.percentile(rank_importance, 90)):.3f}"
        )
        del pooled_coef, abs_median_magnitude

        eff_freq = active_freq * rank_importance
        eff_net_support = net_support * rank_importance
        keep_mask = (eff_freq >= min_freq) & (eff_net_support >= min_net_support)
        floor_keep = _floor_keep(round_id, len(active_idx))
        n_kept = int(np.sum(keep_mask))
        tprint(
            f"    Filter: {n_kept} features pass "
            f"eff_f>={min_freq:.2f} & eff_s>={min_net_support:.2f} "
            f"(floor_keep={floor_keep})"
        )
        if n_kept < floor_keep:
            rank_score = eff_freq * np.maximum(eff_net_support, 1e-8)
            top_k = min(len(rank_score), floor_keep)
            keep_mask = np.zeros(len(rank_score), dtype=bool)
            keep_mask[np.argsort(rank_score)[-top_k:]] = True
            tprint(
                f"    Floor override: keeping top {top_k} by eff_freq*eff_sign_score."
            )
        if int(np.sum(keep_mask)) < 40:
            tprint(
                f"    Only {int(np.sum(keep_mask))} features after pruning (< 40), stopping."
            )
            break

        candidate_active = active_idx[np.asarray(keep_mask, dtype=bool)]
        candidate_oof = np.asarray(chosen["oof"], dtype=np.float32)
        del chosen

        if round_id > 1 and _best_j_final > 0:
            current_j = float(hp_records[0]["J_final"]) if hp_records else 0.0
            current_j = float(best_prev_j)
            if current_j < _best_j_final - best_prev_se:
                tprint(
                    f"    Degradation stop: round J_final={current_j:.4f} "
                    f"< prev({_best_j_final:.4f} - {best_prev_se:.4f})"
                )
                round_history.append(
                    {
                        "round": round_id,
                        "stopped_by_degradation": True,
                        "round_J_final": current_j,
                        "prev_J_final": _best_j_final,
                    }
                )
                active_idx = prev_active_idx
                last_round_oof = prev_round_oof
                break

        _best_alpha = float(best["alpha"])
        _best_l1 = float(best["l1_ratio"])
        best_prev_j = float(best["J_final"])
        best_prev_se = float(best_se)
        _best_j_final = float(best["J_final"])
        _best_j_mean = float(best["J_mean"])
        _best_j_std = float(best["J_std"])
        _best_stability30 = float(best["stability30"])
        del best
        active_idx = candidate_active
        last_round_oof = candidate_oof
        round_history.append(
            {
                "round": round_id,
                "n_features_start": int(x_round.shape[1]),
                "n_features_end": int(len(active_idx)),
                "alpha": _best_alpha,
                "l1_ratio": _best_l1,
                "J_final": _best_j_final,
                "J_mean": _best_j_mean,
                "J_std": _best_j_std,
                "stability30": _best_stability30,
                "min_freq": float(min_freq),
                "min_net_support": float(min_net_support),
                "floor_keep": int(floor_keep),
                "active_freq_mean": float(np.mean(active_freq)),
                "net_support_mean": float(np.mean(net_support)),
            }
        )
        del hp_records
        tprint_performance(f"  Round {round_id}", time.perf_counter() - t0_round)
        gc.collect()

    tprint(
        f"Stage-A Prune complete: {len(active_idx)} features selected, "
        f"{len(round_history)} rounds."
    )
    return {
        "selected_indices": np.asarray(active_idx, dtype=np.int32),
        "pruning_history": round_history,
        "last_round_en_oof": np.asarray(last_round_oof, dtype=np.float32),
    }


def _fit_lgbm_tree_state_bundle(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray | None,
    random_state: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    models = []
    n_specs = len(LEAF_MODEL_SPECS)
    tprint(
        f"  LGBM tree-state bundle: fitting {n_specs} tree models on "
        f"{x_train.shape[0]} rows."
    )

    for si, spec in enumerate(LEAF_MODEL_SPECS):
        leaf_frac = float(spec["leaf_frac"])
        prefix = str(spec["prefix"])
        max_depth = int(spec["max_depth"])
        t0_tree = time.perf_counter()
        fit_idx = _stratified_subsample_indices(
            y_train, max_n=10000, random_state=random_state
        )
        fit_n = max(1, int(len(fit_idx)))
        lgbm = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            max_depth=max_depth,
            min_data_in_leaf=max(50, int(leaf_frac * fit_n)),
            min_sum_hessian_in_leaf=1e-3,
            feature_fraction=0.7,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            min_gain_to_split=0.001,
            max_bin=127,
            n_estimators=500,
            n_jobs=2,
            verbose=-1,
        )
        fit_kwargs = {}
        if y_eval is not None:
            fit_kwargs = {
                "eval_set": [(x_eval, y_eval)],
                "callbacks": [lgb.early_stopping(25, verbose=False)],
            }
        lgbm.fit(x_train[fit_idx], y_train[fit_idx], **fit_kwargs)
        models.append(lgbm)
        tprint_performance(
            f"    Tree-state model {si + 1}/{n_specs} ({prefix})",
            time.perf_counter() - t0_tree,
        )

    t0_features = time.perf_counter()
    feature_scales = _feature_scales(x_train)
    feature_bundle = build_tree_state_features(
        models,
        x_train,
        x_eval,
        feature_scales=feature_scales,
        config=cfg,
    )
    tprint_performance(
        "  Tree-state feature build+prune", time.perf_counter() - t0_features
    )
    meta = dict(feature_bundle["metadata"])
    tprint(
        "  LGBM tree-state bundle done: "
        f"train_tree={feature_bundle['X_tree_features'].shape}, "
        f"eval_tree={feature_bundle['X_valid_tree_features'].shape}, "
        f"raw_features={meta['n_features_raw']}, kept={meta['n_features_kept']}, "
        f"dropped={meta['n_dropped_by_pruning']}"
    )
    return {
        "models": models,
        "train_tree_matrix": feature_bundle["X_tree_features"],
        "eval_tree_matrix": feature_bundle["X_valid_tree_features"],
        "tree_feature_scales": feature_scales,
        "tree_feature_keep_indices": meta.get("keep_after_corr"),
        "tree_feature_metadata": meta,
        "tree_feature_names": feature_bundle["tree_feature_names"],
        "global_tree_feature_names": feature_bundle["global_tree_feature_names"],
        "train_global_tree_matrix": feature_bundle["global_tree_features"],
        "eval_global_tree_matrix": feature_bundle["global_tree_features_eval"],
    }


class RidgeOnLGBMModel:
    def __init__(self) -> None:
        self.lgb_models: list = []
        self.tree_feature_config: dict[str, Any] = {}
        self.tree_feature_scales: np.ndarray | None = None
        self.tree_feature_keep_indices: np.ndarray | None = None
        self.tree_feature_metadata: dict[str, Any] = {}
        self.tree_feature_names: list[str] = []
        self.global_tree_feature_names: list[str] = []
        self.selected_indices: np.ndarray = np.array([], dtype=np.int32)
        self.selected_raw_indices: np.ndarray = np.array([], dtype=np.int32)
        self.selected_tree_indices: np.ndarray = np.array([], dtype=np.int32)
        self.selected_feature_names: list[str] = []
        self.raw_feature_names: list[str] = []
        self.combined_feature_names: list[str] = []
        self.scaler: RobustScaler | None = None
        self.ridge: RidgeClassifier | None = None
        self.uncertainty_modulator: dict[str, Any] | None = None
        self.oof_probs: np.ndarray | None = None
        self.pruning_history: list = []
        self.confidence_norm: dict[str, float] = {}
        self.uncertainty_features: dict[str, np.ndarray] = {}

    def predict_proba(self, X) -> np.ndarray:
        x_np = (
            np.asarray(X, dtype=np.float32)
            if not isinstance(X, np.ndarray)
            else X.astype(np.float32)
        )
        if self.ridge is None or self.scaler is None or len(self.selected_indices) == 0:
            return np.full((x_np.shape[0], 2), 0.5, dtype=np.float32)
        tree_cfg = self.tree_feature_config
        if self.lgb_models and tree_cfg.get(
            "use_raw_features_with_tree_features", True
        ):
            tree_bundle = build_tree_state_features(
                self.lgb_models,
                x_np,
                x_np,
                feature_scales=self.tree_feature_scales,
                config=tree_cfg,
            )
            parts = [
                sparse.csr_matrix(x_np),
                sparse.csr_matrix(tree_bundle["X_tree_features"]),
            ]
            x_c = sparse.hstack(parts, format="csr")
        else:
            x_c = sparse.csr_matrix(x_np)
        x_sel = x_c[:, self.selected_indices]
        xs = self.scaler.transform(x_sel) if x_sel.shape[1] > 0 else x_sel
        score = self.ridge.decision_function(xs).astype(np.float32)
        if self.uncertainty_modulator and self.uncertainty_modulator.get("enabled"):
            global_f = tree_bundle.get(
                "global_tree_features", np.zeros((x_np.shape[0], 8), dtype=np.float32)
            )
            p = _predict_uncertainty_modulated(
                score, global_f, self.uncertainty_modulator
            )
        else:
            p = _sigmoid(score)
        return np.column_stack([1.0 - p, p]).astype(np.float32)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_uncertainty_features(self, X) -> dict[str, np.ndarray]:
        x_np = (
            np.asarray(X, dtype=np.float32)
            if not isinstance(X, np.ndarray)
            else X.astype(np.float32)
        )
        n = x_np.shape[0]
        out: dict[str, np.ndarray] = {}
        if self.oof_probs is not None and len(self.oof_probs) == n:
            out["oof_pred"] = self.oof_probs
        if (
            self.ridge is not None
            and self.scaler is not None
            and len(self.selected_indices) > 0
        ):
            p = self.predict(x_np)
            out["pred_mean"] = p
            out["pred_std"] = np.full(n, float(np.std(p)), dtype=np.float32)
            conf = np.abs(p - 0.5) * 2.0
            cn = self.confidence_norm or {"p5": 0.0, "p95": 1.0}
            lo = cn.get("p5", 0.0)
            hi = cn.get("p95", 1.0)
            if hi > lo:
                out["confidence_norm"] = np.clip((conf - lo) / (hi - lo), 0, 1).astype(
                    np.float32
                )
            else:
                out["confidence_norm"] = conf.astype(np.float32)
        else:
            for k in ("pred_mean", "pred_std", "confidence_norm"):
                out[k] = np.full(n, np.nan, dtype=np.float32)
        return out


def _fit_full_model_for_winner(
    x_np: np.ndarray,
    y_bin: np.ndarray,
    w_base: np.ndarray,
    selected_features_from_cv: np.ndarray,
    oof_pred: np.ndarray,
    random_state: int,
    tree_cfg: dict[str, Any],
    x_df: pd.DataFrame,
    selected_feature_names: list[str] | None = None,
) -> RidgeOnLGBMModel:
    tprint("  === Full-data fit for winning model ===")
    t0_full = time.perf_counter()
    model = RidgeOnLGBMModel()

    bundle_full = _fit_lgbm_tree_state_bundle(
        x_np, y_bin, x_np, None, random_state=random_state, cfg=tree_cfg
    )
    model.lgb_models = bundle_full["models"]
    model.tree_feature_config = tree_cfg
    model.tree_feature_scales = bundle_full["tree_feature_scales"]
    model.tree_feature_keep_indices = bundle_full["tree_feature_keep_indices"]
    model.tree_feature_metadata = bundle_full["tree_feature_metadata"]
    model.tree_feature_names = bundle_full["tree_feature_names"]
    model.global_tree_feature_names = bundle_full["global_tree_feature_names"]
    n_raw_full = x_np.shape[1] if tree_cfg["use_raw_features_with_tree_features"] else 0
    full_parts = []
    if n_raw_full > 0:
        full_parts.append(sparse.csr_matrix(x_np))
    full_parts.append(sparse.csr_matrix(bundle_full["train_tree_matrix"]))
    x_full_c = sparse.hstack(full_parts, format="csr")
    tprint(f"  Full combined matrix: {x_full_c.shape}")

    tprint(
        f"  Using {len(selected_features_from_cv)} features from CV (no re-pruning)."
    )

    model.scaler = RobustScaler(with_centering=False)
    model.raw_feature_names = list(x_df.columns)
    raw_names = (
        model.raw_feature_names
        if tree_cfg["use_raw_features_with_tree_features"]
        else []
    )
    model.combined_feature_names = raw_names + model.tree_feature_names
    if selected_feature_names:
        name_to_idx = {name: i for i, name in enumerate(model.combined_feature_names)}
        missing = [name for name in selected_feature_names if name not in name_to_idx]
        if missing:
            sample = ", ".join(missing[:8])
            raise ValueError(
                "Full-data RidgeOnLGBM feature space is missing "
                f"{len(missing)}/{len(selected_feature_names)} CV-selected features "
                f"(sample: {sample})."
            )
        model.selected_indices = np.asarray(
            [name_to_idx[name] for name in selected_feature_names], dtype=np.int32
        )
    else:
        idx = np.asarray(selected_features_from_cv, dtype=np.int32)
        if np.any((idx < 0) | (idx >= x_full_c.shape[1])):
            bad = idx[(idx < 0) | (idx >= x_full_c.shape[1])]
            raise IndexError(
                "CV-selected RidgeOnLGBM feature indices are out of range for "
                f"full-data matrix with {x_full_c.shape[1]} columns "
                f"(sample bad indices: {bad[:8].tolist()})."
            )
        model.selected_indices = idx
    model.selected_raw_indices = model.selected_indices[
        model.selected_indices < n_raw_full
    ]
    model.selected_tree_indices = model.selected_indices[
        model.selected_indices >= n_raw_full
    ]
    tprint(
        f"  Final feature split: {len(model.selected_raw_indices)} raw + "
        f"{len(model.selected_tree_indices)} tree."
    )
    x_selected = x_full_c[:, model.selected_indices]
    xs = (
        model.scaler.fit_transform(x_selected)
        if x_selected.shape[1] > 0
        else x_selected
    )
    w_final_full = _compute_weight_distillation(
        y_bin, oof_pred, oof_pred, is_classifier=True
    )
    model.ridge = RidgeClassifier(alpha=1.0, random_state=random_state)
    model.ridge.fit(xs, y_bin, sample_weight=w_final_full)

    train_score_base = model.ridge.decision_function(xs).astype(np.float32)
    p_train_base = _sigmoid(train_score_base)
    model.uncertainty_modulator = None
    if bool(tree_cfg.get("enable_uncertainty_score_modulation", True)):
        model.uncertainty_modulator = _fit_ridge_uncertainty_modulator(
            train_score_base,
            bundle_full["train_global_tree_matrix"],
            y_bin,
            sample_weight=w_final_full,
            random_state=random_state + 997,
            min_improvement=float(tree_cfg.get("uncertainty_min_improvement", 0.0)),
            correction_weight=float(tree_cfg.get("uncertainty_correction_weight", 0.2)),
        )

    model.selected_feature_names = [
        model.combined_feature_names[i] for i in model.selected_indices
    ]
    model.selected_raw_feature_names = [
        model.combined_feature_names[i]
        for i in model.selected_indices
        if i < n_raw_full
    ]
    model.selected_tree_feature_names = [
        model.combined_feature_names[i]
        for i in model.selected_indices
        if i >= n_raw_full
    ]

    conf_raw = np.abs(p_train_base - 0.5) * 2.0
    model.confidence_norm = {
        "p5": float(np.percentile(conf_raw, 5)) if len(conf_raw) else 0.0,
        "p95": float(np.percentile(conf_raw, 95)) if len(conf_raw) else 1.0,
    }

    del x_full_c, bundle_full, xs, x_selected
    gc.collect()

    tprint_performance("Full-data fit", time.perf_counter() - t0_full)
    return model


def train_ridge_on_lgbm_candidate(
    X,
    y,
    sample_weight=None,
    random_state=42,
    tree_feature_config: dict[str, Any] | None = None,
):
    tprint("RidgeOnLGBM: starting candidate training.")
    t0_total = time.perf_counter()
    tree_cfg = _tree_feature_config(**(tree_feature_config or {}))
    y_bin = np.asarray(y >= 0.5, dtype=np.int8)
    x_df = X if hasattr(X, "columns") else pd.DataFrame(X)
    x_np = x_df.to_numpy(dtype=np.float32)
    n = len(y_bin)
    pos_rate = float(np.mean(y_bin))
    tprint(f"  Input: {n} samples, {x_np.shape[1]} features, pos_rate={pos_rate:.3f}.")
    if lgb is None or n < 200:
        tprint("  Skipping: lightgbm unavailable or n < 200.")
        return None

    w_base = (
        np.ones(n, dtype=np.float32)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float32)
    )

    race_idx = _stratified_subsample_indices(
        y_bin, max_n=min(RACE_MAX_ROWS, n), random_state=random_state + 101
    )
    x_race = x_np[race_idx]
    y_race = y_bin[race_idx]
    w_race = w_base[race_idx]
    tprint(f"  Race subsample: {len(race_idx)} rows.")

    local_idx = np.arange(len(y_race), dtype=np.int32)
    select_local, eval_local = train_test_split(
        local_idx,
        test_size=RACE_EVAL_FRACTION,
        stratify=y_race,
        random_state=random_state + 202,
    )
    select_local = np.asarray(select_local, dtype=np.int32)
    eval_local = np.asarray(eval_local, dtype=np.int32)
    x_select = x_race[select_local]
    y_select = y_race[select_local]
    w_select = w_race[select_local]
    x_eval = x_race[eval_local]
    y_eval = y_race[eval_local]
    tprint(
        "  Honest prune/eval split: "
        f"select_train={len(select_local)}, eval={len(eval_local)}. "
        f"Stage-A pruning runs once on select_train with {EN_PRUNE_CV_SPLITS} CV folds."
    )

    t0_select = time.perf_counter()
    tprint("  Fitting LGBM tree-state bundle for honest selection/eval split...")
    bundle = _fit_lgbm_tree_state_bundle(
        x_select,
        y_select,
        x_eval,
        y_eval,
        random_state=random_state,
        cfg=tree_cfg,
    )
    raw_eval_pred = bundle["models"][0].predict_proba(x_eval)[:, 1].astype(np.float32)
    raw_lgbm_metrics = _metric_pack(y_eval, raw_eval_pred)

    tprint("  Building combined sparse matrix for single Stage-A pruning run...")
    select_parts = []
    eval_parts = []
    n_raw_total = (
        x_select.shape[1] if tree_cfg["use_raw_features_with_tree_features"] else 0
    )
    if n_raw_total > 0:
        select_parts.append(sparse.csr_matrix(x_select))
        eval_parts.append(sparse.csr_matrix(x_eval))
    select_parts.append(sparse.csr_matrix(bundle["train_tree_matrix"]))
    eval_parts.append(sparse.csr_matrix(bundle["eval_tree_matrix"]))
    x_select_c = sparse.hstack(select_parts, format="csr")
    x_eval_c = sparse.hstack(eval_parts, format="csr")
    tprint(f"  Combined matrix: select={x_select_c.shape}, eval={x_eval_c.shape}")

    tprint("  Running single Stage-A ElasticNet pruning pass...")
    prune = _stage_a_prune(
        x_select_c,
        y_select,
        random_state=random_state,
        n_raw_total=n_raw_total,
        prior_round_pred=None,
        initial_features=None,
        max_rounds=7,
    )
    selected_features = np.asarray(prune["selected_indices"], dtype=np.int32)
    pruning_history = list(prune.get("pruning_history", []))
    last_en_oof = np.asarray(prune["last_round_en_oof"], dtype=np.float32)
    tprint(
        f"  Single Stage-A selection complete: {len(selected_features)} features "
        f"selected ({len(prune['pruning_history'])} rounds)."
    )

    sel_raw = selected_features[selected_features < n_raw_total]
    sel_tree = selected_features[selected_features >= n_raw_total]
    tprint(f"  Selected: {len(sel_raw)} raw + {len(sel_tree)} tree features.")

    scaler = RobustScaler(with_centering=False)
    xsel = x_select_c[:, selected_features]
    xev = x_eval_c[:, selected_features]
    xsel_s = scaler.fit_transform(xsel) if xsel.shape[1] > 0 else xsel
    xev_s = scaler.transform(xev) if xev.shape[1] > 0 else xev

    ridge = RidgeClassifier(alpha=1.0, random_state=random_state)
    w_distillation = _compute_weight_distillation(
        y_select, last_en_oof, last_en_oof, is_classifier=True
    )
    w_final = (w_distillation * w_select).astype(np.float32, copy=False)
    ridge.fit(xsel_s, y_select, sample_weight=w_final)
    score_select_base = ridge.decision_function(xsel_s).astype(np.float32)
    score_eval_base = ridge.decision_function(xev_s).astype(np.float32)
    if bool(tree_cfg.get("enable_uncertainty_score_modulation", True)):
        unc_mod = _fit_ridge_uncertainty_modulator(
            score_select_base,
            bundle["train_global_tree_matrix"],
            y_select,
            sample_weight=w_final,
            random_state=random_state + 313,
            min_improvement=float(tree_cfg.get("uncertainty_min_improvement", 0.0)),
            correction_weight=float(tree_cfg.get("uncertainty_correction_weight", 0.2)),
        )
        eval_pred = _predict_uncertainty_modulated(
            score_eval_base, bundle["eval_global_tree_matrix"], unc_mod
        )
    else:
        eval_pred = _sigmoid(score_eval_base)

    oof_race = np.full(len(y_race), np.nan, dtype=np.float32)
    raw_lgbm2p_oof_race = np.full(len(y_race), np.nan, dtype=np.float32)
    oof_race[eval_local] = eval_pred.astype(np.float32)
    raw_lgbm2p_oof_race[eval_local] = raw_eval_pred
    ridge_eval_metrics = _metric_pack(y_eval, eval_pred)
    tprint_performance(
        "  Honest RidgeOnLGBM selection+eval", time.perf_counter() - t0_select
    )

    tprint(
        f"  Race OOF complete. Raw LGBM lift30={raw_lgbm_metrics.get('lift30', 0):.3f}, "
        f"auc={raw_lgbm_metrics.get('auc', 0):.3f}"
    )

    oof_full = np.full(n, np.nan, dtype=np.float32)
    oof_full[race_idx] = oof_race
    oof_for_full_fit = np.where(
        np.isfinite(oof_full), oof_full, float(np.mean(y_bin))
    ).astype(np.float32)

    selected_features_union = np.array(selected_features, dtype=np.int32)
    raw_names = (
        list(x_df.columns) if tree_cfg["use_raw_features_with_tree_features"] else []
    )
    combined_feature_names = raw_names + list(bundle["tree_feature_names"])
    selected_feature_names = [
        combined_feature_names[i] for i in selected_features_union
    ]
    agg_oof = _aggregate_j([ridge_eval_metrics])
    metrics = dict(ridge_eval_metrics)
    metrics.update(agg_oof)
    j_final_oof = float(metrics.get("J_final", 0.0))
    metrics["J_final_oof"] = j_final_oof
    metrics["J_Score"] = j_final_oof
    metrics["feature_count"] = int(len(selected_features_union))
    metrics["n_raw_features_kept"] = int(np.sum(selected_features_union < n_raw_total))
    metrics["n_leaf_features_kept"] = int(
        np.sum(selected_features_union >= n_raw_total)
    )
    metrics["race_n"] = int(len(race_idx))
    metrics["oof_coverage"] = float(
        np.sum(np.isfinite(oof_full)) / max(metrics["race_n"], 1)
    )

    for key in [
        "J_final_oof",
        "lift30",
        "auc_correct_30",
        "stability30",
        "auc",
        "brier",
        "ece",
        "top30_correctness_rate",
        "overall_correctness_rate",
        "feature_count",
        "oof_std",
        "oof_coverage",
    ]:
        tprint(f"    {key}: {metrics.get(key)}")

    del bundle, prune, x_select_c, x_eval_c, xsel, xev, xsel_s, xev_s
    gc.collect()
    tprint_performance("RidgeOnLGBM total training", time.perf_counter() - t0_total)

    return {
        "model": None,
        "metrics": metrics,
        "raw_lgbm_2p_metrics": raw_lgbm_metrics,
        "raw_lgbm_metrics": raw_lgbm_metrics,
        "oof_probs": oof_full,
        "pruning_history": pruning_history,
        "last_round_en_oof": oof_race,
        "selected_features_from_cv": selected_features_union,
        "selected_feature_names": selected_feature_names,
        "full_fit_needed": True,
        "tree_cfg": tree_cfg,
        "race_idx": race_idx,
        "oof_race": oof_for_full_fit,
    }
