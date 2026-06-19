"""Feature engineering for current-regime specialist selection.

This module builds a compact feature set for distinguishing the current local
regime from historical rows. It is intentionally separate from
``regime_specialist_similarity``: this file decides which raw-state features
are useful regime discriminators, while the similarity module uses those
features and optional discriminator scores to rank analogue windows.
"""

from __future__ import annotations

import itertools
import math
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REGIME_FEATURE_ENGINEERING_SCHEMA_VERSION = "regime_specialist_feature_engineering_v1"


def _log(message: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] Regime feature engineering: {message}", flush=True)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.1f}s"

EXCLUDE_TOKENS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "forward",
    "pnl",
    "wallet",
    "exit_price",
    "entry_price",
    "prediction",
    "pred_",
    "_pred",
    "uncertainty",
    "entropy_uncertainty",
    "error_risk",
    "correctness",
    "model_score",
    "deployment_score",
    "similarity_to_current",
    "regime_specialist",
    "drift",
    "knn",
    "nearest",
    "neighbor",
    "distance",
    "reconstruction",
    "anomaly",
    "rarity",
    "psi",
    "ks",
    "wasserstein",
    "mahalanobis",
)

EXCLUDE_EXACT_NAMES: tuple[str, ...] = (
    "timestamp",
    "time",
    "symbol",
    "asset",
    "pair",
    "instrument",
    "instrument_id",
    "strategy_id",
    "id",
)

APPROVED_MARKET_TOKENS: tuple[str, ...] = (
    "volatility",
    "volume",
    "dollar_volume",
    "liquidity",
    "spread",
    "funding",
    "open_interest",
    "oi_",
    "breadth",
    "dispersion",
    "correlation",
    "corr",
    "trend",
    "momentum",
    "range",
    "atr",
    "return",
    "carry",
    "rank",
    "tier",
    "entropy",
)


@dataclass(frozen=True)
class RegimeFeatureEngineeringConfig:
    min_finite_coverage: float = 0.70
    max_missingness_gap: float = 0.30
    max_dominant_value_share: float = 0.98
    min_unique_values: int = 8
    univariate_folds: int = 5
    univariate_subsample_per_class: int = 8000
    corr_subsample_per_class: int = 12000
    relief_subsample_per_class: int = 6000
    random_state: int = 42

    min_sign_consistency: float = 0.60
    min_fold_pass_rate: float = 0.50
    corr_cluster_threshold: float = 0.96
    max_cluster_features: int = 80
    max_backfill_pool: int = 150
    max_relief_lgbm_only: int = 50
    max_pair_parent_features: int = 50
    max_pair_candidates: int = 2500
    max_pairs_kept: int = 200
    max_final_features: int = 40
    max_model_features: int = 50

    lgbm_enabled: bool = True
    elasticnet_enabled: bool = True
    lgbm_max_samples_per_class: int = 25000
    elasticnet_max_samples_per_class: int = 30000
    grouped_cv_folds: int = 5
    grouped_cv_repeats: int = 3
    one_se_feature_sizes: tuple[int, ...] = (10, 20, 30, 40, 50)
    permutation_repeats: int = 2
    max_permutation_features: int = 80
    max_permutation_rows: int = 4000
    max_shap_rows: int = 4000
    drift_window_days: float = 28.0
    max_drift_raw_features: int = 80
    drift_window_max_rows: int = 20000
    drift_knn_max_rows: int = 4000
    drift_knn_chunk_pairs: int = 2_000_000
    max_tail_pair_features: int = 20
    pca_components: int = 8
    ae_components: int = 8
    ae_max_samples: int = 20000
    ae_max_iter: int = 80
    domain_score_smoothing_enabled: bool = True
    domain_score_ewma_half_life_days: float = 1.0
    domain_score_ewma_max_days: float = 4.0
    run_validation_diagnostics: bool = False
    shadow_random_state_offset: int = 7919
    eps: float = 1e-12


@dataclass
class PairFeatureSpec:
    name: str
    left: str
    right: str
    term: str
    score: float
    parent_quality: float
    incremental_lift: float
    shadow_margin: float
    stability: float
    sign_consistency: float = 0.0
    fold_pass_rate: float = 0.0


@dataclass
class RegimeFeatureEngineeringArtifact:
    schema_version: str
    selected_features: list[str]
    selected_raw_features: list[str]
    selected_pair_features: list[str]
    selected_drift_features: list[str]
    lgbm_features: list[str]
    elasticnet_features: list[str]
    pair_features: list[PairFeatureSpec]
    lgbm_feature_scores: dict[str, float]
    elasticnet_feature_scores: dict[str, float]
    final_feature_scores: dict[str, float]
    row_scores: pd.DataFrame
    materialized_features: pd.DataFrame
    materialized_feature_groups: dict[str, list[str]]
    feature_report: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _is_excluded_feature(name: str) -> bool:
    low = str(name).lower()
    if low in EXCLUDE_EXACT_NAMES:
        return True
    for token in EXCLUDE_TOKENS:
        if token in {"psi", "ks"}:
            if re.search(rf"(^|_){re.escape(token)}($|_)", low):
                return True
            continue
        if token in low:
            return True
    return False


def _is_approved_market_feature(name: str) -> bool:
    low = str(name).lower()
    return any(token in low for token in APPROVED_MARKET_TOKENS)


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _safe_materialized_name(prefix: str, name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not safe:
        safe = "feature"
    return f"{prefix}{safe}"


def _candidate_pool(
    frame: pd.DataFrame,
    candidate_features: Sequence[str] | None,
) -> list[str]:
    if candidate_features is None:
        raw = list(frame.columns)
    else:
        raw = [str(col) for col in candidate_features if str(col) in frame.columns]
    out: list[str] = []
    for col in raw:
        if _is_excluded_feature(col):
            continue
        if candidate_features is None and not _is_approved_market_feature(col):
            continue
        out.append(col)
    return _stable_unique(out)


def _numeric_matrix(
    frame: pd.DataFrame, columns: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    cols: list[str] = []
    arrays: list[np.ndarray] = []
    for col in columns:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        arr = vals.to_numpy(dtype=np.float32, copy=True)
        cols.append(str(col))
        arrays.append(arr)
    if not arrays:
        return np.zeros((len(frame), 0), dtype=np.float32), []
    return np.column_stack(arrays).astype(np.float32, copy=False), cols


def _clean_columns(
    matrix: np.ndarray,
    columns: Sequence[str],
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return (
            np.zeros((len(y), 0), dtype=np.float32),
            [],
            {"kept": 0, "dropped": len(columns)},
        )
    sample = (
        np.asarray(sample_mask, dtype=bool)
        if sample_mask is not None
        else np.ones(len(y), dtype=bool)
    )
    if sample.size != len(y) or not bool(sample.any()):
        sample = np.ones(len(y), dtype=bool)
    keep: list[int] = []
    reasons: dict[str, int] = {
        "coverage": 0,
        "missingness_gap": 0,
        "constant": 0,
        "dominant": 0,
    }
    y_bool = y.astype(bool)
    y_sample = y_bool[sample]
    for j, col in enumerate(columns):
        vals = matrix[sample, j]
        finite = np.isfinite(vals)
        coverage = float(np.mean(finite)) if finite.size else 0.0
        if coverage < float(config.min_finite_coverage):
            reasons["coverage"] += 1
            continue
        miss_pos = (
            1.0 - float(np.mean(finite[y_sample])) if bool(y_sample.any()) else 1.0
        )
        miss_neg = (
            1.0 - float(np.mean(finite[~y_sample])) if bool((~y_sample).any()) else 1.0
        )
        if abs(miss_pos - miss_neg) > float(config.max_missingness_gap):
            reasons["missingness_gap"] += 1
            continue
        finite_vals = vals[finite]
        if finite_vals.size < int(config.min_unique_values):
            reasons["constant"] += 1
            continue
        rounded = np.round(finite_vals.astype(np.float64), 8)
        unique, counts = np.unique(rounded, return_counts=True)
        if unique.size < int(config.min_unique_values):
            reasons["constant"] += 1
            continue
        if float(np.max(counts) / max(np.sum(counts), 1)) > float(
            config.max_dominant_value_share
        ):
            reasons["dominant"] += 1
            continue
        keep.append(j)
    kept = (
        matrix[:, keep].astype(np.float32, copy=False)
        if keep
        else np.zeros((len(y), 0), dtype=np.float32)
    )
    kept_cols = [str(columns[j]) for j in keep]
    return (
        kept,
        kept_cols,
        {
            "kept": len(kept_cols),
            "dropped": len(columns) - len(kept_cols),
            "reasons": reasons,
        },
    )


def _per_symbol_robust_z(
    matrix: np.ndarray,
    symbols: Sequence[Any] | None,
    fit_mask: np.ndarray,
    eps: float,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return arr
    fit = np.asarray(fit_mask, dtype=bool)
    if fit.size != arr.shape[0] or not bool(fit.any()):
        fit = np.ones(arr.shape[0], dtype=bool)

    def _center_scale(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals = np.asarray(rows, dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, np.nan)
        if vals.ndim != 2 or vals.shape[0] == 0:
            return (
                np.zeros(arr.shape[1], dtype=np.float32),
                np.ones(arr.shape[1], dtype=np.float32),
            )
        center = np.nanmedian(vals, axis=0)
        center = np.where(np.isfinite(center), center, 0.0)
        mad = np.nanmedian(np.abs(vals - center.reshape(1, -1)), axis=0)
        scale = 1.4826 * mad
        std = np.nanstd(vals, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > eps), scale, std)
        scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)
        return center.astype(np.float32), scale.astype(np.float32)

    global_center, global_scale = _center_scale(arr[fit])
    if symbols is None:
        filled = np.where(np.isfinite(arr), arr, global_center.reshape(1, -1))
        z = (filled - global_center.reshape(1, -1)) / np.maximum(
            global_scale.reshape(1, -1),
            eps,
        )
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    symbols_arr = pd.Series(symbols).astype(str).to_numpy()
    if symbols_arr.size != arr.shape[0]:
        filled = np.where(np.isfinite(arr), arr, global_center.reshape(1, -1))
        z = (filled - global_center.reshape(1, -1)) / np.maximum(
            global_scale.reshape(1, -1),
            eps,
        )
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    out = np.empty_like(arr, dtype=np.float32)
    for sym in pd.unique(symbols_arr):
        mask = symbols_arr == sym
        fit_sym = mask & fit
        if int(np.sum(fit_sym)) < 10:
            center = global_center
            scale = global_scale
        else:
            center, scale = _center_scale(arr[fit_sym])
        raw = arr[mask]
        filled = np.where(np.isfinite(raw), raw, center.reshape(1, -1))
        z = (filled - center.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), eps)
        out[mask] = np.clip(z, -8.0, 8.0).astype(np.float32)
    return out.astype(np.float32, copy=False)


def _balanced_sample_indices(
    y: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y_bool = np.asarray(y, dtype=bool)
    pos = np.flatnonzero(y_bool)
    neg = np.flatnonzero(~y_bool)
    if pos.size == 0 or neg.size == 0:
        return np.arange(len(y_bool), dtype=np.int64)
    n_pos = min(int(max_per_class), pos.size)
    n_neg = min(int(max_per_class), neg.size)
    pos_sel = rng.choice(pos, size=n_pos, replace=False)
    neg_sel = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos_sel, neg_sel]).astype(np.int64)
    rng.shuffle(idx)
    return idx


def _balanced_sample_from_indices(
    y: np.ndarray,
    indices: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return idx
    local = _balanced_sample_indices(y[idx], max_per_class, rng)
    return idx[local]


def _stratified_fold_indices(
    y: np.ndarray,
    indices: np.ndarray,
    *,
    n_splits: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return []
    labels = np.asarray(y, dtype=bool)[idx]
    pos = idx[labels]
    neg = idx[~labels]
    if pos.size == 0 or neg.size == 0:
        return [idx]
    folds = max(2, min(int(n_splits), int(pos.size), int(neg.size)))
    pos = pos.copy()
    neg = neg.copy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    out: list[np.ndarray] = []
    for fold in range(folds):
        val = np.concatenate([pos[fold::folds], neg[fold::folds]]).astype(np.int64)
        if val.size:
            rng.shuffle(val)
            out.append(val)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals, kind="mergesort")
    sorted_vals = vals[order]
    ranks = np.empty(len(vals), dtype=np.float64)
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        # Average of 1-indexed ranks i+1 through j.
        avg_rank = 0.5 * (float(i + 1) + float(j))
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _auc_lift(values: np.ndarray, y: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    labels = np.asarray(y, dtype=bool)
    mask = np.isfinite(vals)
    vals = vals[mask]
    labels = labels[mask]
    n_pos = int(np.sum(labels))
    n_neg = int(np.sum(~labels))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = _average_ranks(vals)
    pos_rank_sum = float(np.sum(ranks[labels]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(2.0 * abs(auc - 0.5))


def _ks_stat(values: np.ndarray, y: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    labels = np.asarray(y, dtype=bool)
    mask = np.isfinite(vals)
    vals = vals[mask]
    labels = labels[mask]
    pos = np.sort(vals[labels])
    neg = np.sort(vals[~labels])
    if pos.size == 0 or neg.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([pos, neg])))
    pos_cdf = np.searchsorted(pos, grid, side="right") / max(pos.size, 1)
    neg_cdf = np.searchsorted(neg, grid, side="right") / max(neg.size, 1)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def _median_shift(values: np.ndarray, y: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    labels = np.asarray(y, dtype=bool)
    pos = vals[labels & np.isfinite(vals)]
    neg = vals[(~labels) & np.isfinite(vals)]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    return float(np.nanmedian(pos) - np.nanmedian(neg))


def _group_values(
    frame: pd.DataFrame,
    timestamp_col: str,
    active_len: int,
) -> np.ndarray:
    if timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        if ts.notna().any():
            return ts.dt.floor("D").astype("int64").to_numpy(dtype=np.int64, copy=False)
    return np.arange(active_len, dtype=np.int64)


def _repeated_grouped_fold_indices(
    y: np.ndarray,
    fit_indices: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    repeats: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    fit_idx = np.asarray(fit_indices, dtype=np.int64)
    if fit_idx.size == 0:
        return []
    if groups.size != len(y):
        groups = np.arange(len(y), dtype=np.int64)
    group_vals = np.asarray(groups)[fit_idx]
    unique_groups = np.unique(group_vals)
    if unique_groups.size < 2:
        folds = _stratified_fold_indices(y, fit_idx, n_splits=n_splits, rng=rng)
        return [
            (np.setdiff1d(fit_idx, valid, assume_unique=False), valid)
            for valid in folds
        ]
    n = max(2, min(int(n_splits), int(unique_groups.size)))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _repeat in range(max(1, int(repeats))):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        fold_groups = [list() for _ in range(n)]
        fold_rows = np.zeros(n, dtype=np.int64)
        fold_pos = np.zeros(n, dtype=np.int64)
        for group in shuffled:
            rows = fit_idx[group_vals == group]
            pos = int(np.sum(y[rows]))
            score = fold_rows + 2 * np.abs(fold_pos - np.mean(fold_pos))
            dest = int(np.argmin(score))
            fold_groups[dest].append(group)
            fold_rows[dest] += len(rows)
            fold_pos[dest] += pos
        for groups_for_fold in fold_groups:
            if not groups_for_fold:
                continue
            valid_mask = np.isin(group_vals, np.asarray(groups_for_fold))
            valid_idx = fit_idx[valid_mask]
            train_idx = fit_idx[~valid_mask]
            if valid_idx.size == 0 or train_idx.size == 0:
                continue
            if (
                np.unique(y[valid_idx].astype(int)).size < 2
                or np.unique(y[train_idx].astype(int)).size < 2
            ):
                continue
            out.append((train_idx.astype(np.int64), valid_idx.astype(np.int64)))
    return out


def _rank01(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=np.float64)
    max_v = float(np.nanmax(arr)) if arr.size else 0.0
    if not np.isfinite(max_v) or max_v <= 0.0:
        return {key: 0.0 for key in values}
    return {key: float(max(value, 0.0) / max_v) for key, value in values.items()}


def _predict_proba_1d(model: Any, matrix: np.ndarray) -> np.ndarray:
    pred = model.predict_proba(matrix)
    if pred.ndim == 2 and pred.shape[1] > 1:
        return pred[:, 1].astype(np.float32, copy=False)
    return np.asarray(pred).reshape(-1).astype(np.float32, copy=False)


class _LightGBMBoosterClassifier:
    def __init__(self, booster: Any):
        self.booster_ = booster

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        pred = np.asarray(self.booster_.predict(matrix), dtype=np.float32).reshape(-1)
        pred = np.clip(pred, 0.0, 1.0)
        return np.column_stack([1.0 - pred, pred]).astype(np.float32)


class _ScaledClassifier:
    def __init__(self, model: Any, center: np.ndarray, scale: np.ndarray):
        self.model = model
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    def _transform(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float32)
        center = self.center.reshape(1, -1)
        scale = np.maximum(self.scale.reshape(1, -1), 1e-12)
        x = np.where(np.isfinite(x), x, center)
        return ((x - center) / scale).astype(np.float32)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._transform(matrix))


def _balanced_binary_weights(y_train: np.ndarray, eps: float) -> np.ndarray:
    labels = np.asarray(y_train, dtype=bool)
    n_pos = max(float(np.sum(labels)), eps)
    n_neg = max(float(np.sum(~labels)), eps)
    total = float(len(labels))
    weights = np.where(labels, total / (2.0 * n_pos), total / (2.0 * n_neg))
    return weights.astype(np.float32)


def _train_lightgbm_core_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, np.ndarray, str]:
    import lightgbm as lgb

    train_set = lgb.Dataset(
        x_train,
        label=y_train.astype(int),
        weight=_balanced_binary_weights(y_train, 1e-12),
        free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "max_depth": 3,
        "num_leaves": 8,
        "learning_rate": 0.03,
        "min_child_samples": 20,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "feature_fraction": 0.7,
        "lambda_l1": 0.5,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "num_threads": 1,
    }
    booster = lgb.train(params, train_set, num_boost_round=200)
    importance = np.asarray(
        booster.feature_importance(importance_type="gain"), dtype=np.float64
    )
    return _LightGBMBoosterClassifier(booster), importance, "lightgbm_core"


def _train_lgbm_like_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, np.ndarray, str]:
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            max_depth=3,
            learning_rate=0.03,
            n_estimators=200,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            class_weight="balanced",
            reg_alpha=0.5,
            reg_lambda=10.0,
            n_jobs=1,
            random_state=int(seed),
            verbosity=-1,
        )
        model.fit(x_train, y_train.astype(int))
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
        return model, importance, "lightgbm"
    except Exception:
        try:
            return _train_lightgbm_core_model(x_train, y_train, seed=seed)
        except Exception:
            pass
    try:
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=1,
        )
        model.fit(x_train, y_train.astype(int))
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
        return model, importance, "extratrees_fallback"
    except Exception:
        raise


def _train_elasticnet_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    eps: float,
) -> tuple[Any, np.ndarray, str, float]:
    from sklearn.linear_model import LogisticRegression

    x_scaled, center, scale = _robust_standardize_matrix(
        x_train,
        np.ones(len(x_train), dtype=bool),
        eps,
    )
    best_model = None
    best_score = -np.inf
    for alpha in (0.5, 1.0, 2.0):
        c_val = 1.0 / max(float(alpha), eps)
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=c_val,
            class_weight="balanced",
            max_iter=500,
            random_state=int(seed),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", message=".*coef_ did not converge.*")
            model.fit(x_scaled, y_train.astype(int))
        pred = _predict_proba_1d(model, x_scaled)
        score = _auc_lift(pred, y_train)
        if score > best_score:
            best_score = score
            best_model = model
    if best_model is None:
        raise RuntimeError("no_elasticnet_model")
    coef = np.abs(np.asarray(best_model.coef_, dtype=np.float64).ravel())
    return (
        _ScaledClassifier(best_model, center, scale),
        coef,
        "elasticnet",
        float(best_score),
    )


def _train_model(
    kind: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    eps: float,
) -> tuple[Any, np.ndarray, str]:
    if kind == "elasticnet":
        model, importance, model_name, _score = _train_elasticnet_model(
            x_train,
            y_train,
            seed=seed,
            eps=eps,
        )
        return model, importance, model_name
    return _train_lgbm_like_model(x_train, y_train, seed=seed)


def _lightgbm_contribution_importance(
    model: Any,
    x_valid: np.ndarray,
    *,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if x_valid.size == 0:
        return np.zeros(x_valid.shape[1] if x_valid.ndim == 2 else 0, dtype=np.float64)
    try:
        if len(x_valid) > int(max_rows) > 0:
            rows = rng.choice(
                np.arange(len(x_valid)), size=int(max_rows), replace=False
            )
            x_use = x_valid[rows]
        else:
            x_use = x_valid
        booster = getattr(model, "booster_", None)
        if booster is None:
            return np.zeros(x_valid.shape[1], dtype=np.float64)
        contrib = booster.predict(x_use, pred_contrib=True)
        contrib = np.asarray(contrib, dtype=np.float64)
        if contrib.ndim != 2 or contrib.shape[1] < x_valid.shape[1]:
            return np.zeros(x_valid.shape[1], dtype=np.float64)
        return np.nanmean(np.abs(contrib[:, : x_valid.shape[1]]), axis=0)
    except Exception:
        return np.zeros(x_valid.shape[1], dtype=np.float64)


def _permutation_importance(
    model: Any,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_indices: Sequence[int],
    *,
    repeats: int,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if (
        x_valid.ndim != 2
        or x_valid.shape[1] == 0
        or np.unique(y_valid.astype(int)).size < 2
    ):
        return np.zeros(x_valid.shape[1] if x_valid.ndim == 2 else 0, dtype=np.float64)
    x_eval = x_valid
    y_eval = y_valid
    if len(x_eval) > int(max_rows) > 0:
        rows = rng.choice(np.arange(len(x_eval)), size=int(max_rows), replace=False)
        x_eval = x_eval[rows]
        y_eval = y_eval[rows]
    if np.unique(y_eval.astype(int)).size < 2:
        return np.zeros(x_valid.shape[1], dtype=np.float64)
    base = _auc_lift(_predict_proba_1d(model, x_eval), y_eval)
    out = np.zeros(x_valid.shape[1], dtype=np.float64)
    for idx in feature_indices:
        if idx < 0 or idx >= x_valid.shape[1]:
            continue
        drops: list[float] = []
        x_perm = x_eval.copy()
        original_col = x_perm[:, idx].copy()
        for _ in range(max(1, int(repeats))):
            x_perm[:, idx] = rng.permutation(original_col)
            drops.append(
                max(base - _auc_lift(_predict_proba_1d(model, x_perm), y_eval), 0.0)
            )
        x_perm[:, idx] = original_col
        out[idx] = float(np.nanmean(drops)) if drops else 0.0
    return out


def _cv_rank_and_scores(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    y: np.ndarray,
    fit_mask: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    max_per_class: int,
    config: RegimeFeatureEngineeringConfig,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    stage_start = time.perf_counter()
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": "empty_matrix"},
        )
    rng = np.random.default_rng(
        int(config.random_state) + (101 if kind == "lgbm" else 211)
    )
    fit_idx = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    if fit_idx.size == 0 or np.unique(y[fit_idx].astype(int)).size < 2:
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": "single_class_fit_rows"},
        )
    x_all = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    folds = _repeated_grouped_fold_indices(
        y,
        fit_idx,
        groups,
        n_splits=int(config.grouped_cv_folds),
        repeats=int(config.grouped_cv_repeats),
        rng=rng,
    )
    if not folds:
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": "no_valid_grouped_cv_folds"},
        )
    _log(
        f"{kind} CV ranking start: rows={matrix.shape[0]} features={matrix.shape[1]} "
        f"fit_rows={fit_idx.size} folds={len(folds)} max_per_class={int(max_per_class)}"
    )
    oof_sum = np.zeros(len(y), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.float64)
    importance_sum = np.zeros(x_all.shape[1], dtype=np.float64)
    shap_sum = np.zeros(x_all.shape[1], dtype=np.float64)
    permutation_sum = np.zeros(x_all.shape[1], dtype=np.float64)
    fold_presence = np.zeros(x_all.shape[1], dtype=np.float64)
    fold_scores: list[float] = []
    model_name = kind
    for fold_n, (train_pool, valid_idx) in enumerate(folds):
        fold_start = time.perf_counter()
        train_idx = _balanced_sample_from_indices(
            y, train_pool, int(max_per_class), rng
        )
        if train_idx.size == 0 or np.unique(y[train_idx].astype(int)).size < 2:
            _log(
                f"{kind} CV fold {fold_n + 1}/{len(folds)} skipped: "
                f"train_rows={train_idx.size} valid_rows={valid_idx.size}"
            )
            continue
        try:
            model, importance, model_name = _train_model(
                kind,
                x_all[train_idx],
                y[train_idx],
                seed=int(config.random_state) + fold_n,
                eps=float(config.eps),
            )
        except Exception as exc:
            _log(
                f"{kind} CV fold {fold_n + 1}/{len(folds)} model failed after "
                f"{_elapsed(fold_start)}: {exc}"
            )
            continue
        pred = _predict_proba_1d(model, x_all[valid_idx])
        oof_sum[valid_idx] += pred
        oof_count[valid_idx] += 1.0
        fold_scores.append(_auc_lift(pred, y[valid_idx]))
        if importance.size == x_all.shape[1]:
            importance_sum += np.maximum(importance, 0.0)
            fold_presence += (np.asarray(importance) > 0.0).astype(np.float64)
        ranked = np.argsort(-np.maximum(importance, 0.0))[
            : int(config.max_permutation_features)
        ]
        permutation_sum += _permutation_importance(
            model,
            x_all[valid_idx],
            y[valid_idx],
            ranked,
            repeats=int(config.permutation_repeats),
            max_rows=int(config.max_permutation_rows),
            rng=rng,
        )
        if kind == "lgbm" and str(model_name).startswith("lightgbm"):
            shap_sum += _lightgbm_contribution_importance(
                model,
                x_all[valid_idx],
                max_rows=int(config.max_shap_rows),
                rng=rng,
            )
        _log(
            f"{kind} CV fold {fold_n + 1}/{len(folds)} done: "
            f"train_rows={train_idx.size} valid_rows={valid_idx.size} "
            f"auc_lift={fold_scores[-1]:.4f} elapsed={_elapsed(fold_start)}"
        )
    if not fold_scores:
        _log(f"{kind} CV ranking failed: no valid CV models after {_elapsed(stage_start)}")
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {
                "enabled": False,
                "reason": "no_valid_cv_models",
                "folds_attempted": int(len(folds)),
            },
        )
    final_idx = _balanced_sample_from_indices(y, fit_idx, int(max_per_class), rng)
    if final_idx.size == 0 or np.unique(y[final_idx].astype(int)).size < 2:
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": "empty_final_fit"},
        )
    try:
        final_start = time.perf_counter()
        _log(f"{kind} final fit start: rows={final_idx.size} features={x_all.shape[1]}")
        final_model, final_importance, model_name = _train_model(
            kind,
            x_all[final_idx],
            y[final_idx],
            seed=int(config.random_state),
            eps=float(config.eps),
        )
        scores = _predict_proba_1d(final_model, x_all)
        _log(f"{kind} final fit done: model={model_name} elapsed={_elapsed(final_start)}")
    except Exception as exc:
        _log(f"{kind} final fit failed after {_elapsed(stage_start)}: {exc}")
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": f"model_failed: {exc}"},
        )
    oof_mask = oof_count > 0
    scores[oof_mask] = (
        oof_sum[oof_mask] / np.maximum(oof_count[oof_mask], 1.0)
    ).astype(np.float32)
    n_folds = max(len(fold_scores), 1)
    importance_score = _rank01(
        dict(zip(feature_names, importance_sum + np.maximum(final_importance, 0.0)))
    )
    permutation_score = _rank01(dict(zip(feature_names, permutation_sum)))
    shap_score = _rank01(dict(zip(feature_names, shap_sum)))
    presence_score = {
        str(name): float(fold_presence[i] / n_folds)
        for i, name in enumerate(feature_names)
    }
    combined = {
        str(name): (
            0.35 * importance_score.get(str(name), 0.0)
            + 0.30 * permutation_score.get(str(name), 0.0)
            + 0.20 * shap_score.get(str(name), 0.0)
            + 0.15 * presence_score.get(str(name), 0.0)
        )
        for name in feature_names
    }
    diagnostics = {
        "enabled": True,
        "model": model_name,
        "sample_rows": int(len(final_idx)),
        "oof_rows": int(np.sum(oof_mask)),
        "fold_count": int(len(fold_scores)),
        "fold_auc_lift_mean": float(np.nanmean(fold_scores)) if fold_scores else 0.0,
        "fold_auc_lift_std": float(np.nanstd(fold_scores)) if fold_scores else 0.0,
        "importance": importance_score,
        "permutation": permutation_score,
        "shap": shap_score,
        "fold_presence": presence_score,
        "compute_limits": {
            "max_permutation_features": int(config.max_permutation_features),
            "max_permutation_rows": int(config.max_permutation_rows),
            "permutation_repeats": int(config.permutation_repeats),
            "max_shap_rows": int(config.max_shap_rows),
        },
    }
    _log(
        f"{kind} CV ranking complete: model={model_name} valid_folds={len(fold_scores)} "
        f"oof_rows={int(np.sum(oof_mask))} mean_auc_lift={diagnostics['fold_auc_lift_mean']:.4f} "
        f"elapsed={_elapsed(stage_start)}"
    )
    return np.asarray(scores, dtype=np.float32), combined, diagnostics


def _one_se_select_features(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    rank_scores: Mapping[str, float],
    y: np.ndarray,
    fit_mask: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    max_per_class: int,
    config: RegimeFeatureEngineeringConfig,
) -> tuple[list[str], dict[str, Any]]:
    stage_start = time.perf_counter()
    ranked = [
        str(name)
        for name, _score in sorted(
            rank_scores.items(), key=lambda item: item[1], reverse=True
        )
        if str(name) in set(map(str, feature_names))
    ]
    if not ranked:
        ranked = list(map(str, feature_names))
    col_to_idx = {str(col): i for i, col in enumerate(feature_names)}
    sizes = sorted(
        {
            int(s)
            for s in config.one_se_feature_sizes
            if int(s) > 0 and int(s) <= len(ranked)
        }
    )
    if not sizes or sizes[-1] < min(len(ranked), int(config.max_model_features)):
        sizes.append(min(len(ranked), int(config.max_model_features)))
    rng = np.random.default_rng(
        int(config.random_state) + (503 if kind == "lgbm" else 907)
    )
    fit_idx = np.flatnonzero(np.asarray(fit_mask, dtype=bool))
    folds = _repeated_grouped_fold_indices(
        y,
        fit_idx,
        groups,
        n_splits=int(config.grouped_cv_folds),
        repeats=int(config.grouped_cv_repeats),
        rng=rng,
    )
    if not folds:
        _log(
            f"{kind} one-SE skipped: no grouped CV folds; "
            f"features={len(ranked)} elapsed={_elapsed(stage_start)}"
        )
        return ranked[: min(len(ranked), int(config.max_model_features))], {
            "enabled": False,
            "reason": "no_valid_grouped_cv_folds",
        }
    results: list[dict[str, float]] = []
    x_all = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    _log(
        f"{kind} one-SE start: candidate_features={len(ranked)} sizes={sizes} "
        f"folds={len(folds)} max_per_class={int(max_per_class)}"
    )
    for size in sizes:
        size_start = time.perf_counter()
        selected = ranked[:size]
        idx_cols = np.asarray(
            [col_to_idx[name] for name in selected if name in col_to_idx],
            dtype=np.int64,
        )
        if idx_cols.size == 0:
            continue
        scores: list[float] = []
        for fold_n, (train_pool, valid_idx) in enumerate(folds):
            train_idx = _balanced_sample_from_indices(
                y, train_pool, int(max_per_class), rng
            )
            if train_idx.size == 0 or np.unique(y[train_idx].astype(int)).size < 2:
                continue
            try:
                model, _importance, _model_name = _train_model(
                    kind,
                    x_all[train_idx][:, idx_cols],
                    y[train_idx],
                    seed=int(config.random_state) + fold_n + size,
                    eps=float(config.eps),
                )
                pred = _predict_proba_1d(model, x_all[valid_idx][:, idx_cols])
                scores.append(_auc_lift(pred, y[valid_idx]))
            except Exception:
                continue
        mean = float(np.nanmean(scores)) if scores else 0.0
        std = float(np.nanstd(scores)) if scores else 0.0
        se = std / math.sqrt(max(len(scores), 1))
        results.append({"size": float(size), "mean": mean, "std": std, "se": se})
        _log(
            f"{kind} one-SE size={size} done: valid_folds={len(scores)} "
            f"mean_auc_lift={mean:.4f} se={se:.4f} elapsed={_elapsed(size_start)}"
        )
    if not results:
        _log(f"{kind} one-SE failed: no CV results after {_elapsed(stage_start)}")
        return ranked[: min(len(ranked), int(config.max_model_features))], {
            "enabled": False,
            "reason": "no_cv_results",
        }
    best = max(results, key=lambda row: row["mean"])
    threshold = best["mean"] - best["se"]
    eligible = [row for row in results if row["mean"] >= threshold]
    chosen = min(eligible, key=lambda row: row["size"]) if eligible else best
    size = int(chosen["size"])
    _log(
        f"{kind} one-SE complete: best_size={int(best['size'])} "
        f"chosen_size={size} best_mean={float(best['mean']):.4f} "
        f"threshold={float(threshold):.4f} elapsed={_elapsed(stage_start)}"
    )
    return ranked[:size], {
        "enabled": True,
        "kind": kind,
        "results": results,
        "best_size": int(best["size"]),
        "best_mean": float(best["mean"]),
        "best_se": float(best["se"]),
        "chosen_size": size,
        "threshold": float(threshold),
    }


def _univariate_prefilter(
    matrix: np.ndarray,
    columns: Sequence[str],
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
) -> pd.DataFrame:
    stage_start = time.perf_counter()
    rng = np.random.default_rng(int(config.random_state))
    rows: list[dict[str, Any]] = []
    folds = max(int(config.univariate_folds), 1)
    y_arr = np.asarray(y)
    _log(
        f"univariate prefilter start: rows={matrix.shape[0]} features={matrix.shape[1]} "
        f"folds={folds} subsample_per_class={int(config.univariate_subsample_per_class)}"
    )
    fold_samples = [
        (
            _balanced_sample_indices(
                y_arr, int(config.univariate_subsample_per_class), rng
            ),
            _balanced_sample_indices(
                y_arr, int(config.univariate_subsample_per_class), rng
            ),
            _balanced_sample_indices(
                y_arr, int(config.univariate_subsample_per_class), rng
            ),
        )
        for _fold in range(folds)
    ]
    for j, col in enumerate(columns):
        auc_vals: list[float] = []
        ks_vals: list[float] = []
        shift_vals: list[float] = []
        pass_count = 0
        signs: list[float] = []
        for idx_auc, idx_ks, idx_shift in fold_samples:
            auc = _auc_lift(matrix[idx_auc, j], y_arr[idx_auc])
            ks = _ks_stat(matrix[idx_ks, j], y_arr[idx_ks])
            shift = _median_shift(matrix[idx_shift, j], y_arr[idx_shift])
            auc_vals.append(auc)
            ks_vals.append(ks)
            shift_vals.append(shift)
            signs.append(float(np.sign(shift)) if abs(shift) > config.eps else 0.0)
            if auc >= 0.06 or ks >= 0.08 or abs(shift) >= 0.25:
                pass_count += 1
        auc_arr = np.asarray(auc_vals, dtype=np.float64)
        ks_arr = np.asarray(ks_vals, dtype=np.float64)
        shift_arr = np.asarray(shift_vals, dtype=np.float64)
        mean_shift = float(np.nanmean(shift_arr)) if shift_arr.size else 0.0
        dominant_sign = (
            float(np.sign(mean_shift)) if abs(mean_shift) > config.eps else 0.0
        )
        sign_consistency = (
            float(np.mean([sign == dominant_sign for sign in signs if sign != 0.0]))
            if any(sign != 0.0 for sign in signs)
            else 0.0
        )
        auc_mean = float(np.nanmean(auc_arr)) if auc_arr.size else 0.0
        auc_std = float(np.nanstd(auc_arr)) if auc_arr.size else 0.0
        ks_mean = float(np.nanmean(ks_arr)) if ks_arr.size else 0.0
        ks_std = float(np.nanstd(ks_arr)) if ks_arr.size else 0.0
        shift_std = float(np.nanstd(shift_arr)) if shift_arr.size else 0.0
        fold_pass_rate = float(pass_count / max(folds, 1))
        auc_lcb = auc_mean - 0.5 * auc_std
        ks_lcb = ks_mean - 0.5 * ks_std
        shift_lcb = abs(mean_shift) - 0.5 * shift_std
        drop = (
            (sign_consistency < 0.60 and auc_mean < 0.20)
            or fold_pass_rate < 0.50
            or (auc_mean < 0.04 and ks_mean < 0.06 and abs(mean_shift) < 0.20)
            or auc_lcb < 0.02
            or ks_lcb < 0.03
            or shift_lcb < 0.10
        )
        auc_component = float(np.clip(auc_mean / 0.30, 0.0, 1.0))
        ks_component = float(np.clip(ks_mean / 0.35, 0.0, 1.0))
        shift_component = float(np.clip(abs(mean_shift) / 1.50, 0.0, 1.0))
        stability_component = 0.50 * sign_consistency + 0.50 * fold_pass_rate
        noise_penalty = (
            0.25 * auc_std / max(auc_mean, 1e-6)
            + 0.25 * ks_std / max(ks_mean, 1e-6)
            + 0.25 * shift_std / max(abs(mean_shift), 1e-6)
        )
        noise_penalty = float(np.clip(noise_penalty, 0.0, 1.0))
        score = (
            stability_component
            * (0.35 * auc_component + 0.35 * ks_component + 0.30 * shift_component)
            * (1.0 - 0.50 * noise_penalty)
        )
        rows.append(
            {
                "feature": str(col),
                "auc_lift_mean": auc_mean,
                "auc_lift_std": auc_std,
                "ks_mean": ks_mean,
                "ks_std": ks_std,
                "median_shift_mean": mean_shift,
                "median_shift_std": shift_std,
                "sign_consistency": sign_consistency,
                "fold_pass_rate": fold_pass_rate,
                "auc_lift_lcb": auc_lcb,
                "ks_lcb": ks_lcb,
                "shift_lcb": shift_lcb,
                "univariate_score": float(score),
                "selected_univariate": not bool(drop),
            }
        )
        if (j + 1) % 100 == 0 or (j + 1) == len(columns):
            _log(
                f"univariate prefilter progress: features={j + 1}/{len(columns)} "
                f"elapsed={_elapsed(stage_start)}"
            )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("univariate_score", ascending=False).reset_index(
            drop=True
        )
    survivors = int(report["selected_univariate"].sum()) if not report.empty else 0
    _log(
        f"univariate prefilter complete: survivors={survivors}/{len(columns)} "
        f"elapsed={_elapsed(stage_start)}"
    )
    return report


def _corrcoef_subsample(
    matrix: np.ndarray, y: np.ndarray, config: RegimeFeatureEngineeringConfig
) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    rng = np.random.default_rng(int(config.random_state) + 17)
    idx = _balanced_sample_indices(y, int(config.corr_subsample_per_class), rng)
    sub = matrix[idx].astype(np.float64, copy=True)
    sub = np.where(np.isfinite(sub), sub, 0.0)
    if sub.shape[0] < 3:
        return np.eye(sub.shape[1], dtype=np.float32)
    corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def _correlation_cluster_select(
    matrix: np.ndarray,
    columns: Sequence[str],
    report: pd.DataFrame,
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
) -> tuple[list[str], list[str], dict[str, Any]]:
    survivors = (
        report.loc[report["selected_univariate"], "feature"].astype(str).tolist()
    )
    if not survivors:
        return [], list(columns), {"selected": 0, "removed": len(columns)}
    col_to_idx = {col: i for i, col in enumerate(columns)}
    survivor_idx = [col_to_idx[col] for col in survivors if col in col_to_idx]
    survivor_cols = [col for col in survivors if col in col_to_idx]
    sub = (
        matrix[:, survivor_idx]
        if survivor_idx
        else np.zeros((len(y), 0), dtype=np.float32)
    )
    corr = np.abs(_corrcoef_subsample(sub, y, config))
    score_map = dict(
        zip(report["feature"].astype(str), report["univariate_score"].astype(float))
    )
    selected: list[str] = []
    selected_local: list[int] = []
    removed: list[str] = []
    for local_i, col in enumerate(survivor_cols):
        if len(selected) >= int(config.max_cluster_features):
            removed.append(col)
            continue
        if not selected_local:
            selected.append(col)
            selected_local.append(local_i)
            continue
        max_corr = (
            float(np.max(corr[local_i, selected_local])) if selected_local else 0.0
        )
        if max_corr <= float(config.corr_cluster_threshold):
            selected.append(col)
            selected_local.append(local_i)
        else:
            weaker = min(selected, key=lambda f: score_map.get(f, 0.0))
            if score_map.get(col, 0.0) > score_map.get(weaker, 0.0):
                removed.append(weaker)
                replace_pos = selected.index(weaker)
                selected[replace_pos] = col
                selected_local[replace_pos] = local_i
            else:
                removed.append(col)
    removed.extend(
        [col for col in columns if col not in selected and col not in removed]
    )
    return (
        selected,
        _stable_unique(removed),
        {"selected": len(selected), "removed": len(removed)},
    )


def _low_correlation_backfill(
    matrix: np.ndarray,
    columns: Sequence[str],
    selected: Sequence[str],
    removed: Sequence[str],
    report: pd.DataFrame,
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
) -> list[str]:
    if not selected or not removed:
        return []
    col_to_idx = {col: i for i, col in enumerate(columns)}
    selected_idx = [col_to_idx[col] for col in selected if col in col_to_idx]
    removed_idx = [col_to_idx[col] for col in removed if col in col_to_idx]
    if not selected_idx or not removed_idx:
        return []
    rng = np.random.default_rng(int(config.random_state) + 31)
    idx = _balanced_sample_indices(y, int(config.relief_subsample_per_class), rng)
    sel = matrix[idx][:, selected_idx].astype(np.float64, copy=True)
    rem = matrix[idx][:, removed_idx].astype(np.float64, copy=True)
    sel = np.where(np.isfinite(sel), sel, 0.0)
    rem = np.where(np.isfinite(rem), rem, 0.0)
    sel = sel - np.nanmean(sel, axis=0, keepdims=True)
    rem = rem - np.nanmean(rem, axis=0, keepdims=True)
    denom = np.maximum(
        np.sqrt(np.sum(rem * rem, axis=0, keepdims=True)).T
        @ np.sqrt(np.sum(sel * sel, axis=0, keepdims=True)),
        float(config.eps),
    )
    corr = np.abs((rem.T @ sel) / denom)
    min_corr = np.min(corr, axis=1) if corr.size else np.ones(len(removed_idx))
    score_map = dict(
        zip(report["feature"].astype(str), report["univariate_score"].astype(float))
    )
    candidates = [
        (removed[i], float(min_corr[i]), float(score_map.get(removed[i], 0.0)))
        for i in range(len(removed_idx))
    ]
    candidates.sort(key=lambda item: (item[1], -item[2]))
    return [name for name, _corr, _score in candidates[: int(config.max_backfill_pool)]]


def _pair_term_values(x: np.ndarray, y: np.ndarray, term: str) -> np.ndarray:
    if term == "product":
        return x * y
    if term == "abs_diff":
        return np.abs(x - y)
    if term == "co_extreme":
        return ((np.abs(x) > 1.0) & (np.abs(y) > 1.0)).astype(np.float32)
    if term == "same_tail":
        return (((x > 1.0) & (y > 1.0)) | ((x < -1.0) & (y < -1.0))).astype(np.float32)
    if term == "opposite_tail":
        return (((x > 1.0) & (y < -1.0)) | ((x < -1.0) & (y > 1.0))).astype(np.float32)
    return np.zeros_like(x, dtype=np.float32)


def _relationship_shift(
    x: np.ndarray, z: np.ndarray, y: np.ndarray, eps: float
) -> float:
    labels = np.asarray(y, dtype=bool)
    cur = labels
    hist = ~labels
    if int(np.sum(cur)) < 3 or int(np.sum(hist)) < 3:
        return 0.0

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(mask)) < 3:
            return 0.0
        val = float(np.corrcoef(a[mask], b[mask])[0, 1])
        return val if np.isfinite(val) else 0.0

    c_cur = np.arctanh(np.clip(_corr(x[cur], z[cur]), -0.999, 0.999))
    c_hist = np.arctanh(np.clip(_corr(x[hist], z[hist]), -0.999, 0.999))
    return float(min(abs(c_cur - c_hist), 3.0) / 3.0)


def _generate_pair_features(
    matrix: np.ndarray,
    columns: Sequence[str],
    parents: Sequence[str],
    report: pd.DataFrame,
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
) -> list[PairFeatureSpec]:
    stage_start = time.perf_counter()
    col_to_idx = {col: i for i, col in enumerate(columns)}
    score_map = dict(
        zip(report["feature"].astype(str), report["univariate_score"].astype(float))
    )
    parent_cols = [col for col in parents if col in col_to_idx][
        : int(config.max_pair_parent_features)
    ]
    total_candidate_pairs = min(
        int(config.max_pair_candidates),
        int(len(parent_cols) * max(len(parent_cols) - 1, 0) / 2),
    )
    candidates: list[PairFeatureSpec] = []
    pair_iter = itertools.combinations(parent_cols, 2)
    rng = np.random.default_rng(
        int(config.random_state) + int(config.shadow_random_state_offset)
    )
    folds = max(int(config.univariate_folds), 1)
    _log(
        f"pair feature search start: parents={len(parent_cols)} "
        f"candidate_pairs={total_candidate_pairs} folds={folds}"
    )
    y_arr = np.asarray(y)
    fold_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _fold in range(folds):
        idx = _balanced_sample_indices(
            y_arr, int(config.relief_subsample_per_class), rng
        )
        if idx.size == 0 or np.unique(y_arr[idx].astype(int)).size < 2:
            continue
        shadow = y_arr[idx].copy()
        rng.shuffle(shadow)
        fold_samples.append((idx, y_arr[idx], shadow))
    for pair_n, (left, right) in enumerate(pair_iter):
        if pair_n >= int(config.max_pair_candidates):
            break
        i = col_to_idx[left]
        j = col_to_idx[right]
        x = matrix[:, i].astype(np.float32, copy=False)
        z = matrix[:, j].astype(np.float32, copy=False)
        parent_quality = math.sqrt(
            max(score_map.get(left, 0.0), 0.0) * max(score_map.get(right, 0.0), 0.0)
        )
        relationship = _relationship_shift(x, z, y, float(config.eps))
        pair_fold_cache: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        ] = []
        for idx, y_idx, shadow in fold_samples:
            x_idx = x[idx]
            z_idx = z[idx]
            parent_score = max(_auc_lift(x_idx, y_idx), _auc_lift(z_idx, y_idx))
            pair_fold_cache.append((x_idx, z_idx, y_idx, shadow, parent_score))
        if not pair_fold_cache:
            continue
        term_scores: list[dict[str, float | str]] = []
        for term in ("product", "abs_diff", "co_extreme", "same_tail", "opposite_tail"):
            fold_scores: list[float] = []
            fold_incremental: list[float] = []
            fold_passes = 0
            signs: list[float] = []
            shadow_scores: list[float] = []
            for x_idx, z_idx, y_idx, shadow, parent_score in pair_fold_cache:
                vals_idx = _pair_term_values(x_idx, z_idx, term)
                score = _auc_lift(vals_idx, y_idx)
                incremental = float(score - parent_score)
                fold_scores.append(float(score))
                fold_incremental.append(incremental)
                signs.append(float(np.sign(_median_shift(vals_idx, y_idx))))
                if score >= 0.06 or incremental > 0.0:
                    fold_passes += 1
                shadow_scores.append(_auc_lift(vals_idx, shadow))
            if not fold_scores:
                continue
            mean_score = float(np.nanmean(fold_scores))
            mean_incremental = (
                float(np.nanmean(fold_incremental)) if fold_incremental else 0.0
            )
            fold_pass_rate = float(fold_passes / max(len(fold_scores), 1))
            dominant_sign = (
                float(np.sign(np.nanmean([s for s in signs if s != 0.0])))
                if any(s != 0.0 for s in signs)
                else 0.0
            )
            sign_consistency = (
                float(np.mean([s == dominant_sign for s in signs if s != 0.0]))
                if any(s != 0.0 for s in signs)
                else 0.0
            )
            shadow_p95 = (
                float(np.nanpercentile(shadow_scores, 95.0)) if shadow_scores else 0.0
            )
            stability = 0.5 * sign_consistency + 0.5 * fold_pass_rate
            shadow_margin = float(mean_score - shadow_p95)
            raw_score = stability * (
                0.25 * relationship
                + 0.25 * mean_score
                + 0.20 * max(mean_incremental, 0.0)
            ) + 0.10 * max(shadow_margin, 0.0)
            term_scores.append(
                {
                    "term": term,
                    "score": float(raw_score),
                    "mean_score": mean_score,
                    "incremental": mean_incremental,
                    "shadow_margin": shadow_margin,
                    "stability": stability,
                    "sign_consistency": sign_consistency,
                    "fold_pass_rate": fold_pass_rate,
                }
            )
        if not term_scores:
            continue
        best = max(term_scores, key=lambda item: float(item["score"]))
        candidates.append(
            PairFeatureSpec(
                name=f"duo__{left}__{best['term']}__{right}",
                left=left,
                right=right,
                term=str(best["term"]),
                score=float(best["score"]),
                parent_quality=float(parent_quality),
                incremental_lift=float(best["incremental"]),
                shadow_margin=float(best["shadow_margin"]),
                stability=float(best["stability"]),
                sign_consistency=float(best["sign_consistency"]),
                fold_pass_rate=float(best["fold_pass_rate"]),
            )
        )
        if (pair_n + 1) % 250 == 0 or (pair_n + 1) == total_candidate_pairs:
            _log(
                f"pair feature search progress: evaluated={pair_n + 1}/{total_candidate_pairs} "
                f"candidate_terms={len(candidates)} elapsed={_elapsed(stage_start)}"
            )
    if not candidates:
        _log(f"pair feature search complete: no candidates elapsed={_elapsed(stage_start)}")
        return []
    kept: list[PairFeatureSpec] = []
    scores = np.asarray([p.score for p in candidates], dtype=np.float64)
    top_score = float(np.nanmax(scores)) if scores.size else 0.0
    for pair in sorted(candidates, key=lambda p: p.score, reverse=True):
        engineered_top_ranked = pair.score >= top_score - float(config.eps)
        if (
            pair.sign_consistency >= 0.70
            and pair.fold_pass_rate >= 0.60
            and pair.shadow_margin > 0.0
            and (pair.incremental_lift > 0.0 or engineered_top_ranked)
        ):
            kept.append(pair)
        if len(kept) >= int(config.max_pairs_kept):
            break
    _log(
        f"pair feature search complete: candidates={len(candidates)} kept={len(kept)} "
        f"elapsed={_elapsed(stage_start)}"
    )
    return kept


def _matrix_for_feature_names(
    base_matrix: np.ndarray,
    base_columns: Sequence[str],
    feature_names: Sequence[str],
    pair_specs: Sequence[PairFeatureSpec],
    extra_matrix: np.ndarray | None = None,
    extra_columns: Sequence[str] | None = None,
) -> np.ndarray:
    col_to_idx = {col: i for i, col in enumerate(base_columns)}
    extra_col_to_idx = {
        col: i
        for i, col in enumerate(extra_columns or [])
        if extra_matrix is not None and extra_matrix.ndim == 2
    }
    pair_by_name = {pair.name: pair for pair in pair_specs}
    arrays: list[np.ndarray] = []
    for name in feature_names:
        if name in col_to_idx:
            arrays.append(
                base_matrix[:, col_to_idx[name]].astype(np.float32, copy=False)
            )
            continue
        if extra_matrix is not None and name in extra_col_to_idx:
            arrays.append(
                extra_matrix[:, extra_col_to_idx[name]].astype(np.float32, copy=False)
            )
            continue
        pair = pair_by_name.get(name)
        if pair and pair.left in col_to_idx and pair.right in col_to_idx:
            x = base_matrix[:, col_to_idx[pair.left]].astype(np.float32, copy=False)
            y = base_matrix[:, col_to_idx[pair.right]].astype(np.float32, copy=False)
            arrays.append(
                _pair_term_values(x, y, pair.term).astype(np.float32, copy=False)
            )
    if not arrays:
        return np.zeros((base_matrix.shape[0], 0), dtype=np.float32)
    return np.column_stack(arrays).astype(np.float32, copy=False)


def _fit_lgbm_like_scores(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
    fit_mask: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not bool(config.lgbm_enabled):
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": "empty_or_disabled"},
        )
    fit = (
        np.asarray(fit_mask, dtype=bool)
        if fit_mask is not None
        else np.ones(len(y), dtype=bool)
    )
    if fit.size != len(y) or not bool(fit.any()):
        fit = np.ones(len(y), dtype=bool)
    group_arr = (
        np.asarray(groups) if groups is not None else np.arange(len(y), dtype=np.int64)
    )
    _initial_scores, rank_scores, rank_diag = _cv_rank_and_scores(
        matrix,
        feature_names,
        y,
        fit,
        group_arr,
        kind="lgbm",
        max_per_class=int(config.lgbm_max_samples_per_class),
        config=config,
    )
    if not rank_diag.get("enabled", False):
        return _initial_scores, rank_scores, rank_diag
    selected, one_se_diag = _one_se_select_features(
        matrix,
        feature_names,
        rank_scores,
        y,
        fit,
        group_arr,
        kind="lgbm",
        max_per_class=int(config.lgbm_max_samples_per_class),
        config=config,
    )
    col_to_idx = {str(col): i for i, col in enumerate(feature_names)}
    idx = [col_to_idx[name] for name in selected if name in col_to_idx]
    if not idx:
        return _initial_scores, rank_scores, rank_diag
    scores, selected_scores, selected_diag = _cv_rank_and_scores(
        matrix[:, idx],
        selected,
        y,
        fit,
        group_arr,
        kind="lgbm",
        max_per_class=int(config.lgbm_max_samples_per_class),
        config=config,
    )
    diagnostics = dict(selected_diag)
    diagnostics["initial_rank"] = rank_diag
    diagnostics["one_se"] = one_se_diag
    diagnostics["selected_features"] = selected
    diagnostics["rfe_method"] = "repeated_grouped_cv_one_se"
    return scores, selected_scores, diagnostics


def _fit_elasticnet_scores(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    y: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
    fit_mask: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not bool(config.elasticnet_enabled):
        return (
            np.full(len(y), 0.5, dtype=np.float32),
            {},
            {"enabled": False, "reason": "empty_or_disabled"},
        )
    fit = (
        np.asarray(fit_mask, dtype=bool)
        if fit_mask is not None
        else np.ones(len(y), dtype=bool)
    )
    if fit.size != len(y) or not bool(fit.any()):
        fit = np.ones(len(y), dtype=bool)
    group_arr = (
        np.asarray(groups) if groups is not None else np.arange(len(y), dtype=np.int64)
    )
    _initial_scores, rank_scores, rank_diag = _cv_rank_and_scores(
        matrix,
        feature_names,
        y,
        fit,
        group_arr,
        kind="elasticnet",
        max_per_class=int(config.elasticnet_max_samples_per_class),
        config=config,
    )
    if not rank_diag.get("enabled", False):
        return _initial_scores, rank_scores, rank_diag
    selected, one_se_diag = _one_se_select_features(
        matrix,
        feature_names,
        rank_scores,
        y,
        fit,
        group_arr,
        kind="elasticnet",
        max_per_class=int(config.elasticnet_max_samples_per_class),
        config=config,
    )
    col_to_idx = {str(col): i for i, col in enumerate(feature_names)}
    idx = [col_to_idx[name] for name in selected if name in col_to_idx]
    if not idx:
        return _initial_scores, rank_scores, rank_diag
    scores, selected_scores, selected_diag = _cv_rank_and_scores(
        matrix[:, idx],
        selected,
        y,
        fit,
        group_arr,
        kind="elasticnet",
        max_per_class=int(config.elasticnet_max_samples_per_class),
        config=config,
    )
    diagnostics = dict(selected_diag)
    diagnostics["initial_rank"] = rank_diag
    diagnostics["one_se"] = one_se_diag
    diagnostics["selected_features"] = selected
    diagnostics["rfe_method"] = "repeated_grouped_cv_one_se"
    return scores, selected_scores, diagnostics


def _robust_standardize_matrix(
    matrix: np.ndarray, fit_mask: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(matrix, dtype=np.float64)
    fit = np.asarray(fit_mask, dtype=bool)
    if fit.size != arr.shape[0] or not bool(fit.any()):
        fit = np.ones(arr.shape[0], dtype=bool)
    fit_vals = arr[fit]
    fit_vals = np.where(np.isfinite(fit_vals), fit_vals, np.nan)
    center = np.nanmedian(fit_vals, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    mad = np.nanmedian(np.abs(fit_vals - center.reshape(1, -1)), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(fit_vals, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)
    z = (
        np.where(np.isfinite(arr), arr, center.reshape(1, -1)) - center.reshape(1, -1)
    ) / np.maximum(scale.reshape(1, -1), eps)
    return (
        np.clip(z, -8.0, 8.0).astype(np.float32),
        center.astype(np.float32),
        scale.astype(np.float32),
    )


def _pca_ae_embedding(
    matrix: np.ndarray,
    active_mask: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros((len(matrix), 0), dtype=np.float32), {
            "enabled": False,
            "reason": "empty_matrix",
        }
    rng = np.random.default_rng(int(config.random_state) + 1237)
    z, _center, _scale = _robust_standardize_matrix(
        matrix, active_mask, float(config.eps)
    )
    active_idx = np.flatnonzero(active_mask)
    if active_idx.size == 0:
        active_idx = np.arange(len(z), dtype=np.int64)
    train_idx = active_idx
    if len(train_idx) > int(config.ae_max_samples) > 0:
        train_idx = rng.choice(
            train_idx, size=int(config.ae_max_samples), replace=False
        )
    pca_dim = max(1, min(int(config.pca_components), z.shape[1], len(train_idx)))
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_dim, random_state=int(config.random_state))
        pca.fit(z[train_idx])
        pca_emb = pca.transform(z).astype(np.float32)
        pca_reason = "sklearn_pca"
    except Exception:
        train = z[train_idx].astype(np.float64)
        u, s, vt = np.linalg.svd(
            train - np.mean(train, axis=0, keepdims=True), full_matrices=False
        )
        comp = vt[:pca_dim].T
        pca_emb = ((z - np.mean(train, axis=0, keepdims=True)) @ comp).astype(
            np.float32
        )
        pca_reason = "svd_fallback"
    if pca_emb.shape[1] < int(config.pca_components):
        pad = np.zeros(
            (len(z), int(config.pca_components) - pca_emb.shape[1]), dtype=np.float32
        )
        pca_emb = np.column_stack([pca_emb, pad]).astype(np.float32)
    ae_dim = max(1, min(int(config.ae_components), z.shape[1], 16))
    ae_emb = np.zeros((len(z), int(config.ae_components)), dtype=np.float32)
    ae_reason = "unavailable"
    try:
        from sklearn.neural_network import MLPRegressor

        hidden = (
            max(16, min(64, z.shape[1] * 2)),
            ae_dim,
            max(16, min(64, z.shape[1] * 2)),
        )
        ae = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=int(config.ae_max_iter),
            early_stopping=True,
            random_state=int(config.random_state),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            ae.fit(z[train_idx], z[train_idx])
        h = z.astype(np.float32)
        for layer_i, (coef, intercept) in enumerate(zip(ae.coefs_, ae.intercepts_)):
            h = h @ np.asarray(coef, dtype=np.float32) + np.asarray(
                intercept, dtype=np.float32
            ).reshape(1, -1)
            if layer_i < len(ae.coefs_) - 1:
                h = np.maximum(h, 0.0).astype(np.float32)
            if layer_i == 1:
                ae_raw = h.astype(np.float32)
                if ae_raw.shape[1] < int(config.ae_components):
                    pad = np.zeros(
                        (len(z), int(config.ae_components) - ae_raw.shape[1]),
                        dtype=np.float32,
                    )
                    ae_raw = np.column_stack([ae_raw, pad]).astype(np.float32)
                ae_emb = ae_raw[:, : int(config.ae_components)]
                ae_reason = "mlp_autoencoder"
                break
    except Exception as exc:
        ae_reason = f"failed: {exc}"
    emb = np.column_stack(
        [
            pca_emb[:, : int(config.pca_components)],
            ae_emb[:, : int(config.ae_components)],
        ]
    ).astype(np.float32)
    return emb, {
        "enabled": True,
        "pca": pca_reason,
        "autoencoder": ae_reason,
        "pca_components": int(config.pca_components),
        "ae_components": int(config.ae_components),
        "train_rows": int(len(train_idx)),
    }


def _window_ids_from_timestamps(
    frame: pd.DataFrame,
    timestamp_col: str,
    historical_mask: np.ndarray,
    window_days: float,
) -> np.ndarray:
    out = np.full(len(frame), -1, dtype=np.int64)
    if timestamp_col not in frame.columns or not bool(historical_mask.any()):
        out[historical_mask] = 0
        return out
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    hist_ts = ts[historical_mask & ts.notna().to_numpy(dtype=bool)]
    if hist_ts.empty:
        out[historical_mask] = 0
        return out
    anchor = hist_ts.min()
    seconds = (ts - anchor).dt.total_seconds().to_numpy(dtype=np.float64)
    win_sec = max(float(window_days) * 24.0 * 3600.0, 1.0)
    valid = historical_mask & np.isfinite(seconds)
    out[valid] = np.floor(np.maximum(seconds[valid], 0.0) / win_sec).astype(np.int64)
    return out


def _ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.sort(np.asarray(a, dtype=np.float64)[np.isfinite(a)])
    bb = np.sort(np.asarray(b, dtype=np.float64)[np.isfinite(b)])
    if aa.size == 0 or bb.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([aa, bb])))
    return float(
        np.max(
            np.abs(
                np.searchsorted(aa, grid, side="right") / aa.size
                - np.searchsorted(bb, grid, side="right") / bb.size
            )
        )
    )


def _psi_two_sample(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    vals = np.concatenate([aa[np.isfinite(aa)], bb[np.isfinite(bb)]])
    if vals.size < 10:
        return 0.0
    edges = np.unique(np.nanpercentile(vals, np.linspace(0.0, 100.0, 11)))
    if edges.size < 3:
        return 0.0
    pa, _ = np.histogram(aa[np.isfinite(aa)], bins=edges)
    pb, _ = np.histogram(bb[np.isfinite(bb)], bins=edges)
    p = pa.astype(np.float64) / max(float(np.sum(pa)), eps)
    q = pb.astype(np.float64) / max(float(np.sum(pb)), eps)
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return float(np.sum((p - q) * np.log(p / q)))


def _effective_rank_from_corr(corr: np.ndarray, eps: float) -> tuple[float, float]:
    if corr.ndim != 2 or corr.shape[0] < 2:
        return 0.0, 0.0
    eig = np.maximum(np.linalg.eigvalsh(corr.astype(np.float64)), 0.0)
    total = float(np.sum(eig))
    if total <= eps:
        return 0.0, 0.0
    p = eig / total
    p = p[p > eps]
    eff = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
    pc1 = float(np.max(eig) / total) if eig.size else 0.0
    return eff, pc1


def _corr_matrix(matrix: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return np.eye(max(arr.shape[1] if arr.ndim == 2 else 1, 1), dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    fill = np.nanmedian(arr, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    arr = np.where(np.isfinite(arr), arr, fill.reshape(1, -1))
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    shrink = 0.10
    corr = (1.0 - shrink) * corr + shrink * np.eye(corr.shape[0])
    return np.clip(corr, -1.0, 1.0)


def _subsample_matrix_rows(matrix: np.ndarray, max_rows: int) -> np.ndarray:
    arr = np.asarray(matrix)
    if int(max_rows) <= 0 or len(arr) <= int(max_rows):
        return arr
    idx = np.linspace(0, len(arr) - 1, int(max_rows)).round().astype(np.int64)
    return arr[idx]


def _subsample_positions(positions: np.ndarray, max_rows: int) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if int(max_rows) <= 0 or pos.size <= int(max_rows):
        return pos
    idx = np.linspace(0, pos.size - 1, int(max_rows)).round().astype(np.int64)
    return pos[idx]


def _mean_nearest_distance(
    a: np.ndarray,
    b: np.ndarray,
    eps: float,
    *,
    max_rows: int = 4000,
    chunk_pairs: int = 2_000_000,
) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    aa = _subsample_matrix_rows(np.asarray(a, dtype=np.float32), int(max_rows))
    bb = _subsample_matrix_rows(np.asarray(b, dtype=np.float32), int(max_rows))
    if aa.size == 0 or bb.size == 0:
        return 0.0
    feature_count = max(int(aa.shape[1]), 1)
    rows_per_chunk = max(1, int(chunk_pairs) // max(len(bb), 1))
    nearest_chunks: list[np.ndarray] = []
    bb_norm = np.sum(bb * bb, axis=1, keepdims=True).T
    for start in range(0, len(aa), rows_per_chunk):
        block = aa[start : start + rows_per_chunk]
        d = np.maximum(
            np.sum(block * block, axis=1, keepdims=True)
            + bb_norm
            - 2.0 * (block @ bb.T),
            0.0,
        )
        nearest_chunks.append(
            np.sqrt(np.min(d, axis=1) / feature_count).astype(np.float32)
        )
    if not nearest_chunks:
        return 0.0
    nearest = np.concatenate(nearest_chunks)
    return float(np.nanmean(nearest)) if nearest.size else 0.0


def _generate_current_relative_drift_features(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    columns: Sequence[str],
    current_mask: np.ndarray,
    historical_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    timestamp_col: str,
    config: RegimeFeatureEngineeringConfig,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    stage_start = time.perf_counter()
    raw_cols = list(columns)[: int(config.max_drift_raw_features)]
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not raw_cols:
        return (
            np.zeros((len(frame), 0), dtype=np.float32),
            [],
            {"enabled": False, "reason": "empty_raw_state"},
        )
    col_idx = np.arange(len(raw_cols), dtype=np.int64)
    x = matrix[:, col_idx].astype(np.float32, copy=False)
    cur = current_mask & active_mask
    hist = historical_mask & active_mask
    if int(np.sum(cur)) < 3 or int(np.sum(hist)) < 3:
        return (
            np.zeros((len(frame), 0), dtype=np.float32),
            [],
            {"enabled": False, "reason": "insufficient_rows"},
        )
    emb, emb_diag = _pca_ae_embedding(x, active_mask, config)
    window_ids = _window_ids_from_timestamps(
        frame, timestamp_col, hist, float(config.drift_window_days)
    )
    valid_window_ids = sorted(set(window_ids[window_ids >= 0]))
    _log(
        f"current-relative drift start: rows={len(frame)} raw_features={len(raw_cols)} "
        f"current_rows={int(np.sum(cur))} historical_rows={int(np.sum(hist))} "
        f"windows={len(valid_window_ids)} embedding={emb_diag}"
    )
    max_window_rows = int(config.drift_window_max_rows)
    cur_pos = _subsample_positions(np.flatnonzero(cur), max_window_rows)
    x_cur = x[cur_pos]
    emb_cur = emb[cur_pos]
    feature_names = [
        "drift_ks_mean",
        "drift_log_psi_mean",
        "drift_median_shift_abs_mean",
        "drift_iqr_log_ratio_abs_mean",
        "drift_tail_share_shift_mean",
        "drift_knn_symmetric_distance",
        "drift_corr_frobenius_distance",
        "drift_eigen_concentration_delta",
        "drift_effective_rank_delta",
        "drift_extreme_share_delta",
        "drift_pair_coextreme_delta",
    ]
    out = np.zeros((len(frame), len(feature_names)), dtype=np.float32)
    current_corr = _corr_matrix(x_cur, float(config.eps))
    current_eff, current_pc1 = _effective_rank_from_corr(
        current_corr, float(config.eps)
    )
    current_extreme = np.nanmean(np.abs(x_cur) > 1.0, axis=0)
    pair_count = min(int(config.max_tail_pair_features), x.shape[1])
    current_pairs: dict[tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(pair_count), 2):
        current_pairs[(i, j)] = float(
            np.nanmean((np.abs(x_cur[:, i]) > 1.0) & (np.abs(x_cur[:, j]) > 1.0))
        )
    rows_by_window: dict[int, np.ndarray] = {}
    raw_window_values: list[np.ndarray] = []
    processed_windows = 0
    for window_pos, window_id in enumerate(valid_window_ids):
        mask = window_ids == int(window_id)
        pos = np.flatnonzero(mask)
        if pos.size < 3:
            continue
        rows_by_window[int(window_id)] = pos
        stat_pos = _subsample_positions(pos, max_window_rows)
        cand = x[stat_pos]
        ks_vals = []
        psi_vals = []
        shift_vals = []
        iqr_vals = []
        tail_vals = []
        for j in range(x.shape[1]):
            cur_col = x_cur[:, j]
            cand_col = cand[:, j]
            ks_vals.append(_ks_two_sample(cur_col, cand_col))
            psi_vals.append(
                math.log1p(
                    max(_psi_two_sample(cur_col, cand_col, float(config.eps)), 0.0)
                )
            )
            cur_med = float(np.nanmedian(cur_col))
            cand_med = float(np.nanmedian(cand_col))
            shift_vals.append(min(abs(cand_med - cur_med), 3.0) / 3.0)
            cur_iqr = float(
                np.nanpercentile(cur_col, 75) - np.nanpercentile(cur_col, 25)
            )
            cand_iqr = float(
                np.nanpercentile(cand_col, 75) - np.nanpercentile(cand_col, 25)
            )
            iqr_vals.append(
                abs(
                    math.log(
                        max(cand_iqr, float(config.eps))
                        / max(cur_iqr, float(config.eps))
                    )
                )
            )
            tail_vals.append(
                abs(
                    float(np.nanmean(np.abs(cand_col) > 1.0))
                    - float(np.nanmean(np.abs(cur_col) > 1.0))
                )
            )
        cand_emb = emb[stat_pos]
        knn = 0.5 * (
            _mean_nearest_distance(
                cand_emb,
                emb_cur,
                float(config.eps),
                max_rows=int(config.drift_knn_max_rows),
                chunk_pairs=int(config.drift_knn_chunk_pairs),
            )
            + _mean_nearest_distance(
                emb_cur,
                cand_emb,
                float(config.eps),
                max_rows=int(config.drift_knn_max_rows),
                chunk_pairs=int(config.drift_knn_chunk_pairs),
            )
        )
        cand_corr = _corr_matrix(cand, float(config.eps))
        corr_dist = float(
            np.linalg.norm(cand_corr - current_corr, ord="fro")
            / math.sqrt(max(cand_corr.size, 1))
        )
        cand_eff, cand_pc1 = _effective_rank_from_corr(cand_corr, float(config.eps))
        cand_extreme = np.nanmean(np.abs(cand) > 1.0, axis=0)
        pair_deltas = []
        for i, j in itertools.combinations(range(pair_count), 2):
            val = float(
                np.nanmean((np.abs(cand[:, i]) > 1.0) & (np.abs(cand[:, j]) > 1.0))
            )
            pair_deltas.append(abs(val - current_pairs.get((i, j), 0.0)))
        vals = np.asarray(
            [
                float(np.nanmean(ks_vals)),
                float(np.nanmean(psi_vals)),
                float(np.nanmean(shift_vals)),
                float(np.nanmean(iqr_vals)),
                float(np.nanmean(tail_vals)),
                knn,
                corr_dist,
                abs(cand_pc1 - current_pc1),
                abs(cand_eff - current_eff),
                float(np.nanmean(np.abs(cand_extreme - current_extreme))),
                float(np.nanmean(pair_deltas)) if pair_deltas else 0.0,
            ],
            dtype=np.float32,
        )
        raw_window_values.append(vals)
        out[pos] = vals.reshape(1, -1)
        processed_windows += 1
        if processed_windows % 5 == 0 or window_pos + 1 == len(valid_window_ids):
            _log(
                f"current-relative drift progress: processed_windows={processed_windows}/"
                f"{len(valid_window_ids)} assigned_rows={int(sum(len(v) for v in rows_by_window.values()))} "
                f"elapsed={_elapsed(stage_start)}"
            )
    if raw_window_values:
        raw = np.vstack(raw_window_values).astype(np.float64)
        scaled = raw.copy()
        # KS, tail shares, median shift are already bounded-ish. Scale PSI/KNN/cov/eigen by historical windows.
        for col in (1, 3, 5, 6, 7, 8, 10):
            vals = raw[:, col]
            scale = (
                float(np.nanmedian(vals[np.isfinite(vals)]))
                if np.isfinite(vals).any()
                else 1.0
            )
            if not np.isfinite(scale) or scale <= float(config.eps):
                q25, q75 = (
                    np.nanpercentile(vals[np.isfinite(vals)], [25, 75])
                    if np.isfinite(vals).any()
                    else (0.0, 1.0)
                )
                scale = float(q75 - q25)
            scale = scale if np.isfinite(scale) and scale > float(config.eps) else 1.0
            scaled[:, col] = vals / scale
        for row_i, (_window_id, pos) in enumerate(rows_by_window.items()):
            out[pos] = np.clip(scaled[row_i], 0.0, 10.0).astype(np.float32)
    diagnostics = {
        "enabled": True,
        "available": True,
        "feature_count": int(len(feature_names)),
        "raw_state_feature_count": int(len(raw_cols)),
        "window_count": int(len(rows_by_window)),
        "families": {
            "univariate_distribution": 5,
            "knn_manifold": 1,
            "covariance_correlation": 3,
            "tail_coextreme": 2,
        },
        "compute_limits": {
            "drift_knn_max_rows": int(config.drift_knn_max_rows),
            "drift_knn_chunk_pairs": int(config.drift_knn_chunk_pairs),
            "max_drift_raw_features": int(config.max_drift_raw_features),
            "drift_window_max_rows": int(config.drift_window_max_rows),
            "max_tail_pair_features": int(config.max_tail_pair_features),
        },
        "embedding": emb_diag,
    }
    _log(
        f"current-relative drift complete: windows={len(rows_by_window)} "
        f"features={len(feature_names)} elapsed={_elapsed(stage_start)}"
    )
    return out.astype(np.float32, copy=False), feature_names, diagnostics


def _validation_auc_lift(
    matrix: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    groups: np.ndarray,
    config: RegimeFeatureEngineeringConfig,
) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return {"mean": 0.0, "std": 0.0, "folds": 0.0}
    scores, _rank, diag = _cv_rank_and_scores(
        matrix,
        [f"f_{i}" for i in range(matrix.shape[1])],
        y,
        fit_mask,
        groups,
        kind="elasticnet",
        max_per_class=int(config.elasticnet_max_samples_per_class),
        config=config,
    )
    mask = np.asarray(fit_mask, dtype=bool) & np.isfinite(scores)
    return {
        "mean": _auc_lift(scores[mask], y[mask]) if int(np.sum(mask)) >= 4 else 0.0,
        "std": float(diag.get("fold_auc_lift_std", 0.0)),
        "folds": float(diag.get("fold_count", 0)),
    }


def _timestamp_aggregate_and_smooth_scores(
    frame: pd.DataFrame,
    raw_scores: np.ndarray,
    *,
    timestamp_col: str,
    config: RegimeFeatureEngineeringConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.clip(
        np.nan_to_num(
            np.asarray(raw_scores, dtype=np.float32), nan=0.5, posinf=1.0, neginf=0.0
        ),
        0.0,
        1.0,
    )
    if (
        not bool(config.domain_score_smoothing_enabled)
        or timestamp_col not in frame.columns
        or len(raw) != len(frame)
    ):
        return (
            raw,
            raw,
            {
                "enabled": False,
                "reason": "disabled_or_missing_timestamp",
                "aggregation": "none",
            },
        )
    ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    valid = ts.notna().to_numpy(dtype=bool)
    if not bool(valid.any()):
        return (
            raw,
            raw,
            {
                "enabled": False,
                "reason": "no_valid_timestamps",
                "aggregation": "none",
            },
        )
    tmp = pd.DataFrame(
        {
            "_ts": ts,
            "_score": raw,
        },
        index=frame.index,
    )
    grouped = tmp.loc[valid].groupby("_ts", sort=True)["_score"]
    timestamp_score = grouped.mean().astype(np.float32)
    unique_ts = timestamp_score.index
    unique_values = timestamp_score.to_numpy(dtype=np.float32)
    unique_ns = unique_ts.astype("int64").to_numpy(dtype=np.int64)
    half_life_days = max(
        float(config.domain_score_ewma_half_life_days), float(config.eps)
    )
    max_days = max(float(config.domain_score_ewma_max_days), 0.0)
    if unique_values.size == 0:
        return (
            raw,
            raw,
            {
                "enabled": False,
                "reason": "empty_timestamp_groups",
                "aggregation": "timestamp_mean",
            },
        )
    day_ns = 24.0 * 3600.0 * 1_000_000_000.0
    max_age_ns = max_days * day_ns
    smoothed_unique = np.zeros_like(unique_values, dtype=np.float32)
    for i, current_ns in enumerate(unique_ns):
        if max_days > 0.0:
            start = int(
                np.searchsorted(unique_ns, int(current_ns - max_age_ns), side="left")
            )
        else:
            start = i
        age_days = (current_ns - unique_ns[start : i + 1]).astype(np.float64) / day_ns
        weights = np.power(0.5, age_days / half_life_days)
        vals = unique_values[start : i + 1].astype(np.float64)
        denom = float(np.sum(weights))
        smoothed_unique[i] = float(
            np.sum(vals * weights) / max(denom, float(config.eps))
        )
    ts_to_agg = pd.Series(unique_values, index=unique_ts)
    ts_to_smooth = pd.Series(smoothed_unique, index=unique_ts)
    aggregated = ts.map(ts_to_agg).to_numpy(dtype=np.float32)
    smoothed = ts.map(ts_to_smooth).to_numpy(dtype=np.float32)
    aggregated = np.where(np.isfinite(aggregated), aggregated, raw).astype(np.float32)
    smoothed = np.where(np.isfinite(smoothed), smoothed, raw).astype(np.float32)
    diagnostics = {
        "enabled": True,
        "aggregation": "timestamp_mean",
        "smoothing": "causal_ewma_finite_window",
        "half_life_days": float(half_life_days),
        "max_days": float(max_days),
        "timestamp_count": int(len(unique_values)),
        "valid_rows": int(valid.sum()),
        "raw_std": float(np.nanstd(raw)) if raw.size else 0.0,
        "timestamp_aggregated_std": (
            float(np.nanstd(aggregated)) if aggregated.size else 0.0
        ),
        "smoothed_std": float(np.nanstd(smoothed)) if smoothed.size else 0.0,
    }
    return (
        aggregated.astype(np.float32),
        np.clip(smoothed, 0.0, 1.0).astype(np.float32),
        diagnostics,
    )


def build_regime_specialist_feature_engineering_artifact(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    candidate_features: Sequence[str] | None = None,
    unsupervised_regime_artifact: Any | None = None,
    current_mask: Sequence[bool],
    historical_mask: Sequence[bool] | None = None,
    config: RegimeFeatureEngineeringConfig = RegimeFeatureEngineeringConfig(),
) -> RegimeFeatureEngineeringArtifact:
    total_start = time.perf_counter()
    _log(
        f"artifact build start: rows={len(frame)} cols={len(frame.columns)} "
        f"candidate_features={'all' if candidate_features is None else len(candidate_features)} "
        f"validation={bool(config.run_validation_diagnostics)} max_final_features={int(config.max_final_features)}"
    )
    unsupervised_regime_diag: dict[str, Any] = {
        "enabled": bool(unsupervised_regime_artifact is not None),
        "used": False,
    }
    if unsupervised_regime_artifact is not None:
        try:
            from .unsupervised_regime_learning.regime_models import (
                regime_artifact_assessment_summary,
            )

            unsupervised_regime_diag = regime_artifact_assessment_summary(
                unsupervised_regime_artifact
            )
        except Exception as exc:
            unsupervised_regime_diag = {
                "enabled": True,
                "used": False,
                "reason": f"artifact_assessment_failed:{type(exc).__name__}",
            }
    index = frame.index
    y = np.asarray(current_mask, dtype=bool)
    if y.size != len(frame):
        y = np.zeros(len(frame), dtype=bool)
    if historical_mask is None:
        eligible = np.ones(len(frame), dtype=bool)
    else:
        eligible = np.asarray(historical_mask, dtype=bool) | y
        if eligible.size != len(frame):
            eligible = np.ones(len(frame), dtype=bool)
    active = eligible & np.isfinite(y.astype(float))
    if not bool(active.any()):
        active = np.ones(len(frame), dtype=bool)
    _log(
        f"label masks ready: current_rows={int(np.sum(y))} "
        f"historical_or_current_rows={int(np.sum(eligible))} active_rows={int(np.sum(active))}"
    )
    stage_start = time.perf_counter()
    candidates = _candidate_pool(frame, candidate_features)
    _log(
        f"candidate pool built: candidates={len(candidates)} elapsed={_elapsed(stage_start)}"
    )
    stage_start = time.perf_counter()
    raw_matrix, raw_columns = _numeric_matrix(frame, candidates)
    _log(
        f"numeric matrix built: shape={raw_matrix.shape} columns={len(raw_columns)} "
        f"elapsed={_elapsed(stage_start)}"
    )
    stage_start = time.perf_counter()
    clean_matrix, clean_columns, cleaning_diag = _clean_columns(
        raw_matrix,
        raw_columns,
        y,
        config,
        sample_mask=active,
    )
    del raw_matrix
    _log(
        f"cleaning complete: kept={len(clean_columns)} dropped={cleaning_diag.get('dropped', 0)} "
        f"reasons={cleaning_diag.get('reasons', {})} elapsed={_elapsed(stage_start)}"
    )
    fit_mask = eligible & ~y
    symbols = frame[symbol_col].to_numpy() if symbol_col in frame.columns else None
    stage_start = time.perf_counter()
    norm_matrix = _per_symbol_robust_z(
        clean_matrix, symbols, fit_mask, float(config.eps)
    )
    del clean_matrix
    _log(
        f"per-symbol robust scaling complete: shape={norm_matrix.shape} "
        f"symbols={'none' if symbols is None else len(pd.unique(pd.Series(symbols).astype(str)))} "
        f"elapsed={_elapsed(stage_start)}"
    )
    active_idx = np.flatnonzero(active)
    norm_active = norm_matrix[active_idx]
    y_active = y[active_idx]
    report = _univariate_prefilter(norm_active, clean_columns, y_active, config)
    stage_start = time.perf_counter()
    clustered, removed, cluster_diag = _correlation_cluster_select(
        norm_active,
        clean_columns,
        report,
        y_active,
        config,
    )
    _log(
        f"correlation clustering complete: selected={len(clustered)} removed={len(removed)} "
        f"diag={cluster_diag} elapsed={_elapsed(stage_start)}"
    )
    stage_start = time.perf_counter()
    backfill_pool = _low_correlation_backfill(
        norm_active,
        clean_columns,
        clustered,
        removed,
        report,
        y_active,
        config,
    )
    _log(
        f"low-correlation backfill complete: backfill={len(backfill_pool)} "
        f"elapsed={_elapsed(stage_start)}"
    )
    relief_lgbm_only = backfill_pool[: int(config.max_relief_lgbm_only)]
    stable_sorted = (
        report.sort_values("univariate_score", ascending=False)["feature"]
        .astype(str)
        .tolist()
        if not report.empty
        else []
    )
    parents = [f for f in stable_sorted if f in clustered][
        : int(config.max_pair_parent_features)
    ]
    stage_start = time.perf_counter()
    _log(
        f"pair generation start: parents={len(parents)} max_candidates={int(config.max_pair_candidates)} "
        f"max_pairs_kept={int(config.max_pairs_kept)}"
    )
    pairs = _generate_pair_features(
        norm_active, clean_columns, parents, report, y_active, config
    )
    pair_names = [pair.name for pair in pairs]
    _log(
        f"pair generation complete: pairs={len(pairs)} elapsed={_elapsed(stage_start)}"
    )
    del norm_active, y_active
    raw_state_for_drift = _stable_unique(clustered + backfill_pool)[
        : int(config.max_drift_raw_features)
    ]
    raw_state_idx = [
        clean_columns.index(col) for col in raw_state_for_drift if col in clean_columns
    ]
    drift_source_matrix = (
        norm_matrix[:, raw_state_idx]
        if raw_state_idx
        else np.zeros((len(frame), 0), dtype=np.float32)
    )
    stage_start = time.perf_counter()
    _log(
        f"drift feature generation start: raw_state_features={len(raw_state_for_drift)} "
        f"source_shape={drift_source_matrix.shape}"
    )
    drift_matrix, drift_feature_names, drift_diag = (
        _generate_current_relative_drift_features(
            frame,
            drift_source_matrix,
            raw_state_for_drift,
            y,
            fit_mask,
            active,
            timestamp_col=timestamp_col,
            config=config,
        )
    )
    del drift_source_matrix
    _log(
        f"drift feature generation complete: drift_features={len(drift_feature_names)} "
        f"matrix_shape={drift_matrix.shape} elapsed={_elapsed(stage_start)}"
    )
    groups = _group_values(frame, timestamp_col, len(frame))
    # Current-relative drift features are distance-to-current diagnostics. Feeding
    # them to the current-vs-history discriminator is tautological because current
    # rows are the zero-distance reference. Keep them for similarity/drift blocks,
    # but exclude them from discriminator training and validation.
    lgbm_feature_names = _stable_unique(
        clustered + pair_names + relief_lgbm_only
    )
    elasticnet_feature_names = _stable_unique(clustered + pair_names)
    lgbm_matrix = _matrix_for_feature_names(
        norm_matrix,
        clean_columns,
        lgbm_feature_names,
        pairs,
        extra_matrix=drift_matrix,
        extra_columns=drift_feature_names,
    )
    _log(
        f"LGBM discriminator matrix ready: shape={lgbm_matrix.shape} "
        f"features={len(lgbm_feature_names)} enabled={bool(config.lgbm_enabled)}"
    )
    lgbm_scores, lgbm_feature_scores, lgbm_diag = _fit_lgbm_like_scores(
        lgbm_matrix,
        lgbm_feature_names,
        y,
        config,
        fit_mask=active,
        groups=groups,
    )
    del lgbm_matrix
    _log(
        f"LGBM discriminator complete: enabled={bool(lgbm_diag.get('enabled', False))} "
        f"selected={len(lgbm_diag.get('selected_features', []))} "
        f"scores={len(lgbm_feature_scores)}"
    )
    elastic_matrix = _matrix_for_feature_names(
        norm_matrix,
        clean_columns,
        elasticnet_feature_names,
        pairs,
        extra_matrix=drift_matrix,
        extra_columns=drift_feature_names,
    )
    _log(
        f"ElasticNet discriminator matrix ready: shape={elastic_matrix.shape} "
        f"features={len(elasticnet_feature_names)} enabled={bool(config.elasticnet_enabled)}"
    )
    elastic_scores, elastic_feature_scores, elastic_diag = _fit_elasticnet_scores(
        elastic_matrix,
        elasticnet_feature_names,
        y,
        config,
        fit_mask=active,
        groups=groups,
    )
    del elastic_matrix
    _log(
        f"ElasticNet discriminator complete: enabled={bool(elastic_diag.get('enabled', False))} "
        f"selected={len(elastic_diag.get('selected_features', []))} "
        f"scores={len(elastic_feature_scores)}"
    )
    all_features = _stable_unique(
        list(lgbm_feature_scores) + list(elastic_feature_scores)
    )
    final_scores = {
        feature: 0.66 * float(lgbm_feature_scores.get(feature, 0.0))
        + 0.33 * float(elastic_feature_scores.get(feature, 0.0))
        for feature in all_features
    }
    if not final_scores:
        score_map = (
            dict(
                zip(
                    report["feature"].astype(str),
                    report["univariate_score"].astype(float),
                )
            )
            if not report.empty
            else {}
        )
        final_scores = {
            feature: float(score_map.get(feature, 0.0)) for feature in clustered
        }
    pair_name_set = {pair.name for pair in pairs}
    final_selectable = set(clean_columns) | pair_name_set
    selected = [
        feature
        for feature, _score in sorted(
            final_scores.items(), key=lambda item: item[1], reverse=True
        )
        if feature in final_selectable
    ][: int(config.max_final_features)]
    if len(selected) < int(config.max_final_features):
        for feature in clustered:
            if feature not in selected and feature in clean_columns:
                selected.append(feature)
            if len(selected) >= int(config.max_final_features):
                break
    selected_raw = [feature for feature in selected if feature in clean_columns]
    selected_pairs = [feature for feature in selected if feature in pair_name_set]
    selected_drift: list[str] = []
    _log(
        f"final feature selection complete: selected={len(selected)} raw={len(selected_raw)} "
        f"pairs={len(selected_pairs)} drift={len(selected_drift)}"
    )
    active_score_parts: list[np.ndarray] = []
    if bool(lgbm_diag.get("enabled", False)):
        active_score_parts.append(lgbm_scores.astype(np.float32, copy=False))
    if bool(elastic_diag.get("enabled", False)):
        active_score_parts.append(elastic_scores.astype(np.float32, copy=False))
    if active_score_parts:
        blended_raw = np.clip(
            np.nanmean(np.vstack(active_score_parts), axis=0), 0.0, 1.0
        )
    else:
        blended_raw = np.full(len(frame), 0.5, dtype=np.float32)
    timestamp_blended, smoothed_blended, score_smoothing_diag = (
        _timestamp_aggregate_and_smooth_scores(
            frame,
            blended_raw,
            timestamp_col=timestamp_col,
            config=config,
        )
    )
    _log(
        f"domain scores blended: lgbm_enabled={bool(lgbm_diag.get('enabled', False))} "
        f"elasticnet_enabled={bool(elastic_diag.get('enabled', False))} "
        f"smoothing={score_smoothing_diag}"
    )
    row_scores = pd.DataFrame(
        {
            "regime_lgbm_current_likeness": lgbm_scores.astype(np.float32),
            "regime_elasticnet_current_likeness": elastic_scores.astype(np.float32),
            "regime_domain_current_likeness_raw": blended_raw.astype(np.float32),
            "regime_domain_current_likeness_timestamp_mean": timestamp_blended.astype(
                np.float32
            ),
            "regime_domain_current_likeness_ewma": smoothed_blended.astype(np.float32),
            "regime_domain_current_likeness": smoothed_blended.astype(np.float32),
        },
        index=index,
    )
    materialized_arrays: dict[str, np.ndarray] = {}
    materialized_groups: dict[str, list[str]] = {
        "raw_state": [],
        "pair_geometry": [],
        "generated_drift": [],
        "score": [],
    }
    clean_col_to_idx = {col: i for i, col in enumerate(clean_columns)}
    for feature in selected_raw:
        idx = clean_col_to_idx.get(feature)
        if idx is None:
            continue
        name = _safe_materialized_name("fe_raw__", feature)
        materialized_arrays[name] = norm_matrix[:, idx].astype(np.float32, copy=False)
        materialized_groups["raw_state"].append(name)
    if selected_pairs:
        pair_matrix = _matrix_for_feature_names(
            norm_matrix,
            clean_columns,
            selected_pairs,
            pairs,
        )
        for j, feature in enumerate(selected_pairs):
            if j >= pair_matrix.shape[1]:
                continue
            name = _safe_materialized_name("fe_pair__", feature)
            materialized_arrays[name] = pair_matrix[:, j].astype(np.float32, copy=False)
            materialized_groups["pair_geometry"].append(name)
    drift_col_to_idx = {col: i for i, col in enumerate(drift_feature_names)}
    for feature in drift_feature_names:
        idx = drift_col_to_idx.get(feature)
        if idx is None:
            continue
        name = _safe_materialized_name("fe_drift__", feature)
        materialized_arrays[name] = drift_matrix[:, idx].astype(np.float32, copy=False)
        materialized_groups["generated_drift"].append(name)
    for col in row_scores.columns:
        name = _safe_materialized_name("fe_score__", col)
        materialized_arrays[name] = row_scores[col].to_numpy(
            dtype=np.float32, copy=False
        )
        materialized_groups["score"].append(name)
    materialized = pd.DataFrame(materialized_arrays, index=index, dtype=np.float32)
    materialized_group_counts = {
        key: int(len(value)) for key, value in materialized_groups.items()
    }
    _log(
        f"materialized features built: columns={materialized.shape[1]} "
        f"groups={materialized_group_counts}"
    )
    empty_validation = {
        "mean": 0.0,
        "std": 0.0,
        "folds": 0.0,
        "available": False,
        "reason": "disabled",
    }
    validation_diag: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled",
        "raw": empty_validation,
        "drift": empty_validation,
        "raw_plus_drift": empty_validation,
    }
    drift_validation_unavailable = {
        "mean": 0.0,
        "std": 0.0,
        "folds": 0.0,
        "available": False,
        "reason": "current_relative_drift_is_label_conditioned_not_discriminator_valid",
    }
    if bool(config.run_validation_diagnostics):
        stage_start = time.perf_counter()
        _log(
            f"validation diagnostics start: raw_features={len(clustered)} "
            f"drift_features={len(drift_feature_names)} "
            f"drift_classifier_validation=skipped_current_relative"
        )
        raw_validation_matrix = _matrix_for_feature_names(
            norm_matrix,
            clean_columns,
            clustered,
            [],
        )
        validation_diag = {
            "enabled": True,
            "raw": _validation_auc_lift(
                raw_validation_matrix,
                y,
                active,
                groups,
                config,
            ),
            "drift": dict(drift_validation_unavailable),
            "raw_plus_drift": dict(drift_validation_unavailable),
        }
        del raw_validation_matrix
        _log(
            f"validation diagnostics complete: raw={validation_diag.get('raw')} "
            f"drift={validation_diag.get('drift')} "
            f"raw_plus_drift={validation_diag.get('raw_plus_drift')} "
            f"elapsed={_elapsed(stage_start)}"
        )
    diagnostics = {
        "schema_version": REGIME_FEATURE_ENGINEERING_SCHEMA_VERSION,
        "candidate_count": int(len(candidates)),
        "cleaning": cleaning_diag,
        "univariate_survivors": (
            int(report["selected_univariate"].sum()) if not report.empty else 0
        ),
        "cluster": cluster_diag,
        "backfill_pool_count": int(len(backfill_pool)),
        "relief_lgbm_only_count": int(len(relief_lgbm_only)),
        "pair_count": int(len(pairs)),
        "drift": drift_diag,
        "lgbm": lgbm_diag,
        "elasticnet": elastic_diag,
        "domain_score_smoothing": score_smoothing_diag,
        "current_relative_drift_discriminator_policy": {
            "included_in_discriminator": False,
            "included_in_validation": False,
            "materialized_for_similarity": bool(len(drift_feature_names) > 0),
            "reason": "current_relative_drift_features_are_distance_to_current_and_would_tautologically_separate_current_rows",
        },
        "selected_feature_count": int(len(selected)),
        "selected_raw_feature_count": int(len(selected_raw)),
        "selected_pair_feature_count": int(len(selected_pairs)),
        "selected_drift_feature_count": int(len(selected_drift)),
        "materialized_feature_count": int(materialized.shape[1]),
        "materialized_feature_groups": {
            key: int(len(value)) for key, value in materialized_groups.items()
        },
        "feature_counts_by_family": {
            "raw_selected": int(len(clustered)),
            "pair": int(len(pair_names)),
            "lgbm_only_relief": int(len(relief_lgbm_only)),
            "drift": int(len(drift_feature_names)),
        },
        "model_candidate_feature_count": {
            "lgbm": int(len(lgbm_feature_names)),
            "elasticnet": int(len(elasticnet_feature_names)),
        },
        "validation": validation_diag,
        "unsupervised_regime_learning": unsupervised_regime_diag,
    }
    _log(
        f"artifact build complete: selected={len(selected)} materialized={materialized.shape[1]} "
        f"elapsed={_elapsed(total_start)}"
    )
    return RegimeFeatureEngineeringArtifact(
        schema_version=REGIME_FEATURE_ENGINEERING_SCHEMA_VERSION,
        selected_features=selected,
        selected_raw_features=selected_raw,
        selected_pair_features=selected_pairs,
        selected_drift_features=selected_drift,
        lgbm_features=(
            list(lgbm_diag.get("selected_features", []))
            if bool(lgbm_diag.get("enabled", False))
            else []
        ),
        elasticnet_features=(
            list(elastic_diag.get("selected_features", []))
            if bool(elastic_diag.get("enabled", False))
            else []
        ),
        pair_features=pairs,
        lgbm_feature_scores=lgbm_feature_scores,
        elasticnet_feature_scores=elastic_feature_scores,
        final_feature_scores=final_scores,
        row_scores=row_scores,
        materialized_features=materialized,
        materialized_feature_groups=materialized_groups,
        feature_report=report,
        diagnostics=diagnostics,
    )


def build_regime_specialist_frozen_feature_score_artifact(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    candidate_features: Sequence[str],
    current_mask: Sequence[bool],
    historical_mask: Sequence[bool] | None = None,
    config: RegimeFeatureEngineeringConfig = RegimeFeatureEngineeringConfig(),
) -> RegimeFeatureEngineeringArtifact:
    """Train regime discriminators directly on an already frozen model feature set.

    This path is for adding LGBM/ElasticNet current-likeness score features to
    an existing model contract. It intentionally skips feature discovery,
    interaction generation, and current-relative drift generation.
    """

    total_start = time.perf_counter()
    index = frame.index
    y = np.asarray(current_mask, dtype=bool)
    if y.size != len(frame):
        y = np.zeros(len(frame), dtype=bool)
    if historical_mask is None:
        eligible = np.ones(len(frame), dtype=bool)
    else:
        eligible = np.asarray(historical_mask, dtype=bool) | y
        if eligible.size != len(frame):
            eligible = np.ones(len(frame), dtype=bool)
    active = eligible & np.isfinite(y.astype(float))
    if not bool(active.any()):
        active = np.ones(len(frame), dtype=bool)
    fit_mask = eligible & ~y
    candidates = [
        str(feature)
        for feature in _stable_unique(candidate_features)
        if str(feature) in frame.columns
    ]
    _log(
        "frozen-score artifact build start: "
        f"rows={len(frame)} candidate_features={len(candidates)} "
        f"current_rows={int(np.sum(y))} active_rows={int(np.sum(active))}"
    )
    stage_start = time.perf_counter()
    raw_matrix, raw_columns = _numeric_matrix(frame, candidates)
    _log(
        f"frozen-score numeric matrix built: shape={raw_matrix.shape} "
        f"elapsed={_elapsed(stage_start)}"
    )
    stage_start = time.perf_counter()
    clean_matrix, clean_columns, cleaning_diag = _clean_columns(
        raw_matrix,
        raw_columns,
        y,
        config,
        sample_mask=active,
    )
    del raw_matrix
    _log(
        "frozen-score cleaning complete: "
        f"kept={len(clean_columns)} dropped={cleaning_diag.get('dropped', 0)} "
        f"elapsed={_elapsed(stage_start)}"
    )
    symbols = frame[symbol_col].to_numpy() if symbol_col in frame.columns else None
    stage_start = time.perf_counter()
    norm_matrix = _per_symbol_robust_z(
        clean_matrix,
        symbols,
        fit_mask,
        float(config.eps),
    )
    del clean_matrix
    _log(
        "frozen-score robust scaling complete: "
        f"shape={norm_matrix.shape} elapsed={_elapsed(stage_start)}"
    )
    groups = _group_values(frame, timestamp_col, len(frame))
    lgbm_scores, lgbm_feature_scores, lgbm_diag = _fit_lgbm_like_scores(
        norm_matrix,
        clean_columns,
        y,
        config,
        fit_mask=active,
        groups=groups,
    )
    _log(
        "frozen-score LGBM discriminator complete: "
        f"enabled={bool(lgbm_diag.get('enabled', False))} "
        f"selected={len(lgbm_diag.get('selected_features', []))}"
    )
    elastic_scores, elastic_feature_scores, elastic_diag = _fit_elasticnet_scores(
        norm_matrix,
        clean_columns,
        y,
        config,
        fit_mask=active,
        groups=groups,
    )
    _log(
        "frozen-score ElasticNet discriminator complete: "
        f"enabled={bool(elastic_diag.get('enabled', False))} "
        f"selected={len(elastic_diag.get('selected_features', []))}"
    )
    active_score_parts: list[np.ndarray] = []
    if bool(lgbm_diag.get("enabled", False)):
        active_score_parts.append(lgbm_scores.astype(np.float32, copy=False))
    if bool(elastic_diag.get("enabled", False)):
        active_score_parts.append(elastic_scores.astype(np.float32, copy=False))
    if active_score_parts:
        blended_raw = np.clip(
            np.nanmean(np.vstack(active_score_parts), axis=0),
            0.0,
            1.0,
        ).astype(np.float32)
    else:
        blended_raw = np.full(len(frame), 0.5, dtype=np.float32)
    timestamp_blended, smoothed_blended, score_smoothing_diag = (
        _timestamp_aggregate_and_smooth_scores(
            frame,
            blended_raw,
            timestamp_col=timestamp_col,
            config=config,
        )
    )
    row_scores = pd.DataFrame(
        {
            "regime_lgbm_current_likeness": lgbm_scores.astype(np.float32),
            "regime_elasticnet_current_likeness": elastic_scores.astype(np.float32),
            "regime_domain_current_likeness_raw": blended_raw.astype(np.float32),
            "regime_domain_current_likeness_timestamp_mean": timestamp_blended.astype(
                np.float32
            ),
            "regime_domain_current_likeness_ewma": smoothed_blended.astype(np.float32),
            "regime_domain_current_likeness": smoothed_blended.astype(np.float32),
        },
        index=index,
    )
    final_scores = {
        feature: 0.66 * float(lgbm_feature_scores.get(feature, 0.0))
        + 0.33 * float(elastic_feature_scores.get(feature, 0.0))
        for feature in _stable_unique(
            list(lgbm_feature_scores) + list(elastic_feature_scores)
        )
    }
    selected = [
        feature
        for feature, _score in sorted(
            final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ][: int(config.max_final_features)]
    diagnostics = {
        "schema_version": REGIME_FEATURE_ENGINEERING_SCHEMA_VERSION,
        "mode": "frozen_model_features_score_only",
        "feature_discovery_enabled": False,
        "pair_generation_enabled": False,
        "drift_generation_enabled": False,
        "candidate_count": int(len(candidates)),
        "cleaning": cleaning_diag,
        "selected_feature_count": int(len(selected)),
        "selected_raw_feature_count": int(len(selected)),
        "selected_pair_feature_count": 0,
        "selected_drift_feature_count": 0,
        "materialized_feature_count": int(row_scores.shape[1]),
        "materialized_feature_groups": {
            "raw_state": 0,
            "pair_geometry": 0,
            "generated_drift": 0,
            "score": int(row_scores.shape[1]),
        },
        "feature_counts_by_family": {
            "raw_selected": int(len(clean_columns)),
            "pair": 0,
            "lgbm_only_relief": 0,
            "drift": 0,
        },
        "model_candidate_feature_count": {
            "lgbm": int(len(clean_columns)),
            "elasticnet": int(len(clean_columns)),
        },
        "lgbm": lgbm_diag,
        "elasticnet": elastic_diag,
        "domain_score_smoothing": score_smoothing_diag,
        "validation": {"enabled": False, "reason": "score_only_frozen_feature_path"},
        "unsupervised_regime_learning": {"enabled": False, "used": False},
        "elapsed_sec": float(time.perf_counter() - total_start),
    }
    _log(
        "frozen-score artifact build complete: "
        f"selected={len(selected)} elapsed={_elapsed(total_start)}"
    )
    return RegimeFeatureEngineeringArtifact(
        schema_version=REGIME_FEATURE_ENGINEERING_SCHEMA_VERSION,
        selected_features=selected,
        selected_raw_features=list(selected),
        selected_pair_features=[],
        selected_drift_features=[],
        lgbm_features=(
            list(lgbm_diag.get("selected_features", []))
            if bool(lgbm_diag.get("enabled", False))
            else []
        ),
        elasticnet_features=(
            list(elastic_diag.get("selected_features", []))
            if bool(elastic_diag.get("enabled", False))
            else []
        ),
        pair_features=[],
        lgbm_feature_scores=lgbm_feature_scores,
        elasticnet_feature_scores=elastic_feature_scores,
        final_feature_scores=final_scores,
        row_scores=row_scores,
        materialized_features=row_scores.copy(),
        materialized_feature_groups={"score": list(row_scores.columns)},
        feature_report=pd.DataFrame(),
        diagnostics=diagnostics,
    )
