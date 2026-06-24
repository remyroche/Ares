#!/usr/bin/env python3
"""Run contextual meta-stack trials against the current meta baseline.

The trials keep the existing meta contract: unchanged ``y_bin`` label, one
binary probability output, and chronological out-of-fold validation.  The added
signals are ordinary meta input features, generated inside each outer training
fold so validation rows do not define their own context, failure, period, or
novelty transforms.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from scripts import run_canonical_context_retrain_experiment as canon
from scripts import run_one_head_contextual_meta_ablation as ctx
from scripts.diagnose_meta_recent_failures import (
    _base_models_for_head,
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _normalise_keys,
    _prepare_model_matrix,
    lgb,
)


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
CANONICAL_10 = tuple(canon.MODEL_STATE + canon.MARKET_STATE)
MODEL_STATE = tuple(canon.MODEL_STATE)
MARKET_STATE = tuple(canon.MARKET_STATE)

TRIAL_REFIT = "T0_current_stack_refit"
TRIAL_CANONICAL = "T1_canonical_context_10"
TRIAL_FAILURE = "T2_stacked_failure_q"
TRIAL_LEAF = "T3_leaf_structural_support"
TRIAL_PERIOD = "T4_difficult_period_q"
TRIAL_NOVELTY_MAHAL = "T5a_period_novelty_mahalanobis"
TRIAL_NOVELTY_GMM = "T5b_period_novelty_gmm"
TRIAL_ALL = "T6_all_requested_blocks"
BASELINE_TRIAL = "baseline_current_meta_oof"

TRIALS = (
    TRIAL_REFIT,
    TRIAL_CANONICAL,
    TRIAL_FAILURE,
    TRIAL_LEAF,
    TRIAL_PERIOD,
    TRIAL_NOVELTY_MAHAL,
    TRIAL_NOVELTY_GMM,
    TRIAL_ALL,
)

TRIAL_SEED_OFFSET = {trial: 1000 + 137 * i for i, trial in enumerate(TRIALS, start=1)}

TIMESTAMP_FEATURE_KEYWORDS = (
    "mkt",
    "market",
    "breadth",
    "dispersion",
    "vol",
    "rv",
    "atr",
    "liquidity",
    "volume",
    "fund",
    "oi",
    "eigen",
    "tail",
    "cs_",
    "xs",
    "cross",
    "corr",
    "cov",
    "svd",
    "knn",
    "support",
    "path",
    "novelty",
    "regime",
)

FORBIDDEN_FEATURE_TOKENS = (
    "y_bin",
    "target",
    "barrier",
    "future",
    "forward",
    "pnl",
    "profit",
    "outcome",
    "leaf_target",
    "realized_outcome",
    "exit_",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_auc(y: np.ndarray, score: np.ndarray, min_rows: int = 30) -> float:
    yy = np.asarray(y)
    ss = np.asarray(score, dtype=np.float64)
    mask = (yy >= 0) & np.isfinite(ss)
    if int(mask.sum()) < int(min_rows) or len(np.unique(yy[mask])) < 2:
        return np.nan
    return float(roc_auc_score(yy[mask], ss[mask]))


def _rank_pct_by_timestamp(timestamps: pd.Series, score: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    arr = np.asarray(score, dtype=np.float64)
    out = np.full(len(arr), np.nan, dtype=np.float32)
    frame = pd.DataFrame({"timestamp": ts, "score": arr})
    for _, idx in frame.groupby("timestamp", sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        values = arr[ids]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        valid_ids = ids[finite]
        if len(valid_ids) == 1:
            out[valid_ids] = 1.0
        else:
            ranks = pd.Series(values[finite]).rank(method="first").to_numpy(dtype=np.float64)
            out[valid_ids] = ((ranks - 1.0) / max(float(len(valid_ids) - 1), 1.0)).astype(np.float32)
    return out


def _timestamp_centered_features(timestamps: pd.Series, values: np.ndarray, prefix: str) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    val = np.asarray(values, dtype=np.float32)
    frame = pd.DataFrame({"timestamp": ts, "value": val})
    mean = frame.groupby("timestamp", sort=False)["value"].transform("mean").to_numpy(dtype=np.float32)
    return pd.DataFrame(
        {
            prefix: val,
            f"{prefix}_percentile_by_timestamp": _rank_pct_by_timestamp(ts, val),
            f"{prefix}_minus_timestamp_mean": (val - mean).astype(np.float32, copy=False),
        }
    )


def _logit(values: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr)).astype(np.float32)


def _select_numeric_columns(frame: pd.DataFrame, max_columns: int) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        name = str(col)
        low = name.lower()
        if name in {"timestamp", "symbol"}:
            continue
        if any(tok in low for tok in FORBIDDEN_FEATURE_TOKENS):
            continue
        if not any(tok in low for tok in TIMESTAMP_FEATURE_KEYWORDS):
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().mean() < 0.05:
            continue
        cols.append(name)
    if not cols:
        return []
    stats = []
    for col in cols:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        var = float(np.nanvar(finite)) if finite.size else 0.0
        coverage = float(finite.size / max(len(values), 1))
        stats.append((coverage * math.log1p(max(var, 0.0)), col))
    stats.sort(reverse=True)
    return [col for _score, col in stats[: int(max_columns)]]


def _timestamp_feature_table(
    row_features: pd.DataFrame,
    timestamps: pd.Series,
    *,
    max_columns: int,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    selected = _select_numeric_columns(row_features, max_columns=max_columns)
    for col in CANONICAL_10:
        if col in row_features.columns and col not in selected:
            selected.insert(0, col)
    selected = list(dict.fromkeys(selected))[: int(max_columns)]
    if not selected:
        return pd.DataFrame(index=pd.Index(sorted(ts.dropna().unique()), name="timestamp"))
    numeric = row_features.loc[:, selected].apply(pd.to_numeric, errors="coerce").astype("float32")
    numeric["_timestamp"] = ts
    grouped_mean = numeric.groupby("_timestamp", sort=True)[selected].mean()
    grouped_mean.columns = [f"mean__{c}" for c in selected]
    std_cols = [c for c in selected if c in CANONICAL_10 or any(tok in c.lower() for tok in ("breadth", "dispersion", "vol", "fund", "oi", "liquidity"))]
    grouped_std = numeric.groupby("_timestamp", sort=True)[std_cols].std(ddof=0) if std_cols else pd.DataFrame(index=grouped_mean.index)
    if not grouped_std.empty:
        grouped_std.columns = [f"std__{c}" for c in std_cols]
    out = pd.concat([grouped_mean, grouped_std], axis=1, copy=False)
    return _downcast_numeric(out.replace([np.inf, -np.inf], np.nan))


def _align_timestamp_features(row_timestamps: pd.Series, timestamp_features: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(row_timestamps, utc=True, errors="coerce").reset_index(drop=True)
    aligned = timestamp_features.reindex(ts.to_numpy()).reset_index(drop=True)
    return _downcast_numeric(aligned)


def _fit_basic_classifier(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    seed: int,
    max_depth: int,
    n_estimators: int,
    min_child_fraction: float,
) -> tuple[np.ndarray, dict[str, Any], Any | None]:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    y_train = np.asarray(y_train, dtype=np.int8)
    y_valid = np.asarray(y_valid, dtype=np.int8)
    if len(x_train) < 100 or len(x_valid) < 20 or len(np.unique(y_train[y_train >= 0])) < 2:
        p = float(np.nanmean(y_train[y_train >= 0])) if np.any(y_train >= 0) else 0.5
        return np.full(len(x_valid), p, dtype=np.float32), {"reason": "constant_insufficient_rows_or_classes", "feature_count": 0}, None
    x_all = pd.concat([x_train, x_valid], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        p = float(np.nanmean(y_train[y_train >= 0]))
        return np.full(len(x_valid), p, dtype=np.float32), {"reason": "constant_empty_matrix", "feature_count": 0}, None
    x_prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
    x_tr = x_prepared.iloc[: len(x_train)]
    x_va = x_prepared.iloc[len(x_train) :]
    min_child = max(25, int(math.ceil(float(min_child_fraction) * len(x_train))))
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        max_depth=int(max_depth),
        num_leaves=max(4, min(24, 2 ** int(max_depth))),
        min_child_samples=int(min_child),
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
    )
    callbacks = []
    if len(np.unique(y_valid[y_valid >= 0])) >= 2 and len(y_valid) >= 30:
        callbacks = [lgb.early_stopping(35, verbose=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(
            x_tr,
            y_train,
            eval_set=[(x_va, y_valid)] if callbacks else None,
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )
    pred = clf.predict_proba(x_va)[:, 1].astype(np.float32, copy=False)
    diag = {
        "reason": "",
        "feature_count": int(len(keep_cols)),
        "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0),
        "valid_auc": _safe_auc(y_valid, pred),
    }
    return pred, diag, clf


def _fit_final_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    timestamps_train: pd.Series,
    *,
    seed: int,
    max_train_rows: int,
    max_depth: int,
    n_estimators: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_train = np.asarray(y_train, dtype=np.int8)
    valid_train = np.flatnonzero(y_train >= 0)
    if int(max_train_rows) > 0 and len(valid_train) > int(max_train_rows):
        sampled = canon._period_stratified_train_sample(
            timestamps=timestamps_train.reset_index(drop=True),
            y=np.maximum(y_train, 0),
            train_idx=valid_train,
            max_rows=int(max_train_rows),
            seed=int(seed),
        )
    else:
        sampled = valid_train
    pred, diag, _model = _fit_basic_classifier(
        x_train.iloc[sampled].reset_index(drop=True),
        y_train[sampled],
        x_valid.reset_index(drop=True),
        y_valid,
        seed=seed,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_child_fraction=0.025,
    )
    diag["train_rows"] = int(len(sampled))
    diag["valid_rows"] = int(len(x_valid))
    return pred, diag


def _canonical_fold_frames(
    raw: pd.DataFrame,
    fold: canon.FoldContext,
    definitions: dict[str, dict[str, Any]],
    *,
    trailing_window: int,
    min_periods: int,
    min_resolved_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_raw = raw.iloc[fold.train_idx].copy()
    valid_raw = raw.iloc[fold.valid_idx].copy()
    train_ctx, train_diag = canon._build_canonical_frame(
        train_raw,
        definitions,
        trailing_window=trailing_window,
        min_periods=min_periods,
        min_resolved_features=min_resolved_features,
    )
    combined = pd.concat([train_raw, valid_raw], axis=0)
    combined = combined.assign(__orig_idx=combined.index)
    combined = combined.sort_values(["timestamp", "symbol", "__orig_idx"], kind="mergesort")
    valid_all, valid_diag = canon._build_canonical_frame(
        combined.drop(columns=["__orig_idx"]),
        definitions,
        trailing_window=trailing_window,
        min_periods=min_periods,
        min_resolved_features=min_resolved_features,
    )
    valid_all.index = combined["__orig_idx"].to_numpy()
    train_ctx = train_ctx.reindex(train_raw.index).reset_index(drop=True).loc[:, list(CANONICAL_10)]
    valid_ctx = valid_all.reindex(valid_raw.index).reset_index(drop=True).loc[:, list(CANONICAL_10)]
    return _downcast_numeric(train_ctx), _downcast_numeric(valid_ctx), {
        "train_output_feature_count": int(train_diag.get("output_feature_count", 0)),
        "valid_output_feature_count": int(valid_diag.get("output_feature_count", 0)),
    }


def _assemble_base_selected_matrix(
    *,
    head: Any,
    panel: pd.DataFrame,
    base_bundle: dict[str, Any],
    feature_dir: Path,
    transform_cache: Path | None,
    symbol_columns: dict[str, set[str]],
) -> pd.DataFrame:
    _models, base_features = _base_models_for_head(base_bundle, head)
    if not base_features:
        return pd.DataFrame(index=panel.index)
    fake_race = type("FakeRace", (), {})()
    fake_best = type("FakeBest", (), {})()
    fake_best.selected_features = list(base_features)
    fake_best.get_training_meta_features = lambda: pd.DataFrame(index=panel.index)
    fake_best.model_effectiveness_history_defaults_ = {}
    fake_best.feature_stats_train = {}
    fake_race.best_model = fake_best
    selected, _coverage, _summary = ctx._assemble_selected_matrix(
        panel=panel,
        race=fake_race,
        feature_dir=feature_dir,
        transform_cache=transform_cache,
        symbol_columns=symbol_columns,
    )
    return _downcast_numeric(selected)


def _leaf_depth_value_maps(model: Any) -> tuple[list[dict[int, int]], list[dict[int, float]]]:
    depth_maps: list[dict[int, int]] = []
    value_maps: list[dict[int, float]] = []
    try:
        dumped = model.booster_.dump_model()
        trees = dumped.get("tree_info", [])
    except Exception:
        return depth_maps, value_maps

    def visit(node: dict[str, Any], depth: int, depth_map: dict[int, int], value_map: dict[int, float]) -> None:
        if "leaf_index" in node:
            leaf_id = int(node.get("leaf_index", -1))
            depth_map[leaf_id] = int(depth)
            value_map[leaf_id] = float(node.get("leaf_value", np.nan))
            return
        left = node.get("left_child")
        right = node.get("right_child")
        if isinstance(left, dict):
            visit(left, depth + 1, depth_map, value_map)
        if isinstance(right, dict):
            visit(right, depth + 1, depth_map, value_map)

    for info in trees:
        depth_map: dict[int, int] = {}
        value_map: dict[int, float] = {}
        root = info.get("tree_structure", {})
        if isinstance(root, dict):
            visit(root, 0, depth_map, value_map)
        depth_maps.append(depth_map)
        value_maps.append(value_map)
    return depth_maps, value_maps


def _model_feature_matrix(model: Any, x: pd.DataFrame) -> pd.DataFrame:
    feature_names = list(getattr(model, "feature_name_", []) or [])
    if not feature_names and hasattr(model, "booster_"):
        try:
            feature_names = [str(name) for name in model.booster_.feature_name()]
        except Exception:
            feature_names = []
    feature_names = feature_names or list(x.columns)
    out = x.copy()
    for col in feature_names:
        if col not in out.columns:
            out[col] = np.nan
    return _prepare_model_matrix(out.loc[:, feature_names])


def _leaf_structural_fold_features(
    *,
    models: list[Any],
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    prefix: str,
    max_models: int,
    tree_stride: int,
    max_trees: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not models or x_train.empty:
        return pd.DataFrame(index=x_train.index), pd.DataFrame(index=x_valid.index), {
            f"{prefix}_leaf_models_used": 0,
            f"{prefix}_leaf_trees_used": 0,
        }
    train_supports: list[np.ndarray] = []
    valid_supports: list[np.ndarray] = []
    train_depths: list[np.ndarray] = []
    valid_depths: list[np.ndarray] = []
    train_rarity: list[np.ndarray] = []
    valid_rarity: list[np.ndarray] = []
    train_leaf_values: list[np.ndarray] = []
    valid_leaf_values: list[np.ndarray] = []
    used_models = 0
    used_trees = 0
    for model in list(models)[: int(max_models)]:
        if not hasattr(model, "booster_"):
            continue
        x_all = pd.concat([x_train, x_valid], axis=0, ignore_index=True, copy=False)
        try:
            x_use = _model_feature_matrix(model, x_all)
            leaves = model.booster_.predict(x_use, pred_leaf=True)
        except Exception:
            continue
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        train_leaves = leaves[: len(x_train)]
        valid_leaves = leaves[len(x_train) :]
        depth_maps, value_maps = _leaf_depth_value_maps(model)
        tree_ids = list(range(0, train_leaves.shape[1], max(1, int(tree_stride))))[: int(max_trees)]
        if not tree_ids:
            continue
        used_models += 1
        for tree_id in tree_ids:
            tr_leaf = train_leaves[:, tree_id].astype(np.int32, copy=False)
            va_leaf = valid_leaves[:, tree_id].astype(np.int32, copy=False)
            codes, counts = np.unique(tr_leaf, return_counts=True)
            count_map = {int(k): int(v) for k, v in zip(codes, counts)}
            count_values = np.asarray(list(count_map.values()), dtype=np.float64)
            if count_values.size:
                pct_ranks = pd.Series(count_values).rank(method="average", pct=True).to_numpy(dtype=np.float32)
                pct_map = {int(k): float(v) for k, v in zip(count_map.keys(), pct_ranks)}
            else:
                pct_map = {}
            n_train = max(len(tr_leaf), 1)
            n_leaf = max(len(count_map), 1)

            def map_count(values: np.ndarray, as_percentile: bool) -> np.ndarray:
                if as_percentile:
                    return np.asarray([pct_map.get(int(v), 0.0) for v in values], dtype=np.float32)
                return np.asarray([count_map.get(int(v), 0) for v in values], dtype=np.float32)

            tr_count = map_count(tr_leaf, as_percentile=False)
            va_count = map_count(va_leaf, as_percentile=False)
            train_supports.append(map_count(tr_leaf, as_percentile=True))
            valid_supports.append(map_count(va_leaf, as_percentile=True))
            train_rarity.append((-np.log((tr_count + 1.0) / (float(n_train) + float(n_leaf)))).astype(np.float32))
            valid_rarity.append((-np.log((va_count + 1.0) / (float(n_train) + float(n_leaf)))).astype(np.float32))
            depth_map = depth_maps[tree_id] if tree_id < len(depth_maps) else {}
            value_map = value_maps[tree_id] if tree_id < len(value_maps) else {}
            train_depths.append(np.asarray([depth_map.get(int(v), np.nan) for v in tr_leaf], dtype=np.float32))
            valid_depths.append(np.asarray([depth_map.get(int(v), np.nan) for v in va_leaf], dtype=np.float32))
            train_leaf_values.append(np.asarray([value_map.get(int(v), np.nan) for v in tr_leaf], dtype=np.float32))
            valid_leaf_values.append(np.asarray([value_map.get(int(v), np.nan) for v in va_leaf], dtype=np.float32))
            used_trees += 1
    if not train_supports:
        return pd.DataFrame(index=x_train.index), pd.DataFrame(index=x_valid.index), {
            f"{prefix}_leaf_models_used": int(used_models),
            f"{prefix}_leaf_trees_used": 0,
        }

    def summarize(
        supports: list[np.ndarray],
        depths: list[np.ndarray],
        rarity: list[np.ndarray],
        leaf_values: list[np.ndarray],
        index: pd.Index,
    ) -> pd.DataFrame:
        support_m = np.vstack(supports).T.astype(np.float32, copy=False)
        depth_m = np.vstack(depths).T.astype(np.float32, copy=False)
        rarity_m = np.vstack(rarity).T.astype(np.float32, copy=False)
        value_m = np.vstack(leaf_values).T.astype(np.float32, copy=False)
        support_med = np.nanmedian(support_m, axis=1)
        support_q25 = np.nanquantile(support_m, 0.25, axis=1)
        support_min = np.nanmin(support_m, axis=1)
        path_rarity = np.nanmean(rarity_m, axis=1)
        depth_mean = np.nanmean(depth_m, axis=1)
        depth_median = np.nanmedian(depth_m, axis=1)
        depth_max = np.nanmax(depth_m, axis=1)
        disagreement = np.nanstd(support_m, axis=1)
        margin_disp = np.nanstd(value_m, axis=1)
        rarity_scale = np.nanquantile(path_rarity[np.isfinite(path_rarity)], 0.95) if np.isfinite(path_rarity).any() else 1.0
        margin_scale = np.nanquantile(margin_disp[np.isfinite(margin_disp)], 0.95) if np.isfinite(margin_disp).any() else 1.0
        rarity_norm = np.clip(path_rarity / max(float(rarity_scale), 1e-6), 0.0, 1.0)
        margin_norm = np.clip(margin_disp / max(float(margin_scale), 1e-6), 0.0, 1.0)
        uncertainty = np.nanmean(
            np.column_stack(
                [
                    1.0 - np.clip(support_med, 0.0, 1.0),
                    rarity_norm,
                    np.clip(disagreement, 0.0, 1.0),
                    margin_norm,
                ]
            ),
            axis=1,
        )
        return pd.DataFrame(
            {
                f"{prefix}_leaf_support_percentile_median": support_med,
                f"{prefix}_leaf_support_percentile_q25": support_q25,
                f"{prefix}_leaf_support_percentile_min": support_min,
                f"{prefix}_leaf_occupancy_novelty": 1.0 - np.clip(support_med, 0.0, 1.0),
                f"{prefix}_leaf_depth_mean": depth_mean,
                f"{prefix}_leaf_depth_median": depth_median,
                f"{prefix}_leaf_depth_max": depth_max,
                f"{prefix}_leaf_path_rarity": path_rarity,
                f"{prefix}_leaf_tree_disagreement": disagreement,
                f"{prefix}_leaf_margin_dispersion": margin_disp,
                f"{prefix}_leaf_structural_uncertainty_proxy": uncertainty,
            },
            index=index,
        ).astype("float32")

    return (
        _downcast_numeric(summarize(train_supports, train_depths, train_rarity, train_leaf_values, x_train.index)),
        _downcast_numeric(summarize(valid_supports, valid_depths, valid_rarity, valid_leaf_values, x_valid.index)),
        {f"{prefix}_leaf_models_used": int(used_models), f"{prefix}_leaf_trees_used": int(used_trees)},
    )


def _baseline_timestamp_hr30(timestamps: pd.Series, y: np.ndarray, score: np.ndarray) -> pd.Series:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    frame = pd.DataFrame({"timestamp": ts, "y": np.asarray(y), "score": np.asarray(score, dtype=np.float64)})
    rows: list[tuple[pd.Timestamp, float]] = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        g = group.loc[(group["y"] >= 0) & np.isfinite(group["score"])].copy()
        if len(g) < 3:
            continue
        k = max(1, int(math.ceil(0.30 * len(g))))
        ids = np.argsort(g["score"].to_numpy(dtype=np.float64), kind="mergesort")[::-1][:k]
        rows.append((pd.Timestamp(timestamp), float(np.mean(g["y"].to_numpy(dtype=np.float32)[ids]))))
    if not rows:
        return pd.Series(dtype=np.float32)
    return pd.Series(dict(rows), dtype="float32").sort_index()


def _difficult_period_labels(hr30: pd.Series, *, short_window: int, long_window: int, quantile: float) -> pd.Series:
    if hr30.empty:
        return pd.Series(dtype=np.int8)
    hr = hr30.sort_index().astype(float)
    expected = hr.expanding(min_periods=max(10, int(short_window // 3))).median().shift(1)
    expected = expected.fillna(float(hr.median()))
    roll_short = hr.rolling(int(short_window), min_periods=max(5, int(short_window // 3))).mean()
    roll_long = hr.rolling(int(long_window), min_periods=max(10, int(long_window // 3))).mean()
    surprise = np.minimum(roll_short - expected, roll_long - expected)
    finite = surprise[np.isfinite(surprise)]
    if len(finite) < 20:
        threshold = float(hr.quantile(float(quantile)))
        label = hr <= threshold
    else:
        threshold = float(finite.quantile(float(quantile)))
        label = surprise <= threshold
    return label.fillna(False).astype(np.int8)


def _fit_period_classifier_features(
    *,
    z_train: pd.DataFrame,
    z_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    valid_timestamps: pd.Series,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    period_params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_hr = _baseline_timestamp_hr30(train_timestamps, y_train, baseline_train)
    labels = _difficult_period_labels(
        train_hr,
        short_window=int(args.period_short_window),
        long_window=int(args.period_long_window),
        quantile=float(args.period_difficult_quantile),
    )
    train_z = z_train.reindex(labels.index).copy()
    valid_z = z_valid.copy()
    if labels.empty or len(np.unique(labels.to_numpy(dtype=np.int8))) < 2 or train_z.dropna(how="all").shape[0] < 30:
        q_train_ts = pd.Series(float(labels.mean()) if len(labels) else 0.5, index=z_train.index, dtype="float32")
        q_valid_ts = pd.Series(float(labels.mean()) if len(labels) else 0.5, index=z_valid.index, dtype="float32")
        diag = {"period_classifier_reason": "constant_insufficient_period_labels", "period_label_rate": float(labels.mean()) if len(labels) else np.nan}
    else:
        params = dict(period_params or {})
        pred_train, diag, model = _fit_basic_classifier(
            train_z.reset_index(drop=True),
            labels.to_numpy(dtype=np.int8),
            z_train.reset_index(drop=True),
            np.zeros(len(z_train), dtype=np.int8),
            seed=seed,
            max_depth=int(params.get("max_depth", 3)),
            n_estimators=int(params.get("n_estimators", int(args.period_n_estimators))),
            min_child_fraction=float(params.get("min_child_fraction", 0.025)),
        )
        q_train_ts = pd.Series(pred_train, index=z_train.index, dtype="float32")
        if model is None:
            q_valid_ts = pd.Series(np.nanmean(pred_train), index=z_valid.index, dtype="float32")
        else:
            x_all = pd.concat([train_z, z_valid], axis=0, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            keep_cols = [c for c in train_z.columns if pd.to_numeric(train_z[c], errors="coerce").notna().mean() > 0.02]
            x_prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
            pred_valid = model.predict_proba(x_prepared.iloc[len(train_z) :])[:, 1].astype(np.float32, copy=False)
            q_valid_ts = pd.Series(pred_valid, index=z_valid.index, dtype="float32")
        diag.update({"period_label_rate": float(labels.mean()), "period_label_timestamps": int(len(labels))})
    train_features_ts, valid_features_ts = _time_series_probability_features(
        q_train_ts,
        q_valid_ts,
        stem="q_period_norm",
        include_logit=True,
    )
    train_features = _align_timestamp_features(train_timestamps, train_features_ts)
    valid_features = _align_timestamp_features(valid_timestamps, valid_features_ts)
    return train_features, valid_features, diag


def _hpo_period_classifier_params(
    *,
    z_train: pd.DataFrame,
    train_timestamps: pd.Series,
    y_train: np.ndarray,
    baseline_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bounded chronological HPO for the timestamp-level period classifier."""
    default = {
        "max_depth": 3,
        "n_estimators": int(args.period_n_estimators),
        "min_child_fraction": 0.025,
    }
    train_hr = _baseline_timestamp_hr30(train_timestamps, y_train, baseline_train)
    labels = _difficult_period_labels(
        train_hr,
        short_window=int(args.period_short_window),
        long_window=int(args.period_long_window),
        quantile=float(args.period_difficult_quantile),
    )
    z = z_train.reindex(labels.index).copy()
    valid_rows = z.dropna(how="all").shape[0]
    if labels.empty or valid_rows < 80 or len(np.unique(labels.to_numpy(dtype=np.int8))) < 2:
        return default, {"period_hpo_status": "skipped_insufficient_labels", "period_hpo_label_rows": int(len(labels))}
    order = np.argsort(pd.to_datetime(labels.index, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]"), kind="mergesort")
    split = int(max(40, min(len(order) - 20, round(len(order) * 0.75))))
    tr_ids = order[:split]
    va_ids = order[split:]
    y_arr = labels.to_numpy(dtype=np.int8)
    if len(np.unique(y_arr[tr_ids])) < 2 or len(np.unique(y_arr[va_ids])) < 2:
        return default, {"period_hpo_status": "skipped_insufficient_split_classes", "period_hpo_label_rows": int(len(labels))}
    candidates = [
        {"max_depth": 2, "n_estimators": 120, "min_child_fraction": 0.050},
        {"max_depth": 3, "n_estimators": 160, "min_child_fraction": 0.035},
        {"max_depth": 3, "n_estimators": int(args.period_n_estimators), "min_child_fraction": 0.025},
        {"max_depth": 4, "n_estimators": 160, "min_child_fraction": 0.050},
    ]
    best = default
    best_score = -np.inf
    rows: list[dict[str, Any]] = []
    for i, params in enumerate(candidates):
        pred, diag, _model = _fit_basic_classifier(
            z.iloc[tr_ids].reset_index(drop=True),
            y_arr[tr_ids],
            z.iloc[va_ids].reset_index(drop=True),
            y_arr[va_ids],
            seed=int(seed + i * 97),
            max_depth=int(params["max_depth"]),
            n_estimators=int(params["n_estimators"]),
            min_child_fraction=float(params["min_child_fraction"]),
        )
        auc = _safe_auc(y_arr[va_ids], pred, min_rows=20)
        score = float(np.nan_to_num(auc, nan=0.5)) - 0.0005 * float(params["max_depth"])
        rows.append({**params, "auc": auc, "objective": score, "reason": diag.get("reason", "")})
        if score > best_score:
            best_score = score
            best = dict(params)
    return best, {
        "period_hpo_status": "selected",
        "period_hpo_label_rows": int(len(labels)),
        "period_hpo_train_timestamps": int(len(tr_ids)),
        "period_hpo_valid_timestamps": int(len(va_ids)),
        "period_hpo_best_objective": float(best_score),
        "period_hpo_selected_params": json.dumps(best, sort_keys=True),
        "period_hpo_candidates": json.dumps(rows, default=_json_default),
    }


def _time_series_probability_features(
    train_score: pd.Series,
    valid_score: pd.Series,
    *,
    stem: str,
    include_logit: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_score = train_score.sort_index().astype(float)
    valid_score = valid_score.sort_index().astype(float)
    combined = pd.concat([train_score, valid_score], axis=0).sort_index()
    lag = combined.shift(12)
    recent = combined.rolling(24, min_periods=1).max()
    change = combined - lag
    out = pd.DataFrame(
        {
            stem: combined,
            f"{stem}_12h_lag": lag,
            f"{stem}_change_12h": change,
            f"{stem}_recent_max_24h": recent,
        },
        index=combined.index,
    )
    if include_logit:
        out[f"logit_{stem.replace('_norm', '')}"] = _logit(out[stem].to_numpy(dtype=np.float64))
    return (
        _downcast_numeric(out.reindex(train_score.index)),
        _downcast_numeric(out.reindex(valid_score.index)),
    )


def _fit_period_novelty_features(
    *,
    z_train: pd.DataFrame,
    z_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    valid_timestamps: pd.Series,
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = z_train.replace([np.inf, -np.inf], np.nan)
    valid = z_valid.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in train.columns if pd.to_numeric(train[c], errors="coerce").notna().mean() > 0.05]
    diag: dict[str, Any] = {"novelty_mode": mode, "novelty_feature_count": int(len(keep_cols))}
    if len(keep_cols) < 2 or len(train) < 30:
        train_score = pd.Series(np.nan, index=z_train.index, dtype="float32")
        valid_score = pd.Series(np.nan, index=z_valid.index, dtype="float32")
    else:
        train_arr = train.loc[:, keep_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        valid_arr = valid.loc[:, keep_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        med = np.nanmedian(train_arr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        train_arr = np.where(np.isfinite(train_arr), train_arr, med)
        valid_arr = np.where(np.isfinite(valid_arr), valid_arr, med)
        mean = np.nanmean(train_arr, axis=0)
        scale = np.nanstd(train_arr, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        train_zv = (train_arr - mean) / scale
        valid_zv = (valid_arr - mean) / scale
        if mode == "mahalanobis":
            try:
                cov = LedoitWolf().fit(train_zv)
                train_raw = cov.mahalanobis(train_zv)
                valid_raw = cov.mahalanobis(valid_zv)
                diag["novelty_reason"] = ""
            except Exception as exc:
                train_raw = np.sum(train_zv * train_zv, axis=1)
                valid_raw = np.sum(valid_zv * valid_zv, axis=1)
                diag["novelty_reason"] = f"mahalanobis_fallback_l2:{exc}"
        else:
            n_components = max(1, min(int(args.gmm_components), max(1, len(train_zv) // 80)))
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type="diag",
                    reg_covar=1e-4,
                    max_iter=120,
                    random_state=int(seed),
                )
                gmm.fit(train_zv)
                train_raw = -gmm.score_samples(train_zv)
                valid_raw = -gmm.score_samples(valid_zv)
                diag["gmm_components"] = int(n_components)
                diag["novelty_reason"] = ""
            except Exception as exc:
                train_raw = np.sum(train_zv * train_zv, axis=1)
                valid_raw = np.sum(valid_zv * valid_zv, axis=1)
                diag["novelty_reason"] = f"gmm_fallback_l2:{exc}"
        train_score = pd.Series(train_raw.astype(np.float32), index=z_train.index)
        valid_score = pd.Series(valid_raw.astype(np.float32), index=z_valid.index)
    finite_train = train_score[np.isfinite(train_score)]
    if len(finite_train) >= 10:
        ranks_train = pd.Series(train_score).rank(pct=True).to_numpy(dtype=np.float32)
        valid_pct = np.searchsorted(np.sort(finite_train.to_numpy(dtype=np.float64)), valid_score.to_numpy(dtype=np.float64), side="right")
        valid_pct = valid_pct.astype(np.float64) / max(float(len(finite_train)), 1.0)
    else:
        ranks_train = np.full(len(train_score), np.nan, dtype=np.float32)
        valid_pct = np.full(len(valid_score), np.nan, dtype=np.float32)
    train_norm = pd.Series(ranks_train, index=train_score.index, dtype="float32")
    valid_norm = pd.Series(valid_pct.astype(np.float32), index=valid_score.index, dtype="float32")
    train_ts, valid_ts = _time_series_probability_features(
        train_norm,
        valid_norm,
        stem="period_novelty_score",
        include_logit=False,
    )
    train_ts = train_ts.rename(
        columns={
            "period_novelty_score_12h_lag": "period_novelty_lag_12h",
            "period_novelty_score_change_12h": "period_novelty_change_24h",
            "period_novelty_score_recent_max_24h": "period_novelty_recent_max",
        }
    )
    valid_ts = valid_ts.rename(
        columns={
            "period_novelty_score_12h_lag": "period_novelty_lag_12h",
            "period_novelty_score_change_12h": "period_novelty_change_24h",
            "period_novelty_score_recent_max_24h": "period_novelty_recent_max",
        }
    )
    train_ts["period_novelty_percentile"] = train_norm.reindex(train_ts.index).astype("float32")
    valid_ts["period_novelty_percentile"] = valid_norm.reindex(valid_ts.index).astype("float32")
    return _align_timestamp_features(train_timestamps, train_ts), _align_timestamp_features(valid_timestamps, valid_ts), diag


def _fit_failure_stack_features(
    *,
    source_train: pd.DataFrame,
    source_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    valid_timestamps: pd.Series,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rank = _rank_pct_by_timestamp(train_timestamps, anchor_train)
    target = ((rank >= float(args.failure_rank_threshold)) & (np.asarray(y_train, dtype=np.int8) == 0)).astype(np.int8)
    if len(np.unique(target)) < 2 or int(target.sum()) < 20:
        q_train = np.full(len(source_train), float(np.mean(target)) if len(target) else 0.0, dtype=np.float32)
        q_valid = np.full(len(source_valid), float(np.mean(target)) if len(target) else 0.0, dtype=np.float32)
        diag = {"failure_classifier_reason": "constant_insufficient_high_conf_fail", "failure_label_rate": float(np.mean(target)) if len(target) else np.nan}
    else:
        pred_valid, diag, model = _fit_basic_classifier(
            source_train.reset_index(drop=True),
            target,
            source_valid.reset_index(drop=True),
            np.zeros(len(source_valid), dtype=np.int8),
            seed=seed,
            max_depth=3,
            n_estimators=int(args.failure_n_estimators),
            min_child_fraction=0.025,
        )
        if model is None:
            q_train = np.full(len(source_train), np.nanmean(pred_valid), dtype=np.float32)
        else:
            x_all = pd.concat([source_train, source_train], axis=0, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            keep_cols = [c for c in source_train.columns if pd.to_numeric(source_train[c], errors="coerce").notna().mean() > 0.02]
            x_prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
            q_train = model.predict_proba(x_prepared.iloc[: len(source_train)])[:, 1].astype(np.float32, copy=False)
        q_valid = pred_valid
        diag = {f"failure_classifier_{k}": v for k, v in diag.items()}
        diag["failure_label_rate"] = float(np.mean(target))
        diag["failure_label_positive_count"] = int(target.sum())
    train_feat = _timestamp_centered_features(train_timestamps, q_train, "q_fail_norm")
    valid_feat = _timestamp_centered_features(valid_timestamps, q_valid, "q_fail_norm")
    return _downcast_numeric(train_feat), _downcast_numeric(valid_feat), diag


def _combine_features(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [f.reset_index(drop=True) for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(index=frames[0].index if frames else None)
    out = pd.concat(parts, axis=1, copy=False)
    out = out.loc[:, ~out.columns.duplicated()]
    return _downcast_numeric(out)


def _tail_timestamp_rows(
    *,
    head: str,
    trial: str,
    timestamps: pd.Series,
    y: np.ndarray,
    score: np.ndarray,
    baseline_score: np.ndarray,
    min_timestamp_rows: int,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "idx": np.arange(len(y), dtype=np.int64),
            "timestamp": ts,
            "y": np.asarray(y, dtype=np.int8),
            "score": np.asarray(score, dtype=np.float64),
            "baseline_score": np.asarray(baseline_score, dtype=np.float64),
        }
    )
    rows: list[dict[str, Any]] = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        g = group.loc[(group["y"] >= 0) & np.isfinite(group["score"]) & np.isfinite(group["baseline_score"])]
        if len(g) < int(min_timestamp_rows):
            continue
        yy = g["y"].to_numpy(dtype=np.float32)
        ss = g["score"].to_numpy(dtype=np.float64)
        bb = g["baseline_score"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "head": head,
            "trial": trial,
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "week": pd.Timestamp(timestamp).to_period("W").start_time.strftime("%Y-%m-%d"),
            "eligible_rows": int(len(g)),
        }
        for pct in (10, 20, 30):
            frac = pct / 100.0
            k = max(1, int(math.ceil(frac * len(g))))
            cur = np.argsort(ss, kind="mergesort")[::-1][:k]
            base = np.argsort(bb, kind="mergesort")[::-1][:k]
            row[f"hr_at_{pct}"] = float(np.mean(yy[cur])) if len(cur) else np.nan
            row[f"baseline_hr_at_{pct}"] = float(np.mean(yy[base])) if len(base) else np.nan
            row[f"delta_hr_at_{pct}"] = row[f"hr_at_{pct}"] - row[f"baseline_hr_at_{pct}"]
            row[f"selected_at_{pct}"] = int(k)
            row[f"hit_count_at_{pct}"] = float(np.sum(yy[cur]))
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_trial_metrics(
    *,
    head: str,
    trial: str,
    timestamps: pd.Series,
    y: np.ndarray,
    score: np.ndarray,
    baseline_score: np.ndarray,
    min_timestamp_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ts_rows = _tail_timestamp_rows(
        head=head,
        trial=trial,
        timestamps=timestamps,
        y=y,
        score=score,
        baseline_score=baseline_score,
        min_timestamp_rows=min_timestamp_rows,
    )
    mask = (np.asarray(y) >= 0) & np.isfinite(score) & np.isfinite(baseline_score)
    row: dict[str, Any] = {
        "head": head,
        "trial": trial,
        "rows": int(mask.sum()),
        "timestamp_count": int(ts_rows["timestamp"].nunique()) if not ts_rows.empty else 0,
        "auc": _safe_auc(y, score),
        "baseline_auc": _safe_auc(y, baseline_score),
    }
    row["delta_auc"] = row["auc"] - row["baseline_auc"] if np.isfinite(row["auc"]) and np.isfinite(row["baseline_auc"]) else np.nan
    for pct in (10, 20, 30):
        col = f"hr_at_{pct}"
        base_col = f"baseline_hr_at_{pct}"
        if ts_rows.empty:
            row[f"hr_at_{pct}_global_ts_weighted"] = np.nan
            row[f"baseline_hr_at_{pct}_global_ts_weighted"] = np.nan
            row[f"delta_hr_at_{pct}_global_ts_weighted"] = np.nan
        else:
            row[f"hr_at_{pct}_global_ts_weighted"] = float(pd.to_numeric(ts_rows[col], errors="coerce").mean())
            row[f"baseline_hr_at_{pct}_global_ts_weighted"] = float(pd.to_numeric(ts_rows[base_col], errors="coerce").mean())
            row[f"delta_hr_at_{pct}_global_ts_weighted"] = float(pd.to_numeric(ts_rows[f"delta_hr_at_{pct}"], errors="coerce").mean())
        for q_name, q in (("q05", 0.05), ("q10", 0.10), ("q25", 0.25), ("q50", 0.50)):
            if ts_rows.empty:
                row[f"hr_at_{pct}_week_{q_name}"] = np.nan
                row[f"baseline_hr_at_{pct}_week_{q_name}"] = np.nan
                row[f"delta_hr_at_{pct}_week_{q_name}"] = np.nan
            else:
                weekly = ts_rows.groupby("week", sort=True)[col].mean()
                base_weekly = ts_rows.groupby("week", sort=True)[base_col].mean()
                delta_weekly = ts_rows.groupby("week", sort=True)[f"delta_hr_at_{pct}"].mean()
                row[f"hr_at_{pct}_week_{q_name}"] = float(weekly.quantile(q)) if len(weekly) else np.nan
                row[f"baseline_hr_at_{pct}_week_{q_name}"] = float(base_weekly.quantile(q)) if len(base_weekly) else np.nan
                row[f"delta_hr_at_{pct}_week_{q_name}"] = float(delta_weekly.quantile(q)) if len(delta_weekly) else np.nan
    week_auc: list[float] = []
    if mask.any():
        tmp = pd.DataFrame(
            {
                "week": pd.to_datetime(timestamps, utc=True, errors="coerce").dt.to_period("W").dt.start_time.astype(str),
                "y": np.asarray(y),
                "score": np.asarray(score, dtype=np.float64),
            }
        )
        for _, group in tmp.loc[mask].groupby("week", sort=True):
            auc = _safe_auc(group["y"].to_numpy(dtype=np.int8), group["score"].to_numpy(dtype=np.float64), min_rows=20)
            if np.isfinite(auc):
                week_auc.append(float(auc))
    for q_name, q in (("q05", 0.05), ("q10", 0.10), ("q25", 0.25), ("q50", 0.50)):
        row[f"auc_week_{q_name}"] = float(np.nanquantile(week_auc, q)) if week_auc else np.nan
    return row, ts_rows


def _write_markdown_report(out_dir: Path, summary: pd.DataFrame, fold_diag: pd.DataFrame, args: argparse.Namespace) -> Path:
    report = out_dir / "contextual_meta_stack_trials_report.md"
    lines = [
        "# Contextual Meta Stack Trials",
        "",
        "Chronological out-of-fold trials against the current meta OOF baseline.",
        "",
        "## Validation Contract",
        "",
        "- Target: unchanged `y_bin`.",
        "- Output: one binary meta probability per trial.",
        "- Splits: chronological expanding folds with embargo.",
        "- Fold-fitted blocks: canonical context, failure stack, leaf structure, difficult-period state, and novelty.",
        "- Leaf structural features use training leaf populations only and no realized leaf outcomes.",
        "- Difficult-period and novelty features are timestamp-level; rows-per-timestamp/session/universe composition are not model inputs.",
        "",
        "## Head/Trial Summary",
        "",
    ]
    if summary.empty:
        lines.append("No metrics produced.")
    else:
        display_cols = [
            "head",
            "trial",
            "rows",
            "timestamp_count",
            "auc",
            "delta_auc",
            "hr_at_30_global_ts_weighted",
            "delta_hr_at_30_global_ts_weighted",
            "hr_at_20_global_ts_weighted",
            "delta_hr_at_20_global_ts_weighted",
            "hr_at_10_global_ts_weighted",
            "delta_hr_at_10_global_ts_weighted",
            "hr_at_30_week_q05",
            "hr_at_30_week_q10",
            "hr_at_30_week_q25",
            "hr_at_30_week_q50",
        ]
        cols = [c for c in display_cols if c in summary.columns]
        lines.append(summary[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Trial Definitions",
            "",
            f"- `{TRIAL_REFIT}`: current meta feature stack refit under the same chronological fold/training procedure; use this to isolate incremental feature-block effects.",
            f"- `{TRIAL_CANONICAL}`: current meta feature stack plus the 10 canonical model-state and market-state context variables.",
            f"- `{TRIAL_FAILURE}`: current stack plus `q_fail_norm`, timestamp percentile, and timestamp demeaned failure-risk score.",
            f"- `{TRIAL_LEAF}`: current stack plus `base_leaf_*` and `anchor_meta_leaf_*` support, novelty, depth, rarity, and structural uncertainty proxies.",
            f"- `{TRIAL_PERIOD}`: current stack plus timestamp-level difficult-period probability transforms.",
            f"- `{TRIAL_NOVELTY_MAHAL}` / `{TRIAL_NOVELTY_GMM}`: current stack plus separate period novelty scores.",
            f"- `{TRIAL_ALL}`: current stack plus all requested blocks.",
            "",
            "## Artifacts",
            "",
            f"- Summary CSV: `{out_dir / 'trial_summary.csv'}`",
            f"- Timestamp tail metrics: `{out_dir / 'trial_timestamp_metrics.csv'}`",
            f"- Fold diagnostics: `{out_dir / 'trial_fold_diagnostics.csv'}`",
            f"- OOF scores: `{out_dir / 'trial_oof_scores.parquet'}`",
            "",
        ]
    )
    if not fold_diag.empty:
        lines.extend(
            [
                "## Fold Diagnostics Snapshot",
                "",
                fold_diag.head(40).to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    with report.open("w") as fh:
        fh.write("\n".join(lines))
    return report


def run(args: argparse.Namespace) -> Path:
    out_dir = _ensure_dir(Path(args.output_dir))
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    feature_dir = Path(args.feature_dir)
    report_dir = Path(args.report_dir)
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    canonical_defs = canon._load_canonical_definitions(Path(args.canonical_reduction))
    if not canonical_defs:
        raise RuntimeError("No canonical definitions could be loaded")
    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    wanted = set(str(x) for x in (args.only_head or HEADS))
    heads = [h for h in heads if h.head in wanted]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)
    symbol_columns = _feature_store_union(feature_dir)

    summary_rows: list[dict[str, Any]] = []
    timestamp_metric_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    period_hpo_params: dict[str, Any] | None = None
    period_hpo_diag: dict[str, Any] = {"period_hpo_status": "disabled" if not bool(args.period_hpo) else "pending"}

    for head in heads:
        print(f"[context_stack_trials] head={head.head}", flush=True)
        panel = _downcast_numeric(_normalise_keys(pd.read_parquet(head.meta_oof_path)), exclude=["timestamp", "symbol"])
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
            print(f"[context_stack_trials] sampled head={head.head} rows={len(panel)}", flush=True)
        race = meta_models[head.meta_key]
        current_x, raw = ctx._assemble_head_context(
            head=head,
            panel=panel,
            race=race,
            base_bundle=base_bundle,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
            regime_context=None,
            max_regime_columns=0,
        )
        base_x = _assemble_base_selected_matrix(
            head=head,
            panel=panel,
            base_bundle=base_bundle,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
        y = ctx._meta_target(panel)
        baseline = ctx._current_meta_score(panel)
        folds = canon._make_chrono_folds(panel["timestamp"], int(args.outer_folds), embargo_hours=int(args.embargo_hours))
        fold_valid_mask = np.zeros(len(panel), dtype=bool)
        trial_preds: dict[str, np.ndarray] = {trial: np.full(len(panel), np.nan, dtype=np.float32) for trial in TRIALS}
        for fold in folds:
            tr = np.asarray(fold.train_idx, dtype=np.int64)
            va = np.asarray(fold.valid_idx, dtype=np.int64)
            fold_valid_mask[va] = True
            train_raw = raw.iloc[tr].reset_index(drop=True)
            valid_raw = raw.iloc[va].reset_index(drop=True)
            x_train_current = current_x.iloc[tr].reset_index(drop=True)
            x_valid_current = current_x.iloc[va].reset_index(drop=True)
            y_train = y[tr]
            y_valid = y[va]
            ts_train = panel["timestamp"].iloc[tr].reset_index(drop=True)
            ts_valid = panel["timestamp"].iloc[va].reset_index(drop=True)
            canonical_train, canonical_valid, canonical_diag = _canonical_fold_frames(
                raw,
                fold,
                canonical_defs,
                trailing_window=int(args.trailing_window),
                min_periods=int(args.min_periods),
                min_resolved_features=int(args.min_resolved_features),
            )
            context_source_train = _combine_features(x_train_current, canonical_train)
            context_source_valid = _combine_features(x_valid_current, canonical_valid)
            failure_train, failure_valid, failure_diag = _fit_failure_stack_features(
                source_train=context_source_train,
                source_valid=context_source_valid,
                train_timestamps=ts_train,
                valid_timestamps=ts_valid,
                y_train=y_train,
                anchor_train=baseline[tr],
                seed=int(args.seed + 101 * fold.fold_id),
                args=args,
            )
            meta_models_list = list(getattr(getattr(race, "best_model", None), "models", []) or [])
            meta_leaf_train, meta_leaf_valid, meta_leaf_diag = _leaf_structural_fold_features(
                models=meta_models_list,
                x_train=x_train_current,
                x_valid=x_valid_current,
                prefix="anchor_meta_leaf",
                max_models=int(args.leaf_max_models),
                tree_stride=int(args.leaf_tree_stride),
                max_trees=int(args.leaf_max_trees),
            )
            base_models, _base_features = _base_models_for_head(base_bundle, head)
            base_leaf_train, base_leaf_valid, base_leaf_diag = _leaf_structural_fold_features(
                models=base_models,
                x_train=base_x.iloc[tr].reset_index(drop=True),
                x_valid=base_x.iloc[va].reset_index(drop=True),
                prefix="base_leaf",
                max_models=int(args.leaf_max_models),
                tree_stride=int(args.leaf_tree_stride),
                max_trees=int(args.leaf_max_trees),
            )
            leaf_train = _combine_features(base_leaf_train, meta_leaf_train)
            leaf_valid = _combine_features(base_leaf_valid, meta_leaf_valid)
            z_source_train = _combine_features(canonical_train, x_train_current)
            z_source_valid = _combine_features(canonical_valid, x_valid_current)
            z_train = _timestamp_feature_table(z_source_train, ts_train, max_columns=int(args.max_timestamp_features))
            z_valid = _timestamp_feature_table(z_source_valid, ts_valid, max_columns=int(args.max_timestamp_features))
            z_valid = z_valid.reindex(columns=z_train.columns)
            if bool(args.period_hpo) and period_hpo_params is None:
                candidate_period_params, period_hpo_diag = _hpo_period_classifier_params(
                    z_train=z_train,
                    train_timestamps=ts_train,
                    y_train=y_train,
                    baseline_train=baseline[tr],
                    seed=int(args.seed + 1973),
                    args=args,
                )
                if str(period_hpo_diag.get("period_hpo_status", "")) == "selected":
                    period_hpo_params = candidate_period_params
            period_train, period_valid, period_diag = _fit_period_classifier_features(
                z_train=z_train,
                z_valid=z_valid,
                train_timestamps=ts_train,
                valid_timestamps=ts_valid,
                y_train=y_train,
                baseline_train=baseline[tr],
                seed=int(args.seed + 211 * fold.fold_id),
                args=args,
                period_params=period_hpo_params,
            )
            novelty_mahal_train, novelty_mahal_valid, novelty_mahal_diag = _fit_period_novelty_features(
                z_train=z_train,
                z_valid=z_valid,
                train_timestamps=ts_train,
                valid_timestamps=ts_valid,
                mode="mahalanobis",
                seed=int(args.seed + 307 * fold.fold_id),
                args=args,
            )
            novelty_gmm_train, novelty_gmm_valid, novelty_gmm_diag = _fit_period_novelty_features(
                z_train=z_train,
                z_valid=z_valid,
                train_timestamps=ts_train,
                valid_timestamps=ts_valid,
                mode="gmm",
                seed=int(args.seed + 401 * fold.fold_id),
                args=args,
            )
            blocks = {
                TRIAL_REFIT: (
                    pd.DataFrame(index=x_train_current.index),
                    pd.DataFrame(index=x_valid_current.index),
                ),
                TRIAL_CANONICAL: (canonical_train, canonical_valid),
                TRIAL_FAILURE: (failure_train, failure_valid),
                TRIAL_LEAF: (leaf_train, leaf_valid),
                TRIAL_PERIOD: (period_train, period_valid),
                TRIAL_NOVELTY_MAHAL: (novelty_mahal_train, novelty_mahal_valid),
                TRIAL_NOVELTY_GMM: (novelty_gmm_train, novelty_gmm_valid),
                TRIAL_ALL: (
                    _combine_features(canonical_train, failure_train, leaf_train, period_train, novelty_mahal_train, novelty_gmm_train),
                    _combine_features(canonical_valid, failure_valid, leaf_valid, period_valid, novelty_mahal_valid, novelty_gmm_valid),
                ),
            }
            common_diag = {
                "head": head.head,
                "fold": int(fold.fold_id),
                "train_rows": int(len(tr)),
                "valid_rows": int(len(va)),
                **canonical_diag,
                **failure_diag,
                **meta_leaf_diag,
                **base_leaf_diag,
                **period_diag,
                **period_hpo_diag,
                **{f"mahal_{k}": v for k, v in novelty_mahal_diag.items()},
                **{f"gmm_{k}": v for k, v in novelty_gmm_diag.items()},
            }
            for trial, (block_train, block_valid) in blocks.items():
                x_train_trial = _combine_features(x_train_current, block_train)
                x_valid_trial = _combine_features(x_valid_current, block_valid)
                pred, fit_diag = _fit_final_fold(
                    x_train_trial,
                    y_train,
                    x_valid_trial,
                    y_valid,
                    ts_train,
                    seed=int(args.seed + 1009 * fold.fold_id + TRIAL_SEED_OFFSET.get(trial, 9999)),
                    max_train_rows=int(args.max_train_rows),
                    max_depth=int(args.max_depth),
                    n_estimators=int(args.n_estimators),
                )
                trial_preds[trial][va] = pred
                fold_rows.append(
                    {
                        **common_diag,
                        "trial": trial,
                        "trial_block_features": int(block_train.shape[1]) if block_train is not None else 0,
                        **{f"final_{k}": v for k, v in fit_diag.items()},
                    }
                )
            print(
                f"[context_stack_trials] head={head.head} fold={fold.fold_id}/{len(folds)} "
                f"train={len(tr)} valid={len(va)}",
                flush=True,
            )

        baseline_fold = baseline.copy()
        baseline_fold[~fold_valid_mask] = np.nan
        baseline_summary, baseline_ts = _aggregate_trial_metrics(
            head=head.head,
            trial=BASELINE_TRIAL,
            timestamps=panel["timestamp"],
            y=y,
            score=baseline_fold,
            baseline_score=baseline_fold,
            min_timestamp_rows=int(args.min_timestamp_rows),
        )
        summary_rows.append(baseline_summary)
        if not baseline_ts.empty:
            timestamp_metric_rows.append(baseline_ts)
        pred_frame = pd.DataFrame(
            {
                "head": head.head,
                "row_id": np.arange(len(panel), dtype=np.int64),
                "timestamp": pd.to_datetime(panel["timestamp"], utc=True, errors="coerce"),
                "symbol": panel["symbol"].astype(str) if "symbol" in panel.columns else "",
                "y_bin": y,
                "baseline_score": baseline_fold,
            }
        )
        for trial, pred in trial_preds.items():
            pred_frame[trial] = pred
            summary, ts_metrics = _aggregate_trial_metrics(
                head=head.head,
                trial=trial,
                timestamps=panel["timestamp"],
                y=y,
                score=pred,
                baseline_score=baseline_fold,
                min_timestamp_rows=int(args.min_timestamp_rows),
            )
            summary_rows.append(summary)
            if not ts_metrics.empty:
                timestamp_metric_rows.append(ts_metrics)
        score_frames.append(pred_frame)

    summary = pd.DataFrame(summary_rows)
    timestamp_metrics = pd.concat(timestamp_metric_rows, axis=0, ignore_index=True) if timestamp_metric_rows else pd.DataFrame()
    fold_diag = pd.DataFrame(fold_rows)
    scores = pd.concat(score_frames, axis=0, ignore_index=True) if score_frames else pd.DataFrame()
    summary.to_csv(out_dir / "trial_summary.csv", index=False)
    timestamp_metrics.to_csv(out_dir / "trial_timestamp_metrics.csv", index=False)
    fold_diag.to_csv(out_dir / "trial_fold_diagnostics.csv", index=False)
    if not scores.empty:
        scores.to_parquet(out_dir / "trial_oof_scores.parquet", index=False)
    manifest = {
        "output_dir": str(out_dir),
        "meta_artifact_dir": str(meta_artifact_dir),
        "baseline_artifact_dir": str(baseline_artifact_dir),
        "feature_dir": str(feature_dir),
        "canonical_reduction": str(args.canonical_reduction),
        "heads": [h.head for h in heads],
        "trials": list((BASELINE_TRIAL,) + TRIALS),
        "outer_folds": int(args.outer_folds),
        "embargo_hours": int(args.embargo_hours),
        "min_timestamp_rows": int(args.min_timestamp_rows),
        "top_tail_metric": "timestamp_weighted_top_10_20_30_percent_by_trial_score",
        "weekly_tail_quantiles": ["q05", "q10", "q25", "q50"],
        "leakage_controls": [
            "chronological folds",
            "validation feature blocks fitted from outer training history",
            "unchanged y_bin target",
            "leaf features use training leaf occupancy only",
            "difficult period classifier excludes row count/session/universe composition inputs",
        ],
    }
    with (out_dir / "trial_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)
    report = _write_markdown_report(out_dir, summary, fold_diag, args)
    print(f"[context_stack_trials] wrote {report}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--transform-cache", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/contextual_meta_stack_trials_20260623")
    parser.add_argument("--only-head", nargs="*", default=list(HEADS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=320)
    parser.add_argument("--failure-rank-threshold", type=float, default=0.70)
    parser.add_argument("--failure-n-estimators", type=int, default=220)
    parser.add_argument("--period-n-estimators", type=int, default=220)
    parser.add_argument("--period-hpo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--period-short-window", type=int, default=72)
    parser.add_argument("--period-long-window", type=int, default=120)
    parser.add_argument("--period-difficult-quantile", type=float, default=0.25)
    parser.add_argument("--max-timestamp-features", type=int, default=80)
    parser.add_argument("--gmm-components", type=int, default=4)
    parser.add_argument("--leaf-max-models", type=int, default=1)
    parser.add_argument("--leaf-tree-stride", type=int, default=3)
    parser.add_argument("--leaf-max-trees", type=int, default=120)
    parser.add_argument("--min-timestamp-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
