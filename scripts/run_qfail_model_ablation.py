#!/usr/bin/env python3
"""q_fail model ablation for anchored meta corrections.

This isolates improvements to the high-confidence failure model:

1. swap-critical target;
2. pairwise/ranker-style failure ranking;
3. anchor-only versus path-controlled incremental signal;
4. timestamp-balanced q_fail training;
6. anchor-rank-banded q_fail models.

The final score remains a single anchored meta probability:
``z_final = logit(meta_0) + delta``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from scripts import run_anchored_reliability_meta_correction as anchored
from scripts import run_canonical_context_retrain_experiment as canon
from scripts import run_contextual_meta_stack_trials as stack
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
from scripts.quantify_bad_regime_archetype_usefulness import _pick_realized_return


HEADS = anchored.HEADS
TRIAL_ANCHOR = "M0_anchor_current_meta"

QFAIL_VARIANTS = (
    "Q0_broad_path_control",
    "Q1_swap_anchor_control",
    "Q2_swap_path_control",
    "Q3_pairwise_ranker_swap_path",
    "Q4_rank_banded_path_control",
)


def _json_default(value: Any) -> Any:
    return ctx._json_default(value)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _equal_timestamp_weights(timestamps: pd.Series, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    weights = np.zeros(len(mask), dtype=np.float32)
    if not mask.any():
        return weights
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    ids = np.flatnonzero(mask)
    frame = pd.DataFrame({"idx": ids, "timestamp": ts.iloc[ids].to_numpy()})
    for _, group in frame.groupby("timestamp", sort=False):
        g = group["idx"].to_numpy(dtype=np.int64)
        weights[g] = 1.0 / max(float(len(g)), 1.0)
    positive = weights[weights > 0.0]
    if positive.size:
        weights *= float(positive.size) / max(float(weights.sum()), 1e-12)
    return np.clip(weights, 0.0, 100.0).astype(np.float32, copy=False)


def _failure_target_broad(
    *,
    y: np.ndarray,
    rank: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str]:
    candidate = np.isfinite(rank) & (rank >= float(args.failure_candidate_rank_threshold)) & (np.asarray(y) >= 0)
    target = np.full(len(y), np.nan, dtype=np.float32)
    target[candidate] = (np.asarray(y, dtype=np.int8)[candidate] == 0).astype(np.float32)
    return candidate, target, "broad_y0_given_rank_ge_50"


def _failure_target_swap(
    *,
    y: np.ndarray,
    rank: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str]:
    y_arr = np.asarray(y, dtype=np.int8)
    positive = np.isfinite(rank) & (rank >= float(args.swap_top_rank_threshold)) & (y_arr == 0)
    negative = (
        np.isfinite(rank)
        & (rank >= float(args.swap_lower_rank_threshold))
        & (rank < float(args.swap_top_rank_threshold))
        & (y_arr == 1)
    )
    candidate = positive | negative
    target = np.full(len(y), np.nan, dtype=np.float32)
    target[positive] = 1.0
    target[negative] = 0.0
    return candidate, target, "swap_critical_top30_false_positive_vs_50_70_true_positive"


def _make_classifier(
    *,
    seed: int,
    max_depth: int,
    n_estimators: int,
    min_child_samples: int,
) -> Any:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        max_depth=int(max_depth),
        num_leaves=max(4, min(24, 2 ** int(max_depth))),
        min_child_samples=int(min_child_samples),
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _fit_predict_binary(
    x_train: pd.DataFrame,
    target: np.ndarray,
    train_mask: np.ndarray,
    x_pred: pd.DataFrame,
    *,
    timestamps_train: pd.Series,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = np.asarray(train_mask, dtype=bool) & np.isfinite(target)
    train_ids = np.flatnonzero(train_mask)
    if train_ids.size < int(args.failure_min_train_rows) or len(np.unique(target[train_ids].astype(np.int8))) < 2:
        p = float(np.nanmean(target[train_ids])) if train_ids.size else 0.5
        return (
            np.full(len(x_train), p, dtype=np.float32),
            np.full(len(x_pred), p, dtype=np.float32),
            {
                "reason": "constant_insufficient_rows_or_classes",
                "train_rows": int(train_ids.size),
                "feature_count": 0,
            },
        )
    keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        p = float(np.nanmean(target[train_ids]))
        return (
            np.full(len(x_train), p, dtype=np.float32),
            np.full(len(x_pred), p, dtype=np.float32),
            {"reason": "constant_empty_matrix", "train_rows": int(train_ids.size), "feature_count": 0},
        )
    x_all = pd.concat([x_train, x_pred], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
    prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
    x_tr = prepared.iloc[: len(x_train)]
    x_va = prepared.iloc[len(x_train) :]
    weights = _equal_timestamp_weights(timestamps_train, train_mask)
    min_child = max(25, int(math.ceil(float(args.failure_min_child_fraction) * len(train_ids))))
    clf = _make_classifier(
        seed=seed,
        max_depth=int(args.failure_max_depth),
        n_estimators=int(args.failure_n_estimators),
        min_child_samples=min_child,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(
            x_tr.iloc[train_ids],
            target[train_ids].astype(np.int8),
            sample_weight=weights[train_ids],
        )
    pred_train = clf.predict_proba(x_tr)[:, 1].astype(np.float32, copy=False)
    pred_valid = clf.predict_proba(x_va)[:, 1].astype(np.float32, copy=False)
    return pred_train, pred_valid, {
        "reason": "",
        "train_rows": int(train_ids.size),
        "feature_count": int(len(keep_cols)),
        "min_child_samples": int(min_child),
        "timestamp_balanced_weight_mean": float(np.nanmean(weights[train_ids])),
        "timestamp_balanced_weight_max": float(np.nanmax(weights[train_ids])),
    }


def _robust_standardized(train_score: np.ndarray, score: np.ndarray) -> np.ndarray:
    base = np.asarray(train_score, dtype=np.float64)
    finite = base[np.isfinite(base)]
    if finite.size < 20:
        med = float(np.nanmedian(base)) if np.isfinite(base).any() else 0.0
        scale = float(np.nanstd(base)) if np.isfinite(base).any() else 1.0
    else:
        med = float(np.nanmedian(finite))
        q25, q75 = np.nanquantile(finite, [0.25, 0.75])
        scale = float(q75 - q25)
    scale = scale if np.isfinite(scale) and scale > 1e-6 else 1.0
    return ((np.asarray(score, dtype=np.float64) - med) / scale).astype(np.float32, copy=False)


def _crossfit_classifier_increment(
    *,
    controls_train: pd.DataFrame,
    full_train: pd.DataFrame,
    controls_valid: pd.DataFrame,
    full_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    target: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    q_ctrl_train = np.full(len(target), np.nan, dtype=np.float32)
    q_full_train = np.full(len(target), np.nan, dtype=np.float32)
    inner_folds = canon._make_chrono_folds(train_timestamps.reset_index(drop=True), int(args.inner_folds), embargo_hours=int(args.inner_embargo_hours))
    for inner in inner_folds:
        tr = np.asarray(inner.train_idx, dtype=np.int64)
        va = np.asarray(inner.valid_idx, dtype=np.int64)
        _ctrl_tr_pred, ctrl_va_pred, _ = _fit_predict_binary(
            controls_train.iloc[tr].reset_index(drop=True),
            target[tr],
            candidate[tr],
            controls_train.iloc[va].reset_index(drop=True),
            timestamps_train=train_timestamps.iloc[tr].reset_index(drop=True),
            seed=int(seed + 101 * inner.fold_id),
            args=args,
        )
        _full_tr_pred, full_va_pred, _ = _fit_predict_binary(
            full_train.iloc[tr].reset_index(drop=True),
            target[tr],
            candidate[tr],
            full_train.iloc[va].reset_index(drop=True),
            timestamps_train=train_timestamps.iloc[tr].reset_index(drop=True),
            seed=int(seed + 211 * inner.fold_id),
            args=args,
        )
        q_ctrl_train[va] = ctrl_va_pred
        q_full_train[va] = full_va_pred
    default = float(np.nanmean(target[np.asarray(candidate, dtype=bool)])) if np.asarray(candidate, dtype=bool).any() else 0.5
    q_ctrl_train[~np.isfinite(q_ctrl_train)] = default
    q_full_train[~np.isfinite(q_full_train)] = default
    _ctrl_fit_train, q_ctrl_valid, ctrl_diag = _fit_predict_binary(
        controls_train,
        target,
        candidate,
        controls_valid,
        timestamps_train=train_timestamps,
        seed=int(seed + 991),
        args=args,
    )
    _full_fit_train, q_full_valid, full_diag = _fit_predict_binary(
        full_train,
        target,
        candidate,
        full_valid,
        timestamps_train=train_timestamps,
        seed=int(seed + 1991),
        args=args,
    )
    inc_train = anchored._safe_logit(q_full_train) - anchored._safe_logit(q_ctrl_train)
    inc_valid = anchored._safe_logit(q_full_valid) - anchored._safe_logit(q_ctrl_valid)
    eval_mask = np.asarray(candidate, dtype=bool) & np.isfinite(target)
    ctrl_auc = stack._safe_auc(target[eval_mask].astype(np.int8), q_ctrl_train[eval_mask]) if eval_mask.any() else np.nan
    full_auc = stack._safe_auc(target[eval_mask].astype(np.int8), q_full_train[eval_mask]) if eval_mask.any() else np.nan
    return inc_train.astype(np.float32), inc_valid.astype(np.float32), {
        "target_rows": int(eval_mask.sum()),
        "target_positive_count": int(np.nansum(target[eval_mask] == 1.0)),
        "target_positive_rate": float(np.nanmean(target[eval_mask])) if eval_mask.any() else np.nan,
        "control_inner_auc": ctrl_auc,
        "full_inner_auc": full_auc,
        "increment_inner_auc_delta": float(full_auc - ctrl_auc) if np.isfinite(full_auc) and np.isfinite(ctrl_auc) else np.nan,
        "control_feature_count": int(ctrl_diag.get("feature_count", 0)),
        "full_feature_count": int(full_diag.get("feature_count", 0)),
        "control_reason": ctrl_diag.get("reason", ""),
        "full_reason": full_diag.get("reason", ""),
    }


def _fit_predict_ranker(
    x_train: pd.DataFrame,
    target: np.ndarray,
    candidate: np.ndarray,
    x_pred: pd.DataFrame,
    *,
    timestamps_train: pd.Series,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = np.asarray(candidate, dtype=bool) & np.isfinite(target)
    train_ids = np.flatnonzero(train_mask)
    if lgb is None or train_ids.size < int(args.failure_min_train_rows) or len(np.unique(target[train_ids].astype(np.int8))) < 2:
        return _fit_predict_binary(
            x_train,
            target,
            candidate,
            x_pred,
            timestamps_train=timestamps_train,
            seed=seed,
            args=args,
        )
    keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return _fit_predict_binary(
            x_train,
            target,
            candidate,
            x_pred,
            timestamps_train=timestamps_train,
            seed=seed,
            args=args,
        )
    x_all = pd.concat([x_train, x_pred], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
    prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
    x_tr = prepared.iloc[: len(x_train)]
    x_va = prepared.iloc[len(x_train) :]
    ts = pd.to_datetime(timestamps_train, utc=True, errors="coerce").reset_index(drop=True)
    order = np.lexsort((train_ids, ts.iloc[train_ids].astype("int64").to_numpy()))
    ordered_ids = train_ids[order]
    ordered_ts = ts.iloc[ordered_ids].to_numpy()
    groups = pd.Series(ordered_ts).groupby(pd.Series(ordered_ts), sort=False).size().to_numpy(dtype=np.int32)
    if len(groups) < 5:
        return _fit_predict_binary(
            x_train,
            target,
            candidate,
            x_pred,
            timestamps_train=timestamps_train,
            seed=seed,
            args=args,
        )
    weights = _equal_timestamp_weights(timestamps_train, train_mask)
    min_child = max(25, int(math.ceil(float(args.failure_min_child_fraction) * len(ordered_ids))))
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=int(args.failure_n_estimators),
        learning_rate=0.035,
        max_depth=int(args.failure_max_depth),
        num_leaves=max(4, min(24, 2 ** int(args.failure_max_depth))),
        min_child_samples=int(min_child),
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(
            x_tr.iloc[ordered_ids],
            target[ordered_ids].astype(np.float32),
            group=groups,
            sample_weight=weights[ordered_ids],
        )
    return (
        ranker.predict(x_tr).astype(np.float32, copy=False),
        ranker.predict(x_va).astype(np.float32, copy=False),
        {
            "reason": "",
            "ranker_groups": int(len(groups)),
            "train_rows": int(len(ordered_ids)),
            "feature_count": int(len(keep_cols)),
            "min_child_samples": int(min_child),
        },
    )


def _crossfit_ranker_increment(
    *,
    controls_train: pd.DataFrame,
    full_train: pd.DataFrame,
    controls_valid: pd.DataFrame,
    full_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    target: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ctrl_train = np.full(len(target), np.nan, dtype=np.float32)
    full_train_score = np.full(len(target), np.nan, dtype=np.float32)
    inner_folds = canon._make_chrono_folds(train_timestamps.reset_index(drop=True), int(args.inner_folds), embargo_hours=int(args.inner_embargo_hours))
    for inner in inner_folds:
        tr = np.asarray(inner.train_idx, dtype=np.int64)
        va = np.asarray(inner.valid_idx, dtype=np.int64)
        _c_tr, c_va, _ = _fit_predict_ranker(
            controls_train.iloc[tr].reset_index(drop=True),
            target[tr],
            candidate[tr],
            controls_train.iloc[va].reset_index(drop=True),
            timestamps_train=train_timestamps.iloc[tr].reset_index(drop=True),
            seed=int(seed + 303 * inner.fold_id),
            args=args,
        )
        _f_tr, f_va, _ = _fit_predict_ranker(
            full_train.iloc[tr].reset_index(drop=True),
            target[tr],
            candidate[tr],
            full_train.iloc[va].reset_index(drop=True),
            timestamps_train=train_timestamps.iloc[tr].reset_index(drop=True),
            seed=int(seed + 409 * inner.fold_id),
            args=args,
        )
        ctrl_train[va] = c_va
        full_train_score[va] = f_va
    ctrl_train[~np.isfinite(ctrl_train)] = 0.0
    full_train_score[~np.isfinite(full_train_score)] = 0.0
    ctrl_fit_train, ctrl_valid, ctrl_diag = _fit_predict_ranker(
        controls_train,
        target,
        candidate,
        controls_valid,
        timestamps_train=train_timestamps,
        seed=int(seed + 3911),
        args=args,
    )
    full_fit_train, full_valid_score, full_diag = _fit_predict_ranker(
        full_train,
        target,
        candidate,
        full_valid,
        timestamps_train=train_timestamps,
        seed=int(seed + 4909),
        args=args,
    )
    inc_train = _robust_standardized(full_train_score, full_train_score) - _robust_standardized(ctrl_train, ctrl_train)
    inc_valid = _robust_standardized(full_fit_train, full_valid_score) - _robust_standardized(ctrl_fit_train, ctrl_valid)
    eval_mask = np.asarray(candidate, dtype=bool) & np.isfinite(target)
    ctrl_auc = stack._safe_auc(target[eval_mask].astype(np.int8), ctrl_train[eval_mask]) if eval_mask.any() else np.nan
    full_auc = stack._safe_auc(target[eval_mask].astype(np.int8), full_train_score[eval_mask]) if eval_mask.any() else np.nan
    return inc_train.astype(np.float32), inc_valid.astype(np.float32), {
        "target_rows": int(eval_mask.sum()),
        "target_positive_count": int(np.nansum(target[eval_mask] == 1.0)),
        "target_positive_rate": float(np.nanmean(target[eval_mask])) if eval_mask.any() else np.nan,
        "control_inner_auc": ctrl_auc,
        "full_inner_auc": full_auc,
        "increment_inner_auc_delta": float(full_auc - ctrl_auc) if np.isfinite(full_auc) and np.isfinite(ctrl_auc) else np.nan,
        "control_feature_count": int(ctrl_diag.get("feature_count", 0)),
        "full_feature_count": int(full_diag.get("feature_count", 0)),
        "ranker_control_groups": int(ctrl_diag.get("ranker_groups", 0)),
        "ranker_full_groups": int(full_diag.get("ranker_groups", 0)),
    }


def _crossfit_rank_banded_increment(
    *,
    controls_train: pd.DataFrame,
    full_train: pd.DataFrame,
    controls_valid: pd.DataFrame,
    full_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    valid_timestamps: pd.Series,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    anchor_valid: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rank_train = stack._rank_pct_by_timestamp(train_timestamps, anchor_train)
    rank_valid = stack._rank_pct_by_timestamp(valid_timestamps, anchor_valid)
    inc_train = np.zeros(len(y_train), dtype=np.float32)
    inc_valid = np.zeros(len(anchor_valid), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.85), (0.85, 1.01)]
    for band_i, (lo, hi) in enumerate(bands):
        candidate = np.isfinite(rank_train) & (rank_train >= lo) & (rank_train < hi) & (np.asarray(y_train) >= 0)
        target = np.full(len(y_train), np.nan, dtype=np.float32)
        target[candidate] = (np.asarray(y_train, dtype=np.int8)[candidate] == 0).astype(np.float32)
        band_train, band_valid, diag = _crossfit_classifier_increment(
            controls_train=controls_train,
            full_train=full_train,
            controls_valid=controls_valid,
            full_valid=full_valid,
            train_timestamps=train_timestamps,
            target=target,
            candidate=candidate,
            seed=int(seed + 577 * (band_i + 1)),
            args=args,
        )
        train_mask = np.isfinite(rank_train) & (rank_train >= lo) & (rank_train < hi)
        valid_mask = np.isfinite(rank_valid) & (rank_valid >= lo) & (rank_valid < hi)
        inc_train[train_mask] = band_train[train_mask]
        inc_valid[valid_mask] = band_valid[valid_mask]
        rows.append({"band": f"{lo:.2f}_{hi:.2f}", **diag})
    return inc_train, inc_valid, {
        "target_rows": int(sum(row.get("target_rows", 0) for row in rows)),
        "target_positive_count": int(sum(row.get("target_positive_count", 0) for row in rows)),
        "target_positive_rate": float(np.nanmean([row.get("target_positive_rate", np.nan) for row in rows])),
        "control_inner_auc": float(np.nanmean([row.get("control_inner_auc", np.nan) for row in rows])),
        "full_inner_auc": float(np.nanmean([row.get("full_inner_auc", np.nan) for row in rows])),
        "increment_inner_auc_delta": float(np.nanmean([row.get("increment_inner_auc_delta", np.nan) for row in rows])),
        "rank_band_count": len(rows),
        "rank_band_details": json.dumps(rows, default=_json_default),
    }


def _qfail_variant_increment(
    *,
    variant: str,
    anchor_controls_train: pd.DataFrame,
    anchor_controls_valid: pd.DataFrame,
    path_controls_train: pd.DataFrame,
    path_controls_valid: pd.DataFrame,
    full_train: pd.DataFrame,
    full_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    valid_timestamps: pd.Series,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    anchor_valid: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rank_train = stack._rank_pct_by_timestamp(train_timestamps, anchor_train)
    if variant == "Q0_broad_path_control":
        candidate, target, target_name = _failure_target_broad(y=y_train, rank=rank_train, args=args)
        inc_train, inc_valid, diag = _crossfit_classifier_increment(
            controls_train=path_controls_train,
            full_train=full_train,
            controls_valid=path_controls_valid,
            full_valid=full_valid,
            train_timestamps=train_timestamps,
            target=target,
            candidate=candidate,
            seed=seed,
            args=args,
        )
    elif variant == "Q1_swap_anchor_control":
        candidate, target, target_name = _failure_target_swap(y=y_train, rank=rank_train, args=args)
        inc_train, inc_valid, diag = _crossfit_classifier_increment(
            controls_train=anchor_controls_train,
            full_train=full_train,
            controls_valid=anchor_controls_valid,
            full_valid=full_valid,
            train_timestamps=train_timestamps,
            target=target,
            candidate=candidate,
            seed=seed,
            args=args,
        )
    elif variant == "Q2_swap_path_control":
        candidate, target, target_name = _failure_target_swap(y=y_train, rank=rank_train, args=args)
        inc_train, inc_valid, diag = _crossfit_classifier_increment(
            controls_train=path_controls_train,
            full_train=full_train,
            controls_valid=path_controls_valid,
            full_valid=full_valid,
            train_timestamps=train_timestamps,
            target=target,
            candidate=candidate,
            seed=seed,
            args=args,
        )
    elif variant == "Q3_pairwise_ranker_swap_path":
        candidate, target, target_name = _failure_target_swap(y=y_train, rank=rank_train, args=args)
        inc_train, inc_valid, diag = _crossfit_ranker_increment(
            controls_train=path_controls_train,
            full_train=full_train,
            controls_valid=path_controls_valid,
            full_valid=full_valid,
            train_timestamps=train_timestamps,
            target=target,
            candidate=candidate,
            seed=seed,
            args=args,
        )
    elif variant == "Q4_rank_banded_path_control":
        target_name = "rank_banded_y0_given_rank_band"
        inc_train, inc_valid, diag = _crossfit_rank_banded_increment(
            controls_train=path_controls_train,
            full_train=full_train,
            controls_valid=path_controls_valid,
            full_valid=full_valid,
            train_timestamps=train_timestamps,
            valid_timestamps=valid_timestamps,
            y_train=y_train,
            anchor_train=anchor_train,
            anchor_valid=anchor_valid,
            seed=seed,
            args=args,
        )
    else:
        raise ValueError(f"Unknown qfail variant: {variant}")
    diag["qfail_variant"] = variant
    diag["qfail_target"] = target_name
    diag["timestamp_balanced_qfail_training"] = True
    return inc_train, inc_valid, diag


def _score_direct(anchor_valid: np.ndarray, qfail_valid: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    q = pd.to_numeric(qfail_valid.get("q_fail_inc"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return anchored._score_with_delta(anchor_valid, -float(args.r1_lambda_fail) * q, float(args.correction_clip))


def _write_report(out_dir: Path, summary: pd.DataFrame, directional: pd.DataFrame, fold_diag: pd.DataFrame) -> None:
    lines = [
        "# q_fail Model Ablation",
        "",
        "Compares q_fail target/control/ranker/rank-band variants under a single anchored meta score.",
        "",
    ]
    if not summary.empty:
        cols = [
            "head",
            "arm",
            "rows",
            "auc",
            "delta_auc",
            "log_loss",
            "delta_log_loss_improvement",
            "brier",
            "delta_brier_improvement",
        ]
        lines.extend(["## Global Metrics", "", summary[[c for c in cols if c in summary.columns]].to_markdown(index=False, floatfmt=".5f"), ""])
    if not directional.empty:
        cols = [
            "head",
            "arm",
            "timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top10",
            "timestamp_weighted_hr_top20",
            "delta_timestamp_weighted_hr_top20",
            "timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "ndcg_top30",
            "delta_ndcg_top30",
            "top30_entrant_hit_rate",
            "top30_removed_hit_rate",
            "net_correct_trades_gained",
        ]
        lines.extend(["## Directional Metrics", "", directional[[c for c in cols if c in directional.columns]].to_markdown(index=False, floatfmt=".5f"), ""])
    if not fold_diag.empty:
        lines.extend(["## q_fail Diagnostics", "", fold_diag.head(80).to_markdown(index=False, floatfmt=".5f"), ""])
    (out_dir / "qfail_model_ablation_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    out_dir = _ensure_dir(Path(args.output_dir))
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    feature_dir = Path(args.feature_dir)
    report_dir = Path(args.report_dir)
    transform_cache = Path(args.transform_cache) if str(args.transform_cache).strip() else None
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
    directional_rows: list[dict[str, Any]] = []
    directional_timestamp_frames: list[pd.DataFrame] = []
    directional_episode_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []

    for head in heads:
        print(f"[qfail_ablation] head={head.head}", flush=True)
        panel = _downcast_numeric(_normalise_keys(pd.read_parquet(head.meta_oof_path)), exclude=["timestamp", "symbol"])
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
            print(f"[qfail_ablation] sampled head={head.head} rows={len(panel)}", flush=True)
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
        base_x = stack._assemble_base_selected_matrix(
            head=head,
            panel=panel,
            base_bundle=base_bundle,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
        y = ctx._meta_target(panel)
        anchor = ctx._current_meta_score(panel)
        returns = np.asarray(_pick_realized_return(panel), dtype=np.float32)
        folds = canon._make_chrono_folds(panel["timestamp"], int(args.outer_folds), embargo_hours=int(args.embargo_hours))
        bad_episodes, _bad_meta = ctx._load_episode_registry(args.episode_registry, head=head.head, target_name="y_bin")
        fold_valid_mask = np.zeros(len(panel), dtype=bool)
        preds: dict[str, np.ndarray] = {TRIAL_ANCHOR: np.full(len(panel), np.nan, dtype=np.float32)}
        for variant in QFAIL_VARIANTS:
            preds[f"{variant}__direct_R1"] = np.full(len(panel), np.nan, dtype=np.float32)
            preds[f"{variant}__meta1"] = np.full(len(panel), np.nan, dtype=np.float32)
        path_cols: list[str] | None = None
        q_store: list[pd.DataFrame] = []

        for fold in folds:
            tr = np.asarray(fold.train_idx, dtype=np.int64)
            va = np.asarray(fold.valid_idx, dtype=np.int64)
            fold_valid_mask[va] = True
            ts_train = panel["timestamp"].iloc[tr].reset_index(drop=True)
            ts_valid = panel["timestamp"].iloc[va].reset_index(drop=True)
            raw_train = raw.iloc[tr].reset_index(drop=True)
            raw_valid = raw.iloc[va].reset_index(drop=True)
            anchor_controls_train, _ = anchored._anchor_control_features(
                raw_train,
                ts_train,
                anchor[tr],
                max_path_cols=0,
                selected_cols=[],
            )
            anchor_controls_valid, _ = anchored._anchor_control_features(
                raw_valid,
                ts_valid,
                anchor[va],
                max_path_cols=0,
                selected_cols=[],
            )
            path_controls_train, path_cols = anchored._anchor_control_features(
                raw_train,
                ts_train,
                anchor[tr],
                max_path_cols=int(args.max_control_path_features),
                selected_cols=path_cols,
            )
            path_controls_valid, _ = anchored._anchor_control_features(
                raw_valid,
                ts_valid,
                anchor[va],
                max_path_cols=int(args.max_control_path_features),
                selected_cols=path_cols,
            )
            canonical_train, canonical_valid, canonical_diag = stack._canonical_fold_frames(
                raw,
                fold,
                canonical_defs,
                trailing_window=int(args.trailing_window),
                min_periods=int(args.min_periods),
                min_resolved_features=int(args.min_resolved_features),
            )
            meta_models_list = list(getattr(getattr(race, "best_model", None), "models", []) or [])
            meta_leaf_train, meta_leaf_valid, meta_leaf_diag = stack._leaf_structural_fold_features(
                models=meta_models_list,
                x_train=current_x.iloc[tr].reset_index(drop=True),
                x_valid=current_x.iloc[va].reset_index(drop=True),
                prefix="anchor_meta_leaf",
                max_models=int(args.leaf_max_models),
                tree_stride=int(args.leaf_tree_stride),
                max_trees=int(args.leaf_max_trees),
            )
            base_models, _base_features = _base_models_for_head(base_bundle, head)
            base_leaf_train, base_leaf_valid, base_leaf_diag = stack._leaf_structural_fold_features(
                models=base_models,
                x_train=base_x.iloc[tr].reset_index(drop=True),
                x_valid=base_x.iloc[va].reset_index(drop=True),
                prefix="base_leaf",
                max_models=int(args.leaf_max_models),
                tree_stride=int(args.leaf_tree_stride),
                max_trees=int(args.leaf_max_trees),
            )
            leaf_train = stack._combine_features(base_leaf_train, meta_leaf_train)
            leaf_valid = stack._combine_features(base_leaf_valid, meta_leaf_valid)
            full_train = stack._combine_features(path_controls_train, canonical_train, leaf_train)
            full_valid = stack._combine_features(path_controls_valid, canonical_valid, leaf_valid)
            preds[TRIAL_ANCHOR][va] = anchor[va]

            for variant_i, variant in enumerate(QFAIL_VARIANTS, start=1):
                inc_train, inc_valid, qdiag = _qfail_variant_increment(
                    variant=variant,
                    anchor_controls_train=anchor_controls_train,
                    anchor_controls_valid=anchor_controls_valid,
                    path_controls_train=path_controls_train,
                    path_controls_valid=path_controls_valid,
                    full_train=full_train,
                    full_valid=full_valid,
                    train_timestamps=ts_train,
                    valid_timestamps=ts_valid,
                    y_train=y[tr],
                    anchor_train=anchor[tr],
                    anchor_valid=anchor[va],
                    seed=int(args.seed + 1009 * fold.fold_id + 97 * variant_i),
                    args=args,
                )
                q_train = anchored._rank_features_from_values(ts_train, inc_train, "q_fail_inc")
                q_valid = anchored._rank_features_from_values(ts_valid, inc_valid, "q_fail_inc")
                direct_arm = f"{variant}__direct_R1"
                meta_arm = f"{variant}__meta1"
                preds[direct_arm][va] = _score_direct(anchor[va], q_valid, args)
                anchor_train_features = path_controls_train.loc[:, ["anchor_p0", "anchor_logit0", "anchor_rank0_by_timestamp"]]
                anchor_valid_features = path_controls_valid.loc[:, ["anchor_p0", "anchor_logit0", "anchor_rank0_by_timestamp"]]
                correction_train = stack._combine_features(anchor_train_features, q_train, canonical_train, leaf_train)
                correction_valid = stack._combine_features(anchor_valid_features, q_valid, canonical_valid, leaf_valid)
                meta_pred, meta_diag = anchored._fit_correction_model(
                    x_train=correction_train,
                    x_valid=correction_valid,
                    y_train=y[tr],
                    anchor_train=anchor[tr],
                    anchor_valid=anchor[va],
                    timestamps_train=ts_train,
                    seed=int(args.seed + 2003 * fold.fold_id + 131 * variant_i),
                    args=args,
                )
                preds[meta_arm][va] = meta_pred
                q_store.append(
                    pd.DataFrame(
                        {
                            "row_id": va,
                            f"{variant}__q_fail_inc": pd.to_numeric(q_valid.get("q_fail_inc"), errors="coerce").to_numpy(dtype=np.float32),
                        }
                    )
                )
                fold_rows.append(
                    {
                        "head": head.head,
                        "fold": int(fold.fold_id),
                        "variant": variant,
                        "direct_arm": direct_arm,
                        "meta_arm": meta_arm,
                        "train_rows": int(len(tr)),
                        "valid_rows": int(len(va)),
                        "path_control_feature_count": int(len(path_cols or [])),
                        **canonical_diag,
                        **meta_leaf_diag,
                        **base_leaf_diag,
                        **qdiag,
                        **{f"meta1_{k}": v for k, v in meta_diag.items()},
                    }
                )
            print(f"[qfail_ablation] head={head.head} fold={fold.fold_id}/{len(folds)} train={len(tr)} valid={len(va)}", flush=True)

        baseline_fold = anchor.copy()
        baseline_fold[~fold_valid_mask] = np.nan
        for arm, pred in preds.items():
            variant = "anchor_reference" if arm == TRIAL_ANCHOR else "qfail_ablation"
            summary = ctx._overall_metrics(
                head=head.head,
                arm=arm,
                variant=variant,
                y=y,
                pred=pred,
                baseline_pred=baseline_fold,
                returns=returns,
            )
            summary.update(
                {
                    "training_target": "y_bin",
                    "single_output_score": True,
                    "forbidden_targets_used": False,
                    "qfail_ablation": arm != TRIAL_ANCHOR,
                }
            )
            summary_rows.append(summary)
            ts_metrics = ctx._directional_timestamp_metrics(
                head=head.head,
                arm=arm,
                variant=variant,
                panel=panel,
                y=y,
                pred=pred,
                baseline_pred=baseline_fold,
                returns=returns,
                bad_episodes=bad_episodes,
                rank_threshold=float(args.rank_threshold),
                min_timestamp_rows=int(args.directional_min_timestamp_rows),
            )
            if not ts_metrics.empty:
                directional_timestamp_frames.append(ts_metrics)
                agg = ctx._directional_aggregate(ts_metrics)
                if not agg.empty:
                    directional_rows.extend(agg.to_dict(orient="records"))
                ep = ctx._directional_episode_metrics(ts_metrics, bad_episodes)
                if not ep.empty:
                    directional_episode_frames.append(ep)
        score_frame = pd.DataFrame(
            {
                "head": head.head,
                "row_id": np.arange(len(panel), dtype=np.int64),
                "timestamp": pd.to_datetime(panel["timestamp"], utc=True, errors="coerce"),
                "symbol": panel["symbol"].astype(str) if "symbol" in panel.columns else "",
                "y_bin": y,
                "anchor_score": baseline_fold,
            }
        )
        for arm, pred in preds.items():
            score_frame[arm] = pred
        if q_store:
            q_all = pd.concat(q_store, axis=0, ignore_index=True)
            q_all = q_all.groupby("row_id", as_index=False).first()
            score_frame = score_frame.merge(q_all, on="row_id", how="left")
        score_frames.append(score_frame)

    summary = pd.DataFrame(summary_rows)
    directional = pd.DataFrame(directional_rows) if directional_rows else pd.DataFrame(columns=anchored.DIRECTIONAL_SUMMARY_COLUMNS)
    directional_timestamp = pd.concat(directional_timestamp_frames, axis=0, ignore_index=True) if directional_timestamp_frames else pd.DataFrame(columns=anchored.DIRECTIONAL_TIMESTAMP_COLUMNS)
    directional_episode = pd.concat(directional_episode_frames, axis=0, ignore_index=True) if directional_episode_frames else pd.DataFrame()
    fold_diag = pd.DataFrame(fold_rows)
    scores = pd.concat(score_frames, axis=0, ignore_index=True) if score_frames else pd.DataFrame()

    summary.to_csv(out_dir / "qfail_model_ablation_summary.csv", index=False)
    directional.to_csv(out_dir / "qfail_model_ablation_directional_metrics.csv", index=False)
    directional_timestamp.to_csv(out_dir / "qfail_model_ablation_directional_timestamp_metrics.csv", index=False)
    directional_episode.to_csv(out_dir / "qfail_model_ablation_directional_episode_metrics.csv", index=False)
    fold_diag.to_csv(out_dir / "qfail_model_ablation_fold_diagnostics.csv", index=False)
    if not scores.empty:
        scores.to_parquet(out_dir / "qfail_model_ablation_oof_scores.parquet", index=False)
    manifest = {
        "output_dir": str(out_dir),
        "heads": [h.head for h in heads],
        "variants": list(QFAIL_VARIANTS),
        "contract": {
            "target": "y_bin",
            "single_output_score": True,
            "anchor": "current_meta_oof_score",
            "implemented_suggestions": [
                "swap_critical_target",
                "pairwise_lambdarank_swap_variant",
                "anchor_only_vs_path_control_increment",
                "timestamp_balanced_qfail_training",
                "anchor_rank_banded_failure_models",
            ],
        },
        "args": vars(args),
    }
    (out_dir / "qfail_model_ablation_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    _write_report(out_dir, summary, directional, fold_diag)
    print(f"[qfail_ablation] wrote {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--episode-registry", default=str(ctx.DEFAULT_EPISODE_REGISTRY))
    parser.add_argument("--transform-cache", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/qfail_model_ablation_20260623")
    parser.add_argument("--only-head", nargs="*", default=list(HEADS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--inner-embargo-hours", type=int, default=12)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-control-path-features", type=int, default=40)
    parser.add_argument("--failure-candidate-rank-threshold", type=float, default=0.50)
    parser.add_argument("--swap-lower-rank-threshold", type=float, default=0.50)
    parser.add_argument("--swap-top-rank-threshold", type=float, default=0.70)
    parser.add_argument("--failure-min-train-rows", type=int, default=200)
    parser.add_argument("--failure-max-depth", type=int, default=3)
    parser.add_argument("--failure-n-estimators", type=int, default=180)
    parser.add_argument("--failure-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--leaf-max-models", type=int, default=1)
    parser.add_argument("--leaf-tree-stride", type=int, default=3)
    parser.add_argument("--leaf-max-trees", type=int, default=120)
    parser.add_argument("--r1-lambda-fail", type=float, default=0.25)
    parser.add_argument("--correction-clip", type=float, default=0.50)
    parser.add_argument("--correction-max-depth", type=int, default=2)
    parser.add_argument("--correction-n-estimators", type=int, default=140)
    parser.add_argument("--correction-learning-rate", type=float, default=0.035)
    parser.add_argument("--correction-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--directional-min-timestamp-rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
