#!/usr/bin/env python3
"""Fixed-band q_fail and replacement-quality ablation.

This tests the non-circular q_fail formulation:

* F1: failure model inside anchor top 30%;
* F2: replacement-quality model inside the 30-50% anchor band;
* F3: explicit two-model swap rule;
* F4: top-50 residual correctness model with anchor logit as fixed offset;
* F5: F4 plus explicit difficult-period interactions, skipped for short_boll.

The comparison metrics are computed on the anchor top-50 pool.  Baseline top30
means rows with ``rank0 >= 0.70``; candidate top30 means the same timestamp-level
count selected from rows with ``rank0 >= 0.50``.  This lets 30-50% replacement
rows actually enter the selected set, unlike the previous high-rank-only metric.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

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

F0 = "F0_anchor"
F1_CONTROL = "F1_fail30_control_demote"
F1_FULL = "F1_fail30_full_demote"
F1_INC = "F1_fail30_incremental_demote"
F2_CONTROL = "F2_replace_control_promote"
F2_FULL = "F2_replace_full_promote"
F2_INC = "F2_replace_incremental_promote"
F3_CONTROL = "F3_two_model_swap_control"
F3_FULL = "F3_two_model_swap_full"
F4_CONTROL = "F4_top50_residual_controls_only"
F4_QFAIL_INC = "F4_top50_residual_controls_plus_qfail_inc"
F4_STRUCT = "F4_top50_residual_structural"
F5_PERIOD = "F5_top50_residual_period_conditioned"

TRIALS = (
    F0,
    F1_CONTROL,
    F1_FULL,
    F1_INC,
    F2_CONTROL,
    F2_FULL,
    F2_INC,
    F3_CONTROL,
    F3_FULL,
    F4_CONTROL,
    F4_QFAIL_INC,
    F4_STRUCT,
    F5_PERIOD,
)


def _json_default(value: Any) -> Any:
    return ctx._json_default(value)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(y)
    if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(roc_auc_score(y[mask], s[mask]))


def _safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(y)
    if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(average_precision_score(y[mask], s[mask]))


def _safe_logloss(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    s = np.clip(np.asarray(score, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    mask = np.isfinite(s) & np.isfinite(y)
    if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(log_loss(y[mask].astype(int), s[mask], labels=[0, 1]))


def _frame_hash(frame: pd.DataFrame, *, max_rows: int = 20000) -> str:
    if frame is None or frame.empty:
        return "empty"
    f = frame.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    if len(f) > max_rows:
        ids = np.linspace(0, len(f) - 1, max_rows).round().astype(int)
        f = f.iloc[np.unique(ids)].reset_index(drop=True)
    h = hashlib.sha256()
    h.update("\x1f".join(map(str, f.columns)).encode())
    row_hash = pd.util.hash_pandas_object(f, index=False).to_numpy(dtype=np.uint64)
    h.update(row_hash.tobytes())
    return h.hexdigest()[:16]


def _matrix_feature_importance(model: Any, control_cols: set[str]) -> tuple[int, float]:
    if model is None or not hasattr(model, "feature_importances_"):
        return 0, 0.0
    names = list(getattr(model, "feature_name_", []) or [])
    imp = np.asarray(getattr(model, "feature_importances_", []), dtype=np.float64)
    if not names or imp.size != len(names):
        return 0, 0.0
    non_control = np.asarray([str(name) not in control_cols for name in names], dtype=bool)
    used = non_control & (imp > 0)
    total_gain = float(np.sum(imp))
    share = float(np.sum(imp[used]) / total_gain) if total_gain > 0 else 0.0
    return int(np.sum(used)), share


def _rank0(panel: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    if "oof_rank_pct" in panel.columns:
        rank = pd.to_numeric(panel["oof_rank_pct"], errors="coerce").to_numpy(dtype=np.float32)
        if np.isfinite(rank).mean() > 0.5:
            return np.clip(rank, 0.0, 1.0)
    return stack._rank_pct_by_timestamp(panel["timestamp"], score)


def _band_features(rank: np.ndarray) -> pd.DataFrame:
    r = np.asarray(rank, dtype=np.float32)
    return pd.DataFrame(
        {
            "rank_band_50_60": ((r >= 0.50) & (r < 0.60)).astype(np.float32),
            "rank_band_60_70": ((r >= 0.60) & (r < 0.70)).astype(np.float32),
            "rank_band_70_85": ((r >= 0.70) & (r < 0.85)).astype(np.float32),
            "rank_band_85_100": (r >= 0.85).astype(np.float32),
            "rank0_centered": (r - 0.70).astype(np.float32),
        }
    )


def _equal_timestamp_weights(timestamps: pd.Series, train_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(train_mask, dtype=bool)
    weight = np.zeros(len(mask), dtype=np.float32)
    if not mask.any():
        return weight
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    ids = np.flatnonzero(mask)
    frame = pd.DataFrame({"idx": ids, "timestamp": ts.iloc[ids].to_numpy()})
    for _, group in frame.groupby("timestamp", sort=False):
        g = group["idx"].to_numpy(dtype=np.int64)
        weight[g] = 1.0 / max(float(len(g)), 1.0)
    total = float(weight.sum())
    if total > 0:
        weight *= float(mask.sum()) / total
    return np.clip(weight, 0.0, 100.0).astype(np.float32, copy=False)


def _prepare_train_pred_matrix(x_train: pd.DataFrame, x_valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return pd.DataFrame(index=x_train.index), pd.DataFrame(index=x_valid.index), []
    x_all = pd.concat([x_train, x_valid], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
    prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
    return prepared.iloc[: len(x_train)].reset_index(drop=True), prepared.iloc[len(x_train) :].reset_index(drop=True), keep_cols


def _fit_classifier_predict(
    *,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    x_valid: pd.DataFrame,
    timestamps_train: pd.Series,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, Any | None, dict[str, Any]]:
    mask = np.asarray(mask_train, dtype=bool) & np.isfinite(y_train)
    ids = np.flatnonzero(mask)
    if ids.size < int(args.min_train_rows) or len(np.unique(y_train[ids].astype(np.int8))) < 2:
        p = float(np.nanmean(y_train[ids])) if ids.size else 0.5
        return (
            np.full(len(x_train), p, dtype=np.float32),
            np.full(len(x_valid), p, dtype=np.float32),
            None,
            {"reason": "constant_insufficient_rows_or_classes", "train_rows": int(ids.size), "feature_count": 0},
        )
    x_tr, x_va, keep_cols = _prepare_train_pred_matrix(x_train, x_valid)
    if not keep_cols:
        p = float(np.nanmean(y_train[ids]))
        return (
            np.full(len(x_train), p, dtype=np.float32),
            np.full(len(x_valid), p, dtype=np.float32),
            None,
            {"reason": "constant_empty_matrix", "train_rows": int(ids.size), "feature_count": 0},
        )
    weight = _equal_timestamp_weights(timestamps_train, mask)
    min_child = max(25, int(math.ceil(float(args.min_child_fraction) * len(ids))))
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(args.n_estimators),
        learning_rate=0.035,
        max_depth=int(args.max_depth),
        num_leaves=max(4, min(24, 2 ** int(args.max_depth))),
        min_child_samples=min_child,
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
        clf.fit(x_tr.iloc[ids], y_train[ids].astype(np.int8), sample_weight=weight[ids])
    return (
        clf.predict_proba(x_tr)[:, 1].astype(np.float32, copy=False),
        clf.predict_proba(x_va)[:, 1].astype(np.float32, copy=False),
        clf,
        {
            "reason": "",
            "train_rows": int(ids.size),
            "feature_count": int(len(keep_cols)),
            "min_child_samples": int(min_child),
            "weight_mean": float(np.nanmean(weight[ids])),
            "weight_max": float(np.nanmax(weight[ids])),
        },
    )


def _crossfit_two_models(
    *,
    control_train: pd.DataFrame,
    full_train: pd.DataFrame,
    control_valid: pd.DataFrame,
    full_valid: pd.DataFrame,
    target_train: np.ndarray,
    mask_train: np.ndarray,
    timestamps_train: pd.Series,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ctrl_oof = np.full(len(target_train), np.nan, dtype=np.float32)
    full_oof = np.full(len(target_train), np.nan, dtype=np.float32)
    inner = canon._make_chrono_folds(timestamps_train.reset_index(drop=True), int(args.inner_folds), embargo_hours=int(args.inner_embargo_hours))
    for fold in inner:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        _ct, cv, _cm, _cd = _fit_classifier_predict(
            x_train=control_train.iloc[tr].reset_index(drop=True),
            y_train=target_train[tr],
            mask_train=mask_train[tr],
            x_valid=control_train.iloc[va].reset_index(drop=True),
            timestamps_train=timestamps_train.iloc[tr].reset_index(drop=True),
            seed=int(seed + 101 * fold.fold_id),
            args=args,
        )
        _ft, fv, _fm, _fd = _fit_classifier_predict(
            x_train=full_train.iloc[tr].reset_index(drop=True),
            y_train=target_train[tr],
            mask_train=mask_train[tr],
            x_valid=full_train.iloc[va].reset_index(drop=True),
            timestamps_train=timestamps_train.iloc[tr].reset_index(drop=True),
            seed=int(seed + 211 * fold.fold_id),
            args=args,
        )
        ctrl_oof[va] = cv
        full_oof[va] = fv
    default = float(np.nanmean(target_train[mask_train])) if np.asarray(mask_train, dtype=bool).any() else 0.5
    ctrl_oof[~np.isfinite(ctrl_oof)] = default
    full_oof[~np.isfinite(full_oof)] = default
    ctrl_train_pred, ctrl_valid_pred, ctrl_model, ctrl_diag = _fit_classifier_predict(
        x_train=control_train,
        y_train=target_train,
        mask_train=mask_train,
        x_valid=control_valid,
        timestamps_train=timestamps_train,
        seed=int(seed + 991),
        args=args,
    )
    full_train_pred, full_valid_pred, full_model, full_diag = _fit_classifier_predict(
        x_train=full_train,
        y_train=target_train,
        mask_train=mask_train,
        x_valid=full_valid,
        timestamps_train=timestamps_train,
        seed=int(seed + 1991),
        args=args,
    )
    del ctrl_train_pred, full_train_pred
    eval_mask = np.asarray(mask_train, dtype=bool) & np.isfinite(target_train)
    corr = float(np.corrcoef(ctrl_oof[eval_mask], full_oof[eval_mask])[0, 1]) if int(eval_mask.sum()) > 5 else np.nan
    inc = anchored._safe_logit(full_oof) - anchored._safe_logit(ctrl_oof)
    non_control_count, non_control_share = _matrix_feature_importance(full_model, set(control_train.columns))
    diag = {
        "control_hash": _frame_hash(control_train),
        "full_hash": _frame_hash(full_train),
        "matrix_hashes_differ": _frame_hash(control_train) != _frame_hash(full_train),
        "pred_corr_full_control": corr,
        "inc_std": float(np.nanstd(inc[eval_mask])) if eval_mask.any() else np.nan,
        "inc_q05": float(np.nanquantile(inc[eval_mask], 0.05)) if eval_mask.any() else np.nan,
        "inc_q50": float(np.nanquantile(inc[eval_mask], 0.50)) if eval_mask.any() else np.nan,
        "inc_q95": float(np.nanquantile(inc[eval_mask], 0.95)) if eval_mask.any() else np.nan,
        "control_auc": _safe_auc(target_train[eval_mask], ctrl_oof[eval_mask]) if eval_mask.any() else np.nan,
        "full_auc": _safe_auc(target_train[eval_mask], full_oof[eval_mask]) if eval_mask.any() else np.nan,
        "control_pr_auc": _safe_pr_auc(target_train[eval_mask], ctrl_oof[eval_mask]) if eval_mask.any() else np.nan,
        "full_pr_auc": _safe_pr_auc(target_train[eval_mask], full_oof[eval_mask]) if eval_mask.any() else np.nan,
        "control_logloss": _safe_logloss(target_train[eval_mask], ctrl_oof[eval_mask]) if eval_mask.any() else np.nan,
        "full_logloss": _safe_logloss(target_train[eval_mask], full_oof[eval_mask]) if eval_mask.any() else np.nan,
        "incremental_auc": np.nan,
        "incremental_pr_auc": np.nan,
        "incremental_logloss_improvement": np.nan,
        "non_control_features_used": int(non_control_count),
        "non_control_importance_share": float(non_control_share),
        "target_rows": int(eval_mask.sum()),
        "target_positive_rate": float(np.nanmean(target_train[eval_mask])) if eval_mask.any() else np.nan,
        **{f"control_{k}": v for k, v in ctrl_diag.items()},
        **{f"full_{k}": v for k, v in full_diag.items()},
    }
    if np.isfinite(diag["full_auc"]) and np.isfinite(diag["control_auc"]):
        diag["incremental_auc"] = float(diag["full_auc"] - diag["control_auc"])
    if np.isfinite(diag["full_pr_auc"]) and np.isfinite(diag["control_pr_auc"]):
        diag["incremental_pr_auc"] = float(diag["full_pr_auc"] - diag["control_pr_auc"])
    if np.isfinite(diag["full_logloss"]) and np.isfinite(diag["control_logloss"]):
        diag["incremental_logloss_improvement"] = float(diag["control_logloss"] - diag["full_logloss"])
    return (
        {
            "control_oof": ctrl_oof,
            "full_oof": full_oof,
            "control_valid": ctrl_valid_pred,
            "full_valid": full_valid_pred,
        },
        diag,
    )


def _apply_demote(anchor_score: np.ndarray, q_fail: np.ndarray, rank: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    active = np.asarray(rank, dtype=np.float32) >= 0.70
    delta = np.zeros(len(anchor_score), dtype=np.float32)
    delta[active] = -float(args.fail_lambda) * (np.asarray(q_fail, dtype=np.float32)[active] - 0.5)
    return anchored._score_with_delta(anchor_score, delta, float(args.correction_clip))


def _apply_promote(anchor_score: np.ndarray, q_replace: np.ndarray, rank: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    active = (np.asarray(rank, dtype=np.float32) >= 0.50) & (np.asarray(rank, dtype=np.float32) < 0.70)
    delta = np.zeros(len(anchor_score), dtype=np.float32)
    delta[active] = float(args.replace_lambda) * (np.asarray(q_replace, dtype=np.float32)[active] - 0.5)
    return anchored._score_with_delta(anchor_score, delta, float(args.correction_clip))


def _apply_swap_rule(
    *,
    timestamps: pd.Series,
    anchor_score: np.ndarray,
    rank: np.ndarray,
    q_fail: np.ndarray,
    q_replace: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    z = anchored._safe_logit(anchor_score).astype(np.float32, copy=True)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    frame = pd.DataFrame({"timestamp": ts, "rank": rank})
    for _, idx in frame.groupby("timestamp", sort=False).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        top = ids[np.isfinite(rank[ids]) & (rank[ids] >= 0.70)]
        repl = ids[np.isfinite(rank[ids]) & (rank[ids] >= 0.50) & (rank[ids] < 0.70)]
        if top.size == 0 or repl.size == 0:
            continue
        keep_prob = 1.0 - np.asarray(q_fail, dtype=np.float32)[top]
        repl_prob = np.asarray(q_replace, dtype=np.float32)[repl]
        top_order = top[np.argsort(keep_prob, kind="mergesort")]
        repl_order = repl[np.argsort(repl_prob, kind="mergesort")[::-1]]
        max_swaps = min(len(top_order), len(repl_order), int(args.max_swaps_per_timestamp))
        for i in range(max_swaps):
            remove_id = top_order[i]
            add_id = repl_order[i]
            if repl_prob[np.where(repl == add_id)[0][0]] > keep_prob[np.where(top == remove_id)[0][0]] + float(args.swap_delta):
                z[remove_id] -= float(args.swap_logit_delta)
                z[add_id] += float(args.swap_logit_delta)
    return anchored._sigmoid(z)


def _fit_residual_model(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    anchor_valid: np.ndarray,
    rank_train: np.ndarray,
    rank_valid: np.ndarray,
    timestamps_train: pd.Series,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = (np.asarray(rank_train, dtype=np.float32) >= 0.50) & (np.asarray(y_train) >= 0)
    ids = np.flatnonzero(mask)
    if ids.size < int(args.min_train_rows) or len(np.unique(y_train[ids])) < 2:
        return np.asarray(anchor_valid, dtype=np.float32), {"reason": "constant_insufficient_rows_or_classes", "train_rows": int(ids.size)}
    x_tr, x_va, keep_cols = _prepare_train_pred_matrix(x_train, x_valid)
    if not keep_cols:
        return np.asarray(anchor_valid, dtype=np.float32), {"reason": "empty_matrix", "train_rows": int(ids.size)}
    p0 = np.clip(np.asarray(anchor_train, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    denom = np.clip(p0 * (1.0 - p0), 1e-3, None)
    residual = np.clip((np.asarray(y_train, dtype=np.float64) - p0) / denom, -2.0, 2.0).astype(np.float32)
    weight = _equal_timestamp_weights(timestamps_train, mask) * denom.astype(np.float32)
    min_child = max(50, int(math.ceil(float(args.residual_min_child_fraction) * len(ids))))
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(args.residual_n_estimators),
        learning_rate=0.035,
        max_depth=int(args.residual_max_depth),
        num_leaves=max(4, min(12, 2 ** int(args.residual_max_depth))),
        min_child_samples=min_child,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(x_tr.iloc[ids], residual[ids], sample_weight=np.where(weight[ids] > 0, weight[ids], 1.0))
    delta = np.clip(reg.predict(x_va).astype(np.float32, copy=False), -float(args.correction_clip), float(args.correction_clip))
    active_valid = np.asarray(rank_valid, dtype=np.float32) >= 0.50
    out_delta = np.zeros(len(anchor_valid), dtype=np.float32)
    out_delta[active_valid] = delta[active_valid]
    pred = anchored._sigmoid(anchored._safe_logit(anchor_valid) + out_delta)
    return pred, {
        "reason": "",
        "train_rows": int(ids.size),
        "feature_count": int(len(keep_cols)),
        "delta_mean": float(np.nanmean(out_delta[active_valid])) if active_valid.any() else np.nan,
        "delta_std": float(np.nanstd(out_delta[active_valid])) if active_valid.any() else np.nan,
    }


def _interaction_with_period(features: pd.DataFrame, period_features: pd.DataFrame) -> pd.DataFrame:
    if "q_period_inc" not in period_features.columns:
        return features
    q = pd.to_numeric(period_features["q_period_inc"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out = features.copy()
    cols = [c for c in features.columns if any(tok in c for tok in ("prediction_", "leaf_occupancy", "leaf_path", "leverage_", "liquidity_"))]
    for col in cols[:24]:
        vals = pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=np.float32)
        out[f"{col}__x__q_period_inc"] = (vals * q).astype(np.float32, copy=False)
    out = pd.concat([out, period_features.reset_index(drop=True)], axis=1, copy=False)
    return _downcast_numeric(out)


def _ndcg_for_selection(y_pool: np.ndarray, selected_local: np.ndarray, k: int) -> float:
    if k <= 0 or selected_local.size == 0:
        return np.nan
    gains = np.asarray(y_pool, dtype=np.float64)[selected_local[:k]]
    denom = np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    dcg = float(np.sum(gains / denom))
    ideal = np.sort(np.asarray(y_pool, dtype=np.float64))[::-1][: len(gains)]
    ideal_dcg = float(np.sum(ideal / denom)) if len(ideal) else 0.0
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else np.nan


def _fixed_pool_timestamp_metrics(
    *,
    head: str,
    arm: str,
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    anchor_score: np.ndarray,
    rank0: np.ndarray,
    min_timestamp_rows: int,
) -> pd.DataFrame:
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").reset_index(drop=True)
    frame = pd.DataFrame({"idx": np.arange(len(y), dtype=np.int64), "timestamp": ts, "rank0": rank0})
    rows: list[dict[str, Any]] = []
    for timestamp, group in frame.groupby("timestamp", sort=True):
        ids_all = group["idx"].to_numpy(dtype=np.int64)
        pool = ids_all[(y[ids_all] >= 0) & np.isfinite(pred[ids_all]) & np.isfinite(anchor_score[ids_all]) & np.isfinite(rank0[ids_all]) & (rank0[ids_all] >= 0.50)]
        if len(pool) < int(min_timestamp_rows):
            continue
        yy = y[pool].astype(np.float32)
        pp = pred[pool].astype(np.float64)
        rr = rank0[pool].astype(np.float64)
        aa = anchor_score[pool].astype(np.float64)
        row: dict[str, Any] = {
            "head": head,
            "arm": arm,
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "week": pd.Timestamp(timestamp).to_period("W").start_time.strftime("%Y-%m-%d"),
            "pool_rows": int(len(pool)),
        }
        for pct, thresh in ((10, 0.90), (20, 0.80), (30, 0.70)):
            base_local = np.flatnonzero(rr >= thresh)
            if len(base_local):
                base_local = base_local[np.argsort(aa[base_local], kind="mergesort")[::-1]]
            k = int(len(base_local))
            if k <= 0:
                continue
            cand_local = np.argsort(pp, kind="mergesort")[::-1][:k]
            row[f"selected_count_top{pct}"] = int(k)
            row[f"hr_top{pct}"] = float(np.mean(yy[cand_local]))
            row[f"baseline_hr_top{pct}"] = float(np.mean(yy[base_local]))
            row[f"delta_hr_top{pct}"] = row[f"hr_top{pct}"] - row[f"baseline_hr_top{pct}"]
            row[f"hit_count_top{pct}"] = float(np.sum(yy[cand_local]))
            row[f"baseline_hit_count_top{pct}"] = float(np.sum(yy[base_local]))
            if pct == 30:
                cand_set = set(cand_local.tolist())
                base_set = set(base_local.tolist())
                entrants = np.array(sorted(cand_set - base_set), dtype=np.int64)
                removed = np.array(sorted(base_set - cand_set), dtype=np.int64)
                union = cand_set | base_set
                row["ndcg_top30"] = _ndcg_for_selection(yy, cand_local, k)
                row["baseline_ndcg_top30"] = _ndcg_for_selection(yy, base_local, k)
                row["delta_ndcg_top30"] = row["ndcg_top30"] - row["baseline_ndcg_top30"] if np.isfinite(row["ndcg_top30"]) and np.isfinite(row["baseline_ndcg_top30"]) else np.nan
                row["top30_jaccard"] = float(len(cand_set & base_set) / len(union)) if union else np.nan
                row["top30_entrant_count"] = int(len(entrants))
                row["top30_removed_count"] = int(len(removed))
                row["top30_entrant_hit_rate"] = float(np.mean(yy[entrants])) if len(entrants) else np.nan
                row["top30_removed_hit_rate"] = float(np.mean(yy[removed])) if len(removed) else np.nan
                row["net_correct_trades_gained"] = float(np.sum(yy[entrants]) - np.sum(yy[removed])) if len(entrants) or len(removed) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _fixed_pool_aggregate(ts_metrics: pd.DataFrame) -> pd.DataFrame:
    if ts_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (head, arm), group in ts_metrics.groupby(["head", "arm"], sort=True):
        row: dict[str, Any] = {"head": head, "arm": arm, "timestamp_count": int(len(group)), "pool_rows": int(group["pool_rows"].sum())}
        for pct in (10, 20, 30):
            if f"hr_top{pct}" in group.columns:
                row[f"timestamp_weighted_hr_top{pct}"] = float(pd.to_numeric(group[f"hr_top{pct}"], errors="coerce").mean())
                row[f"baseline_timestamp_weighted_hr_top{pct}"] = float(pd.to_numeric(group[f"baseline_hr_top{pct}"], errors="coerce").mean())
                row[f"delta_timestamp_weighted_hr_top{pct}"] = float(pd.to_numeric(group[f"delta_hr_top{pct}"], errors="coerce").mean())
        row["ndcg_top30"] = float(pd.to_numeric(group.get("ndcg_top30"), errors="coerce").mean())
        row["baseline_ndcg_top30"] = float(pd.to_numeric(group.get("baseline_ndcg_top30"), errors="coerce").mean())
        row["delta_ndcg_top30"] = float(pd.to_numeric(group.get("delta_ndcg_top30"), errors="coerce").mean())
        row["top30_entrant_hit_rate"] = _weighted_rate(group.get("top30_entrant_hit_rate"), group.get("top30_entrant_count"))
        row["top30_removed_hit_rate"] = _weighted_rate(group.get("top30_removed_hit_rate"), group.get("top30_removed_count"))
        row["net_correct_trades_gained"] = float(pd.to_numeric(group.get("net_correct_trades_gained"), errors="coerce").fillna(0.0).sum())
        weekly_hits = group.groupby("week", sort=True)["hit_count_top30"].sum()
        weekly_counts = group.groupby("week", sort=True)["selected_count_top30"].sum()
        weekly_hr = weekly_hits / weekly_counts.replace(0, np.nan)
        row["weekly_hr_top30_q10"] = float(weekly_hr.quantile(0.10)) if len(weekly_hr) else np.nan
        row["weekly_hr_top30_q25"] = float(weekly_hr.quantile(0.25)) if len(weekly_hr) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_rate(rate: pd.Series | None, count: pd.Series | None) -> float:
    if rate is None or count is None:
        return np.nan
    r = pd.to_numeric(rate, errors="coerce")
    c = pd.to_numeric(count, errors="coerce").fillna(0.0)
    denom = float(c.sum())
    if denom <= 0:
        return np.nan
    return float((r.fillna(0.0) * c).sum() / denom)


def _write_report(out_dir: Path, comparison: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = ["# Fixed-Band q_fail Ablation", ""]
    if not comparison.empty:
        cols = [
            "head",
            "arm",
            "delta_auc",
            "delta_log_loss_improvement",
            "delta_timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top20",
            "delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "top30_entrant_hit_rate",
            "top30_removed_hit_rate",
            "net_correct_trades_gained",
        ]
        lines.extend(["## Comparison", "", comparison[[c for c in cols if c in comparison.columns]].to_markdown(index=False, floatfmt=".6f"), ""])
    if not diag.empty:
        cols = [
            "head",
            "fold",
            "model",
            "pred_corr_full_control",
            "inc_std",
            "incremental_auc",
            "incremental_pr_auc",
            "incremental_logloss_improvement",
            "non_control_features_used",
            "non_control_importance_share",
            "matrix_hashes_differ",
        ]
        lines.extend(["## Diagnostics", "", diag[[c for c in cols if c in diag.columns]].head(80).to_markdown(index=False, floatfmt=".6f"), ""])
    (out_dir / "fixed_band_qfail_ablation_report.md").write_text("\n".join(lines))


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
    fixed_ts_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []

    for head in heads:
        print(f"[fixed_band_qfail] head={head.head}", flush=True)
        panel = _downcast_numeric(_normalise_keys(pd.read_parquet(head.meta_oof_path)), exclude=["timestamp", "symbol"])
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
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
        anchor_score = ctx._current_meta_score(panel)
        rank0_all = _rank0(panel, anchor_score)
        returns = np.asarray(_pick_realized_return(panel), dtype=np.float32)
        folds = canon._make_chrono_folds(panel["timestamp"], int(args.outer_folds), embargo_hours=int(args.embargo_hours))
        fold_valid_mask = np.zeros(len(panel), dtype=bool)
        preds = {trial: np.full(len(panel), np.nan, dtype=np.float32) for trial in TRIALS}
        path_cols: list[str] | None = None

        for fold in folds:
            tr = np.asarray(fold.train_idx, dtype=np.int64)
            va = np.asarray(fold.valid_idx, dtype=np.int64)
            fold_valid_mask[va] = True
            ts_train = panel["timestamp"].iloc[tr].reset_index(drop=True)
            ts_valid = panel["timestamp"].iloc[va].reset_index(drop=True)
            raw_train = raw.iloc[tr].reset_index(drop=True)
            raw_valid = raw.iloc[va].reset_index(drop=True)
            rank_train = rank0_all[tr]
            rank_valid = rank0_all[va]
            anchor_ctrl_train, path_cols = anchored._anchor_control_features(
                raw_train, ts_train, anchor_score[tr], max_path_cols=int(args.max_control_path_features), selected_cols=path_cols
            )
            anchor_ctrl_valid, _ = anchored._anchor_control_features(
                raw_valid, ts_valid, anchor_score[va], max_path_cols=int(args.max_control_path_features), selected_cols=path_cols
            )
            controls_only_train, _ = anchored._anchor_control_features(
                raw_train, ts_train, anchor_score[tr], max_path_cols=0, selected_cols=[]
            )
            controls_only_valid, _ = anchored._anchor_control_features(
                raw_valid, ts_valid, anchor_score[va], max_path_cols=0, selected_cols=[]
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
            structural_train = stack._combine_features(canonical_train, leaf_train)
            structural_valid = stack._combine_features(canonical_valid, leaf_valid)
            full_train = stack._combine_features(anchor_ctrl_train, structural_train)
            full_valid = stack._combine_features(anchor_ctrl_valid, structural_valid)
            band_train = _band_features(rank_train)
            band_valid = _band_features(rank_valid)

            # F1: top30 conditional failure model.
            fail_mask = (rank_train >= 0.70) & (y[tr] >= 0)
            fail_target = np.full(len(tr), np.nan, dtype=np.float32)
            fail_target[fail_mask] = (y[tr][fail_mask] == 0).astype(np.float32)
            fail_pred, fail_diag = _crossfit_two_models(
                control_train=anchor_ctrl_train,
                full_train=full_train,
                control_valid=anchor_ctrl_valid,
                full_valid=full_valid,
                target_train=fail_target,
                mask_train=fail_mask,
                timestamps_train=ts_train,
                seed=int(args.seed + 101 * fold.fold_id),
                args=args,
            )
            # F2: 30-50 replacement-quality model.
            repl_mask = (rank_train >= 0.50) & (rank_train < 0.70) & (y[tr] >= 0)
            repl_target = np.full(len(tr), np.nan, dtype=np.float32)
            repl_target[repl_mask] = (y[tr][repl_mask] == 1).astype(np.float32)
            repl_pred, repl_diag = _crossfit_two_models(
                control_train=anchor_ctrl_train,
                full_train=full_train,
                control_valid=anchor_ctrl_valid,
                full_valid=full_valid,
                target_train=repl_target,
                mask_train=repl_mask,
                timestamps_train=ts_train,
                seed=int(args.seed + 211 * fold.fold_id),
                args=args,
            )
            preds[F0][va] = anchor_score[va]
            preds[F1_CONTROL][va] = _apply_demote(anchor_score[va], fail_pred["control_valid"], rank_valid, args)
            preds[F1_FULL][va] = _apply_demote(anchor_score[va], fail_pred["full_valid"], rank_valid, args)
            fail_inc_valid = anchored._safe_logit(fail_pred["full_valid"]) - anchored._safe_logit(fail_pred["control_valid"])
            preds[F1_INC][va] = anchored._score_with_delta(anchor_score[va], -float(args.fail_lambda) * fail_inc_valid, float(args.correction_clip))
            preds[F2_CONTROL][va] = _apply_promote(anchor_score[va], repl_pred["control_valid"], rank_valid, args)
            preds[F2_FULL][va] = _apply_promote(anchor_score[va], repl_pred["full_valid"], rank_valid, args)
            repl_inc_valid = anchored._safe_logit(repl_pred["full_valid"]) - anchored._safe_logit(repl_pred["control_valid"])
            preds[F2_INC][va] = anchored._score_with_delta(anchor_score[va], float(args.replace_lambda) * repl_inc_valid, float(args.correction_clip))
            preds[F3_CONTROL][va] = _apply_swap_rule(
                timestamps=ts_valid,
                anchor_score=anchor_score[va],
                rank=rank_valid,
                q_fail=fail_pred["control_valid"],
                q_replace=repl_pred["control_valid"],
                args=args,
            )
            preds[F3_FULL][va] = _apply_swap_rule(
                timestamps=ts_valid,
                anchor_score=anchor_score[va],
                rank=rank_valid,
                q_fail=fail_pred["full_valid"],
                q_replace=repl_pred["full_valid"],
                args=args,
            )
            f4_control_train = stack._combine_features(band_train, controls_only_train.loc[:, ["anchor_rank0_by_timestamp"]])
            f4_control_valid = stack._combine_features(band_valid, controls_only_valid.loc[:, ["anchor_rank0_by_timestamp"]])
            fail_inc_train = anchored._safe_logit(fail_pred["full_oof"]) - anchored._safe_logit(fail_pred["control_oof"])
            repl_inc_train = anchored._safe_logit(repl_pred["full_oof"]) - anchored._safe_logit(repl_pred["control_oof"])
            qinc_train = pd.DataFrame(
                {
                    "q_fail30_inc": fail_inc_train.astype(np.float32, copy=False),
                    "q_replace_inc": repl_inc_train.astype(np.float32, copy=False),
                }
            )
            qinc_valid = pd.DataFrame(
                {
                    "q_fail30_inc": fail_inc_valid.astype(np.float32, copy=False),
                    "q_replace_inc": repl_inc_valid.astype(np.float32, copy=False),
                }
            )
            f4_qinc_train = stack._combine_features(f4_control_train, qinc_train)
            f4_qinc_valid = stack._combine_features(f4_control_valid, qinc_valid)
            f4_struct_train = stack._combine_features(band_train, structural_train)
            f4_struct_valid = stack._combine_features(band_valid, structural_valid)
            f4c_pred, f4c_diag = _fit_residual_model(
                x_train=f4_control_train,
                x_valid=f4_control_valid,
                y_train=y[tr],
                anchor_train=anchor_score[tr],
                anchor_valid=anchor_score[va],
                rank_train=rank_train,
                rank_valid=rank_valid,
                timestamps_train=ts_train,
                seed=int(args.seed + 307 * fold.fold_id),
                args=args,
            )
            f4q_pred, f4q_diag = _fit_residual_model(
                x_train=f4_qinc_train,
                x_valid=f4_qinc_valid,
                y_train=y[tr],
                anchor_train=anchor_score[tr],
                anchor_valid=anchor_score[va],
                rank_train=rank_train,
                rank_valid=rank_valid,
                timestamps_train=ts_train,
                seed=int(args.seed + 353 * fold.fold_id),
                args=args,
            )
            f4s_pred, f4s_diag = _fit_residual_model(
                x_train=f4_struct_train,
                x_valid=f4_struct_valid,
                y_train=y[tr],
                anchor_train=anchor_score[tr],
                anchor_valid=anchor_score[va],
                rank_train=rank_train,
                rank_valid=rank_valid,
                timestamps_train=ts_train,
                seed=int(args.seed + 409 * fold.fold_id),
                args=args,
            )
            preds[F4_CONTROL][va] = f4c_pred
            preds[F4_QFAIL_INC][va] = f4q_pred
            preds[F4_STRUCT][va] = f4s_pred
            if head.head != "short_boll":
                z_source_train = stack._combine_features(canonical_train, current_x.iloc[tr].reset_index(drop=True))
                z_source_valid = stack._combine_features(canonical_valid, current_x.iloc[va].reset_index(drop=True))
                z_train = stack._timestamp_feature_table(z_source_train, ts_train, max_columns=int(args.max_timestamp_features))
                z_valid = stack._timestamp_feature_table(z_source_valid, ts_valid, max_columns=int(args.max_timestamp_features)).reindex(columns=z_train.columns)
                nuisance_train = anchored._timestamp_nuisance_table(ts_train, panel["symbol"].iloc[tr] if "symbol" in panel.columns else None)
                nuisance_valid = anchored._timestamp_nuisance_table(ts_valid, panel["symbol"].iloc[va] if "symbol" in panel.columns else None)
                period_train, period_valid, period_diag = anchored._period_increment_features(
                    z_train=z_train,
                    z_valid=z_valid,
                    nuisance_train=nuisance_train,
                    nuisance_valid=nuisance_valid,
                    train_timestamps=ts_train,
                    valid_timestamps=ts_valid,
                    y_train=y[tr],
                    anchor_train=anchor_score[tr],
                    seed=int(args.seed + 503 * fold.fold_id),
                    args=args,
                )
                f5_train = _interaction_with_period(f4_struct_train, period_train)
                f5_valid = _interaction_with_period(f4_struct_valid, period_valid)
                f5_pred, f5_diag = _fit_residual_model(
                    x_train=f5_train,
                    x_valid=f5_valid,
                    y_train=y[tr],
                    anchor_train=anchor_score[tr],
                    anchor_valid=anchor_score[va],
                    rank_train=rank_train,
                    rank_valid=rank_valid,
                    timestamps_train=ts_train,
                    seed=int(args.seed + 601 * fold.fold_id),
                    args=args,
                )
                preds[F5_PERIOD][va] = f5_pred
            else:
                period_diag = {"period_reason": "skipped_for_short_boll"}
                f5_diag = {"reason": "skipped_for_short_boll"}
                preds[F5_PERIOD][va] = np.nan
            common = {
                "head": head.head,
                "fold": int(fold.fold_id),
                "train_rows": int(len(tr)),
                "valid_rows": int(len(va)),
                **canonical_diag,
                **meta_leaf_diag,
                **base_leaf_diag,
            }
            diag_rows.append({**common, "model": "F1_fail30", **fail_diag})
            diag_rows.append({**common, "model": "F2_replace", **repl_diag})
            diag_rows.append({**common, "model": F4_CONTROL, **{f"resid_{k}": v for k, v in f4c_diag.items()}})
            diag_rows.append({**common, "model": F4_QFAIL_INC, **{f"resid_{k}": v for k, v in f4q_diag.items()}})
            diag_rows.append({**common, "model": F4_STRUCT, **{f"resid_{k}": v for k, v in f4s_diag.items()}})
            diag_rows.append({**common, "model": F5_PERIOD, **period_diag, **{f"resid_{k}": v for k, v in f5_diag.items()}})
            print(f"[fixed_band_qfail] head={head.head} fold={fold.fold_id}/{len(folds)} train={len(tr)} valid={len(va)}", flush=True)

        baseline_fold = anchor_score.copy()
        baseline_fold[~fold_valid_mask] = np.nan
        score_frame = pd.DataFrame(
            {
                "head": head.head,
                "row_id": np.arange(len(panel), dtype=np.int64),
                "timestamp": pd.to_datetime(panel["timestamp"], utc=True, errors="coerce"),
                "symbol": panel["symbol"].astype(str) if "symbol" in panel.columns else "",
                "y_bin": y,
                "rank0": rank0_all,
                "anchor_score": baseline_fold,
            }
        )
        for arm, pred in preds.items():
            if arm == F5_PERIOD and head.head == "short_boll":
                continue
            score_frame[arm] = pred
            summary = ctx._overall_metrics(
                head=head.head,
                arm=arm,
                variant="fixed_band_qfail",
                y=y,
                pred=pred,
                baseline_pred=baseline_fold,
                returns=returns,
            )
            summary_rows.append(summary)
            ts_metrics = _fixed_pool_timestamp_metrics(
                head=head.head,
                arm=arm,
                panel=panel,
                y=y,
                pred=pred,
                anchor_score=baseline_fold,
                rank0=rank0_all,
                min_timestamp_rows=int(args.pool_min_timestamp_rows),
            )
            if not ts_metrics.empty:
                fixed_ts_frames.append(ts_metrics)
        score_frames.append(score_frame)

    summary = pd.DataFrame(summary_rows)
    fixed_ts = pd.concat(fixed_ts_frames, axis=0, ignore_index=True) if fixed_ts_frames else pd.DataFrame()
    fixed_metrics = _fixed_pool_aggregate(fixed_ts)
    if fixed_metrics.empty and not {"head", "arm"}.issubset(fixed_metrics.columns):
        fixed_metrics = pd.DataFrame(columns=["head", "arm"])
    diag = pd.DataFrame(diag_rows)
    scores = pd.concat(score_frames, axis=0, ignore_index=True) if score_frames else pd.DataFrame()
    comparison = summary.merge(fixed_metrics, on=["head", "arm"], how="left")
    comparison.to_csv(out_dir / "fixed_band_qfail_comparison_table.csv", index=False)
    summary.to_csv(out_dir / "fixed_band_qfail_global_summary.csv", index=False)
    fixed_metrics.to_csv(out_dir / "fixed_band_qfail_pool_metrics.csv", index=False)
    fixed_ts.to_csv(out_dir / "fixed_band_qfail_pool_timestamp_metrics.csv", index=False)
    diag.to_csv(out_dir / "fixed_band_qfail_diagnostics.csv", index=False)
    if not scores.empty:
        scores.to_parquet(out_dir / "fixed_band_qfail_oof_scores.parquet", index=False)
    manifest = {
        "output_dir": str(out_dir),
        "heads": [h.head for h in heads],
        "trials": list(TRIALS),
        "contract": {
            "target": "unchanged_y_bin",
            "fixed_pool": "rank0>=0.50",
            "baseline_top30": "rank0>=0.70",
            "candidate_top30": "same timestamp count selected from rank0>=0.50",
            "short_boll_f5": "skipped",
        },
        "args": vars(args),
    }
    (out_dir / "fixed_band_qfail_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    _write_report(out_dir, comparison, diag)
    print(f"[fixed_band_qfail] wrote {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--transform-cache", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/fixed_band_qfail_ablation_20260623")
    parser.add_argument("--only-head", nargs="*", default=list(HEADS))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--inner-embargo-hours", type=int, default=12)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-control-path-features", type=int, default=40)
    parser.add_argument("--max-timestamp-features", type=int, default=80)
    parser.add_argument("--leaf-max-models", type=int, default=1)
    parser.add_argument("--leaf-tree-stride", type=int, default=3)
    parser.add_argument("--leaf-max-trees", type=int, default=120)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--min-child-fraction", type=float, default=0.025)
    parser.add_argument("--fail-lambda", type=float, default=0.35)
    parser.add_argument("--replace-lambda", type=float, default=0.35)
    parser.add_argument("--swap-delta", type=float, default=0.02)
    parser.add_argument("--swap-logit-delta", type=float, default=0.50)
    parser.add_argument("--max-swaps-per-timestamp", type=int, default=20)
    parser.add_argument("--correction-clip", type=float, default=0.50)
    parser.add_argument("--residual-max-depth", type=int, default=2)
    parser.add_argument("--residual-n-estimators", type=int, default=160)
    parser.add_argument("--residual-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--period-short-window", type=int, default=72)
    parser.add_argument("--period-long-window", type=int, default=120)
    parser.add_argument("--period-difficult-quantile", type=float, default=0.25)
    parser.add_argument("--period-min-train-timestamps", type=int, default=80)
    parser.add_argument("--period-max-depth", type=int, default=3)
    parser.add_argument("--period-n-estimators", type=int, default=160)
    parser.add_argument("--period-min-child-fraction", type=float, default=0.035)
    parser.add_argument("--pool-min-timestamp-rows", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
